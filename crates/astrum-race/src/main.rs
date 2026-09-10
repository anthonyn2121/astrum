//! `astrum-race` binary entrypoint.
//!
//! Loads a race scenario, randomizes the gate course from its seed, and either
//! prints the course (`--dump-course`) or flies it, serving telemetry + race state
//! over Zenoh and taking commands back. All the app-building and system logic lives
//! in the `astrum_race` library, where the integration tests exercise it; this file
//! is only the process wiring: scenario load, session open, `run()`.

use std::path::PathBuf;

use astrum_net::Session;
use astrum_race::course::Course;
use astrum_race::scenario::RaceScenario;
use astrum_race::{build_app_with_course, generate_course};
use bevy::prelude::App;
use clap::Parser;

/// Headless drone-racing environment: an F450 quadrotor and a Monte-Carlo
/// randomized gate course.
#[derive(Parser)]
#[command(name = "astrum-race", version, about)]
struct Cli {
    /// Race scenario TOML (world origin, sim rate, drone, course envelope).
    #[arg(env = "ASTRUM_RACE_SCENARIO", default_value = "config/race.toml")]
    scenario: PathBuf,

    /// Override `sim.seed` — a different seed is a different course.
    #[arg(long)]
    seed: Option<u64>,

    /// Override `course.gates`.
    #[arg(long)]
    gates: Option<u32>,

    /// Print the sampled course and exit, without touching JSBSim. Handy for
    /// eyeballing what a seed draws, and for feeding a track to a renderer.
    #[arg(long)]
    dump_course: bool,

    /// Zenoh endpoint to listen on: telemetry + race state out, commands in.
    #[arg(long, env = "ASTRUM_RACE_LISTEN", default_value_t = astrum_net::DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,

    /// Run with no transport at all — physics only, nothing published, no port
    /// bound. Useful when the combat sim already holds the default endpoint.
    #[arg(long)]
    no_net: bool,
}

/// Print every gate, plus the leg and turn it sits between.
fn dump(course: &Course) {
    let legs = course.leg_lengths_m();
    let turns = course.turn_angles_deg();
    let climbs = course.climb_angles_deg();
    println!("# {}", course.summary());
    println!("# idx      east     north       alt     yaw    leg    turn   climb");
    for (i, g) in course.gates.iter().enumerate() {
        // A leg and its climb belong to the gate they *leave*; a turn belongs to the
        // gate it happens at, so gate 0 has neither.
        let cell = |v: Option<&f64>| {
            v.map(|x| format!("{x:7.1}")).unwrap_or_else(|| "      -".to_string())
        };
        println!(
            "{:5} {:9.2} {:9.2} {:9.2} {:7.1} {} {} {}",
            i,
            g.center.x,
            g.center.y,
            g.center.z,
            g.yaw_rad.to_degrees(),
            cell(legs.get(i)),
            cell(i.checked_sub(1).and_then(|k| turns.get(k))),
            cell(climbs.get(i)),
        );
    }
}

fn main() {
    let cli = Cli::parse();
    let mut scenario = match RaceScenario::load(&cli.scenario) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-race] scenario error: {e}");
            std::process::exit(1);
        }
    };
    if let Some(seed) = cli.seed {
        scenario.sim.seed = seed;
    }
    if let Some(gates) = cli.gates {
        scenario.course.gates = gates;
    }

    let course = match generate_course(&scenario) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[astrum-race] {e}");
            std::process::exit(1);
        }
    };

    if cli.dump_course {
        dump(&course);
        return;
    }

    let mut app = build_app_with_course(scenario, course);
    if !cli.no_net {
        install_net(&mut app, &cli.listen);
    }
    app.run();
}

/// Open the Zenoh session on `listen` and activate the bridge systems.
///
/// A failure to open is logged, not fatal: the environment still flies, just with
/// nothing published and no commands accepted. That matters because the default
/// endpoint is the *same* one `astrum-sim` listens on — running a race alongside a
/// combat sim degrades to a silent race rather than an exit.
fn install_net(app: &mut App, listen: &str) {
    match Session::listen(listen) {
        Ok(session) => {
            println!("[astrum-race] zenoh session listening: {listen}");
            astrum_race::net::install(app, &session);
        }
        Err(e) => eprintln!("[astrum-race] zenoh listen {listen} failed: {e} — flying offline"),
    }
}
