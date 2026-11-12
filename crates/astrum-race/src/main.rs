//! `astrum-race` binary: load a race scenario, fix its course, and run it as an
//! `astrum-sim` run — the Zenoh bridge (telemetry out, commands in, controller
//! presence, the control plane, the scenario queryable), optional recording to an
//! MCAP bag, signals ending the run cleanly — plus the race's own course and state
//! on `astrum/race/*`. Every startup failure is fatal.
//!
//! The scenario served and recorded is the *resolved* one: every gate explicit, the
//! course seed spelled out (see `ResolvedRace::to_toml`).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::Ordering;

use astrum_net::{ScenarioServer, Session};
use astrum_race::course::Course;
use astrum_race::{build_race_app, RaceScenario};
use astrum_sim::components::{SimClock, SimParams};
use astrum_sim::sweep::stopped_line;
use astrum_sim::systems::StopRequested;
use astrum_sim::{record, run_to_exit, RunMode};
use bevy::prelude::*;
use clap::Parser;

/// Headless drone-racing environment: an F450 quadrotor, a gate course, and the
/// astrum-sim bus.
#[derive(Parser)]
#[command(name = "astrum-race", version, about)]
struct Cli {
    /// Race scenario TOML (an astrum-sim scenario plus a `[course]` table).
    #[arg(env = "ASTRUM_SCENARIO", default_value = "config/scenario.race.toml")]
    scenario: PathBuf,

    /// Zenoh endpoint to listen on.
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value_t = astrum_net::DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,

    /// Course seed, overriding `course.seed` / `sim.seed`: a different seed is a
    /// different course. Not valid with explicit `[[course.gate]]`.
    #[arg(long)]
    seed: Option<u64>,

    /// Override `course.gates` (the number of sampled gates).
    #[arg(long)]
    gates: Option<u32>,

    /// Print the course and exit, without touching JSBSim or the network. Handy for
    /// eyeballing what a seed draws.
    #[arg(long)]
    dump_course: bool,

    /// FDM stepping threads, overriding `sim.threads` (0 = all cores).
    #[arg(long, env = "ASTRUM_THREADS")]
    threads: Option<usize>,

    /// Pace as a multiple of real time, overriding `sim.speed` (1 = real time,
    /// 0 = as fast as the host allows).
    #[arg(long, env = "ASTRUM_SPEED")]
    speed: Option<f64>,

    /// Record every tick's telemetry, the commands received and the events
    /// (gates passed, crashes, the end) to this MCAP bag, with the resolved
    /// scenario as its metadata.
    #[arg(long, env = "ASTRUM_RECORD")]
    record: Option<PathBuf>,

    /// Run with no transport at all: nothing published, no port bound (a recording
    /// still works). For physics-only runs.
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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut race = match RaceScenario::load(&cli.scenario) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[astrum-race] scenario error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(n) = cli.threads {
        race.sim.sim.threads = n;
    }
    if let Some(speed) = cli.speed {
        race.sim.sim.speed = speed;
    }
    if let Some(gates) = cli.gates {
        if !race.course.gates.is_empty() {
            eprintln!(
                "[astrum-race] --gates has no effect on a scenario with explicit [[course.gate]]"
            );
            return ExitCode::FAILURE;
        }
        race.course.params.gates = gates;
    }
    let resolved = match race.course.validate().and_then(|()| {
        race.sim.validate()?;
        race.resolve(cli.seed)
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[astrum-race] {e}");
            return ExitCode::FAILURE;
        }
    };
    if cli.dump_course {
        dump(&resolved.course);
        return ExitCode::SUCCESS;
    }

    let scenario_toml = resolved.to_toml();
    let session = if cli.no_net {
        None
    } else {
        match Session::listen(&cli.listen) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("[astrum-race] zenoh listen {} failed: {e}", cli.listen);
                return ExitCode::FAILURE;
            }
        }
    };
    let recording = match &cli.record {
        Some(path) => match record::start(path, &scenario_toml) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("[astrum-race] cannot record to {}: {e}", path.display());
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let stop = StopRequested::default();
    let flag = stop.0.clone();
    if let Err(e) = ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::Relaxed) {
            std::process::exit(130); // second signal: give up on a clean exit
        }
    }) {
        eprintln!("[astrum-race] cannot install the signal handler: {e}");
        return ExitCode::FAILURE;
    }
    let speed = resolved.scenario.sim.speed;
    let mut app = build_race_app(resolved, RunMode::Server);
    let _scenario_server = match &session {
        Some(session) => {
            let served = ScenarioServer::serve(session, scenario_toml).and_then(|server| {
                astrum_sim::net::connect(&mut app, session)?;
                astrum_race::net::connect(&mut app, session)?;
                Ok(server)
            });
            match served {
                Ok(server) => {
                    println!("[astrum-race] zenoh session listening: {}", cli.listen);
                    Some(server)
                }
                Err(e) => {
                    eprintln!("[astrum-race] zenoh setup failed: {e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        None => None,
    };
    match speed {
        s if s > 0.0 => println!("[astrum-race] pace: {s}× real time"),
        _ => println!("[astrum-race] pace: unpaced (as fast as possible)"),
    }
    app.insert_resource(stop);
    let (recorder, recording) = recording.unzip();
    if let Some(r) = recorder {
        app.insert_resource(r);
    }
    let started = std::time::Instant::now();
    let mut code = match run_to_exit(&mut app) {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    };
    let sim_s = app.world().resource::<SimClock>().time_s(app.world().resource::<SimParams>());
    let wall_s = started.elapsed().as_secs_f64();
    println!("{}", stopped_line(app.world().resource::<SimClock>().tick, sim_s, wall_s));
    // Dropping the app drops the recorder, closing the queue `finish` waits on.
    drop(app);
    if let Some(recording) = recording {
        let path = recording.path().display().to_string();
        match recording.finish() {
            Ok(s) => println!(
                "[astrum-race] recorded {} telemetry, {} command, {} event messages → {path}",
                s.telemetry, s.commands, s.events
            ),
            Err(e) => {
                eprintln!("[astrum-race] recording {path} failed: {e}");
                code = ExitCode::FAILURE;
            }
        }
    }
    code
}
