//! `astrum-sim` binary: load a scenario, build the app, open the Zenoh session
//! (telemetry out, commands in, controller presence, the control plane and the
//! scenario queryable), optionally record to an MCAP bag, and run. Every
//! startup failure is fatal: a sim that cannot serve anything exits non-zero.
//! SIGINT/SIGTERM end the run cleanly (a second one exits at once).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::Ordering;

use astrum_net::{ScenarioServer, Session};
use astrum_sim::components::{SimClock, SimParams};
use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::stopped_line;
use astrum_sim::systems::StopRequested;
use astrum_sim::{build_app, net, record, run_to_exit, RunMode};
use bevy::prelude::*;
use clap::Parser;

/// Headless multi-agent flight simulator: loads a scenario and serves telemetry
/// (and ingests commands) over Zenoh.
#[derive(Parser)]
#[command(name = "astrum-sim", version, about)]
struct Cli {
    /// Scenario TOML to run.
    #[arg(env = "ASTRUM_SCENARIO", default_value = "config/scenario.example.toml")]
    scenario: PathBuf,

    /// Zenoh endpoint to listen on.
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value_t = astrum_net::DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,

    /// FDM stepping threads, overriding `sim.threads` (0 = all cores).
    #[arg(long, env = "ASTRUM_THREADS")]
    threads: Option<usize>,

    /// Pace as a multiple of real time, overriding `sim.speed` (1 = real time,
    /// 10 = ten times faster, 0 = as fast as the host allows).
    #[arg(long, env = "ASTRUM_SPEED")]
    speed: Option<f64>,

    /// Record every tick's telemetry (and the commands received) to this MCAP
    /// bag, in-process: lossless at any speed (`astrum-bag-play` and
    /// `astrum-web-export` read it).
    #[arg(long, env = "ASTRUM_RECORD")]
    record: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut scenario = match Scenario::load(&cli.scenario) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] scenario error: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(n) = cli.threads {
        scenario.sim.threads = n;
    }
    if let Some(speed) = cli.speed {
        scenario.sim.speed = speed;
    }
    if let Err(e) = scenario.validate() {
        eprintln!("[astrum-sim] {e}");
        return ExitCode::FAILURE;
    }
    let scenario_toml = scenario.to_toml();
    let session = match Session::listen(&cli.listen) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] zenoh listen {} failed: {e}", cli.listen);
            return ExitCode::FAILURE;
        }
    };
    let recording = match &cli.record {
        Some(path) => match record::start(path, &scenario_toml) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("[astrum-sim] cannot record to {}: {e}", path.display());
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
        eprintln!("[astrum-sim] cannot install the signal handler: {e}");
        return ExitCode::FAILURE;
    }
    let _scenario_server = match ScenarioServer::serve(&session, scenario_toml) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] scenario queryable failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[astrum-sim] zenoh session listening: {}", cli.listen);
    match scenario.sim.speed {
        s if s > 0.0 => println!("[astrum-sim] pace: {s}× real time"),
        _ => println!("[astrum-sim] pace: unpaced (as fast as possible)"),
    }

    let mut app = build_app(scenario, RunMode::Server);
    if let Err(e) = net::connect(&mut app, &session) {
        eprintln!("[astrum-sim] zenoh setup failed: {e}");
        return ExitCode::FAILURE;
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
                "[astrum-sim] recorded {} telemetry, {} command messages → {path}",
                s.telemetry, s.commands
            ),
            Err(e) => {
                eprintln!("[astrum-sim] recording {path} failed: {e}");
                code = ExitCode::FAILURE;
            }
        }
    }
    code
}
