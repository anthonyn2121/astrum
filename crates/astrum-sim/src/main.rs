//! `astrum-sim` binary: load a scenario, build the app, open the Zenoh session
//! (telemetry out, commands in, scenario queryable) and run. Every startup
//! failure is fatal: a sim that cannot serve anything exits non-zero.

use std::path::PathBuf;
use std::process::ExitCode;

use astrum_net::{CommandReceiver, ScenarioServer, Session, TelemetryPublisher};
use astrum_sim::scenario::Scenario;
use astrum_sim::{build_app, RunMode};
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
    let scenario_toml = scenario.to_toml();
    let session = match Session::listen(&cli.listen) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] zenoh listen {} failed: {e}", cli.listen);
            return ExitCode::FAILURE;
        }
    };
    let receiver = match CommandReceiver::bind(&session) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[astrum-sim] command subscribe failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let _scenario_server = match ScenarioServer::serve(&session, scenario_toml) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] scenario queryable failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("[astrum-sim] zenoh session listening: {}", cli.listen);

    let mut app = build_app(scenario, RunMode::Server);
    app.insert_non_send(TelemetryPublisher::new(&session));
    app.insert_non_send(receiver);
    match app.run() {
        AppExit::Success => ExitCode::SUCCESS,
        AppExit::Error(code) => ExitCode::from(code.get()),
    }
}
