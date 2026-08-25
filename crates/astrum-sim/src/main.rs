//! `astrum-sim` binary entrypoint.
//!
//! Loads a scenario, builds the headless simulation app ([`astrum_sim::build_app`]),
//! opens the Zenoh session (telemetry out + commands in), and runs the loop. All
//! the app-building and system logic lives in the `astrum_sim` library; the
//! integration tests in `tests/` exercise it there. This file is only the process
//! wiring: scenario load, session open, `run()`.

use std::path::PathBuf;

use astrum_net::{CommandReceiver, Session, TelemetryPublisher};
use astrum_sim::build_app;
use astrum_sim::scenario::Scenario;
use bevy::prelude::*;
use clap::Parser;

/// Headless multi-agent flight simulator: loads a scenario and serves telemetry
/// (and ingests commands) over Zenoh.
#[derive(Parser)]
#[command(name = "astrum-sim", version, about)]
struct Cli {
    /// Scenario TOML to run (the fleet + world origin).
    #[arg(env = "ASTRUM_SCENARIO", default_value = "config/scenario.example.toml")]
    scenario: PathBuf,

    /// Zenoh endpoint to listen on (telemetry out + commands in).
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value_t = astrum_net::DEFAULT_LISTEN_ENDPOINT.to_string())]
    listen: String,
}

/// Open the Zenoh session on `listen` and install the telemetry publisher +
/// command receiver as `NonSend` resources, activating the bridge systems.
/// Telemetry and commands share the one session. A failure to open is logged,
/// not fatal — the bridge systems stay inert.
fn install_net(app: &mut App, listen: &str) {
    let session = match Session::listen(listen) {
        Ok(s) => {
            println!("[astrum-sim] zenoh session listening: {listen}");
            s
        }
        Err(e) => {
            eprintln!("[astrum-sim] zenoh listen {listen} failed: {e}");
            return;
        }
    };

    app.insert_non_send(TelemetryPublisher::new(&session));
    match CommandReceiver::bind(&session) {
        Ok(r) => {
            app.insert_non_send(r);
        }
        Err(e) => eprintln!("[astrum-sim] command subscribe failed: {e}"),
    }
}

fn main() {
    let cli = Cli::parse();
    let scenario = match Scenario::load(&cli.scenario) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] scenario error: {e}");
            std::process::exit(1);
        }
    };
    let mut app = build_app(scenario);
    install_net(&mut app, &cli.listen);
    app.run();
}
