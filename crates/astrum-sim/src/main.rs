//! `astrum-sim` binary entrypoint.
//!
//! Loads a scenario, builds the headless simulation app ([`astrum_sim::build_app`]),
//! opens the Zenoh session (telemetry out + commands in), and runs the loop. All
//! the app-building and system logic lives in the `astrum_sim` library; the
//! integration tests in `tests/` exercise it there. This file is only the process
//! wiring: scenario load, session open, `run()`.

use astrum_net::{CommandReceiver, Session, TelemetryPublisher};
use astrum_sim::build_app;
use astrum_sim::scenario::Scenario;
use bevy::prelude::*;

/// Open the Zenoh session (LISTEN endpoint from `$ASTRUM_ZENOH_LISTEN`, default
/// in `astrum_net`) and install the telemetry publisher + command receiver as
/// `NonSend` resources, activating the bridge systems. Telemetry and commands
/// share the one session. A failure to open is logged, not fatal — the bridge
/// systems stay inert.
fn install_net(app: &mut App) {
    let listen = std::env::var("ASTRUM_ZENOH_LISTEN")
        .unwrap_or_else(|_| astrum_net::DEFAULT_LISTEN_ENDPOINT.to_string());

    let session = match Session::listen(&listen) {
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
    let scenario = match Scenario::load_from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum-sim] scenario error: {e}");
            std::process::exit(1);
        }
    };
    let mut app = build_app(scenario);
    install_net(&mut app);
    app.run();
}
