//! `astrum-sim` binary entrypoint.
//!
//! Loads a scenario, builds the headless simulation app ([`astrum_sim::build_app`]),
//! binds the ZMQ telemetry/command sockets, and runs the loop. All the app-building
//! and system logic lives in the `astrum_sim` library; the integration tests in
//! `tests/` exercise it there. This file is only the process wiring: scenario load,
//! socket bind, `run()`.

use astrum_net::{CommandReceiver, TelemetryPublisher};
use astrum_sim::build_app;
use astrum_sim::scenario::Scenario;
use bevy::prelude::*;

/// Bind the telemetry `PUB` and command `PULL` sockets and install them as
/// `NonSend` resources, activating the bridge systems. Endpoints come from
/// `$ASTRUM_TELEMETRY_BIND` / `$ASTRUM_COMMAND_BIND` (defaults in `astrum_net`).
/// A bind failure is logged, not fatal — the bridge systems stay inert.
fn install_net(app: &mut App) {
    let tlm = std::env::var("ASTRUM_TELEMETRY_BIND")
        .unwrap_or_else(|_| astrum_net::DEFAULT_TELEMETRY_ENDPOINT.to_string());
    let cmd = std::env::var("ASTRUM_COMMAND_BIND")
        .unwrap_or_else(|_| astrum_net::DEFAULT_COMMAND_ENDPOINT.to_string());

    let ctx = astrum_net::context();
    match TelemetryPublisher::bind(&ctx, &tlm) {
        Ok(p) => {
            app.insert_non_send(p);
            println!("[astrum-sim] telemetry PUB bound: {tlm}");
        }
        Err(e) => eprintln!("[astrum-sim] telemetry bind {tlm} failed: {e}"),
    }
    match CommandReceiver::bind(&ctx, &cmd) {
        Ok(r) => {
            app.insert_non_send(r);
            println!("[astrum-sim] command PULL bound: {cmd}");
        }
        Err(e) => eprintln!("[astrum-sim] command bind {cmd} failed: {e}"),
    }
    // Keep the context alive for the app's lifetime (sockets reference it).
    app.insert_non_send(ctx);
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
