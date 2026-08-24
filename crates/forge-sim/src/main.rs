//! `forge-sim` binary entrypoint.
//!
//! Loads a scenario, builds the headless simulation app ([`forge_sim::build_app`]),
//! binds the ZMQ telemetry/command sockets, and runs the loop. All the app-building
//! and system logic lives in the `forge_sim` library; the integration tests in
//! `tests/` exercise it there. This file is only the process wiring: scenario load,
//! socket bind, `run()`.

use bevy::prelude::*;
use forge_net::{CommandReceiver, TelemetryPublisher};
use forge_sim::build_app;
use forge_sim::scenario::Scenario;

/// Bind the telemetry `PUB` and command `PULL` sockets and install them as
/// `NonSend` resources, activating the bridge systems. Endpoints come from
/// `$FORGE_TELEMETRY_BIND` / `$FORGE_COMMAND_BIND` (defaults in `forge_net`).
/// A bind failure is logged, not fatal — the bridge systems stay inert.
fn install_net(app: &mut App) {
    let tlm = std::env::var("FORGE_TELEMETRY_BIND")
        .unwrap_or_else(|_| forge_net::DEFAULT_TELEMETRY_ENDPOINT.to_string());
    let cmd = std::env::var("FORGE_COMMAND_BIND")
        .unwrap_or_else(|_| forge_net::DEFAULT_COMMAND_ENDPOINT.to_string());

    let ctx = forge_net::context();
    match TelemetryPublisher::bind(&ctx, &tlm) {
        Ok(p) => {
            app.insert_non_send(p);
            println!("[forge-sim] telemetry PUB bound: {tlm}");
        }
        Err(e) => eprintln!("[forge-sim] telemetry bind {tlm} failed: {e}"),
    }
    match CommandReceiver::bind(&ctx, &cmd) {
        Ok(r) => {
            app.insert_non_send(r);
            println!("[forge-sim] command PULL bound: {cmd}");
        }
        Err(e) => eprintln!("[forge-sim] command bind {cmd} failed: {e}"),
    }
    // Keep the context alive for the app's lifetime (sockets reference it).
    app.insert_non_send(ctx);
}

fn main() {
    let scenario = match Scenario::load_from_env() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[forge-sim] scenario error: {e}");
            std::process::exit(1);
        }
    };
    let mut app = build_app(scenario);
    install_net(&mut app);
    app.run();
}
