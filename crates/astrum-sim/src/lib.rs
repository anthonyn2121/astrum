//! `astrum-sim` — the ASTRUM headless Bevy ECS state server, as a library.
//!
//! Stands up a headless Bevy app (`MinimalPlugins`, no render/winit) that spawns
//! one entity per scenario aircraft — each owning an independent JSBSim FDM — and
//! advances them one deterministic step per loop, deriving a Cartesian ENU pose
//! from each aircraft's geodetic state, with a Zenoh telemetry/command bridge.
//!
//! The app-building logic lives here (rather than in `main.rs`) so the binary and
//! the integration tests in `tests/` share the exact same [`build_app`]; tests
//! simulate the real main loop by pumping [`bevy::app::App::update`].

pub mod components;
pub mod net;
pub mod scenario;
pub mod systems;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use std::time::Duration;

use astrum_geo::{Geodetic, LocalFrame};
use components::{SimParams, WorldFrame};
use net::{autopilot_step, ingest_commands, publish_telemetry};
use scenario::Scenario;
use systems::{
    pace, report_progress, spawn_fleet, step_fdm, stop_at_max_time, sync_geo_transform, Diag, Pacer,
};

/// Build the headless simulation app from a scenario (does not run it). Shared by
/// the `astrum-sim` binary and the integration tests, which drive it via manual
/// `update()` — the same [`Startup`] + [`Update`] schedule the binary runs.
pub fn build_app(scenario: Scenario) -> App {
    let rate = scenario.sim.rate_hz;
    let dt = 1.0 / rate;
    let params = SimParams {
        rate_hz: rate,
        dt,
        max_time_s: scenario.sim.max_time_s,
        realtime: scenario.sim.realtime,
    };

    let origin = Geodetic::from_degrees(
        scenario.world.origin_lat_deg,
        scenario.world.origin_lon_deg,
        scenario.world.origin_alt_m,
    );
    let frame = WorldFrame(LocalFrame::new(origin));

    println!(
        "[astrum-sim] ENU origin: {:.6}, {:.6} @ {:.1} m",
        scenario.world.origin_lat_deg, scenario.world.origin_lon_deg, scenario.world.origin_alt_m
    );

    // Physics steps exactly once per runner loop; the `pace` system throttles the
    // loop (we drive the runner unthrottled — Bevy's own sleep is a no-op here).
    // Pacing only changes wall-clock timing, never the number of steps or their
    // dt, so the trajectory is fully deterministic.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO)))
        .insert_resource(scenario)
        .insert_resource(params)
        .insert_resource(frame)
        .insert_resource(Diag::new())
        .insert_resource(Pacer::new(dt, params.realtime))
        .add_systems(Startup, spawn_fleet)
        .add_systems(
            Update,
            (
                // Commands first (they set this tick's inputs), then physics,
                // then derive pose, then publish the fresh state.
                ingest_commands,
                autopilot_step,
                step_fdm,
                sync_geo_transform,
                publish_telemetry,
                report_progress,
                stop_at_max_time,
                pace,
            )
                .chain(),
        );
    app
}
