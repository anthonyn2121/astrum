//! `astrum-sim` — the ASTRUM headless Bevy ECS state server, as a library.
//!
//! [`build_app`] stands up a headless Bevy app (no plugins) that spawns one
//! entity per scenario aircraft — each owning an independent JSBSim FDM — and
//! advances them one deterministic step per update, deriving the world-grid
//! state from each aircraft's geodetic state. The binary adds the Zenoh bridge
//! and drives it with [`run_to_exit`], as the sweep runner ([`sweep`]) does for
//! many runs at once; tests and the embedded RL env ([`env`]) pump the same app.
//!
//! # One update == one tick, in this order
//! `ingest_commands` → `expire_leases` → `autopilot_step` → `fire_guns` (rounds
//! leave at `t_k`) → `step_fdm` (all FDMs in parallel to `t_k+1`) →
//! `freeze_dead` → `advance_clock` → `step_bullets` → `resolve_hits` →
//! `apply_damage` → `end_on_elimination` → `publish_telemetry` →
//! `report_progress` / `stop_at_max_time` → `pace` → `exit_on_fatal`.
//! Startup spawns the fleet and publishes tick 0.

pub mod combat;
pub mod components;
pub mod env;
pub mod net;
pub mod pilot;
pub mod record;
pub mod scenario;
pub mod step_pool;
pub mod sweep;
pub mod systems;

use bevy::prelude::*;

use astrum_geo::{Geodetic, LocalFrame};
use combat::systems::{apply_damage, end_on_elimination, fire_guns, resolve_hits, step_bullets};
use combat::{CombatParams, CombatRng, CombatState};
use components::{SimClock, SimParams, SimStatus, WorldFrame};
use net::{ingest_commands, publish_telemetry, LastPublished, RefusalLog};
use pilot::{autopilot_step, expire_leases};
use scenario::Scenario;
use systems::{
    advance_clock, exit_on_fatal, exit_on_signal, freeze_dead, pace, report_progress, spawn_fleet,
    step_fdm, stop_at_max_time, Diag, Pacer, StepPool,
};

/// Resolve the scenario's timing into [`SimParams`].
pub fn sim_params(scenario: &Scenario) -> SimParams {
    let s = &scenario.sim;
    let dt = 1.0 / s.rate_hz;
    SimParams {
        rate_hz: s.rate_hz,
        dt,
        max_time_s: s.max_time_s,
        speed: s.speed,
        lease_ticks: (s.command_lease_s > 0.0).then(|| (s.command_lease_s / dt).ceil() as u64),
        lockstep: s.lockstep,
        lockstep_timeout_s: s.lockstep_timeout_s,
    }
}

/// Number of distinct teams in a scenario (gates combat termination).
pub fn team_count(scenario: &Scenario) -> u8 {
    let mut teams: Vec<u8> = scenario.aircraft.iter().map(|a| a.team).collect();
    teams.sort_unstable();
    teams.dedup();
    teams.len() as u8
}

/// The world grid of a scenario.
pub fn world_frame(scenario: &Scenario) -> WorldFrame {
    let w = &scenario.world;
    WorldFrame(LocalFrame::new(Geodetic::from_degrees(
        w.origin_lat_deg,
        w.origin_lon_deg,
        w.origin_alt_hae_m,
    )))
}

/// How the app is driven.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    /// The `astrum-sim` binary: progress reports, `sim.speed` pacing, and an
    /// exit request when the episode ends (time cap or engagement decided).
    Server,
    /// One run of a sweep: pacing and exit requests, but no progress reports
    /// (many run at once).
    Batch,
    /// Stepped by a caller (tests, the RL env): none of the above.
    Embedded,
}

fn is_server(mode: Res<RunMode>) -> bool {
    *mode == RunMode::Server
}

/// Runs that end themselves: paced, and stopped at their time cap or decision.
fn runs_to_exit(mode: Res<RunMode>) -> bool {
    matches!(*mode, RunMode::Server | RunMode::Batch)
}

/// Drive `app` one update (= one tick) at a time until it requests an exit, as
/// the `astrum-sim` binary and the sweep runner do (tests and the RL env step it
/// themselves). The caller keeps the app, so the final world can be inspected.
pub fn run_to_exit(app: &mut App) -> AppExit {
    app.finish();
    app.cleanup();
    loop {
        app.update();
        if let Some(exit) = app.should_exit() {
            return exit;
        }
    }
}

/// Build the headless simulation app from a validated scenario (does not run
/// it). `sim.threads` sets the FDM stepping threads (`0` = all cores), capped
/// at the fleet size.
pub fn build_app(scenario: Scenario, mode: RunMode) -> App {
    let params = sim_params(&scenario);
    let threads = match scenario.sim.threads {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
    .min(scenario.aircraft.len());

    let mut app = App::new();
    // No plugins: physics steps exactly once per update and `pace` throttles wall
    // time. `MinimalPlugins`' task pools, `Time` and frame counter are unused
    // (FDM threads are our own `StepPool`; the clock is `SimClock`) and cost
    // ~1.9 µs of every update (measured), so they are left out; `run_to_exit`
    // is the run loop.
    app.insert_resource(params)
        .insert_resource(world_frame(&scenario))
        .insert_resource(SimClock::default())
        .insert_resource(SimStatus::default())
        .insert_resource(StepPool::new(threads))
        .insert_resource(Diag::default())
        .insert_resource(Pacer::new(params.dt, params.speed))
        .insert_resource(CombatParams::default())
        .insert_resource(CombatRng::seeded(scenario.sim.seed))
        .insert_resource(CombatState { initial_living_teams: team_count(&scenario) })
        .insert_resource(LastPublished::default())
        .insert_resource(RefusalLog::default())
        .insert_resource(scenario)
        .insert_resource(mode)
        .add_systems(
            Startup,
            (spawn_fleet, publish_telemetry.run_if(SimStatus::running), exit_on_fatal).chain(),
        )
        .add_systems(
            Update,
            (
                (
                    ingest_commands,
                    expire_leases,
                    autopilot_step,
                    fire_guns,
                    step_fdm,
                    freeze_dead,
                    advance_clock,
                    step_bullets,
                    resolve_hits,
                    apply_damage,
                    end_on_elimination.run_if(runs_to_exit),
                    publish_telemetry,
                    report_progress.run_if(is_server),
                    stop_at_max_time.run_if(runs_to_exit),
                )
                    .chain()
                    .distributive_run_if(SimStatus::running),
                pace.run_if(runs_to_exit),
                exit_on_signal,
                exit_on_fatal,
            )
                .chain(),
        );
    app
}
