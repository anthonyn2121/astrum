//! Shared fixtures + ECS query helpers for the `astrum-race` integration tests.
//!
//! Tests build a real app with [`astrum_race::build_app_with_course`] and advance it
//! with [`bevy::app::App::update`] — the same `Startup` + `Update` schedule the
//! binary's main loop runs.
#![allow(dead_code)] // each test binary uses only a subset of these helpers

use astrum_race::components::{Drone, DroneState, RaceProgress};
use astrum_race::control::{FlightCommand, RawOverride};
use astrum_race::course::CourseParams;
use astrum_race::scenario::{DroneSpec, RaceScenario, Sim, World};
use bevy::prelude::*;
use std::sync::{Mutex, MutexGuard};

/// Serialises JSBSim FDM construction across a test binary's parallel `#[test]`s.
/// JSBSim has process-global init state, so constructing two `FGFDMExec`s from two
/// threads at once races and segfaults. Cargo runs a binary's tests in parallel, so
/// any test that stands up an app (→ an FDM in `spawn_drone`) takes this guard as
/// its first line and holds it for its whole duration, keeping a plain `cargo test
/// --workspace` reliable without `--test-threads=1`. Each test binary links its own
/// copy of `common`, so this static serialises that binary's tests.
static JSBSIM_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the process-wide JSBSim guard. Bind it (`let _fdm = fdm_guard();`) so it
/// lives for the whole test — the returned `MutexGuard` is `#[must_use]`, so
/// dropping it on the spot would defeat the point. Poison-tolerant so one failing
/// test can't wedge the rest.
pub fn fdm_guard() -> MutexGuard<'static, ()> {
    JSBSIM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A race scenario at UMore Park: one F450, fastest mode (no pacing sleep), no time
/// limit, seed 1, and whatever course envelope is passed in.
pub fn race_scenario(course: CourseParams) -> RaceScenario {
    RaceScenario {
        world: World { origin_lat_deg: 44.7258, origin_lon_deg: -93.0759, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 1 },
        drone: DroneSpec { id: 1, model: "F450".into(), start_behind_m: 10.0, start_alt_m: None },
        course,
    }
}

/// The drone's synced state, if it spawned.
pub fn drone_state(app: &mut App) -> Option<DroneState> {
    let mut q = app.world_mut().query::<(&Drone, &DroneState)>();
    let world = app.world();
    q.iter(world).next().map(|(_, s)| *s)
}

/// The current race progress.
pub fn progress(app: &App) -> RaceProgress {
    *app.world().resource::<RaceProgress>()
}

/// The drone's raw-actuator override state, if it spawned.
pub fn raw_override(app: &mut App) -> Option<RawOverride> {
    let mut q = app.world_mut().query::<(&Drone, &RawOverride)>();
    let world = app.world();
    q.iter(world).next().map(|(_, r)| *r)
}

/// The drone's current high-level flight command, if it spawned.
pub fn flight_command(app: &mut App) -> Option<FlightCommand> {
    let mut q = app.world_mut().query::<(&Drone, &FlightCommand)>();
    let world = app.world();
    q.iter(world).next().map(|(_, c)| *c)
}

/// A flat course at `alt_m`, giving a test as much room below the drone as it
/// needs. The default 1.5–4 m racing band is far too low to watch a drone fall and
/// recover without hitting the ground.
pub fn flat_course_at(alt_m: f64) -> CourseParams {
    CourseParams { alt_min_m: alt_m, alt_max_m: alt_m, ..Default::default() }
}
