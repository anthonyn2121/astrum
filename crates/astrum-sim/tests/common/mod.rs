//! Shared fixtures + ECS query helpers for the `astrum-sim` integration tests.
//!
//! Every test builds a real app with [`astrum_sim::build_app`] and advances it with
//! [`bevy::app::App::update`] — i.e. it drives the same `Startup` + `Update`
//! schedule the binary's main loop runs, just without opening a real session (or
//! over an in-process Zenoh session). These helpers construct scenarios and read
//! entity state back out of the world.
#![allow(dead_code)] // each test binary uses only a subset of these helpers

use astrum_fdm::prop;
use astrum_sim::components::{Aircraft, FdmInstance, GeoState, VehicleKind};
use astrum_sim::net::AutopilotState;
use astrum_sim::scenario::{AircraftSpec, Scenario, Sim, World};
use bevy::prelude::*;
use std::sync::{Mutex, MutexGuard};

/// Serialises JSBSim FDM construction across a test binary's parallel `#[test]`s.
/// JSBSim has process-global init state, so constructing two `FGFDMExec`s from two
/// threads at once races and segfaults (and stepping/dropping distinct instances
/// concurrently is likewise never done in the single-threaded sim). Cargo runs a
/// binary's tests in parallel, and every astrum-sim test builds a fleet (→ FDMs in
/// `spawn_fleet`), so each takes this guard as its first line and holds it for its
/// whole duration — keeping a plain `cargo test --workspace` reliable without
/// `--test-threads=1`. Each test binary links its own copy of `common`, so this
/// static serialises that binary's tests. See astrum-fdm/tests/fdm.rs for the twin.
static JSBSIM_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the process-wide JSBSim guard. Bind it (`let _fdm = fdm_guard();`) so it
/// lives for the whole test — the returned `MutexGuard` is itself `#[must_use]`, so
/// dropping it on the spot would defeat the point. Poison-tolerant so one failing
/// test can't wedge the rest.
pub fn fdm_guard() -> MutexGuard<'static, ()> {
    JSBSIM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// One aircraft heading east (90°), fastest mode (no pacing sleep), no time limit.
pub fn single_ship() -> Scenario {
    Scenario {
        world: World { origin_lat_deg: 37.6188, origin_lon_deg: -122.3756, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 0 },
        aircraft: vec![AircraftSpec {
            id: 1,
            model: "c172x".into(),
            kind: None,
            lat_deg: 37.6188,
            lon_deg: -122.3756,
            alt_ft: 5000.0,
            heading_deg: 90.0,
            airspeed_kts: 100.0,
            team: 0,
            health: 100.0,
            ammo: None,
        }],
    }
}

/// One F450 quadrotor hovering (≈0 airspeed) at 1000 ft, heading east, fastest
/// mode, no time limit. `kind` is left to inference (F450 → multirotor).
pub fn single_quad() -> Scenario {
    Scenario {
        world: World { origin_lat_deg: 44.7258, origin_lon_deg: -93.0759, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 0 },
        aircraft: vec![AircraftSpec {
            id: 1,
            model: "F450".into(),
            kind: None,
            lat_deg: 44.7258,
            lon_deg: -93.0759,
            alt_ft: 1000.0,
            heading_deg: 90.0,
            airspeed_kts: 0.0,
            team: 0,
            health: 100.0,
            ammo: None,
        }],
    }
}

/// Two aircraft from the same point on opposite headings (90° / 270°), fastest
/// mode, no time limit — the harness drives steps manually.
pub fn two_ship() -> Scenario {
    Scenario {
        world: World { origin_lat_deg: 37.6188, origin_lon_deg: -122.3756, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 0 },
        aircraft: vec![
            AircraftSpec {
                id: 1,
                model: "c172x".into(),
                kind: None,
                lat_deg: 37.6188,
                lon_deg: -122.3756,
                alt_ft: 5000.0,
                heading_deg: 90.0,
                airspeed_kts: 100.0,
                team: 0,
                health: 100.0,
                ammo: None,
            },
            AircraftSpec {
                id: 2,
                model: "c172x".into(),
                kind: None,
                lat_deg: 37.6188,
                lon_deg: -122.3756,
                alt_ft: 5000.0,
                heading_deg: 270.0,
                airspeed_kts: 100.0,
                team: 0,
                health: 100.0,
                ammo: None,
            },
        ],
    }
}

/// A mixed fleet of `n` aircraft, alternating fixed-wing (`c172x`, odd ids) and
/// quadrotor (`F450`, even ids). They're staggered across the grid so no two
/// share a start position: fixed-wing craft cruise at 5000 ft / 100 kts on fanned
/// headings, quads hover (0 kts) at 1000 ft. `kind` is left to inference. Fastest
/// mode, no time limit — the scale test drives steps manually.
pub fn mixed_fleet(n: u32) -> Scenario {
    let (base_lat, base_lon) = (44.7258, -93.0759);
    let aircraft = (1..=n)
        .map(|id| {
            let quad = id % 2 == 0;
            let row = (id - 1) as f64; // stagger start positions along the grid
            AircraftSpec {
                id,
                model: if quad { "F450" } else { "c172x" }.into(),
                kind: None,
                lat_deg: base_lat + row * 0.002,
                lon_deg: base_lon + row * 0.002,
                alt_ft: if quad { 1000.0 } else { 5000.0 },
                heading_deg: (id as f64 * 30.0) % 360.0,
                airspeed_kts: if quad { 0.0 } else { 100.0 },
                team: 0,
                health: 100.0,
                ammo: None,
            }
        })
        .collect();
    Scenario {
        world: World { origin_lat_deg: base_lat, origin_lon_deg: base_lon, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 0 },
        aircraft,
    }
}

/// `(id, ENU translation, sim time)` for every aircraft, sorted by id.
pub fn snapshot(app: &mut App) -> Vec<(u32, Vec3, f64)> {
    let mut state = app.world_mut().query::<(&Aircraft, &Transform, &GeoState)>();
    let world = app.world();
    let mut rows: Vec<(u32, Vec3, f64)> =
        state.iter(world).map(|(a, t, g)| (a.id, t.translation, g.sim_time_s)).collect();
    rows.sort_by_key(|r| r.0);
    rows
}

/// Force an aircraft's autopilot on with the given targets (as if an autopilot
/// command had arrived), so a test can exercise the inner loop without a session.
pub fn engage_ap(app: &mut App, id: u32, heading_rad: f64, alt_msl_m: f64, airspeed_mps: f64) {
    let mut q = app.world_mut().query::<(&Aircraft, &mut AutopilotState)>();
    let world = app.world_mut();
    for (a, mut ap) in q.iter_mut(world) {
        if a.id == id {
            *ap = AutopilotState { engaged: true, heading_rad, alt_msl_m, airspeed_mps };
        }
    }
}

/// An aircraft's current true heading (ψ, rad).
pub fn heading_of(app: &mut App, id: u32) -> f64 {
    let mut q = app.world_mut().query::<(&Aircraft, &GeoState)>();
    let world = app.world();
    q.iter(world).find(|(a, _)| a.id == id).map(|(_, g)| g.psi_rad).unwrap()
}

/// An aircraft's current altitude (m MSL).
pub fn alt_of(app: &mut App, id: u32) -> f64 {
    let mut q = app.world_mut().query::<(&Aircraft, &GeoState)>();
    let world = app.world();
    q.iter(world).find(|(a, _)| a.id == id).map(|(_, g)| g.alt_m).unwrap()
}

/// An aircraft's current roll/pitch `(φ, θ)` in radians.
pub fn attitude_of(app: &mut App, id: u32) -> (f64, f64) {
    let mut q = app.world_mut().query::<(&Aircraft, &GeoState)>();
    let world = app.world();
    q.iter(world).find(|(a, _)| a.id == id).map(|(_, g)| (g.phi_rad, g.theta_rad)).unwrap()
}

/// An aircraft's resolved [`VehicleKind`].
pub fn kind_of(app: &mut App, id: u32) -> VehicleKind {
    let mut q = app.world_mut().query::<(&Aircraft, &VehicleKind)>();
    let world = app.world();
    *q.iter(world).find(|(a, _)| a.id == id).map(|(_, k)| k).unwrap()
}

/// Read an aircraft's `fcs/aileron-cmd-norm` and its autopilot state.
pub fn aircraft_probe(app: &mut App, id: u32) -> (f64, AutopilotState) {
    let mut q = app.world_mut().query::<(&Aircraft, &FdmInstance, &AutopilotState)>();
    let world = app.world();
    q.iter(world)
        .find(|(a, _, _)| a.id == id)
        .map(|(_, inst, ap)| (inst.0.get(prop::AILERON_CMD), *ap))
        .expect("aircraft present")
}
