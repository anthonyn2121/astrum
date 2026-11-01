//! Shared builders for the astrum-sim integration tests. No locking: astrum-fdm
//! serialises JSBSim's lifecycle itself, so tests run in parallel.
#![allow(dead_code)] // each test binary uses a different subset

use astrum_proto::{AutopilotTargets, Control, RawActuators, Telemetry};
use astrum_sim::env::Action;
use astrum_sim::scenario::{AircraftSpec, Scenario, Sim, World};

pub const DT: f64 = 1.0 / 120.0;

pub const ORIGIN: (f64, f64) = (37.6188, -122.3756);

pub fn spec(
    id: u32,
    model: &str,
    lat_deg: f64,
    heading_deg: f64,
    alt_hae_m: f64,
    tas_mps: f64,
) -> AircraftSpec {
    AircraftSpec {
        id,
        model: model.into(),
        kind: None,
        autopilot_profile: None,
        lat_deg,
        lon_deg: ORIGIN.1,
        alt_hae_m,
        heading_deg,
        tas_mps,
        trim: None,
        lockstep: false,
        team: 0,
        health: 100.0,
        ammo: None,
    }
}

pub fn scenario(aircraft: Vec<AircraftSpec>) -> Scenario {
    Scenario {
        world: World { origin_lat_deg: ORIGIN.0, origin_lon_deg: ORIGIN.1, origin_alt_hae_m: 0.0 },
        sim: Sim {
            rate_hz: 120.0,
            max_time_s: 0.0,
            realtime: false,
            seed: 0,
            threads: 1,
            command_lease_s: 0.5,
            lockstep: false,
            lockstep_timeout_s: 30.0,
        },
        aircraft,
    }
}

/// One c172x eastbound at 1524 m, 55 m/s TAS.
pub fn single_ship() -> Scenario {
    scenario(vec![spec(1, "c172x", ORIGIN.0, 90.0, 1524.0, 55.0)])
}

/// Two c172x from the same point on opposite headings.
pub fn two_ship() -> Scenario {
    scenario(vec![
        spec(1, "c172x", ORIGIN.0, 90.0, 1524.0, 55.0),
        spec(2, "c172x", ORIGIN.0, 270.0, 1524.0, 55.0),
    ])
}

/// One F450 quadrotor hovering at 300 m, facing east.
pub fn single_quad() -> Scenario {
    scenario(vec![spec(1, "F450", ORIGIN.0, 90.0, 300.0, 0.0)])
}

/// Two F-16s on opposing teams, 600 m apart in trail, northbound.
pub fn two_f16_opposing() -> Scenario {
    let mut lead = spec(1, "f16", ORIGIN.0 + 0.0054, 0.0, 4572.0, 180.0);
    lead.team = 1;
    let chaser = spec(2, "f16", ORIGIN.0, 0.0, 4572.0, 200.0);
    scenario(vec![lead, chaser])
}

/// `n` aircraft alternating c172x (odd ids) and F450 (even ids), staggered.
pub fn mixed_fleet(n: u32) -> Scenario {
    let aircraft = (1..=n)
        .map(|id| {
            let row = f64::from(id - 1);
            let quad = id % 2 == 0;
            let mut a = spec(
                id,
                if quad { "F450" } else { "c172x" },
                ORIGIN.0 + row * 0.002,
                f64::from(id * 30 % 360),
                if quad { 300.0 } else { 1524.0 },
                if quad { 0.0 } else { 55.0 },
            );
            a.lon_deg = ORIGIN.1 + row * 0.002;
            a
        })
        .collect();
    scenario(aircraft)
}

pub fn of(obs: &[Telemetry], id: u32) -> Telemetry {
    *obs.iter().find(|t| t.aircraft_id == id).expect("aircraft present in the observation")
}

pub fn raw(aileron: f64, elevator: f64, throttle: f64) -> Action {
    Action {
        control: Control::Raw(RawActuators { aileron, elevator, rudder: 0.0, throttle }),
        trigger: false,
    }
}

pub fn autopilot(heading_rad: f64, alt_hae_m: f64, tas_mps: f64) -> Action {
    Action {
        control: Control::Autopilot(AutopilotTargets { heading_rad, alt_hae_m, tas_mps }),
        trigger: false,
    }
}

pub fn wrap(a: f64) -> f64 {
    astrum_sim::pilot::wrap_pi(a)
}
