//! Scale: a 40-aircraft mixed fleet (20 c172x + 20 F450) flown by the built-in
//! autopilots, stepped with 1 and with 4 threads. Parallel stepping must give a
//! bit-identical result — the FDM instances are independent.

mod common;

use astrum_proto::{Telemetry, VehicleKind};
use astrum_sim::env::SimEnv;
use common::{autopilot, mixed_fleet, DT};

fn fly(threads: usize) -> Vec<Telemetry> {
    let mut s = mixed_fleet(40);
    s.sim.threads = threads;
    let mut env = SimEnv::new(s).unwrap();
    let start = env.observe();
    let targets: Vec<_> = start
        .iter()
        .map(|t| (t.aircraft_id, autopilot(t.attitude.psi_rad, t.alt_hae_m + 15.0, 55.0)))
        .collect();
    let mut obs = start;
    for _ in 0..(10.0 / DT) as usize {
        obs = env.step(&targets).unwrap().0;
    }
    obs
}

#[test]
fn mixed_fleet_of_40_flies_identically_on_1_and_4_threads() {
    let serial = fly(1);
    assert_eq!(serial.len(), 40);
    for t in &serial {
        assert!(
            t.pos_enu.e.is_finite() && t.alt_hae_m.is_finite(),
            "aircraft {} diverged",
            t.aircraft_id
        );
        if t.vehicle_kind == VehicleKind::Multirotor {
            assert!(
                t.attitude.phi_rad.abs() < 0.1 && t.attitude.theta_rad.abs() < 0.1,
                "quad {} level",
                t.aircraft_id
            );
        } else {
            assert!(t.tas_mps > 45.0, "plane {} keeps flying", t.aircraft_id);
        }
    }
    assert_eq!(fly(4), serial, "thread count must not change the trajectory");
}
