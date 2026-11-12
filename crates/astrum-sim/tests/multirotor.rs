//! Multirotor (F450): classified by model, hover-hold autopilot climbs to and
//! holds a commanded altitude and heading while staying level.

mod common;

use astrum_proto::VehicleKind;
use astrum_sim::env::SimEnv;
use common::{autopilot, single_quad, wrap, DT};

#[test]
fn multirotor_autopilot_holds_hover() {
    let mut env = SimEnv::new(single_quad()).unwrap();
    let start = env.observe()[0];
    assert_eq!(start.vehicle_kind, VehicleKind::Multirotor);
    assert_eq!(start.combat.ammo, 0, "multirotors are unarmed");

    let target_alt = start.alt_hae_m + 30.0;
    let target_hdg = 180f64.to_radians();
    let mut t = start;
    for _ in 0..(30.0 / DT) as usize {
        t = env.step(&[(1, autopilot(target_hdg, target_alt, 0.0))]).unwrap().0[0];
    }
    assert!(
        (t.alt_hae_m - target_alt).abs() < 1.0,
        "holds {target_alt:.0} m, got {:.2}",
        t.alt_hae_m
    );
    assert!(t.attitude.phi_rad.abs() < 0.05 && t.attitude.theta_rad.abs() < 0.05, "stays level");
    assert!(wrap(target_hdg - t.attitude.psi_rad).abs() < 0.05, "yaws to and holds 180°");
}
