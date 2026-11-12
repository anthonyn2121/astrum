//! Ballistic fire control: hits on the boresight, misses off it, lead for a
//! crossing target, and range limits — all with the sim's own round model.

use astrum_fdm::FdmState;
use astrum_geo::{Enu, Geodetic, LocalFrame};
use astrum_sim::combat::{gun_solution, CombatParams, Gun};
use astrum_sim::components::AircraftState;

const ORIGIN: (f64, f64) = (37.6188, -122.3756);
const DT: f64 = 1.0 / 120.0;

fn frame() -> LocalFrame {
    LocalFrame::new(Geodetic::from_degrees(ORIGIN.0, ORIGIN.1, 3000.0))
}

fn northbound(n: f64, e: f64, vel_e: f64, vel_n: f64) -> AircraftState {
    AircraftState {
        fdm: FdmState {
            lat_deg: ORIGIN.0,
            lon_deg: ORIGIN.1,
            alt_hae_m: 3000.0,
            ..Default::default()
        },
        pos_enu: Enu { e, n, u: 0.0 },
        prev_pos_enu: Enu { e, n, u: 0.0 },
        vel_enu: Enu { e: vel_e, n: vel_n, u: 0.0 },
    }
}

fn solve(target: AircraftState) -> astrum_sim::combat::GunSolution {
    let shooter = northbound(0.0, 0.0, 0.0, 200.0);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    gun_solution(&frame(), &shooter, &gun, &CombatParams::default(), &target, DT)
}

#[test]
fn target_in_trail_on_the_boresight_is_a_hit() {
    let s = solve(northbound(400.0, 0.0, 0.0, 200.0));
    assert!(s.hit && s.in_range, "{s:?}");
    // ~0.4 s of flight drops the round under a metre, a few mrad of aim correction.
    assert!(s.miss_m < 2.0 && s.aim_error_rad < 0.005, "{s:?}");
    assert!((s.tof_s - 400.0 / 1030.0).abs() < 0.05, "tof {}", s.tof_s);
}

#[test]
fn target_off_the_nose_is_a_miss_with_aim_toward_it() {
    let s = solve(northbound(400.0, 60.0, 0.0, 200.0));
    assert!(!s.hit && s.in_range, "{s:?}");
    assert!(s.aim_dir.x > 0.1, "aim swings east toward the target, {:?}", s.aim_dir);
    assert!((s.aim_error_rad - (60f64 / 400.0).atan()).abs() < 0.01, "{}", s.aim_error_rad);
}

#[test]
fn crossing_target_needs_lead_ahead_of_its_motion() {
    // Target 500 m ahead crossing eastbound at 200 m/s: the aim point leads it.
    let s = solve(northbound(500.0, 0.0, 200.0, 0.0));
    assert!(!s.hit, "the boresight shot falls behind a crossing target");
    let bearing_e = 0.0;
    assert!(s.aim_dir.x > bearing_e + 0.05, "lead toward the target's motion, {:?}", s.aim_dir);
    assert!(s.in_range);
}

#[test]
fn distant_target_is_out_of_range() {
    let s = solve(northbound(6000.0, 0.0, 0.0, 0.0));
    assert!(!s.in_range && !s.hit, "{s:?}");
}
