//! Unit tests for the free-projectile ballistics core
//! (`astrum_sim::combat::ballistics`). Low-level and Bevy-App-free — they call
//! the pure integrator directly and check the three modelled forces in isolation
//! (drop, drag, Coriolis) plus the atmosphere and Earth-rate helpers, mirroring
//! the style of `astrum-fdm/tests/fdm.rs`.
//!
//! Run:  `cargo test -p astrum-sim --test ballistics -- --nocapture`

use astrum_sim::combat::ballistics::{
    acceleration, air_density, earth_rate_enu, step_projectile, Ballistics, DVec3, G0, OMEGA_EARTH,
};

const DT: f64 = 1.0 / 120.0; // the sim's physics step
const STEPS: u32 = 120; // → 1.0 s of flight
const T: f64 = STEPS as f64 * DT;

/// Integrate a round for `STEPS`, returning its final `(pos, vel)`. `base_alt_m`
/// is the ENU-origin altitude, so the density lookup sees `base + pos.z`.
fn fly(pos0: DVec3, vel0: DVec3, base_alt_m: f64, b: &Ballistics, omega: DVec3) -> (DVec3, DVec3) {
    let (mut pos, mut vel) = (pos0, vel0);
    for _ in 0..STEPS {
        (pos, vel) = step_projectile(pos, vel, base_alt_m + pos.z, b, omega, DT);
    }
    (pos, vel)
}

#[test]
fn vacuum_drop_matches_free_fall() {
    // No drag, no Earth rate: pure gravity. Fire horizontally (due East).
    let b = Ballistics { drag_factor: 0.0 };
    let v0 = 300.0;
    let (pos, vel) = fly(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), 1000.0, &b, DVec3::ZERO);

    // Semi-implicit Euler integrates vertical velocity exactly: v_z = −g·t.
    assert!((vel.z + G0 * T).abs() < 1e-9, "v_z should be −g·t, got {}", vel.z);
    // Horizontal velocity is untouched and range is exactly v0·t (v_x constant).
    assert!((vel.x - v0).abs() < 1e-9, "v_x should be unchanged, got {}", vel.x);
    assert!((pos.x - v0 * T).abs() < 1e-6, "range should be v0·t, got {}", pos.x);

    // Drop ≈ ½·g·t². Semi-implicit Euler runs a touch deep (g·t·dt/2 ≈ 4 cm at
    // 1 s); allow a small margin rather than asserting the exact closed form.
    let drop = -pos.z;
    let free_fall = 0.5 * G0 * T * T;
    assert!(
        (drop - free_fall).abs() < 0.06,
        "drop should track ½·g·t² ({free_fall:.3} m), got {drop:.3} m"
    );
}

#[test]
fn drag_bleeds_speed_and_shortens_range() {
    // A ~20 mm-class round fired due East at sea level (max density).
    let b = Ballistics::new(0.3, std::f64::consts::PI * 0.01 * 0.01, 0.10);
    let v0 = 1000.0;

    // Horizontal velocity must decrease every step (drag always opposes it),
    // and end well below the muzzle speed.
    let (mut pos, mut vel) = (DVec3::ZERO, DVec3::new(v0, 0.0, 0.0));
    let mut prev_vx = vel.x;
    for _ in 0..STEPS {
        (pos, vel) = step_projectile(pos, vel, pos.z, &b, DVec3::ZERO, DT);
        assert!(vel.x < prev_vx, "drag must reduce horizontal speed each step");
        prev_vx = vel.x;
    }
    assert!(vel.x < v0 * 0.9, "1 s of drag should bleed real speed, v_x={}", vel.x);
    assert!(vel.x > 0.0, "the round should still be moving forward, v_x={}", vel.x);

    // Compared to a drag-free round it falls short of the vacuum range v0·t.
    let vacuum = Ballistics { drag_factor: 0.0 };
    let (pos_vac, _) = fly(DVec3::ZERO, DVec3::new(v0, 0.0, 0.0), 0.0, &vacuum, DVec3::ZERO);
    assert!(pos.x < pos_vac.x, "drag range {} should be < vacuum range {}", pos.x, pos_vac.x);
    assert!((pos_vac.x - v0 * T).abs() < 1e-6, "sanity: vacuum range should be v0·t");
}

#[test]
fn coriolis_deflects_right_in_northern_hemisphere() {
    // Fire due North with no drag; only gravity + Earth Coriolis act. In the
    // northern hemisphere the round should deflect to its right (East, +X); in
    // the southern hemisphere, to its left (West, −X).
    let b = Ballistics { drag_factor: 0.0 };
    let north = DVec3::new(0.0, 500.0, 0.0);

    let (pos_n, _) = fly(DVec3::ZERO, north, 0.0, &b, earth_rate_enu(45f64.to_radians()));
    assert!(pos_n.x > 0.0, "N-hemisphere northward round deflects East, got e={}", pos_n.x);
    assert!(pos_n.x < 1.0, "Coriolis is a ~cm effect at WVR, got e={} m", pos_n.x);

    let (pos_s, _) = fly(DVec3::ZERO, north, 0.0, &b, earth_rate_enu((-45f64).to_radians()));
    assert!(pos_s.x < 0.0, "S-hemisphere northward round deflects West, got e={}", pos_s.x);
}

#[test]
fn air_density_matches_isa() {
    // Reference ISA densities (kg/m³).
    assert!((air_density(0.0) - 1.225).abs() < 0.005, "SL density, got {}", air_density(0.0));
    assert!((air_density(5_000.0) - 0.7364).abs() < 0.01, "5 km, got {}", air_density(5_000.0));
    assert!((air_density(11_000.0) - 0.3639).abs() < 0.01, "11 km, got {}", air_density(11_000.0));

    // Monotonically thinning with altitude, and clamped below sea level.
    assert!(air_density(0.0) > air_density(5_000.0));
    assert!(air_density(5_000.0) > air_density(11_000.0));
    assert!(air_density(11_000.0) > air_density(15_000.0));
    assert!((air_density(-500.0) - air_density(0.0)).abs() < 1e-12, "clamps below SL");
}

#[test]
fn earth_rate_enu_orientation() {
    // At the equator Earth's axis lies in the local horizontal, pointing North.
    let eq = earth_rate_enu(0.0);
    assert!(eq.x.abs() < 1e-15 && eq.z.abs() < 1e-12, "equator rate should be all-North");
    assert!((eq.y - OMEGA_EARTH).abs() < 1e-12, "equator North component should be Ω");

    // At the pole it points straight Up.
    let pole = earth_rate_enu(90f64.to_radians());
    assert!(pole.y.abs() < 1e-12, "polar rate should have no North component");
    assert!((pole.z - OMEGA_EARTH).abs() < 1e-12, "polar Up component should be Ω");
}

#[test]
fn acceleration_terms_compose() {
    // At rest at sea level the only force is gravity: straight down at g.
    let b = Ballistics::new(0.3, 3.0e-4, 0.10);
    let a = acceleration(DVec3::ZERO, 0.0, &b, earth_rate_enu(45f64.to_radians()));
    assert!(a.x.abs() < 1e-12 && a.y.abs() < 1e-12, "no drag/Coriolis at rest");
    assert!((a.z + G0).abs() < 1e-12, "gravity should be −g, got {}", a.z);
}
