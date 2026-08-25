//! Physics-side integration tests. Each builds the real headless app and pumps
//! `App::update()` — the same schedule the binary's main loop runs — to verify
//! the fleet steps deterministically on the ENU grid and the built-in autopilot
//! tracks a commanded heading + altitude.

mod common;

use astrum_sim::build_app;
use common::{alt_of, engage_ap, fdm_guard, heading_of, single_ship, snapshot, two_ship};

#[test]
fn fleet_steps_and_positions_evolve_on_the_grid() {
    let _fdm = fdm_guard();
    let mut app = build_app(two_ship());

    // First update runs Startup (spawn) + one step.
    for _ in 0..12 {
        app.update();
    }
    let early = snapshot(&mut app);
    assert_eq!(early.len(), 2, "both aircraft should spawn");

    for _ in 0..228 {
        app.update();
    }
    let late = snapshot(&mut app);

    // Sim advanced ~2 s of deterministic 120 Hz steps.
    assert!((late[0].2 - 2.0).abs() < 0.05, "sim time ~2s, got {}", late[0].2);

    // Each aircraft's ENU position moved appreciably (aircraft are flying).
    for i in 0..2 {
        let moved = (late[i].1 - early[i].1).length();
        assert!(moved > 10.0, "aircraft {} should move on the grid, moved {moved} m", late[i].0);
        assert!(late[i].1.is_finite(), "aircraft {} position must be finite", late[i].0);
    }

    // Opposite headings from the same start -> independent FDMs -> distinct
    // positions, proving per-entity isolation carries through the ECS.
    let sep = (late[0].1 - late[1].1).length();
    assert!(sep > 10.0, "opposite-heading aircraft should separate, sep {sep} m");
}

#[test]
fn autopilot_tracks_commanded_heading() {
    use std::f64::consts::PI;
    let wrap = |a: f64| (a + PI).rem_euclid(2.0 * PI) - PI;

    let _fdm = fdm_guard();
    let mut app = build_app(single_ship());
    app.update(); // spawn
    engage_ap(&mut app, 1, 0.0, 1500.0, 50.0); // target due north, 1500 m

    let start_err = wrap(0.0 - heading_of(&mut app, 1)).abs(); // ~π/2 (90° off)
    for _ in 0..3600 {
        app.update();
    } // 30 s @ 120 Hz — enough for a ~90° bank-limited turn to settle
    let psi = heading_of(&mut app, 1);
    let end_err = wrap(0.0 - psi).abs();

    assert!(psi.is_finite(), "heading stayed finite (no divergence)");
    assert!(
        end_err < start_err * 0.25,
        "AP should turn toward target: start_err={start_err:.3} end_err={end_err:.3}"
    );
    // Proportional loop settles to a small steady-state heading error (~4°).
    assert!(end_err < 0.15, "AP should settle near target heading, err={end_err:.3} rad");

    // The altitude-hold loop should keep it near the commanded 1500 m.
    let alt = alt_of(&mut app, 1);
    assert!((alt - 1500.0).abs() < 100.0, "AP should hold ~1500 m, got {alt:.0} m");
}
