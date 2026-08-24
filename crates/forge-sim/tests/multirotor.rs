//! Multirotor (F450 quadrotor) integration test. Builds the real headless app,
//! confirms the quad is spawned as a [`VehicleKind::Multirotor`] (which engages
//! `fcs/ScasEngage`), then engages the built-in hover-hold autopilot and asserts
//! it climbs to a commanded altitude and holds it while staying near level — the
//! quad-aware analogue of `simulation::autopilot_tracks_commanded_heading`.

mod common;

use common::{alt_of, attitude_of, engage_ap, fdm_guard, heading_of, kind_of, single_quad};
use forge_sim::build_app;
use forge_sim::components::VehicleKind;

#[test]
fn multirotor_autopilot_holds_hover() {
    use std::f64::consts::PI;
    let wrap = |a: f64| (a + PI).rem_euclid(2.0 * PI) - PI;

    let _fdm = fdm_guard();
    let mut app = build_app(single_quad());
    app.update(); // spawn + engage SCAS

    // The F450 was classified (by inference) as a multirotor.
    assert_eq!(kind_of(&mut app, 1), VehicleKind::Multirotor, "F450 should be a multirotor");

    // Command a 30 m climb, hold the initial heading (east), hover in place.
    let start_alt = alt_of(&mut app, 1);
    let start_hdg = heading_of(&mut app, 1);
    let target_alt = start_alt + 30.0;
    engage_ap(&mut app, 1, start_hdg, target_alt, 0.0);

    for _ in 0..3600 {
        app.update();
    } // 30 s @ 120 Hz

    let alt = alt_of(&mut app, 1);
    let (phi, theta) = attitude_of(&mut app, 1);
    let hdg_err = wrap(start_hdg - heading_of(&mut app, 1)).abs();

    eprintln!(
        "[quad] start_alt={start_alt:.1} target={target_alt:.1} final_alt={alt:.1} \
         phi={:.2}° theta={:.2}° hdg_err={:.2}°",
        phi.to_degrees(),
        theta.to_degrees(),
        hdg_err.to_degrees()
    );

    assert!(alt.is_finite() && phi.is_finite() && theta.is_finite(), "state stayed finite");
    // Climbed to and holds the commanded altitude.
    assert!((alt - target_alt).abs() < 5.0, "should hold ~{target_alt:.0} m, got {alt:.1} m");
    // Stayed essentially level (hover), no attitude divergence.
    assert!(
        phi.abs() < 0.09 && theta.abs() < 0.09,
        "should stay near level, φ={phi:.3} θ={theta:.3}"
    );
    // Held heading.
    assert!(hdg_err < 0.15, "should hold heading, err={hdg_err:.3} rad");
}
