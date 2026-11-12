//! Frame and hit-test geometry: the body→NED attitude convention (ψ=0 North,
//! +ψ East, +θ nose up, +φ right wing down), the local-level tilt away from the
//! grid origin, and the relative-motion contact solver.

use astrum_geo::{Enu, Geodetic, LocalFrame, Ned};
use astrum_sim::combat::ballistics::DVec3;
use astrum_sim::combat::geometry::{
    body_to_grid, body_to_ned, closest_approach, first_contact, grid_to_body, ned_to_body,
};

fn ned(v: Ned) -> DVec3 {
    DVec3::new(v.n, v.e, v.d)
}

fn close(a: DVec3, b: DVec3) -> bool {
    (a - b).length() < 1e-9
}

#[test]
fn nose_and_wing_point_the_right_way() {
    assert!(close(ned(body_to_ned(0.0, 0.0, 0.0, DVec3::X)), DVec3::X), "ψ=0 faces North");
    assert!(
        close(ned(body_to_ned(0.0, 0.0, 90f64.to_radians(), DVec3::X)), DVec3::Y),
        "+ψ faces East"
    );
    let climb = ned(body_to_ned(0.0, 30f64.to_radians(), 0.0, DVec3::X));
    assert!((climb.z + 0.5).abs() < 1e-12, "+θ raises the nose (−Down)");
    let wing = ned(body_to_ned(90f64.to_radians(), 0.0, 0.0, DVec3::Y));
    assert!(close(wing, DVec3::Z), "+φ drops the right wing (+Down)");
}

#[test]
fn combined_roll_and_pitch_compose_in_3_2_1_order() {
    // Banked 60° right, pitched 20° up, heading 45°: the nose is still the
    // pitch-only nose (roll is applied last), the wing is tilted by the roll.
    let (phi, theta, psi) = (60f64.to_radians(), 20f64.to_radians(), 45f64.to_radians());
    let nose = ned(body_to_ned(phi, theta, psi, DVec3::X));
    let nose_no_roll = ned(body_to_ned(0.0, theta, psi, DVec3::X));
    assert!(close(nose, nose_no_roll));
    let wing = ned(body_to_ned(phi, theta, psi, DVec3::Y));
    assert!((wing.z - phi.sin() * theta.cos()).abs() < 1e-12, "wing dips by sinφ·cosθ");
    assert!(wing.dot(nose).abs() < 1e-12, "axes stay orthogonal");
    // The inverse rotation undoes it.
    let v = DVec3::new(1.0, -2.0, 3.0);
    let back = ned_to_body(phi, theta, psi, body_to_ned(phi, theta, psi, v));
    assert!(close(back, v));
}

#[test]
fn body_to_grid_accounts_for_the_local_level_tilt() {
    let frame = LocalFrame::new(Geodetic::from_degrees(40.0, -105.0, 0.0));
    let origin = frame.origin();
    // At the origin, level-east nose is grid east.
    let at_origin = body_to_grid(&frame, &origin, 0.0, 0.0, 90f64.to_radians(), DVec3::X);
    assert!(close(at_origin, DVec3::X));
    // 50 km east, a level eastbound nose points ~0.45° below the grid horizon.
    let far = frame.enu_to_geodetic(&Enu { e: 50_000.0, n: 0.0, u: 0.0 });
    let tilted = body_to_grid(&frame, &far, 0.0, 0.0, 90f64.to_radians(), DVec3::X);
    assert!(tilted.z < -0.007 && tilted.z > -0.009, "tilt {}", tilted.z);
    let back = grid_to_body(&frame, &far, 0.0, 0.0, 90f64.to_radians(), tilted);
    assert!(close(back, DVec3::X));
}

#[test]
fn first_contact_solves_relative_motion() {
    let r = 8.0;
    let z = DVec3::ZERO;
    // Already inside at the start of the step.
    assert_eq!(
        first_contact(z, DVec3::X, DVec3::new(3.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0), r),
        Some(0.0)
    );
    // Head-on closure: contact when the gap first shrinks to r.
    let s = first_contact(
        z,
        DVec3::new(10.0, 0.0, 0.0),
        DVec3::new(20.0, 0.0, 0.0),
        DVec3::new(18.0, 0.0, 0.0),
        r,
    )
    .unwrap();
    assert!((s - 1.0).abs() < 1e-12, "gap 20 → 8 after 12 m of closure, s={s}");
    // Moving together in parallel never closes.
    let v = DVec3::new(10.0, 0.0, 0.0);
    assert_eq!(
        first_contact(z, v, DVec3::new(0.0, 20.0, 0.0), DVec3::new(0.0, 20.0, 0.0) + v, r),
        None
    );
    // Passing outside the radius misses; the closest approach reports where.
    let (s, d) = closest_approach(
        z,
        DVec3::new(10.0, 0.0, 0.0),
        DVec3::new(5.0, 9.0, 0.0),
        DVec3::new(5.0, 9.0, 0.0),
    );
    assert!((s - 0.5).abs() < 1e-12 && (d - 9.0).abs() < 1e-12);
    assert_eq!(
        first_contact(
            z,
            DVec3::new(10.0, 0.0, 0.0),
            DVec3::new(5.0, 9.0, 0.0),
            DVec3::new(5.0, 9.0, 0.0),
            r
        ),
        None
    );
}
