//! Unit tests for the combat frame-geometry helpers
//! (`astrum_sim::combat::geometry`). These lock the body→ENU attitude sign
//! convention against the replay renderer's (ψ=0→North, +ψ→East, +θ→nose-up,
//! +φ→right-wing-down) and the swept segment-distance used by the hit test.
//! App-free, mirroring `tests/ballistics.rs`.
//!
//! Run:  `cargo test -p astrum-sim --test combat_geometry -- --nocapture`

use astrum_sim::combat::ballistics::DVec3;
use astrum_sim::combat::geometry::{body_to_enu, boresight_enu, segment_point_dist_sq};

const EPS: f64 = 1e-12;

fn close(a: DVec3, b: DVec3) -> bool {
    (a - b).length() < 1e-9
}

#[test]
fn boresight_points_the_right_way() {
    // Wings-level, ψ=0 → nose due North (+Y).
    assert!(close(boresight_enu(0.0, 0.0, 0.0), DVec3::Y), "ψ=0 should face North");

    // +ψ yaws toward East (+X).
    let east = boresight_enu(0.0, 0.0, 90f64.to_radians());
    assert!(close(east, DVec3::X), "ψ=+90° should face East, got {east:?}");

    // +θ pitches the nose up (+U): facing North, 45° up.
    let climb = boresight_enu(0.0, 45f64.to_radians(), 0.0);
    assert!(climb.z > 0.0, "+θ should tilt the nose up, got u={}", climb.z);
    assert!(climb.y > 0.0, "still generally northbound, got n={}", climb.y);
    assert!((climb.z - 45f64.to_radians().sin()).abs() < 1e-9, "up = sinθ");
}

#[test]
fn body_axes_map_into_enu() {
    // Wings-level facing North: starboard (+Y body) points East (+X world).
    let stbd = body_to_enu(0.0, 0.0, 0.0, DVec3::Y);
    assert!(close(stbd, DVec3::X), "level starboard should point East, got {stbd:?}");

    // Roll +90° right: the right wing (starboard) now points straight Down.
    let rolled = body_to_enu(90f64.to_radians(), 0.0, 0.0, DVec3::Y);
    assert!(close(rolled, DVec3::new(0.0, 0.0, -1.0)), "+φ drops starboard down, got {rolled:?}");

    // The rotation preserves length (it's an orthonormal DCM).
    let v = DVec3::new(1.0, -2.0, 3.0);
    let out = body_to_enu(0.3, -0.4, 1.1, v);
    assert!((out.length() - v.length()).abs() < 1e-9, "body→ENU must preserve length");
}

#[test]
fn muzzle_offset_lies_along_the_boresight() {
    // A purely-forward body offset must resolve to the boresight direction,
    // scaled — the muzzle sits ahead of the aircraft on the nose vector.
    let (phi, theta, psi) = (0.2, -0.15, 2.0);
    let fwd = body_to_enu(phi, theta, psi, DVec3::new(4.0, 0.0, 0.0));
    let bore = boresight_enu(phi, theta, psi);
    assert!(close(fwd, bore * 4.0), "forward offset should be 4·boresight");
}

#[test]
fn segment_distance_matches_geometry() {
    let a = DVec3::new(0.0, 0.0, 0.0);
    let b = DVec3::new(10.0, 0.0, 0.0);

    // Point beside the middle of the segment: perpendicular distance.
    let mid = segment_point_dist_sq(a, b, DVec3::new(5.0, 3.0, 0.0));
    assert!((mid - 9.0).abs() < EPS, "perpendicular dist² should be 9, got {mid}");

    // Point beyond an endpoint clamps to that endpoint, not the infinite line.
    let past = segment_point_dist_sq(a, b, DVec3::new(20.0, 0.0, 0.0));
    assert!((past - 100.0).abs() < EPS, "should clamp to endpoint b, got {past}");

    // A point exactly on the segment has zero distance.
    let on = segment_point_dist_sq(a, b, DVec3::new(4.0, 0.0, 0.0));
    assert!(on < EPS, "on-segment point should be distance 0, got {on}");

    // Degenerate (zero-length) segment falls back to point distance.
    let degen = segment_point_dist_sq(a, a, DVec3::new(3.0, 4.0, 0.0));
    assert!((degen - 25.0).abs() < EPS, "degenerate segment → point dist² 25, got {degen}");
}
