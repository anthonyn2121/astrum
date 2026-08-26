//! Small frame-geometry helpers shared by the combat systems, kept pure and
//! App-free so the body→world attitude convention (the single biggest
//! correctness risk in placing a muzzle and boresight) can be unit-tested in
//! isolation (`tests/combat_geometry.rs`).

use super::ballistics::DVec3;

/// Rotate a body-axes vector — nose `+X`, starboard `+Y`, down `+Z` — into the
/// ENU world frame, given the aircraft's aerospace 3-2-1 Euler angles (roll `φ`,
/// pitch `θ`, yaw `ψ`; `ψ=0` faces North, `+ψ` yaws East, `+θ` pitches the nose
/// up, `+φ` drops the right wing).
///
/// This is the exact body→NED direction-cosine matrix the replay renderer uses
/// (`web/js/meshes.js::attitudeToQuaternion`), re-expressed straight into ENU via
/// `(East, North, Up) = (E, N, −D)`. Keeping the two in lockstep means a round
/// leaves the muzzle along the same nose vector the renderer draws.
pub fn body_to_enu(phi: f64, theta: f64, psi: f64, v_body: DVec3) -> DVec3 {
    let (sph, cph) = phi.sin_cos();
    let (sth, cth) = theta.sin_cos();
    let (sps, cps) = psi.sin_cos();

    // Each body-axis unit vector expressed in NED = (North, East, Down).
    let nose = DVec3::new(cth * cps, cth * sps, -sth);
    let starboard = DVec3::new(sph * sth * cps - cph * sps, sph * sth * sps + cph * cps, sph * cth);
    let down = DVec3::new(cph * sth * cps + sph * sps, cph * sth * sps - sph * cps, cph * cth);

    // v in NED (packed as x=N, y=E, z=D), then NED→ENU.
    let ned = v_body.x * nose + v_body.y * starboard + v_body.z * down;
    DVec3::new(ned.y, ned.x, -ned.z)
}

/// The aircraft boresight (nose, `+X` body) as a unit vector in ENU.
pub fn boresight_enu(phi: f64, theta: f64, psi: f64) -> DVec3 {
    body_to_enu(phi, theta, psi, DVec3::X)
}

/// Squared distance from point `c` to the segment `[a, b]`. Used for the swept
/// bullet-vs-target hit test: comparing this against the target radius² over the
/// whole per-tick segment stops a fast round tunnelling clean through a target
/// between two integration steps.
pub fn segment_point_dist_sq(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= f64::EPSILON {
        return (c - a).length_squared(); // degenerate segment → point distance
    }
    let t = ((c - a).dot(ab) / len2).clamp(0.0, 1.0);
    let proj = a + t * ab;
    (c - proj).length_squared()
}
