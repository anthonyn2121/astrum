//! Frame geometry for the combat systems, kept pure so the attitude convention
//! (the single biggest correctness risk in placing a muzzle and boresight) and
//! the hit test can be unit-tested in isolation (`tests/combat_geometry.rs`).

use astrum_geo::{Enu, Geodetic, LocalFrame, Ned};

use super::ballistics::DVec3;

/// Body-axes unit vectors (nose `+X`, starboard `+Y`, down `+Z`) expressed in
/// the local-level NED frame, for aerospace 3-2-1 Euler angles (`ψ=0` faces
/// North, `+ψ` yaws East, `+θ` pitches the nose up, `+φ` drops the right wing).
/// Returned as `[nose, starboard, down]`, each packed `(x=N, y=E, z=D)`.
fn body_axes_ned(phi: f64, theta: f64, psi: f64) -> [DVec3; 3] {
    let (sph, cph) = phi.sin_cos();
    let (sth, cth) = theta.sin_cos();
    let (sps, cps) = psi.sin_cos();
    [
        DVec3::new(cth * cps, cth * sps, -sth),
        DVec3::new(sph * sth * cps - cph * sps, sph * sth * sps + cph * cps, sph * cth),
        DVec3::new(cph * sth * cps + sph * sps, cph * sth * sps - sph * cps, cph * cth),
    ]
}

/// Rotate a body-axes vector into the local-level NED frame.
pub fn body_to_ned(phi: f64, theta: f64, psi: f64, v: DVec3) -> Ned {
    let [x, y, z] = body_axes_ned(phi, theta, psi);
    let n = v.x * x + v.y * y + v.z * z;
    Ned { n: n.x, e: n.y, d: n.z }
}

/// Rotate a local-level NED vector into body axes.
pub fn ned_to_body(phi: f64, theta: f64, psi: f64, v: Ned) -> DVec3 {
    let [x, y, z] = body_axes_ned(phi, theta, psi);
    let w = DVec3::new(v.n, v.e, v.d);
    DVec3::new(x.dot(w), y.dot(w), z.dot(w))
}

fn enu(v: Enu) -> DVec3 {
    DVec3::new(v.e, v.n, v.u)
}

/// Rotate a body-axes vector of an aircraft at `at` into the world grid. Away
/// from the grid origin, local level is tilted relative to the grid, so this
/// goes body → local NED → grid rather than swapping axes.
pub fn body_to_grid(
    frame: &LocalFrame,
    at: &Geodetic,
    phi: f64,
    theta: f64,
    psi: f64,
    v: DVec3,
) -> DVec3 {
    enu(frame.ned_to_enu(at, &body_to_ned(phi, theta, psi, v)))
}

/// Rotate a world-grid vector into the body axes of an aircraft at `at`.
pub fn grid_to_body(
    frame: &LocalFrame,
    at: &Geodetic,
    phi: f64,
    theta: f64,
    psi: f64,
    v: DVec3,
) -> DVec3 {
    let ned = frame.enu_to_ned(at, &Enu { e: v.x, n: v.y, u: v.z });
    ned_to_body(phi, theta, psi, ned)
}

/// Earliest fraction `s ∈ [0, 1]` of a step at which two points moving
/// linearly (`a0 → a1` and `b0 → b1` over the same interval) come within `r`
/// of each other, or `None`. Working in relative motion is what makes the test
/// correct against a moving target: a round's swept segment is tested against
/// where the target *was* at each instant, not where it ends the step.
pub fn first_contact(a0: DVec3, a1: DVec3, b0: DVec3, b1: DVec3, r: f64) -> Option<f64> {
    let d0 = a0 - b0;
    let v = (a1 - a0) - (b1 - b0);
    let c = d0.length_squared() - r * r;
    if c <= 0.0 {
        return Some(0.0);
    }
    let a = v.length_squared();
    if a <= f64::EPSILON {
        return None;
    }
    let b = 2.0 * d0.dot(v);
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let s = (-b - disc.sqrt()) / (2.0 * a);
    (0.0..=1.0).contains(&s).then_some(s)
}

/// Closest approach of two linearly moving points over one step:
/// `(fraction s ∈ [0,1], distance)`.
pub fn closest_approach(a0: DVec3, a1: DVec3, b0: DVec3, b1: DVec3) -> (f64, f64) {
    let d0 = a0 - b0;
    let v = (a1 - a0) - (b1 - b0);
    let a = v.length_squared();
    let s = if a <= f64::EPSILON { 0.0 } else { (-d0.dot(v) / a).clamp(0.0, 1.0) };
    (s, (d0 + s * v).length())
}
