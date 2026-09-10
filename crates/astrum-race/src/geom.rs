//! Pure geometry used by the course sampler's clearance tests and by collision
//! detection: segment distances, the oriented box, and overlap tests against it.
//!
//! Rejecting a self-intersecting track means asking two questions repeatedly:
//! *how close does this leg pass to that gate frame?* and *how close do these two
//! legs pass to each other?* Both reduce to a squared distance between segments.
//! Deciding whether the drone hit something means asking whether a piece of gate
//! frame stock overlaps a part of the airframe — a box against a box for its centre
//! section, a box against a sphere for each rotor (see [`crate::airframe`]). All of it
//! is kept App-free here so it can be unit-tested against hand-worked cases
//! (`tests/geom.rs`).
//!
//! The segment–segment and box–box routines follow Ericson, *Real-Time Collision
//! Detection* §5.1.9 and §4.4.1. The distance work is in squared distance: no square
//! roots on the sampler's hot path, and callers compare against a squared threshold.

use bevy::math::DVec3;

/// Tolerance added to the box–box rotation magnitudes, guarding the near-parallel
/// case where an edge–edge cross product degenerates to (numerically) nothing and
/// its normalised axis would be meaningless. Ericson's fudge factor: it widens both
/// boxes by a hair, so the test errs towards reporting contact.
const OBB_EPS: f64 = 1e-12;

/// Squared distance from point `c` to the segment `[a, b]`.
pub fn segment_point_dist_sq(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= f64::EPSILON {
        return (c - a).length_squared(); // degenerate segment → point distance
    }
    let t = ((c - a).dot(ab) / len2).clamp(0.0, 1.0);
    (c - (a + t * ab)).length_squared()
}

/// Squared distance between the segments `[p1, q1]` and `[p2, q2]`.
///
/// Minimises `‖(p1 + s·d1) − (p2 + t·d2)‖²` over `s, t ∈ [0, 1]`, clamping each
/// parameter to its segment and re-solving for the other so the result is the true
/// closest approach even when the infinite lines meet outside both segments.
/// Parallel segments fall back to the `s = 0` edge, then the clamped `t` recovery.
pub fn segment_segment_dist_sq(p1: DVec3, q1: DVec3, p2: DVec3, q2: DVec3) -> f64 {
    let d1 = q1 - p1; // direction of segment 1
    let d2 = q2 - p2; // direction of segment 2
    let r = p1 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);

    // Either or both segments degenerate to a point.
    if a <= f64::EPSILON && e <= f64::EPSILON {
        return r.length_squared();
    }
    if a <= f64::EPSILON {
        return segment_point_dist_sq(p2, q2, p1);
    }
    if e <= f64::EPSILON {
        return segment_point_dist_sq(p1, q1, p2);
    }

    let c = d1.dot(r);
    let b = d1.dot(d2);
    let denom = a * e - b * b; // ≥ 0; zero iff the segments are parallel

    // s along segment 1: the line-line solution when it exists, else pin to p1 and
    // let the t recovery below find the closest point.
    let s = if denom > f64::EPSILON { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };

    // t for that s, clamped; if the clamp moved t, re-solve s for the new t.
    let t_unclamped = (b * s + f) / e;
    let (s, t) = if t_unclamped < 0.0 {
        ((-c / a).clamp(0.0, 1.0), 0.0)
    } else if t_unclamped > 1.0 {
        (((b - c) / a).clamp(0.0, 1.0), 1.0)
    } else {
        (s, t_unclamped)
    };

    ((p1 + s * d1) - (p2 + t * d2)).length_squared()
}

/// A box with an arbitrary orientation — the shape both a piece of gate frame stock
/// and the F450's airframe reduce to.
///
/// Handedness is irrelevant to everything here: only dot products and *absolute*
/// projections are taken, and a change of orthonormal basis preserves dot products
/// whichever way it turns. That is what lets a hull be re-expressed in a gate's
/// left-handed local triple (see [`crate::gate`]) and tested there directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obb {
    /// Centre of the box.
    pub center: DVec3,
    /// The box's own axes: orthonormal, in the same frame as `center`.
    pub axes: [DVec3; 3],
    /// Half-extent along each of `axes`, in order.
    pub half_extents: DVec3,
}

impl Obb {
    /// A box aligned with the coordinate axes.
    pub fn aabb(center: DVec3, half_extents: DVec3) -> Self {
        Self { center, axes: [DVec3::X, DVec3::Y, DVec3::Z], half_extents }
    }

    /// Half-width of the box's shadow on `dir` — its support radius. `dir` need not
    /// be a unit vector; the result scales with its length.
    pub fn projection_radius(&self, dir: DVec3) -> f64 {
        self.axes[0].dot(dir).abs() * self.half_extents.x
            + self.axes[1].dot(dir).abs() * self.half_extents.y
            + self.axes[2].dot(dir).abs() * self.half_extents.z
    }

    /// The lowest the box reaches on the frame's third axis (world *up*, for a hull
    /// in ENU).
    pub fn min_z(&self) -> f64 {
        self.center.z - self.projection_radius(DVec3::Z)
    }

    /// Re-express the box in another orthonormal frame: `origin` becomes the new
    /// zero and `basis` becomes the new coordinate axes. Shape is untouched — only
    /// the numbers describing it change.
    pub fn to_basis(&self, origin: DVec3, basis: [DVec3; 3]) -> Self {
        let project = |v: DVec3| DVec3::new(v.dot(basis[0]), v.dot(basis[1]), v.dot(basis[2]));
        Self {
            center: project(self.center - origin),
            axes: [project(self.axes[0]), project(self.axes[1]), project(self.axes[2])],
            half_extents: self.half_extents,
        }
    }
}

/// Does a sphere overlap an oriented box?
///
/// Exact, and cheaper than the box–box test: measure the centre's offset in the box's
/// own axes, clamp each component to the box, and compare what is left against the
/// radius. That leftover *is* the vector from the closest point on the box to the
/// centre, so this is the true point-to-box distance. Touching counts as overlap.
pub fn sphere_overlaps_obb(center: DVec3, radius: f64, b: &Obb) -> bool {
    let d = center - b.center;
    let local = DVec3::new(d.dot(b.axes[0]), d.dot(b.axes[1]), d.dot(b.axes[2]));
    let outside = (local.abs() - b.half_extents).max(DVec3::ZERO);
    outside.length_squared() <= radius * radius
}

/// Do two oriented boxes overlap? Separating-axis test, Ericson §4.4.1.
///
/// Two convex bodies are disjoint iff some axis exists on which their shadows do
/// not; for boxes it suffices to try 15 candidates — each box's three face normals,
/// and the nine cross products of their edge directions. Both boxes must be given in
/// the *same* frame.
///
/// Exact for boxes, so the only tolerance is [`OBB_EPS`], which biases the answer
/// towards contact. Touching faces count as overlap.
pub fn obb_overlap(a: &Obb, b: &Obb) -> bool {
    // r[i][j] projects b's axis j onto a's axis i; abs_r is what the radius sums use.
    let mut r = [[0.0f64; 3]; 3];
    let mut abs_r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a.axes[i].dot(b.axes[j]);
            abs_r[i][j] = r[i][j].abs() + OBB_EPS;
        }
    }
    // The centre offset, in a's frame.
    let d = b.center - a.center;
    let t = [d.dot(a.axes[0]), d.dot(a.axes[1]), d.dot(a.axes[2])];
    let ae = [a.half_extents.x, a.half_extents.y, a.half_extents.z];
    let be = [b.half_extents.x, b.half_extents.y, b.half_extents.z];

    // a's three face normals.
    for i in 0..3 {
        let rb = be[0] * abs_r[i][0] + be[1] * abs_r[i][1] + be[2] * abs_r[i][2];
        if t[i].abs() > ae[i] + rb {
            return false;
        }
    }
    // b's three face normals.
    for j in 0..3 {
        let ra = ae[0] * abs_r[0][j] + ae[1] * abs_r[1][j] + ae[2] * abs_r[2][j];
        let tj = t[0] * r[0][j] + t[1] * r[1][j] + t[2] * r[2][j];
        if tj.abs() > ra + be[j] {
            return false;
        }
    }

    // The nine edge-pair cross products, a[i] x b[j]. Each is written out rather
    // than formed as a vector: in a's frame the cross product's components *are*
    // entries of r, so the projections collapse to these products.
    for i in 0..3 {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        for j in 0..3 {
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let ra = ae[i1] * abs_r[i2][j] + ae[i2] * abs_r[i1][j];
            let rb = be[j1] * abs_r[i][j2] + be[j2] * abs_r[i][j1];
            let tij = t[i2] * r[i1][j] - t[i1] * r[i2][j];
            if tij.abs() > ra + rb {
                return false;
            }
        }
    }
    true
}
