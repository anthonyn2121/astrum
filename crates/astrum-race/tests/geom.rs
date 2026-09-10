//! The geometry primitives, checked against hand-worked cases. The segment distances
//! are the course sampler's only collision test, and the box/sphere overlaps are the
//! crash detector's: an error in the first would silently let a self-intersecting
//! course through, and one in the second would silently let the drone fly through a
//! gate's frame.

use astrum_race::geom::{
    obb_overlap, segment_point_dist_sq, segment_segment_dist_sq, sphere_overlaps_obb, Obb,
};
use astrum_race::rng::RaceRng;
use bevy::math::DVec3;

fn d(sq: f64) -> f64 {
    sq.sqrt()
}

/// A box rotated by yaw then pitch — enough to exercise the non-axis-aligned paths
/// without reproducing the airframe's whole attitude convention here.
fn tilted(center: DVec3, half_extents: DVec3, yaw: f64, pitch: f64) -> Obb {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x = DVec3::new(cy * cp, sy * cp, sp);
    let y = DVec3::new(-sy, cy, 0.0);
    Obb { center, axes: [x, y, x.cross(y)], half_extents }
}

#[test]
fn point_to_segment_uses_the_perpendicular_or_an_endpoint() {
    let a = DVec3::new(0.0, 0.0, 0.0);
    let b = DVec3::new(10.0, 0.0, 0.0);

    // Perpendicular foot inside the segment.
    assert!((d(segment_point_dist_sq(a, b, DVec3::new(5.0, 3.0, 0.0))) - 3.0).abs() < 1e-12);
    // Beyond either end → the endpoint distance, not the infinite line's.
    assert!((d(segment_point_dist_sq(a, b, DVec3::new(-4.0, 3.0, 0.0))) - 5.0).abs() < 1e-12);
    assert!((d(segment_point_dist_sq(a, b, DVec3::new(14.0, 3.0, 0.0))) - 5.0).abs() < 1e-12);
    // On the segment.
    assert!(segment_point_dist_sq(a, b, DVec3::new(2.5, 0.0, 0.0)) < 1e-24);
    // Degenerate segment → point-to-point.
    assert!((d(segment_point_dist_sq(a, a, DVec3::new(0.0, 0.0, 7.0))) - 7.0).abs() < 1e-12);
}

#[test]
fn segment_to_segment_handles_crossing_skew_and_parallel() {
    // Crossing in the same plane → zero.
    let x = segment_segment_dist_sq(
        DVec3::new(-5.0, 0.0, 0.0),
        DVec3::new(5.0, 0.0, 0.0),
        DVec3::new(0.0, -5.0, 0.0),
        DVec3::new(0.0, 5.0, 0.0),
    );
    assert!(x < 1e-18, "crossing segments are 0 apart, got {}", d(x));

    // Skew: the same crossing lifted 4 m → exactly the vertical offset.
    let s = segment_segment_dist_sq(
        DVec3::new(-5.0, 0.0, 0.0),
        DVec3::new(5.0, 0.0, 0.0),
        DVec3::new(0.0, -5.0, 4.0),
        DVec3::new(0.0, 5.0, 4.0),
    );
    assert!((d(s) - 4.0).abs() < 1e-12, "skew distance should be 4, got {}", d(s));

    // Parallel, offset sideways → the offset (the degenerate-denominator branch).
    let p = segment_segment_dist_sq(
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(10.0, 0.0, 0.0),
        DVec3::new(2.0, 3.0, 0.0),
        DVec3::new(8.0, 3.0, 0.0),
    );
    assert!((d(p) - 3.0).abs() < 1e-12, "parallel offset should be 3, got {}", d(p));

    // Collinear but disjoint → the gap between the near endpoints.
    let c = segment_segment_dist_sq(
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(4.0, 0.0, 0.0),
        DVec3::new(9.0, 0.0, 0.0),
        DVec3::new(12.0, 0.0, 0.0),
    );
    assert!((d(c) - 5.0).abs() < 1e-12, "collinear gap should be 5, got {}", d(c));

    // Closest approach outside both segments → endpoint-to-endpoint (3-4-5).
    let e = segment_segment_dist_sq(
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(-10.0, 0.0, 0.0),
        DVec3::new(3.0, 4.0, 0.0),
        DVec3::new(13.0, 4.0, 0.0),
    );
    assert!((d(e) - 5.0).abs() < 1e-12, "endpoint pair should be 5 apart, got {}", d(e));
}

#[test]
fn segment_to_segment_is_symmetric_and_handles_points() {
    let (p1, q1) = (DVec3::new(1.0, 2.0, 3.0), DVec3::new(-4.0, 6.0, 1.0));
    let (p2, q2) = (DVec3::new(0.0, -3.0, 5.0), DVec3::new(7.0, 2.0, -2.0));
    let ab = segment_segment_dist_sq(p1, q1, p2, q2);
    let ba = segment_segment_dist_sq(p2, q2, p1, q1);
    assert!((ab - ba).abs() < 1e-9, "distance must not depend on argument order");

    // Either segment collapsed to a point → the point-to-segment answer.
    let pt = DVec3::new(2.0, 2.0, 2.0);
    assert!(
        (segment_segment_dist_sq(pt, pt, p2, q2) - segment_point_dist_sq(p2, q2, pt)).abs() < 1e-12
    );
    assert!(
        (segment_segment_dist_sq(p1, q1, pt, pt) - segment_point_dist_sq(p1, q1, pt)).abs() < 1e-12
    );
    // Both collapsed → plain point distance.
    assert!((segment_segment_dist_sq(pt, pt, p1, p1) - (p1 - pt).length_squared()).abs() < 1e-12);
}

// --- The oriented box, and overlaps against it -------------------------------

#[test]
fn a_box_projects_its_support_radius_and_reports_its_floor() {
    let b = Obb::aabb(DVec3::new(1.0, 2.0, 5.0), DVec3::new(0.5, 0.25, 0.125));
    // On a coordinate axis, the radius is just that half-extent.
    assert!((b.projection_radius(DVec3::X) - 0.5).abs() < 1e-12);
    assert!((b.projection_radius(DVec3::Z) - 0.125).abs() < 1e-12);
    // On a diagonal, it is the sum of the projections — the corner, not the face.
    let diag = DVec3::new(1.0, 1.0, 0.0).normalize();
    assert!((b.projection_radius(diag) - 0.75 / 2.0f64.sqrt()).abs() < 1e-12);
    // `min_z` is the bottom face for an axis-aligned box...
    assert!((b.min_z() - (5.0 - 0.125)).abs() < 1e-12);
    // ...and a corner once it is tilted: stand the same box on end.
    let end = tilted(b.center, b.half_extents, 0.0, std::f64::consts::FRAC_PI_2);
    assert!((end.min_z() - (5.0 - 0.5)).abs() < 1e-12, "on end, got {}", end.min_z());
}

#[test]
fn changing_basis_moves_the_numbers_and_not_the_shape() {
    let b = tilted(DVec3::new(3.0, -1.0, 2.0), DVec3::new(0.4, 0.2, 0.1), 0.7, -0.3);
    let basis = [DVec3::Y, DVec3::Z, DVec3::X]; // a permutation is still orthonormal
    let moved = b.to_basis(DVec3::new(3.0, -1.0, 2.0), basis);

    // The origin was the box's own centre, so in the new frame it sits at zero.
    assert!(moved.center.length() < 1e-12, "centre should be the new origin, got {}", moved.center);
    // Shape is untouched, and the axes stay an orthonormal triad.
    assert_eq!(moved.half_extents, b.half_extents);
    for a in moved.axes {
        assert!((a.length() - 1.0).abs() < 1e-12);
    }
    // A support radius is basis-independent once the direction is carried across too.
    let dir = DVec3::new(1.0, -2.0, 0.5).normalize();
    let dir_local = DVec3::new(dir.dot(basis[0]), dir.dot(basis[1]), dir.dot(basis[2]));
    assert!((moved.projection_radius(dir_local) - b.projection_radius(dir)).abs() < 1e-12);
}

#[test]
fn boxes_overlap_when_they_share_space_and_not_when_they_are_apart() {
    let a = Obb::aabb(DVec3::ZERO, DVec3::splat(0.5));

    assert!(obb_overlap(&a, &a), "a box overlaps itself");
    // Faces exactly touching counts as contact — a crash test must not need overlap.
    assert!(obb_overlap(&a, &Obb::aabb(DVec3::new(1.0, 0.0, 0.0), DVec3::splat(0.5))));
    // A hair apart on a face axis does not.
    assert!(!obb_overlap(&a, &Obb::aabb(DVec3::new(1.001, 0.0, 0.0), DVec3::splat(0.5))));
    // Corner to corner is the same rule: exact contact is contact, a gap is not.
    assert!(obb_overlap(&a, &Obb::aabb(DVec3::splat(0.9), DVec3::splat(0.4))));
    assert!(!obb_overlap(&a, &Obb::aabb(DVec3::splat(0.95), DVec3::splat(0.4))));
    // One wholly inside another still overlaps.
    assert!(obb_overlap(&a, &Obb::aabb(DVec3::ZERO, DVec3::splat(0.05))));

    // Orientation is what decides it, not position: a 1.4 m plank half a metre from a
    // wall is nowhere near it end-on, and buried in it swung round to 45 deg, from the
    // same centre. This is the whole reason the airframe's hull carries its attitude.
    let wall = Obb::aabb(DVec3::new(0.0, 0.55, 0.0), DVec3::new(0.7, 0.05, 0.7));
    let plank = |yaw| tilted(DVec3::ZERO, DVec3::new(0.7, 0.05, 0.05), yaw, 0.0);
    assert!(!obb_overlap(&plank(0.0), &wall), "end-on, the plank reaches 0.05 m of 0.50");
    assert!(
        obb_overlap(&plank(std::f64::consts::FRAC_PI_4), &wall),
        "at 45 deg it reaches 0.53 m and must strike"
    );
}

/// The separating-axis test must not depend on argument order, and it must never
/// report *disjoint* for two boxes that provably share a point. The shared point comes
/// from sampling one box's interior, which is an independent check: it knows nothing
/// about axes.
#[test]
fn box_overlap_is_symmetric_and_never_misses_a_shared_point() {
    let mut rng = RaceRng::seeded(20260908);
    let mut shared = 0;
    for _ in 0..2000 {
        let a = tilted(
            DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            DVec3::new(rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6)),
            rng.uniform(-3.2, 3.2),
            rng.uniform(-1.5, 1.5),
        );
        let b = tilted(
            DVec3::new(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            DVec3::new(rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6)),
            rng.uniform(-3.2, 3.2),
            rng.uniform(-1.5, 1.5),
        );

        let ab = obb_overlap(&a, &b);
        assert_eq!(ab, obb_overlap(&b, &a), "overlap depended on argument order");

        // Walk a grid through a's interior; anything inside b is a witness.
        let n = 6;
        let inside_b = |p: DVec3| {
            let d = p - b.center;
            (0..3).all(|i| d.dot(b.axes[i]).abs() <= b.half_extents[i])
        };
        let witness = (0..=n).flat_map(|i| {
            (0..=n).flat_map(move |j| {
                (0..=n).map(move |k| {
                    let f = |v: i32| (f64::from(v) / f64::from(n)) * 2.0 - 1.0;
                    (f(i), f(j), f(k))
                })
            })
        });
        for (u, v, w) in witness {
            let p = a.center
                + a.axes[0] * (u * a.half_extents.x)
                + a.axes[1] * (v * a.half_extents.y)
                + a.axes[2] * (w * a.half_extents.z);
            if inside_b(p) {
                assert!(ab, "boxes share the point {p} but were reported disjoint");
                shared += 1;
                break;
            }
        }
    }
    // The generator has to actually produce overlaps, or the assertion above is idle.
    assert!(shared > 200, "only {shared} of 2000 pairs overlapped — the test is not loaded");
}

#[test]
fn a_sphere_meets_a_box_at_the_true_distance_to_its_surface() {
    let b = Obb::aabb(DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0));

    // Straight off a face: contact exactly when the radius spans the gap.
    assert!(!sphere_overlaps_obb(DVec3::new(2.0, 0.0, 0.0), 0.999, &b));
    assert!(sphere_overlaps_obb(DVec3::new(2.0, 0.0, 0.0), 1.0, &b), "touching counts");
    assert!(sphere_overlaps_obb(DVec3::new(2.0, 0.0, 0.0), 1.001, &b));

    // Off a corner the distance is the diagonal, not the axis gap — a 3-4-5 in the
    // xy-plane, 5 units from the corner at (1, 1, 0).
    let corner = DVec3::new(1.0 + 3.0, 1.0 + 4.0, 0.0);
    assert!(!sphere_overlaps_obb(corner, 4.999, &b));
    assert!(sphere_overlaps_obb(corner, 5.001, &b));
    // A test that only clamped one axis at a time would wrongly call this a hit.
    assert!(!sphere_overlaps_obb(corner, 4.5, &b));

    // A centre inside the box is a hit at any radius, including zero.
    assert!(sphere_overlaps_obb(DVec3::new(0.5, -0.5, 0.25), 0.0, &b));

    // And it is measured in the box's own axes, not the world's. Take one flat slab and
    // the same slab stood on end, and put a sphere above both: flat, the 0.25 m
    // half-extent faces up and there is 1.05 m of clearance; on end, the 1.0 m one
    // does and the same sphere is 0.3 m away.
    let flat = Obb::aabb(DVec3::ZERO, DVec3::new(1.0, 1.0, 0.25));
    let end = tilted(DVec3::ZERO, DVec3::new(1.0, 1.0, 0.25), 0.0, std::f64::consts::FRAC_PI_2);
    let above = DVec3::new(0.0, 0.0, 1.3);
    assert!(!sphere_overlaps_obb(above, 1.04, &flat));
    assert!(sphere_overlaps_obb(above, 1.06, &flat));
    assert!(!sphere_overlaps_obb(above, 0.29, &end));
    assert!(sphere_overlaps_obb(above, 0.31, &end));
}
