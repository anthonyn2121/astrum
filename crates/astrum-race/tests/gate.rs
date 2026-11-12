//! Gate hardware and geometry: the 46x3" / 40x3" frame, its local frame, and the
//! opening test a flythrough is scored with.

use astrum_race::gate::{
    frame_radius_m, Gate, BAR_IN, BAR_M, IN_TO_M, OPENING_IN, OPENING_M, OUTER_HEIGHT_M,
    OUTER_WIDTH_M, TOP_BAR_M, UPRIGHT_M,
};
use bevy::math::DVec3;

/// The gate's dimensions, and the identity that makes them consistent: the 46" top
/// bar is the 40" opening plus both 3" uprights, all derived from the one
/// `OPENING_IN` constant.
///
/// The literals below are the guard against someone retyping a constant: each
/// is the inch figure converted by hand.
#[test]
fn frame_dimensions_are_self_consistent() {
    assert_eq!(OPENING_IN, 40.0);
    assert_eq!(BAR_IN, 3.0);
    assert!((OPENING_M - 1.016).abs() < 1e-12, "40 in opening = 1.016 m, got {OPENING_M}");
    assert!((BAR_M - 0.0762).abs() < 1e-12, "3 in stock = 0.0762 m, got {BAR_M}");
    assert!((TOP_BAR_M - 1.1684).abs() < 1e-12, "46 in top bar = 1.1684 m, got {TOP_BAR_M}");
    assert_eq!(UPRIGHT_M, OPENING_M, "the uprights' length is the opening's height");

    // The identity: 46" = 40" + 2 x 3". The constants are derived from OPENING_IN, so
    // this holds exactly, not just to a tolerance.
    assert_eq!(TOP_BAR_M, OPENING_M + 2.0 * BAR_M, "top bar spans the opening plus both uprights");
    assert!((TOP_BAR_M - 46.0 * IN_TO_M).abs() < 1e-12);
    assert_eq!(OUTER_WIDTH_M, TOP_BAR_M);
    assert!((OUTER_HEIGHT_M - 43.0 * IN_TO_M).abs() < 1e-12, "outer height = 40 in + 3 in");
}

/// The gate's axes are an orthonormal basis with the normal on the commanded heading
/// (ENU: 0 = North, + = East) and up along world up.
///
/// Handedness is worth stating precisely, because it is easy to assert the wrong
/// identity: `right` is *starboard of the flight direction*, so `(normal, up, right)`
/// is the right-handed cycle and the local coordinate triple `(x=right, y=up,
/// z=normal)` is therefore **left**-handed — the same arrangement as a camera basis.
#[test]
fn gate_axes_are_orthonormal_and_follow_the_heading() {
    for yaw_deg in [0.0, 37.0, 90.0, 180.0, 275.0, -140.0] {
        let g = Gate::new(0, DVec3::new(3.0, -4.0, 2.0), f64::to_radians(yaw_deg));
        let (n, r, u) = (g.normal(), g.right(), g.up());

        for (name, v) in [("normal", n), ("right", r), ("up", u)] {
            assert!((v.length() - 1.0).abs() < 1e-12, "{name} must be a unit vector");
        }
        assert!(n.dot(r).abs() < 1e-12 && n.dot(u).abs() < 1e-12 && r.dot(u).abs() < 1e-12);
        // The full cycle, so no single sign can be wrong without failing here.
        assert!((n.cross(u) - r).length() < 1e-12, "normal x up must be right");
        assert!((u.cross(r) - n).length() < 1e-12, "up x right must be the normal");
        assert!((r.cross(n) - u).length() < 1e-12, "right x normal must be up");
        assert!(n.z.abs() < 1e-12, "gates stand upright: the normal stays horizontal");
    }

    // Heading convention, spelled out — including which way "right" is.
    let north = Gate::new(0, DVec3::ZERO, 0.0);
    assert!((north.normal() - DVec3::Y).length() < 1e-12, "yaw 0 points North (+Y)");
    assert!((north.right() - DVec3::X).length() < 1e-12, "flying North, right is East (+X)");
    let east = Gate::new(0, DVec3::ZERO, std::f64::consts::FRAC_PI_2);
    assert!((east.normal() - DVec3::X).length() < 1e-12, "yaw 90 points East (+X)");
    assert!((east.right() + DVec3::Y).length() < 1e-12, "flying East, right is South (-Y)");
}

/// `to_local` and `to_world` are inverses, and the local basis means what it says.
#[test]
fn local_and_world_round_trip() {
    let g = Gate::new(2, DVec3::new(-12.0, 8.5, 3.0), 1.1);
    for p in [
        DVec3::ZERO,
        DVec3::new(0.3, -0.2, 1.7),
        DVec3::new(-5.0, 0.0, 0.0),
        DVec3::new(1.0, 2.0, 3.0),
    ] {
        let round = g.to_world(g.to_local(p));
        assert!((round - p).length() < 1e-9, "to_world(to_local(p)) must be p, got {round:?}");
    }

    // The centre is the local origin; one metre downstream is +z local.
    assert!(g.to_local(g.center).length() < 1e-12);
    let downstream = g.center + g.normal();
    assert!((g.to_local(downstream) - DVec3::Z).length() < 1e-12);
    // Up in the world is +y (up) in the gate.
    assert!((g.to_local(g.center + DVec3::Z) - DVec3::Y).length() < 1e-12);
}

/// The opening is a 40" square about the centre: inside is inside, and a hair
/// outside either edge is not.
#[test]
fn opening_contains_only_the_hole() {
    let g = Gate::new(0, DVec3::new(0.0, 0.0, 2.0), 0.4);
    let h = g.half_opening_m();
    assert!((2.0 * h - OPENING_M).abs() < 1e-12);

    assert!(g.opening_contains(DVec3::ZERO), "the centre is in the opening");
    assert!(g.opening_contains(DVec3::new(h - 1e-6, h - 1e-6, 0.0)), "just inside a corner");
    assert!(!g.opening_contains(DVec3::new(h + 1e-6, 0.0, 0.0)), "just past the upright");
    assert!(!g.opening_contains(DVec3::new(0.0, h + 1e-6, 0.0)), "just under the top bar");
    // The along-normal coordinate is the caller's business (it owns the crossing).
    assert!(g.opening_contains(DVec3::new(0.0, 0.0, 50.0)));

    // The sill is the lowest hardware: half an opening below the centre.
    assert!((g.sill_m() - (g.center.z - h)).abs() < 1e-12);
}

/// The four opening corners are a 40" square in the gate's plane.
#[test]
fn opening_corners_form_the_square() {
    let g = Gate::new(1, DVec3::new(4.0, 4.0, 2.5), -0.8);
    let c = g.opening_corners();

    for (i, p) in c.iter().enumerate() {
        let local = g.to_local(*p);
        assert!(local.z.abs() < 1e-12, "corner {i} lies in the gate plane");
        assert!((local.x.abs() - g.half_opening_m()).abs() < 1e-12);
        assert!((local.y.abs() - g.half_opening_m()).abs() < 1e-12);
    }
    // Consecutive corners are one side apart; the diagonal is side x sqrt(2).
    for i in 0..4 {
        let side = (c[(i + 1) % 4] - c[i]).length();
        assert!((side - OPENING_M).abs() < 1e-9, "side {i} should be {OPENING_M} m, got {side}");
    }
    assert!(((c[2] - c[0]).length() - OPENING_M * 2f64.sqrt()).abs() < 1e-9);
}

/// The three bars sit *around* the opening, never in it, and add up to the assembled
/// frame: two 40" uprights flanking the hole, a 46" bar across the top.
#[test]
fn bars_surround_the_opening_without_blocking_it() {
    let g = Gate::new(0, DVec3::new(0.0, 0.0, 2.0), 0.9);
    let bars = g.bars();
    let h = g.half_opening_m();

    // Nothing intrudes into the opening: sample the hole densely.
    //
    // The *interior* of the hole, precisely. An upright's inner face and the opening's
    // edge are the same plane, and both `opening_contains` and `Bar::contains_local`
    // are closed (`<=`), so a point exactly on that plane is claimed by both by
    // construction — a tie over a measure-zero boundary, not hardware in the way. Which
    // way the tie falls is down to how the two expressions round, so it is not
    // something to assert either way; sample a nanometre inside instead.
    let inner = h - 1e-9;
    for i in 0..=20 {
        for j in 0..=20 {
            let x = -inner + 2.0 * inner * (i as f64 / 20.0);
            let y = -inner + 2.0 * inner * (j as f64 / 20.0);
            let p = DVec3::new(x, y, 0.0);
            assert!(
                !bars.iter().any(|b| b.contains_local(p)),
                "hardware must not intrude into the opening at ({x:.3}, {y:.3})"
            );
        }
    }

    // Left and right uprights: 3" thick, 40" tall, flanking the hole.
    let (left, right, top) = (bars[0], bars[1], bars[2]);
    assert!((left.center.x + (h + BAR_M / 2.0)).abs() < 1e-12, "left upright hugs the opening");
    assert!((right.center.x - (h + BAR_M / 2.0)).abs() < 1e-12, "right upright hugs the opening");
    for up in [left, right] {
        assert!((2.0 * up.half_extents.x - BAR_M).abs() < 1e-12, "3 in thick");
        assert!((2.0 * up.half_extents.y - UPRIGHT_M).abs() < 1e-12, "40 in tall");
    }
    // Top bar: 46" across, 3" deep, resting on the uprights above the hole.
    assert!((2.0 * top.half_extents.x - TOP_BAR_M).abs() < 1e-12, "46 in across");
    assert!((2.0 * top.half_extents.y - BAR_M).abs() < 1e-12, "3 in thick");
    assert!((top.center.y - (h + BAR_M / 2.0)).abs() < 1e-12, "sits directly on the opening");
    // It spans exactly the uprights' outer faces.
    assert!((top.half_extents.x - (right.center.x + right.half_extents.x)).abs() < 1e-12);

    // Overall envelope: 46" wide, 43" tall, all above the sill.
    let lowest = bars.iter().map(|b| b.center.y - b.half_extents.y).fold(f64::MAX, f64::min);
    let highest = bars.iter().map(|b| b.center.y + b.half_extents.y).fold(f64::MIN, f64::max);
    assert!((lowest + h).abs() < 1e-12, "the frame bottoms out at the sill");
    assert!((highest - lowest - OUTER_HEIGHT_M).abs() < 1e-12, "43 in tall overall");
}

/// The bounding sphere the course sampler keeps legs out of really does contain
/// every piece of hardware — and isn't wastefully bigger than it needs to be.
#[test]
fn frame_radius_bounds_the_hardware() {
    let g = Gate::new(0, DVec3::new(1.0, 2.0, 3.0), 0.3);
    let r = frame_radius_m();
    let mut worst: f64 = 0.0;

    for b in g.bars() {
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    let corner = b.center
                        + DVec3::new(
                            sx * b.half_extents.x,
                            sy * b.half_extents.y,
                            sz * b.half_extents.z,
                        );
                    // The sphere is about the opening's centre, and the frame's depth
                    // is not part of the planar radius the sampler reasons about.
                    let d = corner.x.hypot(corner.y);
                    worst = worst.max(d);
                    assert!(d <= r + 1e-12, "hardware corner at {d:.4} m escapes radius {r:.4} m");
                }
            }
        }
    }
    assert!(worst > 0.9 * r, "radius {r:.4} m should be snug: worst corner {worst:.4} m");
    // And the number itself: hypot(23 in, 20 in + 3 in) — the top-bar corner, 23" out
    // and 23" up from the opening's centre.
    assert!((r - 23.0 * IN_TO_M * 2f64.sqrt()).abs() < 1e-12, "frame radius {r}");
    assert!((r - 0.826_183_563_1).abs() < 1e-9, "frame radius {r}");
}

/// The Bevy transform places the gate where the gate says it is, with the sim's yaw
/// convention (`Quat::from_rotation_z(-psi)`).
#[test]
fn transform_matches_the_pose() {
    let g = Gate::new(3, DVec3::new(-7.25, 11.5, 2.75), 0.65);
    let tf = g.transform();
    assert!((tf.translation - g.center.as_vec3()).length() < 1e-5);
    // Rotating +Y (North, the yaw-0 normal) by the transform gives the gate normal.
    let rotated = tf.rotation * bevy::math::Vec3::Y;
    assert!(
        (rotated - g.normal().as_vec3()).length() < 1e-5,
        "transform yaw should agree with normal(): {rotated:?} vs {:?}",
        g.normal()
    );
}
