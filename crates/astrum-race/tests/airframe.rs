//! How big the F450 is, and what fits through a gate.
//!
//! The dimension assertions are deliberately written as literals rather than as
//! algebra over the constants: they are a guard against the model files and this crate
//! drifting apart, so they have to fail if someone edits `Propulsion.xml` or retypes a
//! constant. Each one names the file it came from.
//!
//! The collision cases are all written as **passes**, not as static poses. A gate's
//! stock is only 7.6 cm deep while the drone is 33 cm long, so at the instant its CG
//! is in the plane of a gate none of its four rotors are — putting the hull at one
//! hand-picked point and asking "does it hit?" mostly answers a question about the
//! along-normal coordinate. [`flies_clean`] sweeps a whole pass through the plane, so a
//! verdict means what it reads as.

use astrum_race::airframe::{
    body_axes, Hull, ARM_M, GEAR_DEPTH_M, GEAR_SPAN_M, HALF_SPAN_M, HEIGHT_M, PROP_DIAMETER_M,
    PROP_RADIUS_M, ROTOR_PLANE_M, SAMPLE_STEP_M, WIDTH_M,
};
use astrum_race::gate::{Gate, BAR_M, IN_TO_M, OPENING_M};
use bevy::math::DVec3;

/// A gate at the origin facing North, so gate-local `(x, y, z)` reads as
/// `(East, Up, North)` and a hand-placed offset means what it says.
fn north_gate() -> Gate {
    Gate::new(0, DVec3::ZERO, 0.0)
}

/// Fly the whole way through a gate's plane at a fixed offset and attitude: does
/// anything touch?
///
/// `right`/`up` are the CG's gate-local offsets from the middle of the opening;
/// `psi_rel` is the drone's heading *relative to the gate's normal*, so 0 is square to
/// the opening. The sweep runs 1.2 m either side of the plane in 1 cm steps — four
/// times as much travel as the airframe is long, and finer than the environment's own
/// [`SAMPLE_STEP_M`].
fn flies_clean(gate: &Gate, right: f64, up: f64, phi: f64, theta: f64, psi_rel: f64) -> bool {
    (-120..=120).all(|i| {
        let at = gate.to_world(DVec3::new(right, up, f64::from(i) * 0.01));
        !Hull::at(at, phi, theta, gate.yaw_rad + psi_rel).hits_gate(gate)
    })
}

/// [`flies_clean`] for a level pass square to the opening, which is most cases.
fn level_pass_is_clean(gate: &Gate, right: f64, up: f64) -> bool {
    flies_clean(gate, right, up, 0.0, 0.0, 0.0)
}

#[test]
fn dimensions_come_from_the_jsbsim_model() {
    // Propulsion.xml: four thrusters at (∓0.1651, ±0.1651, +0.025) m from the CG.
    assert_eq!(ARM_M, 0.1651);
    assert_eq!(ROTOR_PLANE_M, 0.025);
    // engine/DJI_9450.xml: <diameter unit="IN"> 9.4 </diameter>.
    assert!((PROP_DIAMETER_M - 0.23876).abs() < 1e-12, "9.4 in prop, got {PROP_DIAMETER_M}");
    assert!((PROP_RADIUS_M - 0.11938).abs() < 1e-12);
    // Gear.xml: contacts at |x| = |y| = 0.1016, all at z = -0.1524.
    assert_eq!(GEAR_SPAN_M, 0.1016);
    assert_eq!(GEAR_DEPTH_M, 0.1524);

    // What those add up to: arm plus blade tip each way, feet to rotor plane.
    assert!((HALF_SPAN_M - 0.28448).abs() < 1e-12, "half span {HALF_SPAN_M}");
    assert!((WIDTH_M - 0.56896).abs() < 1e-12, "tip-to-tip width {WIDTH_M}");
    assert!((HEIGHT_M - 0.1774).abs() < 1e-12, "gear feet to rotor plane {HEIGHT_M}");
    // The drone is a wide flat cross: three times as wide as it is tall.
    const { assert!(WIDTH_M / HEIGHT_M > 3.0) };
}

/// The two clearances the whole task hangs on, stated once: **10.9 cm** square to the
/// opening, **4.1 cm** at the worst heading.
///
/// The second number is the one that sizes the gate. Its worst case is 45° of crab,
/// where the drone presents a corner and reaches the tip circle rather than the half
/// span — and the tip circle is what a 25" opening could not pass, which is why the
/// opening is 31". Because 4.1 cm is positive, heading no longer constrains a pass
/// (`a_gate_fits_at_any_heading`).
///
/// This is also still why the hull is five parts rather than one shape:
/// * a **box** across the tip circle presents its 0.402 m diagonal when crabbed, which
///   the 31" opening does *not* clear — the box model would give back exactly what the
///   enlargement bought;
/// * a **sphere** on the CG now fits the opening, but it reaches the tip circle
///   *downwards* too, where the airframe stops at the gear — 20 cm of fiction against
///   the ground plane, five times the budget.
#[test]
fn the_drone_fits_the_opening_at_any_heading() {
    let half_opening = OPENING_M / 2.0;
    let clearance = half_opening - HALF_SPAN_M;
    assert!((clearance - 0.10922).abs() < 1e-12, "clearance per side {clearance}");

    // The worst heading: a corner leads, so the reach is the tip circle.
    let tip_circle = ARM_M * 2.0f64.sqrt() + PROP_RADIUS_M;
    assert!((tip_circle - 0.35287).abs() < 1e-5, "tip circle radius {tip_circle}");
    let crabbed = half_opening - tip_circle;
    assert!((crabbed - 0.04083).abs() < 1e-5, "worst-heading clearance {crabbed}");
    assert!(crabbed > 0.0, "the opening has to pass the airframe at any heading");
    // And it is worth more than the old 25" gate's best case, which is the test that
    // says the new size is an improvement and not just a different number.
    assert!(crabbed > 25.0 * IN_TO_M / 2.0 - HALF_SPAN_M, "worse than a square 25 in pass");

    // Why not model the hull as one box: crabbed, its diagonal still does not fit.
    let box_diagonal = HALF_SPAN_M * 2.0f64.sqrt();
    assert!((box_diagonal - 0.40232).abs() < 1e-5, "box diagonal {box_diagonal}");
    assert!(box_diagonal > half_opening, "a single box would not fit a crabbed pass");

    // Why not model it as one sphere: it would reach the tip circle below the CG,
    // where the airframe stops at the gear feet.
    assert!(
        tip_circle - GEAR_DEPTH_M > 4.0 * crabbed,
        "a CG sphere's ground error ({:.3} m) must be the reason it is rejected",
        tip_circle - GEAR_DEPTH_M
    );
}

/// The body axes are an orthonormal triad that tracks the attitude, and it is
/// `(forward, right, up)` — left-handed, because the right-handed aero triad puts
/// `down` third.
#[test]
fn body_axes_are_orthonormal_and_follow_the_attitude() {
    for (phi, theta, psi) in [
        (0.0, 0.0, 0.0),
        (0.4, -0.3, 1.1),
        (-1.2, 0.5, -2.4),
        (std::f64::consts::FRAC_PI_2, 0.0, 0.0),
    ] {
        let [f, r, u] = body_axes(phi, theta, psi);
        for a in [f, r, u] {
            assert!((a.length() - 1.0).abs() < 1e-12, "not a unit vector: {a}");
        }
        assert!(f.dot(r).abs() < 1e-12);
        assert!(f.dot(u).abs() < 1e-12);
        assert!(r.dot(u).abs() < 1e-12);
        assert!(
            (f.cross(r).dot(u) + 1.0).abs() < 1e-12,
            "(forward, right, up) must be left-handed, got {}",
            f.cross(r).dot(u)
        );
    }

    // Level and facing North: nose North, right wing East, top up.
    let [f, r, u] = body_axes(0.0, 0.0, 0.0);
    assert!((f - DVec3::Y).length() < 1e-12);
    assert!((r - DVec3::X).length() < 1e-12);
    assert!((u - DVec3::Z).length() < 1e-12);

    // Rolled 90 deg right: the right wing points at the ground and the top faces East.
    let [_, r, u] = body_axes(std::f64::consts::FRAC_PI_2, 0.0, 0.0);
    assert!((r + DVec3::Z).length() < 1e-12, "right wing should point down, got {r}");
    assert!((u - DVec3::X).length() < 1e-12, "the top should face East, got {u}");

    // Pitched 90 deg up: the nose points at the sky, the top faces aft (South).
    let [f, _, u] = body_axes(0.0, std::f64::consts::FRAC_PI_2, 0.0);
    assert!((f - DVec3::Z).length() < 1e-12, "nose should point up, got {f}");
    assert!((u + DVec3::Y).length() < 1e-12, "the top should face South, got {u}");
}

/// The four rotors sit on the arm diagonals at the rotor plane, and yaw carries them
/// round. Getting the arm layout wrong is the one error the gate tests could not
/// catch, because it changes the width without changing anything else.
#[test]
fn the_rotors_sit_on_the_arms_above_the_cg() {
    let h = Hull::at(DVec3::new(0.0, 0.0, 5.0), 0.0, 0.0, 0.0);
    for c in h.rotors {
        // All four at the rotor plane, all four one arm out on each horizontal axis.
        assert!((c.z - (5.0 + ROTOR_PLANE_M)).abs() < 1e-12, "rotor off the plane: {c}");
        assert!((c.x.abs() - ARM_M).abs() < 1e-12, "rotor {c} not one arm East/West");
        assert!((c.y.abs() - ARM_M).abs() < 1e-12, "rotor {c} not one arm North/South");
    }
    // Four distinct corners, not one repeated.
    for (i, a) in h.rotors.iter().enumerate() {
        for b in &h.rotors[i + 1..] {
            assert!((*a - *b).length() > 1e-9, "duplicate rotor at {a}");
        }
    }
    assert_eq!(h.rotor_radius, PROP_RADIUS_M);

    // Yawed 45 deg, the arms line up with the axes: the diagonal is the widest the
    // drone ever is, `ARM_M * sqrt(2)` out to each motor.
    let turned = Hull::at(DVec3::ZERO, 0.0, 0.0, std::f64::consts::FRAC_PI_4);
    let widest = turned.rotors.iter().map(|c| c.x.abs()).fold(0.0, f64::max);
    assert!((widest - ARM_M * 2.0f64.sqrt()).abs() < 1e-12, "widest reach {widest}");
}

/// The hull hangs off the CG rather than being centred on it: 15.2 cm of leg below,
/// 2.5 cm of rotor above. Getting this wrong by 15 cm would eat four times the gate
/// clearance.
#[test]
fn the_lowest_point_is_the_gear_until_the_drone_banks() {
    let level = Hull::at(DVec3::new(0.0, 0.0, 5.0), 0.0, 0.0, 0.0);
    assert!(
        (level.min_z() - (5.0 - GEAR_DEPTH_M)).abs() < 1e-12,
        "level, the feet are the lowest point, got {}",
        level.min_z()
    );

    // On its side, a rotor is lower than the feet ever were: the arm has swung down.
    let knife = Hull::at(DVec3::new(0.0, 0.0, 5.0), std::f64::consts::FRAC_PI_2, 0.0, 0.0);
    assert!(
        (knife.min_z() - (5.0 - HALF_SPAN_M)).abs() < 1e-12,
        "knife-edge should reach a half span down, got {}",
        knife.min_z()
    );
    assert!(knife.min_z() < level.min_z(), "banking must lower the hull, not raise it");
}

/// Flying through the middle of the opening, level, touches nothing — at the drone's
/// own height and a bit either side of it.
#[test]
fn a_centred_level_pass_touches_nothing() {
    let g = north_gate();
    for up in [-0.20, -0.05, 0.0, 0.05, 0.20] {
        for right in [-0.10, 0.0, 0.10] {
            assert!(level_pass_is_clean(&g, right, up), "clean pass at ({right}, {up}) hit");
        }
    }
}

/// Drift wide by more than the clearance and the props find an upright. The threshold
/// is the geometry, not a tuned number: `half opening − half span`.
#[test]
fn drifting_wide_strikes_an_upright() {
    let g = north_gate();
    let clearance = OPENING_M / 2.0 - HALF_SPAN_M;
    for side in [-1.0, 1.0] {
        assert!(
            level_pass_is_clean(&g, side * (clearance - 0.005), 0.0),
            "5 mm inside the clearance should be clean"
        );
        assert!(
            !level_pass_is_clean(&g, side * (clearance + 0.005), 0.0),
            "5 mm past the clearance should strike the upright"
        );
    }
}

/// Riding high finds the top bar. What reaches it is a blade tip at the rotor plane,
/// so the margin is the bar's underside less `ROTOR_PLANE_M + PROP_RADIUS_M` — 0.249 m,
/// well under the 0.3937 m the opening looks like it offers.
#[test]
fn riding_high_strikes_the_top_bar() {
    let g = north_gate();
    let headroom = OPENING_M / 2.0 - ROTOR_PLANE_M - PROP_RADIUS_M;
    assert!((headroom - 0.24932).abs() < 1e-5, "headroom {headroom}");
    assert!(level_pass_is_clean(&g, 0.0, headroom - 0.005), "just under the bar should be clean");
    assert!(!level_pass_is_clean(&g, 0.0, headroom + 0.005), "just into the bar should strike");
}

/// There is nothing at all below the sill, so a gate can be flown under. The margin is
/// again a blade tip: it has to clear the bottom of the uprights, not the CG.
#[test]
fn flying_under_a_gate_touches_nothing() {
    let g = north_gate();
    let below = -(OPENING_M / 2.0 + PROP_RADIUS_M + ROTOR_PLANE_M);
    assert!((below + 0.53808).abs() < 1e-5, "sill clearance {below}");
    // Right across the frame's width, including directly under both uprights.
    for i in -50..=50 {
        let right = f64::from(i) * 0.01;
        assert!(level_pass_is_clean(&g, right, below - 0.01), "under the frame at {right} hit");
    }
    // And the reason that margin is not just the sill: pass 5 cm higher and a blade
    // tip catches the bottom corner of an upright.
    assert!(!level_pass_is_clean(&g, -0.24, below + 0.05));
}

/// **Banking and pitching a centred drone through a gate is legal**, at every roll
/// angle and every sane pitch.
///
/// This is where a single-box hull is least honest: rolled 45 deg and pitched 30, a box
/// across the tip circle claims 0.371 m of vertical reach against the airframe's real
/// 0.293 m. In the old 25" gate that 7.7 cm of fiction made every racing attitude a
/// false crash outright; in a 31" gate it instead shrinks the vertical corridor at that
/// attitude from 10.1 cm to 2.3 cm. Four rotor discs and a core is what keeps the
/// corridor the size the hardware says it is.
#[test]
fn banking_and_pitching_still_fit() {
    let g = north_gate();
    // Any roll at all, knife-edge and inverted included.
    for roll_deg in (0..360).step_by(5) {
        assert!(
            flies_clean(&g, 0.0, 0.0, f64::from(roll_deg).to_radians(), 0.0, 0.0),
            "roll {roll_deg} should fit"
        );
    }
    // Any pitch.
    for pitch_deg in (-90..=90).step_by(5) {
        assert!(
            flies_clean(&g, 0.0, 0.0, 0.0, f64::from(pitch_deg).to_radians(), 0.0),
            "pitch {pitch_deg} should fit"
        );
    }
    // And the two together, over the range a racing quad actually uses.
    for roll_deg in [15, 30, 45] {
        for pitch_deg in [10, 20] {
            assert!(
                flies_clean(
                    &g,
                    0.0,
                    0.0,
                    f64::from(roll_deg).to_radians(),
                    f64::from(pitch_deg).to_radians(),
                    0.0
                ),
                "roll {roll_deg} pitch {pitch_deg} should fit"
            );
        }
    }
}

/// **A centred gate fits at any heading**, which is the property the 31" opening was
/// chosen to buy and the reason it is not the 25" of the brief.
///
/// In a 25" hole this test read the other way round: past ±13° of crab the drone
/// presented its 0.706 m diagonal into a 0.635 m opening and clipped an upright. That
/// capped the approach at ±13° while the course sampler is allowed a 120° turn — whose
/// bisected normal leaves a crossing up to 60° off — so a controller could be tracking
/// a gate's centre perfectly and still be flown into it.
///
/// The margin is not large: 4.1 cm at the worst heading against 10.9 cm square on, so
/// heading still costs lateral corridor. It just no longer costs the whole of it.
#[test]
fn a_gate_fits_at_any_heading() {
    let g = north_gate();
    // Every 5 deg through a full turn, including the 45 deg worst case and the 60 deg
    // that a 120 deg course turn can leave on the approach.
    for crab_deg in (0..360).step_by(5) {
        assert!(
            flies_clean(&g, 0.0, 0.0, 0.0, 0.0, f64::from(crab_deg).to_radians()),
            "{crab_deg} deg of crab should fit a 31 in opening"
        );
    }

    // The margin at the worst heading, measured rather than asserted from arithmetic:
    // at 45 deg the drone reaches the tip circle, so this much lateral offset is all
    // that is left. Either side of it the verdict flips.
    let crabbed = OPENING_M / 2.0 - (ARM_M * 2.0f64.sqrt() + PROP_RADIUS_M);
    let worst = std::f64::consts::FRAC_PI_4;
    for side in [-1.0, 1.0] {
        assert!(
            flies_clean(&g, side * (crabbed - 0.005), 0.0, 0.0, 0.0, worst),
            "5 mm inside the worst-heading clearance should be clean"
        );
        assert!(
            !flies_clean(&g, side * (crabbed + 0.005), 0.0, 0.0, 0.0, worst),
            "5 mm past the worst-heading clearance should clip an upright"
        );
    }
}

/// A hull that never reaches the gate's plane cannot hit it, however far off centre.
#[test]
fn a_hull_away_from_the_plane_never_hits() {
    let g = north_gate();
    // Dead in line with the middle of an upright, so this offset is one that *would*
    // strike if the hull ever reached the plane.
    let into_an_upright = OPENING_M / 2.0 + BAR_M / 2.0;
    for along in [-40.0, -1.0, -0.5, 0.5, 1.0, 40.0] {
        let at = g.to_world(DVec3::new(into_an_upright, 0.0, along));
        assert!(!Hull::at(at, 0.0, 0.0, 0.0).hits_gate(&g), "{along} m from the plane hit");
    }
}

/// A gate's yaw is honoured: the same relative geometry gives the same verdict at any
/// heading and any position, which is what says the test is done in the gate's frame
/// and not the world's.
#[test]
fn strikes_are_judged_in_the_gates_own_frame() {
    let clearance = OPENING_M / 2.0 - HALF_SPAN_M;
    for yaw_deg in [0.0f64, 37.0, 90.0, 180.0, 275.0, -140.0] {
        let g = Gate::new(0, DVec3::new(-8.0, 21.5, 3.25), yaw_deg.to_radians());
        assert!(level_pass_is_clean(&g, 0.0, 0.0), "yaw {yaw_deg}: a centred pass is clean");
        assert!(
            !level_pass_is_clean(&g, clearance + 0.01, 0.0),
            "yaw {yaw_deg}: a wide pass should strike"
        );
    }
}

/// The sample step is what makes a once-per-tick test sound, and this is the arithmetic
/// it rests on: a gap smaller than the stock plus a rotor's own diameter cannot be
/// stepped over, because the rotor would have to be clear on both sides at once.
#[test]
fn the_sample_step_cannot_tunnel_the_thinnest_stock() {
    let smallest_part = PROP_DIAMETER_M;
    assert!(
        SAMPLE_STEP_M < BAR_M + smallest_part,
        "{SAMPLE_STEP_M} m samples could step over {BAR_M} m of stock"
    );
    // With better than 2x of margin, so the bound is not a coincidence of the numbers.
    assert!(SAMPLE_STEP_M * 2.0 < BAR_M + smallest_part);
    // And the core section is thicker still in every direction, so the rotor is the
    // binding case.
    const { assert!(HEIGHT_M > SAMPLE_STEP_M && GEAR_SPAN_M * 2.0 > SAMPLE_STEP_M) };
}
