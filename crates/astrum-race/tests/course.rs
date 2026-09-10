//! Monte-Carlo course generation. Every rule the sampler is supposed to enforce is
//! re-checked here *independently* — the sampler's own `accepts` decides whether to
//! keep a draw, these assertions decide whether it was right — and swept over
//! thousands of seeds, because a randomized generator that is correct on seed 0 says
//! very little about seed 137. The sweep earns its keep: the escalating-backtrack
//! policy exists because seed 60 of the tighter envelope wedged at 11 of 12 gates.

use astrum_race::course::{generate, Course, CourseError, CourseParams};
use astrum_race::gate::{frame_radius_m, OPENING_M};
use astrum_race::geom::{segment_point_dist_sq, segment_segment_dist_sq};

/// Seeds swept by the property tests. Generation is pure arithmetic — no FDM, no
/// world — so this whole sweep is a fraction of a second, and a rare wedge is worth
/// far more than the milliseconds.
const SEEDS: u64 = 5000;

fn wrap_pi(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// Horizontal heading of each leg (rad; 0 = North, + = East).
fn leg_headings(c: &Course) -> Vec<f64> {
    c.gates
        .windows(2)
        .map(|w| {
            let d = w[1].center - w[0].center;
            d.x.atan2(d.y)
        })
        .collect()
}

/// Assert every invariant a generated course must satisfy, re-derived from the gates
/// rather than trusted from the sampler.
fn assert_flyable(c: &Course, p: &CourseParams) {
    let eps = 1e-6;
    assert_eq!(c.len(), p.gates as usize, "should place exactly the gates asked for");

    // Gate 0 anchors the volume at the horizontal origin.
    assert!(
        c.gates[0].center.x.abs() < 1e-12 && c.gates[0].center.y.abs() < 1e-12,
        "gate 0 should sit on the ENU origin"
    );

    // Containment: altitude band, sill clear of the ground, inside the cylinder.
    for g in &c.gates {
        assert!(
            g.center.z >= p.alt_min_m - eps && g.center.z <= p.alt_max_m + eps,
            "gate {} at {:.3} m escapes the altitude band [{}, {}]",
            g.index,
            g.center.z,
            p.alt_min_m,
            p.alt_max_m
        );
        assert!(g.sill_m() >= 0.0, "gate {} sill is underground ({:.3} m)", g.index, g.sill_m());
        let horiz = g.center.x.hypot(g.center.y);
        assert!(
            horiz <= p.course_radius_m + eps,
            "gate {} is {:.2} m out, past the {} m course radius",
            g.index,
            horiz,
            p.course_radius_m
        );
    }

    // Flyability: leg lengths, turns, climb angles.
    for (i, len) in c.leg_lengths_m().iter().enumerate() {
        assert!(
            *len >= p.leg_min_m - eps && *len <= p.leg_max_m + eps,
            "leg {i} is {len:.3} m, outside [{}, {}]",
            p.leg_min_m,
            p.leg_max_m
        );
    }
    for (i, turn) in c.turn_angles_deg().iter().enumerate() {
        assert!(
            *turn <= p.max_turn_deg + eps,
            "turn at gate {} is {turn:.2} deg, past the {} deg limit",
            i + 1,
            p.max_turn_deg
        );
    }
    for (i, climb) in c.climb_angles_deg().iter().enumerate() {
        assert!(
            climb.abs() <= p.max_climb_deg + eps,
            "leg {i} climbs {climb:.2} deg, past the {} deg limit",
            p.max_climb_deg
        );
    }

    // Non-overlap: gate centres apart.
    for i in 0..c.len() {
        for j in (i + 1)..c.len() {
            let gap = (c.gates[j].center - c.gates[i].center).length();
            assert!(
                gap >= p.min_gate_gap_m - eps,
                "gates {i} and {j} are only {gap:.3} m apart (min {})",
                p.min_gate_gap_m
            );
        }
    }

    // Non-overlap: a leg must clear every frame it is not flying through.
    let clear = frame_radius_m() + p.min_leg_gate_gap_m;
    for (li, w) in c.gates.windows(2).enumerate() {
        for (gi, g) in c.gates.iter().enumerate() {
            if gi == li || gi == li + 1 {
                continue; // the leg's own two gates
            }
            let d = segment_point_dist_sq(w[0].center, w[1].center, g.center).sqrt();
            assert!(d >= clear - eps, "leg {li} passes {d:.3} m from gate {gi} (min {clear:.3} m)");
        }
    }

    // Non-overlap: non-adjacent legs must not cross.
    let legs: Vec<_> = c.gates.windows(2).map(|w| (w[0].center, w[1].center)).collect();
    for i in 0..legs.len() {
        for j in (i + 2)..legs.len() {
            let d = segment_segment_dist_sq(legs[i].0, legs[i].1, legs[j].0, legs[j].1).sqrt();
            assert!(
                d >= p.min_leg_leg_gap_m - eps,
                "legs {i} and {j} pass {d:.3} m apart (min {})",
                p.min_leg_leg_gap_m
            );
        }
    }

    // Orientation: each gate's normal bisects its legs, so the crossing angle is at
    // most half the turn — that is what makes a 120 deg turn flyable through a
    // 0.7874 m hole.
    let h = leg_headings(c);
    assert!(wrap_pi(c.gates[0].yaw_rad - h[0]).abs() < 1e-9, "gate 0 faces its outgoing leg");
    let last = c.len() - 1;
    assert!(
        wrap_pi(c.gates[last].yaw_rad - h[last - 1]).abs() < 1e-9,
        "the last gate faces its incoming leg"
    );
    for i in 1..last {
        let (into, out) =
            (wrap_pi(h[i - 1] - c.gates[i].yaw_rad), wrap_pi(h[i] - c.gates[i].yaw_rad));
        assert!(
            (into + out).abs() < 1e-9,
            "gate {i}'s normal should bisect its legs ({:.4} vs {:.4} rad)",
            into,
            out
        );
        assert!(
            into.abs().to_degrees() <= p.max_turn_deg / 2.0 + eps,
            "gate {i} is crossed {:.2} deg off its normal",
            into.abs().to_degrees()
        );
    }
}

#[test]
fn default_envelope_is_flyable_across_seeds() {
    let p = CourseParams::default();
    for seed in 0..SEEDS {
        let c = generate(seed, &p).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_flyable(&c, &p);
    }
}

/// A deliberately dense envelope: 12 gates in a 45 m bowl with wide clearances, so
/// the sampler has to work rather than accept its first draw.
fn tight_envelope() -> CourseParams {
    CourseParams {
        gates: 12,
        leg_min_m: 10.0,
        leg_max_m: 14.0,
        max_turn_deg: 90.0,
        max_climb_deg: 10.0,
        alt_min_m: 2.0,
        alt_max_m: 3.0,
        min_gate_gap_m: 9.0,
        min_leg_gate_gap_m: 2.0,
        min_leg_leg_gap_m: 3.0,
        course_radius_m: 45.0,
        start_heading_deg: 42.0,
        ..CourseParams::default()
    }
}

/// The limits are parameters, not constants: a tighter, denser envelope has to hold
/// just as exactly as the default one.
#[test]
fn tighter_envelope_is_flyable_across_seeds() {
    let p = tight_envelope();
    for seed in 0..SEEDS {
        let c = generate(seed, &p).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_flyable(&c, &p);
        assert!(
            (c.gates[0].yaw_rad.to_degrees() - 42.0).abs() < 1e-9,
            "the entry heading is fixed, not drawn"
        );
    }
}

/// Same seed, same course — and the sampler's cost is reported, not hidden.
#[test]
fn generation_is_deterministic() {
    let p = CourseParams::default();
    for seed in [0, 1, 42, 9_999_999] {
        let a = generate(seed, &p).unwrap();
        let b = generate(seed, &p).unwrap();
        assert_eq!(a.gates, b.gates, "seed {seed} must reproduce exactly");
        assert_eq!((a.samples, a.backtracks), (b.samples, b.backtracks));
        assert_eq!(a.seed, seed);
        assert!(a.samples >= p.gates - 1, "every gate after the first costs a sample");
    }
}

/// Different seeds are different courses — the point of randomizing.
#[test]
fn different_seeds_give_different_courses() {
    let p = CourseParams::default();
    let base = generate(0, &p).unwrap();
    let mut distinct = 0;
    for seed in 1..50 {
        if generate(seed, &p).unwrap().gates != base.gates {
            distinct += 1;
        }
    }
    assert_eq!(distinct, 49, "every seed should draw its own course");
}

/// The sampler must actually use its envelope, not hug one corner of it: a
/// generator that only ever draws 8 m legs and gentle turns would pass every
/// invariant above and still be useless for training.
#[test]
fn sampling_spans_the_envelope() {
    let p = CourseParams::default();
    let (mut min_leg, mut max_leg) = (f64::MAX, f64::MIN);
    let (mut min_alt, mut max_alt) = (f64::MAX, f64::MIN);
    let (mut left, mut right, mut hard) = (0, 0, 0);

    for seed in 0..SEEDS {
        let c = generate(seed, &p).unwrap();
        for len in c.leg_lengths_m() {
            min_leg = min_leg.min(len);
            max_leg = max_leg.max(len);
        }
        for g in &c.gates {
            min_alt = min_alt.min(g.center.z);
            max_alt = max_alt.max(g.center.z);
        }
        let h = leg_headings(&c);
        for w in h.windows(2) {
            let turn = wrap_pi(w[1] - w[0]).to_degrees();
            if turn < -5.0 {
                left += 1;
            } else if turn > 5.0 {
                right += 1;
            }
            if turn.abs() > 100.0 {
                hard += 1;
            }
        }
    }

    // Leg lengths and altitudes reach into both ends of their ranges.
    assert!(min_leg < p.leg_min_m + 0.5, "shortest leg drawn was {min_leg:.2} m");
    assert!(max_leg > p.leg_max_m - 0.5, "longest leg drawn was {max_leg:.2} m");
    assert!(min_alt < p.alt_min_m + 0.3, "lowest gate drawn was {min_alt:.2} m");
    assert!(max_alt > p.alt_max_m - 0.3, "highest gate drawn was {max_alt:.2} m");
    // Courses turn both ways, and sometimes near the limit.
    assert!(left > 100 && right > 100, "turns should go both ways (L {left}, R {right})");
    assert!(hard > 20, "some turns should approach the limit (got {hard} over 100 deg)");
}

/// A zero-width altitude band is a legal request (a deliberately flat course), not
/// an infeasible one — the altitude is drawn inside the band rather than rejected
/// against it, so the band can be a single value.
#[test]
fn flat_course_is_allowed() {
    let p = CourseParams { alt_min_m: 2.5, alt_max_m: 2.5, ..CourseParams::default() };
    let c = generate(7, &p).unwrap();
    assert_flyable(&c, &p);
    for g in &c.gates {
        assert!((g.center.z - 2.5).abs() < 1e-12, "every gate should be at 2.5 m");
    }
    assert!(c.climb_angles_deg().iter().all(|a| a.abs() < 1e-9), "a flat course does not climb");
}

/// Likewise a zero turn limit: a straight line of gates, every one facing the same
/// way.
#[test]
fn straight_course_is_allowed() {
    let p = CourseParams { gates: 3, max_turn_deg: 0.0, ..CourseParams::default() };
    let c = generate(3, &p).unwrap();
    assert_flyable(&c, &p);
    assert!(c.turn_angles_deg().iter().all(|t| *t < 1e-9), "no turns were permitted");
    for g in &c.gates {
        assert!(g.yaw_rad.abs() < 1e-9, "every gate faces the start heading (North)");
    }
}

/// A contradictory envelope is rejected as a parameter error — a clear message about
/// the input, not an exhausted sampler.
#[test]
fn contradictory_parameters_are_rejected() {
    let base = CourseParams::default();
    let cases = [
        ("too few gates", CourseParams { gates: 1, ..base }),
        ("inverted legs", CourseParams { leg_min_m: 20.0, leg_max_m: 8.0, ..base }),
        ("zero-length legs", CourseParams { leg_min_m: 0.0, ..base }),
        ("turn at 180", CourseParams { max_turn_deg: 180.0, ..base }),
        ("negative turn", CourseParams { max_turn_deg: -10.0, ..base }),
        ("vertical climb", CourseParams { max_climb_deg: 90.0, ..base }),
        ("inverted altitudes", CourseParams { alt_min_m: 5.0, alt_max_m: 2.0, ..base }),
        ("sill underground", CourseParams { alt_min_m: OPENING_M / 4.0, ..base }),
        ("negative clearance", CourseParams { min_leg_leg_gap_m: -1.0, ..base }),
        ("gap wider than a leg", CourseParams { min_gate_gap_m: 12.0, ..base }),
        ("volume smaller than a leg", CourseParams { course_radius_m: 5.0, ..base }),
        ("no tries", CourseParams { tries_per_gate: 0, ..base }),
    ];
    for (name, p) in cases {
        match generate(0, &p) {
            Err(CourseError::Params(_)) => {}
            other => panic!("{name} should be a parameter error, got {other:?}"),
        }
        assert!(p.validate().is_err(), "{name} should fail validation directly too");
    }
}

/// A legal-but-unsatisfiable envelope fails as *infeasible*, reporting how far it
/// got — 60 gates 8 m apart will not fit in a 20 m-radius bowl, and the sampler must
/// say so instead of spinning.
#[test]
fn overpacked_course_reports_infeasible() {
    let p = CourseParams {
        gates: 60,
        min_gate_gap_m: 8.0,
        leg_min_m: 8.0,
        leg_max_m: 20.0,
        course_radius_m: 20.0,
        tries_per_gate: 50,
        max_backtracks: 50,
        ..CourseParams::default()
    };
    assert!(p.validate().is_ok(), "the envelope itself is legal");
    match generate(0, &p) {
        Err(CourseError::Infeasible { placed, wanted, samples, backtracks }) => {
            assert_eq!(wanted, 60);
            assert!(placed < 60, "it cannot have placed them all");
            assert!(samples > 0 && backtracks > 0, "it should have tried and backtracked");
            // The budget bounds the work: no unbounded search.
            assert!(
                samples <= (p.max_backtracks + p.gates) * p.tries_per_gate,
                "sampling must stay inside its budget, spent {samples}"
            );
        }
        other => panic!("an overpacked course should be infeasible, got {other:?}"),
    }
}

/// Regression: seed 60 of the dense envelope used to wedge at 11 of 12 gates, burning
/// its whole budget. The dead end was two gates deep, and undoing one gate at a time
/// only ever re-sampled the same trap.
///
/// The subtlety worth protecting: "how long have we been stuck" must be measured
/// against the *longest* course found so far. Counting any successful placement as
/// progress — including the one that merely re-fills the gate just popped — pins the
/// unwind at one gate forever, which is exactly the failure this replaced.
#[test]
fn a_dead_end_deeper_than_one_gate_is_escaped() {
    let p = tight_envelope();
    let c = generate(60, &p).expect("seed 60 must not wedge the sampler");
    assert_flyable(&c, &p);
    assert!(c.backtracks > 0, "this seed is only interesting because it does backtrack");
    assert!(
        c.backtracks < p.max_backtracks,
        "it must escape well inside the budget, used {} of {}",
        c.backtracks,
        p.max_backtracks
    );
}

/// The reported metrics are the ones the summary line quotes.
#[test]
fn course_metrics_agree_with_the_gates() {
    let p = CourseParams::default();
    let c = generate(11, &p).unwrap();
    let legs = c.leg_lengths_m();

    assert_eq!(legs.len(), c.len() - 1);
    assert_eq!(c.turn_angles_deg().len(), c.len() - 2);
    assert_eq!(c.climb_angles_deg().len(), c.len() - 1);
    assert!((c.total_length_m() - legs.iter().sum::<f64>()).abs() < 1e-12);
    assert!(!c.is_empty());

    let s = c.summary();
    assert!(s.contains("8 gates") && s.contains("seed 11"), "summary should say what it is: {s}");
}
