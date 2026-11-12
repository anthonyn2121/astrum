//! Gate scoring: what counts as flying through a gate.
//!
//! `track_gates` is driven here over hand-placed positions in a rig — the race's own
//! tick systems and no JSBSim, so the drone can be put exactly where a case needs it.
//! That matters because the interesting inputs are the ones a real flight rarely
//! produces on demand: a tick that steps clean over the 7 cm frame, a backwards
//! crossing, a pass outside the hole.

mod common;

use astrum_proto::{EndReason, EventKind};
use bevy::math::DVec3;
use common::{exited, fly_to, progress, rig, straight_course};

fn passed(events: &[EventKind]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::GatePassed { gate, .. } => Some(*gate),
            _ => None,
        })
        .collect()
}

fn missed(events: &[EventKind]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::GateMissed { gate, .. } => Some(*gate),
            _ => None,
        })
        .collect()
}

/// Flying straight down the line takes both gates, in order, and the last one ends
/// the run as `RaceComplete`.
#[test]
fn straight_flight_passes_every_gate() {
    let mut app = rig(straight_course(), DVec3::new(0.0, -10.0, 3.0));
    let mut events = Vec::new();
    for i in 1..=60 {
        // -10 m to +50 m North, 1 m a tick: through gate 0 at y=0, gate 1 at y=20.
        events.extend(fly_to(&mut app, DVec3::new(0.0, -10.0 + f64::from(i), 3.0)));
        if exited(&app) {
            break;
        }
    }
    let p = progress(&app);
    assert_eq!(passed(&events), [0, 1], "both gates are scored, in order");
    assert_eq!((p.passed, p.missed), (2, 0));
    assert!(p.finished(), "the course is complete");
    assert!(exited(&app), "and completing it ends the run");
    assert!(
        events
            .contains(&EventKind::RunEnded { reason: EndReason::RaceComplete, message: "".into() }),
        "the end is reported as a completed race: {events:?}"
    );
    // The pass events carry where the CG pierced the gate: dead centre here.
    let offsets: Vec<(f64, f64)> = events
        .iter()
        .filter_map(|e| match e {
            EventKind::GatePassed { offset_x_m, offset_y_m, .. } => {
                Some((*offset_x_m, *offset_y_m))
            }
            _ => None,
        })
        .collect();
    assert!(offsets.iter().all(|(x, y)| x.abs() < 1e-9 && y.abs() < 1e-9), "{offsets:?}");
}

/// A tick that steps clean over the frame still counts. This is the case a
/// point-inside-the-frame test would miss: 20 m/s at 120 Hz is 17 cm a tick, and a
/// coarser controller step can be metres.
#[test]
fn a_single_long_step_across_the_plane_counts() {
    let mut app = rig(straight_course(), DVec3::new(0.0, -8.0, 3.0));
    let events = fly_to(&mut app, DVec3::new(0.0, 8.0, 3.0)); // 16 m in one tick
    let p = progress(&app);
    assert_eq!(passed(&events), [0], "the crossing is on the segment, not at the sample points");
    assert_eq!((p.passed, p.next_gate), (1, 1));
}

/// A pass is reported where the CG pierced the plane, in the gate's frame.
#[test]
fn a_pass_reports_where_the_cg_went_through() {
    let mut app = rig(straight_course(), DVec3::new(0.15, -5.0, 3.0));
    // `right` of a North-facing gate is East (+x); up is +z.
    let events = fly_to(&mut app, DVec3::new(0.15, 5.0, 3.1));
    let [EventKind::GatePassed { gate: 0, offset_x_m, offset_y_m, .. }] = events[..] else {
        panic!("one pass expected, got {events:?}");
    };
    assert!((offset_x_m - 0.15).abs() < 1e-9, "x {offset_x_m}");
    assert!((offset_y_m - 0.05).abs() < 1e-9, "y {offset_y_m}"); // halfway up the climb
}

/// Going round the outside is a miss, not a pass — and the gate stays pending.
#[test]
fn passing_beside_the_opening_is_a_miss() {
    let mut app = rig(straight_course(), DVec3::new(1.5, -5.0, 3.0));
    // 1.5 m to the right of the centreline: past the opening's half-width, but well
    // inside the radius where a crossing is still *about* this gate.
    let events = fly_to(&mut app, DVec3::new(1.5, 5.0, 3.0));
    let p = progress(&app);
    assert_eq!(missed(&events), [0]);
    assert_eq!((p.passed, p.missed, p.next_gate), (0, 1, 0), "a miss must not advance the course");

    // Coming back (round the outside, as the frame is solid) and doing it properly
    // still scores.
    fly_to(&mut app, DVec3::new(1.5, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0));
    let events = fly_to(&mut app, DVec3::new(0.0, 5.0, 3.0));
    assert_eq!(passed(&events), [0], "a second attempt should score");
}

/// Crossing the plane far from the gate is not a miss — it is unrelated flying.
#[test]
fn crossing_the_plane_far_away_is_not_scored() {
    let mut app = rig(straight_course(), DVec3::new(25.0, -5.0, 3.0));
    let events = fly_to(&mut app, DVec3::new(25.0, 5.0, 3.0));
    let p = progress(&app);
    assert!(events.is_empty(), "{events:?}");
    assert_eq!((p.passed, p.missed), (0, 0), "25 m off to the side is not an attempt on the gate");
}

/// Only downstream crossings count: backing through a gate scores nothing.
#[test]
fn flying_backwards_through_a_gate_scores_nothing() {
    let mut app = rig(straight_course(), DVec3::new(0.0, 5.0, 3.0));
    let events = fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0)); // through the hole, wrong way
    let p = progress(&app);
    assert!(events.is_empty(), "{events:?}");
    assert_eq!((p.passed, p.missed, p.next_gate), (0, 0, 0), "the wrong way is not a pass");
}

/// Gates must be taken in order: flying through gate 1 first scores nothing, and
/// gate 0 is still the one being watched.
#[test]
fn gates_must_be_taken_in_order() {
    let mut app = rig(straight_course(), DVec3::new(0.0, 15.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 25.0, 3.0)); // straight through gate 1
    let p = progress(&app);
    assert_eq!((p.passed, p.next_gate), (0, 0), "gate 1 does not count while gate 0 is pending");

    // Gate 0 is behind the drone now; coming back round scores it, then gate 1.
    fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 5.0, 3.0));
    assert_eq!(progress(&app).next_gate, 1, "gate 0 scored, gate 1 now pending");
    fly_to(&mut app, DVec3::new(0.0, 15.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 25.0, 3.0));
    assert!(progress(&app).finished(), "and then the course is complete");
}

/// The opening is scored at the height it is at: ducking under the sill is a miss.
#[test]
fn flying_under_the_sill_is_a_miss() {
    let mut app = rig(straight_course(), DVec3::new(0.0, -5.0, 1.5));
    // Gate centre is 3 m up; 1.5 m is 1.5 m under the centre, far under the sill.
    let events = fly_to(&mut app, DVec3::new(0.0, 5.0, 1.5));
    let p = progress(&app);
    assert_eq!(missed(&events), [0]);
    assert_eq!((p.passed, p.missed), (0, 1), "under the gate is not through it");
}

/// A slanted crossing is scored where the path actually pierces the plane, not at
/// either end of the tick.
#[test]
fn the_crossing_point_is_interpolated() {
    let mut app = rig(straight_course(), DVec3::new(0.0, -4.0, 4.0));
    // Both sample points are outside the opening vertically — over the top bar and
    // under the sill — but the straight line between them crosses y = 0 at z = 3:
    // dead centre. (Shallow enough that the hull clears the top bar on the way.)
    fly_to(&mut app, DVec3::new(0.0, 4.0, 2.0));
    assert_eq!(progress(&app).passed, 1, "the pass is decided on the interpolated crossing");
}
