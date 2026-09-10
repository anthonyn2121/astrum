//! Gate scoring: what counts as flying through a gate.
//!
//! `track_gates` is driven here over hand-placed positions in a minimal app — no
//! JSBSim, so the drone can be put exactly where a case needs it. That matters
//! because the interesting inputs are the ones a real flight rarely produces on
//! demand: a tick that steps clean over the 7 cm frame, a backwards crossing, a pass
//! outside the hole.

use astrum_race::components::{Drone, DroneState, RaceCourse, RaceProgress};
use astrum_race::course::{Course, CourseParams};
use astrum_race::gate::Gate;
use astrum_race::systems::track_gates;
use bevy::math::DVec3;
use bevy::prelude::*;

/// A two-gate course, both facing North, 20 m apart at 3 m — hand-built rather than
/// sampled, so a case's geometry is exactly what it says.
fn straight_course() -> Course {
    Course {
        gates: vec![
            Gate::new(0, DVec3::new(0.0, 0.0, 3.0), 0.0),
            Gate::new(1, DVec3::new(0.0, 20.0, 3.0), 0.0),
        ],
        seed: 0,
        params: CourseParams::default(),
        samples: 0,
        backtracks: 0,
    }
}

/// A minimal app: the gate tracker, one drone, and the course above.
fn scoring_app(course: Course) -> App {
    let mut app = App::new();
    app.insert_resource(RaceCourse(course))
        .insert_resource(RaceProgress::default())
        .add_systems(Update, track_gates);
    app.world_mut().spawn((Drone { id: 1 }, DroneState::default()));
    app
}

/// Move the drone to `pos` (ENU) and run one tick of scoring.
fn fly_to(app: &mut App, pos: DVec3) {
    let mut q = app.world_mut().query::<&mut DroneState>();
    {
        let world = app.world_mut();
        for mut st in q.iter_mut(world) {
            st.pos_enu = pos;
        }
    }
    app.update();
}

fn progress(app: &App) -> RaceProgress {
    *app.world().resource::<RaceProgress>()
}

/// Flying straight down the line takes both gates, in order.
#[test]
fn straight_flight_passes_every_gate() {
    let mut app = scoring_app(straight_course());
    for i in 0..=60 {
        // -10 m to +50 m North, 1 m a tick: through gate 0 at y=0, gate 1 at y=20.
        fly_to(&mut app, DVec3::new(0.0, -10.0 + i as f64, 3.0));
    }
    let p = progress(&app);
    assert_eq!(p.passed, 2, "both gates should be scored");
    assert_eq!(p.missed, 0);
    assert!(p.finished, "the course should be complete");
}

/// A tick that steps clean over the frame still counts. This is the case a
/// point-inside-the-frame test would miss: 20 m/s at 120 Hz is 17 cm a tick, and a
/// coarser controller step can be metres.
#[test]
fn a_single_long_step_across_the_plane_counts() {
    let mut app = scoring_app(straight_course());
    fly_to(&mut app, DVec3::new(0.0, -8.0, 3.0)); // establishes the reference tick
    fly_to(&mut app, DVec3::new(0.0, 8.0, 3.0)); // 16 m in one tick, straight through
    let p = progress(&app);
    assert_eq!(p.passed, 1, "the crossing is on the segment, not at the sample points");
    assert_eq!(p.next_gate, 1);
}

/// Going round the outside is a miss, not a pass — and the gate stays pending.
#[test]
fn passing_beside_the_opening_is_a_miss() {
    let mut app = scoring_app(straight_course());
    // 1.5 m to the right of the centreline: past the 0.39 m half-opening, but well
    // inside the radius where a crossing is still *about* this gate.
    fly_to(&mut app, DVec3::new(1.5, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(1.5, 5.0, 3.0));
    let p = progress(&app);
    assert_eq!((p.passed, p.missed, p.next_gate), (0, 1, 0), "a miss must not advance the course");

    // Coming back and doing it properly still scores.
    fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 5.0, 3.0));
    assert_eq!(progress(&app).passed, 1, "a second attempt should score");
}

/// Crossing the plane far from the gate is not a miss — it is unrelated flying.
#[test]
fn crossing_the_plane_far_away_is_not_scored() {
    let mut app = scoring_app(straight_course());
    fly_to(&mut app, DVec3::new(25.0, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(25.0, 5.0, 3.0));
    let p = progress(&app);
    assert_eq!((p.passed, p.missed), (0, 0), "25 m off to the side is not an attempt on the gate");
}

/// Only downstream crossings count: backing through a gate scores nothing.
#[test]
fn flying_backwards_through_a_gate_scores_nothing() {
    let mut app = scoring_app(straight_course());
    fly_to(&mut app, DVec3::new(0.0, 5.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0)); // through the hole, wrong way
    let p = progress(&app);
    assert_eq!((p.passed, p.missed, p.next_gate), (0, 0, 0), "the wrong way is not a pass");
}

/// Gates must be taken in order: flying through gate 1 first scores nothing, and
/// gate 0 is still the one being watched.
#[test]
fn gates_must_be_taken_in_order() {
    let mut app = scoring_app(straight_course());
    fly_to(&mut app, DVec3::new(0.0, 15.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 25.0, 3.0)); // straight through gate 1
    let p = progress(&app);
    assert_eq!((p.passed, p.next_gate), (0, 0), "gate 1 does not count while gate 0 is pending");

    // Gate 0 is behind the drone now; coming back round scores it, then gate 1.
    fly_to(&mut app, DVec3::new(0.0, -5.0, 3.0));
    fly_to(&mut app, DVec3::new(0.0, 5.0, 3.0));
    assert_eq!(progress(&app).next_gate, 1, "gate 0 scored, gate 1 now pending");
    fly_to(&mut app, DVec3::new(0.0, 15.0, 3.0)); // the new gate's reference tick
    fly_to(&mut app, DVec3::new(0.0, 25.0, 3.0));
    assert!(progress(&app).finished, "and then the course is complete");
}

/// The opening is scored at the height it is at: ducking under the sill is a miss.
#[test]
fn flying_under_the_sill_is_a_miss() {
    let mut app = scoring_app(straight_course());
    // Gate centre is 3 m up with a 0.39 m half-opening; 1.5 m is under the sill.
    fly_to(&mut app, DVec3::new(0.0, -5.0, 1.5));
    fly_to(&mut app, DVec3::new(0.0, 5.0, 1.5));
    let p = progress(&app);
    assert_eq!((p.passed, p.missed), (0, 1), "under the gate is not through it");
}

/// A slanted crossing is scored where the path actually pierces the plane, not at
/// either end of the tick.
#[test]
fn the_crossing_point_is_interpolated() {
    let mut app = scoring_app(straight_course());
    // Both sample points are well outside the opening laterally, but the straight
    // line between them crosses y = 0 at x = 0 — dead centre.
    fly_to(&mut app, DVec3::new(-4.0, -4.0, 3.0));
    fly_to(&mut app, DVec3::new(4.0, 4.0, 3.0));
    assert_eq!(progress(&app).passed, 1, "the pass is decided on the interpolated crossing");
}
