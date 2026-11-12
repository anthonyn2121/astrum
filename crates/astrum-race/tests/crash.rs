//! Crash termination: what ends a run, and what the run is then reported as.
//!
//! Most of this is driven over hand-placed poses in a rig — the race's own tick
//! systems and no JSBSim — because hand-placing is the only way to get the
//! interesting cases on demand: JSBSim will not fly a drone into an upright at a
//! chosen offset just because a test would like it to. The geometry underneath is
//! covered in `airframe.rs`; what is tested here is the *behaviour*: that a contact
//! sets the right outcome, reports it, ends the run, and beats the gate scorer to it.
//!
//! The pose cases are all written as **passes**. At the instant the drone's CG is in a
//! gate's plane, all four of its rotors — the widest thing about it — are 0.165 m away
//! from that plane, so a single hand-picked pose in the opening touches nothing however
//! badly it is placed. A verdict here has to come from flying through.
//!
//! The last test flies the real thing into the ground with JSBSim, which is what says
//! the whole chain works in the app the binary runs.

mod common;

use astrum_proto::{EndReason, EventKind, RaceOutcome};
use astrum_race::airframe::{Hull, HALF_SPAN_M};
use astrum_race::components::RaceProgress;
use astrum_race::course::Course;
use astrum_race::gate::{Gate, OPENING_M};
use astrum_race::scenario::RaceScenario;
use astrum_sim::events::EventLog;
use astrum_sim::RunMode;
use bevy::math::DVec3;
use bevy::prelude::*;
use common::{
    exited, fdm_guard, flat_course, fly_to, place, progress, rig, scenario_toml, straight_course,
};

/// How far off a gate's centreline the drone can be and still fit, square on.
const CLEARANCE_M: f64 = OPENING_M / 2.0 - HALF_SPAN_M;

/// A rig on [`straight_course`] with the drone 5 m upstream of gate 0.
fn crash_rig() -> App {
    rig(straight_course(), DVec3::new(0.0, -5.0, 2.0))
}

/// Fly a gate at a fixed lateral offset and crab angle, in 5 cm ticks — about what a
/// 120 Hz step covers at racing speed. Stops early if the run ends, since a crashed
/// drone does not keep flying, and reports how it went along with the events.
fn fly_the_gate(
    gate: &Gate,
    course: Course,
    right: f64,
    crab_rad: f64,
) -> (App, RaceProgress, Vec<EventKind>) {
    let mut app = rig(course, gate.to_world(DVec3::new(right, 0.0, -0.5)));
    let mut events = Vec::new();
    for i in -10..=10 {
        let at = gate.to_world(DVec3::new(right, 0.0, f64::from(i) * 0.05));
        events.extend(place(&mut app, at, 0.0, 0.0, gate.yaw_rad + crab_rad));
        if progress(&app).finished() {
            break;
        }
    }
    let p = progress(&app);
    (app, p, events)
}

fn crashed(events: &[EventKind]) -> Vec<(RaceOutcome, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::Crashed { outcome, gate, .. } => Some((*outcome, *gate)),
            _ => None,
        })
        .collect()
}

fn run_ended(events: &[EventKind]) -> Vec<EndReason> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::RunEnded { reason, .. } => Some(*reason),
            _ => None,
        })
        .collect()
}

/// The ground ends a run the moment the gear touches, and that ends it for good.
///
/// The threshold is the airframe, not a tuned tolerance: level, the feet are 0.1524 m
/// below the CG, so `min_z <= 0` fires at exactly that CG altitude.
#[test]
fn touching_the_ground_ends_the_run() {
    let mut app = crash_rig();

    fly_to(&mut app, DVec3::new(0.0, -5.0, 1.0));
    assert_eq!(progress(&app).outcome, RaceOutcome::Running, "1 m up is not a crash");
    fly_to(&mut app, DVec3::new(0.0, -5.0, 0.16));
    assert_eq!(progress(&app).outcome, RaceOutcome::Running, "8 mm of daylight is still flying");
    assert!(!exited(&app));

    let events = fly_to(&mut app, DVec3::new(0.0, -5.0, 0.1524));
    let p = progress(&app);
    assert_eq!(p.outcome, RaceOutcome::CrashedGround, "the feet reached the ground plane");
    assert!(exited(&app), "a crash must end the run");
    assert_eq!(
        crashed(&events),
        [(RaceOutcome::CrashedGround, 0)],
        "reported once, flying to gate 0"
    );
    assert_eq!(run_ended(&events), [EndReason::Crashed]);
}

/// It is the lowest part of the *hull* that has to clear the ground, not the CG. At
/// 0.20 m a level drone has 5 cm to spare; on its side, an arm has swung 8 cm below the
/// floor and the same CG altitude is a crash.
#[test]
fn the_ground_test_uses_the_lowest_part_of_the_hull() {
    let mut level = crash_rig();
    fly_to(&mut level, DVec3::new(0.0, -5.0, 0.2));
    assert_eq!(progress(&level).outcome, RaceOutcome::Running, "level at 0.20 m clears the ground");

    let mut knife = crash_rig();
    place(&mut knife, DVec3::new(0.0, -5.0, 0.2), std::f64::consts::FRAC_PI_2, 0.0, 0.0);
    assert_eq!(
        progress(&knife).outcome,
        RaceOutcome::CrashedGround,
        "rolled onto its side, a rotor is below the floor"
    );
}

/// Drifting wide of the opening puts a prop into an upright, which ends the run as a
/// gate strike and scores nothing.
#[test]
fn clipping_a_gate_ends_the_run() {
    let course = straight_course();
    let gate = course.gates[0];
    let (app, p, events) = fly_the_gate(&gate, course, CLEARANCE_M + 0.02, 0.0);
    assert_eq!(p.outcome, RaceOutcome::CrashedGate, "2 cm past the clearance is a strike");
    assert_eq!(p.passed, 0, "a strike is not a pass");
    assert!(exited(&app), "a crash must end the run");
    assert_eq!(crashed(&events), [(RaceOutcome::CrashedGate, 0)], "names the gate it struck");
    assert_eq!(run_ended(&events), [EndReason::Crashed]);
}

/// Down the middle, nothing is touched and the gate scores — the negative control for
/// every case above. A crash detector that fired here would make the course unflyable.
///
/// Including **crabbed**, at the 60 deg a 120 deg course turn can leave on the approach
/// and at the 45 deg that is the airframe's worst heading. That is the property the
/// opening was sized for, checked here in the app rather than only in the geometry.
#[test]
fn flying_the_middle_of_a_gate_never_crashes() {
    let course = straight_course();
    let gate = course.gates[0];

    for crab_deg in [0.0, 45.0, 60.0, -60.0] {
        let (app, p, _) = fly_the_gate(&gate, course.clone(), 0.0, f64::to_radians(crab_deg));
        assert_eq!(p.outcome, RaceOutcome::Running, "{crab_deg} deg crabbed is not a crash");
        assert_eq!((p.passed, p.next_gate), (1, 1), "{crab_deg} deg crabbed still counts");
        assert!(!p.finished() && !exited(&app), "there is another gate to fly");
    }
}

/// Clipping a gate while the CG goes through the hole is a **crash, not a pass**.
///
/// This is what the order of the race's tick buys, and it is not a corner case: the
/// scorer asks where the CG pierced the plane, and the flyable corridor is narrower
/// than the hole. The CG may be anywhere in ±`OPENING_M/2` and score, but the props
/// reach `HALF_SPAN_M` either side of it, so anything past `CLEARANCE_M` is a strike.
/// 30 cm off centre is inside the hole and well into an upright. The centred pass is
/// the control — same track, same two ticks, same crossing, and it scores.
#[test]
fn a_crash_beats_a_pass_on_the_same_tick() {
    let course = straight_course();
    let gate = course.gates[0];
    let one_tick_through = |right: f64| {
        let mut app = rig(course.clone(), gate.to_world(DVec3::new(right, 0.0, -0.5)));
        let events = place(&mut app, gate.to_world(DVec3::new(right, 0.0, 0.5)), 0.0, 0.0, 0.0);
        (progress(&app), events)
    };

    let wide = 0.30;
    assert!(wide < OPENING_M / 2.0, "the CG must cross inside the hole, or there is no race");
    assert!(wide > CLEARANCE_M, "and outside the corridor, or there is no crash");

    let (clipped, events) = one_tick_through(wide);
    assert_eq!(clipped.outcome, RaceOutcome::CrashedGate, "a prop is in the upright");
    assert_eq!((clipped.passed, clipped.next_gate), (0, 0), "the scorer must not credit a crash");
    assert!(
        !events.iter().any(|e| matches!(e, EventKind::GatePassed { .. })),
        "and must not announce one: {events:?}"
    );

    let (centred, events) = one_tick_through(0.0);
    assert_eq!(centred.outcome, RaceOutcome::Running);
    assert_eq!(centred.passed, 1, "the same crossing, taken centred, is a pass");
    assert!(events.iter().any(|e| matches!(e, EventKind::GatePassed { gate: 0, .. })));
}

/// Every gate is solid, not just the pending one: you can clip the gate you have
/// already passed on the way out, or the next one round a tight corner.
#[test]
fn striking_a_gate_you_have_not_reached_still_crashes() {
    let course = straight_course();
    let ahead = course.gates[1];
    let (_, p, events) = fly_the_gate(&ahead, course, CLEARANCE_M + 0.02, 0.0);
    assert_eq!(
        p.outcome,
        RaceOutcome::CrashedGate,
        "gate 1's frame is solid while gate 0 is pending"
    );
    assert_eq!(p.next_gate, 0, "and it was never the gate being scored");
    assert_eq!(crashed(&events), [(RaceOutcome::CrashedGate, 1)]);
}

/// A tick long enough to step clean over a gate still crashes, because the step is
/// subdivided and the true hull tested along it. Both endpoints of this tick are
/// verified clear first — without substepping there would be nothing to find.
#[test]
fn a_long_step_cannot_tunnel_a_gate() {
    let course = straight_course();
    let gate = course.gates[0];
    // 4 m in one tick, 5 cm past the clearance: 480 km/h at 120 Hz, but the same
    // arithmetic covers a controller stepping at 10 Hz.
    let right = CLEARANCE_M + 0.05;
    let (from, to) =
        (gate.to_world(DVec3::new(right, 0.0, -2.0)), gate.to_world(DVec3::new(right, 0.0, 2.0)));
    for at in [from, to] {
        assert!(
            !Hull::at(at, 0.0, 0.0, gate.yaw_rad).hits_gate(&gate),
            "the endpoint at {at} is clear, so only the step between them can be a crash"
        );
    }

    let mut app = rig(course, from);
    assert_eq!(progress(&app).outcome, RaceOutcome::Running, "the tick starts clear of the gate");
    fly_to(&mut app, to);
    assert_eq!(
        progress(&app).outcome,
        RaceOutcome::CrashedGate,
        "the drone teleported through a gate"
    );
}

/// The whole chain, on the real thing: nothing flies the drone, so JSBSim drops it
/// from the gate height to the ground. This is the test that says crash termination
/// works in the app the binary runs — a real FDM, the sim's own step and pose, the
/// real order, the real end of the run.
#[test]
fn an_unflown_drone_crashes_and_the_run_ends_as_crashed() {
    let _fdm = fdm_guard();
    let race = RaceScenario::from_toml_str(&scenario_toml("", "", &flat_course(30.0, 6)))
        .unwrap()
        .resolve(None)
        .unwrap();
    let mut app = astrum_race::build_race_app(race, RunMode::Batch);
    app.finish();
    app.cleanup();

    app.update();
    let start = app.world().resource::<RaceProgress>();
    assert_eq!(start.outcome, RaceOutcome::Running);

    let mut events = Vec::new();
    let mut ticks = 0;
    while app.should_exit().is_none() && ticks < 7200 {
        app.update();
        events.extend(app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()));
        ticks += 1;
    }

    assert!(ticks < 7200, "the fall never ended the run");
    let p = progress(&app);
    assert_eq!(p.outcome, RaceOutcome::CrashedGround, "it fell to the ground");
    assert_eq!(p.passed, 0, "it went down, not through a gate");
    assert_eq!(crashed(&events), [(RaceOutcome::CrashedGround, 0)]);
    assert_eq!(run_ended(&events), [EndReason::Crashed], "the end is reported as a crash");
    assert!(matches!(app.should_exit(), Some(AppExit::Success)));
}
