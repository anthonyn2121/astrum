//! Crash termination: what ends a run, and what the run is then reported as.
//!
//! Most of this is driven over hand-placed poses in a minimal app — the geometry
//! underneath is covered in `airframe.rs`, so what is being tested here is the
//! *behaviour*: that a contact sets the right [`Outcome`], that it ends the run, and
//! that it beats the gate scorer to it. Hand-placing is the only way to get the
//! interesting cases on demand: JSBSim will not fly a drone into an upright at a
//! chosen offset just because a test would like it to.
//!
//! The pose cases are all written as **passes**. At the instant the drone's CG is in a
//! gate's plane, all four of its rotors — the widest thing about it — are 0.165 m away
//! from that plane, so a single hand-picked pose in the opening touches nothing however
//! badly it is placed. A verdict here has to come from flying through.
//!
//! The last test flies the real thing into the ground with JSBSim, which is what says
//! the whole chain works in the app the binary runs.

mod common;

use astrum_race::airframe::Hull;
use astrum_race::components::{Drone, DroneState, Outcome, RaceCourse, RaceProgress};
use astrum_race::control::FlightCommand;
use astrum_race::course::{Course, CourseParams};
use astrum_race::gate::{Gate, OPENING_M};
use astrum_race::systems::{detect_crash, stop_when_finished, track_gates};
use astrum_race::{build_app, build_app_with_course};
use bevy::math::DVec3;
use bevy::prelude::*;
use common::{drone_state, fdm_guard, flat_course_at, progress, race_scenario};

/// How far off a gate's centreline the drone can be and still fit: 10.9 cm, square on.
const CLEARANCE_M: f64 = OPENING_M / 2.0 - astrum_race::airframe::HALF_SPAN_M;

/// Two gates facing North, 20 m apart at 3 m — the same track `scoring.rs` uses, so a
/// case's geometry is exactly what it says rather than whatever a seed drew.
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

/// A minimal app holding the three systems whose *order* is the thing under test:
/// crash detection, then gate scoring, then the single exit. No JSBSim, so the drone
/// goes exactly where a case puts it.
fn crash_app(course: Course) -> App {
    let mut app = App::new();
    app.insert_resource(RaceCourse(course))
        .insert_resource(RaceProgress::default())
        .add_systems(Update, (detect_crash, track_gates, stop_when_finished).chain());
    app.world_mut().spawn((Drone { id: 1 }, DroneState::default()));
    app
}

/// Put the drone at `pos` in the given attitude and run one tick.
fn place(app: &mut App, pos: DVec3, phi: f64, theta: f64, psi: f64) {
    let mut q = app.world_mut().query::<&mut DroneState>();
    {
        let world = app.world_mut();
        for mut st in q.iter_mut(world) {
            st.pos_enu = pos;
            st.phi_rad = phi;
            st.theta_rad = theta;
            st.psi_rad = psi;
        }
    }
    app.update();
}

/// [`place`] for a level drone facing North, which is most cases.
fn fly_to(app: &mut App, pos: DVec3) {
    place(app, pos, 0.0, 0.0, 0.0);
}

/// Fly a gate at a fixed lateral offset and crab angle, in 5 cm ticks — about what a
/// 120 Hz step covers at racing speed. Stops early if the run ends, since a crashed
/// drone does not keep flying, and reports how it went.
fn fly_the_gate(app: &mut App, gate: &Gate, right: f64, crab_rad: f64) -> RaceProgress {
    for i in -10..=10 {
        let at = gate.to_world(DVec3::new(right, 0.0, f64::from(i) * 0.05));
        place(app, at, 0.0, 0.0, gate.yaw_rad + crab_rad);
        if progress(app).finished {
            break;
        }
    }
    progress(app)
}

fn exited(app: &App) -> bool {
    app.should_exit().is_some()
}

/// The ground ends a run the moment the gear touches, and that ends it for good.
///
/// The threshold is the airframe, not a tuned tolerance: level, the feet are 0.1524 m
/// below the CG, so `min_z <= 0` fires at exactly that CG altitude.
#[test]
fn touching_the_ground_ends_the_run() {
    let mut app = crash_app(straight_course());

    fly_to(&mut app, DVec3::new(0.0, -5.0, 2.0));
    assert_eq!(progress(&app).outcome, Outcome::Running, "5 m up is not a crash");
    fly_to(&mut app, DVec3::new(0.0, -5.0, 0.16));
    assert_eq!(progress(&app).outcome, Outcome::Running, "8 mm of daylight is still flying");
    assert!(!exited(&app));

    fly_to(&mut app, DVec3::new(0.0, -5.0, 0.1524));
    let p = progress(&app);
    assert_eq!(p.outcome, Outcome::CrashedGround, "the feet reached the ground plane");
    assert!(p.finished && exited(&app), "a crash must end the run");
}

/// It is the lowest part of the *hull* that has to clear the ground, not the CG. At
/// 0.20 m a level drone has 5 cm to spare; on its side, an arm has swung 8 cm below the
/// floor and the same CG altitude is a crash.
#[test]
fn the_ground_test_uses_the_lowest_part_of_the_hull() {
    let mut level = crash_app(straight_course());
    fly_to(&mut level, DVec3::new(0.0, -5.0, 0.2));
    assert_eq!(progress(&level).outcome, Outcome::Running, "level at 0.20 m clears the ground");

    let mut knife = crash_app(straight_course());
    place(&mut knife, DVec3::new(0.0, -5.0, 0.2), std::f64::consts::FRAC_PI_2, 0.0, 0.0);
    assert_eq!(
        progress(&knife).outcome,
        Outcome::CrashedGround,
        "rolled onto its side, a rotor is below the floor"
    );
}

/// Drifting wide of the opening puts a prop into an upright, which ends the run as a
/// gate strike and scores nothing.
#[test]
fn clipping_a_gate_ends_the_run() {
    let course = straight_course();
    let gate = course.gates[0];
    let mut app = crash_app(course);

    let p = fly_the_gate(&mut app, &gate, CLEARANCE_M + 0.02, 0.0);
    assert_eq!(p.outcome, Outcome::CrashedGate, "2 cm past the clearance is a strike");
    assert_eq!(p.passed, 0, "a strike is not a pass");
    assert!(exited(&app), "a crash must end the run");
}

/// Down the middle, nothing is touched and the gate scores — the negative control for
/// every case above. A crash detector that fired here would make the course unflyable.
///
/// Including **crabbed**, at the 60 deg a 120 deg course turn can leave on the approach
/// and at the 45 deg that is the airframe's worst heading. That is the property the 31"
/// opening was sized for, checked here in the app rather than only in the geometry.
#[test]
fn flying_the_middle_of_a_gate_never_crashes() {
    let course = straight_course();
    let gate = course.gates[0];

    for crab_deg in [0.0, 45.0, 60.0, -60.0] {
        let mut app = crash_app(course.clone());
        let p = fly_the_gate(&mut app, &gate, 0.0, f64::to_radians(crab_deg));
        assert_eq!(p.outcome, Outcome::Running, "{crab_deg} deg crabbed is not a crash");
        assert_eq!((p.passed, p.next_gate), (1, 1), "{crab_deg} deg crabbed still counts");
        assert!(!p.finished && !exited(&app), "there is another gate to fly");
    }
}

/// Clipping a gate while the CG goes through the hole is a **crash, not a pass**.
///
/// This is what the system ordering in `build_app` buys, and it is not a corner case:
/// the scorer asks where the CG pierced the plane, and the flyable corridor is far
/// narrower than the hole. In a 0.7874 m opening the CG may be anywhere in ±0.3937 m
/// and score, but the props reach 0.28448 m either side of it, so anything past
/// ±0.10922 m is a strike. 20 cm off centre is comfortably inside the hole and
/// comfortably into an upright. The centred pass below is the control — same track,
/// same two ticks, same crossing, and it scores.
#[test]
fn a_crash_beats_a_pass_on_the_same_tick() {
    let course = straight_course();
    let gate = course.gates[0];
    let one_tick_through = |right: f64| {
        let mut app = crash_app(course.clone());
        place(&mut app, gate.to_world(DVec3::new(right, 0.0, -0.5)), 0.0, 0.0, gate.yaw_rad);
        place(&mut app, gate.to_world(DVec3::new(right, 0.0, 0.5)), 0.0, 0.0, gate.yaw_rad);
        progress(&app)
    };

    let wide = 0.20;
    assert!(wide < OPENING_M / 2.0, "the CG must cross inside the hole, or there is no race");
    assert!(wide > CLEARANCE_M, "and outside the corridor, or there is no crash");

    let clipped = one_tick_through(wide);
    assert_eq!(clipped.outcome, Outcome::CrashedGate, "a prop is in the upright");
    assert_eq!((clipped.passed, clipped.next_gate), (0, 0), "the scorer must not credit a crash");

    let centred = one_tick_through(0.0);
    assert_eq!(centred.outcome, Outcome::Running);
    assert_eq!(centred.passed, 1, "the same crossing, taken centred, is a pass");
}

/// Every gate is solid, not just the pending one: you can clip the gate you have
/// already passed on the way out, or the next one round a tight corner.
#[test]
fn striking_a_gate_you_have_not_reached_still_crashes() {
    let course = straight_course();
    let ahead = course.gates[1];
    let mut app = crash_app(course);

    let p = fly_the_gate(&mut app, &ahead, CLEARANCE_M + 0.02, 0.0);
    assert_eq!(p.outcome, Outcome::CrashedGate, "gate 1's frame is solid while gate 0 is pending");
    assert_eq!(p.next_gate, 0, "and it was never the gate being scored");
}

/// A tick long enough to step clean over a gate still crashes, because the step is
/// subdivided and the true hull tested along it. Both endpoints of this tick are
/// verified clear first — without substepping there would be nothing to find.
#[test]
fn a_long_step_cannot_tunnel_a_gate() {
    let course = straight_course();
    let gate = course.gates[0];
    // 4 m in one tick, 9 cm wide of the 10.9 cm clearance: 480 km/h at 120 Hz, but the
    // same arithmetic covers a controller stepping at 10 Hz.
    let (from, to) =
        (gate.to_world(DVec3::new(0.2, 0.0, -2.0)), gate.to_world(DVec3::new(0.2, 0.0, 2.0)));
    for at in [from, to] {
        assert!(
            !Hull::at(at, 0.0, 0.0, gate.yaw_rad).hits_gate(&gate),
            "the endpoint at {at} is clear, so only the step between them can be a crash"
        );
    }

    let mut app = crash_app(course);
    fly_to(&mut app, from);
    assert_eq!(progress(&app).outcome, Outcome::Running, "the tick starts clear of the gate");
    fly_to(&mut app, to);
    assert_eq!(progress(&app).outcome, Outcome::CrashedGate, "the drone teleported through a gate");
}

/// A run that never launched ends immediately as [`Outcome::SpawnFailed`] rather than
/// spinning forever: with no drone there is no clock, so no other terminator can fire.
///
/// A real app, but an empty course, so `spawn_drone` bails before it builds an FDM —
/// which is why this one does not need the JSBSim guard.
#[test]
fn a_failed_spawn_stops_the_app_at_once() {
    let scenario = race_scenario(CourseParams::default());
    let empty = Course {
        gates: vec![],
        seed: scenario.sim.seed,
        params: scenario.course,
        samples: 0,
        backtracks: 0,
    };
    let mut app = build_app_with_course(scenario, empty);
    app.update();

    assert_eq!(progress(&app).outcome, Outcome::SpawnFailed);
    assert!(exited(&app), "an app with nothing to fly must not run on");
}

/// The whole chain, on the real thing: command the drone below the ground and JSBSim
/// flies it there. This is the test that says crash termination works in the app the
/// binary runs — a real FDM, the real pose sync, the real system order.
#[test]
fn flying_into_the_ground_ends_a_real_run() {
    let _fdm = fdm_guard();
    let mut app = build_app(race_scenario(flat_course_at(30.0))).unwrap();
    app.update();
    let start = drone_state(&mut app).unwrap();
    assert!(
        start.pos_enu.z > 25.0,
        "should start near the gate height, at {:.1} m",
        start.pos_enu.z
    );

    // Ask for an altitude no amount of collective can hold. Nothing rewrites the
    // command (no transport, no raw lease), so this stands for the rest of the run.
    {
        let mut q = app.world_mut().query::<&mut FlightCommand>();
        let world = app.world_mut();
        for mut cmd in q.iter_mut(world) {
            cmd.alt_m = -50.0;
        }
    }

    // 60 s at 120 Hz is far longer than a 30 m descent needs.
    let mut steps = 0;
    while app.should_exit().is_none() && steps < 7200 {
        app.update();
        steps += 1;
    }
    let end = drone_state(&mut app).unwrap();

    assert!(steps < 7200, "the descent never terminated the run");
    let p = progress(&app);
    assert_eq!(p.outcome, Outcome::CrashedGround, "it flew into the ground");
    assert_eq!(p.passed, 0, "it went down, not through a gate");
    // Caught at the ground plane, not after falling through it, and not still at height.
    assert!(
        end.pos_enu.z > 0.0 && end.pos_enu.z < 0.2,
        "should be caught as the gear touches, at {:.3} m",
        end.pos_enu.z
    );
    assert!(end.vel_enu.z < 0.0, "should still be descending, at {:.2} m/s", end.vel_enu.z);
    // JSBSim's own gear model agrees with the hull: it logs GEAR_CONTACT on the same
    // step, which is what says `min_z <= 0` needs no tolerance constant.

    // First writer wins, so the verdict is stable. (A manual `update()` runs the whole
    // schedule regardless of `AppExit` — that only stops `app.run()`'s loop — so the FDM
    // does step on; what must not change is how the run is reported.)
    app.update();
    assert_eq!(progress(&app).outcome, Outcome::CrashedGround, "the outcome must not be rewritten");
    assert!(exited(&app), "and the exit stands");
}
