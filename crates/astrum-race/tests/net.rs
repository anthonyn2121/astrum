//! The Zenoh bridge, driven through the real app: the course and progress a
//! controller sees, the telemetry it flies from, and the commands it sends back.
//!
//! Every test runs on one in-process [`Session`], where a publisher delivers to a
//! subscriber on the same session synchronously — so no broker, no ports, and no
//! sleeping. Retry loops guard only against residual scheduling lag.
//!
//! These tests are the reason `astrum_race::net::install` lives in the library
//! rather than in `main.rs`: the bridge is wired here exactly as the binary wires
//! it.
//!
//! One trap worth knowing, since it decides the shape of the fixtures below: a
//! **telemetry** subscriber on the same session that nobody drains will block the
//! publisher once its FIFO channel fills (~256 frames, about 2 s of sim), because
//! `CongestionControl::Drop` governs Zenoh's network queues and not a local handler.
//! So only the telemetry test declares one; the long-running command tests use
//! [`race_app`], which opens the transport without any subscriber. The racing
//! channels are immune — they are rings of one.

mod common;

use astrum_net::{CommandSender, RaceSubscriber, Session, TelemetrySubscriber};
use astrum_proto::{
    CommandMsg, Control, RaceCourseMsg, RaceOutcome, RaceStateMsg, TelemetrySample, VehicleKind,
};
use astrum_race::components::{Outcome, RaceProgress};
use astrum_race::{build_app_with_course, course, generate_course};
use bevy::prelude::App;
use common::{
    drone_state, fdm_guard, flat_course_at, flight_command, progress, race_scenario, raw_override,
};

/// The drone id every fixture scenario uses.
const DRONE_ID: u32 = 1;

/// Stand up a racing app with its transport installed on a fresh in-process
/// session, and no subscribers.
fn race_app(params: course::CourseParams) -> (App, Session) {
    let scenario = race_scenario(params);
    let course = generate_course(&scenario).expect("fixture course must be feasible");
    let mut app = build_app_with_course(scenario, course);
    let session = Session::in_process().expect("in-process session");
    astrum_race::net::install(&mut app, &session);
    (app, session)
}

/// The same, plus a race subscriber declared *before* the first tick, so nothing
/// the app publishes on tick 0 is missed.
fn watched_app(params: course::CourseParams) -> (App, Session, RaceSubscriber) {
    let (app, session) = race_app(params);
    let mut race = RaceSubscriber::connect(&session).unwrap();
    race.subscribe(DRONE_ID).unwrap();
    (app, session, race)
}

/// Pump the app until both racing channels have delivered, or give up.
fn pump_for_race(
    app: &mut App,
    race: &RaceSubscriber,
    max_ticks: u32,
) -> (Option<RaceCourseMsg>, Option<RaceStateMsg>) {
    let (mut course, mut state) = (None, None);
    for _ in 0..max_ticks {
        app.update();
        course = course.or(race.try_recv_course().unwrap());
        state = state.or(race.try_recv_state().unwrap());
        if course.is_some() && state.is_some() {
            return (course, state);
        }
    }
    (course, state)
}

#[test]
fn the_course_reaches_a_subscriber_and_matches_the_generated_track() {
    let _fdm = fdm_guard();
    let (mut app, _session, race) = watched_app(Default::default());
    let (course, _) = pump_for_race(&mut app, &race, 300);
    let course = course.expect("no course was published");

    // Compare against the generator, not against constants: the point of the
    // message is that a controller sees *this seed's* track.
    let expected = generate_course(&race_scenario(Default::default())).unwrap();
    assert_eq!(course.drone_id, DRONE_ID);
    assert_eq!(course.seed, expected.seed);
    assert_eq!(course.gates.len(), expected.gates.len());
    for (got, want) in course.gates.iter().zip(&expected.gates) {
        assert_eq!(got.index, want.index);
        assert!((got.center_e - want.center.x).abs() < 1e-12);
        assert!((got.center_n - want.center.y).abs() < 1e-12);
        assert!((got.center_u - want.center.z).abs() < 1e-12);
        assert!((got.yaw_rad - want.yaw_rad).abs() < 1e-12);
    }
    // Hardware travels with the course so a controller need not hard-code it.
    assert!((course.opening_m - astrum_race::gate::OPENING_M).abs() < 1e-12);
    assert!((course.bar_m - astrum_race::gate::BAR_M).abs() < 1e-12);
}

#[test]
fn the_course_is_published_on_the_very_first_tick() {
    // A controller that is already listening must not wait a republish period for
    // the track it needs before it can do anything at all.
    let _fdm = fdm_guard();
    let (mut app, _session, race) = watched_app(Default::default());
    app.update();
    assert!(race.try_recv_course().unwrap().is_some(), "the first tick published no course");
}

#[test]
fn the_course_is_republished_for_a_late_joiner() {
    // Zenoh has no late-joiner replay, so a controller started after the race must
    // still receive the track. Subscribe *after* the app has been running.
    let _fdm = fdm_guard();
    let expected_seed = race_scenario(Default::default()).sim.seed;
    let (mut app, session) = race_app(Default::default());

    // Run for a while with nobody listening — the first publication is lost.
    for _ in 0..150 {
        app.update();
    }

    let mut race = RaceSubscriber::connect(&session).unwrap();
    race.subscribe(DRONE_ID).unwrap();
    // One republish period is 1 simulated second = 120 ticks at the fixture rate.
    let (got, _) = pump_for_race(&mut app, &race, 400);
    assert_eq!(got.map(|c| c.seed), Some(expected_seed), "a late joiner never got the course");
}

#[test]
fn race_state_mirrors_the_progress_resource() {
    let _fdm = fdm_guard();
    let (mut app, _session, race) = watched_app(Default::default());
    let gate_count = app.world().resource::<astrum_race::components::RaceCourse>().0.gates.len();

    // Drive the resource directly: this test is about the projection onto the wire,
    // not about flying, and the scoring rules are covered in `scoring.rs`.
    *app.world_mut().resource_mut::<RaceProgress>() = RaceProgress {
        next_gate: 3,
        passed: 3,
        missed: 2,
        last_pass_s: 4.25,
        ..Default::default()
    };

    let (_, state) = pump_for_race(&mut app, &race, 300);
    let state = state.expect("no race state was published");
    assert_eq!(state.drone_id, DRONE_ID);
    assert_eq!(state.pending_gate, 3);
    assert_eq!(state.passed, 3);
    assert_eq!(state.missed, 2);
    assert_eq!(state.gate_count, gate_count as u32);
    assert!((state.last_pass_s - 4.25).abs() < 1e-12);
    assert!(!state.finished);
    // The resource is the truth; the message must not have perturbed it.
    assert_eq!(progress(&app).passed, 3);
}

/// How a run ended reaches the controller. A crash is the one thing an off-process
/// controller cannot work out for itself — it has the course and the telemetry, but not
/// the airframe's hull — so if the outcome does not travel, a crashed drone just looks
/// like one whose telemetry stopped.
#[test]
fn the_outcome_that_ended_the_run_reaches_a_subscriber() {
    let _fdm = fdm_guard();
    let (mut app, _session, race) = watched_app(Default::default());
    *app.world_mut().resource_mut::<RaceProgress>() = RaceProgress {
        next_gate: 2,
        passed: 2,
        finished: true,
        outcome: Outcome::CrashedGate,
        ..Default::default()
    };

    let (_, state) = pump_for_race(&mut app, &race, 300);
    let state = state.expect("no race state was published");
    assert!(state.finished, "the wire must say the run is over");
    assert_eq!(state.outcome, RaceOutcome::CrashedGate, "and why");
    assert_eq!(state.passed, 2, "with the score it ended on");
}

/// A run still going says so, rather than leaving a controller to infer it from an
/// unset field.
#[test]
fn a_running_race_reports_no_outcome() {
    let _fdm = fdm_guard();
    let (mut app, _session, race) = watched_app(Default::default());
    let (_, state) = pump_for_race(&mut app, &race, 300);
    let state = state.expect("no race state was published");
    assert!(!state.finished);
    assert_eq!(state.outcome, RaceOutcome::Running);
}

#[test]
fn telemetry_reaches_a_subscriber_as_a_multirotor() {
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(Default::default());
    // The only FIFO subscriber in the file, declared before the first tick and read
    // out within a few frames — see the note at the top of the module.
    let mut telemetry = TelemetrySubscriber::connect(&session).unwrap();
    telemetry.subscribe(DRONE_ID).unwrap();

    let mut got: Option<TelemetrySample> = None;
    for _ in 0..300 {
        app.update();
        if let Some(s) = telemetry.try_recv().unwrap() {
            got = Some(s);
            break;
        }
    }
    let got = got.expect("no telemetry was published");
    let st = drone_state(&mut app).expect("the drone must have spawned");

    assert_eq!(got.aircraft_id, DRONE_ID);
    // The class matters on the wire: a subscriber picks its model and its control
    // interpretation from it without reading the scenario.
    assert_eq!(got.vehicle_kind, VehicleKind::Multirotor);
    // Published from the same synced pose the gate scoring uses, within a tick.
    assert!((got.enu_e - st.pos_enu.x).abs() < 1.0);
    assert!((got.enu_n - st.pos_enu.y).abs() < 1.0);
    assert!((got.enu_u - st.pos_enu.z).abs() < 1.0);
    // ENU velocity is converted to the wire's NED, not copied component-wise.
    assert!((got.vel_down_mps + st.vel_enu.z).abs() < 1.0);
}

#[test]
fn a_raw_command_takes_over_the_actuators() {
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(flat_course_at(30.0));
    let tx = CommandSender::connect(&session);

    // Let the station hold settle, then take the collective away. If raw control
    // did *not* win, the inner loop would keep holding altitude.
    for _ in 0..120 {
        app.update();
    }
    let before = drone_state(&mut app).unwrap().pos_enu.z;

    let cut = CommandMsg {
        target_id: DRONE_ID,
        control: Control::Raw { aileron: 0.0, elevator: 0.0, rudder: 0.0, throttle: 0.0 },
    };
    // Refresh every tick, so the lease never lapses during the measurement.
    for _ in 0..120 {
        tx.send(&cut).unwrap();
        app.update();
    }

    let raw = raw_override(&mut app).expect("the drone must carry a RawOverride");
    assert!(raw.active, "a raw command did not take effect");
    assert_eq!(raw.throttle, 0.0);
    let after = drone_state(&mut app).unwrap().pos_enu.z;
    assert!(after < before - 0.5, "collective 0 did not descend: {before:.2} m -> {after:.2} m");
}

#[test]
fn a_raw_command_is_clamped_to_the_valid_range() {
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(flat_course_at(30.0));
    let tx = CommandSender::connect(&session);

    let wild = CommandMsg {
        target_id: DRONE_ID,
        control: Control::Raw { aileron: 5.0, elevator: -5.0, rudder: 5.0, throttle: 5.0 },
    };
    for _ in 0..60 {
        tx.send(&wild).unwrap();
        app.update();
    }

    let raw = raw_override(&mut app).unwrap();
    assert!(raw.active);
    assert_eq!((raw.aileron, raw.elevator, raw.rudder, raw.throttle), (1.0, -1.0, 1.0, 1.0));
}

#[test]
fn a_command_for_another_drone_is_ignored() {
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(flat_course_at(30.0));
    let tx = CommandSender::connect(&session);

    let elsewhere = CommandMsg {
        target_id: DRONE_ID + 99,
        control: Control::Raw { aileron: 0.0, elevator: 0.0, rudder: 0.0, throttle: 0.0 },
    };
    for _ in 0..120 {
        tx.send(&elsewhere).unwrap();
        app.update();
    }
    assert!(!raw_override(&mut app).unwrap().active, "a command for another id was applied");
}

#[test]
fn a_raw_lease_lapses_and_the_inner_loop_recovers() {
    // A quadrotor's collective is the only thing holding it up, so a latched raw
    // command plus a dead controller is a guaranteed crash. The lease must expire
    // and hand the drone back to the inner loop, holding where it ended up.
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(flat_course_at(30.0));
    let tx = CommandSender::connect(&session);

    for _ in 0..120 {
        app.update();
    }
    let cut = CommandMsg {
        target_id: DRONE_ID,
        control: Control::Raw { aileron: 0.0, elevator: 0.0, rudder: 0.0, throttle: 0.0 },
    };
    // A quarter-second of free fall, then the "controller" dies mid-descent.
    for _ in 0..30 {
        tx.send(&cut).unwrap();
        app.update();
    }
    assert!(raw_override(&mut app).unwrap().active, "the raw command never took effect");
    let falling = drone_state(&mut app).unwrap();
    assert!(falling.vel_enu.z < -0.5, "expected a descent, got {:.2} m/s", falling.vel_enu.z);

    // Step until the lease lapses — the lease is timed in *simulated* seconds, so the
    // exact tick is deterministic and worth pinning rather than sleeping past.
    let mut revert = None;
    for _ in 0..240 {
        app.update();
        if !raw_override(&mut app).unwrap().active {
            revert = Some(drone_state(&mut app).unwrap());
            break;
        }
    }
    let revert = revert.expect("the raw lease never lapsed");

    // Reverting means holding *here*, not returning to the pre-override command:
    // `hold_here` reads the FDM before the step, so it trails this tick's synced
    // altitude by one dt of descent.
    let cmd = flight_command(&mut app).unwrap();
    assert!(
        (cmd.alt_m - revert.pos_enu.z).abs() < 0.2,
        "hold target {:.2} m is not the revert altitude {:.2} m",
        cmd.alt_m,
        revert.pos_enu.z
    );

    // And the inner loop must actually catch it: 8 s is far longer than the descent
    // it has to arrest.
    for _ in 0..960 {
        app.update();
    }
    let recovered = drone_state(&mut app).unwrap();
    assert!(recovered.pos_enu.z > 5.0, "the drone fell to {:.2} m", recovered.pos_enu.z);
    assert!(
        recovered.vel_enu.z.abs() < 1.0,
        "the descent was not arrested: {:.2} m/s",
        recovered.vel_enu.z
    );
    assert!(
        (recovered.pos_enu.z - cmd.alt_m).abs() < 2.0,
        "settled at {:.2} m, not the {:.2} m it reverted to",
        recovered.pos_enu.z,
        cmd.alt_m
    );
}

#[test]
fn an_autopilot_command_maps_onto_the_flight_command() {
    // `AutopilotTargets` is a fixed-wing idiom (MSL altitude, scalar airspeed).
    // It is accepted as the coordinated-flight subset: forward along the heading.
    let _fdm = fdm_guard();
    let (mut app, session) = race_app(flat_course_at(30.0));
    let tx = CommandSender::connect(&session);

    // The fixture's ENU datum is at 0 m, so MSL and ENU altitude coincide here;
    // `origin_alt_m` is what carries the general case.
    let target = CommandMsg {
        target_id: DRONE_ID,
        // Due East, so the velocity must land entirely on ENU +x.
        control: Control::Autopilot {
            heading_rad: std::f64::consts::FRAC_PI_2,
            alt_msl_m: 25.0,
            airspeed_mps: 4.0,
        },
    };
    for _ in 0..60 {
        tx.send(&target).unwrap();
        app.update();
    }

    let cmd = flight_command(&mut app).unwrap();
    assert!((cmd.alt_m - 25.0).abs() < 1e-9, "alt {}", cmd.alt_m);
    assert!((cmd.heading_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert!((cmd.vel_enu.x - 4.0).abs() < 1e-9, "east {}", cmd.vel_enu.x);
    assert!(cmd.vel_enu.y.abs() < 1e-9, "north {}", cmd.vel_enu.y);
    // And it hands control back to the inner loop.
    assert!(!raw_override(&mut app).unwrap().active);
}

#[test]
fn the_app_runs_unchanged_with_no_session() {
    // The bridge systems all take `Option<NonSend<..>>`, so a plain physics app must
    // step exactly as before — this is what keeps every other test in this crate
    // free of a transport.
    let _fdm = fdm_guard();
    let scenario = race_scenario(Default::default());
    let course = generate_course(&scenario).unwrap();
    let mut app = build_app_with_course(scenario, course);
    for _ in 0..120 {
        app.update();
    }
    let st = drone_state(&mut app).expect("the drone must still spawn without a session");
    assert!(st.sim_time_s > 0.9, "sim time {:.3}", st.sim_time_s);
    assert!(!raw_override(&mut app).unwrap().active);
}
