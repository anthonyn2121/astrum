//! Zenoh transport integration tests over a single in-process session: command
//! fan-in round-trips, telemetry delivery + per-aircraft key filtering, and the
//! racing course/progress channels.
//!
//! A publisher and subscriber on one [`Session`] deliver locally and
//! synchronously (Zenoh invokes same-session subscriber callbacks inline in the
//! `put`), so these are deterministic; the short retry loops just guard against
//! any residual scheduling lag.

use astrum_net::{
    CommandReceiver, CommandSender, RacePublisher, RaceSubscriber, Session, TelemetryPublisher,
    TelemetrySubscriber,
};
use astrum_proto::{
    CommandMsg, Control, RaceCourseMsg, RaceGateSpec, RaceStateMsg, TelemetrySample,
};

fn sample(id: u32) -> TelemetrySample {
    TelemetrySample {
        aircraft_id: id,
        sim_time_s: 1.0,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_m: 1524.0,
        enu_e: 1.0,
        enu_n: 2.0,
        enu_u: 3.0,
        vel_north_mps: 50.0,
        vel_east_mps: 0.0,
        vel_down_mps: 0.0,
        phi_rad: 0.0,
        theta_rad: 0.0,
        psi_rad: 0.5,
        airspeed_mps: 51.0,
        health: 0,
        vehicle_kind: astrum_proto::VehicleKind::FixedWing,
    }
}

#[test]
fn command_fan_in_roundtrip() {
    let session = Session::in_process().unwrap();
    let rx = CommandReceiver::bind(&session).unwrap();
    let tx = CommandSender::connect(&session);

    let cmd = CommandMsg {
        target_id: 42,
        control: Control::Raw { aileron: 0.1, elevator: -0.2, rudder: 0.0, throttle: 0.8 },
    };
    tx.send(&cmd).unwrap();

    // Only one command was sent, so retry-draining (without resending) yields
    // exactly one copy once it lands.
    let mut drained = Vec::new();
    for _ in 0..1000 {
        drained = rx.drain();
        if !drained.is_empty() {
            break;
        }
    }
    assert_eq!(drained, vec![cmd]);
}

#[test]
fn telemetry_delivery_roundtrip() {
    let session = Session::in_process().unwrap();
    let pubr = TelemetryPublisher::new(&session);
    let mut sub = TelemetrySubscriber::connect(&session).unwrap();
    sub.subscribe(7).unwrap();

    let s = sample(7);
    let mut got = None;
    for _ in 0..1000 {
        pubr.publish(&s).unwrap();
        if let Some(r) = sub.try_recv().unwrap() {
            got = Some(r);
            break;
        }
    }
    assert_eq!(got, Some(s));
}

#[test]
fn telemetry_key_filter_excludes_others() {
    let session = Session::in_process().unwrap();
    let pubr = TelemetryPublisher::new(&session);
    let mut sub = TelemetrySubscriber::connect(&session).unwrap();
    sub.subscribe(7).unwrap();

    // Publish for a different id; the key filter must exclude it. Establish the
    // subscription first by confirming id 7 gets through.
    let mine = sample(7);
    let mut established = false;
    for _ in 0..1000 {
        pubr.publish(&mine).unwrap();
        if sub.try_recv().unwrap().is_some() {
            established = true;
            break;
        }
    }
    assert!(established, "subscription never established");

    // Drain any residual id-7 frames queued during establishment.
    while sub.try_recv().unwrap().is_some() {}

    // Now a burst for id 9 must not be delivered (key astrum/telemetry/9 does not
    // match the subscription astrum/telemetry/7).
    for _ in 0..50 {
        pubr.publish(&sample(9)).unwrap();
    }
    assert!(sub.try_recv().unwrap().is_none());
}

// --- Racing task state -------------------------------------------------------

fn race_course(drone_id: u32) -> RaceCourseMsg {
    RaceCourseMsg {
        drone_id,
        seed: 7,
        opening_m: 0.635,
        bar_m: 0.0762,
        gates: (0..6)
            .map(|i| RaceGateSpec {
                index: i,
                center_e: i as f64,
                center_n: i as f64 * 10.0,
                center_u: 2.5,
                yaw_rad: 0.1 * i as f64,
            })
            .collect(),
    }
}

fn race_state(drone_id: u32, passed: u32) -> RaceStateMsg {
    RaceStateMsg {
        drone_id,
        sim_time_s: passed as f64,
        pending_gate: passed,
        passed,
        missed: 0,
        gate_count: 6,
        last_pass_s: 0.0,
        finished: false,
        outcome: astrum_proto::RaceOutcome::Running,
    }
}

#[test]
fn race_course_and_state_delivery_roundtrip() {
    let session = Session::in_process().unwrap();
    let pubr = RacePublisher::new(&session);
    let mut sub = RaceSubscriber::connect(&session).unwrap();
    sub.subscribe(1).unwrap();

    let (course, state) = (race_course(1), race_state(1, 2));
    let mut got_course = None;
    let mut got_state = None;
    for _ in 0..1000 {
        pubr.publish_course(&course).unwrap();
        pubr.publish_state(&state).unwrap();
        got_course = got_course.or(sub.try_recv_course().unwrap());
        got_state = got_state.or(sub.try_recv_state().unwrap());
        if got_course.is_some() && got_state.is_some() {
            break;
        }
    }
    assert_eq!(got_course, Some(course));
    assert_eq!(got_state, Some(state));
}

#[test]
fn race_reads_are_latest_wins_not_a_queue() {
    // Both racing channels carry *state*, so a backlog is staleness: a read must
    // return the newest message and discard what it superseded. Publishing a
    // strictly increasing sequence makes that observable.
    let session = Session::in_process().unwrap();
    let pubr = RacePublisher::new(&session);
    let mut sub = RaceSubscriber::connect(&session).unwrap();
    sub.subscribe(1).unwrap();

    // Establish the subscription, then clear whatever landed while it came up.
    let mut established = false;
    for _ in 0..1000 {
        pubr.publish_state(&race_state(1, 0)).unwrap();
        if sub.try_recv_state().unwrap().is_some() {
            established = true;
            break;
        }
    }
    assert!(established, "subscription never established");
    while sub.try_recv_state().unwrap().is_some() {}

    for passed in 1..=20 {
        pubr.publish_state(&race_state(1, passed)).unwrap();
    }
    assert_eq!(sub.try_recv_state().unwrap().map(|s| s.passed), Some(20));
    // The whole backlog is consumed, not just one entry off the front.
    assert!(sub.try_recv_state().unwrap().is_none());
}

#[test]
fn race_key_filter_excludes_other_drones() {
    let session = Session::in_process().unwrap();
    let pubr = RacePublisher::new(&session);
    let mut sub = RaceSubscriber::connect(&session).unwrap();
    sub.subscribe(1).unwrap();

    let mut established = false;
    for _ in 0..1000 {
        pubr.publish_state(&race_state(1, 0)).unwrap();
        if sub.try_recv_state().unwrap().is_some() {
            established = true;
            break;
        }
    }
    assert!(established, "subscription never established");
    while sub.try_recv_state().unwrap().is_some() {}
    while sub.try_recv_course().unwrap().is_some() {}

    for _ in 0..50 {
        pubr.publish_state(&race_state(9, 3)).unwrap();
        pubr.publish_course(&race_course(9)).unwrap();
    }
    assert!(sub.try_recv_state().unwrap().is_none(), "drone 9's state leaked into drone 1's read");
    assert!(
        sub.try_recv_course().unwrap().is_none(),
        "drone 9's course leaked into drone 1's read"
    );
}

#[test]
fn race_channels_are_independent() {
    // The course and progress subscribers are separate declarations; a course
    // publication must not satisfy a state read, or a controller would block on
    // one channel while the other is live.
    let session = Session::in_process().unwrap();
    let pubr = RacePublisher::new(&session);
    let mut sub = RaceSubscriber::connect(&session).unwrap();
    sub.subscribe_all().unwrap();

    let mut got = None;
    for _ in 0..1000 {
        pubr.publish_course(&race_course(4)).unwrap();
        got = sub.try_recv_course().unwrap();
        if got.is_some() {
            break;
        }
    }
    assert!(got.is_some(), "subscribe_all did not receive a course");
    assert!(sub.try_recv_state().unwrap().is_none(), "a course satisfied a state read");
}

#[test]
fn race_reads_before_subscribing_are_an_error() {
    let session = Session::in_process().unwrap();
    let sub = RaceSubscriber::connect(&session).unwrap();
    assert!(sub.try_recv_course().is_err());
    assert!(sub.try_recv_state().is_err());
}
