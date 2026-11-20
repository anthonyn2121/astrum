//! The race on the wire: the course and the progress a controller sees, and the race
//! events on the sim's event stream — over one in-process session, driven through the
//! real app.

mod common;

use std::sync::{Arc, Mutex};

use astrum_net::zenoh::Wait;
use astrum_net::{keys, RaceSubscriber, Session};
use astrum_proto::{decode_event, EndReason, EventKind, RaceOutcome};
use astrum_race::build_race_app;
use astrum_race::scenario::RaceScenario;
use astrum_sim::RunMode;
use bevy::prelude::*;
use common::{fdm_guard, flat_course, scenario_toml};

/// A controller's view of a race: the course on the first tick with each gate's explicit
/// basis, the state every tick, and — when nobody flies the drone and it hits the ground —
/// a final state that says so, plus the race events and the end of the run.
#[test]
fn a_controller_sees_the_course_the_state_and_the_crash() {
    let _fdm = fdm_guard();
    let race = RaceScenario::from_toml_str(&scenario_toml("", "", &flat_course(1.5, 4)))
        .unwrap()
        .resolve(None)
        .unwrap();
    let gates = race.course.gates.clone();

    let session = Session::in_process().unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let _sub = session
        .raw()
        .declare_subscriber(keys::EVENTS)
        .callback(move |s| {
            sink.lock().unwrap().push(decode_event(&s.payload().to_bytes()).unwrap())
        })
        .wait()
        .unwrap();
    let race_sub = RaceSubscriber::new(&session, 1).unwrap();

    let mut app = build_race_app(race, RunMode::Batch);
    astrum_sim::net::connect(&mut app, &session).unwrap();
    astrum_race::net::connect(&mut app, &session).unwrap();
    app.finish();
    app.cleanup();

    app.update(); // Startup publishes the course and the initial state, then tick 1
    let course = race_sub.try_recv_course().expect("the course goes out on the first tick");
    assert_eq!(course.seed, 1);
    assert_eq!(course.gates.len(), 4);
    for (wire, gate) in course.gates.iter().zip(&gates) {
        assert_eq!(wire.index, gate.index);
        assert_eq!(
            (wire.center.e, wire.center.n, wire.center.u),
            (gate.center.x, gate.center.y, gate.center.z)
        );
        // The explicit basis: normal along the yaw, right starboard of it, up up.
        let (s, c) = wire.yaw_rad.sin_cos();
        assert_eq!((wire.normal.e, wire.normal.n, wire.normal.u), (s, c, 0.0));
        assert_eq!((wire.right.e, wire.right.n, wire.right.u), (c, -s, 0.0));
        assert_eq!((wire.up.e, wire.up.n, wire.up.u), (0.0, 0.0, 1.0));
    }
    let first = race_sub.try_recv_state().expect("and the state");
    assert_eq!((first.aircraft_id, first.gate_count, first.pending_gate), (1, 4, 0));
    assert!(!first.finished);

    let mut last = first;
    let mut ticks = 0;
    while app.should_exit().is_none() && ticks < 600 {
        app.update();
        if let Some(s) = race_sub.try_recv_state() {
            assert!(s.sim_time_s >= last.sim_time_s, "the state moves forward in time");
            last = s;
        }
        ticks += 1;
    }
    assert!(ticks < 600, "an unflown drone at 1.5 m ends the run in about half a second");
    assert!(last.finished);
    assert_eq!((last.outcome, last.passed, last.missed), (RaceOutcome::CrashedGround, 0, 0));

    let kinds: Vec<EventKind> = events.lock().unwrap().iter().map(|e| e.kind.clone()).collect();
    assert!(
        kinds.contains(&EventKind::Crashed {
            aircraft_id: 1,
            outcome: RaceOutcome::CrashedGround,
            gate: 0
        }),
        "{kinds:?}"
    );
    assert!(
        kinds.contains(&EventKind::RunEnded { reason: EndReason::Crashed, message: String::new() }),
        "{kinds:?}"
    );
}
