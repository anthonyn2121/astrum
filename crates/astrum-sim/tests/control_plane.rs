//! The control plane over an in-process session: pause / step / speed
//! requests, the status reply, and controller presence (liveliness tokens).

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use astrum_net::{
    fetch_status, send_control, CommandSender, ControlRequest, LatestTelemetry, NetError,
    PresenceToken, Selection, Session,
};
use astrum_proto::{Control, ControlMode, RawActuators};
use astrum_sim::components::{SimClock, SimStatus};
use astrum_sim::net::LastPublished;
use astrum_sim::scenario::Scenario;
use astrum_sim::{build_app, net, RunMode};
use bevy::prelude::*;
use common::{of, two_ship};

const T: Duration = Duration::from_secs(2);

fn app_on(session: &Session, scenario: Scenario, mode: RunMode) -> App {
    let mut app = build_app(scenario, mode);
    net::connect(&mut app, session).unwrap();
    app
}

fn tick(app: &App) -> u64 {
    app.world().resource::<SimClock>().tick
}

fn fatal(app: &App) -> Option<String> {
    app.world().resource::<SimStatus>().fatal.clone()
}

/// Update `app` until `done` holds, for at most `limit` of wall time.
fn update_until(app: &mut App, limit: Duration, done: impl Fn(&App) -> bool) -> Duration {
    let started = Instant::now();
    while !done(app) && started.elapsed() < limit {
        app.update();
        std::thread::sleep(Duration::from_millis(2));
    }
    started.elapsed()
}

#[test]
fn pause_holds_the_tick_and_step_runs_exactly_n() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship(), RunMode::Embedded);
    app.update(); // startup + tick 1
    send_control(&session, ControlRequest::Pause, T).unwrap();
    app.update(); // the pause lands before the tick
    assert_eq!(tick(&app), 1);
    app.update();
    assert_eq!(tick(&app), 1, "a paused sim does not tick");
    let s = fetch_status(&session, T).unwrap();
    assert_eq!((s.state.as_str(), s.tick, s.aircraft), ("paused", 1, 2));

    send_control(&session, ControlRequest::Step(5), T).unwrap();
    for _ in 0..8 {
        app.update();
    }
    assert_eq!(tick(&app), 6, "step 5 runs five ticks, then pauses");
    assert_eq!(fetch_status(&session, T).unwrap().state, "paused");

    send_control(&session, ControlRequest::Resume, T).unwrap();
    app.update();
    app.update();
    assert_eq!(tick(&app), 8);
    assert_eq!(fetch_status(&session, T).unwrap().state, "running");
}

#[test]
fn invalid_control_requests_are_refused() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship(), RunMode::Embedded);
    app.update();
    for bad in [
        ControlRequest::Step(0),
        ControlRequest::Speed(1e-6),
        ControlRequest::Speed(-1.0),
        ControlRequest::Speed(f64::NAN),
    ] {
        match send_control(&session, bad, T) {
            Err(NetError::Refused(_)) => {}
            other => panic!("{bad:?} must be refused, got {other:?}"),
        }
    }
    app.update();
    assert_eq!(tick(&app), 2, "nothing was applied");
}

#[test]
fn live_speed_changes_set_the_pace() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship(), RunMode::Batch); // paced, unlike Embedded
    app.update();
    send_control(&session, ControlRequest::Speed(1.0), T).unwrap();
    let started = Instant::now();
    for _ in 0..61 {
        app.update();
    }
    // 60 ticks at 120 Hz in real time is 0.5 s of wall clock (pacing only waits).
    let real_time = started.elapsed();
    assert!(real_time >= Duration::from_millis(480), "1× ran 0.5 s of sim in {real_time:?}");

    send_control(&session, ControlRequest::Speed(0.0), T).unwrap();
    let started = Instant::now();
    for _ in 0..61 {
        app.update();
    }
    let unpaced = started.elapsed();
    assert!(unpaced < real_time / 2, "unpaced took {unpaced:?}, real time {real_time:?}");
}

#[test]
fn a_lost_lockstep_controller_fails_the_run_at_once() {
    let mut s = two_ship();
    s.sim.lockstep = true;
    s.sim.lockstep_timeout_s = 10.0;
    s.aircraft[0].lockstep = true;
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, s, RunMode::Embedded);

    // A controller that answers every tick until told to quit, then goes away.
    let quit = Arc::new(AtomicBool::new(false));
    let controller = {
        let (session, quit) = (session.clone(), Arc::clone(&quit));
        std::thread::spawn(move || {
            let _token = PresenceToken::declare(&session, 1).unwrap();
            let mut sub = LatestTelemetry::subscribe(&session, Selection::Aircraft(1)).unwrap();
            let mut tx = CommandSender::new(&session, 1).unwrap();
            let raw = Control::Raw(RawActuators {
                aileron: 0.0,
                elevator: 0.0,
                rudder: 0.0,
                throttle: 0.7,
            });
            while !quit.load(Ordering::Relaxed) {
                if sub.wait(Duration::from_millis(5)).is_ok() {
                    tx.send(raw, false, sub.latest(1).unwrap().tick).unwrap();
                }
            }
        })
    };
    app.update(); // startup publishes tick 0; its wait ends with the controller's answer
    for _ in 0..10 {
        app.update();
    }
    assert_eq!(tick(&app), 11);
    assert!(fatal(&app).is_none());

    quit.store(true, Ordering::Relaxed);
    controller.join().unwrap(); // dropping its token: the controller is gone
    let took = update_until(&mut app, Duration::from_secs(5), |app| fatal(app).is_some());
    let why = fatal(&app).expect("the run fails");
    assert!(why.contains("aircraft 1 disconnected"), "{why}");
    assert!(took < Duration::from_secs(1), "failed after {took:?}, not the 10 s lockstep timeout");
}

#[test]
fn a_lost_controller_puts_its_aircraft_in_failsafe_before_its_lease_ends() {
    let mut s = two_ship();
    s.sim.command_lease_s = 30.0;
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, s, RunMode::Embedded);
    let token = PresenceToken::declare(&session, 1).unwrap();
    let mut tx = CommandSender::new(&session, 1).unwrap();
    let raw =
        Control::Raw(RawActuators { aileron: 0.1, elevator: 0.0, rudder: 0.0, throttle: 0.7 });
    let mode = |app: &App| of(&app.world().resource::<LastPublished>().0, 1).control_mode;
    app.update();
    tx.send(raw, false, tick(&app)).unwrap();
    update_until(&mut app, T, |app| mode(app) == ControlMode::Raw);
    assert_eq!(mode(&app), ControlMode::Raw);
    let present = || fetch_status(&session, T).unwrap().controllers;
    update_until(&mut app, T, |_| present() == [1]);
    assert_eq!(present(), [1], "the status lists the present controller");

    drop(token);
    let took =
        update_until(&mut app, Duration::from_secs(5), |app| mode(app) == ControlMode::Failsafe);
    assert_eq!(mode(&app), ControlMode::Failsafe);
    assert!(took < Duration::from_secs(1), "failsafe after {took:?}, lease is 30 s");
    assert!(fatal(&app).is_none(), "a non-lockstep aircraft does not end the run");
}
