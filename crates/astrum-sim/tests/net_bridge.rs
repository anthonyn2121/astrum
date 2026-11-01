//! The Zenoh bridge end to end over an in-process session: commands in (raw,
//! autopilot, refusals), telemetry out, and lockstep reproducibility with a
//! controller running in another thread.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use astrum_net::zenoh::Wait;
use astrum_net::{
    keys, CommandReceiver, CommandSender, LatestTelemetry, Selection, Session, TelemetryPublisher,
};
use astrum_proto::{
    encode_command, AutopilotTargets, Command, Control, ControlMode, RawActuators, Telemetry,
};
use astrum_sim::components::SimStatus;
use astrum_sim::net::LastPublished;
use astrum_sim::{build_app, RunMode};
use bevy::prelude::*;
use common::{of, two_ship};

fn app_on(session: &Session, scenario: astrum_sim::scenario::Scenario) -> App {
    let mut app = build_app(scenario, RunMode::Embedded);
    app.insert_non_send(TelemetryPublisher::new(session));
    app.insert_non_send(CommandReceiver::bind(session).unwrap());
    app
}

fn raw(a: f64) -> Control {
    Control::Raw(RawActuators { aileron: a, elevator: 0.0, rudder: 0.0, throttle: 0.7 })
}

fn last(app: &App) -> Vec<Telemetry> {
    app.world().resource::<LastPublished>().0.clone()
}

#[test]
fn commands_in_telemetry_out() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship());
    let fleet = LatestTelemetry::subscribe(&session, Selection::Fleet).unwrap();
    let mut tx1 = CommandSender::new(&session, 1).unwrap();
    let mut tx2 = CommandSender::new(&session, 2).unwrap();
    app.update();

    for _ in 0..60 {
        let tick = last(&app)[0].tick;
        tx1.send(raw(0.5), false, tick).unwrap();
        tx2.send(
            Control::Autopilot(AutopilotTargets {
                heading_rad: 1.0,
                alt_hae_m: 1600.0,
                tas_mps: 55.0,
            }),
            false,
            tick,
        )
        .unwrap();
        app.update();
    }

    let obs = last(&app);
    assert_eq!(of(&obs, 1).control_mode, ControlMode::Raw);
    assert!(of(&obs, 1).rates.p_rad_s > 0.1, "raw aileron reached aircraft 1");
    assert_eq!(of(&obs, 2).control_mode, ControlMode::Autopilot);
    // The subscriber sees the same latest state the sim published.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(fleet.latest(1), Some(of(&obs, 1)));
    assert_eq!(fleet.latest(2), Some(of(&obs, 2)));
}

#[test]
fn forged_stale_and_future_commands_are_refused() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship());
    for _ in 0..200 {
        app.update();
    }
    let now = last(&app)[0].tick;
    let put = |key: String, c: Command| session.raw().put(key, encode_command(&c)).wait().unwrap();
    // Key says aircraft 1, payload says 2.
    put(
        keys::command(1),
        Command { target_id: 2, seq: 1, telemetry_tick: now, trigger: false, control: raw(1.0) },
    );
    // Answers telemetry older than the 0.5 s lease.
    put(
        keys::command(1),
        Command {
            target_id: 1,
            seq: 2,
            telemetry_tick: now - 100,
            trigger: false,
            control: raw(1.0),
        },
    );
    // Answers a tick the sim has not reached.
    put(
        keys::command(2),
        Command {
            target_id: 2,
            seq: 3,
            telemetry_tick: now + 5,
            trigger: false,
            control: raw(1.0),
        },
    );
    for _ in 0..5 {
        app.update();
    }
    for t in last(&app) {
        assert_eq!(
            t.control_mode,
            ControlMode::Uncommanded,
            "aircraft {} took a refused command",
            t.aircraft_id
        );
    }
}

/// Run a lockstep scenario with a controller thread whose law depends only on
/// the telemetry it receives. Returns the final telemetry and the ticks the
/// controller answered.
fn lockstep_run(ticks: usize, control_rate_hz: Option<f64>) -> (Vec<Telemetry>, Vec<u64>) {
    let mut s = two_ship();
    s.sim.lockstep = true;
    s.sim.lockstep_timeout_s = 10.0;
    s.aircraft[0].lockstep = true;
    s.aircraft[0].control_rate_hz = control_rate_hz;
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, s);

    let stop = Arc::new(AtomicBool::new(false));
    let controller = {
        let (session, stop) = (session.clone(), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut sub = LatestTelemetry::subscribe(&session, Selection::Aircraft(1)).unwrap();
            let mut tx = CommandSender::new(&session, 1).unwrap();
            let mut answered: Vec<u64> = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                if sub.wait(Duration::from_millis(20)).is_err() {
                    continue;
                }
                let t = sub.latest(1).unwrap();
                if answered.last() != Some(&t.tick) {
                    // A deterministic law of the received state only.
                    let ail = (-2.0 * t.attitude.phi_rad + 0.3 * (t.tick as f64 * 0.01).sin())
                        .clamp(-1.0, 1.0);
                    tx.send(raw(ail), false, t.tick).unwrap();
                    answered.push(t.tick);
                }
            }
            answered
        })
    };
    app.update(); // startup publishes tick 0, then waits for its answer
    for _ in 0..ticks {
        app.update();
        assert!(app.world().resource::<SimStatus>().fatal.is_none());
    }
    stop.store(true, Ordering::Relaxed);
    let answered = controller.join().unwrap();
    (last(&app), answered)
}

#[test]
fn lockstep_makes_networked_runs_reproducible() {
    let (a, answered) = lockstep_run(240, None);
    let (b, _) = lockstep_run(240, None);
    assert_eq!(of(&a, 1).tick, 241);
    assert_eq!(of(&a, 1).control_mode, ControlMode::Raw);
    // The sim waited on ticks 0..=240; tick 241's answer may or may not have
    // been sent before the controller stopped.
    assert_eq!(answered[..241], (0..=240).collect::<Vec<_>>(), "every tick is a control tick");
    assert_eq!(a, b, "a lockstep run with a deterministic controller replays exactly");
}

#[test]
fn decimated_lockstep_publishes_and_waits_only_on_control_ticks() {
    // 30 Hz control at 120 Hz physics: telemetry and the lockstep wait every 4th
    // tick, the last command held in between — and still reproducible.
    let (a, answered) = lockstep_run(240, Some(30.0));
    let (b, _) = lockstep_run(240, Some(30.0));
    assert_eq!(of(&a, 1).tick, 241);
    assert_eq!(of(&a, 1).control_mode, ControlMode::Raw);
    assert_eq!(answered, (0..=240).step_by(4).collect::<Vec<_>>());
    assert_eq!(a, b);
}

#[test]
fn lockstep_without_a_controller_is_fatal() {
    let mut s = two_ship();
    s.sim.lockstep = true;
    s.sim.lockstep_timeout_s = 0.3;
    s.aircraft[1].lockstep = true;
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, s);
    app.update();
    let fatal =
        app.world().resource::<SimStatus>().fatal.clone().expect("lockstep wait must time out");
    assert!(fatal.contains("from aircraft {2}"), "{fatal}");
    assert_eq!(last(&app)[0].tick, 0, "no step ran after the timeout");
}
