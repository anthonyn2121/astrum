//! The Zenoh bridge end to end over an in-process session: commands in (raw,
//! multirotor, autopilot, refusals, ownership), telemetry and tick ends out
//! (and their order), and lockstep reproducibility with a controller running in
//! another thread.

mod common;

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use astrum_net::zenoh::Wait;
use astrum_net::{
    keys, send_control, CommandSender, ControlRequest, LatestTelemetry, PresenceToken, Selection,
    Session,
};
use astrum_proto::{
    decode_event, decode_telemetry, decode_tick_end, encode_command, AutopilotTargets, Command,
    Control, ControlMode, MultirotorRaw, RawActuators, Refusal, Telemetry,
};
use astrum_sim::components::{SimClock, SimStatus};
use astrum_sim::net::LastPublished;
use astrum_sim::{build_app, RunMode};
use bevy::prelude::*;
use common::{mixed_fleet, of, two_ship, DT};

fn app_on(session: &Session, scenario: astrum_sim::scenario::Scenario) -> App {
    let mut app = build_app(scenario, RunMode::Embedded);
    astrum_sim::net::connect(&mut app, session).unwrap();
    app
}

fn raw(a: f64) -> Control {
    Control::Raw(RawActuators { aileron: a, elevator: 0.0, rudder: 0.0, throttle: 0.7 })
}

fn last(app: &App) -> Vec<Telemetry> {
    app.world().resource::<LastPublished>().0.clone()
}

/// Telemetry without the controller epochs, which are random per sender.
fn without_epochs(mut v: Vec<Telemetry>) -> Vec<Telemetry> {
    for t in &mut v {
        t.command.owner_epoch = 0;
        t.command.last_refused_epoch = 0;
    }
    v
}

/// What a subscriber to `astrum/**` saw, in arrival order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Seen {
    Telemetry { id: u32, tick: u64 },
    Event { tick: u64 },
    TickEnd { tick: u64 },
}

fn watch_all(
    session: &Session,
) -> (Arc<Mutex<Vec<Seen>>>, astrum_net::zenoh::pubsub::Subscriber<()>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let sub = session
        .raw()
        .declare_subscriber("astrum/**")
        .callback(move |s| {
            let key = s.key_expr().as_str();
            let bytes = s.payload().to_bytes();
            let item = if key == keys::TICK {
                Seen::TickEnd { tick: decode_tick_end(&bytes).unwrap().tick }
            } else if key == keys::EVENTS {
                Seen::Event { tick: decode_event(&bytes).unwrap().tick }
            } else if keys::id_of(key, keys::TELEMETRY_PREFIX).is_some() {
                let t = decode_telemetry(&bytes).unwrap();
                Seen::Telemetry { id: t.aircraft_id, tick: t.tick }
            } else {
                return;
            };
            sink.lock().unwrap().push(item);
        })
        .wait()
        .unwrap();
    (seen, sub)
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
fn forged_stale_and_future_commands_are_refused_and_reported() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship());
    for _ in 0..200 {
        app.update();
    }
    let now = last(&app)[0].tick;
    let put = |key: String, c: Command| session.raw().put(key, encode_command(&c)).wait().unwrap();
    let cmd = |target_id, seq, telemetry_tick, epoch| Command {
        target_id,
        seq,
        telemetry_tick,
        trigger: false,
        epoch,
        control: raw(1.0),
    };
    // Key says aircraft 1, payload says 2.
    put(keys::command(1), cmd(2, 1, now, 7));
    // Answers telemetry older than the 0.5 s lease.
    put(keys::command(1), cmd(1, 2, now - 100, 7));
    // Answers a tick the sim has not reached, then carries no epoch.
    put(keys::command(2), cmd(2, 3, now + 5, 9));
    put(keys::command(2), cmd(2, 4, now, 0));
    app.update();
    for t in last(&app) {
        assert_eq!(
            t.control_mode,
            ControlMode::Uncommanded,
            "aircraft {} took a refused command",
            t.aircraft_id
        );
    }
    let obs = last(&app);
    let one = of(&obs, 1).command;
    assert_eq!((one.refusals, one.last_refusal, one.last_refused_epoch), (1, Refusal::Stale, 7));
    let two = of(&obs, 2).command;
    assert_eq!(
        (two.refusals, two.last_refusal, two.last_refused_epoch),
        (2, Refusal::MissingEpoch, 0)
    );
    assert_eq!((one.owner_epoch, two.owner_epoch), (0, 0));
}

#[test]
fn the_first_applied_epoch_owns_the_aircraft_until_its_lease_expires() {
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, two_ship());
    let mut first = CommandSender::new(&session, 1).unwrap();
    let mut second = CommandSender::new(&session, 1).unwrap();
    app.update();
    for _ in 0..5 {
        let tick = last(&app)[0].tick;
        first.send(raw(0.5), false, tick).unwrap();
        second.send(raw(-0.5), false, tick).unwrap();
        app.update();
    }
    let status = |app: &App| of(&last(app), 1).command;
    let s = status(&app);
    assert_eq!((s.owner_epoch, s.last_seq), (first.epoch(), 5));
    assert_eq!((s.refusals, s.last_refusal), (5, Refusal::NotOwner));
    assert_eq!(s.last_refused_epoch, second.epoch());
    assert!(of(&last(&app), 1).rates.p_rad_s > 0.0, "the owner's aileron flies it");

    // The owner goes quiet: past the 0.5 s lease the aircraft holds, unowned.
    for _ in 0..70 {
        app.update();
    }
    assert_eq!(of(&last(&app), 1).control_mode, ControlMode::Failsafe);
    assert_eq!((status(&app).owner_epoch, status(&app).last_seq), (0, 0));
    // Now the other epoch claims it, with its own seq.
    second.send(raw(-0.5), false, last(&app)[0].tick).unwrap();
    app.update();
    let s = status(&app);
    assert_eq!((s.owner_epoch, s.last_seq), (second.epoch(), 6));
    assert_eq!(of(&last(&app), 1).control_mode, ControlMode::Raw);
}

#[test]
fn multirotor_demands_fly_only_a_multirotor() {
    // Aircraft 1 is a c172x, aircraft 2 an F450.
    let session = Session::in_process().unwrap();
    let mut app = app_on(&session, mixed_fleet(2));
    let mut fw = CommandSender::new(&session, 1).unwrap();
    let mut quad = CommandSender::new(&session, 2).unwrap();
    let demand = Control::Multirotor(MultirotorRaw {
        roll_rate: 0.0,
        pitch_rate: 0.0,
        yaw_rate: 0.0,
        collective: 0.5,
    });
    app.update();
    let tick = last(&app)[0].tick;
    fw.send(demand, false, tick).unwrap();
    quad.send(raw(0.0), false, tick).unwrap();
    app.update();
    let obs = last(&app);
    for id in [1, 2] {
        let t = of(&obs, id);
        assert_eq!(t.control_mode, ControlMode::Uncommanded, "aircraft {id}");
        assert_eq!(t.command.last_refusal, Refusal::WrongVehicleKind, "aircraft {id}");
    }
    quad.send(demand, false, last(&app)[0].tick).unwrap();
    app.update();
    let t = of(&last(&app), 2);
    assert_eq!(t.control_mode, ControlMode::Raw);
    assert_eq!(t.command.owner_epoch, quad.epoch());
}

#[test]
fn telemetry_reports_its_control_period() {
    let mut s = two_ship();
    s.aircraft[0].control_rate_hz = Some(30.0);
    let mut app = build_app(s, RunMode::Embedded);
    app.update();
    let obs = last(&app);
    assert_eq!(of(&obs, 1).control_period_s, 4.0 * DT);
    assert_eq!(of(&obs, 2).control_period_s, DT);
}

#[test]
fn tick_end_follows_the_ticks_telemetry_and_events() {
    let session = Session::in_process().unwrap();
    let (seen, _sub) = watch_all(&session);
    let mut app = app_on(&session, two_ship());
    let mut tx = CommandSender::new(&session, 1).unwrap();
    app.update(); // ticks 0 and 1
    let _token = PresenceToken::declare(&session, 1).unwrap(); // a controller-up event
    for k in 0..20 {
        if k % 5 == 0 {
            // Mode changes (events) on several ticks.
            let control = if k % 10 == 0 {
                raw(0.1)
            } else {
                Control::Autopilot(AutopilotTargets {
                    heading_rad: 1.0,
                    alt_hae_m: 1500.0,
                    tas_mps: 55.0,
                })
            };
            tx.send(control, false, last(&app)[0].tick).unwrap();
        }
        app.update();
    }
    let final_tick = last(&app)[0].tick;
    let seen = seen.lock().unwrap().clone();
    let mut closed = BTreeSet::new();
    let mut samples = 0;
    let mut events_closed = 0;
    for item in &seen {
        match *item {
            Seen::Telemetry { tick, .. } => {
                assert!(!closed.contains(&tick), "telemetry of tick {tick} after its TickEnd");
                samples += 1;
            }
            Seen::Event { tick } => {
                assert!(!closed.contains(&tick), "an event of tick {tick} after its TickEnd");
                events_closed += 1;
            }
            Seen::TickEnd { tick } => {
                assert_eq!(samples, 2, "tick {tick}: both samples before its TickEnd");
                assert_eq!(closed.last().map_or(0, |t| t + 1), tick, "tick ends in order");
                closed.insert(tick);
                samples = 0;
            }
        }
    }
    assert_eq!(closed, (0..=final_tick).collect(), "one TickEnd per published tick");
    assert!(events_closed >= 4, "the mode changes were published");
}

#[test]
fn a_lockstep_wait_re_sends_every_sample_of_the_tick_then_its_end() {
    // A follower that subscribed late must still get its leader's sample: the
    // re-publish is the whole tick, not just the awaited aircraft.
    let mut s = two_ship();
    s.sim.lockstep = true;
    s.sim.lockstep_timeout_s = 0.35;
    s.aircraft[0].lockstep = true;
    s.aircraft[1].lockstep = true;
    let session = Session::in_process().unwrap();
    let (seen, _sub) = watch_all(&session);
    let mut app = app_on(&session, s);
    // Aircraft 1 answers tick 0 at once; aircraft 2 never does.
    CommandSender::new(&session, 1).unwrap().send(raw(0.0), false, 0).unwrap();
    app.update();
    let fatal = app.world().resource::<SimStatus>().fatal.clone().expect("aircraft 2 times out");
    assert!(fatal.contains("from aircraft {2}"), "{fatal}");
    // The run's end (a fatal `run-ended` event) follows; only telemetry and
    // tick ends are of interest here.
    let seen: Vec<Seen> =
        seen.lock().unwrap().iter().copied().filter(|s| !matches!(s, Seen::Event { .. })).collect();
    let copy = [
        Seen::Telemetry { id: 1, tick: 0 },
        Seen::Telemetry { id: 2, tick: 0 },
        Seen::TickEnd { tick: 0 },
    ];
    let copies = seen.chunks(3).take_while(|c| *c == copy).count();
    assert!(copies >= 2, "tick 0 published, then re-published whole: {seen:?}");
    assert_eq!(copies * 3, seen.len(), "nothing but whole copies of tick 0: {seen:?}");
}

#[test]
fn a_refusal_during_a_lockstep_wait_shows_in_the_re_sent_sample() {
    // A controller sending the wrong raw arm must learn so from the re-sent
    // tick, not after the lockstep timeout.
    let mut s = two_ship();
    s.sim.lockstep = true;
    s.sim.lockstep_timeout_s = 0.25;
    s.aircraft[0].lockstep = true;
    let session = Session::in_process().unwrap();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&samples);
    let _sub = session
        .raw()
        .declare_subscriber(keys::telemetry(1))
        .callback(move |s| {
            sink.lock().unwrap().push(decode_telemetry(&s.payload().to_bytes()).unwrap())
        })
        .wait()
        .unwrap();
    let mut app = app_on(&session, s);
    let mut tx = CommandSender::new(&session, 1).unwrap();
    // Multirotor demands for the c172x: refused, so tick 0 stays unanswered.
    let demand = Control::Multirotor(MultirotorRaw {
        roll_rate: 0.0,
        pitch_rate: 0.0,
        yaw_rate: 0.0,
        collective: 0.5,
    });
    tx.send(demand, false, 0).unwrap();
    app.update();
    assert!(app.world().resource::<SimStatus>().fatal.is_some(), "tick 0 was never answered");
    let samples = samples.lock().unwrap().clone();
    assert!(samples.len() >= 2, "tick 0 published, then re-sent");
    assert!(samples.iter().all(|t| t.tick == 0));
    assert_eq!(samples[0].command.refusals, 0, "first published before the command arrived");
    let resent = samples[1].command;
    assert_eq!(
        (resent.refusals, resent.last_refusal, resent.last_refused_epoch),
        (1, Refusal::WrongVehicleKind, tx.epoch())
    );
    assert_eq!(resent.owner_epoch, 0);
    // Everything else is still the tick's state.
    assert_eq!(samples[1], Telemetry { command: resent, ..samples[0] });
}

/// Run a lockstep scenario with a controller thread whose law depends only on
/// the telemetry it receives, for `ticks` ticks after the first update. With
/// `interrupt`, the run is paused at tick 80, held, stepped 7 ticks, and
/// resumed. Returns the final telemetry and the ticks the controller answered.
fn lockstep_run(
    ticks: u64,
    control_rate_hz: Option<f64>,
    interrupt: bool,
) -> (Vec<Telemetry>, Vec<u64>) {
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
    let tick = |app: &App| app.world().resource::<SimClock>().tick;
    let control = |r| send_control(&session, r, Duration::from_secs(2)).unwrap();
    app.update(); // startup publishes tick 0, then waits for its answer
    while tick(&app) < ticks + 1 {
        if interrupt && tick(&app) == 80 {
            control(ControlRequest::Pause);
            for _ in 0..3 {
                app.update(); // held, re-sending tick 80 to the controller
            }
            assert_eq!(tick(&app), 80);
            control(ControlRequest::Step(7));
            for _ in 0..9 {
                app.update();
            }
            assert_eq!(tick(&app), 87);
            control(ControlRequest::Resume);
        }
        app.update();
        assert!(app.world().resource::<SimStatus>().fatal.is_none());
    }
    stop.store(true, Ordering::Relaxed);
    let answered = controller.join().unwrap();
    (without_epochs(last(&app)), answered)
}

#[test]
fn lockstep_makes_networked_runs_reproducible() {
    let (a, answered) = lockstep_run(240, None, false);
    let (b, _) = lockstep_run(240, None, false);
    assert_eq!(of(&a, 1).tick, 241);
    assert_eq!(of(&a, 1).control_mode, ControlMode::Raw);
    // The sim waited on ticks 0..=240; tick 241's answer may or may not have
    // been sent before the controller stopped.
    assert_eq!(answered[..241], (0..=240).collect::<Vec<_>>(), "every tick is a control tick");
    assert_eq!(a, b, "a lockstep run with a deterministic controller replays exactly");
}

#[test]
fn pausing_and_stepping_a_lockstep_run_does_not_change_it() {
    let (straight, _) = lockstep_run(240, None, false);
    let (interrupted, answered) = lockstep_run(240, None, true);
    assert_eq!(answered[..241], (0..=240).collect::<Vec<_>>(), "each tick answered once");
    assert_eq!(straight, interrupted, "pause and step decide only when ticks run");
}

#[test]
fn decimated_lockstep_publishes_and_waits_only_on_control_ticks() {
    // 30 Hz control at 120 Hz physics: telemetry and the lockstep wait every 4th
    // tick, the last command held in between — and still reproducible.
    let (a, answered) = lockstep_run(240, Some(30.0), false);
    let (b, _) = lockstep_run(240, Some(30.0), false);
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
