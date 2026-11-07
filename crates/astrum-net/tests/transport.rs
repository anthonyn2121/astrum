//! Zenoh transport behaviour over a single in-process session (a session's own
//! publishers deliver to its own subscribers). Retry loops guard against residual
//! declaration-propagation lag.

use std::time::Duration;

use astrum_net::{
    fetch_scenario, keys, CommandReceiver, CommandSender, LatestTelemetry, SampledTelemetry,
    ScenarioServer, Selection, Session, TelemetryPublisher, TickPublisher, TickReceiver,
};
use astrum_proto::{encode_command, Command, Control, RawActuators, Telemetry, TickEnd};
use zenoh::Wait;

fn sample(id: u32, tick: u64) -> Telemetry {
    Telemetry { aircraft_id: id, tick, sim_time_s: tick as f64 / 120.0, ..Default::default() }
}

fn raw(a: f64) -> Control {
    Control::Raw(RawActuators { aileron: a, elevator: 0.0, rudder: 0.0, throttle: 0.5 })
}

#[test]
fn commands_arrive_in_order_with_increasing_seq_and_the_senders_epoch() {
    let session = Session::in_process().unwrap();
    let rx = CommandReceiver::bind(&session).unwrap();
    let mut tx = CommandSender::new(&session, 42).unwrap();
    for k in 0..3 {
        tx.send(raw(k as f64 / 10.0), k == 2, 100 + k).unwrap();
    }
    let mut got = Vec::new();
    for _ in 0..1000 {
        got.extend(rx.drain().into_iter().map(Result::unwrap));
        if got.len() == 3 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(got.iter().map(|c| c.seq).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert_eq!(got.iter().map(|c| c.telemetry_tick).collect::<Vec<_>>(), vec![100, 101, 102]);
    assert!(got[2].trigger && !got[0].trigger);
    assert!(got.iter().all(|c| c.target_id == 42));
    assert_ne!(tx.epoch(), 0);
    assert!(got.iter().all(|c| c.epoch == tx.epoch()));
}

#[test]
fn every_sender_gets_its_own_epoch() {
    let session = Session::in_process().unwrap();
    let a = CommandSender::new(&session, 1).unwrap();
    let b = CommandSender::new(&session, 1).unwrap();
    assert_ne!(a.epoch(), b.epoch());
}

#[test]
fn command_whose_payload_disagrees_with_its_key_is_rejected() {
    let session = Session::in_process().unwrap();
    let rx = CommandReceiver::bind(&session).unwrap();
    // Payload says aircraft 2, key says aircraft 1.
    let forged = Command {
        target_id: 2,
        seq: 1,
        telemetry_tick: 0,
        trigger: false,
        epoch: 1,
        control: raw(0.0),
    };
    let mut got = Vec::new();
    for _ in 0..1000 {
        session.raw().put(keys::command(1), encode_command(&forged)).wait().unwrap();
        got = rx.drain();
        if !got.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let err = got[0].clone().unwrap_err();
    assert!(err.reason.contains("targets 2"), "{err}");
}

#[test]
fn latest_telemetry_keeps_only_the_newest_sample_per_aircraft() {
    let session = Session::in_process().unwrap();
    let mut publisher = TelemetryPublisher::new(&session);
    let mut fleet = LatestTelemetry::subscribe(&session, Selection::Fleet).unwrap();
    // Establish the subscription, then publish a burst for two aircraft.
    for _ in 0..1000 {
        publisher.publish(&sample(1, 0)).unwrap();
        if fleet.latest(1).is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    for tick in 1..=50 {
        publisher.publish(&sample(1, tick)).unwrap();
        publisher.publish(&sample(2, tick)).unwrap();
    }
    fleet.wait(Duration::from_secs(1)).unwrap();
    for _ in 0..1000 {
        if fleet.latest(1).map(|t| t.tick) == Some(50)
            && fleet.latest(2).map(|t| t.tick) == Some(50)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    let snap = fleet.snapshot();
    assert_eq!(
        snap.iter().map(|t| (t.aircraft_id, t.tick)).collect::<Vec<_>>(),
        vec![(1, 50), (2, 50)]
    );
}

#[test]
fn aircraft_selection_excludes_other_ids() {
    let session = Session::in_process().unwrap();
    let mut publisher = TelemetryPublisher::new(&session);
    let one = LatestTelemetry::subscribe(&session, Selection::Aircraft(1)).unwrap();
    for _ in 0..1000 {
        publisher.publish(&sample(1, 1)).unwrap();
        publisher.publish(&sample(10, 1)).unwrap();
        if one.latest(1).is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(one.latest(1).is_some());
    assert!(one.latest(10).is_none(), "astrum/telemetry/1 must not match .../10");
}

#[test]
fn wait_times_out_without_new_samples() {
    let session = Session::in_process().unwrap();
    let mut sub = LatestTelemetry::subscribe(&session, Selection::Fleet).unwrap();
    assert_eq!(sub.wait(Duration::from_millis(20)), Err(astrum_net::NetError::Timeout));
}

/// A fleet trail on `session` thinned to `period_s`, already receiving: it has
/// seen tick 0 of every aircraft in `ids` (not yet taken).
fn sampled(
    session: &Session,
    publisher: &mut TelemetryPublisher,
    ids: &[u32],
    period_s: f64,
    cap: usize,
) -> SampledTelemetry {
    let mut sub = SampledTelemetry::subscribe(session, Selection::Fleet, period_s, cap).unwrap();
    for _ in 0..1000 {
        for &id in ids {
            publisher.publish(&sample(id, 0)).unwrap();
        }
        if sub.wait(Duration::from_millis(1)).is_ok() {
            break;
        }
    }
    settle(&mut sub);
    sub
}

/// Wait until `sub` has had no new sample for 50 ms: everything published has
/// been delivered.
fn settle(sub: &mut SampledTelemetry) {
    while sub.wait(Duration::from_millis(50)).is_ok() {}
}

fn ticks(batch: &[Telemetry]) -> Vec<(u32, u64)> {
    batch.iter().map(|t| (t.aircraft_id, t.tick)).collect()
}

#[test]
fn sampled_telemetry_keeps_one_sample_per_period_of_sim_time_plus_the_newest() {
    let session = Session::in_process().unwrap();
    let mut publisher = TelemetryPublisher::new(&session);
    // 120 Hz ticks thinned to 30 Hz: every 4th tick.
    let mut sub = sampled(&session, &mut publisher, &[1, 2], 1.0 / 30.0, 1000);
    for tick in 1..=118 {
        publisher.publish(&sample(2, tick)).unwrap();
        publisher.publish(&sample(1, tick)).unwrap();
    }
    settle(&mut sub);
    // Ids ascending, each in time order; tick 118 is the newest, closer than a
    // period to tick 116.
    let every_4th = (0..=116).step_by(4).chain([118]);
    let want: Vec<_> = every_4th.clone().map(|t| (1, t)).chain(every_4th.map(|t| (2, t))).collect();
    assert_eq!(ticks(&sub.take()), want);

    // Nothing new: each aircraft's newest again.
    assert_eq!(ticks(&sub.take()), [(1, 118), (2, 118)]);
    // The spacing carries on from the last sample taken, not from the grid.
    for tick in 119..=128 {
        publisher.publish(&sample(1, tick)).unwrap();
    }
    settle(&mut sub);
    assert_eq!(ticks(&sub.take()), [(1, 122), (1, 126), (1, 128), (2, 118)]);
    assert_eq!(sub.wait(Duration::from_millis(20)), Err(astrum_net::NetError::Timeout));
}

#[test]
fn sampled_telemetry_caps_each_aircraft_at_its_newest_samples() {
    let session = Session::in_process().unwrap();
    let mut publisher = TelemetryPublisher::new(&session);
    let mut sub = sampled(&session, &mut publisher, &[1], 1e-9, 10);
    for tick in 1..=100 {
        publisher.publish(&sample(1, tick)).unwrap();
    }
    settle(&mut sub);
    assert_eq!(ticks(&sub.take()), (91..=100).map(|t| (1, t)).collect::<Vec<_>>());
    assert_eq!(sub.capped(), 91);
}

#[test]
fn sampled_telemetry_drops_an_aircrafts_old_run_when_its_clock_goes_back() {
    let session = Session::in_process().unwrap();
    let mut publisher = TelemetryPublisher::new(&session);
    let mut sub = sampled(&session, &mut publisher, &[1], 1.0 / 30.0, 1000);
    for tick in (1..=40).chain(0..=8) {
        publisher.publish(&sample(1, tick)).unwrap();
    }
    settle(&mut sub);
    assert_eq!(ticks(&sub.take()), [(1, 0), (1, 4), (1, 8)]);
}

#[test]
fn tick_receiver_returns_the_newest_tick_end() {
    let session = Session::in_process().unwrap();
    let publisher = TickPublisher::new(&session).unwrap();
    let mut rx = TickReceiver::subscribe(&session).unwrap();
    for _ in 0..1000 {
        publisher.publish(&TickEnd { tick: 0, sim_time_s: 0.0 }).unwrap();
        if rx.latest().is_some() {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    rx.wait(Duration::from_secs(1)).unwrap();
    for tick in 1..=5 {
        publisher.publish(&TickEnd { tick, sim_time_s: tick as f64 / 120.0 }).unwrap();
    }
    for _ in 0..1000 {
        if rx.latest().map(|t| t.tick) == Some(5) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        rx.wait(Duration::from_secs(1)).unwrap(),
        TickEnd { tick: 5, sim_time_s: 5.0 / 120.0 }
    );
    assert_eq!(rx.wait(Duration::from_millis(20)), Err(astrum_net::NetError::Timeout));
}

#[test]
fn scenario_fetch_waits_for_a_server_that_appears_later() {
    // A recorder that starts with the sim may ask before the sim has declared
    // its queryable; the fetch must keep asking until its timeout.
    let session = Session::in_process().unwrap();
    let server = {
        let session = session.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            ScenarioServer::serve(&session, "[sim]\nrate_hz = 120.0\n".into()).unwrap()
        })
    };
    let got = fetch_scenario(&session, Duration::from_secs(3)).unwrap();
    let _server = server.join().unwrap();
    assert_eq!(got.as_deref(), Some("[sim]\nrate_hz = 120.0\n"));
}

#[test]
fn scenario_fetch_without_a_server_returns_none_after_the_timeout() {
    let session = Session::in_process().unwrap();
    let started = std::time::Instant::now();
    assert_eq!(fetch_scenario(&session, Duration::from_millis(300)).unwrap(), None);
    assert!(started.elapsed() >= Duration::from_millis(300));
}
