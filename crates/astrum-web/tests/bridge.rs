//! The bridge's attachment lifecycle against real Zenoh hubs: connecting in
//! the background, run ids and notes, clearing on re-attach and detach, and
//! the aircraft-models feed.

mod common;

use std::time::{Duration, Instant};

use astrum_bag::aircraft_models;
use astrum_net::{ScenarioServer, Session, TelemetryPublisher};
use astrum_proto::Telemetry;
use astrum_web::bridge::{Bridge, BridgeConfig, BridgeInfo, BATCH_SAMPLES};
use astrum_web::{models_message, run_message};
use axum::extract::ws::Utf8Bytes;
use common::{config, endpoint, wait_until, Sim};
use tokio::sync::watch;

/// Aircraft ids in a `{"type":"frames",…}` message, once each (a batch may
/// carry several samples of one aircraft, grouped by id).
fn ids(snapshot: &Option<Utf8Bytes>) -> Vec<u64> {
    let Some(text) = snapshot else { return Vec::new() };
    let msg: serde_json::Value = serde_json::from_str(text.as_str()).unwrap();
    let mut ids: Vec<u64> =
        msg["frames"].as_array().unwrap().iter().map(|f| f["id"].as_u64().unwrap()).collect();
    ids.dedup();
    ids
}

/// Ticks of every sample in a `{"type":"frames",…}` message, in order.
fn ticks(snapshot: &Option<Utf8Bytes>) -> Vec<u64> {
    let Some(text) = snapshot else { return Vec::new() };
    let msg: serde_json::Value = serde_json::from_str(text.as_str()).unwrap();
    msg["frames"].as_array().unwrap().iter().map(|f| f["tick"].as_u64().unwrap()).collect()
}

#[test]
fn a_new_bridge_is_detached_at_run_zero() {
    let bridge = Bridge::new(config());
    assert_eq!(bridge.info(), BridgeInfo { endpoint: None, run: 0, connected: false });
    assert!(bridge.session().is_none());
    assert_eq!(bridge.runs().borrow().message.as_str(), run_message(0, None));
    assert!(bridge.snapshots().borrow().is_none());
}

#[test]
fn an_attachment_waits_for_its_sim_and_detach_clears_it() {
    let at = endpoint(0);
    let bridge = Bridge::new(config());
    let runs = bridge.runs();

    assert_eq!(bridge.attach(&at), 1);
    assert_eq!(runs.borrow().message.as_str(), run_message(1, Some(&at)));
    // Nothing listens yet: attached, but not connected, for as long as it takes.
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(bridge.info(), BridgeInfo { endpoint: Some(at.clone()), run: 1, connected: false });
    assert!(bridge.session().is_none());

    let _sim = Session::listen(&at).unwrap();
    wait_until("the bridge to connect", || bridge.info().connected);
    assert!(bridge.session().is_some());

    bridge.detach();
    assert_eq!(bridge.info(), BridgeInfo { endpoint: None, run: 2, connected: false });
    assert!(bridge.session().is_none());
    assert_eq!(runs.borrow().message.as_str(), run_message(2, None));
}

#[test]
fn reattaching_drops_the_old_sims_aircraft() {
    let (a, b) = (endpoint(1), endpoint(2));
    let _sim_a = Sim::flying(&a, 1);
    let _sim_b = Sim::flying(&b, 2);
    let bridge = Bridge::new(config());
    let mut snapshots = bridge.snapshots();

    bridge.attach(&a);
    wait_until("aircraft 1", || ids(&snapshots.borrow_and_update()) == [1]);

    assert_eq!(bridge.attach(&b), 2);
    // Cleared at once, and from then on only the new sim's aircraft.
    assert!(snapshots.borrow_and_update().is_none());
    let mut seen = Vec::new();
    wait_until("aircraft 2", || {
        seen = ids(&snapshots.borrow_and_update());
        assert!(!seen.contains(&1), "aircraft 1 of the old sim reappeared: {seen:?}");
        seen == [2]
    });
    // The old attachment's pumps are gone: aircraft 1 never comes back.
    let until = Instant::now() + Duration::from_secs(1);
    while Instant::now() < until {
        assert_eq!(ids(&snapshots.borrow_and_update()), [2]);
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn detaching_stops_publishing() {
    let at = endpoint(3);
    let _sim = Sim::flying(&at, 7);
    let bridge = Bridge::new(config());
    let mut snapshots = bridge.snapshots();
    bridge.attach(&at);
    wait_until("aircraft 7", || ids(&snapshots.borrow_and_update()) == [7]);

    bridge.detach();
    std::thread::sleep(Duration::from_secs(1));
    assert!(snapshots.borrow().is_none(), "a detached bridge still published a snapshot");
}

#[test]
fn a_batch_holds_at_most_batch_samples_per_aircraft_spread_over_the_whole_burst() {
    let at = endpoint(6);
    let hub = Session::listen(&at).unwrap();
    // One batch per 2 s, so a burst lands in a single batch.
    let bridge = Bridge::new(BridgeConfig { min_period: Duration::from_secs(2), ..config() });
    let mut snapshots = bridge.snapshots();
    bridge.attach(&at);
    let mut telemetry = TelemetryPublisher::new(&hub);
    // One sample per 30 Hz sample period, so the thinning keeps every one.
    let mut publish = |tick: u64| {
        let t = Telemetry {
            aircraft_id: 1,
            tick,
            sim_time_s: tick as f64 / 30.0,
            ..Default::default()
        };
        telemetry.publish(&t).unwrap();
    };
    // The first batch goes out at once; the next waits out the period.
    wait_until("aircraft 1", || {
        publish(0);
        ids(&snapshots.borrow_and_update()) == [1]
    });
    for tick in 1..=1000 {
        publish(tick);
        std::thread::sleep(Duration::from_micros(20));
    }
    let mut batch = Vec::new();
    wait_until("the burst's batch", || {
        batch = ticks(&snapshots.borrow_and_update());
        batch.last() == Some(&1000)
    });
    // 1000 samples do not fit 120: the trail halves until they do (every 16th
    // tick fits) and still covers the whole burst, from its first sample to
    // its newest — not just the newest 120 ticks.
    assert!(batch.len() <= BATCH_SAMPLES, "{} samples", batch.len());
    assert_eq!(batch.first(), Some(&1));
    let widest = batch.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    assert!((1..=17).contains(&widest), "gaps up to {widest} ticks in {batch:?}");
    assert!(batch.len() >= 1000 / 17, "{} samples", batch.len());
}

const F16_PAIR: &str =
    "[[aircraft]]\nid = 1\nmodel = \"f16\"\n[[aircraft]]\nid = 2\nmodel = \"f16\"\n";
const TRAINER: &str = "[[aircraft]]\nid = 1\nmodel = \"c172x\"\n";

/// Poll `rx` until it holds `want` (marking it seen), failing after 10 s.
fn wait_for(rx: &mut watch::Receiver<Option<Utf8Bytes>>, want: &str) {
    wait_until(want, || rx.borrow_and_update().as_ref().map(Utf8Bytes::as_str) == Some(want));
}

#[test]
fn models_publish_each_new_fleet_once_and_survive_unanswered_polls() {
    let at = endpoint(4);
    let hub = Session::listen(&at).unwrap();
    let scenario = ScenarioServer::serve(&hub, F16_PAIR.to_owned()).unwrap();
    let bridge = Bridge::new(config());
    let mut rx = bridge.models();
    bridge.attach(&at);

    let first = models_message(&aircraft_models(F16_PAIR));
    wait_for(&mut rx, &first);
    // The same scenario on every poll is not a change.
    std::thread::sleep(config().models_poll * 5);
    assert!(!rx.has_changed().unwrap());

    // Unanswered polls (each waits out its timeout) keep the last fleet.
    drop(scenario);
    std::thread::sleep(Duration::from_millis(2500));
    assert!(!rx.has_changed().unwrap());
    assert_eq!(rx.borrow().as_ref().map(Utf8Bytes::as_str), Some(first.as_str()));

    // A sim that comes back with another fleet replaces it.
    let _scenario = ScenarioServer::serve(&hub, TRAINER.to_owned()).unwrap();
    wait_for(&mut rx, &models_message(&aircraft_models(TRAINER)));

    // A new run starts without models.
    bridge.attach(&endpoint(5));
    assert!(rx.borrow_and_update().is_none());
}

#[test]
fn a_large_fleets_models_reach_the_bridge() {
    // A 1000-aircraft scenario is ~150 KB: far over one low-latency frame
    // (~48 KiB on loopback), so it only arrives because queries use their own
    // default-transport link, which fragments.
    let at = endpoint(7);
    let hub = Session::listen(&at).unwrap();
    let mut toml = String::from("[sim]\nrate_hz = 120.0\n");
    for id in 1..=1000 {
        toml += &format!(
            "\n[[aircraft]]\nid = {id}\nmodel = \"f16\"\nlat_deg = 37.6188\nlon_deg = -122.3756\n\
             alt_hae_m = 4572.0\nheading_deg = 63.0\ntas_mps = 120.0\nteam = 0\nhealth = 100.0\n"
        );
    }
    assert!(toml.len() > 100_000);
    let _scenario = ScenarioServer::serve(&hub, toml.clone()).unwrap();
    let bridge = Bridge::new(config());
    let mut rx = bridge.models();
    bridge.attach(&at);
    let models = aircraft_models(&toml);
    assert_eq!(models.len(), 1000);
    wait_for(&mut rx, &models_message(&models));
}
