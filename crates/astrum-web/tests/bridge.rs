//! The bridge's attachment lifecycle against real Zenoh hubs: connecting in
//! the background, run ids and notes, clearing on re-attach and detach, and
//! the aircraft-models feed.

mod common;

use std::time::{Duration, Instant};

use astrum_bag::aircraft_models;
use astrum_net::{ScenarioServer, Session};
use astrum_web::bridge::{Bridge, BridgeInfo};
use astrum_web::{models_message, run_message};
use axum::extract::ws::Utf8Bytes;
use common::{config, endpoint, wait_until, Sim};
use tokio::sync::watch;

/// Aircraft ids in a `{"type":"frames",…}` message.
fn ids(snapshot: &Option<Utf8Bytes>) -> Vec<u64> {
    let Some(text) = snapshot else { return Vec::new() };
    let msg: serde_json::Value = serde_json::from_str(text.as_str()).unwrap();
    msg["frames"].as_array().unwrap().iter().map(|f| f["id"].as_u64().unwrap()).collect()
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
