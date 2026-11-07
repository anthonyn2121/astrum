//! The clients' transport against a real hub over TCP: `Session::connect` uses
//! Zenoh's low-latency transport, where a client that falls behind slows the
//! publisher instead of losing samples — and so never trips Zenoh 1.10's
//! congestion latch, which on the default transport dropped every later sample
//! to such a client (with the default transport the burst test fails: ~7000 of
//! the 20000 samples arrive, then none) — and where one message must fit in one
//! frame.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use astrum_net::{
    fetch_scenario, keys, zenoh, NetError, ScenarioServer, Session, TelemetryPublisher,
    MAX_SCENARIO_BYTES,
};
use astrum_proto::{decode_telemetry, Telemetry};
use zenoh::Wait;

/// A hub and a client on this process's port `n` (< 10) of the 20000–39999
/// band (topology.rs uses the band's first port per pid).
fn hub_and_client(n: u32) -> (Session, Session) {
    assert!(n < 10);
    let port = 20_000 + (std::process::id() % 1_900) * 10 + 10 + n;
    let endpoint = format!("tcp/127.0.0.1:{port}");
    let hub = Session::listen(&endpoint).unwrap();
    let client = Session::connect(&endpoint, Duration::from_secs(10)).unwrap();
    (hub, client)
}

#[test]
fn a_slow_client_keeps_receiving_through_a_saturating_burst() {
    let (hub, client) = hub_and_client(0);
    // A subscriber far slower than the publisher: ~20 µs or more per sample.
    let (seen, newest) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (count, last) = (Arc::clone(&seen), Arc::clone(&newest));
    let _sub = client
        .raw()
        .declare_subscriber(keys::TELEMETRY_ALL)
        .callback(move |sample| {
            std::thread::sleep(Duration::from_micros(20));
            let t = decode_telemetry(&sample.payload().to_bytes()).unwrap();
            count.fetch_add(1, Ordering::Relaxed);
            last.store(t.tick, Ordering::Relaxed);
        })
        .wait()
        .unwrap();
    let mut publisher = TelemetryPublisher::new(&hub);
    let sample = |tick| Telemetry { aircraft_id: 1, tick, ..Telemetry::default() };
    // Until the subscription has reached the hub.
    let deadline = Instant::now() + Duration::from_secs(10);
    while seen.load(Ordering::Relaxed) == 0 {
        assert!(Instant::now() < deadline, "the subscription never reached the hub");
        publisher.publish(&sample(0)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
    }
    let before = seen.load(Ordering::Relaxed);

    // The burst: as fast as the hub can publish.
    const BURST: u64 = 20_000;
    for tick in 1..=BURST {
        publisher.publish(&sample(tick)).unwrap();
    }
    // Afterwards the client still hears new samples...
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut tick = BURST;
    while newest.load(Ordering::Relaxed) <= BURST {
        assert!(
            Instant::now() < deadline,
            "no sample after the burst reached the client ({} of {BURST} did)",
            seen.load(Ordering::Relaxed) - before
        );
        tick += 1;
        publisher.publish(&sample(tick)).unwrap();
        std::thread::sleep(Duration::from_millis(10));
    }
    // ...and none of the burst was lost.
    assert!(seen.load(Ordering::Relaxed) - before >= BURST);
}

/// A scenario TOML with `n` aircraft, as `astrum-sim` serves it.
fn fleet(n: u32) -> String {
    let mut toml = String::from("[world]\norigin_lat_deg = 37.6188\norigin_lon_deg = -122.3756\n");
    toml += "origin_alt_hae_m = 0.0\n\n[sim]\nrate_hz = 120.0\nmax_time_s = 60.0\n";
    for id in 1..=n {
        toml += &format!(
            "\n[[aircraft]]\nid = {id}\nmodel = \"f16\"\nlat_deg = 37.6188\nlon_deg = -122.3756\n\
             alt_hae_m = 4572.0\nheading_deg = 63.0\ntas_mps = 120.0\nteam = {}\n",
            id % 2
        );
    }
    toml
}

#[test]
fn a_large_fleets_scenario_fits_one_message_and_reaches_a_client() {
    let (hub, client) = hub_and_client(1);
    let toml = fleet(64);
    let _server = ScenarioServer::serve(&hub, toml.clone()).unwrap();
    assert_eq!(fetch_scenario(&client, Duration::from_secs(5)).unwrap(), Some(toml));
}

#[test]
fn the_largest_servable_scenario_reaches_a_client_and_a_larger_one_is_refused() {
    let (hub, client) = hub_and_client(2);
    let mut toml = fleet(1);
    toml += "\n# padding";
    toml += &"x".repeat(MAX_SCENARIO_BYTES - toml.len() - 1);
    toml += "\n";
    assert_eq!(toml.len(), MAX_SCENARIO_BYTES);
    let server = ScenarioServer::serve(&hub, toml.clone()).unwrap();
    assert_eq!(fetch_scenario(&client, Duration::from_secs(5)).unwrap(), Some(toml.clone()));
    drop(server);

    let err = ScenarioServer::serve(&hub, toml + "#").err().expect("an oversized scenario");
    let NetError::TooLarge(why) = err else { panic!("{err}") };
    assert!(why.contains(&format!("{} bytes", MAX_SCENARIO_BYTES + 1)), "{why}");
}
