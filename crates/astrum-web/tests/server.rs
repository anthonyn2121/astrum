//! Behaviour of the HTTP side of `astrum-web`: tile-path validation and the
//! cache-backed tile proxy, static serving confinement, the WebSocket Origin
//! check, and what one browser's WebSocket carries (run notes, snapshots,
//! events, aircraft models) as the bridge attaches to real Zenoh hubs.

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use astrum_bag::aircraft_models;
use astrum_net::{ScenarioServer, Session};
use astrum_web::bridge::{Bridge, BridgeConfig};
use astrum_web::server::{host_allowed, origin_allowed, parse_tile};
use astrum_web::{models_message, run_message};
use axum::body::{to_bytes, Body};
use axum::http::header::{HOST, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use common::{config, endpoint, Sim, SECRET};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tower::ServiceExt;

/// A server over a temp tree ([`common::app`]) with a detached bridge.
fn app(tmp: &Path) -> Router {
    common::app(tmp, Bridge::new(config()))
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::get(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    (status, to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec())
}

#[test]
fn tile_paths_accept_in_range_coordinates() {
    assert_eq!(parse_tile("0/0/0.jpg"), Some((0, 0, 0)));
    assert_eq!(parse_tile("14/3/5.jpg"), Some((14, 3, 5)));
    assert_eq!(parse_tile("14/3/5"), Some((14, 3, 5)));
    let max = (1u32 << 22) - 1;
    assert_eq!(parse_tile(&format!("22/{max}/{max}.jpg")), Some((22, max, max)));
}

#[test]
fn tile_paths_reject_traversal_junk_and_out_of_range() {
    for bad in [
        "../../etc/passwd",
        "14/../5.jpg",
        "14/3/../../5.jpg",
        "14/3/5.jpg/extra",
        "14/3",
        "",
        "14//5.jpg",
        "+14/3/5.jpg",
        "14/-3/5.jpg",
        "14/3/5.png",
        "14/3/5.jpg.jpg",
        "1/2/0.jpg",  // x out of range for zoom 1
        "1/0/2.jpg",  // y out of range for zoom 1
        "23/0/0.jpg", // zoom beyond the maximum
        "99999999999/0/0.jpg",
    ] {
        assert_eq!(parse_tile(bad), None, "{bad:?} must be rejected");
    }
}

#[tokio::test]
async fn tile_proxy_serves_cache_and_rejects_bad_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let app = app(tmp.path());

    let (status, body) = get(&app, "/tiles/14/3/5.jpg").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, b"\xff\xd8jpeg");

    // Cache-only mode: an uncached tile is simply unavailable.
    assert_eq!(get(&app, "/tiles/14/3/6.jpg").await.0, StatusCode::NOT_FOUND);

    for bad in
        ["/tiles/../../secret.txt", "/tiles/14/3/..%2F..%2F..%2Fsecret.txt", "/tiles/23/0/0.jpg"]
    {
        let (status, body) = get(&app, bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
        assert!(!String::from_utf8_lossy(&body).contains(SECRET));
    }
}

#[tokio::test]
async fn static_files_are_confined_to_the_root() {
    let tmp = tempfile::tempdir().unwrap();
    let app = app(tmp.path());

    let (status, body) = get(&app, "/").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, b"<h1>astrum</h1>");

    for bad in [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/%2E%2E%2Fsecret.txt",
        "/x/../../secret.txt",
        "/..%5csecret.txt",
        "//secret.txt",
    ] {
        let (status, body) = get(&app, bad).await;
        assert_ne!(status, StatusCode::OK, "{bad}");
        assert!(!String::from_utf8_lossy(&body).contains(SECRET), "{bad} leaked the file");
    }
}

fn headers(origin: Option<&str>, host: Option<&str>) -> HeaderMap {
    let mut h = HeaderMap::new();
    if let Some(o) = origin {
        h.insert(ORIGIN, HeaderValue::from_str(o).unwrap());
    }
    if let Some(v) = host {
        h.insert(HOST, HeaderValue::from_str(v).unwrap());
    }
    h
}

#[test]
fn origin_check_accepts_same_host_and_originless_clients() {
    assert!(origin_allowed(&headers(None, Some("localhost:8080"))));
    assert!(origin_allowed(&headers(None, None)));
    assert!(origin_allowed(&headers(Some("http://localhost:8080"), Some("localhost:8080"))));
    assert!(origin_allowed(&headers(Some("http://LocalHost:8080"), Some("localhost:8080"))));
    assert!(origin_allowed(&headers(Some("http://example.com"), Some("example.com"))));
    assert!(origin_allowed(&headers(Some("http://example.com"), Some("example.com:80"))));
    assert!(origin_allowed(&headers(Some("https://example.com"), Some("example.com"))));
    assert!(origin_allowed(&headers(Some("http://[::1]:8080"), Some("[::1]:8080"))));
}

#[test]
fn origin_check_rejects_cross_origin_pages() {
    for (origin, host) in [
        ("http://evil.example", "localhost:8080"),
        ("http://localhost:9999", "localhost:8080"),
        ("http://localhost", "localhost:8080"),
        ("https://example.com", "example.com:80"),
        ("http://localhost.evil.example:8080", "localhost:8080"),
        ("null", "localhost:8080"),
        ("file://", "localhost:8080"),
        ("ws://localhost:8080", "localhost:8080"),
        ("http://user@localhost:8080", "localhost:8080"),
        ("http://localhost:8080/path", "localhost:8080"),
    ] {
        assert!(!origin_allowed(&headers(Some(origin), Some(host))), "{origin} vs {host}");
    }
    // A browser Origin without a Host to compare against is refused.
    assert!(!origin_allowed(&headers(Some("http://localhost:8080"), None)));
}

#[test]
fn host_check_accepts_localhost_ip_literals_and_allowed_names_only() {
    let allowed = ["studio.lan".to_owned()];
    for host in [
        "localhost",
        "localhost:8080",
        "LOCALHOST:8080",
        "127.0.0.1:8080",
        "0.0.0.0",
        "192.168.1.20:8080",
        "[::1]:8080",
        "[fe80::1]",
        "studio.lan:8080",
        "Studio.Lan",
    ] {
        assert!(host_allowed(&headers(None, Some(host)), &allowed), "{host}");
    }
    for host in [
        "evil.example",
        "evil.example:8080",
        "localhost.evil.example",
        "127.0.0.1.nip.io",
        "studio.lan.evil.example",
        "lan",
        "user@localhost:8080",
        "",
    ] {
        assert!(!host_allowed(&headers(None, Some(host)), &allowed), "{host}");
    }
    assert!(!host_allowed(&headers(None, None), &allowed), "a missing Host is refused");
    assert!(!host_allowed(&headers(None, Some("studio.lan")), &[]), "listed names only");
}

/// Send a raw WebSocket upgrade for `/telemetry` naming `host` and read the
/// response head: its status code and whatever bytes followed the head.
async fn upgrade(stream: &mut TcpStream, host: &str, origin: Option<&str>) -> (String, Vec<u8>) {
    let origin = origin.map(|o| format!("Origin: {o}\r\n")).unwrap_or_default();
    let req = format!(
        "GET /telemetry HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n{origin}\r\n"
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    let head_end = loop {
        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "server closed before the end of the response head");
        buf.extend_from_slice(&chunk[..n]);
    };
    let line = String::from_utf8_lossy(&buf).lines().next().unwrap_or_default().to_owned();
    let status = line.split_whitespace().nth(1).unwrap_or_default().to_owned();
    (status, buf.split_off(head_end))
}

/// Status code of a raw WebSocket upgrade naming `host` against a running
/// server.
async fn upgrade_status(addr: std::net::SocketAddr, host: &str, origin: Option<&str>) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    upgrade(&mut stream, host, origin).await.0
}

/// Split one complete server frame (unmasked, unfragmented) off the front of
/// `buf`: its opcode and payload, or `None` until all of it has arrived.
fn take_frame(buf: &mut Vec<u8>) -> Option<(u8, Vec<u8>)> {
    let (&b0, &b1) = (buf.first()?, buf.get(1)?);
    let (len, head) = match b1 & 0x7f {
        126 => (usize::from(u16::from_be_bytes(buf.get(2..4)?.try_into().unwrap())), 4),
        127 => (u64::from_be_bytes(buf.get(2..10)?.try_into().unwrap()) as usize, 10),
        n => (usize::from(n), 2),
    };
    if buf.len() < head + len {
        return None;
    }
    assert!(b0 & 0x80 != 0, "fragmented frame");
    let frame: Vec<u8> = buf.drain(..head + len).collect();
    Some((b0 & 0x0f, frame[head..].to_vec()))
}

/// The next text message on an upgraded socket (`buf` holds bytes already
/// read), skipping control frames such as pings.
async fn next_text(stream: &mut TcpStream, buf: &mut Vec<u8>) -> String {
    let mut chunk = [0u8; 1024];
    loop {
        while let Some((opcode, payload)) = take_frame(buf) {
            match opcode {
                0x1 => return String::from_utf8(payload).unwrap(),
                0x8 => panic!("the server closed the WebSocket"),
                _ => {}
            }
        }
        let n = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
            .await
            .expect("no WebSocket message within 5 s")
            .unwrap();
        assert!(n > 0, "server closed the connection");
        buf.extend_from_slice(&chunk[..n]);
    }
}

#[tokio::test]
async fn websocket_upgrade_enforces_origin_and_host() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app_allowing(tmp.path(), Bridge::new(config()), &["studio.lan"]);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    let here = addr.to_string();

    assert_eq!(upgrade_status(addr, &here, None).await, "101");
    assert_eq!(upgrade_status(addr, &here, Some(&format!("http://{addr}"))).await, "101");
    assert_eq!(upgrade_status(addr, &here, Some("http://evil.example")).await, "403");
    assert_eq!(upgrade_status(addr, &here, Some("http://127.0.0.1:1")).await, "403");
    // Same-origin pages on the names a browser reaches this server by.
    for host in ["localhost:8080", "[::1]:8080", "studio.lan:8080"] {
        assert_eq!(upgrade_status(addr, host, Some(&format!("http://{host}"))).await, "101");
    }
    // A rebound attacker domain is same-origin with itself, but not allowed.
    let rebound = "evil.example:8080";
    assert_eq!(upgrade_status(addr, rebound, Some("http://evil.example:8080")).await, "403");
    assert_eq!(upgrade_status(addr, rebound, None).await, "403");
}

/// Serve `bridge` on an ephemeral port and open a WebSocket to it: the stream
/// and the bytes already read past the upgrade.
async fn browser(tmp: &Path, bridge: Bridge) -> (TcpStream, Vec<u8>) {
    let app = common::app(tmp, bridge);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let (status, buf) = upgrade(&mut stream, &addr.to_string(), None).await;
    assert_eq!(status, "101");
    (stream, buf)
}

const F16_PAIR: &str =
    "[[aircraft]]\nid = 1\nmodel = \"f16\"\n[[aircraft]]\nid = 2\nmodel = \"f16\"\n";
const TRAINER: &str = "[[aircraft]]\nid = 1\nmodel = \"c172x\"\n";

#[tokio::test(flavor = "multi_thread")]
async fn websocket_sends_the_run_then_the_models_on_connect_and_on_every_change() {
    let at = endpoint(0);
    let hub = Session::listen(&at).unwrap();
    let scenario = ScenarioServer::serve(&hub, F16_PAIR.to_owned()).unwrap();
    let bridge = Bridge::new(config());
    bridge.attach(&at);
    let tmp = tempfile::tempdir().unwrap();
    let (mut stream, mut buf) = browser(tmp.path(), bridge.clone()).await;

    assert_eq!(next_text(&mut stream, &mut buf).await, run_message(1, Some(&at)));
    // No telemetry, so the models message is the next thing sent.
    let first = models_message(&aircraft_models(F16_PAIR));
    assert_eq!(next_text(&mut stream, &mut buf).await, first);

    drop(scenario);
    let _scenario = ScenarioServer::serve(&hub, TRAINER.to_owned()).unwrap();
    let second = models_message(&aircraft_models(TRAINER));
    assert_eq!(next_text(&mut stream, &mut buf).await, second);
}

/// The aircraft a `frames` or `events` message is about.
fn aircraft(msg: &serde_json::Value) -> Vec<u64> {
    let (list, key) = match msg["type"].as_str() {
        Some("frames") => ("frames", "id"),
        Some("events") => ("events", "aircraft_id"),
        _ => return Vec::new(),
    };
    msg[list].as_array().unwrap().iter().map(|x| x[key].as_u64().unwrap()).collect()
}

/// Read messages until both a `frames` and an `events` message about aircraft
/// `id` arrived, asserting that every frame and event in between is about it.
async fn until_flying(stream: &mut TcpStream, buf: &mut Vec<u8>, id: u64) {
    let (mut frames, mut events) = (false, false);
    while !(frames && events) {
        let text = next_text(stream, buf).await;
        let msg: serde_json::Value = serde_json::from_str(&text).unwrap();
        let about = aircraft(&msg);
        assert!(about.iter().all(|&a| a == id), "expected only aircraft {id}: {text}");
        match msg["type"].as_str() {
            Some("frames") => frames |= !about.is_empty(),
            Some("events") => events |= !about.is_empty(),
            _ => {}
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn reattaching_announces_a_new_run_and_never_mixes_the_sims() {
    let (a, b) = (endpoint(1), endpoint(2));
    let _sim_a = Sim::flying(&a, 1);
    let _sim_b = Sim::flying(&b, 2);
    let bridge = Bridge::new(config());
    bridge.attach(&a);
    let tmp = tempfile::tempdir().unwrap();
    let (mut stream, mut buf) = browser(tmp.path(), bridge.clone()).await;

    assert_eq!(next_text(&mut stream, &mut buf).await, run_message(1, Some(&a)));
    until_flying(&mut stream, &mut buf, 1).await;

    assert_eq!(bridge.attach(&b), 2);
    let note = run_message(2, Some(&b));
    // Skip the old run's tail, up to the new run's note.
    while next_text(&mut stream, &mut buf).await != note {}
    // After the note: only the new sim's telemetry and events.
    until_flying(&mut stream, &mut buf, 2).await;
}

/// A `frames` message's samples as `(aircraft id, tick)`, in message order;
/// `None` for any other message.
fn samples(text: &str) -> Option<Vec<(u64, u64)>> {
    let msg: serde_json::Value = serde_json::from_str(text).unwrap();
    let frames = msg["frames"].as_array()?;
    Some(frames.iter().map(|f| (f["id"].as_u64().unwrap(), f["tick"].as_u64().unwrap())).collect())
}

/// One aircraft's ticks in a batch.
fn ticks_of(batch: &[(u64, u64)], id: u64) -> Vec<u64> {
    batch.iter().filter(|(a, _)| *a == id).map(|&(_, t)| t).collect()
}

/// [`config`] batching at the default 30 Hz.
fn at_30_hz() -> BridgeConfig {
    BridgeConfig { min_period: Duration::from_secs_f64(1.0 / 30.0), ..config() }
}

#[tokio::test(flavor = "multi_thread")]
async fn faster_than_real_time_a_browser_gets_each_aircrafts_trail_at_the_sample_rate() {
    // 120 Hz ticks at 50× real time, sampled at 30 Hz of sim time: every 4th
    // tick, ~50 per aircraft per batch where one-per-flush gave one.
    let at = endpoint(3);
    let _sim = Sim::paced(&at, &[1, 2], 120.0, 50.0);
    let bridge = Bridge::new(at_30_hz());
    bridge.attach(&at);
    let tmp = tempfile::tempdir().unwrap();
    let (mut stream, mut buf) = browser(tmp.path(), bridge.clone()).await;

    // Per aircraft: (tick, whether it ended its batch). The first batch with
    // both aircraft is only an anchor: the trails may begin before it, and
    // before their spacing does.
    let mut trails: BTreeMap<u64, Vec<(u64, bool)>> = BTreeMap::new();
    let mut batches = 0;
    while trails.get(&1).map_or(0, Vec::len) < 600 {
        let Some(batch) = samples(&next_text(&mut stream, &mut buf).await) else { continue };
        if batches == 0 && ticks_of(&batch, 2).is_empty() {
            continue; // the bridge has not heard aircraft 2 yet
        }
        for id in [1, 2] {
            let ticks = ticks_of(&batch, id);
            assert!(!ticks.is_empty(), "a batch without aircraft {id}: {batch:?}");
            let trail = trails.entry(id).or_default();
            let from = if batches == 0 { ticks.len() - 1 } else { 0 };
            let last = ticks.len() - 1;
            trail.extend(ticks.iter().enumerate().skip(from).map(|(i, &t)| (t, i == last)));
        }
        batches += 1;
    }
    for (id, trail) in &trails {
        for pair in trail.windows(2) {
            let ((a, _), (b, ends_batch)) = (pair[0], pair[1]);
            // Never a gap longer than one sample period, within or across
            // batches; exactly one period except up to a batch's newest.
            assert!(b > a && b - a <= 4, "aircraft {id}: tick {a} then {b}");
            if !ends_batch {
                assert_eq!(b - a, 4, "aircraft {id}: tick {a} then {b}");
            }
        }
    }
    let per_batch = trails[&1].len() / batches;
    assert!(per_batch >= 20, "{per_batch} samples per batch at 50× real time");
}

#[tokio::test(flavor = "multi_thread")]
async fn at_real_time_a_browser_gets_one_sample_per_aircraft_per_batch() {
    // A 50 Hz sim at 1×, batched at 30 Hz: as before thinning, each batch is
    // the aircraft's newest sample (two only when a flush ran a tick late).
    let at = endpoint(4);
    let _sim = Sim::flying(&at, 1);
    let bridge = Bridge::new(at_30_hz());
    bridge.attach(&at);
    let tmp = tempfile::tempdir().unwrap();
    let (mut stream, mut buf) = browser(tmp.path(), bridge.clone()).await;

    let mut counts = Vec::new();
    let mut newest = 0;
    let mut started = None;
    while counts.len() < 45 {
        let Some(batch) = samples(&next_text(&mut stream, &mut buf).await) else { continue };
        let ticks = ticks_of(&batch, 1);
        assert!(ticks.iter().all(|&t| t > newest), "{newest} then {ticks:?}");
        newest = *ticks.last().unwrap();
        counts.push(ticks.len());
        started.get_or_insert_with(std::time::Instant::now);
    }
    let elapsed = started.unwrap().elapsed().as_secs_f64();
    assert!(elapsed >= 44.0 / 30.0 * 0.95, "45 batches in {elapsed} s: over 30 Hz");
    assert!(counts.iter().all(|&n| n <= 2), "{counts:?}");
    let single = counts.iter().filter(|&&n| n == 1).count();
    assert!(single * 10 >= counts.len() * 9, "{counts:?}");
}
