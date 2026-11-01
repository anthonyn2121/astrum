//! Behaviour of the HTTP side of `astrum-web`: tile-path validation and the
//! cache-backed tile proxy, static serving confinement, and the WebSocket
//! Origin check.

use std::path::Path;

use astrum_web::server::{origin_allowed, parse_tile, router, Config, Snapshot};
use axum::body::{to_bytes, Body};
use axum::http::header::{HOST, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, watch};
use tower::ServiceExt;

const SECRET: &str = "top-secret-outside-root";

/// A server over a temp tree: `<tmp>/web/index.html`, `<tmp>/secret.txt`
/// (outside the static root), and a cache-only tile cache holding `14/3/5.jpg`.
fn app(tmp: &Path) -> Router {
    let web = tmp.join("web");
    std::fs::create_dir_all(&web).unwrap();
    std::fs::write(web.join("index.html"), "<h1>astrum</h1>").unwrap();
    std::fs::write(tmp.join("secret.txt"), SECRET).unwrap();
    let cache = tmp.join("tiles");
    std::fs::create_dir_all(cache.join("14/3")).unwrap();
    std::fs::write(cache.join("14/3/5.jpg"), b"\xff\xd8jpeg").unwrap();
    let cfg = Config { root: web, tile_url: String::new(), tile_cache: cache };
    let (_tx, rx) = watch::channel::<Snapshot>(None);
    let (events, _) = broadcast::channel(1);
    router(cfg, rx, events).unwrap()
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

/// Status code of a raw WebSocket upgrade against a running server.
async fn upgrade_status(addr: std::net::SocketAddr, origin: Option<&str>) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let origin = origin.map(|o| format!("Origin: {o}\r\n")).unwrap_or_default();
    let req = format!(
        "GET /telemetry HTTP/1.1\r\nHost: {addr}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n{origin}\r\n"
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 256];
    while !buf.windows(2).any(|w| w == b"\r\n") {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "server closed before a status line");
        buf.extend_from_slice(&chunk[..n]);
    }
    let line = String::from_utf8_lossy(&buf).lines().next().unwrap_or_default().to_owned();
    line.split_whitespace().nth(1).unwrap_or_default().to_owned()
}

#[tokio::test]
async fn websocket_upgrade_enforces_origin() {
    let tmp = tempfile::tempdir().unwrap();
    let app = app(tmp.path());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });

    assert_eq!(upgrade_status(addr, None).await, "101");
    assert_eq!(upgrade_status(addr, Some(&format!("http://{addr}"))).await, "101");
    assert_eq!(upgrade_status(addr, Some("http://evil.example")).await, "403");
    assert_eq!(upgrade_status(addr, Some("http://127.0.0.1:1")).await, "403");
}
