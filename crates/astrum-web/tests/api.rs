//! The JSON API: the guard (Host allow-list on every route; Origin and body
//! type on non-GET ones), control-body validation, the relay to a real sim's
//! control plane (and its refusals), and the bridge status, including while
//! no sim is reachable.

mod common;

use std::time::Duration;

use astrum_net::{ControlRequest, ControlServer, Session, MIN_SPEED};
use astrum_web::api::safe_name;
use astrum_web::bridge::Bridge;
use axum::body::{to_bytes, Body};
use axum::http::header::{CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use common::{config, endpoint, wait_until};
use serde_json::{json, Value};
use tower::ServiceExt;

const CONTROL: &str = "/api/bridge/control";
/// The `Host` a browser on the default bind sends.
const LOCAL: &str = "localhost:8080";

/// Send `req`; the status and the JSON body.
async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    assert_eq!(
        resp.headers().get(CONTENT_TYPE).map(|v| v.to_str().unwrap()),
        Some("application/json"),
        "API responses are JSON ({status})"
    );
    let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    call(app, Request::get(uri).header(HOST, LOCAL).body(Body::empty()).unwrap()).await
}

/// An originless JSON `POST` of `body` to the control route.
async fn control(app: &Router, body: &str) -> (StatusCode, Value) {
    let req = Request::post(CONTROL)
        .header(HOST, LOCAL)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    call(app, req).await
}

fn assert_error(reply: &(StatusCode, Value), status: StatusCode, needle: &str) {
    assert_eq!(reply.0, status, "{}", reply.1);
    let error = reply.1["error"].as_str().unwrap_or_else(|| panic!("no error in {}", reply.1));
    assert!(error.contains(needle), "{error:?} should mention {needle:?}");
}

#[tokio::test]
async fn non_get_routes_refuse_other_origins_and_non_json_bodies() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    let post = |origin: Option<&str>, content_type: Option<&str>| {
        let mut req = Request::post(CONTROL).header(HOST, LOCAL);
        if let Some(o) = origin {
            req = req.header(ORIGIN, o);
        }
        if let Some(t) = content_type {
            req = req.header(CONTENT_TYPE, t);
        }
        req.body(Body::from(r#"{"action":"pause"}"#)).unwrap()
    };
    let json = Some("application/json");

    for origin in ["http://evil.example", "http://localhost:9999", "null"] {
        let reply = call(&app, post(Some(origin), json)).await;
        assert_error(&reply, StatusCode::FORBIDDEN, "cross-origin");
    }
    for content_type in [None, Some("text/plain"), Some("application/x-www-form-urlencoded")] {
        let reply = call(&app, post(Some("http://localhost:8080"), content_type)).await;
        assert_error(&reply, StatusCode::UNSUPPORTED_MEDIA_TYPE, "application/json");
    }
    // Past the guard (same origin, or no Origin at all: a non-browser client),
    // the request reaches the route: nothing is attached.
    for (origin, content_type) in [
        (Some("http://localhost:8080"), json),
        (None, json),
        (None, Some("application/json; charset=utf-8")),
    ] {
        let reply = call(&app, post(origin, content_type)).await;
        assert_error(&reply, StatusCode::SERVICE_UNAVAILABLE, "no sim is attached");
    }
    // GET routes skip the Origin check (they change nothing).
    let reply = call(
        &app,
        Request::get("/api/bridge")
            .header(HOST, LOCAL)
            .header(ORIGIN, "http://evil.example")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(reply.0, StatusCode::OK);
}

#[tokio::test]
async fn every_api_path_and_method_is_guarded() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    let paths = [
        "/api/bridge",
        CONTROL,
        "/api/runs",
        "/api/runs/current",
        "/api/runs/current/log",
        "/api/runs/some-run/replay",
        "/api/runs/some-run/metrics",
        "/api/scenario-check",
        "/api/scenarios/some-name",
        "/api/sweeps/plan",
        "/api/sweeps/saved",
        "/api/sweeps/saved/some-name",
        "/api/nope",
        "/api",
        "/api/",
    ];
    for path in paths {
        for method in [Method::POST, Method::PUT, Method::DELETE, Method::PATCH, Method::OPTIONS] {
            let req = Request::builder()
                .method(method.clone())
                .uri(path)
                .header(HOST, LOCAL)
                .header(ORIGIN, "http://evil.example")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap();
            let reply = call(&app, req).await;
            assert_error(&reply, StatusCode::FORBIDDEN, "cross-origin");
        }
    }
}

#[tokio::test]
async fn only_local_ip_and_allowed_hosts_reach_the_api() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app_allowing(tmp.path(), Bridge::new(config()), &["studio.lan"]);
    let request = |method: Method, uri: &str, host: Option<&str>| {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(h) = host {
            // What a browser on that host sends: a same-origin request.
            req = req.header(HOST, h).header(ORIGIN, format!("http://{h}"));
        }
        req.header(CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"action":"pause"}"#))
            .unwrap()
    };

    // A domain re-pointed at this server (DNS rebinding) is same-origin with
    // it, but not a host it answers for — reads and writes alike.
    for host in [Some("evil.example:8080"), Some("localhost.evil.example:8080"), Some(""), None] {
        for (method, uri) in [(Method::GET, "/api/bridge"), (Method::POST, CONTROL)] {
            let reply = call(&app, request(method, uri, host)).await;
            assert_error(&reply, StatusCode::FORBIDDEN, "is not allowed");
        }
    }
    for host in
        ["localhost:8080", "LocalHost", "127.0.0.1:8080", "10.1.2.3", "[::1]:8080", "Studio.LAN:80"]
    {
        let reply = call(&app, request(Method::GET, "/api/bridge", Some(host))).await;
        assert_eq!(reply.0, StatusCode::OK, "{host}: {}", reply.1);
        let reply = call(&app, request(Method::POST, CONTROL, Some(host))).await;
        assert_error(&reply, StatusCode::SERVICE_UNAVAILABLE, "no sim is attached");
    }
    // Static files hold nothing private and stay open.
    let page = app
        .clone()
        .oneshot(Request::get("/").header(HOST, "evil.example").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(page.status(), StatusCode::OK);
}

#[tokio::test]
async fn control_bodies_are_validated_before_anything_is_sent() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    for (body, needle) in [
        ("not json", "bad control body"),
        ("{}", "action"),
        (r#"{"action":"jump"}"#, "jump"),
        (r#"{"action":"pause","extra":1}"#, "extra"),
        (r#"{"action":"resume","extra":1}"#, "extra"),
        (r#"{"action":"step"}"#, "ticks"),
        (r#"{"action":"step","ticks":0}"#, "ticks >= 1"),
        (r#"{"action":"step","ticks":-1}"#, "bad control body"),
        (r#"{"action":"step","ticks":2.5}"#, "bad control body"),
        (r#"{"action":"step","ticks":3,"extra":1}"#, "extra"),
        (r#"{"action":"speed"}"#, "value"),
        (r#"{"action":"speed","value":"fast"}"#, "bad control body"),
        (r#"{"action":"speed","value":-1}"#, "must be 0 (unpaced) or >= 0.001"),
        (r#"{"action":"speed","value":0.0009}"#, "must be 0 (unpaced) or >= 0.001"),
        (r#"{"action":"speed","value":1e999}"#, "bad control body"),
    ] {
        assert_error(&control(&app, body).await, StatusCode::BAD_REQUEST, needle);
    }
    // The boundaries are valid: they get as far as finding no sim.
    let slowest = format!(r#"{{"action":"speed","value":{MIN_SPEED}}}"#);
    for body in [
        r#"{"action":"pause"}"#,
        r#"{"action":"resume"}"#,
        r#"{"action":"step","ticks":1}"#,
        r#"{"action":"speed","value":0}"#,
        slowest.as_str(),
    ] {
        assert_eq!(control(&app, body).await.0, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    }
}

#[tokio::test]
async fn unknown_routes_methods_and_oversized_bodies_are_json_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    for path in ["/api/nope", "/api", "/api/"] {
        assert_error(&get(&app, path).await, StatusCode::NOT_FOUND, "no such API route");
    }
    assert_error(&get(&app, CONTROL).await, StatusCode::METHOD_NOT_ALLOWED, "method not allowed");
    let huge = format!(r#"{{"action":"pause","pad":"{}"}}"#, "x".repeat(3 << 20));
    let reply = control(&app, &huge).await;
    assert_eq!(reply.0, StatusCode::PAYLOAD_TOO_LARGE, "{}", reply.1);
    assert!(reply.1["error"].is_string(), "{}", reply.1);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_runs_while_no_sim_answers() {
    let at = endpoint(0); // nothing listens here
    let bridge = Bridge::new(config());
    bridge.attach(&at);
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), bridge);
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let page = app.clone().oneshot(Request::get("/").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(page.status(), StatusCode::OK);
    let (status, info) = get(&app, "/api/bridge").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(info, json!({"endpoint": at, "run": 1, "connected": false, "status": null}));
    let reply = control(&app, r#"{"action":"pause"}"#).await;
    assert_error(&reply, StatusCode::SERVICE_UNAVAILABLE, &format!("no sim is answering on {at}"));
}

/// A bridge attached and connected to a hub on `at`, and a server for it.
async fn attached(at: &str) -> (Bridge, Router, tempfile::TempDir) {
    let bridge = Bridge::new(config());
    bridge.attach(at);
    let probe = bridge.clone();
    tokio::task::spawn_blocking(move || {
        wait_until("the bridge to connect", || probe.info().connected)
    })
    .await
    .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), bridge.clone());
    (bridge, app, tmp)
}

#[tokio::test(flavor = "multi_thread")]
async fn control_requests_reach_the_sim_and_its_status_is_shown() {
    let at = endpoint(1);
    let hub = Session::listen(&at).unwrap();
    let sim = ControlServer::serve(&hub, MIN_SPEED).unwrap();
    let (_bridge, app, _tmp) = attached(&at).await;

    for (body, want) in [
        (r#"{"action":"pause"}"#, ControlRequest::Pause),
        (r#"{"action":"step","ticks":3}"#, ControlRequest::Step(3)),
        (r#"{"action":"speed","value":2.5}"#, ControlRequest::Speed(2.5)),
        (r#"{"action":"resume"}"#, ControlRequest::Resume),
    ] {
        let reply = control(&app, body).await;
        assert_eq!(reply, (StatusCode::OK, json!({"ok": true})), "{body}");
        assert_eq!(sim.drain(), [want], "{body}");
    }

    {
        let board = sim.board();
        let mut status = board.lock().unwrap();
        status.state = "paused".to_owned();
        status.tick = 42;
    }
    let (status, info) = get(&app, "/api/bridge").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(info["endpoint"], json!(at));
    assert_eq!(info["run"], json!(1));
    assert_eq!(info["connected"], json!(true));
    assert_eq!(info["status"]["state"], json!("paused"));
    assert_eq!(info["status"]["tick"], json!(42));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_the_sim_refuses_is_a_conflict_with_its_reason() {
    let at = endpoint(2);
    let hub = Session::listen(&at).unwrap();
    // This sim's slowest speed is 10x: 2x passes our checks but not the sim's.
    let sim = ControlServer::serve(&hub, 10.0).unwrap();
    let (_bridge, app, _tmp) = attached(&at).await;

    let reply = control(&app, r#"{"action":"speed","value":2}"#).await;
    assert_error(&reply, StatusCode::CONFLICT, ">= 10");
    assert!(sim.drain().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_detached_bridge_has_no_sim_to_control() {
    let at = endpoint(3);
    let hub = Session::listen(&at).unwrap();
    let sim = ControlServer::serve(&hub, MIN_SPEED).unwrap();
    let (bridge, app, _tmp) = attached(&at).await;
    bridge.detach();

    assert_error(
        &control(&app, r#"{"action":"pause"}"#).await,
        StatusCode::SERVICE_UNAVAILABLE,
        "no sim is attached",
    );
    assert_eq!(
        get(&app, "/api/bridge").await.1,
        json!({"endpoint": null, "run": 2, "connected": false, "status": null})
    );
    assert!(sim.drain().is_empty());
}

#[test]
fn safe_names_are_short_plain_identifiers() {
    for good in ["a", "Z", "0", "example", "my_run-2", "A-", &"x".repeat(64)] {
        assert_eq!(safe_name(good), Ok(good), "{good:?}");
    }
    for bad in [
        "",
        "_hidden",
        "-flag",
        ".",
        "..",
        "a.b",
        "a/b",
        "a\\b",
        "a b",
        "naïve",
        "a\0",
        &"x".repeat(65),
    ] {
        let err = safe_name(bad).unwrap_err();
        assert!(err.contains(&format!("{bad:?}")), "{err:?} should name {bad:?}");
    }
}
