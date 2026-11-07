//! The JSON API: the guard on non-GET routes, control-body validation, the
//! relay to a real sim's control plane (and its refusals), and the bridge
//! status, including while no sim is reachable.

mod common;

use std::time::Duration;

use astrum_net::{ControlRequest, ControlServer, Session};
use astrum_web::api::{safe_name, MIN_SPEED};
use astrum_web::bridge::Bridge;
use axum::body::{to_bytes, Body};
use axum::http::header::{CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{Request, StatusCode};
use axum::Router;
use common::{config, endpoint, wait_until};
use serde_json::{json, Value};
use tower::ServiceExt;

const CONTROL: &str = "/api/bridge/control";

/// Send `req`; the status and the JSON body.
async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    assert_eq!(
        resp.headers().get(CONTENT_TYPE).map(|v| v.to_str().unwrap()),
        Some("application/json"),
        "API responses are JSON"
    );
    let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    call(app, Request::get(uri).body(Body::empty()).unwrap()).await
}

/// An originless JSON `POST` of `body` to the control route.
async fn control(app: &Router, body: &str) -> (StatusCode, Value) {
    let req = Request::post(CONTROL)
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
        let mut req = Request::post(CONTROL).header(HOST, "localhost:8080");
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
    // GET routes are not guarded.
    let reply = call(
        &app,
        Request::get("/api/bridge")
            .header(ORIGIN, "http://evil.example")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(reply.0, StatusCode::OK);
}

#[tokio::test]
async fn control_bodies_are_validated_before_anything_is_sent() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    for (body, needle) in [
        ("not json", "bad control body"),
        ("{}", "action"),
        (r#"{"action":"jump"}"#, "jump"),
        (r#"{"action":"step"}"#, "ticks"),
        (r#"{"action":"step","ticks":0}"#, "ticks must be >= 1"),
        (r#"{"action":"step","ticks":-1}"#, "bad control body"),
        (r#"{"action":"step","ticks":2.5}"#, "bad control body"),
        (r#"{"action":"speed"}"#, "value"),
        (r#"{"action":"speed","value":"fast"}"#, "bad control body"),
        (r#"{"action":"speed","value":-1}"#, "value must be 0 (unpaced) or >= 0.001"),
        (r#"{"action":"speed","value":0.0009}"#, "value must be 0 (unpaced) or >= 0.001"),
        (r#"{"action":"speed","value":1e999}"#, "bad control body"),
        (r#"{"action":"step","ticks":3,"extra":1}"#, "extra"),
    ] {
        assert_error(&control(&app, body).await, StatusCode::BAD_REQUEST, needle);
    }
    // The boundaries are valid: they get as far as finding no sim.
    for body in [
        r#"{"action":"pause"}"#,
        r#"{"action":"resume"}"#,
        r#"{"action":"step","ticks":1}"#,
        r#"{"action":"speed","value":0}"#,
        &format!(r#"{{"action":"speed","value":{MIN_SPEED}}}"#),
    ] {
        assert_eq!(control(&app, body).await.0, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    }
}

#[tokio::test]
async fn unknown_routes_and_methods_are_json_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    assert_error(&get(&app, "/api/nope").await, StatusCode::NOT_FOUND, "no such API route");
    assert_error(&get(&app, CONTROL).await, StatusCode::METHOD_NOT_ALLOWED, "method not allowed");
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
