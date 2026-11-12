//! The studio's scenario writes and checks through the router: `PUT
//! /api/scenarios/<name>` saves a scenario (201 created, 200 replaced, 409
//! without `overwrite`, 400 for a bad name or summary, 422 for an invalid
//! scenario); `POST /api/scenario-check` reports its errors and each trim.

mod common;

use std::path::{Path, PathBuf};

use astrum_sim::scenario::Scenario;
use astrum_web::bridge::Bridge;
use astrum_web::catalogs::Catalogs;
use axum::body::{to_bytes, Body};
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use common::config;
use serde_json::{json, Value};
use tower::ServiceExt;

const LOCAL: &str = "localhost:8080";

/// A config directory holding the repository's controller catalog and nothing else.
fn config_dir(root: &Path) -> PathBuf {
    let dir = root.join("config");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(common::repo_config().join("controllers.toml"), dir.join("controllers.toml"))
        .unwrap();
    dir
}

/// The names of the files in `dir`, sorted.
fn files(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// Send `method uri` with a JSON `body` (sent as is); the status and JSON reply.
async fn send(app: &Router, method: Method, uri: &str, body: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(HOST, LOCAL)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// A scenario document: a c172x and an f16 at level cruise, and a quad (no trim).
fn pair() -> Value {
    json!({
        "world": {"origin_lat_deg": 37.6, "origin_lon_deg": -122.4, "origin_alt_hae_m": 0.0},
        "sim": {"rate_hz": 120.0},
        "aircraft": [
            {"id": 7, "model": "F450", "lat_deg": 37.6, "lon_deg": -122.4,
             "alt_hae_m": 50.0, "heading_deg": 0.0, "tas_mps": 0.0, "team": 1},
            {"id": 2, "model": "f16", "lat_deg": 37.6, "lon_deg": -122.4,
             "alt_hae_m": 3000.0, "heading_deg": 90.0, "tas_mps": 200.0},
            {"id": 1, "model": "c172x", "lat_deg": 37.6, "lon_deg": -122.4,
             "alt_hae_m": 1500.0, "heading_deg": 0.0, "tas_mps": 55.0},
        ]
    })
}

/// `pair()` with `edit` applied to its aircraft list.
fn with_aircraft(edit: impl FnOnce(&mut Vec<Value>)) -> Value {
    let mut doc = pair();
    edit(doc["aircraft"].as_array_mut().unwrap());
    doc
}

/// A `PUT` body of `scenario` with the optional summary and overwrite flag.
fn put_body(scenario: &Value, summary: Option<&str>, overwrite: Option<bool>) -> String {
    let mut body = json!({ "scenario": scenario });
    if let Some(summary) = summary {
        body["summary"] = json!(summary);
    }
    if let Some(overwrite) = overwrite {
        body["overwrite"] = json!(overwrite);
    }
    body.to_string()
}

#[tokio::test]
async fn a_saved_scenario_is_written_with_its_summary_and_reads_back_whole() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let catalogs = Catalogs::load(&dir).unwrap();
    let app = common::app_with_catalogs(tmp.path(), Bridge::new(config()), catalogs.clone());
    let doc = pair();

    let body = put_body(&doc, Some("A pair. Second sentence."), None);
    let (status, entry) = send(&app, Method::PUT, "/api/scenarios/pair", &body).await;
    assert_eq!(status, StatusCode::CREATED, "{entry}");
    assert_eq!(entry["name"], "pair");
    assert_eq!(entry["summary"], "A pair.");
    assert_eq!(entry["error"], Value::Null);

    let text = std::fs::read_to_string(dir.join("scenario.pair.toml")).unwrap();
    assert!(text.starts_with("# A pair. Second sentence.\n"), "{text}");
    let expected: Scenario = serde_json::from_value(doc).unwrap();
    assert_eq!(Scenario::from_toml_str(&text).unwrap(), expected);
    assert_eq!(catalogs.scenario("pair").unwrap().scenario, expected);

    // The listing shows the same entry the save answered with, and no temporary file is left.
    let (_, list) = send(&app, Method::GET, "/api/scenarios", "").await;
    let listed = list.as_array().unwrap().iter().find(|e| e["name"] == "pair").unwrap();
    assert_eq!(listed, &entry);
    assert_eq!(files(&dir), ["controllers.toml", "scenario.pair.toml"]);
}

#[tokio::test]
async fn a_second_save_needs_overwrite_and_then_replaces_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let (status, _) =
        send(&app, Method::PUT, "/api/scenarios/pair", &put_body(&pair(), Some("First."), None))
            .await;
    assert_eq!(status, StatusCode::CREATED);
    let first = std::fs::read_to_string(dir.join("scenario.pair.toml")).unwrap();

    for overwrite in [None, Some(false)] {
        let body = put_body(&with_aircraft(|a| a.truncate(1)), Some("Second."), overwrite);
        let (status, reply) = send(&app, Method::PUT, "/api/scenarios/pair", &body).await;
        assert_eq!(status, StatusCode::CONFLICT, "{reply}");
        assert!(reply["error"].as_str().unwrap().contains("overwrite: true"), "{reply}");
        assert_eq!(std::fs::read_to_string(dir.join("scenario.pair.toml")).unwrap(), first);
    }

    let body = put_body(&with_aircraft(|a| a.truncate(1)), Some("Second."), Some(true));
    let (status, entry) = send(&app, Method::PUT, "/api/scenarios/pair", &body).await;
    assert_eq!(status, StatusCode::OK, "{entry}");
    assert_eq!(entry["summary"], "Second.");
    assert_eq!(entry["aircraft"].as_array().unwrap().len(), 1);
    let text = std::fs::read_to_string(dir.join("scenario.pair.toml")).unwrap();
    assert!(text.starts_with("# Second.\n"), "{text}");
}

#[tokio::test]
async fn bad_names_and_summaries_are_refused_and_nothing_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let doc = pair();
    let long_name = "a".repeat(65);
    for name in ["..%2Fx", "a.b", long_name.as_str()] {
        let uri = format!("/api/scenarios/{name}");
        let (status, reply) = send(&app, Method::PUT, &uri, &put_body(&doc, None, None)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {reply}");
        assert!(reply["error"].as_str().unwrap().contains("must be 1-64"), "{reply}");
    }
    let long_summary = "x".repeat(201);
    for summary in ["two\nlines", "two\rlines", long_summary.as_str()] {
        let body = put_body(&doc, Some(summary), None);
        let (status, reply) = send(&app, Method::PUT, "/api/scenarios/pair", &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    }
    assert_eq!(files(&dir), ["controllers.toml"]);

    // Exactly 200 characters is the longest summary.
    let body = put_body(&doc, Some(&"x".repeat(200)), None);
    let (status, _) = send(&app, Method::PUT, "/api/scenarios/edge", &body).await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn an_invalid_scenario_or_unsafe_model_is_unprocessable() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let duplicate = with_aircraft(|a| a[2]["id"] = json!(7));
    let (status, reply) =
        send(&app, Method::PUT, "/api/scenarios/bad", &put_body(&duplicate, None, None)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("duplicate aircraft id 7"), "{reply}");

    let escaping = with_aircraft(|a| {
        a[1]["model"] = json!("../x");
        a[1]["kind"] = json!("fixed-wing");
        a[1]["autopilot_profile"] = json!("fighter");
    });
    let (status, reply) =
        send(&app, Method::PUT, "/api/scenarios/bad", &put_body(&escaping, None, None)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert_eq!(
        reply["error"],
        "scenario: aircraft 2: model: name \"../x\" must be 1-64 characters of A-Z, a-z, 0-9, \
         '_' or '-', starting with a letter or digit"
    );
    assert_eq!(files(&dir), ["controllers.toml"]);
}

#[tokio::test]
async fn a_body_with_unknown_keys_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let body = json!({"scenario": pair(), "extra": true}).to_string();
    let (status, reply) = send(&app, Method::PUT, "/api/scenarios/pair", &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert_eq!(files(&dir), ["controllers.toml"]);
}

#[tokio::test]
async fn check_reports_each_trim_and_writes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());

    let (status, reply) =
        send(&app, Method::POST, "/api/scenario-check", &json!({"scenario": pair()}).to_string())
            .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["errors"], json!([]));
    // The quad does not trim, so it has no entry; the trimming ones follow id order.
    assert_eq!(
        reply["trim"],
        json!([
            {"aircraft": 1, "model": "c172x", "ok": true, "error": null},
            {"aircraft": 2, "model": "f16", "ok": true, "error": null},
        ])
    );

    let banked = with_aircraft(|a| a[1]["bank_deg"] = json!(60.0));
    let (status, reply) =
        send(&app, Method::POST, "/api/scenario-check", &json!({"scenario": banked}).to_string())
            .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["errors"], json!([]));
    let trim = reply["trim"].as_array().unwrap();
    assert_eq!(trim[0], json!({"aircraft": 1, "model": "c172x", "ok": true, "error": null}));
    assert_eq!(trim[1]["aircraft"], 2);
    assert_eq!(trim[1]["ok"], false, "{reply}");
    assert!(!trim[1]["error"].as_str().unwrap().is_empty(), "{reply}");

    assert_eq!(files(&dir), ["controllers.toml"]);
}

#[tokio::test]
async fn check_reports_an_invalid_scenario_as_one_error_and_no_trims() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let duplicate = with_aircraft(|a| a[2]["id"] = json!(7));
    let unsafe_model = with_aircraft(|a| a[1]["model"] = json!("../x"));
    for (scenario, needle) in [(duplicate, "duplicate aircraft id 7"), (unsafe_model, "model")] {
        let body = json!({"scenario": scenario}).to_string();
        let (status, reply) = send(&app, Method::POST, "/api/scenario-check", &body).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(reply["trim"], json!([]));
        let errors = reply["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 1, "{reply}");
        assert!(errors[0].as_str().unwrap().contains(needle), "{reply}");
    }
}

#[tokio::test]
async fn check_takes_at_most_512_aircraft() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let fleet = |n: u32| {
        let aircraft: Vec<Value> = (1..=n)
            .map(|id| {
                json!({"id": id, "model": "F450", "lat_deg": 37.6, "lon_deg": -122.4,
                       "alt_hae_m": 50.0, "heading_deg": 0.0, "tas_mps": 0.0})
            })
            .collect();
        let mut doc = pair();
        doc["aircraft"] = json!(aircraft);
        json!({"scenario": doc}).to_string()
    };

    let (status, reply) = send(&app, Method::POST, "/api/scenario-check", &fleet(512)).await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply, json!({"errors": [], "trim": []}));

    let (status, reply) = send(&app, Method::POST, "/api/scenario-check", &fleet(513)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("513 aircraft"), "{reply}");
}
