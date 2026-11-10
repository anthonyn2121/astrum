//! The studio's read-only catalog routes: environments, models, controllers
//! and the scenarios saved under the config directory.

mod common;

use std::path::Path;

use astrum_sim::scenario::Scenario;
use astrum_web::bridge::Bridge;
use astrum_web::catalogs::{Catalogs, SaveError};
use axum::body::{to_bytes, Body};
use axum::http::header::HOST;
use axum::http::{Request, StatusCode};
use axum::Router;
use common::config;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::get(uri).header(HOST, "localhost:8080").body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

/// A config directory holding the repository's controller catalog, a valid
/// two-ship scenario `pair`, an invalid scenario `broken`, and files that are
/// not `scenario.<name>.toml` scenarios.
fn config_dir(root: &Path) -> std::path::PathBuf {
    let dir = root.join("config");
    std::fs::create_dir_all(dir.join("demos")).unwrap();
    std::fs::copy(common::repo_config().join("controllers.toml"), dir.join("controllers.toml"))
        .unwrap();
    std::fs::write(
        dir.join("scenario.pair.toml"),
        "# A quad and a jet. Second sentence.\n\
         [world]\norigin_lat_deg = 37.6\norigin_lon_deg = -122.4\norigin_alt_hae_m = 0.0\n\
         [sim]\nrate_hz = 120.0\n\
         [[aircraft]]\nid = 7\nmodel = \"F450\"\nlat_deg = 37.6\nlon_deg = -122.4\n\
         alt_hae_m = 50.0\nheading_deg = 0.0\ntas_mps = 0.0\nteam = 1\n\
         [[aircraft]]\nid = 2\nmodel = \"f16\"\nlat_deg = 37.6\nlon_deg = -122.4\n\
         alt_hae_m = 3000.0\nheading_deg = 90.0\ntas_mps = 200.0\n",
    )
    .unwrap();
    std::fs::write(dir.join("scenario.broken.toml"), "# Not flyable.\n[[aircraft]]\nid = 1\n")
        .unwrap();
    // Not scenarios of the studio: wrong pattern, an unsafe name, a run file
    // elsewhere, a directory named like a scenario.
    std::fs::write(dir.join("sweep.pair.toml"), "base = \"x\"\n").unwrap();
    std::fs::write(dir.join("scenario.a.b.toml"), "").unwrap();
    std::fs::write(dir.join("demos/scenario.nested.toml"), "").unwrap();
    std::fs::create_dir_all(dir.join("scenario.dir.toml")).unwrap();
    dir
}

#[tokio::test]
async fn scenarios_lists_only_named_scenario_files_with_their_fleet_or_error() {
    let tmp = tempfile::tempdir().unwrap();
    let catalogs = Catalogs::load(config_dir(tmp.path())).unwrap();
    let app = common::app_with_catalogs(tmp.path(), Bridge::new(config()), catalogs);

    let (status, list) = get(&app, "/api/scenarios").await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<&str> =
        list.as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["broken", "pair"]);

    let broken = &list[0];
    assert!(broken["error"].as_str().unwrap().contains("parsing scenario"), "{broken}");
    assert_eq!(broken["aircraft"], json!([]));
    assert_eq!(broken["summary"], "Not flyable.");

    let pair = &list[1];
    assert_eq!(pair["error"], Value::Null);
    assert_eq!(pair["environment"], "empty");
    assert_eq!(pair["summary"], "A quad and a jet.");
    assert!(pair["path"].as_str().unwrap().ends_with("config/scenario.pair.toml"));
    assert_eq!(
        pair["aircraft"],
        json!([
            {"id": 2, "model": "f16", "kind": "fixed-wing", "team": 0},
            {"id": 7, "model": "F450", "kind": "multirotor", "team": 1},
        ])
    );
}

#[tokio::test]
async fn a_scenario_comes_back_whole_or_with_the_reason_it_cannot() {
    let tmp = tempfile::tempdir().unwrap();
    let catalogs = Catalogs::load(config_dir(tmp.path())).unwrap();
    let app = common::app_with_catalogs(tmp.path(), Bridge::new(config()), catalogs);

    let (status, doc) = get(&app, "/api/scenarios/pair").await;
    assert_eq!(status, StatusCode::OK, "{doc}");
    assert_eq!(doc["name"], "pair");
    assert_eq!(doc["environment"], "empty");
    assert_eq!(doc["scenario"]["world"]["origin_lat_deg"], 37.6);
    assert_eq!(doc["scenario"]["sim"]["rate_hz"], 120.0);
    // The file's own order, every field the scenario resolves (defaults too).
    let ids: Vec<u64> = doc["scenario"]["aircraft"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_u64().unwrap())
        .collect();
    assert_eq!(ids, [7, 2]);
    assert_eq!(doc["scenario"]["aircraft"][1]["health"], 100.0);

    for (uri, status, needle) in [
        ("/api/scenarios/missing", StatusCode::NOT_FOUND, "no scenario named \"missing\""),
        ("/api/scenarios/broken", StatusCode::UNPROCESSABLE_ENTITY, "scenario.broken.toml"),
        ("/api/scenarios/a.b", StatusCode::BAD_REQUEST, "a.b"),
        ("/api/scenarios/.hidden", StatusCode::BAD_REQUEST, ".hidden"),
    ] {
        let (got, body) = get(&app, uri).await;
        assert_eq!(got, status, "{uri}: {body}");
        assert!(body["error"].as_str().unwrap().contains(needle), "{uri}: {body}");
    }
}

#[tokio::test]
async fn the_bundled_scenarios_are_all_valid_studio_scenarios() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));
    let (status, list) = get(&app, "/api/scenarios").await;
    assert_eq!(status, StatusCode::OK);
    let list = list.as_array().unwrap();
    for name in ["demo", "example", "formation", "quad", "smoke"] {
        let entry = list.iter().find(|s| s["name"] == name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(entry["error"], Value::Null, "{name}");
        assert!(!entry["aircraft"].as_array().unwrap().is_empty(), "{name}");
    }
}

#[tokio::test]
async fn environments_models_and_controllers_describe_what_can_be_flown() {
    let tmp = tempfile::tempdir().unwrap();
    let app = common::app(tmp.path(), Bridge::new(config()));

    let (status, envs) = get(&app, "/api/environments").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(envs.as_array().unwrap().len(), 1);
    assert_eq!(envs[0]["id"], "empty");
    assert_eq!(envs[0]["sim"]["rate_hz"], 120.0);
    assert!(!envs[0]["locations"].as_array().unwrap().is_empty());

    let (status, models) = get(&app, "/api/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        models,
        json!([
            {"model": "c172x", "kind": "fixed-wing", "autopilot_profile": "trainer"},
            {"model": "f16", "kind": "fixed-wing", "autopilot_profile": "fighter"},
            {"model": "F450", "kind": "multirotor", "autopilot_profile": null},
        ])
    );

    let (status, controllers) = get(&app, "/api/controllers").await;
    assert_eq!(status, StatusCode::OK);
    let entries = controllers["entries"].as_array().unwrap();
    let hold = entries.iter().find(|e| e["id"] == "cpp-hold").expect("cpp-hold");
    assert_eq!(hold["kinds"], json!(["fixed-wing"]));
    assert!(hold["params"].as_array().unwrap().iter().any(|p| p["name"] == "heading"));
}

#[test]
fn a_config_dir_without_a_valid_controller_catalog_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let err = Catalogs::load(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("controllers.toml"), "{err}");
    std::fs::write(tmp.path().join("controllers.toml"), "[[controller]]\nid = \"x\"\n").unwrap();
    assert!(Catalogs::load(tmp.path()).is_err());
}

const ONE_SHIP: &str = "[world]\norigin_lat_deg = 37.0\norigin_lon_deg = -122.0\n\
                        origin_alt_hae_m = 0.0\n[sim]\nrate_hz = 120.0\n\
                        [[aircraft]]\nid = 1\nmodel = \"F450\"\nlat_deg = 37.0\n\
                        lon_deg = -122.0\nalt_hae_m = 50.0\nheading_deg = 0.0\ntas_mps = 0.0\n";

#[test]
fn save_scenario_writes_the_file_it_names_and_refuses_what_it_cannot_write() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let catalogs = Catalogs::load(&dir).unwrap();
    let scenario = Scenario::from_toml_str(ONE_SHIP).unwrap();
    let file = dir.join("scenario.solo.toml");

    let (entry, created) = catalogs.save_scenario("solo", &scenario, "One ship.", false).unwrap();
    assert!(created);
    assert_eq!(entry.summary, "One ship.");
    let listed = catalogs.scenarios().unwrap().into_iter().find(|e| e.name == "solo").unwrap();
    assert_eq!(entry, listed);
    let text = std::fs::read_to_string(&file).unwrap();
    assert_eq!(text, format!("# One ship.\n{}", scenario.to_toml()));

    // An empty summary writes no comment line, and replacing the file reports `false`.
    let (entry, created) = catalogs.save_scenario("solo", &scenario, "", true).unwrap();
    assert!(!created);
    assert_eq!(entry.summary, "");
    assert_eq!(std::fs::read_to_string(&file).unwrap(), scenario.to_toml());

    let exists = catalogs.save_scenario("solo", &scenario, "x", false).unwrap_err();
    assert_eq!(exists.to_string(), "scenario \"solo\" exists; send overwrite: true to replace it");

    for name in ["", "a.b", "../x"] {
        let e = catalogs.save_scenario(name, &scenario, "", true).unwrap_err();
        assert!(matches!(e, SaveError::BadName(_)), "{name:?}: {e}");
    }
    let long = "x".repeat(201);
    for summary in ["one\ntwo", "one\rtwo", long.as_str()] {
        let e = catalogs.save_scenario("solo", &scenario, summary, true).unwrap_err();
        assert!(matches!(e, SaveError::BadSummary(_)), "{e}");
    }

    // A directory where the file would go: the rename fails and the temporary file goes too.
    let e = catalogs.save_scenario("dir", &scenario, "", true).unwrap_err();
    assert!(matches!(e, SaveError::Io(_)), "{e}");
    assert!(!dir.join(".scenario.dir.toml.tmp").exists());
    assert!(!dir.join(".scenario.solo.toml.tmp").exists());
}
