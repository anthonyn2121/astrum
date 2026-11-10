//! The studio's sweeps through the router: `POST /api/sweeps/plan` says what a
//! sweep would fly (its runs, their values and the moving aircraft's starting
//! positions) without flying it, and `PUT`/`GET /api/sweeps/saved/<name>`
//! write, read and list `config/sweep.<name>.toml`, which `astrum sweep` runs.

mod common;

use std::path::{Path, PathBuf};

use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::{Sweep, SweepFile};
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
const PLAN: &str = "/api/sweeps/plan";

/// Two F-16s level at 3000 m, heading east, 500 m apart (the second is 500 m north).
fn pair() -> Value {
    json!({
        "world": {"origin_lat_deg": 37.6, "origin_lon_deg": -122.4, "origin_alt_hae_m": 0.0},
        "sim": {"rate_hz": 120.0},
        "aircraft": [
            {"id": 1, "model": "f16", "lat_deg": 37.6, "lon_deg": -122.4,
             "alt_hae_m": 3000.0, "heading_deg": 90.0, "tas_mps": 200.0, "team": 1},
            {"id": 2, "model": "f16", "lat_deg": 37.6044966, "lon_deg": -122.4,
             "alt_hae_m": 3000.0, "heading_deg": 90.0, "tas_mps": 200.0, "team": 1}
        ]
    })
}

/// A config directory: the repository's controller catalog and the saved scenario `pair`.
fn config_dir(root: &Path) -> PathBuf {
    let dir = root.join("config");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(common::repo_config().join("controllers.toml"), dir.join("controllers.toml"))
        .unwrap();
    let scenario: Scenario = serde_json::from_value(pair()).unwrap();
    std::fs::write(dir.join("scenario.pair.toml"), scenario.to_toml()).unwrap();
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
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// A request for `pair` with `max_time_s` 60 and the fields `extra` sets.
fn request(extra: Value) -> Value {
    let mut body = json!({"scenario": "pair", "max_time_s": 60.0});
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

/// The grid of the examples: three forward offsets of aircraft 2, four drawn
/// `right_m` samples each (12 runs).
fn grid() -> Value {
    request(json!({
        "vary": {"aircraft.2.offset.forward_m": [-200.0, 0.0, 200.0]},
        "perturb": [{"aircraft": [2], "right_m": {"uniform": 100.0}}],
        "samples": 4,
        "sample_seed": 7
    }))
}

/// A `PUT /api/sweeps/saved/<name>` body.
fn save_body(req: &Value, summary: &str, overwrite: Option<bool>) -> String {
    let mut body = json!({"request": req, "summary": summary});
    if let Some(overwrite) = overwrite {
        body["overwrite"] = json!(overwrite);
    }
    body.to_string()
}

/// The sweep file's first line and everything after it.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap()
}

#[tokio::test]
async fn a_plan_lists_each_run_with_its_values_and_starting_positions() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());

    let (status, plan) = send(&app, Method::POST, PLAN, &grid().to_string()).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert_eq!(plan["runs"], 12);
    assert_eq!(plan["in_process"], true);
    assert_eq!(plan["trim"], Value::Null);
    assert_eq!(plan["keys"], json!(["aircraft.2.offset.forward_m", "aircraft.2.offset.right_m"]));
    assert_eq!(plan["moving"], json!([2]));

    let rows = plan["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 12);
    let right = |row: &Value| row["values"]["aircraft.2.offset.right_m"].as_f64().unwrap();
    for (i, row) in rows.iter().enumerate() {
        // Runs are numbered from 1; the grid point is the outer loop, the sample the inner.
        assert_eq!(row["run"], json!(i + 1));
        assert_eq!(row["sample"], json!(i % 4 + 1));
        assert_eq!(row["seed"], json!(0));
        let forward = [-200.0, 0.0, 200.0][i / 4];
        assert_eq!(row["values"]["aircraft.2.offset.forward_m"], json!(forward));
        assert!(right(row).abs() <= 100.0, "{row}");
    }
    // Sample k draws the same offset at every grid point, and different samples differ.
    for s in 0..4 {
        for point in 1..3 {
            assert_eq!(right(&rows[point * 4 + s]), right(&rows[s]));
        }
    }
    assert!((1..4).any(|s| right(&rows[s]) != right(&rows[0])));

    // Only aircraft 2 moves, so only it is listed, with the base's other fields.
    let initial = plan["initial"].as_array().unwrap();
    assert_eq!(initial.len(), 12);
    for (i, run) in initial.iter().enumerate() {
        assert_eq!(run["run"], json!(i + 1));
        let aircraft = run["aircraft"].as_array().unwrap();
        assert_eq!(aircraft.len(), 1, "{run}");
        let start = &aircraft[0];
        assert_eq!(start["id"], json!(2));
        assert_eq!(start["model"], json!("f16"));
        // A forward offset follows the level frame, which drops ~3 mm over 200 m of Earth.
        let alt = start["alt_hae_m"].as_f64().unwrap();
        assert!((alt - 3000.0).abs() < 0.01, "{alt}");
        assert_eq!(start["heading_deg"], json!(90.0));
        assert_eq!(start["tas_mps"], json!(200.0));
        assert_eq!(start["bank_deg"], json!(0.0));
    }
    // The same sample at another grid point starts elsewhere, and so does another sample.
    assert_ne!(initial[0]["aircraft"][0]["lon_deg"], initial[4]["aircraft"][0]["lon_deg"]);
    assert_ne!(initial[0]["aircraft"][0]["lat_deg"], initial[1]["aircraft"][0]["lat_deg"]);
}

#[tokio::test]
async fn an_assigned_sweep_is_not_in_process() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = request(json!({"assignments": [{"aircraft": 1, "controller": "py-hold"}]}));

    let (status, plan) = send(&app, Method::POST, PLAN, &req.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert_eq!(plan["in_process"], false);
    assert_eq!(plan["runs"], 1);
    assert_eq!(plan["moving"], json!([]));
}

#[tokio::test]
async fn a_plan_that_cannot_be_asked_for_is_refused_with_its_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());

    let refused = |body: Value| {
        let app = app.clone();
        async move { send(&app, Method::POST, PLAN, &body.to_string()).await }
    };
    let (status, reply) = refused(request(json!({"scenario": "nope"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert!(reply["error"].as_str().unwrap().starts_with("scenario:"), "{reply}");

    for key in ["sim.speed", "aircraft.9.offset.forward_m", "aircraft.2.offset.roll_deg"] {
        let (status, reply) = refused(request(json!({"vary": {key: [0.0]}}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
        assert!(reply["error"].as_str().unwrap().contains(key), "{reply}");
    }

    let (status, reply) =
        refused(request(json!({"vary": {"aircraft.2.offset.forward_m": vec![0.0; 2001]}}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert_eq!(reply["error"], json!("plans 2001 runs; at most 2000"));

    for max_time in [json!(0.0), json!(-5.0)] {
        let (status, reply) = refused(request(json!({"max_time_s": max_time}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    }
    // JSON has no NaN: a NaN body is not a request at all.
    let nan = r#"{"scenario": "pair", "max_time_s": NaN}"#;
    let (status, reply) = send(&app, Method::POST, PLAN, nan).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");

    // The planner's own refusals are unprocessable.
    let (status, reply) = refused(request(json!({
        "samples": 1,
        "perturb": [{"aircraft": [9], "forward_m": {"uniform": 10.0}}]
    })))
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert!(reply["error"].as_str().unwrap().contains("aircraft 9"), "{reply}");

    let (status, reply) = refused(request(json!({
        "vary": {"aircraft.2.offset.bank_deg": [60.0]},
        "check_trim": true
    })))
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    let error = reply["error"].as_str().unwrap();
    assert!(error.contains("run 1") && error.contains("aircraft.2.offset.bank_deg"), "{reply}");
}

#[tokio::test]
async fn the_grid_limit_admits_2000_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = request(json!({"vary": {"aircraft.2.offset.forward_m": vec![0.0; 2000]}}));
    let (status, plan) = send(&app, Method::POST, PLAN, &req.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert_eq!(plan["runs"], 2000);
}

#[tokio::test]
async fn a_checked_trim_reports_how_many_builds_it_trimmed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = request(json!({"check_trim": true}));
    let (status, plan) = send(&app, Method::POST, PLAN, &req.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert!(plan["trim"]["trims"].as_u64().unwrap() > 0, "{plan}");
    assert!(plan["trim"]["wall_s"].as_f64().is_some(), "{plan}");
}

#[tokio::test]
async fn a_saved_sweep_is_written_with_its_summary_and_plans_as_the_plan_did() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = grid();
    let (_, plan) = send(&app, Method::POST, PLAN, &req.to_string()).await;

    let (status, saved) =
        send(&app, Method::PUT, "/api/sweeps/saved/grid", &save_body(&req, "A grid. More.", None))
            .await;
    assert_eq!(status, StatusCode::CREATED, "{saved}");
    assert_eq!(saved["name"], "grid");
    assert_eq!(saved["summary"], "A grid. More.");

    let text = std::fs::read_to_string(dir.join("sweep.grid.toml")).unwrap();
    assert_eq!(first_line(&text), "# A grid. More.", "{text}");
    assert!(!text.contains("astrum-studio:"), "{text}");
    // The file is the CLI's sweep: planning it gives the plan the page showed.
    let sweep = Sweep::from_toml_str(&text, "grid").unwrap();
    assert_eq!(sweep.runs.len() as u64, plan["runs"].as_u64().unwrap());
    assert_eq!(json!(sweep.value_keys), plan["keys"]);
    assert_eq!(sweep.runs[5].values.len(), 2);
    let parsed: SweepFile = toml::from_str(&text).unwrap();
    assert_eq!(parsed.base, dir.join("scenario.pair.toml"));
}

#[tokio::test]
async fn a_saved_sweep_keeps_its_assignments_and_reads_back_as_the_request() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = request(json!({
        "assignments": [{"aircraft": 1, "controller": "py-hold", "params": {"heading": 45}}],
        "vary": {"aircraft.2.offset.heading_deg": [-10.0, 10.0]},
        "perturb": [{"aircraft": [2], "rigid": false, "up_m": {"normal": 20.0}}],
        "samples": 2,
        "sample_seed": 3,
        "min_alt_hae_m": 2000.0
    }));
    let (status, saved) =
        send(&app, Method::PUT, "/api/sweeps/saved/studio", &save_body(&req, "", None)).await;
    assert_eq!(status, StatusCode::CREATED, "{saved}");

    let text = std::fs::read_to_string(dir.join("sweep.studio.toml")).unwrap();
    assert!(text.starts_with("# astrum-studio: [{"), "{text}");

    let (status, got) = send(&app, Method::GET, "/api/sweeps/saved/studio", "").await;
    assert_eq!(status, StatusCode::OK, "{got}");
    assert_eq!(got["name"], "studio");
    assert_eq!(got["summary"], "");
    let want: astrum_web::sweeps::SweepRequest = serde_json::from_value(req).unwrap();
    let back: astrum_web::sweeps::SweepRequest =
        serde_json::from_value(got["request"].clone()).unwrap();
    assert_eq!(back, want, "{got}");
}

#[tokio::test]
async fn saving_needs_overwrite_to_replace_and_bad_names_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let (status, _) =
        send(&app, Method::PUT, "/api/sweeps/saved/grid", &save_body(&grid(), "First.", None))
            .await;
    assert_eq!(status, StatusCode::CREATED);
    let first = std::fs::read_to_string(dir.join("sweep.grid.toml")).unwrap();

    let other = request(json!({"vary": {"aircraft.2.offset.forward_m": [1.0]}}));
    for overwrite in [None, Some(false)] {
        let (status, reply) = send(
            &app,
            Method::PUT,
            "/api/sweeps/saved/grid",
            &save_body(&other, "Second.", overwrite),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{reply}");
        assert_eq!(
            reply["error"],
            json!("sweep \"grid\" exists; send overwrite: true to replace it")
        );
        assert_eq!(std::fs::read_to_string(dir.join("sweep.grid.toml")).unwrap(), first);
    }

    let (status, reply) = send(
        &app,
        Method::PUT,
        "/api/sweeps/saved/grid",
        &save_body(&other, "Second.", Some(true)),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(
        first_line(&std::fs::read_to_string(dir.join("sweep.grid.toml")).unwrap()),
        "# Second."
    );

    for name in ["bad.name", "-dash", "no%2Fslash"] {
        let (status, reply) = send(
            &app,
            Method::PUT,
            &format!("/api/sweeps/saved/{name}"),
            &save_body(&grid(), "", None),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {reply}");
        let (status, _) = send(&app, Method::GET, &format!("/api/sweeps/saved/{name}"), "").await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}");
    }

    for summary in ["two\nlines".to_owned(), "x".repeat(201)] {
        let (status, reply) =
            send(&app, Method::PUT, "/api/sweeps/saved/long", &save_body(&grid(), &summary, None))
                .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    }
    assert_eq!(files(&dir), ["controllers.toml", "scenario.pair.toml", "sweep.grid.toml"]);
}

#[tokio::test]
async fn a_sweep_that_cannot_be_planned_is_not_saved() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = request(json!({
        "samples": 1,
        "perturb": [{"aircraft": [9], "forward_m": {"uniform": 10.0}}]
    }));
    let (status, reply) =
        send(&app, Method::PUT, "/api/sweeps/saved/unplannable", &save_body(&req, "", None)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert_eq!(files(&dir), ["controllers.toml", "scenario.pair.toml"]);
}

#[tokio::test]
async fn the_saved_list_shows_each_sweeps_base_and_the_files_it_cannot_read() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let (status, _) = send(
        &app,
        Method::PUT,
        "/api/sweeps/saved/grid",
        &save_body(&grid(), "A grid. More.", None),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    std::fs::write(dir.join("sweep.broken.toml"), "base = 3\n").unwrap();

    let (status, list) = send(&app, Method::GET, "/api/sweeps/saved", "").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    let entries = list.as_array().unwrap();
    assert_eq!(entries.len(), 2, "{list}");
    assert_eq!(entries[0]["name"], "broken");
    assert_eq!(entries[0]["base"], Value::Null);
    assert!(entries[0]["error"].as_str().is_some(), "{list}");
    assert_eq!(entries[1]["name"], "grid");
    assert_eq!(entries[1]["summary"], "A grid. More.");
    assert_eq!(entries[1]["base"], "pair");
    assert_eq!(entries[1]["error"], Value::Null);
    assert_eq!(entries[1]["path"], json!(dir.join("sweep.grid.toml").display().to_string()));

    let (status, reply) = send(&app, Method::GET, "/api/sweeps/saved/missing", "").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{reply}");

    // Nothing but the sweeps was written: no temporary file is left behind.
    assert_eq!(
        files(&dir),
        ["controllers.toml", "scenario.pair.toml", "sweep.broken.toml", "sweep.grid.toml"]
    );
}

/// The `[set]` table the Sweeps page writes for `pair` with no assignments.
const PAGE_SET: &str = "[set]\n\"sim.max_time_s\" = 60.0\n\"sim.lockstep\" = false\n\"aircraft.1.lockstep\" = false\n\"aircraft.2.lockstep\" = false\n";

/// Write `config/sweep.<name>.toml` as `base = <base>` then `text`.
fn write_sweep(dir: &Path, name: &str, text: &str, base: &Path) {
    let base = serde_json::to_string(&base.display().to_string()).unwrap();
    std::fs::write(dir.join(format!("sweep.{name}.toml")), format!("base = {base}\n{text}"))
        .unwrap();
}

#[tokio::test]
async fn a_summary_with_a_control_character_is_refused_and_nothing_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    for summary in ["a\tb", "two\rlines", "bell\u{7}", "vt\u{b}", "ff\u{c}", "del\u{7f}"] {
        let (status, reply) =
            send(&app, Method::PUT, "/api/sweeps/saved/ctl", &save_body(&grid(), summary, None))
                .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{summary:?}: {reply}");
        assert_eq!(reply["error"], json!("summary: control characters are not allowed"));
    }
    assert_eq!(files(&dir), ["controllers.toml", "scenario.pair.toml"]);
}

#[tokio::test]
async fn a_summary_that_would_read_back_as_the_assignments_line_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    for summary in ["astrum-studio: []", "  astrum-studio: [{}]"] {
        let (status, reply) =
            send(&app, Method::PUT, "/api/sweeps/saved/tag", &save_body(&grid(), summary, None))
                .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{summary:?}: {reply}");
        assert!(reply["error"].as_str().unwrap().contains("astrum-studio:"), "{reply}");
    }
    assert_eq!(files(&dir), ["controllers.toml", "scenario.pair.toml"]);
}

#[tokio::test]
async fn a_hand_written_sweep_is_read_back_only_when_the_page_would_write_the_same_file() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let base = dir.join("scenario.pair.toml");
    // (name, the file after its base line, the refusal's field or "" for 200)
    let cases = [
        ("page", PAGE_SET.to_owned(), ""),
        (
            "controllers",
            format!("controllers = [\"python3 hold.py\"]\n{PAGE_SET}"),
            "`controllers`",
        ),
        ("extra_set", format!("{PAGE_SET}\"sim.rate_hz\" = 60.0\n"), "`set`"),
        ("vary", format!("{PAGE_SET}\n[vary]\n\"aircraft.2.forward_m\" = [1.0]\n"), "vary key"),
        ("out", format!("out = \"runs/x\"\n{PAGE_SET}"), "`out`"),
    ];
    for (name, text, field) in cases {
        write_sweep(&dir, name, &text, &base);
        let (status, reply) =
            send(&app, Method::GET, &format!("/api/sweeps/saved/{name}"), "").await;
        if field.is_empty() {
            assert_eq!(status, StatusCode::OK, "{name}: {reply}");
            continue;
        }
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{name}: {reply}");
        let error = reply["error"].as_str().unwrap();
        assert!(
            error
                .starts_with(&format!("sweep \"{name}\" has settings the Sweeps page cannot edit")),
            "{error}"
        );
        assert!(error.contains(field), "{name}: {error}");
    }
}

#[tokio::test]
async fn a_base_that_is_not_this_config_directorys_saved_scenario_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let elsewhere = tmp.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::copy(dir.join("scenario.pair.toml"), elsewhere.join("scenario.pair.toml")).unwrap();
    write_sweep(&dir, "away", PAGE_SET, &elsewhere.join("scenario.pair.toml"));

    let (status, reply) = send(&app, Method::GET, "/api/sweeps/saved/away", "").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert_eq!(reply["error"], json!("base is not a saved scenario of this config directory"));
    let (status, list) = send(&app, Method::GET, "/api/sweeps/saved", "").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(list[0]["name"], "away");
    assert_eq!(list[0]["base"], Value::Null);
}

#[tokio::test]
async fn a_base_saved_with_lockstep_plans_and_saves_without_controllers() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("config");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(common::repo_config().join("controllers.toml"), dir.join("controllers.toml"))
        .unwrap();
    let mut doc = pair();
    doc["sim"]["lockstep"] = json!(true);
    for a in doc["aircraft"].as_array_mut().unwrap() {
        a["lockstep"] = json!(true);
    }
    let scenario: Scenario = serde_json::from_value(doc).unwrap();
    std::fs::write(dir.join("scenario.lock.toml"), scenario.to_toml()).unwrap();
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    let req = json!({"scenario": "lock", "max_time_s": 60.0});

    let (status, plan) = send(&app, Method::POST, PLAN, &req.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{plan}");
    assert_eq!(plan["runs"], 1);
    let (status, saved) =
        send(&app, Method::PUT, "/api/sweeps/saved/locked", &save_body(&req, "", None)).await;
    assert_eq!(status, StatusCode::CREATED, "{saved}");
}

#[tokio::test]
async fn an_unreadable_saved_sweep_is_listed_with_its_error_and_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = config_dir(tmp.path());
    let app =
        common::app_with_catalogs(tmp.path(), Bridge::new(config()), Catalogs::load(&dir).unwrap());
    std::fs::write(dir.join("sweep.binary.toml"), [0xff, 0xfe, b'\n']).unwrap();

    let (status, list) = send(&app, Method::GET, "/api/sweeps/saved", "").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(list[0]["name"], "binary");
    assert_eq!(list[0]["base"], Value::Null);
    let error = list[0]["error"].as_str().unwrap();
    assert!(error.starts_with("cannot read: "), "{error}");

    let (status, reply) = send(&app, Method::GET, "/api/sweeps/saved/binary", "").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{reply}");
    assert_eq!(reply["error"], json!(error));
}
