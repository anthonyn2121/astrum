//! The studio's web sweeps through the router: `POST /api/sweeps` starts one in
//! the background, `GET /api/sweeps/current` and `DELETE` watch and stop it,
//! and every sweep under `<runs-dir>/sweeps/<id>/` lists, reloads, serves its
//! results and replays its runs. The in-process sweeps fly the `pair` scenario
//! (no sim binary needed); one process-mode sweep flies the real sim with the
//! repository's Python controller, like `tests/runs.rs`' real-sim test.

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astrum_sim::scenario::Scenario;
use astrum_web::bridge::Bridge;
use astrum_web::catalogs::Catalogs;
use astrum_web::sweepruns::{SweepRuns, STOP_GRACE};
use axum::body::{to_bytes, Body};
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Method, Request, StatusCode};
use axum::Router;
use common::config;
use serde_json::{json, Value};
use tower::ServiceExt;

const LOCAL: &str = "localhost:8080";
/// Where process-mode sweeps' runs listen (run N on `BASE_PORT + N - 1`).
const BASE_PORT: u16 = 27840;
/// The base port of the real-sim stop test: the two real-sim tests run at once,
/// so each flies its run 1 on its own port.
const STOP_TEST_PORT: u16 = 27850;

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

/// A temp tree: `config/` (the repository's controller catalog, optionally
/// replaced, and the saved scenario `pair`) and the runs dir the sweeps go in.
struct Tree {
    dir: tempfile::TempDir,
}

impl Tree {
    fn new() -> Self {
        Self::with_controllers(
            &std::fs::read_to_string(common::repo_config().join("controllers.toml")).unwrap(),
        )
    }

    fn with_controllers(controllers: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(config.join("controllers.toml"), controllers).unwrap();
        let scenario: Scenario = serde_json::from_value(pair()).unwrap();
        std::fs::write(config.join("scenario.pair.toml"), scenario.to_toml()).unwrap();
        Self { dir }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn sweeps_dir(&self) -> PathBuf {
        self.path().join("runs").join("sweeps")
    }

    fn catalogs(&self) -> Catalogs {
        Catalogs::load(self.path().join("config")).unwrap()
    }

    /// The router and the sweep runner, flying process-mode sweeps with `sim`.
    fn app(&self, sim: &Path) -> (Router, SweepRuns) {
        self.app_stopping_in(sim, STOP_GRACE, BASE_PORT)
    }

    /// [`app`](Self::app), a stop waiting `grace` before the sweep's processes are killed.
    fn app_stopping_in(&self, sim: &Path, grace: Duration, base_port: u16) -> (Router, SweepRuns) {
        common::app_with_sweeps(
            self.path(),
            Bridge::new(config()),
            self.catalogs(),
            sim,
            base_port,
            grace,
        )
    }

    /// The router and the sweep runner over no sim binary (in-process sweeps only).
    fn app_in_process(&self) -> (Router, SweepRuns) {
        self.app(&self.path().join("no-astrum-sim"))
    }

    /// The sweep directory `id` under the runs dir.
    fn dir_of(&self, id: &str) -> PathBuf {
        self.sweeps_dir().join(id)
    }
}

async fn call(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = to_bytes(resp.into_body(), 64 << 20).await.unwrap();
    let value = serde_json::from_slice(&body)
        .unwrap_or_else(|e| panic!("{status}: not JSON ({e}): {}", String::from_utf8_lossy(&body)));
    (status, value)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    call(app, Request::get(uri).header(HOST, LOCAL).body(Body::empty()).unwrap()).await
}

async fn send(app: &Router, method: Method, uri: &str, body: &Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(HOST, LOCAL)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    call(app, req).await
}

async fn start(app: &Router, body: &Value) -> (StatusCode, Value) {
    send(app, Method::POST, "/api/sweeps", body).await
}

/// A request for `pair`: two runs of 2 s, in-process, merged with `extra`.
fn request(extra: Value) -> Value {
    let mut body = json!({"scenario": "pair", "max_time_s": 2.0});
    for (key, value) in extra.as_object().unwrap() {
        body[key] = value.clone();
    }
    body
}

fn assert_error(reply: &(StatusCode, Value), status: StatusCode, needles: &[&str]) {
    assert_eq!(reply.0, status, "{}", reply.1);
    let error = reply.1["error"].as_str().unwrap_or_else(|| panic!("no error in {}", reply.1));
    for needle in needles {
        assert!(error.contains(needle), "{error:?} should mention {needle:?}");
    }
}

/// Poll `GET /api/sweeps/current` until its sweep satisfies `pred`; that sweep.
async fn wait_current(app: &Router, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let (_, body) = get(app, "/api/sweeps/current").await;
        let sweep = body["sweep"].clone();
        if pred(&sweep) {
            return sweep;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}: {sweep}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn wait_state(app: &Router, want: &str) -> Value {
    wait_current(app, want, |sweep| sweep["state"] == want).await
}

async fn shutdown(sweeps: SweepRuns) {
    tokio::task::spawn_blocking(move || sweeps.shutdown()).await.unwrap();
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

/// A sweep directory with `meta` as its meta.json; the caller adds the rest.
fn plant(tree: &Tree, id: &str, meta: &Value) -> PathBuf {
    let dir = tree.dir_of(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
    dir
}

/// A sweep's meta as the studio writes it; `counts` (ok, failed) once it has ended.
fn meta_of(id: &str, state: &str, total: usize, counts: Option<(usize, usize)>) -> Value {
    let (ok, failed) =
        counts.map_or((Value::Null, Value::Null), |(ok, failed)| (json!(ok), json!(failed)));
    json!({
        "id": id, "scenario": "pair", "state": state, "reason": null, "total": total,
        "in_process": true, "jobs": 1, "started_at": "2026-01-01T00:00:00Z",
        "ended_at": null, "wall_s": null, "keys": [], "ok": ok, "failed": failed
    })
}

/// A `#!/bin/sh` program `body` at `dir/name`, executable.
fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Writes a sim that ignores SIGTERM (the ignore survives `exec`, so only
/// SIGKILL ends it) and a catalog whose one controller `sleeper` sleeps;
/// returns the sim. The tree's catalog is read when its app is built.
fn wedged_sim(tree: &Tree) -> PathBuf {
    let sim = script(tree.path(), "wedged-sim", "trap '' TERM\nexec sleep 1000\n");
    let hold = script(tree.path(), "sleeper", "exec sleep 1000\n");
    let catalog = format!(
        "[[controller]]\nid = \"sleeper\"\nname = \"Sleeper\"\ndescription = \"A stand-in.\"\n\
         command = [{:?}]\nid_flag = \"--id\"\nkinds = [\"fixed-wing\"]\nneeds_fleet = false\n",
        hold.display().to_string()
    );
    std::fs::write(tree.path().join("config").join("controllers.toml"), catalog).unwrap();
    sim
}

/// A process-mode request flying one `sleeper` controller on aircraft 1, long enough to stop.
fn wedged_request() -> Value {
    request(json!({
        "max_time_s": 600.0,
        "assignments": [{"aircraft": 1, "controller": "sleeper"}]
    }))
}

/// Once run 1's sim has been started, give its shell a moment to ignore TERM.
async fn sim_started(tree: &Tree, id: &str) {
    let log = tree.dir_of(id).join("run-0001.sim.log");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !log.exists() {
        assert!(Instant::now() < deadline, "no sim log at {}", log.display());
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
}

#[tokio::test]
async fn an_in_process_sweep_flies_to_its_end_and_is_served() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let body = request(json!({
        "perturb": [{"aircraft": [2], "right_m": {"uniform": 100.0}}],
        "samples": 4,
        "sample_seed": 7
    }));
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["state"], "running");
    assert_eq!(view["total"], 4);
    assert_eq!(view["in_process"], true);
    assert_eq!(view["runs"].as_array().unwrap().len(), 4);
    let id = view["id"].as_str().unwrap().to_owned();
    assert!(id.ends_with("-pair"), "{id}");

    let done = wait_state(&app, "finished").await;
    assert_eq!(done["id"], id.as_str());
    assert_eq!(done["ok"], 4, "{done}");
    for run in done["runs"].as_array().unwrap() {
        assert_eq!(run["status"], "ok", "{run}");
        let sim_time = run["sim_time_s"].as_f64().unwrap();
        assert!((sim_time - 2.0).abs() < 0.05, "{run}");
    }
    assert_eq!(done["keys"], json!(["aircraft.2.offset.right_m"]));

    let (status, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    assert_eq!(status, StatusCode::OK, "{results}");
    let columns: Vec<&str> =
        results["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(&columns[..4], ["run", "status", "seed", "sample"]);
    assert_eq!(columns[4], "aircraft.2.offset.right_m");
    let rows = results["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "{results}");
    let bag = columns.iter().position(|c| *c == "bag").unwrap();
    for row in rows {
        assert_eq!(row[1], "ok", "{row}");
        assert!(row[2].is_number(), "{row}");
        assert!(row[bag].as_str().unwrap().ends_with(".mcap"), "{row}");
    }

    let dir = tree.dir_of(&id);
    let names = files(&dir);
    for expected in ["request.json", "sweep.toml", "meta.json", "summary.csv"] {
        assert!(names.iter().any(|n| n == expected), "{names:?}");
    }
    for run in 1..=4 {
        for ext in ["mcap", "metrics.json"] {
            let name = format!("run-{run:04}.{ext}");
            assert!(dir.join(&name).is_file(), "{name} missing from {names:?}");
        }
    }

    let (status, replay) = get(&app, &format!("/api/sweeps/{id}/runs/3/replay")).await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    let ids: std::collections::BTreeSet<u64> =
        replay["frames"].as_array().unwrap().iter().map(|f| f["id"].as_u64().unwrap()).collect();
    assert!(ids.contains(&1) && ids.contains(&2), "{ids:?}");
    for beyond in ["5", "0", "x"] {
        let reply = get(&app, &format!("/api/sweeps/{id}/runs/{beyond}/replay")).await;
        assert_error(&reply, StatusCode::BAD_REQUEST, &["run"]);
    }
    std::fs::remove_file(dir.join("run-0002.mcap")).unwrap();
    let reply = get(&app, &format!("/api/sweeps/{id}/runs/2/replay")).await;
    assert_error(&reply, StatusCode::NOT_FOUND, &["run 2 of sweep", "has no bag"]);
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_second_sweep_is_refused_while_one_runs() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let long = request(json!({"repeats": 8, "jobs": 1, "max_time_s": 600.0}));
    let (status, first) = start(&app, &long).await;
    assert_eq!(status, StatusCode::CREATED, "{first}");
    let id = first["id"].as_str().unwrap().to_owned();
    let reply = start(&app, &long).await;
    assert_error(&reply, StatusCode::CONFLICT, &["a sweep is already running", &id]);
    wait_current(&app, "the first run to start", |s| s["runs"][0]["status"] == "running").await;
    let running = get(&app, &format!("/api/sweeps/{id}/runs/1/replay")).await;
    assert_error(&running, StatusCode::CONFLICT, &["run 1 of sweep", "still running"]);
    let (status, stopped) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["state"], "stopped");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_stop_mid_run_leaves_the_rest_queued_and_a_new_sweep_starts() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let long = request(json!({"repeats": 8, "jobs": 1, "max_time_s": 600.0}));
    let (status, begun) = start(&app, &long).await;
    assert_eq!(status, StatusCode::CREATED, "{begun}");
    let id = begun["id"].as_str().unwrap().to_owned();
    wait_current(&app, "the first run to start", |s| s["runs"][0]["status"] == "running").await;

    let (status, stopped) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["state"], "stopped");
    let count = |view: &Value, key: &str| view[key].as_u64().unwrap();
    let settled: u64 =
        ["stopped", "queued", "ok", "failed"].iter().map(|k| count(&stopped, k)).sum();
    assert_eq!(settled, count(&stopped, "total"), "{stopped}");
    assert!(count(&stopped, "stopped") >= 1, "the run in flight ends stopped: {stopped}");
    assert!(count(&stopped, "queued") >= 1, "{stopped}");

    let (status, reply) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_error(&(status, reply), StatusCode::NOT_FOUND, &["no sweep is running"]);

    // The meta on disk says stopped; a fresh server over the same dir reads the same sweep.
    let meta: Value =
        serde_json::from_slice(&std::fs::read(tree.dir_of(&id).join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["state"], "stopped", "{meta}");
    shutdown(sweeps).await;
    let (app, sweeps) = tree.app_in_process();
    let (status, reloaded) = get(&app, &format!("/api/sweeps/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{reloaded}");
    for key in ["state", "total", "queued", "ok", "failed", "stopped"] {
        assert_eq!(reloaded[key], stopped[key], "{key}");
    }

    let (status, next) = start(&app, &request(json!({"repeats": 1}))).await;
    assert_eq!(status, StatusCode::CREATED, "{next}");
    let done = wait_state(&app, "finished").await;
    assert_eq!(done["ok"], 1, "{done}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_sweep_on_disk_is_listed_and_reloaded_and_a_running_one_is_interrupted() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let (status, view) = start(&app, &request(json!({"repeats": 2}))).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    wait_state(&app, "finished").await;
    shutdown(sweeps).await;

    // A server started over the same runs dir sees the sweep as it ended.
    let (app, sweeps) = tree.app_in_process();
    let (_, list) = get(&app, "/api/sweeps").await;
    let summary = &list.as_array().unwrap()[0];
    assert_eq!(summary["id"], id.as_str(), "{list}");
    assert_eq!(
        (summary["state"].clone(), summary["ok"].clone(), summary["total"].clone()),
        (json!("finished"), json!(2), json!(2))
    );
    let (status, reloaded) = get(&app, &format!("/api/sweeps/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{reloaded}");
    assert_eq!((reloaded["ok"].clone(), reloaded["queued"].clone()), (json!(2), json!(0)));
    assert_eq!(get(&app, "/api/sweeps/current").await.1, json!({"sweep": null}));
    shutdown(sweeps).await;

    // A meta still saying running, with no server behind it: interrupted.
    let meta_path = tree.dir_of(&id).join("meta.json");
    let mut meta: Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
    meta["state"] = json!("running");
    std::fs::write(&meta_path, meta.to_string()).unwrap();
    let (app, sweeps) = tree.app_in_process();
    let (_, list) = get(&app, "/api/sweeps").await;
    assert_eq!(list[0]["state"], "interrupted", "{list}");
    let (_, interrupted) = get(&app, &format!("/api/sweeps/{id}")).await;
    assert_eq!(interrupted["state"], "interrupted", "{interrupted}");
    assert_eq!((interrupted["ok"].clone(), interrupted["total"].clone()), (json!(2), json!(2)));
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_refused_sweep_names_its_culprit_and_leaves_no_directory() {
    let tree = Tree::new();
    let sim = tree.path().join("no-astrum-sim");
    let (app, sweeps) = tree.app(&sim);
    let no_scenario = json!({"scenario": "missing", "max_time_s": 2.0});
    assert_error(&start(&app, &no_scenario).await, StatusCode::BAD_REQUEST, &["\"missing\""]);
    let bent = request(json!({"vary": {"aircraft.2.offset.bank_deg": [60.0]}}));
    assert_error(
        &start(&app, &bent).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        &["run 1", "aircraft.2.offset.bank_deg"],
    );
    let process = request(json!({
        "assignments": [{"aircraft": 1, "controller": "py-hold"}],
        "repeats": 1
    }));
    assert_error(
        &start(&app, &process).await,
        StatusCode::UNPROCESSABLE_ENTITY,
        &["no astrum-sim at"],
    );
    let left = std::fs::read_dir(tree.sweeps_dir()).map_or(0, Iterator::count);
    assert_eq!(left, 0, "a refused sweep left a directory");
    assert_eq!(get(&app, "/api/sweeps/current").await.1, json!({"sweep": null}));
    shutdown(sweeps).await;
}

#[tokio::test]
async fn unsafe_and_unknown_sweep_ids_are_refused() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    for unsafe_id in ["%2E%2E", "a%2Fb", "%2Ehidden", "a%20b"] {
        for suffix in ["", "/results", "/runs/1/replay"] {
            let reply = get(&app, &format!("/api/sweeps/{unsafe_id}{suffix}")).await;
            assert_error(&reply, StatusCode::BAD_REQUEST, &["sweep id"]);
        }
    }
    assert_error(&get(&app, "/api/sweeps/nope").await, StatusCode::NOT_FOUND, &["\"nope\""]);
    assert_error(
        &get(&app, "/api/sweeps/nope/results").await,
        StatusCode::NOT_FOUND,
        &["\"nope\""],
    );
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_summary_cell_with_commas_and_quotes_reads_back_as_one_cell() {
    let tree = Tree::new();
    let id = "20260101-000000-pair";
    let dir = tree.dir_of(id);
    std::fs::create_dir_all(&dir).unwrap();
    let meta = json!({
        "id": id, "scenario": "pair", "state": "finished", "reason": null, "total": 1,
        "in_process": true, "jobs": 1, "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z", "wall_s": 1.0, "keys": []
    });
    std::fs::write(dir.join("meta.json"), meta.to_string()).unwrap();
    let csv = "run,status,seed,sim_time_s,wall_s,speed_x,bag\n\
               1,\"failed: boom, \"\"quoted\"\"\",7,,,,run-0001.mcap\n";
    std::fs::write(dir.join("summary.csv"), csv).unwrap();

    let (app, sweeps) = tree.app_in_process();
    let (status, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    assert_eq!(status, StatusCode::OK, "{results}");
    assert_eq!(results["columns"].as_array().unwrap().len(), 7);
    let row = &results["rows"][0];
    assert_eq!(row.as_array().unwrap().len(), 7, "{results}");
    assert_eq!(row[1], "failed: boom, \"quoted\"");
    assert_eq!(row[2], 7);
    assert_eq!(row[3], Value::Null);
    assert_eq!(row[6], "run-0001.mcap");

    let (_, view) = get(&app, &format!("/api/sweeps/{id}")).await;
    assert_eq!(view["runs"][0]["status"], "failed", "{view}");
    assert_eq!(view["runs"][0]["reason"], "boom, \"quoted\"", "{view}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_summary_reads_back_cell_by_cell() {
    let tree = Tree::new();
    let id = "20260101-000000-pair";
    // CRLF ends; a u64 seed; `007`, ` 9 ` and `+3` stay text; a quote inside a
    // cell is literal; a quoted cell holds a newline and `""`; a line of just
    // `""` is a row; the last row has no line end.
    let csv = "run,status,seed,sim_time_s,note\r\n\
        1,ok,18446744073709551615,2.5,007\r\n\
        2,ok, 9 ,+3,a\"b\r\n\
        3,\"failed: line one\nline \"\"two\"\"\",-4,1e3,\r\n\
        \"\"\n\
        \n\
        4,ok,1,1.50,end";
    let dir = plant(&tree, id, &meta_of(id, "finished", 5, Some((4, 1))));
    std::fs::write(dir.join("summary.csv"), csv).unwrap();

    let (app, sweeps) = tree.app_in_process();
    let (status, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    assert_eq!(status, StatusCode::OK, "{results}");
    assert_eq!(results["columns"], json!(["run", "status", "seed", "sim_time_s", "note"]));
    assert_eq!(
        results["rows"],
        json!([
            [1, "ok", 18446744073709551615u64, 2.5, "007"],
            [2, "ok", " 9 ", "+3", "a\"b"],
            [3, "failed: line one\nline \"two\"", -4, 1000.0, null],
            [null],
            [4, "ok", 1, 1.5, "end"]
        ])
    );
    shutdown(sweeps).await;
}

#[tokio::test]
async fn an_empty_or_unterminated_summary_reads_back_what_it_holds() {
    let tree = Tree::new();
    let cases = [
        ("20260101-000000-pair", "", json!([]), json!([])),
        ("20260102-000000-pair", "run,status\n", json!(["run", "status"]), json!([])),
        (
            "20260103-000000-pair",
            "run,status,seed\n1,ok,1\n2,\"oops,3\n",
            json!(["run", "status", "seed"]),
            json!([[1, "ok", 1], [2, "oops,3\n"]]),
        ),
    ];
    let (app, sweeps) = tree.app_in_process();
    for (id, csv, columns, rows) in cases {
        let dir = plant(&tree, id, &meta_of(id, "finished", 1, Some((0, 0))));
        std::fs::write(dir.join("summary.csv"), csv).unwrap();
        let (status, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
        assert_eq!(status, StatusCode::OK, "{results}");
        assert_eq!(results["columns"], columns, "{id}");
        assert_eq!(results["rows"], rows, "{id}");
    }
    shutdown(sweeps).await;
}

#[tokio::test]
async fn the_list_skips_a_sweep_it_cannot_read_and_counts_a_finished_one_from_its_meta() {
    let tree = Tree::new();
    let (good, corrupt, unreadable) =
        ("20260101-000000-pair", "20260102-000000-pair", "20260103-000000-pair");
    // Finished, counts in its meta; its summary is a directory, so it cannot be read.
    let dir = plant(&tree, good, &meta_of(good, "finished", 2, Some((2, 0))));
    std::fs::create_dir(dir.join("summary.csv")).unwrap();
    // A meta that is not JSON.
    plant(&tree, corrupt, &meta_of(corrupt, "finished", 1, Some((1, 0))));
    std::fs::write(tree.dir_of(corrupt).join("meta.json"), "{not json").unwrap();
    // Running with no server behind it and no readable summary.
    let dir = plant(&tree, unreadable, &meta_of(unreadable, "running", 1, None));
    std::fs::create_dir(dir.join("summary.csv")).unwrap();

    let (app, sweeps) = tree.app_in_process();
    let (status, list) = get(&app, "/api/sweeps").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    let listed = list.as_array().unwrap();
    assert_eq!(listed.len(), 1, "{list}");
    assert_eq!(listed[0]["id"], good);
    assert_eq!((listed[0]["ok"].clone(), listed[0]["failed"].clone()), (json!(2), json!(0)));
    let reply = get(&app, &format!("/api/sweeps/{corrupt}")).await;
    assert_error(&reply, StatusCode::INTERNAL_SERVER_ERROR, &["not a sweep's meta"]);
    shutdown(sweeps).await;
}

#[tokio::test]
async fn two_starts_at_once_admit_exactly_one() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let long = request(json!({"repeats": 8, "jobs": 1, "max_time_s": 600.0}));
    let starts: Vec<_> = (0..2)
        .map(|_| {
            let (app, long) = (app.clone(), long.clone());
            tokio::spawn(async move { start(&app, &long).await })
        })
        .collect();
    let mut replies = Vec::new();
    for s in starts {
        replies.push(s.await.unwrap());
    }
    let created = replies.iter().filter(|r| r.0 == StatusCode::CREATED).count();
    let refused: Vec<_> = replies.iter().filter(|r| r.0 == StatusCode::CONFLICT).collect();
    assert_eq!((created, refused.len()), (1, 1), "{replies:?}");
    assert_error(refused[0], StatusCode::CONFLICT, &["already"]);
    let (status, stopped) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn jobs_above_256_are_refused_and_256_is_not() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let reply = start(&app, &request(json!({"jobs": 257}))).await;
    assert_error(&reply, StatusCode::BAD_REQUEST, &["jobs: at most 256"]);
    assert_eq!(reply.1["error"], "jobs: at most 256");
    assert_eq!(std::fs::read_dir(tree.sweeps_dir()).map_or(0, Iterator::count), 0);
    let (status, view) = start(&app, &request(json!({"jobs": 256, "repeats": 1}))).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    wait_state(&app, "finished").await;
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_meta_that_cannot_be_rewritten_keeps_the_outcome_and_says_why() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let long = request(json!({"repeats": 8, "jobs": 1, "max_time_s": 600.0}));
    let (status, begun) = start(&app, &long).await;
    assert_eq!(status, StatusCode::CREATED, "{begun}");
    let id = begun["id"].as_str().unwrap().to_owned();
    wait_current(&app, "the first run to start", |s| s["runs"][0]["status"] == "running").await;
    // A directory where meta.json is: the sweep's final rewrite cannot replace it.
    let meta = tree.dir_of(&id).join("meta.json");
    std::fs::remove_file(&meta).unwrap();
    std::fs::create_dir(&meta).unwrap();

    let (status, stopped) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["state"], "stopped", "{stopped}");
    assert!(stopped["reason"].as_str().unwrap_or_default().contains("meta.json"), "{stopped}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_sim_that_ignores_the_stop_is_killed_after_the_grace() {
    let tree = Tree::new();
    let sim = wedged_sim(&tree);
    let (app, sweeps) = tree.app_stopping_in(&sim, Duration::from_millis(300), BASE_PORT);
    let (status, view) = start(&app, &wedged_request()).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["in_process"], false);
    let id = view["id"].as_str().unwrap().to_owned();
    wait_current(&app, "the run to start", |s| s["runs"][0]["status"] == "running").await;
    sim_started(&tree, &id).await;

    let (status, stopped) = tokio::time::timeout(
        Duration::from_secs(30),
        send(&app, Method::DELETE, "/api/sweeps/current", &json!({})),
    )
    .await
    .expect("a stop the sim ignores hangs the sweep");
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["state"], "stopped", "{stopped}");
    assert_eq!(stopped["runs"][0]["status"], "stopped", "{stopped}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn shutdown_ends_a_sweep_whose_sim_ignores_the_stop() {
    let tree = Tree::new();
    let sim = wedged_sim(&tree);
    let (app, sweeps) = tree.app_stopping_in(&sim, Duration::from_millis(300), BASE_PORT);
    let (status, view) = start(&app, &wedged_request()).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    wait_current(&app, "the run to start", |s| s["runs"][0]["status"] == "running").await;
    sim_started(&tree, &id).await;

    tokio::time::timeout(Duration::from_secs(30), shutdown(sweeps))
        .await
        .expect("shutdown hangs on a sim that ignores its stop");
    let (_, current) = get(&app, "/api/sweeps/current").await;
    assert_eq!(current["sweep"]["state"], "stopped", "{current}");
}

/// `target/<profile>/astrum-sim` next to this test's `deps/` directory, or
/// `$ASTRUM_SIM_BIN`; `None` when it has not been built.
fn real_sim() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ASTRUM_SIM_BIN") {
        return Some(path.into());
    }
    let exe = std::env::current_exe().ok()?;
    let sim = exe.parent()?.parent()?.join("astrum-sim");
    sim.is_file().then_some(sim)
}

/// The tree of the real-sim tests: `py-hold` runs the repository's example
/// controller (by absolute path: the sim's directory is not the repository's)
/// and the sim to fly it with. `None` when no sim is built: a skip, which
/// passes only with `ASTRUM_SKIP_REAL_SIM=1`.
fn real_sim_tree() -> Option<(PathBuf, Tree)> {
    let Some(sim) = real_sim() else {
        assert!(
            std::env::var_os("ASTRUM_SKIP_REAL_SIM").is_some_and(|v| v == "1"),
            "no astrum-sim next to the test binaries: build it first (cargo build -p astrum-sim), \
             set ASTRUM_SIM_BIN, or skip this test with ASTRUM_SKIP_REAL_SIM=1"
        );
        return None;
    };
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../controllers/python/example.py");
    let example = example.canonicalize().unwrap();
    let controllers =
        std::fs::read_to_string(common::repo_config().join("controllers.toml")).unwrap().replace(
            "\"controllers/python/example.py\"",
            &format!("{:?}", example.display().to_string()),
        );
    Some((sim, Tree::with_controllers(&controllers)))
}

#[tokio::test]
async fn a_process_sweep_flies_the_real_sim_with_a_real_controller() {
    let Some((sim, tree)) = real_sim_tree() else { return };
    let (app, sweeps) = tree.app(&sim);
    let body = request(json!({
        "max_time_s": 3.0,
        "repeats": 2,
        "assignments": [
            {"aircraft": 1, "controller": "py-hold"},
            {"aircraft": 2, "controller": "py-hold"}
        ]
    }));
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["in_process"], false);
    let done = wait_state(&app, "finished").await;
    assert_eq!(done["ok"], 2, "{done}");
    for run in done["runs"].as_array().unwrap() {
        assert_eq!(run["status"], "ok", "{run}");
    }
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_process_sweep_stopped_mid_run_ends_stopped_promptly() {
    let Some((sim, tree)) = real_sim_tree() else { return };
    // A stop waits 300 ms before any process is killed.
    let (app, sweeps) = tree.app_stopping_in(&sim, Duration::from_millis(300), STOP_TEST_PORT);
    let body = request(json!({
        "max_time_s": 3600.0,
        "repeats": 1,
        "jobs": 1,
        "assignments": [
            {"aircraft": 1, "controller": "py-hold"},
            {"aircraft": 2, "controller": "py-hold"}
        ]
    }));
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    wait_current(&app, "the run to start", |s| s["runs"][0]["status"] == "running").await;
    let (status, stopped) = tokio::time::timeout(
        Duration::from_secs(60),
        send(&app, Method::DELETE, "/api/sweeps/current", &json!({})),
    )
    .await
    .expect("a stop must end the sweep promptly");
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["state"], "stopped", "{stopped}");
    shutdown(sweeps).await;
}

/// `GET /api/sweeps/<id>/tracks` with the query string `query`.
async fn get_tracks(app: &Router, id: &str, query: &str) -> (StatusCode, Value) {
    get(app, &format!("/api/sweeps/{id}/tracks?{query}")).await
}

/// The run numbers of a tracks reply, in order.
fn track_runs(reply: &Value) -> Vec<u64> {
    reply["runs"].as_array().unwrap().iter().map(|r| r["run"].as_u64().unwrap()).collect()
}

/// Start four 2 s in-process `pair` runs, wait until they finish, and return the id.
async fn finished_four(app: &Router) -> String {
    let body = request(json!({
        "samples": 4,
        "sample_seed": 7,
        "perturb": [{"aircraft": [2], "right_m": {"uniform": 100.0}}]
    }));
    let (status, view) = start(app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    wait_state(app, "finished").await;
    view["id"].as_str().unwrap().to_owned()
}

#[tokio::test]
async fn run_tracks_are_thinned_per_aircraft_and_paged() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let id = finished_four(&app).await;

    let (status, all) = get_tracks(&app, &id, "points=10").await;
    assert_eq!(status, StatusCode::OK, "{all}");
    assert_eq!(all["points"], 10);
    assert_eq!(
        (all["from"].as_u64(), all["count"].as_u64(), all["total"].as_u64()),
        (Some(1), Some(100), Some(4))
    );
    assert_eq!(track_runs(&all), [1, 2, 3, 4]);
    for run in all["runs"].as_array().unwrap() {
        assert_eq!(run["status"], "ok", "{run}");
        assert!((run["sim_time_s"].as_f64().unwrap() - 2.0).abs() < 0.05, "{run}");
        let aircraft = run["aircraft"].as_array().unwrap();
        let ids: Vec<u64> = aircraft.iter().map(|a| a["id"].as_u64().unwrap()).collect();
        assert_eq!(ids, [1, 2], "{run}");
        assert_eq!(aircraft[0]["model"], "f16", "{run}");
        for track in aircraft {
            let t = track["t"].as_array().unwrap();
            for key in ["e", "n", "u"] {
                assert_eq!(track[key].as_array().unwrap().len(), t.len(), "{track}");
            }
            assert!(!t.is_empty() && t.len() <= 10, "{track}");
            let times: Vec<f64> = t.iter().map(|x| x.as_f64().unwrap()).collect();
            assert!(times.windows(2).all(|w| w[0] < w[1]), "{track}");
            assert_eq!(track["died_at_s"], Value::Null, "{track}");
        }
    }

    let (status, page) = get_tracks(&app, &id, "from=3&count=2&points=10").await;
    assert_eq!(status, StatusCode::OK, "{page}");
    assert_eq!(
        (page["from"].as_u64(), page["count"].as_u64(), page["total"].as_u64()),
        (Some(3), Some(2), Some(4))
    );
    assert_eq!(track_runs(&page), [3, 4]);
    let (status, past) = get_tracks(&app, &id, "from=9").await;
    assert_eq!(status, StatusCode::OK, "{past}");
    assert!(track_runs(&past).is_empty(), "{past}");
    let (status, default) = get_tracks(&app, &id, "").await;
    assert_eq!(status, StatusCode::OK, "{default}");
    assert_eq!(
        (default["points"].as_u64(), default["from"].as_u64(), default["count"].as_u64()),
        (Some(80), Some(1), Some(100))
    );
    let (status, limits) = get_tracks(&app, &id, "points=2&count=1").await;
    assert_eq!(status, StatusCode::OK, "{limits}");
    let (status, limits) = get_tracks(&app, &id, "points=400&count=500").await;
    assert_eq!(status, StatusCode::OK, "{limits}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn tracks_refuse_bad_parameters_and_ids_by_name() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let id = finished_four(&app).await;
    let bad = [
        ("points=1", "points"),
        ("points=401", "points"),
        ("points=x", "points"),
        ("count=0", "count"),
        ("count=501", "count"),
        ("from=0", "from"),
        ("from=-1", "from"),
    ];
    for (query, name) in bad {
        assert_error(&get_tracks(&app, &id, query).await, StatusCode::BAD_REQUEST, &[name]);
    }
    for unsafe_id in ["%2E%2E", "a%20b"] {
        let reply = get(&app, &format!("/api/sweeps/{unsafe_id}/tracks")).await;
        assert_error(&reply, StatusCode::BAD_REQUEST, &["sweep id"]);
    }
    assert_error(&get_tracks(&app, "nope", "").await, StatusCode::NOT_FOUND, &["\"nope\""]);
    shutdown(sweeps).await;
}

#[tokio::test]
async fn run_tracks_are_cached_beside_their_bags_and_a_corrupt_cache_is_rebuilt() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let id = finished_four(&app).await;
    let dir = tree.dir_of(&id);
    let cache = dir.join("run-0001.tracks-10.json");

    assert!(!cache.exists());
    let (status, first) = get_tracks(&app, &id, "points=10&count=1").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert!(cache.is_file(), "{:?}", files(&dir));
    let (_, again) = get_tracks(&app, &id, "points=10&count=1").await;
    assert_eq!(again, first);

    std::fs::write(&cache, b"{ not tracks").unwrap();
    let (status, rebuilt) = get_tracks(&app, &id, "points=10&count=1").await;
    assert_eq!(status, StatusCode::OK, "{rebuilt}");
    assert_eq!(rebuilt, first);
    let cached: Value = serde_json::from_slice(&std::fs::read(&cache).unwrap()).unwrap();
    assert!(cached.is_array(), "{cached}");
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_run_whose_bag_is_gone_has_no_tracks_even_with_a_cache() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let id = finished_four(&app).await;
    let (status, before) = get_tracks(&app, &id, "points=10").await;
    assert_eq!(status, StatusCode::OK, "{before}");
    assert_eq!(track_runs(&before), [1, 2, 3, 4]);

    std::fs::remove_file(tree.dir_of(&id).join("run-0002.mcap")).unwrap();
    let (status, after) = get_tracks(&app, &id, "points=10").await;
    assert_eq!(status, StatusCode::OK, "{after}");
    assert_eq!(track_runs(&after), [1, 3, 4]);
    assert_eq!(after["total"], 4);
    shutdown(sweeps).await;
}

#[tokio::test]
async fn a_run_that_has_not_finished_has_no_tracks_yet() {
    let tree = Tree::new();
    let (app, sweeps) = tree.app_in_process();
    let long = request(json!({"repeats": 8, "jobs": 1, "max_time_s": 600.0}));
    let (status, begun) = start(&app, &long).await;
    assert_eq!(status, StatusCode::CREATED, "{begun}");
    let id = begun["id"].as_str().unwrap().to_owned();
    wait_current(&app, "the first run to start", |s| s["runs"][0]["status"] == "running").await;

    let (_, before) = get(&app, "/api/sweeps/current").await;
    let (status, reply) = get_tracks(&app, &id, "points=10").await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["total"], 8);
    // Every run in the reply had finished when the sweep was read before the call.
    for run in track_runs(&reply) {
        let entry = &before["sweep"]["runs"][run as usize - 1];
        let status = entry["status"].as_str().unwrap();
        assert!(["ok", "failed", "stopped"].contains(&status), "run {run} is {status}: {reply}");
    }

    let (status, stopped) = send(&app, Method::DELETE, "/api/sweeps/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    shutdown(sweeps).await;
}
