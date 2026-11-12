//! Race scenarios through the studio's live runs: a race flies the `astrum-race`
//! binary (a shell script standing in for it, as `tests/runs.rs` stands in for
//! `astrum-sim`), handed the scenario with its `[course]` and the course seed;
//! the run reports its course, and its recording's replay carries the course
//! the recorded scenario lists.

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astrum_bag::BagWriter;
use astrum_net::keys;
use astrum_proto::{encode_telemetry, Telemetry, VehicleKind};
use astrum_race::scenario::RaceScenario;
use astrum_web::bridge::Bridge;
use astrum_web::catalogs::Catalogs;
use astrum_web::runs::Runs;
use astrum_web::sweepruns::{SweepRuns, STOP_GRACE};
use axum::body::{to_bytes, Body};
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use axum::Router;
use common::config;
use serde_json::{json, Value};
use tower::ServiceExt;

/// Tests write scripts and then spawn them; one at a time, so no other test's
/// fork holds a script open for writing (ETXTBSY).
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const LOCAL: &str = "localhost:8080";
/// Where the race sweeps' runs listen (run N on `BASE_PORT + N - 1`).
const BASE_PORT: u16 = 27880;

/// A temp tree: `config/` (the repository's controllers, the race scenario
/// `race`, the same race with its gates listed as `gated`, and an ordinary
/// scenario `solo`) and a recorded bag whose scenario is the resolved race.
struct Tree {
    dir: tempfile::TempDir,
    /// The resolved race (seed 5) as the bag records it.
    resolved: String,
}

impl Tree {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = common::race_config(dir.path());
        let race = RaceScenario::from_toml_str(common::RACE_SCENARIO).unwrap();
        let resolved = race.resolve(None).unwrap().to_toml();
        std::fs::write(config.join("scenario.gated.toml"), &resolved).unwrap();
        std::fs::write(
            config.join("scenario.solo.toml"),
            "[world]\norigin_lat_deg = 37.0\norigin_lon_deg = -122.0\norigin_alt_hae_m = 0.0\n\
             [sim]\nrate_hz = 120.0\nmax_time_s = 5.0\n\
             [[aircraft]]\nid = 1\nmodel = \"F450\"\nlat_deg = 37.0\nlon_deg = -122.0\n\
             alt_hae_m = 50.0\nheading_deg = 0.0\ntas_mps = 0.0\n",
        )
        .unwrap();
        write_bag(&dir.path().join("fixture.mcap"), &resolved);
        Self { dir, resolved }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn catalogs(&self) -> Catalogs {
        Catalogs::load(self.path().join("config")).unwrap()
    }

    /// A stand-in `astrum-race`: prints its arguments, then runs until
    /// SIGTERM, when it copies the fixture bag to its `--record` path.
    fn race_bin(&self) -> PathBuf {
        let fixture = self.path().join("fixture.mcap");
        let body = format!(
            "echo \"args $*\"\n\
             while [ $# -gt 0 ]; do [ \"$1\" = --record ] && bag=\"$2\"; shift; done\n\
             finish() {{ cp '{}' \"$bag\"; echo finished; exit 0; }}\n\
             trap finish TERM\n\
             while true; do sleep 0.05; done\n",
            fixture.display()
        );
        let path = self.path().join("race");
        std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn app(&self) -> (Router, Runs) {
        // Ordinary runs would fly `astrum-sim`, which does not exist here: a
        // race that reached it would fail to start.
        common::app_with_race_runs(
            self.path(),
            Bridge::new(config()),
            self.catalogs(),
            &self.path().join("no-astrum-sim"),
            &self.race_bin(),
        )
    }
}

/// A bag of one aircraft's telemetry whose recorded scenario is `scenario`.
fn write_bag(path: &Path, scenario: &str) {
    let mut w = BagWriter::create(path, Some(scenario)).unwrap();
    for tick in 1..=60u64 {
        let t = Telemetry {
            aircraft_id: 1,
            tick,
            sim_time_s: tick as f64 / 120.0,
            vehicle_kind: VehicleKind::Multirotor,
            ..Telemetry::default()
        };
        let ns = (t.sim_time_s * 1e9) as u64;
        w.write(&keys::telemetry(1), ns, ns, &encode_telemetry(&t)).unwrap();
    }
    w.finish().unwrap();
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

async fn send(app: &Router, method: &str, uri: &str, body: &Value) -> (StatusCode, Value) {
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
    send(app, "POST", "/api/runs", body).await
}

async fn wait_for_log(app: &Router, what: &str, pred: impl Fn(&[Value]) -> bool) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (_, body) = get(app, "/api/runs/current/log").await;
        let lines = body["lines"].as_array().unwrap().clone();
        if pred(&lines) {
            return lines;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}: {lines:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The stand-in's `args …` line of the current run.
async fn race_args(app: &Router) -> String {
    let lines = wait_for_log(app, "the race's arguments", |lines| {
        lines.iter().any(|l| l["line"].as_str().unwrap().starts_with("args "))
    })
    .await;
    let line = lines.iter().find(|l| l["line"].as_str().unwrap().starts_with("args ")).unwrap();
    assert_eq!(line["source"], "sim");
    line["line"].as_str().unwrap().to_owned()
}

/// Stop the active run and return its final view.
async fn stop(app: &Router) -> Value {
    let (status, run) = send(app, "DELETE", "/api/runs/current", &json!(null)).await;
    assert_eq!(status, StatusCode::OK, "{run}");
    assert_eq!(run["state"], "finished", "{run}");
    run
}

async fn shutdown(runs: Runs) {
    tokio::task::spawn_blocking(move || runs.shutdown()).await.unwrap();
}

#[tokio::test]
async fn a_race_run_flies_the_race_binary_with_the_scenario_and_its_course_seed() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app();

    let (status, view) = start(&app, &json!({"scenario": "race", "seed": 7})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    // The course the run flies: seed 7 (not the scenario's 5), as the preview draws it.
    let course = &view["course"];
    assert_eq!(course["seed"], 7);
    assert_eq!(course["gates"].as_array().unwrap().len(), 4);
    assert!(course["length_m"].as_f64().unwrap() > 0.0, "{course}");
    let (_, preview) = get(&app, "/api/scenarios/race/course?seed=7").await;
    assert_eq!(*course, preview);
    let (_, current) = get(&app, "/api/runs/current").await;
    assert_eq!(current["run"]["course"], preview);

    let dir = view["dir"].as_str().unwrap();
    let args = race_args(&app).await;
    let endpoint = view["endpoint"].as_str().unwrap();
    assert_eq!(
        args,
        format!("args {dir}/scenario.toml --listen {endpoint} --seed 7 --record {dir}/run.mcap")
    );
    // The file it was handed: the ordinary scenario, then the `[course]` envelope.
    let written: toml::Table =
        toml::from_str(&std::fs::read_to_string(format!("{dir}/scenario.toml")).unwrap()).unwrap();
    assert_eq!(written["course"]["gates"].as_integer(), Some(4));
    assert_eq!(written["sim"]["seed"].as_integer(), Some(5));
    assert_eq!(written["aircraft"].as_array().unwrap().len(), 1);

    let done = stop(&app).await;
    assert_eq!(done["course"], preview, "the finished run still reports its course");
    // The bag is the stand-in's fixture: its scenario lists the resolved gates.
    let (status, replay) = get(&app, &format!("/api/runs/{id}/replay")).await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    let recorded = &replay["meta"]["course"];
    assert_eq!(recorded["seed"], 5);
    assert_eq!(recorded["gates"].as_array().unwrap().len(), 4);
    assert_eq!(replay["meta"]["scenario"], tree.resolved);
    // Race metrics are reported for a race recording even before any gate event.
    let (status, metrics) = get(&app, &format!("/api/runs/{id}/metrics")).await;
    assert_eq!(status, StatusCode::OK, "{metrics}");
    assert_eq!(metrics["gates_passed"], 0, "{metrics}");
    assert_eq!(metrics["gates_missed"], 0, "{metrics}");
    shutdown(runs).await;
}

#[tokio::test]
async fn a_race_run_without_a_seed_flies_the_scenarios_own() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app();
    let (status, view) = start(&app, &json!({"scenario": "race"})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["course"]["seed"], 5);
    assert!(race_args(&app).await.contains(" --seed 5 "));
    stop(&app).await;
    shutdown(runs).await;
}

#[tokio::test]
async fn a_seed_only_goes_with_a_race_whose_course_is_drawn() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app();

    let (status, body) = start(&app, &json!({"scenario": "solo", "seed": 3})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"].as_str().unwrap().contains("[course]"), "{body}");
    let (status, body) = start(&app, &json!({"scenario": "gated", "seed": 3})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"].as_str().unwrap().contains("[[course.gate]]"), "{body}");
    let (_, current) = get(&app, "/api/runs/current").await;
    assert_eq!(current["run"], Value::Null, "nothing was started");

    // The listed gates are flown as they are: no seed goes to the binary.
    let (status, view) = start(&app, &json!({"scenario": "gated"})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["course"]["gates"].as_array().unwrap().len(), 4);
    let args = race_args(&app).await;
    assert!(!args.contains("--seed"), "{args}");
    stop(&app).await;

    // An ordinary scenario has no course and flies `astrum-sim` (absent here).
    let (status, body) = start(&app, &json!({"scenario": "solo"})).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(body["error"].as_str().unwrap().contains("no-astrum-sim"), "{body}");
    shutdown(runs).await;
}

// --- Sweeps --------------------------------------------------------------------

impl Tree {
    /// A stand-in `astrum-race` for sweeps: prints its arguments, copies the
    /// fixture bag to its `--record` path and ends by itself.
    fn short_race_bin(&self) -> PathBuf {
        let fixture = self.path().join("fixture.mcap");
        let body = format!(
            "echo \"args $*\"\n\
             while [ $# -gt 0 ]; do [ \"$1\" = --record ] && bag=\"$2\"; shift; done\n\
             sleep 0.1\ncp '{}' \"$bag\"\n\
             echo '[astrum-sim] stopped at tick 300: sim 2.500 s in 0.100 s wall (25.0× real time)'\n",
            fixture.display()
        );
        let path = self.path().join("short-race");
        std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn sweep_app(&self) -> (Router, SweepRuns) {
        common::app_with_race_sweeps(
            self.path(),
            Bridge::new(config()),
            self.catalogs(),
            (&self.path().join("no-astrum-sim"), &self.short_race_bin()),
            BASE_PORT,
            STOP_GRACE,
        )
    }
}

fn sweep_request(extra: Value) -> Value {
    let mut body = json!({"scenario": "race", "max_time_s": 20.0, "repeats": 3});
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

async fn plan(app: &Router, body: &Value) -> (StatusCode, Value) {
    send(app, "POST", "/api/sweeps/plan", body).await
}

fn seeds(plan: &Value, key: &str) -> Vec<Value> {
    plan["rows"].as_array().unwrap().iter().map(|r| r[key].clone()).collect()
}

#[tokio::test]
async fn a_race_sweeps_plan_shows_each_runs_course_seed_by_mode() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, sweeps) = tree.sweep_app();

    // Fixed (the default): the scenario's course seed 5 in every run, while
    // `seed` (the sim's other randomness) moves with the repeat.
    let (status, fixed) = plan(&app, &sweep_request(json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{fixed}");
    assert_eq!(fixed["runs"], 3);
    assert_eq!(fixed["in_process"], false, "a race flies as processes");
    assert_eq!(seeds(&fixed, "course_seed"), [5, 5, 5].map(Value::from));
    assert_eq!(seeds(&fixed, "seed"), [5, 6, 7].map(Value::from));
    let (_, explicit) = plan(&app, &sweep_request(json!({"course_seed_mode": "fixed"}))).await;
    assert_eq!(explicit["rows"], fixed["rows"]);

    // Randomize: repeat r flies the base seed + r.
    let (status, random) =
        plan(&app, &sweep_request(json!({"course_seed_mode": "randomize"}))).await;
    assert_eq!(status, StatusCode::OK, "{random}");
    assert_eq!(seeds(&random, "course_seed"), [5, 6, 7].map(Value::from));

    // An ordinary scenario has no course seed.
    let (status, plain) = plan(&app, &sweep_request(json!({"scenario": "solo"}))).await;
    assert_eq!(status, StatusCode::OK, "{plain}");
    assert_eq!(seeds(&plain, "course_seed"), [const { Value::Null }; 3]);
    shutdown_sweeps(sweeps).await;
}

async fn shutdown_sweeps(sweeps: SweepRuns) {
    tokio::task::spawn_blocking(move || sweeps.shutdown()).await.unwrap();
}

#[tokio::test]
async fn a_race_sweep_refuses_what_does_not_apply_to_a_race() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, sweeps) = tree.sweep_app();
    for (extra, status, needle) in [
        (json!({"scenario": "solo", "course_seed_mode": "fixed"}), 400, "no [course]"),
        (json!({"course_seed_mode": "sometimes"}), 400, "`fixed` or `randomize`"),
        (json!({"scenario": "gated", "course_seed_mode": "randomize"}), 422, "[[course.gate]]"),
        (json!({"vary": {"aircraft.1.offset.right_m": [0.0, 1.0]}}), 422, "offsets"),
        (
            json!({"perturb": [{"aircraft": [1], "right_m": {"uniform": 1.0}}], "samples": 2}),
            422,
            "offsets",
        ),
    ] {
        let reply = plan(&app, &sweep_request(extra.clone())).await;
        assert_eq!(reply.0.as_u16(), status, "{extra}: {}", reply.1);
        assert!(reply.1["error"].as_str().unwrap().contains(needle), "{extra}: {}", reply.1);
    }
    shutdown_sweeps(sweeps).await;
}

#[tokio::test]
async fn a_race_sweep_flies_the_race_binary_per_run_and_reports_course_and_race_columns() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, sweeps) = tree.sweep_app();
    let body = sweep_request(json!({"course_seed_mode": "randomize"}));
    let (status, view) = send(&app, "POST", "/api/sweeps", &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["in_process"], false);
    let id = view["id"].as_str().unwrap().to_owned();
    let deadline = Instant::now() + Duration::from_secs(30);
    let done = loop {
        let (_, current) = get(&app, "/api/sweeps/current").await;
        if current["sweep"]["state"] == "finished" {
            break current["sweep"].clone();
        }
        assert!(Instant::now() < deadline, "the sweep never finished: {current}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(done["ok"], 3, "{done}");

    // Each run's race binary was given its own course seed.
    let out = PathBuf::from(done["out"].as_str().unwrap());
    for (run, seed) in [(1, 5), (2, 6), (3, 7)] {
        let log = std::fs::read_to_string(out.join(format!("run-{run:04}.sim.log"))).unwrap();
        assert!(log.contains(&format!(" --seed {seed} ")), "run {run}: {log}");
        assert!(log.contains(&format!("run-{run:04}.toml --listen ")), "run {run}: {log}");
        let written: toml::Table = toml::from_str(
            &std::fs::read_to_string(out.join(format!("run-{run:04}.toml"))).unwrap(),
        )
        .unwrap();
        assert_eq!(written["course"]["gates"].as_integer(), Some(4));
    }

    let (status, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    assert_eq!(status, StatusCode::OK, "{results}");
    let columns: Vec<&str> =
        results["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    assert_eq!(&columns[..4], ["run", "status", "seed", "course_seed"]);
    let at =
        |name: &str| columns.iter().position(|c| *c == name).unwrap_or_else(|| panic!("{name}"));
    for name in ["gates_passed", "gates_missed", "race_time_s", "race_outcome", "end_reason"] {
        at(name);
    }
    let rows = results["rows"].as_array().unwrap();
    let course_seeds: Vec<Value> = rows.iter().map(|r| r[3].clone()).collect();
    assert_eq!(course_seeds, [5, 6, 7].map(Value::from));
    assert!(rows.iter().all(|r| r[at("gates_passed")] == 0), "{results}");

    // A run's replay carries the course its bag recorded.
    let (status, replay) = get(&app, &format!("/api/sweeps/{id}/runs/2/replay")).await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    assert_eq!(replay["meta"]["course"]["gates"].as_array().unwrap().len(), 4);
    shutdown_sweeps(sweeps).await;
}

#[tokio::test]
async fn a_race_sweep_without_a_race_binary_is_refused_naming_it() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    // The ordinary `astrum-sim` stand-in exists; the race binary does not.
    let (app, sweeps) = common::app_with_race_sweeps(
        tree.path(),
        Bridge::new(config()),
        tree.catalogs(),
        (&tree.short_race_bin(), &tree.path().join("no-astrum-race")),
        BASE_PORT,
        STOP_GRACE,
    );
    let (status, body) = send(&app, "POST", "/api/sweeps", &sweep_request(json!({}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body["error"].as_str().unwrap().contains("no astrum-race at"), "{body}");
    shutdown_sweeps(sweeps).await;
}

#[tokio::test]
async fn a_saved_race_sweep_round_trips_with_its_course_seed_mode() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, sweeps) = tree.sweep_app();
    let request = sweep_request(json!({"course_seed_mode": "randomize"}));
    let (status, saved) = send(
        &app,
        "PUT",
        "/api/sweeps/saved/racing",
        &json!({"request": request, "summary": "Course per repeat."}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{saved}");
    let text = std::fs::read_to_string(tree.path().join("config/sweep.racing.toml")).unwrap();
    assert!(text.contains("course_seed_mode = \"randomize\""), "{text}");
    assert!(text.contains("scenario.race.toml"), "{text}");

    let (status, loaded) = get(&app, "/api/sweeps/saved/racing").await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    assert_eq!(loaded["request"]["course_seed_mode"], "randomize");
    assert_eq!(loaded["request"]["scenario"], "race");
    assert_eq!(loaded["summary"], "Course per repeat.");
    // Loaded and planned again: the same sweep.
    let (_, again) = plan(&app, &loaded["request"]).await;
    let (_, first) = plan(&app, &request).await;
    assert_eq!(again, first);

    // The default mode is not written, and reads back as unset.
    let plain = sweep_request(json!({}));
    let (status, _) =
        send(&app, "PUT", "/api/sweeps/saved/plain", &json!({"request": plain})).await;
    assert_eq!(status, StatusCode::CREATED);
    let text = std::fs::read_to_string(tree.path().join("config/sweep.plain.toml")).unwrap();
    assert!(!text.contains("course_seed_mode"), "{text}");
    let (_, loaded) = get(&app, "/api/sweeps/saved/plain").await;
    assert!(loaded["request"].get("course_seed_mode").is_none(), "{loaded}");
    shutdown_sweeps(sweeps).await;
}

// --- The real astrum-race --------------------------------------------------------

/// `target/<profile>/astrum-race` next to this test's `deps/` directory, or
/// `$ASTRUM_RACE_BIN`; `None` when it has not been built.
fn real_race() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ASTRUM_RACE_BIN") {
        return Some(path.into());
    }
    let exe = std::env::current_exe().ok()?;
    let race = exe.parent()?.parent()?.join("astrum-race");
    race.is_file().then_some(race)
}

#[tokio::test]
async fn a_race_sweep_flies_the_real_race_binary_with_a_real_controller() {
    let _serial = SERIAL.lock().await;
    let Some(race) = real_race() else {
        assert!(
            std::env::var_os("ASTRUM_SKIP_REAL_SIM").is_some_and(|v| v == "1"),
            "no astrum-race next to the test binaries: build it first (cargo build -p \
             astrum-race), set ASTRUM_RACE_BIN, or skip this test with ASTRUM_SKIP_REAL_SIM=1"
        );
        return;
    };
    let tree = Tree::new();
    // The repository's controllers, `py-autopilot` by absolute path: the race's
    // directory is not the repository's.
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../controllers/python/example.py");
    let controllers =
        std::fs::read_to_string(common::repo_config().join("controllers.toml")).unwrap().replace(
            "\"controllers/python/example.py\"",
            &format!("{:?}", example.canonicalize().unwrap().display().to_string()),
        );
    std::fs::write(tree.path().join("config/controllers.toml"), controllers).unwrap();
    let (app, sweeps) = common::app_with_race_sweeps(
        tree.path(),
        Bridge::new(config()),
        tree.catalogs(),
        (&tree.path().join("no-astrum-sim"), &race),
        BASE_PORT + 20,
        STOP_GRACE,
    );
    let body = sweep_request(json!({
        "max_time_s": 3.0,
        "repeats": 2,
        "course_seed_mode": "randomize",
        "assignments": [{
            "aircraft": 1,
            "controller": "py-autopilot",
            "params": {"alt": 2.5, "speed": 0.0}
        }]
    }));
    let (status, view) = send(&app, "POST", "/api/sweeps", &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    let deadline = Instant::now() + Duration::from_secs(120);
    let done = loop {
        let (_, current) = get(&app, "/api/sweeps/current").await;
        if current["sweep"]["state"] != "running" {
            break current["sweep"].clone();
        }
        assert!(Instant::now() < deadline, "the sweep never ended: {current}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(done["state"], "finished", "{done}");
    assert_eq!(done["ok"], 2, "{done}");

    let (_, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    let columns: Vec<&str> =
        results["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let at =
        |name: &str| columns.iter().position(|c| *c == name).unwrap_or_else(|| panic!("{name}"));
    let rows = results["rows"].as_array().unwrap();
    assert_eq!(
        rows.iter().map(|r| r[at("course_seed")].clone()).collect::<Vec<_>>(),
        [5, 6].map(Value::from)
    );
    for row in rows {
        // The drone holds station: no gate is passed, and the run times out.
        assert_eq!(row[at("gates_passed")], 0, "{row}");
        assert_eq!(row[at("race_outcome")], "timed-out", "{row}");
    }
    // Each recording lists its course's gates, so its replay carries them: the
    // drawn course is the one the preview shows for the seed.
    for (run, seed) in [(1, 5), (2, 6)] {
        let (status, replay) = get(&app, &format!("/api/sweeps/{id}/runs/{run}/replay")).await;
        assert_eq!(status, StatusCode::OK, "{replay}");
        let (_, preview) = get(&app, &format!("/api/scenarios/race/course?seed={seed}")).await;
        assert_eq!(replay["meta"]["course"]["gates"], preview["gates"], "run {run}");
        assert_eq!(replay["meta"]["course"]["seed"], seed);
    }
    shutdown_sweeps(sweeps).await;
}

// --- Gate count -------------------------------------------------------------------

#[tokio::test]
async fn a_race_runs_gate_count_reaches_its_course_and_the_scenario_it_flies() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app();

    let (status, view) = start(&app, &json!({"scenario": "race", "seed": 7, "gates": 6})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let course = &view["course"];
    assert_eq!(course["gates"].as_array().unwrap().len(), 6);
    assert!(course["length_m"].as_f64().unwrap() > 0.0, "{course}");
    // The preview of that seed and count is the course the run reports.
    let (status, preview) = get(&app, "/api/scenarios/race/course?seed=7&gates=6").await;
    assert_eq!(status, StatusCode::OK, "{preview}");
    assert_eq!(*course, preview);
    // The file the race binary reads says so, and it is still given the seed.
    let dir = view["dir"].as_str().unwrap();
    let written: toml::Table =
        toml::from_str(&std::fs::read_to_string(format!("{dir}/scenario.toml")).unwrap()).unwrap();
    assert_eq!(written["course"]["gates"].as_integer(), Some(6));
    assert!(race_args(&app).await.contains(" --seed 7 "));
    stop(&app).await;

    // The scenario's own count (4) is the default.
    let (status, view) = start(&app, &json!({"scenario": "race"})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert_eq!(view["course"]["gates"].as_array().unwrap().len(), 4);
    stop(&app).await;
    shutdown(runs).await;
}

#[tokio::test]
async fn a_gate_count_outside_2_to_64_or_for_no_drawn_course_is_refused() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app();
    for (body, needle) in [
        (json!({"scenario": "race", "gates": 1}), "gates must be 2..=64, got 1"),
        (json!({"scenario": "race", "gates": 65}), "gates must be 2..=64, got 65"),
        (json!({"scenario": "solo", "gates": 4}), "gates is for a race"),
        (json!({"scenario": "gated", "gates": 4}), "[[course.gate]]"),
    ] {
        let (status, reply) = start(&app, &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {reply}");
        assert!(reply["error"].as_str().unwrap().contains(needle), "{body}: {reply}");
    }
    let (_, current) = get(&app, "/api/runs/current").await;
    assert_eq!(current["run"], Value::Null, "nothing was started");
    for (uri, needle) in [
        ("/api/scenarios/race/course?gates=1", "2..=64"),
        ("/api/scenarios/race/course?gates=65", "2..=64"),
        ("/api/scenarios/race/course?gates=x", "gates"),
    ] {
        let (status, reply) = get(&app, uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {reply}");
        assert!(reply["error"].as_str().unwrap().contains(needle), "{uri}: {reply}");
    }
    let (status, reply) = get(&app, "/api/scenarios/race/course?gates=64").await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["gates"].as_array().unwrap().len(), 64);
    shutdown(runs).await;
}

#[tokio::test]
async fn a_race_sweeps_gate_count_is_in_its_plan_files_columns_and_saved_form() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, sweeps) = tree.sweep_app();

    let (status, plain) = plan(&app, &sweep_request(json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{plain}");
    assert_eq!(seeds(&plain, "gates"), [4, 4, 4].map(Value::from), "the scenario's own");
    let request = sweep_request(json!({"gates": 6, "course_seed_mode": "randomize"}));
    let (status, six) = plan(&app, &request).await;
    assert_eq!(status, StatusCode::OK, "{six}");
    assert_eq!(seeds(&six, "gates"), [6, 6, 6].map(Value::from));
    let (_, solo) = plan(&app, &sweep_request(json!({"scenario": "solo"}))).await;
    assert_eq!(seeds(&solo, "gates"), [const { Value::Null }; 3]);
    for (extra, status, needle) in [
        (json!({"gates": 1}), 400, "gates must be 2..=64, got 1"),
        (json!({"gates": 65}), 400, "got 65"),
        (json!({"scenario": "solo", "gates": 4}), 400, "gates is for a race"),
        (json!({"scenario": "gated", "gates": 4}), 422, "[[course.gate]]"),
    ] {
        let reply = plan(&app, &sweep_request(extra.clone())).await;
        assert_eq!(reply.0.as_u16(), status, "{extra}: {}", reply.1);
        assert!(reply.1["error"].as_str().unwrap().contains(needle), "{extra}: {}", reply.1);
    }

    // Saved: the file says it and the request reads back with it.
    let (status, _) = send(
        &app,
        "PUT",
        "/api/sweeps/saved/six",
        &json!({"request": request, "summary": "Six gates."}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let text = std::fs::read_to_string(tree.path().join("config/sweep.six.toml")).unwrap();
    assert!(text.contains("gates = 6"), "{text}");
    let (status, loaded) = get(&app, "/api/sweeps/saved/six").await;
    assert_eq!(status, StatusCode::OK, "{loaded}");
    assert_eq!(loaded["request"]["gates"], 6);
    assert_eq!(plan(&app, &loaded["request"]).await.1, six);
    let (_, unset) =
        send(&app, "PUT", "/api/sweeps/saved/four", &json!({"request": sweep_request(json!({}))}))
            .await;
    assert_eq!(unset["name"], "four");
    let (_, loaded) = get(&app, "/api/sweeps/saved/four").await;
    assert!(loaded["request"].get("gates").is_none(), "{loaded}");

    // Flown: every run's file has the count, and the results report it.
    let (status, view) = send(&app, "POST", "/api/sweeps", &request).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    let deadline = Instant::now() + Duration::from_secs(30);
    let done = loop {
        let (_, current) = get(&app, "/api/sweeps/current").await;
        if current["sweep"]["state"] == "finished" {
            break current["sweep"].clone();
        }
        assert!(Instant::now() < deadline, "the sweep never finished: {current}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(done["ok"], 3, "{done}");
    let out = PathBuf::from(done["out"].as_str().unwrap());
    for run in 1..=3 {
        let written: toml::Table = toml::from_str(
            &std::fs::read_to_string(out.join(format!("run-{run:04}.toml"))).unwrap(),
        )
        .unwrap();
        assert_eq!(written["course"]["gates"].as_integer(), Some(6), "run {run}");
    }
    let (_, results) = get(&app, &format!("/api/sweeps/{id}/results")).await;
    assert_eq!(
        &results["columns"].as_array().unwrap()[..5],
        &[json!("run"), json!("status"), json!("seed"), json!("course_seed"), json!("gates")]
    );
    assert!(results["rows"].as_array().unwrap().iter().all(|r| r[4] == 6), "{results}");
    shutdown_sweeps(sweeps).await;
}
