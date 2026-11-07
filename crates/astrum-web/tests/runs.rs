//! Live interactive runs over `/api/runs`: a start checks every assignment
//! (each failure a 400 naming the aircraft or parameter) and starts nothing
//! when one fails; a run flies exactly its controlled aircraft in lockstep,
//! is the only one at a time, logs into a ring buffer read with `since`, and
//! stops on request; its replay and metrics are served once it has finished.
//! The "sim" is mostly a shell script standing in for `astrum-sim` (it copies
//! a prepared bag to `--record` when it ends); one test flies the real one.

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astrum_bag::BagWriter;
use astrum_net::keys;
use astrum_proto::{encode_telemetry, Telemetry, VehicleKind};
use astrum_sim::scenario::Scenario;
use astrum_web::bridge::Bridge;
use astrum_web::catalogs::Catalogs;
use astrum_web::runs::Runs;
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

const CATALOG: &str = r#"
[[controller]]
id = "hold"
name = "Test hold"
description = "Prints its arguments, then idles until stopped."
command = ["sh", "-c", "echo \"ctl $*\"; trap 'exit 0' TERM; while true; do sleep 0.05; done", "ctl"]
id_flag = "--id"
kinds = ["fixed-wing"]

[[controller.param]]
name = "alt"
flag = "--alt"
label = "Altitude"
unit = "m"
type = "number"
default = 1500.0
min = 0.0
max = 10000.0

[[controller]]
id = "follow"
name = "Test follower"
description = "Needs a leader."
command = ["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 0.05; done", "ctl"]
id_flag = "--id"
kinds = ["fixed-wing", "multirotor"]
needs_fleet = true

[[controller.param]]
name = "leader"
flag = "--leader"
label = "Leader"
type = "aircraft"

[[controller]]
id = "quad"
name = "Test quad"
description = "Multirotors only."
command = ["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 0.05; done", "ctl"]
id_flag = "--id"
kinds = ["multirotor"]

[[controller]]
id = "crash"
name = "Test crash"
description = "Fails at once."
command = ["sh", "-c", "exit 3", "ctl"]
id_flag = "--id"
kinds = ["fixed-wing"]

[[controller]]
id = "unbuilt"
name = "Test unbuilt"
description = "Its program was never built."
command = ["build/controllers/cpp/no-such-controller"]
id_flag = "--id"
kinds = ["fixed-wing"]

[[controller]]
id = "unstartable"
name = "Test unstartable"
description = "An executable whose interpreter does not exist."
command = ["@BADINTERP@"]
id_flag = "--id"
kinds = ["fixed-wing"]
"#;

/// A fixed-wing (1) and a multirotor (2); the scenario asks for lockstep on
/// aircraft 2, which a web run overrides.
const DUO: &str = r#"
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_hae_m = 0.0

[sim]
rate_hz = 120.0
max_time_s = 40.0
speed = 1.0
lockstep = true

[[aircraft]]
id = 1
model = "c172x"
lat_deg = 37.6188
lon_deg = -122.3756
alt_hae_m = 1524.0
heading_deg = 90.0
tas_mps = 55.0

[[aircraft]]
id = 2
model = "F450"
lat_deg = 37.6190
lon_deg = -122.3756
alt_hae_m = 100.0
heading_deg = 0.0
tas_mps = 0.0
lockstep = true
"#;

/// A temp tree: `config/` (the test catalog, `scenario.duo.toml` and an
/// invalid `scenario.broken.toml`), a recorded bag to hand out, an executable
/// that cannot start (`badinterp`), and the server's other files.
struct Tree {
    dir: tempfile::TempDir,
}

impl Tree {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        std::fs::create_dir_all(&config).unwrap();
        let badinterp = dir.path().join("badinterp");
        std::fs::write(&badinterp, "#!/no/such/interpreter\n").unwrap();
        std::fs::set_permissions(&badinterp, std::fs::Permissions::from_mode(0o755)).unwrap();
        let catalog = CATALOG.replace("@BADINTERP@", &badinterp.display().to_string());
        std::fs::write(config.join("controllers.toml"), catalog).unwrap();
        std::fs::write(config.join("scenario.duo.toml"), DUO).unwrap();
        std::fs::write(config.join("scenario.broken.toml"), "[sim]\nrate_hz = -1\n").unwrap();
        write_bag(&dir.path().join("fixture.mcap"));
        Self { dir }
    }

    /// Processes still running from this tree's scripts.
    fn leftovers(&self) -> Vec<String> {
        let needle = self.path().display().to_string();
        let mut found = Vec::new();
        for entry in std::fs::read_dir("/proc").unwrap().flatten() {
            let Ok(cmdline) = std::fs::read(entry.path().join("cmdline")) else { continue };
            let cmdline = String::from_utf8_lossy(&cmdline).replace('\0', " ");
            if cmdline.contains(&needle) {
                found.push(cmdline);
            }
        }
        found
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn catalogs(&self) -> Catalogs {
        Catalogs::load(self.path().join("config")).unwrap()
    }

    /// A stand-in sim: prints its arguments and a line every 50 ms; on
    /// SIGTERM, or after `life` (shell `sleep` syntax) if given, it copies
    /// the fixture bag to its `--record` path and exits 0.
    fn sim(&self, life: Option<&str>) -> PathBuf {
        let fixture = self.path().join("fixture.mcap");
        let fixture = fixture.display();
        let body = format!(
            "echo \"args $*\"\n\
             while [ $# -gt 0 ]; do [ \"$1\" = --record ] && bag=\"$2\"; shift; done\n\
             finish() {{ cp '{fixture}' \"$bag\"; echo finished; exit 0; }}\n\
             trap finish TERM\n\
             {}\n",
            match life {
                Some(life) => format!("sleep {life}\nfinish"),
                None => "i=0\nwhile true; do i=$((i+1)); echo \"tick $i\"; sleep 0.05; done".into(),
            }
        );
        let path = self.path().join("sim");
        std::fs::write(&path, format!("#!/bin/sh\n{body}")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn app(&self, sim: &Path) -> (Router, Runs) {
        common::app_with_runs(self.path(), Bridge::new(config()), self.catalogs(), sim)
    }
}

/// One aircraft's telemetry over 0.5 s of sim time.
fn write_bag(path: &Path) {
    let mut w = BagWriter::create(path, Some(DUO)).unwrap();
    for tick in 1..=60u64 {
        let t = Telemetry {
            aircraft_id: 1,
            tick,
            sim_time_s: tick as f64 / 120.0,
            vehicle_kind: VehicleKind::FixedWing,
            alt_hae_m: 1500.0,
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

fn assert_error(reply: &(StatusCode, Value), status: StatusCode, needles: &[&str]) {
    assert_eq!(reply.0, status, "{}", reply.1);
    let error = reply.1["error"].as_str().unwrap_or_else(|| panic!("no error in {}", reply.1));
    for needle in needles {
        assert!(error.contains(needle), "{error:?} should mention {needle:?}");
    }
}

/// Poll the current run's log until `pred` holds for the lines read so far.
async fn wait_for_log(app: &Router, what: &str, pred: impl Fn(&[Value]) -> bool) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let (status, body) = get(app, "/api/runs/current/log").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let lines = body["lines"].as_array().unwrap().clone();
        if pred(&lines) {
            return lines;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}: {lines:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_finished(app: &Router) -> Value {
    wait_state(app, "finished").await
}

async fn shutdown(runs: Runs) {
    tokio::task::spawn_blocking(move || runs.shutdown()).await.unwrap();
}

fn has_line(lines: &[Value], source: &str, prefix: &str) -> bool {
    lines.iter().any(|l| l["source"] == source && l["line"].as_str().unwrap().starts_with(prefix))
}

/// `scenario.duo.toml` as an inline scenario object, edited by `edit`.
fn inline_duo(edit: impl FnOnce(&mut Value)) -> Value {
    let mut duo = serde_json::to_value(toml::from_str::<toml::Value>(DUO).unwrap()).unwrap();
    edit(&mut duo);
    duo
}

async fn wait_state(app: &Router, want: &str) -> Value {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (_, body) = get(app, "/api/runs/current").await;
        if body["run"]["state"] == want {
            return body["run"].clone();
        }
        assert!(Instant::now() < deadline, "the run never was {want}: {body}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn bridge_endpoint(app: &Router) -> Value {
    get(app, "/api/bridge").await.1["endpoint"].clone()
}

#[tokio::test]
async fn a_bad_start_names_the_culprit_and_starts_nothing() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(None));
    let with = |assignments: Value| json!({"scenario": "duo", "assignments": assignments});
    let cases: Vec<(Value, StatusCode, Vec<&str>)> = vec![
        (
            with(json!([{"aircraft": 9, "controller": "hold"}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 9", "not in the scenario"],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "nope"}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "\"nope\""],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "quad"}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "\"quad\"", "fixed-wing"],
        ),
        (
            with(json!([{"aircraft": 2, "controller": "hold"}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 2", "multirotor"],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "hold", "params": {"alt": 20000}}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "`alt`", "outside"],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "hold", "params": {"alt": [1]}}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "`alt`"],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "hold", "params": {"gain": 2}}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "`gain`"],
        ),
        (
            with(json!([{"aircraft": 2, "controller": "follow"}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 2", "`leader`", "required"],
        ),
        (
            with(json!([{"aircraft": 2, "controller": "follow", "params": {"leader": 2}}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 2", "`leader`", "itself"],
        ),
        (
            with(json!([{"aircraft": 2, "controller": "follow", "params": {"leader": 5}}])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 2", "`leader`", "not in the scenario"],
        ),
        (
            with(json!([
                {"aircraft": 1, "controller": "hold"},
                {"aircraft": 1, "controller": "follow", "params": {"leader": 2}}
            ])),
            StatusCode::BAD_REQUEST,
            vec!["aircraft 1", "more than one"],
        ),
        (json!({"scenario": "missing"}), StatusCode::BAD_REQUEST, vec!["\"missing\""]),
        (json!({"scenario": "../duo"}), StatusCode::BAD_REQUEST, vec!["scenario"]),
        (json!({"scenario": 7}), StatusCode::BAD_REQUEST, vec!["scenario"]),
        (json!({"scenario": "broken"}), StatusCode::UNPROCESSABLE_ENTITY, vec!["broken"]),
        (
            json!({"scenario": {"world": {}, "sim": {}, "aircraft": []}}),
            StatusCode::UNPROCESSABLE_ENTITY,
            vec!["inline scenario"],
        ),
        (
            // A model is a directory name under the JSBSim root, never a path.
            json!({"scenario": inline_duo(|duo| {
                let a = &mut duo["aircraft"][0];
                a["model"] = json!("../../../../../tmp/x");
                a["kind"] = json!("fixed-wing");
                a["autopilot_profile"] = json!("trainer");
            })}),
            StatusCode::UNPROCESSABLE_ENTITY,
            vec!["inline scenario", "aircraft 1", "model"],
        ),
        (
            with(json!([{"aircraft": 1, "controller": "unbuilt"}])),
            StatusCode::INTERNAL_SERVER_ERROR,
            vec![
                "aircraft 1",
                "`unbuilt`",
                "build/controllers/cpp/no-such-controller",
                "build the C++ controllers",
            ],
        ),
        (json!({"scenario": "duo", "speed": -1}), StatusCode::BAD_REQUEST, vec!["speed"]),
        (json!({"scenario": "duo", "speed": 1e-9}), StatusCode::BAD_REQUEST, vec!["speed"]),
        (json!({"scenario": "duo", "max_time_s": -5}), StatusCode::BAD_REQUEST, vec!["max_time_s"]),
        (json!({"scenario": "duo", "colour": "red"}), StatusCode::BAD_REQUEST, vec!["colour"]),
    ];
    for (body, status, needles) in &cases {
        let reply = start(&app, body).await;
        assert_error(&reply, *status, needles);
    }
    assert_eq!(get(&app, "/api/runs/current").await, (StatusCode::OK, json!({"run": null})));
    let runs_dir = tree.path().join("runs");
    let left = std::fs::read_dir(&runs_dir).map_or(0, Iterator::count);
    assert_eq!(left, 0, "a refused start left a run directory behind");
    shutdown(runs).await;
}

#[tokio::test]
async fn a_run_flies_its_controlled_aircraft_in_lockstep_alone_until_stopped() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(None));
    let body = json!({
        "scenario": "duo",
        "assignments": [{"aircraft": 1, "controller": "hold"}],
        "speed": 0,
        "max_time_s": 12.5,
    });
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    assert!(id.ends_with("-duo") && astrum_web::api::safe_name(&id).is_ok(), "{id}");
    assert_eq!(view["scenario"], "duo");
    assert!(view["state"] == "starting" || view["state"] == "running", "{view}");
    assert_eq!(view["speed"], 0.0);
    assert_eq!(view["max_time_s"], 12.5);
    assert_eq!(view["lockstep"], true);
    assert_eq!(view["outcome"], Value::Null);
    assert_eq!(view["finished_at_ms"], Value::Null);
    // Defaults are filled in.
    assert_eq!(
        view["assignments"],
        json!([{"aircraft": 1, "controller": "hold", "params": {"alt": 1500.0}}])
    );

    // The sim flies the scenario as written to the run's directory: lockstep
    // for exactly the controlled aircraft, the overrides applied.
    let dir = PathBuf::from(view["dir"].as_str().unwrap());
    assert_eq!(dir, tree.path().join("runs").join(&id));
    let flown = Scenario::load(dir.join("scenario.toml")).unwrap();
    assert!(flown.sim.lockstep);
    let lockstep: Vec<(u32, bool)> = flown.aircraft.iter().map(|a| (a.id, a.lockstep)).collect();
    assert_eq!(lockstep, [(1, true), (2, false)]);
    assert_eq!((flown.sim.speed, flown.sim.max_time_s), (0.0, 12.5));
    assert_eq!(view["bag"], dir.join("run.mcap").display().to_string());
    // The bridge carries the run.
    assert_eq!(bridge_endpoint(&app).await, view["endpoint"]);

    // One run at a time.
    let again = start(&app, &json!({"scenario": "duo"})).await;
    assert_error(&again, StatusCode::CONFLICT, &[&id]);
    for what in ["replay", "metrics"] {
        let reply = get(&app, &format!("/api/runs/{id}/{what}")).await;
        assert_error(&reply, StatusCode::CONFLICT, &[&id]);
    }

    // The sim and the controller (named by its aircraft) log, in order;
    // `since` skips what was read.
    let lines = wait_for_log(&app, "the sim's ticks and the controller", |lines| {
        has_line(lines, "aircraft 1", "ctl --id 1 --alt 1500")
            && lines.iter().filter(|l| l["source"] == "sim").count() >= 4
    })
    .await;
    assert!(has_line(&lines, "sim", "args "), "{lines:?}");
    let seqs: Vec<u64> = lines.iter().map(|l| l["seq"].as_u64().unwrap()).collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{seqs:?}");
    let from = seqs[2];
    let (_, later) = get(&app, &format!("/api/runs/current/log?since={from}")).await;
    let later_seqs: Vec<u64> =
        later["lines"].as_array().unwrap().iter().map(|l| l["seq"].as_u64().unwrap()).collect();
    assert_eq!(later_seqs.first(), Some(&from), "{later}");
    assert!(later_seqs.iter().all(|&s| s >= from));
    let next = later["next"].as_u64().unwrap();
    assert!(next > *later_seqs.last().unwrap());
    let (_, beyond) = get(&app, &format!("/api/runs/current/log?since={}", next + 1000)).await;
    assert_eq!(beyond["lines"], json!([]));
    let bad = get(&app, "/api/runs/current/log?since=soon").await;
    assert_eq!(bad.0, StatusCode::BAD_REQUEST, "{}", bad.1);

    // Stop: the final view, finished and stopped.
    let (status, stopped) = send(&app, "DELETE", "/api/runs/current", &json!({})).await;
    assert_eq!(status, StatusCode::OK, "{stopped}");
    assert_eq!(stopped["id"], id.as_str());
    assert_eq!(stopped["state"], "finished");
    assert_eq!(stopped["outcome"]["status"]["kind"], "stopped", "{stopped}");
    assert!(
        stopped["finished_at_ms"].as_u64().unwrap() >= stopped["started_at_ms"].as_u64().unwrap()
    );
    assert_eq!(get(&app, "/api/runs/current").await.1["run"], stopped);
    let none = send(&app, "DELETE", "/api/runs/current", &json!({})).await;
    assert_error(&none, StatusCode::CONFLICT, &["no run is active"]);
    // Finished, the run no longer holds the bridge: back to `--connect`.
    assert_eq!(bridge_endpoint(&app).await, common::RUNS_CONNECT);

    // Finished: its replay and metrics (written once, to metrics.json).
    let (status, replay) = get(&app, &format!("/api/runs/{id}/replay")).await;
    assert_eq!(status, StatusCode::OK, "{replay}");
    assert_eq!(replay["kind"], "astrum-replay");
    assert_eq!(replay["frames"].as_array().unwrap().len(), 60);
    let (status, metrics) = get(&app, &format!("/api/runs/{id}/metrics")).await;
    assert_eq!(status, StatusCode::OK, "{metrics}");
    assert_eq!(metrics["sim_time_s"], 0.5);
    let written: Value =
        serde_json::from_slice(&std::fs::read(dir.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(written, metrics);

    // Another run may start now.
    let (status, second) = start(&app, &json!({"scenario": "duo"})).await;
    assert_eq!(status, StatusCode::CREATED, "{second}");
    assert_ne!(second["id"], id.as_str());
    shutdown(runs).await;
    assert_eq!(get(&app, "/api/runs/current").await.1["run"]["state"], "finished");
}

#[tokio::test]
async fn an_uncontrolled_inline_run_drops_the_scenarios_lockstep_and_ends_by_itself() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(Some("0.3")));
    let (status, view) = start(&app, &json!({"scenario": inline_duo(|_| {})})).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    assert!(view["id"].as_str().unwrap().ends_with("-inline"), "{view}");
    assert_eq!(view["scenario"], Value::Null);
    assert_eq!(view["lockstep"], false);
    assert_eq!(view["assignments"], json!([]));
    // The scenario's own pace and limit.
    assert_eq!((view["speed"].as_f64(), view["max_time_s"].as_f64()), (Some(1.0), Some(40.0)));
    let dir = PathBuf::from(view["dir"].as_str().unwrap());
    let flown = Scenario::load(dir.join("scenario.toml")).unwrap();
    assert!(!flown.sim.lockstep && flown.aircraft.iter().all(|a| !a.lockstep));

    let finished = wait_finished(&app).await;
    assert_eq!(finished["outcome"]["status"], json!({"kind": "ok"}), "{finished}");
    // Metrics were written when it finished, before anyone asked; the bridge
    // is back on `--connect`.
    assert!(dir.join("metrics.json").is_file());
    assert_eq!(bridge_endpoint(&app).await, common::RUNS_CONNECT);
    shutdown(runs).await;
}

#[tokio::test]
async fn a_failing_controller_is_named_by_its_aircraft() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(None));
    let body = json!({"scenario": "duo", "assignments": [{"aircraft": 1, "controller": "crash"}]});
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let finished = wait_finished(&app).await;
    assert_eq!(
        finished["outcome"]["status"],
        json!({"kind": "failed", "reason": "aircraft 1's controller exited (exit status: 3)"}),
        "{finished}"
    );
    shutdown(runs).await;
}

#[tokio::test]
async fn a_controller_that_cannot_start_fails_the_start_and_leaves_nothing() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(None));
    let body =
        json!({"scenario": "duo", "assignments": [{"aircraft": 1, "controller": "unstartable"}]});
    let reply = start(&app, &body).await;
    assert_error(
        &reply,
        StatusCode::INTERNAL_SERVER_ERROR,
        &["aircraft 1's controller", "badinterp"],
    );
    assert_eq!(get(&app, "/api/runs/current").await.1, json!({"run": null}));
    let left = std::fs::read_dir(tree.path().join("runs")).map_or(0, Iterator::count);
    assert_eq!(left, 0, "the failed start left its run directory");
    assert_eq!(tree.leftovers(), Vec::<String>::new(), "the failed start left processes");
    shutdown(runs).await;
}

#[tokio::test]
async fn replay_and_metrics_need_a_safe_known_id_with_a_recording() {
    let _serial = SERIAL.lock().await;
    let tree = Tree::new();
    let (app, runs) = tree.app(&tree.sim(None));
    let runs_dir = tree.path().join("runs");
    std::fs::create_dir_all(runs_dir.join("unrecorded")).unwrap();
    // A run directory from an earlier server: served like this server's.
    std::fs::create_dir_all(runs_dir.join("earlier")).unwrap();
    std::fs::copy(tree.path().join("fixture.mcap"), runs_dir.join("earlier/run.mcap")).unwrap();
    for what in ["replay", "metrics"] {
        let unknown = get(&app, &format!("/api/runs/nope/{what}")).await;
        assert_error(&unknown, StatusCode::NOT_FOUND, &["nope"]);
        let unrecorded = get(&app, &format!("/api/runs/unrecorded/{what}")).await;
        assert_error(&unrecorded, StatusCode::NOT_FOUND, &["recording"]);
        for unsafe_id in ["a.b", "-x", "%2E%2E"] {
            let reply = get(&app, &format!("/api/runs/{unsafe_id}/{what}")).await;
            assert_error(&reply, StatusCode::BAD_REQUEST, &["run id"]);
        }
        let (status, body) = get(&app, &format!("/api/runs/earlier/{what}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    shutdown(runs).await;
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

#[tokio::test]
async fn the_real_sim_flies_a_saved_scenario_to_its_end() {
    let Some(sim) = real_sim() else {
        assert!(
            std::env::var_os("ASTRUM_SKIP_REAL_SIM").is_some_and(|v| v == "1"),
            "no astrum-sim next to the test binaries: build it first (cargo build -p astrum-sim), \
             set ASTRUM_SIM_BIN, or skip this test with ASTRUM_SKIP_REAL_SIM=1"
        );
        return;
    };
    let _serial = SERIAL.lock().await;
    let tmp = tempfile::tempdir().unwrap();
    let (app, runs) =
        common::app_with_runs(tmp.path(), Bridge::new(config()), common::repo_catalogs(), &sim);
    // Real time, so the run is seen running.
    let body = json!({"scenario": "demo", "speed": 1, "max_time_s": 4});
    let (status, view) = start(&app, &body).await;
    assert_eq!(status, StatusCode::CREATED, "{view}");
    let id = view["id"].as_str().unwrap().to_owned();
    // Running = the bridge, attached to the run's endpoint, reached its sim.
    let running = wait_state(&app, "running").await;
    assert_eq!(running["id"], id.as_str());
    let (_, bridge) = get(&app, "/api/bridge").await;
    assert_eq!(bridge["endpoint"], view["endpoint"], "{bridge}");
    let finished = wait_finished(&app).await;
    assert_eq!(finished["outcome"]["status"], json!({"kind": "ok"}), "{finished}");
    assert_eq!(bridge_endpoint(&app).await, common::RUNS_CONNECT);
    let (status, metrics) = get(&app, &format!("/api/runs/{id}/metrics")).await;
    assert_eq!(status, StatusCode::OK, "{metrics}");
    assert_eq!(metrics["end_reason"], "max-time");
    assert!((metrics["sim_time_s"].as_f64().unwrap() - 4.0).abs() < 0.05, "{metrics}");
    let (status, replay) = get(&app, &format!("/api/runs/{id}/replay")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!replay["frames"].as_array().unwrap().is_empty());
    shutdown(runs).await;
}
