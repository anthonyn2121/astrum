//! The studio's sweeps flown from the browser (`/api/sweeps`): one at a time,
//! owned by the server, the way the live interactive run is ([`crate::runs`]).
//!
//! A start resolves the request ([`crate::sweeps::resolve`]) and plans it with
//! the library. Nothing is flown or written before it has passed the refusals
//! `astrum sweep` makes: no `astrum-sim` for a process-mode sweep, too few
//! ports for its runs, and a trim check that fails (always, unlike the
//! request's `check_trim`). Then `<dir>/<id>/` is created, where `<id>` is
//! `<UTC stamp>-<scenario>` (`-2`, `-3`, … when taken), and holds
//! `request.json` (the request), `sweep.toml` (the sweep file `astrum sweep`
//! reads), `meta.json` (the sweep's state, rewritten when it ends) and what the
//! sweep writes: `summary.csv`, and per run `run-NNNN.mcap` and
//! `run-NNNN.metrics.json`. A supervisor thread runs [`execute`] exactly as the
//! CLI does and keeps the sweep's per-run state in memory for the API.
//!
//! Every sweep directory under the sweeps dir (this server's or an earlier
//! one's) serves its results and its runs' replays. A sweep whose `meta.json`
//! says `running` but which this server does not own was interrupted: it is
//! reported as `interrupted`, and its unfinished runs as `stopped`.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use astrum_sim::launch::{run_processes, Children, ProcessCtx};
use astrum_sim::sweep::{execute, run_in_process, Purpose, RunEvent, Status, Sweep};
use astrum_sim::systems::StopRequested;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::watch;

use crate::api::{error, json};
use crate::catalogs::Catalogs;
use crate::export::export_bag;
use crate::runs::{new_run_dir, now_ms, utc_stamp};
use crate::server::AppState;
use crate::sweeps::{resolve, SweepInputError, SweepRequest};
use crate::ExportDoc;

/// Longest sweep id, in characters.
const ID_MAX_LEN: usize = 128;
/// The request a sweep was started with, inside its directory.
const REQUEST_FILE: &str = "request.json";
/// The sweep file `astrum sweep` reads, inside its directory.
const SWEEP_FILE: &str = "sweep.toml";
/// The sweep's state ([`Meta`]), inside its directory.
const META_FILE: &str = "meta.json";
/// One row per run, rewritten as runs start and finish (`execute`'s).
const SUMMARY_FILE: &str = "summary.csv";
/// Why a sweep whose server went away is reported as `interrupted`.
const INTERRUPTED: &str = "the server stopped before the sweep ended";

/// Where sweeps go, what flies them, and the first port process-mode runs
/// listen on.
#[derive(Debug, Clone)]
pub struct SweepRunsConfig {
    /// Parent of every sweep's directory (`<runs-dir>/sweeps`).
    pub dir: PathBuf,
    /// The `astrum-sim` binary process-mode runs fly on.
    pub sim: PathBuf,
    /// Run N (1-based) of a process-mode sweep listens on
    /// `tcp/127.0.0.1:<base_port + N - 1>`.
    pub base_port: u16,
}

/// A sweep's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SweepState {
    /// Its supervisor is still flying runs.
    Running,
    /// Every run was tried and `execute` returned without a stop request.
    Finished,
    /// Stopped by a stop request; the runs never started stay queued.
    Stopped,
    /// `execute` could not write the results; the reason says why.
    Failed,
    /// Its `meta.json` says running, but no server owns it any more.
    Interrupted,
}

/// A run's status in a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryStatus {
    Queued,
    Running,
    Ok,
    Failed,
    Stopped,
}

/// One run of a sweep as the API reports it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunEntry {
    /// 1-based.
    pub run: usize,
    pub status: EntryStatus,
    /// Why the run failed or stopped.
    pub reason: Option<String>,
    pub sim_time_s: Option<f64>,
    pub wall_s: Option<f64>,
}

impl RunEntry {
    fn queued(run: usize) -> Self {
        Self { run, status: EntryStatus::Queued, reason: None, sim_time_s: None, wall_s: None }
    }
}

/// A sweep's `meta.json`: written at start, rewritten when the sweep ends.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    id: String,
    scenario: String,
    state: SweepState,
    reason: Option<String>,
    total: usize,
    in_process: bool,
    jobs: usize,
    /// `yyyy-mm-ddThh:mm:ssZ` (UTC).
    started_at: String,
    ended_at: Option<String>,
    wall_s: Option<f64>,
    /// The varied and drawn keys: the summary's columns after `sample`.
    keys: Vec<String>,
}

/// A sweep as the API reports it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepView {
    pub id: String,
    pub scenario: String,
    pub state: SweepState,
    pub reason: Option<String>,
    pub total: usize,
    pub queued: usize,
    pub running: usize,
    pub ok: usize,
    pub failed: usize,
    pub stopped: usize,
    pub in_process: bool,
    pub jobs: usize,
    pub started_at: String,
    /// Seconds from start to its end (to now, while it runs).
    pub wall_s: Option<f64>,
    /// The sweep's directory, as the server reaches it.
    pub out: String,
    pub keys: Vec<String>,
    pub runs: Vec<RunEntry>,
}

impl SweepView {
    fn new(meta: Meta, out: &Path, runs: Vec<RunEntry>) -> Self {
        let count = |status: EntryStatus| runs.iter().filter(|r| r.status == status).count();
        Self {
            queued: count(EntryStatus::Queued),
            running: count(EntryStatus::Running),
            ok: count(EntryStatus::Ok),
            failed: count(EntryStatus::Failed),
            stopped: count(EntryStatus::Stopped),
            id: meta.id,
            scenario: meta.scenario,
            state: meta.state,
            reason: meta.reason,
            total: meta.total,
            in_process: meta.in_process,
            jobs: meta.jobs,
            started_at: meta.started_at,
            wall_s: meta.wall_s,
            out: out.display().to_string(),
            keys: meta.keys,
            runs,
        }
    }
}

/// One line of `GET /api/sweeps`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepSummary {
    pub id: String,
    pub scenario: String,
    pub state: SweepState,
    pub total: usize,
    pub ok: usize,
    pub failed: usize,
    pub started_at: String,
    pub wall_s: Option<f64>,
}

impl SweepSummary {
    fn of(view: &SweepView) -> Self {
        Self {
            id: view.id.clone(),
            scenario: view.scenario.clone(),
            state: view.state,
            total: view.total,
            ok: view.ok,
            failed: view.failed,
            started_at: view.started_at.clone(),
            wall_s: view.wall_s,
        }
    }
}

/// Why a request fails: its status and the message the reply carries.
#[derive(Debug)]
struct Refusal(StatusCode, String);

impl Refusal {
    fn respond(self) -> Response {
        error(self.0, &self.1)
    }
}

fn refuse(status: StatusCode, msg: impl Into<String>) -> Refusal {
    Refusal(status, msg.into())
}

impl From<SweepInputError> for Refusal {
    fn from(e: SweepInputError) -> Self {
        match e {
            SweepInputError::BadRequest(msg) => refuse(StatusCode::BAD_REQUEST, msg),
            SweepInputError::Invalid(msg) => refuse(StatusCode::UNPROCESSABLE_ENTITY, msg),
            SweepInputError::Io(msg) => refuse(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    }
}

/// The server's sweep runner. Cheap to clone; clones share it.
#[derive(Clone)]
pub struct SweepRuns(Arc<Inner>);

impl fmt::Debug for SweepRuns {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SweepRuns").field("config", &self.0.cfg).finish_non_exhaustive()
    }
}

struct Inner {
    cfg: SweepRunsConfig,
    children: Children,
    state: Mutex<SweepsState>,
}

#[derive(Default)]
struct SweepsState {
    /// The latest sweep since the server started.
    current: Option<Record>,
    /// A sweep is being started (it has no record yet).
    starting: bool,
    /// Shutting down: no new sweeps.
    closed: bool,
}

/// The server's record of one sweep.
struct Record {
    /// The meta as written at start (its id, scenario, plan).
    base: Meta,
    started_ms: u64,
    out: PathBuf,
    /// Set when the supervisor is done.
    ended: Option<Ended>,
    runs: Vec<RunEntry>,
    stop: StopRequested,
    /// Turns `true` once the supervisor has ended.
    done: watch::Receiver<bool>,
    supervisor: Option<JoinHandle<()>>,
}

/// How a sweep ended.
struct Ended {
    state: SweepState,
    reason: Option<String>,
    at_ms: u64,
}

impl Record {
    fn active(&self) -> bool {
        self.ended.is_none()
    }

    /// The meta as of now, or as the sweep ends (`ended`).
    fn meta(&self, ended: Option<&Ended>) -> Meta {
        let mut meta = self.base.clone();
        match ended {
            Some(e) => {
                meta.state = e.state;
                meta.reason = e.reason.clone();
                meta.ended_at = Some(iso_utc(e.at_ms));
                meta.wall_s = Some(secs(self.started_ms, e.at_ms));
            }
            None => meta.wall_s = Some(secs(self.started_ms, now_ms())),
        }
        meta
    }

    fn view(&self) -> SweepView {
        SweepView::new(self.meta(self.ended.as_ref()), &self.out, self.runs.clone())
    }
}

/// What the supervisor needs to fly one sweep on its own thread.
struct Job {
    sweeps: SweepRuns,
    id: String,
    dir: PathBuf,
    sweep: Sweep,
    stop: StopRequested,
    children: Children,
    sim: PathBuf,
    base_port: u16,
    done: watch::Sender<bool>,
}

impl Job {
    /// Fly every run as `astrum sweep` does, then record how the sweep ended.
    fn run(self) {
        let Self { sweeps, id, dir, sweep, stop, children, sim, base_port, done } = self;
        let on_event = |event: RunEvent| sweeps.record(&id, event);
        let jobs = sweep.jobs(None);
        let outcome = if sweep.in_process() {
            execute(&sweep, jobs, &stop, on_event, |run| {
                run_in_process(run, &dir.join(format!("{}.mcap", run.stem())), &stop)
            })
        } else {
            let ctx = ProcessCtx {
                sweep: &sweep,
                out: &dir,
                sim: &sim,
                base_port,
                stop: &stop,
                children: &children,
            };
            execute(&sweep, jobs, &stop, on_event, |run| run_processes(&ctx, run))
        };
        let (state, reason) = match outcome {
            Ok(_) if stop.requested() => (SweepState::Stopped, None),
            Ok(_) => (SweepState::Finished, None),
            Err(e) => {
                (SweepState::Failed, Some(format!("writing the results to {}: {e}", dir.display())))
            }
        };
        sweeps.end(&id, &dir, Ended { state, reason, at_ms: now_ms() }, &done);
    }
}

impl SweepRuns {
    pub fn new(cfg: SweepRunsConfig) -> Self {
        Self(Arc::new(Inner {
            cfg,
            children: Children::new(),
            state: Mutex::new(SweepsState::default()),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, SweepsState> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The latest sweep, its state brought up to date.
    pub fn current(&self) -> Option<SweepView> {
        self.lock().current.as_ref().map(Record::view)
    }

    /// The latest sweep, if it is `id`.
    fn live(&self, id: &str) -> Option<SweepView> {
        self.lock().current.as_ref().filter(|r| r.base.id == id).map(Record::view)
    }

    /// Ask the active sweep (if any) to stop and refuse new ones: the start of
    /// a server shutdown. Never blocks.
    pub fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        if let Some(record) = state.current.as_ref().filter(|r| r.active()) {
            record.stop.0.store(true, Ordering::Relaxed);
        }
    }

    /// [`close`](Self::close), then block until the active sweep (and one
    /// being started) has ended.
    pub fn shutdown(&self) {
        self.close();
        while self.lock().starting {
            std::thread::sleep(Duration::from_millis(20));
        }
        let supervisor = self.lock().current.as_mut().and_then(|r| r.supervisor.take());
        if let Some(supervisor) = supervisor {
            let _ = supervisor.join();
        }
    }

    /// Check the request, plan it and start the sweep. Blocks (file reads,
    /// JSBSim trims, directory writes): call it on a blocking thread.
    fn start_blocking(&self, req: SweepRequest, catalogs: &Catalogs) -> Result<SweepView, Refusal> {
        let resolved = resolve(&req, catalogs)?;
        let sweep =
            Sweep::plan_scenario(resolved.file, resolved.scenario, &req.scenario, Purpose::Sweep)
                .map_err(|e| refuse(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
        {
            let mut state = self.lock();
            if state.closed {
                return Err(refuse(StatusCode::SERVICE_UNAVAILABLE, "the server is shutting down"));
            }
            if state.starting {
                return Err(refuse(StatusCode::CONFLICT, "another sweep is being started"));
            }
            if let Some(active) = state.current.as_ref().filter(|r| r.active()) {
                return Err(refuse(
                    StatusCode::CONFLICT,
                    format!("a sweep is already running: {}", active.base.id),
                ));
            }
            state.starting = true;
        }
        let started = self.launch(&req, sweep);
        self.lock().starting = false;
        started
    }

    fn launch(&self, req: &SweepRequest, mut sweep: Sweep) -> Result<SweepView, Refusal> {
        let cfg = &self.0.cfg;
        if !sweep.in_process() {
            if !cfg.sim.exists() {
                return Err(refuse(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "no astrum-sim at {} (build it: cargo build -p astrum-sim)",
                        cfg.sim.display()
                    ),
                ));
            }
            if usize::from(cfg.base_port) + sweep.runs.len() > usize::from(u16::MAX) + 1 {
                return Err(refuse(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "base port {} leaves too few ports for {} runs",
                        cfg.base_port,
                        sweep.runs.len()
                    ),
                ));
            }
        }
        sweep.check_trim().map_err(|e| refuse(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;

        std::fs::create_dir_all(&cfg.dir).map_err(|e| {
            refuse(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("creating {}: {e}", cfg.dir.display()),
            )
        })?;
        let started_ms = now_ms();
        let (id, dir) = new_run_dir(&cfg.dir, &utc_stamp(started_ms), &req.scenario)
            .map_err(|e| refuse(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        sweep.out = dir.clone();
        let base = Meta {
            id: id.clone(),
            scenario: req.scenario.clone(),
            state: SweepState::Running,
            reason: None,
            total: sweep.runs.len(),
            in_process: sweep.in_process(),
            jobs: sweep.jobs(None),
            started_at: iso_utc(started_ms),
            ended_at: None,
            wall_s: None,
            keys: sweep.value_keys.clone(),
        };
        if let Err(why) = write_start(&dir, req, &sweep, &base) {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(refuse(StatusCode::INTERNAL_SERVER_ERROR, why));
        }

        let stop = StopRequested::default();
        let (done_tx, done) = watch::channel(false);
        // The supervisor waits for the record below, so its first event finds it.
        let (installed_tx, installed) = mpsc::channel::<()>();
        let runs = (1..=sweep.runs.len()).map(RunEntry::queued).collect();
        let job = Job {
            sweeps: self.clone(),
            id: id.clone(),
            dir: dir.clone(),
            sweep,
            stop: stop.clone(),
            children: self.0.children.clone(),
            sim: cfg.sim.clone(),
            base_port: cfg.base_port,
            done: done_tx,
        };
        let spawned = std::thread::Builder::new().name(format!("sweep-{id}")).spawn(move || {
            if installed.recv().is_ok() {
                job.run();
            }
        });
        let supervisor = match spawned {
            Ok(handle) => handle,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(refuse(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("starting the sweep: {e}"),
                ));
            }
        };
        let record = Record {
            base,
            started_ms,
            out: dir,
            ended: None,
            runs,
            stop,
            done,
            supervisor: Some(supervisor),
        };
        let view = {
            let mut state = self.lock();
            if state.closed {
                record.stop.0.store(true, Ordering::Relaxed);
            }
            let view = record.view();
            state.current = Some(record);
            view
        };
        let _ = installed_tx.send(());
        Ok(view)
    }

    fn record(&self, id: &str, event: RunEvent) {
        let mut state = self.lock();
        let Some(record) = state.current.as_mut().filter(|r| r.base.id == id) else { return };
        match event {
            RunEvent::Started { index } => {
                if let Some(run) = record.runs.get_mut(index) {
                    run.status = EntryStatus::Running;
                }
            }
            RunEvent::Finished { index, outcome, .. } => {
                let Some(run) = record.runs.get_mut(index) else { return };
                let (status, reason) = match outcome.status {
                    Status::Ok => (EntryStatus::Ok, None),
                    Status::Failed(why) => (EntryStatus::Failed, Some(why)),
                    Status::Stopped(why) => (EntryStatus::Stopped, Some(why)),
                };
                run.status = status;
                run.reason = reason;
                run.sim_time_s = outcome.sim_time_s;
                run.wall_s = Some(outcome.wall_s);
            }
        }
    }

    /// The supervisor is done: write the sweep's meta, then mark the sweep
    /// ended, so whoever sees it ended also sees its meta on disk.
    fn end(&self, id: &str, dir: &Path, mut ended: Ended, done: &watch::Sender<bool>) {
        let meta =
            self.lock().current.as_ref().filter(|r| r.base.id == id).map(|r| r.meta(Some(&ended)));
        if let Some(meta) = meta {
            if let Err(why) = write_meta(dir, &meta) {
                ended = Ended { state: SweepState::Failed, reason: Some(why), at_ms: ended.at_ms };
            }
        }
        if let Some(record) = self.lock().current.as_mut().filter(|r| r.base.id == id) {
            record.ended = Some(ended);
        }
        done.send_replace(true);
    }

    /// Ask the active sweep to stop and wait until its supervisor has ended.
    async fn stop_current(&self) -> Result<SweepView, Refusal> {
        let mut done = {
            let state = self.lock();
            let Some(record) = state.current.as_ref().filter(|r| r.active()) else {
                return Err(refuse(StatusCode::NOT_FOUND, "no sweep is running"));
            };
            record.stop.0.store(true, Ordering::Relaxed);
            record.done.clone()
        };
        // A supervisor that died (its sender dropped) leaves the sweep as it is.
        let _ = done.wait_for(|ended| *ended).await;
        self.current()
            .ok_or_else(|| refuse(StatusCode::INTERNAL_SERVER_ERROR, "the sweep vanished"))
    }

    /// The directory of the sweep `id`, which must have a `meta.json`.
    fn dir_of(&self, id: &str) -> Result<PathBuf, Refusal> {
        check_id(id)?;
        let dir = self.0.cfg.dir.join(id);
        if !dir.join(META_FILE).is_file() {
            return Err(refuse(StatusCode::NOT_FOUND, format!("no sweep named {id:?}")));
        }
        Ok(dir)
    }

    /// The sweep `id` as the API reports it: the live one, else from disk.
    fn view_of(&self, id: &str) -> Result<SweepView, Refusal> {
        let dir = self.dir_of(id)?;
        if let Some(view) = self.live(id) {
            return Ok(view);
        }
        disk_view(&dir, read_meta(&dir)?)
    }

    /// Every sweep under the sweeps dir, newest first. Blocks.
    fn list(&self) -> Result<Vec<SweepSummary>, Refusal> {
        let dir = &self.0.cfg.dir;
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(refuse(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("listing {}: {e}", dir.display()),
                ))
            }
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| {
                refuse(StatusCode::INTERNAL_SERVER_ERROR, format!("listing {}: {e}", dir.display()))
            })?;
            let Ok(name) = entry.file_name().into_string() else { continue };
            if check_id(&name).is_ok() && entry.path().join(META_FILE).is_file() {
                ids.push(name);
            }
        }
        // Ids start with their UTC stamp, so reverse name order is newest first.
        ids.sort_unstable_by(|a, b| b.cmp(a));
        ids.iter().map(|id| self.view_of(id).map(|view| SweepSummary::of(&view))).collect()
    }

    /// The sweep's `summary.csv`, parsed. Blocks.
    fn results_of(&self, id: &str) -> Result<Results, Refusal> {
        let dir = self.dir_of(id)?;
        let path = dir.join(SUMMARY_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Results::parse(&text)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                Err(refuse(StatusCode::NOT_FOUND, format!("sweep {id} has no results yet")))
            }
            Err(e) => Err(refuse(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reading {}: {e}", path.display()),
            )),
        }
    }

    /// Run `n` (1-based, as text) of the sweep `id`: its bag as an export
    /// document. Blocks.
    fn replay_of(&self, id: &str, n: &str) -> Result<ExportDoc, Refusal> {
        let dir = self.dir_of(id)?;
        let total = read_meta(&dir)?.total;
        let run =
            n.parse::<usize>().ok().filter(|run| (1..=total).contains(run)).ok_or_else(|| {
                refuse(
                    StatusCode::BAD_REQUEST,
                    format!("run {n:?} of sweep {id}: want a run number from 1 to {total}"),
                )
            })?;
        let running = self
            .lock()
            .current
            .as_ref()
            .filter(|r| r.base.id == id)
            .is_some_and(|r| r.runs[run - 1].status == EntryStatus::Running);
        if running {
            return Err(refuse(
                StatusCode::CONFLICT,
                format!("run {run} of sweep {id} is still running"),
            ));
        }
        let bag = dir.join(format!("run-{run:04}.mcap"));
        if !bag.is_file() {
            return Err(refuse(
                StatusCode::NOT_FOUND,
                format!("run {run} of sweep {id} has no bag"),
            ));
        }
        export_bag(&bag).map_err(|e| {
            refuse(StatusCode::UNPROCESSABLE_ENTITY, format!("{}: {e}", bag.display()))
        })
    }
}

// --- Files and views ------------------------------------------------------------

/// Check a sweep id as a single safe path segment: 1–128 characters of ASCII
/// letters, digits, `.`, `_` or `-`, not starting with `.` (so no `..`).
fn check_id(id: &str) -> Result<(), Refusal> {
    let ok = !id.is_empty()
        && id.len() <= ID_MAX_LEN
        && !id.starts_with('.')
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(refuse(
            StatusCode::BAD_REQUEST,
            format!(
                "sweep id {id:?} must be 1-{ID_MAX_LEN} characters of A-Z, a-z, 0-9, '.', '_' or \
                 '-', not starting with '.'"
            ),
        ))
    }
}

/// `yyyy-mm-ddThh:mm:ssZ` (UTC) of a Unix time in ms, from its stamp.
fn iso_utc(ms: u64) -> String {
    let s = utc_stamp(ms);
    format!("{}-{}-{}T{}:{}:{}Z", &s[0..4], &s[4..6], &s[6..8], &s[9..11], &s[11..13], &s[13..15])
}

/// Seconds from `from_ms` to `to_ms`.
fn secs(from_ms: u64, to_ms: u64) -> f64 {
    to_ms.saturating_sub(from_ms) as f64 / 1000.0
}

/// Write `bytes` to `path` through a temporary file beside it, so a reader
/// sees the old file or the new one.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let name = path.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    std::fs::write(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

fn write_meta(dir: &Path, meta: &Meta) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(meta).map_err(|e| format!("writing {META_FILE}: {e}"))?;
    write_atomic(&dir.join(META_FILE), &bytes)
}

/// The files a started sweep begins with: its request, its sweep file and meta.
fn write_start(dir: &Path, req: &SweepRequest, sweep: &Sweep, meta: &Meta) -> Result<(), String> {
    let request =
        serde_json::to_vec_pretty(req).map_err(|e| format!("writing {REQUEST_FILE}: {e}"))?;
    write_atomic(&dir.join(REQUEST_FILE), &request)?;
    let file = sweep.file.to_toml().map_err(|e| format!("writing {SWEEP_FILE}: {e}"))?;
    write_atomic(&dir.join(SWEEP_FILE), file.as_bytes())?;
    write_meta(dir, meta)
}

fn read_meta(dir: &Path) -> Result<Meta, Refusal> {
    let path = dir.join(META_FILE);
    let text = std::fs::read(&path).map_err(|e| {
        refuse(StatusCode::INTERNAL_SERVER_ERROR, format!("reading {}: {e}", path.display()))
    })?;
    serde_json::from_slice(&text).map_err(|e| {
        refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{}: not a sweep's meta: {e}", path.display()),
        )
    })
}

/// A sweep read from its directory, not owned by a running server: a
/// `running` one was interrupted.
fn disk_view(dir: &Path, mut meta: Meta) -> Result<SweepView, Refusal> {
    if meta.state == SweepState::Running {
        meta.state = SweepState::Interrupted;
        meta.reason = Some(INTERRUPTED.to_owned());
    }
    let runs = read_runs(dir, meta.total)?;
    Ok(SweepView::new(meta, dir, runs))
}

/// The runs of a sweep from its `summary.csv` (all queued without one).
fn read_runs(dir: &Path, total: usize) -> Result<Vec<RunEntry>, Refusal> {
    let mut runs: Vec<RunEntry> = (1..=total).map(RunEntry::queued).collect();
    let path = dir.join(SUMMARY_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(runs),
        Err(e) => {
            return Err(refuse(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reading {}: {e}", path.display()),
            ))
        }
    };
    let rows = parse_csv(&text);
    let Some((header, body)) = rows.split_first() else { return Ok(runs) };
    let column = |name: &str| header.iter().position(|h| h == name);
    let (status_col, sim_col, wall_col) =
        (column("status"), column("sim_time_s"), column("wall_s"));
    for (run, row) in runs.iter_mut().zip(body) {
        let cell = |col: Option<usize>| col.and_then(|c| row.get(c)).map_or("", String::as_str);
        let (status, reason) = parse_status(cell(status_col));
        run.status = status;
        run.reason = reason;
        run.sim_time_s = cell(sim_col).parse().ok();
        run.wall_s = cell(wall_col).parse().ok();
    }
    Ok(runs)
}

/// A summary's `status` cell as a run's status and reason. A run that was
/// still `running` when its server went away is `stopped`.
fn parse_status(cell: &str) -> (EntryStatus, Option<String>) {
    if let Some(why) = cell.strip_prefix("failed: ") {
        return (EntryStatus::Failed, Some(why.to_owned()));
    }
    if let Some(why) = cell.strip_prefix("stopped: ") {
        return (EntryStatus::Stopped, Some(why.to_owned()));
    }
    match cell {
        "" | "not run" => (EntryStatus::Queued, None),
        "ok" => (EntryStatus::Ok, None),
        "running" => (EntryStatus::Stopped, Some(INTERRUPTED.to_owned())),
        other => (EntryStatus::Failed, Some(format!("unknown status {other:?}"))),
    }
}

/// Parse RFC 4180 CSV: a `""` inside quotes is a quote, and commas and
/// newlines inside quotes are cell text. LF or CRLF ends a row; blank lines
/// are skipped. Nothing is rejected: an unclosed quote runs to the end.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    cell.push('"');
                }
                '"' => quoted = false,
                _ => cell.push(c),
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => row.push(std::mem::take(&mut cell)),
            '\n' => {
                if !row.is_empty() || !cell.is_empty() {
                    row.push(std::mem::take(&mut cell));
                    rows.push(std::mem::take(&mut row));
                }
            }
            '\r' => {}
            _ => cell.push(c),
        }
    }
    if !row.is_empty() || !cell.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    rows
}

/// A summary cell as JSON: an integer or a finite real as a number, an empty
/// cell as null, anything else as its text.
fn cell_value(cell: &str) -> Value {
    if cell.is_empty() {
        return Value::Null;
    }
    if let Ok(i) = cell.parse::<i64>() {
        return Value::from(i);
    }
    match cell.parse::<f64>() {
        Ok(f) if f.is_finite() => Value::from(f),
        _ => Value::String(cell.to_owned()),
    }
}

/// A `GET /api/sweeps/<id>/results` reply.
#[derive(Serialize)]
struct Results {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
}

impl Results {
    fn parse(text: &str) -> Self {
        let mut rows = parse_csv(text).into_iter();
        let columns = rows.next().unwrap_or_default();
        let rows = rows.map(|row| row.iter().map(|c| cell_value(c)).collect()).collect();
        Self { columns, rows }
    }
}

// --- HTTP ---------------------------------------------------------------------

#[derive(Serialize)]
struct CurrentBody {
    sweep: Option<SweepView>,
}

fn parse_body(body: Result<Bytes, BytesRejection>) -> Result<SweepRequest, Refusal> {
    let body = body.map_err(|rejection| refuse(rejection.status(), rejection.body_text()))?;
    serde_json::from_slice(&body)
        .map_err(|e| refuse(StatusCode::BAD_REQUEST, format!("bad sweep body: {e}")))
}

/// `POST /api/sweeps`.
pub(crate) async fn start(
    State(state): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let req = match parse_body(body) {
        Ok(req) => req,
        Err(refusal) => return refusal.respond(),
    };
    let (sweeps, catalogs) = (state.sweeps.clone(), state.catalogs.clone());
    match tokio::task::spawn_blocking(move || sweeps.start_blocking(req, &catalogs)).await {
        Ok(Ok(view)) => json(StatusCode::CREATED, &view),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("starting the sweep: {e}")),
    }
}

/// `GET /api/sweeps/current`.
pub(crate) async fn current(State(state): State<AppState>) -> Response {
    json(StatusCode::OK, &CurrentBody { sweep: state.sweeps.current() })
}

/// `DELETE /api/sweeps/current`.
pub(crate) async fn stop(State(state): State<AppState>) -> Response {
    match state.sweeps.stop_current().await {
        Ok(view) => json(StatusCode::OK, &view),
        Err(refusal) => refusal.respond(),
    }
}

/// `GET /api/sweeps`.
pub(crate) async fn list(State(state): State<AppState>) -> Response {
    let sweeps = state.sweeps.clone();
    match tokio::task::spawn_blocking(move || sweeps.list()).await {
        Ok(Ok(sweeps)) => json(StatusCode::OK, &sweeps),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("listing sweeps: {e}")),
    }
}

/// `GET /api/sweeps/<id>`.
pub(crate) async fn view(State(state): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let sweeps = state.sweeps.clone();
    match tokio::task::spawn_blocking(move || sweeps.view_of(&id)).await {
        Ok(Ok(view)) => json(StatusCode::OK, &view),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("reading the sweep: {e}")),
    }
}

/// `GET /api/sweeps/<id>/results`.
pub(crate) async fn results(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
) -> Response {
    let sweeps = state.sweeps.clone();
    match tokio::task::spawn_blocking(move || sweeps.results_of(&id)).await {
        Ok(Ok(results)) => json(StatusCode::OK, &results),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("reading the results: {e}")),
    }
}

/// `GET /api/sweeps/<id>/runs/<n>/replay`: run `n`'s bag as an export document.
pub(crate) async fn replay(
    State(state): State<AppState>,
    UrlPath((id, n)): UrlPath<(String, String)>,
) -> Response {
    let sweeps = state.sweeps.clone();
    match tokio::task::spawn_blocking(move || sweeps.replay_of(&id, &n)).await {
        Ok(Ok(doc)) => json(StatusCode::OK, &doc),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("exporting the run: {e}")),
    }
}
