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
//! one's) serves its results and its runs' replays; the list reads each sweep's
//! `meta.json` only, and skips a sweep whose meta (or, for an interrupted one,
//! summary) cannot be read. A sweep whose `meta.json` says `running` but which
//! this server does not own was interrupted: it is reported as `interrupted`,
//! and its unfinished runs as `stopped`.
//!
//! `GET …/tracks` serves the flight tracks of each finished run ([`astrum_bag::tracks`]), cached beside
//! its bag as `run-NNNN.tracks-<points>.json`: rebuilt when the bag is newer than the cache or the
//! cache does not read.
//!
//! A stop request ends the sweep's processes as it asks them to end: once
//! [`STOP_GRACE`] has passed while the sweep is still active, its processes are
//! killed, and again every second, until the supervisor has ended.

use std::fmt;
use std::io;
use std::ops::RangeInclusive;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use astrum_bag::tracks::{tracks as bag_tracks, AircraftTrack};
use astrum_bag::BagReader;
use astrum_sim::launch::{run_processes, Children, ProcessCtx};
use astrum_sim::sweep::{execute, run_in_process, Purpose, RunEvent, Status, Sweep};
use astrum_sim::systems::StopRequested;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Path as UrlPath, Query, State};
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

/// How long a stop request gets before the sweep's processes are killed.
pub const STOP_GRACE: Duration = Duration::from_secs(5);
/// How often a stopped sweep's processes are killed again until it has ended.
const KILL_EVERY: Duration = Duration::from_secs(1);
/// How often the escalation checks whether the sweep has ended.
const TICK: Duration = Duration::from_millis(10);

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
    /// How long a stop request waits before the sweep's processes are killed.
    pub stop_grace: Duration,
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
    /// Runs ok and failed, written when the sweep ends: `None` while it runs and
    /// in a meta written before they were kept.
    #[serde(default)]
    ok: Option<usize>,
    #[serde(default)]
    failed: Option<usize>,
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
    /// A sweep is being started: its id, once its directory exists.
    starting: Option<Option<String>>,
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
                meta.ok = Some(self.runs.iter().filter(|r| r.status == EntryStatus::Ok).count());
                meta.failed =
                    Some(self.runs.iter().filter(|r| r.status == EntryStatus::Failed).count());
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
    /// Fly every run as `astrum sweep` does, then record how the sweep ended. A
    /// panic while flying ends it `failed` all the same, so it never stays
    /// `running` and blocks the next start.
    fn run(self) {
        let Self { sweeps, id, dir, sweep, stop, children, sim, base_port, done } = self;
        let flown = catch_unwind(AssertUnwindSafe(|| {
            let on_event = |event: RunEvent| sweeps.record(&id, event);
            let jobs = sweep.jobs(None);
            if sweep.in_process() {
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
            }
        }));
        let (state, reason) = match flown {
            Ok(Ok(outcomes)) => {
                // A run that never started or ended stopped makes the sweep
                // stopped; a stop that came after the last run leaves it finished.
                let cut_short = outcomes
                    .iter()
                    .any(|o| o.as_ref().is_none_or(|o| matches!(o.status, Status::Stopped(_))));
                (if cut_short { SweepState::Stopped } else { SweepState::Finished }, None)
            }
            Ok(Err(e)) => {
                (SweepState::Failed, Some(format!("writing the results to {}: {e}", dir.display())))
            }
            Err(_) => (SweepState::Failed, Some("the sweep's supervisor panicked".to_owned())),
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
        while self.lock().starting.is_some() {
            std::thread::sleep(Duration::from_millis(20));
        }
        let supervisor = self.lock().current.as_mut().and_then(|r| r.supervisor.take());
        if let Some(supervisor) = supervisor {
            let _ = supervisor.join();
        }
    }

    /// Check the request, plan it and start the sweep. Blocks (file reads,
    /// JSBSim trims, directory writes): call it on a blocking thread. A
    /// conflicting request is refused before anything is planned.
    fn start_blocking(&self, req: SweepRequest, catalogs: &Catalogs) -> Result<SweepView, Refusal> {
        admit(&self.lock())?;
        let resolved = resolve(&req, catalogs)?;
        let sweep =
            Sweep::plan_scenario(resolved.file, resolved.scenario, &req.scenario, Purpose::Sweep)
                .map_err(|e| refuse(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
        let _starting = self.begin()?;
        self.launch(&req, sweep)
    }

    /// Mark a sweep as starting, after checking again that one may start. The
    /// mark clears when the guard drops, however the start ends.
    fn begin(&self) -> Result<Starting, Refusal> {
        let mut state = self.lock();
        admit(&state)?;
        state.starting = Some(None);
        Ok(Starting(self.clone()))
    }

    /// Start the sweep: its directory, its record and its supervisor. The record
    /// is installed before `meta.json` is written, so a reader never finds the
    /// sweep on disk and not in memory.
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
        self.lock().starting = Some(Some(id.clone()));
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
            ok: None,
            failed: None,
        };
        let stop = StopRequested::default();
        let (done_tx, done) = watch::channel(false);
        // The supervisor flies the job it is sent once the sweep's files are written.
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let supervisor = std::thread::Builder::new()
            .name(format!("sweep-{id}"))
            .spawn(move || {
                if let Ok(job) = job_rx.recv() {
                    job.run();
                }
            })
            .map_err(|e| {
                let _ = std::fs::remove_dir_all(&dir);
                refuse(StatusCode::INTERNAL_SERVER_ERROR, format!("starting the sweep: {e}"))
            })?;
        let escalation = {
            let (stop, done, children) = (stop.clone(), done.clone(), self.0.children.clone());
            let grace = cfg.stop_grace;
            std::thread::Builder::new()
                .name(format!("sweep-{id}-stop"))
                .spawn(move || escalate(stop, children, grace, done))
        };
        if let Err(e) = escalation {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(refuse(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("starting the sweep: {e}"),
            ));
        }
        let runs = (1..=sweep.runs.len()).map(RunEntry::queued).collect();
        let record = Record {
            base: base.clone(),
            started_ms,
            out: dir.clone(),
            ended: None,
            runs,
            stop: stop.clone(),
            done,
            supervisor: Some(supervisor),
        };
        let (previous, view) = {
            let mut state = self.lock();
            if state.closed {
                stop.0.store(true, Ordering::Relaxed);
            }
            let view = record.view();
            (state.current.replace(record), view)
        };
        if let Err(why) = write_start(&dir, req, &sweep, &base) {
            self.lock().current = previous;
            let _ = std::fs::remove_dir_all(&dir);
            return Err(refuse(StatusCode::INTERNAL_SERVER_ERROR, why));
        }
        let job = Job {
            sweeps: self.clone(),
            id,
            dir,
            sweep,
            stop,
            children: self.0.children.clone(),
            sim: cfg.sim.clone(),
            base_port: cfg.base_port,
            done: done_tx,
        };
        // Cannot fail: the supervisor waits for this job.
        let _ = job_tx.send(job);
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
    /// ended, so whoever sees it ended also sees its meta on disk. A meta that
    /// cannot be written keeps the outcome; the write error joins the reason.
    fn end(&self, id: &str, dir: &Path, mut ended: Ended, done: &watch::Sender<bool>) {
        let meta =
            self.lock().current.as_ref().filter(|r| r.base.id == id).map(|r| r.meta(Some(&ended)));
        if let Some(meta) = meta {
            if let Err(why) = write_meta(dir, &meta) {
                ended.reason = Some(match ended.reason.take() {
                    Some(reason) => format!("{reason}; {why}"),
                    None => why,
                });
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
        check_id(id)?;
        if let Some(view) = self.live(id) {
            return Ok(view);
        }
        let dir = self.dir_of(id)?;
        disk_view(&dir, read_meta(&dir)?)
    }

    /// Every sweep under the sweeps dir, newest first. Blocks. A sweep whose
    /// meta (or, for an interrupted one, summary) cannot be read is left out.
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
        Ok(ids.iter().filter_map(|id| self.summary_of(id)).collect())
    }

    /// `id`'s line of the list: from memory for the live sweep, else from its
    /// meta alone. Only a sweep that says `running` (interrupted) or predates
    /// the counts in its meta reads its summary. `None` when that cannot be read.
    fn summary_of(&self, id: &str) -> Option<SweepSummary> {
        if let Some(view) = self.live(id) {
            return Some(SweepSummary::of(&view));
        }
        let dir = self.0.cfg.dir.join(id);
        let meta = read_meta(&dir).ok()?;
        let (ok, failed) = match (meta.state, meta.ok, meta.failed) {
            (SweepState::Running, ..) | (_, None, _) | (_, _, None) => {
                let runs = read_runs(&dir, meta.total).ok()?;
                let count =
                    |status: EntryStatus| runs.iter().filter(|r| r.status == status).count();
                (count(EntryStatus::Ok), count(EntryStatus::Failed))
            }
            (_, Some(ok), Some(failed)) => (ok, failed),
        };
        let state = match meta.state {
            SweepState::Running => SweepState::Interrupted,
            state => state,
        };
        Some(SweepSummary {
            id: meta.id,
            scenario: meta.scenario,
            state,
            total: meta.total,
            ok,
            failed,
            started_at: meta.started_at,
            wall_s: meta.wall_s,
        })
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

    /// The flight tracks (thinned to `query`'s `points`) of the runs in its `from` / `count` range that
    /// finished with a readable bag. Blocks.
    fn tracks_of(&self, id: &str, query: &TracksQuery) -> Result<TracksBody, Refusal> {
        let view = self.view_of(id)?;
        let params = TracksParams::parse(query)?;
        let dir = self.0.cfg.dir.join(id);
        let end = params.from.saturating_add(params.count);
        let runs = view
            .runs
            .iter()
            .filter(|run| (params.from..end).contains(&run.run))
            .filter(|run| {
                matches!(run.status, EntryStatus::Ok | EntryStatus::Failed | EntryStatus::Stopped)
            })
            .filter_map(|run| {
                let aircraft = run_tracks(&dir, run.run, params.points)?;
                Some(RunTracks {
                    run: run.run,
                    status: run.status,
                    sim_time_s: run.sim_time_s,
                    aircraft,
                })
            })
            .collect();
        Ok(TracksBody {
            points: params.points,
            from: params.from,
            count: params.count,
            total: view.total,
            runs,
        })
    }
}

/// Holds a sweep's `starting` mark: clears it when dropped, however the start ends.
struct Starting(SweepRuns);

impl Drop for Starting {
    fn drop(&mut self) {
        self.0.lock().starting = None;
    }
}

/// Refuse a sweep while the server is closing, one is being started or one is
/// active. Plans nothing, so a conflicting request costs next to nothing.
fn admit(state: &SweepsState) -> Result<(), Refusal> {
    if state.closed {
        return Err(refuse(StatusCode::SERVICE_UNAVAILABLE, "the server is shutting down"));
    }
    match &state.starting {
        Some(Some(id)) => {
            return Err(refuse(StatusCode::CONFLICT, format!("a sweep is already running: {id}")))
        }
        Some(None) => return Err(refuse(StatusCode::CONFLICT, "a sweep is already starting")),
        None => {}
    }
    match state.current.as_ref().filter(|r| r.active()) {
        Some(active) => Err(refuse(
            StatusCode::CONFLICT,
            format!("a sweep is already running: {}", active.base.id),
        )),
        None => Ok(()),
    }
}

/// Kill the sweep's processes once a stop is requested and `grace` has passed,
/// and again every [`KILL_EVERY`], until the supervisor has ended (`done`). The
/// thread ends with the sweep: it is never left behind.
fn escalate(stop: StopRequested, children: Children, grace: Duration, done: watch::Receiver<bool>) {
    let mut stop_at: Option<Instant> = None;
    let mut next_kill: Option<Instant> = None;
    loop {
        let finished = *done.borrow();
        // `has_changed` errs once the sender is gone: the sweep never flew.
        if finished || done.has_changed().is_err() {
            return;
        }
        if stop.requested() {
            let now = Instant::now();
            let at = *stop_at.get_or_insert(now + grace);
            if now >= at && next_kill.is_none_or(|t| now >= t) {
                children.kill_all();
                next_kill = Some(now + KILL_EVERY);
            }
        }
        std::thread::sleep(TICK);
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

/// Parse RFC 4180 CSV: a quote opens a cell only at its start; inside quotes
/// `""` is a quote and commas and newlines are cell text. LF or CRLF ends a
/// row; blank lines are skipped, and a line holding only `""` is a row of one
/// empty cell. Nothing is rejected: an unclosed quote runs to the end.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    // The cell has no character of its own yet: a quote here opens it.
    let mut fresh = true;
    // The row has a character or a cell end on it (a blank line has neither).
    let mut open = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\n' && c != '\r' {
            open = true;
        }
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
            '"' if fresh => {
                quoted = true;
                fresh = false;
            }
            ',' => {
                row.push(std::mem::take(&mut cell));
                fresh = true;
            }
            '\n' => {
                if open {
                    row.push(std::mem::take(&mut cell));
                    rows.push(std::mem::take(&mut row));
                }
                open = false;
                fresh = true;
            }
            '\r' => {}
            _ => {
                cell.push(c);
                fresh = false;
            }
        }
    }
    if open {
        row.push(cell);
        rows.push(row);
    }
    rows
}

/// A summary cell as JSON: an empty cell as null; a canonical decimal integer
/// (exact up to `u64`) and a finite plain real as numbers; anything else (a
/// leading zero, a `+`, a space) as its text.
fn cell_value(cell: &str) -> Value {
    if cell.is_empty() {
        return Value::Null;
    }
    if let Ok(i) = cell.parse::<i64>() {
        if i.to_string() == cell {
            return Value::from(i);
        }
    }
    if let Ok(u) = cell.parse::<u64>() {
        if u.to_string() == cell {
            return Value::from(u);
        }
    }
    match cell.parse::<f64>() {
        Ok(f) if f.is_finite() && plain_real(cell) => Value::from(f),
        _ => Value::String(cell.to_owned()),
    }
}

/// Whether `cell` is written `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
fn plain_real(cell: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let body = cell.strip_prefix('-').unwrap_or(cell);
    let (mantissa, exponent) = match body.split_once(['e', 'E']) {
        Some((m, e)) => (m, Some(e)),
        None => (body, None),
    };
    let (int, frac) = match mantissa.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (mantissa, None),
    };
    let int_ok = digits(int) && (int == "0" || !int.starts_with('0'));
    let frac_ok = frac.is_none_or(digits);
    let exp_ok = exponent.is_none_or(|e| digits(e.strip_prefix(['+', '-']).unwrap_or(e)));
    int_ok && frac_ok && exp_ok
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

// --- Flight tracks ------------------------------------------------------------------

/// `count` of `GET /api/sweeps/<id>/tracks` when not given, and its largest.
const TRACKS_COUNT: usize = 100;
const TRACKS_MAX_COUNT: usize = 500;
/// `points` of that route when not given, and its range.
const TRACKS_POINTS: usize = 80;
const TRACKS_MIN_POINTS: usize = 2;
const TRACKS_MAX_POINTS: usize = 400;

/// The query of `GET /api/sweeps/<id>/tracks`, as sent.
#[derive(Deserialize)]
pub(crate) struct TracksQuery {
    from: Option<String>,
    count: Option<String>,
    points: Option<String>,
}

/// The query's parameters, checked.
struct TracksParams {
    from: usize,
    count: usize,
    points: usize,
}

impl TracksParams {
    fn parse(query: &TracksQuery) -> Result<Self, Refusal> {
        let from = match query.from.as_deref() {
            None => 1,
            Some(text) => text.parse::<usize>().ok().filter(|n| *n >= 1).ok_or_else(|| {
                refuse(
                    StatusCode::BAD_REQUEST,
                    format!("from {text:?}: want a whole number from 1"),
                )
            })?,
        };
        Ok(Self {
            from,
            count: bounded(query.count.as_deref(), "count", 1..=TRACKS_MAX_COUNT, TRACKS_COUNT)?,
            points: bounded(
                query.points.as_deref(),
                "points",
                TRACKS_MIN_POINTS..=TRACKS_MAX_POINTS,
                TRACKS_POINTS,
            )?,
        })
    }
}

/// A query parameter as a whole number in `range`, `default` when absent.
fn bounded(
    text: Option<&str>,
    name: &str,
    range: RangeInclusive<usize>,
    default: usize,
) -> Result<usize, Refusal> {
    let Some(text) = text else { return Ok(default) };
    text.parse::<usize>().ok().filter(|n| range.contains(n)).ok_or_else(|| {
        refuse(
            StatusCode::BAD_REQUEST,
            format!(
                "{name} {text:?}: want a whole number from {} to {}",
                range.start(),
                range.end()
            ),
        )
    })
}

/// An aircraft's track as the API reports it and a run's cache holds it: the
/// fields of [`AircraftTrack`].
#[derive(Debug, Serialize, Deserialize)]
struct TrackDoc {
    id: u32,
    model: Option<String>,
    t: Vec<f64>,
    e: Vec<f64>,
    n: Vec<f64>,
    u: Vec<f64>,
    died_at_s: Option<f64>,
}

impl TrackDoc {
    /// Whether the four arrays are one length: a cache that is not is recomputed.
    fn well_formed(&self) -> bool {
        let len = self.t.len();
        self.e.len() == len && self.n.len() == len && self.u.len() == len
    }
}

impl From<AircraftTrack> for TrackDoc {
    fn from(a: AircraftTrack) -> Self {
        Self { id: a.id, model: a.model, t: a.t, e: a.e, n: a.n, u: a.u, died_at_s: a.died_at_s }
    }
}

/// One run in a tracks reply.
#[derive(Serialize)]
struct RunTracks {
    run: usize,
    status: EntryStatus,
    sim_time_s: Option<f64>,
    aircraft: Vec<TrackDoc>,
}

/// A `GET /api/sweeps/<id>/tracks` reply.
#[derive(Serialize)]
struct TracksBody {
    points: usize,
    from: usize,
    count: usize,
    total: usize,
    runs: Vec<RunTracks>,
}

/// Run `run`'s tracks in `dir` thinned to `points`: from its cache when that is
/// not older than its bag, else from the bag, which then fills the cache. `None`
/// when the bag is missing or unreadable.
fn run_tracks(dir: &Path, run: usize, points: usize) -> Option<Vec<TrackDoc>> {
    let bag = dir.join(format!("run-{run:04}.mcap"));
    let cache = dir.join(format!("run-{run:04}.tracks-{points}.json"));
    let bag_time = modified(&bag)?;
    if let Some(tracks) = read_cache(&cache, bag_time) {
        return Some(tracks);
    }
    let reader = BagReader::open(&bag).ok()?;
    let aircraft = bag_tracks(&reader, points).ok()?.aircraft;
    let tracks: Vec<TrackDoc> = aircraft.into_iter().map(TrackDoc::from).collect();
    // The cache only saves the next call its computation, so a write that fails costs nothing.
    if let Ok(bytes) = serde_json::to_vec(&tracks) {
        let _ = write_atomic(&cache, &bytes);
    }
    Some(tracks)
}

/// The cached tracks at `cache`, if it is no older than the bag (`bag_time`) and
/// reads as well-formed tracks; a corrupt one is `None`.
fn read_cache(cache: &Path, bag_time: SystemTime) -> Option<Vec<TrackDoc>> {
    if modified(cache)? < bag_time {
        return None;
    }
    let tracks: Vec<TrackDoc> = serde_json::from_slice(&std::fs::read(cache).ok()?).ok()?;
    tracks.iter().all(TrackDoc::well_formed).then_some(tracks)
}

/// When `path` was last modified, if it can be read.
fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
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

/// `GET /api/sweeps/<id>/tracks?from=&count=&points=`: the flight tracks of the runs in range.
pub(crate) async fn tracks(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Query(query): Query<TracksQuery>,
) -> Response {
    let sweeps = state.sweeps.clone();
    match tokio::task::spawn_blocking(move || sweeps.tracks_of(&id, &query)).await {
        Ok(Ok(body)) => json(StatusCode::OK, &body),
        Ok(Err(refusal)) => refusal.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("reading the tracks: {e}")),
    }
}
