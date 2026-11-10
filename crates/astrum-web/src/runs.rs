//! The studio's live interactive runs (`/api/runs`): one at a time, owned by
//! the server.
//!
//! A run is a scenario (a saved one by name, or an inline document) plus a
//! catalog controller per chosen aircraft. Starting one checks every
//! assignment against the scenario and the catalog, flies exactly the
//! controlled aircraft in lockstep, writes `<runs-dir>/<id>/scenario.toml`,
//! starts the sim on a free loopback port (recording `run.mcap`) with the
//! rendered controllers ([`LiveRun`]) and attaches the [`Bridge`] to it. A
//! monitor thread waits for the run's end, computes its metrics once into
//! `metrics.json`, marks it finished and — if the bridge still carries the
//! run — sends the bridge back to the server's `--connect` endpoint. The
//! run's output lines go to a bounded ring buffer with a monotonic sequence
//! number, its controllers named by their aircraft.
//!
//! Every run directory under the runs dir (this server's or an earlier one's)
//! serves its replay export and metrics; the active run's only once it has
//! finished.

use std::collections::VecDeque;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use astrum_bag::BagReader;
use astrum_net::{speed_ok, MIN_SPEED};
use astrum_sim::launch::{Children, ControllerCmd, LiveRun, LiveSpec, Log, LogLine, LogStream};
use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::Outcome;
use astrum_sim::systems::StopRequested;
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, QueryRejection};
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::watch;

use crate::api::{error, json, safe_name, NAME_MAX_LEN};
use crate::assign::{
    assign, check_programs, scenario_input, Assignment, AssignmentBody, InputError,
};
use crate::bridge::Bridge;
use crate::catalogs::Catalogs;
use crate::export::export_bag;
use crate::server::AppState;

/// Most log lines kept for the current run (the oldest go first).
pub const LOG_CAPACITY: usize = 5000;
/// Longest log line kept (bytes); longer ones are cut at a character boundary.
pub const LOG_LINE_MAX: usize = 4096;
/// The scenario the sim flies, inside a run's directory.
pub const SCENARIO_FILE: &str = "scenario.toml";
/// The run's recording, inside its directory.
pub const BAG_FILE: &str = "run.mcap";
/// The run's metrics ([`astrum_bag::metrics::RunMetrics`]), inside its directory.
pub const METRICS_FILE: &str = "metrics.json";
/// The run-id name part of an inline (unsaved) scenario.
const INLINE_NAME: &str = "inline";

/// Where runs go, what flies them, and where the bridge goes back to.
#[derive(Debug, Clone)]
pub struct RunsConfig {
    /// Parent of every run's directory (`runs/web`).
    pub dir: PathBuf,
    /// The `astrum-sim` binary.
    pub sim: PathBuf,
    /// The server's `--connect` endpoint: the bridge re-attaches there once a
    /// run it was attached to has finished.
    pub connect: String,
}

/// A run's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    /// Its processes are up; the bridge has not reached the sim yet.
    Starting,
    Running,
    /// Every process is reaped; [`RunView::outcome`] says how it ended.
    Finished,
}

/// A run as the API reports it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunView {
    /// `<UTC yyyymmdd-hhmmss>-<scenario name | inline>`: sortable, a safe name.
    pub id: String,
    /// The saved scenario's name; `None` for an inline one.
    pub scenario: Option<String>,
    pub state: RunState,
    /// Where the run's sim listens (the bridge's attachment).
    pub endpoint: String,
    /// The pace it started at (× real time, 0 = as fast as possible).
    pub speed: f64,
    /// The sim-time limit it flies to (s, 0 = until stopped).
    pub max_time_s: f64,
    /// Whether the sim runs in lockstep (exactly when an aircraft is controlled).
    pub lockstep: bool,
    pub assignments: Vec<Assignment>,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub outcome: Option<Outcome>,
    /// The run's directory, as the server reaches it.
    pub dir: String,
    /// Its recording.
    pub bag: String,
}

/// One line of the current run's output.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LogEntry {
    pub seq: u64,
    pub source: String,
    pub stream: LogStream,
    pub line: String,
}

/// Why a run request failed; each maps to one HTTP status.
#[derive(Debug, Clone, PartialEq)]
pub enum RunError {
    /// Bad input: unknown aircraft, controller or scenario name, a kind
    /// mismatch, bad parameters (400).
    BadRequest(String),
    /// The scenario is not valid (422).
    InvalidScenario(String),
    /// Another run is active, or none is (409).
    Conflict(String),
    /// No such run, or nothing recorded (404).
    NotFound(String),
    /// The run's recording cannot be read (422).
    BadRecording(String),
    /// The server is shutting down (503).
    Unavailable(String),
    /// Setting up or starting the run failed (500).
    Failed(String),
}

impl RunError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::InvalidScenario(_) | Self::BadRecording(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Failed(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn respond(&self) -> Response {
        error(self.status(), &self.to_string())
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadRequest(m)
            | Self::InvalidScenario(m)
            | Self::Conflict(m)
            | Self::NotFound(m)
            | Self::BadRecording(m)
            | Self::Unavailable(m)
            | Self::Failed(m) => f.write_str(m),
        }
    }
}

/// The server's run manager. Cheap to clone; clones share it.
#[derive(Clone)]
pub struct Runs(Arc<Inner>);

impl fmt::Debug for Runs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Runs").field("config", &self.0.cfg).finish_non_exhaustive()
    }
}

struct Inner {
    cfg: RunsConfig,
    bridge: Bridge,
    children: Children,
    state: Mutex<RunsState>,
}

#[derive(Default)]
struct RunsState {
    /// The latest run since the server started.
    current: Option<Record>,
    /// A run is being started (it has no record yet).
    starting: bool,
    /// Shutting down: no new runs.
    closed: bool,
    log: LogRing,
}

struct Record {
    view: RunView,
    /// The bridge's run id for this run's attachment.
    bridge_run: u64,
    /// The bridge has reached the sim once.
    answered: bool,
    /// The run's own stop handle.
    stop: StopRequested,
    /// Turns `true` once the run is finished.
    done: watch::Receiver<bool>,
    monitor: Option<JoinHandle<()>>,
}

impl Record {
    fn active(&self) -> bool {
        self.view.outcome.is_none()
    }
}

#[derive(Default)]
struct LogRing {
    /// The run the lines belong to.
    run: String,
    lines: VecDeque<LogEntry>,
    /// The next line's seq: monotonic over the server's lifetime, so a
    /// reader's `since` from an earlier run never hides a new run's lines.
    next: u64,
}

impl LogRing {
    fn push(&mut self, mut line: LogLine) {
        if line.line.len() > LOG_LINE_MAX {
            let mut cut = LOG_LINE_MAX;
            while !line.line.is_char_boundary(cut) {
                cut -= 1;
            }
            line.line.truncate(cut);
        }
        if self.lines.len() == LOG_CAPACITY {
            self.lines.pop_front();
        }
        let LogLine { source, stream, line } = line;
        self.lines.push_back(LogEntry { seq: self.next, source, stream, line });
        self.next += 1;
    }
}

/// A `POST /api/runs` body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StartBody {
    /// A saved scenario's name, or a whole scenario document.
    scenario: Value,
    #[serde(default)]
    assignments: Vec<AssignmentBody>,
    /// Pace (× real time, 0 = as fast as possible); default: the scenario's.
    speed: Option<f64>,
    /// Sim-time limit override (s, 0 = until stopped); default: the scenario's.
    max_time_s: Option<f64>,
}

/// What a checked request flies.
struct Plan {
    name: Option<String>,
    scenario: Scenario,
    /// Named by their aircraft: "aircraft 1's controller", source "aircraft 1".
    controllers: Vec<ControllerCmd>,
    assignments: Vec<Assignment>,
}

impl Runs {
    pub fn new(cfg: RunsConfig, bridge: Bridge) -> Self {
        Self(Arc::new(Inner {
            cfg,
            bridge,
            children: Children::new(),
            state: Mutex::new(RunsState::default()),
        }))
    }

    pub fn config(&self) -> &RunsConfig {
        &self.0.cfg
    }

    fn lock(&self) -> MutexGuard<'_, RunsState> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The latest run, its state brought up to date.
    pub fn current(&self) -> Option<RunView> {
        let info = self.0.bridge.info();
        let mut state = self.lock();
        let record = state.current.as_mut()?;
        if record.active() && !record.answered {
            record.answered = info.run == record.bridge_run && info.connected;
            if record.answered {
                record.view.state = RunState::Running;
            }
        }
        Some(record.view.clone())
    }

    /// The current run's log lines with `seq >= since`, and the seq to ask
    /// for next.
    pub fn log_since(&self, since: u64) -> (Vec<LogEntry>, u64) {
        let state = self.lock();
        let lines = state.log.lines.iter().filter(|l| l.seq >= since).cloned().collect();
        (lines, state.log.next)
    }

    /// Ask the active run (if any) to stop and refuse new ones: the start of
    /// a server shutdown. Never blocks.
    pub fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        if let Some(record) = state.current.as_ref().filter(|r| r.active()) {
            record.stop.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// [`close`](Self::close), then block until the active run (and one
    /// being started) has ended and its metrics are written.
    pub fn shutdown(&self) {
        self.close();
        while self.lock().starting {
            std::thread::sleep(Duration::from_millis(20));
        }
        // A run installed while closing was stopped as it was installed.
        let monitor = self.lock().current.as_mut().and_then(|r| r.monitor.take());
        if let Some(monitor) = monitor {
            let _ = monitor.join();
        }
        self.0.children.kill_all();
    }

    /// Check the request and start the run. Blocks (file reads, process
    /// start): call it on a blocking thread.
    fn start_blocking(&self, body: StartBody, catalogs: &Catalogs) -> Result<RunView, RunError> {
        let plan = resolve(body, catalogs)?;
        {
            let mut state = self.lock();
            if state.closed {
                return Err(RunError::Unavailable("the server is shutting down".into()));
            }
            if state.starting {
                return Err(RunError::Conflict("another run is being started".into()));
            }
            if let Some(active) = state.current.as_ref().filter(|r| r.active()) {
                return Err(RunError::Conflict(format!(
                    "run {} is active; stop it first (DELETE /api/runs/current)",
                    active.view.id
                )));
            }
            state.starting = true;
        }
        let started = self.launch(plan);
        self.lock().starting = false;
        started
    }

    fn launch(&self, plan: Plan) -> Result<RunView, RunError> {
        let root = &self.0.cfg.dir;
        std::fs::create_dir_all(root)
            .map_err(|e| RunError::Failed(format!("creating {}: {e}", root.display())))?;
        let started_at_ms = now_ms();
        let (id, dir) = new_run_dir(
            root,
            &utc_stamp(started_at_ms),
            plan.name.as_deref().unwrap_or(INLINE_NAME),
        )?;
        let port = free_port().map_err(|e| RunError::Failed(format!("picking a port: {e}")));
        let port = match port {
            Ok(port) => port,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(e);
            }
        };
        let listen = format!("tcp/127.0.0.1:{port}");
        let bag = dir.join(BAG_FILE);
        let (tx, rx) = mpsc::channel();
        // The log thread keeps the run's lines in the channel until the run's
        // record (and its log ring) is installed, and discards them if it
        // never is; it ends once the run's processes (the senders) are gone.
        let (log_installed_tx, log_installed) = mpsc::channel::<()>();
        let logs = self.clone();
        let log_id = id.clone();
        let log_thread =
            std::thread::Builder::new().name(format!("run-log-{id}")).spawn(move || {
                let keep = log_installed.recv().is_ok();
                for line in rx {
                    if keep {
                        logs.push_log(&log_id, line);
                    }
                }
            });
        let log_thread = match log_thread {
            Ok(handle) => handle,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(RunError::Failed(format!("starting the log thread: {e}")));
            }
        };
        let spec = LiveSpec {
            sim: self.0.cfg.sim.clone(),
            scenario: plan.scenario.clone(),
            scenario_file: Some(dir.join(SCENARIO_FILE)),
            listen,
            record: Some(bag.clone()),
            companions: Vec::new(),
            controllers: plan.controllers,
            run_timeout: None,
            fail_fast: true,
            log: Log::Channel(tx),
            stop: StopRequested::default(),
        };
        let live = match LiveRun::start(spec, &self.0.children) {
            Ok(live) => live,
            Err(why) => {
                // Nothing runs any more: leave no run behind.
                drop(log_installed_tx);
                let _ = log_thread.join();
                let _ = std::fs::remove_dir_all(&dir);
                return Err(RunError::Failed(format!("starting the run: {why}")));
            }
        };
        let endpoint = live.endpoint().to_owned();
        let stop = live.stop_handle();
        // The monitor finalizes the run only once its record is installed
        // (`installed` sent, or dropped if installing never happens).
        let (installed_tx, installed) = mpsc::channel::<()>();
        let (done_tx, done) = watch::channel(false);
        let monitor = self.clone();
        let (monitor_id, monitor_dir) = (id.clone(), dir.clone());
        let handle = std::thread::Builder::new().name(format!("run-{id}")).spawn(move || {
            let outcome = live.wait();
            let _ = log_thread.join();
            let _ = installed.recv();
            monitor.finish(&monitor_id, &monitor_dir, outcome, &done_tx);
        });
        let handle = match handle {
            Ok(handle) => handle,
            Err(e) => {
                // The unstarted closure (and the run in it) was dropped:
                // dropping a LiveRun stops it and reaps its processes.
                let _ = std::fs::remove_dir_all(&dir);
                return Err(RunError::Failed(format!("starting the run monitor: {e}")));
            }
        };
        let bridge_run = self.0.bridge.attach(&endpoint);
        let view = RunView {
            id: id.clone(),
            scenario: plan.name,
            state: RunState::Starting,
            endpoint,
            speed: plan.scenario.sim.speed,
            max_time_s: plan.scenario.sim.max_time_s,
            lockstep: plan.scenario.sim.lockstep,
            assignments: plan.assignments,
            started_at_ms,
            finished_at_ms: None,
            outcome: None,
            dir: dir.display().to_string(),
            bag: bag.display().to_string(),
        };
        {
            let mut state = self.lock();
            if state.closed {
                stop.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            state.log = LogRing { run: id.clone(), lines: VecDeque::new(), next: state.log.next };
            state.current = Some(Record {
                view: view.clone(),
                bridge_run,
                answered: false,
                stop,
                done,
                monitor: Some(handle),
            });
        }
        let _ = log_installed_tx.send(());
        let _ = installed_tx.send(());
        Ok(view)
    }

    fn push_log(&self, run: &str, line: LogLine) {
        let mut state = self.lock();
        if state.log.run == run {
            state.log.push(line);
        }
    }

    /// The run has ended (every process reaped, its log drained): write its
    /// metrics, send the bridge back to `--connect` if it still carries this
    /// run, then mark the run finished — so whoever sees `finished` also sees
    /// the bridge back home.
    fn finish(&self, id: &str, dir: &Path, outcome: Outcome, done: &watch::Sender<bool>) {
        if dir.join(BAG_FILE).is_file() {
            if let Err(why) = write_metrics(dir) {
                self.push_log(
                    id,
                    LogLine {
                        source: "astrum".into(),
                        stream: LogStream::Stderr,
                        line: format!("metrics: {why}"),
                    },
                );
            }
        }
        let bridge_run =
            self.lock().current.as_ref().filter(|r| r.view.id == id).map(|r| r.bridge_run);
        if bridge_run.is_some_and(|run| self.0.bridge.info().run == run) {
            self.0.bridge.attach(&self.0.cfg.connect);
        }
        if let Some(record) = self.lock().current.as_mut().filter(|r| r.view.id == id) {
            record.view.state = RunState::Finished;
            record.view.finished_at_ms = Some(now_ms());
            record.view.outcome = Some(outcome);
        }
        done.send_replace(true);
    }

    /// Ask the active run to stop and wait until it has finished.
    async fn stop_current(&self) -> Result<RunView, RunError> {
        let mut done = {
            let state = self.lock();
            let Some(record) = state.current.as_ref().filter(|r| r.active()) else {
                let why =
                    if state.starting { "a run is being started" } else { "no run is active" };
                return Err(RunError::Conflict(why.into()));
            };
            record.stop.0.store(true, std::sync::atomic::Ordering::Relaxed);
            record.done.clone()
        };
        // A monitor that died (its sender dropped) leaves the run as it is.
        let _ = done.wait_for(|finished| *finished).await;
        self.current().ok_or_else(|| RunError::Failed("the run vanished".into()))
    }

    /// The directory of the finished run `id` under the runs dir.
    fn finished_dir(&self, id: &str) -> Result<PathBuf, RunError> {
        let id = safe_name(id).map_err(|e| RunError::BadRequest(format!("run id: {e}")))?;
        if self.lock().current.as_ref().is_some_and(|r| r.active() && r.view.id == id) {
            return Err(RunError::Conflict(format!("run {id} has not finished yet")));
        }
        let dir = self.0.cfg.dir.join(id);
        if !dir.is_dir() {
            return Err(RunError::NotFound(format!("no run named {id:?}")));
        }
        if !dir.join(BAG_FILE).is_file() {
            return Err(RunError::NotFound(format!("run {id} has no recording ({BAG_FILE})")));
        }
        Ok(dir)
    }

    /// The run's metrics: `metrics.json`, computed (and written) first if
    /// missing. Blocks.
    fn metrics_blocking(&self, id: &str) -> Result<Value, RunError> {
        let dir = self.finished_dir(id)?;
        let path = dir.join(METRICS_FILE);
        let text = match std::fs::read(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                write_metrics(&dir).map_err(RunError::BadRecording)?;
                std::fs::read(&path)
                    .map_err(|e| RunError::Failed(format!("reading {}: {e}", path.display())))?
            }
            Err(e) => return Err(RunError::Failed(format!("reading {}: {e}", path.display()))),
        };
        serde_json::from_slice(&text)
            .map_err(|e| RunError::Failed(format!("{}: not JSON: {e}", path.display())))
    }
}

/// Compute the bag's metrics into `<dir>/metrics.json` (temp file + rename,
/// so a reader never sees half of it).
fn write_metrics(dir: &Path) -> Result<(), String> {
    let bag = dir.join(BAG_FILE);
    let reader = BagReader::open(&bag).map_err(|e| format!("{}: {e}", bag.display()))?;
    let metrics =
        astrum_bag::metrics::compute(&reader).map_err(|e| format!("{}: {e}", bag.display()))?;
    let bytes = serde_json::to_vec_pretty(&metrics)
        .unwrap_or_else(|e| unreachable!("metrics serialisation failed: {e}"));
    let path = dir.join(METRICS_FILE);
    let temp = dir.join(format!(".{METRICS_FILE}.tmp"));
    std::fs::write(&temp, bytes)
        .and_then(|()| std::fs::rename(&temp, &path))
        .map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Resolve the scenario, apply the overrides and check the assignments.
fn resolve(body: StartBody, catalogs: &Catalogs) -> Result<Plan, RunError> {
    let (name, mut scenario) = scenario_input(body.scenario, catalogs).map_err(|e| match e {
        InputError::Bad(why) => RunError::BadRequest(why),
        InputError::Invalid(why) => RunError::InvalidScenario(why),
        InputError::Io(why) => RunError::Failed(why),
    })?;
    if let Some(speed) = body.speed {
        if !speed_ok(speed, MIN_SPEED) {
            return Err(RunError::BadRequest(format!(
                "speed must be 0 (as fast as possible) or at least {MIN_SPEED}, got {speed}"
            )));
        }
        scenario.sim.speed = speed;
    }
    if let Some(max_time_s) = body.max_time_s {
        if !(max_time_s.is_finite() && max_time_s >= 0.0) {
            return Err(RunError::BadRequest(format!(
                "max_time_s must be 0 (until stopped) or a positive number of seconds, got \
                 {max_time_s}"
            )));
        }
        scenario.sim.max_time_s = max_time_s;
    }
    // The overrides were checked above and `assign` keeps the lockstep rules,
    // so the scenario stays valid.
    let (controllers, assignments) =
        assign(&mut scenario, &body.assignments, catalogs.controllers())
            .map_err(RunError::BadRequest)?;
    check_programs(&assignments, catalogs.controllers()).map_err(RunError::Failed)?;
    let controllers = controllers
        .into_iter()
        .zip(&assignments)
        .map(|(command, a)| ControllerCmd {
            command,
            name: format!("aircraft {}'s controller", a.aircraft),
            source: format!("aircraft {}", a.aircraft),
        })
        .collect();
    Ok(Plan { name, scenario, controllers, assignments })
}

/// Create `<root>/<stamp>-<name>` (or `…-2`, `…-3`, … if taken), cut to a
/// [`safe_name`]'s length.
fn new_run_dir(root: &Path, stamp: &str, name: &str) -> Result<(String, PathBuf), RunError> {
    let base = format!("{stamp}-{name}");
    for n in 1..1000u32 {
        let suffix = if n == 1 { String::new() } else { format!("-{n}") };
        let keep = base.len().min(NAME_MAX_LEN - suffix.len());
        let id = format!("{}{suffix}", &base[..keep]);
        let dir = root.join(&id);
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok((id, dir)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(RunError::Failed(format!("creating {}: {e}", dir.display()))),
        }
    }
    Err(RunError::Failed(format!("too many runs named {base} in {}", root.display())))
}

/// A loopback TCP port no one listens on right now.
fn free_port() -> io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// `yyyymmdd-hhmmss` (UTC) of a Unix time in ms.
fn utc_stamp(ms: u64) -> String {
    let secs = ms / 1000;
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // Civil date from days since 1970-01-01 (H. Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

// --- HTTP ---------------------------------------------------------------------

#[derive(Serialize)]
struct CurrentBody {
    run: Option<RunView>,
}

#[derive(Serialize)]
struct LogBody {
    lines: Vec<LogEntry>,
    next: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LogQuery {
    #[serde(default)]
    since: u64,
}

/// `POST /api/runs`.
pub(crate) async fn start(
    State(state): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => return error(rejection.status(), &rejection.body_text()),
    };
    let body: StartBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad run body: {e}")),
    };
    let (runs, catalogs) = (state.runs.clone(), state.catalogs.clone());
    // Started on its own thread, the run is installed even if the client
    // goes away meanwhile.
    match tokio::task::spawn_blocking(move || runs.start_blocking(body, &catalogs)).await {
        Ok(Ok(view)) => json(StatusCode::CREATED, &view),
        Ok(Err(e)) => e.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("starting the run: {e}")),
    }
}

/// `GET /api/runs/current`.
pub(crate) async fn current(State(state): State<AppState>) -> Response {
    json(StatusCode::OK, &CurrentBody { run: state.runs.current() })
}

/// `DELETE /api/runs/current`.
pub(crate) async fn stop(State(state): State<AppState>) -> Response {
    match state.runs.stop_current().await {
        Ok(view) => json(StatusCode::OK, &view),
        Err(e) => e.respond(),
    }
}

/// `GET /api/runs/current/log?since=<seq>`.
pub(crate) async fn log(
    State(state): State<AppState>,
    query: Result<Query<LogQuery>, QueryRejection>,
) -> Response {
    let since = match query {
        Ok(Query(q)) => q.since,
        Err(rejection) => return error(StatusCode::BAD_REQUEST, &rejection.body_text()),
    };
    let (lines, next) = state.runs.log_since(since);
    json(StatusCode::OK, &LogBody { lines, next })
}

/// `GET /api/runs/<id>/replay`: the run's bag as an export document.
pub(crate) async fn replay(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
) -> Response {
    let runs = state.runs.clone();
    let exported = tokio::task::spawn_blocking(move || {
        let dir = runs.finished_dir(&id)?;
        let bag = dir.join(BAG_FILE);
        export_bag(&bag).map_err(|e| RunError::BadRecording(format!("{}: {e}", bag.display())))
    });
    match exported.await {
        Ok(Ok(doc)) => json(StatusCode::OK, &doc),
        Ok(Err(e)) => e.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("exporting the run: {e}")),
    }
}

/// `GET /api/runs/<id>/metrics`.
pub(crate) async fn metrics(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<String>,
) -> Response {
    let runs = state.runs.clone();
    match tokio::task::spawn_blocking(move || runs.metrics_blocking(&id)).await {
        Ok(Ok(metrics)) => json(StatusCode::OK, &metrics),
        Ok(Err(e)) => e.respond(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("reading the metrics: {e}")),
    }
}
