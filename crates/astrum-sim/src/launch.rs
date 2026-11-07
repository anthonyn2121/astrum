//! Process supervision for runs that fly as processes: `astrum run`'s live
//! run, `astrum sweep`'s process mode ([`run_processes`]) and the web
//! studio's interactive runs share it.
//!
//! A [`LiveRun`] is one run: an `astrum-sim` process listening on an
//! endpoint, its companions (e.g. the viewer) and its controllers, started
//! together ([`LiveRun::start`]) and supervised on a thread of their own. The
//! sim decides the run: it ends with the scenario's `max_time_s`, on a stop
//! request ([`LiveRun::stop`], or the shared [`LiveSpec::stop`]) or after the
//! run timeout — the sim is asked to stop and ends cleanly — and then
//! everything else is stopped. A controller that fails meanwhile fails the
//! run (and, with `fail_fast`, stops it); a companion that exits is only
//! reported.
//!
//! Nothing here is process-global: every child is registered in the
//! caller's [`Children`], so a caller's signal handler can kill them all
//! ([`Children::kill_all`]); a long-lived caller starts as many runs as it
//! likes, one after another or at once.
//!
//! Blocking: [`LiveRun::start`] returns once everything is spawned, but on
//! failure it first stops what it had started (up to the grace period);
//! [`LiveRun::wait`] and dropping an unfinished [`LiveRun`] block until the
//! run's processes are reaped. Async callers run those on a blocking thread.

use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::process::CommandExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::scenario::Scenario;
use crate::sweep::{parse_stopped_line, Outcome, Run, Status, Sweep};
use crate::systems::StopRequested;

/// How long processes get to exit after a stop before they are killed.
const GRACE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);
/// ANSI colours of the terminal tags: `[sim]`, companions, and `[ctlN]` in turn.
const SIM_COLOR: &str = "36";
const COMPANION_COLOR: &str = "35";
const CONTROLLER_COLORS: [&str; 4] = ["32", "33", "34", "91"];

/// The ASTRUM binary `name` next to the current executable (`name` itself,
/// looked up on `PATH`, if that cannot be found).
pub fn sibling(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .unwrap_or_else(|| name.into())
}

/// Where a client on this host reaches a sim listening on `listen`: a wildcard
/// address (`0.0.0.0`, `[::]`) becomes loopback.
pub fn connect_endpoint(listen: &str) -> String {
    listen.replacen("/0.0.0.0:", "/127.0.0.1:", 1).replacen("/[::]:", "/[::1]:", 1)
}

/// A run file's `run_timeout_s` as a [`LiveSpec::run_timeout`] (0 = none).
pub fn run_timeout(run_timeout_s: f64) -> Option<Duration> {
    (run_timeout_s > 0.0).then(|| Duration::from_secs_f64(run_timeout_s))
}

// --- Child registry -------------------------------------------------------------

/// The caller's registry of child processes not yet reaped — pid, and whether
/// it leads its own process group — shared by every run started with it.
/// Clones share the registry.
#[derive(Debug, Clone, Default)]
pub struct Children(Arc<Mutex<Vec<(u32, bool)>>>);

impl Children {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<(u32, bool)>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pids of the children still registered (spawned, not yet reaped).
    pub fn pids(&self) -> Vec<u32> {
        self.lock().iter().map(|&(pid, _)| pid).collect()
    }

    /// SIGKILL every registered child (and the process group it leads) — the
    /// last resort of a second Ctrl-C. The runs then see their processes gone.
    pub fn kill_all(&self) {
        for &(pid, group) in self.lock().iter() {
            send(pid, group, libc::SIGKILL);
        }
    }
}

fn send(pid: u32, group: bool, sig: libc::c_int) {
    let pid = pid as libc::pid_t;
    // SAFETY: kill(2) on a child (or the process group it leads) that we
    // spawned and have not reaped.
    unsafe {
        libc::kill(if group { -pid } else { pid }, sig);
    }
}

// --- Output -------------------------------------------------------------------

/// Which of a child's output streams a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogStream {
    Stdout,
    Stderr,
}

/// One whole line of a run's output (without its line ending).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogLine {
    /// `"sim"`, a companion's source (`"web"`), `"ctl1"`, `"ctl2"`, …; or
    /// `"astrum"` for the supervisor's own messages.
    pub source: String,
    pub stream: LogStream,
    pub line: String,
}

/// Where a run's output goes.
#[derive(Debug, Clone)]
pub enum Log {
    /// Onto this terminal, every line tagged `[source]` in an ANSI colour
    /// (when the stream is a terminal); the supervisor's messages as
    /// `[astrum] …`. Each child leads its own process group, so the
    /// terminal's Ctrl-C reaches only this process, which stops the run in
    /// order.
    Terminal,
    /// As [`LogLine`]s, the supervisor's messages from source `"astrum"`; a
    /// dropped receiver only discards them. Each child leads its own process
    /// group.
    Channel(Sender<LogLine>),
    /// Into `<dir>/<stem>.<source>.log`, one file per process; the
    /// supervisor's messages on this terminal. Children stay in this process
    /// group (a terminal's Ctrl-C reaches them too).
    Files { dir: PathBuf, stem: String },
}

impl Log {
    fn output(&self, source: &str, color: &'static str) -> Output {
        match self {
            Log::Terminal => Output::Tagged { label: source.to_owned(), color },
            Log::Channel(tx) => Output::Channel { source: source.to_owned(), tx: tx.clone() },
            Log::Files { dir, stem } => Output::File(dir.join(format!("{stem}.{source}.log"))),
        }
    }

    /// One of the supervisor's own messages.
    fn say(&self, stream: LogStream, msg: String) {
        match self {
            Log::Channel(tx) => {
                let _ = tx.send(LogLine { source: "astrum".into(), stream, line: msg });
            }
            Log::Terminal | Log::Files { .. } => {
                let _ = match stream {
                    LogStream::Stdout => writeln!(io::stdout().lock(), "[astrum] {msg}"),
                    LogStream::Stderr => writeln!(io::stderr().lock(), "[astrum] {msg}"),
                };
            }
        }
    }
}

/// Where one child's output goes.
enum Output {
    File(PathBuf),
    Tagged { label: String, color: &'static str },
    Channel { source: String, tx: Sender<LogLine> },
}

/// The registries a run's child is listed in: the caller's and the run's own.
#[derive(Clone)]
struct Registries {
    shared: Children,
    own: Children,
}

/// A spawned child and the threads copying its output; reaped (killed if
/// need be) when dropped.
struct Proc {
    /// For messages: "the sim", "controller 2", "the viewer".
    name: String,
    child: Child,
    group: bool,
    copiers: Vec<JoinHandle<()>>,
    registries: Registries,
    reaped: bool,
}

/// The sim time of the last [`crate::sweep::stopped_line`] seen on a stream.
type SimTime = Arc<Mutex<Option<f64>>>;

impl Proc {
    /// Spawn `cmd`, registered in `registries`; with `sim_time`, the sim time
    /// of a stopped line on its (piped) stdout lands there.
    fn spawn(
        name: impl Into<String>,
        mut cmd: Command,
        output: Output,
        registries: &Registries,
        sim_time: Option<&SimTime>,
    ) -> io::Result<Self> {
        cmd.stdin(Stdio::null());
        let group = !matches!(output, Output::File(_));
        match &output {
            Output::File(path) => {
                let log = File::create(path)?;
                cmd.stdout(log.try_clone()?).stderr(log);
            }
            Output::Tagged { .. } | Output::Channel { .. } => {
                cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).process_group(0);
            }
        }
        let mut child = cmd.spawn()?;
        for r in [&registries.shared, &registries.own] {
            r.lock().push((child.id(), group));
        }
        let mut copiers = Vec::new();
        if let Some(out) = child.stdout.take() {
            let mut sink = sink(&output, LogStream::Stdout);
            let watch = sim_time.cloned();
            copiers.push(copy_lines(out, move |line| {
                if let (Some(watch), Some(t)) = (&watch, parse_stopped_line(line)) {
                    *watch.lock().unwrap_or_else(|e| e.into_inner()) = Some(t);
                }
                sink(line);
            }));
        }
        if let Some(err) = child.stderr.take() {
            copiers.push(copy_lines(err, sink(&output, LogStream::Stderr)));
        }
        let registries = registries.clone();
        Ok(Self { name: name.into(), child, group, copiers, registries, reaped: false })
    }

    /// Signal the child (or its process group). Only ever called before
    /// [`Proc::reap`]: until then an exited child stays a zombie, so its pid
    /// and the process group it leads cannot be reused by a stranger.
    fn signal(&self, sig: libc::c_int) {
        if !self.reaped {
            send(self.child.id(), self.group, sig);
        }
    }

    /// The child's exit status once it has exited; `None` while it runs.
    /// Does not reap it (`WNOWAIT`): only [`Proc::reap`] does, after the last
    /// signal to its pid or group.
    fn exited(&mut self) -> io::Result<Option<ExitStatus>> {
        if self.reaped {
            return self.child.try_wait();
        }
        // SAFETY: an all-zero siginfo_t is valid; waitid(2) on our own child
        // with WNOWAIT only reads its state.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        let flags = libc::WEXITED | libc::WNOHANG | libc::WNOWAIT;
        let pid = self.child.id() as libc::id_t;
        if unsafe { libc::waitid(libc::P_PID, pid, &mut info, flags) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: waitid filled in a SIGCHLD siginfo (si_pid 0: no exit yet).
        let (child, status) = unsafe { (info.si_pid(), info.si_status()) };
        if child == 0 {
            return Ok(None);
        }
        // The wait(2) status word ExitStatus wraps.
        let raw = match info.si_code {
            libc::CLD_EXITED => (status & 0xff) << 8,
            libc::CLD_DUMPED => status | 0x80,
            _ => status,
        };
        Ok(Some(ExitStatus::from_raw(raw)))
    }

    fn running(&mut self) -> bool {
        matches!(self.exited(), Ok(None))
    }

    /// Kill the child if it still runs and what is left of the process group
    /// it leads (a background grandchild would hold its output open), take it
    /// out of the registries, then wait for it — the pid is released only
    /// after the last signal — and flush its output.
    fn reap(&mut self) -> Option<ExitStatus> {
        if !self.reaped {
            if self.running() {
                send(self.child.id(), self.group, libc::SIGKILL);
            }
            let pid = self.child.id();
            if self.group {
                send(pid, true, libc::SIGKILL);
            }
            for r in [&self.registries.shared, &self.registries.own] {
                r.lock().retain(|&(p, _)| p != pid);
            }
            self.reaped = true;
            let _ = self.child.wait();
            for copier in self.copiers.drain(..) {
                let _ = copier.join();
            }
        }
        self.child.try_wait().ok().flatten()
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        self.reap();
    }
}

/// Where the lines of one piped stream go.
fn sink(output: &Output, stream: LogStream) -> Box<dyn FnMut(&str) + Send> {
    let to_stderr = stream == LogStream::Stderr;
    match output {
        Output::Tagged { label, color } => {
            let tag = format!("{:<6}", format!("[{label}]"));
            let colored =
                if to_stderr { io::stderr().is_terminal() } else { io::stdout().is_terminal() };
            let tag = if colored { format!("\x1b[{color}m{tag}\x1b[0m") } else { tag };
            Box::new(move |text| {
                let _ = if to_stderr {
                    writeln!(io::stderr().lock(), "{tag} {text}")
                } else {
                    writeln!(io::stdout().lock(), "{tag} {text}")
                };
            })
        }
        Output::Channel { source, tx } => {
            let (source, tx) = (source.clone(), tx.clone());
            Box::new(move |text| {
                let line = LogLine { source: source.clone(), stream, line: text.to_owned() };
                let _ = tx.send(line);
            })
        }
        Output::File(_) => Box::new(|_| {}),
    }
}

/// Copy `from` line by line (lossy UTF-8, line ending removed) into `sink`,
/// until the stream ends.
fn copy_lines(
    from: impl Read + Send + 'static,
    mut sink: impl FnMut(&str) + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut from = BufReader::new(from);
        let mut line = Vec::new();
        while matches!(from.read_until(b'\n', &mut line), Ok(n) if n > 0) {
            let text = String::from_utf8_lossy(&line);
            sink(text.trim_end_matches(['\n', '\r']));
            line.clear();
        }
    })
}

/// SIGTERM every process still running, SIGKILL whatever outlives `GRACE`, and
/// reap them all.
fn terminate_all<'a>(procs: impl IntoIterator<Item = &'a mut Proc>) {
    let mut procs: Vec<&mut Proc> = procs.into_iter().collect();
    for p in procs.iter_mut() {
        if p.running() {
            p.signal(libc::SIGTERM);
        }
    }
    let deadline = Instant::now() + GRACE;
    while Instant::now() < deadline && procs.iter_mut().any(|p| p.running()) {
        std::thread::sleep(POLL);
    }
    for p in procs {
        p.reap();
    }
}

/// Start the rendered controller command lines, pointed at `endpoint`, into
/// `started`.
fn start_controllers(
    commands: &[String],
    endpoint: &str,
    log: &Log,
    registries: &Registries,
    started: &mut Vec<Proc>,
) -> Result<(), String> {
    for (k, line) in commands.iter().enumerate() {
        let argv = shlex::split(line)
            .filter(|argv| !argv.is_empty())
            .ok_or_else(|| format!("controller {}: cannot split `{line}`", k + 1))?;
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]).env("ASTRUM_ZENOH_CONNECT", endpoint);
        let output =
            log.output(&format!("ctl{}", k + 1), CONTROLLER_COLORS[k % CONTROLLER_COLORS.len()]);
        if !matches!(output, Output::File(_)) {
            // Live output: Python flushes every line, not only at exit.
            command.env("PYTHONUNBUFFERED", "1");
        }
        let p = Proc::spawn(format!("controller {}", k + 1), command, output, registries, None)
            .map_err(|e| format!("controller {}: starting `{}`: {e}", k + 1, argv[0]))?;
        started.push(p);
    }
    Ok(())
}

// --- Supervision --------------------------------------------------------------

/// How a run's processes ended, as far as its status goes.
struct Ending {
    /// The first failure: a controller exiting non-zero while the sim ran.
    problem: Option<String>,
    /// Why the sim was asked to stop: a stop request or the run timeout.
    stopped: Option<String>,
    /// The sim was asked to stop because of `problem` (fail-fast): however it
    /// exits then, `problem` is the run's.
    failed_fast: bool,
    sim: Option<ExitStatus>,
}

/// What [`supervise`] watches.
struct Supervision<'a> {
    /// The caller's stop handle and the run's own.
    shared: &'a StopRequested,
    own: &'a StopRequested,
    started: Instant,
    timeout: Option<Duration>,
    fail_fast: bool,
    log: &'a Log,
}

/// Wait for the sim to exit, asking it to stop on a stop request, after the
/// timeout and — with `fail_fast` — once a controller fails. Notes the first
/// controller failure, and reports a companion that exits.
fn supervise(
    sim: &mut Proc,
    controllers: &mut [Proc],
    companions: &mut [Proc],
    how: &Supervision,
) -> Ending {
    let mut problem: Option<String> = None;
    let mut stopped: Option<String> = None;
    let mut failed_fast = false;
    let mut companion_gone = vec![false; companions.len()];
    let sim_status = loop {
        match sim.exited() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                problem.get_or_insert(format!("waiting for the sim: {e}"));
                terminate_all([&mut *sim]);
                break sim.reap();
            }
        }
        for c in controllers.iter_mut() {
            if let Ok(Some(status)) = c.exited() {
                if !status.success() && problem.is_none() {
                    problem = Some(format!("{} exited ({status})", c.name));
                    if how.fail_fast {
                        failed_fast = true;
                        sim.signal(libc::SIGTERM);
                    }
                }
            }
        }
        for (c, gone) in companions.iter_mut().zip(&mut companion_gone) {
            if !*gone {
                if let Ok(Some(status)) = c.exited() {
                    *gone = true;
                    let msg = format!("{} exited ({status}); the run goes on", c.name);
                    how.log.say(LogStream::Stderr, msg);
                }
            }
        }
        if stopped.is_none() {
            if how.shared.requested() || how.own.requested() {
                stopped = Some("stop requested".into());
                sim.signal(libc::SIGTERM); // the sim ends its run and finishes the bag
            } else if let Some(t) = how.timeout.filter(|t| how.started.elapsed() > *t) {
                stopped = Some(format!("run_timeout_s {} exceeded", t.as_secs_f64()));
                sim.signal(libc::SIGTERM);
            }
        }
        std::thread::sleep(POLL);
    };
    Ending { problem, stopped, failed_fast, sim: sim_status }
}

/// A run's status from how it ended; `sim_log` hints where the sim's output is.
/// A sim that failed by itself (it exits 0 when asked to stop) is the run's
/// problem even if a controller exited first: a controller that quits on the
/// sim's `RunEnded(fatal)` must not hide the sim's reason. A sim stopped
/// because a controller failed (fail-fast) that then exits cleanly or dies of
/// that SIGTERM (sent before it could handle it) did not fail by itself: the
/// controller's failure is the run's.
fn status_of(ending: Ending, sim_log: &str) -> Status {
    let terminated = |s: &Option<ExitStatus>| {
        s.is_some_and(|s| s.success() || s.signal() == Some(libc::SIGTERM))
    };
    let fail_fast_stop = ending.failed_fast && terminated(&ending.sim);
    match (ending.problem, ending.stopped, ending.sim) {
        (Some(why), ..) if fail_fast_stop => Status::Failed(why),
        (_, None, Some(s)) if !s.success() => Status::Failed(format!("sim exited ({s}){sim_log}")),
        (Some(why), ..) => Status::Failed(why),
        (None, Some(why), _) => Status::Stopped(why),
        (None, None, Some(s)) if s.success() => Status::Ok,
        (None, None, Some(s)) => Status::Failed(format!("sim exited ({s}){sim_log}")),
        (None, None, None) => Status::Failed("the sim could not be reaped".into()),
    }
}

// --- LiveRun ------------------------------------------------------------------

/// A process started alongside the sim and stopped with it: its exit is
/// reported but never fails the run (the `--watch` viewer).
#[derive(Debug, Clone)]
pub struct Companion {
    /// For messages: "the viewer".
    pub name: String,
    /// Its output's source tag: "web".
    pub source: String,
    pub program: PathBuf,
    pub args: Vec<OsString>,
    /// Said (on stdout, from `astrum`) as soon as it has started, before the
    /// controllers start: "viewer: http://localhost:8080".
    pub announce: Option<String>,
}

/// Everything one [`LiveRun`] starts.
#[derive(Clone)]
pub struct LiveSpec {
    /// The `astrum-sim` binary.
    pub sim: PathBuf,
    /// The complete scenario the sim flies.
    pub scenario: Scenario,
    /// Where to write the scenario for the sim (kept); `None`: a temporary
    /// file, removed when the run ends.
    pub scenario_file: Option<PathBuf>,
    /// The Zenoh endpoint the sim listens on.
    pub listen: String,
    /// Have the sim record the run to this MCAP bag.
    pub record: Option<PathBuf>,
    /// Started after the sim, before the controllers.
    pub companions: Vec<Companion>,
    /// Controller command lines, placeholders already rendered; split like a
    /// shell would (no shell runs them).
    pub controllers: Vec<String>,
    /// Stop the run once it has taken this long (wall clock).
    pub run_timeout: Option<Duration>,
    /// Stop the run as soon as a controller fails (else it flies on and
    /// fails at the end).
    pub fail_fast: bool,
    pub log: Log,
    /// A stop handle the caller shares (a signal handler's, a sweep's): a
    /// request on it stops this run, and every other run sharing it. The run
    /// also has its own ([`LiveRun::stop`]).
    pub stop: StopRequested,
}

/// Why a [`LiveRun`] did not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartError {
    /// The scenario file could not be written or the sim could not start;
    /// nothing runs.
    Setup(String),
    /// A companion or controller could not start; what had started was
    /// stopped.
    Launch(String),
}

impl fmt::Display for StartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartError::Setup(why) | StartError::Launch(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for StartError {}

/// One run's processes, supervised on a thread of their own. Dropping an
/// unfinished run stops it (only it) and blocks until it is reaped.
pub struct LiveRun {
    endpoint: String,
    /// This run's own stop handle (the spec's is the caller's).
    own: StopRequested,
    /// This run's processes, for [`LiveRun::kill`].
    procs: Children,
    started: Instant,
    supervisor: Option<JoinHandle<Outcome>>,
    outcome: Option<Outcome>,
}

impl LiveRun {
    /// Write the scenario, start the sim, the companions and the controllers
    /// (each with `$ASTRUM_ZENOH_CONNECT` = [`LiveRun::endpoint`]), register
    /// them in `children` and return at once. If a companion or controller
    /// cannot start, those already started are stopped first (blocking up to
    /// the grace period).
    pub fn start(spec: LiveSpec, children: &Children) -> Result<Self, StartError> {
        let started = Instant::now();
        let (scenario_path, temp) = match &spec.scenario_file {
            Some(path) => {
                std::fs::write(path, spec.scenario.to_toml())
                    .map_err(|e| StartError::Setup(format!("writing {}: {e}", path.display())))?;
                (path.clone(), None)
            }
            None => {
                let dir = std::env::temp_dir();
                let mut temp = tempfile::Builder::new()
                    .prefix("astrum-run-")
                    .suffix(".toml")
                    .tempfile_in(&dir)
                    .map_err(|e| StartError::Setup(format!("writing {}: {e}", dir.display())))?;
                temp.write_all(spec.scenario.to_toml().as_bytes()).map_err(|e| {
                    StartError::Setup(format!("writing {}: {e}", temp.path().display()))
                })?;
                (temp.path().to_owned(), Some(temp))
            }
        };
        let endpoint = connect_endpoint(&spec.listen);
        let registries = Registries { shared: children.clone(), own: Children::new() };

        let mut sim_cmd = Command::new(&spec.sim);
        sim_cmd.arg(&scenario_path).args(["--listen", &spec.listen]);
        if let Some(bag) = &spec.record {
            sim_cmd.arg("--record").arg(bag);
        }
        let sim_time = SimTime::default();
        let output = spec.log.output("sim", SIM_COLOR);
        let mut sim = Proc::spawn("the sim", sim_cmd, output, &registries, Some(&sim_time))
            .map_err(|e| StartError::Setup(format!("starting {}: {e}", spec.sim.display())))?;
        let mut companions = Vec::new();
        let mut controllers = Vec::new();
        let launched = (|| {
            for c in &spec.companions {
                let mut cmd = Command::new(&c.program);
                cmd.args(&c.args).env("ASTRUM_ZENOH_CONNECT", &endpoint);
                let output = spec.log.output(&c.source, COMPANION_COLOR);
                let p = Proc::spawn(c.name.clone(), cmd, output, &registries, None)
                    .map_err(|e| format!("starting {}: {e}", c.name))?;
                companions.push(p);
                if let Some(msg) = &c.announce {
                    spec.log.say(LogStream::Stdout, msg.clone());
                }
            }
            start_controllers(
                &spec.controllers,
                &endpoint,
                &spec.log,
                &registries,
                &mut controllers,
            )
        })();
        if let Err(why) = launched {
            terminate_all(std::iter::once(&mut sim).chain(&mut companions).chain(&mut controllers));
            return Err(StartError::Launch(why));
        }

        let own = StopRequested::default();
        let own_stop = own.clone();
        let supervisor = std::thread::spawn(move || {
            let how = Supervision {
                shared: &spec.stop,
                own: &own_stop,
                started,
                timeout: spec.run_timeout,
                fail_fast: spec.fail_fast,
                log: &spec.log,
            };
            let ending = supervise(&mut sim, &mut controllers, &mut companions, &how);
            terminate_all(controllers.iter_mut().chain(&mut companions));
            sim.reap();
            drop(temp);
            let (sim_time_s, hint) = match &spec.log {
                Log::Files { dir, stem } => (
                    std::fs::read_to_string(dir.join(format!("{stem}.sim.log")))
                        .ok()
                        .and_then(|log| parse_stopped_line(&log)),
                    format!("; see {stem}.sim.log"),
                ),
                _ => (*sim_time.lock().unwrap_or_else(|e| e.into_inner()), String::new()),
            };
            let status = status_of(ending, &hint);
            Outcome { status, sim_time_s, wall_s: started.elapsed().as_secs_f64() }
        });
        Ok(Self {
            endpoint,
            own,
            procs: registries.own,
            started,
            supervisor: Some(supervisor),
            outcome: None,
        })
    }

    /// Where a client on this host reaches the run's sim.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The run's own stop handle: a request on it stops this run only.
    pub fn stop_handle(&self) -> StopRequested {
        self.own.clone()
    }

    /// Ask this run (only) to stop: the sim ends cleanly, then everything
    /// else. [`LiveRun::wait`] kills what is left after the grace period.
    pub fn stop(&self) {
        self.own.0.store(true, Ordering::Relaxed);
    }

    /// SIGKILL this run's processes (and the process groups they lead) at
    /// once; the run then ends, failed unless it was stopped.
    pub fn kill(&self) {
        self.procs.kill_all();
    }

    /// The run's outcome once every process has been reaped; `None` while it
    /// runs. Never blocks.
    pub fn poll(&mut self) -> Option<&Outcome> {
        if self.supervisor.as_ref().is_some_and(JoinHandle::is_finished) {
            self.outcome = Some(self.join());
        }
        self.outcome.as_ref()
    }

    /// Block until the run has ended and every process is reaped. Once this
    /// run's own stop has been requested ([`LiveRun::stop`]), whatever still
    /// runs after the grace period is killed; a stop through the shared
    /// handle leaves escalation to the caller (e.g. a second Ctrl-C).
    pub fn wait(mut self) -> Outcome {
        self.finish()
    }

    fn finish(&mut self) -> Outcome {
        if let Some(outcome) = self.outcome.take() {
            return outcome;
        }
        let mut deadline = None;
        let mut killed = false;
        while !self.supervisor.as_ref().is_none_or(JoinHandle::is_finished) {
            if self.own.requested() && !killed {
                let at = *deadline.get_or_insert_with(|| Instant::now() + GRACE);
                if Instant::now() >= at {
                    self.kill();
                    killed = true;
                }
            }
            std::thread::sleep(POLL);
        }
        self.join()
    }

    fn join(&mut self) -> Outcome {
        let panicked = || Outcome {
            status: Status::Failed("the run's supervisor panicked".into()),
            sim_time_s: None,
            wall_s: self.started.elapsed().as_secs_f64(),
        };
        match self.supervisor.take().map(JoinHandle::join) {
            Some(Ok(outcome)) => outcome,
            Some(Err(_)) | None => panicked(),
        }
    }
}

impl Drop for LiveRun {
    fn drop(&mut self) {
        if self.supervisor.is_some() {
            self.stop();
            self.finish();
        }
    }
}

// --- Sweep process mode -------------------------------------------------------

/// What every process-mode run of a sweep shares.
pub struct ProcessCtx<'a> {
    pub sweep: &'a Sweep,
    /// The sweep's output directory: every run's scenario, bag and logs.
    pub out: &'a Path,
    /// The `astrum-sim` binary.
    pub sim: &'a Path,
    /// Run N listens on `tcp/127.0.0.1:<base_port + N - 1>`.
    pub base_port: u16,
    pub stop: &'a StopRequested,
    pub children: &'a Children,
}

/// One sweep run as processes: the sim on its own endpoint, recording to
/// `<out>/<stem>.mcap`, plus the sweep's controllers pointed at it, each
/// logging to `<out>/<stem>.<source>.log`. A failing controller fails the run
/// at its end; the run stops on the sweep's stop request.
pub fn run_processes(ctx: &ProcessCtx, run: &Run) -> Outcome {
    let started = Instant::now();
    let stem = run.stem();
    let endpoint = format!("tcp/127.0.0.1:{}", u32::from(ctx.base_port) + run.index as u32);
    let spec = LiveSpec {
        sim: ctx.sim.to_owned(),
        scenario: run.scenario.clone(),
        scenario_file: Some(ctx.out.join(format!("{stem}.toml"))),
        record: Some(ctx.out.join(format!("{stem}.mcap"))),
        companions: Vec::new(),
        controllers: ctx.sweep.file.controllers.iter().map(|c| run.render(c, &endpoint)).collect(),
        listen: endpoint,
        run_timeout: run_timeout(ctx.sweep.file.run_timeout_s),
        fail_fast: false,
        log: Log::Files { dir: ctx.out.to_owned(), stem },
        stop: ctx.stop.clone(),
    };
    match LiveRun::start(spec, ctx.children) {
        Ok(live) => live.wait(),
        Err(why) => Outcome {
            status: Status::Failed(why.to_string()),
            sim_time_s: None,
            wall_s: started.elapsed().as_secs_f64(),
        },
    }
}
