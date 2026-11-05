//! `astrum`: ASTRUM's command line. Fly a scenario or run file live (`run`),
//! many runs at once (`sweep`), list what a directory holds (`list`), and reach
//! the other tools: `ctl`, `web`, `record`, `play` and `export` become the
//! sibling binary (`astrum-ctl`, `astrum-web`, `astrum-bag-record`,
//! `astrum-bag-play`, `astrum-web-export`) with the same arguments.
//!
//! `run` starts `astrum-sim`, the viewer (`--watch`) and the run file's
//! controllers as child processes and streams their output, every line tagged
//! with its source. The sim decides the run: it ends with the scenario's
//! `max_time_s` or on Ctrl-C (the sim stops cleanly, then everything else); a
//! controller that fails meanwhile stops the run and fails it.
//!
//! `sweep` flies many runs at once. Without `controllers` in the sweep file,
//! runs fly in-process (one thread each). With them, each run is an
//! `astrum-sim` process on its own Zenoh endpoint plus the listed controller
//! processes, each logging to a file. Ctrl-C stops starting runs and ends the
//! running ones cleanly; the summary is written either way.
//!
//! Exit status: 0 ok, 1 a run not ok, 2 invalid input; a `run` stopped by
//! Ctrl-C exits 130.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, ExitStatus, Stdio};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astrum_sim::sweep::{
    execute, parse_stopped_line, run_in_process, write_summary, Outcome, Purpose, Run, Status,
    Sweep, SweepFile,
};
use astrum_sim::systems::StopRequested;
use clap::{Args, Parser, Subcommand};

/// ASTRUM's command line: fly scenarios, run files and sweeps, and reach the
/// other ASTRUM tools.
#[derive(Parser)]
#[command(name = "astrum", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fly one scenario or run file live: the sim, its controllers and
    /// (--watch) the 3-D viewer, their output on this terminal.
    Run(RunArgs),
    /// Fly a sweep of scenario variants, many at once.
    Sweep(SweepArgs),
    /// List the scenarios, run files and sweeps under a directory.
    List {
        /// Directory to search, recursively.
        #[arg(default_value = "config")]
        dir: PathBuf,
    },
    /// Control a running sim: status, pause, resume, step, speed (astrum-ctl).
    #[command(disable_help_flag = true)]
    Ctl(Forward),
    /// Serve the 3-D viewer for a live sim or a replay (astrum-web).
    #[command(disable_help_flag = true)]
    Web(Forward),
    /// Record a running sim to an MCAP bag (astrum-bag-record).
    #[command(disable_help_flag = true)]
    Record(Forward),
    /// Replay a bag as a stand-in for the sim (astrum-bag-play).
    #[command(disable_help_flag = true)]
    Play(Forward),
    /// Bake a bag into a replay JSON the browser opens offline (astrum-web-export).
    #[command(disable_help_flag = true)]
    Export(Forward),
}

// The tool's arguments, passed on as they are (`--help` shows the tool's own).
#[derive(Args)]
struct Forward {
    /// Arguments for the tool.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<OsString>,
}

#[derive(Args)]
struct RunArgs {
    /// Scenario or run file (TOML; see `astrum list`).
    file: PathBuf,

    /// Override a value: sim.<field>, world.<field>, aircraft.<id>.<field> or
    /// param.<name>, as TOML (`200`, `true`, `"f16"`; a bare word is a
    /// string). Repeatable.
    #[arg(long = "set", value_name = "KEY=VALUE", value_parser = parse_override)]
    set: Vec<(String, toml::Value)>,

    /// Pace as a multiple of real time (0 = as fast as possible), over the run
    /// file's `speed` and the scenario's `sim.speed`.
    #[arg(long)]
    speed: Option<f64>,

    /// Record the run to this MCAP bag (in the sim's process: lossless).
    #[arg(long, value_name = "BAG")]
    record: Option<PathBuf>,

    /// Also serve the 3-D viewer (astrum-web) for the run.
    #[arg(long)]
    watch: bool,

    /// Zenoh endpoint the sim listens on.
    #[arg(long, env = "ASTRUM_ZENOH_LISTEN", default_value = astrum_net::DEFAULT_LISTEN_ENDPOINT)]
    listen: String,

    /// Check the file and print the resolved scenario (and the controller
    /// commands) without flying it.
    #[arg(long)]
    dry_run: bool,

    /// The astrum-sim binary (default: next to this one).
    #[arg(long, env = "ASTRUM_SIM_BIN")]
    sim: Option<PathBuf>,
}

#[derive(Args)]
struct SweepArgs {
    /// Sweep file (TOML; see config/sweep.*.toml).
    sweep: PathBuf,

    /// Runs at once, overriding the sweep file's `jobs` (0 = one per core).
    #[arg(long, env = "ASTRUM_SWEEP_JOBS")]
    jobs: Option<usize>,

    /// Output directory, overriding the sweep file's `out`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Replace an earlier sweep's results in the output directory.
    #[arg(long)]
    overwrite: bool,

    /// Plan the sweep and print its runs without flying them.
    #[arg(long)]
    dry_run: bool,

    /// Process mode: run N listens on tcp/127.0.0.1:<base-port + N - 1>.
    #[arg(long, default_value_t = 27500)]
    base_port: u16,

    /// Process mode: the astrum-sim binary (default: next to this one).
    #[arg(long, env = "ASTRUM_SIM_BIN")]
    sim: Option<PathBuf>,
}

/// How long processes get to exit after a stop before they are killed.
const GRACE: Duration = Duration::from_secs(5);
const POLL: Duration = Duration::from_millis(10);
/// ANSI colours of the `[ctlN]` tags, in turn (`[sim]` cyan, `[web]` magenta).
const CONTROLLER_COLORS: [&str; 4] = ["32", "33", "34", "91"];

fn main() -> ExitCode {
    match Cli::parse().cmd {
        Cmd::Run(args) => run(&args),
        Cmd::Sweep(args) => sweep(args),
        Cmd::List { dir } => list(&dir),
        Cmd::Ctl(f) => exec_tool("astrum-ctl", &f.args),
        Cmd::Web(f) => exec_tool("astrum-web", &f.args),
        Cmd::Record(f) => exec_tool("astrum-bag-record", &f.args),
        Cmd::Play(f) => exec_tool("astrum-bag-play", &f.args),
        Cmd::Export(f) => exec_tool("astrum-web-export", &f.args),
    }
}

/// The ASTRUM binary `name` next to this one.
fn sibling(name: &str) -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .unwrap_or_else(|| name.into())
}

/// Become the sibling tool `name` with `args` (its exit status is ours).
fn exec_tool(name: &str, args: &[OsString]) -> ExitCode {
    let bin = sibling(name);
    let e = Command::new(&bin).args(args).exec();
    eprintln!("[astrum] cannot run {}: {e}", bin.display());
    ExitCode::from(2)
}

/// `KEY=VALUE`, the value as TOML (else as a plain string).
fn parse_override(s: &str) -> Result<(String, toml::Value), String> {
    let (key, raw) = s.split_once('=').ok_or("expected KEY=VALUE")?;
    let (key, raw) = (key.trim(), raw.trim());
    if key.is_empty() {
        return Err("expected KEY=VALUE".into());
    }
    let value = toml::from_str::<toml::Table>(&format!("v = {raw}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(raw.to_owned()));
    Ok((key.to_owned(), value))
}

// --- Child processes ----------------------------------------------------------

/// Every child not yet reaped — pid, and whether it leads its own process
/// group — so a second Ctrl-C can kill them all before giving up.
static CHILDREN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());

fn children() -> std::sync::MutexGuard<'static, Vec<(u32, bool)>> {
    CHILDREN.lock().unwrap_or_else(|e| e.into_inner())
}

fn send(pid: u32, group: bool, sig: libc::c_int) {
    let pid = pid as libc::pid_t;
    // SAFETY: kill(2) on a child (or the process group it leads) that we
    // spawned and have not reaped.
    unsafe {
        libc::kill(if group { -pid } else { pid }, sig);
    }
}

/// First Ctrl-C (or SIGTERM): request a clean stop and say `what`. Second: kill
/// every child and exit 130.
fn on_ctrl_c(stop: &StopRequested, what: &'static str) -> Result<(), ctrlc::Error> {
    let flag = stop.0.clone();
    ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::Relaxed) {
            for &(pid, group) in children().iter() {
                send(pid, group, libc::SIGKILL);
            }
            std::process::exit(130); // second signal: give up on a clean stop
        }
        eprintln!("{what}");
    })
}

/// Where a child's output goes.
enum Output {
    /// Into a log file (sweeps).
    File(PathBuf),
    /// Onto this terminal, every line tagged `[label]` in an ANSI colour
    /// (`astrum run`). The child leads its own process group, so the
    /// terminal's Ctrl-C reaches only `astrum`, which stops the run in order.
    Tagged(String, &'static str),
}

/// A spawned child and the threads copying its output to the terminal.
struct Proc {
    /// For messages: "the sim", "controller 2", "the viewer".
    name: String,
    child: Child,
    group: bool,
    copiers: Vec<JoinHandle<()>>,
}

impl Proc {
    fn spawn(name: impl Into<String>, mut cmd: Command, output: Output) -> io::Result<Self> {
        cmd.stdin(Stdio::null());
        let group = matches!(output, Output::Tagged(..));
        match &output {
            Output::File(path) => {
                let log = File::create(path)?;
                cmd.stdout(log.try_clone()?).stderr(log);
            }
            Output::Tagged(..) => {
                cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).process_group(0);
            }
        }
        let mut child = cmd.spawn()?;
        children().push((child.id(), group));
        let mut copiers = Vec::new();
        if let Output::Tagged(label, color) = output {
            if let Some(out) = child.stdout.take() {
                copiers.push(copy_tagged(out, &label, color, false));
            }
            if let Some(err) = child.stderr.take() {
                copiers.push(copy_tagged(err, &label, color, true));
            }
        }
        Ok(Self { name: name.into(), child, group, copiers })
    }

    fn signal(&self, sig: libc::c_int) {
        send(self.child.id(), self.group, sig);
    }

    fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Kill the child if it still runs, wait for it, and flush its output.
    fn reap(&mut self) -> Option<ExitStatus> {
        if self.running() {
            self.signal(libc::SIGKILL);
        }
        let status = self.child.wait().ok();
        let pid = self.child.id();
        children().retain(|&(p, _)| p != pid);
        for copier in self.copiers.drain(..) {
            let _ = copier.join();
        }
        status
    }
}

/// Copy `from` line by line onto our stdout (or stderr), each line tagged.
fn copy_tagged(
    from: impl Read + Send + 'static,
    label: &str,
    color: &str,
    to_stderr: bool,
) -> JoinHandle<()> {
    let tag = format!("{:<6}", format!("[{label}]"));
    let colored = if to_stderr { io::stderr().is_terminal() } else { io::stdout().is_terminal() };
    let tag = if colored { format!("\x1b[{color}m{tag}\x1b[0m") } else { tag };
    std::thread::spawn(move || {
        let mut from = BufReader::new(from);
        let mut line = Vec::new();
        while matches!(from.read_until(b'\n', &mut line), Ok(n) if n > 0) {
            let text = String::from_utf8_lossy(&line);
            let text = text.trim_end_matches(['\n', '\r']);
            let _ = if to_stderr {
                writeln!(io::stderr().lock(), "{tag} {text}")
            } else {
                writeln!(io::stdout().lock(), "{tag} {text}")
            };
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

/// Start `run`'s controller commands, pointed at `endpoint`, into `started`.
fn start_controllers(
    commands: &[String],
    run: &Run,
    endpoint: &str,
    output: impl Fn(usize) -> Output,
    started: &mut Vec<Proc>,
) -> Result<(), String> {
    for (k, cmd) in commands.iter().enumerate() {
        let rendered = run.render(cmd, endpoint);
        let argv = shlex::split(&rendered)
            .filter(|argv| !argv.is_empty())
            .ok_or_else(|| format!("controller {}: cannot split `{rendered}`", k + 1))?;
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]).env("ASTRUM_ZENOH_CONNECT", endpoint);
        let output = output(k);
        if matches!(output, Output::Tagged(..)) {
            // Live output: Python flushes every line, not only at exit.
            command.env("PYTHONUNBUFFERED", "1");
        }
        let p = Proc::spawn(format!("controller {}", k + 1), command, output)
            .map_err(|e| format!("controller {}: starting `{}`: {e}", k + 1, argv[0]))?;
        started.push(p);
    }
    Ok(())
}

/// How a run's processes ended, as far as its status goes.
struct Ending {
    /// The first failure: a controller exiting non-zero while the sim ran.
    problem: Option<String>,
    /// Why the sim was asked to stop: a stop request or the run timeout.
    stopped: Option<String>,
    sim: Option<ExitStatus>,
}

/// Wait for the sim to exit, asking it to stop on a stop request, after
/// `timeout` (from `started`) and — with `fail_fast` — once a controller
/// fails. Notes the first controller failure, and reports a viewer that exits.
fn supervise(
    sim: &mut Proc,
    controllers: &mut [Proc],
    viewers: &mut [Proc],
    stop: &StopRequested,
    started: Instant,
    timeout: Option<Duration>,
    fail_fast: bool,
) -> Ending {
    let mut problem: Option<String> = None;
    let mut stopped: Option<String> = None;
    let mut viewer_gone = vec![false; viewers.len()];
    let sim_status = loop {
        match sim.child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                problem.get_or_insert(format!("waiting for the sim: {e}"));
                terminate_all([&mut *sim]);
                break sim.reap();
            }
        }
        for c in controllers.iter_mut() {
            if let Ok(Some(status)) = c.child.try_wait() {
                if !status.success() && problem.is_none() {
                    problem = Some(format!("{} exited ({status})", c.name));
                    if fail_fast {
                        sim.signal(libc::SIGTERM);
                    }
                }
            }
        }
        for (v, gone) in viewers.iter_mut().zip(&mut viewer_gone) {
            if !*gone {
                if let Ok(Some(status)) = v.child.try_wait() {
                    *gone = true;
                    eprintln!("[astrum] {} exited ({status}); the run goes on", v.name);
                }
            }
        }
        if stopped.is_none() {
            if stop.requested() {
                stopped = Some("stop requested".into());
                sim.signal(libc::SIGTERM); // the sim ends its run and finishes the bag
            } else if let Some(t) = timeout.filter(|t| started.elapsed() > *t) {
                stopped = Some(format!("run_timeout_s {} exceeded", t.as_secs_f64()));
                sim.signal(libc::SIGTERM);
            }
        }
        std::thread::sleep(POLL);
    };
    Ending { problem, stopped, sim: sim_status }
}

/// A run's status from how it ended; `sim_log` hints where the sim's output is.
fn status_of(ending: Ending, sim_log: &str) -> Status {
    match (ending.problem, ending.stopped, ending.sim) {
        (Some(why), ..) => Status::Failed(why),
        (None, Some(why), _) => Status::Stopped(why),
        (None, None, Some(s)) if s.success() => Status::Ok,
        (None, None, Some(s)) => Status::Failed(format!("sim exited ({s}){sim_log}")),
        (None, None, None) => Status::Failed("the sim could not be reaped".into()),
    }
}

// --- astrum run ---------------------------------------------------------------

/// A file removed when dropped.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem().map_or("run".into(), |s| s.to_string_lossy().into_owned())
}

/// Plan `args.file` — a run file, or a bare scenario — as one live run.
fn plan_run(args: &RunArgs) -> Result<Sweep, String> {
    let path = &args.file;
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let parse_err = |e: toml::de::Error| format!("parsing {}: {e}", path.display());
    let table: toml::Table = toml::from_str(&text).map_err(parse_err)?;
    // A scenario never has `base` (unknown keys are errors there).
    let mut file = if table.contains_key("base") {
        toml::from_str::<SweepFile>(&text).map_err(parse_err)?
    } else {
        SweepFile::for_scenario(path.clone())
    };
    file.set.extend(args.set.iter().cloned());
    if args.speed.is_some() {
        file.speed = args.speed;
    }
    let sweep = Sweep::plan(file, &file_stem(path), Purpose::Run)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    match sweep.runs.len() {
        1 => Ok(sweep),
        n => Err(format!(
            "{} plans {n} runs; `astrum run` flies one (use `astrum sweep`)",
            path.display()
        )),
    }
}

/// Where a client on this host reaches a sim listening on `listen`: a wildcard
/// address (`0.0.0.0`, `[::]`) becomes loopback.
fn connect_endpoint(listen: &str) -> String {
    listen.replacen("/0.0.0.0:", "/127.0.0.1:", 1).replacen("/[::]:", "/[::1]:", 1)
}

/// The viewer's URL: astrum-web binds `$ASTRUM_WEB_BIND` (default 127.0.0.1:8080).
fn viewer_url() -> String {
    let bind = std::env::var("ASTRUM_WEB_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let port = bind.rsplit_once(':').map_or("8080", |(_, port)| port);
    format!("http://localhost:{port}")
}

fn run(args: &RunArgs) -> ExitCode {
    let sweep = match plan_run(args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[astrum] {e}");
            return ExitCode::from(2);
        }
    };
    let run = &sweep.runs[0];
    let endpoint = connect_endpoint(&args.listen);
    if args.dry_run {
        for (k, cmd) in sweep.file.controllers.iter().enumerate() {
            println!("# controller {}: {}", k + 1, run.render(cmd, &endpoint));
        }
        print!("{}", run.scenario.to_toml());
        return ExitCode::SUCCESS;
    }
    let sim_bin = args.sim.clone().unwrap_or_else(|| sibling("astrum-sim"));
    if !sim_bin.exists() {
        eprintln!("[astrum] no astrum-sim at {} (build it, or pass --sim)", sim_bin.display());
        return ExitCode::from(2);
    }
    // The sim reads the resolved scenario (overrides applied) from a file.
    let scenario =
        TempFile(std::env::temp_dir().join(format!("astrum-run-{}.toml", std::process::id())));
    if let Err(e) = std::fs::write(&scenario.0, run.scenario.to_toml()) {
        eprintln!("[astrum] writing {}: {e}", scenario.0.display());
        return ExitCode::FAILURE;
    }
    let stop = StopRequested::default();
    if let Err(e) = on_ctrl_c(&stop, "[astrum] stopping the run (Ctrl-C again to kill it)") {
        eprintln!("[astrum] cannot install the signal handler: {e}");
        return ExitCode::FAILURE;
    }

    let started = Instant::now();
    let mut sim_cmd = Command::new(&sim_bin);
    sim_cmd.arg(&scenario.0).args(["--listen", &args.listen]);
    if let Some(bag) = &args.record {
        sim_cmd.arg("--record").arg(bag);
    }
    let mut sim = match Proc::spawn("the sim", sim_cmd, Output::Tagged("sim".into(), "36")) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[astrum] starting {}: {e}", sim_bin.display());
            return ExitCode::FAILURE;
        }
    };
    let mut viewers = Vec::new();
    let mut controllers = Vec::new();
    let launched = launch(args, &sweep, &endpoint, &mut viewers, &mut controllers);
    if let Err(why) = launched {
        terminate_all(std::iter::once(&mut sim).chain(&mut viewers).chain(&mut controllers));
        eprintln!("[astrum] FAILED: {}: {why}", args.file.display());
        return ExitCode::FAILURE;
    }
    let timeout =
        (sweep.file.run_timeout_s > 0.0).then(|| Duration::from_secs_f64(sweep.file.run_timeout_s));
    let ending = supervise(&mut sim, &mut controllers, &mut viewers, &stop, started, timeout, true);
    terminate_all(controllers.iter_mut().chain(&mut viewers));
    sim.reap();
    let file = args.file.display();
    match status_of(ending, "") {
        Status::Ok => {
            println!("[astrum] ok: {file}");
            ExitCode::SUCCESS
        }
        Status::Failed(why) => {
            eprintln!("[astrum] FAILED: {file}: {why}");
            ExitCode::FAILURE
        }
        Status::Stopped(why) => {
            eprintln!("[astrum] stopped: {file}: {why}");
            ExitCode::from(130)
        }
    }
}

/// Start what joins the sim: the viewer (`--watch`), then the controllers.
fn launch(
    args: &RunArgs,
    sweep: &Sweep,
    endpoint: &str,
    viewers: &mut Vec<Proc>,
    controllers: &mut Vec<Proc>,
) -> Result<(), String> {
    if args.watch {
        let mut web = Command::new(sibling("astrum-web"));
        web.env("ASTRUM_ZENOH_CONNECT", endpoint);
        let viewer = Proc::spawn("the viewer", web, Output::Tagged("web".into(), "35"))
            .map_err(|e| format!("starting the viewer: {e}"))?;
        viewers.push(viewer);
        println!("[astrum] viewer: {}", viewer_url());
    }
    let color = |k: usize| CONTROLLER_COLORS[k % CONTROLLER_COLORS.len()];
    start_controllers(
        &sweep.file.controllers,
        &sweep.runs[0],
        endpoint,
        |k| Output::Tagged(format!("ctl{}", k + 1), color(k)),
        controllers,
    )
}

// --- astrum sweep -------------------------------------------------------------

struct ProcessCtx<'a> {
    sweep: &'a Sweep,
    out: &'a Path,
    sim: &'a Path,
    base_port: u16,
    stop: &'a StopRequested,
}

/// One run as processes: the sim on its own endpoint, recording, plus the
/// sweep's controllers pointed at it.
fn run_processes(ctx: &ProcessCtx, run: &Run) -> Outcome {
    let started = Instant::now();
    let failed = |why: String| Outcome {
        status: Status::Failed(why),
        sim_time_s: None,
        wall_s: started.elapsed().as_secs_f64(),
    };
    let stem = run.stem();
    let scenario_path = ctx.out.join(format!("{stem}.toml"));
    if let Err(e) = std::fs::write(&scenario_path, run.scenario.to_toml()) {
        return failed(format!("writing {}: {e}", scenario_path.display()));
    }
    let endpoint = format!("tcp/127.0.0.1:{}", u32::from(ctx.base_port) + run.index as u32);
    let log = |name: &str| Output::File(ctx.out.join(format!("{stem}.{name}.log")));
    let mut sim_cmd = Command::new(ctx.sim);
    sim_cmd
        .arg(&scenario_path)
        .args(["--listen", &endpoint, "--record"])
        .arg(ctx.out.join(format!("{stem}.mcap")));
    let mut sim = match Proc::spawn("the sim", sim_cmd, log("sim")) {
        Ok(p) => p,
        Err(e) => return failed(format!("starting {}: {e}", ctx.sim.display())),
    };
    let mut controllers = Vec::new();
    let commands = &ctx.sweep.file.controllers;
    let logs = |k: usize| log(&format!("ctl{}", k + 1));
    if let Err(why) = start_controllers(commands, run, &endpoint, logs, &mut controllers) {
        terminate_all(std::iter::once(&mut sim).chain(&mut controllers));
        return failed(why);
    }

    let timeout = (ctx.sweep.file.run_timeout_s > 0.0)
        .then(|| Duration::from_secs_f64(ctx.sweep.file.run_timeout_s));
    let ending = supervise(&mut sim, &mut controllers, &mut [], ctx.stop, started, timeout, false);
    terminate_all(&mut controllers);
    sim.reap();
    let sim_time_s = std::fs::read_to_string(ctx.out.join(format!("{stem}.sim.log")))
        .ok()
        .and_then(|log| parse_stopped_line(&log));
    let status = status_of(ending, &format!("; see {stem}.sim.log"));
    Outcome { status, sim_time_s, wall_s: started.elapsed().as_secs_f64() }
}

fn sweep(args: SweepArgs) -> ExitCode {
    let mut sweep = match Sweep::load(&args.sweep) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[sweep] {e}");
            return ExitCode::from(2);
        }
    };
    if let Some(out) = args.out {
        sweep.out = out;
    }
    let jobs = sweep.jobs(args.jobs);
    let mode = if sweep.in_process() { "in-process" } else { "processes" };
    println!(
        "[sweep] {} runs ({mode}), {jobs} at once → {}",
        sweep.runs.len(),
        sweep.out.display()
    );
    if !sweep.in_process()
        && usize::from(args.base_port) + sweep.runs.len() > usize::from(u16::MAX) + 1
    {
        eprintln!(
            "[sweep] --base-port {} leaves too few ports for {} runs",
            args.base_port,
            sweep.runs.len()
        );
        return ExitCode::from(2);
    }
    if args.dry_run {
        for run in &sweep.runs {
            let values: Vec<String> = sweep
                .vary_keys
                .iter()
                .zip(&run.values)
                .map(|(k, v)| format!("{k} = {v}"))
                .collect();
            println!("  {} seed {}: {}", run.stem(), run.scenario.sim.seed, values.join(", "));
        }
        return ExitCode::SUCCESS;
    }
    let summary = sweep.out.join("summary.csv");
    if summary.exists() && !args.overwrite {
        eprintln!("[sweep] {} exists; pass --overwrite to replace it", summary.display());
        return ExitCode::from(2);
    }
    if let Err(e) = std::fs::create_dir_all(&sweep.out) {
        eprintln!("[sweep] creating {}: {e}", sweep.out.display());
        return ExitCode::FAILURE;
    }
    let sim = args.sim.unwrap_or_else(|| sibling("astrum-sim"));
    if !sweep.in_process() && !sim.exists() {
        eprintln!("[sweep] no astrum-sim at {} (build it, or pass --sim)", sim.display());
        return ExitCode::from(2);
    }

    let stop = StopRequested::default();
    if let Err(e) = on_ctrl_c(&stop, "[sweep] stop requested: finishing the running runs") {
        eprintln!("[sweep] cannot install the signal handler: {e}");
        return ExitCode::FAILURE;
    }

    let started = Instant::now();
    let outcomes = if sweep.in_process() {
        execute(&sweep, jobs, &stop, true, |run| {
            run_in_process(run, &sweep.out.join(format!("{}.mcap", run.stem())), &stop)
        })
    } else {
        let ctx = ProcessCtx {
            sweep: &sweep,
            out: &sweep.out,
            sim: &sim,
            base_port: args.base_port,
            stop: &stop,
        };
        execute(&sweep, jobs, &stop, true, |run| run_processes(&ctx, run))
    };
    let wall = started.elapsed().as_secs_f64();
    if let Err(e) = write_summary(&summary, &sweep, &outcomes) {
        eprintln!("[sweep] writing {}: {e}", summary.display());
        return ExitCode::FAILURE;
    }
    let ok = outcomes.iter().filter(|o| o.as_ref().is_some_and(|o| o.status == Status::Ok)).count();
    let sim_total: f64 = outcomes.iter().flatten().filter_map(|o| o.sim_time_s).sum();
    println!(
        "[sweep] {ok}/{} runs ok in {wall:.1} s ({:.0} sim-s per wall-s) → {}",
        sweep.runs.len(),
        sim_total / wall,
        summary.display()
    );
    if ok == sweep.runs.len() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

// --- astrum list --------------------------------------------------------------

/// What a TOML file is, in listing order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Run,
    Scenario,
    Sweep,
    Invalid,
}

impl Kind {
    /// `None`: some other TOML file (not ASTRUM's).
    fn of(table: &toml::Table) -> Option<Self> {
        if table.contains_key("base") {
            let grid = table
                .get("vary")
                .and_then(toml::Value::as_table)
                .is_some_and(|v| !v.is_empty())
                || table.get("repeats").and_then(toml::Value::as_integer).is_some_and(|r| r > 1);
            Some(if grid { Kind::Sweep } else { Kind::Run })
        } else if table.contains_key("aircraft") {
            Some(Kind::Scenario)
        } else {
            None
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Run => "run",
            Kind::Scenario => "scenario",
            Kind::Sweep => "sweep",
            Kind::Invalid => "invalid",
        }
    }
}

fn find_toml(dir: &Path, found: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            find_toml(&path, found)?;
        } else if path.extension().is_some_and(|e| e == "toml") {
            found.push(path);
        }
    }
    Ok(())
}

/// The first sentence of the comment block a file starts with.
fn summary(text: &str) -> String {
    let mut words = String::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() && words.is_empty() {
            continue;
        }
        let Some(comment) = line.strip_prefix('#').map(str::trim) else { break };
        if comment.is_empty() {
            if words.is_empty() {
                continue;
            }
            break;
        }
        if !words.is_empty() {
            words.push(' ');
        }
        words.push_str(comment);
    }
    let end = words
        .char_indices()
        .find(|&(i, c)| matches!(c, '.' | '?' | '!') && words[i + 1..].starts_with(' '))
        .map_or(words.len(), |(i, _)| i + 1);
    words.truncate(end);
    words
}

fn list(dir: &Path) -> ExitCode {
    let mut paths = Vec::new();
    if let Err(e) = find_toml(dir, &mut paths) {
        eprintln!("[astrum] reading {}: {e}", dir.display());
        return ExitCode::from(2);
    }
    let mut rows: Vec<(Kind, String, String)> = Vec::new();
    for path in paths {
        let shown = path.display().to_string();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                rows.push((Kind::Invalid, shown, e.to_string()));
                continue;
            }
        };
        match toml::from_str::<toml::Table>(&text) {
            Ok(table) => {
                if let Some(kind) = Kind::of(&table) {
                    rows.push((kind, shown, summary(&text)));
                }
            }
            Err(e) => {
                let why = e.message().to_owned();
                rows.push((Kind::Invalid, shown, why));
            }
        }
    }
    rows.sort();
    let width = rows.iter().map(|(_, p, _)| p.len()).max().unwrap_or(0);
    for (kind, path, about) in &rows {
        println!("{:<8}  {path:<width$}  {about}", kind.name());
    }
    ExitCode::SUCCESS
}
