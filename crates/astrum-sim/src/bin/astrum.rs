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
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::Ordering;
use std::time::Instant;

use astrum_sim::catalog;
use astrum_sim::launch::{
    connect_endpoint, run_processes, run_timeout, sibling, Children, Companion, LiveRun, LiveSpec,
    Log, ProcessCtx,
};
use astrum_sim::sweep::{
    execute, run_in_process, write_summary, Purpose, Status, Sweep, SweepFile,
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

// --- Signals ------------------------------------------------------------------

/// First Ctrl-C (or SIGTERM): request a clean stop and say `what`. Second: kill
/// every child in `children` and exit 130.
fn on_ctrl_c(
    stop: &StopRequested,
    children: &Children,
    what: &'static str,
) -> Result<(), ctrlc::Error> {
    let flag = stop.0.clone();
    let children = children.clone();
    ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::Relaxed) {
            children.kill_all();
            std::process::exit(130); // second signal: give up on a clean stop
        }
        eprintln!("{what}");
    })
}

// --- astrum run ---------------------------------------------------------------

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
    let controllers: Vec<String> =
        sweep.file.controllers.iter().map(|cmd| run.render(cmd, &endpoint)).collect();
    if args.dry_run {
        for (k, line) in controllers.iter().enumerate() {
            println!("# controller {}: {line}", k + 1);
        }
        print!("{}", run.scenario.to_toml());
        return ExitCode::SUCCESS;
    }
    let sim_bin = args.sim.clone().unwrap_or_else(|| sibling("astrum-sim"));
    if !sim_bin.exists() {
        eprintln!("[astrum] no astrum-sim at {} (build it, or pass --sim)", sim_bin.display());
        return ExitCode::from(2);
    }
    let stop = StopRequested::default();
    let children = Children::new();
    let what = "[astrum] stopping the run (Ctrl-C again to kill it)";
    if let Err(e) = on_ctrl_c(&stop, &children, what) {
        eprintln!("[astrum] cannot install the signal handler: {e}");
        return ExitCode::FAILURE;
    }

    let companions = if args.watch {
        vec![Companion {
            name: "the viewer".into(),
            source: "web".into(),
            program: sibling("astrum-web"),
            args: Vec::new(),
        }]
    } else {
        Vec::new()
    };
    let spec = LiveSpec {
        sim: sim_bin,
        scenario: run.scenario.clone(),
        scenario_file: None,
        listen: args.listen.clone(),
        record: args.record.clone(),
        companions,
        controllers,
        run_timeout: run_timeout(sweep.file.run_timeout_s),
        fail_fast: true,
        log: Log::Terminal,
        stop,
    };
    let file = args.file.display();
    let live = match LiveRun::start(spec, &children) {
        Ok(live) => live,
        Err(why) => {
            eprintln!("[astrum] FAILED: {file}: {why}");
            return ExitCode::FAILURE;
        }
    };
    if args.watch {
        println!("[astrum] viewer: {}", viewer_url());
    }
    match live.wait().status {
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

// --- astrum sweep -------------------------------------------------------------

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
    let children = Children::new();
    let what = "[sweep] stop requested: finishing the running runs";
    if let Err(e) = on_ctrl_c(&stop, &children, what) {
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
            children: &children,
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

fn list(dir: &Path) -> ExitCode {
    let entries = match catalog::list(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("[astrum] reading {}: {e}", dir.display());
            return ExitCode::from(2);
        }
    };
    let rows: Vec<(&str, String, &str)> = entries
        .iter()
        .map(|e| {
            let about = e.error.as_deref().unwrap_or(&e.summary);
            (e.kind.name(), e.path.display().to_string(), about)
        })
        .collect();
    let width = rows.iter().map(|(_, p, _)| p.len()).max().unwrap_or(0);
    for (kind, path, about) in &rows {
        println!("{kind:<8}  {path:<width$}  {about}");
    }
    ExitCode::SUCCESS
}
