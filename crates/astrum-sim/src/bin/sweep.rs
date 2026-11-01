//! `astrum-sweep`: fly one base scenario many times — a grid of overrides ×
//! repeats — many runs at once, writing each run's bag and a `summary.csv`.
//!
//! Without `controllers` in the sweep file, runs fly in-process (one thread
//! each). With them, each run is an `astrum-sim` process on its own Zenoh
//! endpoint plus the listed controller processes. Ctrl-C stops starting runs
//! and ends the running ones cleanly; the summary is written either way.
//! Exit status: 0 all runs ok, 1 any run not ok, 2 an invalid sweep.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, ExitStatus, Stdio};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use astrum_sim::sweep::{
    execute, parse_stopped_line, run_in_process, write_summary, Outcome, Run, Status, Sweep,
};
use astrum_sim::systems::StopRequested;
use clap::Parser;

/// Fly a sweep of scenario variants, many at once.
#[derive(Parser)]
#[command(name = "astrum-sweep", version, about)]
struct Cli {
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

fn signal(child: &Child, sig: libc::c_int) {
    // SAFETY: kill(2) on a child we spawned and have not reaped.
    unsafe {
        libc::kill(child.id() as libc::pid_t, sig);
    }
}

/// SIGTERM, then SIGKILL after `GRACE`; returns the exit status.
fn terminate(child: &mut Child) -> Option<ExitStatus> {
    if let Ok(Some(status)) = child.try_wait() {
        return Some(status);
    }
    signal(child, libc::SIGTERM);
    let deadline = Instant::now() + GRACE;
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status);
        }
        std::thread::sleep(POLL);
    }
    let _ = child.kill();
    child.wait().ok()
}

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
    let log = |name: &str| File::create(ctx.out.join(format!("{stem}.{name}.log")));
    let (Ok(sim_out), Ok(sim_err)) = (log("sim"), log("sim").and_then(|f| f.try_clone())) else {
        return failed("cannot create the sim log".into());
    };
    let mut sim = match Command::new(ctx.sim)
        .arg(&scenario_path)
        .args(["--listen", &endpoint, "--record"])
        .arg(ctx.out.join(format!("{stem}.mcap")))
        .stdin(Stdio::null())
        .stdout(sim_out)
        .stderr(sim_err)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return failed(format!("starting {}: {e}", ctx.sim.display())),
    };
    let mut controllers: Vec<Child> = Vec::new();
    for (k, cmd) in ctx.sweep.file.controllers.iter().enumerate() {
        let rendered = run.render(cmd, &endpoint);
        let spawned = shlex::split(&rendered)
            .filter(|argv| !argv.is_empty())
            .ok_or_else(|| format!("controller {}: cannot split `{rendered}`", k + 1))
            .and_then(|argv| {
                let out = log(&format!("ctl{}", k + 1)).map_err(|e| e.to_string())?;
                let err = out.try_clone().map_err(|e| e.to_string())?;
                Command::new(&argv[0])
                    .args(&argv[1..])
                    .env("ASTRUM_ZENOH_CONNECT", &endpoint)
                    .stdin(Stdio::null())
                    .stdout(out)
                    .stderr(err)
                    .spawn()
                    .map_err(|e| format!("controller {}: starting `{}`: {e}", k + 1, argv[0]))
            });
        match spawned {
            Ok(c) => controllers.push(c),
            Err(why) => {
                terminate(&mut sim);
                for c in &mut controllers {
                    terminate(c);
                }
                return failed(why);
            }
        }
    }

    let timeout = (ctx.sweep.file.run_timeout_s > 0.0)
        .then(|| Duration::from_secs_f64(ctx.sweep.file.run_timeout_s));
    let mut problem: Option<String> = None;
    let mut stopped: Option<String> = None;
    let sim_status = loop {
        match sim.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {}
            Err(e) => {
                problem.get_or_insert(format!("waiting for the sim: {e}"));
                break terminate(&mut sim);
            }
        }
        for (k, c) in controllers.iter_mut().enumerate() {
            if let Ok(Some(status)) = c.try_wait() {
                if !status.success() && problem.is_none() {
                    problem = Some(format!("controller {} exited ({status})", k + 1));
                }
            }
        }
        if stopped.is_none() {
            if ctx.stop.requested() {
                stopped = Some("stop requested".into());
                signal(&sim, libc::SIGTERM); // the sim ends its run and finishes the bag
            } else if timeout.is_some_and(|t| started.elapsed() > t) {
                stopped = Some(format!("run_timeout_s {} exceeded", ctx.sweep.file.run_timeout_s));
                signal(&sim, libc::SIGTERM);
            }
        }
        std::thread::sleep(POLL);
    };
    for c in &mut controllers {
        terminate(c);
    }
    let sim_time_s = std::fs::read_to_string(ctx.out.join(format!("{stem}.sim.log")))
        .ok()
        .and_then(|log| parse_stopped_line(&log));
    let status = match (problem, stopped, sim_status) {
        (Some(why), ..) => Status::Failed(why),
        (None, Some(why), _) => Status::Stopped(why),
        (None, None, Some(s)) if s.success() => Status::Ok,
        (None, None, Some(s)) => Status::Failed(format!("sim exited ({s}); see {stem}.sim.log")),
        (None, None, None) => Status::Failed("the sim could not be reaped".into()),
    };
    Outcome { status, sim_time_s, wall_s: started.elapsed().as_secs_f64() }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut sweep = match Sweep::load(&cli.sweep) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[sweep] {e}");
            return ExitCode::from(2);
        }
    };
    if let Some(out) = cli.out {
        sweep.out = out;
    }
    let jobs = sweep.jobs(cli.jobs);
    let mode = if sweep.in_process() { "in-process" } else { "processes" };
    println!(
        "[sweep] {} runs ({mode}), {jobs} at once → {}",
        sweep.runs.len(),
        sweep.out.display()
    );
    if !sweep.in_process()
        && usize::from(cli.base_port) + sweep.runs.len() > usize::from(u16::MAX) + 1
    {
        eprintln!(
            "[sweep] --base-port {} leaves too few ports for {} runs",
            cli.base_port,
            sweep.runs.len()
        );
        return ExitCode::from(2);
    }
    if cli.dry_run {
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
    if summary.exists() && !cli.overwrite {
        eprintln!("[sweep] {} exists; pass --overwrite to replace it", summary.display());
        return ExitCode::from(2);
    }
    if let Err(e) = std::fs::create_dir_all(&sweep.out) {
        eprintln!("[sweep] creating {}: {e}", sweep.out.display());
        return ExitCode::FAILURE;
    }
    let sim = cli.sim.unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("astrum-sim")))
            .unwrap_or_else(|| "astrum-sim".into())
    });
    if !sweep.in_process() && !sim.exists() {
        eprintln!("[sweep] no astrum-sim at {} (build it, or pass --sim)", sim.display());
        return ExitCode::from(2);
    }

    let stop = StopRequested::default();
    let flag = stop.0.clone();
    if let Err(e) = ctrlc::set_handler(move || {
        if flag.swap(true, Ordering::Relaxed) {
            std::process::exit(130); // second signal: give up on a clean stop
        }
        eprintln!("[sweep] stop requested: finishing the running runs");
    }) {
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
            base_port: cli.base_port,
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
