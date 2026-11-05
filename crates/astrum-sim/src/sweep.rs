//! Run and sweep files (`astrum run`, `astrum sweep`): fly one base scenario
//! once, or many times — a grid of overrides × repeats — many runs at once,
//! collecting every run's bag plus a `summary.csv`.
//!
//! This module plans a file (parse it, expand its grid, build and validate
//! every run's scenario up front, so a typo fails before anything flies), runs
//! runs in-process ([`run_in_process`], for scenarios nothing external
//! controls), drives a pool of runs ([`execute`]) and writes the summary. The
//! `astrum` binary adds process mode: a sim plus external controllers per run,
//! each on its own Zenoh endpoint, and `astrum run`'s single live run.
//!
//! A run file is a sweep file without a grid (no `vary`, one repeat); a bare
//! scenario is a run file with nothing but `base` ([`SweepFile::for_scenario`]).
//!
//! Override keys: `sim.<field>`, `world.<field>`, `aircraft.<id>.<field>`
//! (scenario values, checked like the scenario itself) and `param.<name>`
//! (free values substituted into controller commands as `{param.<name>}`).

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::components::{SimClock, SimParams, SimStatus};
use crate::record;
use crate::scenario::Scenario;
use crate::systems::StopRequested;
use crate::{build_app, run_to_exit, RunMode};

/// A run or sweep file (TOML). Paths are relative to the working directory.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepFile {
    /// The scenario every run starts from.
    pub base: PathBuf,
    /// Runs per combination of `vary`, run `r` with seed `sim.seed + r`.
    #[serde(default = "one")]
    pub repeats: u64,
    /// Runs at once (`--jobs` overrides); 0 = one per core, divided by the
    /// scenario's `sim.threads`.
    #[serde(default)]
    pub jobs: usize,
    /// Every run's `sim.speed`. Unset: 0 (as fast as possible) in a sweep, the
    /// scenario's own in `astrum run`.
    #[serde(default)]
    pub speed: Option<f64>,
    /// Stop a run that takes longer than this many wall seconds; 0 = never.
    #[serde(default)]
    pub run_timeout_s: f64,
    /// Output directory (`--out` overrides); default `runs/<sweep file stem>`.
    #[serde(default)]
    pub out: Option<PathBuf>,
    /// Commands started for every run (process mode), not through a shell.
    /// Placeholders: `{endpoint}` (the run's Zenoh endpoint, also in
    /// `$ASTRUM_ZENOH_CONNECT`), `{run}` (1-based), `{seed}`, `{param.<name>}`.
    #[serde(default)]
    pub controllers: Vec<String>,
    /// Overrides applied to every run.
    #[serde(default)]
    pub set: BTreeMap<String, toml::Value>,
    /// The grid: every combination of these values is a run (× `repeats`).
    #[serde(default)]
    pub vary: BTreeMap<String, Vec<toml::Value>>,
}

fn one() -> u64 {
    1
}

impl SweepFile {
    /// The run file of a bare scenario: `base` and nothing else.
    pub fn for_scenario(base: PathBuf) -> Self {
        Self {
            base,
            repeats: 1,
            jobs: 0,
            speed: None,
            run_timeout_s: 0.0,
            out: None,
            controllers: Vec::new(),
            set: BTreeMap::new(),
            vary: BTreeMap::new(),
        }
    }
}

/// What a plan is for; the rules a run must meet differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// `astrum sweep`: unattended batches. Every run must end by itself
    /// (`sim.max_time_s > 0`), a lockstep run needs listed controllers, and the
    /// pace defaults to 0 (as fast as possible).
    Sweep,
    /// `astrum run`: one live run. It may run until stopped, a lockstep
    /// scenario may wait for controllers started elsewhere, and the pace
    /// defaults to the scenario's own.
    Run,
}

/// One planned run.
#[derive(Debug, Clone)]
pub struct Run {
    /// 0-based; files and `{run}` use `index + 1`.
    pub index: usize,
    /// The complete, validated scenario this run flies.
    pub scenario: Scenario,
    /// This run's `vary` values, in [`Sweep::vary_keys`] order.
    pub values: Vec<toml::Value>,
    /// `param.<name>` values (from `set` and `vary`), for controller commands.
    pub params: BTreeMap<String, toml::Value>,
}

impl Run {
    /// File stem of this run's outputs (`run-0001`).
    pub fn stem(&self) -> String {
        format!("run-{:04}", self.index + 1)
    }

    /// `cmd` with its placeholders filled in for this run.
    pub fn render(&self, cmd: &str, endpoint: &str) -> String {
        let mut out = cmd
            .replace("{endpoint}", endpoint)
            .replace("{run}", &(self.index + 1).to_string())
            .replace("{seed}", &self.scenario.sim.seed.to_string());
        for (name, value) in &self.params {
            out = out.replace(&format!("{{param.{name}}}"), &plain(value));
        }
        out
    }
}

/// A planned sweep: every run's scenario built and validated.
#[derive(Debug, Clone)]
pub struct Sweep {
    pub file: SweepFile,
    pub vary_keys: Vec<String>,
    pub runs: Vec<Run>,
    pub out: PathBuf,
}

/// An invalid sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepError(pub String);

impl fmt::Display for SweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SweepError {}

fn err(msg: impl Into<String>) -> SweepError {
    SweepError(msg.into())
}

/// Where an override key lands.
enum Target<'k> {
    Section(&'k str, &'k str),
    Aircraft(u32, &'k str),
    Param(&'k str),
}

fn target(key: &str) -> Result<Target<'_>, SweepError> {
    let parts: Vec<&str> = key.split('.').collect();
    match parts.as_slice() {
        ["sim", "speed"] => Err(err("sim.speed is set by `speed` (the file's, or --speed)")),
        [section @ ("sim" | "world"), field] => Ok(Target::Section(section, field)),
        ["aircraft", id, field] => id
            .parse()
            .map(|id| Target::Aircraft(id, field))
            .map_err(|_| err(format!("{key}: `{id}` is not an aircraft id"))),
        ["param", name] if !name.is_empty() => Ok(Target::Param(name)),
        _ => Err(err(format!(
            "unknown key `{key}`: use sim.<field>, world.<field>, aircraft.<id>.<field> or \
             param.<name>"
        ))),
    }
}

/// Apply one override to a scenario (as a TOML value) or the run's params.
fn apply(
    scenario: &mut toml::Value,
    params: &mut BTreeMap<String, toml::Value>,
    key: &str,
    value: &toml::Value,
) -> Result<(), SweepError> {
    match target(key)? {
        Target::Section(section, field) => {
            let table = scenario
                .get_mut(section)
                .and_then(toml::Value::as_table_mut)
                .ok_or_else(|| err(format!("{key}: the scenario has no [{section}]")))?;
            table.insert(field.to_owned(), value.clone());
        }
        Target::Aircraft(id, field) => {
            let aircraft = scenario
                .get_mut("aircraft")
                .and_then(toml::Value::as_array_mut)
                .and_then(|a| {
                    a.iter_mut().find(|a| {
                        a.get("id").and_then(toml::Value::as_integer) == Some(i64::from(id))
                    })
                })
                .and_then(toml::Value::as_table_mut)
                .ok_or_else(|| err(format!("{key}: the base scenario has no aircraft {id}")))?;
            aircraft.insert(field.to_owned(), value.clone());
        }
        Target::Param(name) => {
            params.insert(name.to_owned(), value.clone());
        }
    }
    Ok(())
}

/// A value as a summary cell or command argument (strings unquoted).
fn plain(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

impl Sweep {
    /// Read and plan the sweep file at `path`.
    pub fn load(path: &Path) -> Result<Self, SweepError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| err(format!("reading {}: {e}", path.display())))?;
        let stem = path.file_stem().map_or("sweep".into(), |s| s.to_string_lossy().into_owned());
        Self::from_toml_str(&text, &stem)
    }

    /// Plan a sweep from its TOML text; `name` sets the default output
    /// directory (`runs/<name>`).
    pub fn from_toml_str(text: &str, name: &str) -> Result<Self, SweepError> {
        let file: SweepFile =
            toml::from_str(text).map_err(|e| err(format!("parsing the sweep: {e}")))?;
        Self::plan(file, name, Purpose::Sweep)
    }

    /// Plan `file` for `purpose`; `name` sets the default output directory
    /// (`runs/<name>`).
    pub fn plan(file: SweepFile, name: &str, purpose: Purpose) -> Result<Self, SweepError> {
        if file.repeats == 0 {
            return Err(err("repeats must be >= 1"));
        }
        if !(file.run_timeout_s.is_finite() && file.run_timeout_s >= 0.0) {
            return Err(err(format!("run_timeout_s must be >= 0, got {}", file.run_timeout_s)));
        }
        if let Some((key, _)) = file.vary.iter().find(|(_, values)| values.is_empty()) {
            return Err(err(format!("vary.{key:?} has no values")));
        }
        let base = Scenario::load(&file.base)
            .map_err(|e| err(format!("base scenario {}: {e}", file.base.display())))?;
        let mut base_value = toml::Value::try_from(&base).map_err(|e| err(e.to_string()))?;
        let mut base_params = BTreeMap::new();
        for (key, value) in &file.set {
            apply(&mut base_value, &mut base_params, key, value)?;
        }
        let speed = file.speed.or(match purpose {
            Purpose::Sweep => Some(0.0),
            Purpose::Run => None,
        });

        let vary_keys: Vec<String> = file.vary.keys().cloned().collect();
        let axes: Vec<&Vec<toml::Value>> = file.vary.values().collect();
        let combinations: usize = axes.iter().map(|a| a.len()).product();
        let mut runs = Vec::with_capacity(combinations * file.repeats as usize);
        for combination in 0..combinations {
            // Mixed-radix digits, the last `vary` key fastest.
            let mut rest = combination;
            let mut values = vec![toml::Value::Boolean(false); axes.len()];
            for (slot, axis) in values.iter_mut().zip(&axes).rev() {
                *slot = axis[rest % axis.len()].clone();
                rest /= axis.len();
            }
            let mut value = base_value.clone();
            let mut params = base_params.clone();
            for (key, v) in vary_keys.iter().zip(&values) {
                apply(&mut value, &mut params, key, v)?;
            }
            let scenario: Scenario = value.try_into().map_err(|e: toml::de::Error| {
                err(format!("{}: {}", describe(&vary_keys, &values), e.message()))
            })?;
            for r in 0..file.repeats {
                let mut scenario = scenario.clone();
                if let Some(speed) = speed {
                    scenario.sim.speed = speed;
                }
                scenario.sim.seed = scenario.sim.seed.wrapping_add(r);
                let index = runs.len();
                let run = Run { index, scenario, values: values.clone(), params: params.clone() };
                check_run(&file, purpose, &run).map_err(|e| {
                    err(format!("run {} ({}): {e}", index + 1, describe(&vary_keys, &values)))
                })?;
                runs.push(run);
            }
        }
        let out = file.out.clone().unwrap_or_else(|| Path::new("runs").join(name));
        Ok(Self { file, vary_keys, runs, out })
    }

    /// Whether runs fly in-process (no external controllers).
    pub fn in_process(&self) -> bool {
        self.file.controllers.is_empty()
    }

    /// Runs at once: `jobs` (or the sweep file's), else one per core divided by
    /// the threads each run steps its FDMs on.
    pub fn jobs(&self, jobs: Option<usize>) -> usize {
        match jobs.unwrap_or(self.file.jobs) {
            0 => {
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                let per_run = self.runs.first().map_or(1, |r| r.scenario.sim.threads.max(1));
                (cores / per_run).max(1)
            }
            n => n,
        }
    }
}

fn describe(keys: &[String], values: &[toml::Value]) -> String {
    if keys.is_empty() {
        return "the base scenario".into();
    }
    keys.iter().zip(values).map(|(k, v)| format!("{k} = {v}")).collect::<Vec<_>>().join(", ")
}

/// Rules a run must satisfy beyond the scenario's own validation.
fn check_run(file: &SweepFile, purpose: Purpose, run: &Run) -> Result<(), String> {
    let s = &run.scenario;
    s.validate().map_err(|e| e.to_string())?;
    if purpose == Purpose::Sweep && s.sim.max_time_s <= 0.0 {
        return Err("sim.max_time_s must be > 0 so the run ends".into());
    }
    if file.controllers.is_empty() {
        if purpose == Purpose::Sweep && s.sim.lockstep {
            return Err("sim.lockstep needs controllers to answer it (none are listed)".into());
        }
        if !run.params.is_empty() {
            return Err("param.* values are only used by controllers (none are listed)".into());
        }
    } else if s.sim.speed == 0.0 && !s.sim.lockstep {
        return Err("unpaced runs with external controllers need sim.lockstep (or a `speed` \
                    > 0): otherwise commands land on whichever tick they arrive"
            .into());
    }
    for cmd in &file.controllers {
        let rendered = run.render(cmd, "tcp/127.0.0.1:1");
        if let Some(start) = rendered.find('{') {
            let end = rendered[start..].find('}').map_or(rendered.len(), |e| start + e + 1);
            return Err(format!("unknown placeholder {} in `{cmd}`", &rendered[start..end]));
        }
    }
    Ok(())
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Ok,
    Failed(String),
    /// Stopped by a stop request (Ctrl-C) or its `run_timeout_s`.
    Stopped(String),
}

/// What one run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub status: Status,
    /// Sim time reached, if known.
    pub sim_time_s: Option<f64>,
    pub wall_s: f64,
}

/// The line `astrum-sim` prints when it stops; process mode reads the sim time
/// back from it ([`parse_stopped_line`]).
pub fn stopped_line(tick: u64, sim_s: f64, wall_s: f64) -> String {
    let speed = if wall_s > 0.0 { sim_s / wall_s } else { 0.0 };
    format!("[astrum-sim] stopped at tick {tick}: sim {sim_s:.3} s in {wall_s:.3} s wall ({speed:.1}× real time)")
}

/// The sim time in a [`stopped_line`] found in `log`.
pub fn parse_stopped_line(log: &str) -> Option<f64> {
    let line = log.lines().rev().find(|l| l.starts_with("[astrum-sim] stopped at tick "))?;
    let after = line.split(": sim ").nth(1)?;
    after.split(' ').next()?.parse().ok()
}

/// Fly one run in this process (no external controllers), recording it to `bag`.
pub fn run_in_process(run: &Run, bag: &Path, stop: &StopRequested) -> Outcome {
    let started = Instant::now();
    let failed = |why: String| Outcome {
        status: Status::Failed(why),
        sim_time_s: None,
        wall_s: started.elapsed().as_secs_f64(),
    };
    let (recorder, recording) = match record::start(bag, &run.scenario.to_toml()) {
        Ok(r) => r,
        Err(e) => return failed(format!("recording: {e}")),
    };
    let mut app = build_app(run.scenario.clone(), RunMode::Batch);
    app.insert_resource(recorder);
    app.insert_resource(stop.clone());
    let exit = run_to_exit(&mut app);
    let world = app.world();
    let sim_time_s = world.resource::<SimClock>().time_s(world.resource::<SimParams>());
    let fatal = world.resource::<SimStatus>().fatal.clone();
    drop(app); // closes the recording queue
    let recorded = recording.finish();
    let status = match (fatal, exit, recorded) {
        (Some(why), ..) => Status::Failed(why),
        (None, bevy::app::AppExit::Error(code), _) => Status::Failed(format!("exit code {code}")),
        (None, _, Err(e)) => Status::Failed(format!("recording: {e}")),
        (None, _, Ok(_)) if stop.requested() => Status::Stopped("stop requested".into()),
        (None, _, Ok(_)) => Status::Ok,
    };
    Outcome { status, sim_time_s: Some(sim_time_s), wall_s: started.elapsed().as_secs_f64() }
}

/// Run every run of `sweep` on `jobs` threads with `run_one`, returning each
/// run's outcome (`None`: never started, because a stop was requested). With
/// `progress`, prints a status line about once a second.
pub fn execute(
    sweep: &Sweep,
    jobs: usize,
    stop: &StopRequested,
    progress: bool,
    run_one: impl Fn(&Run) -> Outcome + Sync,
) -> Vec<Option<Outcome>> {
    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let outcomes: Mutex<Vec<Option<Outcome>>> = Mutex::new(vec![None; sweep.runs.len()]);
    let failed = AtomicUsize::new(0);
    let sim_done = Mutex::new(0.0f64);
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..jobs.min(sweep.runs.len()).max(1) {
            s.spawn(|| loop {
                if stop.requested() {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(run) = sweep.runs.get(i) else { return };
                let outcome = run_one(run);
                if outcome.status != Status::Ok {
                    failed.fetch_add(1, Ordering::Relaxed);
                }
                *sim_done.lock().unwrap() += outcome.sim_time_s.unwrap_or(0.0);
                outcomes.lock().unwrap()[i] = Some(outcome);
                finished.fetch_add(1, Ordering::Relaxed);
            });
        }
        if progress {
            let total = sweep.runs.len();
            let mut last = 0;
            while finished.load(Ordering::Relaxed) < total && !stop.requested() {
                std::thread::sleep(Duration::from_millis(100));
                let done = finished.load(Ordering::Relaxed);
                let secs = started.elapsed().as_secs();
                if secs > last || done == total {
                    last = secs;
                    let rate = *sim_done.lock().unwrap() / started.elapsed().as_secs_f64();
                    println!(
                        "[sweep] {done}/{total} done, {} not ok | {rate:.0} sim-s per wall-s",
                        failed.load(Ordering::Relaxed)
                    );
                }
            }
        }
    });
    outcomes.into_inner().unwrap()
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

/// Write `summary.csv`: one row per run — status, seed, the `vary` values, sim
/// time, wall time, speed and the bag.
pub fn write_summary(
    path: &Path,
    sweep: &Sweep,
    outcomes: &[Option<Outcome>],
) -> std::io::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = vec!["run".to_owned(), "status".into(), "seed".into()];
    header.extend(sweep.vary_keys.iter().cloned());
    header.extend(["sim_time_s", "wall_s", "speed_x", "bag"].map(String::from));
    writeln!(f, "{}", header.iter().map(|h| csv_cell(h)).collect::<Vec<_>>().join(","))?;
    for (run, outcome) in sweep.runs.iter().zip(outcomes) {
        let (status, sim, wall) = match outcome {
            None => ("not run".to_owned(), None, None),
            Some(o) => (
                match &o.status {
                    Status::Ok => "ok".to_owned(),
                    Status::Failed(why) => format!("failed: {why}"),
                    Status::Stopped(why) => format!("stopped: {why}"),
                },
                o.sim_time_s,
                Some(o.wall_s),
            ),
        };
        let mut row = vec![(run.index + 1).to_string(), status, run.scenario.sim.seed.to_string()];
        row.extend(run.values.iter().map(plain));
        row.push(sim.map_or(String::new(), |s| format!("{s:.3}")));
        row.push(wall.map_or(String::new(), |w| format!("{w:.3}")));
        row.push(match (sim, wall) {
            (Some(s), Some(w)) if w > 0.0 => format!("{:.1}", s / w),
            _ => String::new(),
        });
        let bag = PathBuf::from(format!("{}.mcap", run.stem()));
        row.push(if outcome.is_some() { bag.display().to_string() } else { String::new() });
        writeln!(f, "{}", row.iter().map(|c| csv_cell(c)).collect::<Vec<_>>().join(","))?;
    }
    f.flush()
}
