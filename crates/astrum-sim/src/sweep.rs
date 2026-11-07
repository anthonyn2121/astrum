//! Run and sweep files (`astrum run`, `astrum sweep`): fly one base scenario
//! once, or many times — a grid of overrides × random samples × repeats — many
//! runs at once, collecting every run's bag, metrics and a `summary.csv`.
//!
//! This module plans a file (parse it, expand its grid, draw its samples, build
//! and validate every run's scenario up front, so a typo fails before anything
//! flies), checks every run's aircraft trim ([`Sweep::check_trim`]), runs runs
//! in-process ([`run_in_process`], for scenarios nothing external controls)
//! and drives a pool of runs ([`execute`]), keeping each run's metrics and the
//! summary up to date. Process mode — a sim plus external controllers per run,
//! each on its own Zenoh endpoint — is [`crate::launch::run_processes`].
//!
//! A run file is a sweep file without a grid (no `vary`, one repeat); a bare
//! scenario is a run file with nothing but `base` ([`SweepFile::for_scenario`]).
//!
//! Override keys: `sim.<field>`, `world.<field>`, `aircraft.<id>.<field>`
//! (scenario values, checked like the scenario itself), `param.<name>` (free
//! values substituted into controller commands as `{param.<name>}`) and
//! `aircraft.<id>.offset.<axis>` (moves or turns the aircraft relative to its
//! initial conditions once every absolute key of the run is applied; see
//! [`crate::perturb`]). `[[perturb]]` tables draw such offsets at random:
//! `samples` draws, sample `k` the same at every grid point (common random
//! numbers), deterministic from `(sample_seed, k)`.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use astrum_bag::metrics::RunMetrics;
use astrum_bag::BagReader;
use serde::{Deserialize, Serialize};

use crate::catalog::METRICS;
use crate::components::{SimClock, SimParams, SimStatus};
use crate::perturb::{self, Axis, Group, Offset, Perturb};
use crate::record;
use crate::rng::SplitMix64;
use crate::scenario::Scenario;
use crate::systems::{build_fdm, StopRequested};
use crate::{build_app, run_to_exit, RunMode};

/// A run or sweep file (TOML). Paths are relative to the working directory.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SweepFile {
    /// The scenario every run starts from.
    pub base: PathBuf,
    /// Runs per grid point and sample, run `r` with seed `sim.seed + r`.
    #[serde(default = "one")]
    pub repeats: u64,
    /// Runs at once (`--jobs` overrides); 0 = one per core, divided by the
    /// scenario's `sim.threads`.
    #[serde(default)]
    pub jobs: usize,
    /// Every run's `sim.speed`. Unset: 0 (as fast as possible) in a sweep, the
    /// scenario's own in `astrum run`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    /// Stop a run that takes longer than this many wall seconds; 0 = never.
    #[serde(default)]
    pub run_timeout_s: f64,
    /// Output directory (`--out` overrides); default `runs/<sweep file stem>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<PathBuf>,
    /// Random draws of the `[[perturb]]` tables per grid point; 0 = none.
    #[serde(default)]
    pub samples: u64,
    /// Seed of the draws: sample `k` depends on `(sample_seed, k)` only.
    #[serde(default)]
    pub sample_seed: u64,
    /// Redraw a sample that puts two aircraft closer than this (3-D, m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_separation_m: Option<f64>,
    /// Redraw a sample that puts an aircraft below this height (HAE, m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_alt_hae_m: Option<f64>,
    /// Commands started for every run (process mode), not through a shell.
    /// Placeholders: `{endpoint}` (the run's Zenoh endpoint, also in
    /// `$ASTRUM_ZENOH_CONNECT`), `{run}` (1-based), `{seed}`, `{param.<name>}`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub controllers: Vec<String>,
    /// Overrides applied to every run.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub set: BTreeMap<String, toml::Value>,
    /// The grid: every combination of these values is a grid point.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub vary: BTreeMap<String, Vec<toml::Value>>,
    /// Random initial-condition offsets, drawn `samples` times.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub perturb: Vec<Perturb>,
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
            samples: 0,
            sample_seed: 0,
            min_separation_m: None,
            min_alt_hae_m: None,
            controllers: Vec::new(),
            set: BTreeMap::new(),
            vary: BTreeMap::new(),
            perturb: Vec::new(),
        }
    }

    /// The file as TOML (parses back to an equal file).
    pub fn to_toml(&self) -> Result<String, SweepError> {
        toml::to_string(self).map_err(|e| err(format!("writing the sweep: {e}")))
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
    /// This run's `vary` values, then its drawn offsets, in
    /// [`Sweep::value_keys`] order.
    pub values: Vec<toml::Value>,
    /// The sample this run flies (0-based; summaries show it 1-based), when
    /// the sweep samples.
    pub sample: Option<u64>,
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
    /// The keys of [`Run::values`]: the `vary` keys (sorted), then the drawn
    /// offset keys ([`perturb::keys`]).
    pub value_keys: Vec<String>,
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
    Offset(u32, Axis),
    Param(&'k str),
}

fn aircraft_id(key: &str, id: &str) -> Result<u32, SweepError> {
    id.parse().map_err(|_| err(format!("{key}: `{id}` is not an aircraft id")))
}

fn target(key: &str) -> Result<Target<'_>, SweepError> {
    let parts: Vec<&str> = key.split('.').collect();
    match parts.as_slice() {
        ["sim", "speed"] => Err(err("sim.speed is set by `speed` (the file's, or --speed)")),
        [section @ ("sim" | "world"), field] => Ok(Target::Section(section, field)),
        ["aircraft", id, field] => Ok(Target::Aircraft(aircraft_id(key, id)?, field)),
        ["aircraft", id, "offset", axis] => {
            let id = aircraft_id(key, id)?;
            let axis = Axis::parse(axis).ok_or_else(|| {
                let names: Vec<&str> = Axis::ALL.iter().map(|a| a.name()).collect();
                err(format!("{key}: unknown offset axis `{axis}` (use {})", names.join(", ")))
            })?;
            Ok(Target::Offset(id, axis))
        }
        ["param", name] if !name.is_empty() => Ok(Target::Param(name)),
        _ => Err(err(format!(
            "unknown key `{key}`: use sim.<field>, world.<field>, aircraft.<id>.<field>, \
             aircraft.<id>.offset.<axis> or param.<name>"
        ))),
    }
}

/// What the overrides of one grid point set besides scenario values.
#[derive(Clone, Default)]
struct Overrides {
    params: BTreeMap<String, toml::Value>,
    offsets: BTreeMap<u32, Offset>,
}

fn has_aircraft(scenario: &toml::Value, id: u32) -> bool {
    scenario.get("aircraft").and_then(toml::Value::as_array).is_some_and(|a| {
        a.iter().any(|a| a.get("id").and_then(toml::Value::as_integer) == Some(i64::from(id)))
    })
}

/// Apply one override to a scenario (as a TOML value), the run's params or
/// its offsets.
fn apply(
    scenario: &mut toml::Value,
    o: &mut Overrides,
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
        Target::Offset(id, axis) => {
            if !has_aircraft(scenario, id) {
                return Err(err(format!("{key}: the base scenario has no aircraft {id}")));
            }
            let v = value
                .as_float()
                .or_else(|| value.as_integer().map(|i| i as f64))
                .filter(|v| v.is_finite())
                .ok_or_else(|| err(format!("{key} must be a finite number, got {value}")))?;
            o.offsets.entry(id).or_default().set(axis, v);
        }
        Target::Param(name) => {
            o.params.insert(name.to_owned(), value.clone());
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

/// One combination of the grid, its offsets applied.
struct Point {
    values: Vec<toml::Value>,
    params: BTreeMap<String, toml::Value>,
    scenario: Scenario,
}

/// One accepted draw: its values and the scenario it makes at every point.
struct Sample {
    values: Vec<toml::Value>,
    scenarios: Vec<Scenario>,
}

/// Draws tried per sample before the separation / altitude limits fail it.
const MAX_DRAWS: usize = 1000;

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

    /// Plan `file` for `purpose`, loading its `base`; `name` sets the default
    /// output directory (`runs/<name>`).
    pub fn plan(file: SweepFile, name: &str, purpose: Purpose) -> Result<Self, SweepError> {
        let base = Scenario::load(&file.base)
            .map_err(|e| err(format!("base scenario {}: {e}", file.base.display())))?;
        Self::plan_scenario(file, base, name, purpose)
    }

    /// Plan `file` for `purpose` from an in-memory `base` (`file.base` is not
    /// read); otherwise exactly [`Self::plan`].
    pub fn plan_scenario(
        file: SweepFile,
        base: Scenario,
        name: &str,
        purpose: Purpose,
    ) -> Result<Self, SweepError> {
        if file.repeats == 0 {
            return Err(err("repeats must be >= 1"));
        }
        if !(file.run_timeout_s.is_finite() && file.run_timeout_s >= 0.0) {
            return Err(err(format!("run_timeout_s must be >= 0, got {}", file.run_timeout_s)));
        }
        if let Some((key, _)) = file.vary.iter().find(|(_, values)| values.is_empty()) {
            return Err(err(format!("vary.{key:?} has no values")));
        }
        check_sampling(&file, &base)?;
        let mut base_value = toml::Value::try_from(&base).map_err(|e| err(e.to_string()))?;
        let mut base_overrides = Overrides::default();
        for (key, value) in &file.set {
            apply(&mut base_value, &mut base_overrides, key, value)?;
        }
        let speed = file.speed.or(match purpose {
            Purpose::Sweep => Some(0.0),
            Purpose::Run => None,
        });

        let vary_keys: Vec<String> = file.vary.keys().cloned().collect();
        let draw_keys = perturb::keys(&file.perturb);
        if let Some(key) = draw_keys.iter().find(|k| file.vary.contains_key(*k)) {
            return Err(err(format!("{key} is both varied and drawn by [[perturb]]")));
        }
        let axes: Vec<&Vec<toml::Value>> = file.vary.values().collect();
        let combinations: usize = axes.iter().map(|a| a.len()).product();
        let per_point = file.samples.max(1) * file.repeats;
        let first_run =
            |point: usize, sample: u64| point as u64 * per_point + sample * file.repeats;
        let mut points = Vec::with_capacity(combinations);
        for combination in 0..combinations {
            // Mixed-radix digits, the last `vary` key fastest.
            let mut rest = combination;
            let mut values = vec![toml::Value::Boolean(false); axes.len()];
            for (slot, axis) in values.iter_mut().zip(&axes).rev() {
                *slot = axis[rest % axis.len()].clone();
                rest /= axis.len();
            }
            let mut value = base_value.clone();
            let mut overrides = base_overrides.clone();
            for (key, v) in vary_keys.iter().zip(&values) {
                apply(&mut value, &mut overrides, key, v)?;
            }
            let at = || {
                format!("run {} ({})", first_run(combination, 0) + 1, describe(&vary_keys, &values))
            };
            let mut scenario: Scenario = value
                .try_into()
                .map_err(|e: toml::de::Error| err(format!("{}: {}", at(), e.message())))?;
            for (&id, &offset) in &overrides.offsets {
                Group { lead: id, members: vec![id], offset }
                    .apply(&mut scenario)
                    .map_err(|e| err(format!("{}: {e}", at())))?;
            }
            points.push(Point { values, params: overrides.params, scenario });
        }

        let samples: Vec<Sample> = (0..file.samples)
            .map(|k| draw_sample(&file, k, &points, &vary_keys, &draw_keys, first_run))
            .collect::<Result<_, _>>()?;

        let value_keys: Vec<String> = vary_keys.iter().chain(&draw_keys).cloned().collect();
        let mut runs = Vec::with_capacity(points.len() * per_point as usize);
        for (p, point) in points.iter().enumerate() {
            let variants: Vec<(Option<u64>, &Scenario, Vec<toml::Value>)> = if samples.is_empty() {
                vec![(None, &point.scenario, point.values.clone())]
            } else {
                samples
                    .iter()
                    .enumerate()
                    .map(|(k, s)| {
                        let values = point.values.iter().chain(&s.values).cloned().collect();
                        (Some(k as u64), &s.scenarios[p], values)
                    })
                    .collect()
            };
            for (sample, scenario, values) in variants {
                for r in 0..file.repeats {
                    let mut scenario = scenario.clone();
                    if let Some(speed) = speed {
                        scenario.sim.speed = speed;
                    }
                    scenario.sim.seed = scenario.sim.seed.wrapping_add(r);
                    let index = runs.len();
                    let run = Run {
                        index,
                        scenario,
                        values: values.clone(),
                        sample,
                        params: point.params.clone(),
                    };
                    check_run(&file, purpose, &run).map_err(|e| {
                        err(format!("run {} ({}): {e}", index + 1, describe(&value_keys, &values)))
                    })?;
                    runs.push(run);
                }
            }
        }
        let out = file.out.clone().unwrap_or_else(|| Path::new("runs").join(name));
        Ok(Self { file, value_keys, runs, out })
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

    /// Build and trim every aircraft whose class trims at every run's initial
    /// conditions, in run order — the sim's own FDM build ([`build_fdm`]), so
    /// a run that would fail its spawn trim fails here, before anything flies.
    /// Identical builds (same aircraft, ICs, rate and seed) are trimmed once.
    /// JSBSim serialises model loads and trims process-wide, so this runs on
    /// the calling thread (about 8 ms per F-16). The error names the first
    /// failing run, its values and the aircraft.
    pub fn check_trim(&self) -> Result<TrimCheck, SweepError> {
        let started = Instant::now();
        let mut seen = HashSet::new();
        for run in &self.runs {
            let s = &run.scenario;
            for a in &s.aircraft {
                let class = a.class().map_err(|e| err(e.to_string()))?;
                if !class.trim {
                    continue;
                }
                let bits = [
                    a.lat_deg,
                    a.lon_deg,
                    a.alt_hae_m,
                    a.heading_deg,
                    a.tas_mps,
                    a.climb_deg,
                    a.bank_deg,
                    s.sim.rate_hz,
                ]
                .map(f64::to_bits);
                if !seen.insert(format!("{}|{}|{class:?}|{bits:?}|{}", a.id, a.model, s.sim.seed)) {
                    continue;
                }
                build_fdm(a, &class, 1.0 / s.sim.rate_hz, s.sim.seed).map_err(|e| {
                    let values = describe(&self.value_keys, &run.values);
                    err(format!("run {} ({values}): {e}", run.index + 1))
                })?;
            }
        }
        Ok(TrimCheck { trims: seen.len(), wall_s: started.elapsed().as_secs_f64() })
    }
}

/// What [`Sweep::check_trim`] did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimCheck {
    /// Distinct aircraft builds trimmed.
    pub trims: usize,
    pub wall_s: f64,
}

/// The sampling keys' own rules and the `[[perturb]]` tables against `base`.
fn check_sampling(file: &SweepFile, base: &Scenario) -> Result<(), SweepError> {
    match (file.samples > 0, file.perturb.is_empty()) {
        (true, true) => return Err(err("samples needs [[perturb]] tables to draw")),
        (false, false) => return Err(err("[[perturb]] needs samples > 0")),
        _ => {}
    }
    if file.perturb.is_empty() {
        for (key, set) in
            [("min_separation_m", file.min_separation_m), ("min_alt_hae_m", file.min_alt_hae_m)]
        {
            if set.is_some() {
                return Err(err(format!("{key} limits [[perturb]] draws (there are none)")));
            }
        }
    }
    if let Some(m) = file.min_separation_m {
        if !(m.is_finite() && m > 0.0) {
            return Err(err(format!("min_separation_m must be > 0, got {m}")));
        }
    }
    if let Some(m) = file.min_alt_hae_m {
        if !m.is_finite() {
            return Err(err(format!("min_alt_hae_m must be finite, got {m}")));
        }
    }
    perturb::validate(&file.perturb, base).map_err(err)
}

/// Draw sample `k` (0-based): the first draw of its stream that keeps every
/// grid point within the separation and altitude limits. `first_run(p, k)` is
/// the 0-based index of the first run of point `p`, sample `k`.
fn draw_sample(
    file: &SweepFile,
    k: u64,
    points: &[Point],
    vary_keys: &[String],
    draw_keys: &[String],
    first_run: impl Fn(usize, u64) -> u64,
) -> Result<Sample, SweepError> {
    let mut rng = SplitMix64::stream(file.sample_seed, k);
    'draw: for _ in 0..MAX_DRAWS {
        let (groups, drawn) = perturb::draw(&file.perturb, &mut rng);
        let drawn: Vec<toml::Value> = drawn.into_iter().map(toml::Value::Float).collect();
        let mut scenarios = Vec::with_capacity(points.len());
        for (p, point) in points.iter().enumerate() {
            let mut scenario = point.scenario.clone();
            for group in &groups {
                group.apply(&mut scenario).map_err(|e| {
                    let keys: Vec<String> = vary_keys.iter().chain(draw_keys).cloned().collect();
                    let values: Vec<toml::Value> =
                        point.values.iter().chain(&drawn).cloned().collect();
                    err(format!(
                        "run {} (sample {}, {}): {e}",
                        first_run(p, k) + 1,
                        k + 1,
                        describe(&keys, &values)
                    ))
                })?;
            }
            let too_low = file
                .min_alt_hae_m
                .is_some_and(|min| scenario.aircraft.iter().any(|a| a.alt_hae_m < min));
            let too_close = file
                .min_separation_m
                .is_some_and(|min| perturb::min_separation_m(&scenario).is_some_and(|d| d < min));
            if too_low || too_close {
                continue 'draw;
            }
            scenarios.push(scenario);
        }
        return Ok(Sample { values: drawn, scenarios });
    }
    let limits: Vec<String> =
        [("min_separation_m", file.min_separation_m), ("min_alt_hae_m", file.min_alt_hae_m)]
            .into_iter()
            .filter_map(|(k, v)| v.map(|v| format!("{k} = {v:?}")))
            .collect();
    Err(err(format!(
        "sample {}: no draw in {MAX_DRAWS} attempts meets {} at every grid point",
        k + 1,
        limits.join(" and ")
    )))
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

/// How a run ended. Serialised as `{"kind": "ok"}` or `{"kind": "failed" |
/// "stopped", "reason": "..."}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", content = "reason", rename_all = "lowercase")]
pub enum Status {
    Ok,
    Failed(String),
    /// Stopped by a stop request (Ctrl-C) or its `run_timeout_s`.
    Stopped(String),
}

/// What one run produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
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

/// What [`execute`] reports about a run, from the worker thread running it.
#[derive(Debug, Clone)]
pub enum RunEvent {
    Started {
        index: usize,
    },
    /// `metrics`: of the run's bag (`None` without a readable one).
    Finished {
        index: usize,
        outcome: Outcome,
        metrics: Option<Box<RunMetrics>>,
    },
}

/// A run's row in the summary.
#[derive(Debug, Clone)]
enum RunState {
    NotRun,
    Running,
    Finished(Outcome, Option<Box<RunMetrics>>),
}

/// The metrics of the bag at `bag`, also written to `json`; `Ok(None)`
/// without a readable bag.
fn record_metrics(bag: &Path, json: &Path) -> std::io::Result<Option<Box<RunMetrics>>> {
    let Ok(reader) = BagReader::open(bag) else { return Ok(None) };
    let Ok(metrics) = astrum_bag::metrics::compute(&reader) else { return Ok(None) };
    let text = serde_json::to_string_pretty(&metrics).map_err(std::io::Error::other)?;
    std::fs::write(json, text + "\n")?;
    Ok(Some(Box::new(metrics)))
}

/// Fly every run of `sweep` on `jobs` threads with `run_one` (which records
/// run `r`'s bag, if any, to `<sweep.out>/<stem>.mcap`), returning each run's
/// outcome (`None`: never started, because a stop was requested). After each
/// run its bag's metrics go to `<sweep.out>/<stem>.metrics.json`; and
/// `<sweep.out>/summary.csv` is rewritten whenever a run starts or finishes,
/// so a reader sees progress. `on_event` hears every start and finish, on the
/// worker thread. Failing to write a result stops starting runs; the error is
/// returned once the running ones end.
pub fn execute(
    sweep: &Sweep,
    jobs: usize,
    stop: &StopRequested,
    on_event: impl Fn(RunEvent) + Sync,
    run_one: impl Fn(&Run) -> Outcome + Sync,
) -> std::io::Result<Vec<Option<Outcome>>> {
    std::fs::create_dir_all(&sweep.out)?;
    let summary = sweep.out.join("summary.csv");
    let board = Mutex::new(vec![RunState::NotRun; sweep.runs.len()]);
    write_summary(&summary, sweep, &board.lock().unwrap())?;
    let next = AtomicUsize::new(0);
    let error: Mutex<Option<std::io::Error>> = Mutex::new(None);
    let fail = |e: std::io::Error| {
        error.lock().unwrap().get_or_insert(e);
    };
    let update = |index: usize, state: RunState| {
        let mut board = board.lock().unwrap();
        board[index] = state;
        if let Err(e) = write_summary(&summary, sweep, &board) {
            fail(e);
        }
    };
    std::thread::scope(|s| {
        for _ in 0..jobs.min(sweep.runs.len()).max(1) {
            s.spawn(|| loop {
                if stop.requested() || error.lock().unwrap().is_some() {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(run) = sweep.runs.get(i) else { return };
                update(i, RunState::Running);
                on_event(RunEvent::Started { index: i });
                let outcome = run_one(run);
                let stem = run.stem();
                let bag = sweep.out.join(format!("{stem}.mcap"));
                let metrics =
                    match record_metrics(&bag, &sweep.out.join(format!("{stem}.metrics.json"))) {
                        Ok(m) => m,
                        Err(e) => {
                            fail(e);
                            None
                        }
                    };
                update(i, RunState::Finished(outcome.clone(), metrics.clone()));
                on_event(RunEvent::Finished { index: i, outcome, metrics });
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let board = board.into_inner().unwrap();
    Ok(board
        .into_iter()
        .map(|s| match s {
            RunState::Finished(outcome, _) => Some(outcome),
            RunState::NotRun | RunState::Running => None,
        })
        .collect())
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

/// A metric value as a summary cell: empty when unknown, reals to 3 decimals.
fn metric_cell(v: Option<&serde_json::Value>) -> String {
    match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Number(n)) if n.is_f64() => {
            n.as_f64().map_or(String::new(), |f| format!("{f:.3}"))
        }
        Some(other) => other.to_string(),
    }
}

/// Write `summary.csv` (via a temporary file, so a reader never sees half of
/// it): one row per run in run order — status, seed, the sample (when the
/// sweep samples), the `vary` values and drawn offsets, sim time, wall time,
/// speed, the bag, then the fleet metrics ([`METRICS`], but for `sim_time_s`,
/// a column already).
fn write_summary(path: &Path, sweep: &Sweep, rows: &[RunState]) -> std::io::Result<()> {
    let tmp = path.with_extension("csv.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    let sampled = sweep.file.samples > 0;
    let mut header = vec!["run".to_owned(), "status".into(), "seed".into()];
    if sampled {
        header.push("sample".into());
    }
    header.extend(sweep.value_keys.iter().cloned());
    header.extend(["sim_time_s", "wall_s", "speed_x", "bag"].map(String::from));
    let metric_ids: Vec<&str> =
        METRICS.iter().copied().filter(|id| !header.iter().any(|h| h == id)).collect();
    header.extend(metric_ids.iter().map(|id| id.to_string()));
    writeln!(f, "{}", header.iter().map(|h| csv_cell(h)).collect::<Vec<_>>().join(","))?;
    for (run, state) in sweep.runs.iter().zip(rows) {
        let (status, outcome, metrics) = match state {
            RunState::NotRun => ("not run".to_owned(), None, None),
            RunState::Running => ("running".to_owned(), None, None),
            RunState::Finished(o, m) => (
                match &o.status {
                    Status::Ok => "ok".to_owned(),
                    Status::Failed(why) => format!("failed: {why}"),
                    Status::Stopped(why) => format!("stopped: {why}"),
                },
                Some(o),
                m.as_ref(),
            ),
        };
        let (sim, wall) = (outcome.and_then(|o| o.sim_time_s), outcome.map(|o| o.wall_s));
        let mut row = vec![(run.index + 1).to_string(), status, run.scenario.sim.seed.to_string()];
        if sampled {
            row.push(run.sample.map_or(String::new(), |k| (k + 1).to_string()));
        }
        row.extend(run.values.iter().map(plain));
        row.push(sim.map_or(String::new(), |s| format!("{s:.3}")));
        row.push(wall.map_or(String::new(), |w| format!("{w:.3}")));
        row.push(match (sim, wall) {
            (Some(s), Some(w)) if w > 0.0 => format!("{:.1}", s / w),
            _ => String::new(),
        });
        let bag = PathBuf::from(format!("{}.mcap", run.stem()));
        row.push(if outcome.is_some() { bag.display().to_string() } else { String::new() });
        let metrics = metrics.map(|m| serde_json::to_value(m).unwrap_or_default());
        row.extend(
            metric_ids.iter().map(|id| metric_cell(metrics.as_ref().and_then(|m| m.get(id)))),
        );
        writeln!(f, "{}", row.iter().map(|c| csv_cell(c)).collect::<Vec<_>>().join(","))?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, path)
}
