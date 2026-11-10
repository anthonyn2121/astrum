//! The studio's sweeps: planning one and saving it, without flying anything.
//!
//! A request names a saved base scenario, optional catalog controllers
//! (assignments, as on the live run), `[[perturb]]` draws and a `vary` grid of
//! offsets. [`resolve`] turns it into the sweep file `astrum sweep` reads, and
//! the library plans that file ([`Sweep::plan_scenario`]), building and
//! validating every run. `POST /api/sweeps/plan` replies with what would fly:
//! each run's values and the moving aircraft's starting positions.
//! `PUT /api/sweeps/saved/<name>` plans the same way and writes the file as
//! `config/sweep.<name>.toml`, runnable with `astrum sweep`; the request's
//! assignments ride along in a header comment, so `GET
//! /api/sweeps/saved/<name>` reads the file back as a request. Planning and
//! trim checks load JSBSim models, so they run on the blocking pool.

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use astrum_sim::perturb::{Axis, Perturb};
use astrum_sim::scenario::{AircraftSpec, Scenario};
use astrum_sim::sweep::{Purpose, Sweep, SweepFile, TrimCheck};
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{error, json, safe_name};
use crate::assign::{assign, check_programs, Assignment, AssignmentBody};
use crate::catalogs::{Catalogs, ScenarioError};
use crate::server::AppState;

/// Most runs one request may plan: each is a scenario built and validated.
pub const MAX_RUNS: u64 = 2000;
/// Longest saved sweep summary, in characters (as a scenario's).
const SUMMARY_MAX_CHARS: usize = 200;
/// The header comment that carries a saved sweep's assignments.
const STUDIO_TAG: &str = "astrum-studio:";

fn one() -> u64 {
    1
}

/// A sweep request: the body of `POST /api/sweeps/plan`, and what a saved
/// sweep is read back as.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepRequest {
    /// A saved scenario's name: the base every run starts from.
    pub scenario: String,
    #[serde(default)]
    pub assignments: Vec<AssignmentBody>,
    #[serde(default)]
    pub perturb: Vec<Perturb>,
    /// The grid: every combination of these values is a grid point. Keys are
    /// `aircraft.<id>.offset.<axis>`.
    #[serde(default)]
    pub vary: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    pub samples: u64,
    #[serde(default)]
    pub sample_seed: u64,
    #[serde(default)]
    pub min_separation_m: Option<f64>,
    #[serde(default)]
    pub min_alt_hae_m: Option<f64>,
    #[serde(default = "one")]
    pub repeats: u64,
    /// Stop each run after this many simulated seconds (> 0).
    pub max_time_s: f64,
    #[serde(default)]
    pub run_timeout_s: f64,
    #[serde(default)]
    pub jobs: usize,
    /// Also trim every aircraft of every run (the library's `check_trim`).
    #[serde(default)]
    pub check_trim: bool,
}

/// Why a request cannot become a sweep file.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepInputError {
    /// The request is wrong (400): an unknown or unsafe scenario name, a vary
    /// key that is not an offset, an assignment the catalog refuses, too many runs.
    BadRequest(String),
    /// The saved base scenario is not a valid scenario (422).
    Invalid(String),
    /// The saved base scenario could not be read (500).
    Io(String),
}

/// A request as the sweep file `astrum sweep` reads.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// The saved base, untouched.
    pub scenario: Scenario,
    pub file: SweepFile,
    /// Each assignment as the run flies it: every catalog parameter, defaults filled in.
    pub assignments: Vec<Assignment>,
}

/// The aircraft and axis of an `aircraft.<id>.offset.<axis>` key.
fn offset_key_parts(key: &str) -> Option<(u32, Axis)> {
    match key.split('.').collect::<Vec<_>>().as_slice() {
        ["aircraft", id, "offset", axis] => Some((id.parse().ok()?, Axis::parse(axis)?)),
        _ => None,
    }
}

/// Check a vary key: an offset of an aircraft the scenario has.
fn check_vary_key(key: &str, scenario: &Scenario) -> Result<(), String> {
    let Some((id, _)) = offset_key_parts(key) else {
        let axes: Vec<&str> = Axis::ALL.iter().map(|a| a.name()).collect();
        return Err(format!(
            "vary key {key:?}: must be aircraft.<id>.offset.<axis>, with <axis> one of {}",
            axes.join(", ")
        ));
    };
    if scenario.aircraft.iter().any(|a| a.id == id) {
        Ok(())
    } else {
        Err(format!("vary key {key:?}: the scenario has no aircraft {id}"))
    }
}

/// The runs a request plans: its grid points × its draws × its repeats,
/// saturating.
fn planned_runs(req: &SweepRequest) -> u64 {
    let points = req
        .vary
        .values()
        .fold(1u64, |n, values| n.saturating_mul(u64::try_from(values.len()).unwrap_or(u64::MAX)));
    points.saturating_mul(req.samples.max(1)).saturating_mul(req.repeats)
}

/// Convert a request to the sweep file the CLI reads, validating its scenario,
/// assignments and vary keys against the saved base. Nothing is planned here.
pub fn resolve(req: &SweepRequest, catalogs: &Catalogs) -> Result<Resolved, SweepInputError> {
    let runs = planned_runs(req);
    if runs > MAX_RUNS {
        return Err(SweepInputError::BadRequest(format!("plans {runs} runs; at most {MAX_RUNS}")));
    }
    if !(req.max_time_s.is_finite() && req.max_time_s > 0.0) {
        return Err(SweepInputError::BadRequest(format!(
            "max_time_s must be > 0 and finite, got {}",
            req.max_time_s
        )));
    }
    let doc = catalogs.scenario(&req.scenario).map_err(|e| match &e {
        ScenarioError::BadName(_) | ScenarioError::NotFound(_) => {
            SweepInputError::BadRequest(format!("scenario: {e}"))
        }
        ScenarioError::Invalid(_) => SweepInputError::Invalid(e.to_string()),
        ScenarioError::Io(_) => SweepInputError::Io(e.to_string()),
    })?;
    for key in req.vary.keys() {
        check_vary_key(key, &doc.scenario).map_err(SweepInputError::BadRequest)?;
    }
    // `assign` checks the assignments on a copy; the sweep's own `set` below
    // carries the lockstep flags it would set.
    let mut checked = doc.scenario.clone();
    let (controllers, assignments) = assign(&mut checked, &req.assignments, catalogs.controllers())
        .map_err(SweepInputError::BadRequest)?;
    check_programs(&assignments, catalogs.controllers()).map_err(SweepInputError::BadRequest)?;

    let mut set =
        BTreeMap::from([("sim.max_time_s".to_owned(), toml::Value::Float(req.max_time_s))]);
    if !req.assignments.is_empty() {
        let controlled: BTreeSet<u32> = req.assignments.iter().map(|a| a.aircraft).collect();
        set.insert("sim.lockstep".to_owned(), toml::Value::Boolean(true));
        for a in &doc.scenario.aircraft {
            set.insert(
                format!("aircraft.{}.lockstep", a.id),
                toml::Value::Boolean(controlled.contains(&a.id)),
            );
        }
    }
    let vary = req
        .vary
        .iter()
        .map(|(key, values)| (key.clone(), values.iter().map(|&v| toml::Value::Float(v)).collect()))
        .collect();
    let base = catalogs.scenario_path(&req.scenario).map_err(SweepInputError::BadRequest)?;
    let file = SweepFile {
        base,
        repeats: req.repeats,
        jobs: req.jobs,
        speed: None,
        run_timeout_s: req.run_timeout_s,
        out: None,
        samples: req.samples,
        sample_seed: req.sample_seed,
        min_separation_m: req.min_separation_m,
        min_alt_hae_m: req.min_alt_hae_m,
        controllers,
        set,
        vary,
        perturb: req.perturb.clone(),
    };
    Ok(Resolved { scenario: doc.scenario, file, assignments })
}

/// Why a request is refused: its status and the message the reply carries.
#[derive(Debug)]
struct Refusal(StatusCode, String);

impl Refusal {
    fn response(self) -> Response {
        error(self.0, &self.1)
    }
}

impl From<SweepInputError> for Refusal {
    fn from(e: SweepInputError) -> Self {
        match e {
            SweepInputError::BadRequest(msg) => Refusal(StatusCode::BAD_REQUEST, msg),
            SweepInputError::Invalid(msg) => Refusal(StatusCode::UNPROCESSABLE_ENTITY, msg),
            SweepInputError::Io(msg) => Refusal(StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
    }
}

fn bad(msg: String) -> Refusal {
    Refusal(StatusCode::BAD_REQUEST, msg)
}

/// Resolve and plan `req`: the sweep the library makes of its file.
fn plan_request(req: &SweepRequest, catalogs: &Catalogs) -> Result<Sweep, Refusal> {
    let resolved = resolve(req, catalogs)?;
    Sweep::plan_scenario(resolved.file, resolved.scenario, &req.scenario, Purpose::Sweep)
        .map_err(|e| Refusal(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))
}

/// A `POST /api/sweeps/plan` reply: what the sweep would fly.
#[derive(Serialize)]
struct PlanView {
    runs: usize,
    in_process: bool,
    keys: Vec<String>,
    rows: Vec<RowView>,
    moving: Vec<u32>,
    initial: Vec<InitialView>,
    trim: Option<TrimReply>,
}

/// One run: its number, seed, sample (1-based) and values by key.
#[derive(Serialize)]
struct RowView {
    run: usize,
    seed: u64,
    sample: Option<u64>,
    values: BTreeMap<String, Value>,
}

/// One run's starting positions of the moving aircraft.
#[derive(Serialize)]
struct InitialView {
    run: usize,
    aircraft: Vec<StartView>,
}

/// An aircraft's initial conditions as the run starts it.
#[derive(Serialize)]
struct StartView {
    id: u32,
    model: String,
    lat_deg: f64,
    lon_deg: f64,
    alt_hae_m: f64,
    heading_deg: f64,
    climb_deg: f64,
    bank_deg: f64,
    tas_mps: f64,
}

fn start(a: &AircraftSpec) -> StartView {
    StartView {
        id: a.id,
        model: a.model.clone(),
        lat_deg: a.lat_deg,
        lon_deg: a.lon_deg,
        alt_hae_m: a.alt_hae_m,
        heading_deg: a.heading_deg,
        climb_deg: a.climb_deg,
        bank_deg: a.bank_deg,
        tas_mps: a.tas_mps,
    }
}

/// What a trim check did.
#[derive(Serialize)]
struct TrimReply {
    trims: usize,
    wall_s: f64,
}

impl From<TrimCheck> for TrimReply {
    fn from(check: TrimCheck) -> Self {
        Self { trims: check.trims, wall_s: check.wall_s }
    }
}

/// A run value as JSON: numbers as numbers, anything else as its text.
fn value_json(v: &toml::Value) -> Value {
    match v {
        toml::Value::Integer(i) => Value::from(*i),
        toml::Value::Float(f) => Value::from(*f),
        toml::Value::String(s) => Value::String(s.clone()),
        other => Value::String(other.to_string()),
    }
}

impl PlanView {
    fn new(req: &SweepRequest, sweep: &Sweep, trim: Option<TrimReply>) -> Self {
        // The aircraft the sweep moves: those its draws name and its grid offsets.
        let moving: BTreeSet<u32> = req
            .perturb
            .iter()
            .flat_map(|p| p.aircraft.iter().copied())
            .chain(req.vary.keys().filter_map(|k| offset_key_parts(k).map(|(id, _)| id)))
            .collect();
        let rows = sweep
            .runs
            .iter()
            .map(|run| RowView {
                run: run.index + 1,
                seed: run.scenario.sim.seed,
                sample: run.sample.map(|k| k + 1),
                values: sweep
                    .value_keys
                    .iter()
                    .zip(&run.values)
                    .map(|(key, v)| (key.clone(), value_json(v)))
                    .collect(),
            })
            .collect();
        let initial = sweep
            .runs
            .iter()
            .map(|run| {
                let mut aircraft: Vec<&AircraftSpec> =
                    run.scenario.aircraft.iter().filter(|a| moving.contains(&a.id)).collect();
                aircraft.sort_by_key(|a| a.id);
                InitialView {
                    run: run.index + 1,
                    aircraft: aircraft.into_iter().map(start).collect(),
                }
            })
            .collect();
        Self {
            runs: sweep.runs.len(),
            in_process: sweep.in_process(),
            keys: sweep.value_keys.clone(),
            rows,
            moving: moving.into_iter().collect(),
            initial,
            trim,
        }
    }
}

/// Plan `req`, and trim-check its runs when asked.
fn plan_view(req: &SweepRequest, catalogs: &Catalogs) -> Result<PlanView, Refusal> {
    let sweep = plan_request(req, catalogs)?;
    let trim = if req.check_trim {
        let check = sweep
            .check_trim()
            .map_err(|e| Refusal(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
        Some(TrimReply::from(check))
    } else {
        None
    };
    Ok(PlanView::new(req, &sweep, trim))
}

/// Parse a request body; a bad body is a 400.
fn parse_body<T: for<'de> Deserialize<'de>>(
    body: Result<Bytes, BytesRejection>,
) -> Result<T, Refusal> {
    let body = body.map_err(|rejection| Refusal(rejection.status(), rejection.body_text()))?;
    serde_json::from_slice(&body).map_err(|e| bad(format!("bad sweep body: {e}")))
}

/// `POST /api/sweeps/plan`: what the sweep would fly, nothing flown.
pub(crate) async fn plan(
    State(state): State<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let req: SweepRequest = match parse_body(body) {
        Ok(req) => req,
        Err(refusal) => return refusal.response(),
    };
    let catalogs = state.catalogs.clone();
    match tokio::task::spawn_blocking(move || plan_view(&req, &catalogs)).await {
        Ok(Ok(view)) => json(StatusCode::OK, &view),
        Ok(Err(refusal)) => refusal.response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("planning the sweep: {e}")),
    }
}

/// A `PUT /api/sweeps/saved/<name>` body.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveBody {
    request: SweepRequest,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    overwrite: bool,
}

/// A saved sweep as its save reply shows it.
#[derive(Serialize)]
struct Saved {
    name: String,
    path: String,
    summary: String,
}

/// One `sweep.<name>.toml` file, valid or not.
#[derive(Serialize)]
pub struct SavedEntry {
    pub name: String,
    /// The file, as the server reaches it (`config/sweep.<name>.toml`).
    pub path: String,
    /// The header comment's first line that is not the assignments line.
    pub summary: String,
    /// The base scenario's name (`scenario.<name>.toml`); `None` otherwise.
    pub base: Option<String>,
    /// Why the file is not a sweep file; `None` when it is.
    pub error: Option<String>,
}

/// A saved sweep read back as a request.
#[derive(Serialize)]
struct SavedSweep {
    name: String,
    summary: String,
    request: SweepRequest,
}

/// `config/sweep.<name>.toml` in the config directory.
fn sweep_path(catalogs: &Catalogs, name: &str) -> PathBuf {
    catalogs.config_dir().join(format!("sweep.{name}.toml"))
}

/// `<name>` of a `sweep.<name>.toml` path whose name is [`safe_name`].
fn sweep_name(path: &Path) -> Option<String> {
    let file = path.file_name()?.to_str()?;
    let name = file.strip_prefix("sweep.")?.strip_suffix(".toml")?;
    safe_name(name).ok().map(str::to_owned)
}

/// `<name>` of a `scenario.<name>.toml` base path.
fn base_name(base: &Path) -> Option<String> {
    let file = base.file_name()?.to_str()?;
    let name = file.strip_prefix("scenario.")?.strip_suffix(".toml")?;
    safe_name(name).ok().map(str::to_owned)
}

/// The comment lines at the top of `text`, each without its `#` and trimmed.
fn header_comments(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .take_while(|line| line.is_empty() || line.starts_with('#'))
        .filter_map(|line| line.strip_prefix('#'))
        .map(str::trim)
}

/// The first header comment that is not the assignments line.
fn header_summary(text: &str) -> String {
    header_comments(text)
        .find(|c| !c.is_empty() && !c.starts_with(STUDIO_TAG))
        .unwrap_or_default()
        .to_owned()
}

/// The JSON of the assignments line, if the header has one.
fn studio_json(text: &str) -> Option<&str> {
    header_comments(text).find_map(|c| c.strip_prefix(STUDIO_TAG).map(str::trim))
}

/// A JSON number as an `f64`, or a TOML integer or float.
fn number(v: &toml::Value) -> Option<f64> {
    v.as_float().or_else(|| v.as_integer().map(|i| i as f64))
}

/// Every `sweep.<name>.toml` in `dir` whose name passes [`safe_name`], sorted by name.
fn saved_entries(dir: &Path) -> io::Result<Vec<SavedEntry>> {
    let mut entries = Vec::new();
    for dirent in std::fs::read_dir(dir)? {
        let path = dirent?.path();
        let Some(name) = sweep_name(&path) else { continue };
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)?;
        let (base, error) = match toml::from_str::<SweepFile>(&text) {
            Ok(file) => (base_name(&file.base), None),
            Err(e) => (None, Some(e.to_string())),
        };
        entries.push(SavedEntry {
            name,
            path: path.display().to_string(),
            summary: header_summary(&text),
            base,
            error,
        });
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(entries)
}

/// `GET /api/sweeps/saved`: every saved sweep, invalid ones too.
pub(crate) async fn list(State(state): State<AppState>) -> Response {
    let catalogs = state.catalogs.clone();
    match tokio::task::spawn_blocking(move || saved_entries(catalogs.config_dir())).await {
        Ok(Ok(entries)) => json(StatusCode::OK, &entries),
        Ok(Err(e)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("listing {}: {e}", state.catalogs.config_dir().display()),
        ),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Read the saved sweep `name` back as a request.
fn read_saved(catalogs: &Catalogs, name: &str) -> Result<SavedSweep, Refusal> {
    let name = safe_name(name).map_err(bad)?;
    let path = sweep_path(catalogs, name);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(Refusal(StatusCode::NOT_FOUND, format!("no sweep named {name:?}")))
        }
        Err(e) => {
            return Err(Refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reading {}: {e}", path.display()),
            ))
        }
    };
    let unprocessable = |msg: String| Refusal(StatusCode::UNPROCESSABLE_ENTITY, msg);
    let file: SweepFile =
        toml::from_str(&text).map_err(|e| unprocessable(format!("{}: {e}", path.display())))?;
    let scenario = base_name(&file.base)
        .ok_or_else(|| unprocessable("base is not a saved scenario".to_owned()))?;
    let max_time_s = file
        .set
        .get("sim.max_time_s")
        .and_then(number)
        .ok_or_else(|| unprocessable("set has no sim.max_time_s (a number)".to_owned()))?;
    let mut vary = BTreeMap::new();
    for (key, values) in &file.vary {
        let numbers = values
            .iter()
            .map(|v| {
                number(v).ok_or_else(|| unprocessable(format!("vary {key:?}: {v} is not a number")))
            })
            .collect::<Result<Vec<f64>, Refusal>>()?;
        vary.insert(key.clone(), numbers);
    }
    let assignments: Vec<AssignmentBody> = match studio_json(&text) {
        Some(json) => serde_json::from_str(json).map_err(|e| {
            unprocessable(format!("{}: the {STUDIO_TAG} line: {e}", path.display()))
        })?,
        None => Vec::new(),
    };
    let request = SweepRequest {
        scenario,
        assignments,
        perturb: file.perturb,
        vary,
        samples: file.samples,
        sample_seed: file.sample_seed,
        min_separation_m: file.min_separation_m,
        min_alt_hae_m: file.min_alt_hae_m,
        repeats: file.repeats,
        max_time_s,
        run_timeout_s: file.run_timeout_s,
        jobs: file.jobs,
        check_trim: false,
    };
    Ok(SavedSweep { name: name.to_owned(), summary: header_summary(&text), request })
}

/// `GET /api/sweeps/saved/<name>`: the saved sweep as a request.
pub(crate) async fn load(
    State(state): State<AppState>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let catalogs = state.catalogs.clone();
    match tokio::task::spawn_blocking(move || read_saved(&catalogs, &name)).await {
        Ok(Ok(saved)) => json(StatusCode::OK, &saved),
        Ok(Err(refusal)) => refusal.response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

/// Refuse a summary that is not one line of at most [`SUMMARY_MAX_CHARS`].
fn check_summary(summary: &str) -> Result<(), Refusal> {
    if summary.contains(['\n', '\r']) {
        return Err(bad("summary must be a single line".to_owned()));
    }
    let chars = summary.chars().count();
    if chars > SUMMARY_MAX_CHARS {
        return Err(bad(format!(
            "summary must be at most {SUMMARY_MAX_CHARS} characters, got {chars}"
        )));
    }
    Ok(())
}

/// The file's text: the summary line, the assignments line (when there are
/// assignments) and the sweep. It must read back as the file it came from.
fn sweep_text(file: &SweepFile, request: &SweepRequest, summary: &str) -> Result<String, Refusal> {
    let internal = |msg: String| Refusal(StatusCode::INTERNAL_SERVER_ERROR, msg);
    let mut text = String::new();
    if !summary.is_empty() {
        text += &format!("# {summary}\n");
    }
    if !request.assignments.is_empty() {
        let assignments = serde_json::to_string(&request.assignments)
            .map_err(|e| internal(format!("writing the assignments: {e}")))?;
        text += &format!("# {STUDIO_TAG} {assignments}\n");
    }
    let body = file.to_toml().map_err(|e| internal(e.to_string()))?;
    let read_back: SweepFile = toml::from_str(&body)
        .map_err(|e| internal(format!("the sweep does not read back: {e}")))?;
    if read_back != *file {
        return Err(internal("the sweep does not read back as it was planned".to_owned()));
    }
    text += &body;
    Ok(text)
}

/// Write `text` to `path` through `tmp` (beside it), so a reader sees the old
/// file or the new one.
fn write_atomic(path: &Path, tmp: &Path, text: &str) -> Result<(), Refusal> {
    let written = std::fs::write(tmp, text).and_then(|()| std::fs::rename(tmp, path));
    written.map_err(|e| {
        // The target is untouched; the temporary file is not wanted.
        let _ = std::fs::remove_file(tmp);
        Refusal(StatusCode::INTERNAL_SERVER_ERROR, format!("saving {}: {e}", path.display()))
    })
}

/// Save `request` as the sweep `name`; `true` in the reply when created.
fn save_sweep(
    catalogs: &Catalogs,
    name: &str,
    request: &SweepRequest,
    summary: &str,
    overwrite: bool,
) -> Result<(Saved, bool), Refusal> {
    let name = safe_name(name).map_err(bad)?;
    check_summary(summary)?;
    let sweep = plan_request(request, catalogs)?;
    let path = sweep_path(catalogs, name);
    let existed = path.exists();
    if existed && !overwrite {
        return Err(Refusal(
            StatusCode::CONFLICT,
            format!("sweep \"{name}\" exists; send overwrite: true to replace it"),
        ));
    }
    let text = sweep_text(&sweep.file, request, summary)?;
    let tmp = catalogs.config_dir().join(format!(".sweep.{name}.toml.tmp"));
    write_atomic(&path, &tmp, &text)?;
    let saved = Saved {
        name: name.to_owned(),
        path: path.display().to_string(),
        summary: summary.to_owned(),
    };
    Ok((saved, !existed))
}

/// `PUT /api/sweeps/saved/<name>`: 201 when created, 200 when replaced.
pub(crate) async fn save(
    State(state): State<AppState>,
    UrlPath(name): UrlPath<String>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let SaveBody { request, summary, overwrite } = match parse_body(body) {
        Ok(body) => body,
        Err(refusal) => return refusal.response(),
    };
    let catalogs = state.catalogs.clone();
    let saved = tokio::task::spawn_blocking(move || {
        save_sweep(&catalogs, &name, &request, &summary, overwrite)
    })
    .await;
    match saved {
        Ok(Ok((saved, true))) => json(StatusCode::CREATED, &saved),
        Ok(Ok((saved, false))) => json(StatusCode::OK, &saved),
        Ok(Err(refusal)) => refusal.response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("saving the sweep: {e}")),
    }
}
