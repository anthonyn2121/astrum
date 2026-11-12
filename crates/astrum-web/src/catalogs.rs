//! What the studio offers the browser: the environments, the aircraft models
//! ASTRUM has profiles for, the controller catalog
//! (`<config>/controllers.toml`) and the scenarios saved as
//! `<config>/scenario.<name>.toml`.
//!
//! The controller catalog is read once, at startup (a broken one stops the
//! server); scenarios are read on every request, so a file saved or edited
//! meanwhile is seen at once.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use astrum_sim::catalog::summary;
use astrum_sim::components::{AutopilotProfile, VehicleKind};
use astrum_sim::controllers::{ControllerCatalog, ControllerError};
use astrum_sim::scenario::{Scenario, KNOWN_MODELS};
use astrum_sim::sweep::Base;
use serde::Serialize;

use crate::api::safe_name;
use crate::race::{self, CourseView};

/// The controller catalog's file name inside the config directory.
pub const CONTROLLERS_FILE: &str = "controllers.toml";

/// The environment of every scenario without a `[course]`.
pub const DEFAULT_ENVIRONMENT: &str = "empty";

/// The environment of a race scenario (one with a `[course]`).
pub const RACE_ENVIRONMENT: &str = "race";

/// Longest scenario summary, in characters.
const SUMMARY_MAX_CHARS: usize = 200;

/// A model ASTRUM has tuned profiles for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelEntry {
    pub model: &'static str,
    pub kind: VehicleKind,
    /// `None` for multirotors.
    pub autopilot_profile: Option<AutopilotProfile>,
}

/// The known-model table, in its order.
pub fn models() -> Vec<ModelEntry> {
    KNOWN_MODELS
        .iter()
        .map(|&(model, kind, autopilot_profile)| ModelEntry { model, kind, autopilot_profile })
        .collect()
}

/// One aircraft of a listed scenario.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AircraftEntry {
    pub id: u32,
    pub model: String,
    pub kind: VehicleKind,
    pub team: u8,
}

/// One `scenario.<name>.toml` file, valid or not.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioEntry {
    pub name: String,
    /// The file, as the server reaches it (`config/scenario.<name>.toml`).
    pub path: String,
    /// The first sentence of the file's leading comment; empty without one.
    pub summary: String,
    pub environment: &'static str,
    /// Why the file is not a valid scenario; `None` when it is.
    pub error: Option<String>,
    /// The fleet, sorted by id; empty when the file is invalid.
    pub aircraft: Vec<AircraftEntry>,
    /// A race scenario's gate count; `None` for any other scenario.
    pub gates: Option<usize>,
}

/// One scenario in full.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioDoc {
    pub name: String,
    pub path: String,
    pub summary: String,
    pub environment: &'static str,
    /// The fleet with each aircraft's resolved vehicle kind (a known model may
    /// leave it out of the file), sorted by id.
    pub fleet: Vec<AircraftEntry>,
    pub scenario: Scenario,
    /// A race scenario's course, drawn from the scenario's own seed
    /// (`course.seed`, else `sim.seed`); `null` for any other scenario.
    pub course: Option<CourseView>,
}

/// Why [`Catalogs::scenario`] has no scenario to give.
#[derive(Debug, Clone, PartialEq)]
pub enum ScenarioError {
    /// The name cannot be a file name ([`safe_name`]).
    BadName(String),
    /// No `scenario.<name>.toml`.
    NotFound(String),
    /// The file exists but is not a valid scenario.
    Invalid(String),
    /// Reading the file failed.
    Io(String),
}

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadName(msg) | Self::Invalid(msg) | Self::Io(msg) => f.write_str(msg),
            Self::NotFound(name) => write!(f, "no scenario named {name:?}"),
        }
    }
}

/// Why [`Catalogs::save_scenario`] wrote nothing.
#[derive(Debug, Clone, PartialEq)]
pub enum SaveError {
    /// The name cannot be a file name ([`safe_name`]).
    BadName(String),
    /// The summary is not one line of at most [`SUMMARY_MAX_CHARS`] characters.
    BadSummary(String),
    /// The file exists and may not be replaced.
    Exists(String),
    /// Writing the file failed.
    Io(String),
}

impl fmt::Display for SaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadName(msg) | Self::BadSummary(msg) | Self::Exists(msg) | Self::Io(msg) => {
                f.write_str(msg)
            }
        }
    }
}

/// The config directory and the controller catalog read from it.
#[derive(Debug, Clone)]
pub struct Catalogs {
    config_dir: PathBuf,
    controllers: ControllerCatalog,
}

impl Catalogs {
    /// Read and validate `<config_dir>/controllers.toml`.
    pub fn load(config_dir: impl Into<PathBuf>) -> Result<Self, ControllerError> {
        let config_dir = config_dir.into();
        let controllers = ControllerCatalog::load(config_dir.join(CONTROLLERS_FILE))?;
        Ok(Self { config_dir, controllers })
    }

    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    pub fn controllers(&self) -> &ControllerCatalog {
        &self.controllers
    }

    /// `<config_dir>/scenario.<name>.toml`, for a `name` that passes
    /// [`safe_name`].
    pub fn scenario_path(&self, name: &str) -> Result<PathBuf, String> {
        let name = safe_name(name)?;
        Ok(self.config_dir.join(format!("scenario.{name}.toml")))
    }

    /// Every `scenario.<name>.toml` directly in the config directory whose
    /// `<name>` passes [`safe_name`], sorted by name; invalid files are listed
    /// with their error.
    pub fn scenarios(&self) -> io::Result<Vec<ScenarioEntry>> {
        let mut entries = Vec::new();
        for dirent in std::fs::read_dir(&self.config_dir)? {
            let path = dirent?.path();
            let Some(name) = scenario_name(&path) else { continue };
            if !path.is_file() {
                continue;
            }
            entries.push(scenario_entry(&path, name)?);
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// The scenario saved as `scenario.<name>.toml`, with its course.
    pub fn scenario(&self, name: &str) -> Result<ScenarioDoc, ScenarioError> {
        let (base, path, text) = self.read_base(name)?;
        let course = match &base.course {
            Some(_) => Some(
                race::course_view(&text, None)
                    .map_err(|e| ScenarioError::Invalid(format!("{}: {e}", path.display())))?,
            ),
            None => None,
        };
        Ok(ScenarioDoc {
            name: name.to_owned(),
            path: path.display().to_string(),
            summary: summary(&text),
            environment: environment_of(&base),
            fleet: fleet(&base.scenario),
            scenario: base.scenario,
            course,
        })
    }

    /// The scenario saved as `scenario.<name>.toml`, as a run or sweep starts
    /// from it: the scenario and, for a race, its `[course]` table.
    pub fn base(&self, name: &str) -> Result<Base, ScenarioError> {
        self.read_base(name).map(|(base, ..)| base)
    }

    /// The validated scenario file `name`, its path and its text.
    fn read_base(&self, name: &str) -> Result<(Base, PathBuf, String), ScenarioError> {
        let path = self.scenario_path(name).map_err(ScenarioError::BadName)?;
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(ScenarioError::NotFound(name.to_owned()))
            }
            Err(e) => return Err(ScenarioError::Io(format!("reading {}: {e}", path.display()))),
        };
        let base = parse_base(&text)
            .map_err(|e| ScenarioError::Invalid(format!("{}: {e}", path.display())))?;
        Ok((base, path, text))
    }

    /// Save `scenario` as `scenario.<name>.toml`, preceded by a `# <summary>`
    /// line unless `summary` is empty. The file is written beside its target
    /// and renamed over it, so a reader sees the old file or the new one. An
    /// existing file is replaced only with `overwrite`. Returns the file's
    /// listing entry and `true` when the file was created (`false`: replaced).
    pub fn save_scenario(
        &self,
        name: &str,
        scenario: &Scenario,
        summary: &str,
        overwrite: bool,
    ) -> Result<(ScenarioEntry, bool), SaveError> {
        let path = self.scenario_path(name).map_err(SaveError::BadName)?;
        if summary.contains(['\n', '\r']) {
            return Err(SaveError::BadSummary("summary must be a single line".into()));
        }
        let chars = summary.chars().count();
        if chars > SUMMARY_MAX_CHARS {
            return Err(SaveError::BadSummary(format!(
                "summary must be at most {SUMMARY_MAX_CHARS} characters, got {chars}"
            )));
        }
        let existed = path.exists();
        if existed && !overwrite {
            return Err(SaveError::Exists(format!(
                "scenario \"{name}\" exists; send overwrite: true to replace it"
            )));
        }
        let header = if summary.is_empty() { String::new() } else { format!("# {summary}\n") };
        let text = format!("{header}{}", scenario.to_toml());
        let tmp = self.config_dir.join(format!(".scenario.{name}.toml.tmp"));
        let written = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, &path));
        if let Err(e) = written {
            // The target is untouched; the temporary file is not wanted.
            let _ = std::fs::remove_file(&tmp);
            return Err(SaveError::Io(format!("saving {}: {e}", path.display())));
        }
        let saved = scenario_entry(&path, name.to_owned())
            .map_err(|e| SaveError::Io(format!("reading {}: {e}", path.display())))?;
        Ok((saved, !existed))
    }
}

/// The listing entry of the scenario file `path`, named `name`.
fn scenario_entry(path: &Path, name: String) -> io::Result<ScenarioEntry> {
    let text = std::fs::read_to_string(path)?;
    let (error, aircraft, environment, gates) = match parse_base(&text) {
        Ok(base) => {
            let gates = base
                .course
                .as_ref()
                .and_then(|_| race::course_view(&text, None).ok())
                .map(|c| c.gates.len());
            (None, fleet(&base.scenario), environment_of(&base), gates)
        }
        Err(e) => (Some(e), Vec::new(), DEFAULT_ENVIRONMENT, None),
    };
    Ok(ScenarioEntry {
        name,
        path: path.display().to_string(),
        summary: summary(&text),
        environment,
        error,
        aircraft,
        gates,
    })
}

/// Parse and validate a scenario file: the scenario and, for a race, its
/// `[course]` (checked by `astrum-race`, which flies it). The error is the
/// message to show.
fn parse_base(text: &str) -> Result<Base, String> {
    let base = Base::from_toml_str(text).map_err(|e| e.to_string())?;
    if base.course.is_some() {
        race::validate(text)?;
    }
    Ok(base)
}

/// The environment a scenario belongs to.
fn environment_of(base: &Base) -> &'static str {
    if base.course.is_some() {
        RACE_ENVIRONMENT
    } else {
        DEFAULT_ENVIRONMENT
    }
}

/// `<name>` of a `scenario.<name>.toml` path whose name is [`safe_name`].
fn scenario_name(path: &Path) -> Option<String> {
    let file = path.file_name()?.to_str()?;
    let name = file.strip_prefix("scenario.")?.strip_suffix(".toml")?;
    safe_name(name).ok().map(str::to_owned)
}

/// A validated scenario's fleet, sorted by id.
fn fleet(scenario: &Scenario) -> Vec<AircraftEntry> {
    let mut fleet: Vec<AircraftEntry> = scenario
        .aircraft
        .iter()
        .filter_map(|a| {
            // A validated scenario's classes all resolve.
            let class = a.class().ok()?;
            Some(AircraftEntry { id: a.id, model: a.model.clone(), kind: class.kind, team: a.team })
        })
        .collect();
    fleet.sort_by_key(|a| a.id);
    fleet
}
