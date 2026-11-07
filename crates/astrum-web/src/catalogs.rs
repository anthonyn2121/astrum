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
use serde::Serialize;

use crate::api::safe_name;

/// The controller catalog's file name inside the config directory.
pub const CONTROLLERS_FILE: &str = "controllers.toml";

/// The environment of every scenario until a scenario can name its own.
pub const DEFAULT_ENVIRONMENT: &str = "empty";

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
}

/// One scenario in full.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioDoc {
    pub name: String,
    pub path: String,
    pub summary: String,
    pub environment: &'static str,
    pub scenario: Scenario,
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
            let text = std::fs::read_to_string(&path)?;
            let (error, aircraft) = match Scenario::from_toml_str(&text) {
                Ok(scenario) => (None, fleet(&scenario)),
                Err(e) => (Some(e.to_string()), Vec::new()),
            };
            entries.push(ScenarioEntry {
                name,
                path: path.display().to_string(),
                summary: summary(&text),
                environment: DEFAULT_ENVIRONMENT,
                error,
                aircraft,
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// The scenario saved as `scenario.<name>.toml`.
    pub fn scenario(&self, name: &str) -> Result<ScenarioDoc, ScenarioError> {
        let path = self.scenario_path(name).map_err(ScenarioError::BadName)?;
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(ScenarioError::NotFound(name.to_owned()))
            }
            Err(e) => return Err(ScenarioError::Io(format!("reading {}: {e}", path.display()))),
        };
        let scenario = Scenario::from_toml_str(&text)
            .map_err(|e| ScenarioError::Invalid(format!("{}: {e}", path.display())))?;
        Ok(ScenarioDoc {
            name: name.to_owned(),
            path: path.display().to_string(),
            summary: summary(&text),
            environment: DEFAULT_ENVIRONMENT,
            scenario,
        })
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
