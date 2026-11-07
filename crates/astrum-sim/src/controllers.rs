//! The controller catalog (`config/controllers.toml`): the SDK controller programs a
//! web-launched run may start, each with typed, bounded parameters. The browser
//! picks an entry and parameter values; [`ControllerSpec::render`] turns them into
//! one command line for `SweepFile.controllers`, so no caller ever supplies a raw
//! command line.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::components::VehicleKind;

/// Errors loading the catalog or rendering an entry. Messages name the offending
/// entry or parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ControllerError {
    /// The catalog file could not be read or parsed, or failed validation.
    Catalog(String),
    /// `render` was given a parameter the entry does not declare.
    UnknownParam(String),
    /// A parameter without a default was not given.
    MissingParam(String),
    /// A parameter value has the wrong type, is out of bounds or is not allowed.
    BadValue { param: String, reason: String },
}

impl fmt::Display for ControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Catalog(msg) => write!(f, "controller catalog: {msg}"),
            Self::UnknownParam(p) => write!(f, "unknown controller param `{p}`"),
            Self::MissingParam(p) => write!(f, "controller param `{p}` is required"),
            Self::BadValue { param, reason } => write!(f, "controller param `{param}`: {reason}"),
        }
    }
}

impl std::error::Error for ControllerError {}

/// The value type of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ParamType {
    /// A finite float (integers are accepted).
    Number,
    /// A whole number.
    Integer,
    /// One of the entry's `choices` strings.
    Choice,
    /// Another aircraft id of the scenario.
    Aircraft,
}

/// The aircraft state a parameter commands, so a UI can prefill it from the
/// aircraft's initial conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamTarget {
    HeadingDeg,
    AltHaeM,
    TasMps,
}

/// One tunable flag of a controller program.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    pub name: String,
    pub flag: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(rename = "type")]
    pub ty: ParamType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<toml::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<ParamTarget>,
}

/// One (program, law) the web studio can launch for an aircraft.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ControllerSpec {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Program and fixed arguments; paths are relative to the repo root (the CWD).
    pub command: Vec<String>,
    /// The flag that takes the aircraft id.
    pub id_flag: String,
    /// The vehicle kinds the law flies.
    pub kinds: Vec<VehicleKind>,
    /// The law reads other aircraft's telemetry (e.g. a formation follower).
    #[serde(default)]
    pub needs_fleet: bool,
    #[serde(default, rename = "param")]
    pub params: Vec<ParamSpec>,
}

/// The parsed, validated catalog.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControllerCatalog {
    entries: Vec<ControllerSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogFile {
    #[serde(default)]
    controller: Vec<ControllerSpec>,
}

impl ControllerCatalog {
    /// Reads and validates a catalog file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ControllerError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ControllerError::Catalog(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
            .map_err(|e| ControllerError::Catalog(format!("{}: {e}", path.display())))
    }

    /// Parses and validates catalog TOML.
    pub fn from_toml_str(text: &str) -> Result<Self, ControllerError> {
        let file: CatalogFile =
            toml::from_str(text).map_err(|e| ControllerError::Catalog(e.to_string()))?;
        let mut ids = BTreeSet::new();
        for spec in &file.controller {
            if !ids.insert(spec.id.as_str()) {
                return Err(ControllerError::Catalog(format!(
                    "duplicate controller id `{}`",
                    spec.id
                )));
            }
            spec.validate()
                .map_err(|e| ControllerError::Catalog(format!("controller `{}`: {e}", spec.id)))?;
        }
        Ok(Self { entries: file.controller })
    }

    /// Entries in catalog order.
    pub fn entries(&self) -> &[ControllerSpec] {
        &self.entries
    }

    /// The entry with this id.
    pub fn get(&self, id: &str) -> Option<&ControllerSpec> {
        self.entries.iter().find(|e| e.id == id)
    }
}

/// An argument that `shlex` can quote and that holds no run-file placeholder brace.
fn check_arg(arg: &str) -> Result<(), String> {
    if arg.contains(['{', '}']) {
        return Err(format!("`{arg}` contains `{{` or `}}`"));
    }
    if arg.contains('\0') {
        return Err("contains a NUL byte".into());
    }
    Ok(())
}

impl ControllerSpec {
    /// Whether the law flies this vehicle kind.
    pub fn flies(&self, kind: VehicleKind) -> bool {
        self.kinds.contains(&kind)
    }

    fn validate(&self) -> Result<(), String> {
        let id_ok = !self.id.is_empty()
            && self.id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        if !id_ok {
            return Err("id must match [a-z0-9-]+".into());
        }
        if self.command.is_empty() || self.command[0].is_empty() {
            return Err("command must name a program".into());
        }
        if self.kinds.is_empty() {
            return Err("kinds must not be empty".into());
        }
        if self.id_flag.is_empty() {
            return Err("id_flag must not be empty".into());
        }
        for arg in self.command.iter().chain([&self.id_flag]) {
            check_arg(arg).map_err(|e| format!("command: {e}"))?;
        }
        let mut names = BTreeSet::new();
        let mut flags = BTreeSet::from([self.id_flag.as_str()]);
        for p in &self.params {
            if !names.insert(p.name.as_str()) {
                return Err(format!("duplicate param name `{}`", p.name));
            }
            if p.flag.is_empty() || !flags.insert(p.flag.as_str()) {
                return Err(format!("param `{}`: flag `{}` is empty or reused", p.name, p.flag));
            }
            p.validate().map_err(|e| format!("param `{}`: {e}", p.name))?;
        }
        Ok(())
    }

    /// Renders one command line (for `SweepFile.controllers`) flying `aircraft_id`:
    /// program, fixed args, `id_flag <id>`, then each param's `flag value` in
    /// catalog order, every argument quoted so `shlex::split` reproduces the argv.
    /// Missing params take their default; `fleet_ids` are the scenario's aircraft.
    pub fn render(
        &self,
        aircraft_id: u32,
        params: &BTreeMap<String, toml::Value>,
        fleet_ids: &[u32],
    ) -> Result<String, ControllerError> {
        if let Some(unknown) = params.keys().find(|k| !self.params.iter().any(|p| &p.name == *k)) {
            return Err(ControllerError::UnknownParam(unknown.clone()));
        }
        let mut argv: Vec<String> = self.command.clone();
        argv.push(self.id_flag.clone());
        argv.push(aircraft_id.to_string());
        for p in &self.params {
            let value = params
                .get(&p.name)
                .or(p.default.as_ref())
                .ok_or_else(|| ControllerError::MissingParam(p.name.clone()))?;
            let text = p
                .format(value, Some((aircraft_id, fleet_ids)))
                .map_err(|reason| ControllerError::BadValue { param: p.name.clone(), reason })?;
            argv.push(p.flag.clone());
            argv.push(text);
        }
        let quoted: Result<Vec<_>, _> = argv.iter().map(|a| shlex::try_quote(a)).collect();
        quoted
            .map(|q| q.join(" "))
            .map_err(|e| ControllerError::Catalog(format!("controller `{}`: {e}", self.id)))
    }
}

impl ParamSpec {
    fn validate(&self) -> Result<(), String> {
        check_arg(&self.flag).map_err(|e| format!("flag: {e}"))?;
        let numeric = matches!(self.ty, ParamType::Number | ParamType::Integer);
        if !numeric && (self.min.is_some() || self.max.is_some()) {
            return Err("min/max apply only to number and integer params".into());
        }
        if self.min.is_some_and(f64::is_nan) || self.max.is_some_and(f64::is_nan) {
            return Err("min/max must not be NaN".into());
        }
        if let (Some(lo), Some(hi)) = (self.min, self.max) {
            if lo > hi {
                return Err(format!("min {lo} exceeds max {hi}"));
            }
        }
        if self.ty == ParamType::Choice {
            if self.choices.is_empty() {
                return Err("a choice param needs choices".into());
            }
            for c in &self.choices {
                check_arg(c).map_err(|e| format!("choice: {e}"))?;
            }
        } else if !self.choices.is_empty() {
            return Err("choices apply only to choice params".into());
        }
        if let Some(d) = &self.default {
            self.format(d, None).map_err(|e| format!("default: {e}"))?;
        }
        Ok(())
    }

    /// Checks `value` and formats it as the flag's argument. `fleet` is
    /// `(aircraft_id, fleet_ids)` at render time; at load time aircraft params are
    /// only type-checked.
    fn format(&self, value: &toml::Value, fleet: Option<(u32, &[u32])>) -> Result<String, String> {
        let in_bounds = |x: f64| -> Result<(), String> {
            if self.min.is_some_and(|lo| x < lo) || self.max.is_some_and(|hi| x > hi) {
                let lo = self.min.map_or("-inf".into(), |v| v.to_string());
                let hi = self.max.map_or("inf".into(), |v| v.to_string());
                return Err(format!("{x} is outside [{lo}, {hi}]"));
            }
            Ok(())
        };
        match (self.ty, value) {
            (ParamType::Number, toml::Value::Float(x)) if x.is_finite() => {
                in_bounds(*x)?;
                Ok(x.to_string())
            }
            (ParamType::Number | ParamType::Integer, toml::Value::Integer(n)) => {
                in_bounds(*n as f64)?;
                Ok(n.to_string())
            }
            (ParamType::Choice, toml::Value::String(s)) => {
                if self.choices.contains(s) {
                    Ok(s.clone())
                } else {
                    Err(format!("`{s}` is not one of {:?}", self.choices))
                }
            }
            (ParamType::Aircraft, toml::Value::Integer(n)) => {
                let id = u32::try_from(*n)
                    .ok()
                    .filter(|&id| id != 0)
                    .ok_or_else(|| format!("aircraft id {n} is not in 1..=4294967295"))?;
                if let Some((own, fleet_ids)) = fleet {
                    if id == own {
                        return Err(format!("aircraft {id} is the controlled aircraft itself"));
                    }
                    if !fleet_ids.contains(&id) {
                        return Err(format!("aircraft {id} is not in the scenario"));
                    }
                }
                Ok(id.to_string())
            }
            (ty, v) => Err(format!("expected a {ty:?} value, got {v}")),
        }
    }
}
