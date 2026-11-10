//! The controllers a run flies: one catalog controller per aircraft.
//!
//! A request names a scenario and, for each controlled aircraft, a catalog
//! controller and its parameters. This module resolves the scenario (a saved
//! one or an inline document), checks every assignment against it and the
//! controller catalog, renders each controller's command line and sets the
//! scenario's lockstep flags, so exactly the controlled aircraft fly in
//! lockstep. Live runs and browser-launched sweeps both start from here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use astrum_sim::controllers::ControllerCatalog;
use astrum_sim::scenario::Scenario;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::safe_name;
use crate::catalogs::{Catalogs, ScenarioError};

/// One requested assignment: an aircraft, a catalog controller and its parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssignmentBody {
    pub aircraft: u32,
    pub controller: String,
    #[serde(default)]
    pub params: BTreeMap<String, Value>,
}

/// One aircraft's controller as the run flies it: every catalog parameter,
/// defaults filled in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Assignment {
    pub aircraft: u32,
    pub controller: String,
    pub params: BTreeMap<String, Value>,
}

/// Why a scenario input was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum InputError {
    /// Not a saved name or a document, or a name that is unknown or unsafe.
    Bad(String),
    /// The scenario is not valid.
    Invalid(String),
    /// The saved scenario could not be read.
    Io(String),
}

/// Resolve a request's scenario: a saved scenario's name (returned with the
/// scenario), or an inline scenario document (no name), validated, with every
/// aircraft's model a safe directory name.
pub fn scenario_input(
    input: Value,
    catalogs: &Catalogs,
) -> Result<(Option<String>, Scenario), InputError> {
    match input {
        Value::String(name) => match catalogs.scenario(&name) {
            Ok(doc) => Ok((Some(doc.name), doc.scenario)),
            Err(e @ (ScenarioError::BadName(_) | ScenarioError::NotFound(_))) => {
                Err(InputError::Bad(format!("scenario: {e}")))
            }
            Err(e @ ScenarioError::Invalid(_)) => Err(InputError::Invalid(e.to_string())),
            Err(e @ ScenarioError::Io(_)) => Err(InputError::Io(e.to_string())),
        },
        doc @ Value::Object(_) => {
            let invalid = |e: String| InputError::Invalid(format!("inline scenario: {e}"));
            let scenario: Scenario =
                serde_json::from_value(doc).map_err(|e| invalid(e.to_string()))?;
            scenario.validate().map_err(|e| invalid(e.to_string()))?;
            // The model names a directory under the JSBSim root: no paths.
            for a in &scenario.aircraft {
                safe_name(&a.model)
                    .map_err(|e| invalid(format!("aircraft {}: model: {e}", a.id)))?;
            }
            Ok((None, scenario))
        }
        _ => Err(InputError::Bad(
            "scenario must be a saved scenario's name or a scenario object".into(),
        )),
    }
}

/// Check that a controller's program can start: a path (relative to the
/// server's working directory) to an executable file, or a name on `PATH`.
pub fn find_program(program: &str) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let executable = |path: &Path| {
        std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    };
    if program.contains('/') {
        if executable(Path::new(program)) {
            return Ok(());
        }
        let hint = if program.starts_with("build/controllers/") {
            " (build the C++ controllers: cmake -S controllers -B build/controllers \
             -DCMAKE_BUILD_TYPE=Release && cmake --build build/controllers -j)"
        } else {
            ""
        };
        return Err(format!("`{program}` not found or not executable{hint}"));
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    if std::env::split_paths(&path).any(|dir| executable(&dir.join(program))) {
        Ok(())
    } else {
        Err(format!("`{program}` not found on PATH"))
    }
}

/// Check that every assignment's controller program can start (see
/// [`find_program`]). Errors name the aircraft and the controller.
pub fn check_programs(
    assignments: &[Assignment],
    catalog: &ControllerCatalog,
) -> Result<(), String> {
    for a in assignments {
        // Rendered, so the catalog has it.
        if let Some(spec) = catalog.get(&a.controller) {
            find_program(&spec.command[0]).map_err(|why| {
                format!("aircraft {}: controller `{}`: {why}", a.aircraft, a.controller)
            })?;
        }
    }
    Ok(())
}

/// Check every assignment against the scenario and the catalog, render the
/// controllers' command lines, and fly exactly the controlled aircraft in
/// lockstep. Errors name the aircraft (and the parameter).
pub fn assign(
    scenario: &mut Scenario,
    assignments: &[AssignmentBody],
    catalog: &ControllerCatalog,
) -> Result<(Vec<String>, Vec<Assignment>), String> {
    let fleet_ids: Vec<u32> = scenario.aircraft.iter().map(|a| a.id).collect();
    let mut controlled = BTreeSet::new();
    let mut commands = Vec::with_capacity(assignments.len());
    let mut views = Vec::with_capacity(assignments.len());
    for a in assignments {
        let id = a.aircraft;
        let Some(spec) = scenario.aircraft.iter().find(|s| s.id == id) else {
            return Err(format!("aircraft {id}: not in the scenario (its fleet is {fleet_ids:?})"));
        };
        if !controlled.insert(id) {
            return Err(format!("aircraft {id}: assigned more than one controller"));
        }
        let Some(controller) = catalog.get(&a.controller) else {
            return Err(format!("aircraft {id}: no controller {:?} in the catalog", a.controller));
        };
        let kind = spec.class().map_err(|e| format!("aircraft {id}: {e}"))?.kind;
        if !controller.flies(kind) {
            let kinds: Vec<&str> =
                controller.kinds.iter().map(|&k| crate::kind_str(k.into())).collect();
            return Err(format!(
                "aircraft {id}: controller {:?} does not fly {} aircraft (it flies {})",
                controller.id,
                crate::kind_str(kind.into()),
                kinds.join(", ")
            ));
        }
        let mut params = BTreeMap::new();
        for (name, value) in &a.params {
            let value = toml_value(value)
                .map_err(|why| format!("aircraft {id}: controller param `{name}`: {why}"))?;
            params.insert(name.clone(), value);
        }
        commands.push(
            controller
                .render(id, &params, &fleet_ids)
                .map_err(|e| format!("aircraft {id}: {e}"))?,
        );
        // Rendered, so every parameter has a value: given or default.
        let effective = controller
            .params
            .iter()
            .filter_map(|p| {
                let value =
                    a.params.get(&p.name).cloned().or_else(|| {
                        p.default.as_ref().and_then(|d| serde_json::to_value(d).ok())
                    })?;
                Some((p.name.clone(), value))
            })
            .collect();
        views.push(Assignment {
            aircraft: id,
            controller: controller.id.clone(),
            params: effective,
        });
    }
    scenario.sim.lockstep = !controlled.is_empty();
    for aircraft in &mut scenario.aircraft {
        aircraft.lockstep = controlled.contains(&aircraft.id);
    }
    Ok((commands, views))
}

/// A JSON parameter value as the catalog's TOML value.
fn toml_value(value: &Value) -> Result<toml::Value, String> {
    match value {
        Value::Number(n) => n
            .as_i64()
            .map(toml::Value::Integer)
            .or_else(|| n.as_f64().map(toml::Value::Float))
            .ok_or_else(|| format!("{n} is not representable")),
        Value::String(s) => Ok(toml::Value::String(s.clone())),
        Value::Bool(b) => Ok(toml::Value::Boolean(*b)),
        other => Err(format!("{other} is not a number or a string")),
    }
}
