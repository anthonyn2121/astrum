//! What there is to fly: the scenario, run and sweep files under a directory
//! (`astrum list`), the JSBSim models installed, and the environments new
//! scenarios are authored in.

use std::io;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::scenario::{Sim, KNOWN_MODELS};

/// What a TOML file is. Declaration order is listing order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FileKind {
    /// A run file: `base` without a grid (no `vary`, at most one repeat and
    /// one sample).
    Run,
    /// A scenario: `aircraft` and no `base`.
    Scenario,
    /// A sweep file: `base` with a non-empty `vary`, `repeats` > 1 or
    /// `samples` > 1.
    Sweep,
    /// Unreadable, or not TOML.
    Invalid,
}

impl FileKind {
    /// The kind of a parsed TOML document; `None`: some other TOML file (not
    /// ASTRUM's).
    pub fn of(table: &toml::Table) -> Option<Self> {
        if table.contains_key("base") {
            let grid =
                table.get("vary").and_then(toml::Value::as_table).is_some_and(|v| !v.is_empty())
                    || ["repeats", "samples"].iter().any(|key| {
                        table.get(*key).and_then(toml::Value::as_integer).is_some_and(|n| n > 1)
                    });
            Some(if grid { FileKind::Sweep } else { FileKind::Run })
        } else if table.contains_key("aircraft") {
            Some(FileKind::Scenario)
        } else {
            None
        }
    }

    /// Lower-case name, as `astrum list` shows it.
    pub fn name(self) -> &'static str {
        match self {
            FileKind::Run => "run",
            FileKind::Scenario => "scenario",
            FileKind::Sweep => "sweep",
            FileKind::Invalid => "invalid",
        }
    }
}

/// One ASTRUM file found by [`list`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CatalogEntry {
    pub path: PathBuf,
    pub kind: FileKind,
    /// The first sentence of the file's leading comment ([`summary`]); empty
    /// if it has none or could not be read.
    pub summary: String,
    /// Why the file is [`FileKind::Invalid`]; `None` otherwise.
    pub error: Option<String>,
}

/// Every scenario, run and sweep file (`*.toml`) under `dir`, recursively, plus
/// the TOML files that cannot be read or parsed (as [`FileKind::Invalid`]).
/// TOML files that are none of ASTRUM's are left out. Sorted by kind, then
/// path.
pub fn list(dir: &Path) -> io::Result<Vec<CatalogEntry>> {
    let mut paths = Vec::new();
    find_toml(dir, &mut paths)?;
    let mut entries = Vec::new();
    for path in paths {
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                let error = Some(e.to_string());
                entries.push(CatalogEntry {
                    path,
                    kind: FileKind::Invalid,
                    summary: String::new(),
                    error,
                });
                continue;
            }
        };
        match toml::from_str::<toml::Table>(&text) {
            Ok(table) => {
                if let Some(kind) = FileKind::of(&table) {
                    entries.push(CatalogEntry { path, kind, summary: summary(&text), error: None });
                }
            }
            Err(e) => {
                let error = Some(e.message().to_owned());
                let summary = summary(&text);
                entries.push(CatalogEntry { path, kind: FileKind::Invalid, summary, error });
            }
        }
    }
    entries.sort_by_cached_key(|e| (e.kind, e.path.display().to_string()));
    Ok(entries)
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

/// The first sentence of the comment block a file starts with (blank lines
/// and empty `#` lines before it skipped; the block's lines joined by spaces).
/// A sentence ends at `.`, `?` or `!` followed by a space; without one, the
/// whole block.
pub fn summary(text: &str) -> String {
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

/// The JSBSim models under `root` (a JSBSim data root): every directory
/// `<root>/aircraft/<name>/` holding `<name>.xml`, sorted by name.
pub fn jsbsim_models(root: &Path) -> io::Result<Vec<String>> {
    let mut models = Vec::new();
    for entry in std::fs::read_dir(root.join("aircraft"))? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else { continue };
        if entry.path().join(format!("{name}.xml")).is_file() {
            models.push(name);
        }
    }
    models.sort();
    Ok(models)
}

/// The fleet-level metric ids every run reports (`astrum_bag::metrics`).
pub const METRICS: &[&str] = &[
    "end_reason",
    "sim_time_s",
    "min_separation_m",
    "min_alt_hae_m",
    "max_bank_deg",
    "max_nz_g",
    "failsafe_entries",
    "controller_losses",
    "command_refusals",
    "hits",
    "kills",
];

/// A world new scenarios are authored in: where it can be placed, the
/// `[sim]` a new scenario starts with, the models it offers and the metrics
/// its runs report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Environment {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// Places to put the scenario's ENU origin; never empty.
    pub locations: Vec<Location>,
    /// Defaults for a new scenario's `[sim]`.
    pub sim: Sim,
    /// Models offered for new aircraft.
    pub models: Vec<String>,
    /// Metric ids its runs report.
    pub metrics: Vec<&'static str>,
}

/// A named scenario origin (`[world]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Location {
    pub name: &'static str,
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    /// Height above the ellipsoid (m).
    pub origin_alt_hae_m: f64,
}

/// Every environment, in display order.
pub fn environments() -> Vec<Environment> {
    vec![empty()]
}

/// The environment with id `id`, if there is one.
pub fn environment(id: &str) -> Option<Environment> {
    environments().into_iter().find(|e| e.id == id)
}

/// Nothing but the fleet: no terrain, obstacles or opponents. Its locations
/// are the origins of the bundled `config/scenario.*.toml`. The `[sim]`
/// defaults are those the bundled bounded scenarios (`demo`, `quad`) share:
/// 120 Hz (every bundled scenario's rate), 40 s so a new scenario ends by
/// itself and can be a sweep's base, real-time pace for watching, and the
/// scenario defaults for the rest. Lockstep stays off: a launch turns it on
/// for the aircraft it attaches controllers to.
fn empty() -> Environment {
    Environment {
        id: "empty",
        name: "Empty world",
        description: "Open sky over the WGS84 ellipsoid, no terrain: just the aircraft you place.",
        locations: vec![
            Location {
                name: "San Francisco International (SFO)",
                origin_lat_deg: 37.6188,
                origin_lon_deg: -122.3756,
                origin_alt_hae_m: 0.0,
            },
            Location {
                name: "Quadrotor demo origin, Minnesota (44.726° N, 93.076° W)",
                origin_lat_deg: 44.7258,
                origin_lon_deg: -93.0759,
                origin_alt_hae_m: 0.0,
            },
        ],
        sim: Sim {
            rate_hz: 120.0,
            max_time_s: 40.0,
            speed: 1.0,
            seed: 0,
            threads: 1,
            command_lease_s: 0.5,
            lockstep: false,
            lockstep_timeout_s: 30.0,
        },
        models: KNOWN_MODELS.iter().map(|(m, ..)| (*m).to_owned()).collect(),
        metrics: METRICS.to_vec(),
    }
}
