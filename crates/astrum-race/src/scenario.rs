//! Scenario configuration for the racing environment: where the world is, how fast
//! it steps, which drone flies, and the envelope the course is drawn from.
//!
//! Deserialised from `config/race.toml`. The `[course]` table is optional — its
//! defaults ([`CourseParams`]) already describe a reasonable track.

use bevy::prelude::Resource;
use serde::Deserialize;
use std::path::Path;

use crate::course::CourseParams;

/// Top-level scenario document.
#[derive(Resource, Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RaceScenario {
    pub world: World,
    pub sim: Sim,
    pub drone: DroneSpec,
    #[serde(default)]
    pub course: CourseParams,
}

/// Local-tangent ENU grid origin (WGS84). The origin's ground plane is the course's
/// `z = 0`, so gate altitudes are heights above the field.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct World {
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub origin_alt_m: f64,
}

/// Global simulation parameters.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sim {
    /// Fixed-step physics rate (Hz); physics advances exactly one step per loop.
    pub rate_hz: f64,
    /// Stop after this many simulated seconds; `0` (default) runs indefinitely.
    #[serde(default)]
    pub max_time_s: f64,
    /// When `true` (default) each step is throttled to `1/rate_hz` of wall time;
    /// when `false` it runs as fast as the host allows. Pacing changes wall-clock
    /// timing only — never the trajectory.
    #[serde(default = "default_realtime")]
    pub realtime: bool,
    /// Seed for course randomization. A fixed seed is a fixed track.
    #[serde(default)]
    pub seed: u64,
}

fn default_realtime() -> bool {
    true
}

/// The racing drone's identity and where it starts.
///
/// The start is derived from the course rather than configured as a lat/lon: the
/// drone lines up on the first gate's approach, which is the only position that
/// makes sense across randomized tracks.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DroneSpec {
    /// Unique id (telemetry key / command target).
    pub id: u32,
    /// JSBSim model name under `<root>/aircraft/<model>/` — the F450 quad.
    pub model: String,
    /// How far upstream of gate 1 to start, along that gate's normal (m).
    #[serde(default = "default_start_behind_m")]
    pub start_behind_m: f64,
    /// Launch altitude (m above the ENU origin plane). `None` (default) starts at
    /// the first gate's centre height, i.e. already on the gate's line.
    #[serde(default)]
    pub start_alt_m: Option<f64>,
}

fn default_start_behind_m() -> f64 {
    10.0
}

impl RaceScenario {
    /// Parse a scenario from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading scenario {}: {e}", path.display()))?;
        Self::from_toml_str(&text)
    }

    /// Parse (and validate) a scenario from a TOML string, so an embedded caller can
    /// build one without touching the filesystem.
    pub fn from_toml_str(text: &str) -> Result<Self, String> {
        let scenario: RaceScenario =
            toml::from_str(text).map_err(|e| format!("parsing race scenario: {e}"))?;
        scenario.validate()?;
        Ok(scenario)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sim.rate_hz <= 0.0 {
            return Err(format!("sim.rate_hz must be > 0, got {}", self.sim.rate_hz));
        }
        if self.drone.model.trim().is_empty() {
            return Err("drone.model must name a JSBSim model".to_string());
        }
        if self.drone.start_behind_m < 0.0 {
            return Err("drone.start_behind_m must not be negative".to_string());
        }
        if let Some(alt) = self.drone.start_alt_m {
            if alt < 0.0 {
                return Err("drone.start_alt_m must not be negative".to_string());
            }
        }
        // Course parameters are checked with the sampler's own rules, so a bad
        // envelope fails at load time rather than after the app is standing.
        self.course.validate().map_err(|e| e.to_string())
    }
}
