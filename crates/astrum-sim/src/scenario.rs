//! Scenario configuration: the geodetic ENU origin, sim rate, and the fleet of
//! aircraft to spawn. Deserialised from `config/scenario.toml` (see
//! `config/scenario.example.toml`).

use bevy::prelude::Resource;
use serde::Deserialize;
use std::path::Path;

use crate::components::VehicleKind;

/// Top-level scenario document.
#[derive(Resource, Debug, Clone, Deserialize)]
pub struct Scenario {
    pub world: World,
    pub sim: Sim,
    #[serde(default)]
    pub aircraft: Vec<AircraftSpec>,
}

/// Local-tangent ENU grid origin (WGS84).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct World {
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    pub origin_alt_m: f64,
}

/// Global simulation parameters.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Sim {
    /// Fixed-step physics rate (Hz). Physics advances exactly one step of
    /// `1/rate_hz` seconds per runner loop, regardless of wall clock.
    pub rate_hz: f64,
    /// Stop after this many simulated seconds; `0` (default) runs indefinitely.
    #[serde(default)]
    pub max_time_s: f64,
    /// When `true` (default) the runner throttles each step to `1/rate_hz` of
    /// wall time (≈ real-time, for live controllers). When `false` it runs as
    /// fast as the host allows (batch / testing). Either way the trajectory —
    /// number of steps and their fixed dt — is identical.
    #[serde(default = "default_realtime")]
    pub realtime: bool,
    /// Seed for the deterministic combat RNG (gun dispersion); any fixed value
    /// makes an engagement reproducible. Default `0`.
    #[serde(default)]
    pub seed: u64,
}

fn default_realtime() -> bool {
    true
}

fn default_health() -> f32 {
    100.0
}

/// One aircraft entity's initial conditions.
#[derive(Debug, Clone, Deserialize)]
pub struct AircraftSpec {
    /// Unique id — Zenoh telemetry key and command target.
    pub id: u32,
    /// JSBSim model name under `<root>/aircraft/<model>/`.
    pub model: String,
    /// Vehicle class. Optional — when omitted it is inferred from `model`
    /// ([`VehicleKind::infer_from_model`]). State it explicitly (`kind =
    /// "multirotor"`) for models ASTRUM doesn't yet know.
    #[serde(default)]
    pub kind: Option<VehicleKind>,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_ft: f64,
    pub heading_deg: f64,
    pub airspeed_kts: f64,
    /// Combat side (design doc §2.2). Aircraft on the same team never damage each
    /// other, and the episode ends when only one team is left alive. Default `0`;
    /// a single-team scenario is non-combat and never self-terminates.
    #[serde(default)]
    pub team: u8,
    /// Starting (and maximum) hit points. Default 100.
    #[serde(default = "default_health")]
    pub health: f32,
    /// Gun ammunition override (rounds). Applies to armed (fixed-wing) aircraft;
    /// `None` (default) keeps the gun's own magazine (see `Gun::m61a1`).
    #[serde(default)]
    pub ammo: Option<u16>,
}

impl AircraftSpec {
    /// The resolved vehicle class: the explicit `kind` if given, else inferred
    /// from the model name.
    pub fn resolved_kind(&self) -> VehicleKind {
        self.kind.unwrap_or_else(|| VehicleKind::infer_from_model(&self.model))
    }
}

impl Scenario {
    /// Parse a scenario from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading scenario {}: {e}", path.display()))?;
        Self::from_toml_str(&text)
    }

    /// Parse (and validate) a scenario from a TOML string. Lets an embedded
    /// caller — e.g. the RL env ([`crate::env`]) or its Python binding — build a
    /// scenario without touching the filesystem.
    pub fn from_toml_str(text: &str) -> Result<Self, String> {
        let scenario: Scenario =
            toml::from_str(text).map_err(|e| format!("parsing scenario: {e}"))?;
        scenario.validate()?;
        Ok(scenario)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sim.rate_hz <= 0.0 {
            return Err(format!("sim.rate_hz must be > 0, got {}", self.sim.rate_hz));
        }
        let mut seen = std::collections::HashSet::new();
        for a in &self.aircraft {
            if !seen.insert(a.id) {
                return Err(format!("duplicate aircraft id {}", a.id));
            }
        }
        Ok(())
    }
}
