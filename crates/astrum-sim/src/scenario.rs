//! Scenario configuration: the geodetic ENU origin, sim timing and command
//! semantics, and the fleet to spawn. Deserialised from TOML (see
//! `config/scenario.example.toml`); unknown keys are errors, and every value is
//! validated before the sim starts.

use astrum_net::{speed_ok, MIN_SPEED};
use bevy::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

use crate::components::{AutopilotProfile, VehicleKind};

/// Top-level scenario document.
#[derive(Resource, Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub world: World,
    pub sim: Sim,
    pub aircraft: Vec<AircraftSpec>,
}

/// Local-tangent ENU grid origin (WGS84).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct World {
    pub origin_lat_deg: f64,
    pub origin_lon_deg: f64,
    /// Height above the ellipsoid (m).
    pub origin_alt_hae_m: f64,
}

/// Global simulation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Sim {
    /// Fixed-step physics rate (Hz). Physics advances exactly one step of
    /// `1/rate_hz` seconds per runner loop, regardless of wall clock.
    pub rate_hz: f64,
    /// Stop after this many simulated seconds; `0` runs indefinitely.
    #[serde(default)]
    pub max_time_s: f64,
    /// Wall-clock pace as a multiple of real time: `1` = real time (for live
    /// controllers and watching), `10` = ten times faster, `0.5` = half speed,
    /// `0` = as fast as the host allows (`astrum-sim --speed` overrides).
    /// Without external controllers (or with `lockstep`) the trajectory is
    /// identical at any speed.
    #[serde(default = "default_speed")]
    pub speed: f64,
    /// Seed for the combat RNG and each aircraft's JSBSim random generator.
    #[serde(default)]
    pub seed: u64,
    /// Threads used to step the FDMs in parallel (`astrum-sim --threads`
    /// overrides). Results are identical for any value.
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// A commanded aircraft that receives no fresh command for this long falls
    /// back to the failsafe hold. `0` disables the lease.
    #[serde(default = "default_lease")]
    pub command_lease_s: f64,
    /// Wait each tick for a command answering the last telemetry from every
    /// aircraft marked `lockstep`, making networked runs reproducible.
    #[serde(default)]
    pub lockstep: bool,
    /// Give up (exit with an error) if a lockstep command is this late.
    #[serde(default = "default_lockstep_timeout")]
    pub lockstep_timeout_s: f64,
}

fn default_speed() -> f64 {
    1.0
}
fn default_threads() -> usize {
    1
}
fn default_lease() -> f64 {
    0.5
}
fn default_lockstep_timeout() -> f64 {
    30.0
}
fn default_health() -> f32 {
    100.0
}
fn is_false(b: &bool) -> bool {
    !*b
}
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// One aircraft entity's identity and initial conditions.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AircraftSpec {
    /// Unique id — the Zenoh key suffix and command target.
    pub id: u32,
    /// JSBSim model name under `<root>/aircraft/<model>/`.
    pub model: String,
    /// Vehicle class; required for models ASTRUM has no profile for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<VehicleKind>,
    /// Built-in fixed-wing autopilot gain set; required for unknown fixed-wing models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autopilot_profile: Option<AutopilotProfile>,
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// Height above the ellipsoid (m).
    pub alt_hae_m: f64,
    /// True heading (deg).
    pub heading_deg: f64,
    /// True airspeed (m/s).
    pub tas_mps: f64,
    /// Flight-path angle (deg, + = climbing), |climb_deg| < 90. A trimmed
    /// aircraft is trimmed into this steady climb/descent; an untrimmed one just
    /// starts with its velocity along it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub climb_deg: f64,
    /// Bank angle (deg, + = right wing down), |bank_deg| < 90. A trimmed
    /// fixed-wing is trimmed into a steady coordinated turn at this bank (exact
    /// at spawn; the F-16's flight controls then roll an uncommanded aircraft
    /// out of it, ≈ 30° → 13° in 20 s while climbing). Steep banks may not trim
    /// (60° on the F-16 does not). An untrimmed aircraft just starts rolled by it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bank_deg: f64,
    /// Trim at spawn for steady flight at `climb_deg`, wings level or — when
    /// `bank_deg` ≠ 0 — in a coordinated turn. Default: fixed-wing yes,
    /// multirotor no. A trim that fails to converge is fatal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trim: Option<bool>,
    /// In a `sim.lockstep` scenario, wait for this aircraft's commands each tick.
    #[serde(default, skip_serializing_if = "is_false")]
    pub lockstep: bool,
    /// Rate (Hz) of this aircraft's telemetry and — in lockstep — of the commands
    /// the sim waits for; the last command holds in between (zero-order hold).
    /// Must divide `sim.rate_hz` evenly. Default: every tick.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_rate_hz: Option<f64>,
    /// Combat side; same-team rounds never damage, and a ≥2-team scenario ends
    /// when one team is left.
    #[serde(default)]
    pub team: u8,
    /// Starting (and maximum) hit points.
    #[serde(default = "default_health")]
    pub health: f32,
    /// Gun magazine override (armed fixed-wing aircraft only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ammo: Option<u16>,
}

/// Models ASTRUM has tuned profiles for: model name (a JSBSim aircraft
/// directory), its vehicle kind, and — fixed-wing only — its autopilot gains.
/// A scenario aircraft of one of these models needs no `kind` or
/// `autopilot_profile`; any other model must state them.
pub const KNOWN_MODELS: &[(&str, VehicleKind, Option<AutopilotProfile>)] = &[
    ("c172x", VehicleKind::FixedWing, Some(AutopilotProfile::Trainer)),
    ("f16", VehicleKind::FixedWing, Some(AutopilotProfile::Fighter)),
    ("F450", VehicleKind::Multirotor, None),
];

/// An aircraft's resolved class: vehicle kind, autopilot profile, trim policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AircraftClass {
    pub kind: VehicleKind,
    /// `Some` for fixed-wing, `None` for multirotors.
    pub profile: Option<AutopilotProfile>,
    pub trim: bool,
}

impl AircraftSpec {
    /// Resolve kind / profile / trim from the explicit fields and the known-model
    /// table. An unknown model must state its `kind` (and, if fixed-wing, its
    /// `autopilot_profile`) — nothing is guessed.
    pub fn class(&self) -> Result<AircraftClass, ScenarioError> {
        let known = KNOWN_MODELS.iter().find(|(m, ..)| *m == self.model);
        let kind = match (self.kind, known) {
            (Some(k), _) => k,
            (None, Some((_, k, _))) => *k,
            (None, None) => {
                let hint = "set kind = \"fixed-wing\" | \"multirotor\"";
                return Err(
                    self.err(format!("model '{}' has no built-in profile; {hint}", self.model))
                );
            }
        };
        let profile = match kind {
            VehicleKind::Multirotor => {
                if self.autopilot_profile.is_some() {
                    return Err(
                        self.err("autopilot_profile applies to fixed-wing aircraft only".into())
                    );
                }
                None
            }
            VehicleKind::FixedWing => match (self.autopilot_profile, known) {
                (Some(p), _) => Some(p),
                (None, Some((_, VehicleKind::FixedWing, Some(p)))) => Some(*p),
                _ => {
                    return Err(self.err(format!(
                        "fixed-wing model '{}' has no built-in autopilot gains; set \
                         autopilot_profile = \"trainer\" | \"fighter\"",
                        self.model
                    )))
                }
            },
        };
        let trim = self.trim.unwrap_or(kind == VehicleKind::FixedWing);
        Ok(AircraftClass { kind, profile, trim })
    }

    /// Ticks between this aircraft's control samples at `rate_hz` (1 = every tick).
    pub fn control_period_ticks(&self, rate_hz: f64) -> u64 {
        self.control_rate_hz.map_or(1, |c| (rate_hz / c).round() as u64)
    }

    fn err(&self, msg: String) -> ScenarioError {
        ScenarioError(format!("aircraft {}: {msg}", self.id))
    }
}

/// An invalid scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioError(pub String);

impl fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ScenarioError {}

fn finite(name: &str, v: f64) -> Result<(), ScenarioError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(ScenarioError(format!("{name} must be finite, got {v}")))
    }
}

impl Scenario {
    /// Parse and validate a scenario file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ScenarioError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| ScenarioError(format!("reading scenario {}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    /// Parse and validate a scenario from TOML text.
    pub fn from_toml_str(text: &str) -> Result<Self, ScenarioError> {
        let scenario: Scenario =
            toml::from_str(text).map_err(|e| ScenarioError(format!("parsing scenario: {e}")))?;
        scenario.validate()?;
        Ok(scenario)
    }

    /// The scenario as TOML (what the sim serves on `astrum/scenario`).
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("a validated scenario always serialises")
    }

    /// Check every value; called by both constructors.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        let w = &self.world;
        finite("world.origin_lat_deg", w.origin_lat_deg)?;
        finite("world.origin_lon_deg", w.origin_lon_deg)?;
        finite("world.origin_alt_hae_m", w.origin_alt_hae_m)?;
        if w.origin_lat_deg.abs() > 90.0 {
            return Err(ScenarioError(format!(
                "world.origin_lat_deg out of range: {}",
                w.origin_lat_deg
            )));
        }
        let s = &self.sim;
        if !(s.rate_hz.is_finite() && s.rate_hz > 0.0) {
            return Err(ScenarioError(format!("sim.rate_hz must be > 0, got {}", s.rate_hz)));
        }
        if !(s.max_time_s.is_finite() && s.max_time_s >= 0.0) {
            return Err(ScenarioError(format!(
                "sim.max_time_s must be >= 0, got {}",
                s.max_time_s
            )));
        }
        if !speed_ok(s.speed, MIN_SPEED) {
            return Err(ScenarioError(format!(
                "sim.speed must be 0 (as fast as possible) or >= {MIN_SPEED}, got {}",
                s.speed
            )));
        }
        if !(s.command_lease_s.is_finite() && s.command_lease_s >= 0.0) {
            return Err(ScenarioError(format!(
                "sim.command_lease_s must be >= 0, got {}",
                s.command_lease_s
            )));
        }
        if !(s.lockstep_timeout_s.is_finite() && s.lockstep_timeout_s > 0.0) {
            return Err(ScenarioError(format!(
                "sim.lockstep_timeout_s must be > 0, got {}",
                s.lockstep_timeout_s
            )));
        }
        if self.aircraft.is_empty() {
            return Err(ScenarioError("scenario has no [[aircraft]]".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for a in &self.aircraft {
            if !seen.insert(a.id) {
                return Err(ScenarioError(format!("duplicate aircraft id {}", a.id)));
            }
            for (name, v) in [
                ("lat_deg", a.lat_deg),
                ("lon_deg", a.lon_deg),
                ("alt_hae_m", a.alt_hae_m),
                ("heading_deg", a.heading_deg),
                ("tas_mps", a.tas_mps),
                ("climb_deg", a.climb_deg),
                ("bank_deg", a.bank_deg),
            ] {
                finite(&format!("aircraft {} {name}", a.id), v)?;
            }
            if a.lat_deg.abs() > 90.0 {
                return Err(a.err(format!("lat_deg out of range: {}", a.lat_deg)));
            }
            for (name, v) in [("climb_deg", a.climb_deg), ("bank_deg", a.bank_deg)] {
                if v.abs() >= 90.0 {
                    return Err(a.err(format!("{name} must be within (-90, 90), got {v}")));
                }
            }
            if a.tas_mps < 0.0 {
                return Err(a.err(format!("tas_mps must be >= 0, got {}", a.tas_mps)));
            }
            if !(a.health.is_finite() && a.health > 0.0) {
                return Err(a.err(format!("health must be > 0, got {}", a.health)));
            }
            if a.lockstep && !s.lockstep {
                return Err(a.err("lockstep = true requires sim.lockstep = true".into()));
            }
            let class = a.class()?;
            if a.ammo.is_some() && class.kind == VehicleKind::Multirotor {
                return Err(a.err("ammo applies to armed fixed-wing aircraft only".into()));
            }
            if let Some(c) = a.control_rate_hz {
                let ratio = s.rate_hz / c;
                if !(c.is_finite() && c > 0.0 && ratio >= 1.0)
                    || (ratio - ratio.round()).abs() > 1e-9 * ratio
                {
                    return Err(a.err(format!(
                        "control_rate_hz must divide sim.rate_hz ({}) evenly, got {c}",
                        s.rate_hz
                    )));
                }
                // One late or lost sample must not trip the failsafe.
                if s.command_lease_s > 0.0 && s.command_lease_s < 2.0 / c {
                    return Err(a.err(format!(
                        "control_rate_hz {c} needs sim.command_lease_s >= {} (two control \
                         periods), got {}",
                        2.0 / c,
                        s.command_lease_s
                    )));
                }
            }
        }
        if s.lockstep && !self.aircraft.iter().any(|a| a.lockstep) {
            return Err(ScenarioError(
                "sim.lockstep = true but no aircraft has lockstep = true".into(),
            ));
        }
        Ok(())
    }
}
