//! A race scenario: an ordinary `astrum-sim` scenario plus a `[course]` table.
//!
//! ```toml
//! [world]  ...                 # the sim's own tables, unchanged
//! [sim]    ...
//! [[aircraft]]                 # exactly one: the F450 (its start pose is overwritten)
//! [course]
//! gates = 8                    # the randomization envelope (`CourseParams`) ...
//! seed = 7                     # ... drawn from this seed (default: `sim.seed`)
//! start_behind_m = 10.0        # launch this far up gate 0's approach
//! [[course.gate]]              # optional: explicit gates, used verbatim, never sampled
//! index = 0
//! east_m = 0.0
//! north_m = 20.0
//! up_m = 2.0
//! yaw_rad = 0.0
//! ```
//!
//! [`RaceScenario`] strips `[course]` and hands the rest to
//! [`astrum_sim::scenario::Scenario`], so every sim table and validation rule is the
//! sim's own. [`RaceScenario::resolve`] then fixes the course — sampled from the
//! seed, or the explicit gates — and aims the drone at it; [`ResolvedRace::to_toml`]
//! writes the *resolved* scenario back out with every gate explicit, which is what a
//! recording stores: replaying that text reproduces the course whatever the sampler
//! has become since.

use std::fmt;
use std::path::Path;

use astrum_geo::Enu;
use astrum_sim::scenario::{Scenario, ScenarioError};
use astrum_sim::world_frame;
use bevy::math::DVec3;
use serde::{Deserialize, Serialize};

use crate::course::{generate, Course, CourseError, CourseParams};
use crate::gate::{Gate, BAR_M, OPENING_M, TOP_BAR_M};

/// Tolerance when comparing a recorded gate dimension with this build's (m).
const DIM_TOL_M: f64 = 1e-9;

/// The only airframe the hull ([`crate::airframe`]) describes.
const RACE_MODEL: &str = "F450";

/// Keys of `[course]` that are not [`CourseParams`].
const SPEC_KEYS: [&str; 6] = ["seed", "start_behind_m", "opening_m", "bar_m", "gate", "length_m"];

/// The most gates a sampled course may have.
pub const MAX_GATES: u32 = 64;

/// The leg lengths a `length_m` spreads over `gates - 1` legs: each leg is drawn
/// from `LEG_MIN_FRAC..LEG_MAX_FRAC` of the mean leg `length_m / (gates - 1)`.
const LEG_MIN_FRAC: f64 = 0.7;
const LEG_MAX_FRAC: f64 = 1.3;

/// Why a race scenario could not be loaded or resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum RaceScenarioError {
    /// The text is not TOML, or `[course]` is malformed.
    Parse(String),
    /// The sim half of the scenario is invalid.
    Scenario(ScenarioError),
    /// The envelope is contradictory or the sampler could not satisfy it.
    Course(CourseError),
    /// A race-specific rule is broken (named in the message).
    Invalid(String),
}

impl fmt::Display for RaceScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(m) | Self::Invalid(m) => f.write_str(m),
            Self::Scenario(e) => e.fmt(f),
            Self::Course(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for RaceScenarioError {}

impl From<ScenarioError> for RaceScenarioError {
    fn from(e: ScenarioError) -> Self {
        Self::Scenario(e)
    }
}

impl From<CourseError> for RaceScenarioError {
    fn from(e: CourseError) -> Self {
        Self::Course(e)
    }
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, RaceScenarioError> {
    Err(RaceScenarioError::Invalid(msg.into()))
}

/// One explicit gate of `[[course.gate]]`: the centre of the opening on the world
/// grid (m) and the heading of its normal (rad; `0` = North, `+` = East).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateSpec {
    pub index: u32,
    pub east_m: f64,
    pub north_m: f64,
    pub up_m: f64,
    pub yaw_rad: f64,
}

impl GateSpec {
    fn of(g: &Gate) -> Self {
        Self {
            index: g.index,
            east_m: g.center.x,
            north_m: g.center.y,
            up_m: g.center.z,
            yaw_rad: g.yaw_rad,
        }
    }

    fn gate(&self) -> Gate {
        Gate::new(self.index, DVec3::new(self.east_m, self.north_m, self.up_m), self.yaw_rad)
    }
}

/// The non-envelope keys of `[course]`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct SpecKeys {
    seed: Option<u64>,
    start_behind_m: f64,
    opening_m: Option<f64>,
    bar_m: Option<f64>,
    gate: Vec<GateSpec>,
    length_m: Option<f64>,
}

impl Default for SpecKeys {
    fn default() -> Self {
        Self {
            seed: None,
            start_behind_m: 10.0,
            opening_m: None,
            bar_m: None,
            gate: Vec::new(),
            length_m: None,
        }
    }
}

/// Which envelope fields a file gave itself. `length_m` derives the others; a field
/// the file states always wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stated {
    pub leg_min_m: bool,
    pub leg_max_m: bool,
    pub course_radius_m: bool,
}
/// The `[course]` table.
#[derive(Debug, Clone, PartialEq)]
pub struct CourseSpec {
    /// The randomization envelope.
    pub params: CourseParams,
    /// Course seed; `None` = `sim.seed`.
    pub seed: Option<u64>,
    /// How far upstream of gate 0, on its centreline and at its height, the drone
    /// starts (m).
    pub start_behind_m: f64,
    /// Gate dimensions the gates were laid out for, as a recording notes them. Absent
    /// = this build's; present and different = the scenario cannot run here.
    pub opening_m: Option<f64>,
    pub bar_m: Option<f64>,
    /// Explicit gates, flight order. Non-empty = used verbatim, nothing is sampled.
    pub gates: Vec<GateSpec>,
    /// Target total length of the gate-to-gate legs (m, 3-D). When set, the leg
    /// limits and the course radius that the file does not state are derived from it
    /// and the gate count ([`CourseSpec::effective_params`]).
    pub length_m: Option<f64>,
    /// Which envelope fields the file stated, and so `length_m` must not derive.
    pub stated: Stated,
}

impl Default for CourseSpec {
    fn default() -> Self {
        let keys = SpecKeys::default();
        Self {
            params: CourseParams::default(),
            seed: keys.seed,
            start_behind_m: keys.start_behind_m,
            opening_m: keys.opening_m,
            bar_m: keys.bar_m,
            gates: keys.gate,
            length_m: keys.length_m,
            stated: Stated::default(),
        }
    }
}

impl CourseSpec {
    fn from_table(mut table: toml::Table) -> Result<Self, RaceScenarioError> {
        let parse = |e: toml::de::Error| RaceScenarioError::Parse(format!("parsing [course]: {e}"));
        let mut keys = toml::Table::new();
        for k in SPEC_KEYS {
            if let Some(v) = table.remove(k) {
                keys.insert(k.to_owned(), v);
            }
        }
        let stated = Stated {
            leg_min_m: table.contains_key("leg_min_m"),
            leg_max_m: table.contains_key("leg_max_m"),
            course_radius_m: table.contains_key("course_radius_m"),
        };
        let keys: SpecKeys = toml::Value::Table(keys).try_into().map_err(parse)?;
        let params: CourseParams = toml::Value::Table(table).try_into().map_err(parse)?;
        let spec = Self {
            params,
            seed: keys.seed,
            start_behind_m: keys.start_behind_m,
            opening_m: keys.opening_m,
            bar_m: keys.bar_m,
            gates: keys.gate,
            length_m: keys.length_m,
            stated,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Check every value of the table (the envelope with the sampler's own rules).
    pub fn validate(&self) -> Result<(), RaceScenarioError> {
        if !(self.start_behind_m.is_finite() && self.start_behind_m >= 0.0) {
            return invalid(format!(
                "course.start_behind_m must be >= 0, got {}",
                self.start_behind_m
            ));
        }
        for (name, given, built) in
            [("opening_m", self.opening_m, OPENING_M), ("bar_m", self.bar_m, BAR_M)]
        {
            if let Some(v) = given {
                if (v - built).abs() > DIM_TOL_M {
                    return invalid(format!(
                        "course.{name} = {v} but this build's gates have {built}: the scenario \
                         was laid out for different gates"
                    ));
                }
            }
        }
        self.effective_params()?;
        Ok(())
    }

    /// The envelope the sampler is given: [`CourseSpec::params`], with — when
    /// `length_m` is set — every field the file did not state derived from it. For a
    /// course of `g` gates the mean leg is `L / (g - 1)`, so
    /// `leg_min_m = 0.7·L/(g-1)`, `leg_max_m = 1.3·L/(g-1)` (the legs sum to about
    /// `L`) and `course_radius_m = max(0.5·L, the default radius)`. Without
    /// `length_m` the envelope is used as written. Explicit gates are not sampled, so
    /// their envelope is only checked as written.
    pub fn effective_params(&self) -> Result<CourseParams, RaceScenarioError> {
        let mut p = self.params;
        if !self.gates.is_empty() {
            p.validate()?;
            return Ok(p);
        }
        if !(2..=MAX_GATES).contains(&p.gates) {
            return invalid(format!(
                "course.gates must be between 2 and {MAX_GATES}, got {}",
                p.gates
            ));
        }
        let Some(length) = self.length_m else {
            p.validate()?;
            return Ok(p);
        };
        if !(length.is_finite() && length > 0.0) {
            return invalid(format!("course.length_m must be > 0, got {length}"));
        }
        let mean_leg = length / f64::from(p.gates - 1);
        let ctx = |what: String| {
            RaceScenarioError::Invalid(format!(
                "course.gates = {} with course.length_m = {length}: {what}",
                p.gates
            ))
        };
        if !self.stated.leg_min_m {
            p.leg_min_m = LEG_MIN_FRAC * mean_leg;
        }
        if !self.stated.leg_max_m {
            p.leg_max_m = LEG_MAX_FRAC * mean_leg;
        }
        if !self.stated.course_radius_m {
            p.course_radius_m = (0.5 * length).max(CourseParams::default().course_radius_m);
        }
        if !self.stated.leg_min_m && p.leg_min_m < p.min_gate_gap_m {
            return Err(ctx(format!(
                "too many gates for the length: the legs would average {mean_leg:.1} m (at \
                 least {:.1} m) but gates must be {} m apart",
                p.leg_min_m, p.min_gate_gap_m
            )));
        }
        if !self.stated.leg_max_m && !self.stated.course_radius_m && p.course_radius_m < p.leg_max_m
        {
            return Err(ctx(format!(
                "too few gates for the length: a leg of up to {:.1} m does not fit the \
                 {:.1} m course radius",
                p.leg_max_m, p.course_radius_m
            )));
        }
        p.validate().map_err(|e| ctx(e.to_string()))?;
        Ok(p)
    }
}

/// A parsed race scenario: the sim's scenario and the course specification.
#[derive(Debug, Clone, PartialEq)]
pub struct RaceScenario {
    pub sim: Scenario,
    pub course: CourseSpec,
}

impl RaceScenario {
    /// Parse a race scenario from a TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RaceScenarioError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            RaceScenarioError::Parse(format!("reading scenario {}: {e}", path.display()))
        })?;
        Self::from_toml_str(&text)
    }

    /// Parse and validate a race scenario. An absent `[course]` is the default
    /// envelope.
    pub fn from_toml_str(text: &str) -> Result<Self, RaceScenarioError> {
        let mut table: toml::Table = text
            .parse()
            .map_err(|e: toml::de::Error| RaceScenarioError::Parse(format!("parsing: {e}")))?;
        let course = match table.remove("course") {
            None => CourseSpec::default(),
            Some(toml::Value::Table(t)) => CourseSpec::from_table(t)?,
            Some(_) => return invalid("[course] must be a table"),
        };
        let sim: Scenario = toml::Value::Table(table)
            .try_into()
            .map_err(|e: toml::de::Error| RaceScenarioError::Parse(format!("parsing: {e}")))?;
        sim.validate()?;
        let [drone] = sim.aircraft.as_slice() else {
            return invalid(format!(
                "a race flies exactly one aircraft, the scenario has {}",
                sim.aircraft.len()
            ));
        };
        if drone.model != RACE_MODEL {
            return invalid(format!(
                "a race flies the {RACE_MODEL} (its hull is that airframe's), not '{}'",
                drone.model
            ));
        }
        if sim.world.origin_alt_hae_m != 0.0 {
            return invalid(format!(
                "world.origin_alt_hae_m must be 0 in a race: the origin plane is the ground \
                 (JSBSim's terrain), got {}",
                sim.world.origin_alt_hae_m
            ));
        }
        Ok(Self { sim, course })
    }

    /// Fix the course and aim the drone at it.
    ///
    /// The course seed is `seed_override`, else `course.seed`, else `sim.seed`. With
    /// explicit `[[course.gate]]` the gates are used verbatim and an override is an
    /// error (it would silently do nothing).
    pub fn resolve(&self, seed_override: Option<u64>) -> Result<ResolvedRace, RaceScenarioError> {
        let explicit = !self.course.gates.is_empty();
        if explicit && seed_override.is_some() {
            return invalid(
                "a seed override has no effect on a scenario with explicit [[course.gate]]",
            );
        }
        let seed = seed_override.or(self.course.seed).unwrap_or(self.sim.sim.seed);
        if i64::try_from(seed).is_err() {
            return invalid(format!("course seed {seed} does not fit a TOML integer (max 2^63-1)"));
        }
        let mut params = self.course.effective_params()?;
        let course = if explicit {
            params.gates = self.course.gates.len() as u32;
            let gates = self.course.gates.iter().map(GateSpec::gate).collect();
            Course::from_gates(gates, seed, params)?
        } else {
            generate(seed, &params)?
        };

        let first = course.gates[0];
        let start = first.center - self.course.start_behind_m * first.normal();
        let geo =
            world_frame(&self.sim).0.enu_to_geodetic(&Enu { e: start.x, n: start.y, u: start.z });
        let mut scenario = self.sim.clone();
        let drone = &mut scenario.aircraft[0];
        drone.lat_deg = geo.lat_deg();
        drone.lon_deg = geo.lon_deg();
        drone.alt_hae_m = geo.alt_hae_m;
        drone.heading_deg = first.yaw_rad.to_degrees();
        drone.tas_mps = 0.0;
        scenario.validate()?;
        let spec = CourseSpec { params, ..self.course.clone() };
        Ok(ResolvedRace { scenario, course, spec })
    }
}

/// A race with its course fixed and the drone placed on it: what is run, and what a
/// recording stores.
#[derive(Debug, Clone)]
pub struct ResolvedRace {
    /// The sim scenario, the drone's start pose applied.
    pub scenario: Scenario,
    pub course: Course,
    spec: CourseSpec,
}

impl ResolvedRace {
    /// The full scenario as TOML: the sim's tables plus `[course]` with the
    /// envelope, the effective `seed`, `start_behind_m`, the gate dimensions and
    /// every gate explicit. [`RaceScenario::from_toml_str`] of this text resolves to
    /// the same course and start.
    pub fn to_toml(&self) -> String {
        let mut table: toml::Table = self
            .scenario
            .to_toml()
            .parse()
            .expect("a validated scenario serialises to TOML it can parse");
        let mut course = match toml::Value::try_from(self.spec.params) {
            Ok(toml::Value::Table(t)) => t,
            _ => unreachable!("course parameters serialise to a table"),
        };
        course.insert(
            "seed".into(),
            toml::Value::Integer(i64::try_from(self.course.seed).expect("checked in resolve")),
        );
        course.insert("start_behind_m".into(), self.spec.start_behind_m.into());
        if let Some(length) = self.spec.length_m {
            course.insert("length_m".into(), length.into());
        }
        course.insert("opening_m".into(), OPENING_M.into());
        course.insert("bar_m".into(), BAR_M.into());
        let gates = self
            .course
            .gates
            .iter()
            .map(|g| toml::Value::try_from(GateSpec::of(g)).expect("a gate serialises"))
            .collect();
        course.insert("gate".into(), toml::Value::Array(gates));
        table.insert("course".into(), toml::Value::Table(course));
        toml::to_string(&table).expect("a scenario table serialises")
    }

    /// The course as the viewer and the web API see it.
    pub fn view(&self) -> CourseView {
        CourseView {
            seed: self.course.seed,
            opening_m: OPENING_M,
            bar_m: BAR_M,
            top_bar_m: TOP_BAR_M,
            length_m: self.course.total_length_m(),
            gates: self
                .course
                .gates
                .iter()
                .map(|g| GateView {
                    index: g.index,
                    east_m: g.center.x,
                    north_m: g.center.y,
                    up_m: g.center.z,
                    yaw_rad: g.yaw_rad,
                })
                .collect(),
        }
    }
}

/// One gate of a [`CourseView`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateView {
    pub index: u32,
    pub east_m: f64,
    pub north_m: f64,
    pub up_m: f64,
    pub yaw_rad: f64,
}

/// A race course as JSON for the web studio: the seed it came from, the gate
/// dimensions and every gate's pose. The gate's local basis is left-handed
/// (`right = (cos yaw, -sin yaw)`, `normal = (sin yaw, cos yaw)`, up = +Z).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CourseView {
    pub seed: u64,
    pub opening_m: f64,
    pub bar_m: f64,
    pub top_bar_m: f64,
    /// The realised total length of the legs, gate to gate (m, 3-D).
    pub length_m: f64,
    pub gates: Vec<GateView>,
}

/// The course of a race scenario's TOML: sampled from its envelope (with
/// `seed_override`, else its own seed) or its explicit gates.
pub fn course_view(
    scenario_toml: &str,
    seed_override: Option<u64>,
) -> Result<CourseView, RaceScenarioError> {
    Ok(RaceScenario::from_toml_str(scenario_toml)?.resolve(seed_override)?.view())
}

/// Does this scenario text have a `[course]` table, i.e. is it a race?
pub fn is_race_scenario(toml_text: &str) -> bool {
    toml_text
        .parse::<toml::Table>()
        .is_ok_and(|t| matches!(t.get("course"), Some(toml::Value::Table(_))))
}
