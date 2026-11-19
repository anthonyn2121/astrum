//! Race scenarios in the studio. A race scenario is an ordinary scenario plus a
//! `[course]` table; `astrum-race` reads it, so this is where the studio asks it
//! whether a scenario is valid and what course a seed draws.

use astrum_race::scenario::RaceScenario;
pub use astrum_race::CourseView;
use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::{scenario_toml, MAX_GATES, MIN_GATES};

/// Check a whole race scenario file (the scenario and its `[course]`) as
/// `astrum-race` reads it.
pub fn validate(text: &str) -> Result<(), String> {
    RaceScenario::from_toml_str(text).map(|_| ()).map_err(|e| e.to_string())
}

/// The course the race scenario `text` draws: from `seed`, else the scenario's
/// own seed (`course.seed`, else `sim.seed`); explicit `[[course.gate]]` are
/// taken as they are.
pub fn course_view(text: &str, seed: Option<u64>) -> Result<CourseView, String> {
    astrum_race::course_view(text, seed).map_err(|e| e.to_string())
}

/// [`course_view`] of `scenario` with its `[course]` table.
pub fn course_view_of(
    scenario: &Scenario,
    course: &toml::Table,
    seed: Option<u64>,
) -> Result<CourseView, String> {
    course_view(&scenario_toml(scenario, Some(course)), seed)
}

/// Check a requested gate count: [`MIN_GATES`]..=[`MAX_GATES`].
pub fn check_gates(gates: u32) -> Result<(), String> {
    if (MIN_GATES..=MAX_GATES).contains(&gates) {
        Ok(())
    } else {
        Err(format!("gates must be {MIN_GATES}..={MAX_GATES}, got {gates}"))
    }
}

pub use astrum_sim::sweep::check_length_m;

/// `course` with `course.gates` and `course.length_m` set to `gates` and
/// `length_m` (those given): the course the race binary draws then has that
/// many gates and that length.
pub fn with_gates(course: &toml::Table, gates: Option<u32>, length_m: Option<f64>) -> toml::Table {
    let mut course = course.clone();
    if let Some(gates) = gates {
        course.insert("gates".to_owned(), toml::Value::Integer(i64::from(gates)));
    }
    if let Some(length_m) = length_m {
        course.insert("length_m".to_owned(), toml::Value::Float(length_m));
    }
    course
}

/// How many gates a `[course]` without a `gates` key draws.
const DEFAULT_GATES: u32 = 8;

/// What a race scenario's own `[course]` asks for: the values a request's
/// `gates`, `length_m` and `seed` override. For the UI's defaults.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CourseParams {
    /// The gate count: the listed gates' number, else `course.gates`, else 8.
    pub gates: u32,
    /// `course.length_m`; `null` when the scenario sets none.
    pub length_m: Option<f64>,
    /// The seed the scenario draws its course from: `course.seed`, else `sim.seed`.
    pub seed: u64,
}

impl CourseParams {
    /// The parameters of `course` (the scenario's `[course]` table) in a scenario
    /// whose `sim.seed` is `sim_seed`.
    pub fn of(course: &toml::Table, sim_seed: u64) -> Self {
        let number = |key: &str| {
            course.get(key).and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
        };
        let listed = course.get("gate").and_then(toml::Value::as_array).map(Vec::len);
        let gates = listed
            .or_else(|| {
                course.get("gates").and_then(toml::Value::as_integer).map(|g| g.max(0) as usize)
            })
            .and_then(|g| u32::try_from(g).ok())
            .unwrap_or(DEFAULT_GATES);
        let seed = course
            .get("seed")
            .and_then(toml::Value::as_integer)
            .and_then(|s| u64::try_from(s).ok())
            .unwrap_or(sim_seed);
        Self { gates, length_m: number("length_m"), seed }
    }
}

/// Whether a `[course]` table lists its gates (`[[course.gate]]`) instead of
/// being drawn from a seed.
pub fn has_explicit_gates(course: &toml::Table) -> bool {
    course.contains_key("gate")
}

/// The course a recorded scenario flew, if it is a race scenario.
pub fn recorded_course(scenario_toml: &str) -> Option<CourseView> {
    astrum_race::is_race_scenario(scenario_toml)
        .then(|| astrum_race::course_view(scenario_toml, None).ok())
        .flatten()
}
