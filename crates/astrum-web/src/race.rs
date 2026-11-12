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

/// `course` with `course.gates` set to `gates` (if given): the course the
/// race binary draws then has that many gates.
pub fn with_gates(course: &toml::Table, gates: Option<u32>) -> toml::Table {
    let mut course = course.clone();
    if let Some(gates) = gates {
        course.insert("gates".to_owned(), toml::Value::Integer(i64::from(gates)));
    }
    course
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
