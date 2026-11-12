//! A race whose drone cannot be spawned ends as `SpawnFailed` and the run as `Fatal`.
//!
//! Its own test binary: it points `JSBSIM_ROOT` at nothing, which the other tests'
//! JSBSim must not see.

mod common;

use astrum_proto::RaceOutcome;
use astrum_race::build_race_app;
use astrum_race::components::RaceProgress;
use astrum_race::scenario::RaceScenario;
use astrum_sim::components::SimStatus;
use astrum_sim::RunMode;
use bevy::prelude::*;
use common::scenario_toml;

#[test]
fn a_drone_that_cannot_spawn_stops_the_run_at_once() {
    std::env::set_var("JSBSIM_ROOT", "/nonexistent");
    let race = RaceScenario::from_toml_str(&scenario_toml("", "", "gates = 3"))
        .unwrap()
        .resolve(None)
        .unwrap();
    let mut app = build_race_app(race, RunMode::Batch);
    app.update();

    assert!(app.world().resource::<SimStatus>().fatal.is_some(), "the sim reports the failure");
    assert_eq!(app.world().resource::<RaceProgress>().outcome, RaceOutcome::SpawnFailed);
    assert!(matches!(app.should_exit(), Some(AppExit::Error(_))), "and the run does not go on");
}
