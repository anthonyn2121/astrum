//! `aircraft_models` reads the fleet's JSBSim models out of a scenario TOML,
//! and `models_message` encodes them for the browser.

use astrum_web::{aircraft_models, models_message, AircraftModel};

fn model(id: u32, model: &str) -> AircraftModel {
    AircraftModel { id, model: model.to_owned() }
}

#[test]
fn models_come_from_the_aircraft_tables_sorted_by_id_ignoring_other_keys() {
    let scenario = r#"
        [world]
        origin_lat_deg = 37.6
        origin_lon_deg = -122.4

        [[aircraft]]
        id = 7
        model = "c172x"
        kind = "fixed-wing"
        team = 2
        [aircraft.initial]
        alt_m = 1500.0

        [[aircraft]]
        id = 1
        model = "f16"
        some_future_key = { nested = true }
    "#;
    assert_eq!(aircraft_models(scenario), vec![model(1, "f16"), model(7, "c172x")]);
}

#[test]
fn entries_missing_an_id_or_model_are_skipped() {
    let scenario = r#"
        [[aircraft]]
        id = 3
        kind = "multirotor"

        [[aircraft]]
        model = "f16"

        [[aircraft]]
        id = 2
        model = "f16nopid"
    "#;
    assert_eq!(aircraft_models(scenario), vec![model(2, "f16nopid")]);
}

#[test]
fn unparseable_or_fleetless_scenarios_have_no_models() {
    assert!(aircraft_models("[[aircraft]\nid = 1").is_empty());
    assert!(aircraft_models("aircraft = 3").is_empty());
    assert!(aircraft_models("[world]\norigin_lat_deg = 37.6\n").is_empty());
    assert!(aircraft_models("").is_empty());
}

#[test]
fn models_message_is_tagged_and_lists_id_and_model() {
    assert_eq!(
        models_message(&[model(1, "f16"), model(2, "c172x")]),
        r#"{"type":"models","models":[{"id":1,"model":"f16"},{"id":2,"model":"c172x"}]}"#
    );
    assert_eq!(models_message(&[]), r#"{"type":"models","models":[]}"#);
}
