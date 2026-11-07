//! `aircraft_models` reads the fleet's JSBSim models out of a scenario TOML.

use astrum_bag::{aircraft_models, AircraftModel};

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
fn a_model_serialises_as_id_and_model() {
    assert_eq!(
        serde_json::to_value(model(1, "f16")).expect("json"),
        serde_json::json!({ "id": 1, "model": "f16" })
    );
}
