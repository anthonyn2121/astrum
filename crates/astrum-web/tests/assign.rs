//! The assignment helpers through their public API: how a request's scenario is
//! resolved (a saved name, an inline document, or a refusal), how assignments
//! are checked against a scenario and the controller catalog (each refusal
//! names its aircraft), which aircraft are put in lockstep, and whether a
//! controller's program can start.

mod common;

use astrum_sim::scenario::Scenario;
use astrum_web::assign::{assign, find_program, scenario_input, AssignmentBody, InputError};
use astrum_web::catalogs::Catalogs;
use common::{repo_catalogs, repo_config};
use serde_json::{json, Value};

/// Three aircraft: 1 and 3 fixed-wing, 2 a multirotor that starts in lockstep
/// (so a test can see it cleared when it is not controlled).
fn fleet_json() -> Value {
    json!({
        "world": {"origin_lat_deg": 37.6188, "origin_lon_deg": -122.3756, "origin_alt_hae_m": 0.0},
        "sim": {"rate_hz": 120.0, "max_time_s": 40.0, "speed": 1.0, "lockstep": true},
        "aircraft": [
            {"id": 1, "model": "c172x", "lat_deg": 37.6188, "lon_deg": -122.3756,
             "alt_hae_m": 1524.0, "heading_deg": 90.0, "tas_mps": 55.0},
            {"id": 2, "model": "F450", "lat_deg": 37.6190, "lon_deg": -122.3756,
             "alt_hae_m": 100.0, "heading_deg": 0.0, "tas_mps": 0.0, "lockstep": true},
            {"id": 3, "model": "c172x", "lat_deg": 37.6192, "lon_deg": -122.3756,
             "alt_hae_m": 1524.0, "heading_deg": 90.0, "tas_mps": 55.0}
        ]
    })
}

fn fleet() -> Scenario {
    scenario_input(fleet_json(), &repo_catalogs()).unwrap().1
}

fn request(value: Value) -> AssignmentBody {
    serde_json::from_value(value).unwrap()
}

#[test]
fn a_saved_name_resolves_to_its_scenario() {
    let (name, scenario) = scenario_input(json!("formation"), &repo_catalogs()).unwrap();
    assert_eq!(name.as_deref(), Some("formation"));
    assert_eq!(scenario.aircraft.len(), 4);
}

#[test]
fn an_unknown_name_is_a_bad_request_naming_it() {
    let err = scenario_input(json!("missing"), &repo_catalogs()).unwrap_err();
    assert!(matches!(&err, InputError::Bad(m) if m.contains("missing")), "{err:?}");
}

#[test]
fn an_invalid_saved_scenario_is_invalid_and_named() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::copy(repo_config().join("controllers.toml"), dir.path().join("controllers.toml"))
        .unwrap();
    std::fs::write(dir.path().join("scenario.broken.toml"), "[world\n").unwrap();
    let catalogs = Catalogs::load(dir.path()).unwrap();
    let err = scenario_input(json!("broken"), &catalogs).unwrap_err();
    assert!(matches!(&err, InputError::Invalid(m) if m.contains("broken")), "{err:?}");
}

#[test]
fn an_inline_document_resolves_without_a_name() {
    let (name, scenario) = scenario_input(fleet_json(), &repo_catalogs()).unwrap();
    assert_eq!(name, None);
    assert_eq!(scenario.aircraft.len(), 3);
}

#[test]
fn an_inline_model_that_is_a_path_is_invalid_naming_the_aircraft() {
    let mut doc = fleet_json();
    doc["aircraft"][0]["kind"] = json!("fixed-wing");
    doc["aircraft"][0]["model"] = json!("../x");
    let err = scenario_input(doc, &repo_catalogs()).unwrap_err();
    assert!(
        matches!(&err, InputError::Invalid(m) if m.contains("aircraft 1") && m.contains("model")),
        "{err:?}"
    );
}

#[test]
fn a_number_is_a_bad_request() {
    let err = scenario_input(json!(7), &repo_catalogs()).unwrap_err();
    assert!(matches!(err, InputError::Bad(_)), "{err:?}");
}

#[test]
fn a_controlled_aircraft_alone_flies_in_lockstep() {
    let catalogs = repo_catalogs();
    let mut scenario = fleet();
    let (commands, views) = assign(
        &mut scenario,
        &[request(json!({"aircraft": 1, "controller": "cpp-hold"}))],
        catalogs.controllers(),
    )
    .unwrap();
    assert!(scenario.sim.lockstep);
    let lockstep: Vec<(u32, bool)> = scenario.aircraft.iter().map(|a| (a.id, a.lockstep)).collect();
    assert_eq!(lockstep, vec![(1, true), (2, false), (3, false)]);
    assert_eq!(commands.len(), 1);
    assert_eq!(views[0].aircraft, 1);
    assert_eq!(views[0].controller, "cpp-hold");
    // Every catalog parameter has a value: the catalog default for `alt`.
    assert_eq!(views[0].params["alt"], json!(1500.0));
}

#[test]
fn every_controlled_aircraft_flies_in_lockstep_and_no_other() {
    let catalogs = repo_catalogs();
    let mut scenario = fleet();
    let requests = [
        request(json!({"aircraft": 1, "controller": "cpp-hold"})),
        request(json!({"aircraft": 3, "controller": "cpp-hold"})),
    ];
    assign(&mut scenario, &requests, catalogs.controllers()).unwrap();
    let lockstep: Vec<(u32, bool)> = scenario.aircraft.iter().map(|a| (a.id, a.lockstep)).collect();
    assert_eq!(lockstep, vec![(1, true), (2, false), (3, true)]);
}

#[test]
fn a_bad_assignment_is_refused_naming_its_aircraft() {
    let catalogs = repo_catalogs();
    let cases: [(Value, &[&str]); 5] = [
        (
            json!([{"aircraft": 9, "controller": "cpp-hold"}]),
            &["aircraft 9", "not in the scenario"],
        ),
        (
            json!([
                {"aircraft": 1, "controller": "cpp-hold"},
                {"aircraft": 1, "controller": "cpp-hold"}
            ]),
            &["aircraft 1", "more than one controller"],
        ),
        (json!([{"aircraft": 1, "controller": "nope"}]), &["aircraft 1", "\"nope\""]),
        (json!([{"aircraft": 2, "controller": "cpp-hold"}]), &["aircraft 2", "does not fly"]),
        (
            json!([{"aircraft": 1, "controller": "cpp-hold", "params": {"alt": [1]}}]),
            &["aircraft 1", "`alt`"],
        ),
    ];
    for (requests, fragments) in cases {
        let requests: Vec<AssignmentBody> = serde_json::from_value(requests).unwrap();
        let err = assign(&mut fleet(), &requests, catalogs.controllers()).unwrap_err();
        for fragment in fragments {
            assert!(err.contains(fragment), "{err}");
        }
    }
}

#[test]
fn an_executable_path_can_start() {
    let me = std::env::current_exe().unwrap();
    assert_eq!(find_program(me.to_str().unwrap()), Ok(()));
}

#[test]
fn a_missing_path_names_the_program() {
    let err = find_program("no/such/astrum-controller").unwrap_err();
    assert_eq!(err, "`no/such/astrum-controller` not found or not executable");
}

#[test]
fn a_missing_bare_name_is_not_on_path() {
    let err = find_program("astrum-no-such-program-xyz").unwrap_err();
    assert_eq!(err, "`astrum-no-such-program-xyz` not found on PATH");
}
