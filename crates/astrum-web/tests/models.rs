//! `models_message` encodes the fleet's JSBSim models (`astrum_bag::aircraft_models`)
//! for the browser.

use astrum_bag::AircraftModel;
use astrum_web::models_message;

fn model(id: u32, model: &str) -> AircraftModel {
    AircraftModel { id, model: model.to_owned() }
}

#[test]
fn models_message_is_tagged_and_lists_id_and_model() {
    assert_eq!(
        models_message(&[model(1, "f16"), model(2, "c172x")]),
        r#"{"type":"models","models":[{"id":1,"model":"f16"},{"id":2,"model":"c172x"}]}"#
    );
    assert_eq!(models_message(&[]), r#"{"type":"models","models":[]}"#);
}
