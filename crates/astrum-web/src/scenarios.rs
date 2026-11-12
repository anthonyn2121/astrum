//! The studio's scenario writes and checks: `PUT /api/scenarios/<name>` saves
//! a scenario as `config/scenario.<name>.toml`, and `POST /api/scenario-check`
//! reports whether a scenario is valid and whether its fixed-wing aircraft
//! trim, without writing anything.
//!
//! A scenario sent as JSON is checked as the sim would read it: its own
//! validation, then each aircraft's model as a plain name (it names a
//! directory under the JSBSim root). The trim checks run on the blocking pool,
//! since each one loads a JSBSim model.

use astrum_sim::scenario::{AircraftSpec, Scenario};
use astrum_sim::trim;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{error, json, safe_name};
use crate::catalogs::SaveError;
use crate::server::AppState;

/// Most aircraft a scenario may have to be checked; each trim is a JSBSim load.
const CHECK_MAX_AIRCRAFT: usize = 512;

/// A `PUT /api/scenarios/<name>` body.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveBody {
    scenario: Value,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    overwrite: bool,
}

/// A `POST /api/scenario-check` body.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckBody {
    scenario: Value,
}

/// The check's answer: what is wrong with the scenario, and each trim.
#[derive(Serialize)]
struct CheckView {
    errors: Vec<String>,
    trim: Vec<TrimView>,
}

/// One aircraft's trim check.
#[derive(Serialize)]
struct TrimView {
    aircraft: u32,
    model: String,
    ok: bool,
    error: Option<String>,
}

/// Parse a scenario sent as JSON and check it: its own validation, then that
/// each aircraft's model is a plain name. The error text is the response's.
fn parse_scenario(doc: Value) -> Result<Scenario, String> {
    let scenario: Scenario = serde_json::from_value(doc).map_err(|e| format!("scenario: {e}"))?;
    scenario.validate().map_err(|e| format!("scenario: {e}"))?;
    for a in &scenario.aircraft {
        safe_name(&a.model).map_err(|e| format!("scenario: aircraft {}: model: {e}", a.id))?;
    }
    Ok(scenario)
}

/// `PUT /api/scenarios/<name>`: 201 when created, 200 when replaced.
pub(crate) async fn save(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => return error(rejection.status(), &rejection.body_text()),
    };
    let body: SaveBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad scenario body: {e}")),
    };
    let SaveBody { scenario, summary, overwrite } = body;
    let scenario = match parse_scenario(scenario) {
        Ok(scenario) => scenario,
        Err(why) => return error(StatusCode::UNPROCESSABLE_ENTITY, &why),
    };
    let catalogs = state.catalogs.clone();
    let saved = tokio::task::spawn_blocking(move || {
        catalogs.save_scenario(&name, &scenario, &summary, overwrite)
    })
    .await;
    match saved {
        Ok(Ok((entry, true))) => json(StatusCode::CREATED, &entry),
        Ok(Ok((entry, false))) => json(StatusCode::OK, &entry),
        Ok(Err(e)) => {
            let status = match e {
                SaveError::BadName(_) | SaveError::BadSummary(_) => StatusCode::BAD_REQUEST,
                SaveError::Exists(_) => StatusCode::CONFLICT,
                SaveError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            error(status, &e.to_string())
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("saving the scenario: {e}")),
    }
}

/// `POST /api/scenario-check`: problems are data (200), not HTTP errors.
pub(crate) async fn check(body: Result<Bytes, BytesRejection>) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => return error(rejection.status(), &rejection.body_text()),
    };
    let body: CheckBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, &format!("bad check body: {e}")),
    };
    let aircraft = body.scenario.get("aircraft").and_then(Value::as_array).map_or(0, Vec::len);
    if aircraft > CHECK_MAX_AIRCRAFT {
        return error(
            StatusCode::BAD_REQUEST,
            &format!(
                "scenario has {aircraft} aircraft; a check takes at most {CHECK_MAX_AIRCRAFT}"
            ),
        );
    }
    let scenario = match parse_scenario(body.scenario) {
        Ok(scenario) => scenario,
        Err(why) => {
            return json(StatusCode::OK, &CheckView { errors: vec![why], trim: Vec::new() })
        }
    };
    match tokio::task::spawn_blocking(move || trim_checks(&scenario)).await {
        Ok(trim) => json(StatusCode::OK, &CheckView { errors: Vec::new(), trim }),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("checking the trims: {e}")),
    }
}

/// The trim check of every aircraft whose class trims, in id order.
fn trim_checks(scenario: &Scenario) -> Vec<TrimView> {
    let mut aircraft: Vec<&AircraftSpec> =
        scenario.aircraft.iter().filter(|a| a.class().is_ok_and(|c| c.trim)).collect();
    aircraft.sort_by_key(|a| a.id);
    aircraft
        .into_iter()
        .map(|a| {
            let result = trim::check_aircraft(scenario, a);
            TrimView {
                aircraft: a.id,
                model: a.model.clone(),
                ok: result.is_ok(),
                error: result.err(),
            }
        })
        .collect()
}
