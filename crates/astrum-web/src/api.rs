//! The JSON API under `/api`.
//!
//! Every response is JSON; a failure is `{"error": "…"}` with a fitting
//! status. The whole API sits behind [`guard`]: a request whose `Host` is not
//! one this server answers for is refused (403, [`host_allowed`]: no DNS
//! rebinding), and on every non-GET route so is a browser request from
//! another origin (403) and a body not declared `Content-Type:
//! application/json` (415) — which a cross-site page cannot send without a
//! CORS preflight this server never grants.
//!
//! * **`GET /api/bridge`** — `{"endpoint", "run", "connected", "status"}`: the
//!   [`BridgeInfo`] plus the sim's `astrum/status` ([`RunStatus`]), or `null`
//!   when no sim answers.
//! * **`POST /api/bridge/control`** — `{"action":"pause"}`,
//!   `{"action":"resume"}`, `{"action":"step","ticks":N}` (N ≥ 1) or
//!   `{"action":"speed","value":X}` (X = 0 for unpaced, or ≥
//!   [`astrum_net::MIN_SPEED`]), relayed to the attached sim: 200
//!   `{"ok":true}`; 400 for a bad body; 409 with the sim's reason when it
//!   refuses; 503 when no sim is reachable.
//! * **`GET /api/environments`** — every environment
//!   ([`astrum_sim::catalog::Environment`]): id, name, description, locations,
//!   sim defaults, models, metric ids.
//! * **`GET /api/models`** — the models ASTRUM has profiles for
//!   ([`ModelEntry`](crate::catalogs::ModelEntry)): model, kind, autopilot
//!   profile.
//! * **`GET /api/controllers`** — the controller catalog (`{"entries": [...]}`,
//!   [`astrum_sim::controllers::ControllerSpec`] each).
//! * **`GET /api/scenarios`** — every `config/scenario.<name>.toml`
//!   ([`ScenarioEntry`](crate::catalogs::ScenarioEntry)): name, path, summary,
//!   environment, error (invalid files are listed too), fleet.
//! * **`GET /api/scenarios/<name>`** — one scenario in full
//!   ([`ScenarioDoc`](crate::catalogs::ScenarioDoc)): 400 for a name that
//!   cannot be a file name, 404 when there is none, 422 when the file is not a
//!   valid scenario.
//! * **`PUT /api/scenarios/<name>`** — save a scenario as
//!   `config/scenario.<name>.toml` ([`crate::scenarios`]): `{"scenario": {…},
//!   "summary": "…", "overwrite": false}` → 201 (created) or 200 (replaced) with
//!   its [`ScenarioEntry`](crate::catalogs::ScenarioEntry); 400 for a bad name,
//!   summary or body, 409 when the file exists and `overwrite` is not set, 422
//!   for an invalid scenario, 500 when it cannot be written.
//! * **`POST /api/scenarios/check`** — `{"scenario": {…}}` → 200
//!   `{"errors": […], "trim": [{"aircraft", "model", "ok", "error"}]}`: the
//!   scenario's errors (then no trims), or whether each aircraft whose class
//!   trims does so at its initial conditions. 400 for a bad body or more than
//!   512 aircraft.
//! * **`POST /api/runs`** — start the live interactive run
//!   ([`crate::runs`]): `{"scenario": "<name>" | {…}, "assignments":
//!   [{"aircraft", "controller", "params"}], "speed", "max_time_s"}` → 201
//!   [`RunView`](crate::runs::RunView); 400 for bad input (the message names
//!   the aircraft / parameter), 409 while a run is active, 422 for an invalid
//!   scenario, 500 when it cannot start.
//! * **`GET /api/runs/current`** — `{"run": RunView | null}`: the latest run
//!   since the server started. **`DELETE`** stops the active one (200, its
//!   final view; 409 when none is active).
//! * **`GET /api/runs/current/log?since=<seq>`** — `{"lines": [{"seq",
//!   "source", "stream", "line"}], "next"}`: the current run's output lines
//!   from `since` on (a bounded ring buffer).
//! * **`GET /api/runs/<id>/replay`**, **`GET /api/runs/<id>/metrics`** — a
//!   run's export document / [`astrum_bag::metrics::RunMetrics`]: 400 for an
//!   unsafe id, 404 for an unknown run (or one without a recording), 409
//!   while it runs.
//!
//! Zenoh's calls, file reads and process starts block, so they run on
//! tokio's blocking pool.
//!
//! [`host_allowed`]: crate::server::host_allowed

use std::time::Duration;

use astrum_net::{fetch_status, send_control, ControlRequest, NetError, RunStatus, MIN_SPEED};
use astrum_sim::catalog::environments;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Path, Request, State};
use axum::http::header::{CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};

use crate::bridge::BridgeInfo;
use crate::catalogs::{models, ScenarioError};
use crate::runs;
use crate::server::{host_allowed, origin_allowed, AppState};

/// Longest a file-like name ([`safe_name`]) may be.
pub const NAME_MAX_LEN: usize = 64;
/// How long `GET /api/bridge` waits for the sim's status: short, because the
/// page polls it and no answer is a valid answer.
const STATUS_TIMEOUT: Duration = Duration::from_millis(500);
/// How long a control request waits for the sim's verdict.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(2);

/// The API's routes; the caller wraps them in [`guard`].
pub(crate) fn router() -> Router<AppState> {
    Router::new()
        .route("/bridge", get(bridge_info))
        .route("/bridge/control", post(control))
        .route("/environments", get(|| async { json(StatusCode::OK, &environments()) }))
        .route("/models", get(|| async { json(StatusCode::OK, &models()) }))
        .route("/controllers", get(controllers))
        .route("/scenarios", get(scenarios))
        .route("/scenarios/check", post(crate::scenarios::check))
        .route("/scenarios/{name}", get(scenario).put(crate::scenarios::save))
        .route("/runs", post(runs::start))
        .route("/runs/current", get(runs::current).delete(runs::stop))
        .route("/runs/current/log", get(runs::log))
        .route("/runs/{id}/replay", get(runs::replay))
        .route("/runs/{id}/metrics", get(runs::metrics))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "no such API route") })
        .method_not_allowed_fallback(|| async {
            error(StatusCode::METHOD_NOT_ALLOWED, "method not allowed on this API route")
        })
}

async fn controllers(State(state): State<AppState>) -> Response {
    json(StatusCode::OK, state.catalogs.controllers())
}

async fn scenarios(State(state): State<AppState>) -> Response {
    let catalogs = state.catalogs.clone();
    match tokio::task::spawn_blocking(move || catalogs.scenarios()).await {
        Ok(Ok(entries)) => json(StatusCode::OK, &entries),
        Ok(Err(e)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("listing {}: {e}", state.catalogs.config_dir().display()),
        ),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn scenario(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let catalogs = state.catalogs.clone();
    let found = match tokio::task::spawn_blocking(move || catalogs.scenario(&name)).await {
        Ok(found) => found,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };
    match found {
        Ok(doc) => json(StatusCode::OK, &doc),
        Err(e) => {
            let status = match e {
                ScenarioError::BadName(_) => StatusCode::BAD_REQUEST,
                ScenarioError::NotFound(_) => StatusCode::NOT_FOUND,
                ScenarioError::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
                ScenarioError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            error(status, &e.to_string())
        }
    }
}

/// Check `name` for use as a file-name component (`config/scenario.<name>.toml`
/// and the like): 1–64 ASCII letters, digits, `_` or `-`, starting with a
/// letter or digit — no separators, dots or hidden files.
pub fn safe_name(name: &str) -> Result<&str, String> {
    let bytes = name.as_bytes();
    let ok = bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        && bytes.len() <= NAME_MAX_LEN
        && bytes.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(name)
    } else {
        Err(format!(
            "name {name:?} must be 1-{NAME_MAX_LEN} characters of A-Z, a-z, 0-9, '_' or '-', \
             starting with a letter or digit"
        ))
    }
}

/// A JSON response.
pub(crate) fn json(status: StatusCode, body: &impl Serialize) -> Response {
    // Plain structs of strings and numbers always serialise (non-finite floats
    // become `null`).
    let bytes = serde_json::to_vec(body)
        .unwrap_or_else(|e| unreachable!("API response serialisation failed: {e}"));
    (status, [(CONTENT_TYPE, "application/json")], bytes).into_response()
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: &'a str,
}

/// A `{"error": msg}` response.
pub(crate) fn error(status: StatusCode, msg: &str) -> Response {
    json(status, &ErrorBody { error: msg })
}

/// Refuse a request for a host this server does not answer for (403); on a
/// non-GET route, also one from another origin (403) or without a JSON body
/// type (415).
pub(crate) async fn guard(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if !host_allowed(req.headers(), &state.allowed_hosts) {
        let host = req.headers().get(HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
        return error(
            StatusCode::FORBIDDEN,
            &format!(
                "host {host:?} is not allowed (localhost, IP addresses and --allowed-hosts are)"
            ),
        );
    }
    if !matches!(*req.method(), Method::GET | Method::HEAD) {
        if !origin_allowed(req.headers()) {
            return error(StatusCode::FORBIDDEN, "cross-origin request refused");
        }
        if !is_json(req.headers()) {
            return error(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "the request body must be Content-Type: application/json",
            );
        }
    }
    next.run(req).await
}

/// The request declares a JSON body (`application/json`, any parameters).
fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
}

#[derive(Serialize)]
struct BridgeView {
    #[serde(flatten)]
    info: BridgeInfo,
    status: Option<RunStatus>,
}

async fn bridge_info(State(state): State<AppState>) -> Response {
    let (info, session) = state.bridge.link();
    let status = match session {
        // The closure owns the session handle, so it is dropped off the runtime too.
        Some(session) => {
            tokio::task::spawn_blocking(move || fetch_status(&session, STATUS_TIMEOUT).ok())
                .await
                .unwrap_or(None)
        }
        None => None,
    };
    json(StatusCode::OK, &BridgeView { info, status })
}

/// A `POST /api/bridge/control` body. Unit-like variants are empty structs so
/// `deny_unknown_fields` refuses extra keys on them too.
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "lowercase", deny_unknown_fields)]
enum ControlBody {
    Pause {},
    Resume {},
    Step { ticks: u64 },
    Speed { value: f64 },
}

/// Parse a control body and check it against the sim's limits.
fn parse_control(body: &[u8]) -> Result<ControlRequest, String> {
    let body: ControlBody =
        serde_json::from_slice(body).map_err(|e| format!("bad control body: {e}"))?;
    let request = match body {
        ControlBody::Pause {} => ControlRequest::Pause,
        ControlBody::Resume {} => ControlRequest::Resume,
        ControlBody::Step { ticks } => ControlRequest::Step(ticks),
        ControlBody::Speed { value } => ControlRequest::Speed(value),
    };
    request.validate(MIN_SPEED)
}

#[derive(Serialize)]
struct Done {
    ok: bool,
}

async fn control(State(state): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) => return error(rejection.status(), &rejection.body_text()),
    };
    let request = match parse_control(&body) {
        Ok(request) => request,
        Err(why) => return error(StatusCode::BAD_REQUEST, &why),
    };
    let (info, session) = state.bridge.link();
    let Some(session) = session else {
        return unavailable(&info);
    };
    let sent =
        tokio::task::spawn_blocking(move || send_control(&session, request, CONTROL_TIMEOUT));
    match sent.await {
        Ok(Ok(())) => json(StatusCode::OK, &Done { ok: true }),
        Ok(Err(NetError::Refused(why))) => error(StatusCode::CONFLICT, &why),
        Ok(Err(NetError::Timeout)) => unavailable(&info),
        Ok(Err(e)) => error(StatusCode::BAD_GATEWAY, &e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, &format!("control relay failed: {e}")),
    }
}

/// 503: no sim answers on the attachment (or nothing is attached).
fn unavailable(info: &BridgeInfo) -> Response {
    let why = match &info.endpoint {
        Some(endpoint) => format!("no sim is answering on {endpoint}"),
        None => "no sim is attached".to_owned(),
    };
    error(StatusCode::SERVICE_UNAVAILABLE, &why)
}
