//! `astrum-web` — the bridge between ASTRUM's binary telemetry wire and the
//! browser-based 3-D replay renderer (`web/`).
//!
//! It defines ONE JSON shape per message kind — [`FrameJson`] for telemetry,
//! [`EventJson`] for `astrum/events`, [`astrum_bag::AircraftModel`] for the
//! fleet's JSBSim models — used by both delivery paths:
//!
//! * **live / replay stream** — the `astrum-web` binary ([`server`]) attaches a
//!   [`bridge::Bridge`] to the live sim *or* `astrum-bag-play`, thins every
//!   aircraft's telemetry by sim time and pushes rate-limited batches of it
//!   ([`frames_message`]), every event ([`events_message`]), the scenario's
//!   aircraft models ([`models_message`]) and each change of attachment
//!   ([`run_message`]) to every connected browser over WebSocket, while also
//!   serving the static renderer and the JSON [`api`] (including the studio's
//!   live interactive runs, [`runs`]).
//! * **offline export** — the `astrum-web-export` binary ([`export`]) reads an
//!   MCAP bag and bakes it into a single [`ExportDoc`] the renderer can open with
//!   no running services.
//!
//! Keeping the shapes here means the live stream and the exporter cannot drift.

pub mod api;
pub mod assign;
pub mod bridge;
pub mod catalogs;
pub mod export;
pub mod runs;
pub mod server;

use astrum_bag::{aircraft_models, AircraftModel};
use astrum_proto::{Enu, Event, EventKind, Telemetry, VehicleKind};
use serde::Serialize;

/// The renderer's name for a vehicle class (matches the scenario `kind` field
/// and the browser's mesh factory).
pub fn kind_str(kind: VehicleKind) -> &'static str {
    match kind {
        VehicleKind::FixedWing => "fixed-wing",
        VehicleKind::Multirotor => "multirotor",
        VehicleKind::Unknown(_) => "unknown",
    }
}

/// One aircraft's state after one sim step, as the browser consumes it: a flat
/// projection of [`Telemetry`] (SI units, angles in radians).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameJson {
    /// Aircraft id (stable across the run).
    pub id: u32,
    /// Vehicle class — selects the browser mesh.
    pub kind: &'static str,
    /// Sim step this state follows.
    pub tick: u64,
    /// Simulation time (s) — the renderer's master playback clock.
    pub sim_time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// Height above the WGS84 ellipsoid (m).
    pub alt_hae_m: f64,
    /// World-grid ENU position (m).
    pub enu_e: f64,
    pub enu_n: f64,
    pub enu_u: f64,
    /// World-grid ENU velocity relative to the Earth (m/s).
    pub vel_e_mps: f64,
    pub vel_n_mps: f64,
    pub vel_u_mps: f64,
    /// 3-2-1 Euler attitude relative to local-level NED (rad).
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    /// Body-axis angular rates (rad/s).
    pub p_rad_s: f64,
    pub q_rad_s: f64,
    pub r_rad_s: f64,
    /// True airspeed (m/s).
    pub tas_mps: f64,
    pub mach: f64,
    pub alpha_rad: f64,
    /// Normal load factor (g).
    pub nz_g: f64,
    pub control_mode: &'static str,
    pub team: u8,
    pub hit_points: f32,
    pub hit_points_max: f32,
    pub alive: bool,
    pub ammo: u16,
}

impl FrameJson {
    /// Project a wire telemetry sample into the browser frame shape.
    pub fn from_telemetry(t: &Telemetry) -> Self {
        Self {
            id: t.aircraft_id,
            kind: kind_str(t.vehicle_kind),
            tick: t.tick,
            sim_time_s: t.sim_time_s,
            lat_deg: t.lat_deg,
            lon_deg: t.lon_deg,
            alt_hae_m: t.alt_hae_m,
            enu_e: t.pos_enu.e,
            enu_n: t.pos_enu.n,
            enu_u: t.pos_enu.u,
            vel_e_mps: t.vel_enu.e,
            vel_n_mps: t.vel_enu.n,
            vel_u_mps: t.vel_enu.u,
            phi_rad: t.attitude.phi_rad,
            theta_rad: t.attitude.theta_rad,
            psi_rad: t.attitude.psi_rad,
            p_rad_s: t.rates.p_rad_s,
            q_rad_s: t.rates.q_rad_s,
            r_rad_s: t.rates.r_rad_s,
            tas_mps: t.tas_mps,
            mach: t.aero.mach,
            alpha_rad: t.aero.alpha_rad,
            nz_g: t.aero.nz_g,
            control_mode: t.control_mode.name(),
            team: t.combat.team,
            hit_points: t.combat.hit_points,
            hit_points_max: t.combat.hit_points_max,
            alive: t.combat.alive,
            ammo: t.combat.ammo,
        }
    }
}

/// One live WebSocket message: a batch of telemetry samples.
#[derive(Serialize)]
struct FramesMessage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    frames: &'a [FrameJson],
}

/// Encode a batch of samples as the live stream's message,
/// `{"type":"frames","frames":[FrameJson…]}`, in the batch's order (the bridge
/// sends several samples per aircraft, time-ordered per aircraft).
pub fn frames_message(samples: &[Telemetry]) -> String {
    let frames: Vec<FrameJson> = samples.iter().map(FrameJson::from_telemetry).collect();
    // Serialising plain structs of numbers / static strings into a String
    // cannot fail (non-finite floats become `null`).
    serde_json::to_string(&FramesMessage { kind: "frames", frames: &frames })
        .unwrap_or_else(|e| unreachable!("FrameJson serialisation failed: {e}"))
}

/// Kind-specific fields of an [`EventJson`], flattened into it (no tag: the
/// event's `kind` names the variant).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EventFields {
    TriggerPressed {
        aircraft_id: u32,
        ammo: u16,
    },
    TriggerReleased {
        aircraft_id: u32,
        rounds_fired: u16,
        ammo: u16,
    },
    /// `hit_points` is the target's after the hit; `pos_enu` `[e, n, u]` (m) is
    /// where the round struck.
    Hit {
        shooter_id: u32,
        target_id: u32,
        damage: f32,
        hit_points: f32,
        pos_enu: [f64; 3],
    },
    /// `killer_id` is `null` when unknown; `pos_enu` is the victim's position.
    Kill {
        victim_id: u32,
        killer_id: Option<u32>,
        pos_enu: [f64; 3],
    },
    /// Control-mode and reason names (`ControlMode::name`, `ModeReason::name`).
    ModeChange {
        aircraft_id: u32,
        from: &'static str,
        to: &'static str,
        reason: &'static str,
    },
    /// `controller-up` / `controller-down`.
    Controller {
        aircraft_id: u32,
    },
    /// `message` is the fatal error text, empty otherwise.
    RunEnded {
        reason: &'static str,
        message: String,
    },
}

/// One run event as the browser consumes it: `{"tick", "sim_time_s", "kind",
/// …kind-specific fields}` (see [`EventFields`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EventJson {
    /// The `Telemetry.tick` that first shows the event.
    pub tick: u64,
    pub sim_time_s: f64,
    /// [`EventKind::name`], e.g. `"hit"`.
    pub kind: &'static str,
    #[serde(flatten)]
    pub fields: EventFields,
}

fn enu(p: Enu) -> [f64; 3] {
    [p.e, p.n, p.u]
}

impl EventJson {
    /// Project a wire event into the browser event shape.
    pub fn from_event(e: &Event) -> Self {
        let fields = match &e.kind {
            &EventKind::TriggerPressed { aircraft_id, ammo } => {
                EventFields::TriggerPressed { aircraft_id, ammo }
            }
            &EventKind::TriggerReleased { aircraft_id, rounds_fired, ammo } => {
                EventFields::TriggerReleased { aircraft_id, rounds_fired, ammo }
            }
            &EventKind::Hit { shooter_id, target_id, damage, hit_points, pos_enu } => {
                EventFields::Hit {
                    shooter_id,
                    target_id,
                    damage,
                    hit_points,
                    pos_enu: enu(pos_enu),
                }
            }
            &EventKind::Kill { victim_id, killer_id, pos_enu } => {
                EventFields::Kill { victim_id, killer_id, pos_enu: enu(pos_enu) }
            }
            &EventKind::ModeChange { aircraft_id, from, to, reason } => EventFields::ModeChange {
                aircraft_id,
                from: from.name(),
                to: to.name(),
                reason: reason.name(),
            },
            &EventKind::ControllerUp { aircraft_id }
            | &EventKind::ControllerDown { aircraft_id } => EventFields::Controller { aircraft_id },
            EventKind::RunEnded { reason, message } => {
                EventFields::RunEnded { reason: reason.name(), message: message.clone() }
            }
        };
        Self { tick: e.tick, sim_time_s: e.sim_time_s, kind: e.kind.name(), fields }
    }
}

/// One live WebSocket message: events in the order they happened.
#[derive(Serialize)]
struct EventsMessage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    events: &'a [EventJson],
}

/// Encode a batch of events as the live stream's message,
/// `{"type":"events","events":[EventJson…]}`.
pub fn events_message(events: &[Event]) -> String {
    let events: Vec<EventJson> = events.iter().map(EventJson::from_event).collect();
    serde_json::to_string(&EventsMessage { kind: "events", events: &events })
        .unwrap_or_else(|e| unreachable!("EventJson serialisation failed: {e}"))
}

/// One live WebSocket message: the fleet's aircraft models.
#[derive(Serialize)]
struct ModelsMessage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    models: &'a [AircraftModel],
}

/// Encode the fleet's aircraft models as the live stream's message,
/// `{"type":"models","models":[{"id":1,"model":"f16"}…]}`.
pub fn models_message(models: &[AircraftModel]) -> String {
    serde_json::to_string(&ModelsMessage { kind: "models", models })
        .unwrap_or_else(|e| unreachable!("AircraftModel serialisation failed: {e}"))
}

/// One live WebSocket message: which attachment the stream now carries.
#[derive(Serialize)]
struct RunMessage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    run: u64,
    endpoint: Option<&'a str>,
}

/// Encode a run change as the live stream's message,
/// `{"type":"run","run":N,"endpoint":"tcp/…"}` (`endpoint` is `null` while
/// nothing is attached). A new `run` means everything sent before it belongs
/// to another sim or run: the browser starts a fresh timeline.
pub fn run_message(run: u64, endpoint: Option<&str>) -> String {
    serde_json::to_string(&RunMessage { kind: "run", run, endpoint })
        .unwrap_or_else(|e| unreachable!("run message serialisation failed: {e}"))
}

/// One exported telemetry sample plus the recorder's receive time.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExportFrame {
    #[serde(flatten)]
    pub frame: FrameJson,
    /// Recorder wall-clock receive time (ns since the Unix epoch).
    pub log_time_ns: u64,
}

/// Summary of an exported bag, for the renderer's UI.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExportMeta {
    pub frames: u64,
    /// Number of exported events.
    pub events: u64,
    /// Distinct aircraft ids, ascending.
    pub aircraft: Vec<u32>,
    /// Sim-time span covered by the frames (s); both 0 for an export without
    /// telemetry.
    pub sim_time_start: f64,
    pub sim_time_end: f64,
    /// The scenario TOML the recorder embedded, if any.
    pub scenario: Option<String>,
    /// The embedded scenario's aircraft models ([`aircraft_models`]), sorted by
    /// id; empty without a (parseable) scenario.
    pub models: Vec<AircraftModel>,
    /// Telemetry or event messages skipped because they failed to decode, or
    /// (telemetry) their payload id disagreed with their topic.
    pub rejected: u64,
}

/// A whole bag baked into one JSON document the browser can open offline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExportDoc {
    /// Format tag the renderer checks on load ([`ExportDoc::KIND`]).
    pub kind: &'static str,
    /// Format version the renderer checks on load ([`ExportDoc::VERSION`]).
    pub version: u32,
    pub meta: ExportMeta,
    /// Every telemetry sample, in bag (log-time) order.
    pub frames: Vec<ExportFrame>,
    /// Every event, by tick (ties in bag order).
    pub events: Vec<EventJson>,
}

impl ExportDoc {
    pub const KIND: &'static str = "astrum-replay";
    pub const VERSION: u32 = 3;

    /// Assemble a document from frames and events in bag order.
    pub fn new(
        frames: Vec<ExportFrame>,
        mut events: Vec<EventJson>,
        scenario: Option<String>,
        rejected: u64,
    ) -> Self {
        let mut aircraft: Vec<u32> = frames.iter().map(|f| f.frame.id).collect();
        aircraft.sort_unstable();
        aircraft.dedup();
        let (sim_time_start, sim_time_end) = frames
            .iter()
            .map(|f| f.frame.sim_time_s)
            .fold(None, |span: Option<(f64, f64)>, t| match span {
                None => Some((t, t)),
                Some((lo, hi)) => Some((lo.min(t), hi.max(t))),
            })
            .unwrap_or((0.0, 0.0));
        events.sort_by_key(|e| e.tick); // stable: ties keep bag order
        let models = scenario.as_deref().map(aircraft_models).unwrap_or_default();
        let meta = ExportMeta {
            frames: frames.len() as u64,
            events: events.len() as u64,
            aircraft,
            sim_time_start,
            sim_time_end,
            scenario,
            models,
            rejected,
        };
        Self { kind: Self::KIND, version: Self::VERSION, meta, frames, events }
    }
}
