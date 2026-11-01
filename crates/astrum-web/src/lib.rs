//! `astrum-web` — the bridge between ASTRUM's binary telemetry wire and the
//! browser-based 3-D replay renderer (`web/`).
//!
//! It defines ONE JSON shape ([`FrameJson`]) used by both delivery paths:
//!
//! * **live / replay stream** — the `astrum-web` binary ([`server`]) tracks the
//!   newest telemetry of every aircraft (the live sim *or* `astrum-bag-play`) and
//!   pushes rate-limited fleet snapshots ([`snapshot_message`]) to every
//!   connected browser over WebSocket, while also serving the static renderer.
//! * **offline export** — the `astrum-web-export` binary ([`export`]) reads an
//!   MCAP bag and bakes it into a single [`ExportDoc`] the renderer can open with
//!   no running services.
//!
//! Keeping the shape here means the live stream and the exporter cannot drift.

pub mod export;
pub mod server;

use astrum_proto::{ControlMode, Telemetry, VehicleKind};
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

/// The renderer's name for who flew the aircraft on a tick.
pub fn control_mode_str(mode: ControlMode) -> &'static str {
    match mode {
        ControlMode::Uncommanded => "uncommanded",
        ControlMode::Raw => "raw",
        ControlMode::Autopilot => "autopilot",
        ControlMode::Failsafe => "failsafe",
        ControlMode::Unknown(_) => "unknown",
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
            control_mode: control_mode_str(t.control_mode),
            team: t.combat.team,
            hit_points: t.combat.hit_points,
            hit_points_max: t.combat.hit_points_max,
            alive: t.combat.alive,
            ammo: t.combat.ammo,
        }
    }
}

/// One live WebSocket message: the newest state of every tracked aircraft.
#[derive(Serialize)]
struct FramesMessage<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    frames: &'a [FrameJson],
}

/// Encode a fleet snapshot as the live stream's message,
/// `{"type":"frames","frames":[FrameJson…]}`.
pub fn snapshot_message(snapshot: &[Telemetry]) -> String {
    let frames: Vec<FrameJson> = snapshot.iter().map(FrameJson::from_telemetry).collect();
    // Serialising plain structs of numbers / static strings into a String
    // cannot fail (non-finite floats become `null`).
    serde_json::to_string(&FramesMessage { kind: "frames", frames: &frames })
        .unwrap_or_else(|e| unreachable!("FrameJson serialisation failed: {e}"))
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
    /// Distinct aircraft ids, ascending.
    pub aircraft: Vec<u32>,
    /// Sim-time span covered (s); both 0 for an empty export.
    pub sim_time_start: f64,
    pub sim_time_end: f64,
    /// The scenario TOML the recorder embedded, if any.
    pub scenario: Option<String>,
    /// Telemetry messages skipped because they failed to decode or their
    /// payload id disagreed with their topic.
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
}

impl ExportDoc {
    pub const KIND: &'static str = "astrum-replay";
    pub const VERSION: u32 = 2;

    /// Assemble a document from frames in bag order.
    pub fn new(frames: Vec<ExportFrame>, scenario: Option<String>, rejected: u64) -> Self {
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
        let meta = ExportMeta {
            frames: frames.len() as u64,
            aircraft,
            sim_time_start,
            sim_time_end,
            scenario,
            rejected,
        };
        Self { kind: Self::KIND, version: Self::VERSION, meta, frames }
    }
}
