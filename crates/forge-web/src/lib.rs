//! `forge-web` — the bridge between FORGE's binary telemetry wire and the
//! browser-based 3-D replay renderer (`web/`).
//!
//! It exposes ONE JSON shape ([`FrameJson`]) used by both delivery paths the
//! project supports:
//!
//! * **live / replay stream** — the `forge-web` binary subscribes to the
//!   telemetry `PUB` (the live sim *or* `forge-bag-play`) and pushes one
//!   [`FrameJson`] per sample to every connected browser over WebSocket, while
//!   also serving the static renderer over HTTP.
//! * **offline export** — the `forge-web-export` binary reads a bag and bakes it
//!   into a single [`ExportDoc`] the renderer can open with no running services.
//!
//! Keeping the shape here (rather than in each binary) means the wire the
//! browser sees is defined once, and the exporter and the live stream can never
//! drift apart.

use forge_proto::{TelemetrySample, VehicleKind};
use serde::Serialize;

/// The renderer's name for a vehicle class (kebab-case, matches the scenario
/// `kind` field and the browser's mesh factory).
pub fn kind_str(kind: VehicleKind) -> &'static str {
    match kind {
        VehicleKind::FixedWing => "fixed-wing",
        VehicleKind::Multirotor => "multirotor",
    }
}

/// One aircraft's state for a single sim step, as the browser consumes it. A
/// direct, self-describing projection of [`TelemetrySample`] (angles stay in
/// radians; the renderer converts). This is the payload of every WebSocket
/// message and of every entry in an [`ExportDoc`].
#[derive(Debug, Clone, Serialize)]
pub struct FrameJson {
    /// Aircraft id (stable across the run).
    pub id: u32,
    /// Vehicle class — selects the browser mesh (`fixed-wing` | `multirotor`).
    pub kind: &'static str,
    /// Simulation time (s) — the renderer's master playback clock.
    pub sim_time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    /// Local ENU grid position (m): east, north, up.
    pub enu_e: f64,
    pub enu_n: f64,
    pub enu_u: f64,
    /// Local NED velocity (m/s).
    pub vel_north_mps: f64,
    pub vel_east_mps: f64,
    pub vel_down_mps: f64,
    /// Body Euler attitude (rad): roll, pitch, yaw/heading.
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    pub airspeed_mps: f64,
    pub health: u8,
}

impl FrameJson {
    /// Project a wire telemetry sample into the browser frame shape.
    pub fn from_sample(s: &TelemetrySample) -> Self {
        Self {
            id: s.aircraft_id,
            kind: kind_str(s.vehicle_kind),
            sim_time_s: s.sim_time_s,
            lat_deg: s.lat_deg,
            lon_deg: s.lon_deg,
            alt_m: s.alt_m,
            enu_e: s.enu_e,
            enu_n: s.enu_n,
            enu_u: s.enu_u,
            vel_north_mps: s.vel_north_mps,
            vel_east_mps: s.vel_east_mps,
            vel_down_mps: s.vel_down_mps,
            phi_rad: s.phi_rad,
            theta_rad: s.theta_rad,
            psi_rad: s.psi_rad,
            airspeed_mps: s.airspeed_mps,
            health: s.health,
        }
    }
}

/// One exported bag frame: the state plus the wall-clock stamp it was recorded
/// at (kept for reference / seeking; playback keys on `sim_time_s`).
#[derive(Debug, Clone, Serialize)]
pub struct ExportFrame {
    #[serde(flatten)]
    pub frame: FrameJson,
    /// Wall-clock receive time from the bag (ns since the Unix epoch).
    pub wall_ns: u64,
}

/// Summary of an exported bag, for the renderer's UI.
#[derive(Debug, Clone, Serialize)]
pub struct ExportMeta {
    pub frames: u64,
    /// Distinct aircraft ids seen, ascending.
    pub aircraft: Vec<u32>,
    /// Sim-time span covered (s).
    pub sim_time_start: f64,
    pub sim_time_end: f64,
}

/// A whole bag baked into one JSON document the browser can open offline.
#[derive(Debug, Clone, Serialize)]
pub struct ExportDoc {
    /// Format tag the renderer checks on load.
    pub kind: &'static str,
    pub version: u32,
    pub meta: ExportMeta,
    pub frames: Vec<ExportFrame>,
}

impl ExportDoc {
    /// The document tag / current version the renderer expects.
    pub const KIND: &'static str = "forge-replay-bag";
    pub const VERSION: u32 = 1;

    /// Assemble a document from bag frames (`(wall_ns, sample)` in bag order).
    pub fn from_frames(frames: Vec<(u64, TelemetrySample)>) -> Self {
        let mut aircraft: Vec<u32> = frames.iter().map(|(_, s)| s.aircraft_id).collect();
        aircraft.sort_unstable();
        aircraft.dedup();
        let (mut t0, mut t1) = (f64::INFINITY, f64::NEG_INFINITY);
        for (_, s) in &frames {
            t0 = t0.min(s.sim_time_s);
            t1 = t1.max(s.sim_time_s);
        }
        if frames.is_empty() {
            t0 = 0.0;
            t1 = 0.0;
        }
        let meta = ExportMeta {
            frames: frames.len() as u64,
            aircraft,
            sim_time_start: t0,
            sim_time_end: t1,
        };
        let frames = frames
            .into_iter()
            .map(|(wall_ns, s)| ExportFrame { frame: FrameJson::from_sample(&s), wall_ns })
            .collect();
        Self { kind: Self::KIND, version: Self::VERSION, meta, frames }
    }
}
