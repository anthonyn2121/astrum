//! ECS components and resources for the racing environment.

use astrum_fdm::Fdm;
use astrum_geo::LocalFrame;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::course::Course;

/// Marks the racing drone and carries its stable id.
#[derive(Component, Debug, Clone, Copy)]
pub struct Drone {
    pub id: u32,
}

/// Owns the drone's JSBSim flight-dynamics instance.
///
/// `Fdm` is `Send + Sync` (see `astrum-fdm`) and Bevy grants only one system
/// exclusive `&mut` access at a time, so stepping it from a system is sound. JSBSim
/// itself is process-global, so scale with processes, never threads.
#[derive(Component)]
pub struct FdmInstance(pub Fdm);

/// The drone's state, read out of JSBSim each tick.
///
/// `pos_enu` is the ENU position the gate tests use; the geodetic values it was
/// derived from are kept alongside because JSBSim, not the ENU grid, is
/// authoritative.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct DroneState {
    /// Geodetic latitude / longitude (deg) and altitude above the ellipsoid (m).
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    /// Position on the world ENU grid (m).
    pub pos_enu: DVec3,
    /// Roll / pitch / yaw (rad).
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    /// Ground velocity in ENU (m/s).
    pub vel_enu: DVec3,
    /// True airspeed (m/s).
    pub airspeed_mps: f64,
    /// Simulation time reported by JSBSim (s).
    pub sim_time_s: f64,
}

/// The shared local-tangent ENU frame the course and the drone live in.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldFrame(pub LocalFrame);

/// Resolved fixed-step timing / termination parameters.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SimParams {
    pub rate_hz: f64,
    pub dt: f64,
    /// Stop after this many simulated seconds; `0` runs indefinitely.
    pub max_time_s: f64,
    /// Throttle each step to real-time.
    pub realtime: bool,
}

/// The generated track, as a resource so any system can consult it without a query.
/// Each gate is *also* spawned as an entity carrying its [`Gate`](crate::gate::Gate).
#[derive(Resource, Debug, Clone)]
pub struct RaceCourse(pub Course);

/// Why a run ended — or that it has not.
///
/// Deliberately carries no payload, so that the value held here is exactly the value
/// published on the wire and the two can never drift. A controller that wants to know
/// *which* gate was struck reads `pending_gate`, which is that gate in all but
/// pathological cases; the exact one is named in the log line at the moment of
/// detection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Outcome {
    /// Still flying.
    #[default]
    Running,
    /// Every gate passed, in order. The only way to win.
    Complete,
    /// The airframe reached the ground plane.
    CrashedGround,
    /// The airframe struck a gate's frame.
    CrashedGate,
    /// `sim.max_time_s` expired with gates still to fly.
    TimedOut,
    /// The drone never launched (no gates, or JSBSim refused the model), so there was
    /// nothing to fly. Distinct from every other outcome because it is *our* fault,
    /// not the pilot's.
    SpawnFailed,
}

/// How far round the course the drone has got, and how the run ended.
///
/// A gate counts only when the drone crosses its plane **inside the opening**, in
/// the downstream direction, and gates must be taken in order — the same rules a
/// race would score by. `last_local` is the previous tick's position in the
/// pending gate's local frame, which is what makes the crossing detectable at all:
/// a 120 Hz tick at racing speed moves metres, so a point-in-volume test would
/// miss the 7 cm-thick plane entirely.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct RaceProgress {
    /// Index of the gate the drone is currently trying to fly through.
    pub next_gate: u32,
    /// Gates passed cleanly, in order.
    pub passed: u32,
    /// Plane crossings that missed the opening (flew round the gate).
    pub missed: u32,
    /// Simulated time the last gate was passed (s).
    pub last_pass_s: f64,
    /// Previous tick's position in the pending gate's local frame, if seen.
    pub last_local: Option<DVec3>,
    /// Set once the run is over, for whatever reason `outcome` names. This is the one
    /// flag every terminator sets and the app's exit condition reads.
    pub finished: bool,
    /// Why the run ended; [`Outcome::Running`] until it does.
    pub outcome: Outcome,
}

impl RaceProgress {
    /// End the run with `outcome`, and report whether this call is what ended it.
    ///
    /// First writer wins. That is not tidiness — it decides races between terminators
    /// that fire on the same tick, and the ordering it enforces is the one a race
    /// would judge by: crash detection runs before gate scoring, so clipping an
    /// upright while the CG crosses inside the opening is a crash, not a pass.
    pub fn finish(&mut self, outcome: Outcome) -> bool {
        if self.finished {
            return false;
        }
        self.outcome = outcome;
        self.finished = true;
        true
    }
}
