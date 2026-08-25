//! ECS components and resources for the ASTRUM state server.

use astrum_fdm::Fdm;
use astrum_geo::LocalFrame;
use bevy::prelude::*;
use serde::Deserialize;

/// Marks an entity as an aircraft and carries its stable id (Zenoh telemetry key
/// / command target).
#[derive(Component, Debug, Clone, Copy)]
pub struct Aircraft {
    pub id: u32,
}

/// Which flight-vehicle class an entity is. Selects the JSBSim initialization
/// (multirotors engage `fcs/ScasEngage`) and which built-in autopilot law
/// [`crate::net::autopilot_step`] applies. The raw actuator path is identical for
/// both — only the *meaning* of the four `fcs/*-cmd-norm` inputs differs
/// (aerodynamic surfaces vs. body-rate + collective demands).
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VehicleKind {
    /// Aerodynamic fixed-wing (e.g. `c172x`): control surfaces + throttle.
    #[default]
    FixedWing,
    /// Multirotor / quadrotor (e.g. `F450`): the FCS turns `aileron/elevator/
    /// rudder` into body roll/pitch/yaw **rate** demands and `throttle` into
    /// **collective** thrust.
    Multirotor,
}

impl VehicleKind {
    /// Best-effort inference from a JSBSim model name, for scenarios that don't
    /// state `kind` explicitly. Known multirotors are matched; everything else is
    /// treated as fixed-wing.
    pub fn infer_from_model(model: &str) -> Self {
        match model {
            "F450" => VehicleKind::Multirotor,
            _ => VehicleKind::FixedWing,
        }
    }
}

/// Owns this entity's independent JSBSim flight-dynamics instance.
///
/// `Fdm` is `Send + Sync` (see `astrum-fdm`); Bevy only ever grants one system
/// exclusive `&mut` access at a time, so stepping it from systems is sound.
#[derive(Component)]
pub struct FdmInstance(pub Fdm);

/// The authoritative per-aircraft state, read out of JSBSim each tick. This is
/// the source of truth telemetry is encoded from; `Transform` is a derived
/// Cartesian convenience view.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct GeoState {
    /// Geodetic latitude (deg).
    pub lat_deg: f64,
    /// Longitude (deg).
    pub lon_deg: f64,
    /// Altitude above the ellipsoid (m).
    pub alt_m: f64,
    /// Roll / pitch / yaw (rad).
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    /// Local NED velocity (m/s).
    pub vel_north_mps: f64,
    pub vel_east_mps: f64,
    pub vel_down_mps: f64,
    /// True airspeed (m/s).
    pub airspeed_mps: f64,
    /// Simulation time reported by JSBSim (s).
    pub sim_time_s: f64,
}

/// The shared local-tangent ENU frame all aircraft positions are expressed in.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldFrame(pub LocalFrame);

/// Resolved fixed-step timing / termination parameters.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SimParams {
    pub rate_hz: f64,
    pub dt: f64,
    /// Stop after this many simulated seconds; `0` runs indefinitely.
    pub max_time_s: f64,
    /// Throttle each step to real-time (see [`crate::scenario::Sim::realtime`]).
    pub realtime: bool,
}
