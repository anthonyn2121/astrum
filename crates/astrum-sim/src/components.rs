//! ECS components and resources for the ASTRUM state server.

use std::collections::HashMap;

use astrum_fdm::{Fdm, FdmState};
use astrum_geo::{Enu, LocalFrame};
use bevy::platform::cell::SyncCell;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Marks an entity as an aircraft and carries its stable id.
#[derive(Component, Debug, Clone, Copy)]
pub struct Aircraft {
    pub id: u32,
}

/// Which flight-vehicle class an entity is. Selects the JSBSim initialisation and
/// the built-in autopilot law. The four `fcs/*-cmd-norm` inputs mean surfaces +
/// throttle on a fixed-wing, body-rate demands + collective on a multirotor.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum VehicleKind {
    FixedWing,
    Multirotor,
}

impl From<VehicleKind> for astrum_proto::VehicleKind {
    fn from(k: VehicleKind) -> Self {
        match k {
            VehicleKind::FixedWing => astrum_proto::VehicleKind::FixedWing,
            VehicleKind::Multirotor => astrum_proto::VehicleKind::Multirotor,
        }
    }
}

/// Gain set of the built-in fixed-wing autopilot.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AutopilotProfile {
    /// Gentle trainer (c172x): 25° bank, 10° pitch.
    Trainer,
    /// Fighter (f16): 60° bank, 24° pitch, load-factor feedforward.
    Fighter,
}

/// This entity's JSBSim instance. [`Fdm`] is `Send` but deliberately not
/// `Sync`; `SyncCell` makes the component `Sync` by only ever handing out
/// exclusive access, which is all the systems need.
#[derive(Component)]
pub struct FdmInstance(pub SyncCell<Fdm>);

impl FdmInstance {
    pub fn new(fdm: Fdm) -> Self {
        Self(SyncCell::new(fdm))
    }

    pub fn fdm(&mut self) -> &mut Fdm {
        self.0.get()
    }
}

/// Which [`StepPool`](crate::systems::StepPool) shard steps this aircraft's FDM,
/// fixed at spawn so an instance's working set stays in one core's cache.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepShard(pub usize);

/// The authoritative per-aircraft state after the latest step, read from
/// JSBSim and mapped onto the world grid (all `f64`).
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
pub struct AircraftState {
    /// JSBSim's state (geodetic position, attitude, local-level NED velocity,
    /// body rates, aero angles, true airspeed).
    pub fdm: FdmState,
    /// World-grid position (m).
    pub pos_enu: Enu,
    /// World-grid position before the latest step (for swept hit tests).
    pub prev_pos_enu: Enu,
    /// Velocity relative to the Earth in the world grid (m/s).
    pub vel_enu: Enu,
}

/// The shared local-tangent ENU frame all aircraft positions are expressed in.
#[derive(Resource, Debug, Clone, Copy)]
pub struct WorldFrame(pub LocalFrame);

/// Resolved timing / termination / command parameters.
#[derive(Resource, Debug, Clone, Copy)]
pub struct SimParams {
    pub rate_hz: f64,
    pub dt: f64,
    /// Stop after this many simulated seconds; `0` runs indefinitely.
    pub max_time_s: f64,
    /// Wall-clock pace as a multiple of real time; `0` = unpaced.
    pub speed: f64,
    /// Command lease in ticks; `None` disables it.
    pub lease_ticks: Option<u64>,
    pub lockstep: bool,
    pub lockstep_timeout_s: f64,
}

/// The sim's clock: the number of completed physics steps. The authoritative
/// time is `tick * dt`, independent of any one aircraft's FDM.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimClock {
    pub tick: u64,
}

impl SimClock {
    pub fn time_s(&self, params: &SimParams) -> f64 {
        self.tick as f64 * params.dt
    }
}

/// A fatal error stops all further simulation (the app exits with an error).
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct SimStatus {
    pub fatal: Option<String>,
}

impl SimStatus {
    pub fn running(status: Res<SimStatus>) -> bool {
        status.fatal.is_none()
    }
}

/// Aircraft id → entity, built at spawn.
#[derive(Resource, Debug, Clone, Default)]
pub struct AircraftIndex(pub HashMap<u32, Entity>);
