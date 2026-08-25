//! Zenoh bridge: telemetry out and commands in, plus the optional in-sim
//! autopilot inner loop that services high-level [`Control::Autopilot`] targets.
//!
//! The transport wrappers live in [`astrum_net`]. They're held as Bevy
//! **`NonSend`** resources (the sim runs single-threaded under
//! `ScheduleRunnerPlugin`); all three bridge systems take theirs as
//! `Option<NonSend<..>>` and no-op when it's absent, so the physics-only
//! `build_app` (and its tests) run unchanged without opening a session —
//! networking is switched on purely by inserting the resources.

use astrum_fdm::{prop, Fdm, FT_TO_M};
use astrum_net::{CommandReceiver, TelemetryPublisher};
use astrum_proto::{Control, TelemetrySample};
use bevy::prelude::*;

use crate::components::{Aircraft, FdmInstance, GeoState, VehicleKind};

/// High-level autopilot targets for one aircraft, populated from
/// [`Control::Autopilot`] commands. `engaged` is cleared the moment a raw
/// actuator command arrives, so raw control always wins.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct AutopilotState {
    pub engaged: bool,
    pub heading_rad: f64,
    pub alt_msl_m: f64,
    pub airspeed_mps: f64,
}

/// Drain queued commands and apply them to the targeted aircraft.
///
/// * [`Control::Raw`] → written straight to the `fcs/*-cmd-norm` inputs and
///   disengages the autopilot (raw overrides).
/// * [`Control::Autopilot`] → stored in [`AutopilotState`]; [`autopilot_step`]
///   turns the targets into control deflections each tick.
pub fn ingest_commands(
    receiver: Option<NonSend<CommandReceiver>>,
    mut q: Query<(&Aircraft, &mut FdmInstance, &mut AutopilotState)>,
) {
    let Some(receiver) = receiver else { return };
    let commands = receiver.drain();
    if commands.is_empty() {
        return;
    }
    for cmd in commands {
        for (ac, mut inst, mut ap) in &mut q {
            if ac.id != cmd.target_id {
                continue;
            }
            match cmd.control {
                Control::Raw { aileron, elevator, rudder, throttle } => {
                    ap.engaged = false;
                    let f = &mut inst.0;
                    f.set(prop::AILERON_CMD, aileron.clamp(-1.0, 1.0));
                    f.set(prop::ELEVATOR_CMD, elevator.clamp(-1.0, 1.0));
                    f.set(prop::RUDDER_CMD, rudder.clamp(-1.0, 1.0));
                    f.set(prop::THROTTLE_CMD, throttle.clamp(0.0, 1.0));
                }
                Control::Autopilot { heading_rad, alt_msl_m, airspeed_mps } => {
                    ap.engaged = true;
                    ap.heading_rad = heading_rad;
                    ap.alt_msl_m = alt_msl_m;
                    ap.airspeed_mps = airspeed_mps;
                }
            }
        }
    }
}

// --- Basic built-in autopilot -----------------------------------------------
//
// A convenience inner loop for high-level [`Control::Autopilot`] targets; the
// primary control path is raw actuators, where the controller supplies its own
// law. The law is vehicle-class specific ([`fixed_wing_autopilot`] /
// [`multirotor_autopilot`]) because the four `fcs/*-cmd-norm` inputs mean
// different things on each; [`autopilot_step`] dispatches on [`VehicleKind`].

// Fixed-wing: conservative proportional bank-to-turn for heading, pitch-for-
// altitude, throttle-for-airspeed. Signs measured from the c172x
// (`aileron+ → roll right`, `elevator+ → nose down`).
const MAX_ROLL_RAD: f64 = 25.0 * std::f64::consts::PI / 180.0;
const MAX_PITCH_RAD: f64 = 10.0 * std::f64::consts::PI / 180.0;
const K_HDG_TO_ROLL: f64 = 1.2; // desired roll (rad) per heading error (rad)
const K_ROLL_TO_AIL: f64 = 1.5; // aileron per roll error (rad)
const K_ROLL_RATE: f64 = 0.3; // roll-rate damping
const K_ALT_TO_PITCH: f64 = 0.01; // desired pitch (rad) per altitude error (m)
const K_PITCH_TO_ELEV: f64 = 2.0; // elevator per pitch error (rad)
const K_PITCH_RATE: f64 = 0.5; // pitch-rate damping
const THROTTLE_TRIM: f64 = 0.6;
const K_SPD_TO_THR: f64 = 0.02; // throttle per airspeed error (m/s)

// Multirotor (F450) hover-hold. The FCS interface is a *rate* command:
// `aileron/elevator/rudder` demand body roll/pitch/yaw rates (measured
// +cmd → +rate, ≈0.75 rad/s per unit) and `throttle` is collective (measured
// hover ≈ 0.44). We hold a level hover at the commanded heading and altitude:
// an outer P loop drives roll/pitch attitude to zero and yaw to the target
// heading (both as rate demands, with body-rate damping), and a PD loop on
// altitude sets the collective. The `airspeed` target is meaningless for a
// hovering rotorcraft and is ignored.
const HOVER_COLLECTIVE: f64 = 0.44; // measured throttle that holds altitude
const K_LEVEL_ATT: f64 = 2.5; // roll/pitch cmd per attitude error (rad)
const K_LEVEL_RATE: f64 = 0.3; // roll/pitch-rate damping (cmd per rad/s)
const K_HDG_TO_YAW: f64 = 1.5; // rudder (yaw-rate) cmd per heading error (rad)
const K_YAW_RATE: f64 = 0.2; // yaw-rate damping (cmd per rad/s)
const K_ALT_TO_COLL: f64 = 0.02; // collective per altitude error (m)
const K_CLIMB_RATE: f64 = 0.05; // collective per climb rate (m/s)

/// Wrap an angle to `(-π, π]`.
fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Service engaged autopilots: convert targets to `fcs/*-cmd-norm` deflections.
/// Runs before `step_fdm` so this tick integrates the fresh commands. The law
/// used depends on the entity's [`VehicleKind`].
pub fn autopilot_step(mut q: Query<(&mut FdmInstance, &AutopilotState, &VehicleKind)>) {
    for (mut inst, ap, kind) in &mut q {
        if !ap.engaged {
            continue;
        }
        match kind {
            VehicleKind::FixedWing => fixed_wing_autopilot(&mut inst.0, ap),
            VehicleKind::Multirotor => multirotor_autopilot(&mut inst.0, ap),
        }
    }
}

/// Fixed-wing inner loop (see the `K_*` constants above).
fn fixed_wing_autopilot(f: &mut Fdm, ap: &AutopilotState) {
    let phi = f.get(prop::PHI_RAD);
    let theta = f.get(prop::THETA_RAD);
    let psi = f.get(prop::PSI_RAD);
    let p = f.get(prop::P_RAD_S);
    let qrate = f.get(prop::Q_RAD_S);
    let alt = f.get(prop::ALT_SL_M);
    let airspeed = f.get(prop::VT_FPS) * FT_TO_M;

    // Heading → bank → aileron. aileron+ rolls right (φ+).
    let hdg_err = wrap_pi(ap.heading_rad - psi);
    let roll_target = (K_HDG_TO_ROLL * hdg_err).clamp(-MAX_ROLL_RAD, MAX_ROLL_RAD);
    let aileron = (K_ROLL_TO_AIL * (roll_target - phi) - K_ROLL_RATE * p).clamp(-1.0, 1.0);

    // Altitude → pitch → elevator. elevator+ pitches down (θ−), so a nose-up
    // (positive) pitch error commands negative elevator; +q damping opposes
    // nose-up rate.
    let pitch_target = (K_ALT_TO_PITCH * (ap.alt_msl_m - alt)).clamp(-MAX_PITCH_RAD, MAX_PITCH_RAD);
    let elevator =
        (-K_PITCH_TO_ELEV * (pitch_target - theta) + K_PITCH_RATE * qrate).clamp(-1.0, 1.0);

    // Airspeed → throttle.
    let throttle = (THROTTLE_TRIM + K_SPD_TO_THR * (ap.airspeed_mps - airspeed)).clamp(0.0, 1.0);

    f.set(prop::AILERON_CMD, aileron);
    f.set(prop::ELEVATOR_CMD, elevator);
    f.set(prop::THROTTLE_CMD, throttle);
}

/// Multirotor hover-hold inner loop (see the `HOVER_*` / `K_*` constants above).
fn multirotor_autopilot(f: &mut Fdm, ap: &AutopilotState) {
    let phi = f.get(prop::PHI_RAD);
    let theta = f.get(prop::THETA_RAD);
    let psi = f.get(prop::PSI_RAD);
    let p = f.get(prop::P_RAD_S);
    let qrate = f.get(prop::Q_RAD_S);
    let r = f.get(prop::R_RAD_S);
    let alt = f.get(prop::ALT_SL_M);
    let climb = -f.get(prop::V_DOWN_FPS) * FT_TO_M; // +up, m/s

    // Level roll/pitch: aileron+ → +roll rate, elevator+ → +pitch rate, so a
    // positive attitude commands a negative rate to return to level.
    let aileron = (-K_LEVEL_ATT * phi - K_LEVEL_RATE * p).clamp(-1.0, 1.0);
    let elevator = (-K_LEVEL_ATT * theta - K_LEVEL_RATE * qrate).clamp(-1.0, 1.0);

    // Heading hold: rudder+ → +yaw rate → +ψ, so a positive heading error
    // commands positive rudder; +r damping opposes the yaw rate.
    let hdg_err = wrap_pi(ap.heading_rad - psi);
    let rudder = (K_HDG_TO_YAW * hdg_err - K_YAW_RATE * r).clamp(-1.0, 1.0);

    // Altitude hold via collective (PD: hover trim + P on altitude − D on climb).
    let throttle = (HOVER_COLLECTIVE + K_ALT_TO_COLL * (ap.alt_msl_m - alt) - K_CLIMB_RATE * climb)
        .clamp(0.0, 1.0);

    f.set(prop::AILERON_CMD, aileron);
    f.set(prop::ELEVATOR_CMD, elevator);
    f.set(prop::RUDDER_CMD, rudder);
    f.set(prop::THROTTLE_CMD, throttle);
}

/// Encode each aircraft's post-step state and publish it on the telemetry key
/// `astrum/telemetry/<id>`. Runs after `sync_geo_transform`.
pub fn publish_telemetry(
    publisher: Option<NonSend<TelemetryPublisher>>,
    q: Query<(&Aircraft, &GeoState, &Transform, &VehicleKind)>,
) {
    let Some(publisher) = publisher else { return };
    for (ac, gs, tf, kind) in &q {
        let sample = TelemetrySample {
            aircraft_id: ac.id,
            sim_time_s: gs.sim_time_s,
            lat_deg: gs.lat_deg,
            lon_deg: gs.lon_deg,
            alt_m: gs.alt_m,
            enu_e: tf.translation.x as f64,
            enu_n: tf.translation.y as f64,
            enu_u: tf.translation.z as f64,
            vel_north_mps: gs.vel_north_mps,
            vel_east_mps: gs.vel_east_mps,
            vel_down_mps: gs.vel_down_mps,
            phi_rad: gs.phi_rad,
            theta_rad: gs.theta_rad,
            psi_rad: gs.psi_rad,
            airspeed_mps: gs.airspeed_mps,
            health: 0,
            vehicle_kind: match kind {
                VehicleKind::FixedWing => astrum_proto::VehicleKind::FixedWing,
                VehicleKind::Multirotor => astrum_proto::VehicleKind::Multirotor,
            },
        };
        // A dropped telemetry frame is non-fatal — log once and keep flying.
        if let Err(e) = publisher.publish(&sample) {
            eprintln!("[astrum-sim] telemetry publish failed for aircraft {}: {e}", ac.id);
        }
    }
}
