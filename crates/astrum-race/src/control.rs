//! The drone's inner loop, and the command interface a course-flying controller
//! writes to.
//!
//! # Why this exists
//!
//! An F450 with its motors running and no commands does not hover — it falls, hits
//! the ground and JSBSim declares a crash. A racing environment therefore needs a
//! stabilising inner loop before it needs anything else: something that holds the
//! drone where it was put, so an outer controller (a policy, an MPC, a hand-coded
//! line-follower) can be written in terms of *where to go* rather than motor
//! commands. This module is that loop and nothing more — it will hold a position
//! and fly a commanded velocity, but it does not know a course exists.
//!
//! # The interface
//!
//! An outer controller writes [`FlightCommand`]: a target altitude, a target
//! heading, and a horizontal velocity demand. Zero velocity is station keeping,
//! which is what the drone launches with.
//!
//! A controller that would rather supply its *own* law writes [`RawOverride`]
//! instead — the four normalized channels, straight through, with the loop below
//! standing down. That is the path an off-process controller takes over Zenoh (see
//! [`crate::net`]); unlike the fleet sim's, it is a lease that must be refreshed,
//! for the reason spelled out on [`RawOverride`].
//!
//! # The F450's controls
//!
//! For a multirotor JSBSim's four `fcs/*-cmd-norm` inputs are *not* aerodynamic
//! surfaces: with `fcs/ScasEngage` on, `aileron`/`elevator`/`rudder` demand body
//! roll/pitch/yaw **rates** (about 0.75 rad/s per unit, `+cmd` → `+rate`, so
//! `elevator+` pitches the nose *up*) and `throttle` is **collective** thrust
//! (about 0.44 holds a hover). Everything below is written against those measured
//! conventions.

use astrum_fdm::{prop, Fdm, FT_TO_M};
use astrum_geo::Geodetic;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::components::{FdmInstance, SimParams, WorldFrame};

/// Nominal collective for a hover — a starting guess, not a constant of nature: the
/// real trim moves with take-off weight and air density. `K_ALT_I` finds the truth.
const HOVER_COLLECTIVE: f64 = 0.44;
/// Collective per metre of altitude error, per metre-second of accumulated error,
/// and per m/s of climb rate (the D term).
const K_ALT: f64 = 0.04;
const K_ALT_I: f64 = 0.01;
const K_CLIMB: f64 = 0.09;
/// Integrator clamp (m.s). Bounds the trim the integral term can discover to
/// +/-`K_ALT_I * ALT_I_LIMIT` of collective, so a saturated climb demand or a long
/// hold against a wrong `HOVER_COLLECTIVE` can't wind up without limit.
const ALT_I_LIMIT: f64 = 20.0;
/// Roll/pitch rate demand per radian of attitude error, and its rate damping.
const K_ATT: f64 = 2.5;
const K_ATT_RATE: f64 = 0.3;
/// Yaw rate demand per radian of heading error, and its rate damping.
const K_HDG: f64 = 1.5;
const K_YAW_RATE: f64 = 0.2;
/// Tilt commanded per m/s of horizontal velocity error — a quad accelerates by
/// leaning, so the velocity loop's output is an attitude target.
const K_VEL: f64 = 0.15;
/// Most the drone will lean (rad). Caps how hard the velocity loop can push, and
/// keeps the collective loop's small-angle assumption honest.
const MAX_TILT: f64 = 0.35;
/// How long a raw actuator command stays in force without being refreshed (s).
///
/// See [`RawOverride`] for why a multirotor needs this and a fixed-wing does not.
/// Half a second is ~60 ticks at 120 Hz: long enough that an external controller
/// running at any sane rate never trips it, short enough that a free fall from the
/// racing altitude band is recoverable.
const RAW_TIMEOUT_S: f64 = 0.5;

/// What an outer controller asks the drone to do.
///
/// Altitude is metres on the world ENU grid (not above sea level), heading follows
/// the sim convention (`0` = North, `+` = East), and `vel_enu`'s horizontal part is
/// a ground-velocity demand in m/s — its vertical component is ignored, since climb
/// is commanded through `alt_m`.
#[derive(Component, Debug, Clone, Copy)]
pub struct FlightCommand {
    pub alt_m: f64,
    pub heading_rad: f64,
    pub vel_enu: DVec3,
}

impl FlightCommand {
    /// Hold this altitude and heading, going nowhere.
    pub fn hold(alt_m: f64, heading_rad: f64) -> Self {
        Self { alt_m, heading_rad, vel_enu: DVec3::ZERO }
    }
}

/// The inner loop's memory: the collective loop's integral term.
///
/// Without it the drone settles wherever proportional gain balances the gap between
/// [`HOVER_COLLECTIVE`] and the collective the airframe actually needs — a
/// steady-state altitude error you can shrink with gain but never remove. With it the
/// loop *learns* the hover trim, so the same gains hold altitude at any weight or
/// density altitude.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct InnerLoop {
    /// Accumulated altitude error (m.s), clamped to +/-[`ALT_I_LIMIT`].
    pub alt_integral: f64,
}

/// An external controller flying the drone's actuators directly, bypassing the
/// inner loop.
///
/// Populated from `Control::Raw` commands off the wire (see [`crate::net`]). Raw
/// wins over [`FlightCommand`] while it is `active`, which is the same rule the
/// fleet sim applies — a controller that supplies its own law should not be fighting
/// ours for the same four channels.
///
/// # Why this expires and the fleet sim's does not
///
/// The fleet sim latches raw control indefinitely: a fixed-wing whose controller
/// dies keeps flying. A quadrotor whose controller dies *falls* — collective is the
/// only thing holding it up, and the last commanded value is frozen. So a raw
/// command here is a lease: unrefreshed for [`RAW_TIMEOUT_S`], it lapses and the
/// inner loop resumes, holding whatever altitude and heading the drone had at that
/// moment. A controller running at any reasonable rate never notices.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct RawOverride {
    /// Is an unexpired raw command in force?
    pub active: bool,
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
    pub throttle: f64,
    /// Simulated time the newest raw command was accepted (s).
    pub last_cmd_s: f64,
}

impl RawOverride {
    /// Accept a raw command at simulated time `now`, clamping to the valid ranges.
    pub fn set(&mut self, aileron: f64, elevator: f64, rudder: f64, throttle: f64, now: f64) {
        self.active = true;
        self.aileron = aileron.clamp(-1.0, 1.0);
        self.elevator = elevator.clamp(-1.0, 1.0);
        self.rudder = rudder.clamp(-1.0, 1.0);
        self.throttle = throttle.clamp(0.0, 1.0);
        self.last_cmd_s = now;
    }

    /// Has the lease lapsed as of simulated time `now`?
    pub fn expired(&self, now: f64) -> bool {
        now - self.last_cmd_s > RAW_TIMEOUT_S
    }

    fn write_to(&self, f: &mut Fdm) {
        f.set(prop::AILERON_CMD, self.aileron);
        f.set(prop::ELEVATOR_CMD, self.elevator);
        f.set(prop::RUDDER_CMD, self.rudder);
        f.set(prop::THROTTLE_CMD, self.throttle);
    }
}

/// Wrap an angle to `(-pi, pi]`.
fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// The drone's current ENU altitude and heading, as a station-keeping command.
/// Used when a raw lease lapses: hold where you are, not where you were told to be
/// before an external controller took over.
fn hold_here(f: &Fdm, frame: &astrum_geo::LocalFrame) -> FlightCommand {
    let geo = Geodetic::from_degrees(
        f.get(prop::LAT_GEOD_DEG),
        f.get(prop::LON_GC_DEG),
        f.get(prop::ALT_SL_M),
    );
    FlightCommand::hold(frame.geodetic_to_enu(&geo).u, f.get(prop::PSI_RAD))
}

/// Drive each drone's actuators: an in-force [`RawOverride`] verbatim, otherwise
/// the inner loop servicing its [`FlightCommand`].
///
/// Runs *before* `step_fdm` so the commands set here are the ones this tick
/// integrates, and *after* command ingest so a raw command that arrived this tick
/// takes effect this tick. The inner loop is cascaded: a velocity loop produces
/// attitude targets, an attitude loop turns those into rate demands, and a PI+D loop
/// on altitude sets the collective.
pub fn stabilize(
    frame: Res<WorldFrame>,
    sim: Res<SimParams>,
    mut q: Query<(&mut FdmInstance, &mut FlightCommand, &mut InnerLoop, Option<&mut RawOverride>)>,
) {
    for (mut inst, mut cmd, mut loop_state, raw) in &mut q {
        // A lapsed lease reverts to station keeping *here*, and resets the
        // integrator: whatever trim it had learnt was for a different regime, and
        // the accumulated error over the raw interval is meaningless.
        if let Some(mut raw) = raw {
            if raw.active {
                let now = inst.0.get(prop::SIM_TIME);
                if raw.expired(now) {
                    raw.active = false;
                    *cmd = hold_here(&inst.0, &frame.0);
                    loop_state.alt_integral = 0.0;
                } else {
                    raw.write_to(&mut inst.0);
                    continue;
                }
            }
        }
        apply_command(&mut inst.0, &frame.0, &cmd, &mut loop_state, sim.dt);
    }
}

/// The inner loop itself, as a plain function over one FDM — App-free so it can be
/// reasoned about (and tested) without a world.
pub fn apply_command(
    f: &mut Fdm,
    frame: &astrum_geo::LocalFrame,
    cmd: &FlightCommand,
    state: &mut InnerLoop,
    dt: f64,
) {
    let (phi, theta, psi) = (f.get(prop::PHI_RAD), f.get(prop::THETA_RAD), f.get(prop::PSI_RAD));
    let (p, q, r) = (f.get(prop::P_RAD_S), f.get(prop::Q_RAD_S), f.get(prop::R_RAD_S));
    let vel = DVec3::new(
        f.get(prop::V_EAST_FPS) * FT_TO_M,
        f.get(prop::V_NORTH_FPS) * FT_TO_M,
        -f.get(prop::V_DOWN_FPS) * FT_TO_M,
    );
    // Altitude on the ENU grid, which is what the command is expressed in.
    let geo = Geodetic::from_degrees(
        f.get(prop::LAT_GEOD_DEG),
        f.get(prop::LON_GC_DEG),
        f.get(prop::ALT_SL_M),
    );
    let alt = frame.geodetic_to_enu(&geo).u;

    // Velocity loop → attitude targets. The nose is +body-x, so the drone leans
    // *forward* (nose down, theta < 0) to gain forward speed and rolls right
    // (phi > 0) to slide right.
    let (s, c) = psi.sin_cos();
    let fwd = |v: DVec3| v.x * s + v.y * c;
    let right = |v: DVec3| v.x * c - v.y * s;
    let e_fwd = fwd(cmd.vel_enu) - fwd(vel);
    let e_right = right(cmd.vel_enu) - right(vel);
    let theta_target = (-K_VEL * e_fwd).clamp(-MAX_TILT, MAX_TILT);
    let phi_target = (K_VEL * e_right).clamp(-MAX_TILT, MAX_TILT);

    // Attitude loop → rate demands, damped on the measured body rates.
    let aileron = (K_ATT * (phi_target - phi) - K_ATT_RATE * p).clamp(-1.0, 1.0);
    let elevator = (K_ATT * (theta_target - theta) - K_ATT_RATE * q).clamp(-1.0, 1.0);
    // Heading hold: rudder+ yaws towards +psi, so a positive error takes positive
    // rudder; the r term damps it.
    let rudder = (K_HDG * wrap_pi(cmd.heading_rad - psi) - K_YAW_RATE * r).clamp(-1.0, 1.0);
    // Collective: nominal trim, PI on altitude error, D on climb rate. The integral
    // is what removes the standing sag; clamping it keeps a long saturated demand
    // from winding up.
    let e_alt = cmd.alt_m - alt;
    state.alt_integral = (state.alt_integral + e_alt * dt).clamp(-ALT_I_LIMIT, ALT_I_LIMIT);
    let throttle = (HOVER_COLLECTIVE + K_ALT * e_alt + K_ALT_I * state.alt_integral
        - K_CLIMB * vel.z)
        .clamp(0.0, 1.0);

    f.set(prop::AILERON_CMD, aileron);
    f.set(prop::ELEVATOR_CMD, elevator);
    f.set(prop::RUDDER_CMD, rudder);
    f.set(prop::THROTTLE_CMD, throttle);
}
