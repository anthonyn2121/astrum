//! Who flies each aircraft: commands → [`PilotMode`], the command lease and its
//! failsafe hold, and the built-in autopilot laws. The Zenoh bridge
//! ([`crate::net`]) and the RL env ([`crate::env`]) apply commands through the
//! same [`apply_command`], so both front doors behave identically.

use astrum_fdm::{Controls, FdmState};
use astrum_proto::{AutopilotTargets, Command, Control, ControlMode, EventKind, ModeReason};
use bevy::prelude::*;

use crate::combat::{Dead, Trigger};
use crate::components::{
    Aircraft, AircraftState, AutopilotProfile, SimClock, SimParams, VehicleKind,
};
use crate::events::EventLog;

/// What drives an aircraft's FCS inputs this tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PilotMode {
    /// Never commanded: the controls JSBSim trimmed at spawn are held.
    Uncommanded,
    /// A controller's raw actuators.
    Raw(Controls),
    /// The built-in autopilot flying commanded targets.
    Autopilot(AutopilotTargets),
    /// The command lease expired or the controller went away (`ModeReason`
    /// says which); the autopilot holds the state at that moment.
    Failsafe(AutopilotTargets, ModeReason),
}

impl PilotMode {
    pub fn control_mode(&self) -> ControlMode {
        match self {
            PilotMode::Uncommanded => ControlMode::Uncommanded,
            PilotMode::Raw(_) => ControlMode::Raw,
            PilotMode::Autopilot(_) => ControlMode::Autopilot,
            PilotMode::Failsafe(..) => ControlMode::Failsafe,
        }
    }
}

/// Integrator memory of the autopilot (roll, altitude and airspeed loops).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Integrators {
    pub roll: f64,
    pub alt: f64,
    pub spd: f64,
}

/// Per-aircraft command state.
#[derive(Component, Debug, Clone, Copy)]
pub struct Pilot {
    pub mode: PilotMode,
    /// Tick at which the last command was applied.
    pub last_command_tick: u64,
    /// `seq` of the last applied command.
    pub last_seq: Option<u64>,
    /// In a lockstep scenario, the sim waits for this aircraft's commands.
    pub lockstep: bool,
    /// Ticks between this aircraft's telemetry samples (and, in lockstep, the
    /// commands the sim waits for): `rate_hz / control_rate_hz`, default 1.
    pub control_period_ticks: u64,
    pub integrators: Integrators,
    /// FCS inputs to write before this tick's step (`None` = leave as is).
    pub output: Option<Controls>,
    /// The control mode last reported as an event (see [`log_mode_changes`]).
    pub reported_mode: ControlMode,
}

impl Pilot {
    /// A pilot sampled every tick.
    pub fn new(lockstep: bool) -> Self {
        Self {
            mode: PilotMode::Uncommanded,
            last_command_tick: 0,
            last_seq: None,
            lockstep,
            control_period_ticks: 1,
            integrators: Integrators::default(),
            output: None,
            reported_mode: ControlMode::Uncommanded,
        }
    }

    /// Whether tick `tick` is one of this aircraft's control ticks (telemetry
    /// published; in lockstep, a command awaited). Tick 0 always is.
    pub fn due(&self, tick: u64) -> bool {
        tick.is_multiple_of(self.control_period_ticks)
    }

    /// Flown by a controller's commands (raw or autopilot), not by the sim alone.
    pub fn commanded(&self) -> bool {
        matches!(self.mode, PilotMode::Raw(_) | PilotMode::Autopilot(_))
    }
}

/// Why a command was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Answers telemetry the sim has not published yet (sender bug or sim restart).
    FromTheFuture { telemetry_tick: u64, now: u64 },
    /// Answers telemetry older than the command lease.
    Stale { telemetry_tick: u64, now: u64 },
    /// The same `seq` was already applied.
    Duplicate { seq: u64 },
    /// A NaN/inf autopilot target.
    NonFinite,
    /// No aircraft with that id.
    UnknownAircraft { id: u32 },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::FromTheFuture { telemetry_tick, now } => {
                write!(f, "answers tick {telemetry_tick}, but the sim is at tick {now}")
            }
            Refusal::Stale { telemetry_tick, now } => {
                write!(f, "stale: answers tick {telemetry_tick} at tick {now}")
            }
            Refusal::Duplicate { seq } => write!(f, "duplicate seq {seq}"),
            Refusal::NonFinite => write!(f, "non-finite autopilot target"),
            Refusal::UnknownAircraft { id } => write!(f, "no aircraft {id}"),
        }
    }
}

/// Apply one command to an aircraft at sim tick `now`. A raw command's
/// channels are clamped (NaN → neutral); autopilot targets must be finite.
pub fn apply_command(
    pilot: &mut Pilot,
    trigger: Option<&mut Trigger>,
    cmd: &Command,
    now: u64,
    lease_ticks: Option<u64>,
) -> Result<(), Refusal> {
    if cmd.telemetry_tick > now {
        return Err(Refusal::FromTheFuture { telemetry_tick: cmd.telemetry_tick, now });
    }
    if let Some(lease) = lease_ticks {
        if now - cmd.telemetry_tick > lease {
            return Err(Refusal::Stale { telemetry_tick: cmd.telemetry_tick, now });
        }
    }
    if pilot.last_seq == Some(cmd.seq) {
        return Err(Refusal::Duplicate { seq: cmd.seq });
    }
    let mode = match cmd.control {
        Control::Raw(r) => PilotMode::Raw(
            Controls {
                aileron: r.aileron,
                elevator: r.elevator,
                rudder: r.rudder,
                throttle: r.throttle,
            }
            .clamped(),
        ),
        Control::Autopilot(t) => {
            if !(t.heading_rad.is_finite() && t.alt_hae_m.is_finite() && t.tas_mps.is_finite()) {
                return Err(Refusal::NonFinite);
            }
            PilotMode::Autopilot(t)
        }
    };
    let entering_autopilot =
        !matches!(pilot.mode, PilotMode::Autopilot(_) | PilotMode::Failsafe(..));
    if matches!(mode, PilotMode::Autopilot(_)) && entering_autopilot {
        pilot.integrators = Integrators::default();
    }
    pilot.mode = mode;
    pilot.last_seq = Some(cmd.seq);
    pilot.last_command_tick = now;
    if let Some(t) = trigger {
        t.0 = cmd.trigger;
    }
    Ok(())
}

/// The failsafe target: hold the current heading, altitude and airspeed.
pub fn hold_targets(s: &FdmState) -> AutopilotTargets {
    AutopilotTargets { heading_rad: s.psi_rad, alt_hae_m: s.alt_hae_m, tas_mps: s.tas_mps }
}

/// Switch to the failsafe hold of the current state and release the trigger.
pub fn enter_failsafe(
    pilot: &mut Pilot,
    trigger: Option<&mut Trigger>,
    state: &AircraftState,
    reason: ModeReason,
) {
    pilot.mode = PilotMode::Failsafe(hold_targets(&state.fdm), reason);
    pilot.integrators = Integrators::default();
    if let Some(t) = trigger {
        t.0 = false;
    }
}

/// Expire command leases: a commanded aircraft with no fresh command for longer
/// than the lease switches to the failsafe hold and releases its trigger.
pub fn expire_leases(
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mut q: Query<(&Aircraft, &mut Pilot, Option<&mut Trigger>, &AircraftState), Without<Dead>>,
) {
    let Some(lease) = params.lease_ticks else { return };
    for (ac, mut pilot, trigger, state) in &mut q {
        if pilot.commanded() && clock.tick - pilot.last_command_tick > lease {
            eprintln!(
                "[astrum-sim] aircraft {}: no command for {:.2} s — failsafe hold",
                ac.id,
                (clock.tick - pilot.last_command_tick) as f64 * params.dt
            );
            let reason = ModeReason::LeaseExpired;
            enter_failsafe(&mut pilot, trigger.map(Mut::into_inner), state, reason);
        }
    }
}

/// Report every aircraft whose control mode changed since the last report: a
/// command, an expired lease, or a lost controller (earlier this update).
pub fn log_mode_changes(mut events: ResMut<EventLog>, mut q: Query<(&Aircraft, &mut Pilot)>) {
    for (ac, mut pilot) in &mut q {
        let to = pilot.mode.control_mode();
        if to == pilot.reported_mode {
            continue;
        }
        let reason = match pilot.mode {
            PilotMode::Failsafe(_, reason) => reason,
            _ => ModeReason::Command,
        };
        let from = std::mem::replace(&mut pilot.reported_mode, to);
        events.push(EventKind::ModeChange { aircraft_id: ac.id, from, to, reason });
    }
}

// --- Built-in autopilot ---------------------------------------------------------
//
// Signs measured on the models: aileron+ rolls right (φ+), elevator+ pitches the
// nose down (θ−); on the F450, aileron/elevator/rudder+ demand +p/+q/+r.

const DEG: f64 = std::f64::consts::PI / 180.0;

/// Gains of the fixed-wing inner loop (heading → bank → aileron, altitude →
/// pitch → elevator with climb-rate damping, airspeed → throttle; PI on bank,
/// altitude and airspeed).
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct FixedWingGains {
    /// Bank limit — sets the minimum turn radius `V²/(g·tan φ)`.
    pub max_roll_rad: f64,
    /// Pitch-attitude command limit.
    pub max_pitch_rad: f64,
    pub k_hdg_to_roll: f64,
    pub k_roll_to_ail: f64,
    pub k_roll_rate: f64,
    /// Integral gain on bank error (aileron per rad·s): trims out the steady
    /// aileron the airframe needs to stay wings-level (propeller torque etc.).
    pub ki_roll: f64,
    pub k_alt_to_pitch: f64,
    /// Integral gain on altitude error (rad of pitch per m·s).
    pub ki_alt: f64,
    /// Climb-rate damping (rad of pitch per m/s of climb). Altitude responds to
    /// pitch through `ḣ ≈ V·γ`, so without this term the altitude loop's gain
    /// grows with airspeed; on the f16 it limit-cycles at ±1 g (measured: 27 m /
    /// 10° α peak-to-peak at 100 m/s, 94 m / 20° at 180 m/s).
    pub k_climb_to_pitch: f64,
    /// Nose-up pitch per unit load-factor excess (`1/cos φ − 1`) in a bank.
    pub k_turn_comp: f64,
    pub k_pitch_to_elev: f64,
    pub k_pitch_rate: f64,
    pub throttle_trim: f64,
    pub k_spd_to_thr: f64,
    /// Integral gain on airspeed error (throttle per m/s·s).
    pub ki_spd: f64,
}

impl FixedWingGains {
    pub const fn for_profile(p: AutopilotProfile) -> Self {
        match p {
            AutopilotProfile::Trainer => Self {
                max_roll_rad: 25.0 * DEG,
                max_pitch_rad: 10.0 * DEG,
                k_hdg_to_roll: 1.2,
                k_roll_to_ail: 1.5,
                k_roll_rate: 0.3,
                ki_roll: 0.3,
                k_alt_to_pitch: 0.01,
                ki_alt: 0.0005,
                // The c172x's few-metre residual is its phugoid, which this term
                // does not reduce (measured), so the trainer runs without it.
                k_climb_to_pitch: 0.0,
                k_turn_comp: 0.0,
                k_pitch_to_elev: 2.0,
                k_pitch_rate: 0.5,
                throttle_trim: 0.6,
                k_spd_to_thr: 0.02,
                ki_spd: 0.005,
            },
            AutopilotProfile::Fighter => Self {
                max_roll_rad: 60.0 * DEG,
                max_pitch_rad: 24.0 * DEG,
                // controllers/optimal's `formation` law inverts this gain
                // (`--ap-heading-gain`, default 2.5) to command a bank.
                k_hdg_to_roll: 2.5,
                k_roll_to_ail: 2.5,
                k_roll_rate: 0.4,
                ki_roll: 0.3,
                k_alt_to_pitch: 0.02,
                ki_alt: 0.001,
                k_climb_to_pitch: 0.02,
                k_turn_comp: 0.14,
                k_pitch_to_elev: 2.0,
                k_pitch_rate: 0.5,
                throttle_trim: 0.6,
                k_spd_to_thr: 0.02,
                ki_spd: 0.005,
            },
        }
    }
}

/// Wrap an angle to `[-π, π)`; non-finite input stays non-finite.
pub fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Conditional (clamping) integration: integrate only while the loop output is
/// unsaturated or the error drives it back inside its limits, and keep the
/// integral term's own contribution within `±term_limit`.
fn integrate(
    i: &mut f64,
    err: f64,
    dt: f64,
    ki: f64,
    saturated_hi: bool,
    saturated_lo: bool,
    term_limit: f64,
) {
    if ki == 0.0 || (saturated_hi && err > 0.0) || (saturated_lo && err < 0.0) {
        return;
    }
    let limit = term_limit / ki.abs();
    *i = (*i + err * dt).clamp(-limit, limit);
}

/// Fixed-wing inner loop.
pub fn fixed_wing(
    s: &FdmState,
    t: &AutopilotTargets,
    g: &FixedWingGains,
    i: &mut Integrators,
    dt: f64,
) -> Controls {
    // Heading → bank → aileron.
    let hdg_err = wrap_pi(t.heading_rad - s.psi_rad);
    let roll_target = (g.k_hdg_to_roll * hdg_err).clamp(-g.max_roll_rad, g.max_roll_rad);
    let roll_err = roll_target - s.phi_rad;
    let ail_raw = g.k_roll_to_ail * roll_err + g.ki_roll * i.roll - g.k_roll_rate * s.p_rad_s;
    integrate(&mut i.roll, roll_err, dt, g.ki_roll, ail_raw >= 1.0, ail_raw <= -1.0, 0.3);

    // Altitude → pitch → elevator (PI with climb-rate damping) plus a load-factor
    // feedforward: a level turn at bank φ needs n = 1/cos φ.
    let alt_err = t.alt_hae_m - s.alt_hae_m;
    let climb = -s.vel_down_mps;
    let load_excess = (1.0 / s.phi_rad.cos().abs().max(1e-3) - 1.0).clamp(0.0, 2.0);
    let pitch_raw = g.k_alt_to_pitch * alt_err - g.k_climb_to_pitch * climb
        + g.ki_alt * i.alt
        + g.k_turn_comp * load_excess;
    let pitch_target = pitch_raw.clamp(-g.max_pitch_rad, g.max_pitch_rad);
    integrate(
        &mut i.alt,
        alt_err,
        dt,
        g.ki_alt,
        pitch_raw >= g.max_pitch_rad,
        pitch_raw <= -g.max_pitch_rad,
        0.5 * g.max_pitch_rad,
    );
    let elevator = -g.k_pitch_to_elev * (pitch_target - s.theta_rad) + g.k_pitch_rate * s.q_rad_s;

    // Airspeed → throttle (PI).
    let spd_err = t.tas_mps - s.tas_mps;
    let thr_raw = g.throttle_trim + g.k_spd_to_thr * spd_err + g.ki_spd * i.spd;
    integrate(&mut i.spd, spd_err, dt, g.ki_spd, thr_raw >= 1.0, thr_raw <= 0.0, 0.4);

    Controls { aileron: ail_raw, elevator, rudder: 0.0, throttle: thr_raw }.clamped()
}

// Multirotor (F450) hover-hold: rate demands from attitude errors, collective
// from a PID on altitude. The airspeed target is ignored.
const HOVER_COLLECTIVE: f64 = 0.44; // measured throttle that holds altitude
const K_LEVEL_ATT: f64 = 2.5;
const K_LEVEL_RATE: f64 = 0.3;
const K_HDG_TO_YAW: f64 = 1.5;
const K_YAW_RATE: f64 = 0.2;
const K_ALT_TO_COLL: f64 = 0.02;
const KI_ALT_TO_COLL: f64 = 0.002;
const K_CLIMB_RATE: f64 = 0.05;

/// Multirotor hover-hold inner loop.
pub fn multirotor(s: &FdmState, t: &AutopilotTargets, i: &mut Integrators, dt: f64) -> Controls {
    let aileron = -K_LEVEL_ATT * s.phi_rad - K_LEVEL_RATE * s.p_rad_s;
    let elevator = -K_LEVEL_ATT * s.theta_rad - K_LEVEL_RATE * s.q_rad_s;
    let rudder = K_HDG_TO_YAW * wrap_pi(t.heading_rad - s.psi_rad) - K_YAW_RATE * s.r_rad_s;
    let alt_err = t.alt_hae_m - s.alt_hae_m;
    let climb = -s.vel_down_mps;
    let coll_raw =
        HOVER_COLLECTIVE + K_ALT_TO_COLL * alt_err + KI_ALT_TO_COLL * i.alt - K_CLIMB_RATE * climb;
    integrate(&mut i.alt, alt_err, dt, KI_ALT_TO_COLL, coll_raw >= 1.0, coll_raw <= 0.0, 0.2);
    Controls { aileron, elevator, rudder, throttle: coll_raw }.clamped()
}

/// Compute every aircraft's FCS inputs for this tick from its pilot mode.
pub fn autopilot_step(
    params: Res<SimParams>,
    mut q: Query<
        (&mut Pilot, &VehicleKind, Option<&FixedWingGains>, &AircraftState),
        Without<Dead>,
    >,
) {
    for (mut pilot, kind, gains, state) in &mut q {
        let pilot = &mut *pilot;
        pilot.output = match pilot.mode {
            PilotMode::Uncommanded => None,
            PilotMode::Raw(c) => Some(c),
            PilotMode::Autopilot(t) | PilotMode::Failsafe(t, _) => Some(match kind {
                VehicleKind::FixedWing => {
                    let g = gains.expect("fixed-wing aircraft always carry FixedWingGains");
                    fixed_wing(&state.fdm, &t, g, &mut pilot.integrators, params.dt)
                }
                VehicleKind::Multirotor => {
                    multirotor(&state.fdm, &t, &mut pilot.integrators, params.dt)
                }
            }),
        };
    }
}
