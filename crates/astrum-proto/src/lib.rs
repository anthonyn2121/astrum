//! `astrum-proto` — the ASTRUM Cap'n Proto wire schema (`schema/astrum.capnp`),
//! its generated Rust bindings ([`wire`]), and plain-data mirrors with
//! encode/decode helpers so the rest of the workspace never touches capnp
//! builders directly.
//!
//! Four payloads: [`Telemetry`] (sim → subscribers), [`Command`] (controllers →
//! sim), [`Event`] (sim → subscribers: hits, kills, mode changes, ...) and
//! [`TickEnd`] (sim → subscribers: the end of a tick's publications).
//! Decoding reads straight from the payload slice (no segment copy) and rejects
//! trailing bytes. Enum values from a newer schema decode to an `Unknown(n)`
//! variant instead of being silently mapped to a known one.

/// Generated code from `schema/astrum.capnp`. It refers to itself as
/// `crate::astrum_capnp`, hence the name; use it through the [`wire`] alias.
#[allow(clippy::all, dead_code, unused)]
#[doc(hidden)]
pub mod astrum_capnp {
    include!(concat!(env!("OUT_DIR"), "/astrum_capnp.rs"));
}

/// The generated Cap'n Proto bindings.
pub use astrum_capnp as wire;

pub use astrum_geo::{Enu, Ned};
pub use capnp::Error as DecodeError;

use capnp::message::{Builder, HeapAllocator, ReaderOptions};
use capnp::serialize;

/// Flight-vehicle class. Mirrors the schema's `VehicleKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VehicleKind {
    /// Aerodynamic fixed-wing (control surfaces + throttle).
    #[default]
    FixedWing,
    /// Multirotor (body-rate demands + collective).
    Multirotor,
    /// A value this build's schema does not know (sent by a newer peer).
    Unknown(u16),
}

/// Who flew the aircraft on this tick. Mirrors the schema's `ControlMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlMode {
    /// No command yet: the initial (trimmed) controls are held.
    #[default]
    Uncommanded,
    /// A controller's raw actuators, within the command lease.
    Raw,
    /// The sim's inner loop flying commanded targets.
    Autopilot,
    /// The command lease expired or the aircraft's controller went away; the
    /// sim is holding heading/altitude/speed.
    Failsafe,
    /// A value this build's schema does not know (sent by a newer peer).
    Unknown(u16),
}

/// Aerospace 3-2-1 Euler attitude relative to local-level NED (radians).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Euler {
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
}

/// Body-axis angular rates relative to the Earth (rad/s).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BodyRates {
    pub p_rad_s: f64,
    pub q_rad_s: f64,
    pub r_rad_s: f64,
}

/// Aerodynamic state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Aero {
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub nz_g: f64,
    pub mach: f64,
}

/// Combat status.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CombatStatus {
    pub team: u8,
    pub hit_points: f32,
    pub hit_points_max: f32,
    pub alive: bool,
    pub ammo: u16,
}

/// One aircraft's authoritative state after sim step `tick`. Mirrors `Telemetry`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Telemetry {
    pub aircraft_id: u32,
    pub tick: u64,
    pub sim_time_s: f64,
    /// Geodetic latitude / longitude (deg) and height above the ellipsoid (m).
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_hae_m: f64,
    /// World-grid position (m).
    pub pos_enu: Enu,
    /// Velocity relative to the Earth in the world grid (m/s).
    pub vel_enu: Enu,
    /// Velocity relative to the Earth in the local-level frame (m/s).
    pub vel_ned: Ned,
    pub attitude: Euler,
    pub rates: BodyRates,
    pub aero: Aero,
    /// True airspeed (m/s).
    pub tas_mps: f64,
    pub fault_flags: u8,
    pub vehicle_kind: VehicleKind,
    pub control_mode: ControlMode,
    pub combat: CombatStatus,
    /// Sim time between this aircraft's telemetry samples (s).
    pub control_period_s: f64,
    /// Ownership and command refusals.
    pub command: CommandStatus,
}

/// Why the sim refused a command. Mirrors the schema's `Refusal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Refusal {
    /// No command refused yet.
    #[default]
    None,
    /// `telemetry_tick` is ahead of the sim.
    FromTheFuture,
    /// `telemetry_tick` is older than the command lease.
    Stale,
    /// `seq` not above the owner's last applied seq.
    OutOfOrder,
    /// Another controller instance (epoch) owns the aircraft.
    NotOwner,
    /// Epoch 0.
    MissingEpoch,
    /// A NaN / infinite autopilot target.
    NonFinite,
    /// Raw on a multirotor, or multirotor on a fixed-wing.
    WrongVehicleKind,
    /// A value this build's schema does not know (sent by a newer peer).
    Unknown(u16),
}

/// The sim's command bookkeeping for one aircraft. Mirrors `CommandStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommandStatus {
    /// Epoch of the controller flying it; 0 = unowned.
    pub owner_epoch: u64,
    /// Seq of the owner's last applied command; 0 = none.
    pub last_seq: u64,
    /// Commands refused for this aircraft so far (wraps).
    pub refusals: u32,
    pub last_refusal: Refusal,
    /// The epoch that sent the most recently refused command.
    pub last_refused_epoch: u64,
}

/// The end of a tick's publications on `astrum/tick`. Mirrors `TickEnd`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TickEnd {
    pub tick: u64,
    pub sim_time_s: f64,
}

/// Raw actuator setpoints: surfaces in [-1, 1], throttle in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RawActuators {
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
    pub throttle: f64,
}

/// Targets for the sim's built-in inner loop.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AutopilotTargets {
    pub heading_rad: f64,
    pub alt_hae_m: f64,
    pub tas_mps: f64,
}

/// Raw multirotor demands: body-rate demands in [-1, 1], collective in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MultirotorRaw {
    pub roll_rate: f64,
    pub pitch_rate: f64,
    pub yaw_rate: f64,
    pub collective: f64,
}

/// The control payload of a [`Command`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    /// Fixed-wing aircraft only.
    Raw(RawActuators),
    Autopilot(AutopilotTargets),
    /// Multirotor aircraft only.
    Multirotor(MultirotorRaw),
}

/// A command for one aircraft. Mirrors `Command`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    pub target_id: u32,
    /// Strictly increasing within one epoch.
    pub seq: u64,
    /// The `Telemetry::tick` this command was computed from.
    pub telemetry_tick: u64,
    pub trigger: bool,
    /// Random non-zero id of the sending controller instance; 0 is refused.
    pub epoch: u64,
    pub control: Control,
}

/// Why an aircraft's control mode changed. Mirrors the schema's `ModeReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeReason {
    /// A controller's command (raw or autopilot).
    Command,
    /// No fresh command within the command lease.
    LeaseExpired,
    /// The controller's presence token went away.
    ControllerLost,
    /// A value this build's schema does not know (sent by a newer peer).
    Unknown(u16),
}

/// Why a run ended. Mirrors the schema's `EndReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// `sim.max_time_s` reached.
    MaxTime,
    /// At most one team still has a living aircraft.
    EngagementDecided,
    /// A fatal error (the event's message says which).
    Fatal,
    /// SIGINT / SIGTERM.
    Stopped,
    /// A value this build's schema does not know (sent by a newer peer).
    Unknown(u16),
}

/// What happened. Mirrors the schema's `Event.kind`.
#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    TriggerPressed {
        aircraft_id: u32,
        ammo: u16,
    },
    TriggerReleased {
        aircraft_id: u32,
        rounds_fired: u16,
        ammo: u16,
    },
    /// A round struck `target_id`, leaving it `hit_points`; `pos_enu` is where.
    Hit {
        shooter_id: u32,
        target_id: u32,
        damage: f32,
        hit_points: f32,
        pos_enu: Enu,
    },
    /// `victim_id` was destroyed; `killer_id` fired the last round that hit it.
    Kill {
        victim_id: u32,
        killer_id: Option<u32>,
        pos_enu: Enu,
    },
    ModeChange {
        aircraft_id: u32,
        from: ControlMode,
        to: ControlMode,
        reason: ModeReason,
    },
    ControllerUp {
        aircraft_id: u32,
    },
    ControllerDown {
        aircraft_id: u32,
    },
    /// `message` is the fatal error, empty otherwise.
    RunEnded {
        reason: EndReason,
        message: String,
    },
}

/// Something that happened during a run. Mirrors `Event`.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// The `Telemetry.tick` that first shows it (while paused: the current tick).
    pub tick: u64,
    pub sim_time_s: f64,
    pub kind: EventKind,
}

// Display names, shared by every JSON / Python projection of these values
// (the web export, the RL env's `info["events"]`): kebab-case, `unknown` for a
// value from a newer schema.

impl ControlMode {
    pub fn name(self) -> &'static str {
        match self {
            ControlMode::Uncommanded => "uncommanded",
            ControlMode::Raw => "raw",
            ControlMode::Autopilot => "autopilot",
            ControlMode::Failsafe => "failsafe",
            ControlMode::Unknown(_) => "unknown",
        }
    }
}

impl Refusal {
    pub fn name(self) -> &'static str {
        match self {
            Refusal::None => "none",
            Refusal::FromTheFuture => "from-the-future",
            Refusal::Stale => "stale",
            Refusal::OutOfOrder => "out-of-order",
            Refusal::NotOwner => "not-owner",
            Refusal::MissingEpoch => "missing-epoch",
            Refusal::NonFinite => "non-finite",
            Refusal::WrongVehicleKind => "wrong-vehicle-kind",
            Refusal::Unknown(_) => "unknown",
        }
    }
}

impl ModeReason {
    pub fn name(self) -> &'static str {
        match self {
            ModeReason::Command => "command",
            ModeReason::LeaseExpired => "lease-expired",
            ModeReason::ControllerLost => "controller-lost",
            ModeReason::Unknown(_) => "unknown",
        }
    }
}

impl EndReason {
    pub fn name(self) -> &'static str {
        match self {
            EndReason::MaxTime => "max-time",
            EndReason::EngagementDecided => "engagement-decided",
            EndReason::Fatal => "fatal",
            EndReason::Stopped => "stopped",
            EndReason::Unknown(_) => "unknown",
        }
    }
}

impl EventKind {
    pub fn name(&self) -> &'static str {
        match self {
            EventKind::TriggerPressed { .. } => "trigger-pressed",
            EventKind::TriggerReleased { .. } => "trigger-released",
            EventKind::Hit { .. } => "hit",
            EventKind::Kill { .. } => "kill",
            EventKind::ModeChange { .. } => "mode-change",
            EventKind::ControllerUp { .. } => "controller-up",
            EventKind::ControllerDown { .. } => "controller-down",
            EventKind::RunEnded { .. } => "run-ended",
        }
    }
}

/// Room for one whole message in the first segment (Telemetry is ~50 words), so
/// encoding allocates once instead of zero-filling capnp's default 8 KiB segment.
const FIRST_SEGMENT_WORDS: u32 = 64;

fn builder() -> Builder<HeapAllocator> {
    Builder::new(HeapAllocator::new().first_segment_words(FIRST_SEGMENT_WORDS))
}

fn reader(bytes: &[u8]) -> capnp::Result<capnp::message::Reader<serialize::BufferSegments<&[u8]>>> {
    let mut rest = bytes;
    let msg = serialize::read_message_from_flat_slice(&mut rest, ReaderOptions::new())?;
    if !rest.is_empty() {
        return Err(capnp::Error::failed(format!(
            "{} trailing bytes after the message",
            rest.len()
        )));
    }
    Ok(msg)
}

fn set_enu(mut b: wire::enu::Builder, v: Enu) {
    b.set_e(v.e);
    b.set_n(v.n);
    b.set_u(v.u);
}

fn get_enu(r: wire::enu::Reader) -> Enu {
    Enu { e: r.get_e(), n: r.get_n(), u: r.get_u() }
}

/// Serialise telemetry to one flat Cap'n Proto message.
///
/// # Panics
/// If `vehicle_kind`, `control_mode` or `command.last_refusal` is `Unknown(_)`
/// (only produced by decoding).
pub fn encode_telemetry(t: &Telemetry) -> Vec<u8> {
    let mut msg = builder();
    {
        let mut b = msg.init_root::<wire::telemetry::Builder>();
        b.set_aircraft_id(t.aircraft_id);
        b.set_tick(t.tick);
        b.set_sim_time_s(t.sim_time_s);
        b.set_tas_mps(t.tas_mps);
        b.set_fault_flags(t.fault_flags);
        // `Unknown(n)` only arises from decoding a newer peer, and nothing in the
        // workspace re-encodes decoded telemetry; the typed capnp API cannot emit a
        // value outside this schema, so doing so is a programming error.
        b.set_vehicle_kind(match t.vehicle_kind {
            VehicleKind::FixedWing => wire::VehicleKind::FixedWing,
            VehicleKind::Multirotor => wire::VehicleKind::Multirotor,
            VehicleKind::Unknown(n) => {
                panic!("cannot encode VehicleKind value {n} unknown to this schema")
            }
        });
        b.set_control_mode(wire_control_mode(t.control_mode));
        {
            let mut g = b.reborrow().init_geo();
            g.set_lat_deg(t.lat_deg);
            g.set_lon_deg(t.lon_deg);
            g.set_alt_hae_m(t.alt_hae_m);
        }
        set_enu(b.reborrow().init_enu(), t.pos_enu);
        set_enu(b.reborrow().init_vel_enu(), t.vel_enu);
        {
            let mut v = b.reborrow().init_vel_ned();
            v.set_n(t.vel_ned.n);
            v.set_e(t.vel_ned.e);
            v.set_d(t.vel_ned.d);
        }
        {
            let mut a = b.reborrow().init_attitude();
            a.set_phi_rad(t.attitude.phi_rad);
            a.set_theta_rad(t.attitude.theta_rad);
            a.set_psi_rad(t.attitude.psi_rad);
        }
        {
            let mut r = b.reborrow().init_body_rates();
            r.set_p_rad_s(t.rates.p_rad_s);
            r.set_q_rad_s(t.rates.q_rad_s);
            r.set_r_rad_s(t.rates.r_rad_s);
        }
        {
            let mut a = b.reborrow().init_aero();
            a.set_alpha_rad(t.aero.alpha_rad);
            a.set_beta_rad(t.aero.beta_rad);
            a.set_nz_g(t.aero.nz_g);
            a.set_mach(t.aero.mach);
        }
        {
            let mut c = b.reborrow().init_combat();
            c.set_team(t.combat.team);
            c.set_hit_points(t.combat.hit_points);
            c.set_hit_points_max(t.combat.hit_points_max);
            c.set_alive(t.combat.alive);
            c.set_ammo(t.combat.ammo);
        }
        b.set_control_period_s(t.control_period_s);
        {
            let mut s = b.reborrow().init_command();
            s.set_owner_epoch(t.command.owner_epoch);
            s.set_last_seq(t.command.last_seq);
            s.set_refusals(t.command.refusals);
            s.set_last_refusal(wire_refusal(t.command.last_refusal));
            s.set_last_refused_epoch(t.command.last_refused_epoch);
        }
    }
    serialize::write_message_to_words(&msg)
}

/// Deserialise telemetry.
pub fn decode_telemetry(bytes: &[u8]) -> capnp::Result<Telemetry> {
    let msg = reader(bytes)?;
    let r = msg.get_root::<wire::telemetry::Reader>()?;
    let geo = r.get_geo()?;
    let ned = r.get_vel_ned()?;
    let att = r.get_attitude()?;
    let rates = r.get_body_rates()?;
    let aero = r.get_aero()?;
    let combat = r.get_combat()?;
    let command = r.get_command()?;
    Ok(Telemetry {
        aircraft_id: r.get_aircraft_id(),
        tick: r.get_tick(),
        sim_time_s: r.get_sim_time_s(),
        lat_deg: geo.get_lat_deg(),
        lon_deg: geo.get_lon_deg(),
        alt_hae_m: geo.get_alt_hae_m(),
        pos_enu: get_enu(r.get_enu()?),
        vel_enu: get_enu(r.get_vel_enu()?),
        vel_ned: Ned { n: ned.get_n(), e: ned.get_e(), d: ned.get_d() },
        attitude: Euler {
            phi_rad: att.get_phi_rad(),
            theta_rad: att.get_theta_rad(),
            psi_rad: att.get_psi_rad(),
        },
        rates: BodyRates {
            p_rad_s: rates.get_p_rad_s(),
            q_rad_s: rates.get_q_rad_s(),
            r_rad_s: rates.get_r_rad_s(),
        },
        aero: Aero {
            alpha_rad: aero.get_alpha_rad(),
            beta_rad: aero.get_beta_rad(),
            nz_g: aero.get_nz_g(),
            mach: aero.get_mach(),
        },
        tas_mps: r.get_tas_mps(),
        fault_flags: r.get_fault_flags(),
        vehicle_kind: match r.get_vehicle_kind() {
            Ok(wire::VehicleKind::FixedWing) => VehicleKind::FixedWing,
            Ok(wire::VehicleKind::Multirotor) => VehicleKind::Multirotor,
            Err(capnp::NotInSchema(n)) => VehicleKind::Unknown(n),
        },
        control_mode: control_mode_of(r.get_control_mode()),
        combat: CombatStatus {
            team: combat.get_team(),
            hit_points: combat.get_hit_points(),
            hit_points_max: combat.get_hit_points_max(),
            alive: combat.get_alive(),
            ammo: combat.get_ammo(),
        },
        control_period_s: r.get_control_period_s(),
        command: CommandStatus {
            owner_epoch: command.get_owner_epoch(),
            last_seq: command.get_last_seq(),
            refusals: command.get_refusals(),
            last_refusal: refusal_of(command.get_last_refusal()),
            last_refused_epoch: command.get_last_refused_epoch(),
        },
    })
}

/// Serialise a command to one flat Cap'n Proto message.
pub fn encode_command(c: &Command) -> Vec<u8> {
    let mut msg = builder();
    {
        let mut b = msg.init_root::<wire::command::Builder>();
        b.set_target_id(c.target_id);
        b.set_seq(c.seq);
        b.set_telemetry_tick(c.telemetry_tick);
        b.set_trigger(c.trigger);
        b.set_epoch(c.epoch);
        let ctrl = b.init_control();
        match c.control {
            Control::Raw(r) => {
                let mut w = ctrl.init_raw();
                w.set_aileron(r.aileron);
                w.set_elevator(r.elevator);
                w.set_rudder(r.rudder);
                w.set_throttle(r.throttle);
            }
            Control::Autopilot(a) => {
                let mut w = ctrl.init_autopilot();
                w.set_heading_rad(a.heading_rad);
                w.set_alt_hae_m(a.alt_hae_m);
                w.set_tas_mps(a.tas_mps);
            }
            Control::Multirotor(m) => {
                let mut w = ctrl.init_multirotor();
                w.set_roll_rate(m.roll_rate);
                w.set_pitch_rate(m.pitch_rate);
                w.set_yaw_rate(m.yaw_rate);
                w.set_collective(m.collective);
            }
        }
    }
    serialize::write_message_to_words(&msg)
}

/// Deserialise a command. A control variant from a newer schema is an error.
pub fn decode_command(bytes: &[u8]) -> capnp::Result<Command> {
    use wire::command::control::Which;
    let msg = reader(bytes)?;
    let c = msg.get_root::<wire::command::Reader>()?;
    let control = match c.get_control().which()? {
        Which::Raw(r) => {
            let r = r?;
            Control::Raw(RawActuators {
                aileron: r.get_aileron(),
                elevator: r.get_elevator(),
                rudder: r.get_rudder(),
                throttle: r.get_throttle(),
            })
        }
        Which::Autopilot(a) => {
            let a = a?;
            Control::Autopilot(AutopilotTargets {
                heading_rad: a.get_heading_rad(),
                alt_hae_m: a.get_alt_hae_m(),
                tas_mps: a.get_tas_mps(),
            })
        }
        Which::Multirotor(m) => {
            let m = m?;
            Control::Multirotor(MultirotorRaw {
                roll_rate: m.get_roll_rate(),
                pitch_rate: m.get_pitch_rate(),
                yaw_rate: m.get_yaw_rate(),
                collective: m.get_collective(),
            })
        }
    };
    Ok(Command {
        target_id: c.get_target_id(),
        seq: c.get_seq(),
        telemetry_tick: c.get_telemetry_tick(),
        trigger: c.get_trigger(),
        epoch: c.get_epoch(),
        control,
    })
}

/// Serialise a tick end to one flat Cap'n Proto message.
pub fn encode_tick_end(t: &TickEnd) -> Vec<u8> {
    let mut msg = builder();
    {
        let mut b = msg.init_root::<wire::tick_end::Builder>();
        b.set_tick(t.tick);
        b.set_sim_time_s(t.sim_time_s);
    }
    serialize::write_message_to_words(&msg)
}

/// Deserialise a tick end.
pub fn decode_tick_end(bytes: &[u8]) -> capnp::Result<TickEnd> {
    let msg = reader(bytes)?;
    let r = msg.get_root::<wire::tick_end::Reader>()?;
    Ok(TickEnd { tick: r.get_tick(), sim_time_s: r.get_sim_time_s() })
}

fn wire_refusal(r: Refusal) -> wire::Refusal {
    match r {
        Refusal::None => wire::Refusal::None,
        Refusal::FromTheFuture => wire::Refusal::FromTheFuture,
        Refusal::Stale => wire::Refusal::Stale,
        Refusal::OutOfOrder => wire::Refusal::OutOfOrder,
        Refusal::NotOwner => wire::Refusal::NotOwner,
        Refusal::MissingEpoch => wire::Refusal::MissingEpoch,
        Refusal::NonFinite => wire::Refusal::NonFinite,
        Refusal::WrongVehicleKind => wire::Refusal::WrongVehicleKind,
        Refusal::Unknown(n) => panic!("cannot encode Refusal value {n} unknown to this schema"),
    }
}

fn refusal_of(r: Result<wire::Refusal, capnp::NotInSchema>) -> Refusal {
    match r {
        Ok(wire::Refusal::None) => Refusal::None,
        Ok(wire::Refusal::FromTheFuture) => Refusal::FromTheFuture,
        Ok(wire::Refusal::Stale) => Refusal::Stale,
        Ok(wire::Refusal::OutOfOrder) => Refusal::OutOfOrder,
        Ok(wire::Refusal::NotOwner) => Refusal::NotOwner,
        Ok(wire::Refusal::MissingEpoch) => Refusal::MissingEpoch,
        Ok(wire::Refusal::NonFinite) => Refusal::NonFinite,
        Ok(wire::Refusal::WrongVehicleKind) => Refusal::WrongVehicleKind,
        Err(capnp::NotInSchema(n)) => Refusal::Unknown(n),
    }
}

fn wire_control_mode(m: ControlMode) -> wire::ControlMode {
    match m {
        ControlMode::Uncommanded => wire::ControlMode::Uncommanded,
        ControlMode::Raw => wire::ControlMode::Raw,
        ControlMode::Autopilot => wire::ControlMode::Autopilot,
        ControlMode::Failsafe => wire::ControlMode::Failsafe,
        ControlMode::Unknown(n) => {
            panic!("cannot encode ControlMode value {n} unknown to this schema")
        }
    }
}

fn control_mode_of(m: Result<wire::ControlMode, capnp::NotInSchema>) -> ControlMode {
    match m {
        Ok(wire::ControlMode::Uncommanded) => ControlMode::Uncommanded,
        Ok(wire::ControlMode::Raw) => ControlMode::Raw,
        Ok(wire::ControlMode::Autopilot) => ControlMode::Autopilot,
        Ok(wire::ControlMode::Failsafe) => ControlMode::Failsafe,
        Err(capnp::NotInSchema(n)) => ControlMode::Unknown(n),
    }
}

/// Serialise an event to one flat Cap'n Proto message.
///
/// # Panics
/// If an enum field is `Unknown(_)` (only produced by decoding).
pub fn encode_event(e: &Event) -> Vec<u8> {
    let mut msg = builder();
    {
        let mut b = msg.init_root::<wire::event::Builder>();
        b.set_tick(e.tick);
        b.set_sim_time_s(e.sim_time_s);
        let mut k = b.init_kind();
        match &e.kind {
            EventKind::TriggerPressed { aircraft_id, ammo } => {
                let mut w = k.init_trigger_pressed();
                w.set_aircraft_id(*aircraft_id);
                w.set_ammo(*ammo);
            }
            EventKind::TriggerReleased { aircraft_id, rounds_fired, ammo } => {
                let mut w = k.init_trigger_released();
                w.set_aircraft_id(*aircraft_id);
                w.set_rounds_fired(*rounds_fired);
                w.set_ammo(*ammo);
            }
            EventKind::Hit { shooter_id, target_id, damage, hit_points, pos_enu } => {
                let mut w = k.init_hit();
                w.set_shooter_id(*shooter_id);
                w.set_target_id(*target_id);
                w.set_damage(*damage);
                w.set_hit_points(*hit_points);
                set_enu(w.init_pos_enu(), *pos_enu);
            }
            EventKind::Kill { victim_id, killer_id, pos_enu } => {
                let mut w = k.init_kill();
                w.set_victim_id(*victim_id);
                match killer_id {
                    Some(id) => w.reborrow().init_killer().set_aircraft_id(*id),
                    None => w.reborrow().init_killer().set_unknown(()),
                }
                set_enu(w.init_pos_enu(), *pos_enu);
            }
            EventKind::ModeChange { aircraft_id, from, to, reason } => {
                let mut w = k.init_mode_change();
                w.set_aircraft_id(*aircraft_id);
                w.set_from(wire_control_mode(*from));
                w.set_to(wire_control_mode(*to));
                w.set_reason(match reason {
                    ModeReason::Command => wire::ModeReason::Command,
                    ModeReason::LeaseExpired => wire::ModeReason::LeaseExpired,
                    ModeReason::ControllerLost => wire::ModeReason::ControllerLost,
                    ModeReason::Unknown(n) => {
                        panic!("cannot encode ModeReason value {n} unknown to this schema")
                    }
                });
            }
            EventKind::ControllerUp { aircraft_id } => k.set_controller_up(*aircraft_id),
            EventKind::ControllerDown { aircraft_id } => k.set_controller_down(*aircraft_id),
            EventKind::RunEnded { reason, message } => {
                let mut w = k.init_run_ended();
                w.set_reason(match reason {
                    EndReason::MaxTime => wire::EndReason::MaxTime,
                    EndReason::EngagementDecided => wire::EndReason::EngagementDecided,
                    EndReason::Fatal => wire::EndReason::Fatal,
                    EndReason::Stopped => wire::EndReason::Stopped,
                    EndReason::Unknown(n) => {
                        panic!("cannot encode EndReason value {n} unknown to this schema")
                    }
                });
                w.set_message(message.as_str());
            }
        }
    }
    serialize::write_message_to_words(&msg)
}

/// Deserialise an event. A kind from a newer schema is an error (skip it).
pub fn decode_event(bytes: &[u8]) -> capnp::Result<Event> {
    use wire::event::kind::Which;
    let msg = reader(bytes)?;
    let e = msg.get_root::<wire::event::Reader>()?;
    let kind = match e.get_kind().which()? {
        Which::TriggerPressed(w) => {
            let w = w?;
            EventKind::TriggerPressed { aircraft_id: w.get_aircraft_id(), ammo: w.get_ammo() }
        }
        Which::TriggerReleased(w) => {
            let w = w?;
            EventKind::TriggerReleased {
                aircraft_id: w.get_aircraft_id(),
                rounds_fired: w.get_rounds_fired(),
                ammo: w.get_ammo(),
            }
        }
        Which::Hit(w) => {
            let w = w?;
            EventKind::Hit {
                shooter_id: w.get_shooter_id(),
                target_id: w.get_target_id(),
                damage: w.get_damage(),
                hit_points: w.get_hit_points(),
                pos_enu: get_enu(w.get_pos_enu()?),
            }
        }
        Which::Kill(w) => {
            use wire::kill::killer::Which as Killer;
            let w = w?;
            EventKind::Kill {
                victim_id: w.get_victim_id(),
                killer_id: match w.get_killer().which()? {
                    Killer::Unknown(()) => None,
                    Killer::AircraftId(id) => Some(id),
                },
                pos_enu: get_enu(w.get_pos_enu()?),
            }
        }
        Which::ModeChange(w) => {
            let w = w?;
            EventKind::ModeChange {
                aircraft_id: w.get_aircraft_id(),
                from: control_mode_of(w.get_from()),
                to: control_mode_of(w.get_to()),
                reason: match w.get_reason() {
                    Ok(wire::ModeReason::Command) => ModeReason::Command,
                    Ok(wire::ModeReason::LeaseExpired) => ModeReason::LeaseExpired,
                    Ok(wire::ModeReason::ControllerLost) => ModeReason::ControllerLost,
                    Err(capnp::NotInSchema(n)) => ModeReason::Unknown(n),
                },
            }
        }
        Which::ControllerUp(id) => EventKind::ControllerUp { aircraft_id: id },
        Which::ControllerDown(id) => EventKind::ControllerDown { aircraft_id: id },
        Which::RunEnded(w) => {
            let w = w?;
            EventKind::RunEnded {
                reason: match w.get_reason() {
                    Ok(wire::EndReason::MaxTime) => EndReason::MaxTime,
                    Ok(wire::EndReason::EngagementDecided) => EndReason::EngagementDecided,
                    Ok(wire::EndReason::Fatal) => EndReason::Fatal,
                    Ok(wire::EndReason::Stopped) => EndReason::Stopped,
                    Err(capnp::NotInSchema(n)) => EndReason::Unknown(n),
                },
                message: w.get_message()?.to_string()?,
            }
        }
    };
    Ok(Event { tick: e.get_tick(), sim_time_s: e.get_sim_time_s(), kind })
}
