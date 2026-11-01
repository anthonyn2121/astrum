//! `astrum-proto` — the ASTRUM Cap'n Proto wire schema (`schema/astrum.capnp`),
//! its generated Rust bindings ([`wire`]), and plain-data mirrors with
//! encode/decode helpers so the rest of the workspace never touches capnp
//! builders directly.
//!
//! Two payloads: [`Telemetry`] (sim → subscribers) and [`Command`] (controllers →
//! sim). Decoding reads straight from the payload slice (no segment copy) and
//! rejects trailing bytes. Enum values from a newer schema decode to an
//! `Unknown(n)` variant instead of being silently mapped to a known one.

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

/// The control payload of a [`Command`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    Raw(RawActuators),
    Autopilot(AutopilotTargets),
}

/// A command for one aircraft. Mirrors `Command`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    pub target_id: u32,
    /// Per-sender, strictly increasing.
    pub seq: u64,
    /// The `Telemetry::tick` this command was computed from.
    pub telemetry_tick: u64,
    pub trigger: bool,
    pub control: Control,
}

/// Room for one whole message in the first segment (Telemetry is ~40 words), so
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
/// If `vehicle_kind` or `control_mode` is `Unknown(_)` (only produced by decoding).
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
        b.set_control_mode(match t.control_mode {
            ControlMode::Uncommanded => wire::ControlMode::Uncommanded,
            ControlMode::Raw => wire::ControlMode::Raw,
            ControlMode::Autopilot => wire::ControlMode::Autopilot,
            ControlMode::Failsafe => wire::ControlMode::Failsafe,
            ControlMode::Unknown(n) => {
                panic!("cannot encode ControlMode value {n} unknown to this schema")
            }
        });
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
        control_mode: match r.get_control_mode() {
            Ok(wire::ControlMode::Uncommanded) => ControlMode::Uncommanded,
            Ok(wire::ControlMode::Raw) => ControlMode::Raw,
            Ok(wire::ControlMode::Autopilot) => ControlMode::Autopilot,
            Ok(wire::ControlMode::Failsafe) => ControlMode::Failsafe,
            Err(capnp::NotInSchema(n)) => ControlMode::Unknown(n),
        },
        combat: CombatStatus {
            team: combat.get_team(),
            hit_points: combat.get_hit_points(),
            hit_points_max: combat.get_hit_points_max(),
            alive: combat.get_alive(),
            ammo: combat.get_ammo(),
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
    };
    Ok(Command {
        target_id: c.get_target_id(),
        seq: c.get_seq(),
        telemetry_tick: c.get_telemetry_tick(),
        trigger: c.get_trigger(),
        control,
    })
}
