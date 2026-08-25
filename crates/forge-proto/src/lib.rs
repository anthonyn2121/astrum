//! `forge-proto` — the FORGE Cap'n Proto wire schema and generated Rust
//! bindings, plus ergonomic plain-data structs and encode/decode helpers so the
//! rest of the workspace never touches capnp builders directly.
//!
//! Three payloads share one schema (`schema/forge.capnp`): [`TelemetrySample`]
//! (sim → subscribers), [`CommandMsg`] (controllers → sim), and the bag stream
//! ([`write_bag_frame`] / [`BagReader`]) for record & replay.

use std::io::{BufRead, Write};

/// Generated code from `schema/forge.capnp`.
#[allow(clippy::all, dead_code, unused)]
pub mod forge_capnp {
    include!(concat!(env!("OUT_DIR"), "/forge_capnp.rs"));
}

use capnp::message::{Builder, HeapAllocator, ReaderOptions};
use capnp::serialize;

/// Flight-vehicle class carried alongside telemetry. Mirrors the `VehicleKind`
/// enum in the schema (and `forge_sim::components::VehicleKind`) so a subscriber
/// — notably the 3-D replay renderer — can pick the right model without any
/// out-of-band scenario knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VehicleKind {
    /// Aerodynamic fixed-wing (control surfaces + throttle).
    #[default]
    FixedWing,
    /// Multirotor / quadrotor (body-rate + collective demands).
    Multirotor,
}

/// Plain-data telemetry sample. Mirrors `Telemetry` in the schema.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetrySample {
    pub aircraft_id: u32,
    pub sim_time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    pub enu_e: f64,
    pub enu_n: f64,
    pub enu_u: f64,
    pub vel_north_mps: f64,
    pub vel_east_mps: f64,
    pub vel_down_mps: f64,
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    pub airspeed_mps: f64,
    pub health: u8,
    pub vehicle_kind: VehicleKind,
}

/// The control payload of a [`CommandMsg`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    /// Raw actuator setpoints: aileron/elevator/rudder in [-1,1], throttle in [0,1].
    Raw { aileron: f64, elevator: f64, rudder: f64, throttle: f64 },
    /// High-level autopilot targets.
    Autopilot { heading_rad: f64, alt_msl_m: f64, airspeed_mps: f64 },
}

/// Plain-data command. Mirrors `Command` in the schema.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommandMsg {
    pub target_id: u32,
    pub control: Control,
}

fn build_telemetry(b: &mut forge_capnp::telemetry::Builder, s: &TelemetrySample) {
    b.set_aircraft_id(s.aircraft_id);
    b.set_sim_time_s(s.sim_time_s);
    b.set_airspeed_mps(s.airspeed_mps);
    b.set_health(s.health);
    b.set_vehicle_kind(match s.vehicle_kind {
        VehicleKind::FixedWing => forge_capnp::VehicleKind::FixedWing,
        VehicleKind::Multirotor => forge_capnp::VehicleKind::Multirotor,
    });
    {
        let mut geo = b.reborrow().init_geo();
        geo.set_lat_deg(s.lat_deg);
        geo.set_lon_deg(s.lon_deg);
        geo.set_alt_m(s.alt_m);
    }
    {
        let mut enu = b.reborrow().init_enu();
        enu.set_x(s.enu_e);
        enu.set_y(s.enu_n);
        enu.set_z(s.enu_u);
    }
    {
        let mut vel = b.reborrow().init_vel_ned();
        vel.set_x(s.vel_north_mps);
        vel.set_y(s.vel_east_mps);
        vel.set_z(s.vel_down_mps);
    }
    {
        let mut att = b.reborrow().init_attitude();
        att.set_phi_rad(s.phi_rad);
        att.set_theta_rad(s.theta_rad);
        att.set_psi_rad(s.psi_rad);
    }
}

fn read_telemetry(r: forge_capnp::telemetry::Reader) -> capnp::Result<TelemetrySample> {
    let geo = r.get_geo()?;
    let enu = r.get_enu()?;
    let vel = r.get_vel_ned()?;
    let att = r.get_attitude()?;
    Ok(TelemetrySample {
        aircraft_id: r.get_aircraft_id(),
        sim_time_s: r.get_sim_time_s(),
        lat_deg: geo.get_lat_deg(),
        lon_deg: geo.get_lon_deg(),
        alt_m: geo.get_alt_m(),
        enu_e: enu.get_x(),
        enu_n: enu.get_y(),
        enu_u: enu.get_z(),
        vel_north_mps: vel.get_x(),
        vel_east_mps: vel.get_y(),
        vel_down_mps: vel.get_z(),
        phi_rad: att.get_phi_rad(),
        theta_rad: att.get_theta_rad(),
        psi_rad: att.get_psi_rad(),
        airspeed_mps: r.get_airspeed_mps(),
        health: r.get_health(),
        // Unknown-future variants (from a newer schema) degrade to fixed-wing.
        vehicle_kind: match r.get_vehicle_kind() {
            Ok(forge_capnp::VehicleKind::Multirotor) => VehicleKind::Multirotor,
            _ => VehicleKind::FixedWing,
        },
    })
}

/// Serialise a telemetry sample to a Cap'n Proto message.
pub fn encode_telemetry(s: &TelemetrySample) -> Vec<u8> {
    let mut msg = Builder::new_default();
    build_telemetry(&mut msg.init_root::<forge_capnp::telemetry::Builder>(), s);
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &msg).expect("write to Vec cannot fail");
    buf
}

/// Deserialise a telemetry sample.
pub fn decode_telemetry(bytes: &[u8]) -> capnp::Result<TelemetrySample> {
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    read_telemetry(msg.get_root::<forge_capnp::telemetry::Reader>()?)
}

/// Serialise a command to a Cap'n Proto message.
pub fn encode_command(c: &CommandMsg) -> Vec<u8> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<forge_capnp::command::Builder>();
        b.set_target_id(c.target_id);
        let ctrl = b.init_control();
        match c.control {
            Control::Raw { aileron, elevator, rudder, throttle } => {
                let mut r = ctrl.init_raw();
                r.set_aileron(aileron);
                r.set_elevator(elevator);
                r.set_rudder(rudder);
                r.set_throttle(throttle);
            }
            Control::Autopilot { heading_rad, alt_msl_m, airspeed_mps } => {
                let mut a = ctrl.init_autopilot();
                a.set_heading_rad(heading_rad);
                a.set_alt_msl_m(alt_msl_m);
                a.set_airspeed_mps(airspeed_mps);
            }
        }
    }
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &msg).expect("write to Vec cannot fail");
    buf
}

/// Deserialise a command.
pub fn decode_command(bytes: &[u8]) -> capnp::Result<CommandMsg> {
    use forge_capnp::command::control::Which;
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    let c = msg.get_root::<forge_capnp::command::Reader>()?;
    let control = match c.get_control().which()? {
        Which::Raw(r) => {
            let r = r?;
            Control::Raw {
                aileron: r.get_aileron(),
                elevator: r.get_elevator(),
                rudder: r.get_rudder(),
                throttle: r.get_throttle(),
            }
        }
        Which::Autopilot(a) => {
            let a = a?;
            Control::Autopilot {
                heading_rad: a.get_heading_rad(),
                alt_msl_m: a.get_alt_msl_m(),
                airspeed_mps: a.get_airspeed_mps(),
            }
        }
    };
    Ok(CommandMsg { target_id: c.get_target_id(), control })
}

// --- Bag stream: a sequence of `BagFrame` capnp messages --------------------

fn build_bag_frame(recv_wall_ns: u64, s: &TelemetrySample) -> Builder<HeapAllocator> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<forge_capnp::bag_frame::Builder>();
        b.set_recv_wall_ns(recv_wall_ns);
        build_telemetry(&mut b.init_telemetry(), s);
    }
    msg
}

/// Append one bag frame (wall-clock stamp + telemetry) to a writer.
pub fn write_bag_frame<W: Write>(
    w: &mut W,
    recv_wall_ns: u64,
    s: &TelemetrySample,
) -> capnp::Result<()> {
    serialize::write_message(w, &build_bag_frame(recv_wall_ns, s))
}

/// Streaming reader over a bag file. Yields `(recv_wall_ns, sample)` per frame.
pub struct BagReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> BagReader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    /// Read the next frame, or `None` at clean end-of-stream.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> capnp::Result<Option<(u64, TelemetrySample)>> {
        match serialize::try_read_message(&mut self.reader, ReaderOptions::new())? {
            None => Ok(None),
            Some(msg) => {
                let f = msg.get_root::<forge_capnp::bag_frame::Reader>()?;
                let sample = read_telemetry(f.get_telemetry()?)?;
                Ok(Some((f.get_recv_wall_ns(), sample)))
            }
        }
    }
}
