//! `astrum-proto` — the ASTRUM Cap'n Proto wire schema and generated Rust
//! bindings, plus ergonomic plain-data structs and encode/decode helpers so the
//! rest of the workspace never touches capnp builders directly.
//!
//! Three payloads share one schema (`schema/astrum.capnp`): [`TelemetrySample`]
//! (sim → subscribers), [`CommandMsg`] (controllers → sim), and the bag stream
//! ([`write_bag_frame`] / [`BagReader`]) for record & replay.
//!
//! The drone-racing environment adds two more in the same schema —
//! [`RaceCourseMsg`] and [`RaceStateMsg`] — carrying *task* state (where the gates
//! are, which one is next) that has no place on [`TelemetrySample`]. It reuses
//! telemetry and commands unchanged.

use std::io::{BufRead, Write};

/// Generated code from `schema/astrum.capnp`.
#[allow(clippy::all, dead_code, unused)]
pub mod astrum_capnp {
    include!(concat!(env!("OUT_DIR"), "/astrum_capnp.rs"));
}

use capnp::message::{Builder, HeapAllocator, ReaderOptions};
use capnp::serialize;

/// Flight-vehicle class carried alongside telemetry. Mirrors the `VehicleKind`
/// enum in the schema (and `astrum_sim::components::VehicleKind`) so a subscriber
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

fn build_telemetry(b: &mut astrum_capnp::telemetry::Builder, s: &TelemetrySample) {
    b.set_aircraft_id(s.aircraft_id);
    b.set_sim_time_s(s.sim_time_s);
    b.set_airspeed_mps(s.airspeed_mps);
    b.set_health(s.health);
    b.set_vehicle_kind(match s.vehicle_kind {
        VehicleKind::FixedWing => astrum_capnp::VehicleKind::FixedWing,
        VehicleKind::Multirotor => astrum_capnp::VehicleKind::Multirotor,
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

fn read_telemetry(r: astrum_capnp::telemetry::Reader) -> capnp::Result<TelemetrySample> {
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
            Ok(astrum_capnp::VehicleKind::Multirotor) => VehicleKind::Multirotor,
            _ => VehicleKind::FixedWing,
        },
    })
}

/// Serialise a telemetry sample to a Cap'n Proto message.
pub fn encode_telemetry(s: &TelemetrySample) -> Vec<u8> {
    let mut msg = Builder::new_default();
    build_telemetry(&mut msg.init_root::<astrum_capnp::telemetry::Builder>(), s);
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &msg).expect("write to Vec cannot fail");
    buf
}

/// Deserialise a telemetry sample.
pub fn decode_telemetry(bytes: &[u8]) -> capnp::Result<TelemetrySample> {
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    read_telemetry(msg.get_root::<astrum_capnp::telemetry::Reader>()?)
}

/// Serialise a command to a Cap'n Proto message.
pub fn encode_command(c: &CommandMsg) -> Vec<u8> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<astrum_capnp::command::Builder>();
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
    use astrum_capnp::command::control::Which;
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    let c = msg.get_root::<astrum_capnp::command::Reader>()?;
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

// --- Drone racing: course + progress ----------------------------------------

/// One gate's pose. Mirrors `RaceGate` in the schema.
///
/// The centre is the middle of the *opening*, on the same world ENU grid
/// [`TelemetrySample::enu_e`]/`_n`/`_u` reports; `yaw_rad` is the heading of the
/// gate's **normal**, i.e. the direction of flight through it (`0` = North, `+` =
/// East), so `(sin yaw, cos yaw)` is the approach direction in ENU.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RaceGateSpec {
    pub index: u32,
    pub center_e: f64,
    pub center_n: f64,
    pub center_u: f64,
    pub yaw_rad: f64,
}

/// The racing track. Mirrors `RaceCourse` in the schema.
///
/// Static for a run, and therefore *republished* at a low rate rather than sent
/// once — Zenoh pub/sub has no late-joiner replay, so a controller that starts
/// after the sim would otherwise never receive it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RaceCourseMsg {
    pub drone_id: u32,
    /// The seed that drew this course — its identity.
    pub seed: u64,
    /// Side of each gate's square opening (m).
    pub opening_m: f64,
    /// Frame stock width/depth (m). The hardware lies *outside* the opening.
    pub bar_m: f64,
    pub gates: Vec<RaceGateSpec>,
}

/// Why a run ended, or that it hasn't. Mirrors `RaceOutcome` in the schema (and
/// `astrum_race::components::Outcome`).
///
/// [`Running`](Self::Running) is the default, matching the schema's ordinal 0: a
/// message from a writer that predates the field decodes as "still going".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RaceOutcome {
    /// Still flying.
    #[default]
    Running,
    /// Every gate passed, in order — the only way to win.
    Complete,
    /// The airframe reached the ground plane.
    CrashedGround,
    /// The airframe struck a gate's frame.
    CrashedGate,
    /// The simulated-time limit expired with gates still to fly.
    TimedOut,
    /// The drone never launched, so nothing was ever flown.
    SpawnFailed,
}

/// How the run is going. Mirrors `RaceState` in the schema.
///
/// Carries no vehicle state — that is [`TelemetrySample`]'s job. Every progress
/// figure is a cumulative counter rather than an event, so a subscriber that skips
/// messages still reads the true totals.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RaceStateMsg {
    pub drone_id: u32,
    pub sim_time_s: f64,
    /// Index of the gate being flown at; equals `gate_count` once finished.
    pub pending_gate: u32,
    pub passed: u32,
    pub missed: u32,
    /// Number of gates in the course, so this message reads before the course does.
    pub gate_count: u32,
    pub last_pass_s: f64,
    /// The run is over, for whatever reason `outcome` names.
    pub finished: bool,
    /// Why it ended; [`RaceOutcome::Running`] until it does.
    pub outcome: RaceOutcome,
}

/// Serialise a course to a Cap'n Proto message.
pub fn encode_race_course(c: &RaceCourseMsg) -> Vec<u8> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<astrum_capnp::race_course::Builder>();
        b.set_drone_id(c.drone_id);
        b.set_seed(c.seed);
        b.set_opening_m(c.opening_m);
        b.set_bar_m(c.bar_m);
        let mut gates = b.init_gates(c.gates.len() as u32);
        for (i, g) in c.gates.iter().enumerate() {
            let mut gb = gates.reborrow().get(i as u32);
            gb.set_index(g.index);
            gb.set_yaw_rad(g.yaw_rad);
            // `init_center` consumes the builder, so it goes last.
            let mut ctr = gb.init_center();
            ctr.set_x(g.center_e);
            ctr.set_y(g.center_n);
            ctr.set_z(g.center_u);
        }
    }
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &msg).expect("write to Vec cannot fail");
    buf
}

/// Deserialise a course.
pub fn decode_race_course(bytes: &[u8]) -> capnp::Result<RaceCourseMsg> {
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    let r = msg.get_root::<astrum_capnp::race_course::Reader>()?;
    let mut gates = Vec::new();
    for g in r.get_gates()? {
        let c = g.get_center()?;
        gates.push(RaceGateSpec {
            index: g.get_index(),
            center_e: c.get_x(),
            center_n: c.get_y(),
            center_u: c.get_z(),
            yaw_rad: g.get_yaw_rad(),
        });
    }
    Ok(RaceCourseMsg {
        drone_id: r.get_drone_id(),
        seed: r.get_seed(),
        opening_m: r.get_opening_m(),
        bar_m: r.get_bar_m(),
        gates,
    })
}

/// Serialise a race-progress sample to a Cap'n Proto message.
pub fn encode_race_state(s: &RaceStateMsg) -> Vec<u8> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<astrum_capnp::race_state::Builder>();
        b.set_drone_id(s.drone_id);
        b.set_sim_time_s(s.sim_time_s);
        b.set_pending_gate(s.pending_gate);
        b.set_passed(s.passed);
        b.set_missed(s.missed);
        b.set_gate_count(s.gate_count);
        b.set_last_pass_s(s.last_pass_s);
        b.set_finished(s.finished);
        b.set_outcome(match s.outcome {
            RaceOutcome::Running => astrum_capnp::RaceOutcome::Running,
            RaceOutcome::Complete => astrum_capnp::RaceOutcome::Complete,
            RaceOutcome::CrashedGround => astrum_capnp::RaceOutcome::CrashedGround,
            RaceOutcome::CrashedGate => astrum_capnp::RaceOutcome::CrashedGate,
            RaceOutcome::TimedOut => astrum_capnp::RaceOutcome::TimedOut,
            RaceOutcome::SpawnFailed => astrum_capnp::RaceOutcome::SpawnFailed,
        });
    }
    let mut buf = Vec::new();
    serialize::write_message(&mut buf, &msg).expect("write to Vec cannot fail");
    buf
}

/// Deserialise a race-progress sample.
pub fn decode_race_state(bytes: &[u8]) -> capnp::Result<RaceStateMsg> {
    let msg = serialize::read_message(&mut &bytes[..], ReaderOptions::new())?;
    let r = msg.get_root::<astrum_capnp::race_state::Reader>()?;
    Ok(RaceStateMsg {
        drone_id: r.get_drone_id(),
        sim_time_s: r.get_sim_time_s(),
        pending_gate: r.get_pending_gate(),
        passed: r.get_passed(),
        missed: r.get_missed(),
        gate_count: r.get_gate_count(),
        last_pass_s: r.get_last_pass_s(),
        finished: r.get_finished(),
        // An outcome this build does not know decodes as `Running`, which is the safe
        // reading: it says "no terminal state I can act on", and `finished` still
        // tells a controller to stop flying.
        outcome: match r.get_outcome() {
            Ok(astrum_capnp::RaceOutcome::Complete) => RaceOutcome::Complete,
            Ok(astrum_capnp::RaceOutcome::CrashedGround) => RaceOutcome::CrashedGround,
            Ok(astrum_capnp::RaceOutcome::CrashedGate) => RaceOutcome::CrashedGate,
            Ok(astrum_capnp::RaceOutcome::TimedOut) => RaceOutcome::TimedOut,
            Ok(astrum_capnp::RaceOutcome::SpawnFailed) => RaceOutcome::SpawnFailed,
            _ => RaceOutcome::Running,
        },
    })
}

// --- Bag stream: a sequence of `BagFrame` capnp messages --------------------

fn build_bag_frame(recv_wall_ns: u64, s: &TelemetrySample) -> Builder<HeapAllocator> {
    let mut msg = Builder::new_default();
    {
        let mut b = msg.init_root::<astrum_capnp::bag_frame::Builder>();
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
                let f = msg.get_root::<astrum_capnp::bag_frame::Reader>()?;
                let sample = read_telemetry(f.get_telemetry()?)?;
                Ok(Some((f.get_recv_wall_ns(), sample)))
            }
        }
    }
}
