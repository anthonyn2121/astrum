//! Zenoh bridge: the drone's telemetry and the race's task state out, actuator
//! commands in.
//!
//! # What is reused and what is new
//!
//! Nothing about *flying* a quadrotor needs a new wire format. The drone's state
//! goes out as a plain [`TelemetrySample`] on `astrum/telemetry/<id>`, and a
//! controller flies it with a plain `Control::Raw` [`CommandMsg`] on
//! `astrum/command/<id>` — the same four normalized channels the fleet sim takes,
//! which is why the bundled C++/Python controllers already talk to an F450 (see
//! `make demo-quad`).
//!
//! What has no home on that wire is the *task*: where the gates are, and which one
//! the drone is flying at. [`TelemetrySample`] means "one aircraft's authoritative
//! state" and is what the bag recorder and the 3-D replay consume, so gate progress
//! does not belong on it. It gets its own two messages
//! ([`RaceCourseMsg`] / [`RaceStateMsg`]) on their own key space
//! (`astrum/race/**`), which leaves every existing message and reader untouched.
//!
//! # Switching it on
//!
//! Every system here takes its transport as `Option<NonSend<..>>` and no-ops when
//! it is absent, so [`crate::build_app_with_course`] stays a pure physics app —
//! which is what lets the integration tests run without opening a session.
//! Networking is switched on purely by [`install`]ing the resources.

use astrum_fdm::prop;
use astrum_net::{CommandReceiver, RacePublisher, Session, TelemetryPublisher};
use astrum_proto::{
    CommandMsg, Control, RaceCourseMsg, RaceGateSpec, RaceOutcome, RaceStateMsg, TelemetrySample,
    VehicleKind,
};
use bevy::prelude::*;

use crate::components::{
    Drone, DroneState, FdmInstance, Outcome, RaceCourse, RaceProgress, SimParams, WorldFrame,
};
use crate::control::{FlightCommand, RawOverride};
use crate::gate::{BAR_M, OPENING_M};

/// How often the course is republished, in simulated seconds.
///
/// The course is static, so publishing it once would be enough — except that Zenoh
/// pub/sub has no late-joiner replay, and the normal case is a controller started
/// *after* the environment. Republishing on a timer costs a few hundred bytes a
/// second and makes "start the controller whenever" work.
const COURSE_REPUBLISH_S: f64 = 1.0;

/// Open the racing environment's transport and activate the bridge systems.
///
/// Installs the telemetry publisher, the race-state publisher and the command
/// receiver as `NonSend` resources (the app is single-threaded under
/// `ScheduleRunnerPlugin`). A failed subscription is logged rather than fatal — the
/// affected system simply stays inert, exactly as if networking were off.
pub fn install(app: &mut App, session: &Session) {
    app.insert_non_send(TelemetryPublisher::new(session));
    app.insert_non_send(RacePublisher::new(session));
    match CommandReceiver::bind(session) {
        Ok(r) => {
            app.insert_non_send(r);
        }
        Err(e) => eprintln!("[astrum-race] command subscribe failed: {e}"),
    }
}

/// Publish the drone's post-step state on `astrum/telemetry/<id>`.
///
/// Runs after `sync_pose`, so `DroneState` is this tick's pose rather than last
/// tick's.
pub fn publish_telemetry(
    publisher: Option<NonSend<TelemetryPublisher>>,
    q: Query<(&Drone, &DroneState)>,
) {
    let Some(publisher) = publisher else { return };
    for (drone, st) in &q {
        let sample = TelemetrySample {
            aircraft_id: drone.id,
            sim_time_s: st.sim_time_s,
            lat_deg: st.lat_deg,
            lon_deg: st.lon_deg,
            alt_m: st.alt_m,
            enu_e: st.pos_enu.x,
            enu_n: st.pos_enu.y,
            enu_u: st.pos_enu.z,
            // The wire carries NED velocity; `DroneState` holds ENU.
            vel_north_mps: st.vel_enu.y,
            vel_east_mps: st.vel_enu.x,
            vel_down_mps: -st.vel_enu.z,
            phi_rad: st.phi_rad,
            theta_rad: st.theta_rad,
            psi_rad: st.psi_rad,
            airspeed_mps: st.airspeed_mps,
            health: 0,
            vehicle_kind: VehicleKind::Multirotor,
        };
        // A dropped frame is non-fatal — the subscriber just reads the next one.
        if let Err(e) = publisher.publish(&sample) {
            eprintln!("[astrum-race] telemetry publish failed for drone {}: {e}", drone.id);
        }
    }
}

/// Publish race progress every tick, and the course every [`COURSE_REPUBLISH_S`].
///
/// Runs after `track_gates` and *before* the termination systems, so the tick that
/// completes the course still publishes the state carrying `finished`.
pub fn publish_race(
    publisher: Option<NonSend<RacePublisher>>,
    course: Res<RaceCourse>,
    progress: Res<RaceProgress>,
    params: Res<SimParams>,
    q: Query<(&Drone, &DroneState)>,
    mut ticks: Local<u64>,
) {
    let Some(publisher) = publisher else { return };
    let Some((drone, st)) = q.iter().next() else { return };

    let state = RaceStateMsg {
        drone_id: drone.id,
        sim_time_s: st.sim_time_s,
        pending_gate: progress.next_gate,
        passed: progress.passed,
        missed: progress.missed,
        gate_count: course.0.gates.len() as u32,
        last_pass_s: progress.last_pass_s,
        finished: progress.finished,
        outcome: wire_outcome(progress.outcome),
    };
    if let Err(e) = publisher.publish_state(&state) {
        eprintln!("[astrum-race] race-state publish failed: {e}");
    }

    // The first tick publishes the course, so a controller already waiting has it
    // immediately rather than a republish period later.
    let period = (COURSE_REPUBLISH_S * params.rate_hz).max(1.0) as u64;
    if ticks.is_multiple_of(period) {
        if let Err(e) = publisher.publish_course(&course_msg(drone.id, &course)) {
            eprintln!("[astrum-race] course publish failed: {e}");
        }
    }
    *ticks += 1;
}

/// Project the environment's [`Outcome`] onto the wire's.
///
/// Two enums with the same variants, deliberately: the environment's terminal state is
/// core physics-side truth and does not depend on a transport being installed, exactly
/// as `astrum_sim`'s `VehicleKind` mirrors the schema's. This function is the seam, and
/// its exhaustive `match` is what makes adding a variant to either side a compile
/// error rather than a silent mistranslation.
fn wire_outcome(outcome: Outcome) -> RaceOutcome {
    match outcome {
        Outcome::Running => RaceOutcome::Running,
        Outcome::Complete => RaceOutcome::Complete,
        Outcome::CrashedGround => RaceOutcome::CrashedGround,
        Outcome::CrashedGate => RaceOutcome::CrashedGate,
        Outcome::TimedOut => RaceOutcome::TimedOut,
        Outcome::SpawnFailed => RaceOutcome::SpawnFailed,
    }
}

/// Project the generated course onto the wire.
///
/// Gate hardware dimensions travel with it so a controller sizes its aim from the
/// message rather than hard-coding 0.7874 m.
fn course_msg(drone_id: u32, course: &RaceCourse) -> RaceCourseMsg {
    RaceCourseMsg {
        drone_id,
        seed: course.0.seed,
        opening_m: OPENING_M,
        bar_m: BAR_M,
        gates: course
            .0
            .gates
            .iter()
            .map(|g| RaceGateSpec {
                index: g.index,
                center_e: g.center.x,
                center_n: g.center.y,
                center_u: g.center.z,
                yaw_rad: g.yaw_rad,
            })
            .collect(),
    }
}

/// Drain queued commands and apply them to the targeted drone.
///
/// Runs *before* `stabilize`, so a command that arrives this tick is the one this
/// tick integrates.
///
/// * `Control::Raw` → a [`RawOverride`] lease: the four channels straight through,
///   with the inner loop standing down until the lease lapses.
/// * `Control::Autopilot` → mapped onto [`FlightCommand`]. See
///   [`autopilot_to_flight_command`] for what that mapping can and cannot express.
pub fn ingest_commands(
    receiver: Option<NonSend<CommandReceiver>>,
    frame: Res<WorldFrame>,
    mut q: Query<(&Drone, &FdmInstance, &mut FlightCommand, &mut RawOverride)>,
) {
    let Some(receiver) = receiver else { return };
    let commands = receiver.drain();
    if commands.is_empty() {
        return;
    }
    let origin_alt_m = frame.0.origin().alt_m;
    for cmd in commands {
        for (drone, inst, mut flight, mut raw) in &mut q {
            if drone.id != cmd.target_id {
                continue;
            }
            apply_wire_command(&cmd, &inst.0, origin_alt_m, &mut flight, &mut raw);
        }
    }
}

/// Apply one decoded wire command to a drone's command state. Split out from the
/// system so the mapping is testable without a world.
pub fn apply_wire_command(
    cmd: &CommandMsg,
    fdm: &astrum_fdm::Fdm,
    origin_alt_m: f64,
    flight: &mut FlightCommand,
    raw: &mut RawOverride,
) {
    match cmd.control {
        Control::Raw { aileron, elevator, rudder, throttle } => {
            raw.set(aileron, elevator, rudder, throttle, fdm.get(prop::SIM_TIME));
        }
        Control::Autopilot { heading_rad, alt_msl_m, airspeed_mps } => {
            raw.active = false;
            *flight =
                autopilot_to_flight_command(heading_rad, alt_msl_m, airspeed_mps, origin_alt_m);
        }
    }
}

/// Read the shared `AutopilotTargets` as a [`FlightCommand`].
///
/// `origin_alt_m` is the ENU datum's height above the ellipsoid, which is the whole
/// of the altitude conversion: ENU "up" at the origin *is* the ellipsoid normal, so
/// over a course tens of metres wide the curvature term is sub-millimetre.
///
/// # What this mapping cannot express
///
/// `AutopilotTargets` carries a **scalar** `airspeedMps`, so the only velocity it
/// can ask for is *forward along the commanded heading*. That is the coordinated-
/// flight idiom the shared autopilot was written for, and it is a strict subset of
/// what a quadrotor can do: a racing drone crabs — it holds a heading while
/// translating sideways, which is exactly how you line up on a gate you are already
/// turning past. There is no encoding of that here. A controller that needs it
/// should fly `Control::Raw` and own the law, or write [`FlightCommand`] in-process.
pub fn autopilot_to_flight_command(
    heading_rad: f64,
    alt_msl_m: f64,
    airspeed_mps: f64,
    origin_alt_m: f64,
) -> FlightCommand {
    let (s, c) = heading_rad.sin_cos();
    FlightCommand {
        alt_m: alt_msl_m - origin_alt_m,
        heading_rad,
        vel_enu: bevy::math::DVec3::new(airspeed_mps * s, airspeed_mps * c, 0.0),
    }
}
