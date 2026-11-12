//! Shared fixtures for the `astrum-race` integration tests.
//!
//! Two kinds of app: a **real** one ([`race_app`], built with
//! [`astrum_race::build_race_app`] on a scenario TOML, which flies JSBSim), and a
//! **rig** ([`rig`]) that holds only the race's own tick systems and a drone that a
//! test places by hand, so the interesting inputs — a tick that steps clean over a
//! frame, a pass 20 cm off centre — can be had on demand.
#![allow(dead_code)] // each test binary uses only a subset of these helpers

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use astrum_net::{CommandSender, LatestTelemetry, Selection, Session};
use astrum_proto::{Control, MultirotorRaw, Telemetry};

use astrum_geo::Enu;
use astrum_proto::EventKind;
use astrum_race::components::{Race, RaceCourse, RaceProgress};
use astrum_race::course::{Course, CourseParams};
use astrum_race::gate::Gate;
use astrum_race::systems::tick_systems;
use astrum_sim::combat::Dead;
use astrum_sim::components::{Aircraft, AircraftState, SimClock, SimParams, SimStatus};
use astrum_sim::events::{publish_events, EventLog};
use astrum_sim::pilot::Pilot;
use astrum_sim::RunMode;
use bevy::math::DVec3;
use bevy::prelude::*;

/// Serialises JSBSim FDM construction across a test binary's parallel `#[test]`s.
/// JSBSim has process-global init state, so constructing two `FGFDMExec`s from two
/// threads at once is not safe; every test that stands up a real app takes this guard
/// as its first line and holds it for its whole duration.
static JSBSIM_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the process-wide JSBSim guard. Poison-tolerant so one failing test can't
/// wedge the rest.
pub fn fdm_guard() -> MutexGuard<'static, ()> {
    JSBSIM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub const DT: f64 = 1.0 / 120.0;

/// A race scenario at UMore Park: one F450, unpaced, `[sim]` / `[[aircraft]]` extras
/// and `[course]` body as given (TOML text, without the table headers).
pub fn scenario_toml(sim_extra: &str, aircraft_extra: &str, course_body: &str) -> String {
    scenario_toml_n(&[1], sim_extra, aircraft_extra, course_body)
}

/// [`scenario_toml`] with one F450 per id in `ids` (`aircraft_extra` goes on each).
pub fn scenario_toml_n(
    ids: &[u32],
    sim_extra: &str,
    aircraft_extra: &str,
    course_body: &str,
) -> String {
    let aircraft: String = ids
        .iter()
        .map(|id| {
            format!(
                r#"
[[aircraft]]
id = {id}
model = "F450"
lat_deg = 44.7258
lon_deg = -93.0759
alt_hae_m = 2.0
heading_deg = 0.0
tas_mps = 0.0
{aircraft_extra}
"#
            )
        })
        .collect();
    format!(
        r#"
[world]
origin_lat_deg = 44.7258
origin_lon_deg = -93.0759
origin_alt_hae_m = 0.0

[sim]
rate_hz = 120.0
speed = 0.0
seed = 1
{sim_extra}
{aircraft}
[course]
{course_body}
"#
    )
}

/// A flat course at `alt_m` — all gates at one height, with as much room below the
/// drone as a test needs (the default racing band is far too low to watch a drone
/// fall and recover without touching the ground).
pub fn flat_course(alt_m: f64, gates: u32) -> String {
    format!("gates = {gates}\nalt_min_m = {alt_m}\nalt_max_m = {alt_m}")
}

/// Two gates facing North, 20 m apart at 3 m — hand-built rather than sampled, so a
/// case's geometry is exactly what it says.
pub fn straight_course() -> Course {
    Course::from_gates(
        vec![
            Gate::new(0, DVec3::new(0.0, 0.0, 3.0), 0.0),
            Gate::new(1, DVec3::new(0.0, 20.0, 3.0), 0.0),
        ],
        0,
        CourseParams::default(),
    )
    .unwrap()
}

/// The race systems and one hand-placed drone (aircraft 1) at `start`: no JSBSim, no sim
/// chain. The run mode is `Batch`, so a decided race requests the exit like a binary
/// run does.
pub fn rig(course: Course, start: DVec3) -> App {
    rig_n(course, &[start])
}

/// [`rig`] with several drones, ids `1..=starts.len()`, at `starts`.
pub fn rig_n(course: Course, starts: &[DVec3]) -> App {
    let mut app = App::new();
    let params = SimParams {
        rate_hz: 120.0,
        dt: DT,
        max_time_s: 0.0,
        speed: 0.0,
        lease_ticks: None,
        lockstep: false,
        lockstep_timeout_s: 30.0,
    };
    app.insert_resource(params)
        .insert_resource(SimClock::default())
        .insert_resource(SimStatus::default())
        .insert_resource(EventLog::default())
        .insert_resource(RunMode::Batch)
        .insert_resource(RaceCourse(course))
        .insert_resource(Race::new(1..=starts.len() as u32))
        .add_systems(Update, (tick_systems(), publish_events).chain());
    for (id, start) in (1..).zip(starts) {
        let at = Enu { e: start.x, n: start.y, u: start.z };
        app.world_mut().spawn((
            Aircraft { id },
            AircraftState { pos_enu: at, prev_pos_enu: at, ..Default::default() },
            Pilot::new(false),
        ));
    }
    app
}

/// Move the rig's aircraft `id` to `pos` in the given attitude (every other drone
/// stays where it is) and run one tick, returning the events that tick produced. The
/// tick's step is from where the drone was.
pub fn place_id(
    app: &mut App,
    id: u32,
    pos: DVec3,
    phi: f64,
    theta: f64,
    psi: f64,
) -> Vec<EventKind> {
    let mut q = app.world_mut().query::<(&Aircraft, &mut AircraftState)>();
    for (ac, mut st) in q.iter_mut(app.world_mut()) {
        st.prev_pos_enu = st.pos_enu;
        if ac.id == id {
            st.pos_enu = Enu { e: pos.x, n: pos.y, u: pos.z };
            st.fdm.phi_rad = phi;
            st.fdm.theta_rad = theta;
            st.fdm.psi_rad = psi;
        }
    }
    app.update();
    app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()).collect()
}

/// [`place_id`] for aircraft 1.
pub fn place(app: &mut App, pos: DVec3, phi: f64, theta: f64, psi: f64) -> Vec<EventKind> {
    place_id(app, 1, pos, phi, theta, psi)
}

/// [`place`] for a level drone facing North, which is most cases.
pub fn fly_to(app: &mut App, pos: DVec3) -> Vec<EventKind> {
    place(app, pos, 0.0, 0.0, 0.0)
}

/// [`place_id`] for a level drone facing North.
pub fn fly_id_to(app: &mut App, id: u32, pos: DVec3) -> Vec<EventKind> {
    place_id(app, id, pos, 0.0, 0.0, 0.0)
}

/// Aircraft 1's progress (the only drone's, in a single-drone race).
pub fn progress(app: &App) -> RaceProgress {
    progress_of(app, 1)
}

pub fn progress_of(app: &App, id: u32) -> RaceProgress {
    *app.world().resource::<Race>().drone(id).expect("a racing drone")
}

/// Has the sim been told to freeze (`Dead`) aircraft `id`?
pub fn frozen(app: &mut App, id: u32) -> bool {
    let mut q = app.world_mut().query_filtered::<&Aircraft, With<Dead>>();
    q.iter(app.world()).any(|a| a.id == id)
}

pub fn exited(app: &App) -> bool {
    app.should_exit().is_some()
}

/// Normalised rate demand per rad/s (the F450's FCS: ~0.75 rad/s per unit).
const RATE_PER_UNIT: f64 = 0.75;
/// Collective that roughly hovers the F450 (the altitude loop trims the rest).
const HOVER_COLLECTIVE: f64 = 0.45;

/// Where gate 0 is and how fast to fly at it: enough to be a real pass, slow enough
/// that an untuned attitude loop stays inside the opening.
pub struct Target {
    pub east: f64,
    pub north: f64,
    pub up: f64,
    pub speed_mps: f64,
}

/// Altitude hold, heading hold north, a forward-speed loop through pitch and a
/// lateral loop through roll, with an integrator on the altitude. All state is in
/// `integral`, so the law is a deterministic function of the telemetry sequence.
pub fn law(t: &Telemetry, target: &Target, integral: &mut f64) -> MultirotorRaw {
    let (dt, e) = (1.0 / 120.0, t.pos_enu);
    let v = t.vel_enu;
    let a = t.attitude;

    let up_err = target.up - e.u;
    *integral = (*integral + up_err * dt).clamp(-0.1, 0.1);
    let tilt = (a.phi_rad.cos() * a.theta_rad.cos()).max(0.5);
    let collective =
        ((HOVER_COLLECTIVE + 0.12 * up_err + 0.5 * *integral - 0.10 * v.u) / tilt).clamp(0.0, 1.0);

    let v_des = (0.5 * (target.north - e.n)).clamp(0.0, target.speed_mps);
    let pitch_des = (-0.10 * (v_des - v.n)).clamp(-0.2, 0.2); // nose down to go north
    let roll_des = (0.10 * (target.east - e.e) - 0.15 * v.e).clamp(-0.1, 0.1);
    let yaw_err = (-a.psi_rad + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;

    let rate = |err: f64, kp: f64| (kp * err / RATE_PER_UNIT).clamp(-1.0, 1.0);
    MultirotorRaw {
        roll_rate: rate(roll_des - a.phi_rad, 4.0),
        pitch_rate: rate(pitch_des - a.theta_rad, 4.0),
        yaw_rate: rate(yaw_err, 2.0),
        collective,
    }
}

/// A scripted controller for aircraft `id` in its own thread: it answers every telemetry
/// tick with [`law`] until `stop` is set. Join the handle after setting it.
pub fn spawn_controller(
    session: &Session,
    stop: &Arc<AtomicBool>,
    id: u32,
    target: Target,
) -> JoinHandle<()> {
    let (session, stop) = (session.clone(), Arc::clone(stop));
    std::thread::spawn(move || {
        let mut sub = LatestTelemetry::subscribe(&session, Selection::Aircraft(id)).unwrap();
        let mut tx = CommandSender::new(&session, id).unwrap();
        let (mut last, mut integral) = (None, 0.0);
        while !stop.load(Ordering::Relaxed) {
            if sub.wait(Duration::from_millis(20)).is_err() {
                continue;
            }
            let t = sub.latest(id).unwrap();
            if last != Some(t.tick) {
                let cmd = law(&t, &target, &mut integral);
                tx.send(Control::Multirotor(cmd), false, t.tick).unwrap();
                last = Some(t.tick);
            }
        }
    })
}
