//! Shared fixtures for the `astrum-race` integration tests.
//!
//! Two kinds of app: a **real** one ([`race_app`], built with
//! [`astrum_race::build_race_app`] on a scenario TOML, which flies JSBSim), and a
//! **rig** ([`rig`]) that holds only the race's own tick systems and a drone that a
//! test places by hand, so the interesting inputs — a tick that steps clean over a
//! frame, a pass 20 cm off centre — can be had on demand.
#![allow(dead_code)] // each test binary uses only a subset of these helpers

use std::sync::{Mutex, MutexGuard};

use astrum_geo::Enu;
use astrum_proto::EventKind;
use astrum_race::components::{RaceCourse, RaceProgress};
use astrum_race::course::{Course, CourseParams};
use astrum_race::gate::Gate;
use astrum_race::systems::tick_systems;
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

[[aircraft]]
id = 1
model = "F450"
lat_deg = 44.7258
lon_deg = -93.0759
alt_hae_m = 2.0
heading_deg = 0.0
tas_mps = 0.0
{aircraft_extra}

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

/// The race systems and a hand-placed drone at `start`: no JSBSim, no sim chain. The
/// run mode is `Batch`, so a decided race requests the exit like a binary run does.
pub fn rig(course: Course, start: DVec3) -> App {
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
        .insert_resource(RaceProgress::new(1))
        .add_systems(Update, (tick_systems(), publish_events).chain());
    let at = Enu { e: start.x, n: start.y, u: start.z };
    app.world_mut().spawn((
        Aircraft { id: 1 },
        AircraftState { pos_enu: at, prev_pos_enu: at, ..Default::default() },
        Pilot::new(false),
    ));
    app
}

/// Move the rig's drone to `pos` in the given attitude and run one tick, returning
/// the events that tick produced. The tick's step is from where the drone was.
pub fn place(app: &mut App, pos: DVec3, phi: f64, theta: f64, psi: f64) -> Vec<EventKind> {
    let mut q = app.world_mut().query::<&mut AircraftState>();
    for mut st in q.iter_mut(app.world_mut()) {
        st.prev_pos_enu = st.pos_enu;
        st.pos_enu = Enu { e: pos.x, n: pos.y, u: pos.z };
        st.fdm.phi_rad = phi;
        st.fdm.theta_rad = theta;
        st.fdm.psi_rad = psi;
    }
    app.update();
    app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()).collect()
}

/// [`place`] for a level drone facing North, which is most cases.
pub fn fly_to(app: &mut App, pos: DVec3) -> Vec<EventKind> {
    place(app, pos, 0.0, 0.0, 0.0)
}

pub fn progress(app: &App) -> RaceProgress {
    *app.world().resource::<RaceProgress>()
}

pub fn exited(app: &App) -> bool {
    app.should_exit().is_some()
}
