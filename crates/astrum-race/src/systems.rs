//! The racing environment's systems: course and drone spawn (startup), then the
//! fixed-step chain that flies the drone, derives its ENU pose, and scores gates.

use astrum_fdm::{prop, Fdm, FT_TO_M};
use astrum_geo::{Enu, Geodetic};
use bevy::math::DVec3;
use bevy::prelude::*;
use std::time::{Duration, Instant};

use crate::airframe;
use crate::components::{
    Drone, DroneState, FdmInstance, Outcome, RaceCourse, RaceProgress, SimParams, WorldFrame,
};
use crate::control::{FlightCommand, InnerLoop, RawOverride};
use crate::gate::Gate;
use crate::scenario::RaceScenario;

/// How close to a gate a plane crossing has to be before it counts as a *miss*
/// rather than as unrelated flying. A gate's plane is mathematically infinite;
/// clipping the tally to a few gate-widths keeps "went round the outside" (which is
/// what a pilot would call a miss) distinct from "flew past 30 m away".
const MISS_RADIUS_M: f64 = 3.0;

/// JSBSim data root (`aircraft/`, `engine/`, `systems/`).
fn jsbsim_root() -> String {
    std::env::var("JSBSIM_ROOT").unwrap_or_else(|_| "/opt/jsbsim/share/jsbsim".into())
}

/// Startup: give every gate in the generated course an entity.
///
/// The course itself is already resolved into the [`RaceCourse`] resource by
/// [`crate::build_app`] — generation is fallible, so it happens before the app is
/// standing, not inside a system that could only log its failure.
pub fn spawn_course(mut commands: Commands, course: Res<RaceCourse>) {
    for gate in &course.0.gates {
        commands.spawn((*gate, gate.transform()));
    }
    println!("[astrum-race] course: {}", course.0.summary());
}

/// Startup: put the F450 on the approach to the first gate and start its motors.
///
/// Every bail-out here ends the run as [`Outcome::SpawnFailed`]. Without that the app
/// spins forever: with no drone there is no [`DroneState`], so [`enforce_time_limit`]
/// has nothing to read a clock off and no terminator can ever fire.
pub fn spawn_drone(
    mut commands: Commands,
    scenario: Res<RaceScenario>,
    course: Res<RaceCourse>,
    frame: Res<WorldFrame>,
    params: Res<SimParams>,
    mut progress: ResMut<RaceProgress>,
) {
    let spec = &scenario.drone;
    let Some(first) = course.0.gates.first() else {
        eprintln!("[astrum-race] course has no gates — nothing to launch towards");
        progress.finish(Outcome::SpawnFailed);
        return;
    };

    // Start on the first gate's centreline, upstream of it: whatever the seed drew,
    // the drone begins lined up on gate 1.
    let start = first.center - spec.start_behind_m * first.normal();
    let start = DVec3::new(start.x, start.y, spec.start_alt_m.unwrap_or(start.z));
    let geo = frame.0.enu_to_geodetic(&Enu { e: start.x, n: start.y, u: start.z });

    let mut fdm = Fdm::new();
    fdm.set_root_dir(&jsbsim_root());
    if !fdm.load_model(&spec.model) {
        eprintln!(
            "[astrum-race] drone {}: failed to load model '{}' — not spawning",
            spec.id, spec.model
        );
        progress.finish(Outcome::SpawnFailed);
        return;
    }

    fdm.set(prop::IC_LAT_GEOD_DEG, geo.lat_deg());
    fdm.set(prop::IC_LON_GC_DEG, geo.lon_deg());
    fdm.set(prop::IC_H_SL_FT, geo.alt_m / FT_TO_M);
    fdm.set(prop::IC_PSI_TRUE_DEG, first.yaw_rad.to_degrees());
    fdm.set(prop::IC_VC_KTS, 0.0); // a quad initialises at hover, not cruise
    if !fdm.run_ic() {
        eprintln!("[astrum-race] drone {}: RunIC failed — not spawning", spec.id);
        progress.finish(Outcome::SpawnFailed);
        return;
    }
    // Electric motors have to be told to spin up, and a multirotor's roll/pitch/yaw
    // commands are inert until its stability-augmentation loop is engaged.
    fdm.set(prop::PROPULSION_SET_RUNNING, -1.0);
    fdm.set(prop::SCAS_ENGAGE, 1.0);
    fdm.set_dt(params.dt);

    commands.spawn((
        Drone { id: spec.id },
        FdmInstance(fdm),
        DroneState::default(),
        Transform::default(),
        // Launch holding station on gate 0's line. An outer controller overwrites
        // this to fly the course; until one does, the drone stays put instead of
        // falling out of the sky.
        FlightCommand::hold(start.z, first.yaw_rad),
        InnerLoop::default(),
        // Inert until a `Control::Raw` command arrives off the wire.
        RawOverride::default(),
    ));

    println!(
        "[astrum-race] drone {} ({}) launched {:.1} m upstream of gate 0 at {:.1} m, \
         heading {:.0} deg @ {:.1} Hz (dt={:.6}s)",
        spec.id,
        spec.model,
        spec.start_behind_m,
        start.z,
        first.yaw_rad.to_degrees(),
        params.rate_hz,
        params.dt,
    );
}

/// Advance the drone's flight dynamics by one fixed timestep.
pub fn step_fdm(mut q: Query<&mut FdmInstance>) {
    for mut inst in &mut q {
        inst.0.step();
    }
}

/// Read JSBSim state into [`DroneState`] and derive the ENU [`Transform`].
pub fn sync_pose(
    frame: Res<WorldFrame>,
    mut q: Query<(&FdmInstance, &mut DroneState, &mut Transform)>,
) {
    for (inst, mut st, mut tf) in &mut q {
        let f = &inst.0;
        st.lat_deg = f.get(prop::LAT_GEOD_DEG);
        st.lon_deg = f.get(prop::LON_GC_DEG);
        st.alt_m = f.get(prop::ALT_SL_M);
        st.phi_rad = f.get(prop::PHI_RAD);
        st.theta_rad = f.get(prop::THETA_RAD);
        st.psi_rad = f.get(prop::PSI_RAD);
        st.airspeed_mps = f.get(prop::VT_FPS) * FT_TO_M;
        st.sim_time_s = f.get(prop::SIM_TIME);
        // NED velocity → ENU.
        st.vel_enu = DVec3::new(
            f.get(prop::V_EAST_FPS) * FT_TO_M,
            f.get(prop::V_NORTH_FPS) * FT_TO_M,
            -f.get(prop::V_DOWN_FPS) * FT_TO_M,
        );

        let enu =
            frame.0.geodetic_to_enu(&Geodetic::from_degrees(st.lat_deg, st.lon_deg, st.alt_m));
        st.pos_enu = DVec3::new(enu.e, enu.n, enu.u);
        tf.translation = st.pos_enu.as_vec3();
        tf.rotation = Quat::from_rotation_z(-st.psi_rad as f32);
    }
}

/// End the run if the drone hit something: the ground, or a gate's frame.
///
/// # What counts
///
/// The airframe is the five-part hull [`airframe::Hull`] builds out of the F450's own
/// JSBSim model — four 9.4 in rotor discs on 0.1651 m arms, over a centre section that
/// runs from the gear feet to the rotor plane.
///
/// * **Ground.** Any contact at all. The racing floor is the ENU origin plane, which
///   is also JSBSim's terrain in the shipped scenario (`world.origin_alt_m = 0`), so
///   `min_z() <= 0` fires exactly as the gear feet touch down and needs no tolerance
///   constant. Touching down gently is still a crash: this is a race, and a drone that
///   put its legs on the ground is out of it. (An impact-speed threshold would be a
///   one-line change here, but it would also be a number to tune and a way to game the
///   ending.)
/// * **Gate frame.** The two uprights and the top bar of *any* gate, not just the
///   pending one — you can very easily clip the gate you have already passed, or the
///   next one round a tight corner. The opening is a hole and nothing hangs below the
///   sill, so flying through or underneath a gate touches nothing.
///
/// # Why it samples the step rather than the tick
///
/// One test per tick is not enough: at 20 m/s a 120 Hz step covers 0.17 m, more than
/// the 0.0762 m of frame stock, so the drone would teleport through gates. The usual
/// fix is to inflate the hull to cover the whole step, but that trades away exactly
/// what this environment cannot spare — the clearance is 4.1 cm at the worst heading,
/// and the inflation at 20 m/s would be 8 cm. So the step is subdivided instead and the true
/// hull tested at each sample, at most [`airframe::SAMPLE_STEP_M`] apart. Attitude is
/// held across the samples; at 120 Hz it moves a degree or two.
///
/// Runs before `track_gates` on purpose: see [`RaceProgress::finish`].
pub fn detect_crash(
    course: Res<RaceCourse>,
    mut progress: ResMut<RaceProgress>,
    mut prev_pos: Local<Option<DVec3>>,
    q: Query<&DroneState>,
) {
    if progress.finished {
        return;
    }
    let Some(st) = q.iter().next() else { return };

    // The first tick has nowhere to have come from, so it samples its own pose.
    let from = prev_pos.replace(st.pos_enu).unwrap_or(st.pos_enu);
    let travel = st.pos_enu - from;
    let samples = (travel.length() / airframe::SAMPLE_STEP_M).ceil().max(1.0) as u32;

    for k in 1..=samples {
        let at = from + travel * (f64::from(k) / f64::from(samples));
        let hull = airframe::Hull::at(at, st.phi_rad, st.theta_rad, st.psi_rad);

        if hull.min_z() <= 0.0 {
            progress.finish(Outcome::CrashedGround);
            println!(
                "[astrum-race] CRASH: ground contact at sim {:.2}s, {:.1} m/s \
                 ({:.1} m/s vertical), {} gates passed",
                st.sim_time_s,
                st.vel_enu.length(),
                st.vel_enu.z,
                progress.passed,
            );
            return;
        }

        if let Some(gate) = course.0.gates.iter().find(|g| hull.hits_gate(g)) {
            progress.finish(Outcome::CrashedGate);
            println!(
                "[astrum-race] CRASH: struck gate {}'s frame at sim {:.2}s, {:.1} m/s, \
                 {} gates passed",
                gate.index,
                st.sim_time_s,
                st.vel_enu.length(),
                progress.passed,
            );
            return;
        }
    }
}

/// Score the pending gate: did this tick's motion carry the drone through it?
///
/// Works on the *segment* between the previous and current position, expressed in
/// the gate's local frame. At racing speed a 120 Hz tick covers metres, so a
/// point-inside test against a 7 cm-thick frame would step clean over the gate;
/// instead we look for the sign change of the along-normal coordinate and check
/// where the crossing landed. Gates must be taken in order, downstream.
pub fn track_gates(
    course: Res<RaceCourse>,
    mut progress: ResMut<RaceProgress>,
    q: Query<&DroneState>,
) {
    if progress.finished {
        return;
    }
    let Some(st) = q.iter().next() else { return };
    let Some(gate) = course.0.gates.get(progress.next_gate as usize) else {
        return;
    };

    let local = gate.to_local(st.pos_enu);
    let Some(prev) = progress.last_local else {
        // First tick for this gate: establish the reference, score nothing.
        progress.last_local = Some(local);
        return;
    };
    progress.last_local = Some(local);

    // Downstream crossing of the gate plane only.
    if !(prev.z < 0.0 && local.z >= 0.0) {
        return;
    }
    // Where in the plane the straight-line path pierced it.
    let t = (-prev.z / (local.z - prev.z)).clamp(0.0, 1.0);
    let hit = prev + t * (local - prev);

    if gate.opening_contains(hit) {
        progress.passed += 1;
        progress.last_pass_s = st.sim_time_s;
        progress.next_gate += 1;
        progress.last_local = None; // the next gate's frame is a different basis
        if progress.next_gate as usize >= course.0.gates.len() {
            progress.finish(Outcome::Complete);
        }
        println!(
            "[astrum-race] gate {}/{} passed at sim {:.2}s ({:.2} m/s, {:.2} m from centre)",
            progress.passed,
            course.0.gates.len(),
            st.sim_time_s,
            st.vel_enu.length(),
            hit.x.hypot(hit.y),
        );
        if progress.finished {
            println!(
                "[astrum-race] course complete: {} gates in {:.2}s ({} missed crossings)",
                progress.passed, st.sim_time_s, progress.missed
            );
        }
    } else if hit.x.hypot(hit.y) <= MISS_RADIUS_M {
        // Crossed the gate's plane beside the hole: went round the outside.
        progress.missed += 1;
        println!(
            "[astrum-race] gate {} missed at sim {:.2}s ({:.2} m off centre)",
            progress.next_gate,
            st.sim_time_s,
            hit.x.hypot(hit.y),
        );
    }
}

/// Tracks wall-clock cadence so we can report the achieved step rate.
#[derive(Resource)]
pub struct Diag {
    steps: u64,
    last_log_step: u64,
    last_log: Instant,
}

impl Diag {
    pub fn new() -> Self {
        Self { steps: 0, last_log_step: 0, last_log: Instant::now() }
    }
}

impl Default for Diag {
    fn default() -> Self {
        Self::new()
    }
}

/// Roughly once per simulated second, report progress and achieved step rate.
pub fn report_progress(
    params: Res<SimParams>,
    course: Res<RaceCourse>,
    progress: Res<RaceProgress>,
    mut diag: ResMut<Diag>,
    q: Query<&DroneState>,
) {
    diag.steps += 1;
    let interval = params.rate_hz.max(1.0) as u64;
    if !diag.steps.is_multiple_of(interval) {
        return;
    }

    let now = Instant::now();
    let secs = now.duration_since(diag.last_log).as_secs_f64();
    let did = diag.steps - diag.last_log_step;
    let achieved = if secs > 0.0 { did as f64 / secs } else { 0.0 };
    let st = q.iter().next().copied().unwrap_or_default();
    // Distance to the gate the drone is working on — the one number that says
    // whether it is making progress at all.
    let to_gate = course
        .0
        .gates
        .get(progress.next_gate as usize)
        .map(|g: &Gate| (g.center - st.pos_enu).length())
        .unwrap_or(0.0);

    println!(
        "[astrum-race] step {} | sim {:.1}s | gate {}/{} | {:.1} m to go | {:.1} m/s | \
         alt {:.1} m | {:.1} Hz achieved (target {:.1})",
        diag.steps,
        st.sim_time_s,
        progress.passed,
        course.0.gates.len(),
        to_gate,
        st.vel_enu.length(),
        st.pos_enu.z,
        achieved,
        params.rate_hz,
    );

    diag.last_log = now;
    diag.last_log_step = diag.steps;
}

/// Real-time throttle. Bevy's `ScheduleRunnerPlugin` sleep is a no-op under
/// `default-features = false` (its `no_std` fallback returns immediately), so the
/// loop is paced here with `std::thread::sleep`.
#[derive(Resource)]
pub struct Pacer {
    dt: Duration,
    realtime: bool,
    /// Wall-clock instant the next step should not begin before.
    next: Option<Instant>,
}

impl Pacer {
    pub fn new(dt: f64, realtime: bool) -> Self {
        Self { dt: Duration::from_secs_f64(dt), realtime, next: None }
    }
}

/// Sleep so consecutive steps are `dt` apart (≈ real-time). When physics can't keep
/// up, rebaseline instead of accumulating debt — the sim runs slower than
/// real-time rather than spiralling.
pub fn pace(mut pacer: ResMut<Pacer>) {
    if !pacer.realtime {
        return;
    }
    let now = Instant::now();
    match pacer.next {
        Some(target) if now < target => {
            std::thread::sleep(target - now);
            pacer.next = Some(target + pacer.dt);
        }
        _ => {
            pacer.next = Some(now + pacer.dt);
        }
    }
}

/// Exit once the run is over — completed, crashed, timed out, or never launched.
///
/// The single exit path: every terminator marks [`RaceProgress::finished`] and this
/// turns that into an `AppExit`, so there is one place a run can end and one flag that
/// says it has.
pub fn stop_when_finished(progress: Res<RaceProgress>, mut exit: MessageWriter<AppExit>) {
    if progress.finished {
        exit.write(AppExit::Success);
    }
}

/// End the run once the configured simulated-time limit is reached (`max_time_s > 0`).
///
/// Marks the outcome rather than exiting, so it sits with the other terminal-state
/// writers ahead of `publish_race` and the wire always carries the outcome that ended
/// the run. [`stop_when_finished`] does the exiting.
pub fn enforce_time_limit(
    params: Res<SimParams>,
    mut progress: ResMut<RaceProgress>,
    q: Query<&DroneState>,
) {
    if params.max_time_s <= 0.0 || progress.finished {
        return;
    }
    if let Some(st) = q.iter().next() {
        if st.sim_time_s >= params.max_time_s && progress.finish(Outcome::TimedOut) {
            println!(
                "[astrum-race] reached max_time_s={:.1} (sim {:.2}s) with {} gates passed — exiting",
                params.max_time_s, st.sim_time_s, progress.passed
            );
        }
    }
}
