//! ASTRUM simulation systems: fleet spawn (startup) and the fixed-step update
//! chain that drives every aircraft's flight dynamics and derives its ENU pose.

use astrum_fdm::{prop, Fdm, FT_TO_M};
use astrum_geo::Geodetic;
use bevy::prelude::*;
use std::time::{Duration, Instant};

use crate::components::{Aircraft, FdmInstance, GeoState, SimParams, VehicleKind, WorldFrame};
use crate::net::AutopilotState;
use crate::scenario::Scenario;

/// JSBSim data root (`aircraft/`, `engine/`, `systems/`).
fn jsbsim_root() -> String {
    std::env::var("JSBSIM_ROOT").unwrap_or_else(|_| "/opt/jsbsim/share/jsbsim".into())
}

/// Startup: instantiate one FDM-owning entity per scenario aircraft.
pub fn spawn_fleet(mut commands: Commands, scenario: Res<Scenario>, params: Res<SimParams>) {
    let root = jsbsim_root();
    let mut spawned = 0u32;

    for spec in &scenario.aircraft {
        let mut fdm = Fdm::new();
        fdm.set_root_dir(&root);
        if !fdm.load_model(&spec.model) {
            eprintln!(
                "[astrum-sim] aircraft {}: failed to load model '{}' — skipping",
                spec.id, spec.model
            );
            continue;
        }

        // Initial conditions (see astrum_fdm::prop).
        fdm.set(prop::IC_LAT_GEOD_DEG, spec.lat_deg);
        fdm.set(prop::IC_LON_GC_DEG, spec.lon_deg);
        fdm.set(prop::IC_H_SL_FT, spec.alt_ft);
        fdm.set(prop::IC_PSI_TRUE_DEG, spec.heading_deg);
        fdm.set(prop::IC_VC_KTS, spec.airspeed_kts);
        if !fdm.run_ic() {
            eprintln!("[astrum-sim] aircraft {}: RunIC failed — skipping", spec.id);
            continue;
        }
        // Start all engines running (airborne start). Without this the piston
        // engine is off and throttle commands are inert — a live flight
        // controller's throttle would do nothing. Electric multirotor motors
        // spin up the same way.
        fdm.set(prop::PROPULSION_SET_RUNNING, -1.0);

        // Multirotors need their stability-augmentation loop engaged, otherwise
        // roll/pitch/yaw commands have no effect (see prop::SCAS_ENGAGE).
        let kind = spec.resolved_kind();
        if kind == VehicleKind::Multirotor {
            fdm.set(prop::SCAS_ENGAGE, 1.0);
        }
        fdm.set_dt(params.dt);

        commands.spawn((
            Aircraft { id: spec.id },
            FdmInstance(fdm),
            kind,
            GeoState::default(),
            Transform::default(),
            AutopilotState::default(),
        ));
        spawned += 1;
    }

    println!(
        "[astrum-sim] spawned {spawned}/{} aircraft @ {:.1} Hz (dt={:.6}s)",
        scenario.aircraft.len(),
        params.rate_hz,
        params.dt,
    );
}

/// Advance every aircraft's flight dynamics by one fixed timestep.
pub fn step_fdm(mut q: Query<&mut FdmInstance>) {
    for mut inst in &mut q {
        inst.0.step();
    }
}

/// Read JSBSim state into [`GeoState`] and derive the Cartesian ENU [`Transform`].
pub fn sync_geo_transform(
    frame: Res<WorldFrame>,
    mut q: Query<(&FdmInstance, &mut GeoState, &mut Transform)>,
) {
    for (inst, mut gs, mut tf) in &mut q {
        let f = &inst.0;
        gs.lat_deg = f.get(prop::LAT_GEOD_DEG);
        gs.lon_deg = f.get(prop::LON_GC_DEG);
        gs.alt_m = f.get(prop::ALT_SL_M);
        gs.phi_rad = f.get(prop::PHI_RAD);
        gs.theta_rad = f.get(prop::THETA_RAD);
        gs.psi_rad = f.get(prop::PSI_RAD);
        gs.vel_north_mps = f.get(prop::V_NORTH_FPS) * FT_TO_M;
        gs.vel_east_mps = f.get(prop::V_EAST_FPS) * FT_TO_M;
        gs.vel_down_mps = f.get(prop::V_DOWN_FPS) * FT_TO_M;
        gs.airspeed_mps = f.get(prop::VT_FPS) * FT_TO_M;
        gs.sim_time_s = f.get(prop::SIM_TIME);

        let enu =
            frame.0.geodetic_to_enu(&Geodetic::from_degrees(gs.lat_deg, gs.lon_deg, gs.alt_m));
        tf.translation = Vec3::new(enu.e as f32, enu.n as f32, enu.u as f32);
        // Provisional: yaw about Up (+Z) from true heading. Full pitch/roll body
        // attitude is finalised with the replay renderer; GeoState carries the
        // exact euler angles for downstream (telemetry) use regardless.
        tf.rotation = Quat::from_rotation_z(-gs.psi_rad as f32);
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

/// Roughly once per simulated second, report achieved vs configured step rate.
pub fn report_progress(params: Res<SimParams>, mut diag: ResMut<Diag>, q: Query<&GeoState>) {
    diag.steps += 1;
    let interval = params.rate_hz.max(1.0) as u64;
    if !diag.steps.is_multiple_of(interval) {
        return;
    }

    let now = Instant::now();
    let secs = now.duration_since(diag.last_log).as_secs_f64();
    let did = diag.steps - diag.last_log_step;
    let achieved = if secs > 0.0 { did as f64 / secs } else { 0.0 };
    let sim_time = q.iter().next().map(|g| g.sim_time_s).unwrap_or(0.0);

    println!(
        "[astrum-sim] step {} | sim {:.1}s | {} aircraft | {:.1} Hz achieved (target {:.1})",
        diag.steps,
        sim_time,
        q.iter().count(),
        achieved,
        params.rate_hz,
    );

    diag.last_log = now;
    diag.last_log_step = diag.steps;
}

/// Real-time throttle. Bevy's `ScheduleRunnerPlugin` sleep is a no-op under
/// `default-features = false` (its `no_std` fallback returns immediately), so we
/// pace the loop ourselves with `std::thread::sleep`.
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

/// Sleep so consecutive steps are `dt` apart (≈ real-time). When physics can't
/// keep up (`now >= target`), rebaseline instead of accumulating debt — the sim
/// simply runs slower than real-time rather than spiralling.
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

/// Exit once the configured simulated-time limit is reached (`max_time_s > 0`).
pub fn stop_at_max_time(
    params: Res<SimParams>,
    q: Query<&GeoState>,
    mut exit: MessageWriter<AppExit>,
) {
    if params.max_time_s <= 0.0 {
        return;
    }
    if let Some(g) = q.iter().next() {
        if g.sim_time_s >= params.max_time_s {
            println!(
                "[astrum-sim] reached max_time_s={:.1} (sim {:.2}s) — exiting",
                params.max_time_s, g.sim_time_s
            );
            exit.write(AppExit::Success);
        }
    }
}
