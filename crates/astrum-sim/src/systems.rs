//! Fleet spawn (startup) and the per-tick physics systems: stepping every FDM
//! (in parallel), the sim clock, progress reporting, pacing and termination.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use astrum_fdm::{prop, Controls, Fdm, FdmError, InitialConditions, TrimMode};
use astrum_geo::{Enu, Geodetic, LocalFrame, Ned};
use bevy::prelude::*;

use crate::combat::{Dead, Gun, Health, Team, Trigger};
use crate::components::{
    Aircraft, AircraftIndex, AircraftState, FdmInstance, SimClock, SimParams, SimStatus,
    VehicleKind, WorldFrame,
};
use crate::pilot::{FixedWingGains, Pilot};
use crate::scenario::{AircraftClass, AircraftSpec, Scenario};

/// JSBSim data root (`aircraft/`, `engine/`, `systems/`).
pub fn jsbsim_root() -> PathBuf {
    std::env::var_os("JSBSIM_ROOT").map(PathBuf::from).unwrap_or("/opt/jsbsim/share/jsbsim".into())
}

/// Per-aircraft JSBSim random seed derived from the scenario seed, so every
/// instance's noise is reproducible and distinct.
fn fdm_seed(scenario_seed: u64, id: u32) -> f64 {
    let mixed = scenario_seed ^ (u64::from(id).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (mixed % (1 << 31)) as f64
}

fn initial_conditions(spec: &AircraftSpec) -> InitialConditions {
    InitialConditions {
        lat_geod_deg: spec.lat_deg,
        lon_deg: spec.lon_deg,
        alt_hae_m: spec.alt_hae_m,
        psi_true_deg: spec.heading_deg,
        tas_mps: spec.tas_mps,
    }
}

/// Everything after the ICs are applied: engines on, the multirotor SCAS
/// engaged, the trim (if configured).
fn finish_init(fdm: &mut Fdm, class: &AircraftClass) -> Result<(), FdmError> {
    // An airborne start: without this a piston engine is off and throttle inert.
    fdm.start_engines()?;
    if class.kind == VehicleKind::Multirotor {
        // Without the stability augmentation the quad has no attitude control.
        fdm.set(prop::SCAS_ENGAGE, 1.0)?;
    }
    if class.trim {
        fdm.trim(TrimMode::Full)?;
    }
    Ok(())
}

/// Load, initialise and (optionally) trim one aircraft's FDM.
pub fn build_fdm(
    spec: &AircraftSpec,
    class: &AircraftClass,
    dt: f64,
    seed: u64,
) -> Result<Fdm, String> {
    let ctx = |e: FdmError| format!("aircraft {} ({}): {e}", spec.id, spec.model);
    let mut fdm = Fdm::load(&jsbsim_root(), &spec.model).map_err(ctx)?;
    fdm.set_dt(dt).map_err(ctx)?;
    fdm.set(prop::RANDOM_SEED, fdm_seed(seed, spec.id)).map_err(ctx)?;
    fdm.set_initial_conditions(&initial_conditions(spec)).map_err(ctx)?;
    fdm.run_ic().map_err(ctx)?;
    finish_init(&mut fdm, class).map_err(ctx)?;
    Ok(fdm)
}

/// Return an existing FDM to `spec`'s initial conditions without reloading the
/// model (`ResetToInitialConditions`), re-seeding and re-trimming exactly as
/// [`build_fdm`] does.
pub fn reset_fdm(
    fdm: &mut Fdm,
    spec: &AircraftSpec,
    class: &AircraftClass,
    seed: u64,
) -> Result<(), String> {
    let ctx = |e: FdmError| format!("aircraft {} ({}): {e}", spec.id, spec.model);
    fdm.set(prop::RANDOM_SEED, fdm_seed(seed, spec.id)).map_err(ctx)?;
    fdm.set_initial_conditions(&initial_conditions(spec)).map_err(ctx)?;
    fdm.reset().map_err(ctx)?;
    finish_init(fdm, class).map_err(ctx)
}

/// Map JSBSim's state onto the world grid.
pub fn grid_state(
    frame: &LocalFrame,
    fdm: &mut Fdm,
    prev_pos_enu: Option<Enu>,
) -> Result<AircraftState, FdmError> {
    let s = fdm.state()?;
    let at = Geodetic::from_degrees(s.lat_deg, s.lon_deg, s.alt_hae_m);
    let pos_enu = frame.geodetic_to_enu(&at);
    let vel_enu =
        frame.ned_to_enu(&at, &Ned { n: s.vel_north_mps, e: s.vel_east_mps, d: s.vel_down_mps });
    Ok(AircraftState { fdm: s, pos_enu, prev_pos_enu: prev_pos_enu.unwrap_or(pos_enu), vel_enu })
}

/// The combat/pilot components every aircraft carries, in their initial state.
pub fn initial_components(
    spec: &AircraftSpec,
    class: &AircraftClass,
) -> (Team, Health, Trigger, Pilot, Option<(FixedWingGains, Gun)>) {
    // Fixed-wing aircraft carry the gun and autopilot gains; multirotors are
    // unarmed and use the hover-hold law.
    let armed = class.profile.map(|p| {
        let mut gun = Gun::m61a1();
        if let Some(ammo) = spec.ammo {
            gun.ammo = ammo;
        }
        (FixedWingGains::for_profile(p), gun)
    });
    (Team(spec.team), Health::full(spec.health), Trigger(false), Pilot::new(spec.lockstep), armed)
}

/// Startup: one FDM-owning entity per scenario aircraft. Any failure is fatal.
pub fn spawn_fleet(
    mut commands: Commands,
    scenario: Res<Scenario>,
    params: Res<SimParams>,
    frame: Res<WorldFrame>,
    mut status: ResMut<SimStatus>,
) {
    let mut index = AircraftIndex::default();
    for spec in &scenario.aircraft {
        let class = match spec.class() {
            Ok(c) => c,
            Err(e) => {
                status.fatal = Some(e.to_string());
                return;
            }
        };
        let mut fdm = match build_fdm(spec, &class, params.dt, scenario.sim.seed) {
            Ok(f) => f,
            Err(e) => {
                status.fatal = Some(e);
                return;
            }
        };
        let state = match grid_state(&frame.0, &mut fdm, None) {
            Ok(s) => s,
            Err(e) => {
                status.fatal = Some(format!("aircraft {}: {e}", spec.id));
                return;
            }
        };
        let (team, health, trigger, pilot, armed) = initial_components(spec, &class);
        let mut ent = commands.spawn((
            Aircraft { id: spec.id },
            FdmInstance::new(fdm),
            class.kind,
            state,
            pilot,
            team,
            health,
            trigger,
        ));
        if let Some((gains, gun)) = armed {
            ent.insert((gains, gun));
        }
        index.0.insert(spec.id, ent.id());
    }
    println!(
        "[astrum-sim] spawned {} aircraft @ {:.1} Hz (dt={:.6}s)",
        scenario.aircraft.len(),
        params.rate_hz,
        params.dt
    );
    commands.insert_resource(index);
}

fn advance(
    fdm: &mut Fdm,
    output: Option<Controls>,
    frame: &LocalFrame,
    st: &mut AircraftState,
) -> Result<(), FdmError> {
    if let Some(c) = output {
        fdm.set_controls(c)?;
    }
    fdm.step()?;
    *st = grid_state(frame, fdm, Some(st.pos_enu))?;
    Ok(())
}

/// The threads that step the FDMs; `None` steps on the schedule's own thread.
/// Owned per app (not a process-global pool), so every sim or env in a process
/// gets exactly the parallelism its scenario asks for.
#[derive(Resource)]
pub struct StepPool(pub Option<rayon::ThreadPool>);

impl StepPool {
    pub fn new(threads: usize) -> Self {
        if threads <= 1 {
            return Self(None);
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("astrum-fdm-{i}"))
            .build()
            .expect("spawning the FDM stepping threads");
        Self(Some(pool))
    }
}

type StepItem<'a> = (&'a Aircraft, Mut<'a, FdmInstance>, &'a Pilot, Mut<'a, AircraftState>);

/// Write each living aircraft's FCS inputs, advance its FDM one step and read
/// back its state — in parallel across aircraft when the scenario asks for
/// threads (instances are independent, so the result is identical for any
/// thread count). A dead aircraft is frozen.
pub fn step_fdm(
    frame: Res<WorldFrame>,
    pool: Res<StepPool>,
    mut status: ResMut<SimStatus>,
    mut q: Query<(&Aircraft, &mut FdmInstance, &Pilot, &mut AircraftState), Without<Dead>>,
) {
    use rayon::prelude::*;
    let frame = frame.0;
    let step = |(ac, mut inst, pilot, mut st): StepItem| {
        advance(inst.fdm(), pilot.output, &frame, &mut st).err().map(|e| (ac.id, e))
    };
    let mut failures: Vec<(u32, FdmError)> = match &pool.0 {
        None => q.iter_mut().filter_map(step).collect(),
        Some(pool) => {
            let items: Vec<StepItem> = q.iter_mut().collect();
            pool.install(|| items.into_par_iter().filter_map(step).collect())
        }
    };
    if !failures.is_empty() {
        failures.sort_by_key(|(id, _)| *id);
        let msg: Vec<String> =
            failures.iter().map(|(id, e)| format!("aircraft {id}: {e}")).collect();
        status.fatal = Some(msg.join("; "));
    }
}

/// Dead aircraft keep their last pose; their "previous" position catches up so
/// a frozen wreck never sweeps a hit volume.
pub fn freeze_dead(mut q: Query<&mut AircraftState, With<Dead>>) {
    for mut st in &mut q {
        st.prev_pos_enu = st.pos_enu;
    }
}

/// One step completed.
pub fn advance_clock(mut clock: ResMut<SimClock>) {
    clock.tick += 1;
}

/// Tracks wall-clock cadence so we can report the achieved step rate.
#[derive(Resource)]
pub struct Diag {
    last_log_tick: u64,
    last_log: Instant,
}

impl Default for Diag {
    fn default() -> Self {
        Self { last_log_tick: 0, last_log: Instant::now() }
    }
}

/// Roughly once per simulated second, report achieved vs configured step rate.
pub fn report_progress(
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mut diag: ResMut<Diag>,
    q: Query<(), With<Aircraft>>,
) {
    let interval = params.rate_hz.round().max(1.0) as u64;
    if !clock.tick.is_multiple_of(interval) {
        return;
    }
    let now = Instant::now();
    let secs = now.duration_since(diag.last_log).as_secs_f64();
    let did = clock.tick - diag.last_log_tick;
    let achieved = if secs > 0.0 { did as f64 / secs } else { 0.0 };
    println!(
        "[astrum-sim] tick {} | sim {:.1}s | {} aircraft | {:.1} Hz achieved (target {:.1})",
        clock.tick,
        clock.time_s(&params),
        q.iter().count(),
        achieved,
        params.rate_hz,
    );
    diag.last_log = now;
    diag.last_log_tick = clock.tick;
}

/// Real-time throttle. Bevy's runner sleep is a no-op under
/// `default-features = false`, so the loop is paced here.
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

/// Sleep so consecutive steps are `dt` apart. When physics can't keep up,
/// rebaseline instead of accumulating debt.
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
        _ => pacer.next = Some(now + pacer.dt),
    }
}

/// Exit once the configured simulated-time limit is reached (`max_time_s > 0`).
pub fn stop_at_max_time(
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mut exit: MessageWriter<AppExit>,
) {
    if params.max_time_s > 0.0 && clock.time_s(&params) >= params.max_time_s {
        println!("[astrum-sim] reached max_time_s={:.1} — exiting", params.max_time_s);
        exit.write(AppExit::Success);
    }
}

/// Report a fatal error once and exit with a failure code.
pub fn exit_on_fatal(
    status: Res<SimStatus>,
    mut exit: MessageWriter<AppExit>,
    mut reported: Local<bool>,
) {
    if let Some(msg) = &status.fatal {
        if !*reported {
            eprintln!("[astrum-sim] fatal: {msg}");
            *reported = true;
        }
        exit.write(AppExit::from_code(1));
    }
}
