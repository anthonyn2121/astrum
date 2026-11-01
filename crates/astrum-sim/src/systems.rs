//! Fleet spawn (startup) and the per-tick physics systems: stepping every FDM
//! (in parallel), the sim clock, progress reporting, pacing and termination.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use astrum_fdm::{prop, Controls, Fdm, FdmError, InitialConditions, TrimMode};
use astrum_geo::{Enu, Geodetic, LocalFrame, Ned};
use bevy::prelude::*;

use crate::combat::{Dead, Gun, Health, Team, Trigger};
use crate::components::{
    Aircraft, AircraftIndex, AircraftState, FdmInstance, SimClock, SimParams, SimStatus, StepShard,
    VehicleKind, WorldFrame,
};
use crate::pilot::{FixedWingGains, Pilot};
use crate::scenario::{AircraftClass, AircraftSpec, Scenario};
use crate::step_pool::ShardPool;
use crate::RunMode;

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
    rate_hz: f64,
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
    let pilot = Pilot {
        control_period_ticks: spec.control_period_ticks(rate_hz),
        ..Pilot::new(spec.lockstep)
    };
    (Team(spec.team), Health::full(spec.health), Trigger(false), pilot, armed)
}

/// Startup: one FDM-owning entity per scenario aircraft, dealt round-robin to
/// the [`StepPool`]'s shards in scenario order. Any failure is fatal.
pub fn spawn_fleet(
    mut commands: Commands,
    scenario: Res<Scenario>,
    params: Res<SimParams>,
    frame: Res<WorldFrame>,
    pool: Res<StepPool>,
    mode: Res<RunMode>,
    mut status: ResMut<SimStatus>,
) {
    let mut index = AircraftIndex::default();
    for (rank, spec) in scenario.aircraft.iter().enumerate() {
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
        let (team, health, trigger, pilot, armed) =
            initial_components(spec, &class, params.rate_hz);
        let mut ent = commands.spawn((
            Aircraft { id: spec.id },
            FdmInstance::new(fdm),
            StepShard(rank % pool.threads()),
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
    if *mode == RunMode::Server {
        println!(
            "[astrum-sim] spawned {} aircraft @ {:.1} Hz (dt={:.6}s)",
            scenario.aircraft.len(),
            params.rate_hz,
            params.dt
        );
    }
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
pub struct StepPool(Option<ShardPool>);

impl StepPool {
    /// `threads <= 1` steps on the schedule's thread.
    pub fn new(threads: usize) -> Self {
        Self((threads > 1).then(|| ShardPool::new(threads)))
    }

    /// Shards an aircraft can be assigned to ([`StepShard`]).
    pub fn threads(&self) -> usize {
        self.0.as_ref().map_or(1, ShardPool::threads)
    }
}

type StepItem<'a> = (&'a Aircraft, Mut<'a, FdmInstance>, &'a Pilot, Mut<'a, AircraftState>);

/// Write each living aircraft's FCS inputs, advance its FDM one step and read
/// back its state — on its [`StepShard`]'s worker when the scenario asks for
/// threads (instances are independent, so the result is identical for any
/// thread count). A dead aircraft is frozen.
pub fn step_fdm(
    frame: Res<WorldFrame>,
    pool: Res<StepPool>,
    mut status: ResMut<SimStatus>,
    mut q: Query<
        (&Aircraft, &StepShard, &mut FdmInstance, &Pilot, &mut AircraftState),
        Without<Dead>,
    >,
) {
    let frame = frame.0;
    let step = |(ac, inst, pilot, st): &mut StepItem| {
        advance(inst.fdm(), pilot.output, &frame, st).err().map(|e| (ac.id, e))
    };
    let mut failures: Vec<(u32, FdmError)> = match &pool.0 {
        None => {
            q.iter_mut().filter_map(|(ac, _, inst, p, st)| step(&mut (ac, inst, p, st))).collect()
        }
        Some(pool) => {
            // One contiguous run of items per shard (the query's order follows
            // archetypes, not shards), handed to the shard's worker.
            let mut items: Vec<(usize, StepItem)> =
                q.iter_mut().map(|(ac, shard, inst, p, st)| (shard.0, (ac, inst, p, st))).collect();
            items.sort_unstable_by_key(|(shard, _)| *shard);
            let mut shards: Vec<&mut [(usize, StepItem)]> = Vec::with_capacity(pool.threads());
            let mut rest = items.as_mut_slice();
            for shard in 0..pool.threads() {
                let n = rest.iter().take_while(|(s, _)| *s == shard).count();
                let (mine, tail) = std::mem::take(&mut rest).split_at_mut(n);
                shards.push(mine);
                rest = tail;
            }
            let failed = Mutex::new(Vec::new());
            pool.run(&mut shards, |_, shard| {
                for (_, item) in shard.iter_mut() {
                    if let Some(f) = step(item) {
                        failed.lock().unwrap_or_else(PoisonError::into_inner).push(f);
                    }
                }
            });
            failed.into_inner().unwrap_or_else(PoisonError::into_inner)
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

/// How often the binary reports progress (wall clock).
const REPORT_EVERY: Duration = Duration::from_secs(1);

/// About once per wall-clock second, report the achieved step rate and speed.
pub fn report_progress(
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mut diag: ResMut<Diag>,
    q: Query<(), With<Aircraft>>,
) {
    let now = Instant::now();
    let elapsed = now.duration_since(diag.last_log);
    if elapsed < REPORT_EVERY {
        return;
    }
    let hz = (clock.tick - diag.last_log_tick) as f64 / elapsed.as_secs_f64();
    let target =
        if params.speed > 0.0 { format!("{}×", params.speed) } else { "unpaced".to_owned() };
    println!(
        "[astrum-sim] tick {} | sim {:.1}s | {} aircraft | {:.0} Hz = {:.1}× real time (target {target})",
        clock.tick,
        clock.time_s(&params),
        q.iter().count(),
        hz,
        hz / params.rate_hz,
    );
    diag.last_log = now;
    diag.last_log_tick = clock.tick;
}

/// Wall-clock throttle (`sim.speed`). Bevy's runner sleep is a no-op under
/// `default-features = false`, so the loop is paced here.
#[derive(Resource)]
pub struct Pacer {
    /// Wall time per step, `dt / speed`; `None` runs unpaced.
    period: Option<Duration>,
    /// Wall-clock instant the next step should not begin before.
    next: Option<Instant>,
}

impl Pacer {
    /// `speed` as validated by the scenario: `0` or at least [`MIN_SPEED`].
    pub fn new(dt: f64, speed: f64) -> Self {
        Self { period: (speed > 0.0).then(|| Duration::from_secs_f64(dt / speed)), next: None }
    }
}

/// Slowest supported `sim.speed` other than `0` (unpaced).
pub const MIN_SPEED: f64 = 1e-3;

/// How far the pacer may fall behind its schedule (sleep overshoot, a slow
/// tick) and still catch up, so a high speed factor is met on average; beyond
/// this it re-baselines instead of bursting to recover the debt.
const MAX_PACE_LAG: Duration = Duration::from_millis(50);

/// Hold each step until its slot on the `dt / speed` schedule.
pub fn pace(mut pacer: ResMut<Pacer>) {
    let Some(period) = pacer.period else { return };
    let now = Instant::now();
    let slot = match pacer.next {
        Some(next) if now < next + MAX_PACE_LAG => next,
        _ => now, // first step, or too far behind: start a new schedule here
    };
    if now < slot {
        std::thread::sleep(slot - now);
    }
    pacer.next = Some(slot + period);
}

/// Exit once the configured simulated-time limit is reached (`max_time_s > 0`).
pub fn stop_at_max_time(
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mode: Res<RunMode>,
    mut exit: MessageWriter<AppExit>,
) {
    if params.max_time_s > 0.0 && clock.time_s(&params) >= params.max_time_s {
        if *mode == RunMode::Server {
            println!("[astrum-sim] reached max_time_s={:.1} — exiting", params.max_time_s);
        }
        exit.write(AppExit::Success);
    }
}

/// Set by the binary's SIGINT/SIGTERM handler: the run ends cleanly (bag
/// finished, exit 0) at the end of the current tick.
#[derive(Resource, Clone, Default)]
pub struct StopRequested(pub Arc<AtomicBool>);

impl StopRequested {
    pub fn requested(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Exit once a stop has been requested.
pub fn exit_on_signal(
    stop: Option<Res<StopRequested>>,
    mode: Res<RunMode>,
    mut exit: MessageWriter<AppExit>,
    mut reported: Local<bool>,
) {
    if stop.is_some_and(|s| s.requested()) {
        if !*reported && *mode == RunMode::Server {
            println!("[astrum-sim] stop requested — exiting");
            *reported = true;
        }
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
