//! The embedded, multi-agent environment seam (design doc §8): `reset` / `step`
//! / `observe` over one headless sim, with no network. The PyO3 binding
//! (`crates/astrum-env`) marshals this API; it is also usable straight from Rust.
//!
//! * Observations are the sim's own [`Telemetry`] samples — exactly what the
//!   Zenoh bridge publishes — sorted by aircraft id.
//! * Actions are commands applied through the same [`apply_command`] the bridge
//!   uses (raw actuators, multirotor demands or autopilot targets, plus the
//!   trigger), all from one controller epoch ([`ENV_EPOCH`]), so the ownership,
//!   lease and failsafe rules are identical: an aircraft left out of `step`
//!   keeps its last command until the scenario's `command_lease_s` expires,
//!   then holds.
//! * One `step` is one physics tick. The first observation (after construction
//!   or `reset`) is at tick 1.
//! * [`SimEnv::events`] are the events of the latest tick (hits, kills, mode
//!   changes, ...), the same ones the bridge publishes on `astrum/events`.
//! * [`SimEnv`] is `Send`: environments may be stepped from different threads
//!   (JSBSim's construction race is handled inside `astrum-fdm`).

use astrum_geo::{Geodetic, LocalFrame};
use astrum_proto::{Command, Control, Event, Telemetry};
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;

use crate::combat::ballistics::DVec3;
use crate::combat::geometry::{body_to_grid, grid_to_body};
use crate::combat::systems::engagement_decided;
use crate::combat::{
    gun_solution, Bullet, CombatParams, CombatRng, CombatState, Dead, Gun, GunSolution, Team,
};
use crate::components::{
    AircraftIndex, AircraftState, FdmInstance, SimClock, SimParams, SimStatus, VehicleKind,
    WorldFrame,
};
use crate::events::EventLog;
use crate::net::{LastPublished, RefusalLog};
use crate::pilot::{apply_command, Pilot};
use crate::scenario::{Scenario, ScenarioError};
use crate::systems::{grid_state, initial_components, reset_fdm, Diag, Pacer};
use crate::{build_app, sim_params, team_count, world_frame, RunMode};

/// The controller epoch of every command the env applies (the env is the only
/// controller of its in-process sim).
pub const ENV_EPOCH: u64 = 1;

/// One agent's input for a tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Action {
    pub control: Control,
    /// Gun trigger (armed aircraft only).
    pub trigger: bool,
}

/// Episode flags after a step (Gymnasium's terminated/truncated split).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    /// The engagement was decided: ≥ 2 teams started and ≤ 1 still has a living aircraft.
    pub terminated: bool,
    /// Sim time reached `sim.max_time_s` (never when uncapped).
    pub truncated: bool,
    pub tick: u64,
    pub sim_time_s: f64,
}

impl StepResult {
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Why an env call failed.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvError {
    Scenario(ScenarioError),
    /// The sim hit a fatal error (an FDM failure); the env must be reset.
    Sim(String),
    UnknownAircraft(u32),
    /// The command was refused (e.g. a non-finite autopilot target).
    Refused {
        id: u32,
        reason: String,
    },
}

impl std::fmt::Display for EnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvError::Scenario(e) => write!(f, "{e}"),
            EnvError::Sim(e) => write!(f, "simulation failed: {e}"),
            EnvError::UnknownAircraft(id) => write!(f, "no aircraft {id}"),
            EnvError::Refused { id, reason } => {
                write!(f, "aircraft {id}: action refused: {reason}")
            }
        }
    }
}

impl std::error::Error for EnvError {}

pub type Result<T, E = EnvError> = std::result::Result<T, E>;

/// An embedded multi-agent environment over one scenario.
pub struct SimEnv {
    scenario: Scenario,
    world: World,
    seq: u64,
}

fn run_tick(world: &mut World) {
    world.run_schedule(Main.intern());
    world.clear_trackers();
}

fn fatal(world: &World) -> Option<String> {
    world.resource::<SimStatus>().fatal.clone()
}

/// Build a fresh world for `scenario` and run its first update (spawn + tick 1).
fn fresh_world(scenario: &Scenario) -> Result<World> {
    let mut app = build_app(scenario.clone(), RunMode::Embedded);
    app.finish();
    app.cleanup();
    app.update();
    let world = std::mem::take(app.world_mut());
    match fatal(&world) {
        Some(e) => Err(EnvError::Sim(e)),
        None => Ok(world),
    }
}

/// Whether `next` can reuse `current`'s loaded FDMs: same aircraft (ids, models,
/// classes, order), rate and thread count. ICs, teams, health, ammo, seed and
/// the world origin may all change.
fn same_fleet(current: &Scenario, next: &Scenario) -> bool {
    current.sim.rate_hz == next.sim.rate_hz
        && current.sim.threads == next.sim.threads
        && current.aircraft.len() == next.aircraft.len()
        && current.aircraft.iter().zip(&next.aircraft).all(|(a, b)| {
            a.id == b.id && a.model == b.model && a.class() == b.class() && a.lockstep == b.lockstep
        })
}

/// Return every aircraft and resource to `scenario`'s initial state without
/// reloading any model — equivalent to a fresh build, at a fraction of the cost.
fn reset_world(world: &mut World, scenario: &Scenario) -> Result<()> {
    let params = sim_params(scenario);
    let frame = world_frame(scenario);
    world.insert_resource(params);
    world.insert_resource(frame);
    world.insert_resource(SimClock::default());
    world.insert_resource(SimStatus::default());
    world.insert_resource(Diag::default());
    world.insert_resource(Pacer::new(params.dt, params.speed));
    world.insert_resource(CombatParams::default());
    world.insert_resource(CombatRng::seeded(scenario.sim.seed));
    world.insert_resource(CombatState { initial_living_teams: team_count(scenario) });
    world.insert_resource(LastPublished::default());
    world.insert_resource(RefusalLog::default());
    world.insert_resource(EventLog::default());
    world.insert_resource(scenario.clone());

    let bullets: Vec<Entity> = world.query_filtered::<Entity, With<Bullet>>().iter(world).collect();
    for b in bullets {
        world.despawn(b);
    }

    let index = world.resource::<AircraftIndex>().0.clone();
    for spec in &scenario.aircraft {
        let class = spec.class().map_err(EnvError::Scenario)?;
        let entity = index[&spec.id];
        let mut ent = world.entity_mut(entity);
        let mut inst = ent.get_mut::<FdmInstance>().expect("aircraft entities own an FDM");
        let fdm = inst.fdm();
        reset_fdm(fdm, spec, &class, scenario.sim.seed).map_err(EnvError::Sim)?;
        let state = grid_state(&frame.0, fdm, None).map_err(|e| EnvError::Sim(e.to_string()))?;
        let (team, health, trigger, pilot, armed) =
            initial_components(spec, &class, params.rate_hz);
        ent.insert((state, team, health, trigger, pilot));
        ent.remove::<Dead>();
        if let Some((gains, gun)) = armed {
            ent.insert((gains, gun));
        }
    }
    Ok(())
}

impl SimEnv {
    /// Build an env (validates the scenario) and run its first tick.
    pub fn new(scenario: Scenario) -> Result<Self> {
        scenario.validate().map_err(EnvError::Scenario)?;
        let world = fresh_world(&scenario)?;
        Ok(SimEnv { scenario, world, seq: 0 })
    }

    pub fn from_toml_str(toml: &str) -> Result<Self> {
        Self::new(Scenario::from_toml_str(toml).map_err(EnvError::Scenario)?)
    }

    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    /// The physics step (s).
    pub fn dt(&self) -> f64 {
        self.world.resource::<SimParams>().dt
    }

    /// The world grid observations are expressed in.
    pub fn frame(&self) -> LocalFrame {
        self.world.resource::<WorldFrame>().0
    }

    /// Sorted ids of every aircraft.
    pub fn agent_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.scenario.aircraft.iter().map(|a| a.id).collect();
        ids.sort_unstable();
        ids
    }

    /// Return to the initial state of `scenario` (default: the current one),
    /// optionally overriding its seed. Reuses the loaded FDMs when the fleet is
    /// unchanged; otherwise rebuilds. Deterministic: the same scenario and seed
    /// reproduce the same episode bit for bit.
    pub fn reset(
        &mut self,
        seed: Option<u64>,
        scenario: Option<Scenario>,
    ) -> Result<Vec<Telemetry>> {
        let mut next = scenario.unwrap_or_else(|| self.scenario.clone());
        if let Some(s) = seed {
            next.sim.seed = s;
        }
        next.validate().map_err(EnvError::Scenario)?;
        let reusable = same_fleet(&self.scenario, &next) && fatal(&self.world).is_none();
        if reusable {
            reset_world(&mut self.world, &next)?;
            run_tick(&mut self.world);
            if let Some(e) = fatal(&self.world) {
                return Err(EnvError::Sim(e));
            }
        } else {
            self.world = fresh_world(&next)?;
        }
        self.scenario = next;
        // Every pilot starts unowned with no seq: number commands afresh, so a
        // reset env reports the same command status as a fresh one.
        self.seq = 0;
        Ok(self.observe())
    }

    /// Apply this tick's actions and advance one tick.
    pub fn step(&mut self, actions: &[(u32, Action)]) -> Result<(Vec<Telemetry>, StepResult)> {
        if let Some(e) = fatal(&self.world) {
            return Err(EnvError::Sim(e));
        }
        let now = self.world.resource::<SimClock>().tick;
        let lease = self.world.resource::<SimParams>().lease_ticks;
        let index = self.world.resource::<AircraftIndex>().0.clone();
        for (id, a) in actions {
            let entity = *index.get(id).ok_or(EnvError::UnknownAircraft(*id))?;
            self.seq += 1;
            let cmd = Command {
                target_id: *id,
                seq: self.seq,
                telemetry_tick: now,
                trigger: a.trigger,
                epoch: ENV_EPOCH,
                control: a.control,
            };
            let mut ent = self.world.entity_mut(entity);
            if ent.contains::<Dead>() {
                continue; // a wreck takes no commands
            }
            let mut pilot = *ent.get::<Pilot>().expect("aircraft entities have a Pilot");
            let kind = *ent.get::<VehicleKind>().expect("aircraft entities have a VehicleKind");
            let mut trigger = ent.get::<crate::combat::Trigger>().copied();
            apply_command(&mut pilot, trigger.as_mut(), kind, &cmd, now, lease)
                .map_err(|e| EnvError::Refused { id: *id, reason: e.to_string() })?;
            ent.insert(pilot);
            if let Some(t) = trigger {
                ent.insert(t);
            }
        }
        run_tick(&mut self.world);
        if let Some(e) = fatal(&self.world) {
            return Err(EnvError::Sim(e));
        }
        Ok((self.observe(), self.step_result()))
    }

    /// The latest tick's telemetry, sorted by id.
    pub fn observe(&self) -> Vec<Telemetry> {
        self.world.resource::<LastPublished>().0.clone()
    }

    /// The events of the latest tick (`step`, or the first tick of `reset`), in
    /// the order they happened.
    pub fn events(&self) -> &[Event] {
        self.world.resource::<EventLog>().last()
    }

    fn step_result(&mut self) -> StepResult {
        let params = *self.world.resource::<SimParams>();
        let clock = *self.world.resource::<SimClock>();
        let state = *self.world.resource::<CombatState>();
        let mut living = self.world.query_filtered::<&Team, Without<Dead>>();
        let terminated = engagement_decided(&state, living.iter(&self.world));
        let sim_time_s = clock.time_s(&params);
        StepResult {
            terminated,
            truncated: params.max_time_s > 0.0 && sim_time_s >= params.max_time_s,
            tick: clock.tick,
            sim_time_s,
        }
    }

    fn state_of(&self, id: u32) -> Result<(Entity, AircraftState)> {
        let e = *self
            .world
            .resource::<AircraftIndex>()
            .0
            .get(&id)
            .ok_or(EnvError::UnknownAircraft(id))?;
        Ok((e, *self.world.get::<AircraftState>(e).expect("aircraft entities have a state")))
    }

    /// The ballistic fire-control solution of `shooter` against `target`, or
    /// `None` if the shooter is unarmed.
    pub fn gun_solution(&self, shooter: u32, target: u32) -> Result<Option<GunSolution>> {
        let (se, s) = self.state_of(shooter)?;
        let (_, t) = self.state_of(target)?;
        let Some(gun) = self.world.get::<Gun>(se) else { return Ok(None) };
        let combat = self.world.resource::<CombatParams>();
        Ok(Some(gun_solution(&self.frame(), &s, gun, combat, &t, self.dt())))
    }

    /// Rotate a body-axes vector of aircraft `id` into the world grid.
    pub fn body_to_grid(&self, id: u32, v: [f64; 3]) -> Result<[f64; 3]> {
        let (_, s) = self.state_of(id)?;
        let at = Geodetic::from_degrees(s.fdm.lat_deg, s.fdm.lon_deg, s.fdm.alt_hae_m);
        let r = body_to_grid(
            &self.frame(),
            &at,
            s.fdm.phi_rad,
            s.fdm.theta_rad,
            s.fdm.psi_rad,
            DVec3::from_array(v),
        );
        Ok(r.to_array())
    }

    /// Rotate a world-grid vector into the body axes of aircraft `id`.
    pub fn grid_to_body(&self, id: u32, v: [f64; 3]) -> Result<[f64; 3]> {
        let (_, s) = self.state_of(id)?;
        let at = Geodetic::from_degrees(s.fdm.lat_deg, s.fdm.lon_deg, s.fdm.alt_hae_m);
        let r = grid_to_body(
            &self.frame(),
            &at,
            s.fdm.phi_rad,
            s.fdm.theta_rad,
            s.fdm.psi_rad,
            DVec3::from_array(v),
        );
        Ok(r.to_array())
    }
}
