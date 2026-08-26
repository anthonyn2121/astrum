//! Embedded reinforcement-learning environment seam (design doc §8).
//!
//! [`SimEnv`] wraps [`build_app`](crate::build_app) + [`App::update`] in a plain,
//! **multi-agent** `reset` / `step` / `observe` API — the layer a Python binding
//! (`crates/astrum-env`) marshals with zero Bevy knowledge. It carries no pyo3
//! dependency on purpose, so it stays exercisable from `cargo test` and reusable
//! straight from Rust (e.g. scripted-opponent rollouts or offline dataset bakes).
//!
//! # One `update()` == one tick
//! Recall the sim has no `FixedUpdate`: exactly one physics step of `dt = 1/rate_hz`
//! runs per [`App::update`]. So [`SimEnv::step`] is a single deterministic tick —
//! apply this tick's actions, advance once, read the resulting state back.
//!
//! # What's in scope here (and what isn't)
//! The env surfaces raw *state* — geometry, kinematics, HP, ammo — and the two
//! episode flags (`terminated` / `truncated`). It deliberately computes **no
//! reward** and does no obs normalisation or relative-geometry framing: those are
//! policy/trainer concerns that live in Python (design doc §8), computed from the
//! [`AgentObs`] fields. Keeping this layer a pure state in / state out mapping
//! keeps it reusable across reward designs.
//!
//! # Parallelism model (hard constraint)
//! JSBSim is process-global and not thread-safe (design doc §2.3/§8), so a single
//! [`SimEnv`] owns one [`App`] on one thread and MUST NOT be shared across threads.
//! Scale training with **separate processes** (Gymnasium `AsyncVectorEnv` /
//! Stable-Baselines3 `SubprocVecEnv`), one `SimEnv` per worker — never OS threads.
//! *Within* a process the env is inherently multi-agent (PettingZoo-style): the N
//! aircraft of a scenario step together on one FDM-sequential app, which is what
//! self-play / NvN needs. The Python binding enforces the single-thread rule at
//! runtime with `#[pyclass(unsendable)]`.

use astrum_fdm::prop;
use bevy::prelude::*;

use crate::build_app;
use crate::combat::components::{Dead, Gun, Health, Team, Trigger};
use crate::components::{Aircraft, FdmInstance, GeoState};
use crate::net::AutopilotState;
use crate::scenario::Scenario;

/// One agent's control input for a tick. The four continuous channels are the
/// normalised `fcs/*-cmd-norm` inputs (their *meaning* is vehicle-kind dependent —
/// aerodynamic surfaces for fixed-wing, body-rate + collective demands for a
/// multirotor — but the interface is identical). `fire` is the learned trigger
/// (design doc §8): the policy decides when to shoot, it is not auto-aimed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Action {
    pub aileron: f64,
    pub elevator: f64,
    pub rudder: f64,
    pub throttle: f64,
    pub fire: bool,
}

/// A single aircraft's observable state after a tick. Absolute/world-frame — the
/// trainer derives relative geometry, a gun solution, and normalisation from these
/// (design doc §8). Positions are ENU metres; velocity is local NED; attitude and
/// body rates are radians / rad·s⁻¹.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentObs {
    pub id: u32,
    pub team: u8,
    /// `false` once the aircraft has been marked [`Dead`] (its FDM is frozen).
    pub alive: bool,
    /// ENU position (m): east, north, up.
    pub pos_e: f64,
    pub pos_n: f64,
    pub pos_u: f64,
    /// Local NED velocity (m/s): north, east, down.
    pub vel_n: f64,
    pub vel_e: f64,
    pub vel_d: f64,
    /// Attitude (rad): roll φ, pitch θ, yaw ψ.
    pub phi: f64,
    pub theta: f64,
    pub psi: f64,
    /// Body angular rates (rad/s): roll p, pitch q, yaw r.
    pub p: f64,
    pub q: f64,
    pub r: f64,
    /// Angle of attack α (rad) and sideslip β (rad) — stall/energy state the
    /// trainer can't reconstruct from kinematics alone (design doc §8).
    pub alpha_rad: f64,
    pub beta_rad: f64,
    /// Normal load factor Nz (g); ≈ +1 in level flight, the corner-speed signal.
    pub nz_g: f64,
    /// True airspeed (m/s).
    pub airspeed_mps: f64,
    /// Altitude above the ellipsoid (m).
    pub alt_m: f64,
    /// Remaining hit points.
    pub health: f32,
    /// Rounds remaining; `0` for unarmed aircraft (no [`Gun`]).
    pub ammo: u16,
    /// JSBSim sim time at this observation (s).
    pub sim_time_s: f64,
}

/// The episode-boundary flags returned alongside each step, mirroring the
/// Gymnasium terminated/truncated split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    /// The engagement was *decided*: ≥ 2 teams started and ≤ 1 is still alive.
    /// Matches [`crate::combat::systems::end_on_elimination`]'s predicate.
    pub terminated: bool,
    /// The episode hit its time cap (`sim.max_time_s`). `false` when uncapped.
    pub truncated: bool,
    /// Sim time at the end of this step (s).
    pub sim_time_s: f64,
}

impl StepResult {
    /// Whether the episode is over for either reason.
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Number of distinct teams in a scenario — the value `build_app` captures into
/// `CombatState::initial_living_teams` to gate termination.
fn count_teams(scenario: &Scenario) -> u8 {
    let mut teams: Vec<u8> = scenario.aircraft.iter().map(|a| a.team).collect();
    teams.sort_unstable();
    teams.dedup();
    teams.len() as u8
}

/// Build a fresh app and pump one `update()` so `Startup` runs: `spawn_fleet`
/// creates the entities and the first physics step settles JSBSim's trim. (Bevy
/// runs the `Startup` schedule on the *first* `update()`, so there's no
/// entities-present state before it.)
fn fresh_app(scenario: &Scenario) -> App {
    let mut app = build_app(scenario.clone());
    app.update();
    app
}

/// An embedded, multi-agent RL environment over one ASTRUM scenario. See the
/// [module docs](self) for the parallelism constraint.
pub struct SimEnv {
    scenario: Scenario,
    app: App,
    /// Teams present at spawn; termination only fires when this is ≥ 2.
    initial_living_teams: u8,
    /// `sim.max_time_s`; `0` means uncapped (never truncates).
    max_time_s: f64,
}

impl SimEnv {
    /// Build an env from an in-memory scenario and put it in the reset state (the
    /// fleet spawned and settled by one tick — see [`SimEnv::reset`]).
    pub fn new(scenario: Scenario) -> Self {
        let initial_living_teams = count_teams(&scenario);
        let max_time_s = scenario.sim.max_time_s;
        let app = fresh_app(&scenario);
        SimEnv { scenario, app, initial_living_teams, max_time_s }
    }

    /// Build an env from a scenario TOML string (the natural entry point for the
    /// Python binding, which hands the config over as text).
    pub fn from_toml_str(toml: &str) -> Result<Self, String> {
        Ok(Self::new(Scenario::from_toml_str(toml)?))
    }

    /// Rebuild the world from the original scenario and return the initial
    /// observation. Fully deterministic — same scenario (and `sim.seed`) ⇒ same
    /// start. One settling tick has run (fleet spawned + JSBSim trim), so the
    /// initial obs is at `t = dt`, not `t = 0`.
    pub fn reset(&mut self) -> Vec<AgentObs> {
        self.app = fresh_app(&self.scenario);
        self.observe()
    }

    /// Advance exactly one tick. `actions` is a slice of `(aircraft_id, Action)`;
    /// each listed aircraft has its autopilot disengaged, its four `fcs/*-cmd-norm`
    /// inputs written, and its trigger set for this tick. Aircraft not in the list
    /// keep their previous commands (e.g. a scripted opponent driven elsewhere, or
    /// an aircraft you intend to leave under its last command). Returns the fresh
    /// observation and the episode flags.
    pub fn step(&mut self, actions: &[(u32, Action)]) -> (Vec<AgentObs>, StepResult) {
        self.apply_actions(actions);
        self.app.update();
        let obs = self.observe();
        let result = self.step_result();
        (obs, result)
    }

    /// Sorted ids of every aircraft in the world.
    pub fn agent_ids(&mut self) -> Vec<u32> {
        let mut q = self.app.world_mut().query::<&Aircraft>();
        let world = self.app.world();
        let mut ids: Vec<u32> = q.iter(world).map(|a| a.id).collect();
        ids.sort_unstable();
        ids
    }

    /// Observe every aircraft's current state, sorted by id (a stable per-agent
    /// order for the trainer's fixed-width observation tensor).
    // The observation genuinely reads eight components per aircraft (id, geo state,
    // ENU transform, team, health, optional gun, alive marker, FDM for body rates);
    // `type_complexity` is the usual Bevy-query false positive.
    #[allow(clippy::type_complexity)]
    pub fn observe(&mut self) -> Vec<AgentObs> {
        let mut q = self.app.world_mut().query::<(
            &Aircraft,
            &GeoState,
            &Transform,
            &Team,
            &Health,
            Option<&Gun>,
            Option<&Dead>,
            &FdmInstance,
        )>();
        let world = self.app.world();
        let mut obs: Vec<AgentObs> = q
            .iter(world)
            .map(|(ac, gs, tf, team, hp, gun, dead, inst)| {
                let f = &inst.0;
                AgentObs {
                    id: ac.id,
                    team: team.0,
                    alive: dead.is_none(),
                    pos_e: tf.translation.x as f64,
                    pos_n: tf.translation.y as f64,
                    pos_u: tf.translation.z as f64,
                    vel_n: gs.vel_north_mps,
                    vel_e: gs.vel_east_mps,
                    vel_d: gs.vel_down_mps,
                    phi: gs.phi_rad,
                    theta: gs.theta_rad,
                    psi: gs.psi_rad,
                    p: f.get(prop::P_RAD_S),
                    q: f.get(prop::Q_RAD_S),
                    r: f.get(prop::R_RAD_S),
                    alpha_rad: f.get(prop::ALPHA_RAD),
                    beta_rad: f.get(prop::BETA_RAD),
                    nz_g: f.get(prop::NZ_G),
                    airspeed_mps: gs.airspeed_mps,
                    alt_m: gs.alt_m,
                    health: hp.hp,
                    ammo: gun.map(|g| g.ammo).unwrap_or(0),
                    sim_time_s: gs.sim_time_s,
                }
            })
            .collect();
        obs.sort_unstable_by_key(|o| o.id);
        obs
    }

    /// Apply this tick's actions to the matching aircraft (see [`SimEnv::step`]).
    // Four `&mut` components in the query trips `type_complexity`; the tuple is
    // plain, so allow it.
    #[allow(clippy::type_complexity)]
    fn apply_actions(&mut self, actions: &[(u32, Action)]) {
        if actions.is_empty() {
            return;
        }
        let mut q =
            self.app
                .world_mut()
                .query::<(&Aircraft, &mut FdmInstance, &mut AutopilotState, &mut Trigger)>();
        let world = self.app.world_mut();
        for (ac, mut inst, mut ap, mut trig) in q.iter_mut(world) {
            if let Some((_, a)) = actions.iter().find(|(id, _)| *id == ac.id) {
                // Disengage the autopilot so `autopilot_step` won't overwrite the
                // raw inputs (mirrors the `Control::Raw` path in `ingest_commands`).
                ap.engaged = false;
                let f = &mut inst.0;
                f.set(prop::AILERON_CMD, a.aileron);
                f.set(prop::ELEVATOR_CMD, a.elevator);
                f.set(prop::RUDDER_CMD, a.rudder);
                f.set(prop::THROTTLE_CMD, a.throttle);
                trig.0 = a.fire;
            }
        }
    }

    /// The episode flags for the current world state.
    fn step_result(&mut self) -> StepResult {
        let sim_time_s = self.sim_time();
        let truncated = self.max_time_s > 0.0 && sim_time_s >= self.max_time_s;
        let terminated = self.initial_living_teams >= 2 && self.living_team_count() <= 1;
        StepResult { terminated, truncated, sim_time_s }
    }

    /// Number of teams with at least one living aircraft — the same predicate
    /// `end_on_elimination` uses to decide the engagement.
    fn living_team_count(&mut self) -> usize {
        let mut q = self.app.world_mut().query_filtered::<&Team, (With<Health>, Without<Dead>)>();
        let world = self.app.world();
        let mut teams: Vec<u8> = q.iter(world).map(|t| t.0).collect();
        teams.sort_unstable();
        teams.dedup();
        teams.len()
    }

    /// Current sim time (s). All aircraft step together, so this is the max over
    /// the fleet (a dead aircraft's clock is frozen at its time of death).
    fn sim_time(&mut self) -> f64 {
        let mut q = self.app.world_mut().query::<&GeoState>();
        let world = self.app.world();
        q.iter(world).map(|g| g.sim_time_s).fold(0.0_f64, f64::max)
    }
}
