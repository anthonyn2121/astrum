//! `astrum_env` — the Python binding for the embedded ASTRUM RL dogfight
//! environment (design doc §8b). A thin marshalling layer over the plain-Rust
//! [`astrum_sim::env::SimEnv`]: it owns one headless sim app and exposes a
//! multi-agent `reset` / `step` API to Python. All the ECS/JSBSim work lives in
//! `astrum-sim`; this file only converts types across the FFI boundary.
//!
//! # Usage (Python)
//! ```python
//! import astrum_env
//! env = astrum_env.DogfightEnv(scenario_toml)   # a scenario as a TOML string
//! obs = env.reset()                             # list[Observation], sorted by id
//! # actions: list of (aircraft_id, [aileron, elevator, rudder, throttle, trigger])
//! obs, info = env.step([(1, [0.2, -0.05, 0.0, 0.7, 1.0])])
//! if info.terminated or info.truncated:
//!     obs = env.reset()
//! ```
//!
//! # Parallelism (hard constraint)
//! JSBSim is process-global and not thread-safe, so [`DogfightEnv`] is
//! `#[pyclass(unsendable)]`: PyO3 pins it to the thread that created it and raises
//! if it's touched from another. Scale training with **separate processes**
//! (Gymnasium `AsyncVectorEnv` / SB3 `SubprocVecEnv`), one env per worker — never
//! threads. See [`astrum_sim::env`] for the full rationale.

// pyo3's `#[pymethods]` expands each `PyResult`-returning method into an FFI
// trampoline (in a module-level `const _: () = { … }` block) whose error path does
// an identity `PyErr: Into<PyErr>`. clippy flags that as `useless_conversion` but
// pins the diagnostic to our method signature, and the lint level resolves at the
// generated node's module scope — so it can only be silenced crate-wide, not per
// method. It's macro-generated code we don't own; allow it for the whole binding.
#![allow(clippy::useless_conversion)]

use astrum_sim::env::{Action, AgentObs, SimEnv, StepResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A single aircraft's observed state after a tick — a plain, read-only mirror of
/// [`AgentObs`] with every field exposed as a Python attribute. Positions are ENU
/// metres, velocity is local NED (m/s), attitude/rates are radians / rad·s⁻¹.
#[pyclass(get_all)]
#[derive(Clone, Copy)]
pub struct Observation {
    pub id: u32,
    pub team: u8,
    pub alive: bool,
    pub pos_e: f64,
    pub pos_n: f64,
    pub pos_u: f64,
    pub vel_n: f64,
    pub vel_e: f64,
    pub vel_d: f64,
    pub phi: f64,
    pub theta: f64,
    pub psi: f64,
    pub p: f64,
    pub q: f64,
    pub r: f64,
    pub alpha_rad: f64,
    pub beta_rad: f64,
    pub nz_g: f64,
    pub airspeed_mps: f64,
    pub alt_m: f64,
    pub health: f32,
    pub ammo: u16,
    pub sim_time_s: f64,
}

impl From<AgentObs> for Observation {
    fn from(o: AgentObs) -> Self {
        Observation {
            id: o.id,
            team: o.team,
            alive: o.alive,
            pos_e: o.pos_e,
            pos_n: o.pos_n,
            pos_u: o.pos_u,
            vel_n: o.vel_n,
            vel_e: o.vel_e,
            vel_d: o.vel_d,
            phi: o.phi,
            theta: o.theta,
            psi: o.psi,
            p: o.p,
            q: o.q,
            r: o.r,
            alpha_rad: o.alpha_rad,
            beta_rad: o.beta_rad,
            nz_g: o.nz_g,
            airspeed_mps: o.airspeed_mps,
            alt_m: o.alt_m,
            health: o.health,
            ammo: o.ammo,
            sim_time_s: o.sim_time_s,
        }
    }
}

#[pymethods]
impl Observation {
    fn __repr__(&self) -> String {
        format!(
            "Observation(id={}, team={}, alive={}, pos=({:.1},{:.1},{:.1}), \
             airspeed={:.1}, alt={:.0}, aoa={:.1}deg, Nz={:.2}g, health={:.0}, \
             ammo={}, t={:.3})",
            self.id,
            self.team,
            self.alive,
            self.pos_e,
            self.pos_n,
            self.pos_u,
            self.airspeed_mps,
            self.alt_m,
            self.alpha_rad.to_degrees(),
            self.nz_g,
            self.health,
            self.ammo,
            self.sim_time_s,
        )
    }
}

/// The episode-boundary flags returned by [`DogfightEnv::step`], mirroring the
/// Gymnasium terminated/truncated split.
#[pyclass(get_all)]
#[derive(Clone, Copy)]
pub struct StepInfo {
    /// The engagement was decided (≥ 2 teams started, ≤ 1 remains alive).
    pub terminated: bool,
    /// The episode hit its `sim.max_time_s` cap.
    pub truncated: bool,
    /// Sim time at the end of the step (s).
    pub sim_time_s: f64,
}

impl From<StepResult> for StepInfo {
    fn from(r: StepResult) -> Self {
        StepInfo { terminated: r.terminated, truncated: r.truncated, sim_time_s: r.sim_time_s }
    }
}

#[pymethods]
impl StepInfo {
    /// Whether the episode is over for either reason.
    fn done(&self) -> bool {
        self.terminated || self.truncated
    }

    fn __repr__(&self) -> String {
        format!(
            "StepInfo(terminated={}, truncated={}, sim_time_s={:.3})",
            self.terminated, self.truncated, self.sim_time_s
        )
    }
}

/// Number of continuous+trigger channels in one agent's action vector:
/// `[aileron, elevator, rudder, throttle, trigger]` (design doc §8).
const ACTION_LEN: usize = 5;

/// Parse one agent's action vector into a plain [`Action`]. The four continuous
/// channels are clamped to their valid ranges (surfaces `[-1, 1]`, throttle
/// `[0, 1]`) so a wild policy output can't feed nonsense into JSBSim; the trigger
/// (5th channel) is gated at `>= 0.5` into the learned fire bool.
fn parse_action(v: &[f64]) -> PyResult<Action> {
    if v.len() != ACTION_LEN {
        return Err(PyValueError::new_err(format!(
            "each action must have {ACTION_LEN} elements \
             [aileron, elevator, rudder, throttle, trigger], got {}",
            v.len()
        )));
    }
    Ok(Action {
        aileron: v[0].clamp(-1.0, 1.0),
        elevator: v[1].clamp(-1.0, 1.0),
        rudder: v[2].clamp(-1.0, 1.0),
        throttle: v[3].clamp(0.0, 1.0),
        fire: v[4] >= 0.5,
    })
}

/// An embedded, multi-agent ASTRUM dogfight environment. One instance owns one
/// headless sim; step it once per tick. See the [module docs](self) for the
/// action/observation layout and the single-thread constraint.
#[pyclass(unsendable)]
pub struct DogfightEnv {
    inner: SimEnv,
}

#[pymethods]
impl DogfightEnv {
    /// Build an environment from a scenario given as a TOML string (same schema as
    /// `config/scenario.toml`). Raises `ValueError` if the TOML is invalid.
    #[new]
    fn new(scenario_toml: &str) -> PyResult<Self> {
        let inner = SimEnv::from_toml_str(scenario_toml).map_err(PyValueError::new_err)?;
        Ok(DogfightEnv { inner })
    }

    /// Sorted ids of every aircraft in the scenario (the stable per-agent order of
    /// every observation list).
    fn agent_ids(&mut self) -> Vec<u32> {
        self.inner.agent_ids()
    }

    /// Reset to the scenario's initial conditions and return the initial
    /// observation (one [`Observation`] per aircraft, sorted by id). Deterministic
    /// for a fixed scenario + `sim.seed`.
    fn reset(&mut self) -> Vec<Observation> {
        self.inner.reset().into_iter().map(Observation::from).collect()
    }

    /// Advance exactly one tick. `actions` is a list of `(aircraft_id, action)`
    /// where `action` is `[aileron, elevator, rudder, throttle, trigger]`. Aircraft
    /// omitted from the list keep their previous commands. Returns
    /// `(observations, StepInfo)`. Raises `ValueError` on a malformed action.
    fn step(&mut self, actions: Vec<(u32, Vec<f64>)>) -> PyResult<(Vec<Observation>, StepInfo)> {
        let mut parsed: Vec<(u32, Action)> = Vec::with_capacity(actions.len());
        for (id, v) in &actions {
            parsed.push((*id, parse_action(v)?));
        }
        let (obs, result) = self.inner.step(&parsed);
        let obs = obs.into_iter().map(Observation::from).collect();
        Ok((obs, StepInfo::from(result)))
    }
}

/// The `astrum_env` Python module.
#[pymodule]
fn astrum_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DogfightEnv>()?;
    m.add_class::<Observation>()?;
    m.add_class::<StepInfo>()?;
    Ok(())
}
