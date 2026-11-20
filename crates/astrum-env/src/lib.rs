//! `astrum_env` — Python bindings for the embedded ASTRUM multi-agent
//! environment ([`astrum_sim::env::SimEnv`]): a thin marshalling layer that moves
//! observations and actions as NumPy arrays and runs the sim with the GIL
//! released.
//!
//! ```python
//! import astrum_env, numpy as np
//! env = astrum_env.Env(scenario_toml)            # one sim; thread-safe to use from any thread
//! obs = env.reset(seed=3)                         # float64 (n_agents, len(OBS_FIELDS)), rows = env.agent_ids
//! act = np.zeros((len(env.agent_ids), 5))         # [aileron, elevator, rudder, throttle, trigger]
//! obs, info = env.step(act)                       # info.events: this tick's hits, kills, ...
//! batch = astrum_env.BatchEnv([scenario_toml] * 16)   # 16 sims stepped in parallel per call
//! ```
//!
//! Observation columns are `OBS_FIELDS`; enum columns use the codes in
//! `VEHICLE_KINDS` / `CONTROL_MODES`. Positions/velocities: world grid ENU and
//! local-level NED, SI units, radians. The first observation after construction
//! or `reset` is at tick 1.

use astrum_fdm::FdmState;
use astrum_proto::{
    AutopilotTargets, Control, ControlMode, Event, EventKind, MultirotorRaw, RawActuators,
    Telemetry, VehicleKind,
};
use astrum_sim::combat::{CombatParams, Gun};
use astrum_sim::components::AutopilotProfile;
use astrum_sim::env::{Action, EnvError, SimEnv, StepResult};
use astrum_sim::pilot::{self, FixedWingGains, Integrators};
use astrum_sim::scenario::Scenario;
use numpy::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

/// Observation matrix columns, in order.
const OBS_FIELDS: [&str; 33] = [
    "id",
    "team",
    "alive",
    "vehicle_kind",
    "control_mode",
    "tick",
    "sim_time_s",
    "lat_deg",
    "lon_deg",
    "alt_hae_m",
    "pos_e",
    "pos_n",
    "pos_u",
    "vel_e",
    "vel_n",
    "vel_u",
    "vel_ned_n",
    "vel_ned_e",
    "vel_ned_d",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
    "alpha",
    "beta",
    "nz",
    "mach",
    "tas",
    "hit_points",
    "hit_points_max",
    "ammo",
];
const N_OBS: usize = OBS_FIELDS.len();
/// `[aileron, elevator, rudder, throttle, trigger]`
const N_ACT: usize = 5;
/// `[engage, heading_rad, alt_hae_m, tas_mps]`
const N_AP: usize = 4;

const VEHICLE_KINDS: [&str; 2] = ["fixed-wing", "multirotor"];
const CONTROL_MODES: [&str; 4] = ["uncommanded", "raw", "autopilot", "failsafe"];

fn kind_code(k: VehicleKind) -> f64 {
    match k {
        VehicleKind::FixedWing => 0.0,
        VehicleKind::Multirotor => 1.0,
        VehicleKind::Unknown(n) => f64::from(n),
    }
}

fn mode_code(m: ControlMode) -> f64 {
    match m {
        ControlMode::Uncommanded => 0.0,
        ControlMode::Raw => 1.0,
        ControlMode::Autopilot => 2.0,
        ControlMode::Failsafe => 3.0,
        ControlMode::Unknown(n) => f64::from(n),
    }
}

fn obs_row(t: &Telemetry) -> [f64; N_OBS] {
    [
        f64::from(t.aircraft_id),
        f64::from(t.combat.team),
        if t.combat.alive { 1.0 } else { 0.0 },
        kind_code(t.vehicle_kind),
        mode_code(t.control_mode),
        t.tick as f64,
        t.sim_time_s,
        t.lat_deg,
        t.lon_deg,
        t.alt_hae_m,
        t.pos_enu.e,
        t.pos_enu.n,
        t.pos_enu.u,
        t.vel_enu.e,
        t.vel_enu.n,
        t.vel_enu.u,
        t.vel_ned.n,
        t.vel_ned.e,
        t.vel_ned.d,
        t.attitude.phi_rad,
        t.attitude.theta_rad,
        t.attitude.psi_rad,
        t.rates.p_rad_s,
        t.rates.q_rad_s,
        t.rates.r_rad_s,
        t.aero.alpha_rad,
        t.aero.beta_rad,
        t.aero.nz_g,
        t.aero.mach,
        t.tas_mps,
        f64::from(t.combat.hit_points),
        f64::from(t.combat.hit_points_max),
        f64::from(t.combat.ammo),
    ]
}

fn obs_matrix(obs: &[Telemetry]) -> Array2<f64> {
    let mut m = Array2::zeros((obs.len(), N_OBS));
    for (i, t) in obs.iter().enumerate() {
        m.row_mut(i).assign(&ArrayView1::from(&obs_row(t)));
    }
    m
}

fn col(name: &str) -> usize {
    OBS_FIELDS.iter().position(|f| *f == name).expect("known column")
}

/// The FDM-state view of one observation row (what the autopilot laws read).
fn fdm_state_of(row: ArrayView1<f64>) -> PyResult<FdmState> {
    if row.len() != N_OBS {
        return Err(PyValueError::new_err(format!(
            "observation row must have {N_OBS} columns, got {}",
            row.len()
        )));
    }
    let g = |n: &str| row[col(n)];
    Ok(FdmState {
        lat_deg: g("lat_deg"),
        lon_deg: g("lon_deg"),
        alt_hae_m: g("alt_hae_m"),
        phi_rad: g("phi"),
        theta_rad: g("theta"),
        psi_rad: g("psi"),
        vel_north_mps: g("vel_ned_n"),
        vel_east_mps: g("vel_ned_e"),
        vel_down_mps: g("vel_ned_d"),
        tas_mps: g("tas"),
        p_rad_s: g("p"),
        q_rad_s: g("q"),
        r_rad_s: g("r"),
        alpha_rad: g("alpha"),
        beta_rad: g("beta"),
        nz_g: g("nz"),
        mach: g("mach"),
        sim_time_s: g("sim_time_s"),
    })
}

fn env_err(e: EnvError) -> PyErr {
    match e {
        EnvError::Scenario(e) => PyValueError::new_err(e.to_string()),
        EnvError::Sim(e) => PyRuntimeError::new_err(e),
        EnvError::UnknownAircraft(id) => PyKeyError::new_err(id),
        e @ EnvError::Refused { .. } => PyValueError::new_err(e.to_string()),
    }
}

fn parse_scenario(toml: &str) -> PyResult<Scenario> {
    Scenario::from_toml_str(toml).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Each agent's id and vehicle class, sorted by id (the row order of every
/// observation and action array).
fn agents_of(env: &SimEnv) -> Vec<(u32, VehicleKind)> {
    env.observe().iter().map(|t| (t.aircraft_id, t.vehicle_kind)).collect()
}

/// Turn one env's action rows (aligned with its sorted agents) into commands.
/// The four raw channels are the class's raw arm: actuators on a fixed-wing,
/// rate demands and collective on a multirotor.
fn actions_for(
    agents: &[(u32, VehicleKind)],
    raw: ArrayView2<f64>,
    autopilot: Option<ArrayView2<f64>>,
    commanded: Option<ArrayView1<bool>>,
) -> Result<Vec<(u32, Action)>, String> {
    if raw.dim() != (agents.len(), N_ACT) {
        return Err(format!(
            "actions must have shape ({}, {N_ACT}), got {:?}",
            agents.len(),
            raw.dim()
        ));
    }
    if let Some(ap) = &autopilot {
        if ap.dim() != (agents.len(), N_AP) {
            return Err(format!(
                "autopilot must have shape ({}, {N_AP}), got {:?}",
                agents.len(),
                ap.dim()
            ));
        }
    }
    if let Some(c) = &commanded {
        if c.len() != agents.len() {
            return Err(format!("commanded must have length {}, got {}", agents.len(), c.len()));
        }
    }
    let mut out = Vec::with_capacity(agents.len());
    for (i, &(id, kind)) in agents.iter().enumerate() {
        if commanded.as_ref().is_some_and(|c| !c[i]) {
            continue;
        }
        let r = raw.row(i);
        let control = match &autopilot {
            Some(ap) if ap[[i, 0]] >= 0.5 => Control::Autopilot(AutopilotTargets {
                heading_rad: ap[[i, 1]],
                alt_hae_m: ap[[i, 2]],
                tas_mps: ap[[i, 3]],
            }),
            _ if kind == VehicleKind::Multirotor => Control::Multirotor(MultirotorRaw {
                roll_rate: r[0],
                pitch_rate: r[1],
                yaw_rate: r[2],
                collective: r[3],
            }),
            _ => Control::Raw(RawActuators {
                aileron: r[0],
                elevator: r[1],
                rudder: r[2],
                throttle: r[3],
            }),
        };
        out.push((id, Action { control, trigger: r[4] >= 0.5 }));
    }
    Ok(out)
}

/// Episode flags and events returned by `Env.step`.
#[pyclass(get_all, frozen)]
pub struct StepInfo {
    /// The engagement was decided (≥ 2 teams started, ≤ 1 has a living aircraft).
    pub terminated: bool,
    /// Sim time reached `sim.max_time_s`.
    pub truncated: bool,
    pub tick: u64,
    pub sim_time_s: f64,
    /// This tick's events, in order, as dicts: `tick`, `sim_time_s`, `kind`
    /// (`hit`, `kill`, `trigger-pressed`, `mode-change`, ...) and the kind's
    /// fields, the same shape as the replay export (README, "Events").
    pub events: Py<PyList>,
}

impl StepInfo {
    fn new(py: Python<'_>, r: StepResult, events: &[Event]) -> PyResult<Self> {
        let list = PyList::empty(py);
        for e in events {
            list.append(event_dict(py, e)?)?;
        }
        Ok(StepInfo {
            terminated: r.terminated,
            truncated: r.truncated,
            tick: r.tick,
            sim_time_s: r.sim_time_s,
            events: list.unbind(),
        })
    }
}

/// One event as the flat dict every consumer shares (see [`StepInfo::events`]).
fn event_dict<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("tick", e.tick)?;
    d.set_item("sim_time_s", e.sim_time_s)?;
    d.set_item("kind", e.kind.name())?;
    let enu = |p: astrum_proto::Enu| [p.e, p.n, p.u];
    match &e.kind {
        EventKind::TriggerPressed { aircraft_id, ammo } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("ammo", ammo)?;
        }
        EventKind::TriggerReleased { aircraft_id, rounds_fired, ammo } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("rounds_fired", rounds_fired)?;
            d.set_item("ammo", ammo)?;
        }
        EventKind::Hit { shooter_id, target_id, damage, hit_points, pos_enu } => {
            d.set_item("shooter_id", shooter_id)?;
            d.set_item("target_id", target_id)?;
            d.set_item("damage", damage)?;
            d.set_item("hit_points", hit_points)?;
            d.set_item("pos_enu", enu(*pos_enu))?;
        }
        EventKind::Kill { victim_id, killer_id, pos_enu } => {
            d.set_item("victim_id", victim_id)?;
            d.set_item("killer_id", killer_id)?;
            d.set_item("pos_enu", enu(*pos_enu))?;
        }
        EventKind::ModeChange { aircraft_id, from, to, reason } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("from", from.name())?;
            d.set_item("to", to.name())?;
            d.set_item("reason", reason.name())?;
        }
        EventKind::ControllerUp { aircraft_id } | EventKind::ControllerDown { aircraft_id } => {
            d.set_item("aircraft_id", aircraft_id)?;
        }
        EventKind::RunEnded { reason, message } => {
            d.set_item("reason", reason.name())?;
            d.set_item("message", message)?;
        }
        EventKind::GatePassed { aircraft_id, gate, offset_x_m, offset_y_m } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("gate", gate)?;
            d.set_item("offset_x_m", offset_x_m)?;
            d.set_item("offset_y_m", offset_y_m)?;
        }
        EventKind::GateMissed { aircraft_id, gate } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("gate", gate)?;
        }
        EventKind::Crashed { aircraft_id, outcome, gate } => {
            d.set_item("aircraft_id", aircraft_id)?;
            d.set_item("outcome", outcome.name())?;
            d.set_item("gate", gate)?;
        }
    }
    Ok(d)
}

#[pymethods]
impl StepInfo {
    fn done(&self) -> bool {
        self.terminated || self.truncated
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "StepInfo(terminated={}, truncated={}, tick={}, sim_time_s={:.3}, events={})",
            self.terminated,
            self.truncated,
            self.tick,
            self.sim_time_s,
            self.events.bind(py).len()
        )
    }
}

/// One embedded simulation (see the module docs).
#[pyclass]
pub struct Env {
    inner: SimEnv,
    agents: Vec<(u32, VehicleKind)>,
}

#[pymethods]
impl Env {
    #[new]
    fn new(py: Python<'_>, scenario_toml: &str) -> PyResult<Self> {
        let scenario = parse_scenario(scenario_toml)?;
        let inner = py.detach(|| SimEnv::new(scenario)).map_err(env_err)?;
        let agents = agents_of(&inner);
        Ok(Env { inner, agents })
    }

    /// Sorted aircraft ids; the row order of every observation and action array.
    #[getter]
    fn agent_ids(&self) -> Vec<u32> {
        self.agents.iter().map(|(id, _)| *id).collect()
    }

    /// Physics step (s).
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt()
    }

    /// The current scenario as TOML.
    #[getter]
    fn scenario_toml(&self) -> String {
        self.inner.scenario().to_toml()
    }

    /// Return to the initial state (optionally of a new scenario, optionally
    /// with a new seed). Reuses the loaded aircraft when the fleet is unchanged.
    #[pyo3(signature = (seed=None, scenario_toml=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seed: Option<u64>,
        scenario_toml: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let scenario = scenario_toml.map(parse_scenario).transpose()?;
        let inner = &mut self.inner;
        let obs = py.detach(|| inner.reset(seed, scenario)).map_err(env_err)?;
        self.agents = agents_of(&self.inner);
        Ok(obs_matrix(&obs).into_pyarray(py))
    }

    /// Advance one tick. `actions[i]` = `[aileron, elevator, rudder, throttle,
    /// trigger]` for `agent_ids[i]` (on a multirotor: `[roll_rate, pitch_rate,
    /// yaw_rate, collective, trigger]`; trigger fires at ≥ 0.5). Optional
    /// `autopilot[i]` = `[engage, heading_rad, alt_hae_m, tas_mps]` replaces the
    /// raw channels with autopilot targets where `engage` ≥ 0.5. Rows with
    /// `commanded[i] == False` send nothing (the aircraft keeps its last command
    /// until the lease expires).
    #[pyo3(signature = (actions, autopilot=None, commanded=None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, f64>,
        autopilot: Option<PyReadonlyArray2<'py, f64>>,
        commanded: Option<PyReadonlyArray1<'py, bool>>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, StepInfo)> {
        let parsed = actions_for(
            &self.agents,
            actions.as_array(),
            autopilot.as_ref().map(|a| a.as_array()),
            commanded.as_ref().map(|c| c.as_array()),
        )
        .map_err(PyValueError::new_err)?;
        let inner = &mut self.inner;
        let (obs, result) = py.detach(|| inner.step(&parsed)).map_err(env_err)?;
        let info = StepInfo::new(py, result, self.inner.events())?;
        Ok((obs_matrix(&obs).into_pyarray(py), info))
    }

    /// The latest observation without stepping.
    fn observe<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        obs_matrix(&self.inner.observe()).into_pyarray(py)
    }

    /// Ballistic fire control of `shooter` against `target` (the sim's own round
    /// model): a dict with `miss_m`, `tof_s`, `hit`, `aim_dir` (grid unit
    /// vector), `aim_error_rad`, `in_range`; `None` if the shooter is unarmed.
    fn gun_solution<'py>(
        &self,
        py: Python<'py>,
        shooter: u32,
        target: u32,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(s) = self.inner.gun_solution(shooter, target).map_err(env_err)? else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("miss_m", s.miss_m)?;
        d.set_item("tof_s", s.tof_s)?;
        d.set_item("hit", s.hit)?;
        d.set_item("aim_dir", s.aim_dir.to_array())?;
        d.set_item("aim_error_rad", s.aim_error_rad)?;
        d.set_item("in_range", s.in_range)?;
        Ok(Some(d))
    }

    /// Rotate a body-axes vector (x nose, y starboard, z down) of aircraft `id`
    /// into the world grid (ENU), accounting for the local-level tilt.
    fn body_to_grid(&self, id: u32, v: [f64; 3]) -> PyResult<[f64; 3]> {
        self.inner.body_to_grid(id, v).map_err(env_err)
    }

    /// Rotate a world-grid (ENU) vector into the body axes of aircraft `id`.
    fn grid_to_body(&self, id: u32, v: [f64; 3]) -> PyResult<[f64; 3]> {
        self.inner.grid_to_body(id, v).map_err(env_err)
    }
}

/// `K` independent simulations stepped in parallel (one call, GIL released).
/// Every scenario must have the same number of aircraft.
#[pyclass]
pub struct BatchEnv {
    envs: Vec<SimEnv>,
    agents: Vec<Vec<(u32, VehicleKind)>>,
}

impl BatchEnv {
    fn stack(&self, obs: &[Vec<Telemetry>]) -> Array3<f64> {
        let n = self.agents[0].len();
        let mut out = Array3::zeros((obs.len(), n, N_OBS));
        for (k, o) in obs.iter().enumerate() {
            out.index_axis_mut(numpy::ndarray::Axis(0), k).assign(&obs_matrix(o));
        }
        out
    }
}

#[pymethods]
impl BatchEnv {
    #[new]
    fn new(py: Python<'_>, scenario_tomls: Vec<String>) -> PyResult<Self> {
        if scenario_tomls.is_empty() {
            return Err(PyValueError::new_err("BatchEnv needs at least one scenario"));
        }
        let scenarios =
            scenario_tomls.iter().map(|s| parse_scenario(s)).collect::<PyResult<Vec<_>>>()?;
        let n = scenarios[0].aircraft.len();
        if scenarios.iter().any(|s| s.aircraft.len() != n) {
            return Err(PyValueError::new_err(
                "every scenario in a BatchEnv must have the same number of aircraft",
            ));
        }
        let envs = py
            .detach(|| scenarios.into_par_iter().map(SimEnv::new).collect::<Result<Vec<_>, _>>())
            .map_err(env_err)?;
        let agents = envs.iter().map(agents_of).collect();
        Ok(BatchEnv { envs, agents })
    }

    fn __len__(&self) -> usize {
        self.envs.len()
    }

    /// Per-env sorted aircraft ids.
    #[getter]
    fn agent_ids(&self) -> Vec<Vec<u32>> {
        self.agents.iter().map(|a| a.iter().map(|(id, _)| *id).collect()).collect()
    }

    /// Reset every env (optionally with per-env seeds); returns `(K, n, F)`.
    #[pyo3(signature = (seeds=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if let Some(s) = &seeds {
            if s.len() != self.envs.len() {
                return Err(PyValueError::new_err(format!(
                    "need {} seeds, got {}",
                    self.envs.len(),
                    s.len()
                )));
            }
        }
        let envs = &mut self.envs;
        let obs = py
            .detach(|| {
                envs.par_iter_mut()
                    .enumerate()
                    .map(|(k, e)| e.reset(seeds.as_ref().map(|s| s[k]), None))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(env_err)?;
        Ok(self.stack(&obs).into_pyarray(py))
    }

    /// Step every env one tick. `actions` is `(K, n, 5)`, `autopilot` optional
    /// `(K, n, 4)` (see `Env.step`). Returns `(obs (K, n, F), terminated (K,),
    /// truncated (K,), tick (K,))`.
    #[pyo3(signature = (actions, autopilot=None))]
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray3<'py, f64>,
        autopilot: Option<PyReadonlyArray3<'py, f64>>,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<u64>>,
    )> {
        let act = actions.as_array();
        let ap = autopilot.as_ref().map(|a| a.as_array());
        if act.shape()[0] != self.envs.len() {
            return Err(PyValueError::new_err(format!(
                "actions must have {} rows on axis 0",
                self.envs.len()
            )));
        }
        let mut parsed = Vec::with_capacity(self.envs.len());
        for k in 0..self.envs.len() {
            let a = act.index_axis(numpy::ndarray::Axis(0), k);
            let p = ap.as_ref().map(|x| x.index_axis(numpy::ndarray::Axis(0), k));
            parsed.push(
                actions_for(&self.agents[k], a, p, None)
                    .map_err(|e| PyValueError::new_err(format!("env {k}: {e}")))?,
            );
        }
        let envs = &mut self.envs;
        let results = py
            .detach(|| {
                envs.par_iter_mut()
                    .zip(parsed.par_iter())
                    .map(|(e, a)| e.step(a))
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(env_err)?;
        let obs: Vec<Vec<Telemetry>> = results.iter().map(|(o, _)| o.clone()).collect();
        let term = Array1::from_iter(results.iter().map(|(_, r)| r.terminated));
        let trunc = Array1::from_iter(results.iter().map(|(_, r)| r.truncated));
        let tick = Array1::from_iter(results.iter().map(|(_, r)| r.tick));
        Ok((
            self.stack(&obs).into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            tick.into_pyarray(py),
        ))
    }
}

/// The sim's built-in fixed-wing autopilot law (PI on bank, altitude with
/// climb-rate damping, and airspeed), for scripts that mix it with their own
/// maneuvers. Stateful: one instance per aircraft, `reset()` between episodes.
#[pyclass]
pub struct FixedWingAutopilot {
    gains: FixedWingGains,
    integrators: Integrators,
    dt: f64,
}

#[pymethods]
impl FixedWingAutopilot {
    /// `profile`: `"trainer"` or `"fighter"`; `dt`: the env's step.
    #[new]
    fn new(profile: &str, dt: f64) -> PyResult<Self> {
        let p = match profile {
            "trainer" => AutopilotProfile::Trainer,
            "fighter" => AutopilotProfile::Fighter,
            other => {
                return Err(PyValueError::new_err(format!("unknown autopilot profile '{other}'")))
            }
        };
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PyValueError::new_err("dt must be > 0"));
        }
        Ok(Self { gains: FixedWingGains::for_profile(p), integrators: Integrators::default(), dt })
    }

    fn reset(&mut self) {
        self.integrators = Integrators::default();
    }

    /// `[aileron, elevator, rudder, throttle]` steering the aircraft in `obs_row`
    /// toward the targets.
    fn step(
        &mut self,
        obs_row: PyReadonlyArray1<'_, f64>,
        heading_rad: f64,
        alt_hae_m: f64,
        tas_mps: f64,
    ) -> PyResult<[f64; 4]> {
        let s = fdm_state_of(obs_row.as_array())?;
        let t = AutopilotTargets { heading_rad, alt_hae_m, tas_mps };
        let c = pilot::fixed_wing(&s, &t, &self.gains, &mut self.integrators, self.dt);
        Ok([c.aileron, c.elevator, c.rudder, c.throttle])
    }
}

/// The sim's built-in multirotor hover-hold law (PID on altitude).
#[pyclass]
pub struct MultirotorAutopilot {
    integrators: Integrators,
    dt: f64,
}

#[pymethods]
impl MultirotorAutopilot {
    #[new]
    fn new(dt: f64) -> PyResult<Self> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(PyValueError::new_err("dt must be > 0"));
        }
        Ok(Self { integrators: Integrators::default(), dt })
    }

    fn reset(&mut self) {
        self.integrators = Integrators::default();
    }

    fn step(
        &mut self,
        obs_row: PyReadonlyArray1<'_, f64>,
        heading_rad: f64,
        alt_hae_m: f64,
    ) -> PyResult<[f64; 4]> {
        let s = fdm_state_of(obs_row.as_array())?;
        let t = AutopilotTargets { heading_rad, alt_hae_m, tas_mps: 0.0 };
        let c = pilot::multirotor(&s, &t, &mut self.integrators, self.dt);
        Ok([c.aileron, c.elevator, c.rudder, c.throttle])
    }
}

/// Wrap an angle to `[-π, π)` (the sim's convention).
#[pyfunction]
fn wrap_pi(a: f64) -> f64 {
    pilot::wrap_pi(a)
}

#[pymodule]
fn astrum_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<BatchEnv>()?;
    m.add_class::<StepInfo>()?;
    m.add_class::<FixedWingAutopilot>()?;
    m.add_class::<MultirotorAutopilot>()?;
    m.add_function(wrap_pyfunction!(wrap_pi, m)?)?;
    m.add("OBS_FIELDS", OBS_FIELDS.to_vec())?;
    m.add("VEHICLE_KINDS", VEHICLE_KINDS.to_vec())?;
    m.add("CONTROL_MODES", CONTROL_MODES.to_vec())?;
    let combat = CombatParams::default();
    let gun = Gun::m61a1();
    m.add("HIT_RADIUS_M", combat.hit_radius_m)?;
    m.add("BULLET_TTL_S", combat.bullet_ttl_s)?;
    m.add("DAMAGE_PER_HIT", combat.damage_per_hit)?;
    m.add("MUZZLE_SPEED_MPS", gun.muzzle_speed_mps)?;
    m.add("GUN_RATE_HZ", gun.rate_hz)?;
    m.add("GUN_AMMO", gun.ammo)?;
    Ok(())
}
