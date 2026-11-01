# astrum-env

Python bindings (`astrum_env`) for the embedded ASTRUM multi-agent environment,
plus the numpy-only Gymnasium dogfight adapter (`python/astrum_dogfight`). The
binding is a thin [PyO3](https://pyo3.rs) marshalling layer over
[`astrum_sim::env::SimEnv`]: all ECS/JSBSim work lives in `astrum-sim`; this crate
moves observations and actions as NumPy arrays and runs the sim with the GIL
released.

| Layer | Where | What |
|-------|-------|------|
| Rust seam | `astrum-sim/src/env.rs` (`SimEnv`) | Multi-agent `reset`/`step`/`observe`, fire control, attitude rotations. Tested by `cargo test`. |
| Python binding | `src/lib.rs` (`Env`, `BatchEnv`, `StepInfo`, `FixedWingAutopilot`, `MultirotorAutopilot`) | NumPy in/out; validated by `python/smoke.py`. |
| Gym adapter | `python/astrum_dogfight` | Observation vector, reward, episode boundaries, scripted Tier-1/2/3 opponents. Tested by `python/test_dogfight_env.py`. |

## API

```python
import astrum_env, numpy as np

env = astrum_env.Env(scenario_toml)          # scenario TOML v2
env.agent_ids                                # sorted ids = row order of every array
env.dt                                       # physics step (s) = 1 / sim.rate_hz
obs = env.reset(seed=3)                      # (n, len(OBS_FIELDS)) float64; tick 1
obs = env.reset(scenario_toml=other_toml)    # next episode in a new scenario, same Env
obs, info = env.step(actions)                # actions (n, 5); info: StepInfo
obs, info = env.step(actions, autopilot=ap, commanded=mask)
env.observe()                                # latest obs without stepping
env.gun_solution(shooter, target)            # sim fire control, dict or None (unarmed)
env.body_to_grid(id, v); env.grid_to_body(id, v)

batch = astrum_env.BatchEnv([scenario_toml] * 16)
obs = batch.reset(seeds=list(range(16)))     # (K, n, F)
obs, terminated, truncated, tick = batch.step(actions)   # actions (K, n, 5)
```

- **Actions** `(n, 5)`: `[aileron, elevator, rudder, throttle, trigger]` per row
  (surfaces `[-1, 1]`, throttle `[0, 1]`, trigger fires at `>= 0.5` — a learned
  fire gate, not auto-aim). Optional `autopilot` `(n, 4)` =
  `[engage, heading_rad, alt_hae_m, tas_mps]` replaces the raw channels where
  `engage >= 0.5`. Rows with `commanded[i] == False` send nothing (the aircraft
  keeps its last command until the raw-command lease expires).
- **Observations** `(n, F)`, columns `astrum_env.OBS_FIELDS`: `id`, `team`,
  `alive`, `vehicle_kind` / `control_mode` (codes index `VEHICLE_KINDS` /
  `CONTROL_MODES`), `tick`, `sim_time_s`, `lat_deg`, `lon_deg`, `alt_hae_m`,
  world-grid ENU position `pos_e/n/u` and velocity `vel_e/n/u`, local-level NED
  velocity `vel_ned_n/e/d`, attitude `phi/theta/psi`, body rates `p/q/r`,
  `alpha`, `beta`, `nz`, `mach`, `tas` (true airspeed), `hit_points`,
  `hit_points_max`, `ammo`. SI units, radians. Use the grid columns for relative
  geometry and the NED columns only where local level is meant.
- **`StepInfo`**: `terminated` (≥ 2 teams started, ≤ 1 has a living aircraft),
  `truncated` (`sim.max_time_s` reached), `tick`, `sim_time_s`, `done()`.
- **`gun_solution`** is the sim's own round model (drag, gravity, velocity
  inheritance, muzzle offset): `miss_m`, `tof_s` (boresight round), `hit`,
  `aim_dir` (grid unit vector), `aim_error_rad`, `in_range`. Gun/combat constants
  are exported: `HIT_RADIUS_M`, `BULLET_TTL_S`, `DAMAGE_PER_HIT`,
  `MUZZLE_SPEED_MPS`, `GUN_RATE_HZ`, `GUN_AMMO`.
- **`body_to_grid` / `grid_to_body`** rotate between body axes (x nose,
  y starboard, z down) and the world grid, including the local-level tilt away
  from the grid origin.
- **Autopilot laws**: `FixedWingAutopilot("trainer" | "fighter", env.dt)` and
  `MultirotorAutopilot(env.dt)` are the sim's built-in laws, `step(obs_row, ...)`
  → `[aileron, elevator, rudder, throttle]`. They are stateful (PI integrators):
  one instance per aircraft, `reset()` between episodes. `wrap_pi` is the sim's
  angle wrap.

One `step()` is one physics tick. `reset(seed=s)` is deterministic: the same
scenario and seed reproduce the episode bit-for-bit (also across `BatchEnv`).

## Gym adapter (`astrum_dogfight`)

`DogfightGymEnv` (and `make_tier1_env` / `make_tier2_env` / `make_tier3_env`)
is single-agent from the trainer's view against a scripted opponent. It follows
Gymnasium seeding (`reset(seed=s)` reseeds `np_random`, which draws each
episode's sim seed) and changes scenario per episode with
`reset(options={"scenario_toml": toml})`. Lead, range and fire decisions use
`Env.gun_solution`; attitude uses `Env.body_to_grid`; the opponents' level hold is
`FixedWingAutopilot("fighter", env.dt)`. The hard deck and ceiling are heights
above the ellipsoid — the env has no terrain model. It subclasses
`gymnasium.Env` when gymnasium is installed and runs without it otherwise.

## Threading and parallelism

`Env` is thread-safe and releases the GIL during construction, `reset` and
`step`, so several `Env`s can be driven from Python threads concurrently.
`BatchEnv` steps its K envs in parallel (rayon) inside one call — the cheapest
way to vectorize. Process-level parallelism (Gymnasium `AsyncVectorEnv`, SB3
`SubprocVecEnv`) works as well.

## Build & test (inside the dev image)

```bash
S=$(bash scripts/build-astrum-env.sh)        # builds the cdylib, prints the stage dir
export PYTHONPATH=$S:crates/astrum-env/python
python crates/astrum-env/python/smoke.py     # binding smoke
pytest -q crates/astrum-env/python           # adapter tests
make test-python                             # all Python suites, incl. the two above
```

`scripts/build-astrum-env.sh` runs `cargo build --release -p astrum-env` and
copies `target/release/libastrum_env.so` to `<stage>/astrum_env.so`
(default `target/python`, override with `ASTRUM_ENV_STAGE`).

### Toolchain notes

- PyO3's `extension-module` + `abi3` features mean the build needs no
  `python3-dev` headers and no `maturin`, and does not link libpython — the
  interpreter resolves those symbols at import time.
- `[lib] test = false` keeps `cargo test --workspace` from building a test
  harness for this cdylib (which would fail to link, libpython being undefined).
