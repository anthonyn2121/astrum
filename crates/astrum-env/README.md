# astrum-env

Python bindings for the embedded ASTRUM reinforcement-learning **dogfight
environment** (design doc §8). A thin [pyo3](https://pyo3.rs) marshalling layer
over the plain-Rust seam [`astrum_sim::env::SimEnv`] — all the ECS/JSBSim work
lives in `astrum-sim`; this crate only converts types across the FFI boundary.

## Two layers

| Layer | Where | What |
|-------|-------|------|
| 1. Rust seam | `astrum-sim/src/env.rs` (`SimEnv`, `Action`, `AgentObs`, `StepResult`) | Multi-agent `reset`/`step`/`observe` over one headless sim. No pyo3; tested by `astrum-sim/tests/env.rs`. |
| 2. Python binding | this crate (`DogfightEnv`, `Observation`, `StepInfo`) | `#[pyclass]` wrappers; validated by `python/smoke.py`. |

Keeping the sim logic in layer 1 means it stays exercisable from `cargo test`
(no Python), and the binding stays a trivial, low-risk shim.

## Environment shape (PettingZoo-style, multi-agent)

- **Action** per agent: `[aileron, elevator, rudder, throttle, trigger]`.
  Continuous channels are clamped (surfaces `[-1,1]`, throttle `[0,1]`); `trigger`
  is a **learned** fire gate (thresholded at `>= 0.5`) — the policy learns *when*
  to shoot, it is not auto-aimed (design doc §8).
- **Observation** per agent (`Observation`): id, team, alive, ENU position, NED
  velocity, attitude (φ/θ/ψ), body rates (p/q/r), airspeed, altitude, health,
  ammo, sim time. Relative geometry / gun-solution / normalisation are the
  trainer's job, computed from these absolute fields.
- **Episode flags** (`StepInfo`): `terminated` (engagement decided — ≥2 teams
  started, ≤1 alive), `truncated` (hit `sim.max_time_s`).

One `step()` is exactly one physics tick (`dt = 1/rate_hz`; the sim has no
`FixedUpdate`). `reset()` is deterministic for a fixed scenario + `sim.seed`.

## Parallelism (hard constraint)

JSBSim is process-global and **not thread-safe**, so `DogfightEnv` is
`#[pyclass(unsendable)]` — pinned to its creating thread. Scale training with
**separate processes** (Gymnasium `AsyncVectorEnv` / SB3 `SubprocVecEnv`), one
env per worker — **never** OS threads. Within a process the env is inherently
multi-agent (N aircraft step together on one FDM-sequential app), which is what
self-play / NvN needs.

## Build & run (inside the dev container)

```bash
# Build the extension module + run the Python smoke test:
bash crates/astrum-env/python/run_smoke.sh
```

`run_smoke.sh` runs `cargo build -p astrum-env --release`, then stages the
resulting `target/release/libastrum_env.so` as `astrum_env.so` on `PYTHONPATH`
and runs `python/smoke.py`.

### Toolchain notes

- The pyo3 features `extension-module` + `abi3-py38` mean the build needs **no
  `python3-dev` headers and no `maturin`** and does **not** link libpython — the
  interpreter resolves those symbols at import time. Plain `cargo build` + a
  rename to `astrum_env.so` is enough for the seam. (The dev image has `python3`
  on `PATH`, which pyo3's build script probes for the ABI config.)
- `[lib] test = false` keeps `cargo test --workspace` from building a test
  harness for this cdylib (which would fail to link, libpython being undefined).
- For a real install/packaging step later, `maturin develop` / `maturin build`
  would produce a proper wheel that names the `.so` correctly and installs it
  into the venv — not required to use the seam, but the natural next step for
  distribution. (`maturin` is not yet in the dev image.)
