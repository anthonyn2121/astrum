# controllers/rl — from-scratch RL for the dogfight learner

A learned flight controller for the WVR gun-dogfight, trained from scratch (**no**
stable-baselines3) against the scripted Tier-1/2/3 opponents in the embedded
`astrum_dogfight` env.

**Two learners, both from scratch:**
- **TD-MPC2** (`astrum_rl/tdmpc2/`) — the *primary* learner: a latent world model
  planned through with MPPI, trained off-policy from a 1M replay buffer, one update per
  env step. Chosen over PPO because PPO underperformed on comparable air-combat envs and
  because a single-env/off-policy loop matches JSBSim's process-global constraint. Spec:
  [`docs/dogfight-tdmpc2-design.md`](../../docs/dogfight-tdmpc2-design.md).
- **PPO** (`astrum_rl/ppo.py`) — a cheap sanity baseline, kept for comparison.

This is a **separate uv environment**, on purpose. torch + the CUDA runtime are
multi-GB and GPU-specific; keeping them out of the base `astrum-dev` image (which
every controller shares) means the base stays lean. The image ships only `uv`
(the env manager) and pure-Python `gymnasium`; everything heavy lives in the uv env
defined by `pyproject.toml` here.

## Layout

| Path | What |
|------|------|
| `pyproject.toml`        | uv project: torch (cu128 / Blackwell), gymnasium, numpy; `pytest` in the dev group. |
| `astrum_rl/_paths.py`   | Puts the embedded env (`astrum_env` .so + `astrum_dogfight`) on `sys.path`. |
| `astrum_rl/envs.py`     | Env factories + action maps (PPO **squash**, TD-MPC2 **affine** `map_tdmpc_action`), `ActionRepeat`/`FrameStack`, `wrap_tdmpc`, AsyncVectorEnv thunk. |
| `astrum_rl/ppo.py`      | The from-scratch PPO baseline: actor-critic MLP, GAE, clipped-surrogate update. |
| `astrum_rl/train.py`    | PPO CLI training entrypoint. |
| `astrum_rl/smoke_gpu.py`| GPU + env smoke test. |
| `astrum_rl/tdmpc2/`     | **The TD-MPC2 learner** (subpackage — see below). |
| `vendor/astrum_env.so`  | The staged pyo3 extension (git-ignored; produced by `make rl-env`). |

### `astrum_rl/tdmpc2/` (the primary learner)

| Module | What |
|--------|------|
| `config.py`       | `TDMPC2Config` dataclass + derived `resolved_discount` / `resolved_seed_steps` / `bin_size`. |
| `functional.py`   | `symlog/symexp`, `two_hot/two_hot_inv`, `soft_ce`, `gaussian_logprob`, tanh `squash`, `RunningScale`. |
| `layers.py`       | `SimNorm`, `NormedLinear`, `mlp`, `encoder`, `Ensemble`. |
| `world_model.py`  | The 6 heads (encoder, dynamics, reward, policy prior, Q-ensemble, termination) + Polyak target Qs. |
| `buffer.py`       | **Plain-tensor** episodic ring buffer (cap 1M); rejection-sampled horizon slices; `terminated` vs `truncated` stored separately. |
| `agent.py`        | `TDMPC2`: `act`/`plan` (MPPI), `_estimate_value`, `_td_target`, `update`, `update_pi`, checkpoint I/O (embeds `cfg`). |
| `train.py`        | Single-env off-policy loop: seed warmup → pretrain burst → 1 update/step → curriculum tier swap → checkpoint. |
| `train_parallel.py`| **Throughput scale-out** (design doc §8): K CPU collector processes (one JSBSim env each, acting prior-only) stream whole episodes to a central GPU learner running back-to-back updates; weights broadcast back each `weight_sync_every`. Same model/buffer/update code as `train.py`. |
| `smoke.py`        | End-to-end smoke (tiny model, real JSBSim): seed → update → plan. CPU. |
| `smoke_parallel.py`| End-to-end smoke of `train_parallel` (2 collectors, tiny model, real Tier-1 JSBSim, CPU): spawn → seed → burst → weight sync → prior collection → back-to-back updates → checkpoint reload; asserts steady-state training ran, not just the seed burst. |
| `export.py`       | `PolicyPrior` + TorchScript export (`.pt` + `.json` metadata) for planning-free deployment. |
| `selfplay.py`     | `ScriptedPolicyOpponent`, `OpponentPool`, `make_selfplay_env` (self-play scaffolding). |

Why a **plain-tensor buffer** and an **affine** (not squash) action map: see the
implementation-notes callout at the top of `docs/dogfight-tdmpc2-design.md`.

## GPU note (RTX 5080 / Blackwell)

The 5080 is Blackwell, compute capability **sm_120**. torch needs a **CUDA 12.8+**
build (the `cu128` wheels, `torch>=2.7`) or kernels fail at launch. `pyproject.toml`
pins the `cu128` index for exactly this. The wheel bundles its own CUDA runtime, so
the container stays plain `ubuntu:24.04` — you only need, on the host:

- a recent NVIDIA driver (CUDA 12.8-capable), and
- the `nvidia-container-toolkit`, with a generated CDI spec
  (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`).

The `dogfight-train` compose service grants the GPU **via CDI**
(`devices: ["nvidia.com/gpu=all"]`), which needs neither an `nvidia` Docker *runtime*
in `daemon.json` nor a daemon restart — just the toolkit + a CDI spec. `make rl-smoke`
/ `make rl-train` / `make rl-tdmpc2-train` use that service. (If you prefer the classic
`nvidia` runtime instead, `sudo nvidia-ctk runtime configure --runtime=docker &&
sudo systemctl restart docker`, then swap the `devices:` line back to a
`deploy.resources.reservations.devices` block.)

Verified on an RTX 5080: `torch 2.11.0+cu128`, `cuda available: True`,
`capability sm_120`, GPU matmul launches, and `rl-tdmpc2-train` runs the full
seed → update → MPPI-plan loop on `device cuda`.

**Training needs the GPU; correctness does not.** MPPI planning (6·512·3 latent
forwards/step) plus a per-step horizon-unrolled update makes real TD-MPC2 *training*
(`make rl-tdmpc2-train`) GPU-bound — that path requires the host
`nvidia-container-toolkit` above. Everything else is device-agnostic and CPU-verifiable:
the unit suite (`make rl-test`), the throughput benchmark (`make rl-bench`), and the
end-to-end smoke (`make rl-tdmpc2-smoke`, tiny model vs real JSBSim) all run on the plain
`sim` service with no GPU. If the toolkit isn't installed, you can still develop and
verify the whole pipeline on CPU — only the full training run is blocked.

## Quickstart

From the repo root (inside the dev container, or with `uv` on your host):

```bash
make rl-env            # build the astrum_env .so and stage it into vendor/

# TD-MPC2 (primary) — CPU-OK except training:
make rl-test           # unit suite (world model / buffer / agent / envs / export)  [CPU]
make rl-bench          # random-policy env-throughput benchmark (sizes the run)     [CPU]
make rl-tdmpc2-smoke   # end-to-end smoke: seed → update → MPPI plan vs real JSBSim [CPU]
make rl-tdmpc2-parallel-smoke  # smoke the K-collector scale-out (2 workers, tiny model) [CPU]
make rl-tdmpc2-train   # TD-MPC2 training, tier 1  (needs the GPU compose service)
make rl-tdmpc2-train-parallel ARGS="--workers 6 ..."  # K CPU collectors + central GPU learner (§8)

# PPO (sanity baseline):
make rl-smoke          # torch+CUDA+env smoke  (uses the GPU compose service)
make rl-train          # PPO training tier 1   (override: ARGS="--tier 2 --total-steps 5_000_000")
```

Or drive uv directly in `controllers/rl/`:

```bash
uv sync                                   # create the env from pyproject.toml
uv run pytest                             # unit suite (CPU)
uv run python -m astrum_rl.tdmpc2.smoke   # TD-MPC2 end-to-end smoke (CPU)
uv run python -m astrum_rl.tdmpc2.train --tier 1 --steps 1_000_000
# curriculum schedule instead of a fixed tier:
#   ... --curriculum "1:0,2:200000,3:500000" --steps 1_000_000
uv run python -m astrum_rl.tdmpc2.export --checkpoint run.pt --out policy   # deploy artifact
```

## Two constraints (from the env's design)

1. **Parallelism must be processes, never threads.** JSBSim is process-global and
   not thread-safe (`DogfightEnv` is `#[pyclass(unsendable)]`). Every scale-out uses
   separate `spawn`ed processes — one env each — never threads: PPO via
   `gymnasium.vector.AsyncVectorEnv` (`envs.env_thunk`), TD-MPC2 via the collector
   processes in `train_parallel.py`.
2. **Single-env collection is the design point for TD-MPC2**, not a limitation: it
   collects off-policy from one env into a 1M replay buffer, one update per step
   (design doc §1). Multi-process collectors feeding a shared buffer are an *optional*
   throughput scale-out — **now implemented** in `train_parallel.py` (design doc §8):
   K CPU collectors act prior-only and stream whole episodes to a central GPU learner
   that runs updates back-to-back, breaking the single-env loop's plan↔update
   serialization. For **PPO**, single-env *is* a bottleneck — vectorizing it is the
   first PPO scaling task; mind gymnasium ≥1.0's NEXT_STEP autoreset (see the note in
   `ppo.py`).

See `docs/dogfight-env-onboarding.md` and `crates/astrum-env/README.md` for the env
itself (observation/action layout, reward, opponents).
