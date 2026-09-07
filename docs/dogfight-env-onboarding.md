# Onboarding: the RL dogfight environment

**Audience:** a developer new to this codebase who needs to train (or hand-code) a
policy that flies the F-16 in a 1v1 gun fight.
**Companion doc:** [`dogfight-environment-design.md`](dogfight-environment-design.md)
is the *why* and the full design; this page is the *how do I start*.
**Date:** 2026-08-25

Everything here runs inside the `astrum-dev` Docker image — the host has no Rust,
JSBSim, or Python deps. Prefix commands with:

```bash
docker compose run --rm --no-deps sim bash -lc '<command>'
```

---

## 1. The mental model: a two-layer seam

There are two things named "env," and the split is the single most important thing
to understand:

| Layer | File | What it is | Do you touch it? |
|-------|------|-----------|------------------|
| **1. Rust sim seam** | `astrum-sim/src/env.rs` (`SimEnv`) → pyo3 wrapper `astrum-env/src/lib.rs` (`DogfightEnv`) | A pure **state-in / state-out** physics sim. One `step()` = one JSBSim tick. Knows nothing about rewards or observations. | Rarely — it's the stable floor. |
| **2. Python trainer adapter** | `crates/astrum-env/python/astrum_dogfight/` | Adds everything RL needs: the **observation vector**, the **reward**, the **episode boundaries**, and the **scripted opponent**. | **Yes — this is where you live.** Pure numpy Python. |

The key idea: the Rust sim hands you *raw absolute state* (positions, attitudes,
health, ammo). Turning that into "am I on his six, inside gun range, with a lead
solution?" is the **trainer's** job, done in `geometry.py`. The sim deliberately
omits reward and observation shaping so the physics stays reusable and testable
from `cargo test` with no Python involved.

Read `crates/astrum-env/README.md` first — it is the crispest statement of this
split and the build/toolchain notes.

---

## 2. First 30 minutes: build it and watch an episode

**Step 0 — build the extension and run the smoke test.** `astrum_env` is a
compiled Rust `.so`; this script builds it, stages it on `PYTHONPATH`, and runs a
minimal sanity check:

```bash
docker compose run --rm --no-deps sim bash -lc 'bash crates/astrum-env/python/run_smoke.sh'
```

**Step 1 — run the adapter's own test suite** (10 tests, no gymnasium required):

```bash
docker compose run --rm --no-deps sim bash -lc 'bash crates/astrum-env/python/run_gym_test.sh'
```

If both pass, the whole stack works. `test_dogfight_env.py` is also the best
worked example in the repo: **`test_live_rollout`** is a complete minimal episode
you can copy verbatim, and the `test_tier3_*` tests show the shooting opponent
end-to-end.

---

## 3. The Gymnasium interface

Standard Gymnasium 5-tuple API (`env.py`). Minimal loop:

```python
from astrum_dogfight.env import make_tier1_env   # or make_tier2_env / make_tier3_env
import numpy as np

env = make_tier1_env(max_time_s=30.0)
obs, info = env.reset()                            # obs: float32[31]

done = False
while not done:
    action = my_policy(obs)                        # float32[5] — see below
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Action — `float32[5]`

`[aileron, elevator, rudder, throttle, trigger]`

- Bounds (the `action_space`): `aileron/elevator/rudder ∈ [-1, 1]`,
  `throttle/trigger ∈ [0, 1]`.
- **These are FLCS commands, not raw control-surface deflections.** `elevator = -1`
  commands a full **+9 g pull**; `aileron` is a **roll-rate** command (~188°/s at
  full). The flight-control system enforces the ~30° AoA / 9 g envelope so you
  can't depart the jet. This trips up everyone once — see the design doc §8 and the
  `f16-raw-is-flcs-nrt` note.
- `trigger` is a **learned fire gate**, thresholded at `≥ 0.5` into a real M61A1
  gun burst. The policy learns *when* to shoot; there is **no auto-aim**.

### Observation — `float32[31]`

Egocentric and normalized (built by `geometry.build_observation`). You do **not**
assemble this by hand — the env returns it — but know the layout so you know what
the policy sees:

- **14 own-state** — airspeed, altitude-above-deck, `sin/cos(φ)`, `sin/cos(θ)`,
  body rates p/q/r, α, β, load factor Nz, health, ammo.
- **14 relative-geometry** — range, `sin/cos` of the bandit's bearing off your nose
  (azimuth + elevation), aspect angle, heading-crossing angle, closure, relative
  altitude, bandit airspeed / health / alive-flag.
- **3 WEZ / gun aids** — in-range flag, gun-solution quality ∈ [0,1], and the
  cosine of the lead error.

Absolute heading ψ is deliberately dropped so a learned policy generalizes across
compass directions.

### Reward and episode flags

Computed by `reward.py`. Dense terms accrue every step; terminal events have strict
precedence:

Values below are **reward-v4** (see design doc §8.3 for the v1→v4 rationale):

| Event | Reward | Flag | `info["event"]` |
|-------|-------:|------|-----------------|
| Hard-deck bust (< 300 m AGL) | −200 | terminated | `hard_deck` |
| Kill (bandit dies) | +150 | terminated | `kill` (or `trade`) |
| Death (learner dies) | −60 | terminated | `killed` |
| Mutual kill (both die same tick) | −25 | terminated | `trade` |
| Bubble exit (> 15 km, or over ceiling) | −50 | truncated | `left_bubble` |
| Damage dealt / taken | ±0.5 / HP | — | `damage_dealt` / `damage_taken` |
| Dense tracking (nose-on × range-closeness, in TRACK_RANGE) | `+K_TRACK·track·range_factor` | — | — |
| Dense gun-solution (in-WEZ quality) | `+K_SOLUTION·gun_quality` | — | — |
| ↳ cap: cumulative dense tracking / episode | `≤ TRACK_BUDGET (50 ≈ R_KILL/3)` | — | `info["track_used"]` |
| Fire a round in / out of solution | `+0.05` / `−0.005` per round | — | — |
| Time | −0.001 each step | — | — |

`info["event"]` tells you which terminal fired; `info["phi"]` is the current
geometry potential. Hard-deck wins over everything, then kill/death, then bubble.
Reward-v2 replaced the old potential-based `K_GEOM·(γΦ′−Φ)` shaping (which telescoped
to nothing and let the policy park at a passive draw) with the non-telescoping dense
tracking terms above; reward-v3 then raised the in-WEZ fire bonus so the policy
actually pulls the trigger. reward-v4 caps the *cumulative* tracking at `TRACK_BUDGET`
(≈`R_KILL`/3) per episode — v2/v3 left it unbounded (`≈2·R_KILL` over a full episode),
so a nose-on track-draw could out-earn a kill and the passive-draw trap reincarnated at
a higher reward level. The cap is enforced in `DogfightGymEnv.step` (`reward.compute`
stays stateless); `K_GEOM` still lives in `constants.py` but is unused.

### Picking the opponent (difficulty tier)

One constructor swaps the whole opponent behavior — the observation/action/reward
contract is identical across tiers, so a policy trained against one runs against
any:

| Factory | Opponent | Behavior |
|---------|----------|----------|
| `make_tier1_env()` | `Tier1Opponent` | Straight-and-level dummy — pure offense practice. Non-shooting. |
| `make_tier2_env(maneuver="break"\|"split_s"\|"alternate")` | `Tier2Opponent` | Predictable defenders: breaks or Split-Ses when it detects you in its rear hemisphere. Non-shooting. |
| `make_tier3_env(posture="offensive"\|"defensive")` | `Tier3Opponent` | 5-state BFM FSM that **shoots back** through the real gun pipeline. `"offensive"` starts it on your six; `"defensive"` starts it ahead of you. |

All three accept scenario keyword args, e.g. `make_tier1_env(separation_m=1500.0,
alt_ft=18000.0, airspeed_kts=380.0, max_time_s=45.0)` (see `scenario.py` for the
full list).

---

## 4. What to build first

The env is complete; **everything above it (the learner's policy) is not**. Good
first tasks, in order:

1. **A hand-coded pursuit policy for the learner.** Reimplement lead pursuit on the
   learner side — read the observation, roll the lift vector onto the bandit, pull,
   pull the trigger inside the WEZ. This forces you to internalize the action and
   observation layouts without needing a training framework, and gives you a
   scripted baseline to benchmark a *learned* policy against later. The reference
   to learn from is `Tier3Opponent._pursue` and `wez_fire` in `opponent.py`.
2. **A random-action sanity harness** — confirm episodes terminate for the right
   reasons (watch `info["event"]`), and that reward is finite every step.
3. **Only then, learning.** A from-scratch PPO baseline now lives in
   `controllers/rl/` (its own uv env with torch/GPU — **no** stable-baselines3).
   Build the env and train with `make rl-env` then `make rl-train`; see that
   directory's README.

---

## 5. Two hard constraints — read before you scale

1. **Parallelism must be multiprocessing, never threads.** JSBSim is process-global
   and **not thread-safe**, so `DogfightEnv` is `#[pyclass(unsendable)]` — pinned to
   its creating thread. Scale training with `SubprocVecEnv` / `AsyncVectorEnv`, one
   env per process. Using OS threads will corrupt the FDM.
2. **Training deps are split across two envs.** The base `astrum-dev` image now has
   pure-Python `gymnasium` (so the adapter uses a real `spaces.Box` / `gymnasium.Env`)
   plus `uv`. The heavy, GPU-specific bits (torch + CUDA) live in a **separate uv
   env** under `controllers/rl/` — not the base image — built on demand with
   `make rl-env`. We train with our own PPO, so stable-baselines3 is intentionally
   absent.

---

## 6. File map

| File | What's in it |
|------|--------------|
| `astrum-env/README.md` | The two-layer seam, build/toolchain, parallelism rule. |
| `python/astrum_dogfight/env.py` | `DogfightGymEnv` + the `make_tierN_env` factories. |
| `python/astrum_dogfight/geometry.py` | `AState`, relative geometry (`relative`), lead-pursuit helpers, and `build_observation`. |
| `python/astrum_dogfight/reward.py` | The reward table + episode-boundary predicates. |
| `python/astrum_dogfight/opponent.py` | Scripted Tier-1/2/3 opponents; `wez_fire`. |
| `python/astrum_dogfight/scenario.py` | TOML scenario builders (`tier1_scenario`, `tier3_scenario`). |
| `python/astrum_dogfight/constants.py` | Every tunable: reward magnitudes, WEZ ranges, FSM thresholds, normalizers. |
| `python/test_dogfight_env.py` | 9 tests — and the best worked examples of the API. |
| `docs/dogfight-environment-design.md` | The full design rationale (§8 covers the RL seam). |
