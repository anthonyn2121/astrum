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
| **1. Rust sim seam** | `astrum-sim/src/env.rs` (`SimEnv`) → PyO3 binding `astrum-env/src/lib.rs` (`astrum_env.Env`, `astrum_env.BatchEnv`) | A pure **state-in / state-out** physics sim. One `step()` = one JSBSim tick. Knows nothing about rewards or observations. | Rarely — it's the stable floor. |
| **2. Python trainer adapter** | `crates/astrum-env/python/astrum_dogfight/` | Adds everything RL needs: the **observation vector**, the **reward**, the **episode boundaries**, and the **scripted opponent**. | **Yes — this is where you live.** Pure numpy Python. |

The key idea: the Rust sim hands you *raw absolute state* (positions, attitudes,
hit points, ammo). Turning that into "am I on his six, inside gun range, with a lead
solution?" is the **trainer's** job, done in `geometry.py`. The sim deliberately
omits reward and observation shaping so the physics stays reusable and testable
from `cargo test` with no Python involved.

Read `crates/astrum-env/README.md` first — it is the crispest statement of this
split, the binding API, and the build/threading notes.

---

## 2. First 30 minutes: build it and watch an episode

**Step 0 — build the extension.** `astrum_env` is a compiled Rust `.so`;
`scripts/build-astrum-env.sh` builds it (`cargo build --release -p astrum-env`)
and prints the staging directory to put on `PYTHONPATH`.

**Step 1 — run the Python tests** (controllers + the adapter's 15 tests + the
binding smoke check, no gymnasium required):

```bash
make test-python
```

or, by hand in a dev shell:

```bash
S=$(bash scripts/build-astrum-env.sh)
export PYTHONPATH="$S:crates/astrum-env/python:$PYTHONPATH"
pytest -q crates/astrum-env/python && python crates/astrum-env/python/smoke.py
```

If they pass, the whole stack works. `test_dogfight_env.py` is also the best
worked example in the repo: `_rollout` / **`test_seeded_rollouts_are_identical`**
is a complete minimal episode you can copy, and the `test_tier3_*` tests show the
shooting opponent end-to-end.

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
  can't depart the jet. This trips up everyone once — see the design doc §2.4 and §8.
- `trigger` is a **learned fire gate**, thresholded at `≥ 0.5` into a real M61A1
  gun burst. The policy learns *when* to shoot; there is **no auto-aim**.

### Observation — `float32[31]`

Egocentric and normalized (built by `geometry.build_observation`). You do **not**
assemble this by hand — the env returns it — but know the layout so you know what
the policy sees:

- **14 own-state** — true airspeed, height above the hard deck (HAE), `sin/cos(φ)`,
  `sin/cos(θ)`, body rates p/q/r, α, β, load factor Nz, hit points, ammo.
- **14 relative-geometry** — range, `sin/cos` of the bandit's bearing off your nose
  (azimuth + elevation), aspect angle, heading-crossing angle, closure, relative
  altitude, bandit airspeed / hit points / alive-flag.
- **3 WEZ / gun aids** — in-range flag, gun-solution quality ∈ [0,1], and the
  cosine of the lead error.

Absolute heading ψ is deliberately dropped so a learned policy generalizes across
compass directions.

### Reward and episode flags

Computed by `reward.py`. Dense terms accrue every step; terminal events have strict
precedence:

| Event | Reward | Flag | `info["event"]` |
|-------|-------:|------|-----------------|
| Hard-deck bust (< 300 m HAE; there is no terrain model) | −200 | terminated | `hard_deck` |
| Kill (bandit dies) | +100 | terminated | `kill` (or `trade`) |
| Death (learner dies) | −100 | terminated | `killed` |
| Bubble exit (> 15 km, or over ceiling) | −50 | truncated | `left_bubble` |
| Damage dealt / taken | ±0.5 / HP | — | `damage_dealt` / `damage_taken` |
| Geometry shaping | `K_GEOM·(γΦ′−Φ)` | — | — |
| Time | small negative each step | — | — |

`info["event"]` tells you which terminal fired; `info["phi"]` is the current
geometry potential. Hard-deck wins over everything, then kill/death, then bubble.

`info["events"]` (plural) is the sim's own record of the tick: a list of dicts such
as `{"kind": "hit", "shooter_id": 2, "target_id": 1, "hit_points": 96.0, "pos_enu": [...], ...}`,
`kill` (with `killer_id`), `trigger-pressed` / `trigger-released` (with `rounds_fired`)
and `mode-change`. Use it to debug a reward or score an engagement: unlike the HP
deltas behind `damage_dealt` / `damage_taken`, it says who hit whom and where. The
kinds and fields are listed in the README's "Events".

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
alt_hae_m=5500.0, tas_mps=200.0, max_time_s=45.0)` (SI units, height above the
ellipsoid, true airspeed; see `scenario.py` for the full list).

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
3. **Only then, learning.** This needs a training framework installed (see §5)
   before you can `import stable_baselines3`.

---

## 5. Two things to know before you scale

1. **Parallelism: threads are fine, `BatchEnv` is cheapest.** The JSBSim wrapper
   (`astrum-fdm`) owns a process-wide lock around JSBSim's global initialisation,
   so `astrum_env.Env` is `Send` and releases the GIL during construction, `reset`
   and `step` — several envs can run on Python threads concurrently.
   `astrum_env.BatchEnv` steps K envs in parallel (rayon) inside one call.
   Process-level vectorisation (`SubprocVecEnv` / `AsyncVectorEnv`) also works.
   Keep `sim.threads = 1` (the default) in batched scenarios — per-env FDM
   workers would oversubscribe the cores the batch already uses.
2. **gymnasium and stable-baselines3 are not in the dev image**
   (`docker/requirements.txt` ships numpy/scipy only). The adapter is numpy-only
   and subclasses `gymnasium.Env` *only if it's importable*, so it runs offline
   today, but training needs those deps added to the image first.

---

## 6. File map

| File | What's in it |
|------|--------------|
| `astrum-env/README.md` | The two-layer seam, binding API, build and threading notes. |
| `python/astrum_dogfight/env.py` | `DogfightGymEnv` + the `make_tierN_env` factories. |
| `python/astrum_dogfight/geometry.py` | `AState`, relative geometry (`relative`), lead-pursuit helpers, and `build_observation`. |
| `python/astrum_dogfight/reward.py` | The reward table + episode-boundary predicates. |
| `python/astrum_dogfight/opponent.py` | Scripted Tier-1/2/3 opponents; `wez_fire`. |
| `python/astrum_dogfight/scenario.py` | TOML scenario builders (`tier1_scenario`, `tier3_scenario`). |
| `python/astrum_dogfight/constants.py` | Every tunable: reward magnitudes, WEZ ranges, FSM thresholds, normalizers. |
| `python/test_dogfight_env.py` | 15 tests — and the best worked examples of the API. |
| `python/smoke.py` | Binding smoke check (`Env`, `BatchEnv`, autopilots, fire control). |
| `docs/dogfight-environment-design.md` | The full design rationale (§8 covers the RL seam). |
