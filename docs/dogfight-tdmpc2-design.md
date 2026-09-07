# Dogfight learner: TD-MPC2 design

**Status:** design locked 2026-08-25; **P0–P5 implemented** the same day in
`controllers/rl/astrum_rl/tdmpc2/` (separate uv env; see that README). This doc is the
spec; it supersedes the from-scratch PPO baseline (`controllers/rl/astrum_rl/ppo.py`)
as the *primary* learner. PPO stays as a cheap sanity baseline.

> **Implementation notes (built vs. this spec):**
> - **Buffer uses plain torch tensors, not `tensordict`** (§4) — one fewer dependency,
>   and the from-scratch goal wants the slicing logic explicit. A flat ring buffer with
>   rejection-sampled horizon windows (see `buffer.py`).
> - **Bounded actions use an affine rescale, not the PPO squash wrapper** (§3.2.2, §6) —
>   TD-MPC2 already acts in `[−1,1]`; a tanh/sigmoid squash would cap `±1` at native
>   `±0.76`, so the jet could never command a full-scale surface / 9 g pull.
> - The empirical acceptance tests for **P4/P5** (beats PPO / the Tier-3 bot; runs in the
>   live controller) require **GPU training**. All code is device-agnostic and was
>   verified for correctness on CPU (unit suite + end-to-end smokes). **GPU is now
>   enabled** (2026-08-26): the host `nvidia-container-toolkit` grants the RTX 5080 to the
>   `dogfight-train` service via CDI (see README), and `rl-tdmpc2-train` was confirmed to
>   run the full seed → update → MPPI-plan loop on `device cuda` (torch 2.11.0+cu128,
>   sm_120) against Tier 3.
> - **Reward + robustness iterations since (2026-08-26):** the first curriculum run
>   plateaued at a passive draw, which drove **reward-v2 → reward-v3 → reward-v4** (env-side;
>   env doc §8.3) and a learner-side **NaN-guard** (§6). Advancement was moved from a step
>   schedule to a **kill-rate competence gate** (§8, env doc §8.5). The gate-led run
>   curriculum-03 got first kills ~150k updates in but the kill-rate stalled at ~2 % because
>   the (then-unbounded) dense tracking reward could out-earn a kill — diagnosed and fixed by
>   **reward-v4**, which caps cumulative tracking at ≈`R_KILL`/3 so a kill strictly dominates
>   a track-draw. curriculum-03 was stopped.
> - **curriculum-04 + guided-fire override (2026-08-27):** the fresh gate-led Tier-1 run on
>   reward-v4 confirmed the cap was *necessary but not sufficient* — over 6.7M steps the return
>   climbed to the tracking budget (~+50) and then plateaued flat, kill-rate stuck at 0–2 %.
>   The policy parks at the (now-capped) track-draw because the safe standing reward still beats
>   the *expected value* of an imperfect, self-initiated shot (waste-fire + return-fire risk):
>   an **exploration/commitment** bottleneck, not a reward-magnitude one. Fix = a collector-side
>   **guided-fire exploration override** (§8.6): early in each tier, when the agent is on a good
>   gun solution (`obs[gun_quality] ≥ q`), force the trigger with probability `p` annealing `p→0`.
>   This seeds the buffer with *fire-in-WEZ* transitions so the value model can learn the shot
>   pays; it is off-policy-native (TD-MPC2 stores and learns from the forced action) so no
>   core-loop change. Implemented behind `--fire-override-p0/-steps/-quality` (default off).
> - **curriculum-05/06 → distance-curriculum stage ladder (2026-08-27):** the override alone
>   still scored no kills (curriculum-05: 100 % bubble-exits — the wall was pursuit-overshoot
>   *flight* geometry under the trigger problem), which drove a close-start bootstrap
>   (`--separation-m`/`--airspeed-kts`) — but a 400 m spawn just *hands* the kill, and the same
>   number means the opposite thing once the gate crosses into Tier 3 (bot on our six). The fix
>   is a **ladder of (tier, separation, gate-metric) stages** (§8.7): close→far on offense (learn
>   to *chase*), far→near on defense (convert with room, then **survive** the point-blank break),
>   with the gate metric sliding kill→survive while reward-v4 stays fixed. The gate now demands
>   **sustained** mastery (≥ bar over a 100-episode window, min 120 played) so a lucky streak
>   can't promote, and `--resume` warm-starts the hard Tier-3 rungs without re-running the offense
>   curriculum. Tier-only `--tier`/`--curriculum` are retired from the parallel trainer. Scoring
>   the Tier-3 head-to-head and the P4/P5 acceptance still follow.

This is the model side of the RL seam described in `docs/dogfight-environment-design.md`
§8. That doc owns the **environment** (obs/action/reward/opponents); this doc owns
the **agent**. Read §8 there first.

> **Reference:** Hansen, Su, Wang, *TD-MPC2: Scalable, Robust World Models for
> Continuous Control* (ICLR 2024). Code: <https://github.com/nicklashansen/tdmpc2>.
> We reimplement from scratch — check that repo's LICENSE before lifting any code
> verbatim; treat it only as a reference for shapes/hyperparameters.

---

## 1. Why TD-MPC2 (not PPO) for this environment

PPO underperformed on comparable air-combat environments elsewhere. Two structural
reasons make TD-MPC2 the better fit *here specifically* — not just a bigger hammer:

1. **Single-env, off-policy collection matches our hard constraint.** JSBSim is
   process-global and not thread-safe (§2.3 of the env design doc), so we cannot run
   the large synchronous parallelism PPO relies on for throughput. TD-MPC2 collects
   from **one environment** with a **1M-transition replay buffer** and does **one
   gradient update per env step** (after a seed warmup). A single env starves PPO;
   it is TD-MPC2's design point.
2. **Robust to reward scale + sparse terminal reward.** Our reward spans −200 (hard
   deck) to +150 (kill) with dense ±0.5/HP and a bounded non-telescoping tracking /
   fire-discipline signal in between (env doc §8.3, reward-v3).
   TD-MPC2 regresses **both reward and value as two-hot categorical distributions
   over a symlog-compressed axis**, which removes the reward-scaling brittleness that
   sinks model-free methods, and its **model-based planning + TD value learning**
   gives far better credit assignment for the sparse kill than a policy-gradient
   return estimate.

Secondary wins: sample efficiency (high replay reuse), one hyperparameter set across
tasks (less tuning), native continuous control, and a clean deployment story
(§9 — the policy prior runs planning-free at 120 Hz).

---

## 2. Algorithm recap (grounded in the reference)

TD-MPC2 learns a **latent world model** and **plans through it** at act time.

### 2.1 World model (all MLPs; `L`=latent_dim, `A`=action_dim, `M`=mlp_dim, `B`=num_bins)

| Head | Signature | Notes |
|---|---|---|
| Encoder | `obs → z ∈ R^L` | 2 layers width `enc_dim`; output normalized by **SimNorm**. |
| Dynamics `next(z,a)` | `[z,a] (L+A) → z' (L)` | 2×`M`; **SimNorm** output → next-state lives in the encoder's normalized latent space. |
| Reward `reward(z,a)` | `[z,a] → B logits` | two-hot target; final layer zero-init. |
| Policy prior `pi(z)` | `z → 2A` | split into `(mean, log_std)`; tanh-**squash**; returns action + log-prob + entropy. |
| Q-ensemble `Q(z,a)` | `[z,a] → B logits`, ×`num_q` | dropout; final layer zero-init. Target set via Polyak. |

**SimNorm**: reshape the latent into groups of `simnorm_dim` (8) and softmax within
each group — bounds the latent, stabilizes the dynamics rollout. **Target Qs** held in
eval mode, updated by `lerp_(·, tau)`. For scalars: `two_hot`/`two_hot_inv` map
between a symlog-compressed scalar and a soft categorical over `num_bins` in
`[vmin,vmax]`; `soft_ce` is the cross-entropy against that soft target.

### 2.2 Planner — MPPI (`plan`)

At each act step (when `mpc=true`):
1. Encode `obs → z`; replicate to `num_samples`.
2. **Seed** `num_pi_trajs` candidate action-sequences by rolling the policy prior
   forward through `next()`; warm-start the rest from the previous step's mean
   (`_prev_mean`, shifted one step), `std = max_std`.
3. For `iterations`: sample `actions = mean + std·ε`, **clamp to [−1,1]**; score each
   with `_estimate_value` (below); take `num_elites` by top-k; reweight by
   `softmax(temperature·(value−max))`; set `mean,std` to the score-weighted
   mean/variance; clamp `std ∈ [min_std,max_std]`.
4. Pick one elite via Gumbel-softmax; return its **first** action (+ exploration noise
   unless eval); store `mean → _prev_mean`.

`_estimate_value(z, actions)`: roll the horizon — `G += discount^t · (1−term) ·
two_hot_inv(reward(z,a))`, advance `z = next(z,a)`; then a terminal bootstrap
`+ discount^H · (1−term) · Q(z, pi(z), 'avg')`.

### 2.3 Update (per env step)

Sample a batch of length-`(H+1)` subtrajectories. Encode the first obs, roll `next()`
forward, and accumulate (each term weighted by `rho^t`):
- **Consistency**: `MSE(z_pred_t, sg[encode(obs_t)])` — the dominant term (coef 20).
- **Reward**: `soft_ce(reward_pred, reward_t)`.
- **Value**: `soft_ce(Q_head, td_target)` summed over the ensemble, where
  `td_target = reward + discount·(1−terminated)·Q(z', pi(z'), 'min', target=True)`.
- **Termination** (only if `episodic`): `BCE(term_logit, terminated_t)`.

Separate **policy update** on detached latents: maximize `scaled_Q + entropy_coef·H`
(Q normalized by a running scale). Then Polyak-update the target Qs.

### 2.4 Reference defaults (`config.yaml`, verbatim)

Planning: `iterations 6 · num_samples 512 · num_elites 64 · num_pi_trajs 24 ·
horizon 3 · min_std 0.05 · max_std 2 · temperature 0.5`.
Training: `steps 10M · batch_size 256 · buffer_size 1M · lr 3e-4 · enc_lr_scale 0.3
· grad_clip_norm 20 · tau 0.01 · rho 0.5`; loss coefs `consistency 20 · reward 0.1 ·
value 0.1 · termination 1`. Discount derived from episode length via
`discount_denom 5`, clamped `[0.95, 0.995]`. Actor: `log_std ∈ [−10,2] ·
entropy_coef 1e-4`. Critic: `num_bins 101 · vmin −10 · vmax +10`. Arch:
`enc_dim 256 · mlp_dim 512 · latent_dim 512 · num_q 5 · dropout 0.01 · simnorm_dim 8`
(default single-task ≈ 5M params). `episodic: false` by default — **we override
(§3, §5).**

---

## 3. Fit analysis for the ASTRUM dogfight

### 3.1 Strong fits
- Single-env/off-policy ⇒ no `AsyncVectorEnv` required for the core loop (multi-process
  collection into a shared buffer stays an *optional* throughput scale-out, §8).
- Two-hot/symlog value+reward ⇒ our −200…+150 reward needs no rescaling
  (`symexp(10) ≈ 22k ≫ 200`, so `vmin/vmax = ±10` covers it).
- Planning + TD value ⇒ credit assignment for the sparse terminal kill; reward-v2's
  dense non-telescoping tracking reward (env doc §8.3, replacing the old policy-invariant
  `K_GEOM·(γΦ′−Φ)` potential shaping) further bootstraps the value function.
- 31-d obs (rates, aspect, closure, WEZ/lead-pip) is close to Markov — good for a
  state world model.

### 3.2 Risks & mitigations (decisions locked here)

1. **Time-scale.** 120 Hz × horizon 3 = 25 ms lookahead — far too short for BFM.
   **Decision:** an **action-repeat wrapper** at ~24 Hz (repeat 5 sim ticks); the long
   horizon is carried by the value function (as in DMC). Also cuts planning cost ~5×.
2. **Trigger discontinuity.** **Decision (locked):** keep the gun trigger as a
   **continuous action in [−1,1], fire at the env boundary when > 0**. The affine action
   map sends the agent's `[−1,1]` trigger to native `[0,1]` (`0 → 0.5`), so trigger `> 0`
   ⇔ native `> 0.5` fires — the "continuous, threshold at env" contract. The two-hot
   reward head absorbs the discrete hit reward. Keep an *auto-fire* ablation (fire on any
   valid WEZ solution) in reserve if trigger learning stalls — not the default.
3. **Termination correctness.** **Decision:** `episodic = true`, and plumb
   **`terminated` (kill / death / hard-deck) separately from `truncated`
   (bubble-exit / time-limit)**. Only `terminated` is a true terminal for the value
   bootstrap and the termination head; `truncated` bootstraps. See §7.
4. **Partial observability.** No recurrence by default. **Decision:** ship Markov
   first; add **obs frame-stacking** (concat last `k`) as the first lever if
   opponent-intent memory is needed. Recurrence is out of scope for v1.
5. **Self-play non-stationarity.** Under self-play the learned dynamics/reward go
   non-stationary and the buffer goes stale. **Decision:** defer to a later phase;
   plan an opponent pool + buffer-refresh cadence then. Against fixed scripted Tiers
   1–3 the world model learns the opponent cleanly.
6. **Compute.** MPPI per step (6·512·3 latent forwards) + one horizon-unrolled update
   is trivially GPU-bound on the RTX 5080; wall-clock is bounded by CPU JSBSim
   stepping. Action-repeat is the main lever; a random-policy throughput benchmark
   (P0) sizes the run.

---

## 4. Integration into ASTRUM

Same `controllers/rl/` uv env — torch (cu128, Blackwell) is already pinned. No new
runtime deps: the buffer is **plain torch tensors** (the reference is TensorDict-based;
we spell the slicing out instead — fewer deps, from-scratch clarity). `pytest` is added
as a dev-group dep for the unit suite.

Reuse the env seam: `astrum_rl._paths` (puts `astrum_env` .so + `astrum_dogfight` on
`sys.path`). TD-MPC2 gets its **own** action map — an **affine `_LinearActionWrapper`**,
*not* the PPO `_SquashActionWrapper` (see the header note / §6). Wrappers in
`astrum_rl/envs.py`:
- **`_LinearActionWrapper(env)`** — expose a `[−1,1]^A` box, affine-map to env ranges.
- **`ActionRepeat(env, n)`** — hold each agent action for `n` sim ticks, sum reward,
  stop early on done; exposes the ~24 Hz control env.
- **`FrameStack(env, k)`** (optional, off by default) — concat the last `k` obs.
- **`wrap_tdmpc(env, …)`** / **`make_tdmpc_env(tier, …)`** — assemble the stack.

New subpackage `controllers/rl/astrum_rl/tdmpc2/` (as built):

| Module | Contents |
|---|---|
| `functional.py` | `symlog/symexp`, `two_hot/two_hot_inv`, `soft_ce`, `clamp_log_std`, `gaussian_logprob`, tanh `squash`, `RunningScale`. |
| `layers.py` | `SimNorm`, `NormedLinear`, `mlp(...)` builder, `encoder(cfg)`, `Ensemble` (a `num_q`-loop; `torch.vmap` is the noted speed upgrade). |
| `world_model.py` | the 6 heads (encoder, dynamics, reward, policy prior, Q-ensemble, termination) + `_target_Qs`; `encode/next/reward/termination/pi/Q/soft_update_target_Qs`. |
| `buffer.py` | episodic **plain-tensor** ring buffer (cap 1M), rejection-sampled horizon slices; stores `terminated` vs `truncated` separately. |
| `agent.py` | `TDMPC2`: `act`/`plan` (MPPI), `_estimate_value`, `_td_target`, `update`, `update_pi`, checkpoint I/O (embeds `cfg`). |
| `config.py` | `TDMPC2Config` dataclass (§5) + derived `resolved_discount`/`resolved_seed_steps`/`bin_size`. |
| `train.py` | single-env off-policy loop (seed warmup + pretrain burst → 1 update/step → curriculum swap → checkpoint). |
| `smoke.py` | end-to-end smoke (tiny model, real JSBSim env): seed → update → plan. |
| `export.py` | `PolicyPrior` + TorchScript export (`.pt` + `.json` metadata) for planning-free deployment. |
| `selfplay.py` | `ScriptedPolicyOpponent`, `OpponentPool`, `make_selfplay_env` (P5 scaffolding). |

Make targets extend the existing pattern: `rl-test`, `rl-bench`, `rl-tdmpc2-smoke`
(CPU, `sim` service), `rl-tdmpc2-train` (GPU `dogfight-train` compose service,
`--profile rl`).

---

## 5. Config — reference default → ASTRUM value

| Param | Reference | ASTRUM | Rationale |
|---|---|---|---|
| `episodic` | false | **true** | Episodes terminate (kill/death/deck). Enables the termination head. |
| `obs_dim` | task | **31** (×`k` if stacked) | `astrum_dogfight.constants.OBS_DIM`. |
| `action_dim` | task | **5** | `[ail,elev,rud,throttle,trigger]`; continuous trigger (§3.2). |
| `action_repeat` | 1–2 | **5** | ~24 Hz control at 120 Hz sim (§3.2). |
| `horizon` | 3 | **3** (try 5) | Value function carries long-term credit; ablate later. |
| `num_bins`/`vmin`/`vmax` | 101 / ±10 | **unchanged** | symlog range already covers ±200. |
| `discount` | derived | **≈0.99–0.995** | From episode length via `discount_denom`; longer scenarios → 0.995. Confirm the derivation formula against source. |
| `model_size` | 5M | **1M → 5M** | 1M for fast iteration on the 5080, scale up once the pipeline is green. |
| `buffer_size` | 1M | **1M** | Trivial RAM (31 floats + 5 action per step). |
| `seed_steps` | derived | **≈5×episode_length** | Random-action warmup; confirm formula against source. |
| `steps` | 10M | **per phase** | P2/P3 short sanity runs; full runs later. |
| `compile` | true | **false → true** | Enable `torch.compile` only after the loop is stable on Blackwell. |

Everything else (planner counts, loss coefs, `rho`, `tau`, `lr`, `simnorm_dim`,
`num_q`, `mlp_dim`, `latent_dim`) starts at the reference defaults — the paper's whole
premise is that one set generalizes; we tune only the env-shaped params above.

---

## 6. Correctness-critical details

> **Hard constraint:** store `terminated` and `truncated` as **separate** booleans on
> every transition. The TD target and termination head use `terminated` only; a
> time-limit / bubble-exit `truncated` must **bootstrap** (it is not a true terminal).
> Collapsing them into one `done` silently teaches the value function that running out
> of time is as bad as dying.

- **Horizon sampling.** The buffer must return contiguous length-`(H+1)` slices that
  do **not** cross an episode boundary (or mask past the boundary). Getting this wrong
  corrupts the consistency/reward rollout.
- **SimNorm** must be applied to the encoder output *and* the dynamics output so
  predicted and encoded latents share one normalized space (the consistency loss
  compares them).
- **two-hot round-trip** (`two_hot_inv(two_hot(x)) ≈ x`) is a required unit test — a
  sign or bin-edge error here silently wrecks value learning.
- **Target Qs** stay in eval mode (dropout off) and update only via Polyak `tau`.
- **Action clamp to [−1,1]** happens inside the planner (and the policy prior tanh-
  squashes); the **affine** `_LinearActionWrapper` then maps `[−1,1] → [low,high]`
  linearly. **Not** a tanh/sigmoid squash: that would cap the agent's `±1` at native
  `±0.76`, making a full-scale surface deflection / 9 g pull unreachable. The buffer
  stores the `[−1,1]` action the agent emitted (what `pi`/`Q` are trained on), never the
  post-map native action.
- **NaN-guard on the update (`agent.py`, added 2026-08-26).** A single divergent batch
  on a hard tier (value targets exploding) can drive one update non-finite and, left
  alone, nuke a multi-hour run — curriculum-02 died exactly this way (last finite update
  80,871 → NaN by 80,908). **Grad clipping cannot save this**: a NaN gradient makes
  `clip_grad_norm_`'s own scale NaN, which then corrupts *every* parameter at
  `optim.step()`. The guard is skip-and-rollback: check `isfinite(total)` pre-step and
  `isfinite(grad_norm)` post-clip; on either failure, skip the step, `load_state_dict`
  back to a periodic (`_nan_snapshot_every` = 100 updates) CPU snapshot of the last
  all-finite weights, count the skip, and log loudly. The learner recovers on the next
  well-formed batch. This is a robustness backstop, not a fix for a genuinely bad reward
  scale (which `vmin/vmax = ±10` already covers).

---

## 7. Phased plan (each phase independently checkpointable)

| Phase | Deliverable | Acceptance test |
|---|---|---|
| **P0** env prep | `ActionRepeat`/`FrameStack` wrappers; transition record with `terminated`+`truncated`; random-policy throughput benchmark. | Wrappers pass step/reset tests; benchmark reports env steps/s (sizes the run). |
| **P1** world model | `functional.py`, `layers.py`, `world_model.py`. | Shape tests for every head; `two_hot` round-trip; SimNorm sums to 1 per group; a dynamics rollout stays finite. |
| **P2** learning (no MPC) | `buffer.py`, `agent.update`, `train.py` with `mpc=false` (act from policy prior). | Tier-1 return trends up over a short run; losses finite; value tracks returns. |
| **P3** planner | `agent.plan` (MPPI). | `mpc=true` matches or beats `mpc=false` on Tier 1 at equal env steps. |
| **P4** curriculum | Tier 1→2→3; tune `horizon`/`action_repeat`/`discount`; enable `torch.compile`. | Beats the PPO baseline and the scripted Tier-3 bot head-to-head. |
| **P5** self-play + deploy | opponent pool + buffer refresh; export policy prior. | Prior-only policy runs in the live Zenoh controller within the 120 Hz budget. |

---

## 8. Throughput scale-out — **implemented** (`train_parallel.py`)

TD-MPC2's UTD=1 tolerates asynchronous data, so the single-env loop's
wall-clock bottleneck (per-step MPPI plan **serialized** with the gradient
update, both time-sharing one GPU → ~49 sps) is broken by running **K collector
processes, one env each** — the process-per-env pattern the JSBSim
process-global constraint already forces — feeding a central learner that runs
GPU updates back-to-back. This is the *only* place `AsyncVectorEnv`-style
multiprocessing re-enters; it is an optimization, not a requirement, and reuses
the exact same model/buffer/update code as the single-env path.

**Architecture (as built):**
- **Collectors** (`_collector_loop`, one `spawn`ed process each): each holds its
  own JSBSim env and a **CPU** copy of the agent, acts **prior-only** (`mpc=false`
  — the throughput default; planning stays on the learner), and streams *whole
  episodes* to the learner over a per-collector `episode_q`. Acting on CPU keeps
  the GPU free for updates. Seeds only the first reset of each env (mirrors
  `train.py`), so initial-condition diversity comes from the distinct per-worker
  seeds rather than re-seeding every episode.
- **Learner** (`run_parallel`, main process): owns the GPU agent + the single
  `ReplayBuffer`, drains episodes via the vectorized `buffer.add_episode`
  (byte-for-byte identical layout to sequential `add`, so `terminated`/`truncated`
  separation and horizon-window sampling are unchanged), runs the seed burst, then
  **1 update per collected env-step** (UTD-scaled) back-to-back. Broadcasts fresh
  weights to each collector every `weight_sync_every` updates as serialized bytes
  over a bounded `weight_q` (maxsize 2); collectors poll and hot-swap between
  episodes.
- **Curriculum** is a ladder of (tier, separation, gate-metric) **stages** (§8.7),
  driven by a shared `stage_value`; the learner advances it on a **sustained-mastery
  competence gate** (`--gate-window`/`--gate-min-episodes`, per-stage bar overridable with
  `--gate-winrate`) whose success metric is per-stage — a *kill* on the offensive/converting
  rungs, mere *survival* on the point-blank defensive rungs — with a per-stage
  `--stage-backstop-steps` step **backstop** so the gate leads (env doc §8.5). Advancement
  is monotonic, one stage at a time; collectors rebuild their env at the new separation on
  change. `--resume` warm-starts a saved run at its recorded stage. Note that `--utd` is a
  **ceiling, not a floor**: the learner is GPU-latency-bound (small batch, sequential horizon rollout,
  Python overhead) at ~79 updates/s while a few prior-only CPU collectors feed thousands
  of steps/s, so the realized UTD pins low (~0.03) and the ceiling never binds — adding
  workers cannot raise it. This costs no wall-clock (the learner never idles), and the
  buffer still churns fast enough to stay near-on-policy.

**Why whole-episode transport, separate processes, and bounded queues:** JSBSim is
process-global and not thread-safe, so collectors *must* be processes (`spawn`, not
`fork`), not threads. Shipping whole episodes (not per-step transitions) amortizes
IPC and lets `add_episode` do a couple of vectorized copies instead of `T+1`
host→device round-trips that would stall the learner. Queues are bounded so a fast
collector can't run the learner out of memory; on shutdown every feeder thread is
`cancel_join_thread()`'d before join to avoid an unflushed-pipe hang.

**Speedup is sublinear in K** — bounded by GPU update latency once collectors
outrun the learner — but removes the plan↔update serialization that caps the
single-env loop. CPU-thread oversubscription is a real trap here: with a tiny model
the learner's default all-core threading is pure dispatch overhead that thrashes
against the collectors, so the learner pins `torch.set_num_threads` to the cores
left over after the collectors.

Smoke: `make rl-tdmpc2-parallel-smoke` (2 collectors, tiny model, real Tier-1
JSBSim, CPU-only, ~40–80 s) exercises the whole pipeline — spawn → seed → burst →
weight broadcast → prior collection → back-to-back updates → checkpoint reload —
and asserts steady-state training actually ran (`updates > seed_steps`), not just
the seed burst. It uses a 2-stage ladder with a tiny backstop so the **stage-change
path** (env rebuild at a new separation + survive-metric logging) runs, then a short
**`--resume`** leg to cover the warm-start (resume load → immediate broadcast → no cold
seed burst). It runs with the guided-fire override on (§8.6) so that path is covered
too. Real runs: `make rl-tdmpc2-train-parallel ARGS="--workers K ..."` (GPU learner +
K CPU collectors).

### 8.6 Guided-fire exploration override (idea 2)

curriculum-04 showed reward-v4's cap was necessary but not sufficient: the policy
climbs to the ~+50 tracking budget and then *parks* there rather than shooting,
because the safe standing reward beats the expected value of an imperfect,
self-initiated shot (waste-fire penalty + return-fire risk). Pure ε-greedy on a
continuous 5-vector almost never produces a *coherent burst while on a solution*,
so the buffer never contains the evidence that "fire-in-WEZ pays" — a chicken-and-egg
exploration trap, not a reward-magnitude one.

The override breaks it collector-side, in `_collector_loop`, with no core-loop change:

- **When:** early in *each* tier (the anneal window restarts on every tier change, so
  each new opponent gets a fresh burst of shooting practice).
- **Trigger:** the agent's chosen action is left untouched *except* that, when it is
  looking at a good gun solution (`obs[gun_quality] ≥ --fire-override-quality`, read
  stack-safely as `obs[-2]`), the trigger is forced on (`action[4] = 1.0`) with
  probability `p = p0·(1 − steps_in_tier / anneal)`.
- **Anneal:** `p` decays linearly `p0 → 0` over `--fire-override-steps` **global**
  control-steps (split `/workers` per collector, matching the `--steps`/`--curriculum`
  step unit), after which the policy is fully on its own.
- **Why it's sound for TD-MPC2:** the forced action is what gets *stored* in the
  episode, and TD-MPC2 is off-policy — the world model and value head learn from the
  forced fire-in-WEZ transitions exactly as from any other data. There is no
  importance-sampling correction to get wrong (unlike an on-policy method). It is
  strictly *data-shaping*, not reward- or policy-shaping.

Flags (default **off** — `--fire-override-p0 0` disables it, preserving prior behavior):
`--fire-override-p0` (initial force probability), `--fire-override-steps` (global anneal
horizon, per tier), `--fire-override-quality` (the `gun_quality` threshold, ~0.6). This
is the "auto-fire ablation kept in reserve" of §3.2, activated — but as an *annealed
exploration prior*, not a hard-wired policy, so the final learned policy still owns the
trigger.

**curriculum-05 (2026-08-27): the override alone was still not enough — and why.** A
gate-led Tier-1 run with the aggressive override (p0 0.5) never scored a kill in ~1900
episodes; a greedy eval of the checkpoint (`eval_policy`, 10 eps) showed the real failure:
**100 % bubble exits**, ~4–14 HP token damage, 0 taken. The policy makes one high-closure
pass, grazes the bandit, overshoots, and coasts out of the 15 km bubble — it never *holds*
a firing solution, so there were no successful bursts for the override to reinforce. The
wall was the **flight** geometry (pursuit-overshoot / won't-re-engage), sitting under the
trigger problem the override targets.

**Fix — a close-start Tier-1 bootstrap (curriculum-06).** The Tier-1 scenario already
spawns the learner on the dummy's six, co-speed; only the 2 km default separation (vs the
1.2 km gun envelope) made it hard. Two `train_parallel` flags now override the start
geometry: `--separation-m` and `--airspeed-kts` (0 = keep the tier default; they forward
straight to `tier1_scenario`/`tier3_scenario`). Starting at **400 m** puts the learner in
an opening solution (`gun_quality ≈ 0.667`, verified), and a validation confirmed that
force-firing while holding level there **kills every time** (opp HP → 0, `event=="kill"`)
at any throttle. So a single held pass now converts to a full kill: the override generates
real kill transitions, the value model learns "solution → fire → +150", and separation can
be **widened again** (a later, wider-start run) once close-range killing is solid. Caveat:
`--separation-m` currently applies to whatever tier is active, so revisit it before the
gate promotes past Tier 1. **§8.7 resolves that caveat by making separation per-stage.**

### 8.7 Distance-curriculum stage ladder

curriculum-06 exposed a confound that a bare tier schedule can't express: **separation is
what actually gates the skill, and its *meaning flips* between the offensive and defensive
tiers.** For Tier 1/2 the learner spawns on the bandit's six and separation is how far it
must **close** — a 400 m spawn hands it a point-blank kill (validated: force-fire while
holding level there kills every time), while a real *chase* only has to be learned farther
out. For Tier 3 the bot opens *behind* the learner and separation is **threat proximity** —
2500 m is room to convert, 450 m is a lethal ambush that must first be *survived*. A single
`--separation-m` applied to "whatever tier is active" (the §8.6 caveat) is therefore wrong
the moment the gate crosses into a tier where the number means the opposite thing.

**The ladder** (`_CHASE_THEN_DEFEND` in `train_parallel.py`, built by `build_stages`) makes
each rung an explicit **(tier, separation, gate-metric, bar)** stage. It climbs **close→far
on offense** (learn to run the bandit down), then **far→near on defense** (convert with
room, then survive the point-blank break):

| stage | tier | sep (m) | metric | bar | rationale |
|------:|:----:|--------:|:-------|----:|:----------|
| 0–4   | 1 | 400 → 700 → 1100 → 1600 → 2000 | kill | 0.70 | passive dummy: master the point-blank kill, then *widen* to learn the chase |
| 5–8   | 2 | 700 → 1100 → 1600 → 2000 | kill | 0.70 | *breaking* bandit: same close→far, now the target maneuvers |
| 9–11  | 3 | 2500 → 1600 → 1100 | kill | 0.60 / 0.55 | bot on the six with room: convert the defensive engagement to a kill |
| 12–13 | 3 | 700 → 450 | **survive** | 0.85 | point-blank ambush: **defeat the shot** — a kill may not be physically reachable, so survival is the milestone |

**The gate metric slides with separation while the reward stays fixed.** Reward-v4 is
unchanged (kill = +150 is always the biggest pull), but *what clears a stage* is
per-stage: a **kill** on the offensive/converting rungs, mere **survival** (any
non-loss terminal — a timeout draw or a kill, i.e. `event ∉ {death, killed, trade,
hard_deck, left_bubble}`) on the two point-blank defensive rungs, where demanding a kill
would wedge the run on a geometry the learner can't yet win from. So the agent is always
*pulled* toward the kill by the reward, but is *promoted* on the achievable milestone at
each distance.

**Sustained mastery, not a lucky streak (the consistency requirement).** A stage clears
only when its success-rate holds **≥ bar over a large recent window** *and* a minimum
number of episodes have been played (defaults `--gate-window 100`, `--gate-min-episodes
120`; bars as tabled). A big window at a high bar makes a hot tail mathematically
insufficient — e.g. a 40-episode win streak riding on 80 prior losses is only 40 % over
the last 100, well under 0.70 (unit-tested in `tests/test_curriculum_gate.py`). A per-stage
`--stage-backstop-steps` floor still forces progress if a gate is genuinely slow, so no
stage can stall the run forever; advancement is `max(backstop, gate)`, monotonic, one rung
at a time. On promotion the per-stage win log is cleared so the next rung's mastery is
measured from scratch.

**Resume.** `--resume <ckpt>` warm-starts model **and** both optimizers, picking up at the
checkpoint's recorded `stage_idx` (override with `--start-stage`); backstops rebase to the
resume stage so a resumed run budgets only the remaining rungs. The resumed policy is
broadcast to collectors immediately, skipping the cold seed/pretrain burst. Every stage
promotion also drops a `…_stageNN_tT_SEPm.pt` milestone checkpoint recording the *next*
stage, so the hard Tier-3 rungs can be iterated on without re-running the T1/T2 offense
curriculum each time.

---

## 9. Deployment

Train with MPC; two deployment options into the live 120 Hz Zenoh flight controller:
1. **Policy prior only** (`mpc=false`) — a single `pi(z)` forward per step; easily
   real-time; some performance cost vs planning.
2. **Reduced MPC** — smaller `num_samples`/`iterations` + action-repeat to fit budget.
Export the trained `WorldModel` weights; a thin controller loads them and runs `act`.

---

## 10. Open questions
- ~~Exact reference formulas for derived `discount` and `seed_steps`.~~ **Resolved:**
  `discount = clamp((L/d − 1)/(L/d), 0.95, 0.995)` with `d = discount_denom = 5`, and
  `seed_steps = max(1000, 5·episode_length)` — matching the published `config.py`.
  `train.py` sets `episode_length` from the env (`max_time_s·rate_hz/action_repeat`).
- Whether Markov obs suffices or frame-stacking is needed — **still empirical** (needs
  GPU training runs, P4); the `FrameStack`/`frame_stack` plumbing is in place, off by
  default.
- Self-play opponent-pool design — **scaffolded** (`selfplay.py`: `OpponentPool` with
  uniform/recency-biased sampling); the training-loop cadence (when to snapshot into the
  pool, buffer-refresh) is the remaining empirical choice, left out of `train.py`.

## 11. References
- Env seam & reward: `docs/dogfight-environment-design.md` §8.
- Training env & PPO baseline: `controllers/rl/README.md`, `astrum_rl/ppo.py`.
- TD-MPC2 paper + code (see the callout in the header).
