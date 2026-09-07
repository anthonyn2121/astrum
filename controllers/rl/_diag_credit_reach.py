"""D5 — critic CALIBRATION: how far back does this critic price a kill it actually achieves?

    cd controllers/rl && uv run python _diag_credit_reach.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --cell chase --aspects 30,55 --episodes 6 --jitter 0.5 --seed 0

WHAT IS MEASURED

Everything is read on the checkpoint's own GREEDY trajectory (`eval_mode=True`, so the rollout is
deterministic given the seed and the discounted return-to-go is EXACT, not a sample). On a greedy
roll the flown action IS the actor mean, so `Q(z,a)` and `Q(z,pi(z))` coincide by construction (up to
the `[-1,1]` clamp) — both are printed anyway, because a divergence would mean the clamp is biting.

  * `Qsa`  = `model.Q(z_t, a_t, return_type="avg")` — the ensemble MEAN over all 5 heads.
    **Never `"min"`**: that is `q[randperm(num_q)[:2]].min(0)` (`world_model.py:105`), a fresh random
    2-of-5 subsample per call — non-reproducible AND deliberately pessimistic, so a "gap" read off it
    would be part pessimism bonus. `two_hot_inv` un-symlogs it, so `Qsa` is in reward units.
  * `Gmc`  = the exact discounted return-to-go `sum_k gamma^k r_{t+k}` over the SAME trajectory,
    using the env's own per-control-step reward (the ActionRepeat sum of 5 sim ticks — exactly what
    `train_parallel.py:1133` stores in the buffer), at the gamma READ OUT OF the saved cfg.
  * `Gboot` = `Gmc + gamma^(T-t) * Q(z_T, pi(z_T), "avg")` on TRUNCATED (timeout) episodes only.
    This is the comparison the learner is actually fitted to: `_td_target` (`agent.py:298-304`)
    multiplies the bootstrap by `(1 - terminated)`, NOT `(1 - done)`, so a timeout is a CONTINUING
    state for this critic and the bare `Gmc` of a timeout episode is biased low by construction.
    On kill/deck/death episodes `terminated=1` at the last step, so `Gboot == Gmc` exactly.
  * `gap`  = `Qsa - G`. **SIGN CONVENTION: negative gap = the critic UNDER-prices what the policy
    actually goes on to collect**, i.e. the outcome has not propagated back that far.

Bucketing is by TIME-TO-TERMINAL, `(T - t) * dt` with `dt = action_repeat / rate_hz = 41.7 ms`
(24 Hz; action_repeat 5 at 120 Hz). Every bucket is quoted in BOTH seconds and control steps
(0.5 s = 12 steps, 1 s = 24, 2 s = 48, 4 s = 96, 8 s = 192) because three prior candidates in this
program mis-sized their own intervention 5x by assuming 208 ms. Every curve is split by TERMINAL
TYPE (`eval_policy._classify`: kill / deck / death / timeout), never pooled.

THREE DISAMBIGUATING READS, because a raw Q-vs-return gap alone cannot tell you WHY

A gap can grow with time-to-terminal for three different reasons and only one of them is a credit
REACH problem. These separate them, and all three use code that already exists:

  1. `td1` = `r_t + gamma*(1-terminated_t)*Q_target(z_{t+1}, pi_mu(z_{t+1}), "avg") - Qsa_t`, the
     ONE-STEP Bellman residual of the critic against its own target network — i.e. exactly the
     quantity `agent.update` drives to zero, evaluated on-policy. If `td1 ~ 0` at every
     time-to-terminal while the MC gap is large and grows, the per-step operator is converged and the
     error is ACCUMULATING down the bootstrap chain: that is the credit-reach signature. If `td1` is
     itself large at small time-to-terminal, the critic has not even fitted the local reward and
     bootstrap length is not the story. (Training's target uses `"min"` of 2 random heads; `"avg"` is
     used here for reproducibility and the min-vs-avg pessimism is quantified separately, below.)
  2. `rhat` = `two_hot_inv(model.reward(z_t, a_t))` vs the realized `r_t`. The reward head and the Q
     heads are trained by the same `soft_ce` on the same latents (`agent.py:329-335`). If `rhat`
     tracks the terminal reward spike but `Qsa` does not, the failure is specific to the VALUE
     regression, not to the encoder or to reward rarity.
  3. `pess` = `Q_target(z,pi,"avg") - Q_target(z,pi,"min")`, the per-backup pessimism the `"min"`
     target injects. Pooled over thousands of steps its mean is stable even though each draw is
     random. This is the mechanism that would make an otherwise-converged critic under-price a
     distant outcome by a factor that compounds with the number of backups, so it must be sized
     before a gap is blamed on horizon.

  Free bonus: `_estimate_value` (`agent.py:270-291`) — the H-step gamma-discounted rollout of the
  LEARNED dynamics + reward head with a terminal Q bootstrap, driven by the model's own pi-mean
  chain — against `Q(z, pi(z))` on the same states. `cfg.mpc` is False in every shipped checkpoint so
  this path is dead in training, but it is callable; if `Vest == Vpi` the model+critic carry no extra
  multi-step information and n-step has nothing to add. The termination head's logit at `z_1` is
  printed alongside because `_estimate_value` is terminal-aware only if that head ever fires.

WHAT IT EXPLICITLY DOES NOT MEASURE

* **This is NOT BFM evidence and must never be quoted as such.** Reward-term sums and
  discounted-return traces along a trajectory are blacklisted as behaviour evidence
  (`bfm_detect.py:37-48`) because a return trace says what the reward paid, not what the jet flew.
  The return is used here ONLY to calibrate the critic against ITSELF — Q vs the return the same
  greedy policy realizes on the same trajectory — which is a self-consistency check on the learner,
  not a claim about maneuvering. A future reader must not read these curves as a behaviour result.
* It does not measure off-policy calibration. Q evaluates pi; the return here is generated by that
  same pi. A gap against a SCRIPTED stick's MC return would be guaranteed and would mean
  "off-policy", not "bootstrap too short" — that category error is why the original form of this
  candidate was closed (handoff C-6). This form has no such gap by construction.
* It does not measure the critic on the collection (exploring) distribution, where the TD updates
  actually happen. Greedy is the out-of-band read this program judges on, and the one that makes the
  return exact.
* `_estimate_value` bootstraps with `model.pi(z)[1]` — a SAMPLE, not the mean (`agent.py:290`). At the
  actor's measured sigma that is close to the mean but not identical, so a few tenths of `Vest - Vpi`
  is sampler noise, not model disagreement. Reported as-is: the point is to test the shipped
  function, not a cleaned-up variant.
* It says nothing about whether a LONGER bootstrap would fix any gap it finds. n-step over this
  uniform off-policy buffer has no importance weight anywhere (`buffer.py`), so it would shift the
  fixed point toward Q^mu; that is a separate, deferred question (handoff D6 judging note).

REWARD-MODE MATCHING (the detail that would silently void this measurement)

The return-to-go is only comparable to Q if it is computed under the reward the critic was TRAINED
on. All three checkpoints record `provenance.outcome_mode = True`, and no existing verdict harness
passes `outcome_mode` (they judge ground-truth HP, which does not care). This harness reads each
checkpoint's own provenance and sets `outcome_mode` per checkpoint. `outcome_mode` touches
`reward.py` only — not termination, not `info["event"]`, not the opponent — so the trajectory and the
kill count are bit-identical to the recorded verdicts, and matching seeds should reproduce them.

The chase cell is IN distribution for c96: it launched with `--rehearse-workers 2` of 6 workers, and
the rehearsal stage (`train_parallel.py:4762-4779`) is exactly this geometry — native T3, sep 700 m,
aspect band [0,55]deg, `max_time_s=60`, kill-gated. So roughly a third of c96's buffer is this cell.

⚠ Residual confound, stated up front and NOT fixable here: B1 (2026-09-05) changed `reward.py`'s WEZ
payout to the sim's own p_hit curve. c96 was trained under the current `reward.py`; **c95 and c87 were
not**. So for the two D6 controls the MC return is computed under a reward their critic never saw,
which inflates the LEVEL of their gap. It should not systematically distort how the gap SCALES with
time-to-terminal, which is the quantity of interest, but the controls' levels are not comparable to
c96's and are reported only as "does the shape appear in an old checkpoint too".

PRE-REGISTERED INTERPRETATION (handoff section 6, D5)

* **Expected / null:** the critic is roughly calibrated — |gap| under ~30% of |G| at 2 s, growing
  modestly out to 4-8 s — and `Vest` agrees with `Vpi`. Outcome: bootstrap length stays closed, and
  the model+critic have no multi-step information n-step could exploit.
* **The wrong-way outcome that would matter:** a large, systematically NEGATIVE gap at 3-6 s on
  trajectories that DO terminate in a kill. That would reopen bootstrap length as a live defect
  against `tdmpc2-credit-horizon-ceiling.md:65-67` (~25 Bellman backups per transition => credit
  propagates ~25 control steps ~ 1 s of flight, strictly 1-step TD, no n-step, no lambda, no
  prioritisation). A dedicated 3-6 s x kill-only cell is printed for exactly this test.
* The interesting quantity is HOW THE GAP SCALES with time-to-terminal, not its level. A constant
  offset across buckets is a value-scale artifact; a gap that shrinks toward zero only as
  time-to-terminal shrinks is a credit-reach failure; a gap that is already full-size in the last
  0.5 s before the terminal is neither — it is a failure to fit the terminal reward at all, and
  `td1`/`rhat` are there to tell those apart.

D6: the FIRST checkpoint is the run, the rest are controls, all in ONE invocation on the same env
build / seeds / thresholds. Runs on CPU so it never contends with a training run.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2 import functional as fn
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

# Time-to-terminal bucket upper edges, in SECONDS. Quoted in steps too (dt = 41.7 ms).
TTT_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0, math.inf)
STEP_S = 5.0 / 120.0  # one control step at the trainer's ar5 @ 120 Hz
# The pre-registered wrong-way window: a large negative gap HERE on kill episodes reopens
# bootstrap length.
WRONGWAY_LO, WRONGWAY_HI = 3.0, 6.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D5: on-policy critic calibration vs time-to-terminal.")
    p.add_argument(
        "--checkpoints",
        required=True,
        help="comma list; RUN FIRST, D6 controls after (c95, c87). Same invocation, same seeds.",
    )
    p.add_argument("--cell", default="chase", choices=("chase", "headon"))
    p.add_argument("--aspects", default="30,55", help="chase cell nominal start aspects, deg")
    p.add_argument("--laterals", default="1400", help="headon cell lateral offset, metres")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--separation", type=float, default=None, help="default 700 chase / 3000 headon")
    p.add_argument("--max-time-s", type=float, default=None, help="default 60 chase / 180 headon")
    p.add_argument("--me-dalt-ft", type=float, default=None, help="default 0 chase / 2000 headon")
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in this program")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument(
        "--reward-mode",
        default="auto",
        choices=("auto", "outcome", "dense"),
        help="auto = read each checkpoint's provenance.outcome_mode (the correct choice: the MC "
        "return must be under the reward the critic was trained on)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _cell_defaults(args) -> None:
    d = (
        dict(separation=700.0, max_time_s=60.0, me_dalt_ft=0.0)
        if args.cell == "chase"
        else dict(separation=3000.0, max_time_s=180.0, me_dalt_ft=2000.0)
    )
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def _cells(args) -> list[tuple[str, float]]:
    if args.cell == "chase":
        return [(f"asp{float(a):.0f}", float(a)) for a in args.aspects.split(",") if a.strip()]
    out = []
    for x in args.laterals.split(","):
        if x.strip():
            lat = float(x)
            out.append((f"lat{lat:.0f}m", math.degrees(math.asin(min(1.0, lat / args.separation)))))
    return out


@torch.no_grad()
def _pi_rollout_actions(model, z: torch.Tensor, horizon: int) -> torch.Tensor:
    """`(H, B, A)` action sequence from the model's own pi MEAN chain through learned dynamics.

    Mean (not sample) so the sequence fed to `_estimate_value` is deterministic; `plan()` seeds with
    the sample (`agent.py:225`), which would add sampler noise to a calibration number.
    """
    acts = []
    _z = z
    for _ in range(horizon):
        mu = model.pi(_z)[0]
        acts.append(mu)
        _z = model.next(_z, mu)
    return torch.stack(acts, dim=0)


@torch.no_grad()
def _roll(agent, env, args, episodes: int, seed0: int) -> list[dict]:
    """Greedy episodes; then batch-evaluate every critic/model quantity on the stored trajectory.

    Flying and measuring are split on purpose: the flight must go one `act()` at a time, but every
    head is an MLP with no batch-coupled layer, so evaluating them over the whole episode at once is
    numerically identical and far cheaper. `obs` is stored T+1 long so `z_{t+1}` (the TD target's
    latent) and the truncation bootstrap `Q(z_T, pi(z_T))` come from the same encode call.
    """
    dt = args.action_repeat / args.rate_hz
    gamma = agent.discount
    H = agent.cfg.horizon
    M = agent.model
    eps_out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0 = True
        obs_hist = [np.asarray(obs, dtype=np.float32)]
        acts, rews, terms = [], [], []
        event = None
        terminated = truncated = False
        while True:
            a_t = agent.act(obs, t0=t0, eval_mode=True)
            t0 = False
            a = a_t.numpy()
            acts.append(np.asarray(a, dtype=np.float32))
            obs, r, terminated, truncated, info = env.step(a)
            obs_hist.append(np.asarray(obs, dtype=np.float32))
            rews.append(float(r))
            terms.append(float(terminated))
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break

        OB = torch.as_tensor(np.stack(obs_hist), dtype=torch.float32, device=agent.device)
        A = torch.as_tensor(np.stack(acts), dtype=torch.float32, device=agent.device)
        z_all = M.encode(OB)  # (T+1, latent)
        z, z_next = z_all[:-1], z_all[1:]
        mu = M.pi(z)[0]
        mu_next = M.pi(z_next)[0]
        q = M.Q(z, A, return_type="avg").reshape(-1)
        qpi = M.Q(z, mu, return_type="avg").reshape(-1)
        rhat = fn.two_hot_inv(M.reward(z, A), agent.cfg).reshape(-1)
        qt_avg = M.Q(z_next, mu_next, return_type="avg", target=True).reshape(-1)
        qt_min = M.Q(z_next, mu_next, return_type="min", target=True).reshape(-1)
        vest = agent._estimate_value(z, _pi_rollout_actions(M, z, H)).reshape(-1)
        termlogit = M.termination(M.next(z, mu)).reshape(-1)
        q_end = float(M.Q(z_all[-1:], M.pi(z_all[-1:])[0], return_type="avg").reshape(-1)[0])

        r = np.asarray(rews, dtype=np.float64)
        tm = np.asarray(terms, dtype=np.float64)
        T = r.size
        gmc = np.zeros(T, dtype=np.float64)
        acc = 0.0
        for i in range(T - 1, -1, -1):
            acc = r[i] + gamma * acc
            gmc[i] = acc
        if truncated and not terminated:
            gboot = gmc + q_end * gamma ** (T - np.arange(T, dtype=np.float64))
        else:
            gboot = gmc.copy()

        qn = qt_avg.cpu().numpy().astype(np.float64)
        qv = q.cpu().numpy().astype(np.float64)
        eps_out.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "T": T,
                "dur_s": T * dt,
                "ttt": (T - np.arange(T, dtype=np.float64)) * dt,
                "q": qv,
                "qpi": qpi.cpu().numpy().astype(np.float64),
                "rhat": rhat.cpu().numpy().astype(np.float64),
                "vest": vest.cpu().numpy().astype(np.float64),
                "termlogit": termlogit.cpu().numpy().astype(np.float64),
                "pess": (qt_avg - qt_min).cpu().numpy().astype(np.float64),
                "td1": r + gamma * (1.0 - tm) * qn - qv,
                "r": r,
                "gmc": gmc,
                "gboot": gboot,
                "q_end": q_end,
                "ret": float(r.sum()),
            }
        )
    return eps_out


def _p(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def _cat(eps: list[dict], key: str) -> np.ndarray:
    return np.concatenate([e[key] for e in eps]) if eps else np.zeros(0)


def _buckets(eps: list[dict]) -> list[tuple[str, str, np.ndarray]]:
    """`(label_s, label_steps, boolean mask over the pooled steps)` per time-to-terminal bucket."""
    ttt = _cat(eps, "ttt")
    out, lo = [], 0.0
    for hi in TTT_EDGES:
        lab_s = f"{lo:g}-{hi:g}s" if math.isfinite(hi) else f">{lo:g}s"
        lo_st = int(round(lo / STEP_S))
        lab_st = f"{lo_st}-{int(round(hi / STEP_S))}st" if math.isfinite(hi) else f">{lo_st}st"
        out.append((lab_s, lab_st, (ttt > lo) & (ttt <= hi)))
        lo = hi
    return out


def _print_curve(eps: list[dict], key: str, indent: str = "     ") -> None:
    gg, qq = _cat(eps, key), _cat(eps, "q")
    td1, rh, rr, pe = _cat(eps, "td1"), _cat(eps, "rhat"), _cat(eps, "r"), _cat(eps, "pess")
    print(
        f"{indent}{'ttt':>9} {'steps':>9} {'n':>6} {'medG':>9} {'medQ':>9} {'gap p50':>9} "
        f"{'gap p10':>9} {'gap p90':>9} {'|gap|/|G|':>9} {'P(Q<G)':>7} | {'medr':>7} "
        f"{'medrhat':>8} {'td1 p50':>8} {'td1 p90':>8} {'pess':>6}"
    )
    for lab_s, lab_st, m in _buckets(eps):
        if not m.any():
            print(f"{indent}{lab_s:>9} {lab_st:>9} {0:>6}  (empty)")
            continue
        gap = qq[m] - gg[m]
        mg = _p(np.abs(gg[m]), 50)
        rel = _p(np.abs(gap), 50) / mg if mg > 1e-9 else float("nan")
        print(
            f"{indent}{lab_s:>9} {lab_st:>9} {int(m.sum()):>6} {_p(gg[m], 50):>9.2f} "
            f"{_p(qq[m], 50):>9.2f} {_p(gap, 50):>9.2f} {_p(gap, 10):>9.2f} {_p(gap, 90):>9.2f} "
            f"{rel:>9.2f} {float((gap < 0).mean()):>7.2f} | {_p(rr[m], 50):>7.2f} "
            f"{_p(rh[m], 50):>8.2f} {_p(td1[m], 50):>8.2f} {_p(td1[m], 90):>8.2f} "
            f"{_p(pe[m], 50):>6.2f}"
        )


def _print_terminal(eps: list[dict], indent: str = "     ") -> None:
    """The terminal transition on its own, vs every other step.

    At `t = T-1` of a TERMINATED episode `terminated = 1`, so the TD target collapses to `r_T`
    exactly — no bootstrap, no horizon, nothing to reach back through. `td1` there is therefore the
    critic's pure regression error on the terminal reward, and it is the one place a horizon
    explanation cannot apply. A median `|td1|` that is small everywhere except here localises the
    entire Q-vs-return gap to a single point defect.
    """
    term = [e for e in eps if e["terminated"]]
    if not term:
        print(f"{indent}TERMINAL TRANSITION: no terminated episodes in this cell (all truncated)")
        return
    rT = np.array([e["r"][-1] for e in term])
    qT = np.array([e["q"][-1] for e in term])
    tdT = np.array([e["td1"][-1] for e in term])
    rhT = np.array([e["rhat"][-1] for e in term])
    body = np.concatenate([e["td1"][:-1] for e in term if e["T"] > 1])
    print(
        f"{indent}TERMINAL TRANSITION (terminated=1 => target IS r, no bootstrap): n={rT.size} "
        f"med r_T {np.median(rT):+.2f} med Q_T {np.median(qT):+.2f} "
        f"med td1_T {np.median(tdT):+.2f} [min {tdT.min():+.2f} max {tdT.max():+.2f}]"
    )
    print(
        f"{indent}   REWARD head on the SAME (z_T, a_T): med rhat_T {np.median(rhT):+.2f} "
        f"vs r_T {np.median(rT):+.2f} => err {np.median(rhT - rT):+.2f} "
        f"({np.median(rhT) / np.median(rT) * 100.0 if abs(np.median(rT)) > 1e-6 else float('nan'):.1f}% "
        f"of it recovered). Same latent, same soft_ce, same rho weight as the Q heads — so if this "
        f"tracks the spike and Q does not, the defect is in the VALUE regression, not the encoder."
    )
    print(
        f"{indent}   vs every NON-terminal step of the same episodes: n={body.size} "
        f"td1 p50 {_p(body, 50):+.3f} p90 {_p(np.abs(body), 90):+.3f} (|td1|) "
        f"mean {float(body.mean()):+.3f}"
    )


def _print_approach(eps: list[dict], key: str, indent: str = "     ") -> None:
    """Q and the realized return at FIXED step offsets before the terminal, on terminated episodes.

    The bucketed curve pools a range of offsets; this pins the shape. If Q *falls* as the terminal
    approaches while the realized return does not, the critic is telling the actor (`update_pi`
    maximizes Q) that the last second of a scoring sequence is getting worse.
    """
    term = [e for e in eps if e["terminated"]]
    if not term:
        return
    offs = [1, 2, 3, 6, 12, 24, 48, 96, 192]
    print(
        f"{indent}APPROACH PROFILE (k control steps before the terminal, terminated eps only, "
        f"n={len(term)}):"
    )
    print(f"{indent}   {'k':>5} {'k*dt':>7} {'medQ':>9} {'medG':>9} {'med gap':>9}")
    for k in offs:
        qv = np.array([e["q"][-k] for e in term if e["T"] >= k])
        gv = np.array([e[key][-k] for e in term if e["T"] >= k])
        if qv.size == 0:
            continue
        print(
            f"{indent}   {k:>5} {k * STEP_S:>6.2f}s {np.median(qv):>9.2f} {np.median(gv):>9.2f} "
            f"{np.median(qv - gv):>9.2f}"
        )


def _print_wrongway(eps: list[dict], key: str, indent: str = "     ") -> None:
    """The pre-registered test: signed gap in the 3-6 s window, on episodes that END IN A KILL."""
    kills = [e for e in eps if e["outcome"] == "kill"]
    lo_st, hi_st = int(round(WRONGWAY_LO / STEP_S)), int(round(WRONGWAY_HI / STEP_S))
    head = f"{indent}WRONG-WAY CELL ({WRONGWAY_LO:g}-{WRONGWAY_HI:g}s = {lo_st}-{hi_st}st x kill):"
    if not kills:
        print(f"{head} no kill episodes in this cell")
        return
    ttt, qq, gg = _cat(kills, "ttt"), _cat(kills, "q"), _cat(kills, key)
    td1 = _cat(kills, "td1")
    m = (ttt >= WRONGWAY_LO) & (ttt <= WRONGWAY_HI)
    if not m.any():
        print(
            f"{head} 0 steps — the longest kill episode is only "
            f"{max(e['dur_s'] for e in kills):.2f}s, so this window does not exist"
        )
        return
    gap = qq[m] - gg[m]
    print(
        f"{head} n={int(m.sum())} medG {_p(gg[m], 50):.2f} medQ {_p(qq[m], 50):.2f} "
        f"gap p50 {_p(gap, 50):+.2f} [p10 {_p(gap, 10):+.2f} p90 {_p(gap, 90):+.2f}] "
        f"P(Q<G) {float((gap < 0).mean()):.2f} | td1 p50 {_p(td1[m], 50):+.2f}"
    )


def _print_model_check(eps: list[dict], indent: str = "     ") -> None:
    """`_estimate_value` (H-step learned rollout + bootstrap) vs `Q(z, pi(z))`, same states."""
    if not eps:
        return
    vpi, vest = _cat(eps, "qpi"), _cat(eps, "vest")
    q, tl, pe = _cat(eps, "q"), _cat(eps, "termlogit"), _cat(eps, "pess")
    rh, rr = _cat(eps, "rhat"), _cat(eps, "r")
    d = vest - vpi
    den = _p(np.abs(vpi), 50)
    print(
        f"{indent}_estimate_value vs Q(z,pi): n={vpi.size} medVpi {_p(vpi, 50):.2f} "
        f"medVest {_p(vest, 50):.2f} diff p50 {_p(d, 50):+.3f} [p10 {_p(d, 10):+.3f} "
        f"p90 {_p(d, 90):+.3f}] med|diff|/med|Vpi| "
        f"{(_p(np.abs(d), 50) / den if den > 1e-9 else float('nan')):.3f} "
        f"corr {float(np.corrcoef(vpi, vest)[0, 1]) if vpi.size > 2 else float('nan'):+.4f}"
    )
    print(
        f"{indent}Q(z,a_flown) vs Q(z,pi_mu): max|diff| {float(np.abs(q - vpi).max()):.4f} "
        f"(greedy => the flown action IS the mean; a nonzero value means the [-1,1] clamp bites)"
    )
    dr = rh - rr
    print(
        f"{indent}reward head: med r {_p(rr, 50):+.3f} med rhat {_p(rh, 50):+.3f} | "
        f"err p50 {_p(dr, 50):+.3f} [p10 {_p(dr, 10):+.3f} p90 {_p(dr, 90):+.3f}] "
        f"corr {float(np.corrcoef(rr, rh)[0, 1]) if rr.size > 2 else float('nan'):+.4f}"
    )
    print(
        f"{indent}min-of-2 pessimism Q_t(avg)-Q_t(min): p50 {_p(pe, 50):+.3f} "
        f"mean {float(pe.mean()):+.3f} p90 {_p(pe, 90):+.3f} (per BACKUP; pooled over {pe.size} steps)"
    )
    print(
        f"{indent}termination head at z_1: logit p50 {_p(tl, 50):+.3f} "
        f"[min {tl.min():+.3f} max {tl.max():+.3f}] P(logit>0) {float((tl > 0).mean()):.4f}"
    )


def _print_episodes(eps: list[dict], indent: str = "     ") -> None:
    print(
        f"{indent}{'ep':>3} {'outcome':>9} {'T':>5} {'dur_s':>7} {'undisc ret':>11} "
        f"{'G_0(mc)':>9} {'G_0(boot)':>10} {'Q_0':>9} {'r_last':>9} {'Q_last':>9} {'Q_end':>9}"
    )
    for i, e in enumerate(eps):
        print(
            f"{indent}{i:>3} {e['outcome']:>9} {e['T']:>5} {e['dur_s']:>7.2f} "
            f"{e['ret']:>11.2f} {e['gmc'][0]:>9.2f} {e['gboot'][0]:>10.2f} "
            f"{e['q'][0]:>9.2f} {e['r'][-1]:>9.2f} {e['q'][-1]:>9.2f} {e['q_end']:>9.2f}"
        )


def main(argv=None) -> None:
    args = parse_args(argv)
    _cell_defaults(args)
    _paths.require_astrum_env()
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    cells = _cells(args)

    print(
        f"D5 CREDIT REACH (critic calibration) | cell {args.cell} | sep {args.separation:.0f}m "
        f"mt{args.max_time_s:.0f}s jitter{args.jitter:g} perch+{args.me_dalt_ft:.0f}ft "
        f"ar{args.action_repeat} | {args.episodes} eps/cell | dt {dt * 1000:.1f} ms "
        f"({1.0 / dt:.1f} Hz) | seed {args.seed}"
    )
    print(
        "gap = Q(z,a,'avg') - return-to-go. NEGATIVE gap = the critic UNDER-prices what the policy "
        "goes on to collect. 'avg' = mean of all 5 Q heads; 'min' is banned here (random 2-of-5)."
    )
    print(
        "td1 = r + g*(1-term)*Q_target(z',pi) - Q(z,a): the ONE-STEP residual the update drives to "
        "zero. td1~0 with a growing MC gap = accumulation down the chain (credit reach). td1 large "
        "at small ttt = the local fit itself failed, and horizon is not the story."
    )
    print(
        "NOT BFM EVIDENCE: a discounted-return trace is blacklisted as behaviour evidence "
        "(bfm_detect.py:37-48). The return is used only to calibrate the critic against itself."
    )
    print("D6: the FIRST checkpoint is the run; the rest are controls.")

    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        raw_cfg = sd["cfg"]
        cfg = TDMPC2Config(**raw_cfg)
        cfg.mpc = False
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        prov = sd.get("provenance") or {}
        if args.reward_mode == "auto":
            outcome_mode = bool(prov.get("outcome_mode", False))
            src = "provenance"
        else:
            outcome_mode = args.reward_mode == "outcome"
            src = "--reward-mode"
        # Which code path produced gamma: an explicit cfg.discount, or the episode_length derivation
        # in `TDMPC2Config.resolved_discount()`.
        if raw_cfg.get("discount") is not None:
            gpath = f"cfg.discount set EXPLICITLY = {raw_cfg['discount']}"
        else:
            frac = cfg.episode_length / cfg.discount_denom
            gpath = (
                f"DERIVED by resolved_discount(): episode_length={cfg.episode_length} / "
                f"denom={cfg.discount_denom} => frac={frac:g}, (frac-1)/frac={(frac - 1) / frac:.6f}"
                f", clamped to [{cfg.discount_min}, {cfg.discount_max}]"
            )
        tag = "RUN" if ci == 0 else "CONTROL"
        print(f"\n{'=' * 118}")
        print(
            f"[{tag}] {ck} (stage_idx={int(sd.get('stage_idx', -1))}, "
            f"collected={int(sd.get('collected', -1)):,})"
        )
        print(f"       gamma = {agent.discount:.6f}  <- {gpath}")
        print(
            f"       H={cfg.horizon} ({cfg.horizon * dt * 1000:.0f} ms), rho={cfg.rho}, "
            f"num_q={cfg.num_q}, episodic={cfg.episodic}, symlog two-hot bins "
            f"[{cfg.vmin},{cfg.vmax}]x{cfg.num_bins}"
        )
        print(
            f"       reward: outcome_mode={outcome_mode} (from {src}); 1/(1-gamma) = "
            f"{1.0 / (1.0 - agent.discount):.1f} steps = {dt / (1.0 - agent.discount):.1f} s "
            f"effective horizon"
        )
        print("=" * 118)

        for label, aspect in cells:
            kw = dict(
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
                outcome_mode=outcome_mode,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            if args.opp_tier > 0:
                kw["opponent_tier"] = args.opp_tier
            if args.cell == "headon":
                kw["opp_heading_deg"] = 180.0
            env = make_tdmpc_env(3, **kw)
            eps = _roll(agent, env, args, args.episodes, args.seed + int(round(aspect)) * 1000)
            env.close()

            counts = {
                o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})
            }
            print(f"\n-- {label} (bearing {aspect:.1f}°) | {len(eps)} eps | outcomes {counts} --")
            _print_episodes(eps)
            for group in ("kill", "deck", "death", "timeout", "bubble"):
                sub = [e for e in eps if e["outcome"] == group]
                if not sub:
                    continue
                key = "gboot" if group in ("timeout", "bubble") else "gmc"
                which = (
                    "G_boot (Q-bootstrapped tail: a timeout is a CONTINUING state for this critic)"
                    if key == "gboot"
                    else "G_mc (exact; terminated => no bootstrap)"
                )
                print(
                    f"\n   [{group}] {len(sub)} eps, {sum(e['T'] for e in sub)} control steps | "
                    f"G = {which}"
                )
                _print_curve(sub, key)
                if key == "gboot":
                    print("   ...the same cell against the BARE MC return (no bootstrap):")
                    _print_curve(sub, "gmc")
                _print_wrongway(sub, key)
                _print_terminal(sub)
                _print_approach(sub, key)
            print()
            _print_model_check(eps, indent="   ")


if __name__ == "__main__":
    main()
