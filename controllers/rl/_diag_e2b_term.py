"""E2b GATE — does the termination head know anything, and is it calibrated?

    cd controllers/rl && uv run python -u _diag_e2b_term.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --aspects 0,30,55 --episodes 20 --jitter 0.5 --log runs/e2b/gate.txt

WHY THIS EXISTS, AND WHAT IT IS NOT

E2a measured that the episode-final transition is ρ-starved 7.000× and that the termination head — whose
BCE is a plain `mean`, not ρ-weighted — sees **one positive per 3,305 elements** (rate 3.03e-4). It called
the ×3 fix "the first mechanism for the dead termination head". The handoff's E2b pre-check then
established that **nothing consumes the head's prediction**: `cfg.mpc = False` in every parallel run
(`train_parallel.py:4635`, `args.no_mpc = not args.collector_mpc`, and no launch line passes
`--collector-mpc`), so the latch `dead = (termination(z) > 0.0)` in `_estimate_value` (`agent.py:290`) is
reachable only from `plan()` → only from `act()` under `if self.cfg.mpc` (`agent.py:203`); and `_td_target`
(`agent.py:299-304`) uses the **buffer's ground-truth `terminated`**, never a prediction.

So E2b's case reduces to two halves:
  (a) the terminal's share of the reward/value supervision, 0.013% → 0.091%;
  (b) a ×3 auxiliary BCE signal whose ONLY route to behaviour is shaping the shared latent.

**This harness gates (b), and nothing else.** If the head already discriminates real terminals, the
auxiliary signal is not starved and (b) is not worth a 5M run — E2b collapses to (a), which is a
small-share fix and must be written up as one. This measures a HEAD, not a policy: it cannot say anything
about kills, BFM, or the ≥55° wall, and it must not be quoted as if it could.

WHAT IS MEASURED, AND WHY THE LABEL IS ON THE POST-STEP OBS

Training pairs `term_logits = termination(zs[1:])` against `terminated` (H,B) (`agent.py:339-341`), and
`zs` is the encoded obs sequence, so `termination(encode(obs_{t+1}))` predicts `terminated_t` — the head's
question is "is THIS state terminal?", asked of the state arrived IN. This harness therefore labels the
**post-step** observation, matching the training pairing exactly rather than a plausible-looking off-by-one.

Truncated episodes (`timeout`, bubble exit past the clock) carry **no positive label** — by design, only
`terminated` zeroes the bootstrap — so they contribute negatives only. That is not a bug to be tidied away:
it is the same convention the head was trained under, and changing it here would measure a different head.

PRE-REGISTERED, with thresholds set against THIS estimator's noise floor (the E2a lesson: a threshold
below its own estimator's SD is not a test). The positive class is at most one per terminating episode, so
AUC's SE ≈ sqrt(AUC·(1−AUC)/n_pos) — at n_pos ≈ 40 that is ≈ 0.075, hence gates at 0.60 / 0.80 and not at
0.55. `--episodes` is deliberately larger than a verdict's 8 for exactly this reason; the reported
`AUC SE` line lets a reader check the gate was resolvable at the n actually obtained.

  K1  ≥1 tick anywhere with p > 0.5      ⇒ the head is NOT saturated-dead; the hard latch COULD fire.
  K2  AUC(exact terminal) ≤ 0.60         ⇒ the head carries no usable signal ⇒ E2b half (b) is justified.
  K3  AUC ≥ 0.80 while max p < 0.5       ⇒ the head KNOWS but never crosses the hard threshold ⇒ half (b)
                                           is NOT starved (E2b collapses to half (a)), and separately the
                                           threshold — not the diet — is what makes it unusable to a
                                           planner, which matters only if MPC is ever turned on.
  K4  mean p within 2× of EITHER this sample's own terminal rate OR E2a's trained buffer rate
                                           3.03e-4 ⇒ calibrated, so "dead" would be a mis-description: it
                                           is correctly predicting a rare event. Two denominators because
                                           the sample rate moves with episode length and the trained rate
                                           is what a prior-only head would emit — passing on either is the
                                           lenient reading, so a FAILURE here is robust to the choice.

Reported alongside, because the pilot run surfaced it: M5, whether the head's peak belief actually lands
ON the terminal tick. The pilot's p_max was two orders above p_final, so a high AUC can coexist with the
head peaking mid-episode — it would rank terminals above the median tick without pointing at the one that
ends the fight. AUC alone cannot distinguish those, so do not read K3 without M5.

D6 IS MANDATORY: pass the historical checkpoints in the SAME invocation. A head-quality number that also
appears in c87 is a property of the architecture and the base rate, not of anything this program changed.
A control measured in a different session is not a control.

Reported but explicitly MOOT at `cfg.mpc = False`, printed only so a future MPC experiment has the number:
M6, the soft-latch mass Σp over the trained latent depth. Do not cite it as a defect.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from astrum_rl import _paths

_paths.require_astrum_env()

from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DEAD_EVENTS = ("killed", "hard_deck", "trade")

# E2a's measured positive rate in the shipped replay buffer: one terminal per 3,305 BCE elements.
# This is the level a head that learned nothing but the PRIOR would emit, so it is one of M4's two
# calibration denominators. See memory/critic-attenuates-the-kill.md.
TRAIN_POS_RATE = 3.03e-4


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos·n_neg), tie-corrected. No scipy in the container."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(len(xs), dtype=np.float64)
    i = 0
    while i < len(xs):  # average ranks within each tie block
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    ranks = np.empty(len(xs), dtype=np.float64)
    ranks[order] = r
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _youden(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """`(best threshold, Youden's J)` over the observed values — the head's best case."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    best_t, best_j = float("nan"), -1.0
    for t in np.unique(np.concatenate([pos, neg])):
        j = (pos >= t).mean() - (neg >= t).mean()
        if j > best_j:
            best_t, best_j = float(t), float(j)
    return best_t, best_j


def _depth_ramp(agent, t_obs, t_act, j, max_d: int = 3) -> dict:
    """`p` at state `t_obs[j]`, reached by `encode()` (d=0) vs by `d` real `next()` steps.

    Paired by construction: every depth targets the SAME physical state, so the only thing varying is
    how its latent was produced. Uses the actions actually taken, not `pi`, so a policy-drift
    explanation is excluded too.
    """
    row = {}
    with torch.no_grad():
        z0 = agent.model.encode(torch.as_tensor(t_obs[j], dtype=torch.float32).unsqueeze(0))
        row[0] = float(torch.sigmoid(agent.model.termination(z0)).item())
        for d in range(1, min(j, max_d) + 1):
            zd = agent.model.encode(torch.as_tensor(t_obs[j - d], dtype=torch.float32).unsqueeze(0))
            for k in range(j - d, j):
                zd = agent.model.next(
                    zd, torch.as_tensor(t_act[k], dtype=torch.float32).unsqueeze(0)
                )
            row[d] = float(torch.sigmoid(agent.model.termination(zd)).item())
    return row


def _ramp_block(log, label, ramps, note):
    """Print a depth ramp with its PAIRED ratio against the encoded query at the same state."""
    if not ramps:
        return
    log(f"       {label}  (n = {len(ramps)})   {note}")
    log(f"         depth 0  encode(obs_j)               p50 {np.median([r[0] for r in ramps]):.3e}")
    for d in (1, 2, 3):
        v = [r[d] for r in ramps if d in r]
        if not v:
            continue
        pair = [r[d] / max(r[0], 1e-30) for r in ramps if d in r]
        log(
            f"         depth {d}  next^{d}(encode(obs_j-{d}))     p50 {np.median(v):.3e}   "
            f"paired ratio {np.median(pair):11.1f}x   ticks p>0.5: "
            f"{sum(x > 0.5 for x in v)}/{len(v)}"
        )


def _roll(agent, env, base, episodes: int, seed0: int, horizon: int):
    """Greedy episodes. Returns per-tick `p`, the exact-terminal mask, and per-episode rows.

    `p` is taken on EVERY observation the episode visits, including the post-step terminal one, so the
    positive class is exactly the state the head was trained to flag.
    """
    ps, is_term, ep_id = [], [], []
    rows, soft, m7, m8 = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0, event = True, None
        n = 0
        # M7/M8 re-query a state at the latent DEPTH the head was trained at, so they need the real
        # (obs, action) history. Keep the whole episode — 1440 x 41 floats is nothing.
        t_obs: list = [obs.copy()]
        t_act: list = []
        with torch.no_grad():
            z = agent.model.encode(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            ps.append(float(torch.sigmoid(agent.model.termination(z)).item()))
        is_term.append(False)
        ep_id.append(ep)
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            t_act.append(a.copy())
            obs, _r, terminated, truncated, info = env.step(a)
            t_obs.append(obs.copy())
            n += 1
            if info.get("event"):
                event = info["event"]
            with torch.no_grad():
                ob = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                z = agent.model.encode(ob)
                ps.append(float(torch.sigmoid(agent.model.termination(z)).item()))
                # M6, moot at cfg.mpc=False: the termination mass a SOFT latch would accumulate over the
                # trained latent depth, rolled with the policy's own actions from this real latent.
                if terminated or truncated:
                    zz, s = z, 0.0
                    for _ in range(horizon):
                        zz = agent.model.next(zz, agent.model.pi(zz)[1])
                        s += float(torch.sigmoid(agent.model.termination(zz)).item())
                    soft.append(s)
                # M7 — THE QUERY-POINT CONTROL. E2a established the terminal index is reachable from
                # exactly one start and always lands at t=2, so `termination` is supervised on the
                # terminal ONLY via `zs[2]` — a latent produced by two `next()` steps, never by
                # `encode()`. Everything above queries `encode(obs_T)`, which is therefore a point the
                # head may never have been trained on. Re-ask at depth d = 1..3 by encoding obs_{T-d}
                # and rolling forward with the REAL actions taken, so the only thing that differs from
                # the encoded query is how the latent was produced.
                #
                # M8 — THE CONTROL FOR M7, and it is not optional. M7's ramp has TWO possible causes:
                #   (a) the head genuinely fires on the PREDICTED manifold, where it was trained; or
                #   (b) `next()` drift alone reads as "terminal" regardless of state, so depth — not
                #       imminent death — drives p, and the head carries no terminal information on the
                #       predicted manifold either.
                # These are indistinguishable from M7 alone and imply opposite conclusions. M8 runs the
                # IDENTICAL depth ramp anchored mid-episode, where no terminal is near. If p climbs the
                # same way there, it is (b). Program rule: run the control as PART of the measurement.
                if terminated:
                    m7.append(_depth_ramp(agent, t_obs, t_act, n))
                    mid = n // 2
                    if mid >= 4:
                        m8.append(_depth_ramp(agent, t_obs, t_act, mid))
            is_term.append(bool(terminated))
            ep_id.append(ep)
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        # M5: WHERE in the episode the head's belief peaks. The pilot run showed p_max two orders
        # above p_final, i.e. the peak is not at the terminal — so a good AUC can coexist with the
        # head pointing at the wrong tick. Record the argmax position instead of inferring it.
        seg = np.asarray(ps[-n - 1 :])
        k = int(seg.argmax())
        rows.append(
            dict(
                ep=ep,
                event=event or ("timeout" if truncated else "?"),
                terminated=bool(terminated),
                steps=n,
                p_final=ps[-1],
                p_max=float(seg.max()),
                argmax_frac=k / max(len(seg) - 1, 1),
                at_term=bool(k == len(seg) - 1),
                me_hp=float(me.health),
                opp_hp=float(opp.health),
            )
        )
    return np.array(ps), np.array(is_term), np.array(ep_id), rows, np.array(soft), m7, m8


def _block(log, tag, ck, sd, ps, is_term, rows, soft, m7, m8, horizon):
    n_pos, n_neg = int(is_term.sum()), int((~is_term).sum())
    pos, neg = ps[is_term], ps[~is_term]
    auc = _auc(pos, neg)
    se = (
        math.sqrt(max(auc * (1 - auc), 1e-9) / n_pos)
        if n_pos > 0 and not math.isnan(auc)
        else float("nan")
    )
    thr, j = _youden(pos, neg)
    base_rate = n_pos / max(len(ps), 1)
    latch = int((ps > 0.5).sum())

    log("")
    log("=" * 92)
    log(
        f"[{tag}] {ck}  (collected={int(sd.get('collected', -1)):,}, "
        f"mpc={bool(sd['cfg'].get('mpc'))}, episodic={bool(sd['cfg'].get('episodic'))})"
    )
    log("=" * 92)
    log(f"   ticks {len(ps):,}   terminal(+) {n_pos}   non-terminal(-) {n_neg:,}")
    log("")
    log("   M1  p = sigmoid(termination(encode(obs)))")
    log(
        f"       non-terminal   p50 {np.median(neg):.2e}  p90 {np.quantile(neg, 0.9):.2e}  "
        f"p99 {np.quantile(neg, 0.99):.2e}  max {neg.max():.2e}"
    )
    if n_pos:
        log(
            f"       TERMINAL       min {pos.min():.2e}  p50 {np.median(pos):.2e}  "
            f"max {pos.max():.2e}"
        )
    else:
        log("       TERMINAL       none in sample (every episode truncated) -> AUC undefined")
    log("")
    log(f"   M2  ticks with p > 0.5 (the `logit > 0` hard latch): {latch:,} of {len(ps):,}")
    log(f"   M3  AUC(exact terminal) = {auc:.4f}   SE {se:.4f}   (gates 0.60 / 0.80)")
    log(f"       best observed threshold {thr:.3e}  Youden J {j:.4f}")
    # M4 has TWO defensible denominators and the answer must not depend on which I pick:
    #   - this sample's own terminal rate = what a head calibrated to THIS state distribution
    #     should output on average (but it moves with episode length, so it is cell-dependent);
    #   - E2a's measured buffer rate 3.03e-4 = the rate the BCE was actually trained against,
    #     i.e. the value a head that learned only the PRIOR would emit.
    # K4 passes if mean p is within 2x of EITHER, so a failure is robust to the choice.
    r_samp = ps.mean() / base_rate if base_rate > 0 else float("nan")
    r_train = ps.mean() / TRAIN_POS_RATE
    log(
        f"   M4  mean p {ps.mean():.3e}   vs THIS sample's terminal rate {base_rate:.3e} "
        f"-> {r_samp:.4f}x   vs E2a's trained buffer rate {TRAIN_POS_RATE:.2e} -> {r_train:.4f}x"
    )
    term_rows = [r for r in rows if r["terminated"]]
    if term_rows:
        at = sum(r["at_term"] for r in term_rows)
        log(
            f"   M5  argmax p is AT the terminal tick in {at}/{len(term_rows)} terminating episodes; "
            f"median argmax position {np.median([r['argmax_frac'] for r in term_rows]):.3f} of the episode"
        )
        log(
            "       (a high AUC with the peak NOT at the terminal means the head ranks terminals "
            "above\n        the median tick without actually pointing at the one that ends it)"
        )
    # M7/M8: paired within each anchor state, so this is a comparison of PRODUCTION METHOD only.
    if m7:
        log("")
        log("   M7/M8  QUERY-POINT CONTROL — the same physical state, latent produced by encode()")
        log("          vs by real next() steps at the depth E2a says the terminal is supervised at")
        log("          (t=2). M8 is M7's control: the identical ramp anchored MID-EPISODE.")
        _ramp_block(log, "M7  anchored at the TERMINAL", m7, "")
        _ramp_block(log, "M8  anchored MID-EPISODE (control)", m8, "no terminal near this state")
        if m7 and m8:
            r7 = np.median([r[2] / max(r[0], 1e-30) for r in m7 if 2 in r])
            r8 = np.median([r[2] / max(r[0], 1e-30) for r in m8 if 2 in r])
            log("")
            log(
                f"          VERDICT at the trained depth 2: terminal ramp {r7:.1f}x vs mid-episode "
                f"control {r8:.1f}x  ->  ratio-of-ratios {r7 / max(r8, 1e-30):.2f}x"
            )
            log(
                "          If these two are COMPARABLE, the ramp is `next()` drift decoded as"
                " termination\n          and carries no state information -- M7 would then be an"
                " artifact, not a mechanism.\n          If the terminal ramp is much larger, the head"
                " really does fire where it was trained."
            )
    log(
        f"   M6  soft-latch mass Sum p over depth {horizon} from the final latent: "
        f"p50 {np.median(soft) if len(soft) else float('nan'):.3e}  "
        f"max {soft.max() if len(soft) else float('nan'):.3e}   [MOOT at cfg.mpc=False]"
    )
    log("")
    log("   per-episode outcome (the head's number is only interpretable beside what happened)")
    log("     ep  event         term  steps      p_final        p_max  argmax@  meHP  oppHP")
    for r in rows:
        log(
            f"     {r['ep']:2d}  {r['event']:<13s} {'Y' if r['terminated'] else '.':>4s}  "
            f"{r['steps']:5d}  {r['p_final']:.4e}  {r['p_max']:.4e}   "
            f"{r['argmax_frac']:.3f}{'*' if r['at_term'] else ' '}  {r['me_hp']:5.0f}  {r['opp_hp']:5.0f}"
        )
    return dict(
        auc=auc,
        se=se,
        latch=latch,
        n_pos=n_pos,
        mean_p=float(ps.mean()),
        base=base_rate,
        r_samp=r_samp,
        r_train=r_train,
        at_term=sum(r["at_term"] for r in term_rows),
        n_term_eps=len(term_rows),
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", required=True, help="comma-separated; the FIRST is the run")
    p.add_argument("--aspects", default="0,30,55")
    p.add_argument(
        "--episodes", type=int, default=20, help="per aspect; >8 so the AUC gate resolves"
    )
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in a verdict")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default="", help="tee to this path (a fresh container loses /tmp)")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    sink = open(args.log, "w", buffering=1) if args.log else None

    def log(s: str = "") -> None:
        print(s, flush=True)
        if sink:
            sink.write(s + "\n")

    log("#" * 92)
    log("#  E2b GATE -- does the termination head DISCRIMINATE, and is it CALIBRATED?")
    log(
        "#  Gates E2b half (b) (the x3 auxiliary BCE signal) ONLY. Says nothing about kills or BFM."
    )
    log(
        "#  K1 any p>0.5 | K2 AUC<=0.60 => signal starved | K3 AUC>=0.80 & max p<0.5 => NOT starved"
    )
    log("#  K4 mean p within 2x of the base rate => calibrated")
    log("#" * 92)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    aspects = [float(a) for a in args.aspects.split(",") if a.strip()]
    log(
        f"   cell chase | sep {args.separation:.0f} m | mt {args.max_time_s:.0f} s | "
        f"jitter {args.jitter:g} | aspects {aspects} | {args.episodes} eps/aspect"
    )
    log(
        "   D6: the FIRST checkpoint is the run; the rest are controls, same env build and same seeds."
    )

    out = {}
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        if not cfg.episodic:
            log(f"\n[SKIP] {ck}: cfg.episodic is False -> the model has NO termination head.")
            continue
        cfg.mpc = False  # we call encode/termination directly; planning is irrelevant here
        agent = TDMPC2(cfg, torch.device("cpu"))
        agent.load_state_dict(sd)
        agent.model.eval()

        ps_all, term_all, rows_all, soft_all, m7_all, m8_all = [], [], [], [], [], []
        for asp in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=asp,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            env = make_tdmpc_env(3, **kw)
            ps, ist, _epid, rows, soft, m7, m8 = _roll(
                agent,
                env,
                env.unwrapped,
                args.episodes,
                args.seed + int(round(asp)) * 1000,
                cfg.horizon,
            )
            env.close()
            for r in rows:
                r["event"] = f"{r['event']}@a{asp:.0f}"
            ps_all.append(ps)
            term_all.append(ist)
            rows_all += rows
            soft_all.append(soft)
            m7_all += m7
            m8_all += m8

        out[ck] = _block(
            log,
            "RUN" if ci == 0 else "CONTROL",
            ck,
            sd,
            np.concatenate(ps_all),
            np.concatenate(term_all),
            rows_all,
            np.concatenate(soft_all),
            m7_all,
            m8_all,
            cfg.horizon,
        )

    log("")
    log("=" * 92)
    log("   PRE-REGISTERED GATE")
    log("=" * 92)
    for ck, r in out.items():
        k1 = r["latch"] > 0
        k2 = (not math.isnan(r["auc"])) and r["auc"] <= 0.60
        k3 = (not math.isnan(r["auc"])) and r["auc"] >= 0.80 and r["latch"] == 0
        k4 = any(0.5 <= x <= 2.0 for x in (r["r_samp"], r["r_train"]))
        log(
            f"   {ck}\n"
            f"      n_pos {r['n_pos']:3d}  AUC {r['auc']:.4f} +/- {r['se']:.4f}  latch {r['latch']}  "
            f"mean p vs sample {r['r_samp']:.4f}x / vs trained rate {r['r_train']:.4f}x  "
            f"peak-at-terminal {r['at_term']}/{r['n_term_eps']}\n"
            f"      [{'X' if k1 else ' '}] K1 any p>0.5   "
            f"[{'X' if k2 else ' '}] K2 AUC<=0.60 (signal STARVED, E2b(b) justified)   "
            f"[{'X' if k3 else ' '}] K3 AUC>=0.80 & no latch (NOT starved)   "
            f"[{'X' if k4 else ' '}] K4 calibrated"
        )
    log("")
    log("   READ THIS ONLY AS A GATE ON E2b HALF (b). Half (a) -- the terminal's 0.013% -> 0.091%")
    log("   share of the reward/value supervision -- is untouched by anything measured here.")
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
