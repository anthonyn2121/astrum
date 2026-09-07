"""AUDIT of D6a (`_diag_latent_depth.py`) — an independent re-derivation of its load-bearing
numbers by separately written code, plus the two confound tests its own statistic invites.

    cd controllers/rl && uv run python _diag_d6a_audit.py

WHAT IS MEASURED (c96 only; the audited box's RUN arm)

  1. VERBATIM REPRODUCTION. Same checkpoint, same cell, same seed base (17), same 4 episodes,
     same stride/cap, independently written rollout + depth math. If `predL2`/`persL2`/`ratio`
     at depth 1 and 30 do not match `_d6a_latent_depth_clean.txt` to 3 dp, the audited number
     is not reproducible.
  2. THE DENOMINATOR TEST. `ratio = predL2/persL2` is reported by the audited harness as the
     pre-registered decision statistic ("median ratio well under 1" => the lever dies). Its
     denominator is the median ONE-STEP LATENT MOVEMENT of the trajectory being flown. This
     prints predL2 and persL2 SEPARATELY per cell so it is visible whether a 17x swing in the
     ratio across cells comes from the model (numerator) or the flight (denominator).
  3. THE DISTRIBUTION TEST. The consistency loss was trained on the REPLAY BUFFER, collected
     with exploration noise (cfg.explore_noise = "white" for c96). The audited harness measures
     on the GREEDY path only. If the model looks "worse than predict-no-change" only because a
     greedy trajectory barely moves per 41.7 ms step, the stochastic arm — same model, same
     cell, collection-style actions — will move more per step and the ratio will fall. Same
     depths, same episode count, printed side by side.
  4. EFFECTIVE-LOOKAHEAD, the scale-free statistic the ratio was trying to be: the persistence
     depth k* whose median error equals the model's median error at depth k, by linear
     interpolation of the measured persL2(k) curve. "The model's k-step prediction is only as
     wrong as freezing the state for k* steps." **k* < k = the model buys lookahead (it predicts
     k steps at the error cost of k*); k* > k = it is worse than freezing.** `inf` = the model's
     error at depth k exceeds persistence error at EVERY depth <= 30, i.e. worse than freezing
     for the whole 1.25 s window — so the statistic is degenerate exactly where the model is
     worst and must be read alongside predL2 itself. Unlike the ratio it cannot be inflated by a
     slow-moving trajectory, because both sides are read off the SAME persistence curve.

WHAT THIS EXPLICITLY DOES NOT MEASURE

  * Not a re-run of the audited harness's controls (c95/c87). This audits the RUN arm's
    statistic; the controls' presence in the same invocation was verified by reading its output.
  * Not planning quality, not training error, not BFM. No detector is involved.
  * The stochastic arm is NOT c96's exact collection distribution (it is today's `_explore_eps`
    with cfg.explore_noise read off the checkpoint, and no expert override / staging). It tests
    the SENSITIVITY of the ratio to per-step state movement, which is all it is used for.

PRE-REGISTERED INTERPRETATION

  * If (1) reproduces, the audited numbers stand as numbers.
  * If persL2@d1 differs across cells by ~an order of magnitude while predL2@d1 is ~constant,
    `ratio` is a statistic about the trajectory, not about the model, and the audited harness's
    pre-registered decision rule cannot be executed as written.
  * If the stochastic arm's ratio at d1 falls below the greedy arm's by a large factor,
    "the model is worse than persistence" is a property of the greedy path, not of the model.

One control step = 41.7 ms (action_repeat 5 at 120 Hz => 24 Hz). Depth 30 = 1.25 s.
"""

from __future__ import annotations

import numpy as np
import torch
from astrum_rl import _paths  # noqa: F401
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

CK = "runs/curriculum-96/run.pt"
K = 30
STRIDE = 3
MAX_STARTS = 1200
DEPTHS = (1, 2, 3, 5, 10, 20, 30)


def roll(agent, env, episodes, seed0, greedy):
    eps = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        agent._prev_mean.zero_()
        t0 = True
        OB, A = [np.asarray(obs, np.float32)], []
        while True:
            a = agent.act(obs, t0=t0, eval_mode=greedy).numpy()
            t0 = False
            obs, r, term, trunc, _ = env.step(a)
            OB.append(np.asarray(obs, np.float32))
            A.append(np.asarray(a, np.float32))
            if term or trunc:
                break
        eps.append((np.stack(OB), np.stack(A)))
    return eps


@torch.no_grad()
def depth_stats(agent, eps):
    m, dev = agent.model, agent.device
    pred = {k: [] for k in range(1, K + 1)}
    pers = {k: [] for k in range(1, K + 1)}
    cap = max(1, MAX_STARTS // max(1, len(eps)))
    for OB, A in eps:
        obs = torch.as_tensor(OB, device=dev)
        act = torch.as_tensor(A, device=dev)
        T = act.shape[0]
        last = T - 1 - K
        if last < 0:
            continue
        Z = m.encode(obs)
        s = torch.arange(0, last + 1, STRIDE, device=dev)
        if s.numel() > cap:
            s = s[torch.linspace(0, s.numel() - 1, cap).round().long().to(dev)]
        z0 = Z[s]
        z = z0
        for k in range(K):
            z = m.next(z, act[s + k])
            tgt = Z[s + k + 1]
            pred[k + 1].append((z - tgt).norm(dim=-1).cpu().numpy())
            pers[k + 1].append((z0 - tgt).norm(dim=-1).cpu().numpy())
    return (
        {k: np.concatenate(v) for k, v in pred.items()},
        {k: np.concatenate(v) for k, v in pers.items()},
    )


def eff_lookahead(pred, pers, k):
    """persistence depth k* with median persL2(k*) == median predL2(k), linear interp."""
    p = float(np.median(pred[k]))
    curve = np.array([float(np.median(pers[j])) for j in range(1, K + 1)])
    if p <= curve[0]:
        return p / max(curve[0], 1e-9)
    if p >= curve[-1]:
        return float("inf")
    j = int(np.searchsorted(curve, p))
    lo, hi = curve[j - 1], curve[j]
    return j + (p - lo) / max(hi - lo, 1e-9)


def main():
    torch.manual_seed(17)
    np.random.seed(17)
    torch.set_num_threads(8)
    dev = resolve_device("cpu")
    sd = torch.load(CK, map_location=dev, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, dev)
    agent.load_state_dict(sd)
    agent.model.eval()
    om = bool((sd.get("provenance") or {}).get("outcome_mode", False))
    print(
        f"AUDIT of D6a | {CK} collected={int(sd['collected']):,} H={cfg.horizon} rho={cfg.rho} "
        f"mpc={cfg.mpc} explore_noise={cfg.explore_noise} outcome_mode={om} | depths 1..{K} "
        f"(1 step = 41.7 ms, d30 = 1.250 s) | stride {STRIDE} cap {MAX_STARTS}"
    )

    from astrum_rl.envs import make_tdmpc_env

    arms = [
        ("asp55 GREEDY  seed 55017 (audited config)", 55.0, 55017, True, 4),
        ("asp55 GREEDY  seed 90001 (seed sensitivity)", 55.0, 90001, True, 4),
        ("asp55 STOCHASTIC seed 55017 (collection-style)", 55.0, 55017, False, 4),
        ("asp30 GREEDY  seed 30017 (audited config)", 30.0, 30017, True, 4),
    ]
    for label, aspect, seed0, greedy, nep in arms:
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=aspect,
            separation_m=700.0,
            action_repeat=5,
            frame_stack=1,
            max_time_s=60.0,
            rate_hz=120.0,
            outcome_mode=om,
            jitter=0.5,
        )
        eps = roll(agent, env, nep, seed0, greedy)
        env.close()
        pred, pers = depth_stats(agent, eps)
        n = pred[1].size
        print(f"\n-- {label} | ep lens {[int(a.shape[0]) for _, a in eps]} | n_starts {n} --")
        print("   depth   predL2 p50   persL2 p50   ratio p50   ratio(medians)  eff_lookahead k*")
        for k in DEPTHS:
            r = float(np.median(pred[k] / np.maximum(pers[k], 1e-9)))
            rm = float(np.median(pred[k])) / max(float(np.median(pers[k])), 1e-9)
            ks = eff_lookahead(pred, pers, k)
            print(
                f"   {k:5d}  {float(np.median(pred[k])):11.4f}  {float(np.median(pers[k])):11.4f}"
                f"  {r:10.3f}  {rm:15.3f}  {ks:16.2f}"
            )


if __name__ == "__main__":
    main()
