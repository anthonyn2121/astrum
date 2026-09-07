"""D5 AUDIT — is the D5 "negative gap" a critic defect, or two other things it cannot separate?

    cd controllers/rl && uv run python _diag_d5_audit.py --episodes-greedy 6 --episodes-pi 8

WHAT IS MEASURED (three independent cross-checks of `_diag_credit_reach.py`'s headline)

The audited claim is a large, systematically NEGATIVE gap `Q(z_t,a_t,'avg') - G_mc(t)` on c96's
aspect-30 chase kill cell (reported: gap p50 -172 at 0-0.5 s, -128 in the pre-registered 3-6 s x kill
window, `td1` p50 ~0.00 at every non-terminal step). The harness's own docstring asserts (lines 73-79)
that its gap has NO policy-mismatch component "by construction". This audit tests that assertion and
two arithmetic alternatives, all on the SAME checkpoint / cell / seeds.

  A. WHICH POLICY IS Q's FIXED POINT.  `agent._td_target` (`agent.py:299-304`) computes the next
     action as `self.model.pi(next_z)[1]`. Index `[1]` of `WorldModel.pi` is the tanh-squashed
     SAMPLE (`world_model.py:96-98`), not the mean `[0]`. So the operator the critic is trained under
     is the value of the SAMPLING policy pi_sigma, while `_diag_credit_reach` compares it to the
     return realised by the GREEDY mean policy. If E[G_0 | pi_sigma] ~ Q_0, the "gap" is the ordinary
     policy-improvement gap of a correctly-fitted critic and carries no information about bootstrap
     length. Arm S rolls pi_sigma with WHITE eps (`model.pi(z, eps=None)`), which is exactly what
     `_td_target` draws -- NOT `agent.act(eval_mode=False)`, which would add the coloured
     `explore_noise` on top and measure a different policy.

  B. HOW MUCH OF THE CURVE IS JUST THE UNPRICED TERMINAL SPIKE.  `R_KILL = 150.0`
     (`constants.py:213`) arrives in ONE control step, and the reported terminal read is
     `rhat_T = +1.78` vs `r_T = +152.32` (1.2% recovered). At a terminal step `terminated = 1`, so the
     TD target is `r_T` EXACTLY -- no bootstrap, no horizon, nothing for credit reach to explain.
     If the critic prices that spike at ~0, pure discounting forces a gap of `-gamma^(T-1-t) * r_T`
     at every earlier step, with no horizon defect anywhere. This arm prints the measured gap next to
     that arithmetic prediction and the residual.

  C. HOW MUCH IS THE min-OF-2 PESSIMISM FIXED POINT.  The training target uses
     `return_type="min"` = `q[randperm(5)[:2]].min(0)`. `_diag_credit_reach` reports
     `Q_t(avg) - Q_t(min)` PER BACKUP only. A per-backup deficit `d` applied at every backup has a
     fixed-point deficit of `d * sum_k gamma^k` over the remaining episode, which is up to 200x at
     gamma = 0.995. This arm reports the amplified figure.

  Also printed: the actor's post-squash per-channel sigma on the same greedy latents (the size of the
  perturbation arm A's policy actually flies), and `bandit.health` is confirmed present in the obs
  (`geometry.py:414`, index 26) so the terminal spike is NOT unpredictable-in-principle.

WHAT IT EXPLICITLY DOES NOT MEASURE

* It does not measure BFM. Nothing here is behaviour evidence; a discounted-return trace is
  blacklisted for that (`bfm_detect.py:37-48`).
* It does not re-run the D6 controls (c95/c87). Neither control produced a single kill episode in the
  audited output, so the kill-cell curve -- the only place the pre-registered wrong-way test is
  defined -- has no control in the original invocation either. That is reported as a limitation of
  the audited result, not repaired here.
* Arm S is a Monte-Carlo estimate of `V^{pi_sigma}` from few episodes; it is reported with its own
  spread and episode count. It bounds the confound's SIZE; it is not a precise value estimate.
* It does not test whether a longer bootstrap would change anything. n-step over this buffer has no
  importance weight, which is a separate deferred question.

PRE-REGISTERED INTERPRETATION

* If `E[G_0 | pi_sigma]` lands near `Q_0` (~+43.5) while greedy `G_0` is ~+157, then arm A alone
  accounts for the headline and the audited harness's claim of "no policy gap by construction" is
  FALSE. The D5 gap is then not a calibration defect at all.
* If arm B's arithmetic prediction `-gamma^(T-1-t)*r_T` tracks the measured gap's SHAPE, then the
  entire curve is the single unpriced terminal step propagated by discounting -- a rare-event
  regression failure at a step where `terminated=1` and no bootstrap exists, i.e. the opposite of a
  credit-reach finding.
* If instead `E[G_0 | pi_sigma]` is close to greedy's +157 AND the arm-B residual stays large and
  GROWS with time-to-terminal, the audited harness's wrong-way reading survives this audit.
* Either way the audited curve's DIRECTION is checked: the pre-registered credit-reach signature
  needs |gap| to GROW with time-to-terminal. The audited output has it SHRINK (172 -> 120). That is
  read here as a fact about the printed table, not re-measured.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2 import functional as fn
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

STEP_S = 5.0 / 120.0


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/curriculum-96/run.pt")
    p.add_argument("--aspect", type=float, default=30.0)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--episodes-greedy", type=int, default=6)
    p.add_argument("--episodes-pi", type=int, default=8)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def roll(agent, env, seed, greedy: bool):
    """One episode. `greedy` -> actor MEAN (= `_diag_credit_reach`); else the actor SAMPLE with
    WHITE eps, which is exactly the action `_td_target` bootstraps through."""
    M = agent.model
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    acts, rews, obs_hist = [], [], [np.asarray(obs, dtype=np.float32)]
    event, terminated, truncated = None, False, False
    while True:
        o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        z = M.encode(o)
        mu, pi, _, _ = M.pi(z, eps=None)
        a = (mu if greedy else pi).squeeze(0).clamp(-1, 1).cpu().numpy()
        acts.append(np.asarray(a, dtype=np.float32))
        obs, r, terminated, truncated, info = env.step(a)
        obs_hist.append(np.asarray(obs, dtype=np.float32))
        rews.append(float(r))
        if info.get("event"):
            event = info["event"]
        if terminated or truncated:
            break
    return dict(
        outcome=_classify(event, terminated, truncated),
        terminated=bool(terminated),
        obs=np.stack(obs_hist),
        acts=np.stack(acts),
        rews=np.asarray(rews, dtype=np.float64),
    )


def gmc(r, gamma):
    g, acc = np.zeros(r.size), 0.0
    for i in range(r.size - 1, -1, -1):
        acc = r[i] + gamma * acc
        g[i] = acc
    return g


def main(argv=None):
    args = parse_args(argv)
    _paths.require_astrum_env()
    from astrum_rl.envs import make_tdmpc_env

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    mpc_in_ckpt = bool(sd["cfg"].get("mpc", False))
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    M = agent.model
    gamma = agent.discount
    prov = sd.get("provenance") or {}
    outcome_mode = bool(prov.get("outcome_mode", False))

    print(
        f"D5 AUDIT | {args.ckpt} collected={int(sd.get('collected', -1)):,} | "
        f"gamma={gamma:.6f} H={cfg.horizon} rho={cfg.rho} num_q={cfg.num_q} | "
        f"cfg.mpc IN CKPT = {mpc_in_ckpt} (harness forces False) | outcome_mode={outcome_mode} | "
        f"dt={dt * 1000:.1f} ms"
    )
    print(
        "ARM A: Q's fixed point is the value of pi_SAMPLE. _td_target uses model.pi(next_z)[1] "
        "= the SQUASHED SAMPLE (world_model.py:96-98), not [0]=mean. So Q ~ V^{pi_sigma}, while "
        "_diag_credit_reach's G is realised by pi_MEAN. Arm S measures V^{pi_sigma} directly."
    )

    kw = dict(
        posture="offensive",
        aspect_deg=args.aspect,
        separation_m=args.separation,
        action_repeat=args.action_repeat,
        frame_stack=1,
        max_time_s=args.max_time_s,
        rate_hz=args.rate_hz,
        outcome_mode=outcome_mode,
        jitter=args.jitter,
    )
    env = make_tdmpc_env(3, **kw)
    seed0 = args.seed + int(round(args.aspect)) * 1000  # identical to _diag_credit_reach
    print(
        f"\nenv: tier3 chase asp{args.aspect:.0f} sep{args.separation:.0f} mt{args.max_time_s:.0f}"
        f" jitter{args.jitter} | seed base {seed0} (SAME as the audited harness)"
    )

    # ---------------- arm G: reproduce the audited greedy read ----------------
    t0 = time.time()
    G = [roll(agent, env, seed0 + i, greedy=True) for i in range(args.episodes_greedy)]
    print(
        f"\n=== ARM G (greedy = pi MEAN): reproduction of the audited numbers "
        f"[{time.time() - t0:.0f}s] ==="
    )
    print(
        f"{'ep':>3} {'outcome':>9} {'T':>5} {'dur_s':>7} {'G_0(mc)':>9} {'Q_0':>8} "
        f"{'r_T':>9} {'Q_T':>8} {'rhat_T':>8}"
    )
    q0s, g0s = [], []
    for i, e in enumerate(G):
        z = M.encode(torch.as_tensor(e["obs"][:-1], dtype=torch.float32, device=device))
        A = torch.as_tensor(e["acts"], dtype=torch.float32, device=device)
        q = M.Q(z, A, return_type="avg").reshape(-1).cpu().numpy()
        rh = fn.two_hot_inv(M.reward(z, A), cfg).reshape(-1).cpu().numpy()
        g = gmc(e["rews"], gamma)
        e.update(q=q, rhat=rh, g=g)
        q0s.append(q[0])
        g0s.append(g[0])
        print(
            f"{i:>3} {e['outcome']:>9} {e['rews'].size:>5} {e['rews'].size * dt:>7.2f} "
            f"{g[0]:>9.2f} {q[0]:>8.2f} {e['rews'][-1]:>9.2f} {q[-1]:>8.2f} {rh[-1]:>8.2f}"
        )
    kills_g = sum(e["outcome"] == "kill" for e in G)
    print(
        f"  greedy: kills {kills_g}/{len(G)} | mean Q_0 {np.mean(q0s):+.2f} | "
        f"mean G_0 {np.mean(g0s):+.2f} | mean gap at t=0 {np.mean(q0s) - np.mean(g0s):+.2f}"
    )

    # actor sigma on the same latents
    zz = M.encode(
        torch.as_tensor(
            np.concatenate([e["obs"][:-1] for e in G]), dtype=torch.float32, device=device
        )
    )
    mu, _, _, log_std = M.pi(zz, eps=None)
    sig_pre = log_std.exp()
    sig_post = (sig_pre * (1.0 - mu.pow(2))).cpu().numpy()  # local post-squash deviation scale
    print("  actor sigma on these greedy latents (per channel [ail,elev,rud,thr,trig]):")
    print(
        "    pre-squash  p50 "
        + " ".join(f"{v:6.3f}" for v in np.percentile(sig_pre.cpu().numpy(), 50, axis=0))
    )
    print(
        "    post-squash p50 " + " ".join(f"{v:6.3f}" for v in np.percentile(sig_post, 50, axis=0))
    )

    # ---------------- arm S: V^{pi_sigma}, the policy Q is actually fitted to ----------------
    t0 = time.time()
    S = [roll(agent, env, seed0 + i, greedy=False) for i in range(args.episodes_pi)]
    print(
        f"\n=== ARM S (pi SAMPLE, white eps = exactly what _td_target bootstraps through) "
        f"[{time.time() - t0:.0f}s] ==="
    )
    print(f"{'ep':>3} {'outcome':>9} {'T':>5} {'dur_s':>7} {'G_0(mc)':>9} {'undisc':>9}")
    s_g0 = []
    for i, e in enumerate(S):
        g = gmc(e["rews"], gamma)
        s_g0.append(g[0])
        print(
            f"{i:>3} {e['outcome']:>9} {e['rews'].size:>5} {e['rews'].size * dt:>7.2f} "
            f"{g[0]:>9.2f} {e['rews'].sum():>9.2f}"
        )
    s_g0 = np.asarray(s_g0)
    kills_s = sum(e["outcome"] == "kill" for e in S)
    print(
        f"  pi_sigma: kills {kills_s}/{len(S)} | mean G_0 {s_g0.mean():+.2f} "
        f"+/- {s_g0.std(ddof=1) / np.sqrt(s_g0.size):.2f} (sem) "
        f"[min {s_g0.min():+.2f} max {s_g0.max():+.2f}]"
    )
    print(
        f"\n  *** ARM A VERDICT INPUTS: Q_0 = {np.mean(q0s):+.2f} | "
        f"V^pi_sigma (MC) = {s_g0.mean():+.2f} | G_0 greedy = {np.mean(g0s):+.2f}"
    )
    denom = np.mean(g0s) - s_g0.mean()
    frac = (np.mean(g0s) - np.mean(q0s)) / denom if abs(denom) > 1e-9 else float("nan")
    print(
        f"      greedy-minus-pi_sigma spread = {denom:+.2f}; the audited t=0 gap "
        f"({np.mean(q0s) - np.mean(g0s):+.2f}) is {1.0 / frac * 100 if frac else float('nan'):.0f}% "
        f"of it -> Q sits {'INSIDE' if min(s_g0.mean(), np.mean(g0s)) - 5 <= np.mean(q0s) <= max(s_g0.mean(), np.mean(g0s)) + 5 else 'OUTSIDE'} "
        f"the [V^pi_sigma, G_greedy] bracket"
    )

    # ---------------- arm B: the unpriced terminal spike, by pure discounting ----------------
    K = [e for e in G if e["outcome"] == "kill"]
    print("\n=== ARM B: is the whole curve just the ONE unpriced terminal step, discounted? ===")
    print(
        "  predicted gap if the critic prices r_T at 0 and everything else perfectly: "
        "-gamma^(T-1-t)*r_T. terminated=1 at T-1 => target IS r_T; no bootstrap, no horizon."
    )
    if not K:
        print("  no greedy kill episodes -- arm B undefined")
    else:
        print(f"  {'k(steps)':>9} {'k*dt':>7} {'med gap':>9} {'predicted':>10} {'residual':>9}")
        for k in (1, 2, 3, 6, 12, 24, 48, 96, 144):
            gp, pr = [], []
            for e in K:
                if e["rews"].size < k:
                    continue
                gp.append(e["q"][-k] - e["g"][-k])
                pr.append(-(gamma ** (k - 1)) * e["rews"][-1])
            if not gp:
                continue
            print(
                f"  {k:>9} {k * STEP_S:>6.2f}s {np.median(gp):>9.2f} {np.median(pr):>10.2f} "
                f"{np.median(gp) - np.median(pr):>9.2f}"
            )
        rT = np.array([e["rews"][-1] for e in K])
        rhT = np.array([e["rhat"][-1] for e in K])
        qT = np.array([e["q"][-1] for e in K])
        print(
            f"  terminal step: med r_T {np.median(rT):+.2f} med rhat_T {np.median(rhT):+.2f} "
            f"med Q_T {np.median(qT):+.2f} | R_KILL=150.0 (constants.py:213). "
            f"bandit.health IS in the obs (geometry.py:414, idx 26) => the spike is predictable "
            f"in principle, so this is a rare-event REGRESSION failure, not aliasing."
        )

    # ---------------- arm D: the exact identity gap(t) = -sum_k gamma^(k-t) td1_k ----------------
    print("\n=== ARM D: the gap IS the discounted sum of one-step residuals (exact identity) ===")
    print("  gap_t = Q_t - G_t and G_t = r_t + gamma*G_{t+1}, Q_t = r_t + gamma*Q_{t+1} - td1_t")
    print("  => gap_t = -sum_{k>=t} gamma^(k-t) td1_k, with td1 computed against the ONLINE Q.")
    print(
        "  So a 'calibration curve' carries NO horizon information beyond where its residuals sit."
    )
    print("  Counterfactual: substitute the critic's OWN terminal estimate Q_T for the terminal")
    print("  reward r_T in the return. If the curve collapses to ~0, the ENTIRE gap is that one")
    print("  step -- a step where terminated=1, so the TD target is r_T exactly and no bootstrap")
    print("  exists for credit reach to be short of.")
    if K:
        print(
            f"  {'k(steps)':>9} {'k*dt':>7} {'gap vs G':>9} {'gap vs G_cf':>12} {'-td1_T*g^(k-1)':>15}"
        )
        for k in (1, 2, 3, 6, 12, 24, 48, 96):
            raw, cf, pred = [], [], []
            for e in K:
                if e["rews"].size < k:
                    continue
                T = e["rews"].size
                td1T = e["rews"][-1] - e["q"][-1]  # terminated => target IS r_T
                # G_cf replaces the terminal reward with the critic's own terminal Q
                gcf = e["g"][-k] - (gamma ** (k - 1)) * (e["rews"][-1] - e["q"][-1])
                raw.append(e["q"][-k] - e["g"][-k])
                cf.append(e["q"][-k] - gcf)
                pred.append(-(gamma ** (k - 1)) * td1T)
                del T
            print(
                f"  {k:>9} {k * STEP_S:>6.2f}s {np.median(raw):>9.2f} {np.median(cf):>12.2f} "
                f"{np.median(pred):>15.2f}"
            )
        print("  (gap-vs-G_cf is what the calibration curve looks like once the single terminal")
        print(
            "   regression error is removed: that residual is all that is left for any horizon story.)"
        )

    # ---------------- arm C: pessimism, amplified to its fixed point ----------------
    print("\n=== ARM C: min-of-2 pessimism at its FIXED POINT, not per backup ===")
    allz, allA = [], []
    for e in G:
        allz.append(e["obs"][1:])
        allA.append(e["acts"])
    zn = M.encode(torch.as_tensor(np.concatenate(allz), dtype=torch.float32, device=device))
    mun = M.pi(zn, eps=None)[0]
    reps = 8
    pess = np.concatenate(
        [
            (
                M.Q(zn, mun, return_type="avg", target=True)
                - M.Q(zn, mun, return_type="min", target=True)
            )
            .reshape(-1)
            .cpu()
            .numpy()
            for _ in range(reps)
        ]
    )
    Tk = int(np.median([e["rews"].size for e in K])) if K else int(args.max_time_s / dt)
    horiz_sum = (1.0 - gamma**Tk) / (1.0 - gamma)
    print(
        f"  per-backup Q_t(avg)-Q_t(min): mean {pess.mean():+.3f} p50 {np.percentile(pess, 50):+.3f}"
        f" (n={pess.size}, {reps} random 2-of-{cfg.num_q} draws per state)"
    )
    print(
        f"  sum_k gamma^k over a {Tk}-step kill episode = {horiz_sum:.1f}; "
        f"1/(1-gamma) = {1.0 / (1.0 - gamma):.1f}"
    )
    print(
        f"  => fixed-point value deficit from pessimism alone ~ {pess.mean() * horiz_sum:+.1f} "
        f"(episode-length) / {pess.mean() / (1.0 - gamma):+.1f} (infinite-horizon). "
        f"The audited harness reported this PER BACKUP only."
    )
    env.close()


if __name__ == "__main__":
    main()
