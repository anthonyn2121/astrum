"""D2a — read the ACTOR's own exploration amplitude, on the path collection actually executes.

    cd controllers/rl && uv run python _diag_actor_std.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt \
        --cell chase --aspects 30,55 --episodes 4 --jitter 0.5

Why this exists. Every sigma in this program's record — including the `sigma_ail = 0.21` that the
"a 1.5 s coordinated deviation is a ~15-sigma event, P ~ 1e-50" argument rests on — is MPPI's `std[0]`,
read through `_diag_explore_noise._plan_explore`, which is a copy of `TDMPC2.plan()`. `cfg.mpc` is
**False** in every shipped checkpoint, so `plan()` never runs during collection (`agent.py:178`) and
that sigma describes a code path no training run has ever taken. The quantity that actually governs
discovery is the actor's own per-channel spread at `world_model.py:83-91`, and it has never been read.

WHAT IS MEASURED, AND THE ONE THING IT DOES NOT MEASURE

The trajectory is rolled GREEDY (`mu`), so the state distribution is the policy's own on-distribution
flight and is directly comparable to every other measurement in the program. At each policy step the
actor head is additionally evaluated to draw what collection WOULD have executed there. Three layers,
because they differ by more than an order of magnitude and the program has already been burned by
quoting one and reasoning about another:

  1. `sigma_pre`  = `log_std.exp()` — PRE-squash, the raw Gaussian scale. This is the number that is
     comparable to `log_std_min/max` (`config.py:50-51`) and to the MPPI `std`, and the number the
     `min_std = 0.05` floor would clamp — except that floor lives inside `plan()` and so is also dead.
  2. `dev_post`   = `|pi - mu|` AFTER the tanh squash (`world_model.py:90`) — the deviation in ACTION
     units, i.e. what the FCS receives. Saturation attenuates this hardest exactly where `mu` is large,
     which is where a committed maneuver lives, so `sigma_pre` systematically overstates it.
  2b. `|mu|` and `rail` = `frac(|mu| > 0.9)`. Added after the first run showed `elev` carrying an 11×
     smaller `dev_post` than `ail` on only a 1.6× smaller `sigma_pre` — a gap that only tanh saturation
     can produce. This separates the two causes, and they need OPPOSITE fixes: a small `sigma_pre` is
     cured by an actor-side std floor, whereas a mean pinned at the rail squashes ANY `sigma_pre` to
     nothing, so on a railed channel a std floor is inert and the deviation must be injected
     PRE-squash. `elev` is the load axis (`Control::Raw` is an Nz command), i.e. the axis D1/D2/D3's
     `nz > 3` clause depends on, so which cause holds there decides what D2b is allowed to be.
  3. `win36`      = the 1.5 s (36-step) rolling mean of `dev_post`, per channel. This is the statistic
     the P ~ 1e-50 claim is about: 36 independent draws average toward zero as 1/sqrt(36), which is the
     mechanism by which an i.i.d. sampler cannot hold a deviation. Reported as
     `P(|win36| > D)` for D in {0.25, 0.5, 0.75} — a probability with no floor at zero, so a null here
     is distinguishable from a no-op (the D1/D2 event counts are not: they read 0.00 either way).

⚠ What it does NOT measure: rolling stochastically would visit a DIFFERENT state distribution, and the
deviations would compound through the dynamics rather than being independent per-step draws. So `win36`
here is the amplitude an i.i.d. sampler offers at states the greedy policy visits — the correct first
reading, and an upper bound on nothing. Do not report it as "the exploration the collector achieves".

Interpretation is PRE-REGISTERED (handoff section 6, D2a), so that this cannot be read after the fact:
  * `sigma_ail` in [0.15, 0.35]  -> the recorded MPPI figure happens to transfer; the coloured-noise
    lever is live and correctly sized.
  * `sigma_ail` materially BELOW ~0.05 -> the coloured-noise lever is dead on arrival, because its
    window-mean gain scales with sigma. The knob is then the MISSING actor-side std floor: `min_std`
    exists but applies only inside `plan()`.
  * `sigma_ail` ABOVE ~0.5 -> white noise already reaches the required band and the whole "exploration
    cannot propose a maneuver" framing is wrong; the binding constraint is elsewhere.

D6 applies: pass the controls in the SAME invocation. Runs on CPU so it never contends with training.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from astrum_rl import _paths  # noqa: F401  (sys.path shim, matches the other harnesses)
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

CHANNELS = ("ail", "elev", "rud", "thr", "fire")
WINDOW_STEPS = 36  # 1.5 s at the trainer's 24 Hz control cadence (action_repeat 5 / 120 Hz)
BANDS = (0.25, 0.5, 0.75)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D2a: the actor's own exploration amplitude.")
    p.add_argument("--checkpoints", required=True, help="comma list; RUN first, D6 controls after")
    p.add_argument("--cell", default="chase", choices=("chase", "headon"))
    p.add_argument("--aspects", default="30,55")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument(
        "--opening-steps",
        type=int,
        default=0,
        help=(
            "if >0, ALSO report every statistic restricted to the first N policy steps of each "
            "episode. Removes the episode-PHASE confound when comparing a checkpoint that kills "
            "early (short episodes) against one that flies to timeout: without it, the killer's "
            "sample is only the opening seconds while the control's spans the whole engagement. "
            "138 steps = 5.75 s = c96's median asp30 kill length."
        ),
    )
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--me-dalt-ft", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--arm",
        default="greedy",
        choices=("greedy", "exec"),
        help=(
            "greedy (D2a): fly `mu`, and evaluate the actor alongside to read the amplitude an "
            "i.i.d. sampler OFFERS at on-distribution states. exec (D2c): fly what collection "
            "actually executes (`eval_mode=False`), so the deviations compound through the "
            "dynamics and the state distribution shifts with them — the thing D2a explicitly "
            "could not measure. Only `exec` can confirm the coloured seam survives closed loop."
        ),
    )
    p.add_argument(
        "--noise",
        default=None,
        choices=("white", "ou", "pink"),
        help="override cfg.explore_noise for this run (the D2c arm comparison)",
    )
    p.add_argument("--explore-tau-steps", type=float, default=None)
    p.add_argument("--explore-beta", type=float, default=None)
    return p.parse_args(argv)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def _roll(agent, env, episodes: int, seed0: int, arm: str = "greedy") -> list[dict]:
    """Fly `episodes`; record sigma, |mu| and the per-step deviation, per episode.

    `arm='greedy'` flies `mu` and reads the deviation the sampler would have OFFERED there.
    `arm='exec'` flies the sampled action, so `dev` is the deviation actually EXECUTED and the
    trajectory is the one collection really visits.

    Returns one dict of per-step arrays PER EPISODE, so a caller can restrict to an opening
    window without re-flying.
    """
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0 = True
        sig, mus, dev = [], [], []
        while True:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                z = agent.model.encode(o)
                mu, pi, _lp, log_std = agent.model.pi(z)
            sig.append(log_std.exp().squeeze(0).cpu().numpy())
            mus.append(mu.abs().squeeze(0).cpu().numpy())
            if arm == "exec":
                # act() consumes the coloured source, so this IS the flown deviation.
                a = agent.act(obs, t0=t0, eval_mode=False).numpy()
                dev.append(a - mu.squeeze(0).cpu().numpy())
            else:
                a = agent.act(obs, t0=t0, eval_mode=True).numpy()
                dev.append((pi - mu).squeeze(0).cpu().numpy())
            t0 = False
            obs, _r, terminated, truncated, _info = env.step(a)
            if terminated or truncated:
                break
        out.append(
            {
                "sig_pre": np.asarray(sig, dtype=float),
                "mu_abs": np.asarray(mus, dtype=float),
                "dev_post": np.asarray(dev, dtype=float),
            }
        )
    return out


def _truncate(eps: list[dict], n: int) -> list[dict]:
    """Restrict every episode to its first `n` policy steps (n <= 0 => unchanged)."""
    if n <= 0:
        return eps
    return [{k: v[:n] for k, v in e.items()} for e in eps]


def _window_exceed(eps: list[dict], ch: int) -> dict:
    """P(|1.5 s window mean of dev_post| > D). Episodes shorter than the window contribute none."""
    means = []
    for e in eps:
        arr = e["dev_post"]
        if arr.shape[0] < WINDOW_STEPS:
            continue
        c = np.cumsum(np.insert(arr[:, ch], 0, 0.0))
        means.append((c[WINDOW_STEPS:] - c[:-WINDOW_STEPS]) / WINDOW_STEPS)
    if not means:
        return {"n": 0}
    m = np.abs(np.concatenate(means))
    out = {"n": int(m.size), "p50": float(np.median(m)), "max": float(m.max())}
    for d in BANDS:
        out[f"P>{d}"] = float((m > d).mean())
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    from astrum_rl.envs import make_tdmpc_env

    aspects = [float(a) for a in args.aspects.split(",") if a.strip()]
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]

    print(
        f"ACTOR STD ({'D2a' if args.arm == 'greedy' else 'D2c'}) | arm {args.arm} | "
        f"noise {args.noise or 'as-shipped'} | cell {args.cell} | sep {args.separation:.0f}m "
        f"mt{args.max_time_s:.0f}s jitter{args.jitter:g} ar{args.action_repeat} | "
        f"{args.episodes} eps/cell | "
        f"window {WINDOW_STEPS} steps = {WINDOW_STEPS * args.action_repeat / args.rate_hz:.2f} s"
    )
    if args.arm == "exec":
        print(
            "arm=exec: dev_post is the FLOWN deviation (a - mu) on the trajectory collection "
            "really visits. Neff is the statistic that matters: white reads ~36-41, and the "
            "coloured target is <=2. |mu| is still the greedy mean at the visited state."
        )
    print(
        "sigma_pre = log_std.exp() PRE-squash (comparable to the recorded MPPI std=0.21, and to "
        "min_std=0.05 which is DEAD inside plan()); dev_post = |pi-mu| in ACTION units after tanh."
    )
    print("D6: the FIRST checkpoint is the run; the rest are controls.")

    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False  # the executed path, explicitly
        if args.noise is not None:
            cfg.explore_noise = args.noise
        if args.explore_tau_steps is not None:
            cfg.explore_tau_steps = args.explore_tau_steps
        if args.explore_beta is not None:
            cfg.explore_beta = args.explore_beta
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        tag = "RUN" if ci == 0 else "CONTROL"
        print(
            f"\n{'=' * 92}\n[{tag}] {ck} (collected={int(sd.get('collected', -1)):,}, "
            f"cfg.mpc as shipped={bool(sd['cfg'].get('mpc', False))}, "
            f"log_std clamp [{cfg.log_std_min}, {cfg.log_std_max}])\n{'=' * 92}"
        )
        for aspect in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            if args.cell == "headon":
                kw["opp_heading_deg"] = 180.0
            env = make_tdmpc_env(3, **kw)
            eps = _roll(
                agent, env, args.episodes, args.seed + int(round(aspect)) * 1000, arm=args.arm
            )
            env.close()

            views = [("full", eps)]
            if args.opening_steps > 0:
                views.append((f"open{args.opening_steps}", _truncate(eps, args.opening_steps)))
            lens = [e["sig_pre"].shape[0] for e in eps]
            print(
                f"\n-- asp{aspect:g} ({sum(lens)} policy steps over {args.episodes} eps; "
                f"ep lens {lens}) --"
            )
            for vname, view in views:
                if len(views) > 1:
                    print(f"  [{vname}]")
                sig = np.concatenate([e["sig_pre"] for e in view])
                mua = np.concatenate([e["mu_abs"] for e in view])
                dev = np.concatenate([e["dev_post"] for e in view])
                for ch in range(min(len(CHANNELS), sig.shape[1])):
                    sp, dp, ma = sig[:, ch], np.abs(dev[:, ch]), mua[:, ch]
                    w = _window_exceed(view, ch)
                    bands = (
                        "  ".join(f"P>{d}={w[f'P>{d}']:.3f}" for d in BANDS)
                        if w["n"]
                        else "no full window"
                    )
                    # Effective sample count in the window. White noise => N_eff = WINDOW_STEPS.
                    p50w = w.get("p50", float("nan"))
                    neff = (_pct(dp, 50) / p50w) ** 2 if p50w and p50w > 0 else float("nan")
                    print(
                        f"   {CHANNELS[ch]:<5} sigma_pre p50 {_pct(sp, 50):.4f} "
                        f"[p10 {_pct(sp, 10):.4f} p90 {_pct(sp, 90):.4f}]  |  "
                        f"|mu| p50 {_pct(ma, 50):.4f} rail {float((ma > 0.9).mean()):.3f}  |  "
                        f"|dev_post| p50 {_pct(dp, 50):.4f} p90 {_pct(dp, 90):.4f}  |  "
                        f"win{WINDOW_STEPS} |mean| p50 {p50w:.4f} max "
                        f"{w.get('max', float('nan')):.4f} Neff {neff:.1f}  {bands}"
                    )


if __name__ == "__main__":
    main()
