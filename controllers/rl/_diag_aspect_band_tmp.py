"""curriculum-93 pre-check: does the CHASE ladder actually live in the band the un-clip un-flattens?

The c93 edit (`PURSUIT_GATE_HI 100°→180°`, `K_ASPECT 80→144`) was justified on merge measurements —
c91/c92 sat at bandit aspect 147-158°, wholly inside the region where dΨ/dA was exactly 0.000/deg.
But c93 trains on `chase-t3-killgate`, spawning at aspect 0-90°. If a chase episode never goes above
100° aspect, the edit is a NO-OP on the ladder it is being run on, and the run is wasted.

Two distinct quantities, because they answer different questions:

  1. OCCUPANCY — % of ticks above 100° aspect. This is what decides whether the new per-step
     gradient is ever felt. It matters *because* γ=0.995 is only an 8.33 s horizon
     ([[tdmpc2-credit-horizon-ceiling]]): under a short horizon a telescoping potential is NOT
     policy-invariant in practice, so densifying reward EARLY is the entire mechanism.

  2. EPISODE-TOTAL DELTA — the exact change in summed aspect_reward, which telescopes to
       slope · [ (min(a₀,180) − min(aₑ,180)) − (min(a₀,100) − min(aₑ,100)) ]
     For a chase starting at a₀ ≤ 100° this is 0 if the episode ends below 100° aspect and
     −slope·(aₑ−100°) if it ends above it. So at the undiscounted level the edit is precisely
     "a terminal penalty for ending nose-off, and nothing else" — the dense intra-episode signal is
     where any gain has to come from. Printing both keeps that honest instead of assumed.

  3. CONVERSION WORK ABOVE THE OLD CLIP — Σ of aspect DECREASE that happens while above 100°.
     Previously that work paid exactly nothing at the moment it was done. This is the reward mass
     the edit newly delivers on-the-spot, and the number that says whether it is worth 5M steps.

    cd controllers/rl && uv run python -u _diag_aspect_band_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 0,30,55,60,75 --episodes 6
"""

import argparse
from math import degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

OLD_HI = radians(100.0)   # the clip c93 removes
OLD_K = 80.0              # and the weight it was paired with
SLOPE = OLD_K / OLD_HI    # reward per radian — INVARIANT across the edit, by construction


def _psi(a, hi):
    return -min(max(a, 0.0), hi) / hi


def roll(agent, env, base, seed):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    asp = []
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        rg = relative(base._states[base.learner_id], base._states[base.opponent_id])
        asp.append(rg.aspect)
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    A = np.asarray(asp)
    d = np.diff(A)
    above = A[:-1] > OLD_HI
    # Aspect DECREASE (d < 0 ⇒ converting) performed while above the old clip, in degrees.
    work_above = float(-d[above & (d < 0.0)].sum()) * 180.0 / np.pi
    # Exact per-step reward under both constants, summed — the dense signal the learner sees.
    old_sum = SLOPE * float(sum(_psi(A[i + 1], OLD_HI) - _psi(A[i], OLD_HI) for i in range(len(A) - 1)) * OLD_HI)
    new_sum = SLOPE * float(sum(_psi(A[i + 1], radians(180.0)) - _psi(A[i], radians(180.0))
                                for i in range(len(A) - 1)) * radians(180.0))
    return dict(
        event=event, n=len(A), frac_above=float(above.mean()) if len(above) else 0.0,
        a0=degrees(A[0]), a_end=degrees(A[-1]), a_med=degrees(np.median(A)), a_max=degrees(A.max()),
        work_above=work_above, old_sum=old_sum, new_sum=new_sum,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="0,30,55,60,75")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    print(f"ASPECT-BAND {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy | "
          f"chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g} | "
          f"{args.episodes} eps/aspect | old clip {degrees(OLD_HI):.0f}° | slope {SLOPE:.3f}/rad "
          f"(= {SLOPE * np.pi / 180.0:.4f}/deg, unchanged by c93)")
    hdr = (f"AB| {'spawn':>6} {'>100° ticks':>11} {'asp0':>6} {'aspMed':>7} {'aspMax':>7} "
           f"{'aspEnd':>7} {'workAbove°':>10} {'Σold':>8} {'Σnew':>8} {'Δ':>8}  events")
    print(hdr, flush=True)
    print("AB| " + "-" * (len(hdr) - 4), flush=True)
    for a in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=a, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        rows = [roll(agent, env, base, 5000 + int(a) * 100 + i) for i in range(args.episodes)]
        env.close()

        def m(k):
            return float(np.mean([r[k] for r in rows]))

        ev = {}
        for r in rows:
            ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
        print(f"AB| {a:>6.0f} {100.0 * m('frac_above'):>10.1f}% {m('a0'):>6.0f} "
              f"{m('a_med'):>7.0f} {m('a_max'):>7.0f} "
              f"{m('a_end'):>7.0f} {m('work_above'):>10.0f} {m('old_sum'):>8.1f} "
              f"{m('new_sum'):>8.1f} {m('new_sum') - m('old_sum'):>8.1f}  "
              + " ".join(f"{k}×{v}" for k, v in sorted(ev.items())), flush=True)
        # Per-episode, because a MEAN of Δ is not the Δ of the means: the telescoped effect is 0 when
        # an episode both starts and ends below the old 100° clip, and only bites on the ones that don't.
        for i, r in enumerate(rows):
            print(f"AB|   ep{i} a0 {r['a0']:>5.0f} aEnd {r['a_end']:>5.0f} "
                  f">100° {100.0 * r['frac_above']:>5.1f}% work {r['work_above']:>5.0f}° "
                  f"Δ {r['new_sum'] - r['old_sum']:>+7.1f}  {r['event']}", flush=True)
    print()


if __name__ == "__main__":
    main()
