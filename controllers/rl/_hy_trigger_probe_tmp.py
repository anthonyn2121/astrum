"""De-risk c70: does the WINDOWED high-yoyo splice actually FIRE in a neutral (beam) fight vs the
aggressive T3? The splice fires only in the hot-closure overshoot transient — the doctrine trigger
    rng < YOYO_RANGE (900 m)  AND  closure > CLOSURE_HI (120 m/s)
(same as bfm.HighYoYo / the Tier3 opponent). If a co-speed pursuer converting from the beam never
spikes closure past 120 inside 900 m, the high-yoyo seed is a no-op and c70 degrades to "energy
reward + non-advantageous band" only. This rolls the greedy warm-start vs the aggressive T3 at the
c70 neutral aspects and reports, per aspect:
  trig%   = fraction of alive steps satisfying the FULL trigger (rng<900 ∧ closure>120)
  inrng%  = fraction inside 900 m (the range half of the gate)
  cloMax/cloP90/cloMean = closure distribution INSIDE 900 m (m/s) — is 120 even reachable?
  epsTrig = episodes that trigger at least once (the seed needs ≥ some of these)

Greedy is a LOWER BOUND — training exploration jitter generates more varied (more overshooting)
approaches, so the true training trigger rate is higher. Throwaway diagnostic (untracked).

  cd /workspace/controllers/rl && uv run python _hy_trigger_probe_tmp.py \
      --checkpoint runs/curriculum-69/_auto_002.pt --aspects 55,75,90,105 --episodes 12
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aspects", default="55,75,90,105")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-time-s", type=float, default=120.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load(path, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def _roll(agent, env, base, C, seed):
    from astrum_dogfight.geometry import relative

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    rng_s, clo_s = [], []
    t0 = True
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, term, trunc, info = env.step(a)
        me = base._states.get(base.learner_id)
        opp = base._states.get(base.opponent_id)
        if me is not None and opp is not None and me.alive and opp.alive:
            rg = relative(me, opp)
            rng_s.append(rg.rng)
            clo_s.append(rg.closure)
        if term or trunc:
            break
    return np.asarray(rng_s), np.asarray(clo_s)


def main():
    args = parse_args()
    _paths.require_astrum_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    from astrum_dogfight import constants as C
    from astrum_rl.envs import make_tdmpc_env

    agent = _load(args.checkpoint, device)
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]

    print(f"HIGH-YOYO TRIGGER PROBE | {args.checkpoint} | vs T{args.opp_tier or 3} (aggressive) "
          f"sep {args.separation:.0f}m | {args.episodes} eps/aspect jitter{args.jitter:g} "
          f"mt{args.max_time_s:g}s")
    print(f"  trigger := rng < YOYO_RANGE ({C.YOYO_RANGE:.0f} m) AND closure > CLOSURE_HI "
          f"({C.CLOSURE_HI:.0f} m/s)\n")
    hdr = (f"{'aspect':>6} {'trig%':>6} {'inrng%':>6} | {'cloMax':>6} {'cloP90':>6} {'cloMean':>7} "
           f"| {'epsTrig':>8}")
    print(hdr)
    print("-" * len(hdr))

    for aspect in aspects:
        env = make_tdmpc_env(
            3, posture="offensive", aspect_deg=aspect, separation_m=args.separation,
            action_repeat=args.action_repeat, frame_stack=1,
            max_time_s=args.max_time_s, rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
            **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
        )
        base = env.unwrapped
        trig_frac, inrng_frac, clo_in = [], [], []
        eps_trig = 0
        for ep in range(args.episodes):
            rng, clo = _roll(agent, env, base, C, args.seed + int(aspect) * 1000 + ep)
            if len(rng) < 3:
                continue
            in_rng = rng < C.YOYO_RANGE
            trig = in_rng & (clo > C.CLOSURE_HI)
            trig_frac.append(float(trig.mean()))
            inrng_frac.append(float(in_rng.mean()))
            if in_rng.any():
                clo_in.append(clo[in_rng])
            eps_trig += int(trig.any())
        env.close()
        clo_all = np.concatenate(clo_in) if clo_in else np.array([0.0])

        def pc(x):
            return f"{100 * x:4.0f}%"
        print(f"{aspect:>6.0f} {pc(np.mean(trig_frac)):>6} {pc(np.mean(inrng_frac)):>6} | "
              f"{clo_all.max():>6.0f} {np.percentile(clo_all, 90):>6.0f} {clo_all.mean():>7.0f} | "
              f"{eps_trig:>5}/{args.episodes:<2}")
    print("\n  If trig%>0 / epsTrig high, the windowed high-yoyo seed fires and can teach. cloMax<120 "
          "everywhere ⇒ the trigger is unreachable at co-speed → widen the gate before launch.")


if __name__ == "__main__":
    main()
