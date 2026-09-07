"""WHAT IS THE POLICY BEING PAID during the 830 m stalemate? Every reward term, summed per window.

`_diag_stalemate_tmp.py` established that the chase failure at spawn aspect 55/60 is a CHOICE, not a
capability limit: for 25-32 s per episode the learner rides the T3's turn circle (inCircle ratio 1.00,
49% inside) at nz 2.9-3.0 and 32-44° of bank while the T3 holds 88°, flying a 1186-1346 m radius when
9 g at 180 m/s buys 369 m. Where it kills (spawn 30) it pulls 4.2 g / 70° bank and sits inside at 0.69.

Hand-pricing the candidates is guesswork, and this program has been wrong that way before
([[merge-crossing-velocity-ceiling]]'s refuted lead mechanism). So: re-run `reward.compute` on the
actual greedy rollout and SUM EVERY TERM over each window. Whatever is suppressing the pull has to
show up as a negative here, or it is not the reward.

Windows match `_diag_stalemate_tmp.py` exactly so the two tables can be read side by side.
`ALT` is carried because the anti-spiral penalty (K_SPIRAL, taxing nose-low x bank) is gated OFF above
SPIRAL_ALT_BAND_AGL = 2000 m AGL, and nothing has ever checked whether the chase fight is below it.

    cd controllers/rl && uv run python -u _diag_terms_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,55,60 --episodes 6
"""

import argparse
from math import degrees

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

# Every dense term `compute` can put in info, in reward.py order.
TERMS = ["damage_reward", "track_reward", "close_reward", "aim_reward", "aspect_reward",
         "convert_bonus", "energy_reward", "band_reward", "corner_penalty", "rel_energy_reward",
         "lead_reward", "spiral_penalty", "fire_reward", "time_penalty"]


def roll(agent, env, base, seed, ground_alt, outcome_mode):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    prev_me = prev_op = None
    ticks = []          # (rng, closure, agl, bank, nz, total_reward, *terms)
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        if prev_me is not None:
            out = R.compute(prev_me, prev_op, me, op, ground_alt=ground_alt,
                            outcome_mode=outcome_mode)
            rg = relative(me, op)
            ticks.append([rg.rng, rg.closure, me.alt - ground_alt, abs(degrees(me.phi)), me.nz,
                          float(out.reward)] + [float(out.info.get(k, 0.0)) for k in TERMS])
        prev_me, prev_op = me, op
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    return dict(event=event, M=np.asarray(ticks).reshape(-1, 6 + len(TERMS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="30,55,60")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-outcome", action="store_true",
                    help="score with outcome_mode OFF (the run flag c87 trained WITH is ON)")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    dt = 5.0 / 120.0
    om = not args.no_outcome

    print(f"TERMS {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy | outcome_mode="
          f"{om} | chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g} | "
          f"{args.episodes} eps/aspect | spiral gate: AGL < {C.SPIRAL_ALT_BAND_AGL:.0f} m ∧ nose-low "
          f"∧ |bank| > {degrees(C.SPIRAL_BANK_MIN_RAD):.0f}° ∧ gun_q < {C.SPIRAL_SOLUTION_Q:g}",
          flush=True)

    for a in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=a, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        ground_alt = float(base.ground_alt)
        rows = [roll(agent, env, base, 5000 + int(a) * 100 + i, ground_alt, om)
                for i in range(args.episodes)]
        env.close()
        eps = len(rows)
        M = np.concatenate([r["M"] for r in rows], axis=0) if eps else np.zeros((0, 6 + len(TERMS)))
        rng, clo, agl = M[:, 0], M[:, 1], M[:, 2]
        ev = {}
        for r in rows:
            ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
        print(f"TERMS| spawn {a:.0f}  ({' '.join(f'{k}×{v}' for k, v in sorted(ev.items()))})  "
              f"AGL median {np.median(agl):.0f} m, min {agl.min():.0f} m, "
              f"{100.0 * (agl < C.SPIRAL_ALT_BAND_AGL).mean():.0f}% of ticks inside the spiral band",
              flush=True)
        for label, sel in (("STALEMATE", (rng >= 600.0) & (rng <= 1000.0) & (np.abs(clo) < 50.0)),
                           ("KILLBAND", (rng >= 250.0) & (rng <= 600.0)),
                           ("all<1200", rng < 1200.0),
                           ("EPISODE", np.ones(len(M), dtype=bool))):
            if not sel.any():
                print(f"TERMS|   {label:>10}: no ticks", flush=True)
                continue
            S = M[sel]
            tot = S[:, 5].sum() / max(1, eps)
            # Per-EPISODE sums, so terms are directly comparable to R_KILL=150 / R_TIMEOUT=-50.
            parts = [(TERMS[i], S[:, 6 + i].sum() / max(1, eps)) for i in range(len(TERMS))]
            parts = [p for p in parts if abs(p[1]) >= 0.05]
            parts.sort(key=lambda p: -abs(p[1]))
            print(f"TERMS|   {label:>10} {sel.sum() * dt / max(1, eps):>5.1f} s/ep  Σreward "
                  f"{tot:>+8.1f}  " + "  ".join(f"{k.replace('_reward', '').replace('_penalty', ''):}"
                                                f" {v:+.1f}" for k, v in parts), flush=True)
        print("TERMS|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
