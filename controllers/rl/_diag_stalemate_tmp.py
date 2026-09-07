"""The 830 m STALEMATE: is the learner turn-rate LIMITED, or is it CHOOSING not to turn?

`_diag_trigger_tmp.py` localised the chase failure precisely. At spawn aspect 55/60 c87 fires on 100%
of the ticks where it has an honest solution — it is not trigger-shy — but it only HAS a solution for
0.1-0.2 s per 60 s episode, because it spends 25-32 s parked at a median 831 m with closure -23 m/s
and LOS rate ~7 deg/s. Two jets co-rotating, matched, in the 600-900 m band where the pooled hit rate
is exactly 0%. The bucket that kills (spawn 30) sits at 645 m closing +48 m/s and converts 24% of its
in-range time into solutions vs 0.5%.

So the wall is CLOSING, and closing is not being outbid by the reward: K_CLOSE=120 over
TRACK_RANGE=3000 pays +15.2 for 831→450 m, while a 528 m Es deficit costs only 25·528/10000 = 1.3
(absolute) + 50·528/10000 = 2.6 (relative). The gradient is intact and dominant. It is FAILING to
close, not declining to.

That leaves exactly two possibilities, and they have opposite levers:

  CAPABILITY — both jets are at their limit and the radii match ⇒ a genuine geometric stalemate. No
               amount of "turn harder" helps; the only exit is OUT-OF-PLANE (lag displacement / high
               or low yo-yo) to cut across his circle. Lever: representation + curriculum, not gain.
  CHOICE     — the learner is loafing at 3 g while the T3 pulls 8 ⇒ it can close and doesn't. Lever:
               whatever is suppressing the pull.

Measured per tick inside the stalemate window, for BOTH jets:
  * nz (load factor), alpha (AoA), |phi| (bank), airspeed — the limit state.
  * TURN RATE as the honest rotation of the VELOCITY vector, |dV̂/dt| — not the heading rate, so a
    pure vertical pull counts. Decomposed into horizontal-course and flight-path-angle parts, giving
    OOP% = the fraction of the turn that is out of the horizontal plane. This is the direct measure of
    the out-of-plane skill the whole program is trying to teach; a policy flying flat reads ~0%.
  * TURN RADIUS r = V/ω for each, and the ratio.
  * TURN-CIRCLE GEOMETRY: his instantaneous turn centre is c = pos + r·â (â = centripetal unit
    vector), so |me − c|/r_opp < 1 means I am INSIDE his turn circle — the position that actually wins
    a co-rotating fight. Printed as a distribution, because "get inside his turn circle" has never
    once been measured in this program despite being the stated goal.

    cd controllers/rl && uv run python -u _diag_stalemate_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,55,60 --episodes 6
"""

import argparse
from math import asin, atan2, degrees

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DEG = 180.0 / np.pi
NCOL = 18


def _kin(s):
    """(unit velocity, horizontal course rad, flight-path angle rad, speed)."""
    v = np.asarray(s.vel, dtype=np.float64)
    sp = float(np.linalg.norm(v))
    if sp < 1e-6:
        return np.zeros(3), 0.0, 0.0, 0.0
    return v / sp, atan2(v[0], v[1]), asin(float(np.clip(v[2] / sp, -1.0, 1.0))), sp


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _turn(prev, cur, dt):
    """Honest turn rate of the velocity vector + its horizontal/vertical split + centripetal dir."""
    v0, chi0, gam0, sp0 = _kin(prev)
    v1, chi1, gam1, sp1 = _kin(cur)
    if sp0 < 1e-6 or sp1 < 1e-6:
        return 0.0, 0.0, 0.0, np.zeros(3)
    dv = (v1 - v0) / dt                      # dV̂/dt: magnitude = total turn rate (rad/s)
    w = float(np.linalg.norm(dv))
    w_h = abs(_wrap(chi1 - chi0)) / dt * np.cos(0.5 * (gam0 + gam1))  # course rate, projected
    w_v = abs(_wrap(gam1 - gam0)) / dt                                # flight-path-angle rate
    a_hat = dv / w if w > 1e-9 else np.zeros(3)                       # centripetal unit vector
    return w, w_h, w_v, a_hat


def roll(agent, env, base, seed, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    prev_me = prev_op = None
    rows = []
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        if prev_me is not None:
            w_m, wh_m, wv_m, ah_m = _turn(prev_me, me, dt)
            w_o, wh_o, wv_o, ah_o = _turn(prev_op, op, dt)
            r_m = me.airspeed / w_m if w_m > 1e-3 else np.inf
            r_o = op.airspeed / w_o if w_o > 1e-3 else np.inf
            # Inside his turn circle? centre = his pos + r·â, compared against r.
            if np.isfinite(r_o) and r_o < 1e5:
                c_o = np.asarray(op.pos, dtype=np.float64) + r_o * ah_o
                d_in = float(np.linalg.norm(np.asarray(me.pos, dtype=np.float64) - c_o)) / r_o
            else:
                d_in = np.nan
            rows.append((
                rg.rng, rg.closure, degrees(rg.ata), degrees(rg.aspect),
                me.nz, degrees(me.alpha), abs(degrees(me.phi)), me.airspeed,
                op.nz, degrees(op.alpha), abs(degrees(op.phi)), op.airspeed,
                w_m * DEG, w_o * DEG,
                (wv_m / w_m if w_m > 1e-6 else 0.0), (wv_o / w_o if w_o > 1e-6 else 0.0),
                min(r_m, 1e5), d_in,
            ))
        prev_me, prev_op = me, op
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    return dict(event=event, M=np.asarray(rows).reshape(-1, NCOL))


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

    print(f"STALE {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy | vs native T3 | "
          f"chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g} | "
          f"{args.episodes} eps/aspect", flush=True)
    print("STALE| windows: STALEMATE = rng 600-1000 m ∧ |closure| < 50 (the measured park); "
          "KILLBAND = rng 250-600 m (pooled hit% 47-56)", flush=True)
    hdr = ("STALE| {:>5} {:>10} {:>5} {:>10} {:>10} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>9}"
           .format("spawn", "window", "s/ep", "nz me/op", "aoa me/op", "bank m/o", "ω°/s m/o",
                   "OOP%me", "OOP%op", "V m/o", "radius", "inCircle"))
    print(hdr, flush=True)
    print("STALE| " + "-" * (len(hdr) - 7), flush=True)

    for a in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=a, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        rows = [roll(agent, env, base, 5000 + int(a) * 100 + i, dt) for i in range(args.episodes)]
        env.close()
        eps = len(rows)
        M = np.concatenate([r["M"] for r in rows], axis=0) if eps else np.zeros((0, NCOL))
        rng, clo, ata, asp = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
        wins = (
            ("STALEMATE", (rng >= 600.0) & (rng <= 1000.0) & (np.abs(clo) < 50.0)),
            ("KILLBAND", (rng >= 250.0) & (rng <= 600.0)),
            ("all<1200", rng < 1200.0),
        )
        for label, sel in wins:
            if not sel.any():
                print(f"STALE| {a:>5.0f} {label:>10} {0.0:>5.1f}   (no ticks)", flush=True)
                continue
            S = M[sel]

            def q(i):
                return float(np.median(S[:, i]))

            inc = S[:, 17]
            inc = inc[np.isfinite(inc)]
            print("STALE| {:>5.0f} {:>10} {:>5.1f} {:>10} {:>10} {:>9} {:>9} {:>6.0f}% {:>6.0f}% "
                  "{:>7} {:>7.0f} {:>8}".format(
                      a, label, sel.sum() * dt / max(1, eps),
                      f"{q(4):.1f}/{q(8):.1f}", f"{q(5):.0f}/{q(9):.0f}",
                      f"{q(6):.0f}/{q(10):.0f}", f"{q(12):.0f}/{q(13):.0f}",
                      100.0 * q(14), 100.0 * q(15),
                      f"{q(7):.0f}/{q(11):.0f}", q(16),
                      (f"{np.median(inc):.2f}|{100.0 * (inc < 1.0).mean():.0f}%"
                       if len(inc) else "n/a")), flush=True)
        print("STALE|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
