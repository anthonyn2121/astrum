"""WHO BUILDS THE MUTUAL ZOOM — the learner, or the scripted T3?

`_diag_can_anything_kill_tmp.py` closed the chase autopsy with a monotone ladder: spawn 30 kills in
5 s flat (dAGL +96, α 1.9°, 0% saturated); spawn 45 takes 32 s (+1840 m, 32% saturated); spawn 55
never finishes (+3180 m, 50% saturated, 0 kills). Every 60 s episode where nobody dies ends with
BOTH jets ~3 km higher at ~150 m/s and α pinned near the airframe's real ~16.7° ceiling, where
neither can point. No controller — greedy, PursuitExpert, BFMCoach, or a hand-flown energy fight —
fires a single round from that state.

One number in that story does not fit. The T3's DEFENSIVE break is `_roll_to(85°) + max pull`
(`opponent.py:423`), and an 85°-bank max-performance turn puts the lift vector nearly in-plane: it
turns level and if anything DESCENDS. Yet the bandit gains +3525..3780 m and sits ~700 m ABOVE the
learner at the splice instant. So the break is not what is flying it for most of the fight, and two
of the other four laws can climb:

    _pursue     lead pursuit — follows the learner's lift vector wherever it goes, including up
    _high_yoyo  "pitch up out of plane at idle" — climbs BY CONSTRUCTION

Those imply opposite conclusions, so measure instead of reasoning:

  * FSM OCCUPANCY (`Tier3Opponent.state`, a public property, per tick) — which law actually flies
    the bandit for the 60 s, split by the phase of the fight.
  * CLIMB ATTRIBUTION — mean vertical velocity of the BANDIT while in each state. This is what
    separates "the T3 climbs on purpose" from "the T3 is dragged up".
  * WHO COMMITTED FIRST — the first tick at which each jet holds a climb (γ ≥ GAM_UP) for HOLD_S.
    If the learner leads, the zoom is the LEARNER's pathology and is fixable by training it; if the
    bandit leads, the benchmark itself builds the un-shootable state and ~20 ceilings were spent
    fighting an artifact of a state nobody inspected.

γ (flight-path angle from the velocity vector) is used rather than θ, because at 16° α the body can
be nose-high while the jet is not actually climbing — θ would overstate the zoom by roughly α.

    cd controllers/rl && uv run python -u _diag_who_climbs_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,45,55 --episodes 4
"""

import argparse
from math import asin, degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

STATES = ["neutral", "offensive", "high_yoyo", "defensive", "energy_recovery"]
GAM_UP = radians(15.0)   # a committed climb, not a transient bobble
HOLD_S = 1.5             # ...held this long before it counts as "committed"
NCOL = 11


def _gamma(s):
    """Flight-path angle (rad) from the velocity vector — NOT θ, which α inflates by ~16°."""
    v = np.asarray(s.vel, dtype=np.float64)
    sp = float(np.linalg.norm(v))
    return asin(float(np.clip(v[2] / sp, -1.0, 1.0))) if sp > 1e-6 else 0.0


def _first_commit(gam, dt):
    """First index where the climb persists HOLD_S, or None."""
    need = max(1, int(round(HOLD_S / dt)))
    up = np.asarray(gam) >= GAM_UP
    for i in range(len(up) - need):
        if up[i:i + need].all():
            return i
    return None


def roll(agent, env, base, seed, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    rows, states = [], []
    while True:
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        # Read the FSM BEFORE stepping, so the label matches the control that flew this tick.
        states.append(getattr(base.opponent, "state", "?"))
        rows.append((rg.rng, degrees(_gamma(me)), degrees(_gamma(op)),
                     me.alt - base.ground_alt, op.alt - base.ground_alt,
                     me.airspeed, op.airspeed, degrees(me.alpha), degrees(op.alpha),
                     float(me.vel[2]), float(op.vel[2])))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    M = np.asarray(rows).reshape(-1, NCOL)
    return dict(event=event, M=M, S=np.asarray(states, dtype=object),
                t_me=_first_commit(M[:, 1], dt), t_op=_first_commit(M[:, 2], dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="30,45,55")
    ap.add_argument("--episodes", type=int, default=4)
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

    print(f"WHO {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy vs native T3 | "
          f"chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g} | "
          f"{args.episodes} eps/aspect | commit = γ ≥ {degrees(GAM_UP):.0f}° held {HOLD_S:g}s",
          flush=True)

    for asp in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=asp, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        rows = [roll(agent, env, base, 5000 + int(asp) * 100 + i, dt)
                for i in range(args.episodes)]
        M = np.concatenate([r["M"] for r in rows], axis=0)
        S = np.concatenate([r["S"] for r in rows], axis=0)
        print(f"WHO| ==== spawn {asp:.0f}°  ({' '.join(sorted(set(str(r['event']) for r in rows)))})"
              f"  eps {np.mean([len(r['M']) for r in rows]) * dt:.0f}s  "
              f"dAGL me {np.mean([r['M'][-1, 3] - r['M'][0, 3] for r in rows]):+.0f} "
              f"op {np.mean([r['M'][-1, 4] - r['M'][0, 4] for r in rows]):+.0f} ====", flush=True)
        # FSM occupancy + the bandit's climb rate WHILE IN each state: the attribution that says
        # whether the T3 climbs deliberately (high_yoyo / pursue) or is dragged up.
        for st in STATES:
            sel = S == st
            if not sel.any():
                continue
            print(f"WHO|   T3 {st:<16} {100.0 * sel.mean():5.1f}% of ticks | "
                  f"op ẋz {np.mean(M[sel, 10]):+6.1f} m/s γ {np.median(M[sel, 2]):+5.1f}° "
                  f"V {np.median(M[sel, 6]):3.0f} α {np.median(M[sel, 8]):4.1f}° | "
                  f"me ẋz {np.mean(M[sel, 9]):+6.1f} m/s γ {np.median(M[sel, 1]):+5.1f}° "
                  f"V {np.median(M[sel, 5]):3.0f} α {np.median(M[sel, 7]):4.1f}° | "
                  f"rng {np.median(M[sel, 0]):4.0f}m", flush=True)
        # WHO COMMITTED FIRST — the crux.
        for i, r in enumerate(rows):
            tm = "never" if r["t_me"] is None else f"{r['t_me'] * dt:5.1f}s"
            to = "never" if r["t_op"] is None else f"{r['t_op'] * dt:5.1f}s"
            if r["t_me"] is None and r["t_op"] is None:
                lead = "neither climbs"
            elif r["t_op"] is None:
                lead = "LEARNER only"
            elif r["t_me"] is None:
                lead = "BANDIT only"
            else:
                d = (r["t_op"] - r["t_me"]) * dt
                lead = (f"LEARNER first by {d:4.1f}s" if d > 0 else
                        f"BANDIT  first by {-d:4.1f}s")
            print(f"WHO|   ep{i} climb-commit: me {tm}  op {to}  -> {lead}"
                  f"   [{r['event']}]", flush=True)
        env.close()
        print("WHO|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
