"""If the T3's break stops pulling THROUGH the reversal, does spawn 55/60 become winnable?

`_diag_break_rollin_tmp.py` traced the scripted T3 through a full 60 s episode and found the engine
of the un-shootable mutual zoom. `_break` (opponent.py:423) recomputes

    sign = 1.0 if relative(s, target).ata_az >= 0.0 else -1.0

FRESH EVERY TICK with no hysteresis, while holding `_corner_limited_pull(-1.0, ...)` — max pull —
unconditionally. So when the learner works its nose across the bandit's, the sign flips and the
bandit rolls 84° -> 0 -> -84° with max aft stick held the whole way. For the ~4 s it spends near
wings-level the entire max-g demand goes into PITCH:

    t 6s  |φ| 84  θ  +7  γ  +4  V 214        t 9s  |φ|  2  θ +28  γ +13  V 218
    t 7s  |φ| 72  θ  +7  γ  +3  V 214       t 11s  |φ| 51  θ +42  γ +28  V 216
                                            t 13s  |φ| 85  θ +40  γ +35  V 209

after which it is locked in a climbing spiral: γ +38°, α pinned at the 16.0° ceiling, 218 -> 138 m/s
while gaining 3,650 m, going over the top at 8,261 m AGL. The learner follows it up and both end
α-saturated at ~150 m/s where neither can point — the state in which no controller (greedy,
PursuitExpert, BFMCoach, hand-flown energy fight) fires a single round.

Its two sibling laws in the same file both guard against exactly this:

    lead_pursuit  :341  aligned = max(cos(bank_err), 0.0)   "pull hardest once rolled onto the target"
    Tier2 break   :214  -0.25 while rolling, -0.9 only once |φ| >= 0.9 * BREAK_BANK_RAD  (+ latched sign)

so the gate is this codebase's own established convention, and Tier 3's max-performance rewrite
dropped it.

This probe does NOT edit opponent.py — the benchmark must not move until that is a deliberate
decision, because every result in the curriculum chain is scored against the current T3. Instead it
MONKEYPATCHES a gated `_break` for the duration of the run and re-runs the reachability arms:

  NATIVE  the T3 exactly as shipped — the control, reproducing the 0-kill result.
  GATED   identical except the pull is scaled by cos(bank_err), so the bandit rolls to its 85° break
          FIRST and only then pulls to the AoA limit. Nothing else changes: same states, same
          transitions, same BREAK_BANK_RAD, same alpha cap, same throttle.

Read it as: does the bandit still zoom (dAGL, V, α), and does the learner get a gun solution it never
had (kills, dealt, tHit)? If GATED converts and NATIVE does not, the spawn-55/60 wall that resisted
~20 reward levers was an opponent defect, not a missing learner skill.

    cd controllers/rl && uv run python -u _diag_gated_break_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,45,55,60 --episodes 4
"""

import argparse
from math import asin, cos, degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import opponent as OPP  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402


def _gated_break(self, s, target):
    """`Tier3Opponent._break` with the roll-then-pull gate its siblings already have.

    Byte-for-byte the shipped law except for `aligned`: the pull is scaled by cos(bank_err), so a
    reversal rolls at low g and the AoA-limit pull only lands once the lift vector is actually on
    him. `max(..., 0.0)` keeps the scale non-negative when the required bank is more than 90° away
    (mid-reversal), which is precisely the window that currently converts max g into pitch.
    """
    sign = 1.0 if relative(s, target).ata_az >= 0.0 else -1.0
    bank_err = sign * C.BREAK_BANK_RAD - s.phi
    ail = OPP._clamp(
        self.g.k_roll_to_ail * bank_err - self.g.k_roll_rate * s.p, -1.0, 1.0
    )
    aligned = max(cos(bank_err), 0.0)
    elev = OPP._corner_limited_pull(-aligned, s, ease_speed=False)
    return [ail, elev, 0.0, 1.0]


def _gamma(s):
    v = np.asarray(s.vel, dtype=np.float64)
    sp = float(np.linalg.norm(v))
    return degrees(asin(float(np.clip(v[2] / sp, -1.0, 1.0)))) if sp > 1e-6 else 0.0


NCOL = 12


def roll(agent, env, base, seed):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    rows, states, flips, last_sign = [], [], 0, None
    while True:
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        # Count break REVERSALS: the sign the bandit's break law would pick this tick.
        sgn = 1.0 if relative(op, me).ata_az >= 0.0 else -1.0
        if last_sign is not None and sgn != last_sign:
            flips += 1
        last_sign = sgn
        states.append(getattr(base.opponent, "state", "?"))
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        rows.append((rg.rng, rg.lay_miss_m, degrees(rg.aspect), degrees(rg.ata),
                     me.airspeed, op.airspeed, degrees(me.alpha), degrees(op.alpha),
                     me.alt - base.ground_alt, op.alt - base.ground_alt,
                     _gamma(op), abs(degrees(op.phi))))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    op = base._states[base.opponent_id]
    return dict(event=event, M=np.asarray(rows).reshape(-1, NCOL), flips=flips,
                S=np.asarray(states, dtype=object),
                dealt=C.HP_MAX - float(op.health), taken=C.HP_MAX - float(me.health),
                spent=int(C.AMMO_MAX - me.ammo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="30,45,55,60")
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
    native = OPP.Tier3Opponent._break

    print(f"GATE {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy | "
          f"NATIVE vs GATED T3 break | sep {args.separation:.0f}m {args.max_time_s:.0f}s "
          f"jitter {args.jitter:g} | {args.episodes} eps/cell | GATED scales the break pull by "
          f"cos(bank_err) so it rolls to {degrees(C.BREAK_BANK_RAD):.0f}° BEFORE pulling", flush=True)
    hdr = (f"GATE| {'spawn':>5} {'break':<7} {'kills':>5} {'events':<30} {'dealt':>5} {'taken':>5} "
           f"{'rds':>4} {'epS':>4} {'rev':>4} {'rngMin':>6} {'opdAGL':>6} {'medAGL':>6} "
           f"{'opγ':>4} {'opφ':>4} {'opV':>4} {'opα':>4} {'meV':>4} {'meα':>4} {'tHit':>5} {'def%':>5}")
    print(hdr, flush=True)
    print("GATE| " + "-" * (len(hdr) - 6), flush=True)

    for asp in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=asp, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        for label, fn in (("NATIVE", native), ("GATED", _gated_break)):
            OPP.Tier3Opponent._break = fn
            try:
                rows = [roll(agent, env, base, 5000 + int(asp) * 100 + i)
                        for i in range(args.episodes)]
            finally:
                OPP.Tier3Opponent._break = native
            M = np.concatenate([r["M"] for r in rows], axis=0)
            S = np.concatenate([r["S"] for r in rows], axis=0)
            ev = {}
            for r in rows:
                ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
            n = len(rows)
            print(f"GATE| {asp:>5.0f} {label:<7} {sum(r['event'] == 'kill' for r in rows):>5} "
                  f"{' '.join(f'{k}×{v}' for k, v in sorted(ev.items())):<30} "
                  f"{np.mean([r['dealt'] for r in rows]):>5.0f} "
                  f"{np.mean([r['taken'] for r in rows]):>5.0f} "
                  f"{np.mean([r['spent'] for r in rows]):>4.0f} "
                  f"{np.mean([len(r['M']) for r in rows]) * dt:>4.0f} "
                  f"{np.mean([r['flips'] for r in rows]):>4.0f} "
                  f"{np.mean([r['M'][:, 0].min() for r in rows]):>6.0f} "
                  f"{np.mean([r['M'][-1, 9] - r['M'][0, 9] for r in rows]):>+6.0f} "
                  f"{np.median(M[:, 9]):>6.0f} "
                  f"{np.median(M[:, 10]):>+4.0f} {np.median(M[:, 11]):>4.0f} "
                  f"{np.median(M[:, 5]):>4.0f} {np.median(M[:, 7]):>4.1f} "
                  f"{np.median(M[:, 4]):>4.0f} {np.median(M[:, 6]):>4.1f} "
                  f"{(M[:, 1] <= C.HIT_RADIUS_M).sum() * dt / n:>5.1f} "
                  f"{100.0 * (S == 'defensive').mean():>5.0f}", flush=True)
        env.close()
        print("GATE|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
