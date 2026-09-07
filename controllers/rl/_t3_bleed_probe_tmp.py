"""DECISIVE option-B diagnostic: does the aggressive T3 EVER go slow enough to be velocity-matchable
in a NEUTRAL head-on merge, under OPTIMAL pressure?

The whole option-B thesis ([[curriculum-75-energy-merge]]) rests on one physical claim: out-energize
the T3 → it goes SLOW → its velocity becomes matchable → the crossing (and the required gun lead)
NULLS → a rear-quarter tracking shot finally hits ([[merge-crossing-velocity-ceiling]]). If even a
PERFECT-BFM coach (holds energy, survives the merge 0-deck, no training — the best pressure we can
apply) cannot force the T3 slow enough to null the crossing at a rear-quarter position, then the
exploit does NOT exist in this sim and no amount of RL will find it — the neutral merge is a
symmetric geometric draw, full stop.

Flies `BFMCoach` (raw stick, no checkpoint) vs the native aggressive T3 from a head-on IC and reports,
per episode, the numbers that decide it:
  • t3MinV / meMinV          — did the T3 (or we) ever go slow? corner/sustained-turn floor
  • at BEST rear-quarter (min aspect, in a firing band): t3V, meV, |vRel|, lead°, closure, rng
  • minLead@rear             — the smallest gun-lead the coach EVER achieved behind him in-band
                               (this is the crossing: if it never drops below ~10°, no tracking shot)
  • minVrel@close            — smallest |velocity mismatch| at rng<800 (0 = crossing nulled)
  • event / net              — outcome (coach survives but should NOT convert per prior findings)

READ: if minLead@rear stays > ~15° and t3MinV stays high (>~140) across all episodes, option-B is
structurally impossible here — report it and BASELINE is final. If the T3 IS driven slow and the
lead DOES null but the coach just doesn't fire, it's a trigger/timing gap (tractable → keep c75).

  cd controllers/rl && PYTHONPATH="vendor:../../crates/astrum-env/python" uv run python \
      _t3_bleed_probe_tmp.py --episodes 10 --opp-heading 180 --sep 3000 --max-time-s 180
"""
import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C
from astrum_dogfight.bfm import BFMCoach
from astrum_dogfight.geometry import relative
from astrum_rl.envs import make_tdmpc_env

DEG = 180.0 / math.pi
G = 9.80665

_REAR_ASPECT_DEG = 60.0  # in his rear quarter
_FIRE_BAND_M = 1500.0     # in a plausible firing band


def _es(a):
    return max(0.0, a.alt) + a.airspeed * a.airspeed / (2.0 * G)


def run_ep(env, base, seed):
    obs, _ = env.reset(seed=seed)
    coach = BFMCoach()
    event = None
    t3_min_v = me_min_v = 1e9
    min_aspect = 1e9
    # snapshot at the deepest rear-quarter (min aspect in a firing band):
    best = dict(asp=1e9, t3v=float("nan"), mev=float("nan"), vrel=float("nan"),
                lead=float("nan"), clo=float("nan"), rng=float("nan"))
    min_lead_rear = 1e9   # smallest gun-lead ever, while behind him in-band
    min_vrel_close = 1e9  # smallest |v_rel| at rng<800
    peak_rel_es = -1e9
    while True:
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        a = coach.act(me, rel)
        t3_min_v = min(t3_min_v, bd.airspeed)
        me_min_v = min(me_min_v, me.airspeed)
        peak_rel_es = max(peak_rel_es, _es(me) - _es(bd))
        asp_deg = rel.aspect * DEG
        lead_deg = rel.lead_error * DEG
        vrel = float(np.linalg.norm(np.asarray(bd.vel) - np.asarray(me.vel)))
        in_band = bd.alive and rel.rng <= _FIRE_BAND_M
        if in_band and asp_deg < min_aspect:
            min_aspect = asp_deg
            best = dict(asp=asp_deg, t3v=bd.airspeed, mev=me.airspeed, vrel=vrel,
                        lead=lead_deg, clo=rel.closure, rng=rel.rng)
        if in_band and asp_deg < _REAR_ASPECT_DEG:
            min_lead_rear = min(min_lead_rear, lead_deg)
        if bd.alive and rel.rng < 800.0:
            min_vrel_close = min(min_vrel_close, vrel)
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    return dict(event=event, dealt=C.HP_MAX - bd.health, taken=C.HP_MAX - me.health,
                t3_min_v=t3_min_v, me_min_v=me_min_v, min_aspect=min_aspect,
                min_lead_rear=min_lead_rear, min_vrel_close=min_vrel_close,
                peak_rel_es=peak_rel_es, **{f"b_{k}": v for k, v in best.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--opp-heading", type=float, default=180.0)
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    env = make_tdmpc_env(3, posture="offensive", aspect_deg=0.0, separation_m=args.sep,
                         action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                         rate_hz=120.0, jitter=args.jitter, opp_heading_deg=args.opp_heading)
    base = env.unwrapped

    print("=" * 118)
    print(f"T3-BLEED PROBE (perfect BFMCoach, NO training) | head-on opp {args.opp_heading:.0f}° "
          f"sep {args.sep:.0f}m {args.max_time_s:.0f}s jitter {args.jitter} | {args.episodes} eps")
    print("=" * 118)
    print(f"{'ep':>3} {'event':>9} {'dealt':>5} {'taken':>5} {'t3MinV':>6} {'meMinV':>6} "
          f"{'bAsp':>5} {'b_t3V':>6} {'b_meV':>6} {'b_vRel':>6} {'b_lead':>6} {'b_rng':>6} "
          f"{'ldRear':>6} {'vrelCl':>6}")
    rows = []
    for ep in range(args.episodes):
        s = run_ep(env, base, seed=3000 + ep)
        rows.append(s)
        ml = s["min_lead_rear"]; mv = s["min_vrel_close"]
        print(f"{ep:>3} {s['event'] or 'none':>9} {s['dealt']:5.0f} {s['taken']:5.0f} "
              f"{s['t3_min_v']:6.0f} {s['me_min_v']:6.0f} {s['min_aspect']:5.0f} "
              f"{s['b_t3v']:6.0f} {s['b_mev']:6.0f} {s['b_vrel']:6.0f} {s['b_lead']:6.0f} {s['b_rng']:6.0f} "
              f"{(ml if ml < 1e8 else float('nan')):6.0f} {(mv if mv < 1e8 else float('nan')):6.0f}")
    env.close()
    n = len(rows)
    print("-" * 118)
    print(f"AGG {n} eps: kills={sum(r['event']=='kill' for r in rows)} "
          f"losses={sum(r['event'] in ('death','killed','hard_deck','trade') for r in rows)} | "
          f"meanT3MinV={np.mean([r['t3_min_v'] for r in rows]):.0f} (min {np.min([r['t3_min_v'] for r in rows]):.0f}) "
          f"meanMinLeadRear={np.nanmean([r['min_lead_rear'] if r['min_lead_rear']<1e8 else np.nan for r in rows]):.0f}° "
          f"meanMinVrelClose={np.nanmean([r['min_vrel_close'] if r['min_vrel_close']<1e8 else np.nan for r in rows]):.0f}m/s "
          f"meanMinAspect={np.mean([r['min_aspect'] for r in rows]):.0f}°")


if __name__ == "__main__":
    main()
