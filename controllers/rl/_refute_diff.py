"""INDEPENDENT refutation probe 2 -- the POLICY-DIFFERENTIAL test.

The claim is that the one-sided K_LEAD gate (and the gamma=1 shaping bias) TAXES
out-of-plane / displacement-roll maneuvers hard enough to make them unlearnable, and
that PATCH B (`lead_r = raw_lead if in_zone else 0.0`) unblocks them.

A penalty only changes behaviour through its DIFFERENCE between candidate actions.
So: fly the SAME scenario with scripted BFM sticks -- (A) in-plane lag-pursuit tracking
baseline, (B) the same but with a LagDisplacementRoll inserted, (C) HighYoYo (out-of-plane),
(D) straight/no-maneuver -- and measure, for each:
    J_ship  = discounted return under the SHIPPED reward
    J_patch = the same trajectory rescored with PATCH B (out-of-zone lead_r == 0)
Then the incentive shift the patch actually delivers for choosing the roll over the
baseline is  (J_patch(B) - J_patch(A)) - (J_ship(B) - J_ship(A)).
"""
import argparse
import math

import numpy as np

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import env as _denv  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402

GAM = 0.995
REPEAT = 5
DEG = 180.0 / math.pi


def _phi_lead(g):
    return (1.0 + math.cos(g.lead_error)) * 0.5


def _phi_rng(r):
    return -min(max(r, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE


def run(base, policy, max_ctl_steps, fire=True):
    """Drive the RAW DogfightGymEnv at 24 Hz (5 sim ticks per stick command)."""
    obs, _ = base.reset(seed=policy["seed"])
    rows = []
    ctl = 0
    ev = None
    while ctl < max_ctl_steps:
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        act4 = policy["fn"](me, rel, ctl)
        trig = 0.0
        if fire and rel.in_range and rel.lay_feasible and rel.lay_miss_m < C.WEZ_HIT_M:
            trig = 1.0
        native = [float(act4[0]), float(act4[1]), float(act4[2]),
                  float(0.5 * (act4[3] + 1.0)) if act4[3] <= 1.0 else 1.0, trig]
        native[3] = float(min(1.0, max(0.0, act4[3])))
        done = False
        for _ in range(REPEAT):
            me_p = base._states[base.learner_id]
            bd_p = base._states[base.opponent_id]
            obs, r, term, trunc, info = base.step(native)
            me_n = base._states[base.learner_id]
            bd_n = base._states[base.opponent_id]
            row = {"r": float(r), "ctl": ctl, "lead_reward": float(info.get("lead_reward", 0.0))}
            if me_p.alive and bd_p.alive and me_n.alive and bd_n.alive:
                rg = relative(me_n, bd_n)
                pg = relative(me_p, bd_p)
                row["in_zone"] = bool(rg.aspect <= C.CONVERT_ASPECT_RAD and rg.rng <= C.CONVERT_RANGE_MAX)
                row["raw_lead"] = C.K_LEAD * (_phi_lead(rg) - _phi_lead(pg))
                row["raw_close"] = C.K_CLOSE * (_phi_rng(rg.rng) - _phi_rng(pg.rng))
                row["close_reward"] = float(info.get("close_reward", 0.0))
                row["rng"] = rg.rng
                row["aspect"] = rg.aspect
            else:
                row["in_zone"] = False
                row["raw_lead"] = 0.0
                row["raw_close"] = 0.0
                row["close_reward"] = float(info.get("close_reward", 0.0))
                row["rng"] = float("nan")
                row["aspect"] = float("nan")
            rows.append(row)
            if info.get("event"):
                ev = info["event"]
            if term or trunc:
                done = True
                break
        ctl += 1
        if done:
            break
    return rows, ev


def score(rows):
    n = len(rows)
    disc = np.array([GAM ** r["ctl"] for r in rows])
    rew = np.array([r["r"] for r in rows])
    lead = np.array([r["lead_reward"] for r in rows])
    rawl = np.array([r["raw_lead"] for r in rows])
    inz = np.array([r["in_zone"] for r in rows])
    patch_lead = np.where(inz, rawl, 0.0)
    dl = patch_lead - lead
    close = np.array([r["close_reward"] for r in rows])
    rawc = np.array([r["raw_close"] for r in rows])
    dc = rawc - close
    return dict(
        n=n, t=n / 120.0,
        J_ship=float((rew * disc).sum()),
        J_ship_u=float(rew.sum()),
        J_patchB=float(((rew + dl + dc) * disc).sum()),
        d_lead=float((dl * disc).sum()),
        d_close=float((dc * disc).sum()),
        lead_ship=float((lead * disc).sum()),
        lead_ship_u=float(lead.sum()),
        oz_frac=float((~inz).mean()),
        minrng=float(np.nanmin([r["rng"] for r in rows])),
    )


def mk_policies(seed, roll_at, roll_len):
    tracker = bfm.LagPursuitTracker()
    yoyo = bfm.HighYoYo()
    state = {}

    def _init(me, rel):
        if "t" not in state:
            tracker.reset(me, rel)
            state["t"] = True

    def pol_track(me, rel, ctl):
        _init(me, rel)
        a, _ = tracker.step(me, rel)
        return a

    def pol_straight(me, rel, ctl):
        return np.array([0.0, -0.05, 0.0, 1.0], np.float32)

    def pol_roll(me, rel, ctl):
        _init(me, rel)
        if roll_at <= ctl < roll_at + roll_len:
            if "roll" not in state:
                r = bfm.LagDisplacementRoll()
                r.reset(me, rel)
                state["roll"] = r
            a, _ = state["roll"].step(me, rel)
            return a
        a, _ = tracker.step(me, rel)
        return a

    def pol_yoyo(me, rel, ctl):
        _init(me, rel)
        if roll_at <= ctl < roll_at + roll_len:
            if "yy" not in state:
                yoyo.reset(me, rel)
                state["yy"] = True
            a, _ = yoyo.step(me, rel)
            return a
        a, _ = tracker.step(me, rel)
        return a

    return [
        dict(name="A_track", fn=pol_track, seed=seed),
        dict(name="B_disproll", fn=pol_roll, seed=seed),
        dict(name="C_highyoyo", fn=pol_yoyo, seed=seed),
        dict(name="D_straight", fn=pol_straight, seed=seed),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=("chase", "merge"), default="chase")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--roll-at", type=int, default=24)   # control steps (24 = 1 s in)
    ap.add_argument("--roll-len", type=int, default=96)  # 96 ctl = 4 s
    ap.add_argument("--max-time-s", type=float, default=60.0)
    args = ap.parse_args()

    if args.scenario == "chase":
        kw = dict(posture="offensive", opponent_tier=0, aspect_deg=30.0, separation_m=900.0,
                  max_time_s=args.max_time_s, rate_hz=120.0, outcome_mode=True)
    else:
        kw = dict(posture="offensive", opponent_tier=0, aspect_deg=0.0, opp_heading_deg=180.0,
                  separation_m=2200.0, max_time_s=180.0, rate_hz=120.0, outcome_mode=True,
                  lateral_m_max=1400.0, jitter=0.5)
    base = _denv.make_tier3_env(**kw)
    maxctl = int(kw["max_time_s"] * 24) + 10

    agg = {}
    for s in range(args.seeds):
        print(f"\n##### seed {s} scenario={args.scenario} roll_at={args.roll_at} len={args.roll_len}")
        res = {}
        for pol in mk_policies(1000 + s, args.roll_at, args.roll_len):
            rows, ev = run(base, pol, maxctl)
            sc = score(rows)
            res[pol["name"]] = sc
            print(f"  {pol['name']:12s} t={sc['t']:6.1f}s ev={str(ev):12s} minRng={sc['minrng']:6.0f} "
                  f"ozFrac={sc['oz_frac']:.2f} J_ship={sc['J_ship']:+9.2f} J_patchB={sc['J_patchB']:+9.2f} "
                  f"dLead={sc['d_lead']:+7.2f} dClose={sc['d_close']:+7.2f} "
                  f"leadShip(u)={sc['lead_ship_u']:+8.2f}")
        a = res["A_track"]
        for k in ("B_disproll", "C_highyoyo", "D_straight"):
            b = res[k]
            gap_ship = b["J_ship"] - a["J_ship"]
            gap_patch = b["J_patchB"] - a["J_patchB"]
            print(f"   -> {k:12s} vs A: gap_ship {gap_ship:+9.2f}  gap_patchB {gap_patch:+9.2f}  "
                  f"INCENTIVE SHIFT {gap_patch - gap_ship:+8.2f}")
            agg.setdefault(k, []).append((gap_ship, gap_patch))

    print("\n==== MEAN over seeds ====")
    for k, v in agg.items():
        v = np.array(v)
        print(f"  {k:12s} gap_ship {v[:,0].mean():+9.2f}  gap_patchB {v[:,1].mean():+9.2f}  "
              f"shift {(v[:,1]-v[:,0]).mean():+8.2f}")


if __name__ == "__main__":
    main()
