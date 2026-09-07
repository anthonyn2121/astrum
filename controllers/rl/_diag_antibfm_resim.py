"""RE-SIMULATION counterfactual for the 'built about equally' claim.

The claim's estimator freezes both positions and swaps one side's velocity direction. That is a
local sensitivity and cannot say what BUILT the geometry (the claim says so itself). The env exposes
`opponent_tier`, which DECOUPLES the opponent AI from the tier-3 start GEOMETRY (env.py:364-367), so
the honest version is available: re-fly the SAME genome from the SAME initial condition against a
bandit that does not turn toward me, and against a bandit that is not there to be turned toward.

Arms (all merge1500, sim seed 62, 576-step / 24 s horizon, trigger = M._trigger):
  BASE          genome            vs T3 (reactive)          -- must reproduce the dump
  OPP_STRAIGHT  genome            vs T1 (straight & level)   -- HIS turn deleted, IC identical
  OPP_T2        genome            vs T2 (straight, then one break)
  ME_PASSIVE    wings-level hold  vs T3                      -- MY turn deleted, HIS AI intact
  BOTH_STRAIGHT wings-level hold  vs T1                      -- neither turns
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_rl.envs import make_tdmpc_env

SEED = 62
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S * M.RATE_HZ / M.ACTION_REPEAT))
GENOME = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"]
G = M.Genome(6, 3, "bank", True)


def build(opponent_tier: int):
    cell = {k: v for k, v in M.CELLS["merge1500"].items() if k != "lateral_m"}
    env = make_tdmpc_env(
        3,
        action_repeat=M.ACTION_REPEAT,
        frame_stack=1,
        record_stats=False,
        jitter=0.0,
        rate_hz=M.RATE_HZ,
        max_time_s=HORIZON_S + 1.0,
        seed=SEED,
        opponent_tier=opponent_tier,
        **cell,
    )
    return env, env.unwrapped


def genome_ctl(step, me, bd, rel):
    return G.act(GENOME, step, N_STEPS, me)


def level_ctl(step, me, bd, rel):
    """'I do nothing': roll to wings level with the SAME inner P-gain, no pull, seg-0 throttle."""
    ail = float(np.clip(M.bfm.KROLL * (0.0 - float(me.phi)), -1.0, 1.0))
    return np.array([ail, 0.0, 0.0, 0.6023, -1.0], np.float32)


def make_hold_ctl(thr: float = 0.6023):
    """'I do nothing, PROPERLY': wings level AND gamma held at 0 by a P loop on flight-path angle,
    so the passive arm is not secretly a 1.5 km descent (which would flatter the bandit's half)."""

    def ctl(step, me, bd, rel):
        ail = float(np.clip(M.bfm.KROLL * (0.0 - float(me.phi)), -1.0, 1.0))
        v = np.asarray(me.vel, float)
        gam = math.atan2(v[2], math.hypot(v[0], v[1]))
        # elev < 0 = pull (Genome.act: elev = -load, load 1 = max pull). gamma below 0 => pull.
        elev = float(np.clip(-3.0 * (0.0 - gam), -1.0, 1.0))
        return np.array([ail, elev, 0.0, thr, -1.0], np.float32)

    return ctl


def arm(tag: str, opponent_tier: int, ctl) -> dict:
    env, base = build(opponent_tier)
    try:
        r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    finally:
        env.close()
    rng = np.array([s.rng for s in r.trace])
    cpa = np.array([s.cpa_miss for s in r.trace])
    lay = np.array([s.lay_miss_m for s in r.trace])
    leth = np.array([1.0 if s.lethal else 0.0 for s in r.trace])
    fired = np.array([float(a[4]) > 0.0 for a in r.actions])
    me_pos = np.array([st[:3] for st in r.states]) if r.states else None
    op_pos = np.array([st[16:19] for st in r.states]) if r.states else None
    print(
        f"{tag:14s} tier{opponent_tier} n={len(rng):3d} event={str(r.event):12s} kill={r.kill} "
        f"hp={r.hp_dealt:6.1f} tleth={r.tleth_s:5.3f} dalt={r.dalt_m:+7.0f} "
        f"rngMin={rng.min():8.1f} (step {int(rng.argmin())})  cpaMin={cpa.min():8.1f} "
        f"layMin={lay.min():8.1f}  lethTk={int(leth.sum()):3d} fired={int(fired.sum()):3d}"
    )
    return dict(tag=tag, rng=rng, cpa=cpa, me=me_pos, op=op_pos, r=r)


def main() -> None:
    print(f"genome {np.round(GENOME, 6).tolist()}\nN_STEPS {N_STEPS}, seed {SEED}\n")
    out = {}
    for tag, tier, ctl in (
        ("BASE", 3, genome_ctl),
        ("OPP_STRAIGHT", 1, genome_ctl),
        ("OPP_T2", 2, genome_ctl),
        ("ME_PASSIVE", 3, level_ctl),
        ("BOTH_STRAIGHT", 1, level_ctl),
        ("ME_HOLD", 3, make_hold_ctl()),
        ("BOTH_HOLD", 1, make_hold_ctl()),
    ):
        out[tag] = arm(tag, tier, ctl)

    # Is my own trajectory really opponent-independent? (validates the claim's premise AND the
    # position-level counterfactual that reuses my dumped path.)
    b, o = out["BASE"], out["OPP_STRAIGHT"]
    n = min(len(b["me"]), len(o["me"]))
    print(
        f"\nown-ship path BASE vs OPP_STRAIGHT over the common {n} steps: "
        f"max|dpos| = {np.abs(b['me'][:n] - o['me'][:n]).max():.3e} m"
    )
    p, q = out["ME_HOLD"], out["BOTH_HOLD"]
    n2 = min(len(p["me"]), len(q["me"]))
    print(
        f"own-ship path ME_PASSIVE vs BOTH_STRAIGHT over the common {n2} steps: "
        f"max|dpos| = {np.abs(p['me'][:n2] - q['me'][:n2]).max():.3e} m"
    )

    # How far does the bandit's own turn carry him under each arm?
    for tag in ("BASE", "OPP_STRAIGHT", "ME_PASSIVE", "ME_HOLD"):
        d = out[tag]
        tr = d["r"].trace
        print(
            f"{tag:14s} bandit: first 25 steps rng {d['rng'][0]:.0f} -> {d['rng'][24]:.0f} m; "
            f"rng at step 172 = {d['rng'][172] if len(d['rng']) > 172 else float('nan'):.1f} m; "
            f"min cpa {d['cpa'].min():.1f} m at step {int(d['cpa'].argmin())}; "
            f"n_trace {len(tr)}"
        )

    # Time-to-first-<600 m per arm (the gun window), and my own altitude price.
    for tag, d in out.items():
        idx = np.argmax(d["rng"] <= 600.0) if (d["rng"] <= 600.0).any() else -1
        print(
            f"{tag:14s} first rng<=600 m at step {idx} ({idx * M.ACTION_REPEAT / M.RATE_HZ:.3f} s)"
            if idx >= 0
            else f"{tag:14s} never <=600 m"
        )


if __name__ == "__main__":
    main()
