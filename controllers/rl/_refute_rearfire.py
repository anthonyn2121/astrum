"""Refutation check: does the sin(lead_error) aliasing fire REARWARD on all three
non-kill seeds, or only on seed 28?  Also reports lay_feasible and the exact
drag-free CHORD miss 1030*tof*|nose - aim_dir| (= 2*1030*tof*sin(delta/2)),
which is the within-model miss of a round launched along the NOSE.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/crates/astrum-env/python")
sys.path.insert(0, "/workspace/controllers/rl")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.gunlay import intercept  # noqa: E402

GENOME = np.array(
    [
        1.0,
        1.0,
        0.20466533,
        0.42106359,
        0.04260145,
        1.0,
        -1.0,
        0.81890272,
        -1.0,
        -0.45391245,
        -0.49038442,
        0.90924598,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        0.7947789,
        0.53075154,
    ],
    float,
)
CELL = M.CELLS["merge1500"]
N_STEPS = 576
gm = M.Genome(6, 3, "bank", True)


def nose_enu(me):
    cp, sp = math.cos(me.theta), math.sin(me.theta)
    cy, sy = math.cos(me.psi), math.sin(me.psi)
    # psi is heading from North, clockwise; ENU x=E, y=N, z=Up
    return np.array([cp * sy, cp * cy, sp], float)


for seed in (28, 79, 96, 62):
    env, base = M.build_env(CELL, seed, N_STEPS * M.CONTROL_DT_S)
    rec = []

    def ctl(step, me, bd, rel):
        rel2 = rel
        sol = intercept(
            np.asarray(me.pos, float),
            np.asarray(me.vel, float),
            np.asarray(bd.pos, float),
            np.asarray(bd.vel, float),
        )
        n = nose_enu(me)
        d = sol.boresight if sol.feasible else np.zeros(3)
        chord = (
            (1030.0 * sol.tof_s * float(np.linalg.norm(n - d))) if sol.feasible else float("nan")
        )
        rec.append(
            dict(
                step=step,
                fire=M._trigger(rel2) > 0,
                rng=float(rel2.rng),
                ata=math.degrees(rel2.ata),
                lead=math.degrees(rel2.lead_error),
                lay=float(rel2.lay_miss_m),
                tof=float(rel2.lay_tof),
                feas=bool(rel2.lay_feasible),
                chord=chord,
                ammo=int(me.ammo),
                oh=float(bd.health),
                leth=bool(rel2.lay_miss_m < C.HIT_RADIUS_M and rel2.rng <= M.R_LETHAL_M),
            )
        )
        return gm.act(GENOME, step, N_STEPS, me)

    r = M.roll_once(env, base, ctl, N_STEPS, keep=False)
    fires = [q for q in rec if q["fire"]]
    rear = [q for q in fires if q["lead"] > 90.0]
    lethal = [q for q in rec if q["leth"]]
    lethal_rear = [q for q in lethal if q["lead"] > 90.0]
    # rounds consumed: ammo drops one step after the fire step
    amm = [q["ammo"] for q in rec]
    first_rear = rear[0]["step"] if rear else None
    print(
        f"=== seed {seed}: event={r.event} steps={r.steps} rounds={r.rounds} "
        f"tleth={r.tleth_s:.3f}s hp_dealt={r.hp_dealt:.1f}"
    )
    print(
        f"    fire steps {len(fires)}  of which lead_error>90deg: {len(rear)} "
        f"-> steps {[q['step'] for q in rear]}"
    )
    print(
        f"    lethal ticks {len(lethal)}  of which lead_error>90deg: {len(lethal_rear)} "
        f"-> steps {[q['step'] for q in lethal_rear]}  (= {len(lethal_rear) * M.CONTROL_DT_S:.4f} s of tLeth)"
    )
    if first_rear is not None:
        a_at = amm[first_rear]
        a_end = amm[-1]
        hp_at = rec[first_rear]["oh"]
        hp_end = rec[-1]["oh"]
        print(f"    at first rearward fire step {first_rear}: ammo={a_at} opp_health={hp_at:.1f}")
        print(f"    at last recorded step {rec[-1]['step']}: ammo={a_end} opp_health={hp_end:.1f}")
        print(
            f"    => rounds spent from step {first_rear} onward = {a_at - a_end}, HP dealt = {hp_at - hp_end:.1f}"
        )
        pre = 511 - a_at
        print(
            f"    => rounds spent BEFORE step {first_rear} = {pre}, HP dealt = {100.0 - hp_at:.1f} "
            f"({(100.0 - hp_at) / 2.0:.0f} hits, {100.0 * ((100.0 - hp_at) / 2.0) / max(pre, 1):.1f}% of rounds)"
        )
    for q in rear:
        print(
            f"      step {q['step']:4d} rng {q['rng']:7.1f} ata {q['ata']:6.1f} "
            f"lead {q['lead']:7.2f} lay_miss {q['lay']:6.2f} tof {q['tof']:.4f} "
            f"feas {q['feas']} CHORD_miss {q['chord']:9.1f} understated_by {q['chord'] / max(q['lay'], 1e-9):7.1f}x "
            f"leth {q['leth']}"
        )
    if fires:
        print(f"    max lead_error over ALL fire steps = {max(q['lead'] for q in fires):.2f} deg")
