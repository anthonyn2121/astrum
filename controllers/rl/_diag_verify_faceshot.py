"""Adversarial check of the 'face shot is shootable because cpa_miss -> 5.2 m' claim.

Three controls, all flying the SAME honest trigger `_diag_true_mpc._trigger`:
  A  the d3b_best genome on merge1500 (baseline; must reproduce the replay CSV)
  B  a NULL own-ship input (bank target 0, load 0, same throttle as genome seg 0) on merge1500
     -- isolates how much of the cpa_miss collapse is the BANDIT's own commit rather than
     'the lead turn'
  C  the same NULL input on merge0, the COLLISION-COURSE cell (`_diag_true_mpc.CELLS['merge0']`)
     -- the geometry the claim says is un-shootable. Measures its cpa_miss and alpha_req.
  D  the genome on merge0, for completeness.

Prints per-step scalars at the rows that matter; no derived statistic is estimated.
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import constants as C

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
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / M.CONTROL_DT_S))
GM = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)


def genome_ctrl(step, me, bd, rel):
    return GM.act(GENOME, step, N_STEPS, me)


def null_ctrl(thr_cmd):
    """Wings-level, no commanded pull, fixed throttle -- the same inner loop as the genome."""

    def fn(step, me, bd, rel):
        ail = float(np.clip(M.bfm.KROLL * (0.0 - float(me.phi)), -1.0, 1.0))
        return np.array([ail, 0.0, 0.0, thr_cmd, -1.0], np.float32)

    return fn


def fly(label, cell_key, ctrl, seed=62):
    env, base = M.build_env(M.CELLS[cell_key], seed, HORIZON_S)
    rows = []

    def rec(step, me, bd, rel):
        a = np.asarray(ctrl(step, me, bd, rel), np.float32)
        rows.append(
            dict(
                step=step,
                t=step * M.CONTROL_DT_S,
                rng=rel.rng,
                cpa=M.B.cpa_miss_m(me.pos, me.vel, bd.pos, bd.vel),
                alpha_req=math.degrees(rel.alpha_req),
                ata=math.degrees(rel.ata),
                aspect=math.degrees(rel.aspect),
                lay=rel.lay_miss_m,
                lead=math.degrees(rel.lead_error),
                alpha=math.degrees(me.alpha),
                alt=float(me.alt),
                nz=float(me.nz),
                phi=math.degrees(float(me.phi)),
                hp=float(bd.health),
                fire=M._trigger(rel) > 0.0,
                lethal=(rel.rng <= M.R_LETHAL_M and rel.lay_miss_m < C.HIT_RADIUS_M),
            )
        )
        return a

    r = M.roll_once(env, base, rec, N_STEPS, keep=False)
    env.close()
    print(f"\n===== {label}  cell={cell_key} seed={seed} =====")
    print(
        f"  event={r.event} kill={r.kill} hp_dealt={r.hp_dealt:.1f} tleth={r.tleth_s:.3f} "
        f"steps={r.steps} rounds={r.rounds} dalt={r.dalt_m:.0f} m  rows={len(rows)}"
    )
    cmin = min(rows, key=lambda x: x["cpa"])
    rmin = min(rows, key=lambda x: x["rng"])
    print(
        f"  min cpa_miss = {cmin['cpa']:.2f} m @ step {cmin['step']} t {cmin['t']:.3f} "
        f"(rng {cmin['rng']:.1f}, alpha_req {cmin['alpha_req']:.2f} deg)"
    )
    print(
        f"  min rng      = {rmin['rng']:.2f} m @ step {rmin['step']} t {rmin['t']:.3f} "
        f"(cpa {rmin['cpa']:.1f}, alpha_req {rmin['alpha_req']:.2f} deg, ata {rmin['ata']:.1f})"
    )
    inr = [x for x in rows if x["rng"] <= 600.0]
    if inr:
        b = min(inr, key=lambda x: x["alpha_req"])
        print(
            f"  rows with rng<=600 m: {len(inr)} ({len(inr) * M.CONTROL_DT_S:.3f} s); "
            f"min alpha_req there = {b['alpha_req']:.2f} deg @ step {b['step']} "
            f"(rng {b['rng']:.1f}, cpa {b['cpa']:.1f}, own alpha {b['alpha']:.2f})"
        )
    else:
        print("  rows with rng<=600 m: 0")
    fr = [x for x in rows if x["fire"]]
    lt = [x for x in rows if x["lethal"]]
    print(
        f"  fire rows {len(fr)} ({len(fr) * M.CONTROL_DT_S:.3f} s); "
        f"lethal rows {len(lt)} ({len(lt) * M.CONTROL_DT_S:.3f} s)"
    )
    if fr:
        print(
            f"    first fire: step {fr[0]['step']} t {fr[0]['t']:.3f} rng {fr[0]['rng']:.1f} "
            f"lay {fr[0]['lay']:.2f} alpha_req {fr[0]['alpha_req']:.2f}"
        )
        print(
            f"    alpha_req over fire rows: min {min(x['alpha_req'] for x in fr):.2f} "
            f"max {max(x['alpha_req'] for x in fr):.2f}"
        )
    if lt:
        print(
            f"    alpha_req over LETHAL rows: min {min(x['alpha_req'] for x in lt):.2f} "
            f"max {max(x['alpha_req'] for x in lt):.2f}"
        )
    # cpa_miss timeline on a 1 s grid, so 'who closed the CPA' is readable
    grid = [x for x in rows if x["step"] % 24 == 0]
    print("  t(s)  cpa_miss(m)   rng(m)  aspect  ata   alpha_req  alt(m)  phi   nz")
    for x in grid[:13]:
        print(
            f"  {x['t']:4.1f} {x['cpa']:11.1f} {x['rng']:8.1f} {x['aspect']:7.1f} "
            f"{x['ata']:6.1f} {x['alpha_req']:9.2f} {x['alt']:7.0f} {x['phi']:6.1f} {x['nz']:5.2f}"
        )
    return rows


def main():
    a = fly("A genome (baseline)", "merge1500", genome_ctrl)
    fly("B NULL input, merge1500", "merge1500", null_ctrl(0.6023))
    fly("C NULL input, merge0 COLLISION COURSE", "merge0", null_ctrl(0.6023))
    fly("D genome, merge0 COLLISION COURSE", "merge0", genome_ctrl)
    # sanity vs the replay CSV, printed as raw numbers to compare by eye
    print("\n=== A sanity rows (compare to replay_kill_seed62.csv) ===")
    for s in (0, 1, 88, 108, 156, 164, 171, 172):
        x = a[s]
        print(
            f"  step {s:3d} t {x['t']:.3f} cpa {x['cpa']:.3f} rng {x['rng']:.2f} "
            f"alpha_req {x['alpha_req']:.3f} lay {x['lay']:.3f} ata {x['ata']:.3f}"
        )


if __name__ == "__main__":
    main()
