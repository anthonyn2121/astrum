"""VERIFY the proposed D3 magnitude floor by EXECUTING a patched detector on all 8 search seeds.

The claim under test asserts (a) the kill trace is unchanged, (b) pooled merge1500 D3 goes 3 -> 0
under `tr[j].cpa_miss > tr[i].cpa_miss + max(50.0, 0.10*tr[i].cpa_miss)` at bfm_detect.py:481, and
its author states the patched detector was NOT executed. This runs it, plus the base-rate check
(are the three pooled D3 events three DISTINCT events, or one trajectory counted three times?).
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
HZ = 24.0
NST = 576


def d3_holds(tr, floor: bool):
    """detect_turning_room verbatim (bfm_detect.py:462-499), optionally with the magnitude floor."""
    out = []
    if len(tr) < 3:
        return out
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    for i, j in B._runs(hold):
        if tr[j].t - tr[i].t < B.CHI_MIN_HOLD_S:
            continue
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        one_side = len(sides) == 1
        bar = max(50.0, 0.10 * tr[i].cpa_miss) if floor else 0.0
        built = tr[j].cpa_miss > tr[i].cpa_miss + bar
        after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
        cashed = bool(after) and min(tr[k].cpa_miss for k in after) < tr[j].cpa_miss
        exit_idx = min(after, key=lambda k: tr[k].cpa_miss) if after else j
        out.append(
            dict(
                i=i,
                j=j,
                dur=tr[j].t - tr[i].t,
                one_side=one_side,
                built=built,
                cashed=cashed,
                bar=bar,
                cpa_i=tr[i].cpa_miss,
                cpa_j=tr[j].cpa_miss,
                build=tr[j].cpa_miss - tr[i].cpa_miss,
                cash=(tr[exit_idx].cpa_miss - tr[j].cpa_miss) if after else float("nan"),
                cash_margin=(tr[j].cpa_miss - min(tr[k].cpa_miss for k in after))
                if after
                else float("nan"),
                alt_i=tr[i].alt,
                alt_j=tr[j].alt,
                alt_x=tr[exit_idx].alt,
                nz_lo=min(tr[k].nz for k in range(i, j + 1)),
                nz_hi=max(tr[k].nz for k in range(i, j + 1)),
                rng_i=tr[i].rng,
                rng_j=tr[j].rng,
                event=(one_side and built and cashed),
                exit_idx=exit_idx,
            )
        )
    return out


def main():
    z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
    g = np.asarray(z["genome"], np.float64)
    gm = M.Genome(6, 3, "bank", True)
    cell = M.CELLS["merge1500"]
    print(f"genome {g.tolist()}")
    print(f"cell merge1500 = {cell}")

    series = {}
    print("\n=== PER-SEED: real detector vs PATCHED detector (floor executed, not arithmetic)")
    tot_base = tot_floor = 0
    for sd in SEEDS:
        env, base = M.build_env(cell, sd, HZ)
        r = M.roll_once(env, base, lambda st, me, bd, rel: gm.act(g, st, NST, me), NST, keep=True)
        env.close()
        det = B.detect_all(r.trace)
        hb = d3_holds(r.trace, floor=False)
        hf = d3_holds(r.trace, floor=True)
        nb = sum(h["event"] for h in hb)
        nf = sum(h["event"] for h in hf)
        tot_base += nb
        tot_floor += nf
        series[sd] = np.array([s.cpa_miss for s in r.trace], np.float64)
        print(
            f"\n seed {sd}: steps {r.steps} event={r.event} kill={r.kill} hp={r.hp_dealt:.1f} "
            f"dalt {r.dalt_m:+.0f} m | B.detect_all D1={len(det['displacement_rolls'])} "
            f"D2={len(det['yoyos'])} D2c={len(det['yoyos_clean'])} D3={len(det['turning_room'])} "
            f"| my D3 base={nb} patched={nf}"
        )
        assert nb == len(det["turning_room"]), "my re-implementation disagrees with B.detect_all"
        for h, hp in zip(hb, hf):
            print(
                f"   hold {h['i']:3d}-{h['j']:3d} dur {h['dur']:.3f}s nz {h['nz_lo']:.2f}-{h['nz_hi']:.2f} "
                f"cpa {h['cpa_i']:8.1f}->{h['cpa_j']:8.1f} build {h['build']:+8.1f} m "
                f"({100.0 * h['build'] / h['cpa_i']:+7.2f}%) bar {hp['bar']:7.1f} "
                f"margin_to_bar {h['build'] - hp['bar']:+8.1f} m | one_side={h['one_side']} "
                f"built={h['built']}->{hp['built']} cashed={h['cashed']} "
                f"cash_margin {h['cash_margin']:+7.1f} m | "
                f"alt {h['alt_i']:.0f}->{h['alt_j']:.0f} ({h['alt_j'] - h['alt_i']:+.0f} m in "
                f"{h['dur']:.3f}s = {(h['alt_j'] - h['alt_i']) / max(h['dur'], 1e-9):+.0f} m/s) "
                f"alt_exit {h['alt_x']:.0f} ({h['alt_x'] - h['alt_i']:+.0f} m) "
                f"rng {h['rng_i']:.0f}->{h['rng_j']:.0f} | EVENT {h['event']}->{hp['event']}"
            )
    print(f"\n POOLED merge1500 D3: base={tot_base}  patched={tot_floor}")

    print("\n=== BASE RATE: are the three pooled D3 events DISTINCT trajectories?")
    for a in (28, 79, 96):
        for b in (28, 79, 96):
            if a < b:
                n = min(len(series[a]), len(series[b]))
                d = float(np.max(np.abs(series[a][:n] - series[b][:n])))
                print(f"   max|cpa_miss diff| seed {a} vs {b} over {n} ticks = {d:.3e} m")
    n = min(len(series[62]), len(series[28]))
    print(
        f"   max|cpa_miss diff| seed 62 (kill) vs 28 over the {n}-tick overlap = "
        f"{float(np.max(np.abs(series[62][:n] - series[28][:n]))):.3e} m"
    )


if __name__ == "__main__":
    main()
