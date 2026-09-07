"""Compare the proposed MAGNITUDE floor against the sibling detector's own CLOSURE clause on D3.

D1 (`detect_displacement`, bfm_detect.py:348) carries `if tr[i].closure <= 0.0: continue` as clause
(c); D3 (`detect_turning_room`, 462-499) omits it. `cpa_miss_m` (181-195) clamps `t` to 0 once the
CPA is behind the jet, so on those ticks `cpa_miss == rng` and a "build" is range opening. This runs
three D3 variants on the same rollouts: as-is, +magnitude floor, +closure clause.
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

SEEDS = [28, 62]
NST = 576


def d3(tr, variant: str) -> int:
    n = 0
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    for i, j in B._runs(hold):
        if tr[j].t - tr[i].t < B.CHI_MIN_HOLD_S:
            continue
        if len({1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}) != 1:
            continue
        if variant == "closure" and tr[i].closure <= 0.0:
            continue
        bar = max(50.0, 0.10 * tr[i].cpa_miss) if variant == "floor" else 0.0
        if tr[j].cpa_miss <= tr[i].cpa_miss + bar:
            continue
        after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
        if not after or min(tr[k].cpa_miss for k in after) >= tr[j].cpa_miss:
            continue
        n += 1
    return n


def main():
    g = np.asarray(np.load("runs/d3b/d3b_best.npz")["genome"], np.float64)
    gm = M.Genome(6, 3, "bank", True)
    for sd in SEEDS:
        env, base = M.build_env(M.CELLS["merge1500"], sd, 24.0)
        r = M.roll_once(env, base, lambda st, me, bd, rel: gm.act(g, st, NST, me), NST)
        env.close()
        base_n = len(B.detect_all(r.trace)["turning_room"])
        print(
            f" seed {sd}: steps {r.steps} kill={r.kill} | D3 as-is {d3(r.trace, 'asis')} "
            f"(B.detect_all {base_n}) | +magnitude floor {d3(r.trace, 'floor')} "
            f"| +closure clause (D1's own clause c) {d3(r.trace, 'closure')}"
        )


if __name__ == "__main__":
    main()
