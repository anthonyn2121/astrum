"""Is "one D2 candidate entry per fight in ANY merge" a property of line 411, or of ONE trajectory?

Flies several controllers in the SAME cell (merge1500) and counts, per rollout:
  * ticks with closure > CLOSURE_OVERSHOOT and the number of CONTIGUOUS runs above the band
  * D2 rising-edge candidates (bfm_detect.py:411 verbatim)
Nothing here re-implements a detector: `roll_once` is the search's own loop and the samples are
`B.make_sample`, so the traces are the exact objects `detect_yoyos` reads.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && uv run python _diag_d2_edge_generality.py'
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm as BFM
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi


def rising_edges(tr):
    return [
        i
        for i in range(len(tr))
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]


def runs_above(tr):
    mask = [s.closure > B.CLOSURE_OVERSHOOT for s in tr]
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


# ---------------------------------------------------------------------------- controllers
def make_genome_cycled():
    z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
    g = np.asarray(z["genome"], float)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)

    def ctl(step, me, bd, rel):
        return gm.act(g, step % 576, 576, me)  # keep 4 s segments, repeat the 24 s programme

    return "d3b genome, cycled every 24 s", ctl


def make_pursuit():
    def ctl(step, me, bd, rel):
        ail, elev = BFM._point_at(rel, lag_rad=math.radians(2.0))
        ail, elev = BFM._deck_guard(me, ail, elev)
        return np.array([ail, elev, 0.0, 1.0, -1.0], np.float32)

    return "pure pursuit, mil power (closed loop on the bandit)", ctl


def make_lag_track():
    mv = BFM.LagPursuitTracker()
    mv.max_steps = 10**9  # never self-cap: we want the whole horizon

    def ctl(step, me, bd, rel):
        if step == 0:
            mv.reset(me, rel)
        a, _done = mv.step(me, rel)
        return np.array([a[0], a[1], a[2], a[3], -1.0], np.float32)

    return "lag-pursuit tracker (bfm.LagPursuitTracker)", ctl


def make_level():
    def ctl(step, me, bd, rel):
        return np.array([M.bfm.KROLL * (0.0 - float(me.phi)), 0.0, 0.0, 0.7, -1.0], np.float32)

    return "wings-level, no pull (a straight-through merge)", ctl


def main():
    cell = M.CELLS["merge1500"]
    arms = [make_genome_cycled(), make_pursuit(), make_lag_track(), make_level()]
    for horizon_s in (24.0, 60.0, 120.0):
        n_steps = int(round(horizon_s / M.CONTROL_DT_S))
        print(
            f"\n{'=' * 100}\n cell merge1500   horizon {horizon_s:.0f} s = {n_steps} control steps   seed 28\n{'=' * 100}"
        )
        print(
            f" {'arm':<52} {'ticks':>6} {'>band':>6} {'%':>6} {'runs':>5} {'D2cand':>7} {'D2':>4} {'event':>10}"
        )
        for name, ctl in arms:
            env, base = M.build_env(cell, 28, n_steps * M.CONTROL_DT_S)
            try:
                r = M.roll_once(env, base, ctl, n_steps, keep=False)
            finally:
                env.close()
            tr = r.trace
            ra = runs_above(tr)
            ed = rising_edges(tr)
            nb = sum(1 for s in tr if s.closure > B.CLOSURE_OVERSHOOT)
            d2 = B.detect_yoyos(tr)
            print(
                f" {name:<52} {len(tr):>6} {nb:>6} {100.0 * nb / max(1, len(tr)):>5.1f}% "
                f"{len(ra):>5} {len(ed):>7} {len(d2):>4} {str(r.event):>10}"
            )
            if len(ed) > 1:
                print(f"      candidate entry steps: {ed}")
                print(f"      runs above band:       {ra}")
                for i in ed:
                    print(
                        f"        entry {i:>4} t {tr[i].t:6.3f} s  g0 {tr[i].gamma * DEG:+7.2f} deg  clo {tr[i].closure:+7.1f} m/s  rng {tr[i].rng:8.1f} m"
                    )


if __name__ == "__main__":
    main()
