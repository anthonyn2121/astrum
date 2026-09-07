"""Is the prereg's pooled D3 = 3 three distinct turning-room builds, or one trajectory counted 3x?"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
g = np.asarray(z["genome"], float)
gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
cell = M.CELLS[str(z["cell"])]
N = 576

print("seed  steps  D3  events (i0,i1,cpa_entry,cpa_peak,cpa_exit,build_m)")
for sd in (28, 79, 96):
    env, base = M.build_env(cell, sd, N * M.CONTROL_DT_S)
    r = M.roll_once(env, base, lambda s, me, bd, rel: gm.act(g, s, N, me), N, keep=True)
    env.close()
    evs = B.detect_turning_room(r.trace)
    desc = "; ".join(
        f"({e.i0},{e.i1},{e.detail['cpa_entry_m']:.2f},{e.detail['cpa_peak_m']:.2f},"
        f"{e.detail['cpa_exit_m']:.2f},{e.detail['cpa_peak_m'] - e.detail['cpa_entry_m']:+.2f})"
        for e in evs
    )
    print(f"{sd:>4}  {r.steps:>5}  {len(evs):>2}  {desc}")
