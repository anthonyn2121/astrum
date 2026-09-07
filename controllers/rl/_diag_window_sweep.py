import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M

d = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)
g0 = np.asarray(d["genome"], float)
cell = M.CELLS["merge1500"]
gm = M.Genome(6, 3, "bank", True)
HOR = 24.0
NST = int(round(HOR / M.CONTROL_DT_S))
print(
    "n_steps",
    NST,
    "dt",
    M.CONTROL_DT_S,
    "R_LETHAL_M",
    M.R_LETHAL_M,
    "FIRE_LAY_MISS_M",
    M.FIRE_LAY_MISS_M,
    "HIT_RADIUS_M",
    M.C.HIT_RADIUS_M,
)


def fly(g, seed):
    env, base = M.build_env(cell, seed, HOR)
    try:

        def ctl(step, me, bd, rel):
            return gm.act(g, step, NST, me)

        r = M.roll_once(env, base, ctl, NST, keep=True)
    finally:
        env.close()
    fire = sum(1 for a in r.actions if float(a[4]) > 0.0)
    return r, fire


print()
print("SEG-1 THROTTLE SWEEP (gene g[5]; baseline 1.0 => thr 1.00), seed 62")
print(" g[5]   thr1   event      kill  hp    steps  t600_s  tleth_s  fireStp fire_s  rngmin  dalt")
for gv in [1.0, 0.6, 0.2, -0.2, -0.6, -1.0]:
    g = g0.copy()
    g[5] = gv
    r, fire = fly(g, 62)
    print(
        f"{gv:+5.2f}  {0.5 * (gv + 1):5.2f}  {str(r.event):<10} {r.kill:>3}  {r.hp_dealt:5.1f} {r.steps:6d} "
        f"{r.t600_s:7.3f} {r.tleth_s:8.3f} {fire:7d} {fire * M.CONTROL_DT_S:6.3f} {r.rng_min:7.2f} {r.dalt_m:7.0f}"
    )
