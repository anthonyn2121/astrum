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

# Kill-free measurement of the PURE KINEMATIC window: disable the trigger entirely so no HP is
# dealt and the episode cannot terminate on the kill. Everything else is byte-identical.
_orig = M._trigger


def nofire(rel):
    return -1.0


def fly(g, seed, fire_on):
    M._trigger = _orig if fire_on else nofire
    env, base = M.build_env(cell, seed, HOR)
    try:

        def ctl(step, me, bd, rel):
            return gm.act(g, step, NST, me)

        r = M.roll_once(env, base, ctl, NST, keep=True)
    finally:
        env.close()
        M._trigger = _orig
    # recover per-step rng/lay_miss from the detector trace
    rng = np.array([s.rng for s in r.trace], float)
    lm = np.array([s.lay_miss_m for s in r.trace], float)
    return r, rng, lm


print("TRIGGER-DISABLED (no HP dealt => no kill truncation). Pure kinematic window inside 600 m.")
print("  seed  event      steps  t600_s  nIn600  tLay<=15  tLay<8  rngmin  hp_taken")
for seed in (62, 28):
    r, rng, lm = fly(g0, seed, False)
    in600 = rng <= M.R_LETHAL_M
    n = int(in600.sum())
    print(
        f"  {seed:4d}  {str(r.event):<10} {r.steps:5d} {r.t600_s:7.3f} {n:6d} "
        f"{int((in600 & (lm <= 15.0)).sum()) * M.CONTROL_DT_S:8.3f} "
        f"{int((in600 & (lm < 8.0)).sum()) * M.CONTROL_DT_S:7.3f} {rng.min():7.2f} {r.hp_taken:5.1f}"
    )

print()
print("SEG-1 BANK x LOAD sweep, TRIGGER DISABLED, seed 62 -- can ANY open-loop schedule hold")
print("the lethal ring longer than the baseline?  (g[3]=bank gene, g[4]=load gene)")
print("  g3     bank    g4     load   event      steps  t600_s  tLay<=15  rngmin")
base_b, base_l = g0[3], g0[4]
for gb in (base_b, 0.0, 0.8, -0.4):
    for gl in (base_l, 1.0, -1.0):
        g = g0.copy()
        g[3] = gb
        g[4] = gl
        r, rng, lm = fly(g, 62, False)
        in600 = rng <= M.R_LETHAL_M
        tag = "  <= BASELINE" if (gb == base_b and gl == base_l) else ""
        print(
            f"  {gb:+5.2f} {115 * gb:+7.1f} {gl:+5.2f} {0.5 * (gl + 1):6.3f}  {str(r.event):<10} {r.steps:5d} "
            f"{r.t600_s:7.3f} {int((in600 & (lm <= 15.0)).sum()) * M.CONTROL_DT_S:8.3f} {rng.min():7.2f}{tag}"
        )
