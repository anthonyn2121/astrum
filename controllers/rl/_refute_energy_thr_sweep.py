"""Throttle sweep for the D3b energy claim: is native throttle 1.000 "full afterburner", and how
much of the trajectory's Es gain is attributable to the throttle step at all?

Same cell / seed / window as `_refute_energy_baseline.py`. The stick is held at the ENERGY-MAXIMAL
setting (wings level, fully unloaded) so throttle is the ONLY thing varying, and the F100-PW-229
deck is `augmented=1 / augmethod=2 / milthrust 17800 / maxthrust 29000 lbf` — if augmentation
engages inside the commandable throttle range there should be a visible knee near the top of it.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _refute_energy_thr_sweep.py'
"""

from __future__ import annotations

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight.geometry import specific_energy

SEED = 62
N_STEPS = 173
HORIZON_STEPS = 576
HORIZON_S = HORIZON_STEPS * M.CONTROL_DT_S
CELL = "merge1500"
G = 9.80665


def fly(a_thr: float) -> tuple:
    gm = M.Genome(6, 3, "bank", True)
    g = np.array([0.0, -1.0, a_thr] * 6, np.float64)  # wings level, fully unloaded
    env, base = M.build_env(M.CELLS[CELL], SEED, HORIZON_S)
    rec: list[tuple] = []

    def ctl(step, me, bd, rel):
        a = gm.act(g, step, HORIZON_STEPS, me)
        rec.append((specific_energy(me), float(me.alt), float(me.airspeed)))
        return a

    M.roll_once(env, base, ctl, N_STEPS, keep=False)
    env.close()
    dEs = rec[-1][0] - rec[0][0]
    return dEs, rec[-1][1] - rec[0][1], rec[-1][2] - rec[0][2], len(rec)


def main() -> None:
    print(
        f"cell {CELL}  seed {SEED}  window {N_STEPS} steps / {(N_STEPS - 1) * M.CONTROL_DT_S:.3f} s"
    )
    print("stick held wings-level + fully UNLOADED so throttle is the only variable\n")
    print(
        f"{'agent a[3]':>11} {'NATIVE thr':>11} {'dEs (m)':>10} {'dAlt (m)':>10} {'dV (m/s)':>10} {'Ps (m/s)':>10}"
    )
    prev = None
    for a3 in (-1.0, -0.5, 0.0, 0.2047, 0.5, 0.6023, 0.8, 0.9, 0.95, 0.99, 1.0):
        dEs, dAlt, dV, n = fly(a3)
        nat = 0.5 * (a3 + 1.0)
        ps = dEs / ((n - 1) * M.CONTROL_DT_S)
        jump = "" if prev is None else f"   (d vs prev native step: {dEs - prev:+.1f} m)"
        print(
            f"{a3:>+11.4f} {nat:>11.4f} {dEs:>+10.1f} {dAlt:>+10.1f} {dV:>+10.2f} {ps:>+10.1f}{jump}"
        )
        prev = dEs


if __name__ == "__main__":
    main()
