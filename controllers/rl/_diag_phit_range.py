"""p_hit per round vs RANGE, for the lay the fire control can actually COMPUTE.

The whole "0 HP off 230 rounds at a 2 m predicted lay" result turns on one question:
at the ranges the merge is fought (395-526 m closest approach), what fraction of
rounds can physically connect?

Two lays, one variable:
  COMPUTABLE  = gunlay.intercept()      -- constant-velocity, no drag, no drop. This is
                what a real FCS must assume, what geometry.py:227 `lay_miss_m` scores,
                and therefore what every reward/diag number in this program is built on.
  CLAIRVOYANT = gunlay.intercept_path() -- solved against the arc the bandit ACTUALLY
                flew. The upper bound; not available to any policy.

Both are scored by the repo's own tick-exact propagator (drag + gravity + Coriolis +
the real 8 mrad dispersion cone) against the true arc, so the gap between the columns
is exactly the cost of not knowing the future.

Geometry: pure tail chase (the most favourable case) at merge altitude, bandit in a
level n-g turn. HITS_TO_KILL = 50, so `rds` = rounds needed for a kill = 50/p.
"""

from __future__ import annotations

import numpy as np

from astrum_dogfight import gunlay
from astrum_dogfight.gunlay import G0, MUZZLE_SPEED_MPS, SIM_DT

ALT_M = 5000.0  # merge fights at 15,000-17,000 ft
V_TGT = 200.0
V_SHOOTER = 230.0
RANGES = [200.0, 300.0, 395.0, 440.0, 500.0, 600.0, 700.0, 900.0, 1200.0]
GEES = [1.0, 3.0, 5.0]
SAMPLES = 6000


def true_arc(v: float, n: float, horizon_s: float) -> np.ndarray:
    """(M,3) ENU path of a level n-g turn starting at the origin heading +E."""
    if n <= 1.0:
        ticks = np.arange(int(horizon_s / SIM_DT) + 2) * SIM_DT
        return np.stack([v * ticks, np.zeros_like(ticks), np.full_like(ticks, ALT_M)], axis=1)
    omega = G0 * np.sqrt(n * n - 1.0) / v
    r = v / omega
    ticks = np.arange(int(horizon_s / SIM_DT) + 2) * SIM_DT
    th = omega * ticks
    return np.stack([r * np.sin(th), r * (1.0 - np.cos(th)), np.full_like(th, ALT_M)], axis=1)


def main() -> None:
    print(f"HITS_TO_KILL = {gunlay.HITS_TO_KILL} | dispersion {gunlay.DISPERSION_RAD*1e3:.1f} mrad "
          f"| muzzle {MUZZLE_SPEED_MPS:.0f} m/s | alt {ALT_M:.0f} m | Vt {V_TGT:.0f} Vs {V_SHOOTER:.0f}")
    for n in GEES:
        omega = 0.0 if n <= 1.0 else np.degrees(G0 * np.sqrt(n * n - 1.0) / V_TGT)
        print(f"\n=== bandit {n:.0f} g  ({omega:.1f} deg/s turn) " + "=" * 40)
        print(f"{'rng':>6} | {'p COMPUTABLE':>12} {'rds/kill':>9} | {'p CLAIRVOY':>10} {'rds/kill':>9} "
              f"| {'perfect-lay miss':>16}")
        for R in RANGES:
            horizon = 2.5 * R / MUZZLE_SPEED_MPS + 0.05
            arc = true_arc(V_TGT, n, horizon)
            tp, tv = arc[0], (arc[1] - arc[0]) / SIM_DT
            # shooter directly astern, co-heading, at range R
            sp = tp - np.array([R, 0.0, 0.0])
            sv = np.array([V_SHOOTER, 0.0, 0.0])
            path = np.concatenate([np.arange(len(arc))[:, None] * SIM_DT, arc], axis=1)

            comp = gunlay.intercept(sp, sv, tp, tv)
            clair = gunlay.intercept_path(sp, sv, path)
            row = [f"{R:>6.0f} |"]
            for sol in (comp, clair):
                if not sol.feasible:
                    row.append(f"{'infeas':>12} {'-':>9} |")
                    continue
                p, _ = gunlay.p_hit_vec(
                    sol.boresight, sp, sv, tp, tv, samples=SAMPLES, seed=7,
                    target_path=path, alt_offset=0.0,
                )
                need = "never" if p < 1e-4 else f"{gunlay.HITS_TO_KILL / p:>9.0f}"
                row.append(f"{p:>12.3f} {need:>9} |")
            # systematic miss of a "perfect" (lay_miss_m == 0) shot: dispersion off
            p0, miss = gunlay.p_hit_vec(
                comp.boresight, sp, sv, tp, tv, samples=1, seed=0,
                target_path=path, sigma=0.0, alt_offset=0.0,
            )
            row.append(f"{miss:>14.2f} m")
            print(" ".join(row))


if __name__ == "__main__":
    main()
