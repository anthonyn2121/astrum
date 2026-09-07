"""Third probe: (a) price the closure-floor counterexamples in ALTITUDE and load, and
(b) re-fly them on the NON-kill seed 28 to show the window is seed-invariant.

The counterexample must not be a c87-style unpriced descent: report dalt over the window and the
loaded fraction (nz > NZ_LOADED) up to the closure minimum.
"""

from __future__ import annotations

import _diag_d3b_express as E
import numpy as np
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


CASES = [
    (
        "16-grid best   FINE 0.5 b0 / 1.0 b115 / rest b0 [ld 1.00 thr 1.00]",
        [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
    ),
    ("counterexample bank0 pull all the way          [ld 1.00 thr 1.00]", [(24.0, 0.0, 1.0, 1.0)]),
    ("counterexample bank0 pull all the way          [ld 0.60 thr 1.00]", [(24.0, 0.0, 0.6, 1.0)]),
    (
        "counterexample b0 pull 2.0s then bank-115 pull [ld 1.00 thr 1.00]",
        [(2.0, 0.0, 1.0, 1.0), (22.0, -115.0, 1.0, 1.0)],
    ),
]


def main() -> None:
    log("#" * 128)
    log("#  VERIFY 3: altitude price of the closure-floor counterexamples, and seed invariance")
    log("#" * 128)
    log(
        f"\n  window = [0, {B.YOYO_RETURN_S:.0f}] s from the tick-0 entry edge = steps 0..144."
        f"  bars: closure {B.CLOSURE_OVERSHOOT:.0f} m/s, rng {B.YOYO_EXIT_RNG_M:.0f} m, nz {B.NZ_LOADED:.1f} g\n"
    )
    log(
        f"     {'case':<66} {'seed':>4} {'cloMin':>7} {'k*':>4} {'t*':>5} {'rng*':>6} "
        f"{'alt0':>7} {'alt*':>7} {'dalt':>7} {'ldFrac':>6} {'nzMx':>5} {'gam*':>6} {'bled?':>5}"
    )
    per_case: dict[str, list[float]] = {}
    for name, phases in CASES:
        for sd in (62, 28):
            r, rows = E.fly(E.Sched(phases), seed=sd)
            tr = r.trace
            i = 0
            assert tr[0].closure > B.CLOSURE_OVERSHOOT, "tick-0 entry edge assumed"
            win = [k for k in range(i, len(tr)) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
            clo = np.array([tr[k].closure for k in win])
            ks = win[int(np.argmin(clo))]
            loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, ks + 1)]
            bled = tr[ks].closure < B.CLOSURE_OVERSHOOT and tr[ks].rng < B.YOYO_EXIT_RNG_M
            a0, ax = rows[0]["alt"], rows[ks]["alt"]
            log(
                f"     {name:<66} {sd:>4} {float(clo.min()):>+7.1f} {ks:>4} {tr[ks].t:>5.2f} "
                f"{tr[ks].rng:>6.0f} {a0:>7.1f} {ax:>7.1f} {ax - a0:>+7.1f} "
                f"{sum(loaded) / len(loaded):>6.2f} {max(tr[k].nz for k in range(i, ks + 1)):>5.2f} "
                f"{np.degrees(tr[ks].gamma):>+6.1f} {str(bled):>5}"
            )
            per_case.setdefault(name, []).append(float(clo.min()))
    log("")
    for name, v in per_case.items():
        log(
            f"     seed 62 vs 28 window closure floor, {name.split()[0]:<14}: {v[0]:+.6f} vs {v[1]:+.6f}  (delta {v[0] - v[1]:+.3e})"
        )
    path = f"{E.OUT}/verify_d2_closure_floor3.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
