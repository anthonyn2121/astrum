"""Can the PUSH recover D2 at the ONLY up-half duration the 18-dim genome can name (4.0 s)?"""

from __future__ import annotations

import _diag_d3b_express as E
import _diag_refute_c_push as P
from astrum_dogfight import bfm_detect as B

n_steps = int(round(9.0 / E.DT))
print(
    f"{'arm':<12} {'T_up':>5} {'swing':>6} {'gpk':>7} {'tpk':>5} {'ret':>5} {'D2':>3} {'D2c':>4} {'dalt':>7}"
)
for t_up in (3.5, 4.0, 4.5):
    for name, phases in (
        ("A PULL115", [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, 1.0, 1.0)]),
        ("B PUSH0", [(t_up, 0.0, 1.0, 1.0), (30.0, 0.0, -1.0, 1.0)]),
    ):
        r, rows = E.fly(P.Sched2(phases, gate=True), n_steps=n_steps)
        st = P.d2_clauses(r.trace)
        det = B.detect_all(r.trace)
        print(
            f"{name:<12} {t_up:>5.2f} {st.get('swing', float('nan')):>6.1f} "
            f"{st.get('gpk', float('nan')):>+7.1f} {st.get('t_peak', float('nan')):>5.2f} "
            f"{str(st.get('ret')):>5} {len(det['yoyos']):>3} {len(det['yoyos_clean']):>4} {r.dalt_m:>+7.0f}"
        )
