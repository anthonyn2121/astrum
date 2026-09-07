"""Fine sweep of the cause-C counterfactual: pull-only vs push, at 0.1 s resolution, two throttles.

Cause C claims the roll the PULL-ONLY elevator forces is what spends the 6 s D2 window. The test
that can break it: sweep the up-half duration finely and ask whether the PUSH arm (zero roll, the
half-range the decode forfeits) ever scores a D2 where the PULL arm does not.

Throttle is swept too because dgamma/dt = g*n/V: idle bleeds V and RAISES the flight-path rate, and
throttle IS a genome channel, so a pull-only schedule at idle is inside the decode's own command set.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_refute_c_push as P
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def main() -> None:
    n_steps = int(round(9.0 / E.DT))
    log("#" * 100)
    log(
        "#  CAUSE C, FINE SWEEP -- does removing the roll (push) ever produce a D2 where pull cannot?"
    )
    log("#" * 100)
    log(
        f"  cell merge1500 seed {E.SEED}; {n_steps} steps = {n_steps * E.DT:.2f} s; window 6.0 s from tick 0"
    )
    log(
        "  swing >= 30 and |gpk| >= 15 and lf >= 0.5 and a return inside the window are ALL required."
    )
    log("")
    log(
        f"  {'arm':<18} {'thr':>4} {'T_up':>5} {'swing':>6} {'gpk':>7} {'tpk':>5} {'lf':>4} "
        f"{'ret':>5} {'tret':>5} {'g@6.0s':>7} {'dn_rate':>8} {'D2':>3} {'D2c':>4} {'dalt':>7} "
        f"{'vt@pk':>6} {'PASS?':>6}"
    )
    n_rows = 0
    n_pass = 0
    for thr in (1.0, 0.0):
        for t_up in (2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2):
            for name, phases, gate in (
                ("A PULL115", [(t_up, 0.0, 1.0, thr), (30.0, 115.0, 1.0, thr)], True),
                ("B PUSH0", [(t_up, 0.0, 1.0, thr), (30.0, 0.0, -1.0, thr)], True),
                ("C PULL115_NOGATE", [(t_up, 0.0, 1.0, thr), (30.0, 115.0, 1.0, thr)], False),
            ):
                ctl = P.Sched2(phases, gate=gate)
                r, rows = E.fly(ctl, n_steps=n_steps)
                st = P.d2_clauses(r.trace)
                det = B.detect_all(r.trace)
                # gamma at exactly 6.00 s, the window edge
                k6 = min(range(len(r.trace)), key=lambda k: abs(r.trace[k].t - 6.0))
                g6 = math.degrees(r.trace[k6].gamma)
                pk = st.get("t_peak", float("nan"))
                kpk = min(range(len(rows)), key=lambda k: abs(rows[k]["t"] - pk))
                ok = len(det["yoyos"]) > 0
                n_rows += 1
                n_pass += 1 if ok else 0
                log(
                    f"  {name:<18} {thr:>4.1f} {t_up:>5.2f} {st.get('swing', float('nan')):>6.1f} "
                    f"{st.get('gpk', float('nan')):>+7.1f} {pk:>5.2f} {st.get('lf', float('nan')):>4.2f} "
                    f"{str(st.get('ret')):>5} {st.get('t_ret', float('nan')):>5.2f} {g6:>+7.1f} "
                    f"{st.get('down_rate', float('nan')):>+8.2f} {len(det['yoyos']):>3} "
                    f"{len(det['yoyos_clean']):>4} {r.dalt_m:>+7.0f} {rows[kpk]['vt']:>6.1f} "
                    f"{'YES' if ok else 'no':>6}"
                )
    log(f"\n  D2 events over {n_rows} rollouts: {n_pass}. Base rate {n_pass}/{n_rows}.")
    path = f"{E.OUT}/refute_c_push2.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
