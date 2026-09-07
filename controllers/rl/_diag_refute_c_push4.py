"""Cross-seed control for the cause-C counterfactual: is the pull/push D2 split seed-dependent?

`sim.seed` only seeds gun dispersion, and these controllers are opponent-blind, so the split should
reproduce on any seed. That is an assumption until it is measured, so measure it: arms A (pull-only,
the decode) and B (push, zero roll) at the same up-half durations, on the search's other seeds.
"""

from __future__ import annotations

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
    log("#  CAUSE C -- cross-seed control (thr 1.0)")
    log("#" * 100)
    log(
        f"  {'seed':>5} {'arm':<10} {'T_up':>5} {'swing':>6} {'ret':>5} {'tret':>5} {'D2':>3} "
        f"{'D2c':>4} {'failAsp':>8} {'failExit':>9} {'dalt':>7}"
    )
    tot = {"A PULL115": [0, 0], "B PUSH0": [0, 0]}
    for seed in (11, 45, 130):
        for t_up in (2.5, 2.6, 2.7, 2.8):
            for name, phases in (
                ("A PULL115", [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, 1.0, 1.0)]),
                ("B PUSH0", [(t_up, 0.0, 1.0, 1.0), (30.0, 0.0, -1.0, 1.0)]),
            ):
                r, rows = E.fly(P.Sched2(phases, gate=True), n_steps=n_steps, seed=seed)
                st = P.d2_clauses(r.trace)
                det = B.detect_all(r.trace)
                fa = any(e.detail.get("failed_aspect") for e in det["yoyos"])
                fe = any(e.detail.get("failed_exit") for e in det["yoyos"])
                tot[name][0] += 1
                tot[name][1] += len(det["yoyos"])
                log(
                    f"  {seed:>5} {name:<10} {t_up:>5.2f} {st.get('swing', float('nan')):>6.1f} "
                    f"{str(st.get('ret')):>5} {st.get('t_ret', float('nan')):>5.2f} "
                    f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4} {str(fa):>8} {str(fe):>9} "
                    f"{r.dalt_m:>+7.0f}"
                )
    for k, (n, d2) in tot.items():
        log(f"  {k}: {d2} D2 events in {n} rollouts (base rate {d2}/{n})")
    path = f"{E.OUT}/refute_c_push4.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
