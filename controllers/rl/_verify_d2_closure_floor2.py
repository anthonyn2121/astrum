"""Second probe: can a genome-nameable schedule put a gamma RETURN and closure < 120 m/s at the
SAME index inside [0, 6] s?

The 16 schedules of `_diag_d2_express` only ever use bank targets 0 and +115 deg. p[0] = -1 is
nameable (BANK_MAX_DEG * clip(p0,-1,1)), and merge1500 has a 1500 m LATERAL offset, so the two roll
directions are not symmetric. This sweeps the return bank over both signs and the pull-up duration,
and reports, per trajectory:
  retIdx / cloRet / rngRet -- exactly what `bled` reads at the exit index
  bothMin -- the lowest closure among window indices that ALSO have rng < 1500 m
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import numpy as np
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def probe(tr) -> dict:
    n = len(tr)
    ed = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    if not ed:
        return dict(edges=0)
    i = ed[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
    ret = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    inr = [k for k in win if tr[k].rng < B.YOYO_EXIT_RNG_M]
    return dict(
        edges=len(ed),
        i0=i,
        swing=math.degrees(abs(tr[peak].gamma - g0)),
        gpk=math.degrees(tr[peak].gamma),
        lf=sum(loaded) / max(1, len(loaded)),
        ret=ret,
        clo_ret=float(tr[ret].closure) if ret is not None else float("nan"),
        rng_ret=float(tr[ret].rng) if ret is not None else float("nan"),
        clo_min=min(float(tr[k].closure) for k in win),
        both_min=min((float(tr[k].closure) for k in inr), default=float("nan")),
        n_in_rng=len(inr),
    )


def main() -> None:
    log("#" * 126)
    log("#  VERIFY 2: a gamma RETURN and closure < 120 m/s at the SAME index -- both bank signs")
    log("#" * 126)
    log(
        "\n  bothMin = lowest closure among window indices that ALSO satisfy rng < 1500 m, i.e. the"
        "\n  best `bled` a schedule offers ANYWHERE in the window. cloRet/rngRet = what `bled` would"
        "\n  actually read at the exit index D2 finds.\n"
    )
    log(
        f"     {'schedule':<52} {'stp':>4} {'i0':>3} {'swing':>6} {'gpk':>7} {'lf':>4} "
        f"{'ret':>5} {'cloRet':>7} {'rngRet':>7} {'cloMin':>7} {'bothMin':>8} {'nRng':>4} {'D2':>3} {'D2c':>4}"
    )
    rows = []
    for d_up in (2.0, 3.0, 4.0, 5.0):
        for bk in (115.0, -115.0, 90.0, -90.0):
            for th in (1.0, 0.0):
                name = f"b0 pull {d_up:.1f}s then bank{bk:+.0f} pull [thr {th:.2f}]"
                phases = [(d_up, 0.0, 1.0, th), (24.0, bk, 1.0, th)]
                r, _ = E.fly(E.Sched(phases))
                st = probe(r.trace)
                det = B.detect_all(r.trace)
                st.update(
                    name=name, D2=len(det["yoyos"]), D2c=len(det["yoyos_clean"]), steps=r.steps
                )
                rows.append(st)
                log(
                    f"     {name:<52} {r.steps:>4} {st.get('i0', -1):>3} "
                    f"{st.get('swing', float('nan')):>6.1f} {st.get('gpk', float('nan')):>+7.1f} "
                    f"{st.get('lf', float('nan')):>4.2f} {str(st.get('ret')):>5} "
                    f"{st.get('clo_ret', float('nan')):>+7.1f} {st.get('rng_ret', float('nan')):>7.0f} "
                    f"{st.get('clo_min', float('nan')):>+7.1f} {st.get('both_min', float('nan')):>+8.1f} "
                    f"{st.get('n_in_rng', -1):>4} {st['D2']:>3} {st['D2c']:>4}"
                )
    log("")
    have_ret = [s for s in rows if s.get("ret") is not None]
    log(f"     schedules with a gamma RETURN inside the window: {len(have_ret)}/{len(rows)}")
    if have_ret:
        log(
            f"     closure AT the return index over those: min {min(s['clo_ret'] for s in have_ret):+.1f}, "
            f"max {max(s['clo_ret'] for s in have_ret):+.1f} m/s"
        )
    bm = [s["both_min"] for s in rows if not np.isnan(s.get("both_min", float("nan")))]
    log(f"     bothMin over all {len(rows)}: min {min(bm):+.1f} m/s, max {max(bm):+.1f} m/s")
    nb = [
        s for s in rows if not np.isnan(s.get("both_min", float("nan"))) and s["both_min"] < 120.0
    ]
    log(
        f"     schedules with SOME window index satisfying closure<120 AND rng<1500: {len(nb)}/{len(rows)}"
    )
    for s in nb:
        log(f"       {s['name']:<54} bothMin {s['both_min']:+.1f} m/s")
    log(
        f"     D2 over all {len(rows)}: {sorted({s['D2'] for s in rows})}   D2c: {sorted({s['D2c'] for s in rows})}"
    )

    path = f"{E.OUT}/verify_d2_closure_floor2.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
