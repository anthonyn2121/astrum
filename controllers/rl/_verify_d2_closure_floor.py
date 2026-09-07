"""REFUTATION probe for "D2 root cause B: `clean` is geometrically foreclosed".

The claim under test (from runs/d3b/d2_express_report.txt closing lines):
    "Closure FLOOR inside the [0, 6] s window over the same trajectories: min +125 m/s,
     max +375 m/s. `clean` requires closure < 120 m/s at the exit index, so as long as
     the floor stays above 120 m/s no exit inside the window can be clean --
     D2c = 0 is forced by the CELL geometry, not by the schedule."

Three things this script measures, none of them re-implemented:

  A  REPRODUCE. Re-fly the exact 16 schedules from `_diag_d2_express.main` and recompute
     the entry-edge INDEX (the report's "(all of them at step 0)" is a hard-coded string,
     not a measured value), cloMin over the window, and the exit-index closure/range at
     the `ret` index where one exists.

  B  THE UNTESTED CHANNEL. Every one of the 16 schedules is (dur, bank, load=1.0, thr=1.0):
     FULL THROTTLE, MAX PULL, always. Throttle is genome channel 3 (`Genome.act` p[2]) and
     the saved best genome itself commands thr 0.000 on segment 2. So re-fly the closure-
     bleeding bank shapes across thr in {0.0, 0.25, 0.5, 1.0} and load in {0.6, 1.0} and
     read the same closure floor. If the floor crosses 120 m/s the claim's mechanism is a
     SCHEDULE artefact, not cell geometry.

  C  THE CONJUNCTION. `bled` = closure < 120 AND rng < 1500 at the SAME index. Report
     whether any index in the window satisfies both, since the closure clause alone is not
     what `clean` tests.
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


def edges_of(tr) -> list[int]:
    return [
        i
        for i in range(len(tr))
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]


def win_stats(tr) -> dict:
    """Everything `clean` reads, inside YOYO_RETURN_S of the FIRST entry edge."""
    ed = edges_of(tr)
    out = dict(edges=ed, n=len(tr))
    if not ed:
        return out
    i = ed[0]
    win = [k for k in range(i, len(tr)) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    clo = np.array([tr[k].closure for k in win])
    rng = np.array([tr[k].rng for k in win])
    kmin = int(np.argmin(clo))
    both = [
        k for k, c, rr in zip(win, clo, rng) if c < B.CLOSURE_OVERSHOOT and rr < B.YOYO_EXIT_RNG_M
    ]
    out.update(
        i0=i,
        nwin=len(win),
        clo_min=float(clo.min()),
        clo_min_k=win[kmin],
        clo_min_t=tr[win[kmin]].t,
        rng_at_clo_min=float(rng[kmin]),
        rng_min=float(rng.min()),
        n_below_120=int((clo < B.CLOSURE_OVERSHOOT).sum()),
        n_bled_both=len(both),
        first_bled=both[0] if both else None,
        gmin=float(min(math.degrees(tr[k].gamma) for k in win)),
        gmax=float(max(math.degrees(tr[k].gamma) for k in win)),
        nzmax=float(max(tr[k].nz for k in win)),
    )
    return out


def d2_ret(tr) -> tuple[int | None, float, float]:
    """The `ret` index `_diag_d2_express.d2_stats` reports, and the closure/rng THERE."""
    ed = edges_of(tr)
    if not ed:
        return None, float("nan"), float("nan")
    i = ed[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, len(tr)) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    if len(win) < 3:
        return None, float("nan"), float("nan")
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    for k in range(peak + 1, len(tr)):
        if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            return k, float(tr[k].closure), float(tr[k].rng)
    return None, float("nan"), float("nan")


HDR = (
    f"     {'schedule':<46} {'steps':>5} {'edg':>3} {'i0':>3} {'cloMin':>7} {'t@min':>6} "
    f"{'rng@min':>7} {'rngMin':>7} {'n<120':>5} {'nBled':>5} {'ret':>5} {'cloRet':>7} "
    f"{'rngRet':>7} {'D2':>3} {'D2c':>4}"
)


def row(name: str, phases) -> dict:
    r, _ = E.fly(E.Sched(phases))
    st = win_stats(r.trace)
    det = B.detect_all(r.trace)
    ret, clo_ret, rng_ret = d2_ret(r.trace)
    log(
        f"     {name:<46} {r.steps:>5} {len(st['edges']):>3} {st.get('i0', -1):>3} "
        f"{st.get('clo_min', float('nan')):>+7.1f} {st.get('clo_min_t', float('nan')):>6.2f} "
        f"{st.get('rng_at_clo_min', float('nan')):>7.0f} {st.get('rng_min', float('nan')):>7.0f} "
        f"{st.get('n_below_120', -1):>5} {st.get('n_bled_both', -1):>5} "
        f"{str(ret):>5} {clo_ret:>+7.1f} {rng_ret:>7.0f} "
        f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4}"
    )
    st["name"] = name
    st["D2"] = len(det["yoyos"])
    st["D2c"] = len(det["yoyos_clean"])
    st["event"] = r.event
    return st


def main() -> None:
    log("#" * 118)
    log("#  VERIFY: is the [0,6] s closure floor > 120 m/s a CELL fact or a FULL-THROTTLE fact?")
    log("#" * 118)
    log(
        f"\n  constants read from bfm_detect: CLOSURE_OVERSHOOT {B.CLOSURE_OVERSHOOT:.0f} m/s, "
        f"YOYO_RETURN_S {B.YOYO_RETURN_S:.1f} s, YOYO_EXIT_RNG_M {B.YOYO_EXIT_RNG_M:.0f} m, "
        f"NZ_LOADED {B.NZ_LOADED:.1f}\n"
        f"  cell merge1500, seed {E.SEED}, horizon {E.HORIZON_S:.1f} s = {E.N_STEPS} steps "
        f"x {E.DT * 1e3:.3f} ms\n"
    )

    # ---------------------------------------------------------------- A: reproduce the 16
    log(
        "  A  THE 16 SCHEDULES OF `_diag_d2_express.main`, verbatim (all of them load 1.0 / thr 1.0)"
    )
    log(HDR)
    base: list[tuple[str, list]] = []
    for d_up in (1.0, 1.5, 2.0, 2.5, 3.0):
        base.append(
            (
                f"HIGH pull {d_up:.1f}s @bank0 then bank115 pull",
                [(d_up, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
            )
        )
    for d_dn in (2.0, 2.5, 3.0, 3.5, 4.0):
        base.append(
            (
                f"LOW  bank115 pull {d_dn:.1f}s then bank0 pull",
                [(d_dn, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)],
            )
        )
    base.append(
        (
            "SEG  4.0s bank0 pull then 4.0s bank115 pull",
            [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)],
        )
    )
    base.append(
        (
            "SEG  4.0s bank115 pull then 4.0s bank0 pull",
            [(4.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)],
        )
    )
    base.append(
        (
            "FINE 1.0 bank0 / 1.5 bank115 / rest bank0",
            [(1.0, 0.0, 1.0, 1.0), (1.5, 115.0, 1.0, 1.0), (21.5, 0.0, 1.0, 1.0)],
        )
    )
    base.append(
        (
            "FINE 0.5 bank0 / 1.0 bank115 / rest bank0",
            [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
        )
    )
    base.append(
        (
            "FINE 1.5 bank115 / 1.5 bank0 / rest bank115",
            [(1.5, 115.0, 1.0, 1.0), (1.5, 0.0, 1.0, 1.0), (21.0, 115.0, 1.0, 1.0)],
        )
    )
    base.append(
        (
            "FINE 2.5 bank115 / 2.0 bank0 / rest bank115",
            [(2.5, 115.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (19.5, 115.0, 1.0, 1.0)],
        )
    )
    stats_a = [row(n, p) for n, p in base]
    fl = [s["clo_min"] for s in stats_a]
    log(
        f"\n     REPRODUCED closure floor over the 16: min {min(fl):+.1f} m/s, max {max(fl):+.1f} m/s "
        f"(report says min +125, max +375)"
    )
    log(
        f"     entry-edge indices over the 16: {sorted({s['i0'] for s in stats_a})}  (report asserts 'all at step 0')"
    )
    log(f"     D2 over the 16: {[s['D2'] for s in stats_a]}")
    log(f"     D2c over the 16: {[s['D2c'] for s in stats_a]}")

    # ---------------------------------------------------------------- B: the throttle channel
    log("\n  B  SAME BANK SHAPES, THROTTLE AND LOAD SWEPT (both are genome channels; the best")
    log("     genome itself commands thr 0.000 on seg 2 and load 0.000 on seg 4)")
    log(HDR)
    stats_b: list[dict] = []
    shapes = [
        ("bank0 pull all the way", lambda ld, th: [(24.0, 0.0, ld, th)]),
        (
            "FINE 0.5 b0 / 1.0 b115 / rest b0",
            lambda ld, th: [(0.5, 0.0, ld, th), (1.0, 115.0, ld, th), (22.5, 0.0, ld, th)],
        ),
        ("SEG  4.0 b0 / then b115", lambda ld, th: [(4.0, 0.0, ld, th), (20.0, 115.0, ld, th)]),
        ("HIGH 3.0 b0 / then b115", lambda ld, th: [(3.0, 0.0, ld, th), (21.0, 115.0, ld, th)]),
        ("bank60 pull all the way", lambda ld, th: [(24.0, 60.0, ld, th)]),
    ]
    for shp, mk in shapes:
        for ld in (1.0, 0.6):
            for th in (1.0, 0.5, 0.25, 0.0):
                stats_b.append(row(f"{shp} [load {ld:.2f} thr {th:.2f}]", mk(ld, th)))
    flb = [s["clo_min"] for s in stats_b if "clo_min" in s]
    log(
        f"\n     closure floor over the {len(stats_b)} swept trajectories: min {min(flb):+.1f} m/s, max {max(flb):+.1f} m/s"
    )
    below = [s for s in stats_b if s.get("clo_min", 1e9) < B.CLOSURE_OVERSHOOT]
    log(
        f"     trajectories whose window closure floor drops BELOW {B.CLOSURE_OVERSHOOT:.0f} m/s: {len(below)}/{len(stats_b)}"
    )
    for s in below:
        log(
            f"       {s['name']:<48} cloMin {s['clo_min']:+8.1f} m/s at t {s['clo_min_t']:.2f} s "
            f"(step {s['clo_min_k']}), rng there {s['rng_at_clo_min']:.0f} m, "
            f"ticks<120 {s['n_below_120']}, ticks with closure<120 AND rng<1500: {s['n_bled_both']}, "
            f"nzMax {s['nzmax']:.2f} g, gamma [{s['gmin']:+.1f}, {s['gmax']:+.1f}] deg"
        )

    # ---------------------------------------------------------------- C: the conjunction
    log("\n  C  THE CONJUNCTION `bled` ACTUALLY TESTS (closure < 120 AND rng < 1500 at ONE index)")
    allst = stats_a + stats_b
    nb = [s for s in allst if s.get("n_bled_both", 0) > 0]
    log(
        f"     trajectories with >=1 index in the window satisfying BOTH clauses: {len(nb)}/{len(allst)}"
    )
    for s in nb:
        log(f"       {s['name']:<48} first bled index {s['first_bled']}  ticks {s['n_bled_both']}")
    log(
        "\n     NOTE the anti-correlation: the bank-0 max-pull shapes that bleed closure best also"
        "\n     open the range, so a low closure floor and rng < 1500 m are not co-located."
    )
    x = np.array([s["clo_min"] for s in allst if "clo_min" in s])
    y = np.array([s["rng_at_clo_min"] for s in allst if "clo_min" in s])
    log(
        f"     corr(cloMin, rng at cloMin) over {len(x)} trajectories = {np.corrcoef(x, y)[0, 1]:+.3f}"
    )

    path = f"{E.OUT}/verify_d2_closure_floor.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
