"""Probe 3: quantify the shortfall, then try the one corner the grid never tried.

H  MEASURED SHORTFALL, not "mutually exclusive". For each candidate, find
      t30  = first time |gamma - gamma0| >= YOYO_GAMMA_SWING_DEG (30), and
      tret = first time AFTER t30 that |gamma - gamma0| < YOYO_GAMMA_RETURN_DEG (10),
   searched over the WHOLE 12 s rollout, i.e. NOT truncated at YOYO_RETURN_S.
   tret - 6.0 s is the measured margin by which this parameterisation misses D2.

I  THE GRADED ROLL-BACK. Every schedule tried so far reverses in one 115 deg step, and
   elev = -clip(load*max(cos(bank_err),0),0,1) is EXACTLY 0 while |bank_err| >= 90 deg, so a
   one-step reversal throws away ~0.5-0.6 s of pull. A reversal broken into sub-90-deg steps
   (115 -> 60 -> 0) never trips the gate and keeps a partial pull the whole way. If a yo-yo is
   expressible at all, this is where it lives. Sweep it, on both branches.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def shortfall(tr) -> dict:
    g0 = tr[0].gamma
    sw = math.radians(B.YOYO_GAMMA_SWING_DEG)
    rt = math.radians(B.YOYO_GAMMA_RETURN_DEG)
    i30 = next((k for k in range(len(tr)) if abs(tr[k].gamma - g0) >= sw), None)
    if i30 is None:
        return dict(t30=None, tret=None, gmx=math.degrees(max(abs(s.gamma - g0) for s in tr)))
    iret = next((k for k in range(i30 + 1, len(tr)) if abs(tr[k].gamma - g0) < rt), None)
    return dict(
        t30=tr[i30].t,
        tret=(tr[iret].t if iret is not None else None),
        gmx=math.degrees(max(abs(s.gamma - g0) for s in tr)),
    )


def d2_pairs(tr) -> tuple:
    """Real detector output plus the superset (any peak, any return) inside the window."""
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    det = B.detect_all(tr)
    if not edges:
        return det, None, 0
    i = edges[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    pairs = []
    for p in win:
        s = math.degrees(abs(tr[p].gamma - g0))
        if (
            s < B.YOYO_GAMMA_SWING_DEG
            or abs(math.degrees(tr[p].gamma)) < B.YOYO_GAMMA_EXCURSION_DEG
        ):
            continue
        for k in range(p + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                pairs.append((p, k, s))
                break
    best = max(pairs, key=lambda z: z[2]) if pairs else None
    return det, best, len(pairs)


def main() -> None:
    N12 = int(round(12.0 / E.DT))
    log("#" * 118)
    log("#  PROBE 3 -- the measured shortfall, and the graded roll-back the grid never tried")
    log("#" * 118)

    log("\n" + "=" * 118)
    log("  H  MEASURED SHORTFALL (searched over 12 s, NOT truncated at the 6 s D2 window)")
    log("=" * 118)
    log(
        f"     t30 = first |dgamma| >= {B.YOYO_GAMMA_SWING_DEG:.0f} deg; tret = first return to within "
        f"{B.YOYO_GAMMA_RETURN_DEG:.0f} deg AFTER t30.\n"
        f"     D2 needs BOTH inside {B.YOYO_RETURN_S:.1f} s of t=0. margin = {B.YOYO_RETURN_S:.1f} - tret."
    )
    cands = [
        ("HIGH 2.8s bank0 -> bank115", [(2.8, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)]),
        ("HIGH 3.0s bank0 -> bank115", [(3.0, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)]),
        ("SEG 4.0s bank0 -> bank115", [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)]),
        ("LOW 4.2s bank115 -> bank0", [(4.2, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
        ("LOW 4.4s bank115 -> bank0", [(4.4, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
        ("LOW 4.6s bank115 -> bank0", [(4.6, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
        ("LOW 4.8s bank115 -> bank0", [(4.8, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
    ]
    log(f"     {'schedule':<44} {'t30':>6} {'tret':>6} {'margin':>7} {'|dg|max':>8} {'D2':>3}")
    for nm, ph in cands:
        r, rows = E.fly(E.Sched(ph), n_steps=N12)
        sf = shortfall(r.trace)
        det, best, npr = d2_pairs(r.trace)
        m = (B.YOYO_RETURN_S - sf["tret"]) if sf["tret"] is not None else float("nan")
        s30 = "None" if sf["t30"] is None else f"{sf['t30']:.2f}"
        sret = "None" if sf["tret"] is None else f"{sf['tret']:.2f}"
        log(f"     {nm:<44} {s30:>6} {sret:>6} {m:>+7.2f} {sf['gmx']:>8.1f} {len(det['yoyos']):>3}")

    log("\n" + "=" * 118)
    log(
        "  I  GRADED ROLL-BACK -- reversal split into sub-90-deg steps so the roll gate never fires"
    )
    log("=" * 118)
    log(
        f"     {'schedule':<58} {'steps':>5} {'swing':>6} {'ret':>5} {'best':>18} {'#pr':>3} "
        f"{'t30':>5} {'tret':>5} {'margin':>7} {'D2':>3} {'D2c':>4} {'dAlt6s':>7}"
    )
    trials = []
    # LOW branch: descend at 115, then step back 115 -> 60 -> 0 (and 115 -> 70 -> 30 -> 0)
    for d1 in (3.8, 4.0, 4.2, 4.4):
        for d2 in (0.4, 0.8, 1.2):
            trials.append(
                (
                    f"LOWg {d1:.1f} b115 / {d2:.1f} b60 / b0",
                    [(d1, 115.0, 1.0, 1.0), (d2, 60.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)],
                )
            )
    for d1 in (4.0, 4.2, 4.4):
        trials.append(
            (
                f"LOWg3 {d1:.1f} b115 / .3 b70 / .3 b30 / b0",
                [
                    (d1, 115.0, 1.0, 1.0),
                    (0.3, 70.0, 1.0, 1.0),
                    (0.3, 30.0, 1.0, 1.0),
                    (24.0, 0.0, 1.0, 1.0),
                ],
            )
        )
    # HIGH branch: climb at 0, then step 0 -> 60 -> 115
    for d1 in (2.4, 2.8, 3.2):
        for d2 in (0.4, 0.8, 1.2):
            trials.append(
                (
                    f"HIGHg {d1:.1f} b0 / {d2:.1f} b60 / b115",
                    [(d1, 0.0, 1.0, 1.0), (d2, 60.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
                )
            )
    # and the same two with idle throttle on the return, which raises gamma-rate via 1/V
    for d1 in (4.0, 4.2, 4.4):
        trials.append(
            (
                f"LOWg-idle {d1:.1f} b115 / .8 b60 / b0",
                [(d1, 115.0, 1.0, 0.0), (0.8, 60.0, 1.0, 0.0), (24.0, 0.0, 1.0, 0.0)],
            )
        )
    hits = []
    for nm, ph in trials:
        r, rows = E.fly(E.Sched(ph), n_steps=N12)
        tr = r.trace
        det, best, npr = d2_pairs(tr)
        sf = shortfall(tr)
        g0 = tr[0].gamma
        win = [k for k in range(len(tr)) if tr[k].t - tr[0].t <= B.YOYO_RETURN_S]
        peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
        ret = None
        for k in range(peak + 1, len(tr)):
            if tr[k].t - tr[0].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                ret = k
                break
        m = (B.YOYO_RETURN_S - sf["tret"]) if sf["tret"] is not None else float("nan")
        kw = min(len(rows) - 1, int(round(B.YOYO_RETURN_S / E.DT)))
        s30 = "None" if sf["t30"] is None else f"{sf['t30']:.2f}"
        sret = "None" if sf["tret"] is None else f"{sf['tret']:.2f}"
        log(
            f"     {nm:<58} {r.steps:>5} {math.degrees(abs(tr[peak].gamma - g0)):>6.1f} "
            f"{str(ret):>5} {str(best):>18} {npr:>3} "
            f"{s30:>5} {sret:>5} {m:>+7.2f} "
            f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4} "
            f"{rows[kw]['alt'] - rows[0]['alt']:>7.0f}"
        )
        if best or det["yoyos"]:
            hits.append((nm, best, det))

    log("\n" + "=" * 118)
    log("  SUMMARY")
    log("=" * 118)
    if not hits:
        log(
            "     NONE of the graded-roll-back schedules produced a qualifying (swing, return) pair or D2 > 0."
        )
    for nm, best, det in hits:
        log(f"     {nm:<58} best={best} D2={len(det['yoyos'])} D2c={len(det['yoyos_clean'])}")
        for ev in det["yoyos"]:
            log(f"        {ev.kind} t {ev.t0:.3f}-{ev.t1:.3f} {ev.detail}")

    path = f"{E.OUT}/refute_d2_express3.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
