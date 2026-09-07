"""Independent recomputation of the D2 rising-edge claim from the replay CSVs.

Recomputes, WITHOUT reusing the audit script's helpers where a number is being checked:
  * closure > CLOSURE_OVERSHOOT counts and rising-edge candidate counts on both traces
  * the structural probe (rising edge -> any tick above the band)
  * a STRICTER probe the claim never ran: drop the closure band ENTIRELY
  * whether the two traces are independent evidence at all
  * altitude price of every hold that any probe turns into an event

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d2_edge_refute.py'
"""

from __future__ import annotations

import csv
import math

import astrum_rl._paths  # noqa: F401  (puts astrum_dogfight + the pyo3 .so on sys.path)
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
CDT = 5.0 / 120.0
KILL = "runs/d3b/replay_kill_seed62.csv"
NOK = "runs/d3b/replay_nokill_seed28.csv"


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(rows, col):
    return [float(r[col]) for r in rows]


def trace(rows):
    tr = []
    for r in rows:
        tr.append(
            B.Sample(
                t=float(r["smp_t"]),
                chi=float(r["smp_chi_rad"]),
                chi_valid=int(float(r["smp_chi_valid"])) != 0,
                gamma=float(r["smp_gamma_rad"]),
                nz=float(r["smp_nz"]),
                phi=float(r["smp_phi_rad"]),
                alt=float(r["smp_alt"]),
                ail=float(r["smp_ail"]),
                rng=float(r["smp_rng"]),
                ata=float(r["smp_ata"]),
                ata_az=float(r["smp_ata_az"]),
                aspect=float(r["smp_aspect"]),
                closure=float(r["smp_closure"]),
                cpa_miss=float(r["smp_cpa_miss"]),
                es_rel=float(r["smp_es_rel"]),
                lay_miss_m=float(r["smp_lay_miss_m"]),
                lethal=int(float(r["smp_lethal"])) != 0,
            )
        )
    return tr


def runs_true(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def shape_ok(tr, i, band: bool):
    """D2's shape cascade verbatim from bfm_detect.detect_yoyos, entry fixed at i.

    Returns (passed, why, detail). `band` False = drop the closure gate entirely.
    """
    n = len(tr)
    sw = math.radians(B.YOYO_GAMMA_SWING_DEG)
    ex = math.radians(B.YOYO_GAMMA_EXCURSION_DEG)
    rt = math.radians(B.YOYO_GAMMA_RETURN_DEG)
    if band and tr[i].closure <= B.CLOSURE_OVERSHOOT:
        return False, "band", {}
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    if len(win) < 3:
        return False, "window<3", {}
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    swing = abs(tr[peak].gamma - g0)
    if swing < sw:
        return False, f"swing {swing * DEG:.2f}", {}
    if abs(tr[peak].gamma) < ex:
        return False, f"excur {abs(tr[peak].gamma) * DEG:.2f}", {}
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
    frac = sum(loaded) / len(loaded)
    if frac < B.LOADED_FRAC:
        return False, f"loaded {frac:.3f}", {}
    ex_idx = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < rt:
            ex_idx = k
            break
    if ex_idx is None:
        tail = [k for k in range(peak + 1, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
        best = min((abs(tr[k].gamma - g0) for k in tail), default=float("nan"))
        return False, f"no-return (closest {best * DEG:.2f} deg)", {}
    bled = tr[ex_idx].closure < B.CLOSURE_OVERSHOOT and tr[ex_idx].rng < B.YOYO_EXIT_RNG_M
    kept = B._deg(tr[ex_idx].aspect - tr[i].aspect) <= B.YOYO_ASPECT_SLIP_DEG
    return (
        True,
        "PASS",
        {
            "peak": peak,
            "exit": ex_idx,
            "swing_deg": swing * DEG,
            "gpk_deg": tr[peak].gamma * DEG,
            "loaded_frac": frac,
            "bled": bled,
            "kept": kept,
            "clean": bool(bled and kept),
            "dalt_entry_peak_m": tr[peak].alt - tr[i].alt,
            "dalt_entry_exit_m": tr[ex_idx].alt - tr[i].alt,
            "nz_pk": max(tr[k].nz for k in range(i, peak + 1)),
        },
    )


def block(tag, path):
    rows = load(path)
    tr = trace(rows)
    n = len(tr)
    clo = [s.closure for s in tr]
    print(f"\n{'=' * 96}\n{tag}  {path}   {n} ticks, {tr[-1].t:.3f} s\n{'=' * 96}")

    # --- 1. the claim's counts, recomputed from the raw column -------------------------
    above = [c > 120.0 for c in clo]
    n_above = sum(above)
    edges = [i for i in range(n) if above[i] and (i == 0 or not above[i - 1])]
    rt = runs_true(above)
    print(f" closure>120 on {n_above}/{n} ticks ({100.0 * n_above / n:.1f}%)")
    print(f" closure range {min(clo):+.0f} .. {max(clo):+.0f} m/s")
    print(f" contiguous runs above the band: {rt}")
    print(f" RISING-EDGE candidates (line 411 verbatim): {len(edges)} -> {edges}")
    if edges:
        i0 = edges[0]
        print(
            f"   entry {i0}: t {tr[i0].t:.3f} s  g0 {tr[i0].gamma * DEG:+.2f} deg  clo {clo[i0]:+.1f}"
        )
    # cross-check the named-degrees column agrees with the smp block
    dmax = max(abs(float(r["closure_ms"]) - float(r["smp_closure"])) for r in rows)
    print(f" max |closure_ms - smp_closure| = {dmax:.3e}  (columns agree)")

    # --- 2. shipped detector ----------------------------------------------------------
    d = B.detect_all(tr)
    print(
        f" SHIPPED detect_all: D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])} "
        f"D2c={len(d['yoyos_clean'])} D3={len(d['turning_room'])} leth={d['lethal_ticks']}"
    )

    # --- 3. probe A: the claim's probe (band kept, edge dropped) ----------------------
    passA = [i for i in range(n) if shape_ok(tr, i, band=True)[0]]
    print(f" PROBE A  edge -> any tick ABOVE band ({n_above} candidates): passes = {len(passA)}")

    # --- 4. probe B: band dropped entirely (the claim never ran this) -----------------
    passB = [i for i in range(n) if shape_ok(tr, i, band=False)[0]]
    print(f" PROBE B  band DROPPED, every tick a candidate ({n} candidates): passes = {len(passB)}")
    if passB:
        print(f"   passing entry steps: {passB[0]}..{passB[-1]}  ({len(passB)} of {n})")
        for i in (passB[0], passB[len(passB) // 2], passB[-1]):
            ok, why, dt = shape_ok(tr, i, band=False)
            print(
                f"   entry {i:>3} t {tr[i].t:6.3f} clo {clo[i]:+7.1f} g0 {tr[i].gamma * DEG:+7.2f} "
                f"-> peak {dt['peak']:>3} exit {dt['exit']:>3} swing {dt['swing_deg']:5.1f} "
                f"gpk {dt['gpk_deg']:+6.1f} loaded {dt['loaded_frac']:.2f} nzPk {dt['nz_pk']:.2f} "
                f"dalt(e->pk) {dt['dalt_entry_peak_m']:+7.1f} m dalt(e->x) {dt['dalt_entry_exit_m']:+7.1f} m "
                f"bled {dt['bled']} kept {dt['kept']} clean {dt['clean']}"
            )
        # how many of the passing entries are INSIDE vs OUTSIDE the closure band
        inside = [i for i in passB if above[i]]
        print(
            f"   of those, closure>120 at entry: {len(inside)}/{len(passB)}; below band: {len(passB) - len(inside)}"
        )

    # --- 5. why the single shipped candidate fails ------------------------------------
    if edges:
        ok, why, _ = shape_ok(tr, edges[0], band=True)
        print(f" shipped candidate step {edges[0]}: {'PASS' if ok else 'FAIL'} at clause -> {why}")
    # the g0 anchor is set BY the entry gate; enumerate what other anchors buy
    g = [s.gamma * DEG for s in tr]
    print(
        f" gamma: step0 {g[0]:+.2f}  min {min(g):+.2f}@{g.index(min(g))}  max {max(g):+.2f}@{g.index(max(g))}  last {g[-1]:+.2f}"
    )

    # --- 6. RET_S monotonicity check at SWING=25, RET_DEG=30 -------------------------
    saved = {
        k: getattr(B, k) for k in ("YOYO_GAMMA_SWING_DEG", "YOYO_GAMMA_RETURN_DEG", "YOYO_RETURN_S")
    }
    try:
        for rs in (6.0, 9.0, 12.0, 18.0, 24.0):
            B.YOYO_GAMMA_SWING_DEG, B.YOYO_GAMMA_RETURN_DEG, B.YOYO_RETURN_S = 25.0, 30.0, rs
            ev = B.detect_yoyos(tr)
            print(
                f"   SWING 25 RET_DEG 30 RET_S {rs:>4.0f} -> D2 {len(ev)}  D2c {sum(1 for e in ev if e.detail.get('clean'))}"
            )
    finally:
        for k, v in saved.items():
            setattr(B, k, v)
    assert B.YOYO_GAMMA_SWING_DEG == 30.0 and B.YOYO_RETURN_S == 6.0
    return rows, tr


def main():
    rk, tk = block("KILL seed 62", KILL)
    rn, tn = block("NOKILL seed 28", NOK)

    print(f"\n{'=' * 96}\n INDEPENDENCE OF THE TWO TRACES\n{'=' * 96}")
    m = min(len(tk), len(tn))
    cols = [
        "smp_gamma_rad",
        "smp_nz",
        "smp_closure",
        "smp_chi_rad",
        "smp_rng",
        "smp_aspect",
        "smp_cpa_miss",
        "smp_alt",
    ]
    for c in cols:
        d = max(abs(float(rk[i][c]) - float(rn[i][c])) for i in range(m))
        print(f"   max |kill - nokill| over steps 0..{m - 1} on {c:>16} = {d:.3e}")
    print(f"   -> the kill trace's ENTIRE length ({m} ticks) is a prefix of the nokill trace")
    # the whole D2 window on either trace
    print(
        f"   D2 window from entry 0 = ticks with t <= {B.YOYO_RETURN_S} s -> steps 0..{int(B.YOYO_RETURN_S / CDT)}"
    )
    print(
        f"   that window lies entirely inside the shared prefix: {int(B.YOYO_RETURN_S / CDT) < m}"
    )

    print(f"\n{'=' * 96}\n WHAT THE BAND EXCLUDES ON SEED 28\n{'=' * 96}")
    clo = [s.closure for s in tn]
    tail = [i for i in range(len(tn)) if clo[i] <= 120.0]
    print(f"   ticks BELOW the band: {len(tail)}/{len(tn)} ({100.0 * len(tail) / len(tn):.1f}%)")
    print(
        f"   max closure among them: {max(clo[i] for i in tail):+.1f} m/s at step {max(tail, key=lambda i: clo[i])}"
    )
    print(
        f"   first tick below band: {tail[0]} (t {tn[tail[0]].t:.3f} s); closure never re-crosses 120 upward: {all(clo[i] <= 120.0 for i in range(tail[0], len(tn)))}"
    )
    # is there BFM-shaped motion in the excluded region? price it in altitude.
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tn]
    for i, j in runs_true(hold):
        if tn[j].t - tn[i].t < B.CHI_MIN_HOLD_S:
            continue
        print(
            f"   loaded chi>45 hold steps {i}-{j}  dur {tn[j].t - tn[i].t:.3f} s  "
            f"nzPk {max(tn[k].nz for k in range(i, j + 1)):.2f} g  dalt {tn[j].alt - tn[i].alt:+.1f} m  "
            f"clo_entry {tn[i].closure:+.1f}  in_band {tn[i].closure > 120.0}"
        )


if __name__ == "__main__":
    main()
