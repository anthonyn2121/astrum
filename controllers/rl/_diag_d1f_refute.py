"""Independent recomputation of the D1 clause-(f) satisfiability claim.

Reads ONLY the dumped per-step CSVs plus the shipped detector module. No re-implementation of
the detector's clause logic: D1 is exercised by calling B.detect_displacement_rolls on a trace
rebuilt from the `smp_*` columns, with CHI_MIN_HOLD_S monkeypatched when a waiver is wanted.
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/crates/astrum-env/python")
from astrum_dogfight import bfm_detect as B  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"


def _tobool(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0", "yes")


def trace_from_csv(path: str) -> tuple[dict[str, np.ndarray], list[B.Sample]]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols: dict[str, np.ndarray] = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) for r in rows], dtype=float)
        except ValueError:
            cols[k] = np.array([_tobool(r[k]) for r in rows], dtype=bool)
    tr = [
        B.Sample(
            t=float(r["smp_t"]),
            chi=float(r["smp_chi_rad"]),
            chi_valid=_tobool(r["smp_chi_valid"]),
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
            lethal=_tobool(r["smp_lethal"]),
        )
        for r in rows
    ]
    return cols, tr


def hdr(s: str) -> None:
    print("\n" + "=" * 96)
    print("  " + s)
    print("=" * 96)


def all_pairs_min_daspect(asp_deg: np.ndarray, hi_excl: int | None = None):
    """min over i<=j of asp[j]-asp[i]; hi_excl = drop the last `hi_excl` ticks entirely."""
    a = asp_deg if hi_excl in (None, 0) else asp_deg[: len(asp_deg) - hi_excl]
    # running max of prefix gives, for each j, the largest asp[i] with i<=j
    run_max = np.maximum.accumulate(a)
    arg_run_max = np.zeros(len(a), dtype=int)
    cur = 0
    for k in range(len(a)):
        if a[k] > a[cur]:
            cur = k
        arg_run_max[k] = cur
    d = a - run_max
    j = int(np.argmin(d))
    return float(d[j]), int(arg_run_max[j]), j


def window_min_daspect(asp_deg: np.ndarray, dt: float, lo_s: float, hi_s: float):
    """min asp[j]-asp[i] over pairs whose span is in [lo_s, hi_s] seconds."""
    n = len(asp_deg)
    lo = max(1, int(round(lo_s / dt)))
    hi = min(n - 1, int(round(hi_s / dt)))
    best = (0.0, -1, -1)
    for i in range(n):
        j0, j1 = i + lo, min(i + hi, n - 1)
        if j0 > j1:
            continue
        seg = asp_deg[j0 : j1 + 1]
        k = int(np.argmin(seg))
        d = float(seg[k] - asp_deg[i])
        if d < best[0]:
            best = (d, i, j0 + k)
    return best


for name, path in (("KILL seed 62", KILL), ("NOKILL seed 28", NOKILL)):
    cols, tr = trace_from_csv(path)
    asp = cols["aspect_deg"]
    asp_smp = np.degrees(cols["smp_aspect"])
    chi = cols["chi_deg"]
    nz = cols["nz_g"]
    rng = cols["rng_m"]
    t = cols["t_s"]
    dt = float(np.median(np.diff(t)))

    hdr(f"{name}   n={len(t)} ticks   dt={dt * 1000:.3f} ms   file={path}")
    print(f"  aspect_deg vs degrees(smp_aspect): max abs diff = {np.abs(asp - asp_smp).max():.3e}")
    print(
        f"  aspect: step0 {asp[0]:.1f}  min {asp.min():.1f} @step {int(np.argmin(asp))} "
        f"(t {t[int(np.argmin(asp))]:.3f})  max {asp.max():.1f} @step {int(np.argmax(asp))} "
        f"(t {t[int(np.argmax(asp))]:.3f})  last {asp[-1]:.1f}"
    )

    d, i, j = all_pairs_min_daspect(asp)
    print(
        f"  ALL (i<=j) pairs: most negative d_aspect = {d:.1f} deg, steps {i}->{j} (rng {rng[i]:.0f}->{rng[j]:.0f} m)"
    )
    for k in (1, 2, 3, 5):
        d2, i2, j2 = all_pairs_min_daspect(asp, hi_excl=k)
        print(
            f"    excluding the last {k} tick(s): {d2:.1f} deg, steps {i2}->{j2} (rng {rng[i2]:.0f}->{rng[j2]:.0f} m)"
        )

    # windows of a length a real D1 event could have
    for lo_s, hi_s in ((0.5, 5.0), (0.5, 8.5), (0.125, 5.0)):
        d3, i3, j3 = window_min_daspect(asp, dt, lo_s, hi_s)
        print(
            f"  windows spanning {lo_s}-{hi_s} s: most negative d_aspect = {d3:.1f} deg, "
            f"steps {i3}->{j3} (t {t[i3]:.3f}->{t[j3]:.3f}, asp {asp[i3]:.1f}->{asp[j3]:.1f}, "
            f"rng {rng[i3]:.0f}->{rng[j3]:.0f} m)"
        )

    # the D1 hold mask and its runs, verbatim thresholds
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    runs = B._runs(hold)
    print(f"  D1 hold-mask runs (|chi|>{B.CHI_OUT_DEG} & nz>{B.NZ_LOADED}): {runs}")

    # D1 with the duration clause waived, verbatim otherwise, and its d_aspect per event
    saved_hold_s, saved_gain = B.CHI_MIN_HOLD_S, B.ASPECT_GAIN_DEG
    try:
        print(f"  D1 as shipped: {len(B.detect_displacement_rolls(tr))} events")
        B.CHI_MIN_HOLD_S = 0.0
        print(
            f"  D1 with CHI_MIN_HOLD_S=0.0, ASPECT_GAIN_DEG={saved_gain}: {len(B.detect_displacement_rolls(tr))}"
        )
        for gain in (15.0, 10.0, 5.0, 0.0, -1e9):
            B.ASPECT_GAIN_DEG = gain
            evs = B.detect_displacement_rolls(tr)
            tag = "no-op (f)" if gain < 0 else f"needs d_aspect <= {-gain:+.1f}"
            print(f"    ASPECT_GAIN_DEG={gain:>10} ({tag}) -> D1 = {len(evs)}")
            if gain < 0:
                for e in evs:
                    print(
                        f"       event i0={e.i0} i1={e.i1} t {e.t0:.3f}-{e.t1:.3f}  "
                        f"d_aspect {e.d_aspect_deg:+.1f} deg  d_rng {e.d_rng_m:+.0f} m  "
                        f"hold {e.detail.get('hold_s', float('nan')):.3f} s  "
                        f"chiPk {e.detail.get('chi_peak_deg', float('nan')):.1f}  "
                        f"nzPk {e.detail.get('nz_peak', float('nan')):.2f}"
                    )
    finally:
        B.CHI_MIN_HOLD_S, B.ASPECT_GAIN_DEG = saved_hold_s, saved_gain
    print(
        f"  globals restored: CHI_MIN_HOLD_S={B.CHI_MIN_HOLD_S} ASPECT_GAIN_DEG={B.ASPECT_GAIN_DEG}"
    )

    # how often does aspect fall at all, and by how much, on a 1 s / 2 s / 4 s lag
    for lag_s in (1.0, 2.0, 4.0):
        L = int(round(lag_s / dt))
        if L >= len(asp):
            continue
        dd = asp[L:] - asp[:-L]
        print(
            f"  lag {lag_s:.0f} s (L={L}): d_aspect min {dd.min():+.1f} max {dd.max():+.1f}  "
            f"frac <= -15 deg: {float((dd <= -15.0).mean()):.4f} ({int((dd <= -15.0).sum())}/{len(dd)})"
        )
