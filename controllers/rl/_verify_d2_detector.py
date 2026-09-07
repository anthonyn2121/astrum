"""Run the SHIPPED detector on the reconstructed traces and probe which D2 clause is load-bearing.

Also re-derives gamma from the raw ENU velocity columns as a cross-check on `smp_gamma_rad`.
"""

import csv
import math

import astrum_rl  # noqa: F401,E402  (puts the embedded env on sys.path)
import numpy as np
from astrum_dogfight import bfm_detect as B  # noqa: E402

BASE = "/workspace/controllers/rl/runs/d3b/"


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def trace(path):
    tr = []
    for r in rows(path):
        tr.append(
            B.Sample(
                t=float(r["smp_t"]),
                chi=float(r["smp_chi_rad"]),
                chi_valid=r["smp_chi_valid"].strip() in ("1", "True", "true"),
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
                lethal=r["smp_lethal"].strip() in ("1", "True", "true"),
            )
        )
    return tr


for name in ("replay_kill_seed62.csv", "replay_nokill_seed28.csv"):
    path = BASE + name
    rr = rows(path)
    tr = trace(path)
    print("=" * 100)
    print(f"{name}: {len(tr)} ticks")

    # cross-check gamma against the ENU velocity columns
    dg = 0.0
    for r, s in zip(rr, tr):
        v = np.array([float(r["vel_e_ms"]), float(r["vel_n_ms"]), float(r["vel_u_ms"])])
        g = math.asin(float(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
        dg = max(dg, abs(g - s.gamma))
    print(f"  max |gamma(smp) - asin(vel_u/|vel|)| = {dg:.3e} rad")

    d = B.detect_all(tr)
    print(
        f"  SHIPPED detect_all: D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])} "
        f"D2c={len(d['yoyos_clean'])} D3={len(d['turning_room'])} lethal={d['lethal_ticks']}"
    )

    # ---- clause isolation: relax exactly ONE constant at a time, everything else shipped
    orig = {
        k: getattr(B, k)
        for k in (
            "YOYO_GAMMA_SWING_DEG",
            "YOYO_GAMMA_RETURN_DEG",
            "YOYO_GAMMA_EXCURSION_DEG",
            "LOADED_FRAC",
            "CLOSURE_OVERSHOOT",
            "YOYO_RETURN_S",
        )
    }
    try:
        print("  ONE-AT-A-TIME clause relaxation (all other constants SHIPPED):")
        for k, vals in (
            ("YOYO_GAMMA_SWING_DEG", [30.0, 28.5, 28.47, 25.0, 15.0, 5.0, 0.0]),
            ("YOYO_GAMMA_RETURN_DEG", [10.0, 15.0, 19.0, 24.0, 30.0, 45.0, 180.0]),
            ("YOYO_GAMMA_EXCURSION_DEG", [15.0, 0.0]),
            ("LOADED_FRAC", [0.5, 0.0]),
            ("CLOSURE_OVERSHOOT", [120.0, -1e9]),
        ):
            for v in vals:
                for kk, vv in orig.items():
                    setattr(B, kk, vv)
                setattr(B, k, v)
                dd = B.detect_all(tr)
                print(
                    f"    {k:26s} = {v:>10.4g} -> D2={len(dd['yoyos'])} D2c={len(dd['yoyos_clean'])}"
                )
        # both shape clauses fully open, return bar SHIPPED -> isolates the return clause
        for kk, vv in orig.items():
            setattr(B, kk, vv)
        B.YOYO_GAMMA_SWING_DEG = 0.0
        B.YOYO_GAMMA_EXCURSION_DEG = 0.0
        B.LOADED_FRAC = 0.0
        dd = B.detect_all(tr)
        print(
            f"    ALL SHAPE CLAUSES OPEN, return bar shipped (10 deg / 6 s) -> "
            f"D2={len(dd['yoyos'])} D2c={len(dd['yoyos_clean'])}"
        )
    finally:
        for kk, vv in orig.items():
            setattr(B, kk, vv)
    print(
        f"  globals restored: SWING={B.YOYO_GAMMA_SWING_DEG} RET={B.YOYO_GAMMA_RETURN_DEG} "
        f"EXC={B.YOYO_GAMMA_EXCURSION_DEG} LF={B.LOADED_FRAC} CLO={B.CLOSURE_OVERSHOOT} "
        f"RET_S={B.YOYO_RETURN_S}"
    )

    # ---- does the flight path EVER come back to g0, with the peak pinned at the 6 s-window peak?
    t = [s.t for s in tr]
    gam = [s.gamma for s in tr]
    g0 = gam[0]
    win = [k for k in range(len(tr)) if t[k] - t[0] <= B.YOYO_RETURN_S]
    peak = max(win, key=lambda k: abs(gam[k] - g0))
    print(
        f"  peak (6 s window) step {peak} t {t[peak]:.3f} s gamma {math.degrees(gam[peak]):+.4f} deg"
    )
    firsts = [k for k in range(peak + 1, len(tr)) if abs(gam[k] - g0) < math.radians(10.0)]
    if firsts:
        k = firsts[0]
        print(
            f"  FIRST tick after the peak with |gamma-g0| < 10 deg: step {k} t {t[k]:.3f} s "
            f"gamma {math.degrees(gam[k]):+.4f} deg  |gamma-g0| {math.degrees(abs(gam[k] - g0)):.4f} deg "
            f"-> that is {t[k] - t[0]:.3f} s after entry (bar YOYO_RETURN_S={B.YOYO_RETURN_S} s)"
        )
    else:
        bg = min(range(peak + 1, len(tr)), key=lambda k: abs(gam[k] - g0))
        print(
            f"  NEVER within 10 deg of g0 after the peak; closest "
            f"{math.degrees(abs(gam[bg] - g0)):.4f} deg at step {bg} t {t[bg]:.3f} s"
        )
    print()
