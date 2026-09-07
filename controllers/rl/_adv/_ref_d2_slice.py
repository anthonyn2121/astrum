"""Independent refutation attempt of the D2 slice-vs-yoyo claim. Recomputes every cited
number from BOTH the named CSV columns and the smp_* (detector-input) columns, then runs the
REAL bfm_detect.detect_yoyos on a rebuilt Sample trace."""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
from astrum_rl import _paths  # noqa: E402

_paths.ensure_on_path()

from astrum_dogfight import bfm_detect as B  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def trace(d):
    n = d["_n"]
    return [
        B.Sample(
            t=d["smp_t"][k],
            chi=d["smp_chi_rad"][k],
            chi_valid=bool(d["smp_chi_valid"][k]),
            gamma=d["smp_gamma_rad"][k],
            nz=d["smp_nz"][k],
            phi=d["smp_phi_rad"][k],
            alt=d["smp_alt"][k],
            ail=d["smp_ail"][k],
            rng=d["smp_rng"][k],
            ata=d["smp_ata"][k],
            ata_az=d["smp_ata_az"][k],
            aspect=d["smp_aspect"][k],
            closure=d["smp_closure"][k],
            cpa_miss=d["smp_cpa_miss"][k],
            es_rel=d["smp_es_rel"][k],
            lay_miss_m=d["smp_lay_miss_m"][k],
            lethal=bool(d["smp_lethal"][k]),
        )
        for k in range(n)
    ]


def col_consistency(d, label):
    print(f"\n--- {label}: named column vs smp_* (detector input) agreement")
    for named, smp, conv in (
        ("gamma_deg", "smp_gamma_rad", math.degrees),
        ("nz_g", "smp_nz", None),
        ("closure_ms", "smp_closure", None),
        ("chi_deg", "smp_chi_rad", math.degrees),
        ("phi_deg", "smp_phi_rad", math.degrees),
        ("alt_enu_m", "smp_alt", None),
        ("cpa_miss_m", "smp_cpa_miss", None),
    ):
        a = d[named]
        b = np.degrees(d[smp]) if conv else d[smp]
        print(f"    {named:12} vs {smp:14} max|diff| = {np.abs(a - b).max():.6e}")


def clauses(d, label):
    n = d["_n"]
    tr = trace(d)
    clo = np.array([s.closure for s in tr])
    gam = np.array([s.gamma for s in tr])
    nz = np.array([s.nz for s in tr])
    t = np.array([s.t for s in tr])
    alt = d["alt_m"]
    print(f"\n=== {label}: D2 clause reconstruction from the DETECTOR'S OWN INPUTS ({n} rows)")
    print(f"    closure[0] = {clo[0]:.4f} m/s   gamma[0] = {math.degrees(gam[0]):+.6f} deg")
    print(
        f"    closure: min {clo.min():.2f} max {clo.max():.2f}; "
        f"steps closure<=120: {int((clo <= 120.0).sum())}/{n}"
    )
    ent = [
        i
        for i in range(n)
        if clo[i] > B.CLOSURE_OVERSHOOT and (i == 0 or clo[i - 1] <= B.CLOSURE_OVERSHOOT)
    ]
    print(f"    rising-edge entries = {ent}")
    print(
        f"    gamma: min {math.degrees(gam.min()):+.3f} deg at step {int(gam.argmin())} "
        f"(t {t[int(gam.argmin())]:.3f} s), max {math.degrees(gam.max()):+.3f} at step "
        f"{int(gam.argmax())}, last {math.degrees(gam[-1]):+.3f} at t {t[-1]:.3f} s"
    )
    for i in ent:
        g0 = gam[i]
        w = [k for k in range(n) if k >= i and t[k] - t[i] <= B.YOYO_RETURN_S]
        if len(w) < 3:
            print(f"    entry {i}: window < 3 -> detector BREAKS out of the loop")
            continue
        peak = max(w, key=lambda k: abs(gam[k] - g0))
        swing = math.degrees(abs(gam[peak] - g0))
        excur = math.degrees(abs(gam[peak]))
        loaded = nz[i : peak + 1] > B.NZ_LOADED
        frac = loaded.sum() / len(loaded)
        # clause d, restricted to the detector's window
        retw = [
            k for k in w if k > peak and abs(gam[k] - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG)
        ]
        # clause d, unrestricted (does gamma EVER come back on this trace?)
        retall = [
            k
            for k in range(peak + 1, n)
            if abs(gam[k] - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG)
        ]
        print(
            f"\n    entry step {i:3d} t {t[i]:6.3f}s g0 {math.degrees(g0):+8.4f} deg  "
            f"window {w[0]}-{w[-1]} (t<= {t[i] + B.YOYO_RETURN_S:.3f}s)"
        )
        print(
            f"      peak step {peak:3d} t {t[peak]:6.3f}s gamma {math.degrees(gam[peak]):+7.3f} "
            f"deg alt_m {alt[peak]:.2f} alt_enu {d['alt_enu_m'][peak]:.2f}"
        )
        print(
            f"      (a) swing     {swing:7.4f} vs {B.YOYO_GAMMA_SWING_DEG:.1f} -> "
            f"{'PASS' if swing >= B.YOYO_GAMMA_SWING_DEG else 'FAIL'} "
            f"(shortfall {B.YOYO_GAMMA_SWING_DEG - swing:+.4f} deg)"
        )
        print(
            f"      (b) excursion {excur:7.4f} vs {B.YOYO_GAMMA_EXCURSION_DEG:.1f} -> "
            f"{'PASS' if excur >= B.YOYO_GAMMA_EXCURSION_DEG else 'FAIL'}"
        )
        print(
            f"      (c) loadfrac  {frac:7.4f} ({int(loaded.sum())}/{len(loaded)}) vs "
            f"{B.LOADED_FRAC} -> {'PASS' if frac >= B.LOADED_FRAC else 'FAIL'}"
        )
        print(
            f"      (d) return in window -> "
            f"{('PASS at step ' + str(retw[0])) if retw else 'FAIL'}; "
            f"gamma at window end (step {w[-1]}, t {t[w[-1]]:.3f}s) = "
            f"{math.degrees(gam[w[-1]]):+7.3f} deg"
        )
        print(
            "      (d) return ANYWHERE later on this trace -> "
            + (
                f"YES first at step {retall[0]} t {t[retall[0]]:.3f}s "
                f"gamma {math.degrees(gam[retall[0]]):+.3f} deg "
                f"(dt from entry {t[retall[0]] - t[i]:.3f}s)"
                if retall
                else "NO, never"
            )
        )
        # altitude price of the descent, entry->peak
        print(
            f"      PRICE entry->peak: alt_m {alt[i]:.1f} -> {alt[peak]:.1f} "
            f"({alt[peak] - alt[i]:+.1f} m), rng {d['rng_m'][i]:.0f} -> {d['rng_m'][peak]:.0f} m, "
            f"aspect {d['aspect_deg'][i]:.1f} -> {d['aspect_deg'][peak]:.1f} deg"
        )
        # what swing bar would flip (a)?
        print(
            f"      SENSITIVITY: (a) would PASS for any YOYO_GAMMA_SWING_DEG <= {swing:.4f}; "
            f"minimum YOYO_RETURN_S that admits (d) = "
            + (f"{t[retall[0]] - t[i]:.3f} s" if retall else "unreachable on this trace")
        )


def detector(d, label):
    tr = trace(d)
    r = B.detect_all(tr)
    print(f"\n=== {label}: REAL B.detect_all")
    print(
        f"    ticks {r['ticks']} dur {r['duration_s']:.3f}s  D1={len(r['displacement_rolls'])} "
        f"D2={len(r['yoyos'])} D2c={len(r['yoyos_clean'])} D3={len(r['turning_room'])} "
        f"lethalTk={r['lethal_ticks']}"
    )
    for e in r["yoyos"]:
        print(f"      YOYO {e.kind} i0={e.i0} i1={e.i1} detail={e.detail}")
    c = r["commitment"]
    print(
        f"    D4 bank_hold {c['longest_bank_hold_s']:.3f}s dalt {c['bank_hold_dalt_m']:+.1f}m | "
        f"chi_hold {c['longest_chi_hold_s']:.3f}s dalt {c['chi_hold_dalt_m']:+.1f}m"
    )


for path, lab in ((KILL, "KILL seed62"), (NOKILL, "NOKILL seed28")):
    d = load(path)
    col_consistency(d, lab)
    clauses(d, lab)
    detector(d, lab)

# gamma trajectory shape past the kill truncation: seed 28 is the SAME commanded stream
print("\n\n=== 'never came back': gamma on the two traces, step-matched")
k, nk = load(KILL), load(NOKILL)
print(
    f"    max|gamma_deg diff| over the 0-{k['_n'] - 1} overlap = "
    f"{np.abs(k['gamma_deg'] - nk['gamma_deg'][: k['_n']]).max():.4e}"
)
print("    step   t_s   gamma_kill  gamma_nokill   alt_nokill   nz_nokill  clo_nokill  seg")
for s in list(range(0, 576, 12)):
    gk = f"{k['gamma_deg'][s]:+8.2f}" if s < k["_n"] else "     --- "
    print(
        f"    {s:4d} {nk['t_s'][s]:6.3f} {gk}   {nk['gamma_deg'][s]:+8.2f}    "
        f"{nk['alt_m'][s]:8.1f}   {nk['nz_g'][s]:7.2f}  {nk['closure_ms'][s]:8.1f}   "
        f"{int(nk['seg_idx'][s])}"
    )
g = nk["gamma_deg"]
up = [s for s in range(1, 576) if g[s] > -10.0]
print(
    "    seed28: first step with gamma > -10 deg after the dive = "
    + (f"{up[0]} (t {nk['t_s'][up[0]]:.3f}s, gamma {g[up[0]]:+.2f})" if up else "none")
)
print(
    f"    seed28: gamma min {g.min():+.2f} at step {int(g.argmin())}, "
    f"max after step 96 {g[96:].max():+.2f} at step {96 + int(g[96:].argmax())} "
    f"(t {nk['t_s'][96 + int(g[96:].argmax())]:.3f}s)"
)
print(
    f"    seed28 alt_m {nk['alt_m'][0]:.1f} -> {nk['alt_m'][-1]:.1f} "
    f"({nk['alt_m'][-1] - nk['alt_m'][0]:+.1f} m), min {nk['alt_m'].min():.1f} at step "
    f"{int(nk['alt_m'].argmin())}"
)
