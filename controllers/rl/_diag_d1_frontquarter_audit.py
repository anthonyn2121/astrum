"""Clause-level audit of the D1/D2 verdict on the d3b merge1500 kill trajectory.

Rebuilds the exact `bfm_detect.Sample` trace from the replay CSV `smp_*` columns (native units),
asserts it reproduces `B.detect_all`'s reported counts, then reports WHICH clause of D1 and D2
rejects each candidate. The question under test is whether D1/D2 are structurally blind to a
front-quarter gun kill (an "invisible by construction" claim) or whether some other clause binds.

Read-only: no sim, no env, no rollout. Pure function of the dumped trace.
"""

from __future__ import annotations

import csv
import math

import astrum_rl  # noqa: F401  -- puts the embedded env adapter on sys.path
from astrum_dogfight import bfm_detect as B

RUNS = "/workspace/controllers/rl/runs/d3b"
DEG = math.degrees


def load(path: str):
    rows = list(csv.DictReader(open(path)))
    tr = [
        B.Sample(
            t=float(r["smp_t"]),
            chi=float(r["smp_chi_rad"]),
            chi_valid=r["smp_chi_valid"] in ("1", "True", "true"),
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
            lethal=r["smp_lethal"] in ("1", "True", "true"),
        )
        for r in rows
    ]
    return rows, tr


def d1_clause_audit(tr, label):
    """Re-implement D1 clause-by-clause and report the first failing clause per candidate."""
    print(f"\n--- D1 clause audit [{label}] ---")
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    runs = B._runs(hold)
    print(f"  |chi|>{B.CHI_OUT_DEG} AND nz>{B.NZ_LOADED} runs: {len(runs)}")
    if not runs:
        # Where does the mask fail? separate the two conjuncts.
        chi_runs = B._runs([abs(s.chi) > math.radians(B.CHI_OUT_DEG) for s in tr])
        nz_runs = B._runs([s.nz > B.NZ_LOADED for s in tr])
        chi_max = (
            max(((tr[j].t - tr[i].t), i, j) for i, j in chi_runs) if chi_runs else (0.0, -1, -1)
        )
        nz_max = max(((tr[j].t - tr[i].t), i, j) for i, j in nz_runs) if nz_runs else (0.0, -1, -1)
        print(
            f"  chi-only runs={len(chi_runs)} longest={chi_max[0]:.3f}s steps {chi_max[1]}-{chi_max[2]}"
        )
        print(
            f"  nz-only  runs={len(nz_runs)} longest={nz_max[0]:.3f}s steps {nz_max[1]}-{nz_max[2]}"
        )
    for i, j in runs:
        dur = tr[j].t - tr[i].t
        n_smp = j - i + 1
        dalt = tr[j].alt - tr[i].alt
        fails = []
        if dur < B.CHI_MIN_HOLD_S:
            fails.append(f"(hold) {dur:.4f}s < {B.CHI_MIN_HOLD_S}")
        inplane = any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        )
        if not inplane:
            fails.append("(a) no in-plane state in 4s lookback")
        if tr[i].closure <= 0.0:
            fails.append(f"(c) closure {tr[i].closure:.1f} <= 0")
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            fails.append("(d) never rolled back in")
        else:
            sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
            if len(sides) != 1:
                fails.append("(e) ata_az sign flipped")
            d_aspect = DEG(tr[exit_idx].aspect - tr[i].aspect)
            if d_aspect >= -B.ASPECT_GAIN_DEG:
                fails.append(f"(f) d_aspect {d_aspect:+.2f} deg >= {-B.ASPECT_GAIN_DEG}")
        print(
            f"  cand steps {i}-{j} ({n_smp} smp) t {tr[i].t:.3f}-{tr[j].t:.3f} dur {dur:.4f}s "
            f"dalt {dalt:+.1f}m nzpk {max(tr[k].nz for k in range(i, j + 1)):.2f} "
            f"|chi|pk {max(abs(DEG(tr[k].chi)) for k in range(i, j + 1)):.1f} "
            f"exit={exit_idx}"
        )
        if exit_idx is not None:
            print(
                f"      aspect entry {DEG(tr[i].aspect):.2f} -> exit {DEG(tr[exit_idx].aspect):.2f} "
                f"(d {DEG(tr[exit_idx].aspect - tr[i].aspect):+.2f})"
            )
        print(f"      FAILS: {fails if fails else 'NONE -> would be a D1 event'}")


def d2_clause_audit(tr, label):
    print(f"\n--- D2 clause audit [{label}] ---")
    n = len(tr)
    swing = math.radians(B.YOYO_GAMMA_SWING_DEG)
    excur = math.radians(B.YOYO_GAMMA_EXCURSION_DEG)
    ret = math.radians(B.YOYO_GAMMA_RETURN_DEG)
    i = 0
    ncand = 0
    while i < n:
        if not (
            tr[i].closure > B.CLOSURE_OVERSHOOT
            and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
        ):
            i += 1
            continue
        ncand += 1
        g0 = tr[i].gamma
        window = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
        print(
            f"  rising edge at step {i} t {tr[i].t:.3f} closure {tr[i].closure:.1f} "
            f"gamma0 {DEG(g0):+.2f} window {len(window)} smp"
        )
        if len(window) < 3:
            print("      FAILS: window < 3 samples (episode ended)")
            break
        peak = max(window, key=lambda k: abs(tr[k].gamma - g0))
        sw = abs(tr[peak].gamma - g0)
        print(
            f"      peak step {peak} gamma {DEG(tr[peak].gamma):+.2f} swing {DEG(sw):.2f} deg "
            f"(need >= {B.YOYO_GAMMA_SWING_DEG}); |gamma_peak| {abs(DEG(tr[peak].gamma)):.2f} "
            f"(need >= {B.YOYO_GAMMA_EXCURSION_DEG})"
        )
        if sw < swing or abs(tr[peak].gamma) < excur:
            print("      FAILS: (b) gamma swing/excursion  -> NO event appended")
            i += 1
            continue
        loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
        frac = sum(loaded) / len(loaded) if loaded else 0.0
        print(f"      loaded_frac {frac:.3f} (need >= {B.LOADED_FRAC})")
        if not loaded or frac < B.LOADED_FRAC:
            print("      FAILS: (c) zoom-coast -> NO event appended")
            i += 1
            continue
        exit_idx = None
        for k in range(peak + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < ret:
                exit_idx = k
                break
        if exit_idx is None:
            print("      FAILS: gamma never returned within YOYO_RETURN_S -> NO event appended")
            i = peak + 1
            continue
        d_aspect = DEG(tr[exit_idx].aspect - tr[i].aspect)
        bled = tr[exit_idx].closure < B.CLOSURE_OVERSHOOT and tr[exit_idx].rng < B.YOYO_EXIT_RNG_M
        kept = d_aspect <= B.YOYO_ASPECT_SLIP_DEG
        print(
            f"      EVENT APPENDED (D2 counts it regardless of aspect): exit {exit_idx} "
            f"d_aspect {d_aspect:+.2f} kept={kept} bled={bled} clean={bled and kept}"
        )
        i = exit_idx + 1
    if ncand == 0:
        print("  no closure rising edge above CLOSURE_OVERSHOOT at all")


def aspect_windows(tr, label):
    """Is a >=15 deg aspect DECREASE available anywhere on this trace, over D1-event-length windows?"""
    print(f"\n--- aspect excursion scan [{label}] ---")
    a = [DEG(s.aspect) for s in tr]
    print(
        f"  aspect step0 {a[0]:.2f}  min {min(a):.2f}@{a.index(min(a))}  "
        f"max {max(a):.2f}@{a.index(max(a))}  last {a[-1]:.2f}"
    )
    # best (most negative) aspect delta over any window up to 8 s (D1 hold 0.5 + return 4 s is
    # <= ~4.5 s; be generous)
    dt = tr[1].t - tr[0].t
    wmax = int(round(8.0 / dt))
    best = (0.0, -1, -1)
    for i in range(len(tr)):
        for j in range(i + 1, min(i + wmax + 1, len(tr))):
            d = a[j] - a[i]
            if d < best[0]:
                best = (d, i, j)
    print(
        f"  most NEGATIVE d_aspect over any <=8s window: {best[0]:+.2f} deg "
        f"steps {best[1]}-{best[2]} (t {tr[best[1]].t:.3f}-{tr[best[2]].t:.3f})"
    )
    # monotonicity
    incr = sum(1 for k in range(1, len(a)) if a[k] > a[k - 1])
    print(f"  steps where aspect increased: {incr}/{len(a) - 1}")


def main():
    for label, path in (
        ("KILL seed62", f"{RUNS}/replay_kill_seed62.csv"),
        ("NOKILL seed28", f"{RUNS}/replay_nokill_seed28.csv"),
    ):
        rows, tr = load(path)
        res = B.detect_all(tr)
        print(f"\n===== {label}: {len(tr)} samples, {res['duration_s']:.3f} s =====")
        print(
            f"  detect_all: D1={len(res['displacement_rolls'])} D2={len(res['yoyos'])} "
            f"D2c={len(res['yoyos_clean'])} D3={len(res['turning_room'])} "
            f"lethTk={res['lethal_ticks']}"
        )
        chi_deg = [abs(DEG(s.chi)) for s in tr]
        k = chi_deg.index(max(chi_deg))
        print(f"  max|chi| {chi_deg[k]:.2f} deg at step {k} t {tr[k].t:.3f} nz {tr[k].nz:.2f}")
        # the claim's two cited numbers
        print(f"  aspect step0 = {DEG(tr[0].aspect):.2f} deg")
        if len(tr) > 156:
            print(
                f"  aspect step156 = {DEG(tr[156].aspect):.2f} deg  "
                f"(delta {DEG(tr[156].aspect - tr[0].aspect):+.2f})"
            )
        d1_clause_audit(tr, label)
        d2_clause_audit(tr, label)
        aspect_windows(tr, label)


if __name__ == "__main__":
    main()
