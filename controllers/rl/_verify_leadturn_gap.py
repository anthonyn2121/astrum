"""Independent verification of the 'no lead-turn detector' claim, run against the REAL detector.

Rebuilds the exact Sample trace `B.detect_all` reads from the smp_* block of the replay CSVs and
re-runs the detector, so every D1/D3/D4 number below is the module's own arithmetic, not a
re-implementation.
"""
import csv
import sys

sys.path.insert(0, "/workspace/crates/astrum-env/python")

from astrum_dogfight import bfm_detect as B


def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def rebuild(rows):
    tr = []
    for r in rows:
        tr.append(
            B.Sample(
                t=float(r["smp_t"]),
                chi=float(r["smp_chi_rad"]),
                chi_valid=float(r["smp_chi_valid"]) > 0.5,
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
                lethal=float(r["smp_lethal"]) > 0.5,
            )
        )
    return tr


for tag, path in [
    ("KILL seed62", "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"),
    ("NOKILL seed28", "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"),
]:
    rows = load(path)
    tr = rebuild(rows)
    d = B.detect_all(tr)
    c = d["commitment"]
    print(f"=== {tag}  ticks {d['ticks']}  dur {d['duration_s']:.3f} s")
    print(
        f"  D1 {len(d['displacement_rolls'])}  D2 {len(d['yoyos'])}"
        f"  D2c {len(d['yoyos_clean'])}  D3 {len(d['turning_room'])}"
        f"  lethalTicks {d['lethal_ticks']}"
    )
    print(
        f"  D4 bank hold {c['longest_bank_hold_s']:.3f} s  dalt {c['bank_hold_dalt_m']:+.1f} m"
        f"  peak bank {c['bank_max_deg']:.1f} deg"
    )
    print(
        f"  D4 chi  hold {c['longest_chi_hold_s']:.3f} s  dalt {c['chi_hold_dalt_m']:+.1f} m"
        f"  valid_frac {c['valid_frac']:.3f}"
    )
    for ev in d["turning_room"]:
        print(
            f"  D3 event steps {ev.i0}-{ev.i1} t {ev.t0:.3f}-{ev.t1:.3f}"
            f" d_aspect {ev.d_aspect_deg:+.1f} deg  detail {ev.detail}"
        )
    # Every candidate D1 run and the clause that kills it.
    import math

    hold = [
        (abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr
    ]
    for i, j in B._runs(hold):
        dur = tr[j].t - tr[i].t
        verdict = "kept past hold test" if dur >= B.CHI_MIN_HOLD_S else "DROPPED at hold test"
        print(
            f"  D1 candidate run steps {i}-{j}  dur {dur:.3f} s vs CHI_MIN_HOLD_S"
            f" {B.CHI_MIN_HOLD_S} -> {verdict}"
        )
