"""Run the REAL bfm_detect.commitment on both dumped traces, rebuilt from the smp_* columns."""

import csv
import sys

sys.path.insert(0, "/workspace/controllers/rl/vendor")
sys.path.insert(0, "/workspace/crates/astrum-env/python")
from astrum_dogfight import bfm_detect as B


def trace(path):
    tr = []
    for r in csv.DictReader(open(path)):
        tr.append(
            B.Sample(
                t=float(r["smp_t"]),
                chi=float(r["smp_chi_rad"]),
                chi_valid=bool(float(r["smp_chi_valid"])),
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
                lethal=bool(float(r["smp_lethal"])),
            )
        )
    return tr


for tag, p in (
    ("KILL seed62", "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"),
    ("NOKILL seed28", "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"),
):
    tr = trace(p)
    c = B.commitment(tr)
    d = B.detect_all(tr)
    print(f"== {tag}  n={len(tr)}")
    for k in (
        "longest_bank_hold_s",
        "bank_hold_dalt_m",
        "longest_chi_hold_s",
        "chi_hold_dalt_m",
        "bank_max_deg",
        "longest_rate_cmd_run_s",
        "roll_runs",
        "max_abs_window_mean_ail",
    ):
        print(f"   {k:26s} {c[k]}")
    print(
        f"   D1 displacement_rolls={len(d['displacement_rolls'])} D2 yoyos={len(d['yoyos'])} "
        f"D2c yoyos_clean={len(d['yoyos_clean'])} D3 turning_room={len(d['turning_room'])} "
        f"lethal_ticks={d['lethal_ticks']}"
    )
    print(
        f"   AIL_COMMIT={B.AIL_COMMIT} BANK_COMMIT_DEG={B.BANK_COMMIT_DEG} NZ_LOADED={B.NZ_LOADED} "
        f"CHI_ROOM_DEG={B.CHI_ROOM_DEG} COMMIT_WINDOW_S={B.COMMIT_WINDOW_S}"
    )
