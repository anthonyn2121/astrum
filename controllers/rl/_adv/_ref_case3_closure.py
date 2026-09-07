"""Refutation probe for T10 (closure was never bled).

Rebuilds the exact `Sample` trace the detector reads from the smp_* CSV block and runs
`bfm_detect.detect_all`, then probes the yo-yo swing threshold to separate a THRESHOLD
rejection from a MECHANISM rejection.
"""

from __future__ import annotations

import csv
import sys

sys.path.insert(0, "/workspace/controllers/rl")

from astrum_rl import _paths  # noqa: E402

_paths.require_astrum_env()
from astrum_dogfight import bfm_detect as B  # noqa: E402

PATHS = {
    "kill_seed62": "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv",
    "nokill_seed28": "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv",
}


def trace(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [
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
        for r in rows
    ]


def main():
    for name, path in PATHS.items():
        tr = trace(path)
        d = B.detect_all(tr)
        print(f"\n=== {name}: ticks={d['ticks']} dur={d['duration_s']:.3f} s")
        print(
            f"  D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])} "
            f"D2c={len(d['yoyos_clean'])} D3={len(d['turning_room'])} "
            f"lethTk={d['lethal_ticks']}"
        )
        base = B.YOYO_GAMMA_SWING_DEG
        for bar in (30.0, 28.4, 25.0, 20.0, 10.0):
            B.YOYO_GAMMA_SWING_DEG = bar
            ev = B.detect_yoyos(tr)
            kinds = [(e.kind, round(e.detail.get("gamma_swing_deg", 0.0), 2)) for e in ev]
            print(f"    YOYO_GAMMA_SWING_DEG={bar:5.1f} -> D2={len(ev)} {kinds}")
        B.YOYO_GAMMA_SWING_DEG = base
        for bar in (6.0, 10.0, 20.0, 30.0):
            B.YOYO_RETURN_S = bar
            B.YOYO_GAMMA_SWING_DEG = 25.0
            ev = B.detect_yoyos(tr)
            print(
                f"    YOYO_RETURN_S={bar:5.1f} (swing bar 25.0) -> D2={len(ev)} "
                f"{[(e.kind, round(e.detail.get('closure_exit', 0.0), 1), e.detail.get('failed_exit')) for e in ev]}"
            )
        B.YOYO_RETURN_S = 6.0
        B.YOYO_GAMMA_SWING_DEG = base


if __name__ == "__main__":
    main()
