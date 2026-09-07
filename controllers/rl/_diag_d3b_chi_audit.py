"""Rebuild the exact `bfm_detect` trace from the replay dumps and re-run the real detectors.

Verifies the stdlib transcription used in the anti-BFM audit: the `smp_*` columns are the raw
`make_sample` field set, so `Sample(**cols)` reconstructs the trace `detect_all` read.
"""

from __future__ import annotations

import csv

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm_detect as B  # noqa: E402

FIELDS = [
    "t",
    "chi",
    "chi_valid",
    "gamma",
    "nz",
    "phi",
    "alt",
    "ail",
    "rng",
    "ata",
    "ata_az",
    "aspect",
    "closure",
    "cpa_miss",
    "es_rel",
    "lay_miss_m",
    "lethal",
]
COL = {
    "t": "smp_t",
    "chi": "smp_chi_rad",
    "chi_valid": "smp_chi_valid",
    "gamma": "smp_gamma_rad",
    "nz": "smp_nz",
    "phi": "smp_phi_rad",
    "alt": "smp_alt",
    "ail": "smp_ail",
    "rng": "smp_rng",
    "ata": "smp_ata",
    "ata_az": "smp_ata_az",
    "aspect": "smp_aspect",
    "closure": "smp_closure",
    "cpa_miss": "smp_cpa_miss",
    "es_rel": "smp_es_rel",
    "lay_miss_m": "smp_lay_miss_m",
    "lethal": "smp_lethal",
}


def rebuild(path: str) -> list[B.Sample]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    tr = []
    for r in rows:
        kw = {}
        for f in FIELDS:
            v = r[COL[f]].strip()
            if f in ("chi_valid", "lethal"):
                kw[f] = v.lower() in ("1", "true", "1.0")
            else:
                kw[f] = float(v)
        tr.append(B.Sample(**kw))
    return tr


for path, label in [
    ("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv", "KILL seed62"),
    ("/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv", "NOKILL seed28"),
]:
    tr = rebuild(path)
    d = B.detect_all(tr)
    print("=" * 78)
    print(f"{label}: {len(tr)} samples, t {tr[0].t:.3f}..{tr[-1].t:.3f}")
    for k in ("displacement_rolls", "yoyos", "yoyos_clean", "turning_room", "lethal_ticks"):
        v = d.get(k)
        print(f"  {k:20s} = {v if not isinstance(v, list) else len(v)}")
    print(f"  raw keys: {sorted(d.keys())}")
    # bar sweep on the REAL detector body
    for bar in (60.0, 55.0, 50.0, 45.0, 40.0, 35.0):
        B.CHI_OUT_DEG = bar
        print(f"    CHI_OUT_DEG={bar:5.1f} -> D1 = {len(B.detect_displacement_rolls(tr))}")
    B.CHI_OUT_DEG = 60.0
    for bar in (30.0, 28.4, 25.0, 20.0):
        B.YOYO_GAMMA_SWING_DEG = bar
        print(f"    YOYO_GAMMA_SWING_DEG={bar:5.1f} -> D2 = {len(B.detect_yoyos(tr))}")
    B.YOYO_GAMMA_SWING_DEG = 30.0
    for bar in (10.0, 15.0, 20.0, 25.0, 30.0):
        B.YOYO_GAMMA_RETURN_DEG = bar
        print(f"    YOYO_GAMMA_RETURN_DEG={bar:5.1f} -> D2 = {len(B.detect_yoyos(tr))}")
    B.YOYO_GAMMA_RETURN_DEG = 10.0
