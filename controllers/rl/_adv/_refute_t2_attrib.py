"""Attribution: did MY velocity vector swing onto the LOS, or did the LOS swing onto it?"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    return {n: np.array([float(r[n]) for r in rows]) for n in names} | {"_n": len(rows)}


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def ang(a, b):
    return math.degrees(
        math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.clip(np.dot(a, b), -1.0, 1.0)))
    )


d = load(KILL)
n = d["_n"]
me_p = np.c_[d["pos_e_m"], d["pos_n_m"], d["pos_u_m"]]
me_v = np.c_[d["vel_e_ms"], d["vel_n_ms"], d["vel_u_ms"]]
op_p = np.c_[d["opp_pos_e_m"], d["opp_pos_n_m"], d["opp_pos_u_m"]]
los = np.array([unit(op_p[k] - me_p[k]) for k in range(n)])
vh = np.array([unit(me_v[k]) for k in range(n)])

print("=" * 100)
print("  B1  HEADING-CROSSING ANGLE: was this ever anything but a reciprocal collision course?")
print("=" * 100)
off = np.abs(180.0 - np.abs(d["hca_deg"]))
print(
    f"  |180 - |hca|| over all {n} steps: min {off.min():.2f} max {off.max():.2f} median {np.median(off):.2f} deg"
)
print(f"  steps with |180-|hca|| > 10 deg: {int((off > 10).sum())}")
print(
    f"  aspect: step 0 {d['aspect_deg'][0]:.2f} -> step 156 {d['aspect_deg'][156]:.2f} deg "
    f"(180 = the bandit's nose exactly on me)"
)

print()
print("=" * 100)
print("  B2  WHO CLOSED THE POINTING ERROR — my velocity vector, or the line of sight?")
print("=" * 100)
for k in (0, 43, 96, 140, 156):
    print(
        f"  step {k:3d}: P = angle(vhat, los) = {ang(vh[k], los[k]):6.2f} deg | "
        f"vhat moved {ang(vh[0], vh[k]):6.2f} deg since step 0 | "
        f"los moved {ang(los[0], los[k]):6.2f} deg since step 0 | "
        f"P if I had NOT turned = angle(vhat0, los_k) = {ang(vh[0], los[k]):6.2f} deg"
    )
sw_v = sum(ang(vh[k - 1], vh[k]) for k in range(1, 157))
sw_l = sum(ang(los[k - 1], los[k]) for k in range(1, 157))
print(f"  total arc swept 0->156: my velocity vector {sw_v:.2f} deg, the LOS {sw_l:.2f} deg")
print("  (the 'if I had NOT turned' column re-uses the OBSERVED bandit path, so it is a kinematic")
print("   bound, not a re-flight: the T3 is closed-loop on me and would have flown differently.)")
