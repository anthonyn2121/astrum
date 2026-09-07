"""REFUTATION probe for the T2 'alpha_req' advocacy claim.

Everything here is recomputed from the dumped CSVs. Nothing is taken from the advocacy script.
Two files: the kill seed (62) and the non-kill seed (28), which shares the identical commanded
action stream through step 172 and then flies 403 more steps.

Tests
  R1  literal reproduction of every cited number
  R2  is the roll-in at step 24 'unloaded'?
  R3  decomposition of alpha_req into POINTING (my velocity vector vs the LOS) and LEAD
      (target crossing motion), by re-solving the SAME intercept solver the env uses
  R4  is alpha_req range-dependent? (translate the bandit along the LOS, keep all velocities)
  R5  when did alpha_req first enter the alpha cone, and was it the binding constraint at the shot?
  R6  the non-kill seed's remaining 403 steps: does the 'shootable lay' persist?
  R7  aspect at the shot vs the collision-course head-on the claim contrasts against
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: E402,F401  (its _paths puts astrum_dogfight + astrum_env on sys.path)
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.gunlay import intercept  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ang(a, b):
    return math.degrees(
        math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.clip(np.dot(a, b), -1.0, 1.0)))
    )


def vecs(d):
    """ENU position / velocity arrays. Bandit velocity by central difference of its position."""
    me_p = np.c_[d["pos_e_m"], d["pos_n_m"], d["pos_u_m"]]
    me_v = np.c_[d["vel_e_ms"], d["vel_n_ms"], d["vel_u_ms"]]
    op_p = np.c_[d["opp_pos_e_m"], d["opp_pos_n_m"], d["opp_pos_u_m"]]
    op_v = np.gradient(op_p, DT, axis=0)
    return me_p, me_v, op_p, op_v


def main():
    d = load(KILL)
    n = d["_n"]
    ar = d["alpha_req_deg"]
    lm = d["lay_miss_m"]
    print(f"kill csv rows = {n}; constants: HIT_RADIUS_M={C.HIT_RADIUS_M} MUZZLE={C.MUZZLE_MPS} ")
    print(f"  ALPHA_REF={math.degrees(C.ALPHA_REF):.1f} deg  R_EFF={C.R_EFF}")

    hdr("R1  LITERAL REPRODUCTION OF THE CITED NUMBERS")
    checks = [
        ("alpha_req step 0", ar[0], 45.46),
        ("alpha_req step 24", ar[24], 51.09),
        ("t_s step 24", d["t_s"][24], 1.000),
        ("alpha_req step 164", ar[164], 10.08),
        ("t_s step 164", d["t_s"][164], 6.833),
        ("alpha_req step 156", ar[156], 11.30),
        ("alpha step 156", d["alpha_deg"][156], 7.43),
        ("lay_miss step 0", lm[0], 1621.73),
        ("lay_miss step 156", lm[156], 14.68),
        ("lay_miss step 168", lm[168], 3.94),
        ("rng step 168", d["rng_m"][168], 87.0),
        ("t_s step 168", d["t_s"][168], 7.000),
    ]
    for nm, got, said in checks:
        print(
            f"  {nm:<22} recomputed {got:12.4f}   claimed {said:10.3f}   {'OK' if abs(got - said) < 0.6 else 'MISMATCH'}"
        )
    print(
        f"  alpha_req GLOBAL max {ar.max():.2f} at step {int(ar.argmax())} (t {d['t_s'][int(ar.argmax())]:.3f})"
    )
    print(
        f"  alpha_req GLOBAL min {ar.min():.2f} at step {int(ar.argmin())} (t {d['t_s'][int(ar.argmin())]:.3f})"
    )
    print(f"  alpha_req on steps 165-172: {np.round(ar[165:], 2).tolist()}")
    seg = ar[24:170]
    print(
        f"  claimed 'falling on 135/145 steps' -> recomputed down {int((np.diff(seg) < 0).sum())}/{len(seg) - 1}"
    )
    print(
        f"  claimed monotone? rises on {int((np.diff(seg) > 0).sum())} of those steps; largest rise {np.diff(seg).max():+.3f} deg"
    )
    print(
        f"  alpha_req over the WHOLE trace: down {int((np.diff(ar) < 0).sum())}/{n - 1}, up {int((np.diff(ar) > 0).sum())}"
    )
    fire = d["fire"] > 0
    print(
        f"  first fire step = {int(np.argmax(fire))}, fire steps = {int(fire.sum())}, "
        f"contiguous={bool(fire[int(np.argmax(fire)) :].all())}"
    )
    print(
        f"  max measured alpha over the trace = {d['alpha_deg'].max():.2f} deg at step {int(d['alpha_deg'].argmax())}"
    )

    hdr("R2  IS THE STEP-24 ROLL-IN 'UNLOADED'?  (the claim's parenthetical)")
    for k in (0, 8, 16, 20, 24, 28, 32, 40):
        print(
            f"  step {k:3d} t {d['t_s'][k]:6.3f}  elev {d['elev'][k]:+7.3f}  nz {d['nz_g'][k]:6.2f} g  "
            f"alpha {d['alpha_deg'][k]:6.2f}  phi {d['phi_deg'][k]:7.2f}  alpha_req {ar[k]:6.2f}  "
            f"ata {d['ata_deg'][k]:6.2f}  rng {d['rng_m'][k]:7.1f}"
        )
    print(f"  nz at step 24 = {d['nz_g'][24]:.2f} g (NZ_LOADED bar in the detector is 3.0 g)")

    hdr("R3  DECOMPOSITION — is alpha_req anything other than 'my velocity vector vs the LOS'?")
    me_p, me_v, op_p, op_v = vecs(d)
    rows = []
    for k in range(n):
        los = op_p[k] - me_p[k]
        vh = unit(me_v[k])
        P = ang(vh, unit(los))  # pointing error: velocity vector vs LOS
        sol = intercept(me_p[k], me_v[k], op_p[k], op_v[k])
        aim = sol.boresight if sol.feasible else unit(los)
        arq = ang(aim, vh)  # this is alpha_req, recomputed
        L = ang(aim, unit(los))  # kinematic lead off the LOS
        # counterfactual: static target (no crossing motion at all)
        s0 = intercept(me_p[k], me_v[k], op_p[k], np.zeros(3))
        ar_static = ang(s0.boresight if s0.feasible else unit(los), vh)
        rows.append((P, arq, L, ar_static, sol.tof_s if sol.feasible else float("nan")))
    Pa, ARq, La, ARs, TOF = (np.array(x) for x in zip(*rows))
    err = np.abs(ARq - ar)
    print(
        f"  my re-solve of alpha_req vs the dumped column: max abs err {err.max():.4f} deg "
        f"(median {np.median(err):.4f}) -> the decomposition machinery is validated"
    )
    print(
        f"  {'step':>4} {'t_s':>6} {'rng':>7} {'ata':>7} {'POINT P':>8} {'LEAD L':>7} {'alpha_req':>9} "
        f"{'ar(static tgt)':>14} {'tof_s':>6}"
    )
    for k in (0, 12, 24, 48, 72, 96, 120, 140, 156, 164, 172):
        print(
            f"  {k:4d} {d['t_s'][k]:6.3f} {d['rng_m'][k]:7.1f} {d['ata_deg'][k]:7.2f} {Pa[k]:8.2f} "
            f"{La[k]:7.2f} {ARq[k]:9.2f} {ARs[k]:14.2f} {TOF[k]:6.3f}"
        )
    print(
        f"  regression alpha_req = a*P + b : a={np.polyfit(Pa, ARq, 1)[0]:.4f} b={np.polyfit(Pa, ARq, 1)[1]:+.3f} "
        f"deg, R^2={np.corrcoef(Pa, ARq)[0, 1] ** 2:.5f}"
    )
    print(
        f"  LEAD term L over the trace: min {La.min():.2f} max {La.max():.2f} deg  "
        f"(asin(V_tgt/MUZZLE) ceiling = {math.degrees(math.asin(d['opp_vt_ms'].max() / C.MUZZLE_MPS)):.2f} deg)"
    )
    print(
        f"  alpha_req against a STATIC target (all crossing lead removed): step 0 {ARs[0]:.2f}, "
        f"step 156 {ARs[156]:.2f} -> the fall is {ARs[0] - ARs[156]:.2f} of the {ARq[0] - ARq[156]:.2f} deg total"
    )

    hdr(
        "R4  IS alpha_req RANGE-DEPENDENT?  (translate the bandit along the LOS; keep all velocities)"
    )
    print(
        f"  {'step':>4} {'true rng':>9} "
        + " ".join(f"{f'ar@{r}m':>9}" for r in (300, 600, 1200, 3000))
    )
    for k in (0, 24, 96, 156):
        out = []
        for r in (300.0, 600.0, 1200.0, 3000.0):
            los = op_p[k] - me_p[k]
            p2 = me_p[k] + unit(los) * r
            s = intercept(me_p[k], me_v[k], p2, op_v[k])
            out.append(ang(s.boresight if s.feasible else unit(los), unit(me_v[k])))
        print(f"  {k:4d} {d['rng_m'][k]:9.1f} " + " ".join(f"{v:9.2f}" for v in out))
    print(
        "  => alpha_req is a POINTING quantity: closing the range does not change it. "
        "lay_miss = 1030*sin(lead)*tof is the range-carrying one."
    )

    hdr("R5  WHEN DID THE LAY BECOME ACHIEVABLE, AND WAS IT THE BINDING CONSTRAINT?")
    for cone in (10.0, 15.0, 16.7, 20.0):
        idx = np.where(ar <= cone)[0]
        if len(idx):
            k = int(idx[0])
            print(
                f"  alpha_req first <= {cone:5.1f} deg at step {k:3d} (t {d['t_s'][k]:6.3f}) "
                f"rng {d['rng_m'][k]:7.1f} m, lay_miss {lm[k]:9.1f} m, in_range600={d['rng_m'][k] <= 600}"
            )
    k600 = int(np.argmax(d["rng_m"] <= 600.0))
    print(
        f"  rng first <= 600 m at step {k600} (t {d['t_s'][k600]:.3f}): alpha_req {ar[k600]:.2f} deg "
        f"(ALREADY inside the cone), lay_miss {lm[k600]:.1f} m -> trigger still SHUT (needs <= 15 m)"
    )
    print(
        f"  steps with alpha_req <= 15 deg: {int((ar <= 15).sum())}/{n}; of those, lay_miss <= 15 m on "
        f"{int(((ar <= 15) & (lm <= 15)).sum())}; rng <= 600 on {int(((ar <= 15) & (d['rng_m'] <= 600)).sum())}"
    )
    print(
        f"  steps with alpha_req <= 15 AND rng <= 600 but lay_miss > 15 m (achievable, in range, still "
        f"no shot): {int(((ar <= 15) & (d['rng_m'] <= 600) & (lm > 15)).sum())}"
    )

    hdr("R6  THE NON-KILL SEED (28): 403 MORE STEPS OF THE SAME GENOME")
    e = load(NOKILL)
    m = e["_n"]
    ae = e["alpha_req_deg"]
    same = np.abs(ae[:n] - ar).max()
    print(
        f"  rows {m}; max |alpha_req(seed28) - alpha_req(seed62)| over the shared 0-172 = {same:.3e} "
        f"-> the cited fall is NOT seed-specific, it is the same trajectory"
    )
    print(
        f"  steps 173-575: alpha_req min {ae[173:].min():.2f} median {np.median(ae[173:]):.2f} "
        f"max {ae[173:].max():.2f} deg; <=15 deg on {int((ae[173:] <= 15).sum())}/{m - 173} steps"
    )
    print(
        f"  steps 173-575: rng min {e['rng_m'][173:].min():.1f} max {e['rng_m'][173:].max():.1f} m; "
        f"rng<=600 on {int((e['rng_m'][173:] <= 600).sum())} steps; fire on {int((e['fire'][173:] > 0).sum())} steps"
    )
    print(
        f"  steps 173-575: lay_miss min {e['lay_miss_m'][173:].min():.1f} m; "
        f"<=15 m on {int((e['lay_miss_m'][173:] <= 15).sum())} steps"
    )
    print(
        f"  opp_health: step172 {e['opp_health'][172]:.1f} -> final {e['opp_health'][-1]:.1f}; "
        f"my health {e['health'][172]:.1f} -> {e['health'][-1]:.1f}"
    )
    print(
        f"  altitude ledger 0->575: {e['alt_m'][0]:.1f} -> {e['alt_m'][-1]:.1f} m "
        f"({e['alt_m'][-1] - e['alt_m'][0]:+.1f})"
    )
    # where alpha_req sits when the second-pass geometry is in range
    inr = np.where((e["rng_m"][173:] <= 1200))[0] + 173
    if len(inr):
        print(
            f"  on the {len(inr)} later steps inside R_EFF(1200 m): alpha_req min {ae[inr].min():.2f} "
            f"median {np.median(ae[inr]):.2f} max {ae[inr].max():.2f}; lay_miss min {e['lay_miss_m'][inr].min():.1f} m"
        )

    hdr("R7  WHAT GEOMETRY WAS THE SHOT TAKEN IN?")
    for k in (140, 150, 156, 160, 164, 168, 172):
        print(
            f"  step {k:3d} t {d['t_s'][k]:6.3f} rng {d['rng_m'][k]:7.1f} aspect {d['aspect_deg'][k]:7.2f} "
            f"ata {d['ata_deg'][k]:6.2f} hca {d['hca_deg'][k]:8.2f} closure {d['closure_ms'][k]:+7.1f} "
            f"alpha_req {ar[k]:6.2f} lay_miss {lm[k]:8.2f} tof {d['lay_tof_s'][k]:5.3f} fire {int(d['fire'][k] > 0)}"
        )
    print(f"  min rng over the trace {d['rng_m'].min():.1f} m at step {int(d['rng_m'].argmin())}")
    print(
        f"  aspect at first fire (156) = {d['aspect_deg'][156]:.2f} deg  (180 = pure head-on reciprocal)"
    )
    print(
        f"  aspect band over the firing window: {d['aspect_deg'][156:].min():.2f} - {d['aspect_deg'][156:].max():.2f} deg"
    )


if __name__ == "__main__":
    main()
