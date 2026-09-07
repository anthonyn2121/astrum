"""Part 3: is the ata_az zero crossing at step 30 a NOSE motion or a ROLL artefact?

ata_az is defined in geometry.py:150 as the LOS bearing in BODY XY (right +). At phi ~ 120 deg the
body y-axis points nearly vertical, so body 'azimuth' stops measuring horizontal bearing. The
roll-free invariants are `ata` (total angle off boresight) and the horizontal (LOS bearing - ground
track).
"""

from __future__ import annotations

import csv

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    return {n: np.array([float(r[n]) for r in rows]) for n in names} | {"_n": len(rows)}


k = load(KILL)
n = k["_n"]
p_me = np.stack([k["pos_e_m"], k["pos_n_m"]], 1)
p_op = np.stack([k["opp_pos_e_m"], k["opp_pos_n_m"]], 1)
r = p_op - p_me
los_az = np.degrees(np.unwrap(np.arctan2(r[:, 0], r[:, 1])))
course = np.degrees(np.unwrap(np.arctan2(k["vel_e_ms"], k["vel_n_ms"])))
horiz_off = los_az - course  # + = LOS to the RIGHT of my ground track

print("=" * 100)
print("  N. THE ata_az ZERO CROSSING AT STEP 30: nose motion, or the 120 deg of ROLL?")
print("=" * 100)
print(
    f"  {'step':>4} {'t':>6} {'phi':>8} {'ata_az':>8} {'ata_el':>8} {'ata(tot)':>9} {'horizOff':>9} {'course':>8} {'gamma':>7}"
)
for s in (0, 6, 12, 18, 24, 27, 30, 33, 36, 42, 48, 60, 72, 84, 96):
    print(
        f"  {s:4d} {k['t_s'][s]:6.3f} {k['phi_deg'][s]:8.2f} {k['ata_az_deg'][s]:+8.2f} "
        f"{k['ata_el_deg'][s]:+8.2f} {k['ata_deg'][s]:9.2f} {horiz_off[s]:+9.2f} "
        f"{course[s]:+8.2f} {k['gamma_deg'][s]:+7.2f}"
    )
print()
print("  ata (TOTAL angle off boresight, ROLL-INVARIANT) at the crossing:")
print(
    f"    ata[0]={k['ata_deg'][0]:.2f}  ata[24]={k['ata_deg'][24]:.2f}  ata[30]={k['ata_deg'][30]:.2f}  "
    f"ata[36]={k['ata_deg'][36]:.2f} deg"
)
print(
    f"    => at the step where ata_az 'crosses zero' the nose is still {k['ata_deg'][30]:.2f} deg OFF the LOS."
)
print(
    f"    ata_el over steps 0->30: {k['ata_el_deg'][0]:+.2f} -> {k['ata_el_deg'][30]:+.2f} deg "
    f"(the offset MOVED into elevation; |ata_el| grew by {abs(k['ata_el_deg'][30]) - abs(k['ata_el_deg'][0]):+.2f} deg)"
)
print(f"    phi over the same steps: {k['phi_deg'][0]:.2f} -> {k['phi_deg'][30]:.2f} deg of BANK")
print(
    f"    my ground track over the same steps: {course[0]:+.2f} -> {course[30]:+.2f} deg "
    f"(only {course[30] - course[0]:+.2f} deg of actual turn)"
)
print()
print("  HORIZONTAL angle-off (LOS bearing - my ground track), sign-invariant to roll:")
print(
    f"    min over the trace = {horiz_off[:173].min():+.2f} deg at step {int(horiz_off[:173].argmin())}"
)
print(
    f"    steps where it is NEGATIVE (nose LEAD of the LOS horizontally): {(horiz_off < 0).sum()}/{n}"
)
print(
    f"    => horizontally the nose is on the LAG side of the LOS for {(horiz_off > 0).sum()}/{n} steps"
)
print()
print("=" * 100)
print("  O. WHEN DID THE SHOT BECOME AVAILABLE, AND WHY?")
print("=" * 100)
ar = k["alpha_req_deg"]
first = int(np.argmax(ar < 16.7))
print(
    f"  alpha_req < 16.7 deg (the measured usable alpha ceiling) first at step {first} "
    f"(t {k['t_s'][first]:.3f} s, rng {k['rng_m'][first]:.0f} m)"
)
print(
    f"  alpha_req[0]={ar[0]:.2f} -> [24]={ar[24]:.2f} (WORSE) -> [96]={ar[96]:.2f} -> [144]={ar[144]:.2f} -> min {ar.min():.2f} at step {int(ar.argmin())}"
)
print(
    f"  lay_tof_s at the shot: step 156 {k['lay_tof_s'][156]:.4f} s; at step 24 {k['lay_tof_s'][24]:.4f} s"
)
print("  -> required lead angle scales with time of flight, which scales with RANGE.")
print(
    f"     rng 2678 m (step 24) tof {k['lay_tof_s'][24]:.3f} s; rng 325 m (step 156) tof {k['lay_tof_s'][156]:.3f} s "
    f"= {k['lay_tof_s'][24] / k['lay_tof_s'][156]:.1f}x less lead needed purely from closing."
)
print()
print("=" * 100)
print("  P. COUNTERFACTUAL: what does 'commitment' buy over NOT turning?")
print("=" * 100)
print(
    f"  cpa_miss[0] = {k['cpa_miss_m'][0]:.1f} m is the constant-velocity miss from step 0 (bfm_detect.cpa_miss_m)."
)
print(
    f"  The maneuver drove it to {k['cpa_miss_m'][-1]:.1f} m, monotonically "
    f"({(np.diff(k['cpa_miss_m']) < 0).sum()}/{n - 1} steps decreasing)."
)
print(
    "  BASE RATE (memory: gun-lethal-envelope): p_hit/round 0.98 @300 m, 0.20 @600 m, 0.000 @1200 m,"
)
print("  hit sphere 8 m -> a 1622 m predicted miss is 0 HP by construction, and any monotone")
print("  convergence to <8 m is a hit whether or not it is BFM.")
print()
print("=" * 100)
print(
    "  Q. CHI (out-of-plane) DURING THE ALLEGED TURN — the standing rule: gated HOLDS, not extrema"
)
print("=" * 100)
chi = np.abs(k["chi_deg"])
nz = k["nz_g"]
for bar in (60.0, 45.0, 30.0):
    m = (chi > bar) & (nz > 3.0)
    runs = []
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            runs.append((i, j, (j - i + 1) * DT, k["alt_m"][j] - k["alt_m"][i]))
            i = j + 1
        else:
            i += 1
    tot = sum(x[2] for x in runs)
    lg = max(runs, key=lambda x: x[2]) if runs else None
    print(
        f"  |chi|>{bar:4.0f} AND nz>3: runs {len(runs)}, total {tot:.3f} s, longest "
        f"{(lg[2] if lg else 0.0):.3f} s (steps {(lg[0] if lg else -1)}-{(lg[1] if lg else -1)}), "
        f"dalt over longest {(lg[3] if lg else 0.0):+.1f} m  [bar CHI_MIN_HOLD_S = 0.5 s]"
    )
print(
    f"  max|chi| = {chi.max():.1f} deg at step {int(chi.argmax())} (t {k['t_s'][int(chi.argmax())]:.3f}, "
    f"nz {nz[int(chi.argmax())]:.2f} g) -- the INITIAL CONDITION, not a maneuver."
)
