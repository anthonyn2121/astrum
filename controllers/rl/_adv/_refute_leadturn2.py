"""Part 2: discriminate LEAD TURN from LEAD-COLLISION INTERCEPT, and attribute the geometry."""

from __future__ import annotations

import csv

import numpy as np

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


k = load(KILL)
q = load(NOKILL)
n = k["_n"]

hdr("I. 'LEAD PURSUIT, NOT PURE' — was the nose on a LEAD SOLUTION or merely on the near side?")
print("   geometry.py:156  lead_error = nose vs the CORRECTED required boresight; LOW = solution")
print(
    f"   {'step':>4} {'t':>6} {'ata_az':>8} {'lead_az':>8} {'lead_err':>9} {'alpha_req':>9} {'lay_miss_m':>11} {'rng':>7}"
)
for s in list(range(0, 109, 12)) + [120, 132, 144, 156, 168, 172]:
    print(
        f"   {s:4d} {k['t_s'][s]:6.3f} {k['ata_az_deg'][s]:+8.2f} {k['lead_az_deg'][s]:+8.2f} "
        f"{k['lead_error_deg'][s]:9.2f} {k['alpha_req_deg'][s]:9.2f} {k['lay_miss_m'][s]:11.1f} {k['rng_m'][s]:7.0f}"
    )
le = k["lead_error_deg"]
w = slice(36, 109)
print(
    f"\n   lead_error over steps 36-108 (the alleged 'lead pursuit' window): min {le[w].min():.2f} "
    f"max {le[w].max():.2f} mean {le[w].mean():.2f} deg"
)
print(
    f"   lay_miss over the same window: min {k['lay_miss_m'][w].min():.1f} max {k['lay_miss_m'][w].max():.1f} m"
    f"   (gun hit sphere 8 m; fire gate 15 m)"
)
print(
    f"   lead_error first below 15 deg at step {int(np.argmax(le < 15.0))} (t {k['t_s'][int(np.argmax(le < 15.0))]:.3f});"
    f" below 5 deg at step {int(np.argmax(le < 5.0))} (t {k['t_s'][int(np.argmax(le < 5.0))]:.3f})"
)
print(
    f"   alpha_req: step0 {k['alpha_req_deg'][0]:.2f}, max {k['alpha_req_deg'].max():.2f} at step "
    f"{int(k['alpha_req_deg'].argmax())}, min {k['alpha_req_deg'].min():.2f} at step {int(k['alpha_req_deg'].argmin())}"
)

hdr("J. WHO TURNED? attribute the ata_az crossing between MY turn and the BANDIT's")
p_me = np.stack([k["pos_e_m"], k["pos_n_m"]], 1)
p_op = np.stack([k["opp_pos_e_m"], k["opp_pos_n_m"]], 1)
r = p_op - p_me
los_az = np.degrees(np.unwrap(np.arctan2(r[:, 0], r[:, 1])))
course = np.degrees(np.unwrap(np.arctan2(k["vel_e_ms"], k["vel_n_ms"])))
opp = np.degrees(np.unwrap(np.radians(k["opp_psi_deg"])))
print(
    f"   {'step':>4} {'t':>6} {'los_az':>8} {'my_course':>10} {'opp_psi':>9} {'los-course':>11} {'ata_az':>8}"
)
for s in (0, 12, 24, 30, 36, 48, 72, 96, 120, 144, 156, 172):
    print(
        f"   {s:4d} {k['t_s'][s]:6.3f} {los_az[s]:8.2f} {course[s]:10.2f} {opp[s]:9.2f} "
        f"{los_az[s] - course[s]:11.2f} {k['ata_az_deg'][s]:+8.2f}"
    )
print("\n   OVER steps 0-30 (up to the LOS crossing the nose):")
print(
    f"     my course rotated {course[30] - course[0]:+.2f} deg;  the LOS rotated {los_az[30] - los_az[0]:+.2f} deg;"
    f"  bandit heading rotated {opp[30] - opp[0]:+.2f} deg"
)
print(
    f"     ata_az went {k['ata_az_deg'][0]:+.2f} -> {k['ata_az_deg'][30]:+.2f} deg (change {k['ata_az_deg'][30] - k['ata_az_deg'][0]:+.2f})"
)
print(
    f"     share of the crossing owed to MY rotation: {abs(course[30] - course[0]) / (abs(course[30] - course[0]) + abs(los_az[30] - los_az[0])) * 100:.1f}%"
)
print(
    f"\n   WHOLE TRACE 0-172: my course {course[-1] - course[0]:+.2f} deg;  bandit heading {opp[-1] - opp[0]:+.2f} deg;"
    f"  LOS {los_az[-1] - los_az[0]:+.2f} deg"
)
print(
    f"     bandit nz mean {k['opp_nz_g'].mean():.2f} g max {k['opp_nz_g'].max():.2f}; bandit alt "
    f"{k['opp_alt_m'][0]:.0f} -> {k['opp_alt_m'][-1]:.0f} m; bandit vt {k['opp_vt_ms'][0]:.1f} -> {k['opp_vt_ms'][-1]:.1f}"
)
print(
    "     -> the bandit out-rotated me by "
    f"{abs(opp[-1] - opp[0]) - abs(course[-1] - course[0]):+.2f} deg over the same 7.208 s"
)

hdr("K. TURN PERFORMANCE vs THE AIRFRAME: is this 'the corner'?")
v = np.stack([k["vel_e_ms"], k["vel_n_ms"], k["vel_u_ms"]], 1)
vn = v / np.linalg.norm(v, axis=1, keepdims=True)
om = np.r_[0.0, np.degrees(np.arccos(np.clip((vn[1:] * vn[:-1]).sum(1), -1, 1))) / DT]
nz = k["nz_g"]
vt = k["vt_ms"]
print(
    f"   omega_3d max over trace {om.max():.2f} deg/s at step {int(om.argmax())} (nz {nz[int(om.argmax())]:.2f} g, vt {vt[int(om.argmax())]:.1f})"
)
print("   ideal level-turn rate for the MEASURED nz at the MEASURED speed, g*sqrt(nz^2-1)/V:")
for s in (48, 72, 88, 95, 100, 120, 156):
    ideal = 9.81 * np.sqrt(max(nz[s] ** 2 - 1, 0)) / vt[s] * 180 / np.pi
    print(
        f"     step {s:3d}  nz {nz[s]:5.2f} g  vt {vt[s]:6.1f} m/s  ideal {ideal:6.2f} deg/s  measured omega {om[s]:6.2f} deg/s"
    )
print("\n   CORNER TEST: turn rate is maximised at the LOWEST speed that still makes the g limit.")
print(
    f"     vt monotone increasing? up on {(np.diff(vt) > 0).sum()}/{n - 1} steps; {vt[0]:.1f} -> {vt[-1]:.1f} m/s"
)
print(
    f"     alpha max {k['alpha_deg'].max():.2f} deg (measured F-16 usable ceiling ~16.7 deg) -> "
    f"{'AT' if k['alpha_deg'].max() > 15.5 else 'NOT AT'} the lift limit"
)
print(
    f"     elev at full aft (<=-0.999): {(k['elev'] <= -0.999).sum()}/{n} steps = {(k['elev'] <= -0.999).sum() * DT:.3f} s"
)
i = np.where(k["elev"] <= -0.999)[0]
print(
    f"       run steps {i.min()}-{i.max()} (t {k['t_s'][i.min()]:.3f}-{k['t_s'][i.max()]:.3f}), contiguous {len(i) == i.max() - i.min() + 1}"
)
print(
    f"     nz over the ROLL-IN steps 0-24: mean {nz[:25].mean():+.2f} g, min {nz[:25].min():+.2f}, "
    f"steps with nz<1 (unloaded/negative): {(nz[:25] < 1.0).sum()}/25"
)
print(
    f"     -> the 'commitment' phase is flown at mean {nz[:25].mean():+.2f} g, i.e. UNLOADED, for "
    f"{(nz[:25] < 1.0).sum() * DT:.3f} s"
)

hdr("L. THE MANEUVER AS A WHOLE ON SEED 28 — is the executed prefix part of any coherent BFM?")
oc = np.degrees(np.unwrap(np.arctan2(q["vel_e_ms"], q["vel_n_ms"])))
print(
    f"   seed28 course over 576 steps: {oc[0]:+.1f} -> {oc[-1]:+.1f} deg, total signed change {oc[-1] - oc[0]:+.1f} deg"
)
sgn = np.sign(np.r_[0.0, np.diff(oc)])
flips = int((np.diff(np.sign(np.r_[0.0, np.diff(oc)][1:])) != 0).sum())
print(f"   reversals of turn direction (sign changes of course rate): {flips}")
print(
    f"   |phi| > 90 deg on {(np.abs(q['phi_deg']) > 90).sum()}/576 steps; phi sign flips: "
    f"{int((np.diff(np.sign(q['phi_deg'])) != 0).sum())}"
)
print(
    f"   altitude: {q['alt_m'][0]:.0f} -> {q['alt_m'][-1]:.0f} m ({q['alt_m'][-1] - q['alt_m'][0]:+.0f} m) in 24.0 s"
    f"  => mean {(q['alt_m'][-1] - q['alt_m'][0]) / 24.0:+.1f} m/s"
)
print(
    f"   Es: {q['es_m'][0]:.0f} -> {q['es_m'][-1]:.0f} ({q['es_m'][-1] - q['es_m'][0]:+.0f} m); "
    f"es_rel {q['es_rel_m'][0]:.0f} -> {q['es_rel_m'][-1]:.0f}"
)
print(
    f"   rng: min {q['rng_m'].min():.1f} (step {int(q['rng_m'].argmin())}), final {q['rng_m'][-1]:.0f} m"
)
print(
    f"   own health {q['health'][0]:.0f} -> {q['health'][-1]:.0f}; opp health {q['opp_health'][0]:.0f} -> {q['opp_health'][-1]:.0f}"
)
print(
    f"   rounds fired {q['ammo'][0] - q['ammo'][-1]:.0f}; HP dealt {100 - q['opp_health'][-1]:.0f}"
)
print(
    f"   post-merge (steps 173-575): mean aspect {q['aspect_deg'][173:].mean():.1f} deg, "
    f"steps with rng<600 m: {(q['rng_m'][173:] < 600).sum()}, steps with lay_miss<15 m: {(q['lay_miss_m'][173:] < 15).sum()}"
)

hdr("M. IN-SAMPLE STATUS OF THE OUTCOME (for the 'cashed as a gun snapshot' half)")
print("   From runs/d3b/replay_report.txt SEED IDENTIFICATION: kill on 5 of the 8 SEARCH seeds")
print(
    "   (11, 45, 62, 113, 130); no kill on 28, 79, 96 despite 96-98 HP dealt by the SAME actions."
)
print("   The 8 seeds are the seeds the genome was OPTIMISED on -> existence, never a hit rate.")
print(
    "   Kinematics steps 0-172 are seed-invariant (max|diff| 0.000e+00 across all columns checked),"
)
print("   so the KINEMATIC half of the claim is NOT seed-contingent; only the KILL is.")
