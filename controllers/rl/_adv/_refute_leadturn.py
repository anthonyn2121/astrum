"""Independent refutation pass on the 'nose-low slice LEAD TURN' claim.

Every number here is recomputed from the CSVs from scratch (no import of the claim's script).
"""

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
print(f"kill rows {n}   nokill rows {q['_n']}   dt {DT:.6f} s")

hdr("A. VERBATIM CHECK OF EVERY CITED NUMBER")
ail = k["ail"]
run = np.where(ail >= 0.999999)[0]
# contiguous leading run
lead = 0
while lead < n and ail[lead] >= 0.999999:
    lead += 1
print(
    f"A1 ail==+1.0 contiguous leading run: steps 0-{lead - 1} (t 0.000-{k['t_s'][lead - 1]:.4f} s)"
)
print(
    f"   ail[{lead - 1}]={ail[lead - 1]:.6f}  ail[{lead}]={ail[lead]:.6f}  ail[{lead + 1}]={ail[lead + 1]:.6f}"
)
print(
    f"   all steps with ail>=0.999: {np.where(ail >= 0.999)[0].min()}-{np.where(ail >= 0.999)[0].max()}, count {(ail >= 0.999).sum()}"
)
print(
    f"   CLAIM: 'ail=+1.000 held steps 0-23 (t 0.000-0.958 s)'  -> steps 0-{lead - 1}, t {k['t_s'][lead - 1]:.4f}"
)

phi = k["phi_deg"]
print(f"\nA2 phi[24]={phi[24]:.4f} deg (t {k['t_s'][24]:.4f}); CLAIM +88.88 at step 24 t 1.000")
print(f"   phi[32]={phi[32]:.4f} deg (t {k['t_s'][32]:.4f}); CLAIM +119.96 at step 32 t 1.333")
print(
    f"   phi max over trace = {phi.max():.4f} at step {int(phi.argmax())} (t {k['t_s'][int(phi.argmax())]:.4f})"
)

rr = k["roll_rate_dps"]
dphi = np.r_[0.0, np.diff(phi) / DT]
print("\nA3 CLAIM 'peak phi_rate 131.1 deg/s at step 24'")
print(
    f"   diff(phi)/dt : max {dphi.max():.4f} at step {int(dphi.argmax())}; value at step 24 = {dphi[24]:.4f}"
)
print(
    f"   roll_rate_dps: max {rr.max():.4f} at step {int(rr.argmax())}; value at step 24 = {rr[24]:.4f}"
)
print(
    f"   -> the cited 131.1 comes from {'diff(phi)/dt' if abs(dphi.max() - 131.1) < 0.5 else 'roll_rate_dps' if abs(rr.max() - 131.1) < 0.5 else 'NEITHER'}"
)

rng = k["rng_m"]
print(
    f"\nA4 min rng = {rng.min():.4f} m at step {int(rng.argmin())} (t {k['t_s'][int(rng.argmin())]:.4f} s)"
)
print(
    f"   CLAIM: 'min rng 8.36 m at step 172, t 7.167 s'; rng[172]={rng[172]:.4f}, t[172]={k['t_s'][172]:.4f}"
)
print(
    f"   rng[32]={rng[32]:.2f} -> rng[32]-rng[172] = {rng[32] - rng[172]:.2f} m  (CLAIM '2553 m before the pass')"
)
print(f"   t[172]-t[32] = {k['t_s'][172] - k['t_s'][32]:.4f} s  (CLAIM '5.83 s')")

az = k["ata_az_deg"]
print(f"\nA5 ata_az[0] = {az[0]:.4f} deg  (CLAIM +30.13)")
zc = [i for i in range(1, n) if az[i - 1] > 0.0 >= az[i]]
print(f"   zero crossings of ata_az (pos->non-pos): {zc}")
if zc:
    c = zc[0]
    print(
        f"   first crossing step {c}, t {k['t_s'][c]:.4f}, rng {rng[c]:.2f} m  (CLAIM step 30, t 1.250, rng 2585)"
    )
w = slice(36, 109)
print(
    f"   ata_az over steps 36-108: min {az[w].min():.4f}  max {az[w].max():.4f}  (CLAIM 'held -3.10 to -11.96')"
)
print(
    f"   any positive in 36-108? {(az[w] > 0).sum()} steps;  duration {(108 - 36 + 1) * DT:.4f} s (CLAIM ~3.0 s)"
)

ata = k["ata_deg"]
print(
    f"\nA6 ata[0]={ata[0]:.4f} (CLAIM 32.07)  ata[84]={ata[84]:.4f} at t {k['t_s'][84]:.4f} (CLAIM 10.14 at t 3.500)"
)
print(f"   ata[156]={ata[156]:.4f} (CLAIM 2.52 at first trigger)")
fire = k["fire"]
fs = np.where(fire > 0)[0]
print(
    f"   fire>0 steps {fs.min()}-{fs.max()} count {len(fs)}; contiguous={len(fs) == fs.max() - fs.min() + 1}"
)
print(
    f"   ata at first fire step {fs.min()} = {ata[fs.min()]:.4f}; rng {rng[fs.min()]:.2f}; aspect {k['aspect_deg'][fs.min()]:.2f}"
)

hdr("B. IS THE NAME RIGHT? aspect / turning-room / conversion ledger")
asp = k["aspect_deg"]
cpa = k["cpa_miss_m"]
print("B1 aspect_deg (near 180 = head-on, near 0 = on his tail):")
for s in (0, 24, 48, 72, 96, 120, 144, 156, 172):
    print(
        f"    step {s:3d} t {k['t_s'][s]:6.3f}  aspect {asp[s]:7.2f}  ata {ata[s]:6.2f}  ata_az {az[s]:+7.2f}  rng {rng[s]:7.1f}  cpa_miss {cpa[s]:8.1f}  hca {k['hca_deg'][s]:8.2f}"
    )
print(
    f"   aspect: min {asp.min():.2f} at step {int(asp.argmin())}; max {asp.max():.2f} at step {int(asp.argmax())}"
)
print(
    f"   monotone INCREASING toward 180 (head-on)? up on {(np.diff(asp) > 0).sum()}/{n - 1} steps"
)
print(f"\nB2 cpa_miss (constant-velocity predicted miss): {cpa[0]:.1f} -> {cpa[-1]:.1f} m")
print(f"   steps where cpa_miss INCREASED (turning room BUILT): {(np.diff(cpa) > 0).sum()}/{n - 1}")
inc = np.where(np.diff(cpa) > 0)[0]
if len(inc):
    print(
        f"   those steps: {list(inc[:30])} ... max single-step increase {np.diff(cpa).max():.3f} m"
    )
print(
    f"   cpa_miss max {cpa.max():.1f} at step {int(cpa.argmax())}; min {cpa.min():.1f} at step {int(cpa.argmin())}"
)

hdr("C. 'AT THE AIRFRAME CORNER' / 'MAX-PERFORMANCE' — priced in altitude and turn RATE")
nz = k["nz_g"]
alt = k["alt_m"]
vt = k["vt_ms"]
es = k["es_m"]
alpha = k["alpha_deg"]
elev = k["elev"]
# ground track and turn rate
course = np.degrees(np.unwrap(np.arctan2(k["vel_e_ms"], k["vel_n_ms"])))
crate = np.r_[0.0, np.diff(course) / DT]
# total turn rate of the velocity vector (3-D)
v = np.stack([k["vel_e_ms"], k["vel_n_ms"], k["vel_u_ms"]], 1)
vn = v / np.linalg.norm(v, axis=1, keepdims=True)
dots = np.clip((vn[1:] * vn[:-1]).sum(1), -1, 1)
omega = np.r_[0.0, np.degrees(np.arccos(dots)) / DT]
print(
    f"C1 nz: max {nz.max():.3f} g at step {int(nz.argmax())} (t {k['t_s'][int(nz.argmax())]:.3f}, seg {int(k['seg_idx'][int(nz.argmax())])})"
)
print(
    f"   nz over seg0 steps 0-95:  min {nz[:96].min():.2f} max {nz[:96].max():.2f} mean {nz[:96].mean():.2f}"
)
print(
    f"   nz over seg1 steps 96-172: min {nz[96:].min():.2f} max {nz[96:].max():.2f} mean {nz[96:].mean():.2f}"
)
print(f"   steps with nz>3 : {(nz > 3).sum()}/{n} = {(nz > 3).sum() * DT:.3f} s")
print(
    f"   elev: full aft (<=-0.999) on {(elev <= -0.999).sum()} steps; range {elev.min():.4f}..{elev.max():.4f}"
)
print(
    f"   alpha max {alpha.max():.2f} deg at step {int(alpha.argmax())} (program's measured airframe ceiling ~16.7 deg)"
)
print("\nC2 TURN RATE — is this a max-rate turn or an accelerating dive?")
for a, b in [(0, 24), (24, 48), (48, 72), (72, 95), (96, 120), (120, 144), (144, 172)]:
    print(
        f"    steps {a:3d}-{b:3d} t {k['t_s'][a]:5.2f}-{k['t_s'][b]:5.2f}  omega_3d {omega[a + 1 : b + 1].mean():6.2f} deg/s"
        f"  course_rate {crate[a + 1 : b + 1].mean():+7.2f} deg/s  nz {nz[a : b + 1].mean():5.2f} g"
        f"  vt {vt[a]:6.1f}->{vt[b]:6.1f}  alt {alt[a]:7.1f}->{alt[b]:7.1f} ({alt[b] - alt[a]:+7.1f})"
        f"  gamma {k['gamma_deg'][a]:+6.1f}->{k['gamma_deg'][b]:+6.1f}"
    )
print(
    f"\nC3 PRICE THE HOLD: total dalt {alt[-1] - alt[0]:+.1f} m over {n * DT:.3f} s; Es {es[0]:.1f}->{es[-1]:.1f} ({es[-1] - es[0]:+.1f})"
)
print(
    f"   vt {vt[0]:.1f} -> {vt[-1]:.1f} m/s (+{vt[-1] - vt[0]:.1f}); descent rate at last step {k['vel_u_ms'][-1]:.1f} m/s"
)
print(
    f"   omega_3d mean over the 'max-performance' seg0 = {omega[1:96].mean():.2f} deg/s; over seg1 = {omega[96:].mean():.2f} deg/s"
)
print(f"   turn rate at nz peak step {int(nz.argmax())} = {omega[int(nz.argmax())]:.2f} deg/s")

hdr("D. 'STABILISED TRACKING RUN-IN' — is seg 1 stabilised, and is it tracking?")
print(
    f"D1 seg1 (steps 96-172): cmd bank {k['cmd_bank_deg'][100]:.2f} load {k['cmd_load'][100]:.4f} thr {k['cmd_thr'][100]:.4f}"
)
print(
    f"   phi {phi[96]:.2f} -> {phi[172]:.2f} deg; nz {nz[96:].min():.2f}-{nz[96:].max():.2f} g; gamma {k['gamma_deg'][96]:+.1f} -> {k['gamma_deg'][172]:+.1f}"
)
print(
    f"   alt {alt[96]:.1f} -> {alt[172]:.1f} ({alt[172] - alt[96]:+.1f} m); vt {vt[96]:.1f} -> {vt[172]:.1f}"
)
print(f"   omega_3d over 96-172 mean {omega[97:].mean():.2f} deg/s, max {omega[97:].max():.2f}")
print(f"   ail over 96-172: min {ail[96:].min():.4f} max {ail[96:].max():.4f}")
print(f"   lay_miss_m at fire steps: {[round(float(k['lay_miss_m'][s]), 2) for s in fs]}")
print(
    f"   lay_miss_m min over trace {k['lay_miss_m'].min():.2f} at step {int(k['lay_miss_m'].argmin())}"
)
print(f"D2 is any of it CLOSED-LOOP on the bandit? rud identically 0: {np.all(k['rud'] == 0.0)}")
print(
    "   ail is a pure function of phi: ail = clip(KROLL*(phi_tgt - phi)); check monotone vs phi within seg0:"
)
print(f"     corr(ail[0:96], -(phi[0:96])) = {np.corrcoef(ail[:96], -phi[:96])[0, 1]:.6f}")

hdr("E. DOES IT SURVIVE ON THE NON-KILL SEED? (seed 28, 576 steps)")
m = 173
cols = [
    "ail",
    "elev",
    "thr_flown",
    "phi_deg",
    "nz_g",
    "ata_az_deg",
    "ata_deg",
    "aspect_deg",
    "rng_m",
    "cpa_miss_m",
    "alt_m",
    "lay_miss_m",
]
for c in cols:
    dd = np.abs(k[c][:m] - q[c][:m]).max()
    print(f"   max|kill-nokill| over steps 0-172 for {c:12s} = {dd:.3e}")
print(
    f"   seed28 fire steps in 0-172: {np.where(q['fire'][:m] > 0)[0].min()}-{np.where(q['fire'][:m] > 0)[0].max()}"
)
print(
    f"   seed28 opp_health at step 172 = {q['opp_health'][172]:.1f}  vs kill seed {k['opp_health'][172]:.1f}"
)
print(
    f"   seed28 min rng over whole 576 = {q['rng_m'].min():.2f} at step {int(q['rng_m'].argmin())}"
)
print(
    "\nE1 WHAT THE 'MANEUVER' DOES AFTER THE MERGE ON SEED 28 (segs 2-5, never seen on the kill):"
)
for a, b in [(173, 191), (192, 287), (288, 383), (384, 479), (480, 575)]:
    seg = int(q["seg_idx"][b])
    print(
        f"    seg {seg} steps {a:3d}-{b:3d} t {q['t_s'][a]:6.2f}-{q['t_s'][b]:6.2f}  cmdbank {q['cmd_bank_deg'][b]:+7.2f} load {q['cmd_load'][b]:.3f} thr {q['cmd_thr'][b]:.3f}"
        f"  phi {q['phi_deg'][a]:+7.1f}->{q['phi_deg'][b]:+7.1f}  nz {q['nz_g'][a : b + 1].min():5.2f}-{q['nz_g'][a : b + 1].max():5.2f}"
        f"  alt {q['alt_m'][a]:7.1f}->{q['alt_m'][b]:7.1f} ({q['alt_m'][b] - q['alt_m'][a]:+8.1f})"
        f"  rng {q['rng_m'][a]:7.0f}->{q['rng_m'][b]:7.0f}  aspect {q['aspect_deg'][b]:7.2f}"
    )
print(
    f"   seed28 total dalt {q['alt_m'][-1] - q['alt_m'][0]:+.1f} m; final alt {q['alt_m'][-1]:.1f} m; final vt {q['vt_ms'][-1]:.1f}"
)
print(f"   seed28 fire steps AFTER 172: {list(np.where(q['fire'][173:] > 0)[0] + 173)[:40]}")
print(
    f"   seed28 rounds/hp: final opp_health {q['opp_health'][-1]:.1f}, own ammo {q['ammo'][-1]:.0f} (from {q['ammo'][0]:.0f})"
)
print(
    f"   seed28 max nz after step 172: {q['nz_g'][173:].max():.2f}; steps nz>3 after 172: {(q['nz_g'][173:] > 3).sum()}"
)

hdr("F. THE 'PASS' — does a pass even occur? and what is the episode's end condition")
print(
    f"F1 kill trace: last row step 172, rng {rng[172]:.2f} m, closure {k['closure_ms'][172]:+.1f} m/s"
)
print(
    f"   closure sign over the trace: negative (opening) on {(k['closure_ms'] < 0).sum()}/{n} steps"
)
print(
    f"   min |closure| {np.abs(k['closure_ms']).min():.2f} at step {int(np.abs(k['closure_ms']).argmin())}"
)
print(f"   -> range decreases monotonically? {(np.diff(rng) < 0).sum()}/{n - 1} steps decreasing")
print(
    f"F2 on seed 28 the pass DOES happen: rng min {q['rng_m'].min():.2f} at step {int(q['rng_m'].argmin())} (t {q['t_s'][int(q['rng_m'].argmin())]:.3f}),"
    f" then rng at 573 = {q['rng_m'][573]:.0f}"
)
i = int(q["rng_m"].argmin())
print(
    f"   post-pass on seed 28: steps {i}-{min(i + 60, 575)}: aspect {q['aspect_deg'][i]:.1f} -> {q['aspect_deg'][min(i + 60, 575)]:.1f},"
    f" ata {q['ata_deg'][i]:.1f} -> {q['ata_deg'][min(i + 60, 575)]:.1f}, alt {q['alt_m'][i]:.0f} -> {q['alt_m'][min(i + 60, 575)]:.0f}"
)

hdr("G. LOADED HOLD DURATIONS (the program's sanctioned statistic), priced in altitude")


def holds(mask, tarr, altarr, minlen=1):
    out = []
    i = 0
    N = len(mask)
    while i < N:
        if mask[i]:
            j = i
            while j + 1 < N and mask[j + 1]:
                j += 1
            if j - i + 1 >= minlen:
                out.append((i, j, (j - i + 1) * DT, altarr[j] - altarr[i]))
            i = j + 1
        else:
            i += 1
    return out


for name, mask in [
    ("|phi|>45 AND nz>3 (kill)", (np.abs(phi) > 45) & (nz > 3.0)),
    ("|phi|>90 AND nz>3 (kill)", (np.abs(phi) > 90) & (nz > 3.0)),
    ("|chi|>60 AND nz>3 (kill)", (np.abs(k["chi_deg"]) > 60) & (nz > 3.0)),
    ("|chi|>30 AND nz>3 (kill)", (np.abs(k["chi_deg"]) > 30) & (nz > 3.0)),
]:
    hh = holds(mask, k["t_s"], alt)
    tot = sum(h[2] for h in hh)
    print(
        f"   {name:28s} runs {len(hh)}  total {tot:6.3f} s  longest "
        f"{max([h[2] for h in hh], default=0.0):6.3f} s  "
        f"dalt over longest {(max(hh, key=lambda h: h[2])[3] if hh else 0.0):+8.1f} m"
    )
for name, mask in [
    ("|phi|>90 AND nz>3 (seed28)", (np.abs(q["phi_deg"]) > 90) & (q["nz_g"] > 3.0)),
    ("|chi|>60 AND nz>3 (seed28)", (np.abs(q["chi_deg"]) > 60) & (q["nz_g"] > 3.0)),
]:
    hh = holds(mask, q["t_s"], q["alt_m"])
    tot = sum(h[2] for h in hh)
    lg = max(hh, key=lambda h: h[2]) if hh else None
    print(
        f"   {name:28s} runs {len(hh)}  total {tot:6.3f} s  longest "
        f"{(lg[2] if lg else 0.0):6.3f} s  dalt over longest {(lg[3] if lg else 0.0):+8.1f} m"
        f"  steps {(lg[0] if lg else -1)}-{(lg[1] if lg else -1)}"
    )

hdr("H. ONE-CIRCLE / LEAD-TURN GEOMETRY: does the turn go INTO or ACROSS the offset?")
# lateral offset: component of the bandit's position perpendicular to MY initial velocity
p_me = np.stack([k["pos_e_m"], k["pos_n_m"], k["pos_u_m"]], 1)
p_op = np.stack([k["opp_pos_e_m"], k["opp_pos_n_m"], k["opp_pos_u_m"]], 1)
r = p_op - p_me
v0 = v[0] / np.linalg.norm(v[0])
lat0 = (
    r
    @ np.cross(v0, np.array([0.0, 0.0, 1.0]))
    / np.linalg.norm(np.cross(v0, np.array([0.0, 0.0, 1.0])))
)
print("H1 horizontal lateral offset of the bandit vs MY INITIAL velocity axis:")
for s in (0, 24, 48, 72, 96, 120, 144, 172):
    print(
        f"    step {s:3d} t {k['t_s'][s]:6.3f}  lateral {lat0[s]:+9.1f} m  along {r[s] @ v0:+9.1f} m  rng {rng[s]:8.1f}"
    )
print(
    f"   |lateral| at step 0 = {abs(lat0[0]):.1f} m (cell nominal 1500 m); at step 172 = {abs(lat0[172]):.1f} m"
)
print(
    f"H2 my heading vs bandit heading (deg): psi {k['psi_deg'][0]:.2f} -> {k['psi_deg'][172]:.2f};"
    f" opp_psi {k['opp_psi_deg'][0]:.2f} -> {k['opp_psi_deg'][172]:.2f}"
)
print(f"   hca {k['hca_deg'][0]:.2f} -> {k['hca_deg'][172]:.2f} (180 = reciprocal)")
print(f"   my unwrapped course change {course[-1] - course[0]:+.2f} deg over {n * DT:.2f} s")
opp_unwrap = np.degrees(np.unwrap(np.radians(k["opp_psi_deg"])))
print(
    f"   bandit unwrapped psi change {opp_unwrap[-1] - opp_unwrap[0]:+.2f} deg -> bandit "
    f"{'DID' if abs(opp_unwrap[-1] - opp_unwrap[0]) > 10 else 'did NOT'} turn"
)
print(
    f"   opp nz over the trace: min {k['opp_nz_g'].min():.2f} max {k['opp_nz_g'].max():.2f} mean {k['opp_nz_g'].mean():.2f}"
)
