"""Independent refutation check of the 'pull was at the airframe corner' claim.

Pure stdlib. Reads the two per-step dumps directly; recomputes every cited number from the
columns with no reuse of _adv_bfm_case*.py code.
"""

from __future__ import annotations

import csv
import math
import statistics as st

HOST = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"
KILL = f"{HOST}/replay_kill_seed62.csv"
NOKILL = f"{HOST}/replay_nokill_seed28.csv"
DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    d = {k: [float(r[k]) for r in rows] for k in rows[0]}
    d["_n"] = len(rows)
    return d


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def course(d, k):
    return math.degrees(math.atan2(d["vel_e_ms"][k], d["vel_n_ms"][k]))


def unwrap(seq):
    out = [seq[0]]
    for v in seq[1:]:
        p = out[-1]
        while v - p > 180.0:
            v -= 360.0
        while v - p < -180.0:
            v += 360.0
        out.append(v)
    return out


def vel_turn(d, a, b):
    """Total |dv_hat| rotation, and its split into course (horizontal) and gamma (vertical)."""
    tot = 0.0
    for k in range(a, b):
        v0 = (d["vel_e_ms"][k], d["vel_n_ms"][k], d["vel_u_ms"][k])
        v1 = (d["vel_e_ms"][k + 1], d["vel_n_ms"][k + 1], d["vel_u_ms"][k + 1])
        n0 = math.sqrt(sum(x * x for x in v0))
        n1 = math.sqrt(sum(x * x for x in v1))
        dot = sum(x * y for x, y in zip(v0, v1)) / (n0 * n1)
        tot += math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    cs = unwrap([course(d, k) for k in range(a, b + 1)])
    return tot, cs[-1] - cs[0], d["gamma_deg"][b] - d["gamma_deg"][a]


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


k = load(KILL)
n = k["_n"]
print(f"KILL seed62: {n} rows, t 0.000-{k['t_s'][-1]:.3f} s")

hdr("C1  Where does the 'hold steps 28-115' come from? nz>3 runs on the kill trace")
for i, j in runs([v > 3.0 for v in k["nz_g"]]):
    print(
        f"  nz>3 run steps {i:3d}-{j:3d}  t {k['t_s'][i]:6.3f}-{k['t_s'][j]:6.3f}  "
        f"count {j - i + 1:3d}  elapsed {(k['t_s'][j] - k['t_s'][i]):5.3f} s  count*DT {(j - i + 1) * DT:5.3f} s"
    )

hdr("C2  'full aft stick for 2.500 s inside the hold' — recount elev <= -0.999")
a, b = 28, 115
mask = [k["elev"][i] <= -0.999 for i in range(n)]
inhold = [i for i in range(a, b + 1) if mask[i]]
print(f"  hold steps {a}-{b}: {b - a + 1} steps, elapsed {k['t_s'][b] - k['t_s'][a]:.3f} s")
print(
    f"  elev<=-0.999 inside hold: {len(inhold)}/{b - a + 1} steps; count*DT = {len(inhold) * DT:.3f} s; "
    f"ELAPSED span {k['t_s'][inhold[-1]] - k['t_s'][inhold[0]]:.3f} s "
    f"(steps {inhold[0]}-{inhold[-1]}, t {k['t_s'][inhold[0]]:.3f}-{k['t_s'][inhold[-1]]:.3f})"
)
allidx = [i for i in range(n) if mask[i]]
print(
    f"  elev<=-0.999 over WHOLE trace: {len(allidx)} steps, runs {runs(mask)}  "
    f"contiguous={len(allidx) == allidx[-1] - allidx[0] + 1}"
)
print(f"  elev min over hold {min(k['elev'][a : b + 1]):.4f}  max {max(k['elev'][a : b + 1]):.4f}")
seg1 = [i for i in range(96, b + 1)]
print(
    f"  SEGMENT SPLIT of the hold: steps {a}-95 = seg0 ({96 - a} steps, cmd_load "
    f"{k['cmd_load'][50]:.4f}); steps 96-{b} = seg1 ({len(seg1)} steps, cmd_load {k['cmd_load'][100]:.4f})"
)
print(
    f"    full-aft steps in seg0 part: {sum(1 for i in inhold if i < 96)}/{96 - a}   "
    f"in seg1 part: {sum(1 for i in inhold if i >= 96)}/{len(seg1)}"
)
print(
    f"    elev in seg1 part of hold: min {min(k['elev'][96 : b + 1]):.4f} max {max(k['elev'][96 : b + 1]):.4f}"
)

hdr("C3  nz: is 5.73 g 'sustained', and what is the true max the airframe gave?")
band = k["nz_g"][44:97]
print(
    f"  steps 44-96 (the cited 'sustained' band): min {min(band):.3f} max {max(band):.3f} "
    f"median {st.median(band):.3f} mean {st.fmean(band):.3f}"
)
hold_nz = k["nz_g"][a : b + 1]
mx = max(hold_nz)
imx = a + hold_nz.index(mx)
print(f"  hold {a}-{b}: nz max {mx:.3f} g at step {imx} (t {k['t_s'][imx]:.3f} s)")
print(f"  WHOLE trace nz max {max(k['nz_g']):.3f} g at step {k['nz_g'].index(max(k['nz_g']))}")
for i in range(96, 112):
    print(
        f"    step {i:3d} t {k['t_s'][i]:6.3f} seg {int(k['seg_idx'][i])} cmd_load {k['cmd_load'][i]:.4f} "
        f"elev {k['elev'][i]:+.4f} ail {k['ail'][i]:+.4f} nz {k['nz_g'][i]:5.2f} alpha "
        f"{k['alpha_deg'][i]:5.2f} phi {k['phi_deg'][i]:7.2f} q {k['pitch_rate_dps'][i]:+6.1f}"
    )
over = [i for i in range(n) if k["nz_g"][i] > 5.8]
print(f"  steps with nz > 5.8 g: {len(over)} -> {over}")
print(
    f"  nz at the full-aft steps only: min {min(k['nz_g'][i] for i in allidx):.3f} "
    f"max {max(k['nz_g'][i] for i in allidx):.3f} median {st.median([k['nz_g'][i] for i in allidx]):.3f}"
)

hdr("C4  alpha: 12.28 deg vs the ~16.7 deg ceiling")
al = k["alpha_deg"][a : b + 1]
print(
    f"  hold alpha max {max(al):.3f} deg at step {a + al.index(max(al))} "
    f"(t {k['t_s'][a + al.index(max(al))]:.3f} s); whole-trace max {max(k['alpha_deg']):.3f}"
)
print(f"  alpha at nz peak (step {imx}) = {k['alpha_deg'][imx]:.3f} deg")
print(
    f"  margin to 16.7 deg ceiling: {16.7 - max(al):.3f} deg = {100 * (1 - max(al) / 16.7):.1f}% unused"
)

hdr("C5  the turn: 51.5 deg in 3.625 s, and how much of it is HEADING vs DIVE")
tot, dcourse, dgam = vel_turn(k, a, b)
print(
    f"  |dv_hat| total {tot:.2f} deg over steps {a}-{b}; elapsed {k['t_s'][b] - k['t_s'][a]:.3f} s "
    f"=> {tot / (k['t_s'][b] - k['t_s'][a]):.2f} deg/s"
)
print(
    f"  signed COURSE change {dcourse:+.2f} deg;  gamma change {dgam:+.2f} deg "
    f"(gamma {k['gamma_deg'][a]:+.2f} -> {k['gamma_deg'][b]:+.2f})"
)
print(
    f"  alt {k['alt_m'][a]:.1f} -> {k['alt_m'][b]:.1f} ({k['alt_m'][b] - k['alt_m'][a]:+.1f} m)  "
    f"vt {k['vt_ms'][a]:.1f} -> {k['vt_ms'][b]:.1f} ({k['vt_ms'][b] - k['vt_ms'][a]:+.1f})  "
    f"Es {k['es_m'][a]:.1f} -> {k['es_m'][b]:.1f} ({k['es_m'][b] - k['es_m'][a]:+.1f} m)"
)
print(
    f"  mean Ps over hold = {(k['es_m'][b] - k['es_m'][a]) / (k['t_s'][b] - k['t_s'][a]):+.2f} m/s"
)
print(
    f"  mean vertical speed = {(k['alt_m'][b] - k['alt_m'][a]) / (k['t_s'][b] - k['t_s'][a]):+.1f} m/s"
)
rad = []
for i in range(a, b):
    v0 = (k["vel_e_ms"][i], k["vel_n_ms"][i], k["vel_u_ms"][i])
    v1 = (k["vel_e_ms"][i + 1], k["vel_n_ms"][i + 1], k["vel_u_ms"][i + 1])
    n0 = math.sqrt(sum(x * x for x in v0))
    n1 = math.sqrt(sum(x * x for x in v1))
    dot = sum(x * y for x, y in zip(v0, v1)) / (n0 * n1)
    w = math.acos(max(-1.0, min(1.0, dot))) / DT
    rad.append(n0 / max(w, 1e-9))
print(f"  median instantaneous radius vt/omega over the hold = {st.median(rad):.0f} m")
print(
    f"  phi over hold: min {min(k['phi_deg'][a : b + 1]):.2f} max {max(k['phi_deg'][a : b + 1]):.2f} deg "
    f"(|phi|>90 on {sum(1 for i in range(a, b + 1) if abs(k['phi_deg'][i]) > 90)}/{b - a + 1} steps)"
)

hdr("C6  ideal-turn cross-check: what turn rate SHOULD 5.73 g give, and what did it give?")
for i in (44, 60, 80, 96, 100):
    nz = k["nz_g"][i]
    v = k["vt_ms"][i]
    ideal = math.degrees(9.80665 * math.sqrt(max(nz * nz - 1.0, 0.0)) / v)
    print(
        f"  step {i:3d} nz {nz:5.2f} vt {v:6.1f} -> ideal max turn rate {ideal:5.2f} deg/s "
        f"(radius {v * v / (9.80665 * math.sqrt(max(nz * nz - 1, 1e-9))):6.0f} m)"
    )

hdr("C7  DOES IT SURVIVE ON THE NON-KILL SEED? seed 28, all 6 segments")
q = load(NOKILL)
m = q["_n"]
print(f"  seed28: {m} rows, t 0.000-{q['t_s'][-1]:.3f} s")
dmax = max(
    abs(q[c][i] - k[c][i])
    for c in ("nz_g", "alpha_deg", "elev", "phi_deg", "alt_m")
    for i in range(n)
)
print(f"  max |seed28 - seed62| over steps 0-{n - 1} on (nz,alpha,elev,phi,alt) = {dmax:.3e}")
print(
    f"  seed28 nz max {max(q['nz_g']):.3f} g at step {q['nz_g'].index(max(q['nz_g']))} "
    f"(t {q['t_s'][q['nz_g'].index(max(q['nz_g']))]:.3f} s)"
)
print(
    f"  seed28 alpha max {max(q['alpha_deg']):.3f} deg at step {q['alpha_deg'].index(max(q['alpha_deg']))}"
)
fa = [i for i in range(m) if q["elev"][i] <= -0.999]
print(
    f"  seed28 elev<=-0.999 on {len(fa)} steps; runs {runs([q['elev'][i] <= -0.999 for i in range(m)])}"
)
if fa:
    print(
        f"    nz on those steps: min {min(q['nz_g'][i] for i in fa):.3f} "
        f"max {max(q['nz_g'][i] for i in fa):.3f}; alpha max {max(q['alpha_deg'][i] for i in fa):.3f}"
    )
for s in range(6):
    idx = [i for i in range(m) if int(q["seg_idx"][i]) == s]
    if not idx:
        continue
    i0, i1 = idx[0], idx[-1]
    print(
        f"  seg {s}: steps {i0:3d}-{i1:3d} cmd_load {q['cmd_load'][i0]:.4f} cmd_bank "
        f"{q['cmd_bank_deg'][i0]:+7.2f} | nz max {max(q['nz_g'][i0 : i1 + 1]):5.2f} alpha max "
        f"{max(q['alpha_deg'][i0 : i1 + 1]):5.2f} elev min {min(q['elev'][i0 : i1 + 1]):+.4f} "
        f"alt {q['alt_m'][i0]:7.1f} -> {q['alt_m'][i1]:7.1f} Es {q['es_m'][i0]:7.1f} -> {q['es_m'][i1]:7.1f}"
    )
print(
    f"  seed28 steps with nz > 5.8 g: {sum(1 for v in q['nz_g'] if v > 5.8)}/{m}; "
    f"nz > 6.5: {sum(1 for v in q['nz_g'] if v > 6.5)}; nz > 7.0: {sum(1 for v in q['nz_g'] if v > 7.0)}"
)
print(
    f"  seed28 steps with alpha > 12.28: {sum(1 for v in q['alpha_deg'] if v > 12.28)}/{m}; "
    f"alpha > 14: {sum(1 for v in q['alpha_deg'] if v > 14.0)}; alpha > 16: {sum(1 for v in q['alpha_deg'] if v > 16.0)}"
)
