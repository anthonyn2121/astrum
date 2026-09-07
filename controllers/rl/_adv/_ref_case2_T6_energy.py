"""Independent recomputation of the T6 'priced perch cash-in' energy claim.

Claim under test (pro-BFM lens):
  "The energy trade is a textbook perch cash-in, priced: 506.7 m of altitude bought 76.5 deg of
   turn and +31.8 m/s, and the energy advantage over the bandit was never surrendered."

Nothing here imports the sim. It reads only the per-step dumps, with plain stdlib, so every
number is traceable to a (file, column, row) triple.
"""

from __future__ import annotations

import csv
import math

KILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0
G = 9.80665


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: [float(r[n]) for r in rows] for n in names}
    d["_n"] = len(rows)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


def unwrap(deg):
    out = [deg[0]]
    for k in range(1, len(deg)):
        dv = deg[k] - deg[k - 1]
        while dv > 180.0:
            dv -= 360.0
        while dv < -180.0:
            dv += 360.0
        out.append(out[-1] + dv)
    return out


def vhat(d, k):
    e, n, u = d["vel_e_ms"][k], d["vel_n_ms"][k], d["vel_u_ms"][k]
    m = math.sqrt(e * e + n * n + u * u)
    return (e / m, n / m, u / m), m


def ang(a, b):
    dp = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.degrees(math.acos(dp))


def main():
    d = load(KILL)
    n = d["_n"]
    print(
        f"KILL dump: {KILL}\n  rows = {n}  (last step index {int(d['step'][-1])}, "
        f"t {d['t_s'][-1]:.3f} s)"
    )

    # ------------------------------------------------------------------ 1
    hdr("1  THE FOUR HEADLINE NUMBERS, recomputed from the columns")
    a0, a1 = d["alt_m"][0], d["alt_m"][172]
    v0, v1 = d["vt_ms"][0], d["vt_ms"][172]
    e0, e1 = d["es_m"][0], d["es_m"][172]
    print(f"  alt_m  [0] {a0:9.3f} -> [172] {a1:9.3f}   delta {a1 - a0:+9.3f} m  (claim -506.7)")
    print(f"  vt_ms  [0] {v0:9.3f} -> [172] {v1:9.3f}   delta {v1 - v0:+9.3f} m/s (claim +31.8)")
    print(f"  es_m   [0] {e0:9.3f} -> [172] {e1:9.3f}   delta {e1 - e0:+9.3f} m  (claim +283.0)")
    print(
        f"  alt_enu_m [0] {d['alt_enu_m'][0]:9.3f} -> [172] {d['alt_enu_m'][172]:9.3f}  "
        f"delta {d['alt_enu_m'][172] - d['alt_enu_m'][0]:+9.3f} m  (the other altitude column)"
    )
    # identity check: Es == alt + v^2/2g ?
    for k in (0, 86, 172):
        print(
            f"    Es identity step {k:3d}: alt {d['alt_m'][k]:8.2f} + v^2/2g "
            f"{d['vt_ms'][k] ** 2 / (2 * G):8.2f} = {d['alt_m'][k] + d['vt_ms'][k] ** 2 / (2 * G):9.2f}"
            f"   vs es_m {d['es_m'][k]:9.2f}  (diff {d['alt_m'][k] + d['vt_ms'][k] ** 2 / (2 * G) - d['es_m'][k]:+7.3f})"
        )

    # ------------------------------------------------------------------ 2
    hdr("2  '76.5 deg of TURN' -- what rotation is that, exactly?")
    course = unwrap(
        [math.degrees(math.atan2(d["vel_e_ms"][k], d["vel_n_ms"][k])) for k in range(n)]
    )
    print(
        f"  (a) NET horizontal ground-course change, unwrapped, steps 0->172:"
        f"  {course[0]:+8.2f} -> {course[-1]:+8.2f} = {course[-1] - course[0]:+8.2f} deg"
    )
    tot_h = sum(abs(course[k] - course[k - 1]) for k in range(1, n))
    print(f"  (b) CUMULATIVE |horizontal course rate| dt (path integral):        {tot_h:8.2f} deg")
    v_first, _ = vhat(d, 0)
    v_last, _ = vhat(d, 172)
    print(
        f"  (c) NET 3-D angle between velocity vectors v[0] and v[172]:        {ang(v_first, v_last):8.2f} deg"
    )
    tot3 = 0.0
    for k in range(1, n):
        p, _ = vhat(d, k - 1)
        q, _ = vhat(d, k)
        tot3 += ang(p, q)
    print(f"  (d) CUMULATIVE 3-D velocity-vector rotation (path integral):       {tot3:8.2f} deg")
    print(
        f"  gamma_deg [0] {d['gamma_deg'][0]:+7.2f} -> [172] {d['gamma_deg'][172]:+7.2f} "
        f"(min {min(d['gamma_deg']):+7.2f} at step {d['gamma_deg'].index(min(d['gamma_deg']))})"
    )
    print("  => (d) is the claim's 76.5, and the label 'cumulative velocity-vector rotation' is")
    print("     CORRECT. But it is a PATH INTEGRAL that includes the pitch-plane rotation: the net")
    print("     horizontal heading change is only (a). See part2 section A for the split.")

    # ------------------------------------------------------------------ 3
    hdr("3  PHASE LEDGER, recomputed with the claim's own windows and its own mean-Ps rule")
    ps = [0.0] + [(d["es_m"][k] - d["es_m"][k - 1]) / DT for k in range(1, n)]
    for nm, i0, i1 in (
        ("roll-in", 0, 32),
        ("slice pull", 28, 115),
        ("unload/roll-out", 108, 122),
        ("tracking run-in", 123, 172),
        ("WHOLE", 0, 172),
    ):
        seg = ps[i0 + 1 : i1 + 1]
        print(
            f"  {nm:<16} steps {i0:3d}-{i1:3d}  t {d['t_s'][i0]:6.3f}-{d['t_s'][i1]:6.3f}  "
            f"alt {d['alt_m'][i1] - d['alt_m'][i0]:+8.1f} m   vt {d['vt_ms'][i0]:6.1f}->"
            f"{d['vt_ms'][i1]:6.1f} ({d['vt_ms'][i1] - d['vt_ms'][i0]:+6.1f})   "
            f"Es {d['es_m'][i1] - d['es_m'][i0]:+8.1f} m   meanPs {sum(seg) / len(seg):+7.1f} m/s   "
            f"thr {d['thr_flown'][i0]:.3f}->{d['thr_flown'][i1]:.3f}"
        )
    print("\n  claim: pull steps 28-115 costs 151.0 m Es (meanPs -41.7) and 254.6 m alt")
    print("  claim: run-in steps 123-172 spends 202.4 m alt, GAINS 138.9 m Es (meanPs +68.0)")
    print("  windows are NOT a partition: 28-115 U 108-122 U 123-172 leaves 0-27 out and")
    print("  double-counts 108-115.")
    print(
        "  sum of the three phase Es deltas = "
        f"{(d['es_m'][115] - d['es_m'][28]) + (d['es_m'][122] - d['es_m'][108]) + (d['es_m'][172] - d['es_m'][123]):+.1f} m"
        f"  vs whole-trajectory {e1 - e0:+.1f} m"
    )

    # ------------------------------------------------------------------ 4
    hdr("4  THE SPAWN TRANSIENT the claim caveats -- how much of the ledger does it own?")
    print(
        f"  steps 0-32: alt {d['alt_m'][32] - d['alt_m'][0]:+7.1f} m  "
        f"vt {d['vt_ms'][0]:6.1f}->{d['vt_ms'][32]:6.1f} ({d['vt_ms'][32] - d['vt_ms'][0]:+5.1f})  "
        f"Es {d['es_m'][32] - d['es_m'][0]:+7.1f} m   nz {d['nz_g'][0]:.2f}->{d['nz_g'][32]:.2f} g"
    )
    print(f"  Es AFTER the transient, steps 32->172: {d['es_m'][172] - d['es_m'][32]:+7.1f} m")
    print(
        f"  vt AFTER the transient, steps 32->172: {d['vt_ms'][172] - d['vt_ms'][32]:+7.1f} m/s "
        f"(claim quotes +31.8 from step 0)"
    )
    frac = (d["es_m"][32] - d["es_m"][0]) / (e1 - e0) * 100.0
    print(f"  => the transient is {frac:.1f}% of the whole-trajectory +{e1 - e0:.1f} m Es 'gain'")
    print(
        f"  => and {(d['vt_ms'][32] - d['vt_ms'][0]) / (v1 - v0) * 100.0:.1f}% of the +"
        f"{v1 - v0:.1f} m/s, bought with only {d['alt_m'][32] - d['alt_m'][0]:+.1f} m of altitude"
    )

    # ------------------------------------------------------------------ 5
    hdr("5  DID THE ALTITUDE BUY TURN? the turn-rate ledger (this is the load-bearing test)")
    print(
        f"  {'win':<14} {'nz mean':>8} {'vt mean':>8} {'ach dpsi/dt':>12} {'ach 3D w':>10} "
        f"{'level-turn w':>13} {'lift bound w':>13}"
    )
    for nm, i0, i1 in (
        ("roll-in 0-27", 0, 27),
        ("pull 28-115", 28, 115),
        ("run-in 123-172", 123, 172),
    ):
        nzs = d["nz_g"][i0 : i1 + 1]
        vts = d["vt_ms"][i0 : i1 + 1]
        nzm = sum(nzs) / len(nzs)
        vtm = sum(vts) / len(vts)
        dur = (i1 - i0) * DT
        ach_h = (course[i1] - course[i0]) / dur
        r3 = 0.0
        for k in range(i0 + 1, i1 + 1):
            p, _ = vhat(d, k - 1)
            q, _ = vhat(d, k)
            r3 += ang(p, q)
        ach3 = r3 / dur
        lvl = math.degrees(G * math.sqrt(max(nzm * nzm - 1.0, 0.0)) / vtm)
        bnd = math.degrees(G * (nzm + 1.0) / vtm)
        print(
            f"  {nm:<14} {nzm:8.2f} {vtm:8.1f} {ach_h:12.2f} {ach3:10.2f} {lvl:13.2f} {bnd:13.2f}"
        )
    print("  level-turn w = g*sqrt(nz^2-1)/V, the ZERO-altitude-loss coordinated turn rate at the")
    print("  same measured nz and V.  lift bound w = g*(nz+1)/V, the absolute ceiling on velocity-")
    print("  vector rotation rate available from that load factor (lift + gravity, best case).")
    print("\n  the descent's OWN effect on turn rate: w scales as 1/V at fixed nz.")
    print(
        f"    pull entry step 28: vt {d['vt_ms'][28]:6.1f} m/s, nz {d['nz_g'][28]:.2f} g -> "
        f"level w {math.degrees(G * math.sqrt(max(d['nz_g'][28] ** 2 - 1, 0)) / d['vt_ms'][28]):.2f} deg/s"
    )
    print(
        f"    pull exit  step 115: vt {d['vt_ms'][115]:6.1f} m/s, nz {d['nz_g'][115]:.2f} g -> "
        f"level w {math.degrees(G * math.sqrt(max(d['nz_g'][115] ** 2 - 1, 0)) / d['vt_ms'][115]):.2f} deg/s"
    )
    print(
        f"  phi_deg over the pull: {d['phi_deg'][28]:+7.2f} -> {max(d['phi_deg'][28:116]):+7.2f} "
        f"(max) -> {d['phi_deg'][115]:+7.2f};  steps with |phi|>90 deg in 28-115: "
        f"{sum(1 for k in range(28, 116) if abs(d['phi_deg'][k]) > 90.0)}/{116 - 28}"
    )
    print(
        "  bank PAST vertical puts sin(phi) < 1 on the horizontal lift component and cos(phi) < 0"
    )
    print("  on the vertical one: the descent is a CONSEQUENCE of phi > 90, not a purchase.")
    print(
        f"    horizontal lift fraction at phi=115 deg: sin = {math.sin(math.radians(115)):.3f} "
        f"(vs 1.000 at phi=90); vertical lift fraction cos = {math.cos(math.radians(115)):+.3f}"
    )

    # ------------------------------------------------------------------ 6
    hdr("6  WHAT THE ALTITUDE ACTUALLY CORRELATES WITH: range and closure")
    for k in (0, 28, 56, 86, 115, 123, 156, 172):
        print(
            f"    step {k:3d} t {d['t_s'][k]:6.3f}  alt {d['alt_m'][k]:8.1f}  vt {d['vt_ms'][k]:6.1f}  "
            f"rng {d['rng_m'][k]:7.1f}  closure {d['closure_ms'][k]:+7.1f}  aspect "
            f"{d['aspect_deg'][k]:7.2f}  ata {d['ata_deg'][k]:6.2f}  cpa_miss {d['cpa_miss_m'][k]:8.1f}"
        )

    # ------------------------------------------------------------------ 7
    hdr("7  'THE ENERGY ADVANTAGE OVER THE BANDIT WAS NEVER SURRENDERED'")
    er = d["es_rel_m"]
    mn, mni = min(er[:173]), er.index(min(er[:173]))
    mx, mxi = max(er[:173]), er.index(max(er[:173]))
    print(
        f"  es_rel_m [0] {er[0]:8.1f}  min {mn:8.1f} at step {mni} (t {d['t_s'][mni]:.3f} s)  "
        f"max {mx:8.1f} at step {mxi}  [172] {er[172]:8.1f}"
    )
    print(f"  drawdown from the PEAK: {mx - mn:8.1f} m = {(mx - mn) / mx * 100.0:.1f}% of the peak")
    print(
        f"  drawdown below the SPAWN value: {er[0] - mn:+8.1f} m "
        f"({(er[0] - mn) / er[0] * 100.0:.1f}% of the spawn advantage)"
    )
    print(
        f"  bandit: alt {d['opp_alt_m'][0]:8.1f} -> {d['opp_alt_m'][172]:8.1f} "
        f"({d['opp_alt_m'][172] - d['opp_alt_m'][0]:+7.1f})  Es {d['opp_es_m'][0]:8.1f} -> "
        f"{d['opp_es_m'][172]:8.1f} ({d['opp_es_m'][172] - d['opp_es_m'][0]:+7.1f})  vt "
        f"{d['opp_vt_ms'][0]:6.1f} -> {d['opp_vt_ms'][172]:6.1f}"
    )
    print("\n  STRIP THE SPAWN TRANSIENT (steps 32 -> 172), i.e. score only the manoeuvre:")
    print(f"    my Es     {d['es_m'][172] - d['es_m'][32]:+8.1f} m")
    print(f"    bandit Es {d['opp_es_m'][172] - d['opp_es_m'][32]:+8.1f} m")
    print(f"    es_rel    {er[32]:8.1f} -> {er[172]:8.1f} = {er[172] - er[32]:+8.1f} m")
    print(
        f"  and over the transient itself (0 -> 32): es_rel {er[0]:8.1f} -> {er[32]:8.1f} = "
        f"{er[32] - er[0]:+8.1f} m"
    )
    print(
        f"  spawn share of the FINAL advantage: {er[0]:.1f} / {er[172]:.1f} = "
        f"{er[0] / er[172] * 100.0:.1f}%   (me_dalt_ft +2000 = {2000 * 0.3048:.1f} m of free altitude)"
    )

    # ------------------------------------------------------------------ 8
    hdr("8  DOES IT SURVIVE OFF THE ONE KILL? the non-kill seed, same commanded genome")
    k28 = load(NOKILL)
    cols = ["alt_m", "vt_ms", "es_m", "es_rel_m", "opp_es_m", "nz_g", "phi_deg"]
    print(f"  seed 28 rows = {k28['_n']}  (t {k28['t_s'][-1]:.3f} s)")
    print("  max |seed62 - seed28| over the OVERLAP steps 0-172:")
    for c in cols:
        m = max(abs(d[c][k] - k28[c][k]) for k in range(173))
        print(f"    {c:<10} {m:.3e}")
    print("\n  the SAME genome flown to its own 24 s horizon (seed 28, steps 0-575):")
    er28 = k28["es_rel_m"]
    print(
        f"    alt_m  [0] {k28['alt_m'][0]:8.1f} -> [575] {k28['alt_m'][575]:8.1f} "
        f"({k28['alt_m'][575] - k28['alt_m'][0]:+8.1f} m)   min {min(k28['alt_m']):8.1f} at step "
        f"{k28['alt_m'].index(min(k28['alt_m']))}"
    )
    print(
        f"    es_m   [0] {k28['es_m'][0]:8.1f} -> [575] {k28['es_m'][575]:8.1f} "
        f"({k28['es_m'][575] - k28['es_m'][0]:+8.1f} m)"
    )
    print(
        f"    es_rel [0] {er28[0]:8.1f} -> [575] {er28[575]:8.1f}   min {min(er28):8.1f} at step "
        f"{er28.index(min(er28))} (t {k28['t_s'][er28.index(min(er28))]:.3f} s)   "
        f"steps with es_rel < 0: {sum(1 for x in er28 if x < 0)}/{k28['_n']}"
    )
    print(
        f"    vt     [0] {k28['vt_ms'][0]:8.1f} -> [575] {k28['vt_ms'][575]:8.1f}   "
        f"max {max(k28['vt_ms']):8.1f}   min {min(k28['vt_ms']):8.1f}"
    )
    c28 = unwrap(
        [math.degrees(math.atan2(k28["vel_e_ms"][k], k28["vel_n_ms"][k])) for k in range(k28["_n"])]
    )
    print(
        f"    net horizontal course change 0->575: {c28[575] - c28[0]:+8.2f} deg for "
        f"{k28['alt_m'][575] - k28['alt_m'][0]:+.1f} m of altitude"
    )
    print(
        f"    => deg of net turn per 100 m of altitude:  kill window "
        f"{abs(course[172] - course[0]) / abs(a1 - a0) * 100.0:6.2f}   full horizon "
        f"{abs(c28[575] - c28[0]) / abs(k28['alt_m'][575] - k28['alt_m'][0]) * 100.0:6.2f}"
    )

    # ------------------------------------------------------------------ 9
    hdr("9  COMMANDED vs FLOWN, and how many genome dims were expressed")
    segs = sorted({int(x) for x in d["seg_idx"]})
    print(
        f"  seg_idx values present on the kill: {segs}  (of 0..5) -> "
        f"{len(segs) * 3} of 18 genome dims expressed"
    )
    for s in segs:
        ks = [k for k in range(n) if int(d["seg_idx"][k]) == s]
        print(
            f"    seg {s}: steps {ks[0]}-{ks[-1]} ({len(ks)} steps)  cmd bank "
            f"{d['cmd_bank_deg'][ks[0]]:+8.2f}  cmd load {d['cmd_load'][ks[0]]:.4f}  cmd thr "
            f"{d['cmd_thr'][ks[0]]:.4f}   flown thr {d['thr_flown'][ks[0]]:.4f}"
        )
    print("  genome p0 = 1.0 and p1 = 1.0 on seg 0: BOTH SATURATED at the search box edge")


if __name__ == "__main__":
    main()
