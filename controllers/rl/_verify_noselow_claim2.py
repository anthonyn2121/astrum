"""Second pass: the controls the 'nose-low slice' claim omits.

1. Was the downward flight-path rotation AIM WORK (LOS elevation) or waste?
2. The lift-vector allocation from bank angle (the actual 'turning capability' split).
3. Gravity-corrected version of the claim's own metric.
4. Base rates: is 'descends on 100% of hold steps' informative given the whole episode?
5. alt_m vs alt_enu_m bookkeeping for the -507 figure.
6. Segment-pure windows (the claim's hold 1 straddles the seg0/seg1 boundary at step 96).
"""

from __future__ import annotations

import csv
import math
import os

DT = 5.0 / 120.0
NZ_LOADED = 3.0
G = 9.80665
RUN = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def unwrap(seq):
    out = [seq[0]]
    for v in seq[1:]:
        d = v - out[-1]
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        out.append(out[-1] + d)
    return out


def main():
    rows = load(os.path.join(RUN, "replay_kill_seed62.csv"))
    n = len(rows)

    print("### 5. alt bookkeeping, kill seed 62")
    print(f"   row0   alt_m {rows[0]['alt_m']:.4f}  alt_enu_m {rows[0]['alt_enu_m']:.4f}  pos_u_m {rows[0]['pos_u_m']:.4f}")
    print(f"   row172 alt_m {rows[-1]['alt_m']:.4f}  alt_enu_m {rows[-1]['alt_enu_m']:.4f}  pos_u_m {rows[-1]['pos_u_m']:.4f}")
    print(f"   dAlt   alt_m {rows[-1]['alt_m']-rows[0]['alt_m']:+.4f}   alt_enu_m {rows[-1]['alt_enu_m']-rows[0]['alt_enu_m']:+.4f}")
    print("   (claim quotes 5181.6 -> 4674.6 = -507; measured alt_m end is 4674.9, alt_enu_m end is"
          f" {rows[-1]['alt_enu_m']:.1f})")

    print("\n### 1. Was the downward rotation aim work?  hold 1 = steps 28-115")
    for k in (0, 28, 48, 72, 96, 108, 115, 123, 156, 172):
        r = rows[k]
        print(f"   step {k:3d} t {r['t_s']:5.3f}  gam {r['gamma_deg']:+7.2f}  ata {r['ata_deg']:6.2f}"
              f"  ata_el {r['ata_el_deg']:+7.2f}  ata_az {r['ata_az_deg']:+7.2f}"
              f"  dalt_rel {r['dalt_rel_m']:+8.1f}  rng {r['rng_m']:7.1f}"
              f"  lay_miss {r['lay_miss_m']:8.1f}  areq {r['alpha_req_deg']:6.2f}"
              f"  cpa {r['cpa_miss_m']:8.1f}  oppalt {r['opp_alt_m']:7.1f}")
    a, b = 28, 115
    print(f"   HOLD 1: dalt_rel {rows[a]['dalt_rel_m']:+.1f} -> {rows[b]['dalt_rel_m']:+.1f} m"
          f"   |ata_el| {abs(rows[a]['ata_el_deg']):.2f} -> {abs(rows[b]['ata_el_deg']):.2f} deg"
          f"   |ata_az| {abs(rows[a]['ata_az_deg']):.2f} -> {abs(rows[b]['ata_az_deg']):.2f} deg")
    print(f"   opponent alt {rows[0]['opp_alt_m']:.1f} -> {rows[-1]['opp_alt_m']:.1f} m"
          f"  (dAlt {rows[-1]['opp_alt_m']-rows[0]['opp_alt_m']:+.1f} m)"
          f"   opp gamma-equivalent: opp_theta {rows[0]['opp_theta_deg']:+.2f} -> {rows[-1]['opp_theta_deg']:+.2f}")
    # how much of the -255 m is 'chasing the bandit down'?
    print(f"   my dAlt over hold1 {rows[b]['alt_m']-rows[a]['alt_m']:+.1f} m ;"
          f" opp dAlt over hold1 {rows[b]['opp_alt_m']-rows[a]['opp_alt_m']:+.1f} m")

    print("\n### 2. Lift-vector allocation from bank angle over hold 1 (steps 28-115)")
    seg = rows[a : b + 1]
    m = len(seg)
    csum = sum(abs(math.cos(math.radians(r["phi_deg"]))) for r in seg) / m
    ssum = sum(abs(math.sin(math.radians(r["phi_deg"]))) for r in seg) / m
    # nz-weighted: the actual specific force allocated to each axis
    wv = sum(r["nz_g"] * abs(math.cos(math.radians(r["phi_deg"]))) for r in seg) * G * DT
    wh = sum(r["nz_g"] * abs(math.sin(math.radians(r["phi_deg"]))) for r in seg) * G * DT
    print(f"   mean |cos phi| = {csum:.4f} (vertical axis)   mean |sin phi| = {ssum:.4f} (heading axis)")
    print(f"   L1 share of lift on the VERTICAL axis = {csum/(csum+ssum)*100:.1f}%")
    print(f"   nz-weighted impulse: vertical {wv:.1f} m/s, heading {wh:.1f} m/s"
          f"  -> vertical share {wv/(wv+wh)*100:.1f}%")
    print("   NOTE phi mean +108.6 deg > 90 => cos phi < 0 => the lift's vertical component points DOWN")

    print("\n### 3. Gravity-corrected version of the claim's OWN metric, hold 1")
    gam = [r["gamma_deg"] for r in seg]
    vt = [r["vt_ms"] for r in seg]
    psi = unwrap([r["psi_deg"] for r in seg])
    horiz = sum(
        (psi[k + 1] - psi[k]) * math.cos(math.radians(0.5 * (gam[k] + gam[k + 1]))) for k in range(m - 1)
    )
    dgam = gam[-1] - gam[0]
    grav = sum(math.degrees(-G * math.cos(math.radians(gam[k])) / vt[k] * DT) for k in range(m - 1))
    lift_dgam = dgam - grav
    print(f"   dPsi*cos(gam) {horiz:+.2f}  dGamma {dgam:+.2f}  gravity-only dGamma {grav:+.2f}"
          f"  lift-attributable dGamma {lift_dgam:+.2f}")
    nrm_raw = math.hypot(horiz, dgam)
    nrm_lift = math.hypot(horiz, lift_dgam)
    print(f"   claim's metric   |dGam|/hypot = {abs(dgam)/nrm_raw*100:.1f}%   (norm {nrm_raw:.2f})")
    print(f"   gravity-removed  |dGam_lift|/hypot = {abs(lift_dgam)/nrm_lift*100:.1f}%   (norm {nrm_lift:.2f})")
    print(f"   the two shares of the claim's norm sum to {abs(dgam)/nrm_raw*100 + abs(horiz)/nrm_raw*100:.1f}%"
          "  <- a component/norm ratio is NOT a partition")

    print("\n### 4. Base rate for 'descends on 100% of its steps'")
    alt = [r["alt_m"] for r in rows]
    nz = [r["nz_g"] for r in rows]
    unl = [k for k in range(n) if nz[k] <= NZ_LOADED]
    ld = [k for k in range(n) if nz[k] > NZ_LOADED]
    print(f"   loaded rows (nz>3): {len(ld)}/{n}   unloaded rows: {len(unl)}/{n}")
    print(f"   fraction of LOADED rows with gamma<0   = {sum(1 for k in ld if rows[k]['gamma_deg'] < 0)}/{len(ld)}")
    print(f"   fraction of UNLOADED rows with gamma<0 = {sum(1 for k in unl if rows[k]['gamma_deg'] < 0)}/{len(unl)}")
    print(f"   strictly-decreasing alt intervals: {sum(1 for k in range(n-1) if alt[k+1] < alt[k])}/{n-1}")
    # sink rate loaded vs unloaded
    sl = [(alt[k + 1] - alt[k]) / DT for k in range(n - 1) if nz[k] > NZ_LOADED]
    su = [(alt[k + 1] - alt[k]) / DT for k in range(n - 1) if nz[k] <= NZ_LOADED]
    print(f"   mean sink on LOADED intervals {sum(sl)/len(sl):+.2f} m/s ; on UNLOADED {sum(su)/len(su):+.2f} m/s")
    print("   initial condition me_dalt_ft = +2000 (perch): the episode is a descent from step 0 by construction")

    print("\n### 6. segment-pure windows on the kill seed")
    for (lo, hi, lbl) in ((28, 95, "seg0 loaded part 28-95"), (96, 115, "seg1 head of hold1 96-115")):
        s = rows[lo : hi + 1]
        mm = len(s)
        gg = [r["gamma_deg"] for r in s]
        pp = unwrap([r["psi_deg"] for r in s])
        hz = sum((pp[k + 1] - pp[k]) * math.cos(math.radians(0.5 * (gg[k] + gg[k + 1]))) for k in range(mm - 1))
        dg = gg[-1] - gg[0]
        print(f"   {lbl}: n={mm}  {(mm-1)*DT:.3f} s  meanNz {sum(r['nz_g'] for r in s)/mm:.2f}"
              f"  dPsi*cos {hz:+.2f}  dGamma {dg:+.2f}  |dGam|/hypot {abs(dg)/math.hypot(hz,dg)*100:.1f}%"
              f"  dAlt {s[-1]['alt_m']-s[0]['alt_m']:+.1f} m  dEs {s[-1]['es_m']-s[0]['es_m']:+.1f} m")
    print(f"   gamma min over hold1 = {min(r['gamma_deg'] for r in rows[28:116]):+.2f} deg at step"
          f" {28 + min(range(88), key=lambda i: rows[28+i]['gamma_deg'])}"
          "  -> gamma REVERSES inside the claim's hold-1 window")

    print("\n### base rate: episode-level energy, kill seed 62 vs non-kill seed 28")
    r28 = load(os.path.join(RUN, "replay_nokill_seed28.csv"))
    for nm, rr in (("62 kill", rows), ("28 nokill", r28)):
        print(f"   {nm:10s} Es {rr[0]['es_m']:.1f} -> {rr[-1]['es_m']:.1f} m  dEs {rr[-1]['es_m']-rr[0]['es_m']:+.1f} m"
              f"  es_rel {rr[0]['es_rel_m']:.1f} -> {rr[-1]['es_rel_m']:.1f} m"
              f"  vt {rr[0]['vt_ms']:.1f} -> {rr[-1]['vt_ms']:.1f} m/s")


if __name__ == "__main__":
    main()
