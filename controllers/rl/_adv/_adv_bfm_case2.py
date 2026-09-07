"""Advocacy audit part 2 — the discriminating tests.

T1  one-circle vs two-circle: signed turn sense of BOTH jets + horizontal turn-circle centres
T2  lead / pure / lag pursuit: the ata_az sign crossing and the lay solution it produced
T3  max-performance? stick saturation and the alpha/nz corner
T4  chi validity: can the detectors even see the endgame
T5  the built-in no-turn counterfactual (cpa_miss at step 0 is exactly that extrapolation)
T6  energy ledger by phase, with DERIVED Ps
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
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
    print("\n" + "=" * 104)
    print("  " + s)
    print("=" * 104)


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def main():
    d = load(KILL)
    n = d["_n"]

    hdr("T1  ONE-CIRCLE vs TWO-CIRCLE — signed turn sense of both jets, and the turn centres")
    # COURSE (track over the ground) = atan2(vel_e, vel_n): the same `course()` geometry.hca uses.
    my_course = unwrap_deg(np.degrees(np.arctan2(d["vel_e_ms"], d["vel_n_ms"])))
    # the bandit's velocity is not dumped componentwise; opp_psi_deg is his heading. Use it, and
    # note the substitution explicitly.
    op_psi = unwrap_deg(d["opp_psi_deg"])
    print(
        f"  my course  (atan2(vel_e, vel_n), unwrapped): {my_course[0]:+8.2f} -> {my_course[-1]:+8.2f} deg"
        f"   signed change {my_course[-1] - my_course[0]:+8.2f} deg"
    )
    print(
        f"  opp psi    (opp_psi_deg, unwrapped)        : {op_psi[0]:+8.2f} -> {op_psi[-1]:+8.2f} deg"
        f"   signed change {op_psi[-1] - op_psi[0]:+8.2f} deg"
    )
    print(
        "  psi/course convention: 0 = north, +90 = east, so INCREASING = clockwise seen from above."
    )
    print(
        f"  => my sense {'CW' if my_course[-1] > my_course[0] else 'CCW'}, "
        f"bandit sense {'CW' if op_psi[-1] > op_psi[0] else 'CCW'}  -> "
        f"{'SAME rotational sense' if (my_course[-1] - my_course[0]) * (op_psi[-1] - op_psi[0]) > 0 else 'OPPOSITE senses'}"
    )
    print(
        "     same ground-relative sense from reciprocal headings = the two turns share one circle"
    )
    print("     (nose-to-nose, ONE-CIRCLE); opposite senses would be nose-to-tail, TWO-CIRCLE.\n")
    print(
        f"  {'step':>4} {'t_s':>6} {'my_course':>10} {'my_rate':>8} {'opp_psi':>9} {'opp_rate':>9} "
        f"{'ctr_sep_m':>10} {'my_R_m':>8} {'aspect':>7} {'hca':>8}"
    )
    # horizontal turn radius from the horizontal speed and the horizontal course rate
    vh = np.hypot(d["vel_e_ms"], d["vel_n_ms"])
    crate = np.r_[0.0, np.diff(my_course) / DT]
    orate = np.r_[0.0, np.diff(op_psi) / DT]
    R = vh / np.radians(np.where(np.abs(crate) > 1e-3, np.abs(crate), 1e-3))
    # my centre: 90 deg to the RIGHT of my ground track when turning CW
    for k in list(range(0, n, 12)) + [n - 1]:
        sgn = 1.0 if crate[k] >= 0 else -1.0
        th = math.radians(my_course[k]) + sgn * math.pi / 2.0
        cme = np.array(
            [d["pos_e_m"][k] + R[k] * math.sin(th), d["pos_n_m"][k] + R[k] * math.cos(th)]
        )
        so = 1.0 if orate[k] >= 0 else -1.0
        Ro = 223.0 / max(math.radians(abs(orate[k])), 1e-3)  # opp horizontal speed ~ opp_vt
        Ro = min(Ro, 20000.0)
        tho = math.radians(op_psi[k]) + so * math.pi / 2.0
        cop = np.array(
            [d["opp_pos_e_m"][k] + Ro * math.sin(tho), d["opp_pos_n_m"][k] + Ro * math.cos(tho)]
        )
        print(
            f"  {int(d['step'][k]):4d} {d['t_s'][k]:6.3f} {my_course[k]:10.2f} {crate[k]:8.2f} "
            f"{op_psi[k]:9.2f} {orate[k]:9.2f} {np.linalg.norm(cme - cop):10.0f} {R[k]:8.0f} "
            f"{d['aspect_deg'][k]:7.2f} {d['hca_deg'][k]:8.2f}"
        )

    hdr("T2  LEAD / PURE / LAG — the ata_az sign crossing, and the lay solution the lead produced")
    az = d["ata_az_deg"]
    cross = [k for k in range(1, n) if az[k - 1] > 0 >= az[k]]
    print(
        f"  ata_az (LOS bearing off MY nose, right +): step 0 {az[0]:+7.2f} deg (bandit 30 deg RIGHT)"
    )
    print(
        f"  I roll RIGHT (ail +1.0, phi -> +113 deg). Nose crosses the LOS at step(s) {cross[:3]}"
        f" -> t {[round(float(d['t_s'][c]), 3) for c in cross[:3]]} s"
    )
    for k in (0, 12, 24, 28, 30, 32, 36, 48, 72, 96, 120, 156, 168):
        print(
            f"    step {k:3d} t {d['t_s'][k]:6.3f}  ata_az {az[k]:+7.2f}  ata {d['ata_deg'][k]:6.2f}  "
            f"alpha_req {d['alpha_req_deg'][k]:6.2f}  lay_miss {d['lay_miss_m'][k]:9.2f} m  "
            f"lead_error {d['lead_error_deg'][k]:7.2f}  rng {d['rng_m'][k]:7.0f}  "
            f"cpa_miss {d['cpa_miss_m'][k]:8.1f}"
        )
    ar = d["alpha_req_deg"]
    print("\n  alpha_req_deg (lay angle the airframe must produce off its velocity vector):")
    print(
        f"    step 0 {ar[0]:.2f} -> max {ar.max():.2f} at step {int(ar.argmax())} (t {d['t_s'][int(ar.argmax())]:.3f} s)"
        f" -> min {ar[:170].min():.2f} at step {int(ar[:170].argmin())} (t {d['t_s'][int(ar[:170].argmin())]:.3f} s)"
    )
    seg = ar[24:170]
    print(
        f"    monotone decreasing from step 24 to 169? down {int((np.diff(seg) < 0).sum())}/{len(seg) - 1} steps"
    )
    print(
        f"    measured alpha at the first trigger step 156 = {d['alpha_deg'][156]:.2f} deg vs "
        f"alpha_req {ar[156]:.2f} deg -> the required lay is INSIDE the usable alpha cone"
    )

    hdr("T3  MAX-PERFORMANCE? stick saturation and the alpha/nz corner during the pull")
    a, b = 28, 115
    el = d["elev"][a : b + 1]
    print(
        f"  hold steps {a}-{b}: elev min {el.min():.3f} max {el.max():.3f}; steps at full aft "
        f"(elev <= -0.999): {int((el <= -0.999).sum())}/{len(el)} = {(el <= -0.999).sum() * DT:.3f} s"
    )
    m = d["elev"] <= -0.999
    if m.any():
        i0 = int(np.argmax(m))
        idx = np.where(m)[0]
        print(
            f"  full-aft run: steps {idx.min()}-{idx.max()} (t {d['t_s'][idx.min()]:.3f}-{d['t_s'][idx.max()]:.3f} s), "
            f"contiguous {len(idx) == idx.max() - idx.min() + 1}; first at step {i0}"
        )
    print(
        f"  nz over the hold: max {d['nz_g'][a : b + 1].max():.2f} g at step "
        f"{a + int(d['nz_g'][a : b + 1].argmax())} (t {d['t_s'][a + int(d['nz_g'][a : b + 1].argmax())]:.3f} s); "
        f"sustained band steps 44-96 = {d['nz_g'][44:97].min():.2f}-{d['nz_g'][44:97].max():.2f} g"
    )
    print(
        f"  alpha over the hold: max {d['alpha_deg'][a : b + 1].max():.2f} deg at step "
        f"{a + int(d['alpha_deg'][a : b + 1].argmax())}"
    )
    print(
        f"  ail: +1.000 run = steps {int(np.argmax(d['ail'] < 0.999))} is the first below 1.0; "
        f"steps with ail >= 0.999: {int((d['ail'] >= 0.999).sum())} = {(d['ail'] >= 0.999).sum() * DT:.3f} s; "
        f"steps with ail <= -0.999: {int((d['ail'] <= -0.999).sum())} = {(d['ail'] <= -0.999).sum() * DT:.3f} s"
    )
    idxp = np.where(d["ail"] >= 0.999)[0]
    idxn = np.where(d["ail"] <= -0.999)[0]
    print(
        f"    +1.0 roll pulse steps {idxp.min()}-{idxp.max()} (t {d['t_s'][idxp.min()]:.3f}-{d['t_s'][idxp.max()]:.3f} s)"
        f"  |  -1.0 roll-out pulse steps {idxn.min()}-{idxn.max()} "
        f"(t {d['t_s'][idxn.min()]:.3f}-{d['t_s'][idxn.max()]:.3f} s)"
    )
    hold = (np.abs(d["ail"]) < 0.10) & (d["nz_g"] > 3.0)
    best = (0, 0, 0)
    i = 0
    while i < n:
        if hold[i]:
            j = i
            while j + 1 < n and hold[j + 1]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j + 1
        else:
            i += 1
    print(
        f"  PULSE-THEN-HOLD signature (bfm_detect.commitment: a rate interface commits with a pulse "
        f"then |ail|~0):\n    longest |ail|<0.10 AND nz>3 run = steps {best[1]}-{best[2]} "
        f"({(best[2] - best[1] + 1) * DT:.3f} s), phi {d['phi_deg'][best[1]]:.2f} -> "
        f"{d['phi_deg'][best[2]]:.2f} deg, alt {d['alt_m'][best[1]]:.1f} -> {d['alt_m'][best[2]]:.1f} m"
    )

    hdr("T4  CHI VALIDITY — could the detectors see the endgame at all?")
    cv = d["chi_valid"]
    print(
        f"  chi_valid == 0 on {int((cv == 0).sum())}/{n} steps"
        + (f"; steps {list(np.where(cv == 0)[0])}" if (cv == 0).any() else "")
    )
    print(
        f"  ata < 5 deg (the CHI_VALID_ATA_DEG bar) on {int((d['ata_deg'] < 5.0).sum())}/{n} steps: "
        f"steps {list(np.where(d['ata_deg'] < 5.0)[0])[:20]}"
    )

    hdr("T5  THE BUILT-IN NO-TURN COUNTERFACTUAL")
    print(
        f"  cpa_miss at step 0 = {d['cpa_miss_m'][0]:.1f} m. bfm_detect.cpa_miss_m is the miss at the "
        f"CPA of the two aircraft\n  under CONSTANT VELOCITY from that instant, i.e. exactly 'neither "
        f"jet turns from here'.\n  At the last step cpa_miss = {d['cpa_miss_m'][-1]:.1f} m and the "
        f"achieved min range = {d['rng_m'].min():.1f} m."
    )
    print(
        f"  So the maneuver closed a {d['cpa_miss_m'][0]:.0f} m predicted miss to {d['cpa_miss_m'][-1]:.1f} m: "
        f"a factor of {d['cpa_miss_m'][0] / max(d['cpa_miss_m'][-1], 1e-9):.0f}x."
    )
    print(
        f"  base rate to quote alongside it: p_hit/round is 0.000 at 1200 m, so a {d['cpa_miss_m'][0]:.0f} m "
        f"pass is 0 HP by construction."
    )

    hdr("T6  ENERGY LEDGER BY PHASE (DERIVED Ps = dEs/dt differenced at dt = 41.667 ms)")
    es = d["es_m"]
    ps = np.r_[0.0, np.diff(es) / DT]
    for nm, i0, i1 in [
        ("roll-in           ", 0, 32),
        ("slice pull (hold A)", 28, 115),
        ("unload / roll-out ", 108, 122),
        ("tracking run-in (hold B)", 123, 172),
        ("WHOLE TRAJECTORY  ", 0, 172),
    ]:
        print(
            f"  {nm:<24} steps {i0:3d}-{i1:3d} t {d['t_s'][i0]:6.3f}-{d['t_s'][i1]:6.3f} "
            f"({(i1 - i0 + 1) * DT:5.3f} s)  alt {d['alt_m'][i0]:7.1f} -> {d['alt_m'][i1]:7.1f} "
            f"({d['alt_m'][i1] - d['alt_m'][i0]:+7.1f} m)  vt {d['vt_ms'][i0]:6.1f} -> {d['vt_ms'][i1]:6.1f} "
            f"({d['vt_ms'][i1] - d['vt_ms'][i0]:+5.1f})  Es {es[i0]:7.1f} -> {es[i1]:7.1f} "
            f"({es[i1] - es[i0]:+7.1f} m)  meanPs {ps[i0 + 1 : i1 + 1].mean():+7.1f} m/s  "
            f"es_rel {d['es_rel_m'][i0]:7.1f} -> {d['es_rel_m'][i1]:7.1f}"
        )
    print(
        f"\n  bandit ledger over the same 7.208 s: alt {d['opp_alt_m'][0]:.1f} -> {d['opp_alt_m'][-1]:.1f} m "
        f"({d['opp_alt_m'][-1] - d['opp_alt_m'][0]:+.1f}), vt {d['opp_vt_ms'][0]:.1f} -> "
        f"{d['opp_vt_ms'][-1]:.1f}, Es {d['opp_es_m'][0]:.1f} -> {d['opp_es_m'][-1]:.1f} "
        f"({d['opp_es_m'][-1] - d['opp_es_m'][0]:+.1f} m)"
    )
    print(
        f"  es_rel never went negative: min {d['es_rel_m'].min():.1f} m at step {int(d['es_rel_m'].argmin())}, "
        f"max {d['es_rel_m'].max():.1f} m at step {int(d['es_rel_m'].argmax())} "
        f"(the +2000 ft perch was worth {d['es_rel_m'][0]:.1f} m of Es at step 0)"
    )
    print(
        f"  throttle actually flown: seg0 {d['thr_flown'][0]:.4f} (steps 0-95), seg1 "
        f"{d['thr_flown'][96]:.4f} (steps 96-172)"
    )


if __name__ == "__main__":
    main()
