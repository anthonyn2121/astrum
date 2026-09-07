"""Advocacy audit part 3 — the last four discriminators.

T7  the roll GATE: was the unload-to-roll flown, and is it the search's or the decode's?
T8  cross-range rate: v_perp = los_rate * rng (DERIVED) — lead-collision or a rotating LOS?
T9  who out-rated whom: my course rate vs the bandit's heading rate
T10 closure: was any of it ever bled (the barrel-roll / lag-roll / yo-yo purpose)?
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
DT = 5.0 / 120.0
KROLL_GUESS = None


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


def main():
    d = load(KILL)
    n = d["_n"]

    hdr("T7  THE ROLL GATE — unload-to-roll then load-to-turn, and WHOSE idea it was")
    print(
        "  Genome.act (_diag_true_mpc.py:213-218): elev = -clip(load * max(cos(bank_err), 0), 0, 1)"
    )
    print(
        "  so with cmd bank 115 deg and phi = 0, bank_err = 115 deg, cos < 0 -> elev = 0 (UNLOADED),"
    )
    print("  and the pull only arrives as the bank error closes. Measured:")
    print(
        f"  {'step':>4} {'t_s':>6} {'phi':>8} {'bank_err':>9} {'cos(be)':>8} {'elev':>7} {'ail':>6} {'nz':>6} {'alpha':>6}"
    )
    for k in list(range(0, 40, 4)) + [44, 48]:
        be = 115.0 - d["phi_deg"][k]
        print(
            f"  {k:4d} {d['t_s'][k]:6.3f} {d['phi_deg'][k]:8.2f} {be:9.2f} "
            f"{math.cos(math.radians(be)):8.3f} {d['elev'][k]:7.3f} {d['ail'][k]:6.2f} "
            f"{d['nz_g'][k]:6.2f} {d['alpha_deg'][k]:6.2f}"
        )
    m = d["nz_g"] <= 1.0
    idx = np.where(m[:40])[0]
    print(
        f"  nz <= 1.0 g on steps {idx.min()}-{idx.max()} of the roll-in "
        f"(t {d['t_s'][idx.min()]:.3f}-{d['t_s'][idx.max()]:.3f} s); nz min {d['nz_g'][:40].min():.2f} g "
        f"at step {int(d['nz_g'][:40].argmin())}"
    )
    print(
        "  => the unload-then-load sequence IS flown, but it is a hard-coded property of the decode's"
    )
    print(
        "     roll gate, not something the 18-dim search discovered. Credit it to the parameterization."
    )

    hdr("T8  CROSS-RANGE RATE — DERIVED v_perp = radians(los_rate_dps) * rng")
    vp = np.radians(d["los_rate_dps"]) * d["rng_m"]
    ang = np.degrees(np.arctan2(vp, np.maximum(d["closure_ms"], 1e-6)))
    print(
        f"  {'step':>4} {'t_s':>6} {'rng':>7} {'los_rate':>9} {'v_perp':>8} {'closure':>8} {'atan(vp/clo)':>13} {'cpa_miss':>9}"
    )
    for k in [0, 24, 48, 72, 96, 120, 144, 156, 168, 171]:
        print(
            f"  {k:4d} {d['t_s'][k]:6.3f} {d['rng_m'][k]:7.0f} {d['los_rate_dps'][k]:9.2f} "
            f"{vp[k]:8.1f} {d['closure_ms'][k]:8.1f} {ang[k]:13.2f} {d['cpa_miss_m'][k]:9.1f}"
        )
    print(
        f"  v_perp falls {vp[0]:.1f} -> {vp[168]:.1f} m/s (steps 0 -> 168) while closure rises "
        f"{d['closure_ms'][0]:.1f} -> {d['closure_ms'][168]:.1f} m/s:"
    )
    print(
        "  the LOS is being frozen in ANGLE while the range collapses = a lead-collision approach."
    )

    hdr("T9  WHO OUT-RATED WHOM")
    course = np.degrees(np.unwrap(np.arctan2(d["vel_e_ms"], d["vel_n_ms"])))
    crate = np.r_[0.0, np.diff(course) / DT]
    orate = np.r_[0.0, np.diff(np.degrees(np.unwrap(np.radians(d["opp_psi_deg"])))) / DT]
    w = slice(36, 116)
    print("  steps 36-115 (both established in the turn):")
    print(
        f"    my   ground-course rate: median {np.median(crate[w]):6.2f} deg/s  max {crate[w].max():6.2f}"
    )
    print(
        f"    opp  heading rate      : median {np.median(orate[w]):6.2f} deg/s  max {orate[w].max():6.2f}"
    )
    print(
        f"    total over 0-172: mine {course[-1] - course[0]:+7.2f} deg, his "
        f"{d['opp_psi_deg'][-1] - d['opp_psi_deg'][0]:+7.2f} deg"
    )
    print(
        f"    my nz median over 36-115 {np.median(d['nz_g'][w]):.2f} g; his "
        f"{np.median(d['opp_nz_g'][w]):.2f} g (max {d['opp_nz_g'].max():.2f} g)"
    )
    print("  => the BANDIT out-rated me. The kill was not won by winning the turn fight.")

    hdr("T10  WAS ANY CLOSURE EVER BLED? (the purpose of a yo-yo / barrel-roll attack / lag roll)")
    clo = d["closure_ms"]
    dc = np.diff(clo[:172])
    print(
        f"  closure step 0 {clo[0]:.1f} -> max {clo[:172].max():.1f} at step {int(clo[:172].argmax())} "
        f"(t {d['t_s'][int(clo[:172].argmax())]:.3f} s) -> step 171 {clo[171]:.1f} m/s"
    )
    print(
        f"  rising on {int((dc > 0).sum())}/{len(dc)} steps, falling on {int((dc < 0).sum())}/{len(dc)}"
    )
    print(f"  min over steps 0-171: {clo[:172].min():.1f} m/s at step {int(clo[:172].argmin())}")
    print(
        f"  CLOSURE_OVERSHOOT bar is 120.0 m/s; closure is above it on "
        f"{int((clo > 120.0).sum())}/{n} steps = {(clo > 120.0).sum() * DT:.3f} s (base rate 100%)."
    )
    print(
        "  => nothing bled closure anywhere. Every maneuver whose PURPOSE is closure control is out."
    )

    hdr("EXTREMES TABLE — for citation")
    for c, lo in [
        ("aspect_deg", True),
        ("hca_deg", True),
        ("ata_deg", True),
        ("cpa_miss_m", True),
        ("nz_g", False),
        ("alpha_deg", False),
        ("alt_m", True),
        ("es_rel_m", True),
        ("lay_miss_m", True),
        ("alpha_req_deg", True),
    ]:
        v = d[c]
        print(
            f"  {c:14} min {v.min():9.2f} @step {int(v.argmin()):3d} (t {d['t_s'][int(v.argmin())]:6.3f})   "
            f"max {v.max():9.2f} @step {int(v.argmax()):3d} (t {d['t_s'][int(v.argmax())]:6.3f})   "
            f"first {v[0]:9.2f}  last {v[-1]:9.2f}"
        )
    h = np.abs(d["hca_deg"])
    print(
        f"  |hca| min {h.min():.2f} @step {int(h.argmin())} (t {d['t_s'][int(h.argmin())]:.3f} s)"
    )


if __name__ == "__main__":
    main()
