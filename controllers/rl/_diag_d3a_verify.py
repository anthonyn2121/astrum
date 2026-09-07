#!/usr/bin/env python3
"""D3a ADVERSARIAL VERIFICATION -- the two joints the D3a write-up rests on that its own
run did not test. Re-uses `_diag_dwell_budget.py` READ-ONLY (imports `_roll_cell`, `_make_arm`,
`CELLS`) so the airframe, the arms, the env construction and the `lethal` predicate are
byte-identical to the box under review; nothing in that file is modified.

================================================================================
WHAT IS MEASURED
================================================================================
V1  RANGE-MATCHED alpha_req. The D3a write-up attributes a "4-10x geometry loss" to the T3's
    break from the shootable fraction (alpha_req <= cone) inside 600 m, T3 vs the Tier-1
    dummy. But the two arms do not occupy the same ranges: in D3a's own table the T3 arms
    log in400 = 0 of in600 = 483-983 control steps at chase-asp30 (100% of sub-600 m
    exposure in the 400-600 m band) while the dummy arms log 56-76% of theirs INSIDE 400 m --
    and alpha_req is a steep monotone function of range on D3a's own co-circle table
    (12.39 deg at 300 m, 16.56 at 400, 25.01 at 600). So the contrast may be measuring the
    range band, not the break. This bins alpha_req and the shootable fraction by 100 m of
    range and reports T3 vs dummy WITHIN each bin.
V2  A1's SYMMETRY PREMISE. A1 ("both jets are the same f16 at the same load factor, so
    neither out-turns the other") is fed the BANDIT's measured nz (p50 3.97-7.05, p90 up to
    9.31). The harness records the LEARNER's nz (`nz_me`) and its FDM alpha and never prints
    either. If the learner's loaded nz sits well under the bandit's, the co-turn steady state
    is not attainable by the shooter and R* is an optimistic upper bound, not a bound the
    shooter can occupy. Reported side by side on the same ticks.
V3  SEED HYGIENE. D3a's A2 uses `seed0 = seed + 1000*(hash(cell) % 7) + 37*opp_tier`, so the
    T3 arm and its Tier-1 control get DIFFERENT jitter draws and different reset seeds (+37),
    and `hash(str)` is PYTHONHASHSEED-randomised, so the cell offset is not reproducible
    across invocations. Here both opponents get the IDENTICAL seed0 and reset seeds, so the
    control is seed-matched as well as build-matched.

================================================================================
WHAT IS EXPLICITLY *NOT* MEASURED
================================================================================
* No p_hit / E[hits] integral, no NegBinom dwell requirement, no analytic A1 table: those
  halves of D3a are not under test here and are not recomputed.
* No checkpoint is loaded. This says nothing about any learned policy.
* Not a claim that a range-matched gap would make the fight winnable or unwinnable. It only
  tests whether an ATTRIBUTION ("the loss is attributable to the break") is confounded.
* Episode length, train-log kill%, reward sums, tHit and the D5 ratio are read nowhere.

================================================================================
PRE-REGISTERED INTERPRETATION (fixed before the run)
================================================================================
Q1  If, within a shared range bin, the T3 and dummy shootable fractions are comparable
    (ratio <= ~2x) while the pooled unbinned ratio is 4-10x, then D3a's "4-10x geometry loss
    attributable to the break" is dominated by WHERE the arms fly, not by the break's effect
    on pointing at a fixed range. The admissible restatement is "the break denies CLOSURE
    into the band", which is a different (and weaker, because arm-conditional) claim.
Q2  If the gap SURVIVES range matching (T3 shootable fraction still >=4x lower inside the
    same 100 m bin), D3a's attribution stands as written and this objection dies.
Q3  If the learner's loaded nz p50/p90 is materially below the bandit's on the same ticks,
    A1's equal-load premise is violated in the optimistic direction and every R* must be
    labelled an upper bound. If they match, the premise is verified and A1 is unharmed.
Units: one control step = action_repeat 5 / 120 Hz = 41.7 ms; 12 steps = 0.5 s, 36 = 1.5 s.

Run (from `controllers/rl`, inside the sim container, NO --user):
    uv run python _diag_d3a_verify.py 2>&1 | tee _d3a_verify_out.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _diag_dwell_budget as D  # noqa: E402  (READ-ONLY re-use; not modified)
import numpy as np  # noqa: E402

DEG = 180.0 / math.pi
BINS = [(200.0, 300.0), (300.0, 400.0), (400.0, 500.0), (500.0, 600.0)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="adversarial verification of D3a's attributions")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--arms", default="expert,lead_max")
    p.add_argument("--cells", default="chase-asp30,chase-asp55")
    p.add_argument(
        "--cone-deg",
        type=float,
        default=16.7,
        help="the cone D3a's `shoot` column uses (ALPHA_CEIL_DEG)",
    )
    p.add_argument(
        "--cone2-deg",
        type=float,
        default=12.4,
        help="the TIGHTER cone D3a's own A2c measured (alpha p99)",
    )
    return p.parse_args(argv)


def main():
    args = parse_args()
    print("D3a VERIFY | " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    print("one control step = 5/120 Hz = 41.7 ms ; 12 steps = 0.5 s ; 36 steps = 1.5 s")
    print(
        f"cone1 = {args.cone_deg} deg (D3a's ALPHA_CEIL_DEG), "
        f"cone2 = {args.cone2_deg} deg (D3a's own MEASURED alpha p99)"
    )
    print("SEED-MATCHED: T3 and the Tier-1 dummy get the SAME seed0 and the same reset seeds.")
    dt = args.action_repeat / args.rate_hz
    roll_args = SimpleNamespace(
        episodes=args.episodes,
        jitter=args.jitter,
        action_repeat=args.action_repeat,
        rate_hz=args.rate_hz,
    )
    c1 = math.radians(args.cone_deg)
    c2 = math.radians(args.cone2_deg)

    for cell in args.cells.split(","):
        cell = cell.strip()
        if not cell:
            continue
        for arm in args.arms.split(","):
            arm = arm.strip()
            if not arm:
                continue
            print()
            print("=" * 104)
            print(
                f"V1  RANGE-MATCHED alpha_req -- cell {cell} / arm {arm} "
                f"({args.episodes} eps each, identical seeds)"
            )
            print("=" * 104)
            print(
                f"  {'band m':>10} | {'opp':<6} {'steps':>6} {'sec':>7} | "
                f"{'areq p50':>8} {'p25':>7} {'min':>7} | "
                f"{'shoot@' + str(args.cone_deg):>10} {'shoot@' + str(args.cone2_deg):>10} | "
                f"{'lay p50 m':>9} {'tLeth s':>8}"
            )
            eps = {}
            for tier, lbl in ((0, "T3"), (1, "dummy")):
                eps[lbl] = D._roll_cell(arm, tier, cell, roll_args, args.seed)
            rows = {}
            for lo, hi in BINS + [(0.0, 600.0)]:
                band = f"{lo:.0f}-{hi:.0f}"
                for lbl in ("T3", "dummy"):
                    n = s1 = s2 = leth = 0
                    areq, lay = [], []
                    for r in eps[lbl]:
                        g = (r["rng"] >= lo) & (r["rng"] < hi)
                        n += int(g.sum())
                        a = r["alpha_req"][g]
                        areq.extend(a[np.isfinite(a)].tolist())
                        lay.extend(r["lay"][g].tolist())
                        s1 += int((g & (r["alpha_req"] <= c1)).sum())
                        s2 += int((g & (r["alpha_req"] <= c2)).sum())
                        leth += int((g & r["lethal"]).sum())

                    def q(xs, p):
                        return float(np.percentile(xs, p)) * DEG if len(xs) else float("nan")

                    f1 = 100.0 * s1 / n if n else float("nan")
                    f2 = 100.0 * s2 / n if n else float("nan")
                    rows[(band, lbl)] = (n, f1, f2)
                    print(
                        f"  {band:>10} | {lbl:<6} {n:6d} {n * dt:7.1f} | "
                        f"{q(areq, 50):8.1f} {q(areq, 25):7.1f} {q(areq, 0):7.1f} | "
                        f"{f1:9.1f}% {f2:9.1f}% | "
                        f"{(float(np.median(lay)) if lay else float('nan')):9.1f} "
                        f"{leth * dt:8.2f}"
                    )
            print("   areq = angle(gunlay.intercept boresight, MY velocity vector), the quantity")
            print("   the alpha cone bounds. shoot@X = fraction of that band's steps with")
            print("   areq <= X. Bands are DISJOINT, so a T3-vs-dummy comparison within one row")
            print("   is range-matched; the 0-600 row is D3a's unbinned statistic.")
            print(
                f"  {'RATIO dummy/T3 of shoot@' + str(args.cone_deg):<44} | "
                f"{'band':>10} {'T3 %':>7} {'dummy %':>8} {'ratio':>7} "
                f"{'T3 steps':>9} {'dummy steps':>11}"
            )
            for lo, hi in BINS + [(0.0, 600.0)]:
                band = f"{lo:.0f}-{hi:.0f}"
                nT, fT, _ = rows[(band, "T3")]
                nD, fD, _ = rows[(band, "dummy")]
                rat = (fD / fT) if (fT and np.isfinite(fT) and fT > 0) else float("nan")
                print(f"  {'':<44} | {band:>10} {fT:7.1f} {fD:8.1f} {rat:7.2f} {nT:9d} {nD:11d}")

            print()
            print(f"V2  A1's EQUAL-LOAD PREMISE -- cell {cell} / arm {arm} (ticks inside 2500 m)")
            print(
                f"  {'opp':<6} | {'ME nz p50':>9} {'p90':>6} {'max':>6} | "
                f"{'BANDIT nz p50':>13} {'p90':>6} {'max':>6} | "
                f"{'ME alpha p50':>12} {'p90':>6} {'max':>6} | {'ME nose-off-v max':>17}"
            )
            for lbl in ("T3", "dummy"):
                mz, bz, al, nv = [], [], [], []
                for r in eps[lbl]:
                    g = r["rng"] < 2500.0
                    mz.extend(r["nz_me"][g].tolist())
                    bz.extend(r["nz_bd"][g].tolist())
                    al.extend((r["alpha_me"][g] * DEG).tolist())
                    nv.extend((r["nose_off_vel"][g] * DEG).tolist())
                if not mz:
                    continue
                p = lambda xs, q: float(np.percentile(xs, q))  # noqa: E731
                print(
                    f"  {lbl:<6} | {p(mz, 50):9.2f} {p(mz, 90):6.2f} {max(mz):6.2f} | "
                    f"{p(bz, 50):13.2f} {p(bz, 90):6.2f} {max(bz):6.2f} | "
                    f"{p(al, 50):12.2f} {p(al, 90):6.2f} {max(al):6.2f} | {max(nv):17.2f}"
                )
            print("   A1 assumes the SHOOTER flies the bandit's (V, n). If ME nz << BANDIT nz on")
            print("   the same ticks, the co-circle is not a state the shooter can occupy and R*")
            print("   is an upper bound only.")
    print("\ndone.")


if __name__ == "__main__":
    main()
