"""D3b DETECTOR AUDIT, part 2 — price the one D3 positive, and characterise what the kill trace IS.

Part 1 established which clause each detector fails on. This part answers the harder question:
  * the single D3 event in the whole 8-seed search pool — how big is its "turning-room build",
    and where in the fight does it happen? (a criterion with no MAGNITUDE requirement can fire on
    noise);
  * is D1's aspect-gain clause satisfiable AT ALL on the kill trace (max aspect improvement over
    every (i, j) pair);
  * what manoeuvre does the kill trace fly, in the detector's own scalars, so the gap between
    "D1 = D2 = 0" and "no BFM" can be stated precisely.

No repo file is modified.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
CDT = 5.0 / 120.0

SMP = [
    ("t", float),
    ("chi_rad", float),
    ("chi_valid", bool),
    ("gamma_rad", float),
    ("nz", float),
    ("phi_rad", float),
    ("alt", float),
    ("ail", float),
    ("rng", float),
    ("ata", float),
    ("ata_az", float),
    ("aspect", float),
    ("closure", float),
    ("cpa_miss", float),
    ("es_rel", float),
    ("lay_miss_m", float),
    ("lethal", bool),
]


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    tr = []
    for r in rows:
        kw = {}
        for raw, cast in SMP:
            v = r[f"smp_{raw}"]
            kw[raw.removesuffix("_rad")] = (int(float(v)) != 0) if cast is bool else float(v)
        tr.append(B.Sample(**kw))
    return rows, tr


def main() -> None:
    log = print
    kill_rows, kill = load("runs/d3b/replay_kill_seed62.csv")
    nk_rows, nk = load("runs/d3b/replay_nokill_seed28.csv")

    log("#" * 100)
    log(
        "#  AUDIT 2 — magnitude of the D3 positive; satisfiability of D1(f); what the kill trace IS"
    )
    log("#" * 100)

    # -------------------------------------------------------------------------------------
    log("")
    log("A. THE ONE D3 POSITIVE IN THE 8-SEED POOL (seed 28; seeds 79/96 are the same trajectory)")
    for ev in B.detect_turning_room(nk):
        i, j = ev.i0, ev.i1
        d = ev.detail
        build = d["cpa_peak_m"] - d["cpa_entry_m"]
        cash = d["cpa_peak_m"] - d["cpa_exit_m"]
        log(
            f"   event {ev.kind}  i0={i} i1={j}  t {ev.t0:.3f}-{ev.t1:.3f} s  "
            f"hold {d['hold_s']:.3f} s"
        )
        log(
            f"     cpa_miss  entry {d['cpa_entry_m']:.1f} m -> peak {d['cpa_peak_m']:.1f} m "
            f"-> exit {d['cpa_exit_m']:.1f} m"
        )
        log(
            f"     BUILD = +{build:.1f} m  ({build / d['cpa_entry_m'] * 100:.2f}% of the entry CPA)   "
            f"CASH = -{cash:.1f} m   <- the ONLY magnitude the clause at line 481 requires is > 0"
        )
        log(
            f"     priced: rng {nk[i].rng:.0f} -> {nk[j].rng:.0f} m   alt {nk[i].alt:.0f} -> "
            f"{nk[j].alt:.0f} m ({nk[j].alt - nk[i].alt:+.0f} m)   d_aspect {ev.d_aspect_deg:+.1f} deg   "
            f"d_rng {ev.d_rng_m:+.0f} m   d_es {ev.d_es_m:+.0f} m"
        )
        log(
            f"     nz over the hold {min(nk[k].nz for k in range(i, j + 1)):.2f}-"
            f"{max(nk[k].nz for k in range(i, j + 1)):.2f} g   "
            f"|chi| {min(abs(nk[k].chi) for k in range(i, j + 1)) * DEG:.1f}-"
            f"{max(abs(nk[k].chi) for k in range(i, j + 1)) * DEG:.1f} deg   "
            f"lethal ticks inside {sum(1 for k in range(i, j + 1) if nk[k].lethal)}"
        )
    # how large a build requirement kills it
    log("")
    log("   sensitivity of that event to a MINIMUM-BUILD requirement (m and % of entry CPA):")
    for i, j in B._runs(
        [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in nk]
    ):
        if (nk[j].t - nk[i].t) < B.CHI_MIN_HOLD_S:
            continue
        b = nk[j].cpa_miss - nk[i].cpa_miss
        log(
            f"     hold {i}-{j}: build {b:+.1f} m  ({b / nk[i].cpa_miss * 100:+.2f}%)  -> "
            f"survives a >0 m bar: {b > 0}; a >50 m bar: {b > 50}; a >10% bar: {b > 0.10 * nk[i].cpa_miss}"
        )

    # -------------------------------------------------------------------------------------
    log("")
    log("B. IS D1 CLAUSE (f) SATISFIABLE ON THE KILL TRACE?  (needs d_aspect <= -15.0 deg)")
    asp = np.array([s.aspect for s in kill]) * DEG
    n = len(asp)
    best = 0.0
    arg = (0, 0)
    for i in range(n):
        k = int(np.argmin(asp[i:])) + i
        if asp[k] - asp[i] < best:
            best, arg = asp[k] - asp[i], (i, k)
    log(
        f"   aspect: step0 {asp[0]:.1f} deg, min {asp.min():.1f} deg at step {int(asp.argmin())}, "
        f"max {asp.max():.1f} deg at step {int(asp.argmax())}, last {asp[-1]:.1f} deg"
    )
    log(
        f"   most negative d_aspect over ALL (i<=j) pairs: {best:+.1f} deg, steps {arg[0]}->{arg[1]} "
        f"(t {kill[arg[0]].t:.3f}->{kill[arg[1]].t:.3f} s, rng {kill[arg[0]].rng:.0f}->"
        f"{kill[arg[1]].rng:.0f} m)"
    )
    log(
        f"   excluding the final 2 ticks (the fly-through at rng < 30 m): "
        f"{min(asp[j] - asp[i] for i in range(n - 2) for j in range(i, n - 2)):+.1f} deg"
    )
    log(
        "   => on the KILL trace the aspect-gain clause is not merely unmet, it is essentially"
        " UNSATISFIABLE: this pass never moves toward the bandit's six."
    )

    # -------------------------------------------------------------------------------------
    log("")
    log("C. WHAT THE KILL TRACE IS, in the detector's own scalars")
    log("   the single loaded run (nz > 3 g), which is the whole manoeuvre:")
    m_nz = [s.nz > B.NZ_LOADED for s in kill]
    i, j = max(B._runs(m_nz), key=lambda ij: ij[1] - ij[0])
    log(
        f"     steps {i}-{j}  t {kill[i].t:.3f}-{kill[j].t:.3f} s  ({kill[j].t - kill[i].t:.3f} s)  "
        f"nz {min(kill[k].nz for k in range(i, j + 1)):.2f}-"
        f"{max(kill[k].nz for k in range(i, j + 1)):.2f} g  "
        f"alt {kill[i].alt:.0f} -> {kill[j].alt:.0f} m ({kill[j].alt - kill[i].alt:+.0f} m)"
    )
    log(
        f"     {'step':>5} {'t_s':>6} {'nz':>5} {'phi':>7} {'chi':>7} {'gam':>7} {'ata':>6} "
        f"{'asp':>6} {'rng':>6} {'cpa':>7} {'hca':>6} {'lead':>6} {'areq':>6} {'alt':>6}"
    )
    by_step = {int(r["step"]): r for r in kill_rows}
    for k in list(range(i, j + 1, 12)) + [j, len(kill) - 1]:
        r = by_step[k]
        log(
            f"     {k:>5} {kill[k].t:>6.3f} {kill[k].nz:>5.2f} {kill[k].phi * DEG:>+7.1f} "
            f"{kill[k].chi * DEG:>+7.1f} {kill[k].gamma * DEG:>+7.1f} {kill[k].ata * DEG:>6.1f} "
            f"{kill[k].aspect * DEG:>6.1f} {kill[k].rng:>6.0f} {kill[k].cpa_miss:>7.1f} "
            f"{float(r['hca_deg']):>6.1f} {float(r['lead_error_deg']):>6.1f} "
            f"{float(r['alpha_req_deg']):>6.1f} {kill[k].alt:>6.0f}"
        )
    log("")
    log("   sign-constant loaded |chi| runs on the kill trace (the D4 chi-hold family), priced:")
    for bar in (20.0, 30.0, 45.0, 60.0):
        mask = [abs(s.chi) > math.radians(bar) and s.nz > B.NZ_LOADED for s in kill]
        out = []
        for a, b in B._runs(mask):
            s0 = 1 if kill[a].chi > 0 else -1
            e = a
            while e + 1 <= b and (1 if kill[e + 1].chi > 0 else -1) == s0:
                e += 1
            out.append((a, e))
        if out:
            a, e = max(out, key=lambda ab: ab[1] - ab[0])
            log(
                f"     |chi| > {bar:>4.0f} deg & nz>3, sign-constant: longest {e - a + 1} ticks = "
                f"{kill[e].t - kill[a].t:.3f} s (steps {a}-{e}, t {kill[a].t:.3f}-{kill[e].t:.3f} s), "
                f"dalt {kill[e].alt - kill[a].alt:+.0f} m, chi {kill[a].chi * DEG:+.1f} -> "
                f"{kill[e].chi * DEG:+.1f} deg"
            )
        else:
            log(f"     |chi| > {bar:>4.0f} deg & nz>3: no run")
    log("")
    log("   hold-duration convention: the detectors measure tr[j].t - tr[i].t, which understates a")
    log(f"   K-tick run by exactly one control step ({CDT * 1e3:.3f} ms).")
    for k in (3, 13, 21):
        log(f"     {k} ticks -> physical {k * CDT:.3f} s, detector reports {(k - 1) * CDT:.3f} s")

    # -------------------------------------------------------------------------------------
    log("")
    log("D. D2 RETURN CLAUSE, evaluated past its own window (is the kill truncating it?)")
    g0 = kill[0].gamma
    log(
        f"   entry g0 = {g0 * DEG:+.2f} deg at step 0; YOYO_RETURN_S window ends at t = 6.000 s "
        f"(step 144), the episode ends at t = {kill[-1].t:.3f} s (step {len(kill) - 1})"
    )
    for k in (109, 144, len(kill) - 1):
        log(
            f"     step {k:>4} t {kill[k].t:>6.3f} s: gamma {kill[k].gamma * DEG:>+7.2f} deg, "
            f"|gamma - g0| = {abs(kill[k].gamma - g0) * DEG:.2f} deg "
            f"(bar {B.YOYO_GAMMA_RETURN_DEG:.0f} deg)"
        )
    log(
        "   => the window expired 1.167 s BEFORE termination and the flight path was still 23.89 deg"
        " from g0; extending the window to the last tick still leaves 18.03 deg. The kill does not"
        " truncate this clause — the flight path simply never comes back."
    )
    log("")
    log("E. D2c (clean) — the bleed half, on the kill trace")
    log(
        f"   closure at the last tick {kill[-1].closure:+.0f} m/s vs CLOSURE_OVERSHOOT "
        f"{B.CLOSURE_OVERSHOOT:.0f}; min closure over the whole trace "
        f"{min(s.closure for s in kill):+.0f} m/s"
    )
    log(
        "   => `bled = closure_exit < 120 and rng_exit < 1500` can never be True on this trace: the"
        " closure never drops below +373 m/s. Any D2 event registered by loosening the shape clauses"
        " would still be tagged failed_exit, so D2c = 0 is doubly earned."
    )
    log("")
    log(f"   files read: {os.path.abspath('runs/d3b/replay_kill_seed62.csv')}")


if __name__ == "__main__":
    main()
