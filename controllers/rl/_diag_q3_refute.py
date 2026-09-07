"""Q3 REFUTATION CHECK -- is the minimal D1 three-phase shape writable at 4.0 s segment grain?

The claim under test: "the MINIMAL D1 shape needs ~1.83 s across 3 phases (0.667 + 0.500 + 0.667),
i.e. ~0.61 s phase resolution, so at 4.0 s grain the minimal three-phase form is NOT writable
phase-by-phase."

Everything here is imported from `_diag_true_mpc` (Genome, roll_once, build_env, CELLS) and the real
`bfm_detect`, so nothing is reimplemented.

Three measurements:
  (1) Re-fly the 4.0 s-grain genome recorded in runs/d3b/d1_exists_report.txt and ask the REAL
      detector for D1. If D1 >= 1 the "not writable at 4.0 s grain" half of the claim is dead.
  (2) For each detected D1, decompose the event into the three phases the claim names and report
      each phase's WALL TIME and the SEGMENT INDEX it lives in, plus the altitude price of the hold
      and the min nz inside it (a hold must be loaded to count).
  (3) Truncate the genome to 3 segments of content (segs 3-5 replaced by an unloaded wings-level
      command) to test whether THREE 4.0 s phases suffice.

Also recomputes the claim's own arithmetic against the detector constants and measures how fast
`chi` (a RELATIVE angle) can actually slew, which is the step the claim converts with a BANK roll
time.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_q3_refute.py'
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

SEGMENTS = 6
CHANNELS = 3
N_STEPS = 576
CELL = M.CELLS["merge1500"]
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]

# The genome printed in runs/d3b/d1_exists_report.txt ("ENCODED 18-dim genome"), 4.0 s segments.
G_D1 = np.array(
    [
        1.0,
        1.0,
        0.2,
        0.582792,
        1.0,
        0.2,
        1.0,
        1.0,
        0.2,
        0.872769,
        1.0,
        0.2,
        -0.007301,
        1.0,
        0.2,
        0.312102,
        1.0,
        0.2,
    ],
    float,
)
# Same first three segments; segs 3-5 blanked to wings-level UNLOADED (load 0 -> elev 0).
G_D1_3SEG = G_D1.copy()
for s in (3, 4, 5):
    G_D1_3SEG[3 * s : 3 * s + 3] = [0.0, -1.0, 0.2]

# The incumbent, for reference (runs/d3b/d3b_best.npz).
G_INC = np.array(
    [
        1.0,
        1.0,
        0.20466533,
        0.42106359,
        0.04260145,
        1.0,
        -1.0,
        0.81890272,
        -1.0,
        -0.45391245,
        -0.49038442,
        0.90924598,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        0.7947789,
        0.53075154,
    ],
    float,
)

GM = M.Genome(SEGMENTS, CHANNELS, "bank", True)


def fly(g: np.ndarray, seed: int):
    env, base = M.build_env(CELL, seed, N_STEPS * M.CONTROL_DT_S)

    def ctl(step, me, bd, rel):
        return GM.act(g, step, N_STEPS, me)

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    return r


def seg_of(step: int) -> int:
    return GM.seg_of(step, N_STEPS)


def phase_report(tr, ev) -> str:
    """Decompose a detected displacement roll into the claim's three phases, measured."""
    i0, i1 = ev.i0, ev.i1  # i0 = hold start, i1 = the RETURN index (exit_idx)
    # recover the hold end: the last tick of the loaded out-of-plane run starting at i0
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    j = i0
    while j + 1 < len(tr) and hold[j + 1]:
        j += 1
    # phase 1: the most recent in-plane tick before the hold (clause a)
    k_in = None
    for k in range(i0 - 1, -1, -1):
        if tr[i0].t - tr[k].t > B.CHI_LOOKBACK_S:
            break
        if abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG):
            k_in = k
            break
    nz_hold = [tr[k].nz for k in range(i0, j + 1)]
    lines = [
        f"    HOLD      steps {i0}-{j}  t {tr[i0].t:6.3f}-{tr[j].t:6.3f}  "
        f"dur {tr[j].t - tr[i0].t:5.3f} s  segs {seg_of(i0)}-{seg_of(j)}  "
        f"nz {min(nz_hold):.2f}-{max(nz_hold):.2f} g  "
        f"|chi| {abs(math.degrees(tr[i0].chi)):.1f}->{abs(math.degrees(tr[j].chi)):.1f}",
        f"    ALT PRICE of the hold: {tr[j].alt - tr[i0].alt:+.0f} m over "
        f"{tr[j].t - tr[i0].t:.3f} s",
    ]
    if k_in is None:
        lines.append("    PHASE 1 (in-plane lookback): NONE FOUND (should be impossible)")
    else:
        lines.append(
            f"    PHASE 1 in-plane tick  step {k_in} t {tr[k_in].t:6.3f} "
            f"|chi| {abs(math.degrees(tr[k_in].chi)):.1f} seg {seg_of(k_in)}  "
            f"-> gap to hold start {tr[i0].t - tr[k_in].t:5.3f} s "
            f"(deadline CHI_LOOKBACK_S {B.CHI_LOOKBACK_S:.1f} s)"
        )
    lines.append(
        f"    PHASE 3 return tick    step {i1} t {tr[i1].t:6.3f} "
        f"|chi| {abs(math.degrees(tr[i1].chi)):.1f} valid {tr[i1].chi_valid} seg {seg_of(i1)}  "
        f"-> gap from hold end {tr[i1].t - tr[j].t:5.3f} s "
        f"(deadline CHI_RETURN_S {B.CHI_RETURN_S:.1f} s)"
    )
    # a validity-gated version of clause (d): ignore the chi=0 sentinel
    k_ret = None
    for k in range(j + 1, len(tr)):
        if tr[k].t - tr[j].t > B.CHI_RETURN_S:
            break
        if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
            k_ret = k
            break
    if k_ret is None:
        lines.append("    PHASE 3 (chi_valid only): NO valid return inside the window")
    else:
        lines.append(
            f"    PHASE 3 (chi_valid only) step {k_ret} t {tr[k_ret].t:6.3f} "
            f"|chi| {abs(math.degrees(tr[k_ret].chi)):.1f} seg {seg_of(k_ret)}  "
            f"-> gap {tr[k_ret].t - tr[j].t:5.3f} s"
        )
    span_t = (tr[i1].t - tr[k_in].t) if k_in is not None else float("nan")
    lines.append(
        f"    THREE-PHASE SPAN (in-plane tick -> return tick) = {span_t:.3f} s; "
        f"segments touched {seg_of(k_in) if k_in is not None else -1}..{seg_of(i1)}"
    )
    lines.append(
        f"    d_aspect {ev.d_aspect_deg:+.1f} deg  d_rng {ev.d_rng_m:+.0f} m  "
        f"d_es {ev.d_es_m:+.0f} m  detail {ev.detail}"
    )
    return "\n".join(lines)


def chi_slew(tr, tag: str) -> None:
    """How fast can `chi` actually move, and does it need roll authority to do it?"""
    mx, mi = 0.0, -1
    for i in range(1, len(tr)):
        if not (tr[i].chi_valid and tr[i - 1].chi_valid):
            continue
        d = abs(abs(math.degrees(tr[i].chi)) - abs(math.degrees(tr[i - 1].chi)))
        rate = d / (tr[i].t - tr[i - 1].t)
        if rate > mx:
            mx, mi = rate, i
    if mi > 0:
        print(
            f"  {tag}: max |d|chi|/dt| over valid pairs = {mx:.1f} deg/s at step {mi} "
            f"t {tr[mi].t:.3f} (nz {tr[mi].nz:.2f} g, |ail| {abs(tr[mi].ail):.2f})"
        )

    def fastest(lo_pred, hi_pred, loaded: bool):
        best, bij = float("inf"), None
        for i, s in enumerate(tr):
            if not s.chi_valid or not lo_pred(abs(math.degrees(s.chi))):
                continue
            for j in range(i + 1, len(tr)):
                if not tr[j].chi_valid:
                    continue
                if loaded and tr[j].nz <= B.NZ_LOADED:
                    break
                if hi_pred(abs(math.degrees(tr[j].chi))):
                    if tr[j].t - tr[i].t < best:
                        best, bij = tr[j].t - tr[i].t, (i, j)
                    break
                if lo_pred(abs(math.degrees(tr[j].chi))):
                    break
        return best, bij

    for name, lo, hi, loaded in (
        ("20->60 (out leg, 40 deg)", lambda c: c < 20.0, lambda c: c > 60.0, False),
        ("20->60 loaded nz>3", lambda c: c < 20.0, lambda c: c > 60.0, True),
        ("60->30 (back leg, 30 deg)", lambda c: c > 60.0, lambda c: c < 30.0, False),
    ):
        best, bij = fastest(lo, hi, loaded)
        if bij is None:
            print(f"  {tag}: fastest {name}: none present in this trace")
        else:
            print(f"  {tag}: fastest {name} = {best:.3f} s (steps {bij[0]}->{bij[1]})")


def audit(tag: str, g: np.ndarray, seeds: list[int]) -> None:
    print("=" * 100)
    print(f"  {tag}")
    print("=" * 100)
    for s in range(SEGMENTS):
        p = g[3 * s : 3 * s + 3]
        print(
            f"    seg {s}  t {4.0 * s:5.1f}-{4.0 * s + 4.0:5.1f} s  "
            f"bank {M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)):+8.2f} deg  "
            f"load {0.5 * (float(np.clip(p[1], -1, 1)) + 1):.4f}  "
            f"thr {0.5 * (float(np.clip(p[2], -1, 1)) + 1):.4f}"
        )
    for seed in seeds:
        r = fly(g, seed)
        d = B.detect_all(r.trace)
        n1 = len(d["displacement_rolls"])
        print(
            f"\n  seed {seed}: steps {r.steps} event={r.event} kill={r.kill} "
            f"hp={r.hp_dealt:.0f} dalt={r.dalt_m:+.0f} m  "
            f"D1={n1} D2={len(d['yoyos'])} D2c={len(d['yoyos_clean'])} "
            f"D3={len(d['turning_room'])}  score={M.score(r)[0]:+.3f}"
        )
        for ev in d["displacement_rolls"]:
            print(f"  D1 event t {ev.t0:.3f}-{ev.t1:.3f} ({ev.duration_s:.3f} s):")
            print(phase_report(r.trace, ev))
        if seed == 62:
            chi_slew(r.trace, f"{tag} seed 62")


def main() -> None:
    print("#" * 100)
    print("#  Q3 CHECK -- minimal D1 shape vs the 4.0 s segment grain")
    print("#" * 100)
    print(
        f"  detector constants (bfm_detect.py:66-74): CHI_INPLANE {B.CHI_INPLANE_DEG} "
        f"CHI_OUT {B.CHI_OUT_DEG} CHI_RETURN {B.CHI_RETURN_DEG} "
        f"CHI_MIN_HOLD_S {B.CHI_MIN_HOLD_S} CHI_LOOKBACK_S {B.CHI_LOOKBACK_S} "
        f"CHI_RETURN_S {B.CHI_RETURN_S} NZ_LOADED {B.NZ_LOADED}"
    )
    print(
        f"  claim's arithmetic: out leg needs {B.CHI_OUT_DEG - B.CHI_INPLANE_DEG:.0f} deg, "
        f"back leg needs {B.CHI_OUT_DEG - B.CHI_RETURN_DEG:.0f} deg "
        f"(the claim says 40 deg for BOTH)"
    )
    print(
        f"  one segment = {N_STEPS // SEGMENTS} steps = "
        f"{N_STEPS // SEGMENTS * M.CONTROL_DT_S:.3f} s; horizon "
        f"{N_STEPS * M.CONTROL_DT_S:.1f} s\n"
    )
    audit("A. 4.0 s-GRAIN D1 WITNESS (runs/d3b/d1_exists_report.txt genome)", G_D1, [62, 28])
    audit("B. SAME, segs 3-5 BLANKED (only THREE 4.0 s phases carry content)", G_D1_3SEG, [62, 28])
    audit("C. INCUMBENT (d3b_best.npz) for reference", G_INC, [62])


if __name__ == "__main__":
    main()
