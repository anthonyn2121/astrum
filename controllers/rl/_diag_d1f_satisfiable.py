"""Is D1 clause (f) satisfiable at HEAD-ON / HIGH ASPECT in the D3b search space?

Independent reproduction. Flies two 18-dim genomes at CELL merge1500, sim seed 62, through the
REAL `Genome.act` / `roll_once` (nothing reimplemented), then runs the shipped
`B.detect_displacement_rolls` and reports, per event, the ENTRY aspect and the d_aspect that
clause (f) tests -- plus the range profile inside the window so a "fly-through" gain can be told
apart from a turn.

Genome A = the incumbent kill genome, read from runs/d3b/d3b_best.npz.
Genome B = the 18-dim encoding printed in runs/d3b/d1_exists_report.txt (retyped here from that
           report; the point of this script is to fly it MYSELF and see what the detector says).
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = math.degrees

G_INCUMBENT = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"].astype(float)

# retyped from d1_exists_report.txt "ENCODED 18-dim genome"
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
    dtype=float,
)


def fly(g: np.ndarray, seed: int):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(E.CELL, seed, E.N_STEPS * E.DT)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, E.N_STEPS, me)

    r = M.roll_once(env, base, ctl, E.N_STEPS, keep=True)
    env.close()
    return r


def report(label: str, g: np.ndarray, seed: int) -> None:
    r = fly(g, seed)
    tr = r.trace
    asp = np.array([DEG(s.aspect) for s in tr])
    rng = np.array([s.rng for s in tr])
    det = B.detect_all(tr)
    print("\n" + "=" * 96)
    print(
        f"  {label}   cell=merge1500  seed={seed}  steps={len(tr)}  event={r.event}  kill={r.kill}"
    )
    print("=" * 96)
    print(
        f"  D1={len(det['displacement_rolls'])} D2={len(det['yoyos'])} "
        f"D2c={len(det['yoyos_clean'])} D3={len(det['turning_room'])} lethTk={det['lethal_ticks']}"
    )
    print(
        f"  aspect: step0 {asp[0]:.1f}  min {asp.min():.1f}  max {asp.max():.1f}  last {asp[-1]:.1f}"
    )
    print(f"  rng: min {rng.min():.0f} m @step {int(np.argmin(rng))}  last {rng[-1]:.0f} m")
    for ev in det["displacement_rolls"]:
        i, j = ev.i0, ev.i1
        seg_rng = rng[i : j + 1]
        print(
            f"  D1 EVENT i0={i} i1={j}  t {ev.t0:.3f}-{ev.t1:.3f}\n"
            f"     ENTRY aspect {asp[i]:.1f} deg  rng {tr[i].rng:.0f} m  closure {tr[i].closure:+.0f} m/s\n"
            f"     EXIT  aspect {asp[j]:.1f} deg  rng {tr[j].rng:.0f} m  closure {tr[j].closure:+.0f} m/s\n"
            f"     clause (f) d_aspect = {ev.d_aspect_deg:+.1f} deg   bar <= {-B.ASPECT_GAIN_DEG:+.1f} deg   "
            f"-> {'PASS' if ev.d_aspect_deg < -B.ASPECT_GAIN_DEG else 'FAIL'}"
            f"  (margin {abs(ev.d_aspect_deg) / B.ASPECT_GAIN_DEG:.1f}x)\n"
            f"     hold {ev.detail['hold_s']:.3f} s  chi_peak {ev.detail['chi_peak_deg']:.1f} deg  "
            f"nz_peak {ev.detail['nz_peak']:.2f} g  ALT PRICE {tr[j].alt - tr[i].alt:+.0f} m  "
            f"d_es {ev.d_es_m:+.0f} m\n"
            f"     rng inside the window: min {seg_rng.min():.0f} m @step {i + int(np.argmin(seg_rng))} "
            f"(a fly-through if this is far below the endpoints)"
        )
    # how big an ASPECT_GAIN bar would this trace still clear?
    saved = B.ASPECT_GAIN_DEG
    try:
        line = []
        for bar in (15.0, 30.0, 60.0, 100.0, 140.0, 160.0):
            B.ASPECT_GAIN_DEG = bar
            line.append(f"{bar:.0f}->{len(B.detect_displacement_rolls(tr))}")
        print("  D1 count vs ASPECT_GAIN_DEG bar:  " + "  ".join(line))
    finally:
        B.ASPECT_GAIN_DEG = saved
    print(f"  restored ASPECT_GAIN_DEG={B.ASPECT_GAIN_DEG}")


report("GENOME A - incumbent kill genome (npz)", G_INCUMBENT, 62)
report("GENOME B - the D1 genome from d1_exists_report.txt", G_D1, 62)
