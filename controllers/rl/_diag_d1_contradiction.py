"""Resolve the contradiction found by _diag_d3b_express.py P2/P3.

On the L=4.0 s iter-2 sweep trace, my hand walk of D1's six clauses says the hold at steps 127-189
PASSES ALL CLAUSES, while `B.detect_all` reports displacement_rolls = 0 on the same trace object.
One of the two is wrong. This rebuilds that exact trace (the sim is deterministic, so repeating the
iteration chain reproduces it bit for bit) and steps through `detect_displacement_rolls` verbatim,
printing every intermediate the real function computes.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B


def chain(L: float, iters: int = 3):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    g = np.load(f"{E.OUT}/d3b_best.npz")["genome"]

    class GenomeCtl:
        def cmd(self, t):
            s = gm.seg_of(int(round(t / E.DT)), E.N_STEPS)
            p = g[s * 3 : (s + 1) * 3]
            return (
                M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)),
                0.5 * (float(np.clip(p[1], -1, 1)) + 1.0),
                0.5 * (float(np.clip(p[2], -1, 1)) + 1.0),
            )

        def __call__(self, step, me, bd, rel):
            return gm.act(g, step, E.N_STEPS, me)

    r, rows = E.fly(GenomeCtl())
    last = None
    for _ in range(iters):
        t, req = E.build_required_bank(rows, E.CHI_TGT_DEG)
        phases = E.project(t, req, L, E.HORIZON_S)
        r, rows = E.fly(E.Sched(phases))
        last = (r, rows, phases)
    return last


def walk(tr) -> None:
    """`detect_displacement_rolls` re-executed line by line with every intermediate printed."""
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    print(f"  trace len {len(tr)}, hold mask true on {sum(hold)} ticks, runs:")
    for i, j in B._runs(hold):
        dur = tr[j].t - tr[i].t
        print(f"\n  RUN steps {i}-{j}  t {tr[i].t:.3f}-{tr[j].t:.3f}  dur {dur:.3f} s")
        if dur < B.CHI_MIN_HOLD_S:
            print(f"    REJECT duration {dur:.3f} < {B.CHI_MIN_HOLD_S}")
            continue
        lookback = [
            k
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
            and abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
        ]
        print(f"    (a) in-plane lookback hits: {lookback[:5]} -> {'ok' if lookback else 'REJECT'}")
        if not lookback:
            continue
        print(f"    (c) closure[{i}] = {tr[i].closure:+.2f} -> {'ok' if tr[i].closure > 0 else 'REJECT'}")
        if tr[i].closure <= 0.0:
            continue
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                print(f"    (d) scan stopped at k={k} (t-tj = {tr[k].t - tr[j].t:.3f} > {B.CHI_RETURN_S})")
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        print(f"    (d) exit_idx = {exit_idx}" + (f" t {tr[exit_idx].t:.3f} |chi| {math.degrees(abs(tr[exit_idx].chi)):.1f}" if exit_idx else " -> REJECT"))
        if exit_idx is None:
            continue
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
        print(f"    (e) ata_az sides over {i}..{exit_idx} = {sides} -> {'ok' if len(sides) == 1 else 'REJECT'}")
        if len(sides) != 1:
            continue
        d_aspect = math.degrees(tr[exit_idx].aspect - tr[i].aspect)
        print(
            f"    (f) aspect {math.degrees(tr[i].aspect):.2f} -> {math.degrees(tr[exit_idx].aspect):.2f} "
            f"= {d_aspect:+.2f} deg; needs <= {-B.ASPECT_GAIN_DEG:.0f} -> "
            f"{'ok' if d_aspect < -B.ASPECT_GAIN_DEG else 'REJECT'}"
        )
        if d_aspect >= -B.ASPECT_GAIN_DEG:
            continue
        print("    ==> EVENT (all six clauses pass)")


def main() -> None:
    for L in (4.0, 1.0):
        print("=" * 100)
        print(f"  L = {L} s, iteration chain length 3")
        print("=" * 100)
        r, rows, phases = chain(L)
        print(f"  steps {r.steps} event {r.event}")
        print("  phases (duration_s, bank_deg, load, thr):")
        for k, p in enumerate(phases[:8]):
            print(f"    {k}: {p[0]:.2f} s bank {p[1]:+8.2f} load {p[2]:.2f} thr {p[3]:.2f}")
        d1 = B.detect_displacement_rolls(r.trace)
        det = B.detect_all(r.trace)
        print(
            f"\n  B.detect_displacement_rolls -> {len(d1)} event(s); "
            f"B.detect_all['displacement_rolls'] -> {len(det['displacement_rolls'])}"
        )
        for ev in d1:
            print(f"    {ev.kind} i {ev.i0}-{ev.i1} t {ev.t0:.3f}-{ev.t1:.3f} d_aspect {ev.d_aspect_deg:+.1f} detail {ev.detail}")
        print("\n  VERBATIM WALK:")
        walk(r.trace)
        print()


if __name__ == "__main__":
    main()
