"""HOLD-DURATION CONVENTION AUDIT — is the one-control-step understatement real, and is it material?

The claim under test: "the detectors measure `tr[j].t - tr[i].t`, which understates a K-tick run by
exactly one control step (41.667 ms)", asserted non-material.

Two things the replay CSV cannot answer, and this script therefore re-flies rather than re-reads:
  * `_diag_d3b_replay.py:180` builds its own sample with `B.make_sample(step * M.CONTROL_DT_S, ...)`
    — a MULTIPLIED time. The search's `roll_once` (`_diag_true_mpc.py:289,307`) feeds an ACCUMULATED
    `t += CONTROL_DT_S`. The two differ by up to ~1e-14 s, which is exactly the size that decides
    `tr[j].t - tr[i].t < CHI_MIN_HOLD_S` for a 13-tick run.
  * Whether the convention flips any verdict needs the run-length HISTOGRAM over all eight search
    seeds, not the two dumped ones. A flip needs a run at K = 12 (or K = 13 under accumulation).

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_holdconv.py'
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
NPZ = "runs/d3b/d3b_best.npz"
N_STEPS = 576
DT = M.CONTROL_DT_S


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


def sign_runs(tr, value, bar, need_valid):
    """`commitment._longest_hold`'s run structure, verbatim, but returning ALL runs."""

    def ok(s):
        if need_valid and not s.chi_valid:
            return False
        return abs(value(s)) > bar and s.nz > B.NZ_LOADED

    out, i, n = [], 0, len(tr)
    while i < n:
        if not ok(tr[i]):
            i += 1
            continue
        s = 1 if value(tr[i]) > 0 else -1
        j = i
        while j + 1 < n and ok(tr[j + 1]) and (1 if value(tr[j + 1]) > 0 else -1) == s:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def flip(tr, i, j):
    """Does the convention change the CHI_MIN_HOLD_S verdict for this run?"""
    k = j - i + 1
    endpoint_pass = not (tr[j].t - tr[i].t < B.CHI_MIN_HOLD_S)
    count_pass = not (k * DT < B.CHI_MIN_HOLD_S)
    return endpoint_pass != count_pass


def report(tag, tr, rr):
    for i, j in sorted(rr, key=lambda ij: -(ij[1] - ij[0])):
        k = j - i + 1
        ep = tr[j].t - tr[i].t
        print(
            f"    {tag:22s} steps {i:4d}-{j:4d} K={k:3d} ticks"
            f"  endpoint {ep!r:24s} ({ep:.4f} s)  K*dt {k * DT:.4f}"
            f"  pass{'Y' if not ep < B.CHI_MIN_HOLD_S else 'n'}"
            f"  FLIP={flip(tr, i, j)}"
        )
    return [(j - i + 1) for i, j in rr]


def main() -> None:
    g = np.load(NPZ, allow_pickle=False)["genome"]
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    cell = M.CELLS["merge1500"]
    all_k: dict[str, list[int]] = {"D1(chi>60)": [], "D3(chi>45)": [], "D4bank": [], "D4chi": []}
    flips = 0
    for sd in SEEDS:
        env, base = M.build_env(cell, sd, N_STEPS * DT)
        r = M.roll_once(env, base, lambda st, me, bd, rel: gm.act(g, st, N_STEPS, me), N_STEPS)
        env.close()
        tr = r.trace
        n = len(tr)
        drift = max(abs(tr[k].t - k * DT) for k in range(n))
        print(
            f"\nSEED {sd:4d}  event={r.event!s:9s} steps={n:4d}  tLeth={r.tleth_s:.3f} s"
            f"  max|t_accumulated - step*dt| = {drift:.3e} s   t[-1]={tr[-1].t!r}"
        )
        d = B.detect_all(tr)
        print(
            f"    detect_all: D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])}"
            f" D2c={len(d['yoyos_clean'])} D3={len(d['turning_room'])}"
            f" lethalTicks={d['lethal_ticks']}"
        )
        for ev in d["turning_room"] + d["displacement_rolls"]:
            print(
                f"    EVENT {ev.kind:16s} i0={ev.i0} i1={ev.i1}"
                f"  hold_s={ev.detail.get('hold_s', float('nan')):.4f}"
                f"  duration_s={ev.duration_s:.4f}"
            )
        m1 = [abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED for s in tr]
        m3 = [abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED for s in tr]
        all_k["D1(chi>60)"] += report("D1 mask chi>60", tr, runs(m1))
        all_k["D3(chi>45)"] += report("D3 mask chi>45", tr, runs(m3))
        all_k["D4bank"] += report(
            "D4 |phi|>45 signconst",
            tr,
            sign_runs(tr, lambda s: s.phi, math.radians(B.BANK_COMMIT_DEG), False),
        )
        all_k["D4chi"] += report(
            "D4 |chi|>45 signconst",
            tr,
            sign_runs(tr, lambda s: s.chi, math.radians(B.CHI_ROOM_DEG), True),
        )
        flips += sum(
            flip(tr, i, j)
            for rr in (
                runs(m1),
                runs(m3),
                sign_runs(tr, lambda s: s.phi, math.radians(B.BANK_COMMIT_DEG), False),
                sign_runs(tr, lambda s: s.chi, math.radians(B.CHI_ROOM_DEG), True),
            )
            for i, j in rr
        )
    print("\n" + "=" * 100)
    print("RUN-LENGTH HISTOGRAM over all 8 seeds (K = ticks in the maximal run)")
    for k, v in all_k.items():
        print(f"  {k:12s} n_runs={len(v):3d}  K values: {sorted(v, reverse=True)}")
        print(f"               K == 12 (always flips): {v.count(12)}   K == 13: {v.count(13)}")
    print(f"  TOTAL verdict FLIPS between the two conventions, all masks, all seeds: {flips}")

    print("\nBOUNDARY PHASE TEST on the ACCUMULATED t the search actually feeds the detector")
    for n in (173, 576):
        ts, x = [], 0.0
        for _ in range(n):
            ts.append(x)
            x += DT
        for k in (12, 13, 14):
            gap = k - 1
            offs = n - gap
            rej = sum(1 for i in range(offs) if ts[i + gap] - ts[i] < 0.5)
            lo = min(ts[i + gap] - ts[i] for i in range(offs))
            hi = max(ts[i + gap] - ts[i] for i in range(offs))
            print(
                f"  n={n:4d} K={k:3d}: rejected by '< CHI_MIN_HOLD_S' at {rej:4d}/{offs} start"
                f" offsets   endpoint diff in [{lo!r}, {hi!r}]"
            )


if __name__ == "__main__":
    main()
