"""Which D1 clause rejects `recorded_two` -- the ONLY control arm that reaches the D1 hold bar?

Clause letters follow the docstring of `bfm_detect.detect_displacement_rolls`:
  (a) in-plane (<CHI_INPLANE_DEG) within CHI_LOOKBACK_S before entry
  (c) closure > 0 at entry
  (d) |chi| returns under CHI_RETURN_DEG within CHI_RETURN_S after the hold
  (e) sign(ata_az) constant across entry..exit
  (f) aspect improves by >= ASPECT_GAIN_DEG
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "/workspace/controllers/rl")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

DEG = 180.0 / math.pi
CELL = M.CELLS["merge1500"]
NSTEPS = 576
DT = M.CONTROL_DT_S
OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def runs_of(mask):
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


def audit(arm, seed):
    env, base = M.build_env(CELL, seed, NSTEPS * DT)
    r = M.roll_once(env, base, M.make_control(arm), NSTEPS, keep=True)
    env.close()
    tr = r.trace
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    log(f"   === {arm} seed {seed}: D1 clause-by-clause on every loaded out-of-plane run ===")
    for i, j in runs_of(hold):
        dur = tr[j].t - tr[i].t
        log(
            f"     run steps {i}-{j}  t {tr[i].t:.3f}-{tr[j].t:.3f} s  dur {dur:.3f} s  "
            f"|chi| {abs(tr[i].chi) * DEG:.1f}->{abs(tr[j].chi) * DEG:.1f}  "
            f"nz {min(tr[k].nz for k in range(i, j + 1)):.2f}-{max(tr[k].nz for k in range(i, j + 1)):.2f} g  "
            f"alt {tr[i].alt:.0f}->{tr[j].alt:.0f} m ({tr[j].alt - tr[i].alt:+.0f})"
        )
        if dur < B.CHI_MIN_HOLD_S:
            log(f"       HOLD:  {dur:.3f} s < {B.CHI_MIN_HOLD_S} s  -> rejected on DURATION")
            continue
        log(f"       HOLD:  {dur:.3f} s >= {B.CHI_MIN_HOLD_S} s  PASS")
        # (a)
        look = [k for k in range(i - 1, -1, -1) if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S]
        a_ok = any(abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG) for k in look)
        mn = min((abs(tr[k].chi) * DEG for k in look), default=float("nan"))
        log(
            f"       (a) in-plane <{B.CHI_INPLANE_DEG:.0f} deg within {B.CHI_LOOKBACK_S} s "
            f"lookback: min|chi| in window = {mn:.1f} deg -> {'PASS' if a_ok else 'FAIL'}"
        )
        # (c)
        c_ok = tr[i].closure > 0.0
        log(
            f"       (c) closure at entry = {tr[i].closure:+.1f} m/s -> {'PASS' if c_ok else 'FAIL'}"
        )
        # (d)
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        log(
            f"       (d) |chi| back under {B.CHI_RETURN_DEG:.0f} deg within {B.CHI_RETURN_S} s: "
            f"{'PASS at step ' + str(exit_idx) if exit_idx is not None else 'FAIL'}"
        )
        if exit_idx is None:
            continue
        # (e)
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
        log(
            f"       (e) sign(ata_az) constant i..exit: {sorted(sides)} -> {'PASS' if len(sides) == 1 else 'FAIL'}"
        )
        # (f)
        gain = (tr[i].aspect - tr[exit_idx].aspect) * DEG
        log(
            f"       (f) aspect gain = {gain:+.1f} deg (need >= {B.ASPECT_GAIN_DEG:.0f}) "
            f"[{tr[i].aspect * DEG:.1f} -> {tr[exit_idx].aspect * DEG:.1f}] "
            f"-> {'PASS' if gain >= B.ASPECT_GAIN_DEG else 'FAIL'}"
        )
        log(
            f"       PRICE: alt {tr[i].alt:.0f} -> {tr[exit_idx].alt:.0f} m "
            f"({tr[exit_idx].alt - tr[i].alt:+.0f} m), rng {tr[i].rng:.0f} -> {tr[exit_idx].rng:.0f} m, "
            f"es_rel {tr[i].es_rel:+.0f} -> {tr[exit_idx].es_rel:+.0f} m"
        )
    log("")


def main():
    for arm in ("recorded_two", "recorded", "inplane"):
        audit(arm, 62)
    with open("/workspace/controllers/rl/runs/d3b/refute_arms_clause2.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
