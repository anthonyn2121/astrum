"""Negative control for the PROPOSED 'lead-turn detector'.

Fly the SAME cell (merge1500, sim_seed 62, 24 s) with controllers that contain NO lead turn by
construction, then run the proposed detector spec verbatim on each trace.

  PP  pure pursuit: null `chi` (put the lift vector ON the bandit) and pull max g. Zero lead --
      the nose chases the bandit's PRESENT position, never its future one.
  PPm pure pursuit at half load.
  SL  wings-level, no pull. Sanity check that the nz>3 gate does something.
"""

import math
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import bfm_detect as B

NZ_LOADED, MIN_HOLD_S, PASS_RNG = B.NZ_LOADED, 0.5, B.LETHAL_RANGE_M
N_STEPS = int(round(24.0 * M.RATE_HZ / M.ACTION_REPEAT))


def runs(f):
    out = []
    i = 0
    n = len(f)
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and f[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def proposed(rec, label):
    t = rec["t"]
    nz = rec["nz"]
    az = rec["az"]
    ar = rec["ar"]
    cpa = rec["cpa"]
    rng = rec["rng"]
    clo = rec["clo"]
    alt = rec["alt"]
    asp = rec["asp"]
    chi = rec["chi"]
    ev = []
    for i, j in runs([x > NZ_LOADED for x in nz]):
        if t[j] - t[i] < MIN_HOLD_S:
            continue
        if clo[i] <= 0.0:
            continue
        if rng[i] <= PASS_RNG:
            continue
        flips = [k for k in range(i, j) if (az[k] >= 0) != (az[k + 1] >= 0)]
        if not flips:
            continue
        if not ar[j] < ar[i]:
            continue
        if not cpa[j] < cpa[i]:
            continue
        ev.append((i, j, flips))
    print(f"  {label}: PROPOSED LEAD-TURN EVENTS = {len(ev)}")
    for i, j, flips in ev:
        print(
            f"    [{i:3d},{j:3d}] t {t[i]:.3f}-{t[j]:.3f}s dur={t[j] - t[i]:.3f}s nzmax={max(nz[i : j + 1]):.2f}g "
            f"max|chi|={max(abs(math.degrees(c)) for c in chi[i : j + 1]):.1f}deg"
        )
        print(
            f"        alpha_req {math.degrees(ar[i]):.2f}->{math.degrees(ar[j]):.2f}  "
            f"cpa {cpa[i]:.1f}->{cpa[j]:.1f}  aspect {math.degrees(asp[i]):.2f}->{math.degrees(asp[j]):.2f} "
            f"(d={math.degrees(asp[j] - asp[i]):+.2f})  dalt={alt[j] - alt[i]:+.1f}m  "
            f"flip|ata_az|={[round(math.degrees(min(abs(az[k]), abs(az[k + 1]))), 3) for k in flips]}"
        )
    return len(ev)


def make(kind):
    rec = {k: [] for k in ("t", "nz", "az", "ar", "cpa", "rng", "clo", "alt", "asp", "chi")}

    def ctrl(step, me, bd, rel):
        chi, valid = B.out_of_plane_angle(me, bd)
        rec["t"].append(step * M.ACTION_REPEAT / M.RATE_HZ)
        rec["nz"].append(float(me.nz))
        rec["az"].append(float(rel.ata_az))
        rec["ar"].append(float(rel.alpha_req))
        rec["rng"].append(float(rel.rng))
        rec["clo"].append(float(rel.closure))
        rec["alt"].append(float(me.pos[2]))
        rec["asp"].append(float(rel.aspect))
        rec["chi"].append(chi)
        rec["cpa"].append(B.cpa_miss_m(me.pos, me.vel, bd.pos, bd.vel))
        if kind == "SL":
            ail = float(np.clip(bfm.KROLL * (0.0 - float(me.phi)), -1.0, 1.0))
            return np.array([ail, 0.0, 0.0, 0.6, -1.0], np.float32)
        load = 1.0 if kind == "PP" else 0.5
        ail = float(np.clip(-bfm.KROLL * chi, -1.0, 1.0)) if valid else 0.0
        mag = max(math.cos(chi), 0.0) if valid else 1.0
        elev = -float(np.clip(load * mag, 0.0, 1.0))
        return np.array([ail, elev, 0.0, 1.0, -1.0], np.float32)

    return ctrl, rec


env, base = M.build_env(M.CELLS["merge1500"], 62, 24.0)
for kind in ("PP", "PPm", "SL"):
    ctrl, rec = make(kind)
    r = M.roll_once(env, base, ctrl, N_STEPS, keep=False)
    d = B.detect_all(r.trace)
    print(
        f"\n=== {kind}: steps={r.steps} event={r.event} kill={r.kill} hp_dealt={r.hp_dealt:.1f} "
        f"rng_min={r.rng_min:.1f} dalt={r.dalt_m:.1f} | DEFAULT detectors "
        f"D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])} D2c={len(d['yoyos_clean'])} "
        f"D3={len(d['turning_room'])} ==="
    )
    proposed(rec, kind)
env.close()
