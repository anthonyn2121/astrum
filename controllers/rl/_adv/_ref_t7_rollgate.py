"""T7 refutation harness — is the unload-to-roll the DECODE's or the SEARCH's, and is it
load-bearing?

Arms, all flown on all eight search seeds with the d3b_best genome held fixed:
  BASE          exact genome, roll_gate=True                 (validation: must reproduce the npz)
  NOGATE        exact genome, roll_gate=False                (elev = -load always: NO unload at all)
  BANK<x>       roll_gate=True, seg-0 bank target overridden (all other segments untouched)

BANK sweep answers the claim's "the maneuver is CLIPPED by the parameterization": if banks past
BANK_MAX_DEG=115 score no better, the clip did not truncate anything. Banks at/below 90 deg remove
the gate's unload entirely (cos(bank_err) >= 0 from step 0), which separates "bank past vertical"
from "unload".
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm  # noqa: E402

NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
N_STEPS = 576
SEGS, CH = 6, 3


class Ctl:
    """Genome.act, re-expressed with the per-segment bank target in DEGREES and an explicit gate."""

    def __init__(self, banks, loads, thrs, gate):
        self.banks, self.loads, self.thrs, self.gate = banks, loads, thrs, bool(gate)
        self.log = []

    def __call__(self, step, me, bd, rel):
        s = min(SEGS - 1, (step * SEGS) // N_STEPS)
        phi_tgt = math.radians(self.banks[s])
        load, thr = self.loads[s], self.thrs[s]
        be = phi_tgt - float(me.phi)
        ail = float(np.clip(bfm.KROLL * be, -1.0, 1.0))
        mag = max(math.cos(be), 0.0) if self.gate else 1.0
        elev = -float(np.clip(load * mag, 0.0, 1.0))
        self.log.append(
            (
                step,
                math.degrees(float(me.phi)),
                float(me.nz),
                elev,
                ail,
                float(me.alt),
                float(me.airspeed),
                float(rel.rng),
                float(rel.ata) * M.DEG,
            )
        )
        return np.array([ail, elev, 0.0, thr, -1.0], np.float32)


def decode(g):
    banks = [M.BANK_MAX_DEG * float(np.clip(g[s * CH + 0], -1, 1)) for s in range(SEGS)]
    loads = [0.5 * (float(np.clip(g[s * CH + 1], -1, 1)) + 1.0) for s in range(SEGS)]
    thrs = [0.5 * (float(np.clip(g[s * CH + 2], -1, 1)) + 1.0) for s in range(SEGS)]
    return banks, loads, thrs


def fly(banks, loads, thrs, gate, seed):
    env, base = M.build_env(M.CELLS["merge1500"], seed, N_STEPS * M.CONTROL_DT_S)
    c = Ctl(banks, loads, thrs, gate)
    r = M.roll_once(env, base, c, N_STEPS, keep=True)
    env.close()
    return r, np.asarray(c.log, float)


def metrics(r, log):
    elev, nz, phi, alt = log[:, 3], log[:, 2], log[:, 1], log[:, 5]
    n = len(log)
    # unload = the leading run of steps at exactly zero elevator command
    k = 0
    while k < n and abs(elev[k]) < 1e-12:
        k += 1
    # nz-gated: leading run with nz <= 1.0 g
    j = 0
    while j < n and nz[j] <= 1.0:
        j += 1
    # roll-in is measured by the caller (it needs the seg-0 bank target) via rollin_step().
    return dict(
        unload_steps=k,
        unload_s=k * M.CONTROL_DT_S,
        unload_alt_m=(alt[k - 1] - alt[0]) if k else 0.0,
        nz_le1_steps=j,
        nz_le1_s=j * M.CONTROL_DT_S,
        nz_min_seg0=float(nz[: min(96, n)].min()),
        nz_max_seg0=float(nz[: min(96, n)].max()),
        phi_max=float(np.abs(phi).max()),
    )


def rollin_step(log, target_deg, tol=5.0):
    phi = log[:, 1]
    hit = np.where(np.abs(phi - target_deg) <= tol)[0]
    return int(hit[0]) if len(hit) else -1


def main():
    z = np.load(NPZ, allow_pickle=True)
    g = np.asarray(z["genome"], float)
    ref_actions = np.asarray(z["actions"], np.float32)
    banks, loads, thrs = decode(g)
    print(f"  genome            {np.array2string(g, precision=6, max_line_width=200)}")
    print(f"  decoded banks     {['%.2f' % b for b in banks]}")
    print(f"  decoded loads     {['%.4f' % v for v in loads]}")
    print(f"  decoded thrs      {['%.4f' % v for v in thrs]}")

    arms = [("BASE", banks, True), ("NOGATE", banks, False)]
    for b0 in (60.0, 85.0, 90.0, 100.0, 130.0, 145.0, 160.0):
        arms.append((f"BANK{b0:.0f}", [b0] + banks[1:], True))

    print("\n" + "=" * 118)
    print("  PER-ARM, PER-SEED  (fit = M.score, the search's own objective)")
    print("=" * 118)
    print(
        f"  {'arm':<9} {'seed':>5} {'event':>9} {'kill':>4} {'hp':>6} {'tLeth':>6} {'st':>4} "
        f"{'dalt':>7} {'rngMn':>6} {'rnds':>5} {'fit':>11} | {'unld_st':>7} {'unld_s':>6} "
        f"{'unld_dalt':>9} {'nz<=1_s':>7} {'nzMx0':>6} {'phiMx':>6} {'roll_in':>7}"
    )
    summary = {}
    for name, bk, gate in arms:
        rows = []
        for sd in SEEDS:
            r, log = fly(bk, loads, thrs, gate, sd)
            mm = metrics(r, log)
            ri = rollin_step(log, bk[0])
            s, _c = M.score(r)
            rows.append((r, mm, s, ri))
            print(
                f"  {name:<9} {sd:>5} {str(r.event):>9} {r.kill:>4} {r.hp_dealt:>6.1f} "
                f"{r.tleth_s:>6.3f} {r.steps:>4} {r.dalt_m:>7.0f} {r.rng_min:>6.0f} "
                f"{r.rounds:>5} {s:>11.2f} | {mm['unload_steps']:>7d} {mm['unload_s']:>6.3f} "
                f"{mm['unload_alt_m']:>9.1f} {mm['nz_le1_s']:>7.3f} {mm['nz_max_seg0']:>6.2f} "
                f"{mm['phi_max']:>6.1f} {ri:>7d}"
            )
            if name == "BASE" and sd == 62:
                n = min(len(r.actions), len(ref_actions))
                d = float(np.max(np.abs(np.asarray(r.actions)[:n] - ref_actions[:n])))
                print(
                    f"      VALIDATION seed 62 vs npz actions: max|d| = {d:.3e} over {n} steps "
                    f"(len mine {len(r.actions)} vs npz {len(ref_actions)})"
                )
        summary[name] = rows
        print("  " + "-" * 114)

    print("\n" + "=" * 118)
    print(
        "  ARM SUMMARY  (8 seeds each; kills are IN-SAMPLE - these are the seeds the search used)"
    )
    print("=" * 118)
    print(
        f"  {'arm':<9} {'kills/8':>7} {'mean hp':>8} {'mean fit':>11} {'mean tLeth':>10} "
        f"{'mean dalt':>9} {'unload_s':>8} {'roll_in_s(mean)':>15}"
    )
    for name, bk, gate in arms:
        rows = summary[name]
        ks = sum(r.kill for r, _m, _s, _i in rows)
        ri = [i for _r, _m, _s, i in rows if i >= 0]
        print(
            f"  {name:<9} {ks:>7d} {np.mean([r.hp_dealt for r, _m, _s, _i in rows]):>8.1f} "
            f"{np.mean([s for _r, _m, s, _i in rows]):>11.2f} "
            f"{np.mean([r.tleth_s for r, _m, _s, _i in rows]):>10.3f} "
            f"{np.mean([r.dalt_m for r, _m, _s, _i in rows]):>9.0f} "
            f"{rows[0][1]['unload_s']:>8.3f} "
            f"{(np.mean(ri) * M.CONTROL_DT_S if ri else float('nan')):>15.3f}"
        )


if __name__ == "__main__":
    main()
