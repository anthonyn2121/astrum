"""REFUTATION CONTROL for the D3b "energy-POSITIVE maneuver" claim.

The claim is that the merge1500 kill trajectory is energy-positive (dEs = +283 m over the 7.167 s
it flew) "because segment 1 descends on full afterburner". Two things are missing from it:

  (a) A BASE RATE. dEs > 0 is only informative against what an ALTERNATIVE stick achieves from the
      same initial condition over the same 7.167 s window. A jet that unloads and runs the engine up
      gains Es whatever the bank angle is; if a wings-level unloaded arm gains MORE, the searched
      maneuver is energy-NEGATIVE relative to the achievable and the sign is an artefact of the
      reference point, not a property of the maneuver.

  (b) A MECHANISM TEST. "because segment 1 descends on full afterburner" is a causal claim. It is
      testable by flying the SAME genome with seg-1 throttle set back to seg-0's value and nothing
      else changed.

Every arm is flown through `_diag_true_mpc`'s own `Genome`/`roll_once`/`build_env` on the SAME cell
(merge1500) and the SAME sim seed (62, the one that produced the npz), and every arm is cut at the
SAME 173 control steps so the comparison window is fixed at 7.167 s.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _refute_energy_baseline.py'
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight.geometry import specific_energy

SEED = 62
N_STEPS = 173  # the kill's own length: steps 0..172, t 0.000..7.167 s -- the CUT
# `Genome.seg_of` is (step*segments)//n_steps, so the genome's 4.0 s segment boundaries only land
# where the search put them if the HORIZON handed to `act` is the searched 576. Cutting the rollout
# at 173 steps must not re-time the genome, so the two are kept separate.
HORIZON_STEPS = 576
HORIZON_S = HORIZON_STEPS * M.CONTROL_DT_S  # 24.0 s
CELL = "merge1500"

G_BEST = np.load("runs/d3b/d3b_best.npz")["genome"].astype(np.float64)


def const(bank_p: float, load_p: float, thr_p: float) -> np.ndarray:
    """A 6-segment genome that commands the same (bank, load, throttle) triple throughout."""
    return np.array([bank_p, load_p, thr_p] * 6, np.float64)


def fly(g: np.ndarray, label: str) -> dict:
    gm = M.Genome(6, 3, "bank", True)
    env, base = M.build_env(M.CELLS[CELL], SEED, HORIZON_S)
    rec: list[dict] = []

    def ctl(step, me, bd, rel):
        a = gm.act(g, step, HORIZON_STEPS, me)
        rec.append(
            dict(
                step=step,
                t=step * M.CONTROL_DT_S,
                alt=float(me.alt),
                vt=float(me.airspeed),
                es=specific_energy(me),
                nz=float(me.nz),
                phi=float(me.phi) * 180.0 / math.pi,
                gam=math.degrees(
                    math.asin(max(-1.0, min(1.0, float(me.vel[2]) / max(1e-6, float(me.airspeed)))))
                ),
                a_thr=float(a[3]),
                elev=float(a[1]),
                opp_es=specific_energy(bd),
            )
        )
        return a

    r = M.roll_once(env, base, ctl, N_STEPS, keep=False)
    env.close()
    es = [row["es"] for row in rec]
    alt = [row["alt"] for row in rec]
    vt = [row["vt"] for row in rec]
    n = len(rec)
    out = dict(
        label=label,
        steps=n,
        event=r.event,
        t_end=rec[-1]["t"],
        dEs=es[-1] - es[0],
        dAlt=alt[-1] - alt[0],
        dV2=vt[-1] ** 2 / (2 * 9.80665) - vt[0] ** 2 / (2 * 9.80665),
        es0=es[0],
        esN=es[-1],
        vtN=vt[-1],
        altN=alt[-1],
        dEs_0_30=es[min(30, n - 1)] - es[0],
        dEs_30_end=es[-1] - es[min(30, n - 1)],
        Ps_mean=(es[-1] - es[0]) / max(1e-9, rec[-1]["t"]),
        hp_dealt=r.hp_dealt,
        rec=rec,
    )
    return out


def main() -> None:
    # What the affine action wrapper actually sends the FDM, so "throttle 0.602" can be checked.
    env, base = M.build_env(M.CELLS[CELL], SEED, HORIZON_S)
    lo = np.asarray(base.action_space.low, float)
    hi = np.asarray(base.action_space.high, float)
    print("BASE (pre-wrapper) env action_space low :", np.round(lo, 4))
    print("BASE (pre-wrapper) env action_space high:", np.round(hi, 4))
    for a3 in (0.6023, 1.0, 0.5, 0.0, -1.0):
        print(
            f"  agent a[3]={a3:+.4f}  ->  NATIVE throttle = {lo[3] + 0.5 * (a3 + 1.0) * (hi[3] - lo[3]):.4f}"
        )
    env.close()

    g_noab = G_BEST.copy()
    g_noab[5] = G_BEST[2]  # seg1 throttle param := seg0's  (the MECHANISM test)
    g_ab0 = G_BEST.copy()
    g_ab0[2] = 1.0  # seg0 throttle := full          (the reverse mechanism test)

    arms = [
        (G_BEST, "d3b_best (the claim)"),
        (g_noab, "d3b, seg1 thr := seg0 thr (no AB step)"),
        (g_ab0, "d3b, seg0 thr := full"),
        (const(0.0, -1.0, 1.0), "CONTROL wings-level, UNLOADED, thr full"),
        (const(0.0, -1.0, -1.0), "CONTROL wings-level, UNLOADED, thr idle"),
        (const(0.0, 0.0, 0.0), "CONTROL wings-level, load 0.5, thr mid"),
        (const(0.0, 1.0, 1.0), "CONTROL wings-level, MAX PULL, thr full"),
        (const(G_BEST[0], 1.0, 1.0), "CONTROL bank 115 MAX PULL thr full"),
        (const(G_BEST[0], G_BEST[1], G_BEST[2]), "CONTROL seg0 command held all window"),
    ]

    res = [fly(g, lbl) for g, lbl in arms]

    print()
    print("=" * 124)
    print(
        f"ENERGY BOOK over a FIXED {N_STEPS}-step / {(N_STEPS - 1) * M.CONTROL_DT_S:.3f} s window, cell {CELL}, sim seed {SEED}"
    )
    print("=" * 124)
    hdr = f"{'arm':<42} {'stp':>4} {'event':>9} {'dEs':>9} {'dAlt':>9} {'dV2/2g':>9} {'Ps':>8} {'dEs[0:30]':>10} {'dEs[30:]':>9} {'vtN':>7}"
    print(hdr)
    print("-" * 124)
    for r in res:
        print(
            f"{r['label']:<42} {r['steps']:>4} {str(r['event']):>9} {r['dEs']:>+9.1f} {r['dAlt']:>+9.1f} "
            f"{r['dV2']:>+9.1f} {r['Ps']:>+8.1f} {r['dEs_0_30']:>+10.1f} {r['dEs_30_end']:>+9.1f} {r['vtN']:>7.1f}"
            if False
            else f"{r['label']:<42} {r['steps']:>4} {str(r['event']):>9} {r['dEs']:>+9.1f} {r['dAlt']:>+9.1f} "
            f"{r['dV2']:>+9.1f} {r['Ps_mean']:>+8.1f} {r['dEs_0_30']:>+10.1f} {r['dEs_30_end']:>+9.1f} {r['vtN']:>7.1f}"
        )
    print("-" * 124)
    base_es0 = res[0]["es0"]
    print(
        f"all arms share es0 = {base_es0:.4f} m (identical IC): "
        + ", ".join(f"{r['es0']:.4f}" for r in res)
    )

    print()
    print(
        "PRE-PULL WINDOW (steps 0-30, t 0.000-1.250 s) IN DETAIL -- is the early Es gain genome-specific?"
    )
    print(f"{'arm':<42} {'nz range':>16} {'dAlt':>8} {'dV':>8} {'dEs':>9} {'Ps':>8}")
    for r in res:
        rec = r["rec"][:31]
        nzs = [x["nz"] for x in rec]
        print(
            f"{r['label']:<42} {min(nzs):>+7.2f}..{max(nzs):<+7.2f} {rec[-1]['alt'] - rec[0]['alt']:>+8.1f} "
            f"{rec[-1]['vt'] - rec[0]['vt']:>+8.2f} {rec[-1]['es'] - rec[0]['es']:>+9.1f} "
            f"{(rec[-1]['es'] - rec[0]['es']) / rec[-1]['t']:>+8.1f}"
        )

    print()
    print(
        "Step-0 commanded action for each arm (agent space) + first 6 nz, to expose the IC transient:"
    )
    for r in res:
        rec = r["rec"]
        print(
            f"  {r['label']:<42} elev0={rec[0]['elev']:+.3f} a_thr0={rec[0]['a_thr']:+.3f} "
            f"nz[0:6]={[round(x['nz'], 2) for x in rec[:6]]}"
        )


if __name__ == "__main__":
    main()
