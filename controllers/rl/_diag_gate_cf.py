"""Counterfactual for the roll-gate claim: re-fly the SAVED d3b genome with roll_gate ON vs OFF.

Nothing is reimplemented -- Genome / build_env / roll_once / score / _trigger all come from
_diag_true_mpc, so the ON arm is the searched trajectory by construction.
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np

DEG = 180.0 / math.pi
NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
CELL = M.CELLS["merge1500"]
HZ = M.RATE_HZ / M.ACTION_REPEAT
DT = 1.0 / HZ


def fly(g, seed, gate, n_steps=576):
    gen = M.Genome(6, 3, "bank", roll_gate=gate)
    env, base = M.build_env(CELL, seed, n_steps / HZ)
    rec = []

    def ctl(step, me, bd, rel):
        a = gen.act(g, step, n_steps, me)
        rec.append(
            dict(
                step=step,
                t=step * DT,
                elev=float(a[1]),
                ail=float(a[0]),
                phi=float(me.phi) * DEG,
                nz=float(me.nz),
                alt=float(me.alt),
                vt=float(me.airspeed),
                rng=float(rel.rng),
                lay=float(rel.lay_miss_m),
                oh=float(bd.health),
            )
        )
        return a

    r = M.roll_once(env, base, ctl, n_steps, keep=True)
    return rec, r


def main():
    d = np.load(NPZ, allow_pickle=True)
    g = np.asarray(d["genome"], np.float64)
    for seed in (62, 28):
        print("=" * 96)
        for gate in (True, False):
            rec, r = fly(g, seed, gate)
            nz = [x["nz"] for x in rec]
            elev = [x["elev"] for x in rec]
            n0 = [i for i, e in enumerate(elev) if abs(e) == 0.0]
            # first step whose nz exceeds 3 g (the detector's NZ_LOADED bar)
            f3 = next((i for i, v in enumerate(nz) if v > 3.0), None)
            fneg = [i for i, v in enumerate(nz) if v < 0.0]
            # roll completion: |phi - 115| <= 5 deg
            frc = next((i for i, x in enumerate(rec) if abs(x["phi"] - 115.0) <= 5.0), None)
            ev = getattr(r, "event", None)
            print(
                f"seed {seed} gate={gate!s:5}  steps {len(rec):3d}  event={ev}  "
                f"kill={getattr(r, 'kill', None)} hp_dealt={getattr(r, 'hp_dealt', None)} "
                f"tleth={getattr(r, 'tleth_s', None)}"
            )
            print(
                f"    elev==0 steps: {len(n0)} (first20 {n0[:20]})   "
                f"min nz over steps 0-29 = {min(nz[:30]):+.3f} g @ step {int(np.argmin(nz[:30]))} "
                f"(t {np.argmin(nz[:30]) * DT:.3f})"
            )
            print(
                f"    nz<0 steps in 0-29: {[i for i in fneg if i < 30]}   "
                f"first nz>3g: step {f3} (t {None if f3 is None else round(f3 * DT, 3)})   "
                f"roll |phi-115|<=5 at step {frc} (t {None if frc is None else round(frc * DT, 3)})"
            )
            print(
                "    first 10 rows: "
                + " ".join(
                    f"[{x['step']} e{x['elev']:+.2f} nz{x['nz']:+.2f} phi{x['phi']:+6.1f}]"
                    for x in rec[:10]
                )
            )
            a30 = rec[min(29, len(rec) - 1)]
            print(
                f"    @step30: phi {a30['phi']:+.1f} nz {a30['nz']:+.2f} alt {a30['alt']:.0f} "
                f"vt {a30['vt']:.1f} rng {a30['rng']:.0f}"
            )


if __name__ == "__main__":
    main()
