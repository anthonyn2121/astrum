"""Counterfactual the d3b dump does not contain: HOLD the +115 deg bank past step 95.

The 'expressible' claim says chi drifts ~20 deg/s under a constant bank target, so a single 4 s
segment sweeps the 40 deg (CHI_INPLANE_DEG 20 -> CHI_OUT_DEG 60) D1 span unaided. Every seed in
runs/d3b changes the bank command at step 96, so the dump cannot answer whether an UNAIDED drift
ever satisfies D1's actual predicate (|chi|>60 AND nz>3 held CHI_MIN_HOLD_S=0.5 s, then back under
CHI_RETURN_DEG=30 within 4 s). This flies the d3b genome's segment-0 command -- bank +115, load 1.0,
thr 0.6023 -- for all six segments on the same cell/seed and measures it.

Control arm: the real d3b genome, same harness, to show the harness reproduces the known result.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/crates/astrum-env/python")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

D3B = np.array(
    [1.0, 1.0, 0.20466533, 0.42106359, 0.04260145, 1.0,
     -1.0, 0.81890272, -1.0, -0.45391245, -0.49038442, 0.90924598,
     1.0, -1.0, -1.0, -1.0, 0.7947789, 0.53075154],
)  # fmt: skip
CONST = np.tile(np.array([1.0, 1.0, 0.20466533]), 6)  # segment 0's command, held for 24 s
N = 576
SEED = 62


def _runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def fly(g, label):
    gm = M.Genome(6, 3, "bank", True)
    env, base = M.build_env(M.CELLS["merge1500"], SEED, 24.0)
    rec = []

    def ctl(step, me, bd, rel):
        a = gm.act(g, step, N, me)
        smp = B.make_sample(step * M.CONTROL_DT_S, me, bd, a, rel)
        rec.append(
            dict(
                step=step,
                t=step * M.CONTROL_DT_S,
                seg=gm.seg_of(step, N),
                cmd_bank=math.degrees(
                    math.radians(M.BANK_MAX_DEG) * float(np.clip(g[gm.seg_of(step, N) * 3], -1, 1))
                ),
                phi=math.degrees(float(me.phi)),
                nz=float(me.nz),
                chi=math.degrees(float(smp.chi)) if smp.chi_valid else float("nan"),
                valid=int(bool(smp.chi_valid)),
                alt=float(me.alt),
                rng=float(rel.rng),
                ata_az=math.degrees(float(rel.ata_az)),
                aspect=math.degrees(float(rel.aspect)),
                closure=float(rel.closure),
                gamma=math.degrees(float(smp.gamma)),
            )
        )
        return a

    r = M.roll_once(env, base, ctl, N, keep=False)
    env.close()
    d = B.detect_all(r.trace)
    print("=" * 96)
    print(
        f"{label}: steps={r.steps} event={r.event} kill={r.kill} hp_dealt={r.hp_dealt:.1f} "
        f"tleth={r.tleth_s:.3f} dalt={r.dalt_m:+.0f} m"
    )
    for k in ("displacement_rolls", "yoyos", "turning_room"):
        for ev in d[k]:
            print(
                f"  EVENT {k}: "
                + " ".join(f"{f}={getattr(ev, f)}" for f in ev.__dataclass_fields__)
            )
    print(
        "  detectors: "
        + " ".join(f"{k}={len(v) if isinstance(v, list) else v}" for k, v in d.items())
    )
    chi = [x["chi"] for x in rec]
    nz = [x["nz"] for x in rec]
    t = [x["t"] for x in rec]
    st = [x["step"] for x in rec]
    print(
        f"  |chi| max {max(abs(c) for c in chi):.1f} deg ; chi_valid all-true={all(x['valid'] for x in rec)}"
    )
    for bar, lab in ((B.CHI_OUT_DEG, "D1 |chi|>60 & nz>3"), (B.CHI_ROOM_DEG, "D3 |chi|>45 & nz>3")):
        m = [abs(c) > bar and n > B.NZ_LOADED for c, n in zip(chi, nz)]
        rr = _runs(m)
        print(f"  {lab}: {len(rr)} runs, ticks {sum(m)}/{len(m)}")
        for i, j in rr:
            print(
                f"     steps {st[i]:3d}-{st[j]:3d} t {t[i]:7.3f}-{t[j]:7.3f} dur {t[j] - t[i]:6.3f} s "
                f"{'>=0.5s' if t[j] - t[i] >= B.CHI_MIN_HOLD_S else 'FAILS 0.5s':>10} "
                f"|chi| {abs(chi[i]):6.1f}->{abs(chi[j]):6.1f} pk {max(abs(chi[k]) for k in range(i, j + 1)):6.1f} "
                f"nz {min(nz[i : j + 1]):.2f}-{max(nz[i : j + 1]):.2f} alt {rec[i]['alt']:.0f}->{rec[j]['alt']:.0f} m"
            )
    print("  exit-clause audit, steps 180-205 (chi_valid=0 defaults chi to 0.0 rad):")
    for x in rec:
        if 180 <= x["step"] <= 205:
            print(
                f"     step {x['step']:3d} t {x['t']:6.3f} valid {x['valid']} chi {x['chi']:9.2f} nz {x['nz']:5.2f} "
                f"phi {x['phi']:8.2f} rng {x['rng']:7.1f} aspect {x['aspect']:7.2f} ata_az {x['ata_az']:8.2f}"
            )
    print(
        "  per-4.0s-segment drift (endpoint d(chi)/dt, d|chi|/dt, total-variation/dt), altitude price:"
    )
    for s in range(6):
        w = [x for x in rec if x["seg"] == s]
        if len(w) < 2:
            print(f"     seg {s}: not flown")
            continue
        c = [x["chi"] for x in w]
        tt = [x["t"] for x in w]
        dt = tt[-1] - tt[0]
        tv = sum(abs(c[k + 1] - c[k]) for k in range(len(c) - 1))
        nzz = [x["nz"] for x in w]
        print(
            f"     seg {s}: steps {w[0]['step']:3d}-{w[-1]['step']:3d} chi {c[0]:8.2f}->{c[-1]:8.2f} "
            f"d(chi)/dt {(c[-1] - c[0]) / dt:+7.2f} d|chi|/dt {(abs(c[-1]) - abs(c[0])) / dt:+7.2f} "
            f"tv/dt {tv / dt:6.2f} deg/s | phi {sum(x['phi'] for x in w) / len(w):+7.1f} "
            f"nz>3 {sum(1 for v in nzz if v > 3)}/{len(nzz)} rng {w[0]['rng']:6.0f}->{w[-1]['rng']:6.0f} "
            f"alt {w[0]['alt']:6.0f}->{w[-1]['alt']:6.0f} ({w[-1]['alt'] - w[0]['alt']:+6.0f} m)"
        )
    return rec


if __name__ == "__main__":
    rec_c = fly(CONST, "CONSTANT +115 BANK / LOAD 1.0 HELD 24 s (counterfactual)")
    rec_d = fly(D3B, "d3b GENOME (control: must reproduce kill at 173 steps, D1=0)")
    print("=" * 96)
    print("overlap check: constant arm vs d3b arm, steps 0-95 (both are segment 0's command)")
    n = min(96, len(rec_c), len(rec_d))
    print(
        "  max |dchi| %.3e deg   max |dphi| %.3e deg   max |dnz| %.3e"
        % (
            max(abs(rec_c[k]["chi"] - rec_d[k]["chi"]) for k in range(n)),
            max(abs(rec_c[k]["phi"] - rec_d[k]["phi"]) for k in range(n)),
            max(abs(rec_c[k]["nz"] - rec_d[k]["nz"]) for k in range(n)),
        )
    )
