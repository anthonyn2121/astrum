"""Independent attempt to REFUTE: "D2 only registers on the d3b kill trace under a joint
relaxation no pilot-facing definition supports, and even then never 'clean'".

Everything is recomputed from a FRESH RE-FLIGHT of the npz genome (trace built by
`M.roll_once` -> `B.make_sample`, i.e. the detector's own trace), and cross-checked against the
replay CSV `smp_*` block. No detector body is re-implemented; thresholds are monkeypatched on the
module and restored.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_refute_d2_joint2.py'
"""

from __future__ import annotations

import csv
import itertools
import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

DEG = 180.0 / math.pi
N_STEPS = 576
CSV = {
    62: "runs/d3b/replay_kill_seed62.csv",
    28: "runs/d3b/replay_nokill_seed28.csv",
}
SMP = [
    ("t", float),
    ("chi_rad", float),
    ("chi_valid", bool),
    ("gamma_rad", float),
    ("nz", float),
    ("phi_rad", float),
    ("alt", float),
    ("ail", float),
    ("rng", float),
    ("ata", float),
    ("ata_az", float),
    ("aspect", float),
    ("closure", float),
    ("cpa_miss", float),
    ("es_rel", float),
    ("lay_miss_m", float),
    ("lethal", bool),
]


def reflight(seed: int):
    z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
    g = np.asarray(z["genome"], float)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    cell = M.CELLS[str(z["cell"])]
    env, base = M.build_env(cell, seed, N_STEPS * M.CONTROL_DT_S)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, N_STEPS, me)

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    return r


def csv_check(seed: int, tr):
    with open(CSV[seed]) as fh:
        rows = list(csv.DictReader(fh))
    print(f"   CSV rows {len(rows)}  vs re-flight samples {len(tr)}")
    worst, worst_f = 0.0, ""
    for raw, cast in SMP:
        name = raw.removesuffix("_rad")
        for a, r in zip(tr, rows):
            v = float(r[f"smp_{raw}"])
            b = float(getattr(a, name))
            if cast is bool:
                v, b = float(v != 0.0), float(b != 0.0)
            if abs(b - v) > worst:
                worst, worst_f = abs(b - v), name
    print(f"   max |reflight - csv| over 17 Sample fields: {worst:.3e} (field {worst_f or 'none'})")


def closure_facts(tr, tag):
    cl = np.array([s.closure for s in tr])
    lo, hi = int(np.argmin(cl)), int(np.argmax(cl))
    print(
        f"   [{tag}] closure: min {cl.min():+.1f} m/s at step {lo} (t {tr[lo].t:.3f}) | "
        f"max {cl.max():+.1f} at step {hi} (t {tr[hi].t:.3f}) | mean {cl.mean():+.1f}"
    )
    below = int((cl <= B.CLOSURE_OVERSHOOT).sum())
    print(f"   [{tag}] ticks with closure <= CLOSURE_OVERSHOOT(120): {below}/{len(cl)}")
    edges = [
        i
        for i in range(len(tr))
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    print(
        f"   [{tag}] D2 rising-edge entry candidates: {edges[:12]}{' ...' if len(edges) > 12 else ''} "
        f"(n={len(edges)}); i==0 vacuous entry included: {0 in edges}"
    )
    for i in edges[:6]:
        print(
            f"        entry step {i} t {tr[i].t:.3f}s closure {tr[i].closure:+.1f} nz {tr[i].nz:.2f} g "
            f"gamma {tr[i].gamma * DEG:+.2f} deg rng {tr[i].rng:.0f} m"
        )
    return cl


def gamma_facts(tr, tag):
    gm_ = np.array([s.gamma for s in tr]) * DEG
    g0 = gm_[0]
    k = int(np.argmax(np.abs(gm_ - g0)))
    print(
        f"   [{tag}] gamma: entry {g0:+.2f} deg, peak excursion {gm_[k]:+.2f} deg at step {k} "
        f"(t {tr[k].t:.3f} s) -> swing {abs(gm_[k] - g0):.2f} deg; final {gm_[-1]:+.2f} deg "
        f"(|final-g0| {abs(gm_[-1] - g0):.2f} deg)"
    )
    post = np.abs(gm_[k + 1 :] - g0)
    if post.size:
        j = int(np.argmin(post)) + k + 1
        print(
            f"   [{tag}] closest RETURN to entry flight path after the peak: {abs(gm_[j] - g0):.2f} deg "
            f"at step {j} (t {tr[j].t:.3f} s)"
        )


def ev_row(tr, e):
    i0, i1 = e.i0, e.i1
    dalt = tr[i1].alt - tr[i0].alt
    d = e.detail
    return (
        f"        {e.kind:<10} entry step {i0} (t {e.t0:.3f}s, nz {tr[i0].nz:.2f} g, "
        f"clos {d['closure_entry']:+.0f}) exit step {i1} (t {e.t1:.3f}s, clos {d['closure_exit']:+.0f}, "
        f"rng {tr[i1].rng:.0f} m) swing {d['gamma_swing_deg']:.2f} peak {d['gamma_peak_deg']:+.2f} "
        f"|exit-g0| {abs(tr[i1].gamma - tr[i0].gamma) * DEG:.2f} loadedFrac {d['loaded_frac']:.2f} "
        f"dalt {dalt:+.0f} m  daspect {e.d_aspect_deg:+.1f} bled {not d['failed_exit']} "
        f"kept {not d['failed_aspect']} CLEAN {d['clean']}"
    )


def sweep18(tr, tag):
    print(f"\n   === 18-COMBO SWEEP, verbatim reproduction ({tag}) ===")
    saved = {k: getattr(B, k) for k in dir(B) if k.isupper()}
    hits = []
    try:
        print(f"     {'SWING':>6} {'RET_DEG':>7} {'RET_S':>6} -> {'D2':>3} {'D2c':>4}")
        for sw, rd, rs in itertools.product((30.0, 25.0, 20.0), (10.0, 20.0, 30.0), (6.0, 12.0)):
            B.YOYO_GAMMA_SWING_DEG, B.YOYO_GAMMA_RETURN_DEG, B.YOYO_RETURN_S = sw, rd, rs
            d2 = B.detect_yoyos(tr)
            nc = sum(1 for e in d2 if e.detail.get("clean"))
            print(
                f"     {sw:>6.0f} {rd:>7.0f} {rs:>6.0f} -> {len(d2):>3} {nc:>4}"
                + ("   <- ret_deg >= swing: RETURN CLAUSE VACUOUS" if rd >= sw else "")
            )
            if d2:
                hits.append((sw, rd, rs))
                for e in d2:
                    print(ev_row(tr, e))
    finally:
        for k, v in saved.items():
            setattr(B, k, v)
    print(f"     combos with D2 >= 1: {hits}")
    return hits


def single_param(tr, tag):
    """Is the relaxation really JOINT? Relax exactly ONE yo-yo threshold at a time."""
    print(f"\n   === SINGLE-PARAMETER RELAXATIONS ({tag}); all others at repo defaults ===")
    saved = {k: getattr(B, k) for k in dir(B) if k.isupper()}
    grid = {
        "YOYO_GAMMA_SWING_DEG": (30.0, 28.5, 28.0, 25.0, 20.0, 10.0, 1.0),
        "YOYO_GAMMA_RETURN_DEG": (10.0, 15.0, 18.0, 19.0, 20.0, 30.0, 60.0, 180.0),
        "YOYO_RETURN_S": (6.0, 8.0, 12.0, 24.0, 60.0),
        "YOYO_GAMMA_EXCURSION_DEG": (15.0, 10.0, 5.0, 0.0),
        "LOADED_FRAC": (0.5, 0.25, 0.0),
        "CLOSURE_OVERSHOOT": (120.0, 200.0, 300.0, 373.0, 400.0, 450.0, 470.0, 480.0, 600.0),
        "YOYO_EXIT_RNG_M": (1500.0, 3000.0, 1e9),
        "YOYO_ASPECT_SLIP_DEG": (10.0, 45.0, 180.0),
    }
    try:
        for k, vals in grid.items():
            out = []
            for v in vals:
                for kk, vv in saved.items():
                    setattr(B, kk, vv)
                setattr(B, k, v)
                d2 = B.detect_yoyos(tr)
                out.append(f"{v:g}->D2 {len(d2)}/D2c {sum(1 for e in d2 if e.detail.get('clean'))}")
            print(f"     {k:<26} " + "  ".join(out))
    finally:
        for kk, vv in saved.items():
            setattr(B, kk, vv)


def clean_reachable(tr, tag):
    """Can `clean` EVER be 1 -- shape clauses fully open, only the closure/aspect exit tests live?"""
    print(f"\n   === IS 'CLEAN' STRUCTURALLY REACHABLE ({tag})? shape clauses fully open ===")
    saved = {k: getattr(B, k) for k in dir(B) if k.isupper()}
    try:
        B.YOYO_GAMMA_SWING_DEG = 1.0
        B.YOYO_GAMMA_RETURN_DEG = 180.0
        B.YOYO_RETURN_S = 60.0
        B.YOYO_GAMMA_EXCURSION_DEG = 0.0
        B.LOADED_FRAC = 0.0
        for co in (120.0, 300.0, 373.0, 400.0, 440.0, 460.0, 470.0, 475.0, 480.0, 490.0, 600.0):
            B.CLOSURE_OVERSHOOT = co
            d2 = B.detect_yoyos(tr)
            nc = sum(1 for e in d2 if e.detail.get("clean"))
            nb = sum(1 for e in d2 if not e.detail.get("failed_exit"))
            nk = sum(1 for e in d2 if not e.detail.get("failed_aspect"))
            print(
                f"     CLOSURE_OVERSHOOT {co:>6.0f} -> D2 {len(d2):>3}  bled {nb:>3}  kept {nk:>3}  CLEAN {nc:>3}"
            )
    finally:
        for kk, vv in saved.items():
            setattr(B, kk, vv)


def main():
    for seed in (62, 28):
        r = reflight(seed)
        tr = r.trace
        tag = f"seed {seed} {'KILL' if r.kill else 'NOKILL'}"
        print("=" * 96)
        print(
            f"  {tag}: {len(tr)} ticks, t {tr[-1].t:.3f} s, event {r.event}, hp_dealt {r.hp_dealt:.1f}, "
            f"tleth {r.tleth_s:.3f}, dalt {r.dalt_m:+.0f} m"
        )
        print("=" * 96)
        csv_check(seed, tr)
        d = B.detect_all(tr)
        print(
            f"   detect_all @ repo defaults: D1 {len(d['displacement_rolls'])} D2 {len(d['yoyos'])} "
            f"D2c {len(d['yoyos_clean'])} D3 {len(d['turning_room'])} lethTk {d['lethal_ticks']}"
        )
        closure_facts(tr, tag)
        gamma_facts(tr, tag)
        sweep18(tr, tag)
        single_param(tr, tag)
        clean_reachable(tr, tag)
        assert B.YOYO_GAMMA_SWING_DEG == 30.0 and B.CLOSURE_OVERSHOOT == 120.0, (
            "globals not restored"
        )
        print("   globals restored -- verified")


if __name__ == "__main__":
    main()
