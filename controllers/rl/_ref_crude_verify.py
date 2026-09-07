"""REFUTATION CHECK for the anti-BFM 'crude family' claim (_diag_d3b_skeptic.py:129-152).

Claim under test: "The maneuver is not a member of any robust crude family, and it is not a single
spiral either: rounding its six expressed numbers to human values still passes at 16 m and still
fires 29 rounds but deals 0.0 HP, while the first segment flown alone flies into the ground on 8/8
seeds."

Every arm here reuses the skeptic's OWN `fly`/`_env_for`/`cell` (imported, not reimplemented) so no
arm can differ on the trigger, the score, or the start. Extra instrumentation is added by wrapping
the controller only.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _ref_crude_verify.py --part A'
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_d3b_skeptic as S  # noqa: E402
import _diag_true_mpc as M  # noqa: E402

DEG = 180.0 / np.pi
N = S.N_STEPS
NPZ = S.NPZ


def enc(bank_deg: float, load: float, thr: float) -> list[float]:
    """Skeptic's own encoder (_diag_d3b_skeptic.py:143-145), verbatim."""
    return [bank_deg / M.BANK_MAX_DEG, 2.0 * load - 1.0, 2.0 * thr - 1.0]


def dec(p) -> tuple[float, float, float]:
    """Inverse: what a 3-vector actually commands."""
    return (
        M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)),
        0.5 * (float(np.clip(p[1], -1, 1)) + 1.0),
        0.5 * (float(np.clip(p[2], -1, 1)) + 1.0),
    )


def fly_rec(g, c, seed, segs=6, n_steps=N):
    """S.fly with a recording controller. Same Genome, same env cache, same roll_once/trigger."""
    gm = M.Genome(segs, 3, "bank", True)
    env, base = S._env_for(c, seed)
    rec = []

    def ctl(step, me, bd, rel):
        a = gm.act(g, min(N - 1, max(0, step)), N, me)
        rec.append(
            dict(
                step=step,
                rng=float(rel.rng),
                lay=float(rel.lay_miss_m),
                fire=float(M._trigger(rel)) > 0.0,
                alt=float(me.alt),
                nz=float(me.nz),
                phi=DEG * float(me.phi),
                aspect=DEG * float(rel.aspect),
                ata=DEG * float(rel.ata),
                oppH=float(bd.health),
                ammo=float(me.ammo),
            )
        )
        return a

    r = M.roll_once(env, base, ctl, n_steps, keep=False)
    return r, rec


def per_seed(tag, g, c, seeds, segs=6):
    rs = []
    print(f"  {tag}")
    for s in seeds:
        t0 = time.time()
        r = S.fly(g, c, s, segs=segs)
        rs.append(r)
        print(
            f"    seed {s:4d}  ev {str(r.event):9s} kill {r.kill}  hp {r.hp_dealt:5.1f}  "
            f"tLeth {r.tleth_s:5.3f}  t600 {r.t600_s:5.3f}  rngMin {r.rng_min:7.1f}  "
            f"rnds {r.rounds:4d}  dalt {r.dalt_m:+7.0f}  dead {int(r.dead)}  steps {r.steps:4d}"
            f"   [{time.time() - t0:.1f}s]",
            flush=True,
        )
    k = sum(r.kill for r in rs)
    print(
        f"    POOL kills {k}/{len(rs)}  hp {np.mean([r.hp_dealt for r in rs]):6.2f}  "
        f"tLeth {np.mean([r.tleth_s for r in rs]):6.3f}  "
        f"rngMin mean {np.mean([r.rng_min for r in rs]):7.1f} min {min(r.rng_min for r in rs):7.1f}  "
        f"rnds {np.mean([r.rounds for r in rs]):6.2f}  "
        f"dalt {np.mean([r.dalt_m for r in rs]):+8.0f}  dead {sum(r.dead for r in rs)}  "
        f"steps {np.mean([r.steps for r in rs]):6.1f}",
        flush=True,
    )
    return rs


def compact(tag, g, c, seeds, segs=6):
    rs = [S.fly(g, c, s, segs=segs) for s in seeds]
    k = sum(r.kill for r in rs)
    hp = [r.hp_dealt for r in rs]
    print(
        f"  {tag:44s} kills {k}/{len(rs)}  hp {np.mean(hp):6.2f} [{min(hp):5.1f},{max(hp):5.1f}]  "
        f"tLeth {np.mean([r.tleth_s for r in rs]):5.3f}  "
        f"rngMin {np.mean([r.rng_min for r in rs]):7.1f}  "
        f"rnds {np.mean([r.rounds for r in rs]):5.1f}  "
        f"dalt {np.mean([r.dalt_m for r in rs]):+7.0f}  dead {sum(r.dead for r in rs)}  "
        f"steps {np.mean([r.steps for r in rs]):5.0f}",
        flush=True,
    )
    return rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="A")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    seeds = S.SEEDS[: a.seeds]
    g = np.asarray(np.load(NPZ, allow_pickle=True)["genome"], float)
    seg0, seg1 = g[0:3], g[3:6]
    print(f"genome {np.round(g, 6).tolist()}")
    print(
        f"  seg0 decode bank {dec(seg0)[0]:8.4f} deg  load {dec(seg0)[1]:.6f}  thr {dec(seg0)[2]:.6f}"
    )
    print(
        f"  seg1 decode bank {dec(seg1)[0]:8.4f} deg  load {dec(seg1)[1]:.6f}  thr {dec(seg1)[2]:.6f}"
    )
    print(f"  seeds {seeds}\n", flush=True)
    c = S.cell(1500.0)

    if a.part == "A":
        print("A. EXACT REPRODUCTION of the cited crude rows, PER SEED")
        per_seed("BASELINE full 18-dim genome", g, c, seeds)
        per_seed("seg0,seg1 repeated (2-seg, 4 s each)", np.concatenate([seg0, seg1] * 3), c, seeds)
        per_seed("seg0 ONLY, all 24 s", np.concatenate([seg0] * 6), c, seeds)
        crude = np.asarray(enc(115.0, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * 5, float)
        per_seed("CRUDE 115/1.0/0.6 -> 45/0.5/1.0", crude, c, seeds)

    if a.part == "B":
        print("B. THE CITED SWEEPS, recomputed (compact rows, same order as the script)")
        for bank in (95.0, 105.0, 115.0, 125.0):
            cr = np.asarray(enc(bank, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * 5, float)
            compact(f"CRUDE bank {bank:5.1f} phase1", cr, c, seeds)
        for tsw in (3.0, 3.5, 4.0, 4.5, 5.0):
            n = max(1, int(round(24.0 / tsw)))
            gg = np.asarray(enc(115.0, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * (n - 1), float)
            compact(
                f"CRUDE switch at {tsw:.1f}s (segs {n} => {24.0 / n:.2f}s each)",
                gg,
                c,
                seeds,
                segs=n,
            )

    if a.part == "C":
        print("C. WHICH of the three roundings breaks it? (one at a time, rest EXACT)")
        for tag, s1 in [
            ("exact seg1 (control)", list(seg1)),
            ("seg1 bank 45 only", [45.0 / M.BANK_MAX_DEG, seg1[1], seg1[2]]),
            ("seg1 load 0.5 only", [seg1[0], 0.0, seg1[2]]),
            ("seg1 thr 1.0 only", [seg1[0], seg1[1], 1.0]),
        ]:
            gg = np.concatenate([seg0, np.asarray(s1, float), g[6:]])
            compact(tag, gg, c, seeds)
        for tag, s0 in [
            ("seg0 thr 0.60 only", [seg0[0], seg0[1], 2.0 * 0.60 - 1.0]),
        ]:
            gg = np.concatenate([np.asarray(s0, float), seg1, g[6:]])
            compact(tag, gg, c, seeds)

    if a.part == "D":
        print("D. IS THE BASIN A KNIFE EDGE? seg-1 bank swept, everything else EXACT")
        print("   (segments 2-5 left at the searched values so only ONE number moves)")
        for bank in (40.0, 42.0, 44.0, 45.0, 46.0, 47.0, 48.0, 48.42, 49.0, 50.0, 52.0, 55.0):
            gg = np.concatenate([seg0, [bank / M.BANK_MAX_DEG], seg1[1:], g[6:]])
            compact(f"seg1 bank {bank:6.2f} deg (exact load/thr)", gg, c, seeds)

    if a.part == "E":
        print("E. CRUDE-FAMILY GRID: 2-seg hand-coded, seg1 bank x load, thr 1.0, seg0 115/1.0/0.6")
        for bank in (45.0, 50.0, 55.0):
            for load in (0.4, 0.5, 0.6, 0.7):
                gg = np.asarray(enc(115.0, 1.0, 0.60) + enc(bank, load, 1.00) * 5, float)
                compact(f"CRUDE seg1 bank {bank:5.1f} load {load:.1f}", gg, c, seeds)

    if a.part == "F":
        print("F. 'FLIES INTO THE GROUND': WHEN? seg0-only, instrumented, seed by seed")
        one = np.concatenate([seg0] * 6)
        for s in seeds:
            r, rec = fly_rec(one, c, s)
            alt0 = rec[0]["alt"]
            i173 = min(172, len(rec) - 1)
            imin = int(np.argmin([q["rng"] for q in rec]))
            print(
                f"    seed {s:4d}  ev {str(r.event):9s} steps {r.steps:4d} ({r.steps * M.CONTROL_DT_S:6.3f}s)"
                f"  dalt {r.dalt_m:+7.0f}  hp {r.hp_dealt:5.1f}  rnds {r.rounds:4d}"
                f"  | at step {i173} (t {i173 * M.CONTROL_DT_S:5.3f}s): alt {rec[i173]['alt']:7.0f} "
                f"(dalt {rec[i173]['alt'] - alt0:+6.0f}) rng {rec[i173]['rng']:7.0f}"
                f"  | rngMin {rec[imin]['rng']:7.1f} at step {imin} (t {imin * M.CONTROL_DT_S:6.3f}s)"
                f" alt there {rec[imin]['alt']:7.0f}",
                flush=True,
            )
        print("\n   seg0 ONLY but horizon TRUNCATED to the kill length (173 steps = 7.208 s)")
        for s in seeds:
            r, rec = fly_rec(one, c, s, n_steps=173)
            print(
                f"    seed {s:4d}  ev {str(r.event):9s} steps {r.steps:4d}  dalt {r.dalt_m:+7.0f}"
                f"  hp {r.hp_dealt:5.1f}  rnds {r.rounds:4d}  rngMin {r.rng_min:7.1f}"
                f"  t600 {r.t600_s:5.3f}  tLeth {r.tleth_s:5.3f}  dead {int(r.dead)}",
                flush=True,
            )

    if a.part == "G":
        print("G. WHERE DID THE CRUDE ARM'S ROUNDS GO? (fire log, baseline vs crude, seed 62)")
        crude = np.asarray(enc(115.0, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * 5, float)
        for tag, gg in [("BASELINE", g), ("CRUDE", crude)]:
            r, rec = fly_rec(gg, c, 62)
            fired = [q for q in rec if q["fire"]]
            imin = int(np.argmin([q["rng"] for q in rec]))
            print(
                f"\n  {tag}: ev {r.event} steps {r.steps} rnds {r.rounds} hp {r.hp_dealt:.1f} "
                f"rngMin {r.rng_min:.1f} m at step {imin} (t {imin * M.CONTROL_DT_S:.3f}s)  "
                f"fire steps {len(fired)}",
                flush=True,
            )
            if fired:
                print(
                    f"    first fire step {fired[0]['step']} t {fired[0]['step'] * M.CONTROL_DT_S:.3f}s "
                    f"rng {fired[0]['rng']:.0f} lay {fired[0]['lay']:.1f} aspect {fired[0]['aspect']:.1f} "
                    f"ata {fired[0]['ata']:.1f}"
                )
                print(
                    f"    last  fire step {fired[-1]['step']} t {fired[-1]['step'] * M.CONTROL_DT_S:.3f}s "
                    f"rng {fired[-1]['rng']:.0f} lay {fired[-1]['lay']:.1f} aspect {fired[-1]['aspect']:.1f} "
                    f"ata {fired[-1]['ata']:.1f}"
                )
                print(
                    f"    fire-step rng   min {min(q['rng'] for q in fired):7.1f}  "
                    f"med {np.median([q['rng'] for q in fired]):7.1f}  "
                    f"max {max(q['rng'] for q in fired):7.1f}"
                )
                print(
                    f"    fire-step lay   min {min(q['lay'] for q in fired):7.2f}  "
                    f"med {np.median([q['lay'] for q in fired]):7.2f}  "
                    f"max {max(q['lay'] for q in fired):7.2f}"
                )
                nleth = sum(1 for q in fired if q["lay"] < 8.0 and q["rng"] <= 600.0)
                print(f"    fire steps with lay < HIT_RADIUS 8 m and rng<=600: {nleth}")
            print(
                f"    at CPA step {imin}: rng {rec[imin]['rng']:.1f} lay {rec[imin]['lay']:.1f} "
                f"fire {rec[imin]['fire']} aspect {rec[imin]['aspect']:.1f} ata {rec[imin]['ata']:.1f} "
                f"nz {rec[imin]['nz']:.2f} phi {rec[imin]['phi']:.1f}"
            )

    if a.part == "H":
        print("H. IS THERE A HUMAN-ROUNDED 2-SEGMENT CODE INSIDE THE BASIN?")
        print("   (pure hand-coded genome: seg0 = 115/1.0/0.60 for 4 s, then seg1 held to 24 s)")
        for bank, load in [
            (48.0, 0.52),
            (48.0, 0.50),
            (48.0, 0.55),
            (48.5, 0.52),
            (49.0, 0.52),
            (47.0, 0.52),
            (47.0, 0.55),
            (50.0, 0.52),
        ]:
            gg = np.asarray(enc(115.0, 1.0, 0.60) + enc(bank, load, 1.00) * 5, float)
            compact(f"HAND 115/1.0/0.60 -> {bank:5.1f}/{load:.2f}/1.00", gg, c, seeds)
        print("\n   load basin at the searched bank 48.4223, everything else EXACT genome")
        for load in (0.50, 0.51, 0.52, 0.521301, 0.53, 0.54, 0.56):
            gg = np.concatenate([seg0, [seg1[0]], [2.0 * load - 1.0], [seg1[2]], g[6:]])
            compact(f"seg1 load {load:.6f} (exact bank/thr)", gg, c, seeds)

    if a.part == "I":
        print("I. HOW FAR DOES THE HAND-CODED 8/8 ARM REACH? (bounding my own counter-claim)")
        hand = np.asarray(enc(115.0, 1.0, 0.60) + enc(48.0, 0.55, 1.00) * 5, float)
        for tag, cc in [
            ("lateral 1500 (as searched)", S.cell(1500.0)),
            ("lateral 1350 (-10%)", S.cell(1350.0)),
            ("lateral 1650 (+10%)", S.cell(1650.0)),
            ("lateral 1200 (-20%)", S.cell(1200.0)),
            ("lateral 1800 (+20%)", S.cell(1800.0)),
            ("MIRRORED lateral -1500", S.mirrored(1500.0)),
        ]:
            compact(f"HAND 48/0.55 @ {tag}", hand, cc, seeds)
        print("\n   the SEARCHED genome at the same cells, for comparison")
        for tag, cc in [
            ("lateral 1500 (as searched)", S.cell(1500.0)),
            ("lateral 1350 (-10%)", S.cell(1350.0)),
            ("lateral 1650 (+10%)", S.cell(1650.0)),
        ]:
            compact(f"SEARCHED @ {tag}", g, cc, seeds)


if __name__ == "__main__":
    main()
