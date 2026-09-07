"""REFUTATION PROBE for the 'geom' claim: does the perturbation test have any POWER?

Four measurements, all through `M.roll_once` (same honest `M._trigger`), nothing reimplemented:

  T1  Is the OWN-SHIP path identical across the perturbed geometries? If it is, the arm did not
      perturb the maneuver at all -- it only MOVED THE TARGET, and the acceptance window under
      test is the gun's aperture, not the maneuver's basin.
  T2  Per-seed detail on the arms the claim aggregates as '0/8': how many DISTINCT trajectories
      does each 8-seed row actually contain? (a 0-round candidate is seed-independent by the
      search's own precheck, _diag_true_mpc.py:511)
  T3  BASIN WIDTH in lateral separation, finer than the claim's +-10%.
  T4  RE-TIMING: at each perturbed geometry, slide the SAME shape in time by one parameter.
      If the kill comes back, the SHAPE survived the perturbation and only the clock had to be
      re-solved -- which the claim's wording ('does not survive any perturbation') denies.
  T5  MIRROR CONTROL: mirrored geometry flown with the bank channel NEGATED (a chirality
      control the claim's MIRRORED arm omits).
"""

from __future__ import annotations

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402

NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
N = 576


def cell(lateral_m: float, separation_m: float = 3000.0, **over) -> dict:
    c = M._merge_cell(lateral_m, separation_m)
    c.update(over)
    return c


def mirrored(lateral_m: float, separation_m: float = 3000.0) -> dict:
    c = M._merge_cell(lateral_m, separation_m)
    c["aspect_deg"] = -c["aspect_deg"]
    return c


_ENVS: dict[tuple, tuple] = {}


def env_for(c: dict, seed: int):
    key = (tuple(sorted((k, float(v)) for k, v in c.items())), seed)
    if key not in _ENVS:
        _ENVS[key] = M.build_env(c, seed, N * M.CONTROL_DT_S)
    return _ENVS[key]


def fly(g, c, seed, shift=0, segs=6, keep=False):
    """Returns (rollout, extras). extras carries min lay_miss inside 600 m and the own-ship path."""
    gm = M.Genome(segs, 3, "bank", True)
    env, base = env_for(c, seed)
    ex = {"lay_min_600": float("inf"), "lay_min_any": float("inf"), "own": []}

    def ctl(step, me, bd, rel):
        ex["own"].append(M._state_vec(me, bd)[0:14].copy())
        ex["lay_min_any"] = min(ex["lay_min_any"], float(rel.lay_miss_m))
        if rel.rng <= M.R_LETHAL_M:
            ex["lay_min_600"] = min(ex["lay_min_600"], float(rel.lay_miss_m))
        return gm.act(np.asarray(g, float), min(N - 1, max(0, step + shift)), N, me)

    r = M.roll_once(env, base, ctl, N, keep=keep)
    ex["own"] = np.asarray(ex["own"], float)
    return r, ex


def agg(tag, pairs):
    rs = [p[0] for p in pairs]
    ex = [p[1] for p in pairs]
    k = sum(r.kill for r in rs)
    hp = [r.hp_dealt for r in rs]
    print(
        f"  {tag:36s} kills {k}/{len(rs)}  hp {np.mean(hp):6.1f}  "
        f"tLeth {np.mean([r.tleth_s for r in rs]):5.3f}s  t600 {np.mean([r.t600_s for r in rs]):5.3f}s "
        f"rngMin {np.mean([r.rng_min for r in rs]):7.0f}m  rnds {np.mean([r.rounds for r in rs]):5.1f} "
        f"layMin600 {np.mean([e['lay_min_600'] for e in ex]):8.1f}m  "
        f"dalt {np.mean([r.dalt_m for r in rs]):+7.0f}m dead {sum(r.dead for r in rs)} "
        f"steps {np.mean([r.steps for r in rs]):5.0f}",
        flush=True,
    )
    return rs, ex


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    g = np.asarray(d["genome"], float)

    # ------------------------------------------------------------------ T1
    print("T1. IS THE OWN-SHIP PATH THE SAME ACROSS ARMS? (pos,vel,euler,pqr,nz,vt; seed 62)")
    ref = fly(g, cell(1500.0), 62)
    arms = {
        "lateral 1350": cell(1350.0),
        "lateral 1650": cell(1650.0),
        "lateral 1200": cell(1200.0),
        "lateral 1800": cell(1800.0),
        "MIRRORED -1500": mirrored(1500.0),
        "sep 2700": cell(1500.0, 2700.0),
        "sep 3300": cell(1500.0, 3300.0),
        "no perch": cell(1500.0, me_dalt_ft=0.0),
        "opp_heading 170": cell(1500.0, opp_heading_deg=170.0),
        "opp_heading 190": cell(1500.0, opp_heading_deg=190.0),
    }
    kept = {}
    for tag, c in arms.items():
        r, ex = fly(g, c, 62)
        kept[tag] = (r, ex)
        n = min(len(ref[1]["own"]), len(ex["own"]))
        dv = float(np.max(np.abs(ref[1]["own"][:n] - ex["own"][:n]))) if n else float("nan")
        print(
            f"  {tag:20s} steps {r.steps:4d}  overlap {n:4d}  "
            f"max|d own-ship state| vs baseline seed62 = {dv:.3e}",
            flush=True,
        )

    # ------------------------------------------------------------------ T2
    print("\nT2. PER-SEED DETAIL: how many DISTINCT rollouts does a '0/8' row contain?")
    for tag, c in [
        ("BASELINE 1500", cell(1500.0)),
        ("lateral 1350", cell(1350.0)),
        ("lateral 1650", cell(1650.0)),
        ("opp_heading 190", cell(1500.0, opp_heading_deg=190.0)),
    ]:
        rows = [fly(g, c, s) for s in SEEDS]
        sig = {}
        for s, (r, e) in zip(SEEDS, rows):
            key = (r.steps, r.rounds, round(r.hp_dealt, 3), round(r.dalt_m, 3))
            sig.setdefault(key, []).append(s)
        print(f"  {tag:18s} distinct (steps,rounds,hp,dalt) signatures = {len(sig)}")
        for key, ss in sig.items():
            print(
                f"      steps {key[0]:4d} rnds {key[1]:4d} hp {key[2]:6.1f} dalt {key[3]:+8.1f} "
                f"<- seeds {ss}"
            )

    # ------------------------------------------------------------------ T3
    print("\nT3. BASIN WIDTH in lateral separation (seed 62; 0-round arms are seed-independent)")
    for lat in (1400, 1425, 1450, 1470, 1485, 1495, 1500, 1505, 1515, 1530, 1550, 1575, 1600):
        agg(f"lateral {lat}", [fly(g, cell(float(lat)), 62)])

    # ------------------------------------------------------------------ T4
    print("\nT4. RE-TIMING THE SAME SHAPE at each perturbed geometry (seed 62 screen)")
    shifts = list(range(-48, 49, 4))
    hits = []
    for tag, c in arms.items():
        best = None
        line = []
        for sh in shifts:
            r, e = fly(g, c, 62, shift=sh)
            line.append((sh, r.kill, r.hp_dealt, r.rounds, e["lay_min_600"]))
            key = (r.kill, r.hp_dealt, -e["lay_min_600"])
            if best is None or key > best[0]:
                best = (key, sh, r, e)
        pos = [x for x in line if x[2] > 0.0]
        print(
            f"  {tag:20s} best shift {best[1]:+4d} ({best[1] * M.CONTROL_DT_S:+.3f}s) "
            f"kill {best[2].kill} hp {best[2].hp_dealt:6.1f} rnds {best[2].rounds:4d} "
            f"layMin600 {best[3]['lay_min_600']:8.1f}m   shifts with hp>0: "
            f"{[(x[0], x[2]) for x in pos]}",
            flush=True,
        )
        if best[2].hp_dealt > 0.0:
            hits.append((tag, c, best[1]))

    print("\nT4b. RE-TIMED ARMS THAT DEALT DAMAGE, confirmed over all 8 seeds")
    if not hits:
        print("  none")
    for tag, c, sh in hits:
        agg(f"{tag} shift {sh:+d}", [fly(g, c, s, shift=sh) for s in SEEDS])

    # ------------------------------------------------------------------ T5
    print("\nT5. MIRROR CONTROL: mirrored geometry, bank channel NEGATED (chirality control)")
    gm_ = g.copy()
    gm_[0::3] = -gm_[0::3]
    agg("MIRRORED geom + MIRRORED genome", [fly(gm_, mirrored(1500.0), s) for s in SEEDS])
    agg("MIRRORED geom, genome unchanged", [fly(g, mirrored(1500.0), s) for s in SEEDS])
    agg("BASELINE geom + MIRRORED genome", [fly(gm_, cell(1500.0), s) for s in SEEDS])


if __name__ == "__main__":
    main()
