"""P1: the BASIN in lateral separation with the full 8-seed n. P2: the OPPONENT REGIME per arm.

P1 answers 'does the kill survive ANY perturbation of the start' at the resolution the claim never
probed (+-10% = +-150 m; here +-2 m .. +-30 m).
P2 answers the claim's own confound caveat: it concedes lateral 1800 leaves the T3's 1659 m threat
gate. Measured here for EVERY arm: the bandit's peak bank and total heading swing, i.e. whether the
T3 actually flew a defensive break (BREAK_BANK_RAD = 85 deg) in that arm at all.
"""

from __future__ import annotations

import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402

NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
N = 576
DEG = 180.0 / math.pi


def cell(lateral_m: float, separation_m: float = 3000.0, **over) -> dict:
    c = M._merge_cell(lateral_m, separation_m)
    c.update(over)
    return c


def mirrored(lateral_m: float) -> dict:
    c = M._merge_cell(lateral_m)
    c["aspect_deg"] = -c["aspect_deg"]
    return c


_ENVS: dict[tuple, tuple] = {}


def env_for(c: dict, seed: int):
    key = (tuple(sorted((k, float(v)) for k, v in c.items())), seed)
    if key not in _ENVS:
        _ENVS[key] = M.build_env(c, seed, N * M.CONTROL_DT_S)
    return _ENVS[key]


def fly(g, c, seed):
    gm = M.Genome(6, 3, "bank", True)
    env, base = env_for(c, seed)
    ex = {"opp_phi": [], "opp_psi": [], "lay600": float("inf")}

    def ctl(step, me, bd, rel):
        ex["opp_phi"].append(abs(float(bd.phi)))
        ex["opp_psi"].append(float(bd.psi))
        if rel.rng <= M.R_LETHAL_M:
            ex["lay600"] = min(ex["lay600"], float(rel.lay_miss_m))
        return gm.act(np.asarray(g, float), step, N, me)

    r = M.roll_once(env, base, ctl, N, keep=False)
    psi = np.unwrap(np.asarray(ex["opp_psi"], float))
    ex["opp_phi_max_deg"] = float(np.max(ex["opp_phi"])) * DEG
    ex["opp_psi_swing_deg"] = float(np.max(psi) - np.min(psi)) * DEG
    ex["broke"] = ex["opp_phi_max_deg"] >= 0.9 * 85.0
    return r, ex


def main() -> None:
    g = np.asarray(np.load(NPZ, allow_pickle=True)["genome"], float)

    print("P1. BASIN IN LATERAL SEPARATION, all 8 seeds (baseline as searched = 1500 m)")
    print(
        f"  {'lateral':>8} {'kills':>7} {'hp':>7} {'tLeth':>7} {'t600':>6} {'rngMin':>7} "
        f"{'rnds':>6} {'lay600':>7} {'dalt':>7} {'steps':>6}"
    )
    for lat in (1470, 1480, 1485, 1490, 1495, 1498, 1500, 1502, 1505, 1510, 1520):
        rows = [fly(g, cell(float(lat)), s) for s in SEEDS]
        rs = [r for r, _ in rows]
        ex = [e for _, e in rows]
        print(
            f"  {lat:8d} {sum(r.kill for r in rs):5d}/8 "
            f"{np.mean([r.hp_dealt for r in rs]):7.1f} "
            f"{np.mean([r.tleth_s for r in rs]):7.3f} "
            f"{np.mean([r.t600_s for r in rs]):6.3f} "
            f"{np.mean([r.rng_min for r in rs]):7.0f} "
            f"{np.mean([r.rounds for r in rs]):6.1f} "
            f"{np.mean([e['lay600'] for e in ex]):7.1f} "
            f"{np.mean([r.dalt_m for r in rs]):+7.0f} "
            f"{np.mean([r.steps for r in rs]):6.0f}",
            flush=True,
        )

    print("\nP2. DID THE T3 ACTUALLY BREAK IN THIS ARM? (seed 62; own-ship path is identical)")
    arms = [
        ("BASELINE lateral 1500", cell(1500.0)),
        ("lateral 1350", cell(1350.0)),
        ("lateral 1650", cell(1650.0)),
        ("lateral 1200", cell(1200.0)),
        ("lateral 1800", cell(1800.0)),
        ("MIRRORED -1500", mirrored(1500.0)),
        ("sep 2700", cell(1500.0, 2700.0)),
        ("sep 3300", cell(1500.0, 3300.0)),
        ("no perch", cell(1500.0, me_dalt_ft=0.0)),
        ("opp_heading 170", cell(1500.0, opp_heading_deg=170.0)),
        ("opp_heading 190", cell(1500.0, opp_heading_deg=190.0)),
    ]
    print(
        f"  {'arm':24s} {'oppPhiMax':>10} {'oppPsiSwing':>12} {'broke>=76.5deg':>15} "
        f"{'steps':>6} {'rngMin':>7}"
    )
    for tag, c in arms:
        r, e = fly(g, c, 62)
        print(
            f"  {tag:24s} {e['opp_phi_max_deg']:10.1f} {e['opp_psi_swing_deg']:12.1f} "
            f"{str(e['broke']):>15} {r.steps:6d} {r.rng_min:7.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
