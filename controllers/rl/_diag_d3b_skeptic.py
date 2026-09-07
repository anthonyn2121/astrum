"""D3b SKEPTIC — is the saved winning genome BFM, or an open-loop slice that fits ONE start?

The genome is open loop with respect to the bandit by construction (`Genome.act` reads own-ship
attitude only, `_diag_true_mpc.py:197-220`), so it cannot be *reacting* to him. That leaves two
readings of the kill:

  (a) it is a maneuver whose SHAPE solves the merge, in which case it should survive small,
      physically meaningless changes to the start and to its own internal timing; or
  (b) it is a fixed schedule that happens to intersect this one bandit's own turn, in which case
      the kill is a co-incidence of geometry and clock and will not survive them.

Nothing here changes any threshold, reward or opponent: every arm flies the SAME genome through the
SAME honest trigger (`M._trigger`, overwritten inside `M.roll_once` for every arm) and the same
`M.score`. Only the START GEOMETRY or the SEGMENT CLOCK moves.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d3b_skeptic.py'
"""

from __future__ import annotations

import argparse

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402

NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
N_STEPS = 576


def cell(lateral_m: float, separation_m: float = 3000.0, **over) -> dict:
    c = M._merge_cell(lateral_m, separation_m)
    c.update(over)
    return c


def mirrored(lateral_m: float, separation_m: float = 3000.0) -> dict:
    """The same start reflected about the learner's own heading: bandit offset LEFT, not right."""
    c = M._merge_cell(lateral_m, separation_m)
    c["aspect_deg"] = -c["aspect_deg"]
    return c


_ENVS: dict[tuple, tuple] = {}


def _env_for(c: dict, seed: int):
    """jitter=0 => reset() re-runs the FIXED scenario, so one env serves every genome at this
    (cell, seed) and the start is bit-identical across arms."""
    key = (tuple(sorted((k, float(v)) for k, v in c.items())), seed)
    if key not in _ENVS:
        _ENVS[key] = M.build_env(c, seed, N_STEPS * M.CONTROL_DT_S)
    return _ENVS[key]


def fly(g: np.ndarray, c: dict, seed: int, shift: int = 0, segs: int = 6) -> M.Rollout:
    gm = M.Genome(segs, 3, "bank", True)
    env, base = _env_for(c, seed)

    def ctl(step, me, bd, rel):
        # shift<0 delays the schedule (seg 0 is held longer); shift>0 runs it early. Clamped so a
        # delayed clock never indexes before segment 0.
        return gm.act(g, min(N_STEPS - 1, max(0, step + shift)), N_STEPS, me)

    return M.roll_once(env, base, ctl, N_STEPS, keep=False)


def row(tag: str, rs: list[M.Rollout]) -> None:
    k = sum(r.kill for r in rs)
    hp = [r.hp_dealt for r in rs]
    print(
        f"  {tag:34s} kills {k}/{len(rs)}  hp {np.mean(hp):6.1f} [{min(hp):5.1f},{max(hp):5.1f}]  "
        f"tLeth {np.mean([r.tleth_s for r in rs]):5.3f}s  t600 {np.mean([r.t600_s for r in rs]):5.3f}s  "
        f"rngMin {np.mean([r.rng_min for r in rs]):7.0f}m  rnds {np.mean([r.rounds for r in rs]):5.1f}  "
        f"dalt {np.mean([r.dalt_m for r in rs]):+7.0f}m  dead {sum(r.dead for r in rs)}  "
        f"steps {np.mean([r.steps for r in rs]):5.0f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    seeds = SEEDS[: a.seeds]
    d = np.load(NPZ, allow_pickle=True)
    g = np.asarray(d["genome"], float)
    print(f"genome {np.round(g, 4).tolist()}\n  seeds {seeds}\n")

    if a.part in ("all", "geom"):
        print("A. START-GEOMETRY PERTURBATION (same genome, same trigger, same score)")
        for tag, c in [
            ("BASELINE lateral 1500 (as searched)", cell(1500.0)),
            ("lateral 1350 (-10%)", cell(1350.0)),
            ("lateral 1650 (+10%)", cell(1650.0)),
            ("lateral 1200 (-20%)", cell(1200.0)),
            ("lateral 1800 (+20%)", cell(1800.0)),
            ("MIRRORED lateral -1500", mirrored(1500.0)),
            ("sep 2700 (-10%), lateral 1500", cell(1500.0, 2700.0)),
            ("sep 3300 (+10%), lateral 1500", cell(1500.0, 3300.0)),
            ("no perch me_dalt_ft 0", cell(1500.0, me_dalt_ft=0.0)),
            ("opp_heading 170 (-10)", cell(1500.0, opp_heading_deg=170.0)),
            ("opp_heading 190 (+10)", cell(1500.0, opp_heading_deg=190.0)),
        ]:
            row(tag, [fly(g, c, s) for s in seeds])

    if a.part in ("all", "fine"):
        print("\nB2. SEGMENT-CLOCK SHIFT, FINE (how wide is the basin, in control steps?)")
        for k in (-2, -1, 0, 1, 2):
            row(
                f"shift {k:+3d} steps ({k * M.CONTROL_DT_S:+.3f} s)",
                [fly(g, cell(1500.0), s, shift=k) for s in seeds],
            )

    if a.part in ("all", "clock"):
        print("\nB. SEGMENT-CLOCK SHIFT (identical geometry; the schedule slid in time)")
        for k in (-24, -12, -6, -3, 0, 3, 6, 12, 24):
            row(
                f"shift {k:+3d} steps ({k * M.CONTROL_DT_S:+.3f} s)",
                [fly(g, cell(1500.0), s, shift=k) for s in seeds],
            )

    if a.part in ("all", "crude"):
        print("\nC. HOW CRUDE CAN THE MANEUVER BE AND STILL KILL?")
        seg0, seg1 = g[0:3], g[3:6]
        one = np.concatenate([seg0] * 6)
        row("seg0 ONLY, all 24 s (one hard bank)", [fly(one, cell(1500.0), s) for s in seeds])
        row(
            "seg1 ONLY, all 24 s", [fly(np.concatenate([seg1] * 6), cell(1500.0), s) for s in seeds]
        )
        two = np.concatenate([seg0, seg1] * 3)
        row("seg0,seg1 repeated (2-seg, 4 s each)", [fly(two, cell(1500.0), s) for s in seeds])

        # a 2-segment genome coded by hand from ROUNDED numbers: bank 115/45, load 1.0/0.5, thr 0.6/1.0
        def enc(bank_deg, load, thr):
            return [bank_deg / M.BANK_MAX_DEG, 2.0 * load - 1.0, 2.0 * thr - 1.0]

        crude = np.asarray(enc(115.0, 1.0, 0.60) * 1 + enc(45.0, 0.5, 1.00) * 5, float)
        row(
            "CRUDE hand-coded 115/1.0/0.6->45/0.5/1.0", [fly(crude, cell(1500.0), s) for s in seeds]
        )
        for bank in (95.0, 105.0, 115.0, 125.0):
            cr = np.asarray(enc(bank, 1.0, 0.60) * 1 + enc(45.0, 0.5, 1.00) * 5, float)
            row(f"CRUDE bank {bank:5.1f} phase1", [fly(cr, cell(1500.0), s) for s in seeds])
        for tsw in (3.0, 3.5, 4.0, 4.5, 5.0):
            n = max(1, int(round(24.0 / tsw)))
            gg = np.asarray(enc(115.0, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * (n - 1), float)
            row(
                f"CRUDE switch at {tsw:.1f}s (segs {n})",
                [fly(gg, cell(1500.0), s, segs=n) for s in seeds],
            )


if __name__ == "__main__":
    main()
