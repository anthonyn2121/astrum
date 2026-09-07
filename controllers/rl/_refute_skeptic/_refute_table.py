"""Independent recomputation of the D3b perturbation table.

Reuses `_diag_d3b_skeptic`'s OWN `fly`/`cell`/`mirrored` so no code path is reimplemented, but
enumerates the arms explicitly, hashes each arm's (genome, cell, shift, segs) so DUPLICATE arms are
detected, and reports per-seed detail plus the `event` histogram so the "failure mode is always the
same" clause can be checked rather than assumed.

  --mode cached  : the skeptic's own env cache (up to 11 live envs in one process)
  --mode fresh   : one env per rollout, closed immediately (the discipline `precheck` uses)
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")

import _diag_d3b_skeptic as S  # noqa: E402
import _diag_true_mpc as M  # noqa: E402

NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"


def enc(bank_deg, load, thr):
    return [bank_deg / M.BANK_MAX_DEG, 2.0 * load - 1.0, 2.0 * thr - 1.0]


def arms(g: np.ndarray, part: str) -> list[tuple[str, str, np.ndarray, dict, int, int]]:
    """(part, tag, genome, cell, shift, segs)."""
    out = []
    if part in ("all", "geom"):
        for tag, c in [
            ("BASELINE lateral 1500 (as searched)", S.cell(1500.0)),
            ("lateral 1350 (-10%)", S.cell(1350.0)),
            ("lateral 1650 (+10%)", S.cell(1650.0)),
            ("lateral 1200 (-20%)", S.cell(1200.0)),
            ("lateral 1800 (+20%)", S.cell(1800.0)),
            ("MIRRORED lateral -1500", S.mirrored(1500.0)),
            ("sep 2700 (-10%), lateral 1500", S.cell(1500.0, 2700.0)),
            ("sep 3300 (+10%), lateral 1500", S.cell(1500.0, 3300.0)),
            ("no perch me_dalt_ft 0", S.cell(1500.0, me_dalt_ft=0.0)),
            ("opp_heading 170 (-10)", S.cell(1500.0, opp_heading_deg=170.0)),
            ("opp_heading 190 (+10)", S.cell(1500.0, opp_heading_deg=190.0)),
        ]:
            out.append(("geom", tag, g, c, 0, 6))
    if part in ("all", "fine"):
        for k in (-2, -1, 0, 1, 2):
            out.append(
                (
                    "fine",
                    f"shift {k:+3d} steps ({k * M.CONTROL_DT_S:+.3f} s)",
                    g,
                    S.cell(1500.0),
                    k,
                    6,
                )
            )
    if part in ("all", "clock"):
        for k in (-24, -12, -6, -3, 0, 3, 6, 12, 24):
            out.append(
                (
                    "clock",
                    f"shift {k:+3d} steps ({k * M.CONTROL_DT_S:+.3f} s)",
                    g,
                    S.cell(1500.0),
                    k,
                    6,
                )
            )
    if part in ("all", "crude"):
        seg0, seg1 = g[0:3], g[3:6]
        out.append(
            (
                "crude",
                "seg0 ONLY, all 24 s (one hard bank)",
                np.concatenate([seg0] * 6),
                S.cell(1500.0),
                0,
                6,
            )
        )
        out.append(
            ("crude", "seg1 ONLY, all 24 s", np.concatenate([seg1] * 6), S.cell(1500.0), 0, 6)
        )
        out.append(
            (
                "crude",
                "seg0,seg1 repeated (2-seg, 4 s each)",
                np.concatenate([seg0, seg1] * 3),
                S.cell(1500.0),
                0,
                6,
            )
        )
        crude = np.asarray(enc(115.0, 1.0, 0.60) * 1 + enc(45.0, 0.5, 1.00) * 5, float)
        out.append(
            ("crude", "CRUDE hand-coded 115/1.0/0.6->45/0.5/1.0", crude, S.cell(1500.0), 0, 6)
        )
        for bank in (95.0, 105.0, 115.0, 125.0):
            cr = np.asarray(enc(bank, 1.0, 0.60) * 1 + enc(45.0, 0.5, 1.00) * 5, float)
            out.append(("crude", f"CRUDE bank {bank:5.1f} phase1", cr, S.cell(1500.0), 0, 6))
        for tsw in (3.0, 3.5, 4.0, 4.5, 5.0):
            n = max(1, int(round(24.0 / tsw)))
            gg = np.asarray(enc(115.0, 1.0, 0.60) + enc(45.0, 0.5, 1.00) * (n - 1), float)
            out.append(
                ("crude", f"CRUDE switch at {tsw:.1f}s (segs {n})", gg, S.cell(1500.0), 0, n)
            )
    return out


def key_of(gen: np.ndarray, c: dict, shift: int, segs: int) -> str:
    blob = json.dumps(
        dict(
            g=[round(float(x), 12) for x in np.asarray(gen, float).ravel()],
            c=sorted((k, round(float(v), 9)) for k, v in c.items()),
            shift=int(shift),
            segs=int(segs),
        ),
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def fly_fresh(gen, c, seed, shift, segs):
    gm = M.Genome(segs, 3, "bank", True)
    env, base = M.build_env(c, seed, S.N_STEPS * M.CONTROL_DT_S)
    try:

        def ctl(step, me, bd, rel):
            return gm.act(gen, min(S.N_STEPS - 1, max(0, step + shift)), S.N_STEPS, me)

        return M.roll_once(env, base, ctl, S.N_STEPS, keep=False)
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    ap.add_argument("--mode", default="cached", choices=("cached", "fresh"))
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    seeds = S.SEEDS[: a.seeds]
    g = np.asarray(np.load(NPZ, allow_pickle=True)["genome"], float)
    rows = arms(g, a.part)
    print(f"# mode={a.mode} part={a.part} seeds={seeds}  arms listed = {len(rows)}")
    print(f"# genome {np.round(g, 6).tolist()}\n")

    seen: dict[str, str] = {}
    print(
        f"{'key':>10} {'dup_of':>10} {'part':>5}  {'arm':<40} "
        f"{'K/8':>5} {'hpMean':>7} {'hpMin':>6} {'hpMax':>6} "
        f"{'tLethMean':>9} {'tLethMax':>8} {'t600':>6} "
        f"{'rngMean':>8} {'rngMin':>7} {'rngMax':>7} "
        f"{'rndMean':>7} {'rndMax':>7} {'daltMean':>9} {'dead':>4} {'stMean':>6}  events"
    )
    for part, tag, gen, c, shift, segs in rows:
        k = key_of(gen, c, shift, segs)
        dup = seen.get(k, "")
        if not dup:
            seen[k] = tag
        rs = [
            (
                S.fly(gen, c, s, shift=shift, segs=segs)
                if a.mode == "cached"
                else fly_fresh(gen, c, s, shift, segs)
            )
            for s in seeds
        ]
        hp = [r.hp_dealt for r in rs]
        tl = [r.tleth_s for r in rs]
        rg = [r.rng_min for r in rs]
        rn = [r.rounds for r in rs]
        ev = collections.Counter(str(r.event) for r in rs)
        print(
            f"{k:>10} {dup[:10]:>10} {part:>5}  {tag:<40} "
            f"{sum(r.kill for r in rs):>5} {np.mean(hp):>7.2f} {min(hp):>6.1f} {max(hp):>6.1f} "
            f"{np.mean(tl):>9.4f} {max(tl):>8.4f} {np.mean([r.t600_s for r in rs]):>6.3f} "
            f"{np.mean(rg):>8.1f} {min(rg):>7.1f} {max(rg):>7.1f} "
            f"{np.mean(rn):>7.1f} {max(rn):>7d} {np.mean([r.dalt_m for r in rs]):>+9.0f} "
            f"{sum(r.dead for r in rs):>4} {np.mean([r.steps for r in rs]):>6.0f}  "
            f"{dict(ev)}",
            flush=True,
        )
    ndup = len(rows) - len(seen)
    print(
        f"\n# DISTINCT arms by (genome, cell, shift, segs) = {len(seen)} of {len(rows)} rows "
        f"listed  => {ndup} duplicate row(s)"
    )


if __name__ == "__main__":
    main()
