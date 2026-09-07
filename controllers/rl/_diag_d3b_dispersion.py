"""D3b VERIFY — is kill-vs-no-kill on the saved genome decided by gun dispersion alone?

The claim under test (anti-BFM lens): "the same geometry deals 96-100 HP on EVERY seed and the
three failures are one or two 2-HP hits short ... missed by 1-2 rounds out of ~68".

Two things it asserts that the 8-seed table cannot establish, and that this script measures:

  (A) "on every seed" — the 8 seeds in the table are the seeds CMA-ES optimised the mean score
      over, so the hp_dealt spread 96-100 is an IN-SAMPLE, selected spread. Re-fly the identical
      genome on seeds the search never saw and read the spread there.
  (B) "out of ~68" — 68 is the KILL seed's round count. The non-kill seeds do not terminate at
      step 172, so they keep flying (segments 2-5, which never executed on the kill) and keep
      firing. Count the rounds each seed actually fires, and the HITS those extra rounds buy.

Also measured, because the claim's mechanism depends on it: whether the GEOMETRY (both jets'
pos/vel/attitude) really is bit-identical across seeds over the shared prefix, or whether the
scripted T3 reacts to damage and the seeds diverge before the shot.

Nothing here is re-implemented: `Genome`, `build_env`, `_trigger`, `roll_once` are imported from
`_diag_true_mpc`, and the recorder is passed AS the controller so rows are sampled inside the
search's own rollout loop.
"""

from __future__ import annotations

import argparse

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402

SEARCH_SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
# Explicit, never hash-derived (D3a defect 4). 40 seeds the search never touched.
FRESH_SEEDS = [1000 + 7 * i for i in range(40)]

NPZ = "runs/d3b/d3b_best.npz"
PREFIX_ROWS = 173  # the kill episode's length: the shared, identical-geometry prefix


def geom_vec(me, bd) -> np.ndarray:
    """Everything that defines the GEOMETRY. Deliberately EXCLUDES health and ammo, which are the
    only channels the gun PRNG can touch, so a nonzero divergence here means the jets flew
    differently."""
    out = []
    for a in (me, bd):
        out.extend(
            [
                *np.asarray(a.pos, float),
                *np.asarray(a.vel, float),
                a.phi,
                a.theta,
                a.psi,
                a.p,
                a.q,
                a.r,
                a.nz,
                a.airspeed,
            ]
        )
    return np.asarray(out, float)


def fly(cell, g, gm, n_steps, seed):
    """Fly the genome at one sim seed, recording per-step rows from inside `roll_once`."""
    env, base = M.build_env(cell, seed, n_steps * M.CONTROL_DT_S)
    rows: list[dict] = []

    def rec(step, me, bd, rel):
        a = gm.act(g, step, n_steps, me)
        rows.append(
            dict(
                step=step,
                t_s=step * M.CONTROL_DT_S,
                fire=M._trigger(rel) > 0.0,
                lethal=bool(rel.lay_miss_m < C.HIT_RADIUS_M and rel.rng <= M.R_LETHAL_M),
                rng=float(rel.rng),
                lay_miss=float(rel.lay_miss_m),
                aspect=float(np.degrees(rel.aspect)),
                ammo=float(me.ammo),
                opp_hp=float(bd.health),
                geom=geom_vec(me, bd),
            )
        )
        return a

    r = M.roll_once(env, base, rec, n_steps, keep=False)
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    post = dict(ammo=float(me.ammo), opp_hp=float(bd.health))
    env.close()
    return r, rows, post


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--cell", default="merge1500")
    ap.add_argument("--horizon-s", type=float, default=24.0)
    ap.add_argument("--fresh", type=int, default=40)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=False)
    g = np.asarray(z["genome"], float)
    cell = M.CELLS[a.cell]
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    n_steps = int(round(a.horizon_s / M.CONTROL_DT_S))
    assert gm.dim == g.size, (gm.dim, g.size)

    seeds = SEARCH_SEEDS + FRESH_SEEDS[: a.fresh]
    print("#" * 100)
    print("#  D3b VERIFY — dispersion-vs-flying, IN-SAMPLE (8) vs FRESH (%d) sim seeds" % a.fresh)
    print("#" * 100)
    print(f"   npz {a.npz}  cell {a.cell}  horizon {a.horizon_s} s = {n_steps} control steps")
    print(f"   genome {np.array2string(g, precision=6, max_line_width=200)}")
    print(f"   HP_MAX {C.HP_MAX}  AMMO_MAX {C.AMMO_MAX}  hits to kill {int(C.HP_MAX / 2.0)}")
    print()

    hdr = (
        f"   {'seed':>6} {'kind':>9} {'ev':>6} {'k':>2} {'hp':>6} {'hits':>5} {'steps':>5} "
        f"{'rnds':>5} {'fireSt':>6} {'lethSt':>6} {'hp@173':>7} {'rn@173':>7} {'rnAfter':>7} "
        f"{'hitAfter':>8} {'geomDiv':>9}"
    )
    print(hdr)
    ref_geom = None
    table = []
    for sd in seeds:
        kind = "in-sample" if sd in SEARCH_SEEDS else "fresh"
        r, rows, post = fly(cell, g, gm, n_steps, sd)
        gm_stack = np.stack([w["geom"] for w in rows])
        if ref_geom is None:
            ref_geom = gm_stack
        k = min(len(gm_stack), len(ref_geom), PREFIX_ROWS)
        div = float(np.max(np.abs(gm_stack[:k] - ref_geom[:k])))
        fire_st = sum(1 for w in rows if w["fire"])
        leth_st = sum(1 for w in rows if w["lethal"])
        hits = int(round(r.hp_dealt / 2.0))
        # state at the end of the shared prefix (row index 172 exists on every seed)
        idx = min(PREFIX_ROWS - 1, len(rows) - 1)
        if len(rows) > PREFIX_ROWS:
            hp173 = C.HP_MAX - rows[PREFIX_ROWS]["opp_hp"]
            rn173 = C.AMMO_MAX - rows[PREFIX_ROWS]["ammo"]
        else:
            hp173 = r.hp_dealt
            rn173 = C.AMMO_MAX - post["ammo"]
        rn_after = r.rounds - rn173
        hit_after = int(round((r.hp_dealt - hp173) / 2.0))
        print(
            f"   {sd:>6} {kind:>9} {str(r.event):>6} {r.kill:>2} {r.hp_dealt:>6.0f} {hits:>5} "
            f"{r.steps:>5} {r.rounds:>5} {fire_st:>6} {leth_st:>6} {hp173:>7.0f} {rn173:>7.0f} "
            f"{rn_after:>7.0f} {hit_after:>8} {div:>9.3e}"
        )
        table.append(
            dict(
                seed=sd,
                kind=kind,
                kill=r.kill,
                hp=r.hp_dealt,
                hits=hits,
                steps=r.steps,
                rounds=r.rounds,
                fire_st=fire_st,
                leth_st=leth_st,
                hp173=hp173,
                rn173=rn173,
                rn_after=rn_after,
                hit_after=hit_after,
                div=div,
                idx=idx,
            )
        )

    def summ(name, sub):
        if not sub:
            return
        hp = np.array([t["hp"] for t in sub], float)
        hits = np.array([t["hits"] for t in sub], float)
        rounds = np.array([t["rounds"] for t in sub], float)
        rn173 = np.array([t["rn173"] for t in sub], float)
        hp173 = np.array([t["hp173"] for t in sub], float)
        ra = np.array([t["rn_after"] for t in sub], float)
        ha = np.array([t["hit_after"] for t in sub], float)
        kills = sum(t["kill"] for t in sub)
        print(f"\n   {name}  n = {len(sub)}")
        print(f"     kills ........... {kills}/{len(sub)}")
        print(
            "     hp_dealt ........ min %.0f  max %.0f  mean %.2f  sd %.2f  distinct %s"
            % (
                hp.min(),
                hp.max(),
                hp.mean(),
                hp.std(ddof=1) if len(sub) > 1 else 0.0,
                sorted(set(hp.tolist())),
            )
        )
        print(
            "     hits ............ min %.0f  max %.0f  mean %.2f (need %d)"
            % (hits.min(), hits.max(), hits.mean(), int(C.HP_MAX / 2.0))
        )
        print(
            "     rounds fired .... min %.0f  max %.0f  mean %.2f"
            % (rounds.min(), rounds.max(), rounds.mean())
        )
        print(
            "     by end of shared prefix (row %d): rounds mean %.2f, hp mean %.2f"
            % (PREFIX_ROWS, rn173.mean(), hp173.mean())
        )
        print(
            "     AFTER the prefix: rounds mean %.2f (max %.0f), hits mean %.2f (max %.0f)"
            "  -> marginal hit rate %d/%d = %.3f"
            % (
                ra.mean(),
                ra.max(),
                ha.mean(),
                ha.max(),
                int(ha.sum()),
                int(ra.sum()),
                (ha.sum() / ra.sum()) if ra.sum() else float("nan"),
            )
        )
        pooled_p = hits.sum() / rounds.sum()
        print(
            "     pooled per-round hit fraction = %d/%d = %.4f"
            % (int(hits.sum()), int(rounds.sum()), pooled_p)
        )
        print(
            "     max geometry divergence vs the first seed over the shared prefix = %.3e"
            % max(t["div"] for t in sub)
        )

    summ(
        "IN-SAMPLE (the 8 seeds CMA-ES selected on)", [t for t in table if t["kind"] == "in-sample"]
    )
    summ("FRESH (never seen by the search)", [t for t in table if t["kind"] == "fresh"])
    allt = table
    print(
        "\n   ALL %d seeds: kills %d/%d, hp_dealt distinct %s"
        % (
            len(allt),
            sum(t["kill"] for t in allt),
            len(allt),
            sorted({t["hp"] for t in allt}),
        )
    )
    print("\n   NOTE: every number above is a re-flight of ONE genome; the fresh block is the only")
    print("   out-of-sample statistic here. The in-sample block may not be quoted as a hit rate.")


if __name__ == "__main__":
    main()
