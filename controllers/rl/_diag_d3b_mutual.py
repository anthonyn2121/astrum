"""D3b MUTUAL-GEOMETRY probe: is the kill pass symmetric ("a mutual face shot")?

The anti-BFM claim reads |hca| in [171.8, 180.0] deg and aspect 147.9 -> 170.7 deg off the
replay CSV and concludes "no angular advantage is gained at any point ... this is a mutual face
shot". `hca` is `_wrap_pi(course(bandit) - course(me))` (geometry.py:215), i.e. the DIFFERENCE of
the two courses, so it is blind by construction to any rotation the two aircraft perform together.
Round counts are also not a clean symmetry test, because the two jets fire on DIFFERENT rules:
mine is `_diag_true_mpc._trigger` (rng <= 600 m AND lay_miss <= 15 m) while Tier-3 uses
`opponent.wez_fire` (in_range AND naive_lead_error < FIRE_LEAD_RAD), an angle-only gate.

So this script measures the rule-free mutual quantity: it re-flies the d3b genome on seed 62 and
evaluates `geometry.relative` in BOTH directions every control step -- mine (me -> bandit) and
HIS (bandit -> me) -- and prints ata / aspect / lay_miss / alpha_req / in_range side by side.
Everything printed is measured in this run; nothing is estimated.
"""

from __future__ import annotations

import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import geometry as G  # noqa: E402

DEG = 180.0 / math.pi
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / M.CONTROL_DT_S))
SEED = 62
GENOME = np.load("runs/d3b/d3b_best.npz")["genome"].astype(float)
GM = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)


def main() -> None:
    env, base = M.build_env(M.CELLS["merge1500"], SEED, HORIZON_S)
    rows: list[dict] = []

    def rec(step, me, bd, rel):
        rel_op = G.relative(bd, me)  # the SAME function, roles swapped
        rows.append(
            dict(
                step=step,
                t=step * M.CONTROL_DT_S,
                rng=float(rel.rng),
                hca=float(rel.hca) * DEG,
                my_ata=float(rel.ata) * DEG,
                his_ata=float(rel_op.ata) * DEG,
                my_aspect=float(rel.aspect) * DEG,
                his_aspect=float(rel_op.aspect) * DEG,
                my_lay=float(rel.lay_miss_m),
                his_lay=float(rel_op.lay_miss_m),
                my_areq=float(rel.alpha_req) * DEG,
                his_areq=float(rel_op.alpha_req) * DEG,
                my_inr=bool(rel.in_range),
                his_inr=bool(rel_op.in_range),
                nz=float(me.nz),
                onz=float(bd.nz),
                psi=float(me.psi) * DEG,
                opsi=float(bd.psi) * DEG,
                alt=float(me.alt),
                hp=float(bd.health),
                myhp=float(me.health),
            )
        )
        return GM.act(GENOME, step, N_STEPS, me)

    r = M.roll_once(env, base, rec, N_STEPS, keep=False)
    env.close()
    print(f"seed {SEED}: steps={len(rows)} event={r.event} kill={r.kill} hp={r.hp_dealt}")

    print("\nstep     t     rng    myATA  hisATA   myLAY    hisLAY   myAREQ hisAREQ  nz  onz  hp myhp")
    for x in rows:
        if x["step"] % 12 == 0 or x["step"] >= 140:
            print(
                "%4d %6.3f %7.1f %7.2f %7.2f %8.1f %9.1f %7.1f %7.1f %4.1f %4.1f %4.0f %4.0f"
                % (
                    x["step"],
                    x["t"],
                    x["rng"],
                    x["my_ata"],
                    x["his_ata"],
                    x["my_lay"],
                    x["his_lay"],
                    x["my_areq"],
                    x["his_areq"],
                    x["nz"],
                    x["onz"],
                    x["hp"],
                    x["myhp"],
                )
            )

    # --- rule-free asymmetry summaries ------------------------------------------------
    n = len(rows)
    mine_better_lay = sum(1 for x in rows if x["my_lay"] < x["his_lay"])
    mine_better_ata = sum(1 for x in rows if x["my_ata"] < x["his_ata"])
    print(f"\nsteps where MY lay_miss < HIS: {mine_better_lay}/{n}")
    print(f"steps where MY ata     < HIS: {mine_better_ata}/{n}")

    # longest run of a strict advantage in BOTH, and what it costs in altitude
    def longest_run(pred):
        best = (0, -1, -1)
        i = 0
        while i < n:
            if pred(rows[i]):
                j = i
                while j + 1 < n and pred(rows[j + 1]):
                    j += 1
                if j - i + 1 > best[0]:
                    best = (j - i + 1, i, j)
                i = j + 1
            else:
                i += 1
        return best

    for label, pred in (
        ("my_lay<his_lay", lambda x: x["my_lay"] < x["his_lay"]),
        ("my_ata<his_ata", lambda x: x["my_ata"] < x["his_ata"]),
        ("both, nz>3", lambda x: x["my_lay"] < x["his_lay"] and x["my_ata"] < x["his_ata"] and x["nz"] > 3.0),
        ("nz>3 (load only)", lambda x: x["nz"] > 3.0),
    ):
        ln, i, j = longest_run(pred)
        if ln:
            print(
                "longest run %-18s %3d steps = %.3f s  (step %d..%d, t %.3f..%.3f) "
                "alt %.0f -> %.0f m (d %+.0f)  psi %+.1f deg  opsi %+.1f deg"
                % (
                    label,
                    ln,
                    ln * M.CONTROL_DT_S,
                    i,
                    j,
                    rows[i]["t"],
                    rows[j]["t"],
                    rows[i]["alt"],
                    rows[j]["alt"],
                    rows[j]["alt"] - rows[i]["alt"],
                    rows[j]["psi"] - rows[i]["psi"],
                    rows[j]["opsi"] - rows[i]["opsi"],
                )
            )
        else:
            print(f"longest run {label:<18} 0 steps")

    # the lethal window (my honest trigger) and the mutual numbers inside it
    fired = [x for x in rows if x["rng"] <= 600.0 and x["my_lay"] <= 15.0]
    if fired:
        print(
            "\nMY trigger-open window: %d steps = %.3f s, step %d..%d, rng %.1f..%.1f m"
            % (
                len(fired),
                len(fired) * M.CONTROL_DT_S,
                fired[0]["step"],
                fired[-1]["step"],
                fired[0]["rng"],
                fired[-1]["rng"],
            )
        )
        print(
            "  inside it: my_lay %.2f..%.2f m ; his_lay %.1f..%.1f m ; "
            "my_ata %.2f..%.2f deg ; his_ata %.2f..%.2f deg"
            % (
                min(x["my_lay"] for x in fired),
                max(x["my_lay"] for x in fired),
                min(x["his_lay"] for x in fired),
                max(x["his_lay"] for x in fired),
                min(x["my_ata"] for x in fired),
                max(x["my_ata"] for x in fired),
                min(x["his_ata"] for x in fired),
                max(x["his_ata"] for x in fired),
            )
        )
    # would HIS trigger ever have opened on the same rule?
    his = [x for x in rows if x["rng"] <= 600.0 and x["his_lay"] <= 15.0]
    print(f"steps where HIS geometry passes MY honest gate (rng<=600 and his_lay<=15): {len(his)}/{n}")
    print(f"min his_lay over the whole pass: {min(x['his_lay'] for x in rows):.1f} m")


if __name__ == "__main__":
    main()
