"""D3b OPPONENT-GATE AUDIT — measure the scripted T3's own trigger, not a proxy for it.

The geometry-lens claim under test says the bandit "lost its 4 deg fire gate" and that its
misses are explained by firing where the sim's p_hit is ~0. Both halves are claims about
`opponent.wez_fire(bandit, me)`, whose input is `relative(bandit, me).naive_lead_error` --
a quantity that is NOT in `replay_kill_seed62.csv` (that CSV's `ata_deg`/`lead_error_deg`
are MY nose-to-LOS / MY corrected lay). So the claim was argued from a reconstructed proxy.

This script measures the gate itself. `env.step` computes the opponent action from
`self._states` BEFORE stepping (env.py:187-190), which is exactly the snapshot the
`roll_once` controller is handed, so calling `wez_fire(bd, me)` inside the controller
returns the bit-identical decision the sim used on that step.

Everything is imported from `_diag_true_mpc`; nothing is reimplemented.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d3b_oppgate.py'
"""

from __future__ import annotations

import csv
import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import p_hit_range, relative
from astrum_dogfight.opponent import wez_fire

DEG = 180.0 / math.pi
NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
OUT = "/workspace/controllers/rl/runs/d3b/oppgate_seed{}.csv"

COLS = [
    "step",
    "t_s",
    "rng_m",
    # --- the bandit's own gate, exactly as `wez_fire` reads it ---------------
    "opp_wez_fire",
    "opp_in_range",
    "opp_naive_lead_deg",
    "opp_lead_corr_deg",
    "opp_lay_miss_m",
    "opp_ata_deg",
    "opp_fsm_state",
    "opp_ammo",
    "opp_health",
    "my_health",
    "my_ammo",
    # --- the honest per-round hit probability at this range ------------------
    "phit_range",
    # --- my side, for the symmetric comparison ------------------------------
    "my_wez_fire_proxy",
    "my_naive_lead_deg",
    "my_lay_miss_m",
    "my_ata_deg",
    "my_trigger",
]


def fly(seed: int, n_steps: int, genome: np.ndarray) -> list[dict]:
    cell = M.CELLS["merge1500"]
    gm = M.Genome(6, 3, "bank", True)
    env, base = M.build_env(cell, seed, n_steps * M.CONTROL_DT_S)
    rows: list[dict] = []
    t = 0.0

    def ctl(step, me, bd, rel):
        nonlocal t
        rg_opp = relative(bd, me)  # me seen from the bandit == what wez_fire reads
        rows.append(
            {
                "step": step,
                "t_s": round(t, 6),
                "rng_m": float(rel.rng),
                "opp_wez_fire": int(bool(wez_fire(bd, me))),
                "opp_in_range": int(bool(rg_opp.in_range)),
                "opp_naive_lead_deg": float(rg_opp.naive_lead_error) * DEG,
                "opp_lead_corr_deg": float(rg_opp.lead_error) * DEG,
                "opp_lay_miss_m": float(rg_opp.lay_miss_m),
                "opp_ata_deg": float(rg_opp.ata) * DEG,
                "opp_fsm_state": str(base.opponent.state),
                "opp_ammo": float(bd.ammo),
                "opp_health": float(bd.health),
                "my_health": float(me.health),
                "my_ammo": float(me.ammo),
                "phit_range": float(p_hit_range(float(rel.rng))),
                "my_wez_fire_proxy": int(bool(wez_fire(me, bd))),
                "my_naive_lead_deg": float(rel.naive_lead_error) * DEG,
                "my_lay_miss_m": float(rel.lay_miss_m),
                "my_ata_deg": float(rel.ata) * DEG,
                "my_trigger": float(M._trigger(rel)),
            }
        )
        t += M.CONTROL_DT_S
        return gm.act(genome, step, n_steps, me)

    r = M.roll_once(env, base, ctl, n_steps, keep=False)
    env.close()
    print(
        f"seed {seed}: event={r.event} kill={r.kill} steps={r.steps} "
        f"hp_dealt={r.hp_dealt:.1f} hp_taken={r.hp_taken:.1f} tleth={r.tleth_s:.3f} "
        f"rounds={r.rounds}"
    )
    return rows


def summarise(seed: int, rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n--- seed {seed}: {n} control steps ---")
    fired = [i for i in range(n - 1) if rows[i]["opp_ammo"] - rows[i + 1]["opp_ammo"] > 0]
    gate = [i for i in range(n) if rows[i]["opp_wez_fire"] == 1]
    print(f"  opp ammo-decrement steps ({len(fired)}): {fired}")
    print(f"  opp wez_fire==1 steps   ({len(gate)}): {gate}")
    print(f"  agree: {sorted(set(fired)) == sorted(set(gate) & set(range(n - 1)))}")
    inr = [i for i in range(n) if rows[i]["opp_in_range"] == 1]
    print(
        f"  opp in_range==1 steps: {len(inr)} of {n}  (first {inr[0] if inr else None}, "
        f"last {inr[-1] if inr else None})"
    )
    # gate-off-while-in-range: WHICH clause is doing the blocking
    blocked = [i for i in inr if rows[i]["opp_wez_fire"] == 0]
    print(f"  in_range but NOT firing: {len(blocked)} steps")
    if blocked:
        lead = [rows[i]["opp_naive_lead_deg"] for i in blocked]
        print(
            f"    their naive_lead over those steps: min {min(lead):.2f} deg "
            f"max {max(lead):.2f} deg  (gate = {C.FIRE_LEAD_RAD * DEG:.1f} deg)"
        )
        under = [i for i in blocked if rows[i]["opp_naive_lead_deg"] < C.FIRE_LEAD_RAD * DEG]
        print(f"    of those, naive_lead ALREADY < 4 deg (so blocked by something else): {under}")
    # expected hits under the honest curve, and under the curve x an aim gate
    eh = 0.0
    for i in fired:
        d = rows[i]["opp_ammo"] - rows[i + 1]["opp_ammo"]
        eh += d * rows[i]["phit_range"]
    print(
        f"  opp rounds {sum(rows[i]['opp_ammo'] - rows[i + 1]['opp_ammo'] for i in fired):.0f}"
        f"  E[hits] under p_hit_range = {eh:.2f} ({2.0 * eh:.1f} HP)"
    )
    print(f"  ACTUAL HP taken by me: {rows[0]['my_health'] - rows[-1]['my_health']:.1f}")
    lay = [rows[i]["opp_lay_miss_m"] for i in fired]
    print(
        f"  opp lay_miss_m over its firing steps: min {min(lay):.1f} max {max(lay):.1f} m"
        f"  (kill sphere = {C.HIT_RADIUS_M:.0f} m)"
    )
    nlead = [rows[i]["opp_naive_lead_deg"] for i in fired]
    print(f"  opp naive_lead over its firing steps: min {min(nlead):.2f} max {max(nlead):.2f} deg")
    print(f"  opp FSM states while firing: {sorted(set(rows[i]['opp_fsm_state'] for i in fired))}")
    print("  opp FSM state timeline: ", end="")
    prev = None
    for i in range(n):
        s = rows[i]["opp_fsm_state"]
        if s != prev:
            print(f"[{i}:{s}]", end="")
            prev = s
    print()


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    g = np.asarray(z["genome"], float)
    for seed, n in ((62, 576), (28, 576)):
        rows = fly(seed, n, g)
        summarise(seed, rows)
        with open(OUT.format(seed), "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {OUT.format(seed)}")


if __name__ == "__main__":
    main()
