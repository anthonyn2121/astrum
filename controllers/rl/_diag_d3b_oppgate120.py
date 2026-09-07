"""D3b OPPONENT GATE AT 120 Hz — the gate's own clock, not the control-step boundary.

`ActionRepeat(repeat=5)` (astrum_rl/envs.py:167-193) calls `AstrumDogfightEnv.step` five times
per 41.667 ms control step, and each of those calls recomputes `opponent.act` -> `wez_fire`
(env.py:190, opponent.py:493). So a control-step-boundary sample of the gate is only 1 of the 5
decisions the sim actually made, and the Rust `fire_guns` accumulator (`rate_hz`=100 rounds/s,
`accum += rate_hz*dt` per 120 Hz tick, combat/systems.rs:65) spawns rounds on the SIM tick.

This wraps `base.opponent.act` so every single 120 Hz gate evaluation is recorded with the exact
range, the exact `naive_lead_error` the gate reads, and the resulting `lay_miss_m`, then reports
the honest per-round hit probability at each tick a round was actually spawned.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d3b_oppgate120.py'
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import p_hit_range, relative

DEG = 180.0 / math.pi
NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"
GATE_DEG = C.FIRE_LEAD_RAD * DEG


def fly(seed: int, n_steps: int, genome: np.ndarray):
    cell = M.CELLS["merge1500"]
    gm = M.Genome(6, 3, "bank", True)
    env, base = M.build_env(cell, seed, n_steps * M.CONTROL_DT_S)

    ticks: list[dict] = []
    orig_act = base.opponent.act

    def spy(s, opponent):
        a = orig_act(s, opponent)
        rg = relative(s, opponent)  # target (me) seen from the bandit
        ticks.append(
            {
                "tick": len(ticks),
                "rng": float(rg.rng),
                "in_range": bool(rg.in_range),
                "naive_deg": float(rg.naive_lead_error) * DEG,
                "lay_miss": float(rg.lay_miss_m),
                "ata_deg": float(rg.ata) * DEG,
                "trig": float(a[4]),
                "ammo": float(s.ammo),
                "fsm": str(base.opponent.state),
                "my_hp": float(opponent.health),
            }
        )
        return a

    base.opponent.act = spy

    def ctl(step, me, bd, rel):
        return gm.act(genome, step, n_steps, me)

    r = M.roll_once(env, base, ctl, n_steps, keep=False)
    env.close()
    return r, ticks


def report(seed: int, r, ticks: list[dict]) -> None:
    n = len(ticks)
    print("=" * 88)
    print(
        f"seed {seed}: event={r.event} kill={r.kill} ctrl_steps={r.steps} sim_ticks={n} "
        f"hp_dealt={r.hp_dealt:.1f} hp_taken={r.hp_taken:.1f} rounds_mine={r.rounds}"
    )
    on = [t for t in ticks if t["trig"] > 0.5]
    inr = [t for t in ticks if t["in_range"]]
    print(f"  gate ON ticks: {len(on)} of {n}   in_range ticks: {len(inr)}")
    if on:
        print(
            f"  gate ON tick range: {on[0]['tick']}..{on[-1]['tick']} "
            f"(t {on[0]['tick'] / 120.0:.3f}..{on[-1]['tick'] / 120.0:.3f} s)"
        )
        # contiguous runs of gate-ON
        runs = []
        start = prev = on[0]["tick"]
        for t in on[1:]:
            if t["tick"] != prev + 1:
                runs.append((start, prev))
                start = t["tick"]
            prev = t["tick"]
        runs.append((start, prev))
        print(f"  gate ON runs (sim ticks): {runs}")
        for a, b in runs:
            ta, tb = ticks[a], ticks[b]
            print(
                f"    run {a}-{b}: t {a / 120.0:.3f}-{b / 120.0:.3f} s  "
                f"({b - a + 1} ticks = {(b - a + 1) / 120.0:.3f} s), "
                f"rng {ta['rng']:.1f} -> {tb['rng']:.1f} m, "
                f"naive_lead {ta['naive_deg']:.2f} -> {tb['naive_deg']:.2f} deg, "
                f"lay_miss {ta['lay_miss']:.1f} -> {tb['lay_miss']:.1f} m, "
                f"p_hit {p_hit_range(ta['rng']):.4f} -> {p_hit_range(tb['rng']):.4f}, "
                f"fsm {ta['fsm']}"
            )
        nd = [t["naive_deg"] for t in on]
        lm = [t["lay_miss"] for t in on]
        print(
            f"  over gate-ON ticks: naive_lead min {min(nd):.3f} max {max(nd):.3f} deg "
            f"(gate {GATE_DEG:.1f}); lay_miss min {min(lm):.2f} max {max(lm):.2f} m "
            f"(kill sphere {C.HIT_RADIUS_M:.0f} m)"
        )
        frac = sum(1 for x in lm if x <= C.HIT_RADIUS_M) / len(lm)
        print(
            f"  fraction of gate-ON ticks whose lay_miss <= 8 m: {frac:.3f} "
            f"({sum(1 for x in lm if x <= C.HIT_RADIUS_M)}/{len(lm)})"
        )
    # rounds actually spawned, per sim tick, from the ammo series
    fired = []
    for i in range(n - 1):
        d = ticks[i]["ammo"] - ticks[i + 1]["ammo"]
        if d > 0:
            fired.append((i, d))
    tot = sum(d for _, d in fired)
    eh = sum(d * p_hit_range(ticks[i]["rng"]) for i, d in fired)
    print(
        f"  rounds spawned: {tot:.0f} on {len(fired)} sim ticks; "
        f"E[hits] under p_hit_range = {eh:.2f} ({2 * eh:.1f} HP); ACTUAL HP taken by me = "
        f"{ticks[0]['my_hp'] - ticks[-1]['my_hp']:.1f}"
    )
    if fired:
        rs = [ticks[i]["rng"] for i, _ in fired]
        print(f"  rng span of spawned rounds: {min(rs):.1f} .. {max(rs):.1f} m")
        # split the burst by p_hit band
        for lo, hi, lab in (
            (0, 300, "<300 m"),
            (300, 500, "300-500"),
            (500, 700, "500-700"),
            (700, 1200, "700-1200"),
        ):
            sel = [(i, d) for i, d in fired if lo <= ticks[i]["rng"] < hi]
            if sel:
                s_eh = sum(d * p_hit_range(ticks[i]["rng"]) for i, d in sel)
                print(f"    {lab:9s}: {sum(d for _, d in sel):5.0f} rounds  E[hits] {s_eh:6.2f}")
    # what blocks the gate while in range
    blocked = [t for t in inr if t["trig"] < 0.5]
    if blocked:
        nd = [t["naive_deg"] for t in blocked]
        print(
            f"  in_range but gate OFF: {len(blocked)} ticks; naive_lead min {min(nd):.3f} "
            f"max {max(nd):.3f} deg; any below {GATE_DEG:.1f}? "
            f"{sum(1 for x in nd if x < GATE_DEG)}"
        )
    # where the gate first goes off after the main burst, and the p_hit there
    if on:
        last_main = None
        for a, b in runs:
            if b - a > 20:
                last_main = b
        if last_main is not None:
            t0 = ticks[last_main]
            t1 = ticks[last_main + 1]
            print(
                f"  gate drops after tick {last_main} (t {last_main / 120.0:.3f} s): "
                f"rng {t0['rng']:.1f}->{t1['rng']:.1f} m, "
                f"naive_lead {t0['naive_deg']:.2f}->{t1['naive_deg']:.2f} deg, "
                f"p_hit {p_hit_range(t0['rng']):.4f}->{p_hit_range(t1['rng']):.4f}"
            )


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    g = np.asarray(z["genome"], float)
    for seed in (62, 28):
        r, ticks = fly(seed, 576, g)
        report(seed, r, ticks)


if __name__ == "__main__":
    main()
