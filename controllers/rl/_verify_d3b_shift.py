"""VERIFY the D3b 'knife-edge clock' claim: reproduce part `fine`, then instrument WHERE the
surviving statistics (rngMin, rounds) actually come from on the +-2 arms.

Part A re-runs `_diag_d3b_skeptic.fly` verbatim (same module, same cell, same seeds) for
shifts -2..+2 and prints the same row, so the claimed numbers can be checked digit for digit.

Part B re-flies the same five arms with a RECORDING controller (the recorder IS the controller
passed to `M.roll_once`, so every quantity is sampled inside the search's own rollout loop, after
`relative()` and before `env.step()`; `fire` is the same `M._trigger(rel)` call `roll_once` writes
into a[4]). It then asks the question the pooled row cannot answer: on the arms that deal 0.0 HP,
WHEN does the closest approach happen and WHEN are the rounds fired -- on the same merge pass the
kill lived on (t <= 8 s, segments 0-1), or later, in segments 2-5 that the kill never flew?

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _verify_d3b_shift.py'
"""

from __future__ import annotations

import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_d3b_skeptic as S  # noqa: E402
import _diag_true_mpc as M  # noqa: E402

OUT = "/workspace/controllers/rl/runs/d3b_shift/verify_shift.txt"
SHIFTS = (-2, -1, 0, 1, 2)
KILL_END_STEP = 173  # the flown length of the saved kill (seed 62)
PASS_T_S = 8.0  # end of genome segment 1 = the whole flown extent of the kill

_LINES: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LINES.append(s)


class Rec:
    """Recording controller. Identical commands to `S.fly`'s `ctl`, plus a per-step dump."""

    def __init__(self, g, shift, n_steps=576, segs=6):
        self.gm = M.Genome(segs, 3, "bank", True)
        self.g, self.shift, self.n = g, int(shift), int(n_steps)
        self.rows: list[dict] = []

    def __call__(self, step, me, bd, rel):
        idx = min(self.n - 1, max(0, step + self.shift))
        a = self.gm.act(self.g, idx, self.n, me)
        self.rows.append(
            dict(
                step=step,
                t_s=step * M.CONTROL_DT_S,
                seg=self.gm.seg_of(idx, self.n),
                cmd_bank_deg=math.degrees(
                    math.radians(M.BANK_MAX_DEG) * float(self.g[3 * self.gm.seg_of(idx, self.n)])
                ),
                fire=float(M._trigger(rel)),
                rng_m=float(rel.rng),
                lay_miss_m=float(rel.lay_miss_m),
                aspect_deg=float(rel.aspect) * M.DEG,
                ata_deg=float(rel.ata) * M.DEG,
                closure_ms=float(rel.closure),
                phi_deg=float(me.phi) * M.DEG,
                nz_g=float(me.nz),
                alt_m=float(me.alt),
                opp_health=float(bd.health),
                ammo=float(me.ammo),
            )
        )
        return a


def part_a(g, seeds):
    log("A. REPRODUCTION of _diag_d3b_skeptic.py part `fine` (lines 112-118), verbatim `fly`/`row`")
    log("   (same module, same M._trigger, same M.score-relevant fields; only `shift` varies)")
    out = {}
    for k in SHIFTS:
        rs = [S.fly(g, S.cell(1500.0), s, shift=k) for s in seeds]
        out[k] = rs
        S.row(f"shift {k:+3d} steps ({k * M.CONTROL_DT_S:+.3f} s)", rs)
    log("")
    log("   PER-SEED detail (the pooled row above cannot show these)")
    log(
        f"     {'shift':>5} {'seed':>5} {'event':>10} {'kill':>4} {'hp':>6} {'tkn':>4} "
        f"{'tLeth':>6} {'t600':>6} {'rngMn':>6} {'rnds':>5} {'dalt':>7} {'steps':>5}"
    )
    for k in SHIFTS:
        for s, r in zip(seeds, out[k]):
            log(
                f"     {k:>+5d} {s:>5} {str(r.event):>10} {r.kill:>4} {r.hp_dealt:>6.1f} "
                f"{r.hp_taken:>4.0f} {r.tleth_s:>6.3f} {r.t600_s:>6.2f} {r.rng_min:>6.0f} "
                f"{r.rounds:>5d} {r.dalt_m:>+7.0f} {r.steps:>5d}"
            )
    return out


def part_b(g, seeds):
    log("")
    log(
        "B. WHERE DO THE SURVIVING STATISTICS COME FROM?  (recording controller, same rollout loop)"
    )
    log(
        f"   PASS window = t <= {PASS_T_S:.1f} s (genome segments 0-1 = the ENTIRE flown extent of "
        f"the saved kill, {KILL_END_STEP} steps = 7.208 s)"
    )
    log("")
    log(
        f"     {'shift':>5} {'seed':>5} {'steps':>5} | {'rngMin':>7} {'@t_s':>6} {'@seg':>4} "
        f"{'layMiss':>7} | {'passCPA':>7} {'@t_s':>6} {'passLay':>7} | "
        f"{'rnds':>5} {'in<=8s':>6} {'after':>6} | {'minLay':>6} {'@rng':>6} {'hp':>5} {'dalt':>7}"
    )
    summary = {}
    for k in SHIFTS:
        agg = []
        for s in seeds:
            env, base = S._env_for(S.cell(1500.0), s)
            rec = Rec(g, k)
            r = M.roll_once(env, base, rec, S.N_STEPS, keep=False)
            d = rec.rows
            assert len(d) == r.steps, (len(d), r.steps)
            rng = np.array([x["rng_m"] for x in d])
            t = np.array([x["t_s"] for x in d])
            fire = np.array([x["fire"] for x in d]) > 0.0
            lay = np.array([x["lay_miss_m"] for x in d])
            i = int(np.argmin(rng))
            m = t <= PASS_T_S
            j = int(np.argmin(np.where(m, rng, np.inf)))
            n_fire_in = int(fire[m].sum())
            n_fire_after = int(fire[~m].sum())
            # min lay_miss achieved INSIDE lethal range (what a kill actually needs)
            inr = rng <= M.R_LETHAL_M
            if inr.any():
                jl = int(np.argmin(np.where(inr, lay, np.inf)))
                minlay, minlay_rng = lay[jl], rng[jl]
            else:
                minlay, minlay_rng = float("nan"), float("nan")
            log(
                f"     {k:>+5d} {s:>5} {r.steps:>5} | {rng[i]:>7.1f} {t[i]:>6.3f} {d[i]['seg']:>4} "
                f"{lay[i]:>7.1f} | {rng[j]:>7.1f} {t[j]:>6.3f} {lay[j]:>7.1f} | "
                f"{r.rounds:>5d} {n_fire_in:>6d} {n_fire_after:>6d} | "
                f"{minlay:>6.1f} {minlay_rng:>6.0f} {r.hp_dealt:>5.1f} {r.dalt_m:>+7.0f}"
            )
            agg.append(
                dict(
                    rngmin=rng[i],
                    t_rngmin=t[i],
                    passcpa=rng[j],
                    t_passcpa=t[j],
                    fire_in=n_fire_in,
                    fire_after=n_fire_after,
                    minlay=minlay,
                    hp=r.hp_dealt,
                    dalt=r.dalt_m,
                    steps=r.steps,
                )
            )
        summary[k] = agg
        log(
            f"     {k:>+5d} {'MEAN':>5} {np.mean([a['steps'] for a in agg]):>5.0f} | "
            f"{np.mean([a['rngmin'] for a in agg]):>7.1f} "
            f"{np.mean([a['t_rngmin'] for a in agg]):>6.3f} {'-':>4} {'-':>7} | "
            f"{np.mean([a['passcpa'] for a in agg]):>7.1f} "
            f"{np.mean([a['t_passcpa'] for a in agg]):>6.3f} {'-':>7} | "
            f"{'-':>5} {np.mean([a['fire_in'] for a in agg]):>6.1f} "
            f"{np.mean([a['fire_after'] for a in agg]):>6.1f} | "
            f"{np.nanmean([a['minlay'] for a in agg]):>6.1f} {'-':>6} "
            f"{np.mean([a['hp'] for a in agg]):>5.1f} {np.mean([a['dalt'] for a in agg]):>+7.0f}"
        )
        log("")
    return summary


def part_c(g, seeds):
    """How many COMMANDED steps actually differ between shift arms, on the flown prefix?"""
    log("C. WHAT DOES THE SHIFT ACTUALLY CHANGE IN THE COMMAND STREAM?")
    gm = M.Genome(6, 3, "bank", True)
    for k in SHIFTS:
        segs = [gm.seg_of(min(575, max(0, st + k)), 576) for st in range(576)]
        base = [gm.seg_of(st, 576) for st in range(576)]
        diff = [st for st in range(576) if segs[st] != base[st]]
        diff_pre = [st for st in diff if st < KILL_END_STEP]
        b01 = next(st for st in range(576) if segs[st] == 1)
        log(
            f"     shift {k:+3d}: seg0->seg1 boundary at step {b01:>3d} "
            f"(t {b01 * M.CONTROL_DT_S:.3f} s); commanded-segment differs from shift 0 on "
            f"{len(diff):>3d}/576 steps, {len(diff_pre):>2d} of them inside the kill's "
            f"{KILL_END_STEP}-step flown window {diff_pre}"
        )
    log("")


def main() -> None:
    d = np.load(S.NPZ, allow_pickle=True)
    g = np.asarray(d["genome"], float)
    seeds = S.SEEDS
    log(f"genome {np.round(g, 6).tolist()}")
    log(f"seeds  {seeds}   N_STEPS {S.N_STEPS}   CONTROL_DT_S {M.CONTROL_DT_S:.6f} s")
    log("")
    part_c(g, seeds)
    part_a(g, seeds)
    part_b(g, seeds)
    with open(OUT, "w") as f:
        f.write("\n".join(_LINES) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
