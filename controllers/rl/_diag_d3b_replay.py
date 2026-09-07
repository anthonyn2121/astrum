"""D3b REPLAY — re-fly the saved winning genome and dump the per-step time series.

`runs/d3b/d3b_best.npz` carries the genome, the flown agent-space actions and the 41-float
observations, but NO per-step state array, so nothing downstream can say what the jet actually DID.
The sim is deterministic (`_diag_true_mpc.precheck` measured replay and cross-seed max|div| = 0.000e+00
at 576 control steps), so a re-flight is EXACT, not approximate, and the missing attitude / position /
relative-geometry series can be recovered by flying the genome again and recording it.

METHOD, and why it cannot drift from the search:
  * Everything is imported from `_diag_true_mpc` (`Genome`, `roll_once`, `build_env`, `_trigger`,
    `CELLS`, `score`) — nothing is reimplemented, so the flown trajectory is the searched one by
    construction rather than by inspection.
  * The recorder is the CONTROLLER. `roll_once` calls `controller(step, me, bd, rel)` at exactly the
    point where the row must be sampled — after `relative()`, before `env.step()` — so the dumped row
    is the state the flown action was computed from, and the trace is 1:1 with `r.trace`
    (`B.make_sample`). The controller returns `Genome.act(...)` verbatim; `roll_once` then overwrites
    `a[4]` with `_trigger(rel)`, and the `fire` column is that same `_trigger(rel)` call, so the CSV
    action row is byte-identical to the flown one.
  * Which seed produced the npz is MEASURED, not assumed: every search seed is re-flown and its
    actions compared elementwise against the npz array.

The last CSV row is the state the KILLING burst was fired from, not the post-kill state: `roll_once`
records the row, then steps, then breaks on `term`. The post-terminal state is printed to stdout
(`POST-TERMINAL`) rather than appended, so the CSV stays 1:1 with the detector trace.

Column groups: `cmd_*`/`ail`/`elev`/`thr_flown`/`fire` = commanded; bare names = own ship; `opp_*` =
bandit; `rng_m`..`lethal` = relative. The `smp_*` block is the raw `B.make_sample()` field set in its
native units (radians, booleans) so a detector audit can rebuild the exact trace the detectors read.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d3b_replay.py'
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B
from astrum_dogfight.geometry import specific_energy

DEG = 180.0 / math.pi
SEARCH_SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]

# The npz-writing rule in `_diag_true_mpc.main`: `max(rows, key=lambda r: (r.kill, r.tleth_s))` over
# the per-seed rows of the FIRST cell in `results`. `max` returns the FIRST maximal element, so ties
# are broken by seed ORDER — which is the whole 172-vs-173 / 0.458-vs-0.500 story.
NPZ_RULE = "max(rows, key=(kill, tleth_s)) over seeds in order; ties -> first seed"

COLUMNS = [
    # --- index -----------------------------------------------------------------------------
    "step",
    "t_s",
    # --- commanded (the genome, decoded) ---------------------------------------------------
    "seg_idx",
    "cmd_bank_deg",
    "cmd_load",
    "cmd_thr",
    "ail",
    "elev",
    "rud",
    "thr_flown",
    "fire",
    # --- own ship --------------------------------------------------------------------------
    "phi_deg",
    "theta_deg",
    "psi_deg",
    "alpha_deg",
    "beta_deg",
    "nz_g",
    "alt_m",
    "alt_enu_m",
    "vt_ms",
    "gamma_deg",
    "roll_rate_dps",
    "pitch_rate_dps",
    "yaw_rate_dps",
    "es_m",
    "health",
    "ammo",
    "pos_e_m",
    "pos_n_m",
    "pos_u_m",
    "vel_e_ms",
    "vel_n_ms",
    "vel_u_ms",
    # --- bandit ----------------------------------------------------------------------------
    "opp_phi_deg",
    "opp_theta_deg",
    "opp_psi_deg",
    "opp_alpha_deg",
    "opp_nz_g",
    "opp_alt_m",
    "opp_vt_ms",
    "opp_es_m",
    "opp_health",
    "opp_ammo",
    "opp_pos_e_m",
    "opp_pos_n_m",
    "opp_pos_u_m",
    # --- relative --------------------------------------------------------------------------
    "rng_m",
    "aspect_deg",
    "ata_deg",
    "ata_az_deg",
    "ata_el_deg",
    "hca_deg",
    "closure_ms",
    "dalt_rel_m",
    "lay_miss_m",
    "lay_tof_s",
    "lead_error_deg",
    "lead_az_deg",
    "lead_el_deg",
    "alpha_req_deg",
    "chi_deg",
    "chi_valid",
    "los_rate_az_dps",
    "los_rate_el_dps",
    "los_rate_dps",
    "cpa_miss_m",
    "es_rel_m",
    "in_range",
    "range_factor_phit",
    "gun_quality",
    "lethal",
    # --- the raw B.make_sample() field set, native units, for the detector audit ------------
    "smp_t",
    "smp_chi_rad",
    "smp_chi_valid",
    "smp_gamma_rad",
    "smp_nz",
    "smp_phi_rad",
    "smp_alt",
    "smp_ail",
    "smp_rng",
    "smp_ata",
    "smp_ata_az",
    "smp_aspect",
    "smp_closure",
    "smp_cpa_miss",
    "smp_es_rel",
    "smp_lay_miss_m",
    "smp_lethal",
]


def decode_genome(g: np.ndarray, gm: M.Genome) -> list[dict]:
    """The 6-row genome decode, using `Genome.act`'s own arithmetic (not a paraphrase of it)."""
    rows = []
    for s in range(gm.segments):
        p = g[s * gm.channels : (s + 1) * gm.channels]
        rows.append(
            dict(
                seg=s,
                bank_deg=M.BANK_MAX_DEG * float(np.clip(p[0], -1.0, 1.0)),
                load=0.5 * (float(np.clip(p[1], -1.0, 1.0)) + 1.0),
                thr=0.5 * (float(np.clip(p[2], -1.0, 1.0)) + 1.0),
                rud=float(np.clip(p[3], -1.0, 1.0)) if gm.channels == 4 else 0.0,
            )
        )
    return rows


def make_recorder(g: np.ndarray, gm: M.Genome, n_steps: int, rows: list):
    """A controller that is ALSO the row sink. Returns `Genome.act` verbatim."""
    dec = decode_genome(g, gm)

    def ctl(step, me, bd, rel):
        a = gm.act(g, step, n_steps, me)
        seg = gm.seg_of(step, n_steps)
        d = dec[seg]
        # `fire` is the SAME call `roll_once` uses to overwrite a[4]; the sample is the SAME
        # constructor the detector trace is built from.
        fire = M._trigger(rel)
        smp = B.make_sample(step * M.CONTROL_DT_S, me, bd, np.append(a[:4], fire), rel)
        rows.append(
            {
                "step": step,
                "t_s": step * M.CONTROL_DT_S,
                "seg_idx": seg,
                "cmd_bank_deg": d["bank_deg"],
                "cmd_load": d["load"],
                "cmd_thr": d["thr"],
                "ail": float(a[0]),
                "elev": float(a[1]),
                "rud": float(a[2]),
                "thr_flown": float(a[3]),
                "fire": fire,
                "phi_deg": me.phi * DEG,
                "theta_deg": me.theta * DEG,
                "psi_deg": me.psi * DEG,
                "alpha_deg": me.alpha * DEG,
                "beta_deg": me.beta * DEG,
                "nz_g": me.nz,
                "alt_m": me.alt,
                "alt_enu_m": float(me.pos[2]),
                "vt_ms": me.airspeed,
                "gamma_deg": smp.gamma * DEG,
                "roll_rate_dps": me.p * DEG,
                "pitch_rate_dps": me.q * DEG,
                "yaw_rate_dps": me.r * DEG,
                "es_m": specific_energy(me),
                "health": me.health,
                "ammo": int(me.ammo),
                "pos_e_m": float(me.pos[0]),
                "pos_n_m": float(me.pos[1]),
                "pos_u_m": float(me.pos[2]),
                "vel_e_ms": float(me.vel[0]),
                "vel_n_ms": float(me.vel[1]),
                "vel_u_ms": float(me.vel[2]),
                "opp_phi_deg": bd.phi * DEG,
                "opp_theta_deg": bd.theta * DEG,
                "opp_psi_deg": bd.psi * DEG,
                "opp_alpha_deg": bd.alpha * DEG,
                "opp_nz_g": bd.nz,
                "opp_alt_m": bd.alt,
                "opp_vt_ms": bd.airspeed,
                "opp_es_m": specific_energy(bd),
                "opp_health": bd.health,
                "opp_ammo": int(bd.ammo),
                "opp_pos_e_m": float(bd.pos[0]),
                "opp_pos_n_m": float(bd.pos[1]),
                "opp_pos_u_m": float(bd.pos[2]),
                "rng_m": rel.rng,
                "aspect_deg": rel.aspect * DEG,
                "ata_deg": rel.ata * DEG,
                "ata_az_deg": rel.ata_az * DEG,
                "ata_el_deg": rel.ata_el * DEG,
                "hca_deg": rel.hca * DEG,
                "closure_ms": rel.closure,
                "dalt_rel_m": rel.rel_alt,
                "lay_miss_m": rel.lay_miss_m,
                "lay_tof_s": rel.lay_tof,
                "lead_error_deg": rel.lead_error * DEG,
                "lead_az_deg": rel.lead_az * DEG,
                "lead_el_deg": rel.lead_el * DEG,
                "alpha_req_deg": rel.alpha_req * DEG,
                "chi_deg": smp.chi * DEG,
                "chi_valid": int(smp.chi_valid),
                "los_rate_az_dps": rel.los_rate_az * DEG,
                "los_rate_el_dps": rel.los_rate_el * DEG,
                "los_rate_dps": rel.los_rate * DEG,
                "cpa_miss_m": smp.cpa_miss,
                "es_rel_m": smp.es_rel,
                "in_range": int(rel.in_range),
                "range_factor_phit": rel.range_factor,
                "gun_quality": rel.gun_quality,
                "lethal": int(smp.lethal),
                "smp_t": smp.t,
                "smp_chi_rad": smp.chi,
                "smp_chi_valid": int(smp.chi_valid),
                "smp_gamma_rad": smp.gamma,
                "smp_nz": smp.nz,
                "smp_phi_rad": smp.phi,
                "smp_alt": smp.alt,
                "smp_ail": smp.ail,
                "smp_rng": smp.rng,
                "smp_ata": smp.ata,
                "smp_ata_az": smp.ata_az,
                "smp_aspect": smp.aspect,
                "smp_closure": smp.closure,
                "smp_cpa_miss": smp.cpa_miss,
                "smp_es_rel": smp.es_rel,
                "smp_lay_miss_m": smp.lay_miss_m,
                "smp_lethal": int(smp.lethal),
            }
        )
        return a

    return ctl


def fly(g, gm, cell, n_steps, seed):
    """One instrumented re-flight. Returns (Rollout, rows, terminal-state dict)."""
    env, base = M.build_env(cell, seed, n_steps * M.CONTROL_DT_S)
    rows: list[dict] = []
    r = M.roll_once(env, base, make_recorder(g, gm, n_steps, rows), n_steps, keep=True)
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    post = dict(
        alt_m=me.alt,
        vt_ms=me.airspeed,
        phi_deg=me.phi * DEG,
        theta_deg=me.theta * DEG,
        health=me.health,
        ammo=int(me.ammo),
        opp_alt_m=bd.alt,
        opp_vt_ms=bd.airspeed,
        opp_health=bd.health,
        rng_m=float(np.linalg.norm(np.asarray(bd.pos) - np.asarray(me.pos))),
    )
    env.close()
    return r, rows, post


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", default="runs/d3b/d3b_best.npz")
    ap.add_argument("--outdir", default="runs/d3b")
    ap.add_argument("--nokill-seed", type=int, default=28)
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    fh = open(args.log, "w") if args.log else None

    def log(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    z = np.load(args.npz, allow_pickle=False)
    cell_name = str(z["cell"])
    g = np.asarray(z["genome"], float)
    a_npz = np.asarray(z["actions"], np.float32)
    o_npz = np.asarray(z["obs"], np.float32)
    npz_kill = int(z["kill"])
    npz_tleth = float(z["tleth_s"])
    npz_hp = float(z["hp_dealt"])

    cell = M.CELLS[cell_name]
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    n_steps = 576  # 24.0 s / CONTROL_DT_S — the searched horizon, re-flown in full

    log("#" * 100)
    log("#  D3b REPLAY — instrumented re-flight of the saved winning genome")
    log("#" * 100)
    log("")
    log(f"   npz .............. {os.path.abspath(args.npz)}")
    log(
        f"   cell ............. {cell_name}  lateral {cell['lateral_m']:.0f} m  "
        f"sep {cell['separation_m']:.0f} m  aspect {cell['aspect_deg']:.2f} deg  "
        f"perch {cell['me_dalt_ft']:.0f} ft  alt {cell['alt_ft']:.0f} ft"
    )
    log(
        f"   npz outcome ...... kill={npz_kill}  tleth_s={npz_tleth:.3f}  hp_dealt={npz_hp:.1f}  "
        f"actions {a_npz.shape}  obs {o_npz.shape}"
    )
    log(
        f"   genome ........... dim {gm.dim} = {gm.segments} segments x {gm.channels} channels  "
        f"param={gm.param}  roll_gate={gm.roll_gate}"
    )
    log(f"   horizon .......... {n_steps} control steps x {M.CONTROL_DT_S * 1e3:.3f} ms = 24.000 s")
    log(f"   npz save rule .... {NPZ_RULE}")

    # ---- genome decode ----------------------------------------------------------------------
    dec = decode_genome(g, gm)
    seg_steps = (
        n_steps // gm.segments
    )  # `seg_of` is (step*segments)//n_steps, i.e. 96 steps/segment
    log("")
    log(
        "   GENOME DECODE (6 segments; `seg_of` = (step*6)//576, so 96 control steps = 4.000 s each)"
    )
    log(
        f"     {'seg':>3} {'steps':>11} {'seconds':>13} {'bank_cmd_deg':>13} {'load':>6} {'thr':>6}"
    )
    for d in dec:
        s0, s1 = d["seg"] * seg_steps, (d["seg"] + 1) * seg_steps - 1
        log(
            f"     {d['seg']:>3} {f'{s0}-{s1}':>11} "
            f"{f'{s0 * M.CONTROL_DT_S:.3f}-{(s1 + 1) * M.CONTROL_DT_S:.3f}':>13} "
            f"{d['bank_deg']:>+13.2f} {d['load']:>6.4f} {d['thr']:>6.4f}"
        )

    # ---- which seed produced the npz --------------------------------------------------------
    log("")
    log("   SEED IDENTIFICATION — every search seed re-flown, actions compared to the npz array")
    log(
        f"     {'seed':>5} {'ev':>10} {'kill':>4} {'hp':>6} {'tLeth':>6} {'steps':>5} "
        f"{'ovl':>5} {'max|dA| ovl':>12} {'len==npz':>8} {'max|dA| full':>13} {'max|dO| ovl':>12}"
    )
    flown = {}
    for sd in SEARCH_SEEDS:
        r, rows, post = fly(g, gm, cell, n_steps, sd)
        a = np.asarray(r.actions, np.float32)
        o = np.asarray(r.obs, np.float32)
        k = min(len(a), len(a_npz))
        d_ovl = float(np.max(np.abs(a[:k] - a_npz[:k]))) if k else float("nan")
        d_obs = float(np.max(np.abs(o[:k] - o_npz[:k]))) if k else float("nan")
        same_len = a.shape == a_npz.shape
        d_full = float(np.max(np.abs(a - a_npz))) if same_len else float("nan")
        flown[sd] = dict(r=r, rows=rows, post=post, d_ovl=d_ovl, d_full=d_full, same_len=same_len)
        log(
            f"     {sd:>5} {str(r.event):>10} {r.kill:>4} {r.hp_dealt:>6.1f} {r.tleth_s:>6.3f} "
            f"{r.steps:>5} {k:>5} {d_ovl:>12.3e} {str(same_len):>8} {d_full:>13.3e} "
            f"{d_obs:>12.3e}"
        )

    exact = [
        sd
        for sd in SEARCH_SEEDS
        if flown[sd]["same_len"]
        and flown[sd]["d_full"] == 0.0
        and flown[sd]["r"].kill == npz_kill
        and abs(flown[sd]["r"].tleth_s - npz_tleth) < 1e-9
        and abs(flown[sd]["r"].hp_dealt - npz_hp) < 1e-9
    ]
    log("")
    log(f"     seeds reproducing the npz EXACTLY (actions, kill, tLeth, hp): {exact}")
    log(
        f"     the npz row is the FIRST of those in seed order -> seed {exact[0] if exact else None}"
    )

    # ---- PER-SEED detector counts ------------------------------------------------------------
    # The prereg reports the SEARCH arm's detector counts POOLED over all 8 episodes, so a nonzero
    # D3 or a chiMx there cannot be attributed to the kill trajectory. Per-seed splits it out.
    log("")
    log("   PER-SEED DETECTOR (the prereg's SEARCH row is the SUM of these 8; splitting it out is")
    log("   the only way to say whether the KILL trajectory itself carries a BFM event)")
    log(
        f"     {'seed':>5} {'kill':>4} {'steps':>5} {'D1':>3} {'D2':>3} {'D2c':>4} {'D3':>3} "
        f"{'lethTk':>6} {'chiMx':>6} {'nzMx':>5} {'dalt':>7}"
    )
    tot = dict(D1=0, D2=0, D2c=0, D3=0, leth=0, chimx=0.0)
    for sd in SEARCH_SEEDS:
        r = flown[sd]["r"]
        det = B.detect_all(r.trace)
        chimx = max((abs(s.chi) for s in r.trace if s.chi_valid), default=0.0) * DEG
        nzmx = max((s.nz for s in r.trace), default=0.0)
        tot["D1"] += len(det["displacement_rolls"])
        tot["D2"] += len(det["yoyos"])
        tot["D2c"] += len(det["yoyos_clean"])
        tot["D3"] += len(det["turning_room"])
        tot["leth"] += det["lethal_ticks"]
        tot["chimx"] = max(tot["chimx"], chimx)
        log(
            f"     {sd:>5} {r.kill:>4} {r.steps:>5} {len(det['displacement_rolls']):>3} "
            f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4} {len(det['turning_room']):>3} "
            f"{det['lethal_ticks']:>6} {chimx:>6.1f} {nzmx:>5.1f} {r.dalt_m:>+7.0f}"
        )
    log(
        f"     {'POOL':>5} {sum(flown[s]['r'].kill for s in SEARCH_SEEDS):>4} {'-':>5} "
        f"{tot['D1']:>3} {tot['D2']:>3} {tot['D2c']:>4} {tot['D3']:>3} {tot['leth']:>6} "
        f"{tot['chimx']:>6.1f}   (compare the prereg SEARCH row for this cell)"
    )

    # ---- the two dumps ----------------------------------------------------------------------
    kill_seed = exact[0] if exact else None
    if kill_seed is None:
        log("     !! NO seed reproduced the npz exactly — dumping the best-matching seed instead")
        kill_seed = min(SEARCH_SEEDS, key=lambda s: (not flown[s]["same_len"], flown[s]["d_ovl"]))
    dumps = [("kill", kill_seed), ("nokill", args.nokill_seed)]
    log("")
    log("   DUMPS")
    paths = []
    for tag, sd in dumps:
        f = flown[sd]
        path = os.path.join(args.outdir, f"replay_{tag}_seed{sd}.csv")
        write_csv(path, f["rows"])
        paths.append(os.path.abspath(path))
        r = f["r"]
        log(
            f"     {os.path.abspath(path)}   rows {len(f['rows'])}  cols {len(COLUMNS)}  "
            f"event={r.event}  kill={r.kill}  hp_dealt={r.hp_dealt:.1f}  tLeth={r.tleth_s:.3f} s  "
            f"rounds={r.rounds}  dalt={r.dalt_m:+.0f} m"
        )
        log(f"       POST-TERMINAL (state AFTER the last step; NOT a CSV row): {f['post']}")
        assert len(f["rows"]) == len(r.trace) == r.steps, "row/trace/step count mismatch"

    # ---- which segments actually flew -------------------------------------------------------
    log("")
    log("   SEGMENTS EXECUTED (a segment 'never executed' iff its first step is past termination)")
    for tag, sd in dumps:
        r = flown[sd]["r"]
        rows = flown[sd]["rows"]
        log(f"     seed {sd} ({tag}): {r.steps} steps = {r.steps * M.CONTROL_DT_S:.3f} s flown")
        for d in dec:
            s0 = d["seg"] * seg_steps
            got = [x for x in rows if x["seg_idx"] == d["seg"]]
            if not got:
                log(
                    f"       seg {d['seg']}  steps {s0}-{s0 + seg_steps - 1}  "
                    f"NEVER EXECUTED (episode ended at step {r.steps - 1})"
                )
            else:
                frac = len(got) / seg_steps
                log(
                    f"       seg {d['seg']}  steps {got[0]['step']}-{got[-1]['step']}  "
                    f"t {got[0]['t_s']:.3f}-{got[-1]['t_s'] + M.CONTROL_DT_S:.3f} s  "
                    f"{len(got)}/{seg_steps} steps ({frac * 100:.0f}%)"
                    f"{'  <- TRUNCATED' if frac < 1.0 else ''}"
                )

    # ---- self-consistency + detector, on the SAME rows --------------------------------------
    log("")
    log("   SELF-CONSISTENCY / DETECTOR (detectors read the trace the CSV `smp_*` block mirrors)")
    for tag, sd in dumps:
        f = flown[sd]
        r = f["r"]
        det = B.detect_all(r.trace)
        chimx = max((abs(s.chi) for s in r.trace if s.chi_valid), default=0.0) * DEG
        nzmx = max((s.nz for s in r.trace), default=0.0)
        s_scalar, comp = M.score(r)
        fired = sum(1 for x in f["rows"] if x["fire"] > 0.0)
        leth = sum(1 for x in f["rows"] if x["lethal"])
        log(
            f"     seed {sd} ({tag}): D1={len(det['displacement_rolls'])} "
            f"D2={len(det['yoyos'])} D2c={len(det['yoyos_clean'])} "
            f"D3={len(det['turning_room'])} lethalTicks={det['lethal_ticks']} "
            f"chiMx={chimx:.1f} deg nzMx={nzmx:.1f} g | trigger-on steps {fired} "
            f"lethal rows {leth} | score {s_scalar:+.3f} {comp}"
        )
        log(
            f"       rng min {min(x['rng_m'] for x in f['rows']):.0f} m   "
            f"alt {f['rows'][0]['alt_m']:.0f} -> {f['rows'][-1]['alt_m']:.0f} m   "
            f"vt {f['rows'][0]['vt_ms']:.0f} -> {f['rows'][-1]['vt_ms']:.0f} m/s   "
            f"opp_health {f['rows'][-1]['opp_health']:.0f} at last row"
        )
    # ---- a readable digest of the kill trajectory (every 0.5 s, plus every trigger-on row) ----
    log("")
    log(f"   KILL-TRAJECTORY DIGEST — seed {kill_seed}, every 12th step (0.5 s) + every fire row")
    log(
        f"     {'step':>4} {'t_s':>6} {'sg':>2} {'cmdphi':>7} {'phi':>7} {'nz':>5} {'alpha':>6} "
        f"{'gam':>6} {'alt':>6} {'vt':>4} {'rng':>6} {'asp':>5} {'ata':>5} {'chi':>6} {'cv':>2} "
        f"{'miss':>7} {'areq':>5} {'clo':>5} {'F':>2} {'ophp':>4}"
    )
    for x in flown[kill_seed]["rows"]:
        if x["step"] % 12 and x["fire"] <= 0.0:
            continue
        log(
            f"     {x['step']:>4} {x['t_s']:>6.3f} {x['seg_idx']:>2} {x['cmd_bank_deg']:>+7.1f} "
            f"{x['phi_deg']:>+7.1f} {x['nz_g']:>5.2f} {x['alpha_deg']:>+6.2f} "
            f"{x['gamma_deg']:>+6.1f} {x['alt_m']:>6.0f} {x['vt_ms']:>4.0f} {x['rng_m']:>6.0f} "
            f"{x['aspect_deg']:>5.1f} {x['ata_deg']:>5.1f} {x['chi_deg']:>+6.1f} "
            f"{x['chi_valid']:>2} {x['lay_miss_m']:>7.1f} {x['alpha_req_deg']:>5.1f} "
            f"{x['closure_ms']:>+5.0f} {int(x['fire'] > 0):>2} {x['opp_health']:>4.0f}"
        )

    ok = (
        flown[kill_seed]["r"].kill == 1
        and abs(flown[kill_seed]["r"].hp_dealt - 100.0) < 1e-9
        and flown[kill_seed]["d_full"] == 0.0
    )
    log("")
    log(
        f"   VERDICT: reproduction {'CONFIRMED' if ok else 'FAILED'} on seed {kill_seed} "
        f"(kill=1, hp_dealt=100.0, max|dActions| = {flown[kill_seed]['d_full']:.3e})"
    )
    log(f"   wrote: {paths}")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
