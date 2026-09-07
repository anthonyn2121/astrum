"""INDEPENDENT verification of the "D1 is comfortably expressible" claim.

Recomputes, from the LITERAL 18 numbers quoted in the claim, every number the claim cites, plus
the controls the claim does not run:

  V1  the vector is inside the searched box [-1,1]^18 (train loop clips: _diag_true_mpc.py:723)
  V2  fly on all 8 search seeds; per-seed outcome + detector counts + rounds fired
  V3  cross-seed IDENTITY: is "8 seeds" 8 measurements or 1 trajectory replicated 8 times?
  V4  the D1 event, clause by clause, with a per-step table over the hold
  V5  WHERE the chi excursion comes from: decompose chi = phi_lift - phi_los and report how much
      of the excursion is own-ship ROLL vs the LOS plane sweeping out from under a constant bank
  V6  the (d) exit sentinel: is the bandit dead AHEAD or dead ASTERN at the exit index?
  V7  validity-gated (d): does the event survive clauses (d)+(e)+(f) if the exit must be a VALID chi?
  V8  price the hold and the episode in altitude, energy and airspeed
  V9  CONTROL: constant-bank max-load genomes (a single 24 s pull, NO roll input at all).
      If those also score D1 >= 1, then D1 = 1 does not require a displacement roll.
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

OUT = "/workspace/controllers/rl/runs/d3b"
CELL = M.CELLS["merge1500"]
DT = M.CONTROL_DT_S
N_STEPS = int(round(24.0 / DT))
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
DEG = 180.0 / math.pi

# --- the vector EXACTLY as the claim writes it (d1_exists_report.txt:15) --------------------------
G_CLAIM = np.array(
    [
        1.0,
        1.0,
        0.2,
        0.582792,
        1.0,
        0.2,
        1.0,
        1.0,
        0.2,
        0.872769,
        1.0,
        0.2,
        -0.007301,
        1.0,
        0.2,
        0.312102,
        1.0,
        0.2,
    ],
    float,
)

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _chi_parts(me, bd):
    """`out_of_plane_angle` re-implemented to expose the TWO terms it differences."""
    v = np.asarray(me.vel, float)
    n = float(np.linalg.norm(v))
    v_hat = v / n if n > 1e-9 else np.zeros(3)
    d = np.asarray(bd.pos, float) - np.asarray(me.pos, float)
    nd = float(np.linalg.norm(d))
    los = d / nd if nd > 1e-9 else np.zeros(3)
    up = np.array([0.0, 0.0, 1.0])
    u = up - float(np.dot(up, v_hat)) * v_hat
    nu = float(np.linalg.norm(u))
    u_ref = u / nu if nu > 1e-9 else np.zeros(3)
    r_ref = np.cross(v_hat, u_ref)
    los_perp = los - float(np.dot(los, v_hat)) * v_hat
    lift = B.lift_vector_enu(me.phi, me.theta, me.psi)
    lift_perp = lift - float(np.dot(lift, v_hat)) * v_hat
    phi_lift = math.atan2(float(np.dot(lift_perp, r_ref)), float(np.dot(lift_perp, u_ref)))
    phi_los = math.atan2(float(np.dot(los_perp, r_ref)), float(np.dot(los_perp, u_ref)))
    return phi_lift, phi_los, float(np.linalg.norm(los_perp))


def fly(g: np.ndarray, seed: int):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(CELL, seed, N_STEPS * DT)
    rows: list[dict] = []

    def ctl(step, me, bd, rel):
        a = gm.act(g, step, N_STEPS, me)
        pl, plos, lpn = _chi_parts(me, bd)
        p = g[gm.seg_of(step, N_STEPS) * 3 : (gm.seg_of(step, N_STEPS) + 1) * 3]
        rows.append(
            dict(
                step=step,
                t=step * DT,
                seg=gm.seg_of(step, N_STEPS),
                cmd_bank=M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)),
                cmd_load=0.5 * (float(np.clip(p[1], -1, 1)) + 1.0),
                cmd_thr=0.5 * (float(np.clip(p[2], -1, 1)) + 1.0),
                ail=float(a[0]),
                elev=float(a[1]),
                phi=float(me.phi) * DEG,
                theta=float(me.theta) * DEG,
                psi=float(me.psi) * DEG,
                alpha=float(getattr(me, "alpha", float("nan"))) * DEG,
                nz=float(me.nz),
                vt=float(me.airspeed),
                alt_state=float(me.alt),
                alt_enu=float(me.pos[2]),
                p_dps=float(me.p) * DEG,
                phi_lift=pl * DEG,
                phi_los=plos * DEG,
                los_perp=lpn,
                ammo=float(me.ammo),
                health=float(me.health),
                opp_health=float(bd.health),
            )
        )
        return a

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    rows = rows[: r.steps]
    for k, row in enumerate(rows):
        s = r.trace[k]
        row.update(
            chi=s.chi * DEG,
            chi_valid=bool(s.chi_valid),
            gamma=s.gamma * DEG,
            rng=s.rng,
            ata=s.ata * DEG,
            ata_az=s.ata_az * DEG,
            aspect=s.aspect * DEG,
            closure=s.closure,
            cpa=s.cpa_miss,
            es_rel=s.es_rel,
            lay=s.lay_miss_m,
            alt_smp=s.alt,
        )
    return r, rows


def counts(r):
    d = B.detect_all(r.trace)
    return (
        len(d["displacement_rolls"]),
        len(d["yoyos"]),
        len(d["yoyos_clean"]),
        len(d["turning_room"]),
        d,
    )


def main() -> None:
    log("#" * 100)
    log("#  INDEPENDENT VERIFICATION of 'D1 comfortably expressible' (d1_exists_report.txt)")
    log("#" * 100)

    # ---------------- V1 in-box + decode ----------------
    log("\nV1  IS THE CLAIMED VECTOR INSIDE THE SEARCHED BOX?")
    log(
        f"    train loop clips candidates to [-1,1] (_diag_true_mpc.py:723). max|g| = {np.abs(G_CLAIM).max():.6f}"
    )
    log(f"    inside box: {bool(np.all(np.abs(G_CLAIM) <= 1.0))}   dim = {G_CLAIM.size}")
    log("    decode (bank_deg, load, thr) per segment, steps and t:")
    for s in range(6):
        p = G_CLAIM[s * 3 : (s + 1) * 3]
        lo, hi = s * 96, s * 96 + 95
        log(
            f"      seg {s}  steps {lo:3d}-{hi:3d}  t {lo * DT:6.3f}-{(hi + 1) * DT:6.3f}  "
            f"bank {M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)):+8.2f}  "
            f"load {0.5 * (float(np.clip(p[1], -1, 1)) + 1.0):.4f}  "
            f"thr {0.5 * (float(np.clip(p[2], -1, 1)) + 1.0):.4f}"
        )

    # ---------------- V2/V3 all seeds ----------------
    log("\nV2  FLOWN ON ALL 8 SEARCH SEEDS (my own rollout, real Genome.act)")
    log(
        f"    {'seed':>5} {'steps':>6} {'event':>10} {'kill':>5} {'hp':>6} {'tLeth':>7} {'rngMn':>7} "
        f"{'dalt':>7} {'rnds':>5} {'D1':>3} {'D2':>3} {'D2c':>4} {'D3':>3} {'score':>11}"
    )
    keep = {}
    pooled = 0
    for sd in SEEDS:
        r, rows = fly(G_CLAIM, sd)
        d1, d2, d2c, d3, det = counts(r)
        sc, comp = M.score(r)
        pooled += d1
        log(
            f"    {sd:>5} {r.steps:>6} {str(r.event):>10} {r.kill:>5} {r.hp_dealt:>6.1f} "
            f"{r.tleth_s:>7.3f} {r.rng_min:>7.0f} {r.dalt_m:>+7.0f} {r.rounds:>5} "
            f"{d1:>3} {d2:>3} {d2c:>4} {d3:>3} {sc:>11.3f}"
        )
        keep[sd] = (r, rows, det, sc, comp)
    log(f"    POOLED D1 over 8 seeds = {pooled}")

    log("\nV3  ARE THE 8 SEEDS 8 MEASUREMENTS, OR ONE TRAJECTORY REPLICATED?")
    a62 = np.asarray(keep[62][0].actions, float)
    s62 = np.asarray(keep[62][0].states, float)
    for sd in SEEDS:
        a = np.asarray(keep[sd][0].actions, float)
        st = np.asarray(keep[sd][0].states, float)
        n = min(len(a), len(a62))
        log(
            f"    seed {sd:>3}: len {len(a):>3}  max|dActions vs 62| {np.abs(a[:n] - a62[:n]).max():.3e}"
            f"   max|dStates vs 62| {np.abs(st[:n] - s62[:n]).max():.3e}"
            f"   rounds {keep[sd][0].rounds}"
        )

    # ---------------- V4 clause walk on seed 62 ----------------
    r62, rows62, det62, sc62, _ = keep[62]
    tr = r62.trace
    log("\nV4  D1 CLAUSE WALK on seed 62 (detector source: bfm_detect.py:316-376)")
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    log(
        f"    hold mask (|chi|>{B.CHI_OUT_DEG:.0f} AND nz>{B.NZ_LOADED:.0f}) true on {sum(hold)}/{len(tr)} ticks"
    )
    for i, j in B._runs(hold):
        dur = tr[j].t - tr[i].t
        log(
            f"    run steps {i}-{j}  t {tr[i].t:.3f}-{tr[j].t:.3f}  dur {dur:.3f} s"
            + ("" if dur >= B.CHI_MIN_HOLD_S else "   (below the 0.5 s bar)")
        )
    for ev in det62["displacement_rolls"]:
        i, j = ev.i0, ev.i1
        hold_end = next(b for a, b in B._runs(hold) if a == i)
        log(
            f"\n    EVENT: i0={i} i1(exit)={j} hold_end={hold_end}  hold_s={ev.detail['hold_s']:.3f} "
            f"(bar {B.CHI_MIN_HOLD_S}, margin {ev.detail['hold_s'] / B.CHI_MIN_HOLD_S:.2f}x)"
        )
        log(
            f"      d_aspect {ev.d_aspect_deg:+.1f}  d_rng {ev.d_rng_m:+.0f}  d_es {ev.d_es_m:+.0f}  "
            f"chi_peak {ev.detail['chi_peak_deg']:.1f}  nz_peak {ev.detail['nz_peak']:.2f}"
        )
        hs = rows62[i : hold_end + 1]
        log(
            f"      segments spanned by the hold: {sorted({q['seg'] for q in hs})}  "
            f"cmd_bank set {sorted({round(q['cmd_bank'], 2) for q in hs})}  "
            f"cmd_load set {sorted({round(q['cmd_load'], 3) for q in hs})}"
        )
        nzs = [q["nz"] for q in hs]
        chis = [abs(q["chi"]) for q in hs]
        phis = [q["phi"] for q in hs]
        ails = [q["ail"] for q in hs]
        log(
            f"      nz over hold: min {min(nzs):.2f} max {max(nzs):.2f} g   (clause (b) needs > 3 on EVERY tick)"
        )
        log(f"      |chi| over hold: min {min(chis):.1f} max {max(chis):.1f} deg")
        log(f"      chi_valid over hold: all True = {all(q['chi_valid'] for q in hs)}")
        log(
            f"      OWN BANK phi over hold: min {min(phis):+.1f} max {max(phis):+.1f} deg  "
            f"range {max(phis) - min(phis):.1f} deg"
        )
        log(
            f"      aileron over hold: min {min(ails):+.3f} max {max(ails):+.3f} "
            f"mean {sum(ails) / len(ails):+.3f}  |ail|>0.5 on "
            f"{sum(1 for a in ails if abs(a) > 0.5)}/{len(ails)} ticks"
        )

        # V5 decomposition
        log("\nV5  WHERE DID THE chi EXCURSION COME FROM?  chi = phi_lift - phi_los")
        pl = np.array([q["phi_lift"] for q in hs])
        plos = np.array([q["phi_los"] for q in hs])
        log(
            f"      phi_lift (own lift roll about v): {pl[0]:+.1f} -> {pl[-1]:+.1f} deg   "
            f"total unwrapped travel {np.abs(np.diff(np.unwrap(np.radians(pl)) * DEG)).sum():.1f} deg"
        )
        log(
            f"      phi_los  (LOS roll about v):      {plos[0]:+.1f} -> {plos[-1]:+.1f} deg   "
            f"total unwrapped travel {np.abs(np.diff(np.unwrap(np.radians(plos)) * DEG)).sum():.1f} deg"
        )
        log(
            f"      d(phi_lift) over hold {np.unwrap(np.radians(pl))[-1] * DEG - pl[0]:+.1f} deg   "
            f"d(phi_los) over hold {np.unwrap(np.radians(plos))[-1] * DEG - plos[0]:+.1f} deg"
        )

        # V6 exit sentinel direction
        log("\nV6  THE (d) EXIT: sentinel, and WHICH WAY is the bandit?")
        for k in (hold_end, j):
            q = rows62[k]
            log(
                f"      step {k:3d} t {q['t']:.3f}  |chi| {abs(q['chi']):.1f} valid {q['chi_valid']}  "
                f"ata {q['ata']:.1f} deg  aspect {q['aspect']:.1f}  rng {q['rng']:.0f} m  "
                f"closure {q['closure']:+.0f} m/s  |los_perp| {q['los_perp']:.4f} "
                f"(valid needs > {math.sin(math.radians(B.CHI_VALID_ATA_DEG)):.4f})"
            )
        log(
            f"      ata > 90 deg means the bandit is BEHIND the nose. ata at exit = {rows62[j]['ata']:.1f} deg"
        )

        # V7 validity-gated (d) + (e) + (f)
        log("\nV7  DOES THE EVENT SURVIVE A VALIDITY-GATED (d)?  (exit must be a VALID |chi|<30)")
        alt_exit = None
        for k in range(hold_end + 1, len(tr)):
            if tr[k].t - tr[hold_end].t > B.CHI_RETURN_S:
                break
            if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                alt_exit = k
                break
        if alt_exit is None:
            log("      NONE inside the return window -> event exists ONLY via the sentinel")
        else:
            sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, alt_exit + 1)}
            dasp = (tr[alt_exit].aspect - tr[i].aspect) * DEG
            log(
                f"      valid exit at step {alt_exit} t {tr[alt_exit].t:.3f} |chi| "
                f"{abs(tr[alt_exit].chi) * DEG:.1f}  ata {tr[alt_exit].ata * DEG:.1f}"
            )
            log(
                f"      (e) sign(ata_az) constant over {i}..{alt_exit}: {len(sides) == 1}  sides={sides}"
            )
            log(
                f"      (f) d_aspect {dasp:+.1f} deg (needs <= -{B.ASPECT_GAIN_DEG:.0f})  "
                f"-> {'PASS' if dasp < -B.ASPECT_GAIN_DEG else 'FAIL'}"
            )
            log(
                f"      => validity-gated event survives: "
                f"{len(sides) == 1 and dasp < -B.ASPECT_GAIN_DEG}"
            )

        # V8 price
        log("\nV8  PRICE THE HOLD")
        a0, a1 = rows62[i], rows62[hold_end]
        dt_h = a1["t"] - a0["t"]
        log(
            f"      alt (Sample.alt = pos[2], what the detector reads): {a0['alt_enu']:.0f} -> "
            f"{a1['alt_enu']:.0f} m = {a1['alt_enu'] - a0['alt_enu']:+.0f} m over {dt_h:.3f} s "
            f"= {(a1['alt_enu'] - a0['alt_enu']) / dt_h:+.1f} m/s"
        )
        log(
            f"      alt (AState.alt, MSL):                             {a0['alt_state']:.0f} -> "
            f"{a1['alt_state']:.0f} m = {a1['alt_state'] - a0['alt_state']:+.0f} m"
        )
        log(
            f"      gamma {a0['gamma']:+.1f} -> {a1['gamma']:+.1f} deg   "
            f"min over hold {min(q['gamma'] for q in hs):+.1f}"
        )
        log(
            f"      vt {a0['vt']:.0f} -> {a1['vt']:.0f} m/s   psi {a0['psi']:+.1f} -> {a1['psi']:+.1f} deg  "
            f"heading travel {np.abs(np.diff(np.unwrap(np.radians([q['psi'] for q in hs])) * DEG)).sum():.1f} deg"
        )
        log(
            f"      rng {a0['rng']:.0f} -> {a1['rng']:.0f} m  closure {a0['closure']:+.0f} -> "
            f"{a1['closure']:+.0f} m/s  aspect {a0['aspect']:.1f} -> {a1['aspect']:.1f} deg"
        )
        log(f"      cpa_miss {a0['cpa']:.1f} -> {a1['cpa']:.1f} m")

    log("\nV8b EPISODE PRICE, seed 62")
    log(
        f"      alt_enu {rows62[0]['alt_enu']:.0f} -> {rows62[-1]['alt_enu']:.0f} m "
        f"= {rows62[-1]['alt_enu'] - rows62[0]['alt_enu']:+.0f} m over {rows62[-1]['t']:.3f} s"
    )
    log(
        f"      r.dalt_m = {r62.dalt_m:+.0f} m   min alt_state over episode "
        f"{min(q['alt_state'] for q in rows62):.0f} m"
    )
    log(
        f"      rounds fired {r62.rounds}  ammo {rows62[0]['ammo']:.0f} -> {rows62[-1]['ammo']:.0f}  "
        f"hp dealt {r62.hp_dealt:.1f}  hp taken {r62.hp_taken:.1f}  tLeth {r62.tleth_s:.3f} s"
    )
    log(f"      score {sc62:+.3f}")

    # incumbent control
    log("\nV8c INCUMBENT (d3b_best.npz) on seed 62, for reference")
    ginc = np.load(f"{OUT}/d3b_best.npz")["genome"]
    rinc, rowsinc = fly(ginc, 62)
    d1i, d2i, d2ci, d3i, _ = counts(rinc)
    sci, _ = M.score(rinc)
    log(
        f"      steps {rinc.steps} event {rinc.event} kill {rinc.kill} hp {rinc.hp_dealt:.0f} "
        f"tLeth {rinc.tleth_s:.3f} dalt {rinc.dalt_m:+.0f} D1 {d1i} D2 {d2i} D2c {d2ci} D3 {d3i} "
        f"score {sci:+.3f}"
    )

    # ---------------- V9 the control the claim never ran ----------------
    log("\nV9  CONTROL: does D1 = 1 require a ROLL at all?  Constant-bank, max-load, 24 s genomes")
    log(
        "    (all six segments identical => ONE bank target, ONE load: no displacement, no reversal)"
    )
    log(
        f"    {'bank_deg':>9} {'D1':>3} {'D2':>3} {'D2c':>4} {'D3':>3} {'hold_s':>7} {'dalt':>7} "
        f"{'event':>10} {'rngMn':>7}"
    )
    for bank_deg in (45.0, 67.02, 90.0, 100.37, 115.0, -67.02, -115.0):
        p0 = float(np.clip(bank_deg / M.BANK_MAX_DEG, -1.0, 1.0))
        g = np.tile(np.array([p0, 1.0, 0.2]), 6)
        r, rws = fly(g, 62)
        d1, d2, d2c, d3, det = counts(r)
        hs = max((e.detail.get("hold_s", 0.0) for e in det["displacement_rolls"]), default=0.0)
        log(
            f"    {bank_deg:>9.2f} {d1:>3} {d2:>3} {d2c:>4} {d3:>3} {hs:>7.3f} {r.dalt_m:>+7.0f} "
            f"{str(r.event):>10} {r.rng_min:>7.0f}"
        )

    with open("/workspace/controllers/rl/runs/d3b/verify_d1_claim.txt", "w") as fh:
        fh.write("\n".join(_LOG) + "\n")


if __name__ == "__main__":
    main()
