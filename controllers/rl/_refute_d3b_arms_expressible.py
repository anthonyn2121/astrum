"""REFUTATION AUDIT -- 'the hand-coded closed-loop arms also score D1=0/D2=0, therefore the
open-loop 4.0 s parameterisation is not what suppressed D1/D2'.

A control that is INCAPABLE of producing the event cannot corroborate anything about why the
event is absent. So this script does not just re-count D1/D2 -- it instruments, per arm, WHICH
clause of D1 and D2 fails, and whether the arm ever reaches the hold PRECONDITION at all
(|chi| > CHI_OUT_DEG while nz > NZ_LOADED). Every hold is priced in altitude.

Run inside the container:
  cd /workspace/controllers/rl && python _refute_d3b_arms_expressible.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

DEG = 180.0 / math.pi
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
CELL = M.CELLS["merge1500"]
NSTEPS = 576
DT = M.CONTROL_DT_S

OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def runs_of(mask):
    """[(i, j)] inclusive runs of True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def fly(arm, seed):
    env, base = M.build_env(CELL, seed, NSTEPS * DT)
    if arm == "SEARCH":
        g = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)["genome"]
        gm = M.Genome(6, 3, "bank", True)
        ctl = lambda step, me, bd, rel: gm.act(g, step, NSTEPS, me)  # noqa: E731
    else:
        ctl = M.make_control(arm)
    r = M.roll_once(env, base, ctl, NSTEPS, keep=True)
    env.close()
    return r


def arm_report(arm, rows):
    tr_all = [r.trace for r in rows]
    per = [B.detect_all(t) for t in tr_all]
    d1 = sum(len(p["displacement_rolls"]) for p in per)
    d2 = sum(len(p["yoyos"]) for p in per)
    d2c = sum(len(p["yoyos_clean"]) for p in per)
    d3 = sum(len(p["turning_room"]) for p in per)
    leth = sum(p["lethal_ticks"] for p in per)
    chimx = max(max((abs(s.chi) for s in t if s.chi_valid), default=0.0) for t in tr_all)
    nzmx = max(max(s.nz for s in t) for t in tr_all)
    dalt = float(np.mean([r.dalt_m for r in rows]))
    kills = sum(r.kill for r in rows)
    log(
        f"     {arm:<14} {len(tr_all):>4} {d1:>4} {d2:>4} {d2c:>4} {d3:>4} {leth:>7} "
        f"{chimx * DEG:>6.1f} {nzmx:>5.1f} {dalt:>+7.0f} {kills:>6d}"
    )
    return dict(d1=d1, d2=d2, d2c=d2c, d3=d3, leth=leth, chimx=chimx * DEG, nzmx=nzmx)


def clause_audit(arm, r, seed):
    tr = r.trace
    n = len(tr)
    chi = np.array([abs(s.chi) * DEG for s in tr])
    val = np.array([s.chi_valid for s in tr])
    nz = np.array([s.nz for s in tr])
    rng = np.array([s.rng for s in tr])
    clo = np.array([s.closure for s in tr])
    gam = np.array([s.gamma * DEG for s in tr])
    alt = np.array([s.alt for s in tr])
    cpa = np.array([s.cpa_miss for s in tr])
    lay = np.array([s.lay_miss_m for s in tr])
    loaded = nz > B.NZ_LOADED

    log(f"   --- {arm} seed {seed}  ({n} steps = {n * DT:.3f} s) ---")
    log(
        f"     alt {alt[0]:.0f} -> {alt[-1]:.0f} m (dalt {alt[-1] - alt[0]:+.0f})   "
        f"rng min {rng.min():.0f} m at step {int(rng.argmin())}   "
        f"laymin {lay.min():.1f} m   rounds {r.rounds}   ticksInRange(<=600) {int((rng <= 600).sum())}"
    )
    # ---- chi ceiling, unconditional and LOAD-GATED ------------------------------------
    iv = np.flatnonzero(val)
    if len(iv):
        k = iv[int(np.argmax(chi[iv]))]
        log(
            f"     max|chi| (valid, ANY nz) = {chi[k]:.1f} deg at step {k} t {k * DT:.3f} s, "
            f"nz {nz[k]:.2f} g   [this is the table's chiMx]"
        )
    m = val & loaded
    im = np.flatnonzero(m)
    if len(im):
        k = im[int(np.argmax(chi[im]))]
        log(
            f"     max|chi| while LOADED (nz>{B.NZ_LOADED:.0f}) = {chi[k]:.1f} deg at step {k} "
            f"t {k * DT:.3f} s, nz {nz[k]:.2f} g   [{int(m.sum())} loaded+valid ticks]"
        )
    else:
        log("     max|chi| while LOADED = NONE (no loaded+valid tick at all)")
    log(f"     loaded ticks (nz>3) = {int(loaded.sum())}/{n} = {100.0 * loaded.mean():.1f}%")

    # ---- D1 precondition: |chi|>60 AND nz>3, held >=0.5 s -----------------------------
    h1 = (chi > B.CHI_OUT_DEG) & loaded
    r1 = runs_of(h1.tolist())
    tot = 0.0
    for i, j in r1:
        tot = max(tot, (j - i) * DT)
    log(
        f"     D1 precondition |chi|>{B.CHI_OUT_DEG:.0f} & nz>3 : {len(r1)} run(s), "
        f"longest {tot:.3f} s vs CHI_MIN_HOLD_S {B.CHI_MIN_HOLD_S:.1f} s "
        f"-> {'REACHES the bar' if tot >= B.CHI_MIN_HOLD_S else 'NEVER reaches the bar'}"
    )
    for i, j in r1[:6]:
        log(
            f"         run steps {i}-{j}  {(j - i) * DT:.3f} s  |chi| {chi[i]:.1f}->{chi[j]:.1f}  "
            f"nz {nz[i : j + 1].min():.2f}-{nz[i : j + 1].max():.2f} g  "
            f"dalt {alt[j] - alt[i]:+.0f} m"
        )
    # how close does it get with the LOAD clause dropped?
    h1u = chi > B.CHI_OUT_DEG
    r1u = runs_of(h1u.tolist())
    lu = max([(j - i) * DT for i, j in r1u], default=0.0)
    log(
        f"     ... same hold with the nz clause DROPPED: {len(r1u)} run(s), longest {lu:.3f} s "
        f"(an unpriced hold; reported only to show the load clause is what binds)"
    )

    # ---- D3 precondition: |chi|>45 AND nz>3 -------------------------------------------
    h3 = (chi > B.CHI_ROOM_DEG) & loaded
    r3 = runs_of(h3.tolist())
    l3 = max([(j - i) * DT for i, j in r3], default=0.0)
    log(
        f"     D3 precondition |chi|>{B.CHI_ROOM_DEG:.0f} & nz>3 : {len(r3)} run(s), longest {l3:.3f} s"
    )
    for i, j in r3[:6]:
        if (j - i) * DT < B.CHI_MIN_HOLD_S:
            continue
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        built = cpa[j] > cpa[i]
        log(
            f"         run steps {i}-{j} {(j - i) * DT:.3f} s  cpa {cpa[i]:.1f}->{cpa[j]:.1f} m "
            f"({'BUILT' if built else 'fell -> rejected by the built clause'})  "
            f"sides {sorted(sides)}  dalt {alt[j] - alt[i]:+.0f} m"
        )

    # ---- D2 precondition: closure rising edge over 120 m/s, then a 30 deg gamma swing --
    edges = [
        i
        for i in range(n)
        if clo[i] > B.CLOSURE_OVERSHOOT and (i == 0 or clo[i - 1] <= B.CLOSURE_OVERSHOOT)
    ]
    log(
        f"     D2 precondition closure>{B.CLOSURE_OVERSHOOT:.0f} m/s rising edges: {len(edges)} "
        f"(closure max {clo.max():.0f} m/s)"
    )
    for i in edges[:6]:
        w = [k for k in range(i, n) if (k - i) * DT <= B.YOYO_RETURN_S]
        pk = max(w, key=lambda k: abs(gam[k] - gam[i]))
        swing = abs(gam[pk] - gam[i])
        ld = loaded[i : pk + 1]
        frac = float(ld.mean()) if len(ld) else 0.0
        why = []
        if swing < B.YOYO_GAMMA_SWING_DEG:
            why.append(f"swing {swing:.1f}<{B.YOYO_GAMMA_SWING_DEG:.0f}")
        if abs(gam[pk]) < B.YOYO_GAMMA_EXCURSION_DEG:
            why.append(f"|gamma_pk| {abs(gam[pk]):.1f}<{B.YOYO_GAMMA_EXCURSION_DEG:.0f}")
        if frac < B.LOADED_FRAC:
            why.append(f"loadedFrac {frac:.2f}<{B.LOADED_FRAC:.2f}")
        log(
            f"         edge step {i} t {i * DT:.3f}s  gamma0 {gam[i]:+.1f}  peak step {pk} "
            f"gamma {gam[pk]:+.1f}  swing {swing:.1f} deg  loadedFrac {frac:.2f}  "
            f"-> {'PASSES gates' if not why else 'fails: ' + ', '.join(why)}"
        )
    return dict(
        d1_pre=tot,
        d1_pre_unloaded=lu,
        d3_pre=l3,
        edges=len(edges),
        chimax_loaded=(chi[im].max() if len(im) else float("nan")),
        rngmin=float(rng.min()),
        rounds=int(r.rounds),
        dalt=float(alt[-1] - alt[0]),
    )


def main():
    log("=" * 92)
    log("REFUTATION AUDIT -- d3b merge1500 control arms: is D1=0/D2=0 EARNED or STRUCTURAL?")
    log("=" * 92)
    log(f"   cell merge1500 = {CELL}")
    log(f"   seeds {SEEDS}   n_steps {NSTEPS}   dt {DT * 1000:.3f} ms")
    log(
        f"   thresholds: CHI_OUT {B.CHI_OUT_DEG} CHI_ROOM {B.CHI_ROOM_DEG} "
        f"NZ_LOADED {B.NZ_LOADED} HOLD {B.CHI_MIN_HOLD_S}s CLOSURE_OVERSHOOT {B.CLOSURE_OVERSHOOT}"
    )

    arms = {}
    for arm in ("recorded", "recorded_two", "inplane"):
        rows = []
        for sd in SEEDS:
            rows.append(fly(arm, sd))
        arms[arm] = rows

    log("")
    log("   (1) PER-SEED IDENTITY -- are the '8 eps' 8 independent episodes or 8 copies of one?")
    log(f"     {'arm':<14} {'max|dAct| vs seed11':>20} {'distinct outcome rows':>24}")
    for arm, rows in arms.items():
        a0 = np.asarray(rows[0].actions, np.float64)
        dmax = 0.0
        for r in rows[1:]:
            a = np.asarray(r.actions, np.float64)
            m = min(len(a0), len(a))
            dmax = max(dmax, float(np.abs(a0[:m] - a[:m]).max()))
        sig = {
            (
                r.steps,
                r.kill,
                round(r.hp_dealt, 3),
                round(r.dalt_m, 3),
                r.rounds,
                round(r.rng_min, 3),
            )
            for r in rows
        }
        log(f"     {arm:<14} {dmax:>20.3e} {len(sig):>24}")

    log("")
    log("   (2) DETECTOR TABLE, recomputed from scratch (compare to d3b_prereg.txt:112-116)")
    log(
        f"     {'arm':<14} {'eps':>4} {'D1':>4} {'D2':>4} {'D2c':>4} {'D3':>4} "
        f"{'lethTk':>7} {'chiMx':>6} {'nzMx':>5} {'dalt':>7} {'kills':>6}"
    )
    tab = {}
    for arm, rows in arms.items():
        tab[arm] = arm_report(arm, rows)

    log("")
    log("   (3) CLAUSE-LEVEL AUDIT -- which clause kills D1/D2, per arm (seed 62 shown; the")
    log("       identity check above establishes whether the other seeds differ at all)")
    det = {}
    for arm, rows in arms.items():
        det[arm] = clause_audit(arm, rows[SEEDS.index(62)], 62)
        log("")

    log("   (4) THE SAME AUDIT ON THE SEARCH ARM, kill seed 62 and non-kill seed 28")
    for sd in (62, 28):
        r = fly("SEARCH", sd)
        det[f"SEARCH{sd}"] = clause_audit("SEARCH", r, sd)
        log("")

    log("=" * 92)
    log("   SUMMARY -- expressibility of the D1 hold, per arm")
    log("=" * 92)
    log(
        f"     {'arm':<14} {'maxChi@load':>12} {'D1preHold(s)':>13} {'D3preHold(s)':>13} "
        f"{'clo edges':>10} {'rngMin':>7} {'rnds':>5} {'dalt':>7}"
    )
    for k, v in det.items():
        log(
            f"     {k:<14} {v['chimax_loaded']:>12.1f} {v['d1_pre']:>13.3f} {v['d3_pre']:>13.3f} "
            f"{v['edges']:>10d} {v['rngmin']:>7.0f} {v['rounds']:>5d} {v['dalt']:>+7.0f}"
        )
    with open("/workspace/controllers/rl/runs/d3b/refute_arms_expressible.txt", "w") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
