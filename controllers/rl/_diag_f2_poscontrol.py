"""F2 — the detector POSITIVE CONTROL (never run before), with a fly-by contrast + the perch re-fly.

Everything this program says about BFM rests on one instrument (`bfm_detect`) and that instrument has
only ever been shown NOT to falsely reject.  Two facts motivate this box:

  * the D3b audit showed D1 = 0 across all 20 threshold combinations on a real gun kill => "not too
    strict" -- but that is a negative result about a trajectory that was never a displacement roll;
  * `runs/d3b/d1_exists_report.txt` shows D1 = 1 on a genome whose "event" is a 890 -> 349 m fly-by
    at +-450 m/s with aspect 175.8 -> 25.3 deg, clause (d) satisfied by the chi-validity SENTINEL
    (bandit dead ahead) rather than by rolling back in.  That is a FALSE POSITIVE, i.e. the opposite
    failure mode, and it means the instrument is unvalidated in BOTH directions.

So: fly the library's own closed-loop primitives AT THE GEOMETRY EACH IS FOR, and ask two separable
questions that the program has so far conflated.

  Q1  does the jet actually FLY the maneuver?      (kinematics: nz, chi hold, gamma swing)
  Q2  does the detector then SEE it?               (D1/D2/D2c/D3)

PRE-REGISTERED READING -- all three outcomes are stated before the run, and they are exclusive:
  (A) Q1 yes and Q2 yes  => the detector is VALIDATED in the positive direction.  Every "zero BFM"
      reading in the program stands as written.
  (B) Q1 yes and Q2 no   => the detector is TOO STRICT on a real maneuver.  Every "zero BFM" reading
      becomes suspect and must be re-derived; the clause attribution below names the clause to fix.
  (C) Q1 no              => the STICK LIBRARY cannot fly the maneuver (Phase A's closure), the
      detector remains UNVALIDATED, and a positive control needs a different generator.  This is
      NOT evidence about the detector -- do not report it as one.

F2b rides along in the same invocation (same env build, same seeds): re-fly the D3b winning genome
with `me_dalt_ft = 0`.  The written prediction is that the kill DISAPPEARS, i.e. it was an exploit of
`BREAK_BANK_RAD = 85 deg` having no LOS-plane term while the +2000 ft perch tilts that plane.

Usage (inside the rl container):
    python /workspace/controllers/rl/_diag_f2_poscontrol.py
"""

from __future__ import annotations

import math
import os

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import bfm_detect as B
from astrum_dogfight import constants as C

OUT = "/workspace/controllers/rl/runs/f2"
SEEDS = (11, 28, 45, 62, 79, 96, 113, 130)  # the D3b search seeds, so F2b is directly comparable
DT = M.CONTROL_DT_S

CHASE_S = 60.0  # the program's chase default (`_diag_credit_reach._cell_defaults`)
MERGE_S = 24.0  # the D3b horizon
N_CHASE = int(round(CHASE_S / DT))
N_MERGE = int(round(MERGE_S / DT))

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


# =================================================================================================
# Cells
# =================================================================================================
def chase_cell(aspect_deg: float) -> dict:
    """The program's chase cell: sep 700 m, co-heading (`opp_heading_deg` unset => both head north),
    no perch.  `aspect_deg` is the bandit's BEARING around the learner, so 0 = dead ahead."""
    return dict(
        separation_m=700.0,
        aspect_deg=float(aspect_deg),
        me_dalt_ft=0.0,
        alt_ft=15000.0,
    )


def merge_cell(perch_ft: float) -> dict:
    c = dict(M.CELLS["merge1500"])
    c.pop("lateral_m", None)
    c["me_dalt_ft"] = float(perch_ft)
    return c


# =================================================================================================
# Controllers
# =================================================================================================
def _pad(a4) -> np.ndarray:
    """bfm.py sticks are 4-vectors [ail, elev, rud, thr]; roll_once overwrites a[4] for every arm."""
    a = np.asarray(a4, np.float32).reshape(-1)
    return np.array([a[0], a[1], a[2], a[3], -1.0], np.float32)


def forced_ctl(factory, entry_pred):
    """Pursue until `entry_pred(me, rel)` fires, then hand the stick to the primitive until it
    reports done, then resume pursuit.  `st` records the maneuver window so the kinematics can be
    read over the maneuver itself and not diluted by the ingress."""
    st = {"phase": "in", "mv": None, "i0": None, "i1": None, "trk": None}

    def _pursue(me, rel):
        if st["trk"] is None:
            st["trk"] = bfm.LagPursuitTracker()
            st["trk"].reset(me, rel)
        a4, _ = st["trk"].step(me, rel)
        return _pad(a4)

    def fn(step, me, bd, rel):
        if st["phase"] == "in":
            if entry_pred(me, rel):
                st["mv"] = factory()
                st["mv"].reset(me, rel)
                st["phase"] = "mv"
                st["i0"] = step
            else:
                return _pursue(me, rel)
        if st["phase"] == "mv":
            a4, done = st["mv"].step(me, rel)
            if done:
                st["phase"] = "out"
                st["i1"] = step
                st["trk"] = None  # fresh tracker for the egress
            return _pad(a4)
        return _pursue(me, rel)

    return fn, st


def coach_ctl():
    st = {"phase": "coach", "i0": 0, "i1": None}
    coach = bfm.BFMCoach()

    def fn(step, me, bd, rel):
        a = np.asarray(coach.act(me, rel), np.float32).reshape(-1)
        return np.array([a[0], a[1], a[2], a[3], -1.0], np.float32)

    return fn, st


def genome_ctl(g: np.ndarray, n_steps: int):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    st = {"phase": "genome", "i0": 0, "i1": None}

    def fn(step, me, bd, rel):
        return gm.act(g, step, n_steps, me)

    return fn, st


# =================================================================================================
# Q1 — did the jet FLY a maneuver?  (read independently of every detector clause)
# =================================================================================================
def kinematics(tr: list[B.Sample], i0: int | None, i1: int | None) -> dict:
    """Raw evidence of a maneuver over [i0, i1] (defaults to the whole trace).

    `chi_hold_s` is the RAW D1 ingredient -- clause (a)+(b) only, i.e. the longest contiguous run of
    (|chi| > CHI_OUT_DEG AND nz > NZ_LOADED).  Comparing it against the D1 verdict is what separates
    "the detector rejected a maneuver" from "there was no maneuver to reject".
    """
    a = 0 if i0 is None else max(0, i0)
    b = (len(tr) - 1) if i1 is None else min(len(tr) - 1, i1)
    if b <= a:
        b = len(tr) - 1
    w = tr[a : b + 1]
    if not w:
        return dict(n=0)
    nz = np.array([s.nz for s in w], float)
    loaded = nz > B.NZ_LOADED
    chi_loaded = [abs(math.degrees(s.chi)) for s in w if s.nz > B.NZ_LOADED and s.chi_valid]
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in w]
    hold_s = 0.0
    for i, j in B._runs(hold):
        hold_s = max(hold_s, w[j].t - w[i].t)
    gam = np.array([math.degrees(s.gamma) for s in w], float)
    return dict(
        n=len(w),
        t0=w[0].t,
        t1=w[-1].t,
        nz_p50=float(np.median(nz)),
        nz_max=float(nz.max()),
        frac_loaded=float(loaded.mean()),
        chi_max_loaded=(max(chi_loaded) if chi_loaded else float("nan")),
        chi_hold_s=hold_s,
        gamma_swing=float(gam.max() - gam.min()),
        rng_p50=float(np.median([s.rng for s in w])),
        clo_max=float(max(s.closure for s in w)),
        dalt=float(w[-1].alt - w[0].alt),
    )


# =================================================================================================
# Q2 diagnosis — which of D1's six clauses kills each candidate hold?
# =================================================================================================
_CLAUSES = (
    "hold_s",
    "(a) lookback",
    "(c) closing",
    "(d) rolled back",
    "(e) same side",
    "(f) angles",
)


def d1_attribute(tr: list[B.Sample]) -> list[dict]:
    """Re-walk `detect_displacement_rolls` and record the FIRST failing clause per candidate hold.

    Mirrors `bfm_detect.detect_displacement_rolls` clause-for-clause (bfm_detect.py:316-378).
    """
    out = []
    if len(tr) < 3:
        return out
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    for i, j in B._runs(hold):
        rec = dict(
            i0=i,
            i1=j,
            t0=tr[i].t,
            hold_s=tr[j].t - tr[i].t,
            chi_peak=max(abs(math.degrees(tr[k].chi)) for k in range(i, j + 1)),
            nz_peak=max(tr[k].nz for k in range(i, j + 1)),
            fail=None,
        )
        if rec["hold_s"] < B.CHI_MIN_HOLD_S:
            rec["fail"] = "hold_s"
            out.append(rec)
            continue
        if not any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        ):
            rec["fail"] = "(a) lookback"
            out.append(rec)
            continue
        if tr[i].closure <= 0.0:
            rec["fail"] = "(c) closing"
            rec["closure"] = tr[i].closure
            out.append(rec)
            continue
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            rec["fail"] = "(d) rolled back"
            out.append(rec)
            continue
        rec["exit"] = exit_idx
        rec["exit_valid"] = bool(tr[exit_idx].chi_valid)
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
        if len(sides) != 1:
            rec["fail"] = "(e) same side"
            out.append(rec)
            continue
        ev = B._event_from(tr, "displacement_roll", i, exit_idx)
        rec["d_aspect"] = ev.d_aspect_deg
        if ev.d_aspect_deg >= -B.ASPECT_GAIN_DEG:
            rec["fail"] = "(f) angles"
            out.append(rec)
            continue
        rec["fail"] = None  # PASSED every clause
        out.append(rec)
    return out


# =================================================================================================
# Arms
# =================================================================================================
_R = math.radians


def _entry_high_yoyo(me, rel):  # BFMCoach branch (b)
    return rel.aspect < _R(90.0) and rel.rng < C.YOYO_RANGE and rel.closure > C.CLOSURE_HI


def _entry_barrel(me, rel):  # BFMCoach branch (b), the hotter half
    return rel.aspect < _R(90.0) and rel.rng < C.YOYO_RANGE and rel.closure > C.CLOSURE_HI + 60.0


def _entry_low_yoyo(me, rel):  # BFMCoach branch (c)
    return (
        rel.aspect < bfm.COACH_REAR_ASPECT
        and rel.rng > C.YOYO_RANGE_EXIT
        and rel.closure < C.CLOSURE_LO
    )


def _entry_lag_roll(me, rel):
    return rel.rng < C.YOYO_RANGE and rel.closure > 0.0


def _entry_lead_turn(me, rel):
    # the band the hand-coded conversion committed in (`merge-leadturn-conversion`: ~1000 m)
    return rel.rng <= 1200.0


def arms(d3b_genome: np.ndarray, flyby_genome: np.ndarray) -> list[dict]:
    """Every arm, with the cell it is FLOWN AT.  A primitive at the wrong geometry is not a control."""
    out: list[dict] = []
    for asp in (0.0, 30.0):
        cell, n = chase_cell(asp), N_CHASE
        tag = f"chase{asp:.0f}"
        out += [
            dict(
                name=f"{tag}:high_yoyo",
                cell=cell,
                n=n,
                kind="forced",
                factory=lambda: bfm.HighYoYo(),
                entry=_entry_high_yoyo,
            ),
            dict(
                name=f"{tag}:barrel",
                cell=cell,
                n=n,
                kind="forced",
                factory=lambda: bfm.BarrelRollAttack(),
                entry=_entry_barrel,
            ),
            dict(
                name=f"{tag}:lag_roll",
                cell=cell,
                n=n,
                kind="forced",
                factory=lambda: bfm.LagDisplacementRoll(),
                entry=_entry_lag_roll,
            ),
            dict(
                name=f"{tag}:low_yoyo",
                cell=cell,
                n=n,
                kind="forced",
                factory=lambda: bfm.LowYoYo(),
                entry=_entry_low_yoyo,
            ),
            dict(name=f"{tag}:coach", cell=cell, n=n, kind="coach"),
        ]
    mc = merge_cell(2000.0)
    out += [
        dict(
            name="merge:lead_turn",
            cell=mc,
            n=N_MERGE,
            kind="forced",
            factory=lambda: bfm.LeadTurn(),
            entry=_entry_lead_turn,
        ),
        dict(name="merge:coach", cell=mc, n=N_MERGE, kind="coach"),
        dict(name="merge:FLYBY(d1exists)", cell=mc, n=N_MERGE, kind="genome", g=flyby_genome),
        dict(name="merge:d3b_perch2000", cell=mc, n=N_MERGE, kind="genome", g=d3b_genome),
        dict(name="merge:d3b_perch0", cell=merge_cell(0.0), n=N_MERGE, kind="genome", g=d3b_genome),
    ]
    return out


def fly(arm: dict, seed: int):
    env, base = M.build_env(arm["cell"], seed, arm["n"] * DT)
    if arm["kind"] == "forced":
        ctl, st = forced_ctl(arm["factory"], arm["entry"])
    elif arm["kind"] == "coach":
        ctl, st = coach_ctl()
    else:
        ctl, st = genome_ctl(arm["g"], arm["n"])
    r = M.roll_once(env, base, ctl, arm["n"], keep=False)
    env.close()
    return r, st


# =================================================================================================
# Main
# =================================================================================================
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    log("#" * 100)
    log("#  F2 — DETECTOR POSITIVE CONTROL (+ fly-by contrast, + the me_dalt_ft=0 re-fly)")
    log("#" * 100)
    log(
        f"\n  seeds {SEEDS}   chase {CHASE_S:.0f} s ({N_CHASE} steps)   merge {MERGE_S:.0f} s "
        f"({N_MERGE} steps)   dt {DT * 1000:.3f} ms"
    )
    log(
        f"  detector bars: CHI_OUT {B.CHI_OUT_DEG:.0f} deg  HOLD {B.CHI_MIN_HOLD_S:.2f} s  "
        f"NZ_LOADED {B.NZ_LOADED:.1f} g  ASPECT_GAIN {B.ASPECT_GAIN_DEG:.0f} deg"
    )
    log(
        "  PRE-REGISTERED: (A) flew+seen => detector validated; (B) flew+unseen => TOO STRICT, every "
        "zero-BFM reading suspect;\n                  (C) did not fly => library cannot generate it, "
        "detector still UNVALIDATED (not evidence about the detector)."
    )

    d3b = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"]
    # the fly-by: the L=4.0 s D1-exists genome, hard-coded from runs/d3b/d1_exists_report.txt so this
    # script does not depend on re-running the P2 chain.
    flyby = np.array(
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

    rows: list[dict] = []
    for arm in arms(d3b, flyby):
        per = []
        for sd in SEEDS:
            r, st = fly(arm, sd)
            det = B.detect_all(r.trace)
            kin_all = kinematics(r.trace, None, None)
            kin_mv = (
                kinematics(r.trace, st.get("i0"), st.get("i1"))
                if st.get("i0") is not None
                else None
            )
            per.append(
                dict(
                    seed=sd,
                    r=r,
                    det=det,
                    kin_all=kin_all,
                    kin_mv=kin_mv,
                    st=dict(st),
                    attr=d1_attribute(r.trace),
                )
            )
        rows.append(dict(arm=arm, per=per))
        k = sum(p["r"].kill for p in per)
        e = sum(1 for p in per if p["st"].get("i0") is not None)
        log(f"    flew {arm['name']:<26} kills {k}/8  entered {e}/8")

    # ---- TABLE 1: outcome + detector -----------------------------------------------------------
    log("\n" + "=" * 100)
    log("TABLE 1 — outcome and detector, pooled over the 8 seeds")
    log("=" * 100)
    log(
        f"  {'arm':<26} {'entry':>6} {'kill':>5} {'hp':>7} {'tLeth':>7} {'rngMn':>7} {'dalt':>8} "
        f"{'D1':>4} {'D2':>4} {'D2c':>4} {'D3':>4} {'decks':>6}"
    )
    for row in rows:
        per, a = row["per"], row["arm"]
        ent = sum(1 for p in per if p["st"].get("i0") is not None)
        log(
            f"  {a['name']:<26} {ent:>5}/8 {sum(p['r'].kill for p in per):>5} "
            f"{np.mean([p['r'].hp_dealt for p in per]):>7.1f} "
            f"{np.mean([p['r'].tleth_s for p in per]):>7.3f} "
            f"{np.mean([p['r'].rng_min for p in per]):>7.0f} "
            f"{np.mean([p['r'].dalt_m for p in per]):>+8.0f} "
            f"{sum(len(p['det']['displacement_rolls']) for p in per):>4} "
            f"{sum(len(p['det']['yoyos']) for p in per):>4} "
            f"{sum(len(p['det']['yoyos_clean']) for p in per):>4} "
            f"{sum(len(p['det']['turning_room']) for p in per):>4} "
            f"{sum(1 for p in per if p['r'].dead):>6}"
        )

    # ---- TABLE 2: Q1, did the jet FLY a maneuver? -----------------------------------------------
    log("\n" + "=" * 100)
    log("TABLE 2 — Q1: did the jet actually FLY one?  (over the MANEUVER WINDOW where one exists)")
    log("=" * 100)
    log(
        f"  {'arm':<26} {'win_s':>7} {'nzP50':>6} {'nzMax':>6} {'frLoad':>7} {'chiMaxL':>8} "
        f"{'chiHold':>8} {'gamSwg':>7} {'rngP50':>7} {'cloMax':>7} {'D1':>4} {'D2':>4}"
    )
    for row in rows:
        per, a = row["per"], row["arm"]
        use = [p["kin_mv"] or p["kin_all"] for p in per]
        use = [u for u in use if u.get("n")]
        if not use:
            log(f"  {a['name']:<26} {'--- never entered ---':>60}")
            continue

        def m(key):
            v = [u[key] for u in use if not (isinstance(u[key], float) and math.isnan(u[key]))]
            return float(np.mean(v)) if v else float("nan")

        log(
            f"  {a['name']:<26} {m('t1') - m('t0'):>7.2f} {m('nz_p50'):>6.2f} {m('nz_max'):>6.2f} "
            f"{m('frac_loaded'):>7.3f} {m('chi_max_loaded'):>8.1f} {m('chi_hold_s'):>8.3f} "
            f"{m('gamma_swing'):>7.1f} {m('rng_p50'):>7.0f} {m('clo_max'):>7.0f} "
            f"{sum(len(p['det']['displacement_rolls']) for p in per):>4} "
            f"{sum(len(p['det']['yoyos']) for p in per):>4}"
        )

    # ---- TABLE 3: Q2, D1 clause attribution ----------------------------------------------------
    log("\n" + "=" * 100)
    log("TABLE 3 — Q2: D1 clause attribution over ALL candidate holds (pooled, 8 seeds/arm)")
    log("  A candidate hold = a contiguous run of (|chi| > CHI_OUT AND nz > NZ_LOADED). 'PASS' = a")
    log("  detected displacement roll. Any other column = the FIRST clause that rejected it.")
    log("=" * 100)
    log(f"  {'arm':<26} {'cands':>6} {'PASS':>5} " + " ".join(f"{c:>15}" for c in _CLAUSES))
    for row in rows:
        per, a = row["per"], row["arm"]
        allc = [c for p in per for c in p["attr"]]
        cnt = {k: sum(1 for c in allc if c["fail"] == k) for k in _CLAUSES}
        log(
            f"  {a['name']:<26} {len(allc):>6} {sum(1 for c in allc if c['fail'] is None):>5} "
            + " ".join(f"{cnt[k]:>15}" for k in _CLAUSES)
        )
        # the interesting ones: holds that cleared the duration bar but were still rejected
        big = [c for c in allc if c["hold_s"] >= B.CHI_MIN_HOLD_S and c["fail"] is not None]
        for c in sorted(big, key=lambda x: -x["hold_s"])[:3]:
            log(
                f"      rejected hold {c['hold_s']:.3f} s  chi_peak {c['chi_peak']:.1f}  "
                f"nz_peak {c['nz_peak']:.2f} g  by {c['fail']}"
                + (f"  d_aspect {c['d_aspect']:+.1f} deg" if "d_aspect" in c else "")
                + (f"  closure {c['closure']:+.0f} m/s" if "closure" in c else "")
            )

    # ---- TABLE 4: F2b ---------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("TABLE 4 — F2b: does the +2000 ft perch make the D3b merge kill?  (per seed)")
    log("=" * 100)
    by = {r["arm"]["name"]: r for r in rows}
    for nm in ("merge:d3b_perch2000", "merge:d3b_perch0"):
        if nm not in by:
            continue
        per = by[nm]["per"]
        log(f"\n  {nm}")
        log(
            f"    {'seed':>5} {'steps':>6} {'event':>10} {'kill':>5} {'hp':>7} {'tLeth':>7} "
            f"{'rngMn':>7} {'rounds':>7} {'dalt':>8}"
        )
        for p in per:
            r = p["r"]
            log(
                f"    {p['seed']:>5} {r.steps:>6} {str(r.event):>10} {r.kill:>5} {r.hp_dealt:>7.1f} "
                f"{r.tleth_s:>7.3f} {r.rng_min:>7.0f} {r.rounds:>7} {r.dalt_m:>+8.0f}"
            )
        log(
            f"    POOLED kills {sum(p['r'].kill for p in per)}/8   hp "
            f"{np.mean([p['r'].hp_dealt for p in per]):.1f}   tLeth "
            f"{np.mean([p['r'].tleth_s for p in per]):.3f} s   rngMn "
            f"{np.mean([p['r'].rng_min for p in per]):.0f} m"
        )
    if "merge:d3b_perch2000" in by and "merge:d3b_perch0" in by:
        a = sum(p["r"].kill for p in by["merge:d3b_perch2000"]["per"])
        b = sum(p["r"].kill for p in by["merge:d3b_perch0"]["per"])
        ha = np.mean([p["r"].hp_dealt for p in by["merge:d3b_perch2000"]["per"]])
        hb = np.mean([p["r"].hp_dealt for p in by["merge:d3b_perch0"]["per"]])
        log(
            f"\n  => perch 2000 ft: {a}/8 kills, {ha:.1f} HP   |   perch 0 ft: {b}/8 kills, {hb:.1f} HP"
        )
        log(
            "  PREDICTION WAS: the kill disappears at perch 0 => "
            + ("CONFIRMED" if b == 0 and a > 0 else "NOT CONFIRMED (read the rows)")
        )

    path = f"{OUT}/f2_report.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
