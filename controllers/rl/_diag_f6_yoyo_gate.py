"""F6 — IS THE YO-YO FAMILY EVEN REACHABLE ON THE MERGE?  (the cheap CPU box F5 left open)

[[t3-turns-in-plane-and-climbs]] withdrew a claim of mine as INVERTED: I had written that "at the
merge, closure 253-453 m/s far exceeds `CLOSURE_HI = 120`, so the overshoot primitives that never
entered on the chase WOULD enter here."  Clearing one clause is not entry.  Whether a merge
`HighYoYo` ever enters was left explicitly UNMEASURED, and it is the prerequisite for any
"teach it a yo-yo" box: a reward or a demo for a maneuver whose ENTRY GATE never opens is a reward
for nothing.  [[bfm-phase-a-closures]] closed cloning because the stick library cannot FLY a yo-yo
at any threshold; this asks the prior question of whether the geometry ever ASKS for one.

The gate under test is `BFMCoach` branch (b), verbatim from `_diag_f2_poscontrol._entry_high_yoyo`:

    aspect < 90 deg   AND   rng < YOYO_RANGE (900 m)   AND   closure > CLOSURE_HI (120 m/s)

and its doctrinal sibling, branch (c) / `_entry_low_yoyo`:

    aspect < COACH_REAR_ASPECT (60 deg)   AND   rng > YOYO_RANGE_EXIT (1400 m)   AND
    closure < CLOSURE_LO (40 m/s)

PRE-REGISTERED PREDICTIONS (written before the run; read the verdict against these, not around
them).  `aspect` is the angle off the bandit's TAIL, so a head-on merge is aspect ~180 deg.

  P1  On a pure lag-pursuit ingress and on the `LeadTurn` arm, the HIGH yo-yo gate NEVER opens over
      the whole 180 s, and it fails on a DIFFERENT clause in each phase: pre-CPA `aspect >= 90` (he
      is nose-on, not overshot) while closure is huge, and post-CPA `closure <= 120` (we are
      separating or co-orbiting) once aspect finally drops.  I.e. the two clauses are never true at
      the same time and the family is unreachable BY GEOMETRY, not by a threshold.
  P2  If it opens at all it is a brief window straddling CPA, where aspect has swung under 90 but
      closure has not yet gone negative.  Reported as window count + first entry time, never as a
      fraction alone.
  P3  The LOW yo-yo gate is the one that CAN open after a conversion (bled closure, reopened range,
      rear aspect) -- i.e. the doctrine answer on this cell is "cut the corner", not "kill closure".
      F4 measured `LeadTurn` ending at aspect 60.6 deg with ata 138.4 deg, a classic overshoot in
      ANGLES with the nose nowhere near him, which is a LOW-yo-yo picture, not a HIGH-yo-yo one.

If P1 is wrong the finding is that the gate DOES open and the primitives simply never got the
chance; that would reopen the yo-yo family and is the reason this is worth 10 minutes of CPU.

SCOPE, stated so the conclusion cannot be over-read.  This measures the gate along the ingresses
this harness can fly -- lag pursuit, and a `LeadTurn` conversion.  A different ingress could open a
gate these do not.  "Unreachable" here means "unreachable from these two ingresses on merge1500 at a
+2000 ft perch", never "unreachable in principle".

Seeds: `build_env` uses `jitter=0.0`, and `_diag_true_mpc.precheck` established that `sim.seed` only
drives GUN DISPERSION -- so for any arm that fires nothing the three seeds are ONE sample and the
rows are bit-identical.  That is checked and reported rather than assumed ([[bfm-learnability]]
mis-filed a 3-seed row as 3 samples once already).
"""

from __future__ import annotations

import math
import os

import _diag_f2_poscontrol as F
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import relative

OUT = "/workspace/controllers/rl/runs/f6"
SEEDS = (11, 28, 45)
DT = M.CONTROL_DT_S
N180 = int(round(180.0 / DT))

_R = math.radians
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


# =================================================================================================
# The two gates, as PREDICATES ON A RECORDED TICK (so the same code scores every arm)
# =================================================================================================
HI_CLAUSES = (
    ("aspect<90", lambda r: r["aspect"] < _R(90.0)),
    ("rng<900", lambda r: r["rng"] < C.YOYO_RANGE),
    ("clo>120", lambda r: r["closure"] > C.CLOSURE_HI),
)
LO_CLAUSES = (
    ("aspect<60", lambda r: r["aspect"] < bfm.COACH_REAR_ASPECT),
    ("rng>1400", lambda r: r["rng"] > C.YOYO_RANGE_EXIT),
    ("clo<40", lambda r: r["closure"] < C.CLOSURE_LO),
)


def gate_report(rec: list[dict], clauses) -> dict:
    """Per-clause satisfaction, the JOINT gate, and the closest miss.

    The closest miss is the point of this function: a joint fraction of 0.000 says the gate never
    opened but not WHY, and "why" is the whole question -- a gate blocked by one clause at one
    phase and a different clause at another is unreachable by geometry, while a gate blocked by the
    same clause everywhere is a threshold argument.
    """
    n = len(rec)
    cols = {name: np.asarray([bool(f(r)) for r in rec]) for name, f in clauses}
    joint = np.ones(n, bool)
    for v in cols.values():
        joint &= v
    # windows = contiguous runs of the joint gate
    windows, i = [], 0
    while i < n:
        if joint[i]:
            j = i
            while j + 1 < n and joint[j + 1]:
                j += 1
            windows.append((rec[i]["t"], rec[j]["t"]))
            i = j + 1
        else:
            i += 1
    met = np.zeros(n, int)
    for v in cols.values():
        met += v.astype(int)
    best = int(np.argmax(met))
    # Which single clause blocks most often among the ticks where the gate is shut?
    shut = ~joint
    blame = {name: float((shut & ~v).mean()) if shut.any() else 0.0 for name, v in cols.items()}
    return dict(
        n=n,
        frac={name: float(v.mean()) for name, v in cols.items()},
        joint=float(joint.mean()),
        windows=windows,
        best_met=int(met[best]),
        best=rec[best],
        blame=blame,
        # the ONLY-clause-missing count: ticks where exactly one clause fails, and which
        near=int((met == len(cols) - 1).sum()),
    )


# =================================================================================================
# Arms
# =================================================================================================
def _never(me, rel) -> bool:
    return False


def two_stage_ctl(second_factory, second_entry, lead_turn: bool):
    """Optional `LeadTurn` first, then WATCH for `second_entry` and hand off to the second primitive
    every time it fires.  Between and after, lag pursuit -- the same ingress `forced_ctl` uses, so
    with `lead_turn=True` and a gate that never opens this arm is bit-identical to the F4
    `lead_turn` arm, and with `lead_turn=False` it is bit-identical to `lagpursuit`.  Both
    equivalences are visible in the output and are the harness's own control.

    ⚠ Records EVERY stage-2 entry, not the last one.  The first version of this probe stored a
    single `s2_i0` and overwrote it on each re-entry, so an arm that entered at t=70.9 s and again
    at t=161.9 s reported "stage2 entered t=161.9s" -- which read as "it barely entered at the end"
    when in fact the trajectory had diverged 90 s earlier.  The count and the FIRST entry are what
    make the row interpretable.
    """
    st = {
        "phase": "in" if lead_turn else "watch",
        "mv": None,
        "lt_i0": None,
        "lt_i1": None,
        "s2": [],  # [(i0, i1|None)] for every entry, in order
        "trk": None,
    }

    def _pursue(me, rel):
        if st["trk"] is None:
            st["trk"] = bfm.LagPursuitTracker()
            st["trk"].reset(me, rel)
        a4, _ = st["trk"].step(me, rel)
        return F._pad(a4)

    def fn(step, me, bd, rel):
        if st["phase"] == "in":
            if F._entry_lead_turn(me, rel):
                st["mv"] = bfm.LeadTurn()
                st["mv"].reset(me, rel)
                st["phase"] = "lt"
                st["lt_i0"] = step
            else:
                return _pursue(me, rel)
        if st["phase"] == "lt":
            a4, done = st["mv"].step(me, rel)
            if done:
                st["phase"] = "watch"
                st["lt_i1"] = step
                st["trk"] = None
            return F._pad(a4)
        if st["phase"] == "watch":
            if second_entry(me, rel):
                st["mv"] = second_factory()
                st["mv"].reset(me, rel)
                st["phase"] = "s2"
                st["s2"].append([step, None])
            else:
                return _pursue(me, rel)
        if st["phase"] == "s2":
            a4, done = st["mv"].step(me, rel)
            if done:
                st["phase"] = "watch"  # re-armed: it may enter again
                st["s2"][-1][1] = step
                st["trk"] = None
            return F._pad(a4)
        return _pursue(me, rel)

    return fn, st


ARMS = (
    # The two pure ingresses, as the controls the two-stage arms are read against.
    dict(name="lagpursuit", kind="forced", factory=lambda: bfm.HighYoYo(), entry=_never),
    dict(name="leadturn", kind="forced", factory=lambda: bfm.LeadTurn(), entry=F._entry_lead_turn),
    # From a LEAD TURN. P1 says the HIGH gate never opens here.
    dict(name="leadturn>highyoyo", kind="two", lt=True, factory=bfm.HighYoYo, entry=F._entry_high_yoyo),
    dict(name="leadturn>lowyoyo", kind="two", lt=True, factory=bfm.LowYoYo, entry=F._entry_low_yoyo),
    # From LAG PURSUIT — added AFTER the first run measured the HIGH gate opening twice on this
    # ingress (t=23.4 s and 34.5 s), refuting P1's "never". These are the arms the result demands:
    # if the gate opens, the question stops being "is it reachable" and becomes "what does it buy".
    dict(name="lagpursuit>highyoyo", kind="two", lt=False, factory=bfm.HighYoYo, entry=F._entry_high_yoyo),
    dict(name="lagpursuit>lowyoyo", kind="two", lt=False, factory=bfm.LowYoYo, entry=F._entry_low_yoyo),
)


def build_ctl(arm: dict):
    if arm["kind"] == "forced":
        return F.forced_ctl(arm["factory"], arm["entry"])
    return two_stage_ctl(arm["factory"], arm["entry"], arm["lt"])


def fly(arm: dict, seed: int) -> dict:
    cell = F.merge_cell(2000.0)
    env, base = M.build_env(cell, seed, N180 * DT)
    ctl, st = build_ctl(arm)
    env.reset()
    rec: list[dict] = []
    event = None
    t = 0.0
    cpa_i, cpa_rng = 0, float("inf")
    for step in range(N180):
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        a = np.asarray(ctl(step, me, bd, rel), np.float32).copy()
        a[4] = M._trigger(rel)
        rec.append(
            dict(
                t=t,
                rng=float(rel.rng),
                aspect=float(rel.aspect),
                ata=float(rel.ata),
                closure=float(rel.closure),
                alt=float(me.pos[2]),
                v=float(me.airspeed),
            )
        )
        if rel.rng < cpa_rng:
            cpa_rng, cpa_i = float(rel.rng), step
        _obs, _r, term, trunc, info = env.step(a)
        t += DT
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    out = dict(
        rec=rec,
        st=st,
        event=event,
        cpa_i=cpa_i,
        cpa_rng=cpa_rng,
        dealt=C.HP_MAX - float(bd.health),
        rounds=int(round(C.AMMO_MAX - float(me.ammo))),
        dalt=float(rec[-1]["alt"] - rec[0]["alt"]),
        aspect_end=math.degrees(rec[-1]["aspect"]),
        ata_end=math.degrees(rec[-1]["ata"]),
    )
    env.close()
    return out


# =================================================================================================
# Reporting
# =================================================================================================
def _fmt_windows(w) -> str:
    if not w:
        return "NEVER"
    return f"{len(w)}x " + ", ".join(f"[{a:.1f},{b:.1f}]s" for a, b in w[:4])


def _fmt_s2(s2: list, rec: list[dict]) -> str:
    """Stage-2 entries as COUNT + FIRST + total time held.

    The count and the first entry are the interpretable numbers: an arm whose trajectory diverges
    from its control at t=70.9 s and re-enters at t=161.9 s is a different story from one that only
    entered once at the very end, and the earlier version of this printer could not tell them apart.
    """
    if not s2:
        return "NEVER entered"
    held = sum(((i1 if i1 is not None else len(rec) - 1) - i0) for i0, i1 in s2) * DT
    times = ", ".join(f"{rec[i0]['t']:.1f}s" for i0, i1 in s2[:5])
    tail = "" if len(s2) <= 5 else f", +{len(s2) - 5} more"
    return f"{len(s2)}x, first t={rec[s2[0][0]]['t']:.1f}s, held {held:.1f}s total (at {times}{tail})"


def report(name: str, r: dict) -> None:
    rec, ci = r["rec"], r["cpa_i"]
    phases = (("whole", rec), ("preCPA", rec[: ci + 1]), ("postCPA", rec[ci:]))
    log(f"\n  {'-' * 108}")
    log(
        f"  ARM {name}   event {r['event']}   CPA {r['cpa_rng']:.0f} m @ t={rec[ci]['t']:.1f}s   "
        f"dalt {r['dalt']:+.0f} m   rounds {r['rounds']}   dealt {r['dealt']:.1f}   "
        f"end aspect {r['aspect_end']:.0f} / ata {r['ata_end']:.0f} deg"
    )
    st = r["st"]
    if "s2" in st:
        lt = "not used" if st.get("lt_i0") is None else f"t={rec[st['lt_i0']]['t']:.1f}s"
        log(f"    stage1 LeadTurn {lt} | stage2 {_fmt_s2(st['s2'], rec)}")
    for gname, clauses in (("HIGH", HI_CLAUSES), ("LOW", LO_CLAUSES)):
        log(f"    {gname} yo-yo gate")
        for pname, sub in phases:
            if not sub:
                continue
            g = gate_report(sub, clauses)
            fr = "  ".join(f"{k} {v:.3f}" for k, v in g["frac"].items())
            log(
                f"      {pname:8s} n{g['n']:5d} | {fr} | JOINT {g['joint']:.3f} "
                f"{_fmt_windows(g['windows'])} | 2-of-3 ticks {g['near']}"
            )
            if pname == "whole":
                b = g["best"]
                blame = "  ".join(f"{k} {v:.2f}" for k, v in g["blame"].items())
                log(
                    f"               best tick {g['best_met']}/3 at t={b['t']:.1f}s: "
                    f"aspect {math.degrees(b['aspect']):.0f} deg  rng {b['rng']:.0f} m  "
                    f"closure {b['closure']:+.0f} m/s   | blocked-by {blame}"
                )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    log("#" * 110)
    log("#  F6 — is the yo-yo ENTRY GATE ever open on the merge?  (see docstring for P1/P2/P3)")
    log("#" * 110)
    log(
        f"\n  cell merge1500 + perch 2000 ft   horizon 180 s ({N180} steps, dt {DT * 1000:.1f} ms)"
        f"   seeds {SEEDS}   jitter 0"
    )
    log(
        f"  HIGH gate: aspect < 90 deg AND rng < {C.YOYO_RANGE:.0f} m AND closure > "
        f"{C.CLOSURE_HI:.0f} m/s     (BFMCoach branch b)"
    )
    log(
        f"  LOW  gate: aspect < {math.degrees(bfm.COACH_REAR_ASPECT):.0f} deg AND rng > "
        f"{C.YOYO_RANGE_EXIT:.0f} m AND closure < {C.CLOSURE_LO:.0f} m/s   (BFMCoach branch c)"
    )
    log("  aspect = angle off the BANDIT'S TAIL, so a head-on merge is ~180 deg.")

    summary: list[tuple[str, dict]] = []
    for arm in ARMS:
        per_seed = [fly(arm, s) for s in SEEDS]
        # Seed independence, MEASURED (jitter=0 + no rounds fired => identical traces).
        same = all(
            len(p["rec"]) == len(per_seed[0]["rec"]) and abs(p["dalt"] - per_seed[0]["dalt"]) < 1e-6
            for p in per_seed[1:]
        )
        fired = any(p["rounds"] > 0 for p in per_seed)
        report(arm["name"], per_seed[0])
        log(
            f"    seeds: {'IDENTICAL (=> n=1, not 3)' if same else 'DIFFER'}"
            f"   rounds/seed {[p['rounds'] for p in per_seed]}"
            f"   dealt/seed {[round(p['dealt'], 1) for p in per_seed]}"
            f"{'' if fired else '   (nothing fired => seed only drives dispersion)'}"
        )
        summary.append((arm["name"], per_seed[0]))

    log("\n" + "=" * 110)
    log("VERDICT — the joint gate over the whole episode, and whether P1 survives")
    log("=" * 110)
    for name, r in summary:
        hi = gate_report(r["rec"], HI_CLAUSES)
        lo = gate_report(r["rec"], LO_CLAUSES)
        st = r["st"]
        s2 = "n/a" if "s2" not in st else (f"{len(st['s2'])}x" if st["s2"] else "no")
        log(
            f"  {name:20s} HIGH joint {hi['joint']:.3f} {_fmt_windows(hi['windows']):28s} | "
            f"LOW joint {lo['joint']:.3f} {_fmt_windows(lo['windows']):28s} | stage2 {s2:4s} | "
            f"dealt {r['dealt']:.1f}"
        )
    log("")
    log("  P1 predicted HIGH joint = 0.000 on `lagpursuit` and `leadturn`, blocked by ASPECT")
    log("     pre-CPA and by CLOSURE post-CPA (the two clauses never co-true).")
    log("  P3 predicted the LOW gate is the one that opens after a conversion.")
    log("  Read the blocked-by rows above: they are what distinguishes 'unreachable by geometry'")
    log("  from 'one threshold too tight'.")

    path = os.path.join(OUT, "f6_yoyo_gate.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
