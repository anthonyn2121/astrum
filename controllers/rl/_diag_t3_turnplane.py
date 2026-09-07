"""F4 — DOES THE T3 BANK INTO AN IN-PLANE TURN AFTER THE PASS, AND IS OUR DIVE FORCED?

Two questions, and they are different questions.

Q_A (the opponent's turn plane).  Read the code first: after a neutral merge the T3 can only be in
`_break` (DEFENSIVE) or `lead_pursuit` (NEUTRAL/OFFENSIVE), and NEITHER law has an altitude term
or an out-of-plane term.  `_break` commands a FIXED `BREAK_BANK_RAD = 85 deg` regardless of where
the LOS is; `lead_pursuit` rolls the lift vector ONTO the lead point and pulls along it.  So on the
code the T3 has no mechanism to fly out of plane on purpose.  What the code does NOT tell us is
whether the resulting flight path is LEVEL: a 85 deg bank needs nz = 1/cos(85) = 11.5 g to hold
altitude, which is past the F-16's 9 g limiter, so a max-bank break MUST descend unless the entry
was nose-high.  Measured here: the T3's own dalt, dEs, gamma, bank, and the split of its turn rate
into HORIZONTAL (d course/dt) and VERTICAL (d gamma/dt) components, post-CPA.

Q_B (is OUR altitude spend forced?).  Every arm this program has recorded on the merge dives:
c95 -14,807 m, c96 -4,670 m, d3b -1,293 m, d3b_perch0 -2,422 m.  The standing claim is "the only
turning currency is altitude".  That claim is UNTESTED against an arm that simply refuses to
descend, so two are added here: `levelturn80` and `levelturn60` bank hard toward the bandit but run
a gamma-hold on the elevator instead of a max-g pull.  If those hold altitude AND stay in the fight
(aspect / ata / range comparable to the diving arms), the dive is ours, not the geometry's.  If they
hold altitude and fall out of the fight, the dive is the price of turn rate and the standing claim
survives -- now with a control behind it.

Everything is flown at the SAME 24 s horizon and the SAME `merge1500` cell as F2/D3b so the rows are
directly comparable to the recorded ones, with one extra 40 s `straight` arm to see the T3's steady
state well past the pass.  `roll_once` is not reused: it records the LEARNER's trace only, and this
question is about the BANDIT.  The trigger rule is `M._trigger` on both, exactly as in F2.
"""

from __future__ import annotations

import math
import os

import _diag_f2_poscontrol as F
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import bfm_detect as B
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import relative, specific_energy

OUT = "/workspace/controllers/rl/runs/f4"
SEEDS = (11, 28, 45)
DT = M.CONTROL_DT_S
N24 = int(round(24.0 / DT))
N40 = int(round(40.0 / DT))

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _gamma(s) -> float:
    v = np.asarray(s.vel, float)
    sp = float(np.linalg.norm(v))
    return math.asin(_clamp(float(v[2]) / sp, -1.0, 1.0)) if sp > 1e-6 else 0.0


def _course(s) -> float:
    v = np.asarray(s.vel, float)
    return math.atan2(float(v[0]), float(v[1]))  # same expression as opponent._course


# =================================================================================================
# Learner arms
# =================================================================================================
def straight_ctl():
    """Null provocation: wings level, gamma-hold at 0, cruise throttle.  If the T3 descends against
    THIS, the descent is intrinsic to the T3 and not a response to anything we do."""

    def fn(step, me, bd, rel):
        ail = bfm._roll_to(me.phi, 0.0)
        elev = _clamp(4.0 * _gamma(me) + 0.5 * float(me.q), -1.0, 1.0)
        return np.array([ail, elev, 0.0, 0.75, -1.0], np.float32)

    return fn, {"phase": "straight", "i0": 0, "i1": None}


def level_turn_ctl(bank_deg: float):
    """Bank hard TOWARD the bandit and hold gamma = 0 on the elevator -- a turn that refuses to
    trade altitude.  Note what this does and does not concede: the elevator is a gamma-hold, so if
    the jet cannot make the bank's level-flight load (1/cos(80) = 5.76 g, 1/cos(60) = 2.00 g) the
    loop saturates at full pull and it descends anyway.  The arm is therefore "spend as little
    altitude as this bank allows", not "cannot descend" -- and the achieved bank/nz are reported so
    that distinction is readable rather than assumed.
    """
    tgt = math.radians(bank_deg)

    def fn(step, me, bd, rel):
        sgn = 1.0 if rel.ata_az >= 0.0 else -1.0
        ail = bfm._roll_to(me.phi, sgn * tgt)
        elev = _clamp(4.0 * _gamma(me) + 0.5 * float(me.q), -1.0, 1.0)
        return np.array([ail, elev, 0.0, 1.0, -1.0], np.float32)

    return fn, {"phase": f"level{bank_deg:.0f}", "i0": 0, "i1": None}


def arms(d3b: np.ndarray) -> list[dict]:
    mc = F.merge_cell(2000.0)
    return [
        dict(name="straight@2000", cell=mc, n=N24, kind="straight"),
        dict(name="straight@2000_40s", cell=mc, n=N40, kind="straight"),
        dict(name="levelturn80@2000", cell=mc, n=N24, kind="level", bank=80.0),
        dict(name="levelturn60@2000", cell=mc, n=N24, kind="level", bank=60.0),
        dict(name="lead_turn@2000", cell=mc, n=N24, kind="forced"),
        dict(name="d3b@2000", cell=mc, n=N24, kind="genome", g=d3b),
        dict(name="d3b@0", cell=F.merge_cell(0.0), n=N24, kind="genome", g=d3b),
    ]


def build_ctl(arm: dict):
    k = arm["kind"]
    if k == "straight":
        return straight_ctl()
    if k == "level":
        return level_turn_ctl(arm["bank"])
    if k == "forced":
        return F.forced_ctl(lambda: bfm.LeadTurn(), F._entry_lead_turn)
    return F.genome_ctl(arm["g"], arm["n"])


# =================================================================================================
# The roll: BOTH jets recorded, plus the T3's FSM state
# =================================================================================================
def fly(arm: dict, seed: int) -> dict:
    env, base = M.build_env(arm["cell"], seed, arm["n"] * DT)
    ctl, _st = build_ctl(arm)
    obs, _ = env.reset()
    rec: list[dict] = []
    event = None
    t = 0.0
    for step in range(arm["n"]):
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        rel_b = relative(bd, me)  # the bandit's own picture of us
        a = np.asarray(ctl(step, me, bd, rel), np.float32).copy()
        a[4] = M._trigger(rel)
        chi_me, val_me = B.out_of_plane_angle(me, bd)
        chi_bd, val_bd = B.out_of_plane_angle(bd, me)
        rec.append(
            dict(
                t=t,
                state=base.opponent.state,
                rng=float(rel.rng),
                ata=float(rel.ata),
                aspect=float(rel.aspect),
                closure=float(rel.closure),
                me_alt=float(me.pos[2]),
                me_gam=_gamma(me),
                me_crs=_course(me),
                me_phi=float(me.phi),
                me_nz=float(me.nz),
                me_v=float(me.airspeed),
                me_es=specific_energy(me),
                me_chi=chi_me,
                me_chi_v=bool(val_me),
                bd_alt=float(bd.pos[2]),
                bd_gam=_gamma(bd),
                bd_crs=_course(bd),
                bd_phi=float(bd.phi),
                bd_nz=float(bd.nz),
                bd_v=float(bd.airspeed),
                bd_es=specific_energy(bd),
                bd_chi=chi_bd,
                bd_chi_v=bool(val_bd),
                bd_ata=float(rel_b.ata),
            )
        )
        obs, _r, term, trunc, info = env.step(a)
        t += DT
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    out = dict(
        rec=rec,
        event=event,
        kill=1 if event == "kill" else 0,
        dead=event in ("hard_deck", "killed", "trade"),
        hp_dealt=C.HP_MAX - float(bd.health),
        hp_taken=C.HP_MAX - float(me.health),
        rounds=int(round(C.AMMO_MAX - float(me.ammo))),
    )
    env.close()
    return out


# =================================================================================================
# Reduction
# =================================================================================================
def _rates(rec: list[dict], key_crs: str, key_gam: str) -> tuple[np.ndarray, np.ndarray]:
    """deg/s magnitudes of horizontal (course) and vertical (gamma) turn rate."""
    crs = np.asarray([r[key_crs] for r in rec], float)
    gam = np.asarray([r[key_gam] for r in rec], float)
    dcrs = np.degrees(np.abs(np.diff(np.unwrap(crs)))) / DT
    dgam = np.degrees(np.abs(np.diff(gam))) / DT
    return dcrs, dgam


def window(rec: list[dict], lo: int, hi: int) -> dict:
    """All the per-window statistics, for one jet pair, over rec[lo:hi]."""
    w = rec[lo:hi]
    if len(w) < 3:
        return dict(n=len(w))
    dcrs_b, dgam_b = _rates(w, "bd_crs", "bd_gam")
    dcrs_m, dgam_m = _rates(w, "me_crs", "me_gam")
    states = [r["state"] for r in w]
    frac = {s: states.count(s) / len(states) for s in sorted(set(states))}
    bchi = np.degrees(np.abs([r["bd_chi"] for r in w if r["bd_chi_v"]]))
    mchi = np.degrees(np.abs([r["me_chi"] for r in w if r["me_chi_v"]]))
    return dict(
        n=len(w),
        dur=w[-1]["t"] - w[0]["t"],
        frac=frac,
        # --- the bandit ---
        bd_dalt=w[-1]["bd_alt"] - w[0]["bd_alt"],
        bd_des=w[-1]["bd_es"] - w[0]["bd_es"],
        bd_phi=float(np.median(np.degrees(np.abs([r["bd_phi"] for r in w])))),
        bd_phi95=float(np.percentile(np.degrees(np.abs([r["bd_phi"] for r in w])), 95)),
        bd_gam=float(np.median(np.degrees([r["bd_gam"] for r in w]))),
        bd_gam_lo=float(np.min(np.degrees([r["bd_gam"] for r in w]))),
        bd_gam_hi=float(np.max(np.degrees([r["bd_gam"] for r in w]))),
        bd_nz=float(np.median([r["bd_nz"] for r in w])),
        bd_dcrs=float(np.median(dcrs_b)),
        bd_dgam=float(np.median(dgam_b)),
        bd_vshare=float(np.median(dgam_b) / max(np.median(dcrs_b) + np.median(dgam_b), 1e-9)),
        bd_chi=float(np.median(bchi)) if len(bchi) else float("nan"),
        bd_chi_vfrac=len(bchi) / len(w),
        # --- us ---
        me_dalt=w[-1]["me_alt"] - w[0]["me_alt"],
        me_des=w[-1]["me_es"] - w[0]["me_es"],
        me_phi=float(np.median(np.degrees(np.abs([r["me_phi"] for r in w])))),
        me_gam=float(np.median(np.degrees([r["me_gam"] for r in w]))),
        me_nz=float(np.median([r["me_nz"] for r in w])),
        me_dcrs=float(np.median(dcrs_m)),
        me_dgam=float(np.median(dgam_m)),
        me_chi=float(np.median(mchi)) if len(mchi) else float("nan"),
        # --- the fight ---
        rng_min=float(np.min([r["rng"] for r in w])),
        ata=float(np.median(np.degrees([r["ata"] for r in w]))),
        ata_end=float(np.degrees(w[-1]["ata"])),
        aspect_end=float(np.degrees(w[-1]["aspect"])),
    )


def summarise(arm: dict, runs: list[dict]) -> None:
    log("")
    log("-" * 108)
    log(f"ARM  {arm['name']}    ({len(runs)} seeds, {arm['n'] * DT:.0f} s)")
    log("-" * 108)
    for seed, r in zip(SEEDS, runs, strict=False):
        rec = r["rec"]
        icpa = int(np.argmin([x["rng"] for x in rec]))
        pre = window(rec, 0, icpa + 1)
        post = window(rec, icpa, len(rec))
        log(
            f"  seed {seed:>3}  ev={str(r['event']):<12} kill={r['kill']} hp={r['hp_dealt']:5.1f} "
            f"rnds={r['rounds']:>3}   CPA t={rec[icpa]['t']:5.2f}s rng={rec[icpa]['rng']:7.1f}m"
        )
        for tag, w in (("PRE ", pre), ("POST", post)):
            if w.get("n", 0) < 3:
                log(f"      {tag}  (too short: n={w.get('n')})")
                continue
            fr = " ".join(f"{k}={v:.0%}" for k, v in w["frac"].items())
            log(
                f"      {tag} {w['dur']:5.2f}s  T3: dalt{w['bd_dalt']:+8.1f}m dEs{w['bd_des']:+8.1f}m "
                f"|phi|{w['bd_phi']:5.1f}/{w['bd_phi95']:5.1f}deg gam{w['bd_gam']:+6.1f}"
                f"[{w['bd_gam_lo']:+6.1f},{w['bd_gam_hi']:+6.1f}] nz{w['bd_nz']:4.2f}g"
            )
            log(
                f"           {'':5}   turn: horiz{w['bd_dcrs']:6.2f}deg/s vert{w['bd_dgam']:6.2f}deg/s "
                f"=> VERTICAL SHARE {w['bd_vshare']:5.1%}   |chi_T3|{w['bd_chi']:6.1f}deg "
                f"(valid {w['bd_chi_vfrac']:.0%})   [{fr}]"
            )
            log(
                f"           {'':5}   ME: dalt{w['me_dalt']:+9.1f}m dEs{w['me_des']:+8.1f}m "
                f"|phi|{w['me_phi']:5.1f}deg gam{w['me_gam']:+6.1f}deg nz{w['me_nz']:4.2f}g "
                f"horiz{w['me_dcrs']:6.2f} vert{w['me_dgam']:6.2f}  |chi|{w['me_chi']:6.1f}deg"
            )
            log(
                f"           {'':5}   fight: rngMn{w['rng_min']:7.1f}m ata_p50{w['ata']:6.1f}deg "
                f"ata_end{w['ata_end']:6.1f}deg aspect_end{w['aspect_end']:6.1f}deg"
            )


def trace(arm_name: str, r: dict, every_s: float = 1.0) -> None:
    rec = r["rec"]
    icpa = int(np.argmin([x["rng"] for x in rec]))
    log("")
    log("=" * 108)
    log(
        f"PER-SECOND TRACE — {arm_name}, seed {SEEDS[0]}   (CPA at t={rec[icpa]['t']:.2f}s, marked *)"
    )
    log("=" * 108)
    log(
        "     t  T3state      rng     ata |  T3: alt   gam   phi    nz    V  |  ME: alt   gam"
        "   phi    nz    V  | chiT3"
    )
    k = max(1, int(round(every_s / DT)))
    for i in range(0, len(rec), k):
        x = rec[i]
        mark = "*" if abs(i - icpa) < k / 2 else " "
        log(
            f"  {x['t']:5.2f}{mark} {x['state']:<11} {x['rng']:7.1f} {math.degrees(x['ata']):6.1f} | "
            f"{x['bd_alt']:7.1f} {math.degrees(x['bd_gam']):+5.1f} {math.degrees(x['bd_phi']):+6.1f} "
            f"{x['bd_nz']:5.2f} {x['bd_v']:5.1f} | "
            f"{x['me_alt']:7.1f} {math.degrees(x['me_gam']):+5.1f} {math.degrees(x['me_phi']):+6.1f} "
            f"{x['me_nz']:5.2f} {x['me_v']:5.1f} | "
            f"{math.degrees(x['bd_chi']):+6.1f}{'' if x['bd_chi_v'] else '?'}"
        )


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    log("#" * 108)
    log("#  F4 — the T3's post-pass turn plane, and whether OUR altitude spend is forced")
    log("#" * 108)
    log("")
    log("CODE FACTS (read, not measured) — the T3 has no out-of-plane and no altitude mechanism:")
    log(f"  _break        : bank commanded to a FIXED {math.degrees(C.BREAK_BANK_RAD):.0f} deg")
    log("                  (opponent.py:456) with NO LOS-plane term; pull = corner-limited max.")
    log("  lead_pursuit  : rolls the lift vector ONTO the lead point (opponent.py:338-342) => its")
    log("                  turn plane CONTAINS us by construction; chi_T3 should sit near 0.")
    log("  neither law has an altitude-hold term (contrast InnerLoop, opponent.py:98-107).")
    lvl = [(60.0, 1 / math.cos(math.radians(60.0))), (80.0, 1 / math.cos(math.radians(80.0)))]
    lvl.append((math.degrees(C.BREAK_BANK_RAD), 1 / math.cos(C.BREAK_BANK_RAD)))
    log(
        "  level-flight load required by bank: " + ", ".join(f"{b:.0f}deg={g:.2f}g" for b, g in lvl)
    )
    log("  => the 85 deg break needs 11.47 g, past the 9 g limiter, so a max-bank break CANNOT")
    log(
        "     hold altitude.  Whether it descends in PRACTICE depends on its entry gamma: measured."
    )

    d3b = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"]
    keep: dict[str, dict] = {}
    for arm in arms(d3b):
        runs = [fly(arm, s) for s in SEEDS]
        summarise(arm, runs)
        keep[arm["name"]] = runs[0]

    for nm in ("straight@2000_40s", "levelturn80@2000", "d3b@2000"):
        if nm in keep:
            trace(nm, keep[nm])

    with open(os.path.join(OUT, "f4_turnplane.txt"), "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    log("")
    log(f"[written] {os.path.join(OUT, 'f4_turnplane.txt')}")


if __name__ == "__main__":
    main()
