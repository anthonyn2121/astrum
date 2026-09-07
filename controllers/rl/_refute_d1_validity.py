"""INDEPENDENT REFUTATION ATTEMPT on the 'detector-validity caveat' attached to the constructed D1.

The claim under test (three parts):
  (1) clause (f) d_aspect = -150.5 deg is the merge PASS (aspect 175.8 @ 890 m -> 25.3 @ 349 m);
  (2) clause (d) is FIRST satisfied at step 190 by the chi = 0.0 sentinel, chi_valid = False;
  (3) "the event survives a validity-gated clause (d) as well: the first chi_valid |chi| < 30 inside
      the return window is step 200, t 8.333, d_aspect -146.8 deg".

Part (3) is the load-bearing rescue and `_diag_d1_exists.d1_audit` never tested it: it checks only
that a valid |chi| < 30 EXISTS in the window and reports d_aspect there. It never re-evaluates
clause (e), which `bfm_detect.detect_displacement_rolls` evaluates over `range(i, exit_idx + 1)`
(bfm_detect.py:361) -- i.e. over the window that MOVING the exit lengthens. So this script re-runs
the whole six-clause predicate at the alternate exit, and also runs a verbatim copy of the detector
with validity gating added to (d), which is the direct measurement.

Everything is recomputed from a fresh re-flight; no number is taken from a prior report.
"""

from __future__ import annotations

import math

import _diag_d1_contradiction as K
import _diag_d1_exists as X
import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []

# the 18-dim vector as PRINTED in runs/d3b/d1_exists_report.txt (precision=6), for cross-check
G_REPORTED = np.array(
    [
        1.0, 1.0, 0.2, 0.582792, 1.0, 0.2, 1.0, 1.0, 0.2,
        0.872769, 1.0, 0.2, -0.007301, 1.0, 0.2, 0.312102, 1.0, 0.2,
    ],
    float,
)  # fmt: skip


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def d1_validity_gated(tr: list[B.Sample]) -> list[B.Event]:
    """`detect_displacement_rolls` VERBATIM, with one change: (d) requires a VALID chi.

    The only edited line is marked. Everything else, including the clause ORDER and the
    `range(i, exit_idx + 1)` span of clause (e), is copied from bfm_detect.py:316-376.
    """
    if len(tr) < 3:
        return []
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    events: list[B.Event] = []
    for i, j in B._runs(hold):
        if tr[j].t - tr[i].t < B.CHI_MIN_HOLD_S:
            continue
        if not any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        ):
            continue
        if tr[i].closure <= 0.0:
            continue
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            # >>> THE ONLY EDIT: a chi = 0.0 sentinel is not a roll back in <<<
            if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            continue
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
        if len(sides) != 1:
            continue
        ev = B._event_from(
            tr,
            "displacement_roll",
            i,
            exit_idx,
            chi_peak_deg=max(abs(math.degrees(tr[k].chi)) for k in range(i, j + 1)),
            nz_peak=max(tr[k].nz for k in range(i, j + 1)),
            hold_s=tr[j].t - tr[i].t,
        )
        if ev.d_aspect_deg >= -B.ASPECT_GAIN_DEG:
            continue
        events.append(ev)
    return events


def clause_walk(tr: list[B.Sample], i: int, j: int, exit_idx: int, tag: str) -> bool:
    """Evaluate all six clauses explicitly at a given (hold, exit) and report each."""
    ok = True
    log(f"\n    --- SIX-CLAUSE WALK, {tag}: hold {i}-{j}, exit {exit_idx} ---")
    dur = tr[j].t - tr[i].t
    c_dur = dur >= B.CHI_MIN_HOLD_S
    log(f"      hold duration {dur:.3f} s >= {B.CHI_MIN_HOLD_S} s          : {c_dur}")
    lb = [
        k
        for k in range(i - 1, -1, -1)
        if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        and abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
    ]
    log(
        f"      (a) in-plane |chi|<{B.CHI_INPLANE_DEG:.0f} in lookback     : {bool(lb)}  "
        f"({len(lb)} ticks, nearest step {max(lb) if lb else None})"
    )
    nz_min = min(tr[k].nz for k in range(i, j + 1))
    log(
        f"      (b) nz > {B.NZ_LOADED} on every hold tick (min {nz_min:.2f} g): {nz_min > B.NZ_LOADED}"
    )
    log(f"      (c) closure at entry {tr[i].closure:+.0f} m/s > 0        : {tr[i].closure > 0.0}")
    c_d = tr[exit_idx].chi_valid and abs(tr[exit_idx].chi) < math.radians(B.CHI_RETURN_DEG)
    log(
        f"      (d) |chi| at exit {abs(math.degrees(tr[exit_idx].chi)):.1f} deg, valid "
        f"{tr[exit_idx].chi_valid}     : raw {abs(math.degrees(tr[exit_idx].chi)) < B.CHI_RETURN_DEG}, "
        f"validity-gated {c_d}"
    )
    sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
    flip = None
    s0 = 1 if tr[i].ata_az >= 0.0 else -1
    for k in range(i, exit_idx + 1):
        if (1 if tr[k].ata_az >= 0.0 else -1) != s0:
            flip = k
            break
    log(
        f"      (e) sign(ata_az) constant over {i}..{exit_idx}   : {len(sides) == 1}  sides={sides}"
        + (
            f"  FIRST FLIP at step {flip} t {tr[flip].t:.3f} "
            f"ata_az {math.degrees(tr[flip - 1].ata_az):+.2f} -> "
            f"{math.degrees(tr[flip].ata_az):+.2f} deg"
            if flip is not None
            else ""
        )
    )
    d_asp = math.degrees(tr[exit_idx].aspect - tr[i].aspect)
    log(
        f"      (f) d_aspect {d_asp:+.1f} deg <= -{B.ASPECT_GAIN_DEG:.0f}          : "
        f"{d_asp < -B.ASPECT_GAIN_DEG}"
    )
    ok = bool(lb) and c_dur and nz_min > B.NZ_LOADED and tr[i].closure > 0.0 and c_d
    ok = ok and len(sides) == 1 and d_asp < -B.ASPECT_GAIN_DEG
    log(f"      => EVENT SURVIVES ALL SIX: {ok}")
    return ok


def unwrapped_travel(tr, i: int, j: int) -> tuple[float, float, float, float]:
    """phi_lift / phi_los about v_hat, unwrapped, over [i, j]. Returns total |travel| of each."""
    from astrum_dogfight.geometry import body_axes_enu

    pl, pls = [], []
    for k in range(i, j + 1):
        me, bd = _ME[k], _BD[k]
        v_hat = B._unit(np.asarray(me.vel, float))
        up = np.array([0.0, 0.0, 1.0])
        u_ref = B._unit(up - float(np.dot(up, v_hat)) * v_hat)
        r_ref = np.cross(v_hat, u_ref)
        _, _, down = body_axes_enu(me.phi, me.theta, me.psi)
        lift = -down
        lp = lift - float(np.dot(lift, v_hat)) * v_hat
        los = B._unit(np.asarray(bd.pos, float) - np.asarray(me.pos, float))
        lsp = los - float(np.dot(los, v_hat)) * v_hat
        pl.append(math.atan2(float(lp @ r_ref), float(lp @ u_ref)))
        pls.append(math.atan2(float(lsp @ r_ref), float(lsp @ u_ref)))
    a = np.degrees(np.unwrap(np.asarray(pl)))
    b = np.degrees(np.unwrap(np.asarray(pls)))
    return a[0], a[-1], b[0], b[-1]


_ME: dict[int, object] = {}
_BD: dict[int, object] = {}


def fly(g: np.ndarray, seed: int, capture: bool = False):
    """Re-fly, optionally capturing the raw AState pair per step for the phi_lift/phi_los split."""
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(E.CELL, seed, E.N_STEPS * E.DT)
    if capture:
        _ME.clear()
        _BD.clear()

    def ctl(step, me, bd, rel):
        if capture:
            import copy

            _ME[step] = copy.deepcopy(me)
            _BD[step] = copy.deepcopy(bd)
        return gm.act(g, step, E.N_STEPS, me)

    r = M.roll_once(env, base, ctl, E.N_STEPS, keep=True)
    env.close()
    return r


def main() -> None:
    log("#" * 100)
    log("#  REFUTATION ATTEMPT: does the constructed D1 event survive a validity-gated clause (d)?")
    log("#" * 100)

    log("\nR0  REBUILD THE GENOME (same code path as _diag_d1_exists, not the printed digits)")
    _r_ref, _rows_ref, phases = K.chain(4.0)
    g = X.encode(phases, 1.0, 0.60)
    log("    reconstructed: " + np.array2string(g, precision=6, separator=", ", max_line_width=200))
    log(f"    max|reconstructed - reported(printed digits)| = {np.abs(g - G_REPORTED).max():.3e}")

    log("\nR1  FLY ALL EIGHT SEARCH SEEDS, DETECTOR VERDICT PER SEED")
    log(
        f"    {'seed':>5} {'steps':>6} {'event':>10} {'kill':>4} {'hp':>5} {'dalt':>7} "
        f"{'D1':>3} {'D1gated':>8} {'maxdA':>10} {'maxdS':>10}"
    )
    ref_a = ref_s = None
    tr62 = None
    pooled = pooled_g = 0
    for sd in (11, 28, 45, 62, 79, 96, 113, 130):
        r = fly(g, sd, capture=False)
        a = np.asarray(r.actions, float)
        s = np.asarray(r.states, float)
        if ref_a is None:
            ref_a, ref_s = a, s
        n = min(len(a), len(ref_a))
        da = float(np.abs(a[:n] - ref_a[:n]).max())
        ds = float(np.abs(s[:n] - ref_s[:n]).max())
        n1 = len(B.detect_displacement_rolls(r.trace))
        ng = len(d1_validity_gated(r.trace))
        pooled += n1
        pooled_g += ng
        log(
            f"    {sd:>5} {r.steps:>6} {str(r.event):>10} {r.kill:>4} {r.hp_dealt:>5.1f} "
            f"{r.dalt_m:>+7.0f} {n1:>3} {ng:>8} {da:>10.3e} {ds:>10.3e}"
        )
        if sd == 62:
            tr62 = r.trace
            r62 = r
    log(
        f"    POOLED over 8 seeds: D1 (as shipped) = {pooled}, D1 (validity-gated (d)) = {pooled_g}"
    )
    log(
        "    NOTE the 8 seeds are BIT-IDENTICAL trajectories (max|dActions| and max|dStates| "
        "columns above), so the pooled 8 is ONE trajectory counted eight times, not 8 measurements."
    )

    log("\nR1b RE-FLY SEED 62 WITH ATTITUDE CAPTURE (deterministic; must match the loop exactly)")
    r62c = fly(g, 62, capture=True)
    a62 = np.asarray(r62c.actions, float)
    log(
        f"    steps {r62c.steps}  max|dActions vs the loop's seed-62 flight| = "
        f"{np.abs(a62 - np.asarray(r62.actions, float)).max():.3e}  "
        f"D1 {len(B.detect_displacement_rolls(r62c.trace))}"
    )
    tr62 = r62c.trace

    tr = tr62
    log("\nR2  THE SHIPPED EVENT ON SEED 62, RECOMPUTED")
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    log(f"    hold mask true on {sum(hold)}/{len(tr)} ticks; runs:")
    for a, b in B._runs(hold):
        log(f"      run {a}-{b}  t {tr[a].t:.3f}-{tr[b].t:.3f}  dur {tr[b].t - tr[a].t:.3f} s")
    evs = B.detect_displacement_rolls(tr)
    log(f"    detect_displacement_rolls -> {len(evs)} event(s)")
    for ev in evs:
        i, xi = ev.i0, ev.i1
        hold_end = next(b for a, b in B._runs(hold) if a == i)
        log(
            f"      i0={i} exit={xi} hold_end={hold_end}  hold_s {ev.detail['hold_s']:.3f}  "
            f"d_aspect {ev.d_aspect_deg:+.1f}  d_rng {ev.d_rng_m:+.0f}  d_es {ev.d_es_m:+.0f}  "
            f"chi_peak {ev.detail['chi_peak_deg']:.1f}  nz_peak {ev.detail['nz_peak']:.2f}"
        )
        log(
            f"      ENTRY step {i} t {tr[i].t:.3f} rng {tr[i].rng:.1f} m aspect "
            f"{math.degrees(tr[i].aspect):.1f} deg closure {tr[i].closure:+.1f} m/s "
            f"alt {tr[i].alt:.1f} m |chi| {abs(math.degrees(tr[i].chi)):.1f} valid {tr[i].chi_valid}"
        )
        log(
            f"      EXIT  step {xi} t {tr[xi].t:.3f} rng {tr[xi].rng:.1f} m aspect "
            f"{math.degrees(tr[xi].aspect):.1f} deg closure {tr[xi].closure:+.1f} m/s "
            f"alt {tr[xi].alt:.1f} m |chi| {abs(math.degrees(tr[xi].chi)):.4f} "
            f"valid {tr[xi].chi_valid}"
        )

        log(
            "\n    (d) SEARCH WALK: every tick from hold_end+1 until the first |chi| < 30 (raw rule)"
        )
        for k in range(hold_end + 1, min(hold_end + 16, len(tr))):
            mark = ""
            if abs(math.degrees(tr[k].chi)) < B.CHI_RETURN_DEG:
                mark = "  <== raw (d) fires here"
            log(
                f"      step {k} t {tr[k].t:.3f} |chi| {abs(math.degrees(tr[k].chi)):7.2f} "
                f"valid {str(tr[k].chi_valid):>5} ata {math.degrees(tr[k].ata):6.1f} "
                f"nz {tr[k].nz:5.2f}{mark}"
            )
            if mark:
                break

        log("\n    VALIDITY-GATED (d): first tick with chi_valid AND |chi| < 30 in the window")
        alt_exit = None
        for k in range(hold_end + 1, len(tr)):
            if tr[k].t - tr[hold_end].t > B.CHI_RETURN_S:
                break
            if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                alt_exit = k
                break
        if alt_exit is None:
            log("      NONE in the window")
        else:
            log(
                f"      step {alt_exit} t {tr[alt_exit].t:.3f} |chi| "
                f"{abs(math.degrees(tr[alt_exit].chi)):.2f} valid {tr[alt_exit].chi_valid} "
                f"aspect {math.degrees(tr[alt_exit].aspect):.1f} deg  "
                f"d_aspect {math.degrees(tr[alt_exit].aspect - tr[i].aspect):+.1f} deg"
            )

        clause_walk(tr, i, hold_end, xi, "SHIPPED exit (sentinel)")
        if alt_exit is not None:
            clause_walk(tr, i, hold_end, alt_exit, "VALIDITY-GATED exit")

        log("\n    ata_az THROUGH THE EXTENDED WINDOW (the clause (e) span that lengthens)")
        for k in range(xi - 2, (alt_exit if alt_exit else xi) + 3):
            if 0 <= k < len(tr):
                log(
                    f"      step {k} t {tr[k].t:.3f} ata_az {math.degrees(tr[k].ata_az):+8.3f} "
                    f"sign {'+' if tr[k].ata_az >= 0 else '-'} ata {math.degrees(tr[k].ata):6.2f} "
                    f"|chi| {abs(math.degrees(tr[k].chi)):7.2f} valid {str(tr[k].chi_valid):>5}"
                )

        log("\n    WHICH POLE DOES THE chi_valid SENTINEL FIRE AT? (ahead, or directly ASTERN?)")
        log(
            "      |los_perp| = |sin(angle between LOS and the VELOCITY vector)|; the test is "
            f"sign-blind, |los_perp| < sin({B.CHI_VALID_ATA_DEG:.0f} deg) = "
            f"{math.sin(math.radians(B.CHI_VALID_ATA_DEG)):.4f}, so it fires at BOTH poles."
        )
        for k in range(188, 203):
            me, bd = _ME[k], _BD[k]
            v_hat = B._unit(np.asarray(me.vel, float))
            los = B._unit(np.asarray(bd.pos, float) - np.asarray(me.pos, float))
            dot = float(los @ v_hat)
            lp = float(np.linalg.norm(los - dot * v_hat))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            log(
                f"      step {k} t {tr[k].t:.3f}  los.v_hat {dot:+.4f}  angle(LOS,v) {ang:7.2f} deg "
                f" |los_perp| {lp:.4f}  chi_valid {str(tr[k].chi_valid):>5}  "
                f"{'ASTERN' if dot < 0 else 'ahead'}"
            )

        log("\n    IS THE (e) FLIP A GENUINE NOSE CROSSING, OR AN ata_az WRAP AT +-180?")
        log(
            "      the clause's stated intent is 'the bandit crossed the nose'; a nose crossing is "
            "ata_az through 0, and |ata_az| is SMALL there. A wrap is ata_az through +-180."
        )
        for lo, hi, tag in (
            (i, xi, f"{i}..{xi} (shipped exit)"),
            (i, alt_exit, f"{i}..{alt_exit} (validity-gated exit)"),
        ):
            if hi is None:
                continue
            raw_flips, nose_flips, wrap_flips = [], [], []
            for k in range(lo + 1, hi + 1):
                s_prev = 1 if tr[k - 1].ata_az >= 0.0 else -1
                s_now = 1 if tr[k].ata_az >= 0.0 else -1
                if s_prev == s_now:
                    continue
                raw_flips.append(k)
                if abs(math.degrees(tr[k].ata_az)) < 90.0:
                    nose_flips.append(k)
                else:
                    wrap_flips.append(k)
            log(
                f"      window {tag}: raw sign flips {len(raw_flips)} at {raw_flips}; "
                f"genuine NOSE crossings (|ata_az| < 90) {len(nose_flips)}; "
                f"wrap flips near +-180 {len(wrap_flips)}"
            )
            log(
                f"        min |ata_az| over the window = "
                f"{min(abs(math.degrees(tr[k].ata_az)) for k in range(lo, hi + 1)):.2f} deg, "
                f"min ata = {min(math.degrees(tr[k].ata) for k in range(lo, hi + 1)):.2f} deg"
            )

        log("\n    GEOMETRY THROUGH THE HOLD (was the bandit ever in front?)")
        log(
            f"      {'step':>5} {'t':>6} {'rng':>7} {'clos':>7} {'ata':>7} {'ata_az':>9} "
            f"{'aspect':>7} {'|chi|':>7} {'valid':>5} {'nz':>5} {'phi':>7}"
        )
        for k in list(range(i, hold_end + 1, 6)) + [hold_end, xi]:
            log(
                f"      {k:>5} {tr[k].t:>6.3f} {tr[k].rng:>7.1f} {tr[k].closure:>+7.0f} "
                f"{math.degrees(tr[k].ata):>7.1f} {math.degrees(tr[k].ata_az):>+9.2f} "
                f"{math.degrees(tr[k].aspect):>7.1f} {abs(math.degrees(tr[k].chi)):>7.1f} "
                f"{str(tr[k].chi_valid):>5} {tr[k].nz:>5.2f} {math.degrees(tr[k].phi):>+7.1f}"
            )
        rr = [tr[k].rng for k in range(len(tr))]
        cpa = int(np.argmin(rr[: xi + 40]))
        log(
            f"      min rng inside [0, {xi + 40}] = {rr[cpa]:.1f} m at step {cpa} t {tr[cpa].t:.3f} "
            f"-- the CPA falls INSIDE the event window [{i}, {xi}]: {i <= cpa <= xi}"
        )
        log(
            f"      ata over the whole hold: min {min(math.degrees(tr[k].ata) for k in range(i, hold_end + 1)):.1f} "
            f"max {max(math.degrees(tr[k].ata) for k in range(i, hold_end + 1)):.1f} deg "
            "(ata > 90 deg = bandit BEHIND the nose)"
        )

        log("\n    WHO ROLLED? own attitude vs the LOS, over the hold")
        pl0, pl1, pls0, pls1 = unwrapped_travel(tr, i, hold_end)
        log(
            f"      phi_lift (own lift roll about v): {pl0:+.1f} -> {pl1:+.1f} deg  "
            f"travel {abs(pl1 - pl0):.1f} deg"
        )
        log(
            f"      phi_los  (LOS roll about v)     : {pls0:+.1f} -> {pls1:+.1f} deg  "
            f"travel {abs(pls1 - pls0):.1f} deg"
        )
        phis = [math.degrees(tr[k].phi) for k in range(i, hold_end + 1)]
        ails = [tr[k].ail for k in range(i, hold_end + 1)]
        log(
            f"      own BANK phi over hold: {min(phis):+.1f} .. {max(phis):+.1f} deg "
            f"(range {max(phis) - min(phis):.1f} deg)"
        )
        log(
            f"      aileron over hold: min {min(ails):+.3f} max {max(ails):+.3f}  "
            f"|ail| > {B.AIL_COMMIT} on {sum(1 for x in ails if abs(x) > B.AIL_COMMIT)}/{len(ails)} ticks"
        )

        log("\n    PRICE THE HOLD IN ALTITUDE (never duration alone)")
        log(
            f"      Sample.alt {tr[i].alt:.0f} -> {tr[hold_end].alt:.0f} m = "
            f"{tr[hold_end].alt - tr[i].alt:+.0f} m over {tr[hold_end].t - tr[i].t:.3f} s "
            f"({(tr[hold_end].alt - tr[i].alt) / (tr[hold_end].t - tr[i].t):+.1f} m/s)"
        )
        log(
            f"      gamma {math.degrees(tr[i].gamma):+.1f} -> "
            f"{math.degrees(tr[hold_end].gamma):+.1f} deg   "
            f"min over hold {min(math.degrees(tr[k].gamma) for k in range(i, hold_end + 1)):+.1f}"
        )
        log(
            f"      cpa_miss {tr[i].cpa_miss:.1f} -> {tr[hold_end].cpa_miss:.1f} m   "
            f"episode dalt {r62.dalt_m:+.0f} m over {r62.steps * E.DT:.3f} s"
        )

    log("\nR3  WHY THE OTHER TWO LONG HOLDS PRODUCE NO EVENT (which clause rejects each)")
    for a, b in B._runs(hold):
        if tr[b].t - tr[a].t < B.CHI_MIN_HOLD_S:
            continue
        if any(ev.i0 == a for ev in evs):
            log(f"    hold {a}-{b}: this IS the shipped event")
            continue
        xi = None
        for k in range(b + 1, len(tr)):
            if tr[k].t - tr[b].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                xi = k
                break
        log(f"    hold {a}-{b} dur {tr[b].t - tr[a].t:.3f} s: raw (d) exit = {xi}")
        if xi is not None:
            clause_walk(tr, a, b, xi, f"hold {a}-{b}")

    path = f"{E.OUT}/refute_d1_validity.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
