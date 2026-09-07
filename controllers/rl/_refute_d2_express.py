"""ADVERSARIAL check of the Q4 "D2/D2c not expressible at ANY resolution" verdict.

Four independent probes:

  A  REPRODUCE the 16 rows of runs/d3b/d2_express_report.txt verbatim (same Sched list, same
     seed, same 576-step horizon) and recompute swing / ret / best / D2 / D2c myself.
  B  Read the gamma(t) trace of the two rows the verdict leans on ('SEG 4.0s bank0 pull then
     4.0s bank115 pull' = the 4.0 s point, 'FINE 0.5 bank0 / 1.0 bank115 / rest bank0' = the
     0.5 s point) and ask whether the failure mode really is IDENTICAL, i.e. whether the
     0.5 s schedule ever COMMANDS a return leg at all.
  C  The untested channels. All 16 published schedules hold load = 1.0 AND thr = 1.0. The
     genome has three channels; throttle is the one that sets V, and gamma-rate = (n cos phi -
     cos gamma) g / V, so a lower throttle rotates FASTER. Sweep throttle and the down-phase
     length on both the HIGH and LOW families, at phase resolutions the published grid skipped
     (d_up 2.0..3.2 by 0.1 -- the published HIGH grid jumps 2.5 -> 3.0, straight over the
     swing = 30 bar).
  D  Closure-edge structure at low throttle: the "the only edge is step 0" premise is measured
     only on full-throttle traces. If idle throttle makes closure dip under 120 m/s and come
     back, a SECOND entry edge exists and the 6 s window is no longer anchored at t = 0.

Everything is flown through _diag_d3b_express.fly -> _diag_true_mpc.build_env/roll_once and
scored with the real astrum_dogfight.bfm_detect.detect_all. No dynamics or detector code is
reimplemented.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []
WIN_S = B.YOYO_RETURN_S


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


# -------------------------------------------------------------------------------------------
# the quantities D2 reads, recomputed independently of _diag_d2_express.d2_stats
# -------------------------------------------------------------------------------------------
def d2_stats(tr) -> dict:
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    out = dict(edges=edges, n=n)
    if not edges:
        return out
    i = edges[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= WIN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
    ret = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > WIN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    # every (peak, return) pair in the window that clears BOTH gamma bars -- a SUPERSET of what
    # the real detector can find, because the detector only ever tests the argmax peak.
    pairs = []
    for p in win:
        sw = math.degrees(abs(tr[p].gamma - g0))
        if (
            sw < B.YOYO_GAMMA_SWING_DEG
            or abs(math.degrees(tr[p].gamma)) < B.YOYO_GAMMA_EXCURSION_DEG
        ):
            continue
        ld = [tr[k].nz > B.NZ_LOADED for k in range(i, p + 1)]
        lf_p = sum(ld) / max(1, len(ld))
        for k in range(p + 1, n):
            if tr[k].t - tr[i].t > WIN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                pairs.append((p, k, sw, lf_p))
                break
    best = max(pairs, key=lambda z: z[2]) if pairs else None
    out.update(
        swing=math.degrees(abs(tr[peak].gamma - g0)),
        gpk=math.degrees(tr[peak].gamma),
        t_peak=tr[peak].t,
        lf=sum(loaded) / max(1, len(loaded)),
        ret=ret,
        best=best,
        npairs=len(pairs),
        clo_min_win=min(tr[k].closure for k in win),
        rng_min_win=min(tr[k].rng for k in win),
        g0=math.degrees(g0),
    )
    return out


PUBLISHED = []
for d_up in (1.0, 1.5, 2.0, 2.5, 3.0):
    PUBLISHED.append(
        (
            f"HIGH pull {d_up:.1f}s @bank0 then bank115 pull",
            [(d_up, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
        )
    )
for d_dn in (2.0, 2.5, 3.0, 3.5, 4.0):
    PUBLISHED.append(
        (
            f"LOW  bank115 pull {d_dn:.1f}s then bank0 pull",
            [(d_dn, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)],
        )
    )
PUBLISHED += [
    (
        "SEG  4.0s bank0 pull then 4.0s bank115 pull",
        [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)],
    ),
    (
        "SEG  4.0s bank115 pull then 4.0s bank0 pull",
        [(4.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)],
    ),
    (
        "FINE 1.0 bank0 / 1.5 bank115 / rest bank0",
        [(1.0, 0.0, 1.0, 1.0), (1.5, 115.0, 1.0, 1.0), (21.5, 0.0, 1.0, 1.0)],
    ),
    (
        "FINE 0.5 bank0 / 1.0 bank115 / rest bank0",
        [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
    ),
    (
        "FINE 1.5 bank115 / 1.5 bank0 / rest bank115",
        [(1.5, 115.0, 1.0, 1.0), (1.5, 0.0, 1.0, 1.0), (21.0, 115.0, 1.0, 1.0)],
    ),
    (
        "FINE 2.5 bank115 / 2.0 bank0 / rest bank115",
        [(2.5, 115.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (19.5, 115.0, 1.0, 1.0)],
    ),
]


def row(name: str, phases, n_steps: int, seed: int = 62, keep_rows: bool = False):
    r, rows = E.fly(E.Sched(phases), n_steps=n_steps, seed=seed)
    st = d2_stats(r.trace)
    det = B.detect_all(r.trace)
    nwin = max(1, int(round(WIN_S / E.DT)))
    kwin = min(nwin, len(rows) - 1)
    dalt = rows[kwin]["alt"] - rows[0]["alt"]
    log(
        f"     {name:<46} {r.steps:>5} {len(st['edges']):>5} {st.get('swing', float('nan')):>6.1f} "
        f"{st.get('gpk', float('nan')):>+7.1f} {st.get('lf', float('nan')):>4.2f} "
        f"{str(st.get('ret')):>5} {str(st.get('best')):>28} {st.get('npairs', 0):>3} "
        f"{st.get('clo_min_win', float('nan')):>7.0f} {st.get('rng_min_win', float('nan')):>7.0f} "
        f"{dalt:>7.0f} {len(det['yoyos']):>3} {len(det['yoyos_clean']):>4}"
    )
    return r, rows, st, det


HDR = (
    f"     {'schedule':<46} {'steps':>5} {'edge':>5} {'swing':>6} {'gpk':>7} {'lf':>4} "
    f"{'ret':>5} {'best(peak,ret,swing,lf)':>28} {'#pr':>3} {'cloMin':>7} {'rngMin':>7} "
    f"{'dAlt6s':>7} {'D2':>3} {'D2c':>4}"
)


def main() -> None:
    N24 = int(round(24.0 / E.DT))
    N10 = int(round(10.0 / E.DT))

    log("#" * 118)
    log("#  REFUTATION PROBE -- 'D2/D2c not expressible at ANY resolution tested' (merge1500, D3b)")
    log("#" * 118)
    log(
        f"  bars: swing>={B.YOYO_GAMMA_SWING_DEG:.0f} excursion>={B.YOYO_GAMMA_EXCURSION_DEG:.0f} "
        f"return<{B.YOYO_GAMMA_RETURN_DEG:.0f} within {B.YOYO_RETURN_S:.0f}s  "
        f"entry closure>{B.CLOSURE_OVERSHOOT:.0f}  loaded_frac>={B.LOADED_FRAC} nz>{B.NZ_LOADED:.0f}  "
        f"clean: closure<{B.CLOSURE_OVERSHOOT:.0f} and rng<{B.YOYO_EXIT_RNG_M:.0f} at exit"
    )
    log(
        "  `best` here is the LARGEST qualifying (peak,return) pair, over ALL peaks; #pr = how many exist."
    )

    # ---------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  A  REPRODUCTION of the 16 published rows (seed 62, 576 steps, identical Sched list)")
    log("=" * 118)
    log(HDR)
    repro = {}
    for name, phases in PUBLISHED:
        r, rows, st, det = row(name, phases, N24)
        repro[name] = (st, det, r)

    # ---------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  B  IS THE FAILURE MODE IDENTICAL AT 4.0 s AND 0.5 s?  gamma(t) inside the 6 s window")
    log("=" * 118)
    for name, phases in [
        (
            "SEG  4.0s bank0 pull then 4.0s bank115 pull",
            [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)],
        ),
        (
            "FINE 0.5 bank0 / 1.0 bank115 / rest bank0",
            [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
        ),
        (
            "FINE 1.0 bank0 / 1.5 bank115 / rest bank0",
            [(1.0, 0.0, 1.0, 1.0), (1.5, 115.0, 1.0, 1.0), (21.5, 0.0, 1.0, 1.0)],
        ),
    ]:
        r, rows = E.fly(E.Sched(phases), n_steps=N10)
        log(f"\n  {name}")
        log("     t(s)  cmdBank  cmdLoad   phi     nz   gamma   dgamma/dt(deg/s)   vt    alt")
        for k in range(0, min(len(rows), int(round(WIN_S / E.DT)) + 1), 12):
            w = rows[k]
            if k + 12 < len(rows):
                rate = (rows[k + 12]["gamma"] - w["gamma"]) / (12 * E.DT)
            else:
                rate = float("nan")
            log(
                f"     {w['t']:5.2f} {w['cmd_bank']:>8.1f} {w['cmd_thr']:>8.2f} {w['phi']:>7.1f} "
                f"{w['nz']:>6.2f} {w['gamma']:>7.1f} {rate:>16.2f} {w['vt']:>6.1f} {w['alt']:>7.0f}"
            )
        gam = [w["gamma"] for w in rows[: int(round(WIN_S / E.DT)) + 1]]
        d = [gam[k + 1] - gam[k] for k in range(len(gam) - 1)]
        rev = sum(1 for k in range(1, len(d)) if d[k] * d[k - 1] < 0.0)
        log(
            f"     -> gamma in window: min {min(gam):+.1f} max {max(gam):+.1f}; dgamma sign reversals {rev}"
        )
        # honest, simple: is the LAST commanded phase inside the window a return-leg command?
        sch = E.Sched(phases)
        log(
            f"     -> commanded bank at t=3,4,5,6 s: {[sch.cmd(t)[0] for t in (3.0, 4.0, 5.0, 6.0)]}"
        )

    # ---------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  C1 HIGH family, FINE d_up sweep the published grid skipped (2.5 -> 3.0 jumps the bar)")
    log("=" * 118)
    log(HDR)
    hits = []
    for thr in (1.0, 0.6, 0.0):
        for d_up in (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2):
            nm = f"HIGH {d_up:.1f}s bank0 pull -> bank115 pull, thr {thr:.1f}"
            r, rows, st, det = row(nm, [(d_up, 0.0, 1.0, thr), (24.0, 115.0, 1.0, thr)], N10)
            if st.get("best") or det["yoyos"]:
                hits.append((nm, st, det))

    log("\n" + "=" * 118)
    log(
        "  C2 HIGH family, 3-phase: up / over-and-down / LEVEL OFF (keeps the argmax on the up peak)"
    )
    log("=" * 118)
    log(HDR)
    for thr in (1.0, 0.0):
        for d_up in (2.2, 2.6, 3.0):
            for d_dn in (1.5, 2.0, 2.5, 3.0):
                nm = f"HIGH3 {d_up:.1f}/{d_dn:.1f} thr{thr:.1f} lvl"
                phases = [(d_up, 0.0, 1.0, thr), (d_dn, 115.0, 1.0, thr), (24.0, 0.0, 0.25, thr)]
                r, rows, st, det = row(nm, phases, N10)
                if st.get("best") or det["yoyos"]:
                    hits.append((nm, st, det))

    log("\n" + "=" * 118)
    log("  C3 LOW family, FINE d_dn sweep + throttle (published grid stops at 4.0 s, swing 29.0)")
    log("=" * 118)
    log(HDR)
    for thr in (1.0, 0.6, 0.0):
        for d_dn in (3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8):
            nm = f"LOW {d_dn:.1f}s bank115 pull -> bank0 pull, thr {thr:.1f}"
            r, rows, st, det = row(nm, [(d_dn, 115.0, 1.0, thr), (24.0, 0.0, 1.0, thr)], N10)
            if st.get("best") or det["yoyos"]:
                hits.append((nm, st, det))

    log("\n" + "=" * 118)
    log(
        "  C4 sign flip (-115): the bandit is 1500 m to ONE side, so roll direction is not symmetric"
    )
    log("=" * 118)
    log(HDR)
    for thr in (1.0, 0.0):
        for d_up in (2.4, 2.8, 3.2):
            nm = f"HIGH- {d_up:.1f}s bank0 -> bank-115, thr {thr:.1f}"
            r, rows, st, det = row(nm, [(d_up, 0.0, 1.0, thr), (24.0, -115.0, 1.0, thr)], N10)
            if st.get("best") or det["yoyos"]:
                hits.append((nm, st, det))
        for d_dn in (3.8, 4.2, 4.6):
            nm = f"LOW-  {d_dn:.1f}s bank-115 -> bank0, thr {thr:.1f}"
            r, rows, st, det = row(nm, [(d_dn, -115.0, 1.0, thr), (24.0, 0.0, 1.0, thr)], N10)
            if st.get("best") or det["yoyos"]:
                hits.append((nm, st, det))

    # ---------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  D  CLOSURE-EDGE STRUCTURE over the full 24 s -- is step 0 really the only entry edge?")
    log("=" * 118)
    log(
        f"     {'schedule':<46} {'steps':>5} {'#edges':>6} {'edge steps (t s)':<40} "
        f"{'cloMin24':>9} {'D2':>3} {'D2c':>4}"
    )
    for nm, phases in [
        ("full-thr max pull, bank 115 all 24 s", [(24.0, 115.0, 1.0, 1.0)]),
        ("IDLE thr, max pull, bank 115 all 24 s", [(24.0, 115.0, 1.0, 0.0)]),
        ("IDLE thr, max pull, bank 0 all 24 s", [(24.0, 0.0, 1.0, 0.0)]),
        ("IDLE thr, HIGH 2.8/rest", [(2.8, 0.0, 1.0, 0.0), (24.0, 115.0, 1.0, 0.0)]),
        ("IDLE thr, LOW 4.2/rest", [(4.2, 115.0, 1.0, 0.0), (24.0, 0.0, 1.0, 0.0)]),
        (
            "IDLE thr, hard break bank 115 then level",
            [(6.0, 115.0, 1.0, 0.0), (24.0, 0.0, 0.3, 0.0)],
        ),
    ]:
        r, rows = E.fly(E.Sched(phases), n_steps=N24)
        tr = r.trace
        edges = [
            i
            for i in range(len(tr))
            if tr[i].closure > B.CLOSURE_OVERSHOOT
            and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
        ]
        det = B.detect_all(tr)
        es = ", ".join(f"{i}({tr[i].t:.1f})" for i in edges[:6])
        log(
            f"     {nm:<46} {r.steps:>5} {len(edges):>6} {es:<40} "
            f"{min(s.closure for s in tr):>9.0f} {len(det['yoyos']):>3} {len(det['yoyos_clean']):>4}"
        )

    # ---------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  SUMMARY of anything that cleared the gamma bars WITH a return, or scored D2 > 0")
    log("=" * 118)
    if not hits:
        log("     NONE -- every probe in C1..C4 has best = None and D2 = 0.")
    for nm, st, det in hits:
        log(
            f"     {nm:<52} best={st.get('best')} npairs={st.get('npairs')} D2={len(det['yoyos'])} D2c={len(det['yoyos_clean'])}"
        )
        for ev in det["yoyos"]:
            log(f"        {ev.kind} t {ev.t0:.3f}-{ev.t1:.3f} {ev.detail}")

    path = f"{E.OUT}/refute_d2_express.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
