"""ADVERSARIAL re-derivation of the 'detector is SOUND on this trajectory' claim.

Independent of _diag_d3b_detector_audit.py: rebuilds the Sample trace from the CSV smp_* block,
re-runs detect_all, re-derives every cited number, and then probes the clauses the shipped audit
did NOT vary.
"""

from __future__ import annotations

import csv
import math
import sys

sys.path.insert(0, "/workspace/crates/astrum-env/python")

from astrum_dogfight import bfm_detect as B  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"

SMP = [
    ("t", float),
    ("chi_rad", float),
    ("chi_valid", bool),
    ("gamma_rad", float),
    ("nz", float),
    ("phi_rad", float),
    ("alt", float),
    ("ail", float),
    ("rng", float),
    ("ata", float),
    ("ata_az", float),
    ("aspect", float),
    ("closure", float),
    ("cpa_miss", float),
    ("es_rel", float),
    ("lay_miss_m", float),
    ("lethal", bool),
]


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def trace(rows):
    tr = []
    for r in rows:
        kw = {}
        for name, cast in SMP:
            v = r[f"smp_{name}"]
            kw[name.removesuffix("_rad")] = (int(float(v)) != 0) if cast is bool else float(v)
        tr.append(B.Sample(**kw))
    return tr


def runs(mask):
    return B._runs(mask)


def d1_holds(tr, chi_out, nz_loaded=3.0):
    m = [(abs(s.chi) > math.radians(chi_out) and s.nz > nz_loaded) for s in tr]
    return runs(m)


def d1_clause_cascade(tr, chi_out, hold_s, nz_loaded=3.0, aspect_gain=15.0, waive_dur=False):
    """Verbatim D1 logic, instrumented: which clause kills each candidate run."""
    out = []
    for i, j in d1_holds(tr, chi_out, nz_loaded):
        rec = dict(run=(i, j), dur=tr[j].t - tr[i].t, fails=[])
        if not waive_dur and rec["dur"] < hold_s:
            rec["fails"].append("dur")
        # (a) in-plane lookback
        ok_a = any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        )
        if not ok_a:
            rec["fails"].append("a")
        if tr[i].closure <= 0.0:
            rec["fails"].append("c")
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            rec["fails"].append("d")
            rec["d_aspect"] = None
        else:
            sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
            if len(sides) != 1:
                rec["fails"].append("e")
            da = math.degrees(tr[exit_idx].aspect - tr[i].aspect)
            rec["d_aspect"] = da
            if da >= -aspect_gain:
                rec["fails"].append("f")
        rec["chi_pk"] = max(abs(math.degrees(tr[k].chi)) for k in range(i, j + 1))
        rec["nz_pk"] = max(tr[k].nz for k in range(i, j + 1))
        out.append(rec)
    return out


def sweep_d1(tr, chi_outs, holds, aspect_gain=15.0):
    rows = []
    old_out, old_hold, old_ag = B.CHI_OUT_DEG, B.CHI_MIN_HOLD_S, B.ASPECT_GAIN_DEG
    try:
        for co in chi_outs:
            for hs in holds:
                B.CHI_OUT_DEG, B.CHI_MIN_HOLD_S, B.ASPECT_GAIN_DEG = co, hs, aspect_gain
                n = len(B.detect_displacement_rolls(tr))
                casc = d1_clause_cascade(tr, co, hs, aspect_gain=aspect_gain)
                reached_f = sum(
                    1 for r in casc if r["d_aspect"] is not None and "dur" not in r["fails"]
                )
                only_f = sum(1 for r in casc if r["fails"] == ["f"])
                rows.append((co, hs, aspect_gain, n, len(casc), reached_f, only_f))
    finally:
        B.CHI_OUT_DEG, B.CHI_MIN_HOLD_S, B.ASPECT_GAIN_DEG = old_out, old_hold, old_ag
    return rows


def sweep_d2(tr, swings, rets, ret_ss):
    rows = []
    o = (B.YOYO_GAMMA_SWING_DEG, B.YOYO_GAMMA_RETURN_DEG, B.YOYO_RETURN_S)
    try:
        for sw in swings:
            for rd in rets:
                for rs in ret_ss:
                    B.YOYO_GAMMA_SWING_DEG, B.YOYO_GAMMA_RETURN_DEG, B.YOYO_RETURN_S = sw, rd, rs
                    ev = B.detect_yoyos(tr)
                    clean = [e for e in ev if e.detail.get("clean")]
                    rows.append((sw, rd, rs, len(ev), len(clean), ev))
    finally:
        B.YOYO_GAMMA_SWING_DEG, B.YOYO_GAMMA_RETURN_DEG, B.YOYO_RETURN_S = o
    return rows


def d3_cascade(tr, chi_room=45.0, nz_loaded=3.0, hold_s=0.5):
    out = []
    m = [(abs(s.chi) > math.radians(chi_room) and s.nz > nz_loaded) for s in tr]
    for i, j in runs(m):
        rec = dict(run=(i, j), dur=tr[j].t - tr[i].t, fails=[])
        if rec["dur"] < hold_s:
            rec["fails"].append("dur")
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        if len(sides) != 1:
            rec["fails"].append("side")
        rec["cpa_i"], rec["cpa_j"] = tr[i].cpa_miss, tr[j].cpa_miss
        if tr[j].cpa_miss <= tr[i].cpa_miss:
            rec["fails"].append("built")
        after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
        if not after or min(tr[k].cpa_miss for k in after) >= tr[j].cpa_miss:
            rec["fails"].append("cashed")
        rec["chi_pk"] = max(abs(math.degrees(tr[k].chi)) for k in range(i, j + 1))
        rec["nz_pk"] = max(tr[k].nz for k in range(i, j + 1))
        out.append(rec)
    return out


def main():
    for tag, path in (("KILL seed62", KILL), ("NOKILL seed28", NOKILL)):
        rows = load(path)
        tr = trace(rows)
        alt = [float(r["alt_enu_m"]) for r in rows]
        print("=" * 100)
        print(f"{tag}   {len(tr)} ticks   t {tr[0].t:.3f}..{tr[-1].t:.3f} s")
        print("=" * 100)
        r = B.detect_all(tr)
        print(
            f"  detect_all AS SHIPPED: D1={len(r['displacement_rolls'])} "
            f"D2={len(r['yoyos'])} D2c={len(r['yoyos_clean'])} "
            f"D3={len(r['turning_room'])} lethal_ticks={r['lethal_ticks']}"
        )

        # ---- aspect: the clause the shipped sweep never varied ------------------
        asp = [math.degrees(s.aspect) for s in tr]
        print(
            f"  aspect: {asp[0]:.1f} -> {asp[-1]:.1f} deg, min {min(asp):.1f}, max {max(asp):.1f}; "
            f"best d_aspect from t=0 = {min(asp) - asp[0]:+.1f} deg  (clause f needs <= -15.0)"
        )
        # global best fall over ANY (i,j) pair with i<j
        best, bij = 0.0, None
        for i in range(len(asp)):
            for j in range(i + 1, len(asp)):
                d = asp[j] - asp[i]
                if d < best:
                    best, bij = d, (i, j)
        print(
            f"  best aspect FALL over any (i<j) pair on the whole trace: {best:+.1f} deg at {bij}"
        )

        # ---- D1 clause cascade at shipped thresholds ----------------------------
        print("  D1 candidate holds at CHI_OUT=60 NZ=3 (verbatim clause cascade):")
        for rec in d1_clause_cascade(tr, 60.0, 0.5):
            da = "None" if rec["d_aspect"] is None else f"{rec['d_aspect']:+.1f}"
            print(
                f"     run {rec['run']}  dur {rec['dur']:.3f}s  chiPk {rec['chi_pk']:.1f}  "
                f"nzPk {rec['nz_pk']:.2f}  d_aspect {da}  FAILS={rec['fails'] or 'NONE'}"
            )

        # ---- my own 20-combo chi sweep, PLUS what it actually tested ------------
        print("  D1 sweep (chi_out x hold_s) with ASPECT_GAIN=15.0 (shipped):")
        print("     CHI_OUT HOLD_S  D1  nHolds  reachedF  onlyF-fails")
        for co, hs, ag, n, nh, rf, of in sweep_d1(tr, [60, 50, 45, 40, 30], [0.5, 0.3, 0.2, 0.125]):
            print(f"     {co:7.0f} {hs:6.3f} {n:3d} {nh:7d} {rf:9d} {of:12d}")

        # ---- the SAME sweep with the binding clause (f) neutralised -------------
        print(
            "  D1 sweep with ASPECT_GAIN_DEG = -180 (clause f neutralised, everything else verbatim):"
        )
        print("     CHI_OUT HOLD_S  D1")
        for co, hs, ag, n, nh, rf, of in sweep_d1(
            tr, [60, 50, 45, 40, 30], [0.5, 0.3, 0.2, 0.125], aspect_gain=-180.0
        ):
            print(f"     {co:7.0f} {hs:6.3f} {n:3d}")

        # ---- D2 -----------------------------------------------------------------
        entries = [
            i
            for i in range(len(tr))
            if tr[i].closure > B.CLOSURE_OVERSHOOT
            and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
        ]
        hot = sum(1 for s in tr if s.closure > B.CLOSURE_OVERSHOOT)
        print(
            f"  D2 entry gate: closure>120 on {hot}/{len(tr)} ticks "
            f"({100.0 * hot / len(tr):.1f}%); RISING-EDGE candidate entries = {len(entries)} at {entries}"
        )
        if entries:
            i = entries[0]
            g0 = tr[i].gamma
            win = [k for k in range(i, len(tr)) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
            pk = max(win, key=lambda k: abs(tr[k].gamma - g0))
            sw = math.degrees(abs(tr[pk].gamma - g0))
            print(
                f"     entry {i}: g0 {math.degrees(g0):+.2f} deg; peak step {pk} (t {tr[pk].t:.3f}s) "
                f"gamma {math.degrees(tr[pk].gamma):+.2f}, swing {sw:.2f} deg "
                f"vs SWING={B.YOYO_GAMMA_SWING_DEG} -> margin {B.YOYO_GAMMA_SWING_DEG - sw:+.2f} deg "
                f"({100.0 * (B.YOYO_GAMMA_SWING_DEG - sw) / B.YOYO_GAMMA_SWING_DEG:.1f}% of threshold)"
            )
            tail = [k for k in range(pk + 1, len(tr)) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
            if tail:
                bk = min(tail, key=lambda k: abs(tr[k].gamma - g0))
                print(
                    f"     closest return in tail: step {bk} |gamma-g0| "
                    f"{math.degrees(abs(tr[bk].gamma - g0)):.2f} deg vs RET={B.YOYO_GAMMA_RETURN_DEG}"
                )
        print("  D2 sweep (swing x ret_deg x ret_s):  D2 / D2clean, and the fired event PRICED")
        flips = 0
        tot = 0
        for sw, rd, rs, n, nc, evs in sweep_d2(tr, [30, 25, 20], [10, 20, 30], [6, 12]):
            tot += 1
            if n:
                flips += 1
            extra = ""
            for e in evs:
                dalt = alt[e.i1] - alt[e.i0]
                extra += (
                    f"   [{e.kind} i {e.i0}->{e.i1} t {e.t0:.2f}->{e.t1:.2f}s "
                    f"gpk {e.detail['gamma_peak_deg']:+.1f} swing {e.detail['gamma_swing_deg']:.1f} "
                    f"dalt {dalt:+.0f} m d_aspect {e.d_aspect_deg:+.1f} "
                    f"failed_exit={e.detail['failed_exit']} failed_aspect={e.detail['failed_aspect']}]"
                )
            print(f"     sw {sw:3d} ret {rd:3d} ret_s {rs:3d} -> D2 {n} D2c {nc}{extra}")
        print(f"     D2 flipped ON in {flips}/{tot} relaxed combinations (base: 0 as shipped)")

        # ---- D3 -----------------------------------------------------------------
        print("  D3 candidate holds (verbatim clause cascade):")
        for rec in d3_cascade(tr):
            print(
                f"     run {rec['run']}  dur {rec['dur']:.3f}s  chiPk {rec['chi_pk']:.1f}  "
                f"nzPk {rec['nz_pk']:.2f}  cpa {rec['cpa_i']:.1f}->{rec['cpa_j']:.1f}  "
                f"FAILS={rec['fails'] or 'NONE'}  (nFails={len(rec['fails'])})"
            )
        cpa = [s.cpa_miss for s in tr]
        rises = sum(1 for k in range(1, len(cpa)) if cpa[k] > cpa[k - 1])
        print(f"  cpa_miss: {cpa[0]:.1f} -> {cpa[-1]:.1f} m; ticks RISING {rises}/{len(cpa) - 1}")

        # ---- price the D3-bar hold in altitude ---------------------------------
        for i, j in runs([(abs(s.chi) > math.radians(45.0) and s.nz > 3.0) for s in tr]):
            if tr[j].t - tr[i].t >= 0.5:
                print(
                    f"  PRICED HOLD steps {i}-{j}: {tr[j].t - tr[i].t:.3f} s, "
                    f"dalt {alt[j] - alt[i]:+.1f} m, nz {min(tr[k].nz for k in range(i, j + 1)):.2f}"
                    f"..{max(tr[k].nz for k in range(i, j + 1)):.2f} g"
                )
        print()


if __name__ == "__main__":
    main()
