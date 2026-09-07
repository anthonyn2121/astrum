"""D3b DETECTOR AUDIT — instrument every clause of D1/D2/D2c/D3 against the replayed kill trace.

The whole program concludes "the agent does no BFM" from D1 = D2 = 0. If the instrument is too
strict, that body of conclusions is suspect. This script does not re-implement any detector: it
rebuilds the exact `bfm_detect.Sample` trace the detectors read (from the `smp_*` block of the
replay CSVs, VERIFIED field-by-field against a fresh re-flight), then walks each detector's clause
cascade by hand, printing "criterion X requires >= A, trace achieved B, at its best around step S".

Threshold probes monkeypatch `bfm_detect` module globals at call time (the detector bodies read the
globals inside the function, so this is the sensitivity probe the module's own header invites) and
restore them afterwards. NO repo file is modified.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_d3b_detector_audit.py'
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
CDT = 5.0 / 120.0  # ACTION_REPEAT / RATE_HZ, the control step (s)

SMP_FIELDS = [
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


def load_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def trace_from_rows(rows: list[dict]) -> list[B.Sample]:
    """Rebuild the exact detector trace from the CSV `smp_*` block."""
    tr = []
    for r in rows:
        kw = {}
        for name, cast in SMP_FIELDS:
            v = r[f"smp_{name}"]
            kw[name.removesuffix("_rad")] = (int(float(v)) != 0) if cast is bool else float(v)
        tr.append(B.Sample(**kw))
    return tr


# ==========================================================================================
# fidelity check: does the reconstructed trace equal the trace the sim produced?
# ==========================================================================================
def verify_against_reflight(rows: list[dict], tr: list[B.Sample], seed: int, log) -> None:
    import _diag_true_mpc as M

    z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
    g = np.asarray(z["genome"], float)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    cell = M.CELLS[str(z["cell"])]
    n_steps = 576
    env, base = M.build_env(cell, seed, n_steps * M.CONTROL_DT_S)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, n_steps, me)

    r = M.roll_once(env, base, ctl, n_steps, keep=True)
    env.close()
    ref = r.trace
    log(f"   re-flight seed {seed}: {len(ref)} samples vs {len(tr)} reconstructed")
    assert len(ref) == len(tr), "length mismatch"
    worst = {}
    for raw, cast in SMP_FIELDS:
        name = raw.removesuffix("_rad")
        if cast is bool:
            d = sum(1 for a, b in zip(ref, tr) if getattr(a, name) != getattr(b, name))
            worst[name] = float(d)
        else:
            worst[name] = max(abs(getattr(a, name) - getattr(b, name)) for a, b in zip(ref, tr))
    bad = {k: v for k, v in worst.items() if v > 1e-9}
    log(f"   max |reconstructed - reflight| over all 17 Sample fields: {max(worst.values()):.3e}")
    log(f"   fields differing by > 1e-9: {bad if bad else 'NONE'}")
    d_ref = B.detect_all(ref)
    d_rec = B.detect_all(tr)
    for k in ("displacement_rolls", "yoyos", "yoyos_clean", "turning_room"):
        assert len(d_ref[k]) == len(d_rec[k]), f"{k} differs"
    log(
        "   detect_all on the re-flight trace == detect_all on the reconstructed trace: "
        f"D1 {len(d_ref['displacement_rolls'])} D2 {len(d_ref['yoyos'])} "
        f"D2c {len(d_ref['yoyos_clean'])} D3 {len(d_ref['turning_room'])}  CONFIRMED"
    )


# ==========================================================================================
# helpers
# ==========================================================================================
def runs_of(mask) -> list[tuple[int, int]]:
    return B._runs(list(mask))


def longest_run(mask) -> tuple[int, int, int]:
    rs = runs_of(mask)
    if not rs:
        return (-1, -1, 0)
    i, j = max(rs, key=lambda ij: ij[1] - ij[0])
    return (i, j, j - i + 1)


def fmt_run(tr, i, j) -> str:
    if i < 0:
        return "none"
    return (
        f"steps {i}-{j} ({j - i + 1} ticks, {tr[j].t - tr[i].t:.3f} s, "
        f"t {tr[i].t:.3f}-{tr[j].t:.3f} s)"
    )


# ==========================================================================================
# D1 cascade
# ==========================================================================================
def audit_d1(tr, log, tag) -> None:
    n = len(tr)
    chi_out = math.radians(B.CHI_OUT_DEG)
    m_chi = [abs(s.chi) > chi_out for s in tr]
    m_nz = [s.nz > B.NZ_LOADED for s in tr]
    hold = [a and b for a, b in zip(m_chi, m_nz)]

    log("")
    log(f"   D1 displacement_rolls  ({tag})   -- bfm_detect.py:316-376")
    log(
        f"     ingredient A  |chi| > CHI_OUT_DEG={B.CHI_OUT_DEG:.0f} deg      : "
        f"{sum(m_chi)}/{n} ticks ({sum(m_chi) / n * 100:.1f}%)   longest run {fmt_run(tr, *longest_run(m_chi)[:2])}"
    )
    log(
        f"     ingredient B  nz > NZ_LOADED={B.NZ_LOADED:.0f} g                : "
        f"{sum(m_nz)}/{n} ticks ({sum(m_nz) / n * 100:.1f}%)   longest run {fmt_run(tr, *longest_run(m_nz)[:2])}"
    )
    log(
        f"     CO-OCCURRENCE (the `hold` mask, line 336)          : "
        f"{sum(hold)}/{n} ticks   longest run {fmt_run(tr, *longest_run(hold)[:2])}"
    )
    mx = max(range(n), key=lambda k: abs(tr[k].chi))
    log(
        f"     max |chi| overall: {abs(tr[mx].chi) * DEG:.1f} deg at step {mx} (t {tr[mx].t:.3f} s), "
        f"nz there {tr[mx].nz:+.2f} g, chi_valid {tr[mx].chi_valid}"
    )
    mxl = [k for k in range(n) if m_nz[k]]
    if mxl:
        k = max(mxl, key=lambda k: abs(tr[k].chi))
        log(
            f"     max |chi| among LOADED (nz>3) ticks: {abs(tr[k].chi) * DEG:.1f} deg at step {k} "
            f"(t {tr[k].t:.3f} s), nz {tr[k].nz:.2f} g   <- the best simultaneous state"
        )
    log(
        f"     CLAUSE hold >= CHI_MIN_HOLD_S={B.CHI_MIN_HOLD_S:.2f} s: "
        f"trace best = {longest_run(hold)[2] * CDT if longest_run(hold)[2] else 0.0:.3f} s "
        f"({longest_run(hold)[2]} ticks) -> "
        f"{'PASS' if (longest_run(hold)[2] - 1) * CDT >= B.CHI_MIN_HOLD_S else 'FAIL'}"
        f"  (the detector measures tr[j].t - tr[i].t, i.e. ticks-1)"
    )

    # Which clause binds if the hold-duration clause is waived? Re-run the cascade over EVERY run.
    log("     per-run clause cascade with the DURATION clause WAIVED (all other clauses verbatim):")
    log(
        f"       {'run':>11} {'dur_s':>6} {'chiPk':>6} {'nzPk':>5} {'(a)inpl':>8} {'(c)clos':>8} "
        f"{'(d)back':>8} {'(e)side':>8} {'(f)dasp':>9} {'verdict':>10}"
    )
    for i, j in runs_of(hold):
        a_ok = any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        )
        c_ok = tr[i].closure > 0.0
        exit_idx = None
        for k in range(j + 1, n):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        d_ok = exit_idx is not None
        e_ok = False
        dasp = float("nan")
        if d_ok:
            sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
            e_ok = len(sides) == 1
            dasp = (tr[exit_idx].aspect - tr[i].aspect) * DEG
        f_ok = bool(d_ok and dasp < -B.ASPECT_GAIN_DEG)
        dur = tr[j].t - tr[i].t
        fails = [
            nm
            for nm, ok in (
                ("dur", dur >= B.CHI_MIN_HOLD_S),
                ("a", a_ok),
                ("c", c_ok),
                ("d", d_ok),
                ("e", e_ok),
                ("f", f_ok),
            )
            if not ok
        ]
        log(
            f"       {f'{i}-{j}':>11} {dur:>6.3f} "
            f"{max(abs(tr[k].chi) for k in range(i, j + 1)) * DEG:>6.1f} "
            f"{max(tr[k].nz for k in range(i, j + 1)):>5.2f} "
            f"{str(a_ok):>8} {str(c_ok):>8} {str(d_ok):>8} {str(e_ok):>8} "
            f"{dasp:>+9.1f} {('FAIL ' + ','.join(fails)) if fails else 'PASS':>10}"
        )
    log(
        "       (f) requires d_aspect <= -ASPECT_GAIN_DEG = -15.0 deg; aspect convention is "
        "geometry.py:152 `LOW = his 6 o'clock`, so a PASS needs aspect to FALL 15 deg."
    )
    log(
        f"     whole-trace aspect: {tr[0].aspect * DEG:.1f} deg at step 0 -> "
        f"{tr[-1].aspect * DEG:.1f} deg at step {n - 1}; min "
        f"{min(s.aspect for s in tr) * DEG:.1f} deg, max {max(s.aspect for s in tr) * DEG:.1f} deg; "
        f"best d_aspect from step 0 = {(min(s.aspect for s in tr) - tr[0].aspect) * DEG:+.1f} deg"
    )


# ==========================================================================================
# D2 cascade
# ==========================================================================================
def audit_d2(tr, log, tag) -> None:
    n = len(tr)
    ret = math.radians(B.YOYO_GAMMA_RETURN_DEG)
    above = [s.closure > B.CLOSURE_OVERSHOOT for s in tr]
    edges = [i for i in range(n) if above[i] and (i == 0 or not above[i - 1])]
    log("")
    log(f"   D2 yoyos  ({tag})   -- bfm_detect.py:382-456")
    log(
        f"     closure > CLOSURE_OVERSHOOT={B.CLOSURE_OVERSHOOT:.0f} m/s on "
        f"{sum(above)}/{n} ticks ({sum(above) / n * 100:.1f}%);  closure range "
        f"{min(s.closure for s in tr):+.0f} .. {max(s.closure for s in tr):+.0f} m/s"
    )
    log(
        f"     ENTRY GATE is a RISING EDGE (line 411): candidate entries on this trace = "
        f"{len(edges)} -> steps {edges}"
    )
    for i in edges:
        g0 = tr[i].gamma
        window = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
        log(
            f"     -- entry step {i} (t {tr[i].t:.3f} s): g0 = {g0 * DEG:+.2f} deg, "
            f"closure {tr[i].closure:+.0f} m/s, window {len(window)} ticks "
            f"(t {tr[i].t:.3f}-{tr[window[-1]].t:.3f} s)"
        )
        if len(window) < 3:
            log("        len(window) < 3 -> `break`, loop ENDS")
            continue
        peak = max(window, key=lambda k: abs(tr[k].gamma - g0))
        sw = abs(tr[peak].gamma - g0) * DEG
        ex = abs(tr[peak].gamma) * DEG
        log(
            f"        peak swing step {peak} (t {tr[peak].t:.3f} s): gamma {tr[peak].gamma * DEG:+.2f} deg,"
            f" swing {sw:.2f} deg, |gamma| {ex:.2f} deg"
        )
        log(
            f"        CLAUSE swing >= YOYO_GAMMA_SWING_DEG={B.YOYO_GAMMA_SWING_DEG:.0f}: got "
            f"{sw:.2f} -> {'PASS' if sw >= B.YOYO_GAMMA_SWING_DEG else f'FAIL by {B.YOYO_GAMMA_SWING_DEG - sw:.2f} deg'}"
        )
        log(
            f"        CLAUSE |gamma_peak| >= YOYO_GAMMA_EXCURSION_DEG={B.YOYO_GAMMA_EXCURSION_DEG:.0f}: got "
            f"{ex:.2f} -> {'PASS' if ex >= B.YOYO_GAMMA_EXCURSION_DEG else 'FAIL'}"
        )
        loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
        lf = sum(loaded) / len(loaded) if loaded else 0.0
        log(
            f"        CLAUSE loaded_frac >= LOADED_FRAC={B.LOADED_FRAC:.2f} over steps {i}-{peak}: got "
            f"{lf:.3f} ({sum(loaded)}/{len(loaded)}) -> {'PASS' if lf >= B.LOADED_FRAC else 'FAIL'}"
        )
        # return clause, evaluated regardless of the earlier verdicts
        exit_idx = None
        for k in range(peak + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < ret:
                exit_idx = k
                break
        tail = [k for k in range(peak + 1, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
        closest = min(tail, key=lambda k: abs(tr[k].gamma - g0)) if tail else None
        log(
            f"        CLAUSE return |gamma-g0| < YOYO_GAMMA_RETURN_DEG={B.YOYO_GAMMA_RETURN_DEG:.0f} "
            f"within YOYO_RETURN_S={B.YOYO_RETURN_S:.0f} s: "
            + (
                f"exit at step {exit_idx} (t {tr[exit_idx].t:.3f} s) -> PASS"
                if exit_idx is not None
                else (
                    f"never; closest approach to g0 in the tail is "
                    f"{abs(tr[closest].gamma - g0) * DEG:.2f} deg at step {closest} "
                    f"(t {tr[closest].t:.3f} s) -> FAIL by "
                    f"{abs(tr[closest].gamma - g0) * DEG - B.YOYO_GAMMA_RETURN_DEG:.2f} deg"
                    if closest is not None
                    else "no tail -> FAIL"
                )
            )
        )
    # how much of the trace can NEVER be an entry
    if len(edges) <= 1:
        log(
            "     => on this trace the yo-yo detector has AT MOST ONE shot, anchored at the tick"
            " where closure first exceeds 120 m/s; any yo-yo flown later in a sustained hot pass"
            " cannot produce an entry candidate at all."
        )
    log(
        f"     gamma over the whole trace: {tr[0].gamma * DEG:+.2f} deg at step 0 -> "
        f"{tr[-1].gamma * DEG:+.2f} deg; min {min(s.gamma for s in tr) * DEG:+.2f}, "
        f"max {max(s.gamma for s in tr) * DEG:+.2f} deg"
    )


# ==========================================================================================
# D3 cascade
# ==========================================================================================
def audit_d3(tr, log, tag) -> None:
    n = len(tr)
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    log("")
    log(f"   D3 turning_room  ({tag})   -- bfm_detect.py:462-499")
    log(
        f"     hold mask |chi| > CHI_ROOM_DEG={B.CHI_ROOM_DEG:.0f} AND nz > {B.NZ_LOADED:.0f}: "
        f"{sum(hold)}/{n} ticks; runs -> {runs_of(hold)}"
    )
    log(
        f"       {'run':>11} {'dur_s':>6} {'chiPk':>6} {'nzPk':>5} {'side':>6} {'cpa_i':>8} "
        f"{'cpa_j':>8} {'built':>6} {'cashed':>7} {'verdict':>16}"
    )
    for i, j in runs_of(hold):
        dur = tr[j].t - tr[i].t
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        built = tr[j].cpa_miss > tr[i].cpa_miss
        after = [k for k in range(j + 1, n) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
        cashed = bool(after) and min(tr[k].cpa_miss for k in after) < tr[j].cpa_miss
        fails = [
            nm
            for nm, ok in (
                ("dur", dur >= B.CHI_MIN_HOLD_S),
                ("side", len(sides) == 1),
                ("built", built),
                ("cashed", cashed),
            )
            if not ok
        ]
        log(
            f"       {f'{i}-{j}':>11} {dur:>6.3f} "
            f"{max(abs(tr[k].chi) for k in range(i, j + 1)) * DEG:>6.1f} "
            f"{max(tr[k].nz for k in range(i, j + 1)):>5.2f} "
            f"{('one' if len(sides) == 1 else 'BOTH'):>6} {tr[i].cpa_miss:>8.1f} "
            f"{tr[j].cpa_miss:>8.1f} {str(built):>6} {str(cashed):>7} "
            f"{('FAIL ' + ','.join(fails)) if fails else 'PASS':>16}"
        )
    cp = [s.cpa_miss for s in tr]
    dcp = np.diff(cp)
    log(
        f"     cpa_miss: {cp[0]:.1f} m at step 0 -> {cp[-1]:.1f} m at step {n - 1}; max {max(cp):.1f} m "
        f"at step {int(np.argmax(cp))}; ticks where cpa_miss RISES: {int((dcp > 0).sum())}/{len(dcp)}"
    )


# ==========================================================================================
# chi conditioning
# ==========================================================================================
def audit_chi_valid(rows, tr, log, tag) -> None:
    n = len(tr)
    valid = [s.chi_valid for s in tr]
    log("")
    log(f"   chi_valid conditioning  ({tag})   -- bfm_detect.py:127-178, gate at line 168")
    log(
        f"     chi_valid True on {sum(valid)}/{n} ticks ({sum(valid) / n * 100:.2f}%);  "
        f"False on {n - sum(valid)}"
    )
    # the ACTUAL quantity the gate tests: the angle between the LOS and the VELOCITY vector
    me_p = np.array([[float(r["pos_e_m"]), float(r["pos_n_m"]), float(r["pos_u_m"])] for r in rows])
    me_v = np.array(
        [[float(r["vel_e_ms"]), float(r["vel_n_ms"]), float(r["vel_u_ms"])] for r in rows]
    )
    op_p = np.array(
        [[float(r["opp_pos_e_m"]), float(r["opp_pos_n_m"]), float(r["opp_pos_u_m"])] for r in rows]
    )
    ata_nose = np.array([float(r["ata_deg"]) for r in rows])
    los = op_p - me_p
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    vh = me_v / np.linalg.norm(me_v, axis=1, keepdims=True)
    cos_v = np.clip((los * vh).sum(axis=1), -1.0, 1.0)
    ata_v = np.degrees(np.arccos(cos_v))  # LOS-to-VELOCITY angle: what the gate actually reads
    log(
        f"     the gate is |los_perp| < sin(CHI_VALID_ATA_DEG={B.CHI_VALID_ATA_DEG:.0f} deg), i.e. the "
        "LOS-to-VELOCITY angle < 5 deg. That is NOT `rel.ata` (measured about the NOSE)."
    )
    log(
        f"     LOS-to-VELOCITY angle: min {ata_v.min():.2f} deg at step {int(ata_v.argmin())}, "
        f"median {np.median(ata_v):.2f}, max {ata_v.max():.2f} deg"
    )
    log(
        f"     nose-referenced rel.ata:  min {ata_nose.min():.2f} deg at step {int(ata_nose.argmin())},"
        f" median {np.median(ata_nose):.2f}, max {ata_nose.max():.2f} deg"
    )
    below5_v = int((ata_v < B.CHI_VALID_ATA_DEG).sum())
    below5_n = int((ata_nose < B.CHI_VALID_ATA_DEG).sum())
    log(
        f"     ticks with LOS-to-VELOCITY < 5 deg: {below5_v}  (these SHOULD be invalid);  "
        f"ticks with nose-ata < 5 deg: {below5_n}"
    )
    log(
        f"     ticks reported chi_valid=True while nose-ata < 5 deg: "
        f"{sum(1 for k in range(n) if valid[k] and ata_nose[k] < B.CHI_VALID_ATA_DEG)}"
        "   <- the prior 'valid near boresight' observation, explained: chi is DEFINED about the"
        " velocity vector, and alpha separates the two references."
    )

    # numeric conditioning: perturb the LOS by 1 deg and measure the resulting |dchi|
    def chi_of(p_me, v_me, p_op):
        up = np.array([0.0, 0.0, 1.0])
        vhat = v_me / np.linalg.norm(v_me)
        los = p_op - p_me
        los = los / np.linalg.norm(los)
        u_ref = up - float(up @ vhat) * vhat
        u_ref = u_ref / np.linalg.norm(u_ref)
        r_ref = np.cross(vhat, u_ref)
        lp = los - float(los @ vhat) * vhat
        return math.atan2(float(lp @ r_ref), float(lp @ u_ref))

    def cond_deg(k, eps_deg=1.0):
        """max |d phi_los| (deg) from rotating the LOS by eps_deg in the two reference directions."""
        p_me, v_me, p_op = me_p[k], me_v[k], op_p[k]
        base = chi_of(p_me, v_me, p_op)
        rng = float(np.linalg.norm(p_op - p_me))
        vhat = v_me / np.linalg.norm(v_me)
        up = np.array([0.0, 0.0, 1.0])
        u_ref = up - float(up @ vhat) * vhat
        u_ref = u_ref / np.linalg.norm(u_ref)
        r_ref = np.cross(vhat, u_ref)
        worst = 0.0
        for d in (u_ref, r_ref, -u_ref, -r_ref):
            p2 = p_op + d * (rng * math.tan(math.radians(eps_deg)))
            worst = max(worst, abs(B._wrap_pi(chi_of(p_me, v_me, p2) - base)) * DEG)
        return worst

    log(
        "     CONDITIONING: |d chi| produced by displacing the bandit 1.0 deg of arc "
        "(the phi_los half of chi only; the lift half is exact):"
    )
    k_mx = max(range(n), key=lambda k: abs(tr[k].chi))
    probe = sorted({0, k_mx, int(ata_v.argmin()), n - 1, n // 2})
    for k in probe:
        log(
            f"       step {k:>4} t {tr[k].t:>6.3f} s  chi {tr[k].chi * DEG:>+7.1f} deg  "
            f"ata_v {ata_v[k]:>6.2f} deg  nz {tr[k].nz:>+5.2f} g  valid {tr[k].chi_valid}  "
            f"|dchi|/1deg = {cond_deg(k):>6.2f} deg  (1/sin(ata_v) bound "
            f"{1.0 / max(math.sin(math.radians(ata_v[k])), 1e-9):>7.1f})"
        )
    log(
        f"       max |chi| tick is step {k_mx}: ata_v {ata_v[k_mx]:.2f} deg -> "
        f"{'WELL-CONDITIONED' if ata_v[k_mx] > 20 else 'DEGENERATE'}"
    )


# ==========================================================================================
# threshold sensitivity (D6-style probe; globals restored)
# ==========================================================================================
def sensitivity(tr, log, tag) -> None:
    log("")
    log(f"   THRESHOLD SENSITIVITY  ({tag}) -- monkeypatched module globals, restored after")
    saved = {k: getattr(B, k) for k in dir(B) if k.isupper()}
    try:
        log(f"     {'CHI_OUT':>8} {'HOLD_S':>7} {'NZ':>5} -> {'D1':>3}")
        for chi_out in (60.0, 50.0, 45.0, 40.0, 30.0):
            for hold_s in (0.5, 0.3, 0.2, 0.125):
                B.CHI_OUT_DEG = chi_out
                B.CHI_MIN_HOLD_S = hold_s
                d1 = len(B.detect_displacement_rolls(tr))
                log(f"     {chi_out:>8.0f} {hold_s:>7.3f} {B.NZ_LOADED:>5.1f} -> {d1:>3}")
        for k, v in saved.items():
            setattr(B, k, v)
        log(f"     {'SWING':>8} {'RET_DEG':>7} {'RET_S':>6} -> {'D2':>3} {'D2c':>4}")
        for sw in (30.0, 25.0, 20.0):
            for rd in (10.0, 20.0, 30.0):
                for rs in (6.0, 12.0):
                    B.YOYO_GAMMA_SWING_DEG = sw
                    B.YOYO_GAMMA_RETURN_DEG = rd
                    B.YOYO_RETURN_S = rs
                    d2 = B.detect_yoyos(tr)
                    log(
                        f"     {sw:>8.0f} {rd:>7.0f} {rs:>6.0f} -> {len(d2):>3} "
                        f"{sum(1 for e in d2 if e.detail.get('clean')):>4}"
                    )
        for k, v in saved.items():
            setattr(B, k, v)
        # does D2 fire if the rising-edge gate is replaced by "any tick above the band"?
        log("     rising-edge gate replaced by ANY tick above the closure band (structural probe):")
        n = len(tr)
        found = []
        sw = math.radians(B.YOYO_GAMMA_SWING_DEG)
        ex = math.radians(B.YOYO_GAMMA_EXCURSION_DEG)
        rt = math.radians(B.YOYO_GAMMA_RETURN_DEG)
        for i in range(n):
            if tr[i].closure <= B.CLOSURE_OVERSHOOT:
                continue
            g0 = tr[i].gamma
            win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
            if len(win) < 3:
                continue
            peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
            if abs(tr[peak].gamma - g0) < sw or abs(tr[peak].gamma) < ex:
                continue
            loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
            if not loaded or sum(loaded) / len(loaded) < B.LOADED_FRAC:
                continue
            ok = any(
                abs(tr[k].gamma - g0) < rt
                for k in range(peak + 1, n)
                if tr[k].t - tr[i].t <= B.YOYO_RETURN_S
            )
            if ok:
                found.append(i)
        log(
            f"       entry ticks that would pass ALL of D2's shape clauses: {len(found)}"
            + (f" -> first at step {found[0]} (t {tr[found[0]].t:.3f} s)" if found else "")
        )
    finally:
        for k, v in saved.items():
            setattr(B, k, v)
        assert B.CHI_OUT_DEG == 60.0 and B.YOYO_GAMMA_SWING_DEG == 30.0, "globals not restored"
    log("     globals restored (CHI_OUT_DEG 60.0, YOYO_GAMMA_SWING_DEG 30.0) -- verified")


# ==========================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-csv", default="runs/d3b/replay_kill_seed62.csv")
    ap.add_argument("--nokill-csv", default="runs/d3b/replay_nokill_seed28.csv")
    ap.add_argument("--verify-reflight", type=int, default=62)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    fh = open(args.log, "w") if args.log else None

    def log(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    log("#" * 100)
    log("#  D3b DETECTOR AUDIT — clause-level instrumentation of D1/D2/D2c/D3")
    log("#" * 100)
    log("")
    log("   THRESHOLDS AS SHIPPED (bfm_detect.py:66-91)")
    for k in (
        "CHI_INPLANE_DEG",
        "CHI_OUT_DEG",
        "CHI_ROOM_DEG",
        "CHI_RETURN_DEG",
        "CHI_MIN_HOLD_S",
        "CHI_RETURN_S",
        "CHI_LOOKBACK_S",
        "NZ_LOADED",
        "LOADED_FRAC",
        "CLOSURE_OVERSHOOT",
        "YOYO_GAMMA_SWING_DEG",
        "YOYO_GAMMA_EXCURSION_DEG",
        "YOYO_RETURN_S",
        "YOYO_GAMMA_RETURN_DEG",
        "YOYO_EXIT_RNG_M",
        "YOYO_ASPECT_SLIP_DEG",
        "ASPECT_GAIN_DEG",
        "BANK_COMMIT_DEG",
        "CHI_VALID_ATA_DEG",
        "LETHAL_RANGE_M",
    ):
        log(f"     {k:<28} = {getattr(B, k)}")

    for tag, path in (("KILL seed 62", args.kill_csv), ("NOKILL seed 28", args.nokill_csv)):
        rows = load_rows(path)
        tr = trace_from_rows(rows)
        log("")
        log("=" * 100)
        log(f"  {tag}   {os.path.abspath(path)}   {len(tr)} ticks, {tr[-1].t:.3f} s")
        log("=" * 100)
        if tag.startswith("KILL"):
            log("")
            log("   FIDELITY: reconstructed-from-CSV trace vs a fresh re-flight")
            verify_against_reflight(rows, tr, args.verify_reflight, log)
        det = B.detect_all(tr)
        cm = det["commitment"]
        log("")
        log(
            f"   detect_all: D1={len(det['displacement_rolls'])} D2={len(det['yoyos'])} "
            f"D2c={len(det['yoyos_clean'])} D3={len(det['turning_room'])} "
            f"lethal_ticks={det['lethal_ticks']}"
        )
        log(
            f"   D4: bank hold {cm['longest_bank_hold_s']:.3f} s (dalt {cm['bank_hold_dalt_m']:+.0f} m), "
            f"chi hold {cm['longest_chi_hold_s']:.3f} s (dalt {cm['chi_hold_dalt_m']:+.0f} m), "
            f"valid_frac {cm['valid_frac']:.4f}, chi_hist {cm['chi_hist']}"
        )
        audit_d1(tr, log, tag)
        audit_d2(tr, log, tag)
        audit_d3(tr, log, tag)
        audit_chi_valid(rows, tr, log, tag)
        sensitivity(tr, log, tag)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
