"""Advocacy audit: is the D3b merge1500 kill trajectory a nameable BFM?

Pure post-hoc analysis of the two per-step dumps produced by _diag_d3b_replay.py. No sim, no
re-flight: every number is a function of the CSV columns, and every derived column is labelled
DERIVED with the arithmetic that produced it. numpy only (no pandas in the image).

Standing evidence rules obeyed:
  * no |chi| median anywhere; chi appears ONLY as load-gated hold durations / band residence
  * every hold priced in altitude (alt_m, the geodetic column `geometry` itself uses)
  * every row carries a step index and a time
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0  # ACTION_REPEAT / RATE_HZ seconds per control step

NZ_LOADED = 3.0
CHI_INPLANE = 20.0
CHI_ROOM = 45.0
CHI_OUT = 60.0
CHI_MIN_HOLD_S = 0.5
YOYO_SWING = 30.0
YOYO_EXCUR = 15.0
YOYO_RETURN_S = 6.0
YOYO_GAMMA_RETURN = 10.0
CLOSURE_OVERSHOOT = 120.0


def runs(mask) -> list[tuple[int, int]]:
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


def hdr(s: str) -> None:
    print("\n" + "=" * 104)
    print("  " + s)
    print("=" * 104)


def load(path: str) -> dict:
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    v = np.stack([d["vel_e_ms"], d["vel_n_ms"], d["vel_u_ms"]], axis=1)
    sp = np.linalg.norm(v, axis=1)
    vh = v / sp[:, None]
    dot = np.clip(np.sum(vh[1:] * vh[:-1], axis=1), -1.0, 1.0)
    step_turn = np.degrees(np.arccos(dot))
    d["turn_rate_dps"] = np.r_[step_turn[0] / DT, step_turn / DT]  # DERIVED |dv_hat/dt|
    d["turn_cum_deg"] = np.r_[0.0, np.cumsum(step_turn)]  # DERIVED
    d["Ps_ms"] = np.r_[0.0, np.diff(d["es_m"]) / DT]  # DERIVED dEs/dt
    d["phi_rate_dps"] = np.r_[0.0, np.diff(d["phi_deg"]) / DT]  # DERIVED
    d["chi_rate_dps"] = np.r_[0.0, np.diff(d["chi_deg"]) / DT]  # DERIVED
    d["turn_radius_m"] = sp / np.radians(np.maximum(d["turn_rate_dps"], 1e-6))  # DERIVED vt/omega
    d["_n"] = len(rows)
    return d


def loaded_holds(d: dict, label: str) -> None:
    hdr(f"{label}: LOAD-GATED HOLDS (nz > {NZ_LOADED} g) — each priced in altitude")
    nz = d["nz_g"]
    print(
        f"{'i0':>4} {'i1':>4} {'t0':>6} {'t1':>6} {'dur_s':>6} {'dalt_m':>8} {'dvt':>6} {'dEs_m':>7} "
        f"{'turn_d':>7} {'rate':>6} {'radius':>7} {'dgam':>7} {'nzMin':>6} {'nzMax':>6} "
        f"{'phi0':>7} {'phi1':>7} {'chi0':>7} {'chi1':>7} {'rng0':>6} {'rng1':>6}"
    )
    for i, j in runs(nz > NZ_LOADED):
        dur = d["t_s"][j] - d["t_s"][i]
        turn = d["turn_cum_deg"][j] - d["turn_cum_deg"][i]
        print(
            f"{i:4d} {j:4d} {d['t_s'][i]:6.3f} {d['t_s'][j]:6.3f} {dur:6.3f} "
            f"{d['alt_m'][j] - d['alt_m'][i]:8.1f} {d['vt_ms'][j] - d['vt_ms'][i]:6.1f} "
            f"{d['es_m'][j] - d['es_m'][i]:7.1f} {turn:7.1f} {turn / max(dur, 1e-6):6.1f} "
            f"{np.median(d['turn_radius_m'][i : j + 1]):7.0f} "
            f"{d['gamma_deg'][j] - d['gamma_deg'][i]:7.1f} {nz[i : j + 1].min():6.2f} "
            f"{nz[i : j + 1].max():6.2f} {d['phi_deg'][i]:7.1f} {d['phi_deg'][j]:7.1f} "
            f"{d['chi_deg'][i]:7.1f} {d['chi_deg'][j]:7.1f} {d['rng_m'][i]:6.0f} {d['rng_m'][j]:6.0f}"
        )
        if dur < CHI_MIN_HOLD_S:
            continue
        ach = np.abs(d["chi_deg"][i : j + 1])
        alt = d["alt_m"][i : j + 1]
        tt = d["t_s"][i : j + 1]
        for lo, hi, nm in [
            (0.0, CHI_INPLANE, f"IN-PLANE |chi|<{CHI_INPLANE:.0f}"),
            (CHI_INPLANE, CHI_ROOM, f"oblique {CHI_INPLANE:.0f}-{CHI_ROOM:.0f}"),
            (CHI_ROOM, CHI_OUT, f"D3 band {CHI_ROOM:.0f}-{CHI_OUT:.0f}"),
            (CHI_OUT, 181.0, f"D1 band |chi|>{CHI_OUT:.0f}"),
        ]:
            m = (ach >= lo) & (ach < hi)
            if not m.any():
                print(f"        band {nm:<22}   0 steps = 0.000 s")
                continue
            lr = runs(m)
            lg = max(lr, key=lambda r: r[1] - r[0])
            print(
                f"        band {nm:<22} {int(m.sum()):3d} steps = {m.sum() * DT:5.3f} s  "
                f"alt price {alt[m][-1] - alt[m][0]:+8.1f} m  longest run "
                f"{lg[1] - lg[0] + 1:3d} steps = {(lg[1] - lg[0] + 1) * DT:5.3f} s "
                f"(steps {i + lg[0]}-{i + lg[1]}, t {tt[lg[0]]:.3f}-{tt[lg[1]]:.3f} s, "
                f"alt {alt[lg[0]]:.1f} -> {alt[lg[1]]:.1f} m)"
            )


def bank_holds(d: dict, label: str) -> None:
    hdr(f"{label}: BANK holds priced (|phi|>90 => the lift vector is BELOW the horizon)")
    for bar in (45.0, 90.0, 110.0):
        m = (np.abs(d["phi_deg"]) > bar) & (d["nz_g"] > NZ_LOADED)
        if not m.any():
            print(f"  |phi|>{bar:5.1f} AND nz>3 : 0 steps")
            continue
        for i, j in runs(m):
            if j - i < 2:
                continue
            print(
                f"  |phi|>{bar:5.1f} AND nz>3 : steps {i:3d}-{j:3d} t {d['t_s'][i]:6.3f}-"
                f"{d['t_s'][j]:6.3f} ({(j - i + 1) * DT:5.3f} s) alt {d['alt_m'][i]:7.1f} -> "
                f"{d['alt_m'][j]:7.1f} ({d['alt_m'][j] - d['alt_m'][i]:+7.1f} m) vt "
                f"{d['vt_ms'][i]:5.1f} -> {d['vt_ms'][j]:5.1f} Es {d['es_m'][i]:7.1f} -> "
                f"{d['es_m'][j]:7.1f} ({d['es_m'][j] - d['es_m'][i]:+6.1f} m) turn "
                f"{d['turn_cum_deg'][j] - d['turn_cum_deg'][i]:6.1f} deg  gamma "
                f"{d['gamma_deg'][i]:+6.1f} -> {d['gamma_deg'][j]:+6.1f}"
            )


def roll_in(d: dict) -> None:
    hdr("THE ROLL-IN: how fast the lift vector was placed, and that it then STAYED")
    print(
        f"{'step':>4} {'t_s':>6} {'phi':>8} {'phi_rate':>9} {'ail':>6} {'nz':>5} {'alpha':>6} "
        f"{'chi':>8} {'chi_rate':>9} {'ata':>6} {'aspect':>7} {'rng':>6} {'alt':>7} {'gamma':>7}"
    )
    for k in list(range(0, 56, 4)) + [60, 72, 84, 96]:
        print(
            f"{int(d['step'][k]):4d} {d['t_s'][k]:6.3f} {d['phi_deg'][k]:8.2f} "
            f"{d['phi_rate_dps'][k]:9.1f} {d['ail'][k]:6.2f} {d['nz_g'][k]:5.2f} "
            f"{d['alpha_deg'][k]:6.2f} {d['chi_deg'][k]:8.2f} {d['chi_rate_dps'][k]:9.1f} "
            f"{d['ata_deg'][k]:6.2f} {d['aspect_deg'][k]:7.2f} {d['rng_m'][k]:6.0f} "
            f"{d['alt_m'][k]:7.1f} {d['gamma_deg'][k]:7.2f}"
        )
    a, b = 36, 96
    print(
        f"\n  ESTABLISHED TURN, steps {a}-{b} (t {d['t_s'][a]:.3f}-{d['t_s'][b]:.3f} s):\n"
        f"    phi  {d['phi_deg'][a : b + 1].min():7.2f} .. {d['phi_deg'][a : b + 1].max():7.2f} deg"
        f"   |phi_rate| max {np.abs(d['phi_rate_dps'][a : b + 1]).max():6.2f} deg/s\n"
        f"    chi  {d['chi_deg'][a : b + 1].min():7.2f} .. {d['chi_deg'][a : b + 1].max():7.2f} deg"
        f"   chi_rate mean {d['chi_rate_dps'][a : b + 1].mean():+6.2f} deg/s\n"
        f"    nz   {d['nz_g'][a : b + 1].min():5.2f} .. {d['nz_g'][a : b + 1].max():5.2f} g"
        f"   alpha {d['alpha_deg'][a : b + 1].min():5.2f} .. {d['alpha_deg'][a : b + 1].max():5.2f} deg\n"
        f"    ail  {d['ail'][a : b + 1].min():6.3f} .. {d['ail'][a : b + 1].max():6.3f}"
        f"   elev {d['elev'][a : b + 1].min():6.3f} .. {d['elev'][a : b + 1].max():6.3f}\n"
        f"    => bank HELD to +-{(d['phi_deg'][a : b + 1].max() - d['phi_deg'][a : b + 1].min()) / 2:.2f} deg"
        f" while chi drifts {d['chi_deg'][b] - d['chi_deg'][a]:+.1f} deg: the LOS moved, not the lift vector."
    )


def d2_reconstruction(d: dict, label: str) -> None:
    hdr(f"{label}: D2 (yo-yo) clause-by-clause — where exactly the detector says no")
    clo, t = d["closure_ms"], d["t_s"]
    gam = np.radians(d["gamma_deg"])
    n = d["_n"]
    entries = [
        i
        for i in range(n)
        if clo[i] > CLOSURE_OVERSHOOT and (i == 0 or clo[i - 1] <= CLOSURE_OVERSHOOT)
    ]
    print(
        f"  closure > {CLOSURE_OVERSHOOT} m/s rising edges (D2 entry gate) = steps {entries[:8]}\n"
        f"  closure at step 0 = {clo[0]:.1f} m/s, so step 0 IS an entry via the `i == 0` branch "
        f"(bfm_detect.py:411)"
    )
    for i in entries[:4]:
        g0 = gam[i]
        w = [k for k in range(i, n) if t[k] - t[i] <= YOYO_RETURN_S]
        peak = max(w, key=lambda k: abs(gam[k] - g0))
        swing = math.degrees(abs(gam[peak] - g0))
        excur = math.degrees(abs(gam[peak]))
        loaded = [d["nz_g"][k] > NZ_LOADED for k in range(i, peak + 1)]
        frac = sum(loaded) / len(loaded)
        ret = [k for k in w if k > peak and abs(gam[k] - g0) < math.radians(YOYO_GAMMA_RETURN)]
        print(
            f"\n  entry step {i:3d} t {t[i]:6.3f} s  gamma0 {math.degrees(g0):+7.3f} deg  "
            f"window steps {w[0]}-{w[-1]} (t <= {t[i] + YOYO_RETURN_S:.3f} s)\n"
            f"    peak       step {peak:3d} t {t[peak]:6.3f} s gamma {math.degrees(gam[peak]):+7.2f} deg "
            f"alt {d['alt_m'][peak]:7.1f} m\n"
            f"    (a) swing      {swing:6.2f} deg vs bar {YOYO_SWING:.1f} -> "
            f"{'PASS' if swing >= YOYO_SWING else 'FAIL'} (shortfall {YOYO_SWING - swing:+6.2f} deg)\n"
            f"    (b) excursion  {excur:6.2f} deg vs bar {YOYO_EXCUR:.1f} -> "
            f"{'PASS' if excur >= YOYO_EXCUR else 'FAIL'}\n"
            f"    (c) loadedfrac {frac:6.3f} ({sum(loaded)}/{len(loaded)} steps) vs bar 0.5 -> "
            f"{'PASS' if frac >= 0.5 else 'FAIL'}\n"
            f"    (d) return within {YOYO_GAMMA_RETURN:.0f} deg of gamma0 inside window -> "
            f"{('PASS at step ' + str(ret[0])) if ret else 'FAIL (never returns)'}\n"
            f"        gamma at window end (step {w[-1]}, t {t[w[-1]]:.3f} s) = "
            f"{math.degrees(gam[w[-1]]):+7.2f} deg"
        )


def geometry_arc(d: dict) -> None:
    hdr("WHAT THE TURN BOUGHT: pursuit curve, LOS rate, turning room, heading crossing")
    cols = [
        "step",
        "t_s",
        "seg_idx",
        "rng_m",
        "ata_deg",
        "ata_az_deg",
        "aspect_deg",
        "hca_deg",
        "closure_ms",
        "los_rate_dps",
        "cpa_miss_m",
        "lay_miss_m",
        "alpha_req_deg",
        "turn_cum_deg",
        "turn_rate_dps",
        "turn_radius_m",
        "es_m",
        "es_rel_m",
        "Ps_ms",
        "alt_m",
        "vt_ms",
        "gamma_deg",
        "phi_deg",
        "nz_g",
        "chi_deg",
    ]
    print(" ".join(f"{c[:9]:>9}" for c in cols))
    for k in sorted(set(list(range(0, d["_n"], 12)) + [d["_n"] - 1])):
        print(" ".join(f"{d[c][k]:9.2f}" for c in cols))
    print("\n  monotonicity checks:")
    for c in ("cpa_miss_m", "rng_m", "turn_cum_deg", "es_m", "es_rel_m"):
        v = d[c]
        dv = np.diff(v)
        print(
            f"    {c:14} first {v[0]:9.1f} last {v[-1]:9.1f}  up {int((dv > 0).sum())}/{len(dv)} "
            f"down {int((dv < 0).sum())}/{len(dv)}  max {v.max():9.1f} at step {int(v.argmax())}"
        )
    h = np.abs(d["hca_deg"])
    print(
        f"    hca_deg        first {d['hca_deg'][0]:9.2f} last {d['hca_deg'][-1]:9.2f}  |hca| min "
        f"{h.min():.2f} at step {int(h.argmin())} (t {d['t_s'][int(h.argmin())]:.3f} s)"
    )
    lr = d["los_rate_dps"]
    print(
        f"    los_rate_dps   first {lr[0]:6.2f} min {lr.min():6.2f} at step {int(lr.argmin())} "
        f"(t {d['t_s'][int(lr.argmin())]:.3f} s) last {lr[-1]:6.2f}"
    )
    print(
        f"    turn_rate_dps  max {d['turn_rate_dps'].max():6.2f} at step "
        f"{int(d['turn_rate_dps'].argmax())}; median over steps 36-96 "
        f"{np.median(d['turn_rate_dps'][36:97]):6.2f} deg/s"
    )


def opponent(d: dict, label: str) -> None:
    hdr(f"{label}: DID THE BANDIT MANEUVER? (a BFM claim needs an adversary that moves)")
    n = d["_n"]
    print(
        f"  opp_nz_g  min {d['opp_nz_g'].min():6.3f} max {d['opp_nz_g'].max():6.3f}  "
        f"steps opp_nz>{NZ_LOADED}: {int((d['opp_nz_g'] > NZ_LOADED).sum())}/{n}  "
        f"steps opp_nz>1.5: {int((d['opp_nz_g'] > 1.5).sum())}/{n}"
    )
    print(
        f"  opp_phi   min {d['opp_phi_deg'].min():8.2f} max {d['opp_phi_deg'].max():8.2f} deg  "
        f"steps |opp_phi|>45: {int((np.abs(d['opp_phi_deg']) > 45).sum())}/{n}"
    )
    psi = d["opp_psi_deg"]
    print(
        f"  opp_psi   first {psi[0]:8.3f} last {psi[-1]:8.3f} deg  |change| {abs(psi[-1] - psi[0]):7.3f}"
    )
    print(
        f"  opp_alt   {d['opp_alt_m'][0]:8.1f} -> {d['opp_alt_m'][-1]:8.1f} m  opp_vt "
        f"{d['opp_vt_ms'][0]:6.1f} -> {d['opp_vt_ms'][-1]:6.1f} m/s  opp_Es "
        f"{d['opp_es_m'][0]:7.1f} -> {d['opp_es_m'][-1]:7.1f} m"
    )
    print(
        f"  opp_ammo  {int(d['opp_ammo'][0])} -> {int(d['opp_ammo'][-1])} (fired "
        f"{int(d['opp_ammo'][0] - d['opp_ammo'][-1])})  my health {d['health'][0]:.1f} -> "
        f"{d['health'][-1]:.1f}"
    )
    m = d["opp_nz_g"] > NZ_LOADED
    if m.any():
        i = int(np.argmax(m))
        print(
            f"  first bandit load > 3 g: step {i} t {d['t_s'][i]:.3f} s rng {d['rng_m'][i]:.0f} m "
            f"opp_phi {d['opp_phi_deg'][i]:.1f} deg"
        )
    m2 = np.abs(d["opp_phi_deg"]) > 15.0
    if m2.any():
        i = int(np.argmax(m2))
        print(
            f"  first |opp_phi| > 15 deg: step {i} t {d['t_s'][i]:.3f} s rng {d['rng_m'][i]:.0f} m"
        )
    else:
        print("  |opp_phi| never exceeds 15 deg on this trace")


def fire_window(d: dict) -> None:
    hdr("THE SHOT: trigger window and the geometry when it opened")
    f = np.where(d["fire"] > 0)[0]
    print(
        f"  fire rows {len(f)} = {len(f) * DT:.3f} s, steps {f[0]}-{f[-1]}, t {d['t_s'][f[0]]:.3f}-"
        f"{d['t_s'][f[-1]]:.3f} s;  lethal rows {int(d['lethal'].sum())} = "
        f"{d['lethal'].sum() * DT:.3f} s"
    )
    i = int(f[0])
    print(
        f"  FIRST trigger step {i} t {d['t_s'][i]:.3f}: rng {d['rng_m'][i]:.1f} m aspect "
        f"{d['aspect_deg'][i]:.2f} ata {d['ata_deg'][i]:.2f} hca {d['hca_deg'][i]:.2f} closure "
        f"{d['closure_ms'][i]:+.1f} m/s lay_miss {d['lay_miss_m'][i]:.2f} m alpha_req "
        f"{d['alpha_req_deg'][i]:.2f} alpha {d['alpha_deg'][i]:.2f} nz {d['nz_g'][i]:.2f} phi "
        f"{d['phi_deg'][i]:.2f} gamma {d['gamma_deg'][i]:.2f} alt {d['alt_m'][i]:.1f} cpa_miss "
        f"{d['cpa_miss_m'][i]:.1f} m"
    )
    print(
        f"  bandit health {d['opp_health'][0]:.0f} -> {d['opp_health'][-1]:.0f}; my ammo "
        f"{int(d['ammo'][0])} -> {int(d['ammo'][-1])}"
    )


def main() -> None:
    k = load(KILL)
    print(f"KILL trace {KILL}: {k['_n']} rows, t 0.000-{k['t_s'][-1]:.3f} s, dt {DT * 1000:.3f} ms")
    roll_in(k)
    loaded_holds(k, "KILL seed 62")
    bank_holds(k, "KILL seed 62")
    geometry_arc(k)
    d2_reconstruction(k, "KILL seed 62")
    opponent(k, "KILL seed 62")
    fire_window(k)

    nk = load(NOKILL)
    hdr("NEGATIVE CONTROL — seed 28: identical commanded stream, all 6 segments, NO kill")
    print(f"  {nk['_n']} rows, t 0.000-{nk['t_s'][-1]:.3f} s")
    loaded_holds(nk, "NOKILL seed 28")
    d2_reconstruction(nk, "NOKILL seed 28")
    opponent(nk, "NOKILL seed 28")
    print(
        f"\n  seed 28: alt {nk['alt_m'][0]:.1f} -> {nk['alt_m'][-1]:.1f} m "
        f"({nk['alt_m'][-1] - nk['alt_m'][0]:+.1f}), Es {nk['es_m'][0]:.1f} -> {nk['es_m'][-1]:.1f} m, "
        f"turn_cum {nk['turn_cum_deg'][-1]:.1f} deg, rng min {nk['rng_m'].min():.0f} m"
    )


if __name__ == "__main__":
    main()
