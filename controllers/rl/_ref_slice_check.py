"""REFUTATION AUDIT of the 'it is a SLICE, not a level turn' claim.

Independent recomputation from the two CSV dumps. Nothing is reused from _adv_bfm_case.py:
every statistic is recomputed from the raw columns here.
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0
G = 9.80665
NZ_LOADED = 3.0


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def runs(mask):
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


def kin(d):
    """DERIVED kinematics, all from vel_[enu]_ms + alt_m + attitude."""
    v = np.stack([d["vel_e_ms"], d["vel_n_ms"], d["vel_u_ms"]], axis=1)
    sp = np.linalg.norm(v, axis=1)
    vh = v / sp[:, None]
    # total velocity-vector rotation per step: use the CROSS product norm (better conditioned
    # near dot==1 than arccos(dot)) and cross-check against arccos.
    cr = np.cross(vh[:-1], vh[1:])
    ang_cross = np.arcsin(np.clip(np.linalg.norm(cr, axis=1), -1.0, 1.0))
    ang_acos = np.arccos(np.clip(np.sum(vh[1:] * vh[:-1], axis=1), -1.0, 1.0))
    d["_step_turn_deg"] = np.degrees(ang_cross)
    d["_step_turn_acos_deg"] = np.degrees(ang_acos)
    d["turn_cum_deg"] = np.r_[0.0, np.cumsum(np.degrees(ang_acos))]  # matches the claim's recipe
    d["turn_cum_cross_deg"] = np.r_[0.0, np.cumsum(np.degrees(ang_cross))]
    # HEADING-only rotation (horizontal projection of the velocity vector)
    hd = np.degrees(np.arctan2(d["vel_e_ms"], d["vel_n_ms"]))
    dhd = np.diff(hd)
    dhd = (dhd + 180.0) % 360.0 - 180.0
    d["hdg_deg"] = hd
    d["hdg_cum_deg"] = np.r_[0.0, np.cumsum(np.abs(dhd))]
    d["hdg_rate_dps"] = np.r_[0.0, dhd / DT]
    # gamma-only rotation
    d["gam_cum_deg"] = np.r_[0.0, np.cumsum(np.abs(np.diff(d["gamma_deg"])))]
    d["turn_rate_dps"] = np.r_[0.0, np.degrees(ang_acos) / DT]
    d["Ps_ms"] = np.r_[0.0, np.diff(d["es_m"]) / DT]
    d["vt_chk"] = sp
    # LIFT-VECTOR UP COMPONENT, from Euler angles + alpha (lift is perpendicular to V in the
    # plane of symmetry: lift_hat = x_b*sin(alpha) - z_b*cos(alpha)).
    ph = np.radians(d["phi_deg"])
    th = np.radians(d["theta_deg"])
    al = np.radians(d["alpha_deg"])
    # NED down-components of the body axes
    xb_d = -np.sin(th)
    zb_d = np.cos(ph) * np.cos(th)
    lift_up = -(np.sin(al) * xb_d - np.cos(al) * zb_d)  # up = -down
    d["lift_up"] = lift_up
    d["lift_up_nozalpha"] = np.cos(ph) * np.cos(th)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


def window(d, i, j, tag):
    dur_steps = (j - i + 1) * DT
    dur_t = d["t_s"][j] - d["t_s"][i]
    print(f"\n  {tag}: steps {i}-{j}")
    print(f"    t {d['t_s'][i]:.3f} -> {d['t_s'][j]:.3f} s")
    print(f"    t[j]-t[i]          = {dur_t:.4f} s   <- the honest elapsed time")
    print(f"    (j-i+1)*DT         = {dur_steps:.4f} s   <- what the script printed")
    print(
        f"    alt_m  {d['alt_m'][i]:9.2f} -> {d['alt_m'][j]:9.2f}  ({d['alt_m'][j] - d['alt_m'][i]:+.2f} m)"
    )
    print(
        f"    vt_ms  {d['vt_ms'][i]:9.2f} -> {d['vt_ms'][j]:9.2f}  ({d['vt_ms'][j] - d['vt_ms'][i]:+.2f} m/s)"
    )
    print(
        f"    es_m   {d['es_m'][i]:9.2f} -> {d['es_m'][j]:9.2f}  ({d['es_m'][j] - d['es_m'][i]:+.2f} m)"
    )
    print(f"    gamma  {d['gamma_deg'][i]:9.2f} -> {d['gamma_deg'][j]:9.2f} deg")
    print(f"    phi    {d['phi_deg'][i]:9.2f} -> {d['phi_deg'][j]:9.2f} deg")
    print(
        f"    nz     min {d['nz_g'][i : j + 1].min():.3f} max {d['nz_g'][i : j + 1].max():.3f} mean {d['nz_g'][i : j + 1].mean():.3f}"
    )
    tot = d["turn_cum_deg"][j] - d["turn_cum_deg"][i]
    totx = d["turn_cum_cross_deg"][j] - d["turn_cum_cross_deg"][i]
    hcum = d["hdg_cum_deg"][j] - d["hdg_cum_deg"][i]
    gcum = d["gam_cum_deg"][j] - d["gam_cum_deg"][i]
    print(
        f"    TOTAL vel-vector rotation (arccos recipe) = {tot:.2f} deg  ({tot / dur_t:.2f} deg/s)"
    )
    print(f"    TOTAL vel-vector rotation (cross  recipe) = {totx:.2f} deg  (cross-check)")
    print(
        f"      of which HEADING (sum|dpsi_v|)          = {hcum:.2f} deg  ({hcum / dur_t:.2f} deg/s)"
    )
    print(
        f"      of which GAMMA   (sum|dgamma|)          = {gcum:.2f} deg  ({gcum / dur_t:.2f} deg/s)"
    )
    print(
        f"      net heading change {d['hdg_deg'][j] - d['hdg_deg'][i]:+.2f} deg, net gamma change {d['gamma_deg'][j] - d['gamma_deg'][i]:+.2f} deg"
    )
    # LEVEL-TURN COUNTERFACTUAL at the same nz, same speed: heading rate = g*sqrt(nz^2-1)/V
    nz = d["nz_g"][i : j + 1]
    vt = d["vt_ms"][i : j + 1]
    lvl = np.degrees(G * np.sqrt(np.maximum(nz**2 - 1.0, 0.0)) / vt)
    # what the overbanked turn actually delivers horizontally: nz*sin(phi)*g/(V*cos gamma)
    ph = np.radians(d["phi_deg"][i : j + 1])
    gm = np.radians(d["gamma_deg"][i : j + 1])
    horiz = np.degrees(G * nz * np.abs(np.sin(ph)) / (vt * np.maximum(np.cos(gm), 1e-6)))
    print(
        f"    LEVEL-TURN counterfactual heading rate at the SAME nz,V: mean {lvl.mean():.2f} deg/s (min {lvl.min():.2f} max {lvl.max():.2f})"
    )
    print(
        f"    analytic horizontal rate of the FLOWN bank              : mean {horiz.mean():.2f} deg/s"
    )
    print(
        f"    measured mean |heading rate| over the window            : {np.abs(d['hdg_rate_dps'][i + 1 : j + 1]).mean():.2f} deg/s"
    )
    # GRAVITY's share of the velocity-vector rotation
    grav_rate = np.degrees(G * np.cos(gm) / vt)
    print(
        f"    gravity-only rotation rate g*cos(gamma)/V: mean {grav_rate.mean():.2f} deg/s  => gravity share of the {tot / dur_t:.2f} deg/s total = {100 * grav_rate.mean() / (tot / dur_t):.1f}%"
    )
    print(
        f"    lift_up (Euler+alpha) min {d['lift_up'][i : j + 1].min():+.4f} max {d['lift_up'][i : j + 1].max():+.4f}  steps with lift_up<0: {int((d['lift_up'][i : j + 1] < 0).sum())}/{j - i + 1}"
    )
    print(
        f"    cpa_miss_m {d['cpa_miss_m'][i]:.1f} -> {d['cpa_miss_m'][j]:.1f}   rng_m {d['rng_m'][i]:.0f} -> {d['rng_m'][j]:.0f}"
    )
    print(
        f"    thr commanded {d['cmd_thr'][i]:.4f} .. {d['cmd_thr'][j]:.4f}   cmd_bank {d['cmd_bank_deg'][i]:.2f} .. {d['cmd_bank_deg'][j]:.2f}  cmd_load {d['cmd_load'][i]:.4f} .. {d['cmd_load'][j]:.4f}"
    )


def bank_holds(d, label):
    hdr(f"{label}: |phi|>bar AND nz>3 runs (the claim's own gate), recomputed")
    for bar in (45.0, 90.0, 110.0):
        m = (np.abs(d["phi_deg"]) > bar) & (d["nz_g"] > NZ_LOADED)
        rr = [(i, j) for (i, j) in runs(m) if j - i >= 2]
        print(f"\n  --- bar {bar} : {int(m.sum())} steps total, {len(rr)} runs of >=3 steps")
        for i, j in rr:
            window(d, i, j, f"|phi|>{bar:.0f} & nz>3")


def main():
    k = kin(load(KILL))
    nk = kin(load(NOKILL))
    print(f"KILL seed62: {k['_n']} rows, t 0 -> {k['t_s'][-1]:.3f} s")
    print(f"NOKILL seed28: {nk['_n']} rows, t 0 -> {nk['t_s'][-1]:.3f} s")

    hdr("PRECHECK: are the first 173 steps of seed 28 identical to seed 62?")
    cols = [
        "phi_deg",
        "nz_g",
        "alt_m",
        "vt_ms",
        "gamma_deg",
        "es_m",
        "ail",
        "elev",
        "thr_flown",
        "theta_deg",
    ]
    for c in cols:
        dv = np.abs(k[c] - nk[c][: k["_n"]])
        print(f"  {c:12} max|diff| over steps 0-{k['_n'] - 1} = {dv.max():.6e}")
    for c in ["opp_health", "health", "rng_m", "cpa_miss_m"]:
        dv = np.abs(k[c] - nk[c][: k["_n"]])
        print(
            f"  {c:12} max|diff| = {dv.max():.6e} (first differing step "
            f"{int(np.argmax(dv > 0)) if (dv > 0).any() else -1})"
        )

    bank_holds(k, "KILL seed 62")

    hdr("PHI PLATEAU over steps 36-96 (the claim's '113.08-119.38')")
    a, b = 36, 96
    print(
        f"  phi min {k['phi_deg'][a : b + 1].min():.4f} max {k['phi_deg'][a : b + 1].max():.4f} deg"
    )
    print(
        f"  steps 36-96: theta min {k['theta_deg'][a : b + 1].min():.2f} max {k['theta_deg'][a : b + 1].max():.2f}"
    )
    print(
        f"  lift_up over 36-96: min {k['lift_up'][a : b + 1].min():+.4f} max {k['lift_up'][a : b + 1].max():+.4f}"
    )
    print(
        f"  cos(phi)cos(theta) over 36-96: min {k['lift_up_nozalpha'][a : b + 1].min():+.4f} max {k['lift_up_nozalpha'][a : b + 1].max():+.4f}"
    )
    # when does phi first cross 90 / 110, and what is the whole-episode phi history
    for bar in (90.0, 110.0):
        m = np.abs(k["phi_deg"]) > bar
        idx = np.where(m)[0]
        print(
            f"  |phi|>{bar}: {len(idx)} steps, first {idx[0]} (t {k['t_s'][idx[0]]:.3f}), last {idx[-1]} (t {k['t_s'][idx[-1]]:.3f})"
        )

    hdr("WHOLE-EPISODE ALTITUDE / ENERGY LEDGER")
    for nm, d in (("KILL seed62", k), ("NOKILL seed28", nk)):
        print(
            f"  {nm}: alt {d['alt_m'][0]:.1f} -> {d['alt_m'][-1]:.1f} ({d['alt_m'][-1] - d['alt_m'][0]:+.1f} m)  "
            f"Es {d['es_m'][0]:.1f} -> {d['es_m'][-1]:.1f} ({d['es_m'][-1] - d['es_m'][0]:+.1f} m)  "
            f"vt {d['vt_ms'][0]:.1f} -> {d['vt_ms'][-1]:.1f}  turn_cum {d['turn_cum_deg'][-1]:.1f} deg  "
            f"hdg_cum {d['hdg_cum_deg'][-1]:.1f} deg"
        )

    hdr("NOKILL seed 28: the SAME |phi|>110 & nz>3 gate over the FULL 24 s")
    bank_holds(nk, "NOKILL seed 28")

    hdr("NOKILL seed 28: every nz>3 loaded hold, priced in altitude")
    for i, j in runs(nk["nz_g"] > NZ_LOADED):
        if j - i < 2:
            continue
        dur = nk["t_s"][j] - nk["t_s"][i]
        print(
            f"  steps {i:3d}-{j:3d} t {nk['t_s'][i]:6.3f}-{nk['t_s'][j]:6.3f} ({dur:6.3f} s) "
            f"alt {nk['alt_m'][i]:7.1f} -> {nk['alt_m'][j]:7.1f} ({nk['alt_m'][j] - nk['alt_m'][i]:+8.1f} m) "
            f"Es {nk['es_m'][j] - nk['es_m'][i]:+8.1f} m  vt {nk['vt_ms'][i]:5.1f}->{nk['vt_ms'][j]:5.1f}  "
            f"turn {nk['turn_cum_deg'][j] - nk['turn_cum_deg'][i]:6.1f} hdg {nk['hdg_cum_deg'][j] - nk['hdg_cum_deg'][i]:6.1f} "
            f"|phi|max {np.abs(nk['phi_deg'][i : j + 1]).max():6.1f} rng {nk['rng_m'][i]:5.0f}->{nk['rng_m'][j]:5.0f}"
        )

    hdr("SEG-BY-SEG on the NOKILL seed (segments 2-5 never ran on the kill)")
    for s in range(6):
        m = nk["seg_idx"] == s
        if not m.any():
            continue
        idx = np.where(m)[0]
        i, j = int(idx[0]), int(idx[-1])
        print(
            f"  seg {s}: steps {i:3d}-{j:3d} cmd_bank {nk['cmd_bank_deg'][i]:+8.2f} cmd_load {nk['cmd_load'][i]:.4f} "
            f"cmd_thr {nk['cmd_thr'][i]:.4f} | alt {nk['alt_m'][i]:7.1f}->{nk['alt_m'][j]:7.1f} ({nk['alt_m'][j] - nk['alt_m'][i]:+8.1f}) "
            f"Es {nk['es_m'][j] - nk['es_m'][i]:+8.1f} nz mean {nk['nz_g'][i : j + 1].mean():5.2f} "
            f"|phi| max {np.abs(nk['phi_deg'][i : j + 1]).max():6.1f} rng {nk['rng_m'][i]:5.0f}->{nk['rng_m'][j]:5.0f}"
        )

    hdr("SANITY: level-turn vs flown, tabulated across the plateau")
    print(
        f"{'step':>5}{'t':>7}{'phi':>8}{'theta':>8}{'nz':>6}{'vt':>7}{'gam':>8}{'liftup':>8}"
        f"{'hdgrate':>9}{'lvlrate':>9}{'totrate':>9}{'gravrate':>9}"
    )
    for kk in range(24, 112, 4):
        nz, vt = k["nz_g"][kk], k["vt_ms"][kk]
        gm = math.radians(k["gamma_deg"][kk])
        lvl = math.degrees(G * math.sqrt(max(nz * nz - 1.0, 0.0)) / vt)
        grav = math.degrees(G * math.cos(gm) / vt)
        print(
            f"{int(k['step'][kk]):5d}{k['t_s'][kk]:7.3f}{k['phi_deg'][kk]:8.2f}{k['theta_deg'][kk]:8.2f}"
            f"{nz:6.2f}{vt:7.1f}{k['gamma_deg'][kk]:8.2f}{k['lift_up'][kk]:8.3f}"
            f"{k['hdg_rate_dps'][kk]:9.2f}{lvl:9.2f}{k['turn_rate_dps'][kk]:9.2f}{grav:9.2f}"
        )


if __name__ == "__main__":
    main()
