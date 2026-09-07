"""Second pass: chi shape across the whole hold, energy price, full-hold frame attribution,
and the exact D1 clause that rejects the one >60 deg loaded hold the genome ever flies (seed 28).
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0
NZ_LOADED, CHI_INPLANE, CHI_ROOM, CHI_OUT = 3.0, 20.0, 45.0, 60.0
CHI_MIN_HOLD_S, CHI_RETURN_DEG, CHI_RETURN_S, CHI_LOOKBACK_S = 0.5, 30.0, 4.0, 4.0
ASPECT_GAIN_DEG = 15.0


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


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


def body_axes_enu(phi, theta, psi):  # geometry.py:96-102 verbatim
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    nose = np.array([cth * sps, cth * cps, sth])
    starboard = np.array([sph * sth * sps + cph * cps, sph * sth * cps - cph * sps, -sph * cth])
    down = np.array([cph * sth * sps - sph * cps, cph * sth * cps + sph * sps, -cph * cth])
    return nose, starboard, down


def vecs(d, i):
    _, _, down = body_axes_enu(
        math.radians(d["phi_deg"][i]),
        math.radians(d["theta_deg"][i]),
        math.radians(d["psi_deg"][i]),
    )
    lift = -down
    v = np.array([d["vel_e_ms"][i], d["vel_n_ms"][i], d["vel_u_ms"][i]])
    vh = v / np.linalg.norm(v)
    los = np.array(
        [
            d["opp_pos_e_m"][i] - d["pos_e_m"][i],
            d["opp_pos_n_m"][i] - d["pos_n_m"][i],
            d["opp_pos_u_m"][i] - d["pos_u_m"][i],
        ]
    )
    return lift / np.linalg.norm(lift), los / np.linalg.norm(los), vh


def ang(u, v):
    return math.degrees(math.acos(float(np.clip(u @ v, -1, 1))))


def full_hold_attribution(d, a, b, label):
    hdr(f"{label}: INERTIAL motion over the FULL load-gated hold, steps {a}-{b}")
    L0, S0, V0 = vecs(d, a)
    L1, S1, V1 = vecs(d, b)
    cl = cs = cv = 0.0
    p = vecs(d, a)
    for k in range(a + 1, b + 1):
        c = vecs(d, k)
        cl += ang(p[0], c[0])
        cs += ang(p[1], c[1])
        cv += ang(p[2], c[2])
        p = c
    print(
        f"  t {d['t_s'][a]:.3f} -> {d['t_s'][b]:.3f} s ({(b - a) * DT:.3f} s endpoint)\n"
        f"    net rotation  lift {ang(L0, L1):7.2f} deg   LOS {ang(S0, S1):7.2f} deg   "
        f"vel {ang(V0, V1):7.2f} deg\n"
        f"    cumulative    lift {cl:7.2f} deg   LOS {cs:7.2f} deg   vel {cv:7.2f} deg\n"
        f"    ratio lift/LOS  net {ang(L0, L1) / max(ang(S0, S1), 1e-9):5.2f}x   "
        f"cumulative {cl / max(cs, 1e-9):5.2f}x"
    )
    print(
        f"    price: alt {d['alt_m'][a]:7.1f} -> {d['alt_m'][b]:7.1f} "
        f"({d['alt_m'][b] - d['alt_m'][a]:+7.1f} m)  Es {d['es_m'][a]:7.1f} -> {d['es_m'][b]:7.1f} "
        f"({d['es_m'][b] - d['es_m'][a]:+7.1f} m)  es_rel {d['es_rel_m'][a]:+7.1f} -> "
        f"{d['es_rel_m'][b]:+7.1f}  vt {d['vt_ms'][a]:.1f} -> {d['vt_ms'][b]:.1f} m/s"
    )
    print(
        f"    geometry: rng {d['rng_m'][a]:7.1f} -> {d['rng_m'][b]:7.1f}  cpa_miss "
        f"{d['cpa_miss_m'][a]:7.1f} -> {d['cpa_miss_m'][b]:7.1f}  ata {d['ata_deg'][a]:6.2f} -> "
        f"{d['ata_deg'][b]:6.2f}  aspect {d['aspect_deg'][a]:6.2f} -> {d['aspect_deg'][b]:6.2f}  "
        f"gamma {d['gamma_deg'][a]:+6.2f} -> {d['gamma_deg'][b]:+6.2f}"
    )


def chi_shape(d, a, b, label):
    hdr(f"{label}: chi through the hold — is there an out-AND-BACK shape? (steps {a}-{b})")
    print(
        f"{'step':>5} {'t_s':>6} {'chi':>8} {'nz':>5} {'phi':>8} {'alt':>7} {'cpa':>8} {'rng':>7} {'ata_az':>8}"
    )
    for k in range(a, b + 1, 4):
        print(
            f"{k:5d} {d['t_s'][k]:6.3f} {d['chi_deg'][k]:8.2f} {d['nz_g'][k]:5.2f} "
            f"{d['phi_deg'][k]:8.2f} {d['alt_m'][k]:7.1f} {d['cpa_miss_m'][k]:8.1f} "
            f"{d['rng_m'][k]:7.1f} {d['ata_az_deg'][k]:8.2f}"
        )
    ach = np.abs(d["chi_deg"][a : b + 1])
    print(
        f"  |chi| argmax step {a + int(ach.argmax())} ({ach.max():.2f} deg); "
        f"|chi| at hold end step {b} = {ach[-1]:.2f} deg"
    )


def d1_clauses(d, label):
    hdr(f"{label}: D1 clause-by-clause on EVERY loaded |chi|>60 run")
    ach = np.abs(d["chi_deg"])
    t = d["t_s"]
    n = d["_n"]
    hold = (ach > CHI_OUT) & (d["nz_g"] > NZ_LOADED)
    for i, j in runs(hold):
        dur = t[j] - t[i]
        lb = any(
            ach[k] < CHI_INPLANE for k in range(i - 1, -1, -1) if t[i] - t[k] <= CHI_LOOKBACK_S
        )
        clo = d["closure_ms"][i] > 0.0
        exit_idx = None
        for k in range(j + 1, n):
            if t[k] - t[j] > CHI_RETURN_S:
                break
            if ach[k] < CHI_RETURN_DEG:
                exit_idx = k
                break
        sides = None
        dasp = None
        if exit_idx is not None:
            sides = {1 if d["ata_az_deg"][k] >= 0 else -1 for k in range(i, exit_idx + 1)}
            dasp = d["aspect_deg"][exit_idx] - d["aspect_deg"][i]
        print(
            f"  run {i}-{j} (t {t[i]:.3f}-{t[j]:.3f}, endpt {dur:.3f} s, chi peak {ach[i : j + 1].max():.2f})\n"
            f"    hold >= {CHI_MIN_HOLD_S} s        : {'PASS' if dur >= CHI_MIN_HOLD_S else 'FAIL'}\n"
            f"    (a) in-plane lookback : {'PASS' if lb else 'FAIL'}\n"
            f"    (c) closing at entry  : {'PASS' if clo else 'FAIL'} (closure {d['closure_ms'][i]:+.1f} m/s)\n"
            f"    (d) rolled back in    : {('PASS at step ' + str(exit_idx)) if exit_idx is not None else 'FAIL'}\n"
            f"    (e) ata_az one-sided  : {sides} {'PASS' if sides and len(sides) == 1 else 'FAIL'}\n"
            f"    (f) aspect gain <= -{ASPECT_GAIN_DEG:.0f} deg : "
            f"{('d_aspect ' + format(dasp, '+.2f') + ' -> ' + ('PASS' if dasp < -ASPECT_GAIN_DEG else 'FAIL')) if dasp is not None else 'n/a'}"
        )


def endpoint_sensitivity(d, label):
    hdr(f"{label}: is 'phi held' robust to the window? (the claim picks steps 36-96)")
    for a, b in [(36, 96), (46, 95), (36, 95), (28, 115), (46, 115), (60, 115)]:
        phi = d["phi_deg"][a : b + 1]
        chi = d["chi_deg"][a : b + 1]
        print(
            f"  steps {a:3d}-{b:3d}: phi {phi.min():7.2f}..{phi.max():7.2f} "
            f"(halfrange {(phi.max() - phi.min()) / 2:5.2f} deg)  chi {chi[0]:+7.2f} -> {chi[-1]:+7.2f} "
            f"({chi[-1] - chi[0]:+7.2f})  alt {d['alt_m'][a]:7.1f} -> {d['alt_m'][b]:7.1f} "
            f"({d['alt_m'][b] - d['alt_m'][a]:+7.1f} m)  seg {int(d['seg_idx'][a])}->{int(d['seg_idx'][b])}"
        )
    print(
        "\n  NOTE step 96 is the FIRST step of genome segment 1 (bank cmd +115.00 -> +48.42 deg),"
        "\n  i.e. the claim's 'established turn' window ends exactly at the command discontinuity."
    )
    for k in (94, 95, 96, 97, 100, 104, 108, 112, 115):
        print(
            f"    step {k:3d} t {d['t_s'][k]:6.3f} seg {int(d['seg_idx'][k])} cmd_bank "
            f"{d['cmd_bank_deg'][k]:+8.2f} phi {d['phi_deg'][k]:8.2f} chi {d['chi_deg'][k]:+8.2f} "
            f"nz {d['nz_g'][k]:5.2f}"
        )


def main():
    k = load(KILL)
    nk = load(NOKILL)
    full_hold_attribution(k, 28, 115, "KILL seed 62")
    chi_shape(k, 28, 120, "KILL seed 62")
    endpoint_sensitivity(k, "KILL seed 62")
    d1_clauses(k, "KILL seed 62")
    d1_clauses(nk, "NOKILL seed 28")
    full_hold_attribution(nk, 499, 575, "NOKILL seed 28 (the ONE >60 deg loaded hold)")


if __name__ == "__main__":
    main()
