"""Independent re-measurement of the chi_valid / chi-conditioning claim on the d3b artifact.

Reads ONLY the dumped per-step CSVs plus bfm_detect.py itself. Recomputes the validity gate from
raw geometry, the LOS-to-velocity angle, the conditioning amplification at EVERY tick (not a
sampled few), and runs the decisive counterfactual: re-run detect_all with the gate REMOVED
(chi computed everywhere) and see whether any detector count moves.
"""

from __future__ import annotations

import csv
import math
import sys
import types

import numpy as np

# Import the detector submodule WITHOUT executing astrum_dogfight/__init__.py, which pulls in the
# compiled `astrum_env` sim extension. bfm_detect itself needs only `constants` and `geometry`.
_PKG = "/workspace/crates/astrum-env/python/astrum_dogfight"
sys.path.insert(0, "/workspace/crates/astrum-env/python")
_stub = types.ModuleType("astrum_dogfight")
_stub.__path__ = [_PKG]
sys.modules["astrum_dogfight"] = _stub
import astrum_dogfight.bfm_detect as B  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def frame(v_hat):
    up = np.array([0.0, 0.0, 1.0])
    u_ref = _unit(up - float(np.dot(up, v_hat)) * v_hat)
    if not u_ref.any():
        north = np.array([0.0, 1.0, 0.0])
        u_ref = _unit(north - float(np.dot(north, v_hat)) * v_hat)
    return u_ref, np.cross(v_hat, u_ref)


def chi_of(los_hat, v_hat, phi, theta, psi):
    """chi with NO validity gate (radians)."""
    u_ref, r_ref = frame(v_hat)
    lp = los_hat - float(np.dot(los_hat, v_hat)) * v_hat
    lift = B.lift_vector_enu(phi, theta, psi)
    liftp = lift - float(np.dot(lift, v_hat)) * v_hat
    if float(np.linalg.norm(liftp)) < 1e-6 or float(np.linalg.norm(lp)) < 1e-12:
        return None
    a = math.atan2(float(np.dot(liftp, r_ref)), float(np.dot(liftp, u_ref)))
    b = math.atan2(float(np.dot(lp, r_ref)), float(np.dot(lp, u_ref)))
    return B._wrap_pi(a - b)


def rotate(u, axis, ang):
    axis = _unit(axis)
    return (
        u * math.cos(ang)
        + np.cross(axis, u) * math.sin(ang)
        + axis * float(np.dot(axis, u)) * (1 - math.cos(ang))
    )


def max_dchi(los_hat, v_hat, phi, theta, psi, arc_deg, n=720):
    """Max |d chi| over all displacement DIRECTIONS of the bandit, 'arc_deg' of arc."""
    base = chi_of(los_hat, v_hat, phi, theta, psi)
    if base is None:
        return float("nan")
    # build an orthonormal pair perpendicular to los
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, los_hat))) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = _unit(tmp - float(np.dot(tmp, los_hat)) * los_hat)
    e2 = np.cross(los_hat, e1)
    worst = 0.0
    for k in range(n):
        th = 2 * math.pi * k / n
        axis = e1 * math.cos(th) + e2 * math.sin(th)  # rotation axis perp to los
        los2 = _unit(rotate(los_hat, axis, math.radians(arc_deg)))
        c2 = chi_of(los2, v_hat, phi, theta, psi)
        if c2 is None:
            continue
        worst = max(worst, abs(math.degrees(B._wrap_pi(c2 - base))))
    return worst


def geom(row):
    me_p = np.array([float(row["pos_e_m"]), float(row["pos_n_m"]), float(row["pos_u_m"])])
    me_v = np.array([float(row["vel_e_ms"]), float(row["vel_n_ms"]), float(row["vel_u_ms"])])
    op_p = np.array(
        [float(row["opp_pos_e_m"]), float(row["opp_pos_n_m"]), float(row["opp_pos_u_m"])]
    )
    los = op_p - me_p
    rng = float(np.linalg.norm(los))
    los_hat = _unit(los)
    v_hat = _unit(me_v)
    lp = float(np.linalg.norm(los_hat - float(np.dot(los_hat, v_hat)) * v_hat))
    ata_v = math.degrees(math.acos(float(np.clip(np.dot(los_hat, v_hat), -1.0, 1.0))))
    return me_p, me_v, op_p, rng, los_hat, v_hat, lp, ata_v


def build_trace(rows, ungate=False):
    """Trace exactly as the detector read it (from the smp_* block), or with the gate removed."""
    tr = []
    for row in rows:
        _, _, _, _, los_hat, v_hat, _, _ = geom(row)
        chi = float(row["smp_chi_rad"])
        valid = row["smp_chi_valid"].strip().lower() in ("1", "true", "yes")
        if ungate:
            c = chi_of(
                los_hat,
                v_hat,
                math.radians(float(row["phi_deg"])),
                math.radians(float(row["theta_deg"])),
                math.radians(float(row["psi_deg"])),
            )
            if c is not None:
                chi, valid = c, True
        tr.append(
            B.Sample(
                t=float(row["smp_t"]),
                chi=chi,
                chi_valid=valid,
                gamma=float(row["smp_gamma_rad"]),
                nz=float(row["smp_nz"]),
                phi=float(row["smp_phi_rad"]),
                alt=float(row["smp_alt"]),
                ail=float(row["smp_ail"]),
                rng=float(row["smp_rng"]),
                ata=float(row["smp_ata"]),
                ata_az=float(row["smp_ata_az"]),
                aspect=float(row["smp_aspect"]),
                closure=float(row["smp_closure"]),
                cpa_miss=float(row["smp_cpa_miss"]),
                es_rel=float(row["smp_es_rel"]),
                lay_miss_m=float(row["smp_lay_miss_m"]),
                lethal=row["smp_lethal"].strip().lower() in ("1", "true", "yes"),
            )
        )
    return tr


def report(name, path):
    rows = load(path)
    n = len(rows)
    print("=" * 100)
    print(f"{name}   {path}   {n} ticks")

    sin5 = math.sin(math.radians(B.CHI_VALID_ATA_DEG))
    valid_csv = [r["smp_chi_valid"].strip().lower() in ("1", "true", "yes") for r in rows]
    ata_v_all, lp_all, rng_all, gate_all = [], [], [], []
    chi_recomp_err = 0.0
    for r in rows:
        _, _, _, rng, los_hat, v_hat, lp, ata_v = geom(r)
        ata_v_all.append(ata_v)
        lp_all.append(lp)
        rng_all.append(rng)
        gate_all.append(lp < sin5)  # True == the gate TRIPS == invalid
        if not (lp < sin5):
            c = chi_of(
                los_hat,
                v_hat,
                math.radians(float(r["phi_deg"])),
                math.radians(float(r["theta_deg"])),
                math.radians(float(r["psi_deg"])),
            )
            chi_recomp_err = max(chi_recomp_err, abs(math.degrees(c) - float(r["chi_deg"])))
    ata_v_all = np.array(ata_v_all)
    lp_all = np.array(lp_all)
    rng_all = np.array(rng_all)

    nv = sum(valid_csv)
    print(f"  chi_valid in dump      : True {nv}/{n} ({100 * nv / n:.2f}%)   False {n - nv}")
    ntrip = sum(gate_all)
    print(f"  gate |los_perp|<sin(5) : trips on {ntrip}/{n}  -> invalid {ntrip}")
    print(
        f"  agreement dump vs gate : {sum(1 for a, b in zip(valid_csv, gate_all) if a == (not b))}/{n}"
    )
    print(f"  max |chi_recomp - chi_deg| over valid ticks = {chi_recomp_err:.3e} deg")
    lo = int(np.sum(ata_v_all < B.CHI_VALID_ATA_DEG))
    hi = int(np.sum(ata_v_all > 180.0 - B.CHI_VALID_ATA_DEG))
    print(
        f"  ata_v (LOS-to-VELOCITY): min {ata_v_all.min():.2f} @step {int(np.argmin(ata_v_all))}"
        f"  max {ata_v_all.max():.2f} @step {int(np.argmax(ata_v_all))}"
    )
    print(
        f"  ticks ata_v < 5 deg    : {lo}      ticks ata_v > 175 deg : {hi}   (BOTH trip the gate)"
    )

    inv = [i for i, v in enumerate(valid_csv) if not v]
    if inv:
        print(f"  INVALID tick indices   : {inv[0]}..{inv[-1]}  count {len(inv)}")
        print(
            f"    ata_v on invalid     : min {ata_v_all[inv].min():.3f}  max {ata_v_all[inv].max():.3f}"
        )
        print(
            f"    rng   on invalid     : min {rng_all[inv].min():.1f} m  max {rng_all[inv].max():.1f} m"
        )
        runs = []
        s = inv[0]
        for a, b in zip(inv, inv[1:] + [None]):
            if b != (a + 1 if a is not None else None):
                runs.append((s, a))
                s = b
        print(f"    contiguous runs      : {runs}")

    # conditioning at EVERY tick
    amp = np.full(n, np.nan)
    for i, r in enumerate(rows):
        _, _, _, _, los_hat, v_hat, lp, _ = geom(r)
        amp[i] = max_dchi(
            los_hat,
            v_hat,
            math.radians(float(r["phi_deg"])),
            math.radians(float(r["theta_deg"])),
            math.radians(float(r["psi_deg"])),
            1.0,
            n=360,
        )
    k = int(np.nanargmax(amp))
    print(f"  CONDITIONING, 1.0 deg of arc, max over directions, ALL {n} ticks:")
    print(
        f"    worst |dchi| = {amp[k]:.2f} deg/deg at step {k} t {float(rows[k]['t_s']):.3f} s"
        f"  ata_v {ata_v_all[k]:.2f}  chi {float(rows[k]['chi_deg']):+.1f}  nz {float(rows[k]['nz_g']):+.2f}"
    )
    order = np.argsort(-np.nan_to_num(amp))[:8]
    for i in order:
        print(
            f"      step {i:4d} t {float(rows[i]['t_s']):6.3f}  ata_v {ata_v_all[i]:7.2f}"
            f"  rng {rng_all[i]:8.1f}  chi {float(rows[i]['chi_deg']):+7.1f}"
            f"  nz {float(rows[i]['nz_g']):+5.2f}  |dchi|/1deg {amp[i]:6.2f}"
            f"  (1/sin bound {1.0 / max(lp_all[i], 1e-9):5.2f})"
        )

    # per-METRE sensitivity: arc subtended by 1 m of bandit position error at this range
    perm = np.array(
        [amp[i] * math.degrees(math.asin(min(1.0, 1.0 / max(rng_all[i], 1e-6)))) for i in range(n)]
    )
    j = int(np.nanargmax(perm))
    print(
        f"  per-METRE sensitivity  : worst {perm[j]:.1f} deg of chi per 1 m of bandit position"
        f" error, step {j} (rng {rng_all[j]:.1f} m)"
    )

    # the audit's specific rows
    for i in (0, 86, 165, 172, 288, 410, 575):
        if i >= n:
            continue
        print(
            f"    AUDIT ROW step {i:4d}: ata_v {ata_v_all[i]:7.2f}  chi {float(rows[i]['chi_deg']):+7.1f}"
            f"  nz {float(rows[i]['nz_g']):+5.2f}  valid {valid_csv[i]}  |dchi|/1deg {amp[i]:5.2f}"
        )

    # exact LOS error needed to manufacture a 45 deg (CHI_ROOM) chi excursion
    for i in (165,):
        if i >= n:
            continue
        _, _, _, _, los_hat, v_hat, _, _ = geom(rows[i])
        p, t_, s_ = (math.radians(float(rows[i][c])) for c in ("phi_deg", "theta_deg", "psi_deg"))
        lo_d, hi_d = 0.0, 20.0
        for _ in range(60):
            mid = 0.5 * (lo_d + hi_d)
            if max_dchi(los_hat, v_hat, p, t_, s_, mid, n=360) >= 45.0:
                hi_d = mid
            else:
                lo_d = mid
        theo = math.degrees(
            math.asin(min(1.0, math.sin(math.radians(ata_v_all[i])) * math.sin(math.radians(45.0))))
        )
        print(
            f"  step {i}: LOS arc error needed to move chi by 45 deg = {hi_d:.2f} deg"
            f"  (closed form arcsin(sin ata_v * sin45) = {theo:.2f});"
            f"  linear extrapolation 45/{amp[i]:.2f} = {45.0 / amp[i]:.2f} deg"
        )

    # ---- decisive counterfactual: does any VERDICT turn on the gate? ----
    tr = build_trace(rows, ungate=False)
    tr_u = build_trace(rows, ungate=True)
    d = B.detect_all(tr)
    du = B.detect_all(tr_u)

    def cnt(x):
        return {
            k2: (len(v) if isinstance(v, list) else v)
            for k2, v in x.items()
            if k2 in ("displacement_rolls", "yoyos", "yoyos_clean", "turning_room", "lethal_ticks")
        }

    print(f"  detect_all AS-DUMPED   : {cnt(d)}")
    print(f"  detect_all GATE REMOVED: {cnt(du)}")
    nd = sum(1 for a, b in zip(tr, tr_u) if abs(a.chi - b.chi) > 1e-12)
    print(f"  ticks whose chi changed when the gate was removed: {nd}")
    if nd:
        mx = max(abs(math.degrees(a.chi - b.chi)) for a, b in zip(tr, tr_u))
        print(f"    max |dchi| from un-gating = {mx:.1f} deg")

    # in-plane predicate abuse: invalid ticks satisfy D1 (a) lookback and (d) return trivially
    ip = sum(1 for i, v in enumerate(valid_csv) if not v)
    print(
        f"  invalid ticks that trivially satisfy 'in-plane' (|chi|<{B.CHI_INPLANE_DEG:.0f}) "
        f"in D1 clause (a)/(d): {ip}"
    )

    # holds at the CHI_ROOM bar and their sensitivity
    chi_deg = np.array([float(r["chi_deg"]) for r in rows])
    nz = np.array([float(r["nz_g"]) for r in rows])
    hold = (np.abs(chi_deg) > B.CHI_ROOM_DEG) & (nz > B.NZ_LOADED)
    runs = []
    i = 0
    while i < n:
        if hold[i]:
            j = i
            while j + 1 < n and hold[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    print(f"  loaded |chi|>{B.CHI_ROOM_DEG:.0f} runs (D3 hold candidates): {runs}")
    for a, b in runs:
        dur = float(rows[b]["t_s"]) - float(rows[a]["t_s"])
        dalt = float(rows[b]["alt_m"]) - float(rows[a]["alt_m"])
        print(
            f"    steps {a}-{b}  {dur:.3f} s  dalt {dalt:+.1f} m"
            f"  cpa {float(rows[a]['cpa_miss_m']):.1f} -> {float(rows[b]['cpa_miss_m']):.1f} m"
            f"  amp at edges {amp[a]:.2f}/{amp[b]:.2f} deg/deg"
            f"  |chi| at edges {abs(chi_deg[a]):.1f}/{abs(chi_deg[b]):.1f}"
        )

    return rows, valid_csv, ata_v_all, amp, d, du


if __name__ == "__main__":
    report("KILL seed 62", KILL)
    report("NOKILL seed 28", NOKILL)
