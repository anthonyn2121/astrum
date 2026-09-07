"""LEAD AUDIT — is the shot's small `alpha_req` a range fact or a geometry fact?

Recomputes, from the dumped per-step state in runs/d3b/replay_*.csv:
  * ang(LOS, own velocity)                       -- the claim's "raw LOS-off-velocity angle"
  * alpha_req (recomputed from gunlay.intercept) -- checked against the dumped column
  * L = ang(aim_dir, LOS)                        -- the TRUE lead angle the lay demands
  * the in-plane / out-of-plane split of L relative to the (LOS, v) plane
  * ang(nose, v) -- the total off-flight-path cone the airframe is actually holding

Nothing is estimated: every angle is a dot/cross product over the dumped ENU vectors, and the
intercept solver is the same `astrum_dogfight.gunlay.intercept` that `geometry.relative` calls.
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight.geometry import body_axes_enu  # noqa: E402
from astrum_dogfight.gunlay import intercept  # noqa: E402

DEG = 180.0 / math.pi
RUNS = "/workspace/controllers/rl/runs/d3b"


def load(path: str) -> list[dict[str, float]]:
    with open(path) as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def ang(a: np.ndarray, b: np.ndarray) -> float:
    return math.atan2(
        float(np.linalg.norm(np.cross(a, b))),
        float(np.clip(np.dot(a, b), -1.0, 1.0)),
    )


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def audit(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    out = []
    for r in rows:
        p = np.array([r["pos_e_m"], r["pos_n_m"], r["pos_u_m"]])
        v = np.array([r["vel_e_ms"], r["vel_n_ms"], r["vel_u_ms"]])
        q = np.array([r["opp_pos_e_m"], r["opp_pos_n_m"], r["opp_pos_u_m"]])
        # bandit velocity is NOT dumped as a vector -> reconstruct from attitude+speed? NO.
        # Instead difference the dumped bandit position at dt (done by the caller); here we
        # only need it for the intercept, so it is passed in via r["_bvel"].
        bv = np.array([r["_bve"], r["_bvn"], r["_bvu"]])
        los = q - p
        rng = float(np.linalg.norm(los))
        lh = unit(los)
        vh = unit(v)
        sol = intercept(p, v, q, bv)
        aim = sol.boresight if sol.feasible else lh
        a_req = ang(aim, vh) * DEG
        a_losv = ang(lh, vh) * DEG
        lead = ang(aim, lh) * DEG
        # split L into the component inside the (LOS, v) plane and out of it
        nrm = np.cross(lh, vh)
        if float(np.linalg.norm(nrm)) > 1e-9:
            nrm = unit(nrm)
            out_of_plane = math.asin(abs(float(np.clip(np.dot(aim, nrm), -1.0, 1.0)))) * DEG
        else:
            out_of_plane = float("nan")
        nose, _, _ = body_axes_enu(
            math.radians(r["phi_deg"]), math.radians(r["theta_deg"]), math.radians(r["psi_deg"])
        )
        out.append(
            dict(
                step=r["step"],
                t=r["t_s"],
                rng_csv=r["rng_m"],
                rng_chk=rng,
                tof_csv=r["lay_tof_s"],
                tof_chk=sol.tof_s,
                areq_csv=r["alpha_req_deg"],
                areq_chk=a_req,
                a_losv=a_losv,
                diff=a_req - a_losv,
                lead_off_los=lead,
                lead_oop=out_of_plane,
                ang_nose_v=ang(nose, vh) * DEG,
                # EXACT identity: |r/t + u| = MUZZLE with u = v_rel, so the transverse part of
                # v_rel alone fixes the lead off the LOS:  sin L = |v_perp| / MUZZLE.
                vperp=math.radians(r["los_rate_dps"]) * rng,
                l_exact=math.degrees(
                    math.asin(min(1.0, math.radians(r["los_rate_dps"]) * rng / 1030.0))
                ),
                alpha=r["alpha_deg"],
                beta=r["beta_deg"],
                ata=r["ata_deg"],
                aspect=r["aspect_deg"],
                losrate=r["los_rate_dps"],
                lead_pred=r["los_rate_dps"] * sol.tof_s,
                lay_miss=r["lay_miss_m"],
                lead_err=r["lead_error_deg"],
                fire=r["fire"],
                nz=r["nz_g"],
                closure=r["closure_ms"],
            )
        )
    return out


def with_bandit_vel(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    """Bandit velocity vector is not a dumped column; central-difference its position.

    dt = ACTION_REPEAT/RATE_HZ = 5/120 s between consecutive rows. This is the ONLY estimated
    quantity in the audit, so it is validated below against the dumped `rng`/`closure`/`lay_tof`.
    """
    dt = 5.0 / 120.0
    n = len(rows)
    for i, r in enumerate(rows):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        span = (j1 - j0) * dt
        for k, c in (("_bve", "opp_pos_e_m"), ("_bvn", "opp_pos_n_m"), ("_bvu", "opp_pos_u_m")):
            r[k] = (rows[j1][c] - rows[j0][c]) / span
    return rows


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "kill"
    path = f"{RUNS}/replay_{'kill_seed62' if which == 'kill' else 'nokill_seed28'}.csv"
    rows = with_bandit_vel(load(path))
    a = audit(rows)
    print(f"=== {path}  ({len(a)} rows) ===")
    # validation of the differenced bandit velocity: |opp speed| vs dumped opp_vt_ms, and
    # recomputed alpha_req / tof vs the dumped columns.
    sp = np.array([math.hypot(math.hypot(r["_bve"], r["_bvn"]), r["_bvu"]) for r in rows])
    vt = np.array([r["opp_vt_ms"] for r in rows])
    print(
        f"bandit speed: differenced vs dumped opp_vt_ms  max|err| = {np.abs(sp - vt).max():.4f} m/s"
    )
    print(f"rng:      max|chk-csv| = {max(abs(r['rng_chk'] - r['rng_csv']) for r in a):.3e} m")
    print(f"lay_tof:  max|chk-csv| = {max(abs(r['tof_chk'] - r['tof_csv']) for r in a):.3e} s")
    print(f"alpha_req max|chk-csv| = {max(abs(r['areq_chk'] - r['areq_csv']) for r in a):.3e} deg")
    lex = np.array([r["l_exact"] for r in a])
    lmeas = np.array([r["lead_off_los"] for r in a])
    print(
        f"IDENTITY  sin L == |v_perp|/1030 : max|L - asin(vperp/1030)| = {np.abs(lex - lmeas).max():.3e} deg"
    )
    hdr = (
        "step    t_s   rng_m   tof_s  areq_csv areq_chk  a(LOS,v)   diff  L=a(aim,LOS) "
        " L_oop  losrate  vperp  L_exact  a(nose,v) alpha  beta   ata  aspect lay_miss lead_err fire"
    )
    print(hdr)
    cited = {24, 96, 144, 164}
    for r in a:
        s = int(r["step"])
        if not (s in cited or s % 8 == 0 or r["fire"] > 0.5 or s == len(a) - 1):
            continue
        mark = " *CITED*" if s in cited else ""
        print(
            f"{s:4d} {r['t']:6.3f} {r['rng_csv']:7.1f} {r['tof_csv']:6.3f} "
            f"{r['areq_csv']:8.2f} {r['areq_chk']:8.2f} {r['a_losv']:9.2f} {r['diff']:6.2f} "
            f"{r['lead_off_los']:12.2f} {r['lead_oop']:6.2f} {r['losrate']:8.2f} "
            f"{r['vperp']:6.1f} {r['l_exact']:7.2f} {r['ang_nose_v']:9.2f} "
            f"{r['alpha']:5.2f} {r['beta']:6.2f} "
            f"{r['ata']:6.2f} {r['aspect']:6.1f} {r['lay_miss']:8.2f} {r['lead_err']:8.2f} "
            f"{int(r['fire']):d}{mark}"
        )
    # correlation structure: does the "diff" track tof, or does the true lead L?
    tof = np.array([r["tof_csv"] for r in a])
    diff = np.array([r["diff"] for r in a])
    lead = np.array([r["lead_off_los"] for r in a])
    alosv = np.array([r["a_losv"] for r in a])
    idx = {int(r["step"]): r for r in a}
    if 24 in idx and 164 in idx:
        s0, s1 = idx[24], idx[164]
        d_req = s1["areq_csv"] - s0["areq_csv"]
        d_los = s1["a_losv"] - s0["a_losv"]
        d_lead = s1["diff"] - s0["diff"]
        print(
            "\nDECOMPOSITION of the alpha_req fall, step 24 -> step 164 (the claim's own endpoints)"
        )
        print(
            f"  alpha_req      {s0['areq_csv']:7.2f} -> {s1['areq_csv']:7.2f}   d = {d_req:+7.2f} deg"
        )
        print(
            f"  ang(LOS,v)     {s0['a_losv']:7.2f} -> {s1['a_losv']:7.2f}   d = {d_los:+7.2f} deg"
            f"   = {100.0 * d_los / d_req:5.1f}% of the fall"
        )
        print(
            f"  lead term      {s0['diff']:7.2f} -> {s1['diff']:7.2f}   d = {d_lead:+7.2f} deg"
            f"   = {100.0 * d_lead / d_req:5.1f}% of the fall"
        )
        print(
            f"  |v_perp|       {s0['vperp']:7.1f} -> {s1['vperp']:7.1f} m/s  "
            f"(ratio {s1['vperp'] / s0['vperp']:.3f}); rng {s0['rng_csv']:.0f} -> {s1['rng_csv']:.0f} m"
        )
    print("\nSUMMARY over all rows")
    print(
        f"  L = ang(aim,LOS):    min {lead.min():.2f}  med {np.median(lead):.2f}  max {lead.max():.2f} deg"
    )
    print(
        f"  ang(LOS,v):          min {alosv.min():.2f}  med {np.median(alosv):.2f}  max {alosv.max():.2f} deg"
    )
    print(
        f"  diff:                min {diff.min():.2f}  med {np.median(diff):.2f}  max {diff.max():.2f} deg"
    )
    print(f"  corr(tof, diff)  = {np.corrcoef(tof, diff)[0, 1]:+.4f}")
    print(f"  corr(tof, L)     = {np.corrcoef(tof, lead)[0, 1]:+.4f}")
    print(f"  corr(tof, a_losv)= {np.corrcoef(tof, alosv)[0, 1]:+.4f}")
    # short-range population: is a small alpha_req automatic once the range is short?
    for lo, hi in ((0.0, 200.0), (200.0, 600.0), (600.0, 1200.0)):
        sel = [r for r in a if lo <= r["rng_csv"] < hi]
        if not sel:
            print(f"  rng [{lo:.0f},{hi:.0f}) m: no rows")
            continue
        ar = np.array([r["areq_csv"] for r in sel])
        al = np.array([r["a_losv"] for r in sel])
        ld = np.array([r["lead_off_los"] for r in sel])
        print(
            f"  rng [{lo:.0f},{hi:.0f}) m: n={len(sel):3d}  alpha_req min {ar.min():6.2f} med "
            f"{np.median(ar):6.2f} max {ar.max():7.2f} | a(LOS,v) med {np.median(al):6.2f} max "
            f"{al.max():7.2f} | L med {np.median(ld):5.2f} max {ld.max():6.2f}"
        )


if __name__ == "__main__":
    main()
