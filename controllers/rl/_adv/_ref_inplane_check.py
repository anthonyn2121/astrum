"""Refutation audit of the 'in-plane w.r.t. LOS / out-of-plane w.r.t. horizon' claim.

Pure post-hoc: every number is a function of the two dumped CSVs. numpy only.
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0

NZ_LOADED = 3.0
CHI_INPLANE = 20.0
CHI_ROOM = 45.0
CHI_OUT = 60.0
CHI_MIN_HOLD_S = 0.5


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


def body_axes_enu(phi, theta, psi):
    """`geometry.py:96-102` VERBATIM. Copied rather than imported because importing the package
    pulls the `astrum_env` Rust extension, which this pure-CSV audit does not need. Validated by
    the `reconstruction check` lines below: chi rebuilt from the dumped attitude/positions matches
    the dumped `chi_deg` column to all printed digits.
    """
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    nose = np.array([cth * sps, cth * cps, sth])
    starboard = np.array([sph * sth * sps + cph * cps, sph * sth * cps - cph * sps, -sph * cth])
    down = np.array([cph * sth * sps - sph * cps, cph * sth * cps + sph * sps, -cph * cth])
    return nose, starboard, down


def check_geom_consistency(d, i):
    """Reconstruct chi from the dumped attitude + positions and compare to chi_deg."""
    phi, th, ps = (
        math.radians(d["phi_deg"][i]),
        math.radians(d["theta_deg"][i]),
        math.radians(d["psi_deg"][i]),
    )
    _, _, down = body_axes_enu(phi, th, ps)
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
    los = los / np.linalg.norm(los)
    up = np.array([0.0, 0.0, 1.0])
    u_ref = up - np.dot(up, vh) * vh
    u_ref /= np.linalg.norm(u_ref)
    r_ref = np.cross(vh, u_ref)
    lp = los - np.dot(los, vh) * vh
    lift_p = lift - np.dot(lift, vh) * vh
    phi_lift = math.atan2(lift_p @ r_ref, lift_p @ u_ref)
    phi_los = math.atan2(lp @ r_ref, lp @ u_ref)
    chi = (phi_lift - phi_los + math.pi) % (2 * math.pi) - math.pi
    return (
        math.degrees(chi),
        math.degrees(phi_lift),
        math.degrees(phi_los),
        lift,
        los,
        vh,
        math.degrees(math.asin(np.clip(np.linalg.norm(lp), -1, 1))),
    )


def part1_recompute(d, label):
    hdr(f"{label}: (1) IS steps 28-115 A CONTIGUOUS nz>3 HOLD? band residence recomputed")
    nz = d["nz_g"]
    hl = runs(nz > NZ_LOADED)
    print(f"  all nz>{NZ_LOADED} runs (i0,i1,count,dur_endpoint_s,dur_count_s):")
    for i, j in hl:
        print(
            f"    {i:4d}-{j:4d}  n={j - i + 1:4d}  t {d['t_s'][i]:6.3f}-{d['t_s'][j]:6.3f}  "
            f"endpt {(d['t_s'][j] - d['t_s'][i]):6.3f}s  count {(j - i + 1) * DT:6.3f}s  "
            f"alt {d['alt_m'][i]:7.1f}->{d['alt_m'][j]:7.1f} ({d['alt_m'][j] - d['alt_m'][i]:+7.1f})"
        )
    # the claimed hold
    i, j = 28, 115
    if i >= d["_n"] or j >= d["_n"]:
        print("  claimed window out of range")
        return
    seg = nz[i : j + 1]
    print(
        f"\n  claimed hold 28-115: nz min {seg.min():.3f} max {seg.max():.3f}; "
        f"all>3 = {bool((seg > NZ_LOADED).all())}; n={j - i + 1} steps; "
        f"t {d['t_s'][i]:.3f}-{d['t_s'][j]:.3f}; endpt dur {(d['t_s'][j] - d['t_s'][i]):.3f} s; "
        f"count dur {(j - i + 1) * DT:.3f} s"
    )
    ach = np.abs(d["chi_deg"][i : j + 1])
    alt = d["alt_m"][i : j + 1]
    val = d["chi_valid"][i : j + 1]
    print(f"  chi_valid over 28-115: {int(val.sum())}/{len(val)} valid; invalid steps ->", end=" ")
    inv = [i + k for k in range(len(val)) if val[k] < 0.5]
    print(inv[:40])
    for lo, hi, nm in [
        (0.0, CHI_INPLANE, "IN-PLANE <20"),
        (CHI_INPLANE, CHI_ROOM, "oblique 20-45"),
        (CHI_ROOM, CHI_OUT, "D3 band 45-60"),
        (CHI_OUT, 181.0, "D1 band >60"),
    ]:
        m = (ach >= lo) & (ach < hi)
        if not m.any():
            print(f"    band {nm:<16} 0 steps")
            continue
        lr = runs(m)
        lg = max(lr, key=lambda r: r[1] - r[0])
        nvalid = int((val[m] > 0.5).sum())
        print(
            f"    band {nm:<16} {int(m.sum()):3d} steps = {m.sum() * DT:5.3f} s "
            f"(valid {nvalid}/{int(m.sum())})  alt {alt[m][0]:.1f}->{alt[m][-1]:.1f} "
            f"({alt[m][-1] - alt[m][0]:+.1f} m)  longest {lg[1] - lg[0] + 1} steps = "
            f"{(lg[1] - lg[0] + 1) * DT:.3f} s (steps {i + lg[0]}-{i + lg[1]}, "
            f"alt {alt[lg[0]]:.1f}->{alt[lg[1]]:.1f})"
        )
    tot = int(len(ach))
    print(
        f"    TOTAL {tot} steps = {tot * DT:.3f} s  (claim quotes hold dur 3.625 s = endpoint "
        f"convention, but sums bands by COUNT = {tot * DT:.3f} s)"
    )
    frac_in = float((ach < CHI_INPLANE).mean())
    print(
        f"    fraction of the hold with |chi| < 20 deg (the claim's own 'in-plane' bar): "
        f"{frac_in:.3f} ({int((ach < CHI_INPLANE).sum())}/{tot})"
    )
    print(
        f"    fraction with |chi| >= 45 deg: "
        f"{float((ach >= CHI_ROOM).mean()):.3f} ({int((ach >= CHI_ROOM).sum())}/{tot}); "
        f"max |chi| in hold {ach.max():.2f} deg at step {i + int(ach.argmax())} "
        f"(nz {nz[i + int(ach.argmax())]:.2f} g)"
    )


def part2_phi_chi(d, label):
    hdr(f"{label}: (2) steps 36-96 -- phi held? chi drift? mean chi_rate?")
    a, b = 36, 96
    phi = d["phi_deg"][a : b + 1]
    chi = d["chi_deg"][a : b + 1]
    phir = np.r_[0.0, np.diff(d["phi_deg"]) / DT]
    chir = np.r_[0.0, np.diff(d["chi_deg"]) / DT]
    print(
        f"  phi min {phi.min():.3f} max {phi.max():.3f} halfrange "
        f"{(phi.max() - phi.min()) / 2:.3f} deg  |phi_rate| max over 36-96 "
        f"{np.abs(phir[a : b + 1]).max():.3f} deg/s at step {a + int(np.abs(phir[a : b + 1]).argmax())}"
    )
    print(
        f"  chi {chi[0]:.3f} -> {chi[-1]:.3f} = {chi[-1] - chi[0]:+.3f} deg; "
        f"mean chi_rate[36:97] {chir[a : b + 1].mean():+.3f} deg/s; "
        f"(chi[96]-chi[36])/(t96-t36) = {(chi[-1] - chi[0]) / (d['t_s'][b] - d['t_s'][a]):+.3f} deg/s"
    )
    # where is the |phi_rate| max, is it 'all settling at 36-40'?
    big = [k for k in range(a, b + 1) if abs(phir[k]) > 5.0]
    print(f"  steps in 36-96 with |phi_rate| > 5 deg/s: {big}")
    print(
        f"  steps in 36-96 with |phi_rate| > 2 deg/s: {[k for k in range(a, b + 1) if abs(phir[k]) > 2.0]}"
    )


def part3_frame_attribution(d, label, a=36, b=96):
    hdr(f"{label}: (3) DID THE LIFT VECTOR MOVE? inertial-frame attribution of the chi drift")
    print(
        "  chi = phi_lift - phi_los, both measured about the (ROTATING) velocity vector.\n"
        "  Test: how far does the LIFT VECTOR move in ENU vs how far does the LOS move in ENU?"
    )
    r0 = check_geom_consistency(d, a)
    r1 = check_geom_consistency(d, b)
    print(
        f"  reconstruction check: chi_dump[{a}] {d['chi_deg'][a]:+8.3f} vs recomputed "
        f"{r0[0]:+8.3f}  (phi_lift {r0[1]:+8.3f}, phi_los {r0[2]:+8.3f})"
    )
    print(
        f"  reconstruction check: chi_dump[{b}] {d['chi_deg'][b]:+8.3f} vs recomputed "
        f"{r1[0]:+8.3f}  (phi_lift {r1[1]:+8.3f}, phi_los {r1[2]:+8.3f})"
    )
    ang = lambda u, v: math.degrees(math.acos(float(np.clip(u @ v, -1, 1))))  # noqa: E731
    L0, L1 = r0[3], r1[3]
    S0, S1 = r0[4], r1[4]
    V0, V1 = r0[5], r1[5]
    print(
        f"\n  INERTIAL (ENU) rotation over steps {a}->{b} (t {d['t_s'][a]:.3f}->{d['t_s'][b]:.3f} s):\n"
        f"    lift vector      moved {ang(L0, L1):7.2f} deg\n"
        f"    line of sight    moved {ang(S0, S1):7.2f} deg\n"
        f"    velocity vector  moved {ang(V0, V1):7.2f} deg"
    )
    # per-step cumulative path lengths (a rotation can double back)
    cumL = cumS = cumV = 0.0
    prev = check_geom_consistency(d, a)
    for k in range(a + 1, b + 1):
        cur = check_geom_consistency(d, k)
        cumL += ang(prev[3], cur[3])
        cumS += ang(prev[4], cur[4])
        cumV += ang(prev[5], cur[5])
        prev = cur
    print(f"    cumulative arc:  lift {cumL:7.2f} deg   LOS {cumS:7.2f} deg   vel {cumV:7.2f} deg")
    # decompose the chi drift: hold LOS fixed vs hold lift fixed
    print(
        "\n  COUNTERFACTUAL DECOMPOSITION of d(chi) (both terms use the true v_hat at each step):"
    )
    print(
        f"    phi_lift {r0[1]:+8.3f} -> {r1[1]:+8.3f} = {(r1[1] - r0[1] + 180) % 360 - 180:+8.3f} deg"
    )
    print(
        f"    phi_los  {r0[2]:+8.3f} -> {r1[2]:+8.3f} = {(r1[2] - r0[2] + 180) % 360 - 180:+8.3f} deg"
    )
    print(
        "    NOTE both phi_* are angles in a frame attached to the rotating velocity vector; a\n"
        "    change in phi_los there does NOT mean the LOS moved in space."
    )
    # how much does phi_los change purely because v_hat rotated? freeze both jets' positions.
    p_me = np.array([d["pos_e_m"][a], d["pos_n_m"][a], d["pos_u_m"][a]])
    p_op = np.array([d["opp_pos_e_m"][a], d["opp_pos_n_m"][a], d["opp_pos_u_m"][a]])
    los_frozen = (p_op - p_me) / np.linalg.norm(p_op - p_me)
    out = []
    for k in (a, b):
        v = np.array([d["vel_e_ms"][k], d["vel_n_ms"][k], d["vel_u_ms"][k]])
        vh = v / np.linalg.norm(v)
        up = np.array([0.0, 0.0, 1.0])
        u_ref = up - (up @ vh) * vh
        u_ref /= np.linalg.norm(u_ref)
        r_ref = np.cross(vh, u_ref)
        lp = los_frozen - (los_frozen @ vh) * vh
        out.append(math.degrees(math.atan2(lp @ r_ref, lp @ u_ref)))
    print(
        f"    FROZEN-GEOMETRY control (positions pinned at step {a}, only v_hat allowed to rotate):\n"
        f"      phi_los would go {out[0]:+8.3f} -> {out[1]:+8.3f} = "
        f"{(out[1] - out[0] + 180) % 360 - 180:+8.3f} deg\n"
        f"      i.e. this much of the phi_los 'swing' is the OWN-SHIP TURN, not bandit/LOS motion."
    )


def part4_d3_clause(d, label):
    hdr(f"{label}: (4) WHY D3 = 0 -- is it the chi clause or the cpa clause?")
    ach = np.abs(d["chi_deg"])
    nz = d["nz_g"]
    hold = (ach > CHI_ROOM) & (nz > NZ_LOADED)
    print(f"  D3 hold mask (|chi|>{CHI_ROOM} AND nz>{NZ_LOADED}) runs:")
    any_pass_chi = False
    for i, j in runs(hold):
        dur = d["t_s"][j] - d["t_s"][i]
        ok = dur >= CHI_MIN_HOLD_S
        if ok:
            any_pass_chi = True
        sides = {1 if d["ata_az_deg"][k] >= 0 else -1 for k in range(i, j + 1)}
        print(
            f"    {i:4d}-{j:4d} n={j - i + 1:3d} t {d['t_s'][i]:6.3f}-{d['t_s'][j]:6.3f} "
            f"endpt dur {dur:6.3f}s  chi_hold_bar(0.5s) {'PASS' if ok else 'FAIL'}  "
            f"sides {sides} {'PASS' if len(sides) == 1 else 'FAIL'}  "
            f"cpa {d['cpa_miss_m'][i]:8.1f} -> {d['cpa_miss_m'][j]:8.1f} "
            f"({'BUILT' if d['cpa_miss_m'][j] > d['cpa_miss_m'][i] else 'NOT BUILT -> REJECT'})  "
            f"alt {d['alt_m'][i]:7.1f}->{d['alt_m'][j]:7.1f}"
        )
    print(
        f"\n  => at least one hold clears the chi+load+duration clause: {any_pass_chi}\n"
        f"     if True, D3=0 is NOT caused by 'the lift vector never moved off the LOS plane'."
    )
    # D1
    hold1 = (ach > CHI_OUT) & (nz > NZ_LOADED)
    print(f"\n  D1 hold mask (|chi|>{CHI_OUT} AND nz>{NZ_LOADED}) runs:")
    for i, j in runs(hold1):
        print(
            f"    {i:4d}-{j:4d} n={j - i + 1:3d} t {d['t_s'][i]:6.3f}-{d['t_s'][j]:6.3f} "
            f"endpt dur {d['t_s'][j] - d['t_s'][i]:6.3f}s vs bar {CHI_MIN_HOLD_S} -> "
            f"{'PASS' if d['t_s'][j] - d['t_s'][i] >= CHI_MIN_HOLD_S else 'FAIL'}  "
            f"chi peak {ach[i : j + 1].max():.2f}"
        )
    # cpa monotonicity
    cpa = d["cpa_miss_m"]
    dv = np.diff(cpa)
    print(
        f"\n  cpa_miss_m: first {cpa[0]:.1f} last {cpa[-1]:.1f} max {cpa.max():.1f} at step "
        f"{int(cpa.argmax())}; rising steps {int((dv > 0).sum())}/{len(dv)}"
    )


def part5_nokill(nk):
    hdr("(5) NON-KILL SEED 28: does the SAME hold appear, and what does the detector say later?")
    part1_recompute(nk, "NOKILL seed 28")
    part2_phi_chi(nk, "NOKILL seed 28")
    part4_d3_clause(nk, "NOKILL seed 28")


def part6_prefix_identity(k, nk):
    hdr("(6) PREFIX IDENTITY: are the two traces the same state through step 172?")
    n = min(k["_n"], nk["_n"])
    for c in ("phi_deg", "chi_deg", "nz_g", "alt_m", "cpa_miss_m", "rng_m", "ata_az_deg"):
        dif = np.abs(k[c][:n] - nk[c][:n])
        print(
            f"  {c:14} max|diff| over steps 0-{n - 1} = {dif.max():.6e} at step {int(dif.argmax())}"
        )


def main():
    k = load(KILL)
    nk = load(NOKILL)
    print(f"KILL rows {k['_n']}, NOKILL rows {nk['_n']}, dt {DT * 1000:.4f} ms")
    part1_recompute(k, "KILL seed 62")
    part2_phi_chi(k, "KILL seed 62")
    part3_frame_attribution(k, "KILL seed 62")
    part4_d3_clause(k, "KILL seed 62")
    part6_prefix_identity(k, nk)
    part5_nokill(nk)
    part3_frame_attribution(nk, "NOKILL seed 28")


if __name__ == "__main__":
    main()
