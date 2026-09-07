"""Adversarial re-derivation of the 'collision course was built about equally' claim.

Everything here is recomputed from the per-step dumps in runs/d3b/, with NO reuse of the
claim's own code. Two classes of counterfactual are reported side by side:

  LOCAL  -- the claim's estimator: freeze both POSITIONS at the evaluation step, swap one
            side's velocity DIRECTION back to its t=0 value, recompute `cpa_miss_m`.
  GLOBAL -- the position-level version: propagate one side ballistically (constant velocity)
            from t=0 and take the minimum range against the other side's ACTUAL dumped path.

The point of running both is that only the second one can speak to what BUILT the geometry.
"""

from __future__ import annotations

import math
import os

import numpy as np

RUNS = "/workspace/controllers/rl/runs/d3b"
KILL = os.path.join(RUNS, "replay_kill_seed62.csv")
NOKILL = os.path.join(RUNS, "replay_nokill_seed28.csv")


def load(path: str) -> dict:
    with open(path) as fh:
        head = fh.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return {k: arr[:, i] for i, k in enumerate(head)}


def cpa_miss_m(pos_me, vel_me, pos_op, vel_op) -> float:
    """bfm_detect.cpa_miss_m, retyped from source (bfm_detect.py:181-194)."""
    r = np.asarray(pos_op, float) - np.asarray(pos_me, float)
    v = np.asarray(vel_op, float) - np.asarray(vel_me, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-9 else max(0.0, -float(r @ v) / vv)
    return float(np.linalg.norm(r + v * t))


def enu(d: dict, who: str) -> np.ndarray:
    if who == "me":
        return np.stack([d["pos_e_m"], d["pos_n_m"], d["pos_u_m"]], 1)
    return np.stack([d["opp_pos_e_m"], d["opp_pos_n_m"], d["opp_pos_u_m"]], 1)


def vel_me(d: dict) -> np.ndarray:
    return np.stack([d["vel_e_ms"], d["vel_n_ms"], d["vel_u_ms"]], 1)


def vel_diff(pos: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference velocity; one-sided at the ends."""
    v = np.zeros_like(pos)
    v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
    v[0] = (pos[1] - pos[0]) / dt
    v[-1] = (pos[-1] - pos[-2]) / dt
    return v


def vel_from_attitude(d: dict) -> np.ndarray:
    """Opponent velocity from psi / (theta-alpha) / vt -- the no-differencing reconstruction."""
    psi = np.radians(d["opp_psi_deg"])
    gam = np.radians(d["opp_theta_deg"] - d["opp_alpha_deg"])
    vt = d["opp_vt_ms"]
    return np.stack(
        [vt * np.cos(gam) * np.sin(psi), vt * np.cos(gam) * np.cos(psi), vt * np.sin(gam)], 1
    )


def rot_to_heading(v: np.ndarray, psi_new: float) -> np.ndarray:
    """Rotate the HORIZONTAL part of v to compass heading psi_new; keep |v_h| and v_u."""
    vh = math.hypot(v[0], v[1])
    return np.array([vh * math.sin(psi_new), vh * math.cos(psi_new), v[2]])


def dir_to_t0(v: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Point v along the FULL 3-D t=0 unit vector, keeping |v|."""
    return float(np.linalg.norm(v)) * u0


def ang(a: np.ndarray, b: np.ndarray) -> float:
    ca = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, ca))))


def unwrap_deg(x: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(x)))


def report(tag: str, path: str, eval_steps: list[int]) -> None:
    d = load(path)
    n = len(d["step"])
    dt = float(d["t_s"][1] - d["t_s"][0])
    pm, po = enu(d, "me"), enu(d, "op")
    vm = vel_me(d)
    vo_fd = vel_diff(po, dt)
    vo_at = vel_from_attitude(d)
    vm_fd = vel_diff(pm, dt)

    print(f"\n{'=' * 92}\n{tag}  ({path})  n={n} steps, dt={dt:.6f} s\n{'=' * 92}")

    # ---- 1. does a reconstructed-velocity CPA reproduce the dumped column? ------------
    for name, vo in (("central-difference", vo_fd), ("psi/gamma/vt", vo_at)):
        err = np.array(
            [abs(cpa_miss_m(pm[i], vm[i], po[i], vo[i]) - d["cpa_miss_m"][i]) for i in range(n)]
        )
        print(
            f"CPA reconstruction ({name:19s}): max|err| {err.max():8.3f} m   "
            f"median {np.median(err):7.3f} m   p95 {np.percentile(err, 95):7.3f} m"
        )
    # own velocity is dumped directly; confirm the dump's own consistency
    print(
        f"own vel dumped-vs-differenced: max angle {max(ang(vm[i], vm_fd[i]) for i in range(1, n - 1)):.3f} deg"
    )

    vo = vo_fd  # the better of the two, chosen by the numbers printed above
    psi_m = unwrap_deg(d["psi_deg"])
    psi_o = unwrap_deg(d["opp_psi_deg"])
    u0m = vm[0] / np.linalg.norm(vm[0])
    u0o = vo[0] / np.linalg.norm(vo[0])

    # ---- 2. bandit onset claims -------------------------------------------------------
    b30 = np.argmax(np.abs(d["opp_phi_deg"]) > 30.0)
    b3g = np.argmax(d["opp_nz_g"] > 3.0)
    print(
        f"bandit |phi|>30 deg first at step {b30} (t {d['t_s'][b30]:.3f} s, "
        f"phi {d['opp_phi_deg'][b30]:.1f}); nz>3 g first at step {b3g} "
        f"(t {d['t_s'][b3g]:.3f} s, nz {d['opp_nz_g'][b3g]:.2f})"
    )

    # ---- 3. per-step counterfactual sweep --------------------------------------------
    print(
        "\nstep    t_s    rng_m  cpaDump   cpaAct   cpaMyHdg  cpaHisHdg  cpaBoth  "
        "cpaMy3D  cpaHis3D   share_me  |dV_me|  |dV_op|  tcpa_s"
    )
    rows = []
    for i in range(n):
        a = cpa_miss_m(pm[i], vm[i], po[i], vo[i])
        cm = cpa_miss_m(pm[i], rot_to_heading(vm[i], math.radians(d["psi_deg"][0])), po[i], vo[i])
        ch = cpa_miss_m(
            pm[i], vm[i], po[i], rot_to_heading(vo[i], math.radians(d["opp_psi_deg"][0]))
        )
        cb = cpa_miss_m(
            pm[i],
            rot_to_heading(vm[i], math.radians(d["psi_deg"][0])),
            po[i],
            rot_to_heading(vo[i], math.radians(d["opp_psi_deg"][0])),
        )
        c3m = cpa_miss_m(pm[i], dir_to_t0(vm[i], u0m), po[i], vo[i])
        c3h = cpa_miss_m(pm[i], vm[i], po[i], dir_to_t0(vo[i], u0o))
        dvm = float(np.linalg.norm(rot_to_heading(vm[i], math.radians(d["psi_deg"][0])) - vm[i]))
        dvo = float(
            np.linalg.norm(rot_to_heading(vo[i], math.radians(d["opp_psi_deg"][0])) - vo[i])
        )
        r = po[i] - pm[i]
        v = vo[i] - vm[i]
        tc = max(0.0, -float(r @ v) / float(v @ v))
        share = (cm - a) / max(1e-9, (cm - a) + (ch - a))
        rows.append((i, a, cm, ch, cb, c3m, c3h, share, dvm, dvo, tc))
        if i in eval_steps or i % 20 == 0 or i == n - 1:
            print(
                f"{i:4d} {d['t_s'][i]:6.3f} {d['rng_m'][i]:8.1f} {d['cpa_miss_m'][i]:8.1f} "
                f"{a:8.1f} {cm:9.1f} {ch:10.1f} {cb:8.1f} {c3m:8.1f} {c3h:9.1f}  "
                f"{share:8.3f} {dvm:8.1f} {dvo:8.1f} {tc:7.3f}"
            )

    # ---- 4. the additivity test the local estimator has to pass -----------------------
    for i in eval_steps:
        _, a, cm, ch, cb, c3m, c3h, share, dvm, dvo, tc = rows[i]
        print(
            f"\n[step {i}, t {d['t_s'][i]:.3f} s, rng {d['rng_m'][i]:.1f} m] "
            f"ADDITIVITY: actual {a:.1f}; +myHdg {cm - a:+.1f}; +hisHdg {ch - a:+.1f}; "
            f"sum {(cm - a) + (ch - a):+.1f}; JOINT {cb - a:+.1f}  "
            f"=> sum/joint = {((cm - a) + (ch - a)) / max(1e-9, cb - a):.3f}"
        )
        print(
            f"    |dV| test: |dV_me| {dvm:.1f} m/s, |dV_op| {dvo:.1f} m/s, t_cpa {tc:.3f} s "
            f"=> |dV|*t_cpa predicts {dvm * tc:.1f} / {dvo * tc:.1f} m "
            f"(measured {cm - a:.1f} / {ch - a:.1f})"
        )
        print(
            f"    turn to step {i}: psi me {psi_m[i] - psi_m[0]:+.1f} deg, op "
            f"{psi_o[i] - psi_o[0]:+.1f} deg | 3-D velocity-vector turn me "
            f"{ang(vm[0], vm[i]):.1f} deg, op {ang(vo[0], vo[i]):.1f} deg"
        )
        print(
            f"    speeds: vt me {d['vt_ms'][i]:.1f}, op {d['opp_vt_ms'][i]:.1f} m/s; "
            f"gamma me {d['gamma_deg'][i]:+.1f} deg"
        )

    sh = np.array([r[7] for r in rows])
    ok = np.array([r[1] < 400.0 for r in rows])  # only where a CPA is meaningful
    print(
        f"\nshare_me over steps with actual CPA<400 m (n={int(ok.sum())}): "
        f"min {sh[ok].min():.3f} median {np.median(sh[ok]):.3f} max {sh[ok].max():.3f}"
    )

    # ---- 5. GLOBAL (position-level) counterfactuals -----------------------------------
    t = d["t_s"]
    pm_str = pm[0] + np.outer(t, vm[0])
    po_str = po[0] + np.outer(t, vo[0])
    rng_act = np.linalg.norm(po - pm, axis=1)
    rng_me_str = np.linalg.norm(po - pm_str, axis=1)
    rng_op_str = np.linalg.norm(po_str - pm, axis=1)
    rng_both = np.linalg.norm(po_str - pm_str, axis=1)
    print(
        f"\nGLOBAL ballistic-propagation min range over the flown window:\n"
        f"  actual                       {rng_act.min():9.1f} m at step {int(rng_act.argmin())}\n"
        f"  I fly straight from t=0      {rng_me_str.min():9.1f} m at step {int(rng_me_str.argmin())}\n"
        f"  HE flies straight from t=0   {rng_op_str.min():9.1f} m at step {int(rng_op_str.argmin())}\n"
        f"  both fly straight from t=0   {rng_both.min():9.1f} m at step {int(rng_both.argmin())}"
    )
    print(f"  max|rng_act - dumped rng_m| = {np.abs(rng_act - d['rng_m']).max():.3f} m")

    # ---- 6. price both turns in altitude ---------------------------------------------
    print(
        f"\nprice of the two turns: me alt {d['alt_m'][0]:.0f} -> {d['alt_m'][-1]:.0f} m "
        f"({d['alt_m'][-1] - d['alt_m'][0]:+.0f}), Es {d['es_m'][0]:.0f} -> {d['es_m'][-1]:.0f} m; "
        f"op alt {d['opp_alt_m'][0]:.0f} -> {d['opp_alt_m'][-1]:.0f} m "
        f"({d['opp_alt_m'][-1] - d['opp_alt_m'][0]:+.0f}), Es {d['opp_es_m'][0]:.0f} -> "
        f"{d['opp_es_m'][-1]:.0f} m"
    )
    load_op = d["opp_nz_g"] > 3.0
    load_me = d["nz_g"] > 3.0
    print(
        f"loaded (nz>3 g) step counts: me {int(load_me.sum())}/{n} "
        f"({load_me.sum() * dt:.3f} s), op {int(load_op.sum())}/{n} ({load_op.sum() * dt:.3f} s)"
    )


if __name__ == "__main__":
    report("KILL SEED 62", KILL, [0, 100, 156, 172])
    report("NON-KILL SEED 28", NOKILL, [0, 100, 156, 172])
