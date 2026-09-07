"""Adversarial verification of the D3b "nose-to-nose / one-circle entry" claim.

Recomputes every cited number from the dumped CSVs and from `geometry.body_axes_enu` verbatim.
Read-only: touches nothing but the two replay CSVs.
"""

from __future__ import annotations

import csv
import importlib
import math
import os
import sys
import types

import numpy as np

# Load astrum_dogfight.geometry VERBATIM without executing the package __init__ (which needs the
# compiled `astrum_env` extension). A shim package object with the real __path__ is enough.
_PKGDIR = "/workspace/crates/astrum-env/python/astrum_dogfight"
_shim = types.ModuleType("astrum_dogfight")
_shim.__path__ = [_PKGDIR]
sys.modules["astrum_dogfight"] = _shim
G = importlib.import_module("astrum_dogfight.geometry")
print(f"geometry loaded from {G.__file__}")

RUN = "/workspace/controllers/rl/runs/d3b"
KILL = os.path.join(RUN, "replay_kill_seed62.csv")
NOKILL = os.path.join(RUN, "replay_nokill_seed28.csv")
DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    cols = {}
    for k in rows[0]:
        try:
            cols[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            cols[k] = np.array([r[k] for r in rows])
    return cols, rows


def ata_of(phi, theta, psi, own_pos, other_pos):
    """ata, ata_az, ata_el (deg) of `other` seen from an aircraft at `own_pos`/attitude."""
    los = np.asarray(other_pos, float) - np.asarray(own_pos, float)
    n = np.linalg.norm(los)
    los_hat = los / n
    nose, stbd, down = G.body_axes_enu(phi, theta, psi)
    lx = float(np.dot(los_hat, nose))
    ly = float(np.dot(los_hat, stbd))
    lz = float(np.dot(los_hat, down))
    ata = math.degrees(math.atan2(math.hypot(ly, lz), lx))
    az = math.degrees(math.atan2(ly, lx))
    el = math.degrees(math.atan2(-lz, math.hypot(lx, ly)))
    return ata, az, el, n


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(a)))


def rate(a_deg):
    """Centred finite difference of an unwrapped angle series, deg/s."""
    u = unwrap_deg(a_deg)
    return np.gradient(u, DT)


def block(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


c, rows = load(KILL)
n = len(rows)
print(f"KILL CSV rows = {n}  (last step {int(c['step'][-1])}, t {c['t_s'][-1]:.4f} s)")

# ----------------------------------------------------------------- A. step-0 boresights
block("A. STEP-0 BORESIGHTS, recomputed from raw pos/attitude columns")
for i in (0,):
    me_pos = (c["pos_e_m"][i], c["pos_n_m"][i], c["pos_u_m"][i])
    op_pos = (c["opp_pos_e_m"][i], c["opp_pos_n_m"][i], c["opp_pos_u_m"][i])
    mine = ata_of(
        math.radians(c["phi_deg"][i]),
        math.radians(c["theta_deg"][i]),
        math.radians(c["psi_deg"][i]),
        me_pos,
        op_pos,
    )
    his = ata_of(
        math.radians(c["opp_phi_deg"][i]),
        math.radians(c["opp_theta_deg"][i]),
        math.radians(c["opp_psi_deg"][i]),
        op_pos,
        me_pos,
    )
    print(
        f"step {i}: dumped ata_az_deg  = {c['ata_az_deg'][i]:+.4f}   ata_deg = {c['ata_deg'][i]:.4f}"
    )
    print(f"        recomputed MY ata_az = {mine[1]:+.4f}   ata = {mine[0]:.4f}  el {mine[2]:+.4f}")
    print(f"        recomputed HIS ata_az= {his[1]:+.4f}   ata = {his[0]:.4f}  el {his[2]:+.4f}")
    print(f"        rng dumped {c['rng_m'][i]:.3f}  recomputed {mine[3]:.3f}")
    print(
        f"        my  psi {c['psi_deg'][i]:+.3f}  phi {c['phi_deg'][i]:+.3f}  theta {c['theta_deg'][i]:+.3f}"
    )
    print(
        f"        opp psi {c['opp_psi_deg'][i]:+.3f}  phi {c['opp_phi_deg'][i]:+.3f}  theta {c['opp_theta_deg'][i]:+.3f}"
    )
    print(f"        dumped aspect {c['aspect_deg'][i]:.3f}  hca {c['hca_deg'][i]:+.3f}")

# his ata_az over the whole trace
his_az = np.zeros(n)
his_ata = np.zeros(n)
for i in range(n):
    _a, _az, _el, _r = ata_of(
        math.radians(c["opp_phi_deg"][i]),
        math.radians(c["opp_theta_deg"][i]),
        math.radians(c["opp_psi_deg"][i]),
        (c["opp_pos_e_m"][i], c["opp_pos_n_m"][i], c["opp_pos_u_m"][i]),
        (c["pos_e_m"][i], c["pos_n_m"][i], c["pos_u_m"][i]),
    )
    his_ata[i], his_az[i] = _a, _az

block("B. BORESIGHT TIME SERIES (deg) — mine vs his, every 12 steps")
print(
    f"{'step':>5}{'t_s':>8}{'rng':>9}{'my_ata':>9}{'my_az':>9}{'his_ata':>9}{'his_az':>9}"
    f"{'hca':>9}{'aspect':>9}{'phi':>9}{'oppphi':>9}{'nz':>7}{'oppnz':>7}"
)
for i in list(range(0, n, 12)) + [n - 1]:
    print(
        f"{int(c['step'][i]):>5}{c['t_s'][i]:>8.3f}{c['rng_m'][i]:>9.1f}"
        f"{c['ata_deg'][i]:>9.2f}{c['ata_az_deg'][i]:>+9.2f}{his_ata[i]:>9.2f}{his_az[i]:>+9.2f}"
        f"{c['hca_deg'][i]:>+9.2f}{c['aspect_deg'][i]:>9.2f}{c['phi_deg'][i]:>+9.2f}"
        f"{c['opp_phi_deg'][i]:>+9.2f}{c['nz_g'][i]:>7.2f}{c['opp_nz_g'][i]:>7.2f}"
    )

# ----------------------------------------------------------------- C/D. bank claims
block("C. BANK CLAIMS")
print(f"my  phi at step 36 = {c['phi_deg'][36]:+.4f}")
print(f"my  phi min/max over steps 0..172 = {c['phi_deg'].min():+.3f} / {c['phi_deg'].max():+.3f}")
i0 = 36
print(
    f"my  phi min/max over steps 36..95  = {c['phi_deg'][36:96].min():+.3f} / {c['phi_deg'][36:96].max():+.3f}"
)
print(
    f"my  phi min/max over steps 36..172 = {c['phi_deg'][36:].min():+.3f} / {c['phi_deg'][36:].max():+.3f}"
)
first_ge = np.argmax(c["phi_deg"] >= 113.1)
print(f"first step with phi >= 113.1 : {int(c['step'][first_ge])} (t {c['t_s'][first_ge]:.3f} s)")
print(f"my  phi at 90 deg crossing   : first step phi>90 = {int(np.argmax(c['phi_deg'] > 90.0))}")
print(f"opp phi at step 24 = {c['opp_phi_deg'][24]:+.4f}")
print(
    f"opp phi[0] = {c['opp_phi_deg'][0]:+.4f}   max = {c['opp_phi_deg'].max():+.4f} at step "
    f"{int(np.argmax(c['opp_phi_deg']))}   min = {c['opp_phi_deg'].min():+.4f}"
)
print(f"opp phi last = {c['opp_phi_deg'][-1]:+.4f}")
print("opp phi every 8 steps:", " ".join(f"{v:+.1f}" for v in c["opp_phi_deg"][::8]))
print(f"opp nz min/max = {c['opp_nz_g'].min():.2f} / {c['opp_nz_g'].max():.2f}")

# ----------------------------------------------------------------- E. heading change
block("E. NET INERTIAL HEADING CHANGE (unwrapped psi)")
mypsi = unwrap_deg(c["psi_deg"])
oppsi = unwrap_deg(c["opp_psi_deg"])
print(
    f"my  psi[0] = {mypsi[0]:+.3f}  psi[172] = {mypsi[-1]:+.3f}  net = {mypsi[-1] - mypsi[0]:+.3f} deg"
    f"  over {c['t_s'][-1]:.3f} s  ->  mean {(mypsi[-1] - mypsi[0]) / c['t_s'][-1]:+.2f} deg/s"
)
print(
    f"opp psi[0] = {oppsi[0]:+.3f}  psi[172] = {oppsi[-1]:+.3f}  net = {oppsi[-1] - oppsi[0]:+.3f} deg"
    f"  over {c['t_s'][-1]:.3f} s  ->  mean {(oppsi[-1] - oppsi[0]) / c['t_s'][-1]:+.2f} deg/s"
)
mr = rate(c["psi_deg"])
orr = rate(c["opp_psi_deg"])
print(f"my  psi-dot: min {mr.min():+.2f}  max {mr.max():+.2f}  median {np.median(mr):+.2f} deg/s")
print(
    f"opp psi-dot: min {orr.min():+.2f}  max {orr.max():+.2f}  median {np.median(orr):+.2f} deg/s"
)
# also on the settled part of the turn only
print(f"my  psi-dot steps 24..172:  min {mr[24:].min():+.2f} max {mr[24:].max():+.2f}")
print(f"opp psi-dot steps 24..172:  min {orr[24:].min():+.2f} max {orr[24:].max():+.2f}")
print(
    f"sign agreement of psi-dot over 0..172: "
    f"{int(np.sum(np.sign(mr) == np.sign(orr)))}/{n} steps same sense"
)
neg_my = np.where(mr < 0)[0]
neg_op = np.where(orr < 0)[0]
print(f"steps where my psi-dot < 0 : {neg_my.tolist()[:20]} (count {neg_my.size})")
print(f"steps where opp psi-dot < 0: {neg_op.tolist()[:20]} (count {neg_op.size})")

# yaw-rate column cross-check (body r, not psi-dot)
print(f"my  yaw_rate_dps min/max = {c['yaw_rate_dps'].min():+.2f}/{c['yaw_rate_dps'].max():+.2f}")

# total turn rate (flight-path rotation), for contrast with psi-dot
vel = np.stack([c["vel_e_ms"], c["vel_n_ms"], c["vel_u_ms"]], axis=1)
vhat = vel / np.linalg.norm(vel, axis=1, keepdims=True)
tot = np.zeros(n - 1)
for i in range(n - 1):
    d = float(np.clip(np.dot(vhat[i], vhat[i + 1]), -1.0, 1.0))
    tot[i] = math.degrees(math.acos(d)) / DT
print(
    f"my TOTAL flight-path turn rate (3-D): min {tot.min():.2f} max {tot.max():.2f} "
    f"mean {tot.mean():.2f} deg/s   [vs psi-dot mean {(mypsi[-1] - mypsi[0]) / c['t_s'][-1]:.2f}]"
)

# ----------------------------------------------------------------- F. hca "pinned"
block("F. IS hca PINNED NEAR RECIPROCAL?")
h = c["hca_deg"]
print(f"hca[0] = {h[0]:+.3f}   min {h.min():+.3f}  max {h.max():+.3f}")
print(f"|hca| min {np.abs(h).min():.3f} max {np.abs(h).max():.3f}")
hu = unwrap_deg(h)
print(
    f"unwrapped hca: [0]={hu[0]:+.3f} [last]={hu[-1]:+.3f} range {hu.min():+.3f}..{hu.max():+.3f} "
    f"excursion {hu.max() - hu.min():.3f} deg"
)
print("|hca| every 12 steps:", " ".join(f"{abs(v):.1f}" for v in h[::12]))
print(
    f"aspect[0] {c['aspect_deg'][0]:.2f}  aspect[last] {c['aspect_deg'][-1]:.2f} "
    f"min {c['aspect_deg'].min():.2f} max {c['aspect_deg'].max():.2f}"
)

# ----------------------------------------------------------------- G. one-circle test
block("G. WAS IT EVER A CIRCLE? instantaneous horizontal turn centres")
vh = np.stack([c["vel_e_ms"], c["vel_n_ms"]], axis=1)
sp = np.linalg.norm(vh, axis=1)
opp_pos = np.stack([c["opp_pos_e_m"], c["opp_pos_n_m"]], axis=1)
me_pos = np.stack([c["pos_e_m"], c["pos_n_m"]], axis=1)
# bandit horizontal velocity by differencing position (no opp vel columns dumped)
opp_vh = np.gradient(opp_pos, DT, axis=0)
opp_sp = np.linalg.norm(opp_vh, axis=1)
my_course = np.degrees(np.arctan2(vh[:, 0], vh[:, 1]))
opp_course = np.degrees(np.arctan2(opp_vh[:, 0], opp_vh[:, 1]))
my_cdot = rate(my_course)
opp_cdot = rate(opp_course)
print(
    f"{'step':>5}{'my_r_m':>10}{'op_r_m':>10}{'centre_gap_m':>14}{'r_sum':>9}{'|r1-r2|':>9}{'rng':>9}"
)
for i in list(range(6, n, 12)):
    r1 = sp[i] / math.radians(abs(my_cdot[i])) if abs(my_cdot[i]) > 0.05 else float("inf")
    r2 = opp_sp[i] / math.radians(abs(opp_cdot[i])) if abs(opp_cdot[i]) > 0.05 else float("inf")

    # centre is 90 deg to the turn side of the course
    def centre(pos, course_deg, r, cdot):
        b = math.radians(course_deg + (90.0 if cdot > 0 else -90.0))
        return pos + r * np.array([math.sin(b), math.cos(b)])

    if math.isfinite(r1) and math.isfinite(r2):
        c1 = centre(me_pos[i], my_course[i], r1, my_cdot[i])
        c2 = centre(opp_pos[i], opp_course[i], r2, opp_cdot[i])
        gap = float(np.linalg.norm(c1 - c2))
        print(
            f"{int(c['step'][i]):>5}{r1:>10.1f}{r2:>10.1f}{gap:>14.1f}{r1 + r2:>9.1f}"
            f"{abs(r1 - r2):>9.1f}{c['rng_m'][i]:>9.1f}"
        )
    else:
        print(f"{int(c['step'][i]):>5}{r1:>10.1f}{r2:>10.1f}{'--':>14}")
print(
    f"\nrange monotone? rng[0]={c['rng_m'][0]:.1f} min={c['rng_m'].min():.1f} "
    f"max={c['rng_m'].max():.1f} last={c['rng_m'][-1]:.1f}"
)
d = np.diff(c["rng_m"])
print(
    f"steps with rng increasing: {int(np.sum(d > 0))} of {n - 1}  (max increase {d.max():+.3f} m/step)"
)
print(f"closure min/max = {c['closure_ms'].min():+.1f}/{c['closure_ms'].max():+.1f} m/s")

# ----------------------------------------------------------------- H. altitude price
block("H. PRICE THE HOLDS IN ALTITUDE (both aircraft)")
print(
    f"my  alt_m  [0]={c['alt_m'][0]:.1f} [95]={c['alt_m'][95]:.1f} [last]={c['alt_m'][-1]:.1f} "
    f"net {c['alt_m'][-1] - c['alt_m'][0]:+.1f} m"
)
print(
    f"opp alt_m  [0]={c['opp_alt_m'][0]:.1f} [95]={c['opp_alt_m'][95]:.1f} [last]={c['opp_alt_m'][-1]:.1f} "
    f"net {c['opp_alt_m'][-1] - c['opp_alt_m'][0]:+.1f} m"
)
print(
    f"my  gamma  [0]={c['gamma_deg'][0]:+.2f} min={c['gamma_deg'].min():+.2f} last={c['gamma_deg'][-1]:+.2f}"
)
print(f"my  nz     min={c['nz_g'].min():.2f} max={c['nz_g'].max():.2f}")
loaded = c["nz_g"] > 3.0
print(f"steps with nz>3.0 (NZ_LOADED): {int(loaded.sum())}/{n} = {loaded.sum() * DT:.3f} s")
oploaded = c["opp_nz_g"] > 3.0
print(f"opp steps with nz>3.0        : {int(oploaded.sum())}/{n} = {oploaded.sum() * DT:.3f} s")

block("H2. CAN THE CLAIMED RATE RANGES BE RECOVERED BY ANY WINDOW?")
for w in (6, 12, 24, 48, 96):
    nb = n // w
    mb = [(mypsi[(k + 1) * w] - mypsi[k * w]) / (w * DT) for k in range(nb)]
    ob = [(oppsi[(k + 1) * w] - oppsi[k * w]) / (w * DT) for k in range(nb)]
    print(
        f"window {w:>3} steps ({w * DT:.3f} s): mine {min(mb):+.2f}..{max(mb):+.2f}   "
        f"his {min(ob):+.2f}..{max(ob):+.2f}"
    )
for lo, hi, lab in ((0, 96, "seg0"), (96, 173, "seg1")):
    print(
        f"{lab}: mine {(mypsi[hi - 1] - mypsi[lo]) / ((hi - 1 - lo) * DT):+.2f}  "
        f"his {(oppsi[hi - 1] - oppsi[lo]) / ((hi - 1 - lo) * DT):+.2f} deg/s"
    )

# ----------------------------------------------------------------- I. seed independence
block("I. IS THE ENTRY SEED-SPECIFIC? seed 62 vs seed 28, steps 0..172")
c28, rows28 = load(NOKILL)
worst = []
for k in c:
    if isinstance(c[k][0], np.str_):
        continue
    a = c[k]
    b = c28[k][: len(a)]
    dd = float(np.max(np.abs(a - b))) if a.dtype.kind == "f" else 0.0
    worst.append((dd, k))
worst.sort(reverse=True)
print("largest |seed62 - seed28| over the overlapping 173 rows, top 8 columns:")
for dd, k in worst[:8]:
    print(f"   {k:>18}  {dd:.6e}")
print(f"opp_health[172]: seed62 {c['opp_health'][-1]:.1f}  seed28 {c28['opp_health'][172]:.1f}")
print(
    f"seed28 entry: ata_az[0]={c28['ata_az_deg'][0]:+.4f}  phi[36]={c28['phi_deg'][36]:+.3f}  "
    f"opp_phi[24]={c28['opp_phi_deg'][24]:+.3f}"
)
m28 = unwrap_deg(c28["psi_deg"])
o28 = unwrap_deg(c28["opp_psi_deg"])
print(f"seed28 net psi 0..172: mine {m28[172] - m28[0]:+.3f}  opp {o28[172] - o28[0]:+.3f}")
print(f"seed28 FULL 24 s net psi: mine {m28[-1] - m28[0]:+.3f}  opp {o28[-1] - o28[0]:+.3f}")
print(f"seed28 hca over full 24 s: min {c28['hca_deg'].min():+.2f} max {c28['hca_deg'].max():+.2f}")
print(
    f"seed28 rng: [0]={c28['rng_m'][0]:.1f} min={c28['rng_m'].min():.1f} max={c28['rng_m'].max():.1f} "
    f"last={c28['rng_m'][-1]:.1f}"
)
print(f"seed28 alt net: mine {c28['alt_m'][-1] - c28['alt_m'][0]:+.1f} m")

# ----------------------------------------------------------------- J. seed 28 full episode
block("J. SEED 28, FULL 24 s — does the 'one-circle' survive past the kill point?")
n28 = len(c28["step"])
his_az28 = np.zeros(n28)
for i in range(n28):
    _a, _az, _el, _r = ata_of(
        math.radians(c28["opp_phi_deg"][i]),
        math.radians(c28["opp_theta_deg"][i]),
        math.radians(c28["opp_psi_deg"][i]),
        (c28["opp_pos_e_m"][i], c28["opp_pos_n_m"][i], c28["opp_pos_u_m"][i]),
        (c28["pos_e_m"][i], c28["pos_n_m"][i], c28["pos_u_m"][i]),
    )
    his_az28[i] = _az
print(
    f"{'step':>5}{'t_s':>8}{'rng':>9}{'my_az':>9}{'his_az':>9}{'hca':>9}{'phi':>9}{'oppphi':>9}"
    f"{'nz':>7}{'alt':>9}{'seg':>5}"
)
for i in list(range(0, n28, 24)) + [n28 - 1]:
    print(
        f"{int(c28['step'][i]):>5}{c28['t_s'][i]:>8.3f}{c28['rng_m'][i]:>9.1f}"
        f"{c28['ata_az_deg'][i]:>+9.2f}{his_az28[i]:>+9.2f}{c28['hca_deg'][i]:>+9.2f}"
        f"{c28['phi_deg'][i]:>+9.2f}{c28['opp_phi_deg'][i]:>+9.2f}{c28['nz_g'][i]:>7.2f}"
        f"{c28['alt_m'][i]:>9.1f}{int(c28['seg_idx'][i]):>5}"
    )
m28 = unwrap_deg(c28["psi_deg"])
o28 = unwrap_deg(c28["opp_psi_deg"])
mr28 = rate(c28["psi_deg"])
or28 = rate(c28["opp_psi_deg"])
print(
    f"\nseed28 FULL net psi: mine {m28[-1] - m28[0]:+.1f} deg  his {o28[-1] - o28[0]:+.1f} deg over "
    f"{c28['t_s'][-1]:.2f} s"
)
print(
    f"seed28 psi-dot sign agreement over 576 steps: "
    f"{int(np.sum(np.sign(mr28) == np.sign(or28)))}/{n28}"
)
print(f"seed28 |hca| min {np.abs(c28['hca_deg']).min():.1f} max {np.abs(c28['hca_deg']).max():.1f}")
print(
    f"seed28 rng [0]={c28['rng_m'][0]:.1f} min={c28['rng_m'].min():.1f} at step "
    f"{int(np.argmin(c28['rng_m']))} max_after_min="
    f"{c28['rng_m'][int(np.argmin(c28['rng_m'])) :].max():.1f} last={c28['rng_m'][-1]:.1f}"
)
print(
    f"seed28 alt net {c28['alt_m'][-1] - c28['alt_m'][0]:+.1f} m   opp alt net "
    f"{c28['opp_alt_m'][-1] - c28['opp_alt_m'][0]:+.1f} m"
)
print(f"seed28 opp_health last {c28['opp_health'][-1]:.1f}  my health last {c28['health'][-1]:.1f}")
