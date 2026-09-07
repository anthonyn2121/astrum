"""Addendum: pin down the two turn-sense reversals and the 0.5 s rate windows."""

from __future__ import annotations

import csv
import os

import numpy as np

RUN = "/workspace/controllers/rl/runs/d3b"
DT = 5.0 / 120.0
with open(os.path.join(RUN, "replay_kill_seed62.csv")) as fh:
    rows = list(csv.DictReader(fh))
c = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "event"}
n = len(rows)


def u(a):
    return np.degrees(np.unwrap(np.radians(a)))


mypsi, oppsi = u(c["psi_deg"]), u(c["opp_psi_deg"])
mr, orr = np.gradient(mypsi, DT), np.gradient(oppsi, DT)
print("THE BANDIT'S REVERSAL WINDOW (steps 52..80)")
print(
    f"{'step':>5}{'t_s':>7}{'oppphi':>8}{'oppnz':>7}{'opp_psidot':>11}{'oppalt':>9}{'oppvt':>8}"
    f"{'my_psidot':>10}{'rng':>8}"
)
for i in range(52, 81):
    print(
        f"{i:>5}{c['t_s'][i]:>7.3f}{c['opp_phi_deg'][i]:>+8.2f}{c['opp_nz_g'][i]:>7.2f}"
        f"{orr[i]:>+11.2f}{c['opp_alt_m'][i]:>9.1f}{c['opp_vt_ms'][i]:>8.1f}{mr[i]:>+10.2f}"
        f"{c['rng_m'][i]:>8.1f}"
    )
print("\nMY DIP (steps 96..110)")
for i in range(96, 111):
    print(
        f"{i:>5}{c['t_s'][i]:>7.3f} phi {c['phi_deg'][i]:>+8.2f} nz {c['nz_g'][i]:>5.2f} "
        f"psidot {mr[i]:>+7.2f} gamma {c['gamma_deg'][i]:>+7.2f} cmd_bank {c['cmd_bank_deg'][i]:>+8.2f}"
    )
print("\n0.5 s (12-step) WINDOW RATE SERIES, deg/s")
for k in range(n // 12):
    a, b = k * 12, (k + 1) * 12
    print(
        f"  win {k} steps {a:>3}-{b:>3} t {a * DT:5.2f}-{b * DT:5.2f}s : mine "
        f"{(mypsi[b] - mypsi[a]) / (12 * DT):+7.2f}  his {(oppsi[b] - oppsi[a]) / (12 * DT):+7.2f}"
    )
print(f"\nopp_nz<0 steps: {np.where(c['opp_nz_g'] < 0)[0].tolist()}")
print(f"opp_nz<3 steps count {int(np.sum(c['opp_nz_g'] < 3.0))}/{n}")
print(f"my ata_az sign change: first step ata_az<0 = {int(np.argmax(c['ata_az_deg'] < 0))}")
print(
    f"my nz>3 contiguous: first {int(np.argmax(c['nz_g'] > 3))}, "
    f"count {int(np.sum(c['nz_g'] > 3))} of {n}"
)
