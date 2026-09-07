"""Conditioning of chi across the WHOLE kill trace, with the firing window called out.

Amplification bound: chi's phi_los half is an atan2 of the LOS projected perpendicular to v_hat, so
displacing the bandit by an angle eps of arc moves phi_los by up to eps / sin(ata_v). Reported per
tick from the CSV geometry columns (no re-flight needed: ata_v is recoverable from the dumped
velocity and the two positions).
"""

from __future__ import annotations

import csv
import math

import numpy as np

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"


def main():
    with open(KILL) as fh:
        rows = list(csv.DictReader(fh))
    ata_v = []
    for r in rows:
        v = np.array([float(r["vel_e_ms"]), float(r["vel_n_ms"]), float(r["vel_u_ms"])])
        d = np.array(
            [
                float(r["opp_pos_e_m"]) - float(r["pos_e_m"]),
                float(r["opp_pos_n_m"]) - float(r["pos_n_m"]),
                float(r["opp_pos_u_m"]) - float(r["pos_u_m"]),
            ]
        )
        vh = v / np.linalg.norm(v)
        dh = d / np.linalg.norm(d)
        ata_v.append(math.degrees(math.acos(float(np.clip(vh @ dh, -1, 1)))))
    amp = [1.0 / math.sin(math.radians(a)) for a in ata_v]
    # `fire` is the agent-space a[4] in [-1, 1]: -1.0 = hold, +1.0 = trigger. Test > 0, not != 0.
    fire = [i for i, r in enumerate(rows) if float(r["fire"]) > 0.0]
    leth = [i for i, r in enumerate(rows) if int(float(r["smp_lethal"])) != 0]
    valid = sum(1 for r in rows if int(float(r["smp_chi_valid"])) != 0)

    def stat(idx, tag):
        a = [ata_v[i] for i in idx]
        g = [amp[i] for i in idx]
        print(
            f"   {tag:28s} n={len(idx):3d}  ata_v min {min(a):6.2f} med "
            f"{float(np.median(a)):6.2f} max {max(a):6.2f} deg   "
            f"amp 1/sin(ata_v) med {float(np.median(g)):5.2f} MAX {max(g):5.2f}"
        )

    print(f"KILL seed 62: {len(rows)} ticks; chi_valid True on {valid}/{len(rows)}")
    print(
        f"   CHI_VALID_ATA_DEG gate = 5.0 deg; ticks with ata_v < 5 deg: "
        f"{sum(1 for a in ata_v if a < 5.0)}"
    )
    stat(list(range(len(rows))), "whole trace")
    stat(fire, f"FIRING ticks {fire[0]}-{fire[-1]}")
    stat(leth, f"LETHAL ticks {leth[0]}-{leth[-1]}")
    stat([i for i in range(88, 109)], "D3-bar hold 88-108")
    worst = int(np.argmax(amp))
    print(
        f"   worst-conditioned tick: step {worst} t {float(rows[worst]['t_s']):.3f} s "
        f"ata_v {ata_v[worst]:.2f} deg amp {amp[worst]:.2f}  chi "
        f"{float(rows[worst]['chi_deg']):+.1f} deg  chi_valid "
        f"{int(float(rows[worst]['smp_chi_valid']))}  rng {float(rows[worst]['rng_m']):.0f} m"
    )
    over = [i for i in range(len(rows)) if amp[i] > 3.0]
    print(
        f"   ticks with amplification > 3.0 (ata_v < 19.5 deg): {len(over)}/{len(rows)}"
        f"  -> {over[0]}..{over[-1]}"
        if over
        else "   none"
    )
    print(
        f"   ALL of them reported chi_valid=True: "
        f"{all(int(float(rows[i]['smp_chi_valid'])) != 0 for i in over)}"
    )


if __name__ == "__main__":
    main()
