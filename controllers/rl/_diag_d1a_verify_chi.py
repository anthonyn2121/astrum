#!/usr/bin/env python3
"""D1a VERIFICATION (adversarial): is `chi` a stable quantity at the ata where the c96/c95
chase/asp30 gate clearances actually happen?

WHAT IS MEASURED

`bfm_detect.out_of_plane_angle` returns `chi = atan2(L.r, L.u) - atan2(LOS.r, LOS.u)` with both
vectors projected perpendicular to the velocity vector. The LOS term's magnitude perpendicular to
`v_hat` is `sin(ata)`, so the CLOCK angle of the projected LOS has gain `1/sin(ata)` with respect to
a transverse angular perturbation of the bandit. The module guards only the hard singularity
(`CHI_VALID_ATA_DEG = 5 deg`); it does not guard the ill-conditioned neighbourhood above it.

D1a's raw output records, on `chase/asp30`:
    c96 gated ticks: ata p50 8.6 deg, p90 9.2 deg, |chi| p50 156.1 deg
    c95 gated ticks: ata p50 8.9 deg, p90 10.9 deg, |chi| p50 138.8 deg
i.e. >=90% of both arms' gate clearances on that sub-cell sit within ~5 deg of the validity cutoff.

This script computes, with the detector's OWN function and fabricated `AState`s:
  (a) the exact numerical sensitivity `d|chi|` to a fixed TRANSVERSE displacement of the bandit
      (metres at fixed range), swept over ata; and
  (b) the full range of `chi` reachable at fixed ata and fixed attacker attitude by moving the
      bandit only around the CLOCK (same ata, same range) -- the span of `chi` values that a single
      "how far off the LOS plane is my lift vector" reading is compatible with.

WHAT THIS DOES NOT MEASURE

It is not a rollout, it does not touch a checkpoint or the sim, and it makes NO claim about any
arm's behaviour. It bounds the CONDITIONING of the statistic at a stated geometry. It cannot say
what fraction of any arm's gated ticks are noise -- only how much `chi` moves per metre there.

PRE-REGISTERED INTERPRETATION

If the amplification at ata ~ 9 deg is <= ~2x the amplification at ata ~ 60 deg, the low-ata gate
clearances are fine and this objection is withdrawn. If it is >= ~5x, then D1a's `|chi| > 45 deg`
clause is being read where the quantity is ill-conditioned, and any per-arm gate fraction dominated
by ata < ~15 deg ticks must be reported as such and must not be compared against an arm whose gated
ticks sit at ata > 100 deg.
"""

from __future__ import annotations

import math

import numpy as np
from astrum_rl import _paths


def make_state(AState, pos, vel, phi, theta, psi):
    return AState(
        id=0,
        team=0,
        alive=True,
        pos=np.asarray(pos, float),
        vel=np.asarray(vel, float),
        phi=phi,
        theta=theta,
        psi=psi,
        p=0.0,
        q=0.0,
        r=0.0,
        alpha=0.0,
        beta=0.0,
        nz=5.0,
        airspeed=float(np.linalg.norm(vel)),
        alt=float(pos[2]),
        health=100.0,
        ammo=500,
    )


def main() -> None:
    _paths.require_astrum_env()
    from astrum_dogfight import bfm_detect as B
    from astrum_dogfight.geometry import AState

    DEG = 180.0 / math.pi
    RNG_M = 500.0  # c96's gated-tick median range on chase/asp30 was 489 m
    PERT_M = 10.0  # a 10 m transverse wobble of the bandit at that range

    # Attacker: level heading north at 200 m/s, banked 60 deg right (a loaded turn).
    me = make_state(AState, [0.0, 0.0, 3000.0], [0.0, 200.0, 0.0], math.radians(60.0), 0.0, 0.0)
    v_hat = np.array([0.0, 1.0, 0.0])
    u_ref = np.array([0.0, 0.0, 1.0])
    r_ref = np.cross(v_hat, u_ref)  # = +E

    print(
        f"CHI_VALID_ATA_DEG = {B.CHI_VALID_ATA_DEG:.1f} deg | rng {RNG_M:.0f} m"
        f" | transverse perturbation {PERT_M:.0f} m ({DEG * PERT_M / RNG_M:.2f} deg)"
    )
    print("ata_deg  1/sin(ata)  |d chi| for the 10 m wobble (deg)   chi span over clock (deg)")
    for ata_deg in (6.0, 8.6, 9.2, 10.9, 15.0, 30.0, 60.0, 90.0, 120.0):
        ata = math.radians(ata_deg)

        # Bandit at (ata, clock) around v_hat.
        def bandit_at(clock_rad, extra_perp=0.0):
            d = math.cos(ata) * v_hat + math.sin(ata) * (
                math.cos(clock_rad) * u_ref + math.sin(clock_rad) * r_ref
            )
            # `extra_perp` displaces the bandit TANGENTIALLY (perpendicular to both v_hat and the
            # radial direction), i.e. the worst case for the clock angle.
            tang = -math.sin(clock_rad) * u_ref + math.cos(clock_rad) * r_ref
            p = me.pos + RNG_M * d + extra_perp * tang
            return make_state(AState, p, [0.0, 200.0, 0.0], 0.0, 0.0, 0.0)

        chis = []
        for k in range(72):
            clock = 2.0 * math.pi * k / 72.0
            c0, v0 = B.out_of_plane_angle(me, bandit_at(clock, 0.0))
            c1, v1 = B.out_of_plane_angle(me, bandit_at(clock, PERT_M))
            if v0 and v1:
                chis.append((abs(c0) * DEG, abs(abs(c1) - abs(c0)) * DEG, c0 * DEG))
        if not chis:
            print(f"{ata_deg:7.1f}  {'INVALID (chi_valid False)':>50s}")
            continue
        dmax = max(c[1] for c in chis)
        dmed = float(np.median([c[1] for c in chis]))
        span = max(c[2] for c in chis) - min(c[2] for c in chis)
        print(
            f"{ata_deg:7.1f}  {1.0 / math.sin(ata):10.2f}  med {dmed:6.2f}  max {dmax:6.2f}"
            f"                {span:8.1f}"
        )

    print()
    print("Read: at fixed ata and fixed ATTITUDE, chi spans the full +/-180 deg purely from the")
    print("bandit's CLOCK position, and the per-metre gain on that clock angle is 1/sin(ata).")


if __name__ == "__main__":
    main()
