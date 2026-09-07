"""PROBE 5(e) — price a REAL out-of-plane conversion in reward units, at the training altitude.

Flies a scripted displacement-roll / nose-low reversal (roll 180, hold max pull) from the
actual c94/c95 start (block 15,000 ft + 2,000 ft perch = 17,000 ft MSL, ground = 0) through
`DogfightGymEnv` with the SAME reward the trainer uses (outcome_mode=True), and sums every
altitude-coupled term: deck cushion, K_SPIRAL, K_ENERGY, K_REL_ENERGY, band, corner.

Also repeats the identical maneuver from a LOWER staged block so the taxes can be compared.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import DogfightGymEnv
from astrum_dogfight.opponent import Tier3Opponent
from astrum_dogfight.scenario import tier3_scenario

FT = 0.3048
KEYS = (
    "deck_penalty",
    "spiral_penalty",
    "energy_reward",
    "rel_energy_reward",
    "band_reward",
    "corner_penalty",
)


def run(block_ft: float, perch_ft: float, secs: float, mode: str) -> None:
    env = DogfightGymEnv(
        tier3_scenario(
            alt_ft=block_ft,
            me_dalt_ft=perch_ft,
            aspect_deg=0.0,
            opp_heading_deg=180.0,
            separation_m=3000.0,
            max_time_s=secs + 5.0,
        ),
        opponent=Tier3Opponent(),
        outcome_mode=True,
    )
    obs, info = env.reset()
    tot = {k: 0.0 for k in KEYS}
    me = env._states[env.learner_id]
    alt0 = me.alt
    v0 = me.airspeed
    n = int(secs * 120)
    for _ in range(n):
        me = env._states[env.learner_id]
        if mode == "splitS":  # roll inverted, max pull -> nose-low reversal
            bank_tgt = math.pi
        elif mode == "level":  # 85 deg bank in-plane break (the control)
            bank_tgt = math.radians(85.0)
        else:  # nose-high egg
            bank_tgt = 0.0
        err = (bank_tgt - me.phi + math.pi) % (2 * math.pi) - math.pi
        ail = float(np.clip(2.0 * err, -1.0, 1.0))
        act = np.array([ail, -1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        obs, r, term, trunc, inf = env.step(act)
        for k in KEYS:
            tot[k] += float(inf.get(k, 0.0))
        if term or trunc:
            break
    me = env._states[env.learner_id]
    dh_ft = (me.alt - alt0) / FT
    print(
        f"  {mode:7s} block {block_ft:6.0f}+{perch_ft:5.0f} ft  {secs:4.1f}s  "
        f"dh {dh_ft:+7.0f} ft  V {v0:5.1f}->{me.airspeed:5.1f}  "
        f"endAGL {me.alt / FT:7.0f} ft  || "
        + "  ".join(f"{k.split('_')[0]}={tot[k]:+7.3f}" for k in KEYS)
        + f"  || SUM={sum(tot.values()):+7.3f}  (R_KILL={C.R_KILL:.0f})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=6.0)
    a = ap.parse_args()
    print(
        "altitude-coupled reward terms over ONE scripted conversion "
        f"(outcome_mode=True, ground=0, DECK_AGL={C.DECK_AGL:.0f} m, "
        f"SPIRAL_ALT_BAND_AGL={C.SPIRAL_ALT_BAND_AGL:.0f} m)"
    )
    for mode in ("level", "splitS", "egg"):
        for block, perch in ((15000.0, 2000.0), (8000.0, 2000.0), (4000.0, 2000.0)):
            run(block, perch, a.secs, mode)


if __name__ == "__main__":
    main()
