"""Reward, penalties and wrapper-enforced episode boundaries (design doc §8.3–§8.4).

Operates on `AState` snapshots and their `RelGeom` (both fabricable), so the
reward table and the boundary predicates are unit-testable with no live sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from . import constants as C
from .geometry import AState, RelGeom, potential


@dataclass
class RewardOut:
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def below_hard_deck(me: AState) -> bool:
    """Below the height-above-ellipsoid floor (no terrain model; see constants)."""
    return me.alt_hae_m < C.DECK_HAE_M


def outside_bubble(me: AState) -> bool:
    """Beyond the horizontal bubble around the grid origin or above its HAE ceiling."""
    return hypot(me.pos[0], me.pos[1]) > C.R_BUBBLE or me.alt_hae_m > C.CEILING_HAE_M


def compute(
    prev_me: AState,
    prev_bandit: AState,
    me: AState,
    bandit: AState,
    rg_prev: RelGeom,
    rg_new: RelGeom,
    gamma: float = C.GAMMA,
) -> RewardOut:
    """One step's reward + episode flags, from the (prev → new) state transition.

    `rg_prev` / `rg_new` are `relative(me, bandit, gun)` before and after.
    Precedence (§8.3–§8.4): hard deck first (the worst outcome), then kill/death,
    then bubble exit; dense terms (damage, shaping, time) always accrue.
    """
    info: dict = {}
    reward = C.TIME_PENALTY

    # Dense: damage dealt / taken (zero-sum in self-play).
    dmg_dealt = max(0.0, prev_bandit.hit_points - bandit.hit_points)
    dmg_taken = max(0.0, prev_me.hit_points - me.hit_points)
    reward += C.DMG_PER_HP * (dmg_dealt - dmg_taken)
    if dmg_dealt:
        info["damage_dealt"] = dmg_dealt
    if dmg_taken:
        info["damage_taken"] = dmg_taken

    # Dense: potential-based geometry shaping γΦ(s') − Φ(s) (telescopes; §8.3).
    phi_new = potential(me, bandit, rg_new)
    reward += C.K_GEOM * (gamma * phi_new - potential(prev_me, prev_bandit, rg_prev))
    info["phi"] = phi_new

    if below_hard_deck(me):
        info["event"] = "hard_deck"
        return RewardOut(reward + C.R_DECK, True, False, info)

    just_killed = prev_bandit.alive and not bandit.alive
    just_died = prev_me.alive and not me.alive
    if just_died:
        reward += C.R_DEATH
        info["event"] = "killed"
    if just_killed:
        reward += C.R_KILL
        info["event"] = "trade" if just_died else "kill"
    if just_died or just_killed:
        return RewardOut(reward, True, False, info)

    if outside_bubble(me):
        info["event"] = "left_bubble"
        return RewardOut(reward + C.R_BUBBLE_EXIT, False, True, info)

    return RewardOut(reward, False, False, info)
