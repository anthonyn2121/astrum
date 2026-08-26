"""Reward, penalties and wrapper-enforced episode boundaries (design doc §8.3–§8.4).

Operates purely on `AState` snapshots (fabricable), so the reward table and the
boundary predicates are unit-testable with no live sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from . import constants as C
from .geometry import AState, potential


@dataclass
class RewardOut:
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def below_hard_deck(me: AState, ground_alt: float = 0.0) -> bool:
    return (me.alt - ground_alt) < C.DECK_AGL


def outside_bubble(me: AState, ground_alt: float = 0.0, center=(0.0, 0.0)) -> bool:
    horiz = hypot(me.pos[0] - center[0], me.pos[1] - center[1])
    return horiz > C.R_BUBBLE or (me.alt - ground_alt) > C.CEILING_AGL


def compute(
    prev_me: AState,
    prev_bandit: AState,
    me: AState,
    bandit: AState,
    ground_alt: float = 0.0,
    gamma: float = C.GAMMA,
) -> RewardOut:
    """One step's reward + episode flags, from the (prev → new) state transition.

    Precedence for the terminal/truncation split (§8.3–§8.4): hard deck first (the
    worst outcome), then kill/death, then bubble exit; dense terms (damage,
    shaping, time) always accrue.
    """
    info: dict = {}
    reward = C.TIME_PENALTY

    # Dense: damage dealt / taken (zero-sum in self-play).
    dmg_dealt = max(0.0, prev_bandit.health - bandit.health)
    dmg_taken = max(0.0, prev_me.health - me.health)
    reward += C.DMG_PER_HP * dmg_dealt - C.DMG_PER_HP * dmg_taken
    if dmg_dealt:
        info["damage_dealt"] = dmg_dealt
    if dmg_taken:
        info["damage_taken"] = dmg_taken

    # Dense: potential-based geometry shaping γΦ(s') − Φ(s) (telescopes; §8.3).
    phi_prev = potential(prev_me, prev_bandit)
    phi_new = potential(me, bandit)
    reward += C.K_GEOM * (gamma * phi_new - phi_prev)
    info["phi"] = phi_new

    terminated = False
    truncated = False

    # Terminal: hard-deck bust (checked first — the single worst outcome).
    if below_hard_deck(me, ground_alt):
        reward += C.R_DECK
        terminated = True
        info["event"] = "hard_deck"
        return RewardOut(reward, terminated, truncated, info)

    # Terminal: kill / death (the newly-dead transition).
    just_killed = prev_bandit.alive and not bandit.alive
    just_died = prev_me.alive and not me.alive
    if just_died:
        reward += C.R_DEATH
        terminated = True
        info["event"] = "killed"
    if just_killed:
        reward += C.R_KILL
        terminated = True
        info["event"] = "kill" if not just_died else "trade"
    if terminated:
        return RewardOut(reward, terminated, truncated, info)

    # Truncate: left the engagement bubble.
    if outside_bubble(me, ground_alt):
        reward += C.R_BUBBLE_EXIT
        truncated = True
        info["event"] = "left_bubble"

    return RewardOut(reward, terminated, truncated, info)
