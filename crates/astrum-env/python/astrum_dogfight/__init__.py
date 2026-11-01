"""Gymnasium-style dogfight training adapter over the embedded ASTRUM env.

This package is the *trainer's* side of the RL seam (design doc §8.2–§8.5). The
Rust binding (`astrum_env.Env`) is a pure state-in/state-out sim with its own
fire control (`Env.gun_solution`), attitude rotations (`Env.body_to_grid`) and
autopilot law (`astrum_env.FixedWingAutopilot`); this package adds what the sim
deliberately omits:

- an egocentric, normalized **observation vector** (`geometry.build_observation`),
- the **reward / penalty** function and the episode boundaries the sim doesn't
  enforce — hard deck + engagement bubble (`reward`),
- a **scripted opponent** driven through the same raw action interface the learner
  uses (`opponent`): Tier-1 straight-and-level dummy, Tier-2 predictable defenders
  that break / Split-S on detecting an attacker on their six, and a Tier-3 rule-based
  FSM tactician with a WEZ-gated gun.

It is numpy-only: it exposes a Gymnasium-compatible surface (`reset() -> (obs,
info)`, `step(a) -> (obs, reward, terminated, truncated, info)`,
`observation_space` / `action_space`) and subclasses `gymnasium.Env` iff
gymnasium is importable.
"""

from .constants import ACTION_DIM, OBS_DIM
from .env import DogfightGymEnv, make_tier1_env, make_tier2_env, make_tier3_env
from .opponent import (
    BreakTurn,
    SplitS,
    Tier1Opponent,
    Tier2Opponent,
    Tier3Opponent,
    wez_fire,
)

__all__ = [
    "DogfightGymEnv",
    "make_tier1_env",
    "make_tier2_env",
    "make_tier3_env",
    "Tier1Opponent",
    "Tier2Opponent",
    "Tier3Opponent",
    "BreakTurn",
    "SplitS",
    "wez_fire",
    "OBS_DIM",
    "ACTION_DIM",
]
