"""From-scratch TD-MPC2 for the ASTRUM dogfight env (design doc §2).

A latent world model (`WorldModel`) planned through with MPPI (`TDMPC2.plan`), trained
off-policy from an episodic replay buffer. No stable-baselines3, no tensordict — the
buffer is plain torch tensors and the algorithm is spelled out in `agent.py`.
"""

from __future__ import annotations

from .agent import TDMPC2
from .buffer import ReplayBuffer
from .config import TDMPC2Config
from .world_model import WorldModel

__all__ = ["TDMPC2Config", "WorldModel", "TDMPC2", "ReplayBuffer"]
