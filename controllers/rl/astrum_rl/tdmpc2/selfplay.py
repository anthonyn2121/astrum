"""Self-play scaffolding (design doc §3, P5 — deferred beyond the scripted curriculum).

The scripted Tier-1/2/3 bots teach BFM fundamentals; self-play is how the policy is
meant to surpass them. This module supplies the pieces without committing the training
loop to a particular self-play schedule:

* `ScriptedPolicyOpponent` — drives the sim's *other* aircraft from an exported policy
  prior (`export.PolicyPrior`), conforming to the `astrum_dogfight` opponent interface
  (`reset(s)` / `act(s, opponent) → native action list`). It builds the same egocentric
  31-float observation the learner sees, from the bot's own point of view.
* `OpponentPool` — a growing set of frozen prior checkpoints to sample opponents from
  (uniform, or recency-biased à la prioritized fictitious self-play).
* `make_selfplay_env` — a dogfight env whose opponent is a pooled/loaded policy, wrapped
  with the same TD-MPC2 stack as `make_tdmpc_env`.

Wiring (left to the trainer, intentionally): periodically `export_policy` the current
prior into a pool dir, and every episode rebuild the env against `pool.sample()`. Kept
out of `train.py` so the P0–P4 curriculum path stays simple and dependency-free.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ..envs import map_tdmpc_action, wrap_tdmpc
from .export import ACTION_HIGH, ACTION_LOW, load_prior


class ScriptedPolicyOpponent:
    """An `astrum_dogfight` opponent flown by an exported TD-MPC2 policy prior."""

    def __init__(
        self,
        prior_path: str,
        *,
        frame_stack: int = 1,
        ground_alt: float = 0.0,
        device: str = "cpu",
    ):
        import torch

        self._torch = torch
        self.prior = load_prior(prior_path, torch.device(device))
        self.device = device
        self.frame_stack = max(1, int(frame_stack))
        self.ground_alt = float(ground_alt)
        self._low = np.asarray(ACTION_LOW, np.float32)
        self._high = np.asarray(ACTION_HIGH, np.float32)
        self._frames: deque = deque(maxlen=self.frame_stack)

    def reset(self, s) -> None:
        # The bandit state isn't available here (interface passes only our own state),
        # so the frame history fills lazily on the first act() call.
        self._frames.clear()

    def _stacked(self, obs: np.ndarray) -> np.ndarray:
        if self.frame_stack == 1:
            return obs
        if not self._frames:
            for _ in range(self.frame_stack):
                self._frames.append(obs)
        else:
            self._frames.append(obs)
        return np.concatenate(list(self._frames)).astype(np.float32)

    def act(self, s, opponent) -> list:
        from astrum_dogfight.geometry import build_observation

        obs = build_observation(s, opponent, self.ground_alt)
        obs = self._stacked(np.asarray(obs, np.float32))
        with self._torch.no_grad():
            x = self._torch.as_tensor(obs, dtype=self._torch.float32).unsqueeze(0)
            a = self.prior(x).squeeze(0).cpu().numpy()
        native = map_tdmpc_action(a, self._low, self._high)
        return native.tolist()


class OpponentPool:
    """A set of frozen policy-prior checkpoints to sample self-play opponents from."""

    def __init__(self, *, frame_stack: int = 1, device: str = "cpu", recency_bias: float = 0.0):
        self.frame_stack = frame_stack
        self.device = device
        self.recency_bias = float(recency_bias)  # 0 = uniform; >0 favors recent snapshots
        self._paths: list[str] = []

    def __len__(self) -> int:
        return len(self._paths)

    def add(self, prior_path: str) -> None:
        if not Path(prior_path).exists():
            raise FileNotFoundError(prior_path)
        self._paths.append(prior_path)

    def add_dir(self, directory: str, pattern: str = "*.pt") -> None:
        for p in sorted(Path(directory).glob(pattern)):
            self._paths.append(str(p))

    def _weights(self) -> np.ndarray:
        n = len(self._paths)
        if self.recency_bias <= 0.0:
            return np.full(n, 1.0 / n)
        w = np.exp(self.recency_bias * np.arange(n))  # older→newer increasing
        return w / w.sum()

    def sample(self, rng: np.random.Generator | None = None) -> ScriptedPolicyOpponent:
        if not self._paths:
            raise RuntimeError("opponent pool is empty; add() a prior first")
        rng = rng or np.random.default_rng()
        idx = int(rng.choice(len(self._paths), p=self._weights()))
        return ScriptedPolicyOpponent(
            self._paths[idx], frame_stack=self.frame_stack, device=self.device
        )


def make_selfplay_env(
    opponent,
    *,
    scenario=None,
    action_repeat: int = 5,
    frame_stack: int = 1,
    record_stats: bool = True,
    **scenario_kwargs,
):
    """A dogfight env vs a policy `opponent` (a `ScriptedPolicyOpponent` or a prior path),
    wrapped with the TD-MPC2 stack. `scenario` defaults to the Tier-1 geometry."""
    from astrum_dogfight.env import DogfightGymEnv
    from astrum_dogfight.scenario import tier1_scenario

    if isinstance(opponent, (str, Path)):
        opponent = ScriptedPolicyOpponent(str(opponent), frame_stack=frame_stack)
    base = DogfightGymEnv(scenario or tier1_scenario(**scenario_kwargs), opponent=opponent)
    return wrap_tdmpc(
        base, action_repeat=action_repeat, frame_stack=frame_stack, record_stats=record_stats
    )
