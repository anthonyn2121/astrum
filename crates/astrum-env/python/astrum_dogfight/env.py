"""`DogfightGymEnv` — the Gymnasium-compatible training adapter (design doc §8).

Single-agent from the trainer's view: the learner flies one f16, a **scripted**
opponent (`opponent.Tier1Opponent` by default) flies the other, driven internally
through the same raw action interface. Wraps `astrum_env.DogfightEnv` and adds the
observation vector (§8.2), reward/penalties (§8.3), and the wrapper-enforced
boundaries (§8.4).

numpy-only: subclasses `gymnasium.Env` iff gymnasium is importable, else exposes
the identical surface over a tiny space shim so it runs in the offline container.
"""

from __future__ import annotations

import astrum_env
import numpy as np

from . import constants as C
from . import reward as _reward
from .geometry import AState, build_observation
from .opponent import Tier1Opponent, Tier2Opponent, Tier3Opponent
from .scenario import tier1_scenario, tier3_scenario

try:  # Gymnasium is optional — present it if installed, shim it otherwise.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
    _HAS_GYM = True
except Exception:  # pragma: no cover - exercised only in the minimal container
    _Base = object
    _HAS_GYM = False


class _Box:
    """Minimal Gymnasium-`Box`-compatible space for the no-gymnasium container."""

    def __init__(self, low, high, shape):
        self.low = np.broadcast_to(np.asarray(low, np.float32), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, np.float32), shape).copy()
        self.shape = tuple(shape)
        self.dtype = np.float32

    def sample(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.uniform(lo, hi).astype(np.float32)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))


def _box(low, high, shape):
    if _HAS_GYM:
        return _spaces.Box(
            low=np.asarray(low, np.float32),
            high=np.asarray(high, np.float32),
            shape=shape,
            dtype=np.float32,
        )
    return _Box(low, high, shape)


class DogfightGymEnv(_Base):
    """A 1v1 gun-dogfight vs a scripted opponent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_toml: str | None = None,
        *,
        learner_id: int = 1,
        opponent_id: int | None = None,
        opponent=None,
        ground_alt: float = 0.0,
        gamma: float = C.GAMMA,
    ):
        super().__init__()
        self._env = astrum_env.DogfightEnv(scenario_toml or tier1_scenario())
        ids = self._env.agent_ids()
        if learner_id not in ids:
            raise ValueError(f"learner_id {learner_id} not in scenario ids {ids}")
        self.learner_id = learner_id
        if opponent_id is None:
            others = [i for i in ids if i != learner_id]
            if len(others) != 1:
                raise ValueError(f"expected exactly one opponent, got ids {ids}")
            opponent_id = others[0]
        self.opponent_id = opponent_id
        self.opponent = opponent or Tier1Opponent()
        self.ground_alt = ground_alt
        self.gamma = gamma

        self.observation_space = _box(-np.inf, np.inf, (C.OBS_DIM,))
        # [aileron, elevator, rudder] ∈ [-1,1]; throttle, trigger ∈ [0,1].
        self.action_space = _box(
            [-1.0, -1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], (C.ACTION_DIM,)
        )
        self._states: dict[int, AState] = {}

    # -- helpers -----------------------------------------------------------
    def _snapshot(self, obs_list) -> dict[int, AState]:
        return {int(o.id): AState.from_obs(o) for o in obs_list}

    def _obs_vec(self) -> np.ndarray:
        return build_observation(
            self._states[self.learner_id], self._states[self.opponent_id], self.ground_alt
        )

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # The scenario carries its own deterministic sim.seed; reset seed is a no-op.
        self._states = self._snapshot(self._env.reset())
        self.opponent.reset(self._states[self.opponent_id])
        return self._obs_vec(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        me_prev = self._states[self.learner_id]
        bandit_prev = self._states[self.opponent_id]

        opp_action = self.opponent.act(bandit_prev, me_prev)
        actions = [
            (self.learner_id, [float(x) for x in action[: C.ACTION_DIM]]),
            (self.opponent_id, [float(x) for x in opp_action]),
        ]
        obs_list, info = self._env.step(actions)
        self._states = self._snapshot(obs_list)
        me = self._states[self.learner_id]
        bandit = self._states[self.opponent_id]

        r = _reward.compute(me_prev, bandit_prev, me, bandit, self.ground_alt, self.gamma)
        terminated = r.terminated or bool(info.terminated)
        truncated = r.truncated or bool(info.truncated)
        info_out = {
            **r.info,
            "sim_terminated": bool(info.terminated),
            "sim_truncated": bool(info.truncated),
            "sim_time_s": float(info.sim_time_s),
        }
        return self._obs_vec(), float(r.reward), terminated, truncated, info_out


def make_tier1_env(**scenario_kwargs) -> DogfightGymEnv:
    """Convenience: a Tier-1 (straight-and-level dummy) dogfight env."""
    return DogfightGymEnv(tier1_scenario(**scenario_kwargs))


def make_tier2_env(*, maneuver: str = "break", **scenario_kwargs) -> DogfightGymEnv:
    """Convenience: a Tier-2 (predictable defensive maneuvers) dogfight env.

    The opponent starts on the learner's tail-side (same setup as Tier 1) and, on
    detecting the learner in its rear hemisphere, breaks or Split-Ses away.
    `maneuver` ∈ {"break", "split_s", "alternate"}.
    """
    rate_hz = scenario_kwargs.get("rate_hz", 120.0)
    opponent = Tier2Opponent(maneuver=maneuver, dt=1.0 / rate_hz)
    return DogfightGymEnv(tier1_scenario(**scenario_kwargs), opponent=opponent)


def make_tier3_env(*, posture: str = "offensive", **scenario_kwargs) -> DogfightGymEnv:
    """Convenience: a Tier-3 (FSM tactician + WEZ gun) dogfight env.

    `posture` picks the starting geometry: "offensive" puts the shooting bot on the
    learner's six (`tier3_scenario`); "defensive" puts it ahead with the learner on
    its tail (`tier1_scenario`), so the bot opens by breaking. The opponent is a
    `Tier3Opponent` either way.
    """
    rate_hz = scenario_kwargs.get("rate_hz", 120.0)
    opponent = Tier3Opponent(dt=1.0 / rate_hz)
    if posture == "offensive":
        scenario = tier3_scenario(**scenario_kwargs)
    elif posture == "defensive":
        scenario = tier1_scenario(**scenario_kwargs)
    else:
        raise ValueError(f"unknown posture {posture!r}")
    return DogfightGymEnv(scenario, opponent=opponent)
