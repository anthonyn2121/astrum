"""`DogfightGymEnv` — the Gymnasium-compatible training adapter (design doc §8).

Single-agent from the trainer's view: the learner flies one f16, a scripted
opponent (`opponent.Tier1Opponent` by default) flies the other through the same
raw action matrix. Wraps `astrum_env.Env` and adds the observation vector
(§8.2), reward/penalties (§8.3) and the wrapper-enforced boundaries (§8.4).

Seeding follows Gymnasium: `reset(seed=s)` reseeds `self.np_random`; every reset
draws the episode's sim seed from it and passes it to `Env.reset(seed=...)`, so a
seeded first reset fixes the whole sequence of episodes. A new scenario for the
next episode is `reset(options={"scenario_toml": toml})` (same env, no rebuild).

numpy-only: subclasses `gymnasium.Env` iff gymnasium is importable, else exposes
the identical surface over a tiny space shim so it runs in the offline container.
"""

from __future__ import annotations

import astrum_env
import numpy as np

from . import constants as C
from . import reward as _reward
from .geometry import AState, GunSolution, build_observation, relative
from .opponent import Tier1Opponent, Tier2Opponent, Tier3Opponent
from .scenario import tier1_scenario, tier3_scenario

try:  # Gymnasium is optional — present it if installed, shim it otherwise.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
    _HAS_GYM = True
except ImportError:  # pragma: no cover - exercised only in the minimal container
    _Base = object
    _HAS_GYM = False

_SIM_SEED_BOUND = 2**63


class _Box:
    """Minimal Gymnasium-`Box`-compatible space for the no-gymnasium container."""

    def __init__(self, low, high, shape):
        self.low = np.broadcast_to(np.asarray(low, np.float32), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, np.float32), shape).copy()
        self.shape = tuple(shape)
        self.dtype = np.float32
        self._rng = np.random.default_rng()

    def seed(self, seed=None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return self._rng.uniform(lo, hi).astype(np.float32)

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
        gamma: float = C.GAMMA,
    ):
        super().__init__()
        self._env = astrum_env.Env(scenario_toml or tier1_scenario())
        self.learner_id = learner_id
        self._requested_opponent = opponent_id
        self._bind_ids()
        self.opponent = opponent or Tier1Opponent()
        self.gamma = gamma
        if not _HAS_GYM:
            self.np_random = np.random.default_rng()

        self.observation_space = _box(-np.inf, np.inf, (C.OBS_DIM,))
        # [aileron, elevator, rudder] ∈ [-1,1]; throttle, trigger ∈ [0,1].
        self.action_space = _box(
            [-1.0, -1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], (C.ACTION_DIM,)
        )
        self._actions = np.zeros((len(self._env.agent_ids), C.ACTION_DIM))

    # -- helpers -----------------------------------------------------------
    def _bind_ids(self) -> None:
        ids = list(self._env.agent_ids)
        if self.learner_id not in ids:
            raise ValueError(f"learner_id {self.learner_id} not in scenario ids {ids}")
        opp = self._requested_opponent
        if opp is None:
            others = [i for i in ids if i != self.learner_id]
            if len(others) != 1:
                raise ValueError(f"expected exactly one opponent, got ids {ids}")
            opp = others[0]
        elif opp not in ids:
            raise ValueError(f"opponent_id {opp} not in scenario ids {ids}")
        self.opponent_id = opp
        self._li = ids.index(self.learner_id)
        self._oi = ids.index(opp)

    def _snapshot(self, obs: np.ndarray) -> None:
        self._me = AState.from_env(self._env, obs[self._li])
        self._bandit = AState.from_env(self._env, obs[self._oi])
        self._gun_me = GunSolution.query(self._env, self.learner_id, self.opponent_id)
        self._gun_opp = GunSolution.query(self._env, self.opponent_id, self.learner_id)
        self._rg = relative(self._me, self._bandit, self._gun_me)

    @property
    def sim(self) -> astrum_env.Env:
        """The wrapped `astrum_env.Env` (read access: gun_solution, rotations)."""
        return self._env

    @property
    def dt(self) -> float:
        return self._env.dt

    @property
    def scenario_toml(self) -> str:
        return self._env.scenario_toml

    def aircraft(self, aircraft_id: int) -> AState:
        """The latest snapshot of the learner or the opponent."""
        if aircraft_id == self.learner_id:
            return self._me
        if aircraft_id == self.opponent_id:
            return self._bandit
        raise KeyError(aircraft_id)

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if _HAS_GYM:
            super().reset(seed=seed)
        else:
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
        sim_seed = int(self.np_random.integers(_SIM_SEED_BOUND))
        scenario = (options or {}).get("scenario_toml")
        obs = self._env.reset(seed=sim_seed, scenario_toml=scenario)
        if scenario is not None:
            self._bind_ids()
            self._actions = np.zeros((len(self._env.agent_ids), C.ACTION_DIM))
        self._snapshot(obs)
        self.opponent.reset(self._bandit, self._env.dt)
        return build_observation(self._me, self._bandit, self._rg), {"sim_seed": sim_seed}

    def step(self, action):
        me_prev, bandit_prev, rg_prev = self._me, self._bandit, self._rg
        self._actions[self._li] = np.asarray(action, dtype=np.float64).reshape(-1)[: C.ACTION_DIM]
        self._actions[self._oi] = self.opponent.act(bandit_prev, me_prev, self._gun_opp)
        obs, info = self._env.step(self._actions)
        self._snapshot(obs)

        r = _reward.compute(
            me_prev, bandit_prev, self._me, self._bandit, rg_prev, self._rg, self.gamma
        )
        info_out = {
            **r.info,
            "sim_terminated": info.terminated,
            "sim_truncated": info.truncated,
            "sim_time_s": info.sim_time_s,
        }
        return (
            build_observation(self._me, self._bandit, self._rg),
            float(r.reward),
            r.terminated or info.terminated,
            r.truncated or info.truncated,
            info_out,
        )


def make_tier1_env(**scenario_kwargs) -> DogfightGymEnv:
    """A Tier-1 (straight-and-level dummy) dogfight env."""
    return DogfightGymEnv(tier1_scenario(**scenario_kwargs))


def make_tier2_env(*, maneuver: str = "break", **scenario_kwargs) -> DogfightGymEnv:
    """A Tier-2 (predictable defensive maneuvers) dogfight env.

    The opponent starts ahead of the learner (Tier-1 geometry) and, on detecting
    it in its rear hemisphere, breaks or Split-Ses away.
    `maneuver` ∈ {"break", "split_s", "alternate"}.
    """
    return DogfightGymEnv(
        tier1_scenario(**scenario_kwargs), opponent=Tier2Opponent(maneuver=maneuver)
    )


def make_tier3_env(*, posture: str = "offensive", **scenario_kwargs) -> DogfightGymEnv:
    """A Tier-3 (FSM tactician + WEZ gun) dogfight env.

    `posture` picks the starting geometry: "offensive" puts the shooting bot on the
    learner's six (`tier3_scenario`); "defensive" puts it ahead with the learner on
    its tail (`tier1_scenario`), so the bot opens by breaking.
    """
    if posture == "offensive":
        scenario = tier3_scenario(**scenario_kwargs)
    elif posture == "defensive":
        scenario = tier1_scenario(**scenario_kwargs)
    else:
        raise ValueError(f"unknown posture {posture!r}")
    return DogfightGymEnv(scenario, opponent=Tier3Opponent())
