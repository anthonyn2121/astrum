"""Env construction + the two policy→env action mappings.

The learner's action space is mixed-range: `[aileron, elevator, rudder] ∈ [-1,1]`
and `[throttle, trigger] ∈ [0,1]` (see `astrum_dogfight.constants.ACTION_DIM`).
Two learners map onto it differently:

* **PPO** (`map_policy_action` / `_SquashActionWrapper`) emits an *unbounded*
  Gaussian in R^5, squashed by tanh (`[0:3]`) and sigmoid (`[3:5]`). The squash is
  treated as part of the env (the stored action/log-prob are the raw sample), which
  keeps the PPO objective simple.
* **TD-MPC2** (`map_tdmpc_action` / `_LinearActionWrapper`) acts in a *bounded*
  `[-1,1]^5` box (its policy prior tanh-squashes and its planner clamps to that box).
  Here the right map is an **affine rescale**, NOT tanh/sigmoid: a tanh would cap the
  agent's `±1` at `±0.76` native, so the jet could never command full-scale surfaces
  / the full 9 g pull. See docs/dogfight-tdmpc2-design.md §3.2/§6.

`make_tdmpc_env` also stacks the TD-MPC2-specific wrappers: an `ActionRepeat`
(act at ~24 Hz over the 120 Hz sim; the value function carries the long horizon) and
an optional flat `FrameStack`. Both preserve the env's separate `terminated`
(kill/death/deck) vs `truncated` (bubble/time) flags — TD-MPC2's value bootstrap and
termination head depend on that split.
"""

from __future__ import annotations

from collections import deque
from typing import Callable

import gymnasium as gym
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier1_env, make_tier2_env, make_tier3_env

_FACTORIES = {1: make_tier1_env, 2: make_tier2_env, 3: make_tier3_env}


def map_policy_action(raw: np.ndarray) -> np.ndarray:
    """Squash an unbounded R^5 policy sample into the env's action ranges.

    `[0:3]` → tanh → [-1, 1] (control-surface commands);
    `[3:5]` → sigmoid → [0, 1] (throttle, trigger). trigger > 0.5 fires the gun.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    out = np.empty(C.ACTION_DIM, dtype=np.float32)
    out[:3] = np.tanh(raw[:3])
    out[3:] = 1.0 / (1.0 + np.exp(-raw[3:]))
    return out


class _SquashActionWrapper(gym.ActionWrapper):
    """Apply `map_policy_action` so the env accepts raw Gaussian samples directly."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # The policy sees an unbounded action space; the wrapper does the squashing.
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(C.ACTION_DIM,), dtype=np.float32
        )

    def action(self, action):
        return map_policy_action(action)


def make_env(tier: int = 1, *, squash: bool = True, record_stats: bool = True, **kwargs):
    """Build one dogfight env (Tier 1/2/3), optionally with the squash + stats wrappers."""
    if tier not in _FACTORIES:
        raise ValueError(f"tier must be one of {sorted(_FACTORIES)}, got {tier}")
    env = _FACTORIES[tier](**kwargs)
    if squash:
        env = _SquashActionWrapper(env)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def env_thunk(tier: int = 1, **kwargs) -> Callable[[], gym.Env]:
    """A no-arg factory for `gymnasium.vector.AsyncVectorEnv` (one env per process).

    JSBSim is process-global AND not thread-safe (design doc §8.6), so vectorized
    rollouts MUST use AsyncVectorEnv (separate processes) — never SyncVectorEnv
    (all envs in one process → shared/corrupted FDM state) or a thread pool.

    Two rules keep the process-global state safe under AsyncVectorEnv:
      1. Build it with `context="spawn"` (fork would copy an already-initialized
         JSBSim into every child):
             AsyncVectorEnv([env_thunk(1) for _ in range(8)], context="spawn")
      2. Never construct a DogfightEnv in the PARENT before the vector env — read
         dims from `astrum_dogfight.constants` (OBS_DIM / ACTION_DIM) instead of
         instantiating one to probe its spaces.
    """

    def _thunk() -> gym.Env:
        return make_env(tier, **kwargs)

    return _thunk


# ======================================================================
# TD-MPC2 side: bounded [-1,1]^5 actions + action-repeat + frame-stack.
# ======================================================================


def map_tdmpc_action(action, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Affine-map a bounded `[-1,1]^A` agent action into the env's native ranges.

    `native = low + (a+1)/2 · (high-low)`, so `a=-1 → low`, `a=+1 → high`, linearly.
    For `[ail,elev,rud]` (native `[-1,1]`) this is the identity; for `[throttle,
    trigger]` (native `[0,1]`) it is `(a+1)/2`. Hence the trigger fires (native
    `≥0.5`) exactly when the agent's trigger `≥0` — the locked "continuous, threshold
    at env, fire when >0" contract. `a` is clipped to `[-1,1]` first (exploration
    noise can nudge it just outside the box).
    """
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return low + 0.5 * (a + 1.0) * (high - low)


def invert_tdmpc_action(native, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of `map_tdmpc_action`: an env-native action back to the agent's `[-1,1]^A`.

    `a = 2·(native−low)/(high−low) − 1`. Used to convert the env's raw `expert_action`
    (surfaced in info for the collector's DAgger-style expert rollouts) into the agent-space
    action the policy/buffer speak — so a stored expert action re-maps to itself through
    `map_tdmpc_action`. Degenerate `high==low` channels map to `-1` (never happens here)."""
    native = np.asarray(native, dtype=np.float32).reshape(-1)
    span = high - low
    a = np.where(span > 0, 2.0 * (native - low) / np.where(span > 0, span, 1.0) - 1.0, -1.0)
    return np.clip(a.astype(np.float32), -1.0, 1.0)


class _LinearActionWrapper(gym.ActionWrapper):
    """Expose a bounded `[-1,1]^A` action space, affine-mapped to the env's ranges.

    The affine (not tanh/sigmoid) map is deliberate: TD-MPC2's policy already lives
    in `[-1,1]`, and a squash would shrink full-scale commands (a `9 g` pull needs
    native elevator `-1`, which tanh's `±1` can never reach). See the module docstring.

    Also converts the env's raw-space `expert_action` (info) to agent `[-1,1]` space on the
    way out — the inverse of this same map — so the collector receives a ready-to-store
    expert action that re-maps to the identical env action when flown.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base = env.action_space
        self._low = np.asarray(base.low, dtype=np.float32)
        self._high = np.asarray(base.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=base.shape, dtype=np.float32)

    def action(self, action):
        return map_tdmpc_action(action, self._low, self._high)

    def _convert_info(self, info):
        ea = info.get("expert_action") if info else None
        if ea is None:
            return info
        return {**info, "expert_action": invert_tdmpc_action(ea, self._low, self._high)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._convert_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        return obs, reward, terminated, truncated, self._convert_info(info)


class ActionRepeat(gym.Wrapper):
    """Hold each agent action for `repeat` sim ticks; sum reward, stop early on done.

    JSBSim steps at 120 Hz but BFM decisions live at ~20-30 Hz, so acting every tick
    both wastes planning compute and gives the value function an impossibly short
    effective horizon. Repeating `repeat≈5` exposes a ~24 Hz control env. The env's
    separate `terminated`/`truncated` flags propagate from whichever inner tick ended
    the episode (we break immediately), so the kill/deck vs bubble/time distinction
    the learner needs is preserved.
    """

    def __init__(self, env: gym.Env, repeat: int = 5):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs = info = None
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Concatenate the last `k` observations into one flat `(k·obs_dim,)` vector.

    Optional (default off): the 31-float obs is close to Markov, but stacking gives
    the MLP encoder a cheap window on opponent *intent* (closure/aspect trend) without
    recurrence. Flat concat — not a `(k, obs_dim)` stack — because the TD-MPC2 encoder
    is a plain MLP. On reset the first frame fills all `k` slots.
    """

    def __init__(self, env: gym.Env, k: int = 2):
        super().__init__(env)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self._frames: deque = deque(maxlen=k)
        space = env.observation_space
        low = np.concatenate([np.asarray(space.low, dtype=np.float32)] * k)
        high = np.concatenate([np.asarray(space.high, dtype=np.float32)] * k)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames)).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(np.asarray(obs, dtype=np.float32))
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(obs, dtype=np.float32))
        return self._stacked(), reward, terminated, truncated, info


def wrap_tdmpc(
    env: gym.Env,
    *,
    action_repeat: int = 5,
    frame_stack: int = 1,
    record_stats: bool = True,
) -> gym.Env:
    """Apply the TD-MPC2 wrapper stack to a raw `DogfightGymEnv`.

    linear `[-1,1]^A` action map → ActionRepeat → (optional) FrameStack →
    RecordEpisodeStatistics (outermost, so `info["episode"]` counts *control* steps).
    Shared by `make_tdmpc_env` (scripted opponents) and the self-play env builder.
    """
    env = _LinearActionWrapper(env)
    if action_repeat and action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    if frame_stack and frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


def make_tdmpc_env(
    tier: int = 1,
    *,
    action_repeat: int = 5,
    frame_stack: int = 1,
    record_stats: bool = True,
    **kwargs,
) -> gym.Env:
    """Build a dogfight env (scripted opponent, given `tier`) with the TD-MPC2 stack."""
    if tier not in _FACTORIES:
        raise ValueError(f"tier must be one of {sorted(_FACTORIES)}, got {tier}")
    return wrap_tdmpc(
        _FACTORIES[tier](**kwargs),
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        record_stats=record_stats,
    )


def tdmpc_env_thunk(tier: int = 1, **kwargs) -> Callable[[], gym.Env]:
    """No-arg factory for `AsyncVectorEnv` (the optional multi-process scale-out, §8).

    Same one-env-per-process / `context="spawn"` rules as `env_thunk` apply — JSBSim
    is process-global and not thread-safe.
    """

    def _thunk() -> gym.Env:
        return make_tdmpc_env(tier, **kwargs)

    return _thunk
