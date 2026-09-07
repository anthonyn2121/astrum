"""TD-MPC2 env wrappers (design doc §3.2/§6): affine action map (NOT squash),
action-repeat reward summing, and flat frame-stacking. Uses a stub env so these run
without the astrum_env .so."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from astrum_rl.envs import (  # noqa: E402
    ActionRepeat,
    FrameStack,
    _LinearActionWrapper,
    invert_tdmpc_action,
    map_tdmpc_action,
)


class _StubEnv(gym.Env):
    """Mixed-range action env mirroring the dogfight: [-1,1]^3 surfaces + [0,1]^2."""

    def __init__(self, obs_dim=4, done_after=None):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(
            np.array([-1, -1, -1, 0, 0], np.float32),
            np.array([1, 1, 1, 1, 1], np.float32),
            (5,),
            np.float32,
        )
        self.obs_dim = obs_dim
        self.done_after = done_after
        self.t = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.full(self.obs_dim, self.t, np.float32), {}

    def step(self, action):
        self.t += 1
        self.last_action = np.asarray(action, np.float32)
        terminated = self.done_after is not None and self.t >= self.done_after
        obs = np.full(self.obs_dim, self.t, np.float32)
        return obs, 1.0, terminated, False, {}


LOW = np.array([-1, -1, -1, 0, 0], np.float32)
HIGH = np.array([1, 1, 1, 1, 1], np.float32)


def test_affine_map_endpoints_and_midpoint():
    # a=-1 → low, a=+1 → high, a=0 → midpoint.
    assert np.allclose(map_tdmpc_action(-np.ones(5), LOW, HIGH), LOW)
    assert np.allclose(map_tdmpc_action(np.ones(5), LOW, HIGH), HIGH)
    assert np.allclose(map_tdmpc_action(np.zeros(5), LOW, HIGH), [0, 0, 0, 0.5, 0.5])


def test_full_scale_elevator_is_reachable():
    """The whole point of affine-over-squash: a=-1 must hit native -1 (full 9 g pull),
    which tanh could never reach (tanh(1)=0.76)."""
    native = map_tdmpc_action([0, -1, 0, 0, 0], LOW, HIGH)
    assert native[1] == pytest.approx(-1.0)


def test_trigger_threshold_at_zero():
    # Continuous trigger, fires (native ≥ 0.5, i.e. > 0) exactly when a_trigger ≥ 0.
    assert map_tdmpc_action([0, 0, 0, 0, 0.0], LOW, HIGH)[4] == pytest.approx(0.5)
    assert map_tdmpc_action([0, 0, 0, 0, -1e-3], LOW, HIGH)[4] < 0.5
    assert map_tdmpc_action([0, 0, 0, 0, 1e-3], LOW, HIGH)[4] > 0.5


def test_map_clips_out_of_box():
    out = map_tdmpc_action([-2, 2, 0, 5, -5], LOW, HIGH)
    assert np.allclose(out, [-1, 1, 0, 1, 0])


def test_linear_wrapper_space_and_mapping():
    env = _LinearActionWrapper(_StubEnv())
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    env.reset()
    env.step(np.array([0, -1, 0, 1, 1], np.float32))
    assert np.allclose(env.env.last_action, [0, -1, 0, 1, 1])


def test_action_repeat_sums_reward_and_holds_action():
    env = ActionRepeat(_StubEnv(), repeat=4)
    env.reset()
    obs, reward, term, trunc, _ = env.step(np.zeros(5, np.float32))
    assert reward == 4.0  # 1.0 per inner tick × 4
    assert obs[0] == 4  # advanced 4 sim ticks


def test_action_repeat_breaks_on_done():
    env = ActionRepeat(_StubEnv(done_after=2), repeat=5)
    env.reset()
    obs, reward, term, trunc, _ = env.step(np.zeros(5, np.float32))
    assert term and reward == 2.0 and obs[0] == 2  # stopped early at the terminal tick


def test_frame_stack_concats_and_fills():
    env = FrameStack(_StubEnv(obs_dim=4), k=3)
    assert env.observation_space.shape == (12,)
    obs, _ = env.reset()
    assert obs.shape == (12,) and np.all(obs == 0)  # reset fills all k slots
    obs, *_ = env.step(np.zeros(5, np.float32))
    # newest frame (t=1) is last; oldest (t=0) first.
    assert np.all(obs[-4:] == 1) and np.all(obs[:4] == 0)


def test_repeat_must_be_positive():
    with pytest.raises(ValueError):
        ActionRepeat(_StubEnv(), repeat=0)


LOW = np.array([-1, -1, -1, 0, 0], np.float32)
HIGH = np.array([1, 1, 1, 1, 1], np.float32)


def test_invert_tdmpc_action_roundtrips_the_forward_map():
    # invert ∘ map == identity on the agent box, and map ∘ invert == identity on native.
    for native in ([-1, 1, 0, 1, 0], [0.3, -0.7, 0.1, 0.6, 1.0], [1, -1, -1, 0, 0]):
        a = invert_tdmpc_action(native, LOW, HIGH)
        assert np.all(a >= -1.0) and np.all(a <= 1.0)
        assert np.allclose(map_tdmpc_action(a, LOW, HIGH), native, atol=1e-6)
    # throttle/trigger: native 0 → agent −1, native 0.5 → agent 0, native 1 → agent +1.
    assert np.allclose(invert_tdmpc_action([0, 0, 0, 0.5, 0.0], LOW, HIGH), [0, 0, 0, 0, -1])


def test_linear_wrapper_converts_expert_action_and_roundtrips():
    # The env surfaces a RAW-space expert action in info; the wrapper must convert it to the
    # agent's [-1,1] space AND flying that converted action back must reproduce the raw one.
    native = np.array([0.5, -0.5, 0.0, 0.25, 1.0], np.float32)

    class _ExpertStub(_StubEnv):
        def reset(self, *, seed=None, options=None):
            obs, _ = super().reset(seed=seed, options=options)
            return obs, {"expert_action": native.copy()}

        def step(self, action):
            obs, r, term, trunc, _ = super().step(action)
            return obs, r, term, trunc, {"expert_action": native.copy()}

    env = _LinearActionWrapper(_ExpertStub())
    _, info = env.reset()
    ea = info["expert_action"]  # now agent-space
    assert np.all(ea >= -1.0) and np.all(ea <= 1.0)
    env.step(ea)  # the wrapper re-maps it → must equal the raw expert action the env emitted
    assert np.allclose(env.env.last_action, native, atol=1e-6)
