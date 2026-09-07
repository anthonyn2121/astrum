"""End-to-end smoke test for TD-MPC2 against the real JSBSim dogfight env.

    cd controllers/rl && uv run python -m astrum_rl.tdmpc2.smoke
    (or `make rl-tdmpc2-smoke`, which builds the .so first)

Uses a deliberately tiny model + planner so it runs in seconds on CPU: build the Tier-1
env, seed the replay buffer with random actions, take a few gradient updates, then act
with the MPPI planner for a handful of steps. Asserts everything stays finite and every
emitted action is inside the `[-1,1]` box. It exercises the whole P2+P3 path (buffer →
update → plan) on genuine sim transitions — not the synthetic unit-test data.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import _paths
from ..envs import make_tdmpc_env
from .agent import TDMPC2
from .buffer import ReplayBuffer
from .config import TDMPC2Config


def _tiny_cfg(obs_dim: int, action_dim: int) -> TDMPC2Config:
    return TDMPC2Config(
        obs_dim=obs_dim,
        action_dim=action_dim,
        enc_dim=64,
        mlp_dim=64,
        latent_dim=64,
        simnorm_dim=8,
        num_q=2,
        horizon=3,
        batch_size=32,
        buffer_size=4096,
        num_samples=64,
        num_pi_trajs=6,
        num_elites=8,
        iterations=3,
        episode_length=288,
        device="cpu",
    )


def run(seed_steps: int = 60, updates: int = 20, plan_steps: int = 10) -> None:
    _paths.require_astrum_env()
    torch.manual_seed(0)
    np.random.seed(0)

    env = make_tdmpc_env(tier=1, action_repeat=5, max_time_s=30.0)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    print(f"env: obs {obs_dim}, action {action_dim}, box {env.action_space.low[:2]}..")

    cfg = _tiny_cfg(obs_dim, action_dim)
    agent = TDMPC2(cfg, torch.device("cpu"))
    buffer = ReplayBuffer(cfg, torch.device("cpu"))

    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "non-finite reset obs"
    buffer.add(obs, np.zeros(action_dim, np.float32), 0.0, 0.0, 0.0, first=True)

    # 1) Seed the buffer with random control.
    for _ in range(seed_steps):
        a = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(a)
        assert np.all(np.isfinite(nxt)) and np.isfinite(r)
        buffer.add(nxt, a, r, term, trunc, first=False)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
            buffer.add(obs, np.zeros(action_dim, np.float32), 0.0, 0.0, 0.0, first=True)
    print(f"seeded {len(buffer)} transitions")

    # 2) A few gradient updates.
    stats = {}
    for _ in range(updates):
        stats = agent.update(buffer)
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"
    print(
        "update ✓  "
        + " ".join(f"{k}={stats[k]:.3f}" for k in ("consistency_loss", "reward_loss", "value_loss"))
    )

    # 3) Plan (MPPI) for a few steps and check the emitted actions are valid.
    t0 = True
    for _ in range(plan_steps):
        a = agent.act(obs, t0=t0, eval_mode=False).numpy()
        t0 = False
        assert a.shape == (action_dim,)
        assert a.min() >= -1.0 - 1e-5 and a.max() <= 1.0 + 1e-5, a
        obs, r, term, trunc, _ = env.step(a)
        assert np.isfinite(r)
        if term or trunc:
            obs, _ = env.reset()
            t0 = True
    print(f"planned {plan_steps} steps ✓")

    env.close()
    print("TD-MPC2 smoke OK")


if __name__ == "__main__":
    run()
