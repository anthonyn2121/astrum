"""Train the dogfight learner with the from-scratch PPO in `ppo.py`.

    cd controllers/rl
    uv run python -m astrum_rl.train --tier 1 --total-steps 1_000_000

Single env for now (correct + simple); see `ppo.py` for the AsyncVectorEnv scaling
note. Prints a running summary each rollout: env steps, mean finished-episode
return, and the PPO loss stats.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from . import _paths
from .envs import make_env
from .ppo import ActorCritic, PPOConfig, Rollout, collect_rollout, update


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="From-scratch PPO for the ASTRUM dogfight.")
    p.add_argument("--tier", type=int, default=1, choices=(1, 2, 3), help="opponent tier")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="total env steps")
    p.add_argument("--rollout-len", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-time-s", type=float, default=60.0, help="episode time limit")
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--save", default="", help="path to save the trained policy (optional)")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device} | tier {args.tier} | rollout {args.rollout_len}")

    env = make_env(tier=args.tier, max_time_s=args.max_time_s)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    cfg = PPOConfig(
        rollout_len=args.rollout_len,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
    )
    rollout = Rollout(cfg.rollout_len, obs_dim, act_dim, device)

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    iteration = 0
    while global_step < args.total_steps:
        obs, ep_returns = collect_rollout(env, agent, rollout, obs, cfg, device)
        stats = update(agent, optimizer, rollout, cfg)
        global_step += cfg.rollout_len
        iteration += 1

        ep_mean = float(np.mean(ep_returns)) if ep_returns else float("nan")
        print(
            f"iter {iteration:4d} | step {global_step:>9,d} | "
            f"eps {len(ep_returns):3d} | ep_return {ep_mean:8.2f} | "
            f"pg {stats['pg_loss']:+.3f} v {stats['v_loss']:.3f} "
            f"ent {stats['entropy']:.3f} kl {stats['approx_kl']:.4f}"
        )

    if args.save:
        torch.save(agent.state_dict(), args.save)
        print(f"saved policy → {args.save}")


if __name__ == "__main__":
    main()
