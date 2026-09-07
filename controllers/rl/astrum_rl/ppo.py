"""A from-scratch PPO (no stable-baselines3) — network + GAE + clipped update.

Deliberately small and readable: an MLP actor-critic with a diagonal Gaussian
policy, generalized advantage estimation, and the clipped surrogate objective.
This is the *starting point* to build on, not a tuned agent.

Scope note — single env by design (for now). JSBSim is process-global and not
thread-safe, so the mandated way to parallelize is `gymnasium.vector.AsyncVectorEnv`
(one env per process; see `envs.env_thunk`). Vectorizing this loop is the next
step; when you do, mind gymnasium ≥1.0's NEXT_STEP autoreset (the step *after* a
done returns the reset obs with the action ignored) — the GAE bookkeeping here
assumes the manual-reset flow in `collect_rollout`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """MLP actor-critic with a state-independent diagonal-Gaussian policy."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            # Small final gain → near-zero initial mean → mild, well-behaved actions.
            _layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Return (action, log_prob, entropy, value). Samples if `action` is None."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(obs).squeeze(-1)


@dataclass
class PPOConfig:
    rollout_len: int = 2048
    update_epochs: int = 10
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    norm_adv: bool = True
    clip_vloss: bool = True


class Rollout:
    """Flat per-step buffers for one env; `.compute_gae` fills advantages/returns."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, act_dim, device=device)
        self.logprobs = torch.zeros(capacity, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.values = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)  # 1.0 if episode ended AT step t
        self.advantages = torch.zeros(capacity, device=device)
        self.returns = torch.zeros(capacity, device=device)
        self.capacity = capacity

    def compute_gae(self, last_value: torch.Tensor, cfg: PPOConfig) -> None:
        lastgaelam = 0.0
        for t in reversed(range(self.capacity)):
            mask = 1.0 - self.dones[t]
            next_value = last_value if t == self.capacity - 1 else self.values[t + 1]
            delta = self.rewards[t] + cfg.gamma * next_value * mask - self.values[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * mask * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values


def collect_rollout(
    env, agent: ActorCritic, rollout: Rollout, obs: np.ndarray, cfg: PPOConfig, device: torch.device
):
    """Fill `rollout` with `capacity` steps from `env`, resetting on episode end.

    Truncation (time/bubble limit) bootstraps the cut-off future into that step's
    reward, so the done-masked GAE treats termination and truncation correctly.
    Returns `(obs, episode_returns)` — the obs to start the next rollout from and
    a list of finished-episode returns seen this rollout (for logging).
    """
    episode_returns: list[float] = []
    for t in range(rollout.capacity):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_t)
        rollout.obs[t] = obs_t.squeeze(0)
        rollout.actions[t] = action.squeeze(0)
        rollout.logprobs[t] = logprob.squeeze(0)
        rollout.values[t] = value.squeeze(0)

        act_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(act_np)
        done = bool(terminated or truncated)
        if truncated and not terminated:
            with torch.no_grad():
                boot = agent.get_value(
                    torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                ).item()
            reward = float(reward) + cfg.gamma * boot

        rollout.rewards[t] = float(reward)
        rollout.dones[t] = 1.0 if done else 0.0

        if "episode" in info:  # RecordEpisodeStatistics posts this on episode end.
            episode_returns.append(float(info["episode"]["r"]))
        if done:
            next_obs, _ = env.reset()
        obs = next_obs

    with torch.no_grad():
        last_value = agent.get_value(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        ).squeeze(0)
    rollout.compute_gae(last_value, cfg)
    return obs, episode_returns


def update(agent: ActorCritic, optimizer, rollout: Rollout, cfg: PPOConfig) -> dict[str, float]:
    """One PPO update over the collected rollout; returns loss stats for logging."""
    n = rollout.capacity
    minibatch = max(1, n // cfg.num_minibatches)
    b_obs = rollout.obs
    b_actions = rollout.actions
    b_logprobs = rollout.logprobs
    b_advantages = rollout.advantages
    b_returns = rollout.returns
    b_values = rollout.values

    idx = np.arange(n)
    approx_kl = pg_loss = v_loss = ent_loss = 0.0
    for _ in range(cfg.update_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, minibatch):
            mb = idx[start : start + minibatch]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            logratio = newlogprob - b_logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean().item()

            mb_adv = b_advantages[mb]
            if cfg.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss_t = torch.max(pg1, pg2).mean()

            if cfg.clip_vloss:
                v_unclipped = (newvalue - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(
                    newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef
                )
                v_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss_t = 0.5 * torch.max(v_unclipped, v_clipped).mean()
            else:
                v_loss_t = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

            ent_loss_t = entropy.mean()
            loss = pg_loss_t - cfg.ent_coef * ent_loss_t + cfg.vf_coef * v_loss_t

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_loss, v_loss, ent_loss = pg_loss_t.item(), v_loss_t.item(), ent_loss_t.item()

    return {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": ent_loss, "approx_kl": approx_kl}
