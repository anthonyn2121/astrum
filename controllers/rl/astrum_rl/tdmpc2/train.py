"""Single-env off-policy training loop for TD-MPC2 (design doc §2.3, §7 P2/P4).

    cd controllers/rl
    uv run python -m astrum_rl.tdmpc2.train --tier 1 --steps 1_000_000

One JSBSim env (it is process-global + not thread-safe, so no in-process vectorization
— see envs.py), one gradient update per environment step after a seeding phase, and an
episodic replay buffer that keeps `terminated` (kill/death/deck) distinct from
`truncated` (bubble/time) so the value bootstrap stays correct (§6).

Curriculum (P4): `--curriculum "1:0,2:200000,3:500000"` swaps the scripted opponent
tier at the given global steps; the world model and replay buffer carry across swaps
(the obs/action spaces are identical), so learning is continuous.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from .. import _paths
from .agent import TDMPC2, resolve_device
from .buffer import ReplayBuffer
from .config import TDMPC2Config


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="From-scratch TD-MPC2 for the ASTRUM dogfight.")
    p.add_argument("--tier", type=int, default=1, choices=(1, 2, 3), help="scripted-opponent tier")
    p.add_argument(
        "--curriculum",
        default="",
        help='tier schedule "tier:step,...", e.g. "1:0,2:200000,3:500000" (overrides --tier)',
    )
    p.add_argument("--steps", type=int, default=1_000_000, help="total env (control) steps")
    p.add_argument("--seed-steps", type=int, default=0, help="random warmup; 0 = derive from cfg")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0, help="episode time limit (sim s)")
    p.add_argument("--rate-hz", type=float, default=120.0, help="sim rate")
    p.add_argument("--no-mpc", action="store_true", help="act with the policy prior (no MPPI)")
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--compile", action="store_true", help="torch.compile the update (P4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--log-every", type=int, default=1000, help="steps between summary lines")
    p.add_argument("--save", default="", help="checkpoint path (saved at the end)")
    p.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (0=off)")
    p.add_argument(
        "--explore-noise",
        default="white",
        choices=("white", "pink", "ou"),
        help="temporal spectrum of the per-control-step exploration noise. 'white' (default) is "
        "the TD-MPC2 reference i.i.d. draw, whose exploration sequence has ~zero autocorrelation, "
        "so a 1.5 s coordinated deviation (a displacement roll) is a ~15-sigma event. 'pink'/'ou' "
        "keep the same marginal variance but put the power at the maneuver timescale.",
    )
    p.add_argument(
        "--explore-beta", type=float, default=1.0, help="pink spectral exponent (1=pink, 2=brown)"
    )
    p.add_argument(
        "--explore-tau-steps",
        type=float,
        default=36.0,
        help="OU correlation time in CONTROL steps (36 ≈ 1.5 s at 24 Hz)",
    )
    return p.parse_args(argv)


def parse_curriculum(spec: str, default_tier: int):
    """Parse "tier:step,..." into a sorted list of `(step, tier)` breakpoints."""
    if not spec:
        return [(0, default_tier)]
    out = []
    for chunk in spec.split(","):
        tier_s, step_s = chunk.split(":")
        out.append((int(step_s), int(tier_s)))
    out.sort()
    if out[0][0] != 0:
        raise ValueError("curriculum must define a tier at step 0")
    return out


def tier_at(schedule, step: int) -> int:
    tier = schedule[0][1]
    for brk, t in schedule:
        if step >= brk:
            tier = t
    return tier


def _build_env(tier: int, args):
    from ..envs import make_tdmpc_env

    return make_tdmpc_env(
        tier=tier,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        max_time_s=args.max_time_s,
        rate_hz=args.rate_hz,
    )


def build_config(args, obs_dim: int, action_dim: int) -> TDMPC2Config:
    # Max control steps per episode, from the env's time limit and control rate.
    episode_length = max(1, math.ceil(args.max_time_s * args.rate_hz / max(1, args.action_repeat)))
    return TDMPC2Config(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        horizon=args.horizon,
        mpc=not args.no_mpc,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        steps=args.steps,
        seed_steps=args.seed_steps or None,
        episode_length=episode_length,
        compile=args.compile,
        device=args.device,
        seed=args.seed,
        # Only the parallel trainer exposes --discount; 0 keeps the episode-length
        # derivation (`resolved_discount()`), which is what every run to date used.
        discount=(getattr(args, "discount", 0.0) or None),
        explore_noise=getattr(args, "explore_noise", "white"),
        explore_beta=getattr(args, "explore_beta", 1.0),
        explore_tau_steps=getattr(args, "explore_tau_steps", 36.0),
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    _paths.require_astrum_env()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    schedule = parse_curriculum(args.curriculum, args.tier)
    tier = schedule[0][1]
    env = _build_env(tier, args)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    cfg = build_config(args, obs_dim, action_dim)
    agent = TDMPC2(cfg, device)
    if cfg.compile:
        agent.update = torch.compile(agent.update)  # type: ignore[method-assign]
    buffer = ReplayBuffer(cfg, device)
    seed_steps = cfg.resolved_seed_steps()

    print(
        f"device {device} | tier {tier} | obs {obs_dim} act {action_dim} | "
        f"mpc {cfg.mpc} | discount {agent.discount:.4f} | seed_steps {seed_steps} | "
        f"episode_length {cfg.episode_length}"
    )

    obs, _ = env.reset(seed=args.seed)
    buffer.add(obs, np.zeros(action_dim, np.float32), 0.0, 0.0, 0.0, first=True)
    ep_return, ep_len, ep_count = 0.0, 0, 0
    recent_returns: list[float] = []
    t0 = True
    last_stats: dict = {}
    t_start = time.monotonic()

    for step in range(1, cfg.steps + 1):
        # Curriculum: swap the scripted opponent tier at a breakpoint (buffer/model persist).
        want_tier = tier_at(schedule, step)
        if want_tier != tier:
            tier = want_tier
            env.close()
            env = _build_env(tier, args)
            obs, _ = env.reset()
            buffer.add(obs, np.zeros(action_dim, np.float32), 0.0, 0.0, 0.0, first=True)
            ep_return, ep_len, t0 = 0.0, 0, True
            print(f"  ↳ step {step:,}: curriculum → tier {tier}")

        if step <= seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, t0=t0, eval_mode=False).numpy()
        t0 = False

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(next_obs, action, reward, terminated, truncated, first=False)
        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        # One update per step after seeding; a pretraining burst at the crossover.
        if step == seed_steps:
            print(f"  pretraining on {seed_steps} seed transitions...")
            for _ in range(seed_steps):
                last_stats = agent.update(buffer)
        elif step > seed_steps:
            last_stats = agent.update(buffer)

        if terminated or truncated:
            ep_count += 1
            recent_returns.append(ep_return)
            obs, _ = env.reset()
            buffer.add(obs, np.zeros(action_dim, np.float32), 0.0, 0.0, 0.0, first=True)
            ep_return, ep_len, t0 = 0.0, 0, True

        if step % args.log_every == 0:
            sps = step / (time.monotonic() - t_start)
            ret = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
            msg = (
                f"step {step:>9,d} | eps {ep_count:4d} | ret(20) {ret:8.2f} | "
                f"buf {len(buffer):>7,d} | {sps:6.1f} sps"
            )
            if last_stats:
                msg += (
                    f" | cons {last_stats['consistency_loss']:.3f} "
                    f"rew {last_stats['reward_loss']:.3f} val {last_stats['value_loss']:.3f} "
                    f"pi {last_stats['pi_loss']:+.3f}"
                )
            print(msg)

        if args.save and args.save_every and step % args.save_every == 0:
            torch.save(agent.state_dict(), args.save)

    if args.save:
        torch.save(agent.state_dict(), args.save)
        print(f"saved checkpoint → {args.save}")
    env.close()


if __name__ == "__main__":
    main()
