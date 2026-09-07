"""P0 throughput benchmark — how fast does ONE process step the dogfight env?

TD-MPC2 collects from a single env (JSBSim is process-global; design doc §8), so a
run's wall-clock is bounded by env-stepping throughput, not the GPU. This measures
that floor with a random policy (no torch) so it isolates the sim from the agent,
and prints a wall-clock estimate for a target step budget.

    cd controllers/rl
    uv run python -m astrum_rl.benchmark --tier 1 --action-repeat 5 --control-steps 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import _paths
from .envs import make_tdmpc_env


def run(
    tier: int = 1,
    *,
    action_repeat: int = 5,
    control_steps: int = 2000,
    seed: int = 0,
    max_time_s: float = 60.0,
) -> dict[str, float]:
    """Step a random policy for `control_steps` control steps; report throughput."""
    _paths.require_astrum_env()
    rng = np.random.default_rng(seed)
    env = make_tdmpc_env(tier, action_repeat=action_repeat, max_time_s=max_time_s)
    adim = env.action_space.shape[0]

    obs, _ = env.reset(seed=seed)
    episodes = 0
    ep_returns: list[float] = []
    t0 = time.perf_counter()
    for _ in range(control_steps):
        action = rng.uniform(-1.0, 1.0, size=adim).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            episodes += 1
            if "episode" in info:
                ep_returns.append(float(info["episode"]["r"]))
            obs, _ = env.reset()
    dt = time.perf_counter() - t0

    control_per_s = control_steps / dt
    sim_per_s = control_per_s * action_repeat
    mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
    print(
        f"tier {tier} | action_repeat {action_repeat} | {control_steps} control steps "
        f"in {dt:.2f}s\n"
        f"  {control_per_s:8.1f} control steps/s   ({sim_per_s:,.0f} sim ticks/s)\n"
        f"  {episodes} episodes, mean return {mean_ret:.2f}"
    )
    for budget in (100_000, 1_000_000):
        hrs = budget / control_per_s / 3600.0
        print(f"  → {budget:>9,d} control steps ≈ {hrs:6.2f} h wall-clock (this process)")
    return {"control_per_s": control_per_s, "sim_per_s": sim_per_s, "episodes": episodes}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-policy throughput benchmark (P0).")
    p.add_argument("--tier", type=int, default=1, choices=(1, 2, 3))
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--control-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.tier,
        action_repeat=args.action_repeat,
        control_steps=args.control_steps,
        seed=args.seed,
        max_time_s=args.max_time_s,
    )


if __name__ == "__main__":
    main()
