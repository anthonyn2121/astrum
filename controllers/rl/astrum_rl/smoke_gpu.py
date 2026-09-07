"""Smoke test: torch sees the GPU, and the dogfight env steps.

Run:  cd controllers/rl && uv run python -m astrum_rl.smoke_gpu
(or `make rl-smoke`, which builds the .so and runs this in the GPU container).

Checks, in order:
  1. torch imports and reports CUDA availability + the device.
  2. The RTX 5080 is Blackwell (sm_120); warn loudly if this torch build lacks
     sm_120 kernels (the classic "wrong CUDA wheel" trap — it imports fine but
     every kernel launch fails at runtime).
  3. A tiny matmul actually runs on the GPU.
  4. A Tier-1 dogfight episode steps with finite obs/reward.
"""

from __future__ import annotations

import numpy as np
import torch

from . import _paths
from .envs import make_env


def check_torch() -> torch.device:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  ! CUDA NOT available — training will fall back to CPU.")
        print("    Check: host NVIDIA driver, nvidia-container-toolkit, and that the")
        print("    container was started with --gpus all (compose: the `dogfight-train`")
        print("    service reserves the GPU).")
        return torch.device("cpu")

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    print(f"  CUDA {torch.version.cuda} | device: {name} (sm_{major}{minor})")

    arches = torch.cuda.get_arch_list()
    print(f"  built for: {', '.join(arches)}")
    want = f"sm_{major}{minor}"
    if major >= 12 and want not in arches and "sm_120" not in arches:
        print(
            f"  !! this torch build lacks {want} kernels — a Blackwell (RTX 50-series)\n"
            "     GPU needs a cu128 wheel (torch>=2.7). Kernels will fail at launch.\n"
            "     Fix the cu128 index in pyproject.toml, then `uv sync --reinstall`."
        )
    return torch.device("cuda")


def check_gpu_matmul(device: torch.device) -> None:
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = (a @ b).sum().item()
    print(f"  matmul on {device}: sum={c:.1f}  ✓")


def check_env(steps: int = 120) -> None:
    _paths.require_astrum_env()
    env = make_env(tier=1, max_time_s=30.0)
    obs, info = env.reset(seed=0)
    assert obs.shape[0] > 0 and np.all(np.isfinite(obs)), "reset obs must be finite"
    total = 0.0
    n = 0
    for _ in range(steps):
        # Raw Gaussian-style sample; the squash wrapper maps it into range.
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        total += reward
        n += 1
        if terminated or truncated:
            break
    print(f"  env: obs_dim={obs.shape[0]}, {n} steps, Σreward={total:.3f}  ✓")


def main() -> None:
    print("== astrum-rl smoke ==")
    device = check_torch()
    if device.type == "cuda":
        check_gpu_matmul(device)
    check_env()
    print("OK")


if __name__ == "__main__":
    main()
