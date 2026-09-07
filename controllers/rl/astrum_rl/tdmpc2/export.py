"""Export a trained TD-MPC2 policy for deployment (design doc §9, P5).

TD-MPC2 acts by MPPI-planning through its world model, but the cheapest deployable
artifact is the **policy prior** alone: a plain `obs → action` MLP (encoder + π mean,
tanh-squashed to `[-1,1]`), no planning, no world model. That is what a real-time
flight controller wants — one forward pass per control step.

`export_policy` writes two files:
  <out>.pt    — a TorchScript module: `(N, obs_dim) float32 → (N, action_dim) in [-1,1]`
  <out>.json  — metadata: obs/action dims, the env's native action bounds (for the
                affine map back to `[ail,elev,rud,thr,trigger]`), frame_stack, repeat.

TorchScript loads in Python *and* in C++/Rust via libtorch, so the deployed controller
needn't import this training package. The full-planner path (better, slower) is still
available by loading the checkpoint into `TDMPC2` and calling `act(mpc=True)`.

    uv run python -m astrum_rl.tdmpc2.export --checkpoint run.pt --out policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .agent import TDMPC2, resolve_device
from .config import TDMPC2Config
from .layers import encoder, mlp
from .world_model import WorldModel

# The env's native action ranges (astrum_dogfight: 3 surfaces in [-1,1], throttle +
# trigger in [0,1]); consumers affine-map the prior's [-1,1] output back to these.
ACTION_LOW = [-1.0, -1.0, -1.0, 0.0, 0.0]
ACTION_HIGH = [1.0, 1.0, 1.0, 1.0, 1.0]


class PolicyPrior(nn.Module):
    """Deterministic `obs → action∈[-1,1]` inference head (encoder + π mean)."""

    def __init__(self, cfg: TDMPC2Config):
        super().__init__()
        self.action_dim = cfg.action_dim
        self.encoder = encoder(cfg)
        self.pi = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2 * cfg.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        mu = self.pi(z)[..., : self.action_dim]
        return torch.tanh(mu)  # deterministic squashed mean, in [-1,1]

    @classmethod
    def from_world_model(cls, model: WorldModel, cfg: TDMPC2Config) -> "PolicyPrior":
        prior = cls(cfg)
        prior.encoder.load_state_dict(model._encoder.state_dict())
        prior.pi.load_state_dict(model._pi.state_dict())
        prior.eval()
        return prior


def load_agent_from_checkpoint(path: str, device: torch.device | None = None):
    """Rebuild a `TDMPC2` (with the exact training architecture) from a checkpoint."""
    device = device or torch.device("cpu")
    sd = torch.load(path, map_location=device, weights_only=False)
    if "cfg" not in sd:
        raise KeyError(
            f"checkpoint {path} has no 'cfg' (saved by an older run); "
            "retrain or pass a matching TDMPC2Config explicitly."
        )
    cfg = TDMPC2Config(**sd["cfg"])
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    return agent, cfg


def export_policy(checkpoint: str, out: str, device: torch.device | None = None) -> tuple[str, str]:
    """Trace the policy prior from a checkpoint to `<out>.pt` + `<out>.json`."""
    device = device or torch.device("cpu")
    agent, cfg = load_agent_from_checkpoint(checkpoint, device)
    prior = PolicyPrior.from_world_model(agent.model, cfg).to(device)

    # torch.jit is deprecated in torch>=2.7 but remains the portable format libtorch
    # (C++/Rust) loads directly; torch.export lacks that broad deployment support yet.
    example = torch.zeros(1, cfg.obs_dim, device=device)
    with torch.no_grad():
        scripted = torch.jit.trace(prior, example)

    out_path = Path(out)
    pt_path = out_path.with_suffix(".pt")
    json_path = out_path.with_suffix(".json")
    scripted.save(str(pt_path))
    meta = {
        "obs_dim": cfg.obs_dim,
        "action_dim": cfg.action_dim,
        "action_low": ACTION_LOW[: cfg.action_dim],
        "action_high": ACTION_HIGH[: cfg.action_dim],
        "frame_stack": cfg.frame_stack,
        "action_repeat": cfg.action_repeat,
        "output_range": [-1.0, 1.0],
        "note": "affine-map output to [low,high]: native = low + 0.5*(a+1)*(high-low)",
    }
    json_path.write_text(json.dumps(meta, indent=2))
    return str(pt_path), str(json_path)


def load_prior(pt_path: str, device: torch.device | None = None):
    """Load an exported TorchScript prior for inference."""
    device = device or torch.device("cpu")
    return torch.jit.load(pt_path, map_location=device).eval()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a TD-MPC2 policy prior for deployment.")
    p.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    p.add_argument("--out", required=True, help="output basename (writes .pt and .json)")
    p.add_argument("--device", default="cpu", help="'cpu' | 'cuda' | 'auto'")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    pt, js = export_policy(args.checkpoint, args.out, resolve_device(args.device))
    print(f"exported policy prior → {pt}\n            metadata     → {js}")


if __name__ == "__main__":
    main()
