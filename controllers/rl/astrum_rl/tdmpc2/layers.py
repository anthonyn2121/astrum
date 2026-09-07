"""Network building blocks for the world model.

`SimNorm` bounds the latent by softmax-over-groups (stabilizes the dynamics rollout);
`NormedLinear` is the LayerNorm+Mish unit the reference stacks; `mlp` assembles them
with an optional activation on the final layer; `Ensemble` runs the Q-heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Simplicial normalization: softmax within each `dim`-sized group of the latent.

    Bounds the latent to the product of simplices — keeps the learned dynamics from
    diverging over a multi-step rollout without collapsing capacity (design doc §2.1).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear → (Dropout) → LayerNorm → activation. The reference's core MLP unit."""

    def __init__(self, in_features: int, out_features: int, act=nn.Mish, dropout: float = 0.0):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = act() if isinstance(act, type) else act
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self) -> str:
        return (
            f"NormedLinear(in={self.in_features}, out={self.out_features}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(in_dim: int, hidden_dims, out_dim: int, act=None, dropout: float = 0.0) -> nn.Sequential:
    """Stack `NormedLinear` hidden layers; final layer is `NormedLinear(act=act)` if
    `act` is given (e.g. `SimNorm` for latents) else a plain `nn.Linear` (for heads).

    Dropout is applied to the first hidden layer only (the reference's convention)."""
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    dims = [in_dim, *hidden_dims, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    if act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def encoder(cfg) -> nn.Sequential:
    """Obs → normalized latent. Final `NormedLinear` uses `SimNorm` as its activation."""
    dims = [cfg.obs_dim] + [cfg.enc_dim] * (cfg.num_enc_layers - 1)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=cfg.dropout * (i == 0)))
    layers.append(NormedLinear(dims[-1], cfg.latent_dim, act=SimNorm(cfg.simnorm_dim)))
    return nn.Sequential(*layers)


class Ensemble(nn.Module):
    """A stack of identically-shaped MLPs; forward returns `(num_q, *batch, out)`.

    A plain loop over the members (num_q≈5) — clear and correct; `torch.vmap` would be
    the speed upgrade if the ensemble ever dominates step time.
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=0)
