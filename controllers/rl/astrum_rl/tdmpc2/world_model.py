"""`WorldModel` — the learned latent model TD-MPC2 plans through (design doc §2.1).

Heads: encoder (obs→z), dynamics (z,a→z'), reward (z,a→two-hot logits), policy prior
(z→squashed action), a Q-ensemble (z,a→two-hot logits) with a Polyak target copy, and
(episodic) a termination head (z→logit). All operate over arbitrary leading dims, so
they accept single steps `(B,·)` or horizon-major sequences `(T,B,·)` unchanged.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from . import functional as fn
from .layers import Ensemble, SimNorm, encoder, mlp


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = encoder(cfg)
        self._dynamics = mlp(
            cfg.latent_dim + cfg.action_dim,
            [cfg.mlp_dim, cfg.mlp_dim],
            cfg.latent_dim,
            act=SimNorm(cfg.simnorm_dim),
            dropout=cfg.dropout,
        )
        self._reward = mlp(
            cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1)
        )
        self._pi = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2 * cfg.action_dim)
        self._Qs = Ensemble(
            [
                mlp(
                    cfg.latent_dim + cfg.action_dim,
                    [cfg.mlp_dim, cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        if cfg.episodic:
            self._termination = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 1)
        else:
            self._termination = None

        # Zero ONLY the prediction heads, as the reference TD-MPC2 does.
        #
        # This used to be `self.apply(_zero_init_last)`, which walks EVERY submodule — and since
        # `NormedLinear` subclasses `nn.Linear`, the old `isinstance` check inside also matched the
        # encoder's and dynamics' final `NormedLinear(act=SimNorm)`. Zeroing those makes the encoder
        # a CONSTANT FUNCTION at init (measured: latent variance across distinct observations
        # exactly 0.0, zero gradient into the encoder's first layer, and a grad norm of ~25.8k —
        # 1300× the clip norm — on its final layer). Warm runs load a checkpoint over this so they
        # were never affected; every from-scratch run was, which confounds curriculum-80's
        # "cold training is disproven" conclusion.
        for head in (self._reward, self._Qs, self._termination):
            if head is not None:
                head.apply(_zero_init_last)
        # A frozen Polyak copy of the Q-ensemble for stable TD targets.
        self._target_Qs = copy.deepcopy(self._Qs).requires_grad_(False)

    # -- heads -------------------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self._encoder(obs)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._reward(torch.cat([z, a], dim=-1))

    def termination(self, z: torch.Tensor) -> torch.Tensor:
        """Termination *logit* for a latent (episodic only)."""
        assert self._termination is not None
        return self._termination(z)

    def pi(self, z: torch.Tensor, eps: torch.Tensor | None = None):
        """Policy prior: returns `(mean, sample, log_prob, log_std)`, all tanh-squashed.

        `eps` overrides the i.i.d. N(0,1) draw with a caller-supplied one of unit marginal
        variance, letting collection substitute a temporally CORRELATED sequence. It is
        applied **pre-squash**, which is the only place it can do any work: D2a measured
        `dev_post ~ sigma*(1-mu^2)`, so a post-squash perturbation of a channel whose mean
        sits at the tanh rail is a no-op, and the roll axis is railed 0.45-0.73 of the
        opening seconds in every checkpoint measured. `None` keeps the reference behaviour
        bit-for-bit, so the update path and every eval are unaffected.
        """
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = fn.clamp_log_std(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        eps = torch.randn_like(mu) if eps is None else eps.to(mu.dtype).expand_as(mu)
        log_pi = fn.gaussian_logprob(eps, log_std)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = fn.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def Q(self, z: torch.Tensor, a: torch.Tensor, return_type: str = "min", target: bool = False):
        """Q head(s). `return_type`: 'all' → two-hot logits `(num_q,*,B)`; 'min' →
        scalar min over 2 random members; 'avg' → scalar mean over the ensemble."""
        assert return_type in ("all", "min", "avg")
        x = torch.cat([z, a], dim=-1)
        net = self._target_Qs if target else self._Qs
        logits = net(x)  # (num_q, *batch, num_bins)
        if return_type == "all":
            return logits
        q = fn.two_hot_inv(logits, self.cfg)  # (num_q, *batch)
        if return_type == "avg":
            return q.mean(0)
        idx = torch.randperm(self.cfg.num_q, device=q.device)[:2]
        return q[idx].min(0).values

    @torch.no_grad()
    def soft_update_target_Qs(self) -> None:
        tau = self.cfg.tau
        for p, tp in zip(self._Qs.parameters(), self._target_Qs.parameters()):
            tp.data.lerp_(p.data, tau)

    # -- parameter groups (encoder gets a scaled LR; pi optimized separately) --
    def model_param_groups(self) -> list[dict]:
        enc = list(self._encoder.parameters())
        rest = (
            list(self._dynamics.parameters())
            + list(self._reward.parameters())
            + list(self._Qs.parameters())
        )
        if self._termination is not None:
            rest += list(self._termination.parameters())
        return [
            {"params": enc, "lr": self.cfg.lr * self.cfg.enc_lr_scale},
            {"params": rest, "lr": self.cfg.lr},
        ]

    def pi_parameters(self):
        return self._pi.parameters()


def _zero_init_last(module: nn.Module) -> None:
    """Zero the final Linear of each head so reward/Q/termination start unbiased.

    The type check is EXACT on purpose: `mlp()` ends a head in a plain `nn.Linear` but ends a
    latent-producing stack in `NormedLinear(act=…)`, and `NormedLinear` subclasses `nn.Linear`,
    so `isinstance` cannot tell the two apart."""
    if isinstance(module, nn.Sequential) and len(module) and type(module[-1]) is nn.Linear:
        nn.init.zeros_(module[-1].weight)
        if module[-1].bias is not None:
            nn.init.zeros_(module[-1].bias)
