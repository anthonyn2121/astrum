"""Stateless math for TD-MPC2: symlog two-hot regression + Gaussian-policy helpers.

These are the pieces that make TD-MPC2 robust to reward scale (design doc §1): value
and reward are regressed as soft two-hot distributions over a symlog-compressed axis,
so the −200…+100 dogfight reward needs no rescaling. `two_hot`/`two_hot_inv` are
inverses in expectation — a required unit test (§6).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Signed log compression: `sign(x)·log(1+|x|)`. Tames large magnitudes."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `symlog`: `sign(x)·(exp(|x|)-1)`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(x: torch.Tensor, cfg) -> torch.Tensor:
    """Scalar(s) → soft two-hot distribution over `num_bins` symlog-spaced bins.

    Places mass on the two bins straddling `symlog(x)`, linearly by distance. Returns
    the input shape with a trailing `num_bins` axis. `num_bins<=1` degenerates to the
    (sym)log scalar itself (no categorical head).
    """
    if cfg.num_bins == 1:
        return symlog(x)
    if cfg.num_bins <= 0:
        return x
    shape = x.shape
    x = x.reshape(-1)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)
    idx_f = (x - cfg.vmin) / cfg.bin_size
    lower = torch.floor(idx_f)
    offset = (idx_f - lower).unsqueeze(-1)
    lower = lower.long()
    lo = lower.clamp(0, cfg.num_bins - 1).unsqueeze(-1)
    hi = (lower + 1).clamp(0, cfg.num_bins - 1).unsqueeze(-1)
    out = torch.zeros(x.size(0), cfg.num_bins, device=x.device, dtype=x.dtype)
    out.scatter_add_(1, lo, 1.0 - offset)
    out.scatter_add_(1, hi, offset)
    return out.reshape(*shape, cfg.num_bins)


def two_hot_inv(logits: torch.Tensor, cfg) -> torch.Tensor:
    """Two-hot *logits* → scalar expectation (softmax · bin-centers, then `symexp`).

    Inverts `two_hot` in expectation: pass logits (the model's raw head output). For a
    probability distribution `p`, `two_hot_inv(log(p))` recovers the encoded scalar.
    """
    if cfg.num_bins == 1:
        return symexp(logits)
    if cfg.num_bins <= 0:
        return logits
    bins = torch.linspace(
        cfg.vmin, cfg.vmax, cfg.num_bins, device=logits.device, dtype=logits.dtype
    )
    probs = torch.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(-1))


def soft_ce(logits: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    """Cross-entropy of predicted two-hot `logits` vs the two-hot of scalar `target`.

    Returns per-element loss (reduces only the `num_bins` axis)."""
    log_probs = F.log_softmax(logits, dim=-1)
    target_dist = two_hot(target, cfg)
    return -(target_dist * log_probs).sum(-1)


def clamp_log_std(log_std: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Squash a raw log-std into `[low, high]` smoothly (tanh, not hard clamp)."""
    return low + 0.5 * (high - low) * (torch.tanh(log_std) + 1.0)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Log-prob of a diagonal-Gaussian sample `x = mu + eps·exp(log_std)` (summed)."""
    residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * eps.shape[-1] * math.log(2 * math.pi)


def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
    """Tanh-squash a Gaussian action into `[-1,1]`, correcting the log-prob (SAC)."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class RunningScale:
    """EMA of the value spread (95th−5th percentile), to normalize the policy loss.

    Keeps the policy-improvement gradient scale-invariant as the two-hot value head's
    magnitude drifts during training (TD-MPC2's stabilizer for `update_pi`).
    """

    def __init__(self, tau: float, device: torch.device):
        self.tau = tau
        self._value = torch.ones(1, dtype=torch.float32, device=device)

    @property
    def value(self) -> float:
        return self._value.item()

    def update(self, x: torch.Tensor) -> None:
        q = torch.quantile(
            x.detach().float().reshape(-1),
            torch.tensor([0.05, 0.95], device=x.device),
        )
        spread = (q[1] - q[0]).clamp_min(1.0)
        self._value.lerp_(spread, self.tau)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._value
