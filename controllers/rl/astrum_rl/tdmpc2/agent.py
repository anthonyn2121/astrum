"""`TDMPC2` — the learner: acting (MPPI plan or policy prior), the joint model update,
the policy-prior improvement step, and the TD target (design doc §2.2/§2.3, §6).

Two optimizers, per the reference: one over the world-model params (encoder at a
scaled LR) and a separate one over the policy prior — the prior is trained only to
maximize the learned Q, decoupled from the model's regression losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import functional as fn
from .config import TDMPC2Config
from .world_model import WorldModel


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ======================================================================
# Exploration noise for the final per-control-step action draw in `plan`.
# ----------------------------------------------------------------------
# The reference TD-MPC2 draw is i.i.d. N(0,1) per control step, so the exploration
# SEQUENCE has ~zero autocorrelation (measured: |rho(k)| < 0.04 for every lag on
# every channel) and the amplitude that survives a T-step window falls as
# 1/sqrt(T). At 24 Hz that makes a 1.5 s (36-step) coordinated deviation — the
# timescale of a displacement roll / out-of-plane reposition — a ~15-sigma event
# (P ~ 1e-50 per window at the measured sigma_ail = 0.21), i.e. UNREACHABLE. White
# noise can only dither around whatever the prior already does; it cannot propose a
# multi-second maneuver. Colored noise keeps the marginal variance (same clamp
# behaviour, same "how far off-policy is one step") but moves the power to the
# maneuver timescale, raising the same event to O(1) per episode.
#
# Prior art (from memory, verify before quoting): pink action noise as an off-policy
# default — Eberhard, Hollenstein, Pinneri, Martius, "Pink Noise Is All You Need:
# Colored Noise Exploration in Deep RL", ICLR 2023; OU action noise — Lillicrap et
# al., DDPG, ICLR 2016; colored noise in CEM/MPC action samples — Pinneri et al.,
# "Sample-efficient Cross-Entropy Method for Real-time Planning", CoRL 2020.
# ======================================================================
class WhiteNoise:
    """i.i.d. N(0,1) — bit-identical to the reference draw (`torch.randn(A)`)."""

    def __init__(self, action_dim: int, device: torch.device):
        self.a = int(action_dim)
        self.device = device

    def reset(self) -> None:
        pass

    def sample(self) -> torch.Tensor:
        return torch.randn(self.a, device=self.device)


class OUNoise:
    """AR(1) / discrete Ornstein-Uhlenbeck with UNIT marginal variance.

    `x <- rho*x + sqrt(1-rho^2)*w`, `rho = exp(-1/tau_steps)`. Marginal N(0,1) exactly,
    so `a_std` keeps its meaning; only the temporal spectrum changes.
    """

    def __init__(self, action_dim: int, device: torch.device, tau_steps: float):
        self.a = int(action_dim)
        self.device = device
        tau = max(1e-3, float(tau_steps))
        self.rho = float(torch.exp(torch.tensor(-1.0 / tau)))
        self.s = float((1.0 - self.rho**2) ** 0.5)
        self.x: torch.Tensor | None = None

    def reset(self) -> None:
        self.x = None

    def sample(self) -> torch.Tensor:
        w = torch.randn(self.a, device=self.device)
        self.x = w if self.x is None else self.rho * self.x + self.s * w
        return self.x


class PinkNoise:
    """Per-episode 1/f^(beta/2)-amplitude sequence, renormalized to unit variance.

    Timmer & Koenig (1995) construction: shape the rFFT of white noise by
    `f^(-beta/2)` (DC dropped so exploration carries no constant bias), inverse-FFT,
    then divide by the realized per-channel std so the marginal variance matches the
    white arm. `beta=1` = pink (1/f power), `beta=2` = brown/red.
    """

    def __init__(self, action_dim: int, device: torch.device, length: int, beta: float = 1.0):
        self.a = int(action_dim)
        self.device = device
        self.n = max(64, int(length))
        self.beta = float(beta)
        self.buf: torch.Tensor | None = None
        self.i = 0

    def _build(self) -> torch.Tensor:
        nfft = 1 << (2 * self.n - 1).bit_length()
        f = torch.fft.rfftfreq(nfft, d=1.0, device=self.device)
        scale = torch.zeros_like(f)
        scale[1:] = f[1:] ** (-self.beta / 2.0)  # scale[0] = 0 ⇒ zero-mean noise
        re = torch.randn(self.a, f.numel(), device=self.device) * scale
        im = torch.randn(self.a, f.numel(), device=self.device) * scale
        x = torch.fft.irfft(torch.complex(re, im), n=nfft, dim=1)[:, : self.n]
        return x / x.std(dim=1, keepdim=True).clamp_min(1e-8)

    def reset(self) -> None:
        self.buf = None
        self.i = 0

    def sample(self) -> torch.Tensor:
        if self.buf is None or self.i >= self.n:
            self.buf = self._build()  # a longer-than-expected episode gets a fresh block
            self.i = 0
        v = self.buf[:, self.i]
        self.i += 1
        return v


# Channels that receive the COLOURED component; the rest get a fresh white draw.
# {0,1} = [ail, elev] — roll and g, the two axes that compose an out-of-plane maneuver.
# Deliberately NOT {2,3,4}: a correlated a[4] latches the trigger (`fire` fires at agent-space
# >= 0.0) into a magazine dump that would confound with the recorded ammo-dump finding, and a
# correlated a[3] holds throttle through the bleed phase of every reposition.
COLOURED_CHANNELS = (0, 1)


def make_explore_noise(cfg: TDMPC2Config, device: torch.device):
    """Build the exploration-noise source named by `cfg.explore_noise`."""
    kind = str(getattr(cfg, "explore_noise", "white")).lower()
    if kind == "white":
        return WhiteNoise(cfg.action_dim, device)
    if kind == "ou":
        # tau is in CONTROL STEPS on purpose: the sim rate is not in cfg, so seconds
        # would smuggle in a 120 Hz assumption. 36 steps = 1.5 s at 24 Hz.
        return OUNoise(cfg.action_dim, device, getattr(cfg, "explore_tau_steps", 36.0))
    if kind == "pink":
        return PinkNoise(
            cfg.action_dim, device, cfg.episode_length + 1, getattr(cfg, "explore_beta", 1.0)
        )
    raise ValueError(f"unknown cfg.explore_noise {kind!r} (expected white|ou|pink)")


class TDMPC2:
    def __init__(self, cfg: TDMPC2Config, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or resolve_device(cfg.device)
        self.model = WorldModel(cfg).to(self.device)
        self.discount = cfg.resolved_discount()

        self.optim = torch.optim.Adam(self.model.model_param_groups())
        self.pi_optim = torch.optim.Adam(self.model.pi_parameters(), lr=cfg.lr, eps=1e-5)
        self.scale = fn.RunningScale(cfg.tau, self.device)

        # MPPI warm-start: last plan's mean, shifted forward each control step.
        self._prev_mean = torch.zeros(cfg.horizon, cfg.action_dim, device=self.device)

        # Exploration noise for the final per-control-step draw (see the classes above).
        # Purely a COLLECTION-time object: no parameters, not checkpointed, `white` is
        # bit-identical to the reference. Reset at every episode start (`act(t0=True)`).
        self.explore_noise = make_explore_noise(cfg, self.device)

        # NaN-guard: a CPU snapshot of the last all-finite weights. On a hard tier the
        # value targets can explode and drive a single update non-finite; grad clipping
        # can't help (a NaN grad makes its own clip scale NaN and corrupts every param at
        # optim.step()). We skip such updates and roll back so one bad batch can't nuke a
        # multi-hour run. Refreshed every `_nan_snapshot_every` verified-finite updates.
        self._last_good_sd: dict | None = None
        self._nan_skips = 0
        self._good_updates = 0
        self._nan_snapshot_every = 100
        self.model.eval()

    # -- acting ------------------------------------------------------------
    def _explore_eps(self) -> torch.Tensor:
        """The pre-squash N(0,1) draw for the policy branch, `[1, A]`.

        `white` returns the reference draw directly — one `randn` of exactly `A` values, so
        the default config stays bit-identical to before this seam existed. Otherwise the
        coloured source supplies `COLOURED_CHANNELS` and the rest keep an independent white
        draw. Both sources have unit marginal variance, so the actor's own `log_std` keeps
        its meaning and only the temporal spectrum changes.
        """
        eps = torch.randn(1, self.cfg.action_dim, device=self.device)
        if isinstance(self.explore_noise, WhiteNoise):
            return eps
        coloured = self.explore_noise.sample()
        for ch in COLOURED_CHANNELS:
            eps[0, ch] = coloured[ch]
        return eps

    @torch.no_grad()
    def act(self, obs, *, t0: bool = False, eval_mode: bool = False) -> torch.Tensor:
        """Return a `[-1,1]^A` action for a single observation (numpy or tensor)."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if t0:
            self.explore_noise.reset()  # a new episode ⇒ a fresh noise trajectory
        z = self.model.encode(obs)
        if self.cfg.mpc:
            return self.plan(z, t0=t0, eval_mode=eval_mode).cpu()
        # eval draws nothing correlated: `mu` is returned, and `eps=None` keeps the RNG
        # stream and the output bit-identical to the reference.
        mu, pi, _, _ = self.model.pi(z, eps=None if eval_mode else self._explore_eps())
        a = mu if eval_mode else pi
        return a.squeeze(0).clamp(-1, 1).cpu()

    # -- planning (MPPI, design doc §2.2 / P3) -----------------------------
    @torch.no_grad()
    def plan(self, z: torch.Tensor, *, t0: bool = False, eval_mode: bool = False) -> torch.Tensor:
        """Model-predictive path-integral control over the latent model.

        Seeds the sample set with `num_pi_trajs` on-policy rollouts, then iteratively
        refits a per-step diagonal Gaussian to the top-`num_elites` return-weighted
        action sequences. Returns the first action of the optimized plan (mean at eval,
        mean+noise while exploring), clamped to the `[-1,1]` box.
        """
        cfg = self.cfg
        H, A = cfg.horizon, cfg.action_dim

        # Policy-prior seed trajectories.
        pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=self.device)
        if cfg.num_pi_trajs > 0:
            _z = z.repeat(cfg.num_pi_trajs, 1)
            for t in range(H - 1):
                pi_actions[t] = self.model.pi(_z)[1]
                _z = self.model.next(_z, pi_actions[t])
            pi_actions[H - 1] = self.model.pi(_z)[1]

        z = z.repeat(cfg.num_samples, 1)
        mean = torch.zeros(H, A, device=self.device)
        std = torch.full((H, A), cfg.max_std, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(H, cfg.num_samples, A, device=self.device)
        if cfg.num_pi_trajs > 0:
            actions[:, : cfg.num_pi_trajs] = pi_actions

        score = None
        for _ in range(cfg.iterations):
            n_sample = cfg.num_samples - cfg.num_pi_trajs
            noise = torch.randn(H, n_sample, A, device=self.device)
            actions[:, cfg.num_pi_trajs :] = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(
                -1, 1
            )
            value = self._estimate_value(z, actions).nan_to_num_(0.0)  # (num_samples, 1)
            elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
            elite_value = value[elite_idxs]  # (num_elites, 1)
            elite_actions = actions[:, elite_idxs]  # (H, num_elites, A)

            max_value = elite_value.max(0).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / (score.sum(0) + 1e-9)  # (num_elites, 1)
            mean = (score.unsqueeze(0) * elite_actions).sum(1) / (score.sum(0) + 1e-9)
            var = (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1) / (
                score.sum(0) + 1e-9
            )
            std = var.sqrt().clamp(cfg.min_std, cfg.max_std)

        # Sample one elite plan (weighted by its score) and take its first action.
        pick = torch.multinomial(score.squeeze(1), 1).item()
        plan_actions = elite_actions[:, pick]  # (H, A)
        self._prev_mean = mean
        a, a_std = plan_actions[0], std[0]
        if not eval_mode:
            # `explore_noise` is unit-marginal-variance in every mode, so `a_std` keeps
            # its meaning; only the noise's TEMPORAL spectrum changes (white ⇒ identical
            # to `torch.randn(A)`, same RNG consumption).
            a = a + a_std * self.explore_noise.sample()
        return a.clamp(-1, 1)

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Discounted return of each action sequence under the model, terminal-aware.

        `G += discount·(1-term)·reward` per step, terminals latch closed so no reward
        accrues past a predicted kill/death; bootstraps the tail with `Q(z_H, π(z_H))`.
        """
        G = torch.zeros(z.shape[0], 1, device=self.device)
        discount = 1.0
        term = torch.zeros(z.shape[0], 1, device=self.device)
        for t in range(self.cfg.horizon):
            reward = fn.two_hot_inv(self.model.reward(z, actions[t]), self.cfg).unsqueeze(-1)
            z = self.model.next(z, actions[t])
            G = G + discount * (1.0 - term) * reward
            if self.cfg.episodic:
                dead = (self.model.termination(z) > 0.0).float()
                term = torch.clamp(term + dead, max=1.0)
            discount *= self.discount
        pi = self.model.pi(z)[1]
        q = self.model.Q(z, pi, return_type="avg").unsqueeze(-1)
        return G + discount * (1.0 - term) * q

    # -- learning ----------------------------------------------------------
    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated) -> torch.Tensor:
        """`r + γ·(1-terminated)·min Q_target(z', π(z'))`. Only *terminated* (not
        truncated) zeroes the bootstrap (design doc §6)."""
        pi = self.model.pi(next_z)[1]
        q = self.model.Q(next_z, pi, return_type="min", target=True)  # (H, B)
        return reward + self.discount * (1.0 - terminated) * q

    def update(self, buffer) -> dict:
        cfg = self.cfg
        obs, action, reward, terminated, truncated = buffer.sample()
        # obs (H+1,B,O); action/reward/terminated/truncated (H,B,·)

        with torch.no_grad():
            next_z = self.model.encode(obs[1:])  # (H,B,latent), consistency + TD targets
            td_targets = self._td_target(next_z, reward, terminated)  # (H,B)

        self.model.train()
        H, B = cfg.horizon, obs.shape[1]
        rho = torch.pow(cfg.rho, torch.arange(H, device=self.device))  # (H,)

        # Latent rollout from the first real observation.
        zs = torch.empty(H + 1, B, cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0])
        zs[0] = z
        consistency = torch.zeros((), device=self.device)
        for t in range(H):
            z = self.model.next(z, action[t])
            zs[t + 1] = z
            consistency = consistency + rho[t] * F.mse_loss(z, next_z[t])

        _zs = zs[:-1]  # (H,B,latent): the latents actions were taken from
        q_logits = self.model.Q(_zs, action, return_type="all")  # (num_q,H,B,bins)
        reward_logits = self.model.reward(_zs, action)  # (H,B,bins)

        reward_ce = fn.soft_ce(reward_logits, reward, cfg).mean(1)  # (H,)
        reward_loss = (rho * reward_ce).sum()
        value_ce = fn.soft_ce(q_logits, td_targets.unsqueeze(0), cfg).mean(2)  # (num_q,H)
        value_loss = (rho.unsqueeze(0) * value_ce).sum()

        if cfg.episodic:
            term_logits = self.model.termination(zs[1:])  # (H,B,1)
            termination_loss = F.binary_cross_entropy_with_logits(
                term_logits, terminated.unsqueeze(-1)
            )
        else:
            termination_loss = torch.zeros((), device=self.device)

        consistency = consistency / H
        reward_loss = reward_loss / H
        value_loss = value_loss / (H * cfg.num_q)

        total = (
            cfg.consistency_coef * consistency
            + cfg.reward_coef * reward_loss
            + cfg.value_coef * value_loss
            + cfg.termination_coef * termination_loss
        )
        # NaN-guard (pre-step): a non-finite loss means the gradient will be non-finite too;
        # stepping would write NaN into every parameter. Skip and roll back instead.
        if not torch.isfinite(total):
            self._on_nan_skip()
            return self._nan_stats(total)

        self.optim.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
        # NaN-guard (post-clip): guards a non-finite grad that slipped past a finite `total`.
        if not torch.isfinite(grad_norm):
            self.optim.zero_grad(set_to_none=True)
            self._on_nan_skip()
            return self._nan_stats(total)
        self.optim.step()

        pi_loss = self.update_pi(zs.detach())
        self.model.soft_update_target_Qs()
        self.model.eval()

        # This update was fully finite — advance the rollback point periodically.
        self._good_updates += 1
        if self._last_good_sd is None or self._good_updates % self._nan_snapshot_every == 0:
            self._snapshot_good()

        return {
            "consistency_loss": consistency.detach().item(),
            "reward_loss": reward_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "termination_loss": termination_loss.detach().item(),
            "pi_loss": float(pi_loss),
            "total_loss": total.detach().item(),
            "grad_norm": float(grad_norm),
            "value_scale": self.scale.value,
        }

    def update_pi(self, zs: torch.Tensor) -> torch.Tensor:
        """Improve the policy prior to maximize the (scale-normalized) learned Q, with
        an entropy bonus. Only the prior's params move; `zs` is detached."""
        cfg = self.cfg
        self.pi_optim.zero_grad(set_to_none=True)
        _, pis, log_pis, _ = self.model.pi(zs)  # (H+1,B,A), (H+1,B,1)
        qs = self.model.Q(zs, pis, return_type="avg")  # (H+1,B)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(cfg.rho, torch.arange(zs.shape[0], device=self.device))  # (H+1,)
        # Maximize Q, minimize -entropy (= +log_pi); per-horizon mean over batch, ρ-weighted.
        per_step = (cfg.entropy_coef * log_pis.squeeze(-1) - qs).mean(1)  # (H+1,)
        pi_loss = (per_step * rho).mean()

        pi_loss.backward()
        nn.utils.clip_grad_norm_(self.model.pi_parameters(), cfg.grad_clip_norm)
        self.pi_optim.step()
        return pi_loss.detach()

    # -- NaN-guard ---------------------------------------------------------
    def _snapshot_good(self) -> None:
        """Cache the current (verified-finite) weights on CPU as the rollback point."""
        self._last_good_sd = {
            k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()
        }

    def _restore_good(self) -> bool:
        """Roll the model back to the last cached finite weights; False if none cached yet."""
        if self._last_good_sd is None:
            return False
        self.model.load_state_dict({k: v.to(self.device) for k, v in self._last_good_sd.items()})
        return True

    def _on_nan_skip(self) -> None:
        """Handle a skipped divergent update: roll back, count it, and make it loud."""
        restored = self._restore_good()
        self._nan_skips += 1
        self.model.eval()
        print(
            f"  ⚠ nan-guard: non-finite update skipped, "
            f"{'rolled back to last good weights' if restored else 'no snapshot yet — weights left as-is'} "
            f"(skip #{self._nan_skips})",
            flush=True,
        )

    def _nan_stats(self, total) -> dict:
        """Stats payload for a skipped update — mirrors update()'s keys so callers/logs don't
        break; the NaNs themselves flag that a rollback fired (and clear on the next update)."""
        nan = float("nan")
        return {
            "consistency_loss": nan,
            "reward_loss": nan,
            "value_loss": nan,
            "termination_loss": nan,
            "pi_loss": nan,
            "total_loss": float(total.detach().item()) if torch.is_tensor(total) else nan,
            "grad_norm": nan,
            "value_scale": self.scale.value,
            "nan_skip": 1,
        }

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict:
        # `cfg` is stored so exporters/loaders can rebuild the exact architecture
        # from the checkpoint alone (load_state_dict ignores it).
        import dataclasses

        return {
            "cfg": dataclasses.asdict(self.cfg),
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model"])
        self.optim.load_state_dict(sd["optim"])
        self.pi_optim.load_state_dict(sd["pi_optim"])
