"""`TDMPC2Config` — every hyperparameter in one dataclass (design doc §2.4/§5).

Defaults are the TD-MPC2 reference values, with the ASTRUM-specific overrides from
the design doc baked in (`episodic=True`, `action_repeat=5`, obs/action dims). The
derived-value helpers (`resolved_discount`, `resolved_seed_steps`, `bin_size`) mirror
the reference's runtime derivations — flagged in the design doc §10 as "confirm
against source"; these match the published formulas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TDMPC2Config:
    # -- env / shapes ------------------------------------------------------
    obs_dim: int = 31  # model input dim (already includes frame_stack; set from env)
    action_dim: int = 5  # [ail, elev, rud, throttle, trigger]
    action_repeat: int = 5  # ~24 Hz control over the 120 Hz sim
    frame_stack: int = 1  # concat last k obs (1 = off); only used to build the env
    episodic: bool = True  # episodes terminate → enable the termination head

    # -- architecture ------------------------------------------------------
    enc_dim: int = 256
    num_enc_layers: int = 2
    mlp_dim: int = 512
    latent_dim: int = 512
    simnorm_dim: int = 8
    num_q: int = 5
    dropout: float = 0.01

    # -- two-hot value/reward head ----------------------------------------
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # -- MPPI planner ------------------------------------------------------
    mpc: bool = True
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5

    # -- policy prior ------------------------------------------------------
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4

    # -- exploration noise (collection only; see agent.make_explore_noise) --
    # "white" = the TD-MPC2 reference i.i.d. draw (default; bit-identical to before).
    # "pink"/"ou" put the same marginal variance at the MANEUVER timescale so a
    # multi-second coordinated deviation is reachable at all. Not shape-bearing and
    # not used by any loss, so it is safe to flip on a warm checkpoint.
    # D2a measured why this matters and sized it: the actor's own sigma is already
    # 0.67-0.90 on the roll axis (not the 0.21 long quoted), so AMPLITUDE was never the
    # limiter -- the measured N_eff over a 36-step window is 34-41 against a white-noise
    # prediction of exactly 36, i.e. the draws are indistinguishable from independent.
    # Do NOT pair this with an actor-side std floor: sigma is already 13-18x min_std, so
    # such a floor could never bind.
    #
    # Pick tau/beta from the MEASURED N_eff (independent samples per 1.5 s window; the
    # actor reads 34-41, this WhiteNoise source reads 40.6 on the same estimator):
    #   ou tau=12  2.27   pink beta=1.0  2.20
    #   ou tau=18  1.86   pink beta=1.5  1.25
    #   ou tau=36  1.41   pink beta=2.0  1.07
    #   ou tau=72  1.23   pink beta=2.5  1.05
    # `ou` is the recommended launch arm: tau is an explicit correlation time, its marginal
    # variance is exactly unit by construction, and the closed-form AR(1) prediction matched
    # measurement to within 3%. Note beta=1 (the pink-noise-paper default) is the WEAKEST
    # coloured option here, so a null from it would be a weak null.
    explore_noise: str = "white"  # white | pink | ou
    explore_beta: float = 1.0  # pink spectral exponent (1 = pink, 2 = brown)
    explore_tau_steps: float = 36.0  # OU correlation time in CONTROL steps (36 ≈ 1.5 s @24 Hz)

    # -- optimization ------------------------------------------------------
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    batch_size: int = 256
    tau: float = 0.01  # target-Q Polyak rate
    rho: float = 0.5  # per-step loss discount over the horizon
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0

    # -- discount schedule (derived from episode length unless set) -------
    discount: float | None = None
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995
    # Control steps per episode; only feeds the discount/seed-step derivations below.
    # train.py overrides this from the env (max_time_s·rate_hz/action_repeat). The 288
    # default ≈ 12 s at 24 Hz — a fallback for unit tests that build a config directly.
    episode_length: int = 288

    # -- training loop -----------------------------------------------------
    buffer_size: int = 1_000_000
    seed_steps: int | None = None  # random warmup; derived if None
    steps: int = 1_000_000

    # -- runtime -----------------------------------------------------------
    device: str = "auto"
    compile: bool = False
    seed: int = 0

    @property
    def bin_size(self) -> float:
        return (self.vmax - self.vmin) / (self.num_bins - 1)

    def resolved_discount(self) -> float:
        """Discount from episode length, `clamp((L/d - 1)/(L/d), dmin, dmax)`."""
        if self.discount is not None:
            return self.discount
        frac = self.episode_length / self.discount_denom
        return min(max((frac - 1) / frac, self.discount_min), self.discount_max)

    def resolved_seed_steps(self) -> int:
        """Random-action warmup before the first update: `max(1000, 5·episode_length)`."""
        if self.seed_steps is not None:
            return self.seed_steps
        return max(1000, 5 * self.episode_length)
