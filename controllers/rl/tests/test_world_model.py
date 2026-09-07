"""WorldModel head shapes + invariants (design doc §2.1)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astrum_rl.tdmpc2 import TDMPC2Config, WorldModel  # noqa: E402


@pytest.fixture
def wm():
    torch.manual_seed(0)
    return TDMPC2Config(), WorldModel(TDMPC2Config())


def test_head_shapes_single_step(wm):
    cfg, model = wm
    B = 8
    obs = torch.randn(B, cfg.obs_dim)
    a = torch.rand(B, cfg.action_dim) * 2 - 1
    z = model.encode(obs)
    assert z.shape == (B, cfg.latent_dim)
    assert model.next(z, a).shape == (B, cfg.latent_dim)
    assert model.reward(z, a).shape == (B, cfg.num_bins)
    assert model.termination(z).shape == (B, 1)


def test_pi_is_squashed(wm):
    cfg, model = wm
    z = torch.randn(16, cfg.latent_dim)
    mu, pi, log_pi, log_std = model.pi(z)
    assert mu.shape == pi.shape == (16, cfg.action_dim)
    assert log_pi.shape == (16, 1)
    assert mu.abs().max() <= 1.0 and pi.abs().max() <= 1.0
    assert (log_std >= cfg.log_std_min - 1e-4).all() and (log_std <= cfg.log_std_max + 1e-4).all()


def test_q_return_types(wm):
    cfg, model = wm
    B = 8
    z = torch.randn(B, cfg.latent_dim)
    a = torch.rand(B, cfg.action_dim) * 2 - 1
    assert model.Q(z, a, "all").shape == (cfg.num_q, B, cfg.num_bins)
    assert model.Q(z, a, "min").shape == (B,)
    assert model.Q(z, a, "avg").shape == (B,)


def test_heads_pass_horizon_major(wm):
    cfg, model = wm
    T, B = 3, 8
    z = torch.randn(T, B, cfg.latent_dim)
    a = torch.rand(T, B, cfg.action_dim) * 2 - 1
    assert model.next(z, a).shape == (T, B, cfg.latent_dim)
    assert model.Q(z, a, "all").shape == (cfg.num_q, T, B, cfg.num_bins)
    assert model.Q(z, a, "min").shape == (T, B)


def test_target_starts_synced_then_diverges_under_update(wm):
    cfg, model = wm
    z = torch.randn(4, cfg.latent_dim)
    a = torch.rand(4, cfg.action_dim) * 2 - 1
    # Fresh target is a deepcopy → identical outputs.
    online = model.Q(z, a, "all")
    target = model.Q(z, a, "all", target=True)
    assert torch.allclose(online, target, atol=1e-6)
    # Perturb online params, soft-update: target moves toward online by tau.
    with torch.no_grad():
        for p in model._Qs.parameters():
            p.add_(torch.randn_like(p))
    model.soft_update_target_Qs()
    moved = model.Q(z, a, "all", target=True)
    assert not torch.allclose(moved, target, atol=1e-6)


def test_reward_and_q_start_at_zero(wm):
    # Zero-init of the final head layer → symexp(0)=0 predictions at init.
    cfg, model = wm
    z = torch.randn(6, cfg.latent_dim)
    a = torch.rand(6, cfg.action_dim) * 2 - 1
    from astrum_rl.tdmpc2 import functional as fn

    assert torch.allclose(fn.two_hot_inv(model.reward(z, a), cfg), torch.zeros(6), atol=1e-5)
    assert torch.allclose(model.Q(z, a, "avg"), torch.zeros(6), atol=1e-5)
