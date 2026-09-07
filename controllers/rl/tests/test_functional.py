"""TD-MPC2 math primitives (design doc §6): two-hot round-trips, SimNorm sums to 1."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astrum_rl.tdmpc2 import functional as fn  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402
from astrum_rl.tdmpc2.layers import SimNorm  # noqa: E402


def _cfg(**kw):
    return TDMPC2Config(**kw)


def test_symlog_symexp_inverse():
    x = torch.linspace(-50, 50, 101)
    assert torch.allclose(fn.symexp(fn.symlog(x)), x, atol=1e-4)


def test_two_hot_is_a_distribution():
    cfg = _cfg()
    x = torch.tensor([-8.0, -0.3, 0.0, 1.7, 8.0])
    d = fn.two_hot(x, cfg)
    assert d.shape == (5, cfg.num_bins)
    assert torch.allclose(d.sum(-1), torch.ones(5), atol=1e-5)
    assert (d >= 0).all()


def test_two_hot_round_trip_in_expectation():
    cfg = _cfg()
    # In-range values (|symlog(x)| <= vmax) recover exactly through the log.
    x = torch.tensor([-9.5, -2.0, -0.1, 0.0, 0.4, 3.3, 9.5])
    probs = fn.two_hot(x, cfg)
    back = fn.two_hot_inv(torch.log(probs + 1e-12), cfg)
    assert torch.allclose(back, x, atol=1e-3)


def test_two_hot_inv_takes_logits():
    # Uniform logits -> softmax uniform -> expectation ~ symexp(0) = 0.
    cfg = _cfg()
    logits = torch.zeros(3, cfg.num_bins)
    assert torch.allclose(fn.two_hot_inv(logits, cfg), torch.zeros(3), atol=1e-5)


def test_num_bins_one_degenerates_to_symlog():
    cfg = _cfg(num_bins=1)
    x = torch.tensor([2.0, -3.0])
    assert torch.allclose(fn.two_hot(x, cfg), fn.symlog(x))
    assert torch.allclose(fn.two_hot_inv(fn.symlog(x), cfg), x, atol=1e-5)


def test_soft_ce_minimized_at_target():
    cfg = _cfg()
    target = torch.tensor([1.5])
    good = torch.log(fn.two_hot(target, cfg) + 1e-9)  # logits matching the target
    bad = torch.log(fn.two_hot(torch.tensor([-4.0]), cfg) + 1e-9)
    assert fn.soft_ce(good, target, cfg).item() < fn.soft_ce(bad, target, cfg).item()


def test_simnorm_groups_sum_to_one():
    cfg = _cfg()
    sn = SimNorm(cfg.simnorm_dim)
    z = sn(torch.randn(4, cfg.latent_dim))
    groups = z.view(4, -1, cfg.simnorm_dim).sum(-1)
    assert torch.allclose(groups, torch.ones_like(groups), atol=1e-5)
    assert (z >= 0).all()


def test_running_scale_tracks_spread():
    rs = fn.RunningScale(tau=1.0, device=torch.device("cpu"))  # tau=1 → jump to latest
    rs.update(torch.linspace(0, 100, 1000))
    assert rs.value > 50.0  # 95th-5th percentile of [0,100] ≈ 90
