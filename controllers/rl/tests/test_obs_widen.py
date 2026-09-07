"""Zero-init encoder-column surgery for an OBS APPEND (`_widen_resume_obs`, curriculum-87).

The observation had been frozen for ~17 consecutive runs, so growing it is the first time this
trainer has had to resume a checkpoint into a WIDER obs. The whole justification for doing that on
a warm champion is a bit-identity claim: pad the encoder's first layer with ZERO columns and the
resumed policy behaves exactly as before on the features it was trained on, so the new signal has
to be earned by gradient descent instead of arriving as a step change. Every historical collapse
in this project began with a warm start that perturbed the policy on contact, so that claim is
load-bearing and tested here rather than argued.

Covers: the padded weight's exact layout (including the per-FRAME placement that frame stacking
requires), true bit-identity of the encoder output, Adam moment padding, and that a SHRINK or a
frame_stack mismatch is refused rather than silently guessed at.
"""

from __future__ import annotations

import pytest
import torch
from astrum_rl.tdmpc2.agent import TDMPC2
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.train_parallel import _widen_resume_obs

OLD, NEW = 31, 41


def _agent(obs_dim: int) -> TDMPC2:
    cfg = TDMPC2Config(
        obs_dim=obs_dim,
        action_dim=5,
        latent_dim=64,
        enc_dim=32,
        num_enc_layers=2,
        mlp_dim=64,
        num_q=2,
        horizon=3,
        batch_size=8,
        device="cpu",
    )
    torch.manual_seed(0)
    return TDMPC2(cfg, torch.device("cpu"))


def _sd(agent: TDMPC2) -> dict:
    return {
        "model": agent.model.state_dict(),
        "optim": agent.optim.state_dict(),
        "pi_optim": agent.pi_optim.state_dict(),
        "cfg": vars(agent.cfg)
        if not hasattr(agent.cfg, "__dataclass_fields__")
        else {f.name: getattr(agent.cfg, f.name) for f in agent.cfg.__dataclass_fields__.values()},
    }


def _step_once(agent: TDMPC2, obs_dim: int) -> None:
    """Put real (non-zero) Adam moments in the encoder group, so padding has something to preserve."""
    z = agent.model.encode(torch.randn(4, obs_dim))
    z.square().mean().backward()
    agent.optim.step()
    agent.optim.zero_grad(set_to_none=True)


def test_widen_is_bit_identical_on_the_trained_features():
    agent = _agent(OLD)
    _step_once(agent, OLD)
    sd = _sd(agent)

    obs_old = torch.randn(6, OLD)
    with torch.no_grad():
        want = agent.model.encode(obs_old)

    _widen_resume_obs(sd, OLD, NEW, frame_stack=1)
    assert sd["cfg"]["obs_dim"] == NEW
    assert sd["model"]["_encoder.0.weight"].shape == (32, NEW)

    wide = _agent(NEW)
    wide.model.load_state_dict(sd["model"])
    # The live env appends the new features; a warm policy must be unmoved by WHATEVER they are,
    # not merely by zeros — the zero columns are in the WEIGHTS, so any values must cancel.
    obs_new = torch.cat([obs_old, torch.randn(6, NEW - OLD) * 10.0], dim=1)
    with torch.no_grad():
        got = wide.model.encode(obs_new)
    assert torch.equal(got, want), "zero-padded encoder must be bit-identical on the old features"


def test_widen_places_columns_per_frame_when_stacked():
    fs = 3
    agent = _agent(OLD * fs)
    sd = _sd(agent)
    w_before = sd["model"]["_encoder.0.weight"].clone()

    _widen_resume_obs(sd, OLD, NEW, frame_stack=fs)
    w = sd["model"]["_encoder.0.weight"]
    assert w.shape == (32, NEW * fs)
    assert sd["cfg"]["obs_dim"] == NEW * fs
    for f in range(fs):
        trained = w[:, f * NEW : f * NEW + OLD]
        pad = w[:, f * NEW + OLD : (f + 1) * NEW]
        assert torch.equal(trained, w_before[:, f * OLD : (f + 1) * OLD]), f"frame {f} remapped"
        assert torch.all(pad == 0.0), f"frame {f} pad must be zero"


def test_widen_pads_adam_moments_and_keeps_history():
    agent = _agent(OLD)
    _step_once(agent, OLD)
    sd = _sd(agent)
    enc0 = sd["optim"]["param_groups"][0]["params"][0]
    before = sd["optim"]["state"][enc0]["exp_avg"].clone()
    assert before.abs().sum() > 0, "fixture must produce non-trivial moments to preserve"

    _widen_resume_obs(sd, OLD, NEW, frame_stack=1)
    after = sd["optim"]["state"][enc0]["exp_avg"]
    assert after.shape == (32, NEW)
    assert torch.equal(after[:, :OLD], before), "trained columns must keep their momentum"
    assert torch.all(after[:, OLD:] == 0.0), "new columns start at a fresh param's zero moment"


def test_widen_refuses_a_layout_it_cannot_infer():
    sd = _sd(_agent(OLD))
    with pytest.raises(SystemExit, match="cannot widen obs"):
        _widen_resume_obs(sd, OLD + 1, NEW, frame_stack=1)  # claimed old width is not the tensor's
