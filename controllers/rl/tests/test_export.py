"""P5: policy-prior export round-trips, and the self-play opponent produces native
actions. The prior export/reload is pure-torch (no sim); the self-play env path needs
the astrum_env .so and is skipped when it isn't importable."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from astrum_rl.tdmpc2.agent import TDMPC2  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402
from astrum_rl.tdmpc2.export import PolicyPrior, export_policy, load_prior  # noqa: E402


def _tiny_cfg(**kw):
    base = dict(
        obs_dim=8,
        action_dim=5,
        enc_dim=32,
        mlp_dim=32,
        latent_dim=32,
        simnorm_dim=4,
        num_q=2,
        horizon=3,
        device="cpu",
    )
    base.update(kw)
    return TDMPC2Config(**base)


def test_prior_matches_agent_pi_mean():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    agent = TDMPC2(cfg, torch.device("cpu"))
    prior = PolicyPrior.from_world_model(agent.model, cfg)

    obs = torch.randn(4, cfg.obs_dim)
    with torch.no_grad():
        want, *_ = agent.model.pi(agent.model.encode(obs))  # mu (already tanh-squashed)
        got = prior(obs)
    assert torch.allclose(want, got, atol=1e-6)
    assert got.abs().max() <= 1.0


def test_export_roundtrip_and_metadata(tmp_path):
    torch.manual_seed(0)
    cfg = _tiny_cfg(action_repeat=5, frame_stack=2)
    agent = TDMPC2(cfg, torch.device("cpu"))
    ckpt = tmp_path / "run.pt"
    torch.save(agent.state_dict(), ckpt)

    pt, js = export_policy(str(ckpt), str(tmp_path / "policy"))
    prior = load_prior(pt)

    obs = torch.randn(3, cfg.obs_dim)
    with torch.no_grad():
        a = prior(obs)
    assert a.shape == (3, cfg.action_dim)
    assert a.abs().max() <= 1.0

    import json

    meta = json.loads((tmp_path / "policy.json").read_text())
    assert meta["obs_dim"] == cfg.obs_dim and meta["action_dim"] == cfg.action_dim
    assert meta["frame_stack"] == 2 and meta["action_repeat"] == 5
    assert meta["action_low"] == [-1, -1, -1, 0, 0]


def test_checkpoint_without_cfg_is_rejected(tmp_path):
    from astrum_rl.tdmpc2.export import load_agent_from_checkpoint

    torch.save({"model": {}}, tmp_path / "old.pt")
    with pytest.raises(KeyError):
        load_agent_from_checkpoint(str(tmp_path / "old.pt"))


def test_opponent_pool_sampling(tmp_path):
    """Pool sampling + the exported-prior opponent's forward path (no sim needed)."""
    from astrum_rl.tdmpc2.selfplay import OpponentPool, ScriptedPolicyOpponent

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    agent = TDMPC2(cfg, torch.device("cpu"))
    ckpt = tmp_path / "run.pt"
    torch.save(agent.state_dict(), ckpt)
    pt, _ = export_policy(str(ckpt), str(tmp_path / "p0"))

    pool = OpponentPool()
    pool.add(pt)
    assert len(pool) == 1
    opp = pool.sample(np.random.default_rng(0))
    assert isinstance(opp, ScriptedPolicyOpponent)

    # Drive it directly with fake AState-likes via build_observation's inputs would need
    # the sim; here we just confirm the loaded prior maps an obs → a native action list.
    with torch.no_grad():
        a = opp.prior(torch.zeros(1, cfg.obs_dim)).squeeze(0).numpy()
    from astrum_rl.envs import map_tdmpc_action

    native = map_tdmpc_action(a, opp._low, opp._high)
    assert native.shape == (cfg.action_dim,)
    assert native[3] >= 0.0 and native[3] <= 1.0  # throttle mapped into [0,1]


# -- curriculum-21 self-play trainer wiring (pure-torch; the sim path is smoke-tested in-container) --
def test_snapshot_prior_writes_a_loadable_pool_entry(tmp_path):
    """The learner's in-process pool snapshot is a TorchScript prior a collector can load and that
    matches the live policy — the round-trip the self-play opponent depends on."""
    from astrum_rl.tdmpc2.train_parallel import _snapshot_prior

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    agent = TDMPC2(cfg, torch.device("cpu"))
    pool = tmp_path / "pool"
    pool.mkdir()

    path = _snapshot_prior(agent, cfg, str(pool), 0)
    assert path.endswith("snap_000000.pt")
    assert not list(pool.glob("*.tmp"))  # atomic: the .tmp staging file is renamed away

    prior = load_prior(path)
    obs = torch.randn(4, cfg.obs_dim)
    with torch.no_grad():
        got = prior(obs)
        want = PolicyPrior.from_world_model(agent.model, cfg)(obs)
    assert torch.allclose(got, want, atol=1e-6)  # the snapshot IS the current policy prior


def test_sample_opponent_from_pool_dir_and_empty(tmp_path):
    """`_sample_opponent` re-globs the pool dir and returns a policy opponent, or None when empty."""
    from astrum_rl.tdmpc2.selfplay import ScriptedPolicyOpponent
    from astrum_rl.tdmpc2.train_parallel import _sample_opponent, _snapshot_prior

    rng = np.random.default_rng(0)
    pool = tmp_path / "pool"
    pool.mkdir()
    assert _sample_opponent(str(pool), 1, 0.0, rng) is None  # empty pool → None (startup race)

    torch.manual_seed(0)
    cfg = _tiny_cfg()
    agent = TDMPC2(cfg, torch.device("cpu"))
    _snapshot_prior(agent, cfg, str(pool), 0)
    opp = _sample_opponent(str(pool), 1, 0.0, rng)
    assert isinstance(opp, ScriptedPolicyOpponent)
    # a stray non-snapshot .pt in the dir must NOT be sampled (pattern is snap_*.pt).
    (pool / "other.pt").write_bytes(b"junk")
    with torch.no_grad():
        a = _sample_opponent(str(pool), 1, 0.0, rng).prior(torch.zeros(1, cfg.obs_dim))
    assert a.shape == (1, cfg.action_dim)  # loaded the real snapshot, not the junk file


def test_prune_pool_keeps_most_recent(tmp_path):
    from astrum_rl.tdmpc2.train_parallel import _prune_pool

    pool = tmp_path / "pool"
    pool.mkdir()
    for i in range(5):
        (pool / f"snap_{i:06d}.pt").write_bytes(b"x")
    _prune_pool(str(pool), max_pool=2)
    kept = sorted(p.name for p in pool.glob("snap_*.pt"))
    assert kept == ["snap_000003.pt", "snap_000004.pt"]  # the two highest-indexed survive
    # max_pool <= 0 is a no-op (unbounded pool).
    _prune_pool(str(pool), max_pool=0)
    assert len(list(pool.glob("snap_*.pt"))) == 2
