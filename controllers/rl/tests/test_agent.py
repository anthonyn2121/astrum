"""TDMPC2 agent: acting shapes/bounds, a running update, and loss decrease on synthetic
data (design doc §7 P2 acceptance). Also the MPPI planner sanity (P3)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astrum_rl.tdmpc2.agent import TDMPC2  # noqa: E402
from astrum_rl.tdmpc2.buffer import ReplayBuffer  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402


def _tiny_cfg(**kw):
    """A small, fast config for CPU unit tests."""
    base = dict(
        obs_dim=8,
        action_dim=3,
        enc_dim=32,
        num_enc_layers=2,
        mlp_dim=32,
        latent_dim=32,
        simnorm_dim=4,
        num_q=2,
        horizon=3,
        batch_size=32,
        buffer_size=2048,
        num_samples=64,
        num_pi_trajs=4,
        num_elites=8,
        iterations=2,
        device="cpu",
    )
    base.update(kw)
    return TDMPC2Config(**base)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _agent(**kw):
    return TDMPC2(_tiny_cfg(**kw), torch.device("cpu"))


def test_act_mpc_shape_and_bounds():
    ag = _agent(mpc=True)
    a = ag.act(torch.randn(ag.cfg.obs_dim), t0=True)
    assert a.shape == (ag.cfg.action_dim,)
    assert a.min() >= -1.0 and a.max() <= 1.0


def test_act_policy_prior_path():
    ag = _agent(mpc=False)
    a = ag.act(torch.randn(ag.cfg.obs_dim), eval_mode=True)
    assert a.shape == (ag.cfg.action_dim,)
    assert a.abs().max() <= 1.0


def test_plan_warm_start_shift():
    ag = _agent(mpc=True)
    obs = torch.randn(ag.cfg.obs_dim)
    ag.act(obs, t0=True)  # seeds _prev_mean
    prev = ag._prev_mean.clone()
    ag.act(obs, t0=False)  # should shift & refit, not error
    assert ag._prev_mean.shape == prev.shape


def _fill_random(ag, buf, n):
    cfg = ag.cfg
    obs = torch.randn(cfg.obs_dim)
    buf.add(obs, torch.zeros(cfg.action_dim), 0.0, 0.0, 0.0, first=True)
    for i in range(n):
        a = torch.rand(cfg.action_dim) * 2 - 1
        nxt = torch.randn(cfg.obs_dim)
        done = (i + 1) % 40 == 0
        buf.add(nxt, a, float(torch.randn(1)), 0.0, float(done), first=False)
        if done:
            buf.add(
                torch.randn(cfg.obs_dim), torch.zeros(cfg.action_dim), 0.0, 0.0, 0.0, first=True
            )


def test_update_runs_and_is_finite():
    ag = _agent()
    buf = ReplayBuffer(ag.cfg, torch.device("cpu"))
    _fill_random(ag, buf, 300)
    stats = ag.update(buf)
    for k in ("consistency_loss", "reward_loss", "value_loss", "termination_loss", "pi_loss"):
        assert stats[k] == stats[k], f"{k} is NaN"  # finite / not-NaN


def test_nan_guard_skips_and_rolls_back():
    """A non-finite update must be skipped and the weights rolled back to the last finite
    snapshot, then recover — one divergent batch (seen on hard curriculum tiers, where value
    targets can explode) must not corrupt a multi-hour run. Grad clipping can't do this: a
    NaN grad makes its own clip scale NaN and propagates into every param at optim.step()."""
    ag = _agent()
    ag._nan_snapshot_every = 1  # snapshot every finite update so the rollback point is current
    buf = ReplayBuffer(ag.cfg, torch.device("cpu"))
    _fill_random(ag, buf, 300)

    ag.update(buf)  # first finite update establishes the snapshot
    good = ag.update(buf)
    assert good.get("nan_skip") is None and ag._nan_skips == 0
    ref = torch.cat([p.detach().flatten() for p in ag.model.parameters()]).clone()

    # Force non-finite value targets → non-finite total. `_td_target` is a plain method on
    # TDMPC2 (not an nn.Module submodule), so it's safe to monkeypatch.
    orig_td = ag._td_target
    ag._td_target = lambda *a, **k: orig_td(*a, **k) * float("nan")
    try:
        stats = ag.update(buf)
    finally:
        ag._td_target = orig_td

    assert stats.get("nan_skip") == 1 and ag._nan_skips == 1, stats
    after = torch.cat([p.detach().flatten() for p in ag.model.parameters()])
    assert torch.isfinite(after).all(), "weights must stay finite after a skipped update"
    assert torch.allclose(after, ref), "weights must roll back to the last finite snapshot"

    rec = ag.update(buf)  # learner recovers on the next well-formed batch
    assert rec.get("nan_skip") is None and rec["reward_loss"] == rec["reward_loss"]


def test_reward_loss_decreases_on_constant_reward():
    """A fixed dataset whose reward is always +1: the reward head must learn it."""
    ag = _agent()
    cfg = ag.cfg
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    # One long episode; obs random-but-fixed, reward ≡ 1.
    obs = torch.randn(cfg.obs_dim)
    buf.add(obs, torch.zeros(cfg.action_dim), 0.0, 0.0, 0.0, first=True)
    torch.manual_seed(1)
    for _ in range(400):
        buf.add(torch.randn(cfg.obs_dim), torch.rand(cfg.action_dim) * 2 - 1, 1.0, 0.0, 0.0, False)

    first = ag.update(buf)["reward_loss"]
    for _ in range(150):
        stats = ag.update(buf)
    assert stats["reward_loss"] < first * 0.6, (first, stats["reward_loss"])


def test_target_q_tracks_online_after_updates():
    ag = _agent()
    buf = ReplayBuffer(ag.cfg, torch.device("cpu"))
    _fill_random(ag, buf, 300)
    before = torch.cat([p.flatten() for p in ag.model._target_Qs.parameters()]).clone()
    for _ in range(20):
        ag.update(buf)
    after = torch.cat([p.flatten() for p in ag.model._target_Qs.parameters()])
    assert not torch.allclose(before, after)  # Polyak-tracked, so it moved


# ---------------------------------------------------------------------------
# D2b: the coloured-exploration seam on the POLICY branch (the path collection
# actually takes, since cfg.mpc is False in every shipped checkpoint).
# ---------------------------------------------------------------------------


def _flown(ag, steps: int, obs):
    """The actions collection would actually execute, from a fixed observation."""
    return torch.stack([ag.act(obs, t0=(i == 0)) for i in range(steps)])


def test_explore_seam_leaves_eval_and_white_bit_identical():
    """`eval_mode` must be untouched, and the DEFAULT (white) arm must not move either.

    `_explore_eps` draws exactly `A` values for white, the same count `randn_like(mu)`
    consumed before the seam existed, so the RNG stream stays aligned.
    """
    obs = torch.randn(_tiny_cfg().obs_dim)

    for eval_mode in (True, False):
        torch.manual_seed(7)
        a = _agent(mpc=False, explore_noise="white")
        torch.manual_seed(11)
        got = torch.stack([a.act(obs, t0=(i == 0), eval_mode=eval_mode) for i in range(5)])

        # Reference: bypass the seam entirely, exactly as the pre-D2b code did.
        torch.manual_seed(7)
        b = _agent(mpc=False, explore_noise="white")
        torch.manual_seed(11)
        ref = []
        for _ in range(5):
            z = b.model.encode(obs.unsqueeze(0))
            mu, pi, _, _ = b.model.pi(z)
            ref.append((mu if eval_mode else pi).squeeze(0).clamp(-1, 1))
        assert torch.equal(got, torch.stack(ref)), f"white/{eval_mode=} arm moved"


def test_eval_mode_ignores_the_coloured_source():
    """Greedy is greedy: flipping the noise kind must not change an `eval_mode` action."""
    obs = torch.randn(_tiny_cfg().obs_dim)
    out = []
    for kind in ("white", "ou", "pink"):
        torch.manual_seed(7)
        ag = _agent(mpc=False, explore_noise=kind, episode_length=64)
        out.append(_flown_eval := ag.act(obs, t0=True, eval_mode=True))
    assert torch.equal(out[0], out[1]) and torch.equal(out[0], out[2])


def _neff36(x: torch.Tensor) -> torch.Tensor:
    """Independent samples per 36-step (1.5 s) window: `var(x) / var(window mean)`.

    This, not a lag-k autocorrelation, is the statistic the lever is about — it is what
    decides whether a coherent 1.5 s deviation is reachable, and it is what D2a measured on
    the actor (ail: 41.2 / 33.9 against a white-noise reference of 36). A sample ACF is the
    wrong instrument here: over a few-hundred-sample block with the sample mean removed it
    is biased low for a long-memory process, which made beta=1 pink read 0.105 at lag 36
    while its N_eff is in fact 2.20.
    """
    c = torch.cumsum(torch.cat([torch.zeros(1, x.shape[1]), x], 0), 0)
    m = (c[36:] - c[:-36]) / 36.0
    return x.var(0) / m.var(0).clamp_min(1e-12)


@pytest.mark.parametrize("kind", ["ou", "pink"])
def test_coloured_noise_reaches_the_flown_action(kind):
    """The lever's whole point, asserted on the EXECUTED action.

    Asserting on the flown action rather than on the noise object is what makes this a test
    of the plumbing instead of the RNG: before D2b the sampler was never consulted on this
    branch at all, so `explore_noise` could report 'pink' while collection sampled white —
    that is the defect being fixed. Measured sources at these settings: white 40.6,
    ou tau=36 1.41, pink beta=1 2.20 (see the config comment), so the bands below are wide.
    """
    steps, obs = 400, torch.randn(_tiny_cfg().obs_dim)

    torch.manual_seed(3)
    white = _flown(_agent(mpc=False, explore_noise="white", episode_length=steps), steps, obs)
    torch.manual_seed(3)
    col = _flown(_agent(mpc=False, explore_noise=kind, episode_length=steps), steps, obs)

    from astrum_rl.tdmpc2.agent import COLOURED_CHANNELS

    nw, nc = _neff36(white), _neff36(col)
    for ch in COLOURED_CHANNELS:
        assert nw[ch] > 15.0, f"white ch{ch} N_eff {nw[ch]:.1f} — expected ~36, harness is off"
        assert nc[ch] < 8.0, f"{kind} ch{ch} N_eff {nc[ch]:.1f} — noise did not reach the action"
        assert nc[ch] < nw[ch] / 3.0, f"{kind} ch{ch}: {nc[ch]:.1f} vs white {nw[ch]:.1f}"


def test_uncoloured_channels_stay_white():
    """a[2]/a[3]/a[4] must keep an independent draw: a correlated trigger latches into a
    magazine dump and a correlated throttle holds power through every bleed phase.

    The fixed observation is load-bearing, not laziness. It pins `mu`, which isolates the
    plumbing: on a real trajectory `dev = a - mu` also carries the slow drift of the squash
    gain `(1-mu^2)` as the aircraft maneuvers, so an uncoloured channel's N_eff falls there
    (rud read 11.0 white vs 3.0 coloured in the D2c exec arm) without any correlation having
    leaked into its draw. Holding `mu` constant separates the two.
    """
    steps, obs = 400, torch.randn(8)
    torch.manual_seed(3)
    ag = _agent(mpc=False, action_dim=5, explore_noise="ou", episode_length=steps)
    n = _neff36(_flown(ag, steps, obs))
    for ch in (2, 3, 4):
        assert n[ch] > 15.0, f"ch{ch} N_eff {n[ch]:.1f} — picked up correlation, must stay white"
