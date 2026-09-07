"""Replay buffer: slice shapes, no boundary/seam crossings, terminated≠truncated (§6)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from astrum_rl.tdmpc2.buffer import ReplayBuffer  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402


def _cfg(**kw):
    base = dict(obs_dim=31, action_dim=5, horizon=3, batch_size=32, buffer_size=64)
    base.update(kw)
    return TDMPC2Config(**base)


def _fill(buf, cfg, episodes, ep_len, *, end="truncated"):
    """Add `episodes` episodes of `ep_len` steps. obs[0]=global counter, obs[1]=ep id;
    the incoming action/reward at slot j both encode j (to check the [1:] convention)."""
    dev = buf.device
    counter = 0
    for ep in range(episodes):
        o = torch.zeros(cfg.obs_dim, device=dev)
        o[0], o[1] = counter, ep
        buf.add(o, torch.zeros(cfg.action_dim, device=dev), 0.0, 0.0, 0.0, first=True)
        counter += 1
        for k in range(ep_len):
            o = torch.zeros(cfg.obs_dim, device=dev)
            o[0], o[1] = counter, ep
            last = k == ep_len - 1
            term = float(last and end == "terminated")
            trunc = float(last and end == "truncated")
            a = torch.full((cfg.action_dim,), float(counter), device=dev)
            buf.add(o, a, float(counter), term, trunc, first=False)
            counter += 1


def test_sample_shapes():
    cfg = _cfg()
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    _fill(buf, cfg, episodes=1, ep_len=40)
    obs, action, reward, terminated, truncated = buf.sample(16)
    H = cfg.horizon
    assert obs.shape == (H + 1, 16, cfg.obs_dim)
    assert action.shape == (H, 16, cfg.action_dim)
    assert reward.shape == terminated.shape == truncated.shape == (H, 16)


def test_slices_are_consecutive_same_episode():
    cfg = _cfg()
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    _fill(buf, cfg, episodes=3, ep_len=15)
    obs, action, reward, _, _ = buf.sample(256)
    # obs[0] is a global monotonic counter: within a valid window it must step by +1.
    counters = obs[..., 0]  # (H+1, B)
    diffs = counters[1:] - counters[:-1]
    assert torch.all(diffs == 1), "window crossed a seam or gap (non-consecutive)"
    # obs[1] is the episode id: constant within a window (never crosses a boundary).
    ep_ids = obs[..., 1]
    assert torch.all(ep_ids == ep_ids[0:1]), "window crossed an episode boundary"


def test_incoming_action_reward_convention():
    # action[t]/reward[t] come from slot s+1+t == obs[t]'s counter + 1.
    cfg = _cfg()
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    _fill(buf, cfg, episodes=2, ep_len=20)
    obs, action, reward, _, _ = buf.sample(128)
    obs_counter = obs[..., 0][:-1]  # (H, B): counter of the state acted from
    assert torch.all(action[..., 0] - obs_counter == 1)
    assert torch.all(reward - obs_counter == 1)


def test_full_ring_still_consecutive():
    # Overfill so the ring wraps several times; windows must never span the write seam.
    cfg = _cfg(buffer_size=48)
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    _fill(buf, cfg, episodes=30, ep_len=9)  # ~300 adds into a 48-slot ring
    assert buf.full
    obs, *_ = buf.sample(512)
    counters = obs[..., 0]
    assert torch.all(counters[1:] - counters[:-1] == 1)
    ep_ids = obs[..., 1]
    assert torch.all(ep_ids == ep_ids[0:1])


def test_terminated_and_truncated_are_distinct():
    cfg = _cfg()
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    _fill(buf, cfg, episodes=1, ep_len=20, end="terminated")
    _fill(buf, cfg, episodes=1, ep_len=20, end="truncated")
    # Exactly one terminated flag and one truncated flag were written.
    assert buf._terminated.sum().item() == 1.0
    assert buf._truncated.sum().item() == 1.0
    # ...and never on the same slot.
    assert torch.all((buf._terminated * buf._truncated) == 0)


def test_raises_before_warmup():
    cfg = _cfg()
    buf = ReplayBuffer(cfg, torch.device("cpu"))
    buf.add(torch.zeros(cfg.obs_dim), torch.zeros(cfg.action_dim), 0.0, 0.0, 0.0, first=True)
    with pytest.raises(RuntimeError):
        buf.sample(4)
