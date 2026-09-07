"""Episodic replay for the single-env off-policy loop (design doc §2.3, §6).

A flat ring buffer of individual transitions. Each stored slot `j` holds `(obs_j,
action_j, reward_j, terminated_j, truncated_j, first_j)` where `action_j/reward_j/
terminated_j` describe the transition that *led into* `obs_j` (the reference's
"incoming" convention), and `first_j` marks an episode's initial observation (added
with a zero action / zero reward that the sampler always drops).

`sample()` returns a horizon-major slice of length `horizon+1`:

    obs         (H+1, B, obs_dim)   states s_i … s_{i+H}
    action      (H,   B, act_dim)   actions a_i … a_{i+H-1}   (slot[1:] — drops incoming)
    reward      (H,   B)            r_i … r_{i+H-1}
    terminated  (H,   B)            was s_{i+t+1} a *real* terminal (kill/death/deck)?
    truncated   (H,   B)            was it a time/bubble cutoff instead?

**Why terminated and truncated are separate (§6, hard constraint):** the value target
zeroes the bootstrap only on `terminated`; a `truncated` step must still bootstrap
`Q(s_{i+H})`. Conflating them teaches the jet that running out the clock is as bad as
dying. The two are stored distinctly and only `terminated` feeds `_td_target`.

Slices never cross an episode boundary (a `first` flag interior to the window is
rejected) nor the ring's write seam, so every returned window is a run of genuinely
consecutive transitions from one episode.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.capacity = int(cfg.buffer_size)
        self.horizon = int(cfg.horizon)
        self.slice_len = self.horizon + 1
        obs_dim, act_dim = cfg.obs_dim, cfg.action_dim

        self._obs = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self._action = torch.empty((self.capacity, act_dim), dtype=torch.float32, device=device)
        self._reward = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self._terminated = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self._truncated = torch.zeros((self.capacity,), dtype=torch.float32, device=device)
        self._first = torch.zeros((self.capacity,), dtype=torch.bool, device=device)

        self._pos = 0  # next write slot == oldest slot once full
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.capacity

    # -- writing -----------------------------------------------------------
    def add(self, obs, action, reward, terminated, truncated, first: bool) -> None:
        """Append one transition. `first=True` marks an episode's initial obs (its
        `action`/`reward`/`terminated` are ignored by the sampler — pass zeros)."""
        i = self._pos
        self._obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self._action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._reward[i] = float(reward)
        self._terminated[i] = float(terminated)
        self._truncated[i] = float(truncated)
        self._first[i] = bool(first)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode(self, obs, action, reward, terminated, truncated) -> int:
        """Append a whole episode at once (used by the parallel trainer's learner).

        Equivalent to one `add(..., first=True)` for `obs[0]` followed by an
        `add(obs[i+1], action[i], reward[i], terminated[i], truncated[i], first=False)`
        per step — but writes all `T+1` slots in a couple of vectorized ops instead of
        `T+1` per-element host→device copies, so ingesting a collector's episode barely
        touches the GPU the learner is trying to keep busy updating. Returns the number
        of *transitions* (`T`), i.e. `len(action)`.

        The stored layout is byte-for-byte what the sequential `add()` path produces:
        slot 0 is the episode's initial obs (`first=True`, zero action/reward/term/trunc
        that the sampler always drops); slots `1..T` carry the incoming action/reward
        that led into `obs[1..T]`. Episode boundaries therefore stay marked, so
        `sample()`'s rejection of interior `first`s keeps horizon windows within one
        episode exactly as before.
        """
        obs = np.asarray(obs, dtype=np.float32)  # (T+1, obs_dim)
        action = np.asarray(action, dtype=np.float32).reshape(len(obs) - 1, -1)  # (T, act_dim)
        t = action.shape[0]
        if t == 0:
            return 0
        if obs.shape[0] != t + 1:
            raise ValueError(f"expected obs of length T+1={t + 1}, got {obs.shape[0]}")
        n = t + 1
        a_dim = self._action.shape[1]

        store_action = np.zeros((n, a_dim), dtype=np.float32)
        store_action[1:] = action
        store_reward = np.zeros((n,), dtype=np.float32)
        store_reward[1:] = np.asarray(reward, dtype=np.float32).reshape(-1)
        store_term = np.zeros((n,), dtype=np.float32)
        store_term[1:] = np.asarray(terminated, dtype=np.float32).reshape(-1)
        store_trunc = np.zeros((n,), dtype=np.float32)
        store_trunc[1:] = np.asarray(truncated, dtype=np.float32).reshape(-1)
        store_first = np.zeros((n,), dtype=bool)
        store_first[0] = True

        idx = torch.as_tensor((self._pos + np.arange(n)) % self.capacity, device=self.device)
        self._obs[idx] = torch.as_tensor(obs, device=self.device)
        self._action[idx] = torch.as_tensor(store_action, device=self.device)
        self._reward[idx] = torch.as_tensor(store_reward, device=self.device)
        self._terminated[idx] = torch.as_tensor(store_term, device=self.device)
        self._truncated[idx] = torch.as_tensor(store_trunc, device=self.device)
        self._first[idx] = torch.as_tensor(store_first, device=self.device)

        self._pos = (self._pos + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        return t

    # -- sampling ----------------------------------------------------------
    def sample(self, batch_size: int | None = None):
        """Draw `batch_size` valid horizon slices (rejection-sampled boundaries)."""
        B = batch_size or self.cfg.batch_size
        if self._size < self.slice_len:
            raise RuntimeError(
                f"buffer has {self._size} < slice_len {self.slice_len}; warm up first"
            )
        starts = self._sample_starts(B)
        offsets = torch.arange(self.slice_len, device=self.device)
        idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.capacity  # (B, L)

        obs = self._obs[idx].transpose(0, 1).contiguous()  # (L, B, obs_dim)
        action = self._action[idx].transpose(0, 1)[1:].contiguous()  # (H, B, act_dim)
        reward = self._reward[idx].transpose(0, 1)[1:].contiguous()  # (H, B)
        terminated = self._terminated[idx].transpose(0, 1)[1:].contiguous()
        truncated = self._truncated[idx].transpose(0, 1)[1:].contiguous()
        return obs, action, reward, terminated, truncated

    def _sample_starts(self, batch_size: int) -> torch.Tensor:
        """Uniformly sample `batch_size` valid slice-start indices.

        A start `s` is valid iff the window `s … s+H` (mod capacity) contains no
        episode boundary after position 0 and does not straddle the write seam. Boundary
        crossings are rare (episode length ≫ H), so we oversample candidates, mask, and
        top up until enough survive.
        """
        L, cap = self.slice_len, self.capacity
        offsets = torch.arange(1, L, device=self.device)  # interior positions 1..H
        out = torch.empty(batch_size, dtype=torch.long, device=self.device)
        filled = 0
        while filled < batch_size:
            need = batch_size - filled
            cand = self._candidate_starts(need * 2 + 16)
            interior = (cand.unsqueeze(1) + offsets.unsqueeze(0)) % cap  # (n, H)
            # Reject if a new episode starts inside the window (crosses an episode edge).
            bad = self._first[interior].any(dim=1)
            # When full, reject windows whose interior hits the write head (oldest slot).
            if self.full:
                bad = bad | (interior == self._pos).any(dim=1)
            good = cand[~bad]
            take = min(good.numel(), need)
            if take:
                out[filled : filled + take] = good[:take]
                filled += take
        return out

    def _candidate_starts(self, n: int) -> torch.Tensor:
        """Random occupied indices that can host a full window without wrapping past
        the valid data. Not-full: `[0, size-L]`. Full: any slot (seam handled by mask)."""
        if not self.full:
            high = self._size - self.slice_len + 1
            return torch.randint(0, high, (n,), device=self.device)
        return torch.randint(0, self.capacity, (n,), device=self.device)
