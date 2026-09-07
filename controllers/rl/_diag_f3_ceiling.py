"""F3 PRE-CHECK — can (horizon, rho) reweighting ever give the terminal enough gradient?

E3 measured the reward head predicting kill-death = 8.1 where truth is 352.9 (43x compression) while
dense terms are accurate to 0.05.  The authored F3 box was `horizon 3 -> 9` with `rho 0.5 -> 1.0` as
one joint edit, on the reasoning that the terminal transition is pinned to the LAST horizon position
and therefore carries loss weight rho^(H-1) = 0.25 today.

The pinning claim is right, but the CONCLUSION does not follow, and this script is the arithmetic that
decides it before spending a 2 h run.

Buffer layout (`buffer.py`, "incoming" convention): an episode of T transitions occupies T+1 slots at
base b; slot b+0 is the initial obs (`first=True`, zero reward, always dropped) and slot b+i carries
the reward/terminated that led INTO obs[i].  So the terminal transition's reward sits in slot b+T, the
episode's LAST slot.  `sample()` takes slots s..s+H and then drops position 0, so reward index t reads
slot s+1+t.  `_sample_starts` rejects any window with a `first` flag in slots s+1..s+H.

  terminal in window at index t  =>  s+1+t = b+T  and  b+T+1 not in [s+1, s+H]
                                 =>  t >= H-1, and t <= H-1  =>  t = H-1 EXACTLY.

So the terminal is reachable, always at the far end, in exactly ONE valid window per episode -- while
the episode offers T-H+1 valid windows total.  Its share of the reward head's gradient is therefore

    share(H, rho, T) = rho^(H-1) / [ (T - H + 1) * sum_{t<H} rho^t ]

and the (T-H+1) factor is the one no in-window reweighting can touch.  The in-window factor
rho^(H-1)/sum(rho^t) is bounded above by 1 (rho -> inf), against the incumbent's 0.25/1.75 = 0.1429,
so the ENTIRE (H, rho) family is capped at 7.00x.  E3 needs 43x.

Both halves are checked here: the closed form against a direct enumeration, and the enumeration
against the SHIPPED `ReplayBuffer._sample_starts` by Monte Carlo, so the conclusion does not rest on
my algebra.
"""

from __future__ import annotations

import torch
from astrum_rl.tdmpc2.buffer import ReplayBuffer


class Cfg:
    def __init__(self, horizon, cap=20000, batch_size=4096):
        self.horizon = horizon
        self.buffer_size = cap
        self.obs_dim = 4
        self.action_dim = 2
        self.batch_size = batch_size


def share_closed(H: int, rho: float, T: int) -> float:
    """rho^(H-1) / [(T-H+1) * sum_{t<H} rho^t]."""
    denom_in = sum(rho**t for t in range(H))
    return rho ** (H - 1) / ((T - H + 1) * denom_in)


def share_enumerated(H: int, rho: float, T: int) -> tuple[float, int, list[int]]:
    """Enumerate every valid start for one episode; return the terminal's weight share.

    Mirrors buffer.py exactly: slots 0..T for one episode (slot 0 `first`), slot T+1 `first` (the next
    episode).  Reward index t of window s reads slot s+1+t.  The terminal reward is slot T.
    """
    first = {0, T + 1}
    w_term, w_all, hits = 0.0, 0.0, []
    for s in range(0, T + 1):
        if any((s + o) in first for o in range(1, H + 1)):
            continue
        if s + H > T + 1:
            continue
        for t in range(H):
            w = rho**t / H
            w_all += w
            if s + 1 + t == T:
                w_term += w
                hits.append(t)
    return (w_term / w_all if w_all else 0.0), len(hits), sorted(set(hits))


def share_empirical(H: int, rho: float, T: int, draws: int = 200_000) -> tuple[float, list[int]]:
    """Monte-Carlo the SHIPPED sampler: fill one episode, draw windows, weight the terminal hits."""
    dev = torch.device("cpu")
    buf = ReplayBuffer(Cfg(H), dev)
    obs = torch.zeros((T + 1, 4))
    act = torch.zeros((T, 2))
    rew = torch.arange(T, dtype=torch.float32)
    term = torch.zeros((T,))
    term[-1] = 1.0  # the terminal transition
    buf.add_episode(obs, act, rew, term, torch.zeros((T,)))
    # Slot T+1 must carry a real `first` (the next episode's initial obs), exactly as in training --
    # otherwise the sampler would accept windows running off the end of this episode.  Add ONE such
    # slot, not a whole second episode: a second episode would double the valid-window count while
    # contributing no terminal, and the empirical share would land at half the closed form for a
    # reason that has nothing to do with the sampler.
    buf.add(torch.zeros(4), torch.zeros(2), 0.0, 0.0, 0.0, first=True)

    rho_v = torch.pow(torch.tensor(rho), torch.arange(H, dtype=torch.float32))  # (H,)
    w_term, w_all, idxs = 0.0, 0.0, []
    done = 0
    while done < draws:
        b = min(4096, draws - done)
        _o, _a, _r, tm, _tr = buf.sample(b)  # tm (H, b)
        w_all += float(rho_v.sum()) * b / H
        hit = tm > 0.5
        if hit.any():
            per_t = hit.sum(dim=1).float()  # (H,)
            w_term += float((rho_v * per_t).sum()) / H
            idxs += torch.nonzero(hit.any(dim=1)).flatten().tolist()
        done += b
    return (w_term / w_all if w_all else 0.0), sorted(set(idxs))


ARMS = [
    ("INCUMBENT  H=3  rho=0.5", 3, 0.5),
    ("F3 as authored H=9 rho=1.0", 9, 1.0),
    ("H=9  rho=0.5", 9, 0.5),
    ("H=3  rho=1.0", 3, 1.0),
    ("H=3  rho=2.0", 3, 2.0),
    ("H=3  rho=4.0", 3, 4.0),
    ("H=5  rho=2.0", 5, 2.0),
    ("H=1  (degenerate ceiling)", 1, 1.0),
]
T_EP = 1440  # 60 s at 24 Hz, the chase cell


def main() -> None:
    print("=" * 100)
    print("F3 PRE-CHECK — the terminal transition's share of the REWARD-HEAD gradient")
    print("=" * 100)
    print(f"\nepisode length T = {T_EP} transitions (60 s at 24 Hz)\n")
    print(
        f"  {'arm':<28} {'closed form':>13} {'enumerated':>13} {'reward idx':>11} {'vs incumbent':>13}"
    )
    base = share_closed(3, 0.5, T_EP)
    for name, H, rho in ARMS:
        c = share_closed(H, rho, T_EP)
        e, nhit, idxs = share_enumerated(H, rho, T_EP)
        print(
            f"  {name:<28} {c:>12.3e} {e:>13.3e} {str(idxs):>11} {c / base:>12.2f}x"
            + ("   <-- 1 window/episode" if nhit == 1 else f"   <-- {nhit} windows!")
        )

    print("\n" + "=" * 100)
    print("CROSS-CHECK against the SHIPPED ReplayBuffer._sample_starts (Monte Carlo)")
    print("=" * 100)
    for name, H, rho in ARMS[:4]:
        emp, idxs = share_empirical(H, rho, T_EP)
        c = share_closed(H, rho, T_EP)
        print(
            f"  {name:<28} closed {c:.3e}   empirical {emp:.3e}   "
            f"ratio {emp / c:.3f}   terminal seen at reward idx {idxs}"
        )

    print("\n" + "=" * 100)
    print("THE CEILING — best possible in-window reweighting, at any (H, rho)")
    print("=" * 100)
    inwin_base = 0.5**2 / sum(0.5**t for t in range(3))
    print(f"  incumbent in-window factor  rho^(H-1)/sum rho^t = {inwin_base:.4f}")
    print("  supremum over all (H, rho)                      = 1.0000  (rho -> inf)")
    print(f"  => the ENTIRE (H, rho) family is capped at      = {1.0 / inwin_base:.2f}x")
    print("  E3 needs                                        = 43x")
    print(
        "\n  The (T - H + 1) factor -- one terminal per ~1440 transitions -- is what dominates, and"
    )
    print("  no in-window reweighting can touch it.  A fix has to change the SAMPLING or add an")
    print("  explicit terminal weight, not the horizon.")

    print("\n" + "=" * 100)
    print("WHAT A TERMINAL IMPORTANCE WEIGHT BUYS (H=3, rho=0.5, weight 1 + lam on terminated)")
    print("=" * 100)
    print(f"  {'lambda':>8} {'share':>12} {'vs incumbent':>14}")
    for lam in (0, 10, 100, 300, 1000, 3000, 10000):
        s = base * (1 + lam) / (1 + base * lam)
        print(f"  {lam:>8} {s:>12.3e} {s / base:>13.2f}x")


if __name__ == "__main__":
    main()
