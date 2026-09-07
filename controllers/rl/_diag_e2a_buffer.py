"""E2a — does the replay sampler starve the TERMINAL transition of gradient?

WHY THIS EXISTS
---------------
D5 reported a second value-path defect (the first, the `two_hot_inv` "log-mean
compresses the kill" claim, was refuted by `_diag_e1a_decode.py`): with
`slice_len = H+1 = 4` and `_sample_starts`' rejection of interior `first` flags
(`buffer.py:142-168`), the terminal stored index is allegedly reachable from exactly
ONE slice start, always landing at position `t = 2`, giving it **527 vs 3,594**
rho-weight = **6.82x less** than a mid-episode index.

That number was agent-reported with no committed harness -- the same provenance as the
claim E1a just killed. `grep 6.82` finds nothing in the repo. So: measure it, with a
pre-registered gate, before anything is built on it.

WHAT IS MEASURED
----------------
M0  EXACT ENUMERATION. Every slice start in the ring is enumerated (no RNG), the real
    validity rule is applied, and rho^t is accumulated onto each stored index. Gives the
    exact per-position weight profile and reachability multiplicity.
M1  THE SHIPPED SAMPLER. The real `ReplayBuffer._sample_starts` is called for many
    batches on a real, full 1e6-slot ring; the same histogram is accumulated. M0 is only
    believable if M1 agrees.
M2  DEPTH. `buffer.sample()` is called and `terminated == 1` is located in the returned
    (H, B) tensor. This is the most direct possible test of "only ever at t = 2" -- it
    reads the shipped tensor, not a reimplementation.
M3  DIET. Expected terminal transitions per 256-sample batch, and updates-per-terminal,
    swept over the terminal-episode rate (which the ratio does NOT depend on).
M4  THE TERMINATION HEAD. `termination_loss` is NOT rho-weighted (`agent.py:337-341`) --
    it is a plain BCE mean. So its diet is the positive-class RATE, not the weight. That
    is a separate quantity and is measured separately.
M5  AGGREGATE SHARE. Terminal rho-weight as a fraction of the whole reward/value loss,
    with the counterfactual. A 7x on a term that is 0.02% of the loss is a different
    claim from a 7x on a term that is 5% of it, and "6.82x" alone does not say which.
M6  LENGTH INVARIANCE. The profile is re-measured at episode lengths 20..1440 to
    establish it is a property of the geometry, not of the run's episode-length mix.

NEGATIVE CONTROL (the program's standing method rule: run the control as PART of the
measurement). The episode's FIRST transition is also a boundary index and is also
reachable from exactly one start -- but at `t = 0`, weight 1.0. If both boundary indices
are equally starved, the finding is "boundaries are down-weighted"; if only the terminal
is, the finding is specifically about DEPTH. Measured, not assumed.

PRE-REGISTERED KILL CONDITIONS (fixed before the first run)
-----------------------------------------------------------
K1  Terminal reachable at t = 0 or t = 1 with nonzero probability
    => "only ever at maximum depth" is FALSE.
K2  Measured terminal:interior rho-weight ratio < 2.0
    => the 6.82x is materially wrong and E2b is CLOSED.
K3  M1 (shipped sampler) disagrees with M0 (enumeration) by > 2% on the ratio
    => M0's model of the code is wrong; trust M1 and re-derive.
K4  Terminal share of total reward-loss rho-weight already >= 1%
    => "starvation" is a mis-frame in aggregate; report it as a re-weighting, not a
       missing signal.
K5  Expected terminal transitions per batch >= 1.0 under the CURRENT sampler
    => every update already sees a kill; the frequency half is not binding.

Run (CPU only -- deliberately, so it does not contend with a training run for the GPU):
    docker compose --profile rl run --rm -T -w /workspace/controllers/rl dogfight-train \
        uv run python _diag_e2a_buffer.py
"""

from __future__ import annotations

import numpy as np
import torch
from astrum_rl.tdmpc2.buffer import ReplayBuffer

# --- the real geometry, read from runs/curriculum-96/run.pt's cfg -----------------
H = 3
RHO = 0.5
BATCH = 256
CAP_REAL = 1_000_000
OBS_DIM = 41
ACT_DIM = 5
MEAN_EP_LEN = 692.9  # measured: runs/curriculum-97-white/train.log, windowed over 2474 eps
EP_CAP = 1440  # --max-time-s 60 at 24 Hz
SEED = 20260906

RHO_POW = np.array([RHO**t for t in range(H)], dtype=np.float64)
SLICE_LEN = H + 1
FULL_WEIGHT = float(RHO_POW.sum())  # 1.75 at H=3, rho=0.5


class _Cfg:
    """Minimal cfg surface `ReplayBuffer` touches."""

    def __init__(self, capacity: int):
        self.buffer_size = capacity
        self.horizon = H
        self.batch_size = BATCH
        self.obs_dim = OBS_DIM
        self.action_dim = ACT_DIM


# --------------------------------------------------------------------------------
# layout construction
# --------------------------------------------------------------------------------
def _episode_lengths(rng: np.random.Generator, total: int) -> tuple[np.ndarray, np.ndarray]:
    """Episode lengths whose mean matches the live run's 692.9, plus terminal flags.

    A mixture: a fraction terminate early (kill/death/deck) with a truncated-exponential
    length, the rest run to the `EP_CAP` clock. `p_term` is solved so the mean lands on
    the measured value, which makes the DIET numbers (M3/M4) reflect the real run rather
    than a guess. The WEIGHT ratio (M0/M1) is invariant to this choice -- M6 proves it.
    """
    # mean = p*E[early] + (1-p)*EP_CAP  ->  solve for p given E[early]
    mean_early = 260.0
    p_term = (EP_CAP - MEAN_EP_LEN) / (EP_CAP - mean_early)
    lengths: list[int] = []
    terms: list[bool] = []
    n = 0
    while n < total:
        if rng.random() < p_term:
            t = int(np.clip(rng.exponential(mean_early), 2 * SLICE_LEN, EP_CAP))
            terms.append(True)
        else:
            t = EP_CAP
            terms.append(False)
        lengths.append(t)
        n += t + 1
    return np.array(lengths, dtype=np.int64), np.array(terms, dtype=bool)


def _fill(buf: ReplayBuffer, lengths: np.ndarray, terms: np.ndarray) -> dict[str, np.ndarray]:
    """Write episodes through the REAL `add_episode`, mirroring per-slot metadata.

    Metadata is written to the same ring indices with the same wrap arithmetic, so an
    overwritten slot's metadata is overwritten too -- exactly like the buffer's own data.
    """
    cap = buf.capacity
    meta = {
        "ep_id": np.full(cap, -1, dtype=np.int64),
        "pos": np.full(cap, -1, dtype=np.int64),
        "ep_T": np.full(cap, -1, dtype=np.int64),
        "is_term": np.zeros(cap, dtype=bool),
        "ep_terminates": np.zeros(cap, dtype=bool),
    }
    zeros_obs_cache: dict[int, np.ndarray] = {}
    for ep_id, (t, is_term_ep) in enumerate(zip(lengths.tolist(), terms.tolist())):
        n = t + 1
        if n not in zeros_obs_cache:
            zeros_obs_cache[n] = np.zeros((n, OBS_DIM), dtype=np.float32)
        obs = zeros_obs_cache[n]
        act = np.zeros((t, ACT_DIM), dtype=np.float32)
        rew = np.full((t,), 0.1, dtype=np.float32)
        term = np.zeros((t,), dtype=np.float32)
        trunc = np.zeros((t,), dtype=np.float32)
        if is_term_ep:
            rew[-1] = 150.0  # R_KILL
            term[-1] = 1.0
        else:
            trunc[-1] = 1.0

        pos0 = buf._pos
        buf.add_episode(obs, act, rew, term, trunc)
        idx = (pos0 + np.arange(n)) % cap
        meta["ep_id"][idx] = ep_id
        meta["pos"][idx] = np.arange(n)
        meta["ep_T"][idx] = t
        meta["ep_terminates"][idx] = is_term_ep
        meta["is_term"][idx] = False
        if is_term_ep:
            meta["is_term"][idx[t]] = True
    return meta


# --------------------------------------------------------------------------------
# M0 -- exact enumeration of every valid start
# --------------------------------------------------------------------------------
def _valid_starts_exact(buf: ReplayBuffer) -> np.ndarray:
    """Every start the shipped rule accepts. Mirrors `_sample_starts` exactly:
    reject if any INTERIOR slot (s+1 .. s+H) carries `first`, and when the ring is full
    also reject if an interior slot is the write head."""
    cap = buf.capacity
    first = buf._first.cpu().numpy()
    if buf.full:
        cand = np.arange(cap, dtype=np.int64)
    else:
        cand = np.arange(0, buf._size - SLICE_LEN + 1, dtype=np.int64)
    interior = (cand[:, None] + np.arange(1, SLICE_LEN)[None, :]) % cap  # (n, H)
    bad = first[interior].any(axis=1)
    if buf.full:
        bad |= (interior == buf._pos).any(axis=1)
    return cand[~bad]


def _accumulate(buf: ReplayBuffer, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rho-weight and reach-count per stored index, from a set of starts.

    Counts first, then scales by rho^t -- a `weights=` bincount would need a float64
    array the size of `starts`, which is 400 MB at the draw counts M1 needs.
    """
    cap = buf.capacity
    weight = np.zeros(cap, dtype=np.float64)
    reach = np.zeros(cap, dtype=np.int64)
    for t in range(H):
        idx = (starts + 1 + t) % cap
        c = np.bincount(idx, minlength=cap)
        weight += RHO_POW[t] * c
        reach += c
    return weight, reach


def _depth_hist(buf: ReplayBuffer, starts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """How many times indices in `mask` are hit at each position t = 0..H-1."""
    cap = buf.capacity
    out = np.zeros(H, dtype=np.int64)
    for t in range(H):
        idx = (starts + 1 + t) % cap
        out[t] = int(mask[idx].sum())
    return out


# --------------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------------
def _stats(v: np.ndarray) -> str:
    if v.size == 0:
        return "n=0"
    return (
        f"n={v.size:>9,d}  mean={v.mean():7.4f}  p10={np.percentile(v, 10):7.4f}  "
        f"p50={np.percentile(v, 50):7.4f}  p90={np.percentile(v, 90):7.4f}  "
        f"min={v.min():7.4f}  max={v.max():7.4f}"
    )


def _line(ch: str = "-", n: int = 92) -> None:
    print(ch * n)


# --------------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cpu")  # deliberate: leave the GPU to the training run

    print()
    _line("=")
    print("E2a  REPLAY-SAMPLER TERMINAL STARVATION AUDIT")
    _line("=")
    print(f"   geometry: H={H}  slice_len={SLICE_LEN}  rho={RHO}  batch={BATCH}  cap={CAP_REAL:,}")
    print(f"   rho^t = {RHO_POW.tolist()}   full interior weight = {FULL_WEIGHT}")
    print(f"   episode mix: mean {MEAN_EP_LEN} steps (measured), clock cap {EP_CAP}")
    print("   PRE-REGISTERED KILLS: K1 t<2 reachable | K2 ratio<2.0 | K3 M0!=M1 by >2%")
    print("                         K4 share>=1% | K5 terminals/batch>=1.0")
    _line("=")

    # ---------------- build a real, FULL ring -----------------------------------
    cfg = _Cfg(CAP_REAL)
    buf = ReplayBuffer(cfg, dev)
    lengths, terms = _episode_lengths(rng, int(CAP_REAL * 1.35))
    meta = _fill(buf, lengths, terms)
    print()
    print(
        f"   filled: {len(lengths):,} episodes  size={len(buf):,}  full={buf.full}  "
        f"write_head={buf._pos:,}"
    )
    print(
        f"   episode lengths: mean={lengths.mean():.1f}  terminating={terms.mean() * 100:.1f}%  "
        f"(the ring holds the most recent {int((meta['ep_id'] >= 0).sum()):,} slots)"
    )

    live = meta["ep_id"] >= 0
    pos, ep_T = meta["pos"], meta["ep_T"]
    is_term = meta["is_term"]
    # index classes (only inside episodes long enough to have a clean interior)
    long_ep = live & (ep_T >= 4 * SLICE_LEN)
    cls = {
        "first transition  (pos 1)": long_ep & (pos == 1),
        "pos 2": long_ep & (pos == 2),
        "pos 3": long_ep & (pos == 3),
        "interior (4..T-4)": long_ep & (pos >= 4) & (pos <= ep_T - 4),
        "pos T-3": long_ep & (pos == ep_T - 3),
        "pos T-2": long_ep & (pos == ep_T - 2),
        "pos T-1": long_ep & (pos == ep_T - 1),
        "LAST transition (pos T)": long_ep & (pos == ep_T),
    }

    # ---------------- M0 ---------------------------------------------------------
    starts = _valid_starts_exact(buf)
    w0, r0 = _accumulate(buf, starts)
    print()
    _line()
    print("   M0  EXACT ENUMERATION -- rho-weight and reach-count by position in episode")
    _line()
    print(
        f"   valid starts: {starts.size:,} of {CAP_REAL:,} slots "
        f"({100.0 * starts.size / CAP_REAL:.2f}%)"
    )
    print()
    print(
        f"   {'position class':<28} {'n':>9}  {'rho-weight':>11}  {'reach':>7}  {'depths t=':>10}"
    )
    prof: dict[str, float] = {}
    for name, m in cls.items():
        if m.sum() == 0:
            continue
        wv, rv = w0[m], r0[m]
        prof[name] = float(wv.mean())
        dh = _depth_hist(buf, starts, m)
        dstr = "/".join(str(int(x)) for x in dh)
        print(
            f"   {name:<28} {int(m.sum()):>9,d}  {wv.mean():>11.4f}  {rv.mean():>7.3f}  {dstr:>10}"
        )

    interior_w = prof.get("interior (4..T-4)", float("nan"))
    term_w = prof.get("LAST transition (pos T)", float("nan"))
    first_w = prof.get("first transition  (pos 1)", float("nan"))
    ratio0 = interior_w / term_w if term_w else float("inf")

    print()
    print(
        f"   >> interior : terminal  rho-weight ratio = {interior_w:.4f} / {term_w:.4f} "
        f"= {ratio0:.3f}x"
    )
    print(
        f"   >> NEGATIVE CONTROL, the other boundary index: first transition weight "
        f"{first_w:.4f} (ratio {interior_w / first_w if first_w else float('nan'):.3f}x)"
    )

    # depth histogram, terminals only
    term_mask = live & is_term
    dh_term = _depth_hist(buf, starts, term_mask)
    tot_t = int(dh_term.sum())
    print()
    print(f"   terminal indices in ring: {int(term_mask.sum()):,}")
    print(
        f"   terminal hits by depth t: t=0 {dh_term[0]:,}   t=1 {dh_term[1]:,}   "
        f"t=2 {dh_term[2]:,}   (total {tot_t:,})"
    )
    if tot_t:
        print(
            "   P(t | slice contains terminal) = "
            + "  ".join(f"t={t}:{dh_term[t] / tot_t:.4f}" for t in range(H))
        )
    dh_first = _depth_hist(buf, starts, live & (pos == 1) & long_ep)
    print(
        f"   first-transition hits by depth: t=0 {dh_first[0]:,}   t=1 {dh_first[1]:,}   "
        f"t=2 {dh_first[2]:,}"
    )

    # ---------------- M1 -- the shipped sampler ---------------------------------
    print()
    _line()
    print("   M1  THE SHIPPED SAMPLER  (`_sample_starts`, real code path)")
    _line()
    # POWER. A boundary class has only ~n_ep members, each reachable from `reach` starts,
    # so its Monte-Carlo SD is set by how many times each individual start gets drawn.
    # With lambda = draws per valid start, SD/mean on a reach-1 class is
    # 1/sqrt(lambda * n_members). The first pass used lambda = 0.10 -> ~8% SD, which
    # cannot be tested against a 2% threshold. Target lambda = TARGET_LAMBDA.
    TARGET_LAMBDA = 50
    total_draws = int(TARGET_LAMBDA * starts.size)
    chunk = 500_000
    w1 = np.zeros(CAP_REAL, dtype=np.float64)
    start_counts = np.zeros(CAP_REAL, dtype=np.int64)
    got = 0
    while got < total_draws:
        n = min(chunk, total_draws - got)
        d = buf._sample_starts(n).cpu().numpy()
        wc, _rc = _accumulate(buf, d)
        w1 += wc
        start_counts += np.bincount(d, minlength=CAP_REAL)
        got += n
    scale = starts.size / got  # sampling is with replacement; rescale to one-per-start
    lam = got / starts.size
    print(f"   drew {got:,} starts  =>  lambda = {lam:.1f} draws per valid start")
    print(
        f"   {'position class':<28} {'M0 weight':>11}  {'M1 weight':>11}  {'rel err':>9}  {'z':>7}"
    )
    per_cls = {}
    for name, m in cls.items():
        if m.sum() == 0:
            continue
        b = float(w1[m].mean() * scale)
        per_cls[name] = b
        a = prof[name]
        # analytic SD of the estimator: per member, weight = sum_t rho^t * K_t,
        # K_t ~ Poisson(lam); Var = lam * sum_t rho^(2t) over the reachable t's.
        reach_n = int(round(float(r0[m].mean())))
        var_one = lam * float(sum(RHO_POW[t] ** 2 for t in range(reach_n)))
        sd = np.sqrt(var_one / int(m.sum())) / lam  # in one-per-start units
        z = (b - a) / max(sd, 1e-12)
        print(
            f"   {name:<28} {a:>11.4f}  {b:>11.4f}  "
            f"{abs(b - a) / max(abs(a), 1e-12):>8.2%}  {z:>7.2f}"
        )
    ratio1 = (
        per_cls["interior (4..T-4)"] / per_cls["LAST transition (pos T)"]
        if per_cls.get("LAST transition (pos T)")
        else float("inf")
    )
    print()
    print(
        f"   >> M1 ratio = {ratio1:.3f}x   vs M0 {ratio0:.3f}x   "
        f"rel disagreement {abs(ratio1 - ratio0) / ratio0:.2%}"
    )

    # M1b -- UNIFORMITY of the sampler itself, at the level of STARTS (not stored
    # indices). If `_sample_starts` were biased against starts near an episode edge, the
    # per-index weights would be biased even though the arithmetic is right. This isolates
    # the two. `good[:take]` keeps candidate order, so a prefix of iid-uniform draws
    # should still be uniform -- measured, not assumed.
    print()
    print("   M1b UNIFORMITY of `_sample_starts` over valid starts (draws per start)")
    sc_valid = start_counts[starts]
    spos, sT = pos[starts], ep_T[starts]
    groups = {
        "start at pos 0 (the `first` slot)": spos == 0,
        "start at pos 1..3 (episode head)": (spos >= 1) & (spos <= 3),
        "start interior (4..T-6)": (spos >= 4) & (spos <= sT - 6),
        "start at T-5..T-3 (episode tail)": (spos >= sT - 5) & (spos <= sT - 3),
        "start hosting the terminal (T-3)": (spos == sT - 3),
    }
    print(f"   {'start class':<36} {'n':>9}  {'mean draws':>11}  {'vs lambda':>10}  {'z':>7}")
    for name, gm in groups.items():
        if gm.sum() == 0:
            continue
        v = sc_valid[gm]
        sd = np.sqrt(lam / v.size)
        print(
            f"   {name:<36} {v.size:>9,d}  {v.mean():>11.3f}  "
            f"{v.mean() / lam:>9.4f}x  {(v.mean() - lam) / max(sd, 1e-12):>7.2f}"
        )
    print(f"   invalid starts drawn (must be 0): {int(start_counts.sum() - sc_valid.sum()):,}")

    # ---------------- M2 -- read the shipped tensor -----------------------------
    print()
    _line()
    print("   M2  DEPTH, READ OFF `buffer.sample()`  (the shipped tensor, not a model)")
    _line()
    n_batches = 20_000
    depth_counts = np.zeros(H, dtype=np.int64)
    n_slices = 0
    n_term_slices = 0
    for _ in range(n_batches):
        _, _, _, terminated, _ = buf.sample(BATCH)
        tn = terminated.cpu().numpy()  # (H, B)
        depth_counts += (tn > 0.5).sum(axis=1).astype(np.int64)
        n_slices += tn.shape[1]
        n_term_slices += int((tn > 0.5).any(axis=0).sum())
    tot = int(depth_counts.sum())
    print(f"   sampled {n_batches:,} batches x {BATCH} = {n_slices:,} slices")
    print(
        f"   `terminated == 1` occurrences by position t: "
        f"t=0 {depth_counts[0]:,}   t=1 {depth_counts[1]:,}   t=2 {depth_counts[2]:,}"
    )
    if tot:
        print(
            "   P(t | terminal present) = "
            + "  ".join(f"t={t}:{depth_counts[t] / tot:.5f}" for t in range(H))
        )
    print(
        f"   slices containing a terminal: {n_term_slices:,} / {n_slices:,} "
        f"= {n_term_slices / n_slices:.6f}"
    )
    print(
        f"   => terminals per {BATCH}-slice batch = {BATCH * n_term_slices / n_slices:.4f}   "
        f"=> 1 terminal every {n_slices / max(n_term_slices, 1):.1f} slices "
        f"= every {n_slices / max(n_term_slices, 1) / BATCH:.2f} updates"
    )
    # The POWERED M0-vs-shipped-code check. This pools all terminal indices instead of
    # asking about one thin position class, so its noise floor is set by the total count
    # (Poisson) rather than by per-start draw variance -- which is why it, not M1's
    # per-class table, is the test that can actually resolve a 2% disagreement.
    exp_rate = float(_depth_hist(buf, starts, term_mask).sum()) / starts.size
    obs_rate = n_term_slices / n_slices
    pois_sd = np.sqrt(max(n_term_slices, 1)) / n_slices
    print()
    print(
        f"   >> POWERED CHECK  P(slice holds a terminal): enumerated {exp_rate:.6e}  "
        f"shipped {obs_rate:.6e}"
    )
    print(
        f"      disagreement {abs(obs_rate - exp_rate) / exp_rate:.2%}   "
        f"Poisson SD {pois_sd / exp_rate:.2%}   z = {(obs_rate - exp_rate) / pois_sd:.2f}"
    )

    # ---------------- M3 -- diet vs terminal rate -------------------------------
    print()
    _line()
    print("   M3  DIET vs terminal-episode rate  (ratio is rate-independent; diet is not)")
    _line()
    n_valid = starts.size
    live_slots = int(live.sum())
    print(f"   {'p(term ep)':>11}  {'terminals/batch':>16}  {'updates per terminal':>21}")
    for p in (0.05, 0.1, 0.2, 0.3, float(terms.mean()), 0.5):
        # one terminal index per terminating episode; each is in exactly `reach` slices
        reach_t = float(r0[term_mask].mean()) if term_mask.sum() else 0.0
        n_eps_in_ring = live_slots / (MEAN_EP_LEN + 1.0)
        per_slice = p * n_eps_in_ring * reach_t / max(n_valid, 1)
        per_batch = BATCH * per_slice
        print(f"   {p:>11.3f}  {per_batch:>16.4f}  {1.0 / max(per_batch, 1e-12):>21.1f}")

    # ---------------- M4 -- the termination head --------------------------------
    print()
    _line()
    print("   M4  THE TERMINATION HEAD  (`agent.py:337-341` -- BCE mean, NOT rho-weighted)")
    _line()
    pos_rate = tot / max(n_slices * H, 1)
    print(
        f"   positive-class rate in the (H,B) `terminated` tensor = {pos_rate:.3e} "
        f"({tot:,} positives in {n_slices * H:,} elements)"
    )
    print(f"   1 positive per {1.0 / max(pos_rate, 1e-12):,.0f} BCE elements")
    print(
        "   counterfactual, terminal reachable at all H positions (E2b): positive rate "
        f"x{H} = {pos_rate * H:.3e}"
    )
    print("   NOTE: the termination loss is NOT rho-weighted, so for THIS head the defect")
    print("         is purely the reachability COUNT (1 vs H), not the depth.")

    # ---------------- M5 -- aggregate share -------------------------------------
    print()
    _line()
    print("   M5  AGGREGATE SHARE of the reward/value loss")
    _line()
    tot_w = float(w0[live].sum())
    term_tot_w = float(w0[term_mask].sum())
    share = term_tot_w / max(tot_w, 1e-12)
    cf_share = (float(term_mask.sum()) * FULL_WEIGHT) / max(tot_w, 1e-12)
    print(f"   total rho-weight over the ring   = {tot_w:,.1f}")
    print(f"   rho-weight on terminal indices   = {term_tot_w:,.1f}   share = {share:.3%}")
    print(
        f"   counterfactual if treated as interior ({FULL_WEIGHT}) = "
        f"{float(term_mask.sum()) * FULL_WEIGHT:,.1f}   share = {cf_share:.3%}"
    )
    print(
        f"   the fix therefore moves the terminal from {share:.3%} to {cf_share:.3%} "
        f"of the loss ({cf_share / max(share, 1e-12):.2f}x)"
    )

    # ---------------- M6 -- length invariance -----------------------------------
    print()
    _line()
    print("   M6  LENGTH INVARIANCE  (is the profile a property of the geometry?)")
    _line()
    print(
        f"   {'ep len T':>9}  {'interior w':>11}  {'terminal w':>11}  {'ratio':>8}  "
        f"{'first w':>9}  {'term depths':>12}"
    )
    for T in (20, 50, 100, 200, 693, 1440):
        cap_s = max(60_000, 8 * (T + 1))
        cfg_s = _Cfg(cap_s)
        bs = ReplayBuffer(cfg_s, dev)
        n_ep = cap_s // (T + 1) + 3
        ls = np.full(n_ep, T, dtype=np.int64)
        ts = np.ones(n_ep, dtype=bool)
        ms = _fill(bs, ls, ts)
        st = _valid_starts_exact(bs)
        ws, _rs = _accumulate(bs, st)
        lv = ms["ep_id"] >= 0
        le = lv & (ms["ep_T"] >= 4 * SLICE_LEN)
        m_int = le & (ms["pos"] >= 4) & (ms["pos"] <= ms["ep_T"] - 4)
        m_ter = lv & ms["is_term"]
        m_fst = le & (ms["pos"] == 1)
        iw = float(ws[m_int].mean()) if m_int.sum() else float("nan")
        tw = float(ws[m_ter].mean()) if m_ter.sum() else float("nan")
        fw = float(ws[m_fst].mean()) if m_fst.sum() else float("nan")
        dh = _depth_hist(bs, st, m_ter)
        print(
            f"   {T:>9}  {iw:>11.4f}  {tw:>11.4f}  {iw / tw if tw else float('nan'):>8.3f}  "
            f"{fw:>9.4f}  {'/'.join(str(int(x)) for x in dh):>12}"
        )

    # ---------------- verdict ---------------------------------------------------
    print()
    _line("=")
    print("   PRE-REGISTERED GATE")
    _line("=")
    k1 = bool(dh_term[0] > 0 or dh_term[1] > 0) or bool(depth_counts[0] > 0 or depth_counts[1] > 0)
    k2 = bool(ratio0 < 2.0)
    k3 = bool(abs(ratio1 - ratio0) / max(ratio0, 1e-12) > 0.02)
    k3_powered = bool(abs(obs_rate - exp_rate) / max(exp_rate, 1e-12) > 0.02)
    k4 = bool(share >= 0.01)
    k5 = bool(BATCH * n_term_slices / n_slices >= 1.0)
    for name, fired, note in (
        (
            "K1 terminal reachable at t<2",
            k1,
            f"depths {dh_term.tolist()} / {depth_counts.tolist()}",
        ),
        ("K2 ratio < 2.0", k2, f"ratio {ratio0:.3f}x"),
        ("K3 M0 vs M1 disagree > 2%", k3, f"{abs(ratio1 - ratio0) / max(ratio0, 1e-12):.2%}"),
        (
            "K3' POWERED, M0 vs sample() > 2%",
            k3_powered,
            f"{abs(obs_rate - exp_rate) / max(exp_rate, 1e-12):.2%} "
            f"(z={(obs_rate - exp_rate) / max(pois_sd, 1e-12):.2f})",
        ),
        ("K4 share >= 1%", k4, f"share {share:.3%}"),
        ("K5 >= 1 terminal per batch", k5, f"{BATCH * n_term_slices / n_slices:.4f}/batch"),
    ):
        print(f"   [{'X' if fired else ' '}] {name:<32} {note}")
    print()
    print("   K3 NOTE: K3 as pre-registered compares two per-class MEANS over ~1.4k-member")
    print("   boundary classes; its own noise floor is 1/sqrt(lambda*n) per class. K3' is the")
    print("   same question asked where the count is large (all terminals pooled, read off the")
    print("   shipped `sample()` tensor) and is the one that can resolve 2%. M1b tests sampler")
    print("   uniformity directly, so a K3 firing with K3' quiet and M1b flat = estimator noise.")
    print()
    print("   DESIGN NOTE for E2b, from M0's depth histogram: the terminal can only reach")
    print("   t = 0 from start s = T-1, whose window spans slots T-1, T, T+1, T+2 -- i.e. it")
    print("   STRADDLES the episode boundary. So 'allow the terminal at t=0' is NOT a one-line")
    print("   relaxation of the rejection rule: positions after the terminal would train on")
    print("   the next episode's transitions. Any E2b must add a post-terminal loss MASK, or")
    print("   re-weight terminal-bearing slices instead of relocating them.")
    _line("=")
    print()


if __name__ == "__main__":
    main()
