"""D5 ADVERSARIAL VERIFICATION — is the measured Q-vs-return gap a CREDIT-REACH defect?

    cd controllers/rl && uv run python _diag_d5_verify.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt \
        --episodes 6 --jitter 0.5 --seed 0

D5 (`_diag_credit_reach.py`, `_d5_credit_reach_out.txt`) measured, on c96 greedy asp30 kills:
`med r_T = +152.32`, `med Q_T = -28.18`, `med td1_T = +180.61`, and a reward head that recovers
"1.2%" of the terminal spike — and its 3-6s x kill wrong-way cell came back `gap p50 -128.09`,
`P(Q<G) 1.00`. This harness does NOT re-measure the gap. It tests the three things that decide what
that gap MEANS, none of which D5 separated.

WHAT IS MEASURED (all on the SAME greedy trajectories, same seeds as D5: seed0 = seed + 1000*aspect)

  1. **Decoder compression.** `Q`/`reward` heads are categorical over 101 symlog-spaced bins, and
     `two_hot_inv` (`functional.py:53-67`) returns `symexp( sum_i p_i * bin_i )` — an expectation
     taken in SYMLOG space and then exponentiated. That is a log-mean: for a head whose posterior is
     a MIXTURE of a rare large spike and a bulk near zero, the decoded scalar is pulled toward the
     bulk by construction. So the same logits are decoded three ways at the terminal transition:
       * `Qdec`   = `symexp(E_p[bin])`  — what the shipped code returns and D5 quoted;
       * `Qlin`   = `E_p[symexp(bin)]`  — the honest linear-space posterior mean;
       * `Qargmax`= `symexp(bin[argmax p])` — the modal prediction;
       plus `P(bin >= symlog(50))`, the posterior mass the head puts on a >=50-unit outcome.
     If `Qlin` (or the high-bin mass) recovers the spike while `Qdec` does not, "the critic has not
     fitted the terminal reward" is a statement about the DECODER, not about credit propagation.
     If `Qlin` misses it too, the head genuinely never learned the spike and the decoder is exonerated.

  2. **Evaluation-point mismatch.** D5 reads `Q(encode(o_{T-1}), a_{T-1})`. Training never fits that
     transition at an encoded latent: with `slice_len = H+1 = 4` and the `_first` rejection in
     `buffer.py:_sample_starts`, the terminal stored index T is reachable from exactly ONE slice start
     (`s = T-3`), where it lands at horizon position `t = 2`, whose latent is a TWO-STEP learned
     rollout `next(next(encode(o_{T-3}), a_{T-3}), a_{T-2})` (`agent.py:317-322`, `_zs = zs[:-1]`).
     Both heads are therefore re-evaluated at that exact training latent and printed beside the
     encoder-latent reading. A spike that appears at the rollout latent and not at the encoder latent
     means D5 measured off-distribution.

  3. **Sign symmetry across terminal types (the negative control D5 already had and did not use).**
     A credit-REACH failure and a magnitude-COMPRESSION failure make the same prediction for a
     positive terminal spike, and OPPOSITE predictions for the level of `td1` at the terminal itself,
     where `terminated = 1` makes the target exactly `r_T` with NO bootstrap (`agent.py:298-304`).
     So the c95 asp55 DECK episode (`r_T = -199.91`) is measured in the same invocation with the same
     code path. Compression predicts a sign-flipped residual of comparable relative size; credit
     reach predicts nothing at all here, because there is no bootstrap to reach through.

  4. **Buffer arithmetic, executed rather than argued.** A real `ReplayBuffer` is loaded with two
     synthetic episodes and `_sample_starts` is drawn 200k times; the histogram of (stored index x
     horizon position) is printed for the last four indices of an episode. This turns "the terminal
     transition is only ever sampled at t=2" into a measured occupancy, and multiplies it by
     `rho^t` to give the terminal transition's total loss weight relative to an interior one.

WHAT IT EXPLICITLY DOES NOT MEASURE

* It is NOT BFM evidence and contains no behaviour claim. Reward-term sums and discounted-return
  traces stay blacklisted (`bfm_detect.py:37-48`); no return trace is printed here at all.
* It does not re-derive D5's calibration curve, and it cannot adjudicate D5's arithmetic — a sibling
  agent has that. It reproduces exactly three scalars (`r_T`, `Q_T`, `td1_T`) as a seed check.
* It cannot see the actual replay buffer (not saved in the checkpoint), so read 4 is the SAMPLING
  arithmetic of the shipped buffer class, not the empirical composition of c96's buffer. It says how
  much gradient weight a terminal transition can receive, not how many there were.
* `Qlin` is not what any trained code path optimizes. It is a diagnostic on the head's posterior; the
  actor maximizes `Qdec` (`agent.py:387` -> `two_hot_inv`), so if the two disagree, `Qdec` is still
  what drives behaviour. That is the point, not a caveat.

PRE-REGISTERED INTERPRETATION

* If `td1_T` is large while `|td1|` at every other time-to-terminal is ~0 (which D5 already reports:
  p50 0.002 on 739 non-terminal steps vs +180.61 at the terminal), then the one-step Bellman operator
  is converged everywhere a bootstrap exists and the entire gap is localized to the ONE transition
  whose target contains no bootstrap. Bootstrap length CANNOT be the mechanism, and D5's "wrong-way
  outcome" must NOT be read as reopening C-6. The kill cell and the deck cell disagreeing only in
  SIGN would confirm this independently.
* If `Qlin`/high-bin mass recovers the spike: the finding is "the symlog two-hot DECODER compresses
  rare terminal spikes", a representation defect with a different fix list than horizon.
* If neither recovers it and the rollout-latent reading matches the encoder-latent one: the finding is
  "the value/reward regression never fit the terminal spike at all", and the 7x-down-weighted single
  sampling window from read 4 is the leading candidate mechanism — still not bootstrap length.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2 import functional as fn
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.buffer import ReplayBuffer
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

STEP_S = 5.0 / 120.0
SPIKE_REF = 50.0  # "large outcome" threshold for the posterior-mass read, in reward units


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial verification of D5's conclusion.")
    p.add_argument("--checkpoints", required=True, help="c96 first (RUN), c95 after (CONTROL)")
    p.add_argument("--aspects", default="30", help="per-checkpoint aspect list, ';'-separated")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--eps-only", default="", help="';'-separated per-cell episode index filters")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _decode3(logits: torch.Tensor, cfg) -> dict:
    """Three decodings of ONE categorical head output plus its high-outcome posterior mass."""
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, dtype=logits.dtype)
    p = torch.softmax(logits, dim=-1)
    dec = float(fn.symexp((p * bins).sum(-1)))
    lin = float((p * fn.symexp(bins)).sum(-1))
    am = int(torch.argmax(p, dim=-1))
    hi = float(p[..., bins >= math.log1p(SPIKE_REF)].sum(-1))
    lo = float(p[..., bins <= -math.log1p(SPIKE_REF)].sum(-1))
    return {
        "dec": dec,
        "lin": lin,
        "argmax": float(fn.symexp(bins[am])),
        "p_hi": hi,
        "p_lo": lo,
        "p_max": float(p[..., am]),
    }


@torch.no_grad()
def _roll_one(agent, env, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    obs_hist = [np.asarray(obs, dtype=np.float32)]
    acts, rews = [], []
    event = None
    terminated = truncated = False
    while True:
        a_t = agent.act(obs, t0=t0, eval_mode=True)
        t0 = False
        a = a_t.numpy()
        acts.append(np.asarray(a, dtype=np.float32))
        obs, r, terminated, truncated, info = env.step(a)
        obs_hist.append(np.asarray(obs, dtype=np.float32))
        rews.append(float(r))
        if info.get("event"):
            event = info["event"]
        if terminated or truncated:
            break
    return {
        "obs": np.stack(obs_hist),
        "act": np.stack(acts),
        "rew": np.asarray(rews, dtype=np.float64),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "outcome": _classify(event, terminated, truncated),
    }


@torch.no_grad()
def _report_terminal(agent, ep: dict, indent: str = "     ") -> None:
    """Everything about the ONE transition whose TD target has no bootstrap."""
    M, cfg = agent.model, agent.cfg
    OB = torch.as_tensor(ep["obs"], dtype=torch.float32, device=agent.device)
    A = torch.as_tensor(ep["act"], dtype=torch.float32, device=agent.device)
    T = A.shape[0]
    r_T = float(ep["rew"][-1])

    z_enc = M.encode(OB[T - 1 : T])  # encoder latent of o_{T-1}: what D5 read
    a_T = A[T - 1 : T]
    q_enc = float(M.Q(z_enc, a_T, return_type="avg").reshape(-1)[0])
    # td1 at the terminal: terminated = 1 => target is exactly r_T, no bootstrap.
    td1_T = r_T - q_enc

    # The latent training ACTUALLY fits this transition at: 2-step learned rollout from o_{T-3}.
    if T >= 3:
        z0 = M.encode(OB[T - 3 : T - 2])
        z1 = M.next(z0, A[T - 3 : T - 2])
        z_roll = M.next(z1, A[T - 2 : T - 1])
    else:
        z_roll = z_enc
    q_roll = float(M.Q(z_roll, a_T, return_type="avg").reshape(-1)[0])
    dz = float(torch.linalg.vector_norm(z_roll - z_enc))

    print(
        f"{indent}terminal transition (t=T-1, terminated=1 => target IS r_T): r_T {r_T:+.2f} "
        f"Qdec(enc) {q_enc:+.2f} td1_T {td1_T:+.2f} ({td1_T / r_T * 100.0:.0f}% of r_T)"
    )
    for name, z in (
        ("encoder z(o_T-1)  [what D5 read]", z_enc),
        ("2-step rollout z  [what training fits]", z_roll),
    ):
        ql = M.Q(z, a_T, return_type="all")  # (num_q, 1, bins)
        rl = M.reward(z, a_T)  # (1, bins)
        qd = [_decode3(ql[k, 0], cfg) for k in range(ql.shape[0])]
        rd = _decode3(rl[0], cfg)
        print(
            f"{indent}  {name}: Q per head dec [" + " ".join(f"{d['dec']:+.1f}" for d in qd) + "]"
            f" -> avg {np.mean([d['dec'] for d in qd]):+.2f}"
        )
        print(
            f"{indent}     Q  Qdec=symexp(E[bin]) {np.mean([d['dec'] for d in qd]):+9.2f} | "
            f"Qlin=E[symexp(bin)] {np.mean([d['lin'] for d in qd]):+9.2f} | "
            f"Qargmax {np.mean([d['argmax'] for d in qd]):+9.2f} | "
            f"P(bin>=symlog{SPIKE_REF:.0f}) {np.mean([d['p_hi'] for d in qd]):.4f} "
            f"P(bin<=-symlog{SPIKE_REF:.0f}) {np.mean([d['p_lo'] for d in qd]):.4f} "
            f"p_max {np.mean([d['p_max'] for d in qd]):.3f}"
        )
        print(
            f"{indent}     r  rdec {rd['dec']:+9.2f} | rlin {rd['lin']:+9.2f} | "
            f"rargmax {rd['argmax']:+9.2f} | P(hi) {rd['p_hi']:.4f} P(lo) {rd['p_lo']:.4f} "
            f"p_max {rd['p_max']:.3f}   (r_T {r_T:+.2f})"
        )
    print(
        f"{indent}  ||z_roll - z_enc|| = {dz:.4f}   Qdec(roll) - Qdec(enc) = {q_roll - q_enc:+.2f}"
    )

    # Is a value of this magnitude even representable? (bin range check, not a fit check.)
    print(
        f"{indent}  representability: symlog(|r_T|) = {math.log1p(abs(r_T)):.3f} vs bin range "
        f"[{cfg.vmin}, {cfg.vmax}] => {'INSIDE' if math.log1p(abs(r_T)) <= cfg.vmax else 'CLIPPED'}"
        f"; top bin decodes to {float(fn.symexp(torch.tensor(cfg.vmax))):.3g}"
    )


@torch.no_grad()
def _report_nonterminal_td1(agent, ep: dict, indent: str = "     ") -> None:
    """`|td1|` on every step EXCEPT the terminal — the localisation check, one line."""
    M = agent.model
    OB = torch.as_tensor(ep["obs"], dtype=torch.float32, device=agent.device)
    A = torch.as_tensor(ep["act"], dtype=torch.float32, device=agent.device)
    r = ep["rew"]
    T = A.shape[0]
    z_all = M.encode(OB)
    z, z_next = z_all[:-1], z_all[1:]
    q = M.Q(z, A, return_type="avg").reshape(-1).cpu().numpy().astype(np.float64)
    qt = (
        M.Q(z_next, M.pi(z_next)[0], return_type="avg", target=True)
        .reshape(-1)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    tm = np.zeros(T)
    tm[-1] = 1.0 if ep["terminated"] else 0.0
    td1 = r + agent.discount * (1.0 - tm) * qt - q
    body = np.abs(td1[:-1])
    print(
        f"{indent}|td1| off the terminal: n={body.size} p50 {np.percentile(body, 50):.3f} "
        f"p90 {np.percentile(body, 90):.3f} max {body.max():.3f} | at the terminal "
        f"{abs(td1[-1]):.2f} => ratio {abs(td1[-1]) / max(np.percentile(body, 50), 1e-9):.0f}x the "
        f"median non-terminal residual"
    )


def _buffer_occupancy(cfg, device) -> None:
    """Execute `_sample_starts` on synthetic episodes: where can a terminal transition land?"""
    c = TDMPC2Config(**{**vars(cfg), "buffer_size": 4096, "batch_size": 4096})
    buf = ReplayBuffer(c, device)
    H = c.horizon
    ep_len = 100
    n_eps = 6
    for _ in range(n_eps):
        obs = np.zeros((ep_len + 1, c.obs_dim), np.float32)
        act = np.zeros((ep_len, c.action_dim), np.float32)
        rew = np.zeros((ep_len,), np.float32)
        term = np.zeros((ep_len,), np.float32)
        term[-1] = 1.0
        buf.add_episode(obs, act, rew, term, np.zeros((ep_len,), np.float32))
    draws = 50
    counts = np.zeros((ep_len + 1, H), dtype=np.int64)  # stored index x horizon position
    for _ in range(draws):
        starts = buf._sample_starts(c.batch_size).cpu().numpy()
        for t in range(H):
            idx = (starts + 1 + t) % buf.capacity
            within = idx % (ep_len + 1)
            np.add.at(counts[:, t], within, 1)
    rho = np.power(c.rho, np.arange(H))
    tot = counts.sum()
    print(
        f"   BUFFER SAMPLING (executed, not argued): {n_eps} synthetic episodes of {ep_len} steps, "
        f"slice_len = H+1 = {H + 1}, {draws * c.batch_size:,} starts drawn"
    )
    print(
        f"   {'stored idx':>11} "
        + " ".join(f"{'t=' + str(t):>10}" for t in range(H))
        + f" {'share':>8} {'rho-weight':>11}"
    )
    for j in list(range(ep_len - 3, ep_len + 1)):
        row = counts[j]
        w = float((row * rho).sum() / max(row.sum(), 1))
        print(
            f"   {j:>11} "
            + " ".join(f"{int(v):>10,}" for v in row)
            + f" {row.sum() / tot * 100.0:>7.3f}% {w:>11.4f}"
        )
    interior = counts[ep_len // 2]
    termrow = counts[ep_len]
    wi = float((interior * rho).sum())
    wt = float((termrow * rho).sum())
    print(
        f"   => a MID-EPISODE index gets {interior.sum():,} draws with total rho-weight {wi:,.0f}; "
        f"the TERMINAL index (idx {ep_len}, the only transition carrying terminated=1 and the "
        f"outcome spike) gets {termrow.sum():,} draws with total rho-weight {wt:,.0f} "
        f"=> {wi / max(wt, 1e-9):.2f}x less gradient weight, and it is reachable at "
        f"t={[t for t in range(H) if termrow[t] > 0]} only (rho^t = "
        f"{[round(float(rho[t]), 4) for t in range(H) if termrow[t] > 0]})."
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    _paths.require_astrum_env()
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    asp_sets = [s.strip() for s in args.aspects.split(";")]
    only_sets = [s.strip() for s in args.eps_only.split(";")] if args.eps_only else []

    print(
        f"D5 ADVERSARIAL VERIFICATION | chase sep {args.separation:.0f}m mt{args.max_time_s:.0f}s "
        f"jitter{args.jitter:g} ar{args.action_repeat} | dt {dt * 1000:.1f} ms | seed {args.seed}"
    )
    print(
        "Question: is D5's Q-vs-return gap a CREDIT-REACH defect? Three separations: decoder "
        "(symexp(E[bin]) vs E[symexp(bin)]), evaluation point (encoder latent vs the 2-step rollout "
        "latent training fits), and terminal-type SIGN (kill spike vs deck spike)."
    )
    print("D6: the FIRST checkpoint is the run; the rest are controls, same invocation/seeds.")

    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        prov = sd.get("provenance") or {}
        outcome_mode = bool(prov.get("outcome_mode", False))
        tag = "RUN" if ci == 0 else "CONTROL"
        print(f"\n{'=' * 112}")
        print(
            f"[{tag}] {ck} (collected={int(sd.get('collected', -1)):,}) gamma={agent.discount:.6f} "
            f"H={cfg.horizon} rho={cfg.rho} num_q={cfg.num_q} bins[{cfg.vmin},{cfg.vmax}]x"
            f"{cfg.num_bins} outcome_mode={outcome_mode}"
        )
        print("=" * 112)
        if ci == 0:
            _buffer_occupancy(cfg, device)

        aspects = [float(a) for a in asp_sets[min(ci, len(asp_sets) - 1)].split(",") if a.strip()]
        only = None
        if only_sets:
            s = only_sets[min(ci, len(only_sets) - 1)]
            only = {int(x) for x in s.split(",") if x.strip()} if s else None
        for aspect in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=1,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
                outcome_mode=outcome_mode,
                jitter=args.jitter,
            )
            env = make_tdmpc_env(3, **kw)
            seed0 = args.seed + int(round(aspect)) * 1000
            print(f"\n-- asp{aspect:.0f} (seed0 {seed0}) --")
            for e in range(args.episodes):
                if only is not None and e not in only:
                    continue
                ep = _roll_one(agent, env, seed0 + e)
                print(
                    f"   ep{e} seed{seed0 + e} outcome={ep['outcome']} T={ep['act'].shape[0]} "
                    f"({ep['act'].shape[0] * dt:.2f}s) undisc_ret {ep['rew'].sum():+.2f} "
                    f"r_last {ep['rew'][-1]:+.2f}"
                )
                if not ep["terminated"]:
                    print("     (truncated: no terminal transition to read)")
                    continue
                _report_terminal(agent, ep)
                _report_nonterminal_td1(agent, ep)
            env.close()


if __name__ == "__main__":
    main()
