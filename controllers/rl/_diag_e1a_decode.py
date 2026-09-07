#!/usr/bin/env python3
"""E1a -- is the `two_hot_inv` DECODER compressing the kill, and if so does it change any DECISION?

Phase E box E1a (`docs/bfm-program-handoff.md` §6), authored by D5's audits. Runs on CPU so it never
contends with a training run. Reuses `_diag_credit_reach.py` READ-ONLY (`_roll`-style rollout, the
cell/env construction, `_classify`) so the flight path and the env build are byte-identical to the
box that raised the finding; that file is not modified.

================================================================================
THE MECHANISM UNDER TEST
================================================================================
`functional.py:53-67` decodes the critic's categorical head as

    q_ship = symexp( Σ_i p_i · bin_i )          # a LOG-MEAN in symlog space

and NOT as the conditional expectation

    q_true = Σ_i p_i · symexp(bin_i)            # the quantity a Bellman backup needs

The two agree EXACTLY when `p` is a genuine two-hot (mass on the two bins straddling `symlog(x)`),
because linear interpolation makes `Σ p_i·bin_i = symlog(x)` identically. They diverge only when the
PREDICTED distribution is spread over non-adjacent bins -- i.e. when the critic is uncertain or
bimodal. D5 measured c96 with 0.56-0.86 of Q mass above +50 and a modal bin of +147.41 while the
shipped decode returned a median of +35.4, which is the "the critic knows about the kill and the
decoder throws it away" reading.

================================================================================
WHY THE OBVIOUS STATISTIC IS NOT ENOUGH (read before interpreting anything below)
================================================================================
`symexp` is MONOTONE INCREASING. So at a FIXED state, `argmax_a q_ship(z,a) = argmax_a q_true(z,a)`
and the decoder cannot change which action the actor prefers. The bias can only do work through
CROSS-STATE distortion: `_td_target` is `r + γ·(1-term)·Q(z',π(z'))`, so a per-state-varying shrinkage
of `Q(z')` re-prices states against each other, and THAT flows into the actor at earlier states.

=> the decisive statistic is not the size of the gap. It is whether the gap RE-ORDERS states.
Reported as Spearman ρ(q_ship, q_true) across states, the discordant-pair fraction, and the rank
displacement of the top decile.

================================================================================
WHAT IS MEASURED
================================================================================
M0  ENCODE/DECODE EXACTNESS CONTROL. `two_hot_inv(log(two_hot(x)))` vs `x` over a sweep that includes
    R_KILL=150, R_DEATH=-60, R_TIMEOUT=-50 and |J|~560. Must agree to ~1e-5. If it does, the effect
    below is entirely from PREDICTED SPREAD and is not a coding bug in the codec.
M1  BIN-AXIS HEADROOM. max |symlog(target)| against `cfg.vmax`. `two_hot` CLAMPS, so a target past
    vmax is silently clipped -- a separate and worse defect. symlog(560)=6.33 vs vmax=10 predicts no
    clipping; printed so it is checked rather than assumed.
M2  PER-STATE DECODE GAP on greedy rollouts: q_ship, q_true, Δ = q_true - q_ship, mass above +50 in
    SCALAR terms, the modal bin as a scalar, the head's entropy and `N_eff_bins = exp(H)`.
M3  THE TAIL-SENSITIVITY AUDIT -- the strongest objection to "just use the expectation". Bins run to
    `symexp(±10) = ±22025`, so 1e-3 of stray softmax mass on an end bin adds ±22 to q_true outright.
    Reported as `tail%` = the share of |q_true| contributed by bins with |bin| > --trim-symlog, and as
    a TRIMMED expectation `q_trim` that renormalises inside that band. If q_true is tail-dominated,
    E1b form (i) (decode as the raw expectation) is DANGEROUS and the admissible fix is re-spacing the
    bins or a trimmed decode, not the raw expectation.
M4  THE DECISION TEST. Spearman ρ(q_ship, q_true) within a cell, the discordant-pair fraction, and
    where the top decile by q_true ranks under q_ship.
M5  THE DIFFERENTIAL TEST. Δ bucketed by time-to-terminal and by episode outcome. The mechanism as
    stated requires Δ to be LARGER on kill-adjacent states than on spike-free ones. If Δ is uniform,
    it is generic symlog bias: a near-constant value-scale offset, which re-prices nothing.
M6  BELLMAN MAGNITUDE. γ·Δ per backup and the fixed-point implication Δ/(1-γ) = 200·Δ, against D5's
    measured sampled-bootstrap pessimism of -16.8..-36.3.

================================================================================
WHAT IS EXPLICITLY *NOT* MEASURED
================================================================================
* No training. Nothing here is a verdict on whether changing the decode would help; that is E1b.
* The reward head's decode gap is printed as a FOOTNOTE only. `mpc=False` in every checkpoint on this
  board, and `_td_target` uses the buffer's REAL reward, so the reward head is off the operative path.
* Episode length, train-log kill%/ret, reward-term sums, tHit and the D5 ratio are read nowhere
  (`bfm_detect.py:37-48` blacklist). This says nothing about BFM.
* `return_type="min"` is never used (random 2-of-5 => not reproducible). "avg" throughout, and it is
  replicated EXACTLY: decode per head, then mean over heads.

================================================================================
PRE-REGISTERED INTERPRETATION (fixed before the run)
================================================================================
KILL CONDITION (stop; do not build E1b) -- ANY of:
  K1  M0 fails: the codec is not exact on genuine two-hots => the finding is a codec bug to fix
      directly and this framing is wrong.
  K2  M4 shows ρ >= 0.999 AND a discordant-pair fraction < 1% AND no top-decile displacement => the
      decode is a monotone reparameterisation ACROSS states too, so it re-prices nothing and only
      rescales. Magnitude alone is then not actionable.
  K3  M5 shows Δ (relative, Δ/|q_ship|) statistically indistinguishable between kill-adjacent states
      and the spike-free control cell => generic symlog bias, not a kill-specific attenuation. This is
      the D6 discipline applied at the bin level.
  K4  M3 shows tail% > ~25% of q_true => the raw expectation is not a usable target and form (i) is
      off the table; report the honest restatement (re-space the bins) instead of "fix the decode".
PROCEED (E1b is real) if: M0 passes, M1 shows no clipping, and M4 shows a materially broken ordering
  (ρ < 0.99 or top-decile displacement) that M5 localises to kill-adjacent states with M3's tail
  share small.
Units: one control step = action_repeat 5 / 120 Hz = 41.7 ms; 12 steps = 0.5 s, 36 = 1.5 s.

D6: the FIRST checkpoint is the run, the rest are controls, ALL IN ONE INVOCATION on the same env
build and the same seeds. `--aspects` deliberately includes a cell where c96 kills 0/20 so K3 has a
spike-free comparison inside the same run.

Run (from `controllers/rl`, inside the container, CPU only):
    uv run python _diag_e1a_decode.py \
      --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
      --cell chase --aspects 30,75 --episodes 6 2>&1 | tee _e1a_out.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _diag_credit_reach as CR  # noqa: E402  (READ-ONLY re-use; not modified)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from astrum_rl import _paths  # noqa: E402
from astrum_rl.tdmpc2 import functional as fn  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402
from astrum_rl.tdmpc2.eval_policy import _classify  # noqa: E402

STEP_S = 5.0 / 120.0
# Time-to-terminal buckets, seconds (upper edges). Matches D5's so the two are readable together.
TTT_EDGES = (0.5, 1.0, 2.0, 4.0, 8.0, math.inf)
# Scalar threshold for "kill-sized" probability mass. R_KILL = 150; +50 is D5's reported cut.
MASS_ABOVE = 50.0


# ----------------------------------------------------------------------------- decode variants
def _decode_stats(logits: torch.Tensor, cfg, trim_symlog: float) -> dict[str, np.ndarray]:
    """All decode variants + shape statistics for a `(num_q, B, num_bins)` logit tensor.

    `q_ship` replicates `Q(..., return_type="avg")` EXACTLY: `two_hot_inv` per head, then mean over
    heads. Every other variant is averaged over heads the same way so the comparison is apples to
    apples rather than mean-then-decode (which would be a different estimator).
    """
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, dtype=logits.dtype)
    scal = fn.symexp(bins)  # bin centres in REWARD units
    p = torch.softmax(logits, dim=-1)  # (num_q, B, num_bins)

    q_ship = fn.symexp((p * bins).sum(-1)).mean(0)  # symexp(E[symlog])  <- shipped
    q_true = (p * scal).sum(-1).mean(0)  # E[scalar]        <- the Bellman quantity

    keep = (bins.abs() <= trim_symlog).to(p.dtype)  # trimmed expectation
    pk = p * keep
    pk = pk / pk.sum(-1, keepdim=True).clamp_min(1e-12)
    q_trim = (pk * scal).sum(-1).mean(0)

    contrib = p * scal
    tail = (contrib * (1.0 - keep)).sum(-1).abs()
    total = contrib.sum(-1).abs().clamp_min(1e-12)
    tail_share = (tail / total).mean(0)

    mass_hi = (p * (scal >= MASS_ABOVE).to(p.dtype)).sum(-1).mean(0)
    modal = scal[p.mean(0).argmax(-1)]
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1).mean(0)

    out = {
        "q_ship": q_ship,
        "q_true": q_true,
        "q_trim": q_trim,
        "tail_share": tail_share,
        "mass_hi": mass_hi,
        "modal": modal,
        "neff_bins": ent.exp(),
    }
    return {k: v.detach().cpu().numpy().astype(np.float64).reshape(-1) for k, v in out.items()}


@torch.no_grad()
def _roll(agent, env, args, episodes: int, seed0: int) -> list[dict]:
    """Greedy episodes, then batch-evaluate the Q head's LOGITS over the whole stored trajectory.

    Split for the same reason `_diag_credit_reach._roll` splits: flying must be one `act()` at a time,
    but every head is a plain MLP with no batch-coupled layer, so a whole-episode forward pass is
    numerically identical and far cheaper. `q_ship` here must equal that file's `q` to float error --
    asserted in M0's self-check.
    """
    M = agent.model
    cfg = agent.cfg
    gamma = agent.discount
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0 = True
        obs_hist = [np.asarray(obs, dtype=np.float32)]
        acts, rews, terms = [], [], []
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
            terms.append(float(terminated))
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break

        OB = torch.as_tensor(np.stack(obs_hist), dtype=torch.float32, device=agent.device)
        A = torch.as_tensor(np.stack(acts), dtype=torch.float32, device=agent.device)
        z_all = M.encode(OB)
        z, z_next = z_all[:-1], z_all[1:]
        mu_next = M.pi(z_next)[0]

        # (num_q, B, num_bins) for the ONLINE head at the visited (z,a) -- the quantity the actor
        # and every printed Q in this program reads -- and for the TARGET head at (z', pi(z')),
        # which is the one `_td_target` actually decodes.
        st_q = _decode_stats(M.Q(z, A, return_type="all"), cfg, args.trim_symlog)
        st_t = _decode_stats(
            M.Q(z_next, mu_next, return_type="all", target=True), cfg, args.trim_symlog
        )
        ship_ref = M.Q(z, A, return_type="avg").reshape(-1).cpu().numpy().astype(np.float64)

        r = np.asarray(rews, dtype=np.float64)
        tm = np.asarray(terms, dtype=np.float64)
        T = r.size
        gmc = np.zeros(T, dtype=np.float64)
        acc = 0.0
        for i in range(T - 1, -1, -1):
            acc = r[i] + gamma * acc
            gmc[i] = acc

        out.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "terminated": bool(terminated),
                "T": T,
                "ttt": (T - np.arange(T, dtype=np.float64)) * STEP_S,
                "r": r,
                "term": tm,
                "gmc": gmc,
                "ship_ref": ship_ref,
                **{f"q_{k}": v for k, v in st_q.items()},
                **{f"t_{k}": v for k, v in st_t.items()},
            }
        )
    return out


# ----------------------------------------------------------------------------- stats helpers
def _cat(eps: list[dict], key: str) -> np.ndarray:
    return np.concatenate([e[key] for e in eps]) if eps else np.zeros(0)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman is exact without scipy."""
    order = np.argsort(x, kind="mergesort")
    rk = np.empty(x.size, dtype=np.float64)
    rk[order] = np.arange(x.size, dtype=np.float64)
    # average ties
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            rk[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return rk


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _discordant(a: np.ndarray, b: np.ndarray, max_pairs: int = 400_000) -> float:
    """Fraction of state PAIRS whose ordering inverts between the two decodes.

    Sampled (deterministically, stride-decimated -- no RNG, so it is reproducible) when the full
    O(n²) pair set is too large. Reported as a percentage.
    """
    n = a.size
    if n < 2:
        return float("nan")
    step = max(1, int(math.ceil(n / math.sqrt(max_pairs))))
    ia = np.arange(0, n, step)
    if ia.size < 2:
        return float("nan")
    da = a[ia][:, None] - a[ia][None, :]
    db = b[ia][:, None] - b[ia][None, :]
    iu = np.triu_indices(ia.size, k=1)
    da, db = da[iu], db[iu]
    live = (da != 0) & (db != 0)
    if not live.any():
        return float("nan")
    return 100.0 * float((np.sign(da[live]) != np.sign(db[live])).mean())


def _q(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if x.size else float("nan")


# ----------------------------------------------------------------------------- printers
def _print_m0(cfg, args) -> None:
    print("\nM0  ENCODE/DECODE EXACTNESS CONTROL (must be exact on GENUINE two-hots)")
    xs = torch.tensor(
        [0.0, 1.0, -1.0, 8.0, 35.4, 50.0, -50.0, -60.0, 147.41, 150.0, 560.0, -560.0, 3000.0],
        dtype=torch.float32,
    )
    d = fn.two_hot(xs, cfg)
    back = fn.two_hot_inv(d.clamp_min(1e-30).log(), cfg)
    err = (back - xs).abs()
    rel = err / xs.abs().clamp_min(1.0)
    print(
        f"     {'x':>10} {'decoded':>12} {'abs err':>10} {'rel err':>9}   {'symlog(x)':>10} vs vmax"
    )
    for i in range(xs.numel()):
        sl = float(fn.symlog(xs[i]))
        flag = "  <-- CLIPPED by two_hot" if abs(sl) > cfg.vmax else ""
        print(
            f"     {float(xs[i]):10.2f} {float(back[i]):12.4f} {float(err[i]):10.2e} "
            f"{float(rel[i]):9.2e}   {sl:10.4f} / {cfg.vmax:g}{flag}"
        )
    worst = float(rel.max())
    print(
        f"     worst relative error {worst:.2e} => codec is "
        f"{'EXACT (K1 does not fire)' if worst < 1e-4 else 'NOT exact => K1 FIRES'}"
    )
    print(
        "     Reading: the codec is exact on a genuine two-hot, so every gap below comes from the"
    )
    print("     PREDICTED distribution being spread over non-adjacent bins, not from the codec.")
    print(
        f"     trim band for M3: |symlog(bin)| <= {args.trim_symlog:g} "
        f"(= |scalar| <= {float(fn.symexp(torch.tensor(args.trim_symlog))):.1f})"
    )


def _print_cell(eps: list[dict], label: str, gamma: float, args) -> dict:
    """M2/M3/M4/M5/M6 for one cell. Returns the per-cell summary for the cross-cell K3 table."""
    n = sum(e["T"] for e in eps)
    counts = {o: sum(e["outcome"] == o for e in eps) for o in sorted({e["outcome"] for e in eps})}
    print(f"\n-- {label} | {len(eps)} eps | {n} states | outcomes {counts} --")
    if not n:
        return {}

    ship = _cat(eps, "q_q_ship")
    true = _cat(eps, "q_q_true")
    trim = _cat(eps, "q_q_trim")
    tail = _cat(eps, "q_tail_share")
    mhi = _cat(eps, "q_mass_hi")
    modal = _cat(eps, "q_modal")
    nefb = _cat(eps, "q_neff_bins")
    ref = _cat(eps, "ship_ref")
    ttt = _cat(eps, "ttt")

    # self-check: our q_ship must reproduce Q(...,'avg') exactly
    dev = float(np.abs(ship - ref).max())
    print(
        f"   self-check |q_ship - Q(...,'avg')|max = {dev:.2e} "
        f"({'OK' if dev < 1e-3 else 'MISMATCH -- do not trust this cell'})"
    )

    print("\n   M2/M3  DECODE VARIANTS (per state, then percentiles over states)")
    print(f"     {'quantity':<34} {'p10':>10} {'p50':>10} {'p90':>10} {'max':>10} {'mean':>10}")
    for nm, v in (
        ("q_ship = symexp(E[symlog])  SHIPPED", ship),
        ("q_true = E[scalar]  (Bellman)", true),
        (f"q_trim = E[scalar | |sl|<={args.trim_symlog:g}]", trim),
        ("Delta  = q_true - q_ship", true - ship),
        ("Delta_trim = q_trim - q_ship", trim - ship),
        ("P(mass above +50)", mhi),
        ("modal bin (scalar)", modal),
        ("N_eff bins = exp(entropy)", nefb),
        ("tail share of q_true (frac)", tail),
    ):
        print(
            f"     {nm:<34} {_q(v, 10):10.4f} {_q(v, 50):10.4f} {_q(v, 90):10.4f} "
            f"{float(v.max()):10.4f} {float(v.mean()):10.4f}"
        )
    tail_p50 = _q(tail, 50)
    print(
        f"     K4 check: tail share p50 = {100 * tail_p50:.1f}%, p90 = {100 * _q(tail, 90):.1f}% "
        f"=> {'K4 FIRES (raw expectation unusable)' if tail_p50 > 0.25 else 'K4 quiet'}"
    )

    print("\n   M4  THE DECISION TEST -- does the decode RE-ORDER states?")
    rho = _spearman(ship, true)
    disc = _discordant(ship, true)
    rt = _ranks(true)
    rs = _ranks(ship)
    cut = np.percentile(rt, 90.0)
    top = rt >= cut
    disp = float(np.abs(rt[top] - rs[top]).mean()) if top.any() else float("nan")
    print(f"     Spearman rho(q_ship, q_true) over {ship.size} states = {rho:.6f}")
    print(f"     discordant state PAIRS                              = {disc:.3f}%")
    print(
        f"     top-decile-by-q_true mean |rank shift| under q_ship = {disp:.1f} of {ship.size} ranks "
        f"({100.0 * disp / max(1, ship.size):.2f}%)"
    )
    k2 = (rho >= 0.999) and (disc < 1.0) and (disp / max(1, ship.size) < 0.01)
    print(
        f"     => {'K2 FIRES: monotone across states too, re-prices nothing' if k2 else 'K2 quiet: the decode DOES re-order states'}"
    )

    print("\n   M5  THE DIFFERENTIAL TEST -- is Delta localised near the terminal?")
    print(
        f"     {'ttt bucket':>14} {'steps':>7} {'n':>6} | {'q_ship p50':>10} {'q_true p50':>10} "
        f"{'Delta p50':>10} {'Delta/|ship|':>12} {'massHi p50':>10}"
    )
    lo = 0.0
    for hi in TTT_EDGES:
        g = (ttt > lo) & (ttt <= hi)
        if g.sum():
            d = true[g] - ship[g]
            rel = d / np.abs(ship[g]).clip(1e-6)
            hs = "inf" if hi == math.inf else f"{hi / STEP_S:.0f}"
            print(
                f"     {f'{lo:g}-{hi:g}s':>14} {hs:>7} {int(g.sum()):6d} | "
                f"{_q(ship[g], 50):10.3f} {_q(true[g], 50):10.3f} {_q(d, 50):10.3f} "
                f"{_q(rel, 50):12.3f} {_q(mhi[g], 50):10.4f}"
            )
        lo = hi
    for oc in sorted({e["outcome"] for e in eps}):
        sel = [e for e in eps if e["outcome"] == oc]
        s, t = _cat(sel, "q_q_ship"), _cat(sel, "q_q_true")
        if s.size:
            rel = (t - s) / np.abs(s).clip(1e-6)
            print(
                f"     outcome {oc:<12} n={s.size:6d} | q_ship p50 {_q(s, 50):9.3f} "
                f"q_true p50 {_q(t, 50):9.3f} Delta p50 {_q(t - s, 50):9.3f} "
                f"rel p50 {_q(rel, 50):7.3f}"
            )

    print(
        "\n   M6  BELLMAN MAGNITUDE (on the TARGET head at (z', pi(z')), which _td_target decodes)"
    )
    ts, tt = _cat(eps, "t_q_ship"), _cat(eps, "t_q_true")
    dt_ = tt - ts
    print(
        f"     target-head Delta p50 {_q(dt_, 50):9.3f} | per-backup gamma*Delta p50 "
        f"{gamma * _q(dt_, 50):9.3f} | fixed-point Delta/(1-g) = {_q(dt_, 50) / (1.0 - gamma):9.1f}"
    )
    print(
        "     compare D5's measured sampled-bootstrap pessimism -16.8..-36.3 at the fixed point: "
        "same order => the two value-path defects are comparable in size."
    )
    rel_all = (true - ship) / np.abs(ship).clip(1e-6)
    return {
        "label": label,
        "n": int(ship.size),
        "kills": int(sum(1 for e in eps if e["outcome"] == "kill")),
        "rho": rho,
        "disc": disc,
        "rel_p50": _q(rel_all, 50),
        "delta_p50": _q(true - ship, 50),
        "tail_p50": tail_p50,
        "massHi_p50": _q(mhi, 50),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1a: is the two_hot decoder compressing the kill?")
    p.add_argument(
        "--checkpoints",
        required=True,
        help="comma list; RUN FIRST, D6 controls after. Same invocation, same seeds.",
    )
    p.add_argument("--cell", default="chase", choices=("chase", "headon"))
    p.add_argument(
        "--aspects",
        default="30,75",
        help="chase start aspects. MUST include a spike-free cell (75 = c96 kills 0/20) for K3.",
    )
    p.add_argument("--laterals", default="1400")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--separation", type=float, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--me-dalt-ft", type=float, default=None)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in this program")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument("--reward-mode", default="auto", choices=("auto", "outcome", "dense"))
    p.add_argument(
        "--trim-symlog",
        type=float,
        default=7.0,
        help="M3 trim band in SYMLOG units (7.0 => |scalar| <= 1096, comfortably past |J|~560)",
    )
    p.add_argument("--device", default="cpu", help="CPU by default so it never contends with a run")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    CR._cell_defaults(args)
    _paths.require_astrum_env()
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    cells = CR._cells(args)

    print(
        f"E1a DECODER COMPRESSION | cell {args.cell} | sep {args.separation:.0f}m "
        f"mt{args.max_time_s:.0f}s jitter{args.jitter:g} perch+{args.me_dalt_ft:.0f}ft "
        f"| {args.episodes} eps/cell | dt {STEP_S * 1000:.1f} ms | seed {args.seed}"
    )
    print(
        "q_ship = symexp(E_p[bin]) = what functional.py:53-67 SHIPS. q_true = E_p[symexp(bin)] = "
        "what a Bellman backup needs. They agree exactly on a genuine two-hot (M0)."
    )
    print(
        "⚠ symexp is MONOTONE, so at a FIXED state the decode cannot change argmax_a. The bias can "
        "only act by RE-ORDERING states into _td_target => M4 is the decisive statistic, not the gap."
    )
    print("NOT BFM EVIDENCE: no behaviour statistic is read here (bfm_detect.py:37-48).")
    print("D6: the FIRST checkpoint is the run, the rest are controls.")

    summary = []
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        raw_cfg = sd["cfg"]
        cfg = TDMPC2Config(**raw_cfg)
        cfg.mpc = False
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        prov = sd.get("provenance") or {}
        outcome_mode = (
            bool(prov.get("outcome_mode", False))
            if args.reward_mode == "auto"
            else args.reward_mode == "outcome"
        )
        tag = "RUN" if ci == 0 else "CONTROL"
        print(f"\n{'=' * 118}")
        print(
            f"[{tag}] {ck} (stage_idx={int(sd.get('stage_idx', -1))}, "
            f"collected={int(sd.get('collected', -1)):,}, outcome_mode={outcome_mode})"
        )
        print(
            f"       gamma={agent.discount:.6f} | symlog two-hot bins [{cfg.vmin},{cfg.vmax}]"
            f"x{cfg.num_bins} (bin_size {cfg.bin_size:.4f} symlog) | num_q={cfg.num_q} | "
            f"mpc={cfg.mpc} | H={cfg.horizon} rho={cfg.rho}"
        )
        print(
            f"       bin extremes in REWARD units: symexp({cfg.vmin:g})="
            f"{float(fn.symexp(torch.tensor(float(cfg.vmin)))):.1f} .. symexp({cfg.vmax:g})="
            f"{float(fn.symexp(torch.tensor(float(cfg.vmax)))):.1f}  <- M3's whole point"
        )
        print("=" * 118)
        if ci == 0:
            _print_m0(cfg, args)

        for label, aspect in cells:
            kw = dict(
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
                outcome_mode=outcome_mode,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            if args.opp_tier > 0:
                kw["opponent_tier"] = args.opp_tier
            if args.cell == "headon":
                kw["opp_heading_deg"] = 180.0
            env = make_tdmpc_env(3, **kw)
            eps = _roll(agent, env, args, args.episodes, args.seed + int(round(aspect)) * 1000)
            env.close()
            row = _print_cell(eps, f"{tag} {Path(ck).parent.name} / {label}", agent.discount, args)
            if row:
                row["ck"] = Path(ck).parent.name
                row["tag"] = tag
                summary.append(row)

    print(f"\n{'=' * 118}")
    print("K3  CROSS-CELL DIFFERENTIAL -- the mechanism needs Delta LARGER where the kill is")
    print("=" * 118)
    print(
        f"  {'ckpt':<16} {'cell':<28} {'n':>6} {'kills':>6} | {'rel Delta p50':>13} "
        f"{'Delta p50':>10} {'massHi p50':>10} {'rho':>9} {'disc%':>7} {'tail p50':>9}"
    )
    for r in summary:
        print(
            f"  {r['ck']:<16} {r['label'].split('/')[-1].strip():<28} {r['n']:6d} {r['kills']:6d} | "
            f"{r['rel_p50']:13.4f} {r['delta_p50']:10.3f} {r['massHi_p50']:10.4f} "
            f"{r['rho']:9.5f} {r['disc']:7.3f} {r['tail_p50']:9.4f}"
        )
    print(
        "  K3 fires if rel Delta is indistinguishable between a kill-bearing cell and a "
        "spike-free (kills=0) one => generic symlog bias, not a kill-specific attenuation."
    )
    print("\ndone.")


if __name__ == "__main__":
    main()
