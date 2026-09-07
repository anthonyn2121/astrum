"""D6a — the free pre-check on `rho`: how deep is the learned dynamics model actually usable?

    cd controllers/rl && uv run python _diag_latent_depth.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --cells chase,headon --aspects 30,55 --episodes 4 --jitter 0.5 --max-depth 30

WHY THIS EXISTS

`rho` (`config.py:86`, 0.5) is the last algorithmic knob the ledger grades UNTESTED that also has a
mechanism. It appears in three places, all as `pow(cfg.rho, arange(...))`:

  * `agent.py:317`  — weights the CONSISTENCY term (coef 20.0) at latent-rollout step t, and the
                      reward (coef 0.1) and value (coef 0.1) cross-entropies at the same t.
  * `agent.py:402`  — weights `update_pi` over H+1 latents.

The claim D6 rests on is "rho caps trained latent-rollout depth at ~10 control steps for every H".
This harness measures the thing that claim is about: the model's own latent prediction error as a
function of rollout depth, 1 → `--max-depth` steps, on recorded on-policy transitions.

WHAT IS MEASURED

For every start index t in a rolled episode: encode `obs_t`, then roll `model.next(z, a)` forward
using the STORED actions `a_t … a_{t+k-1}` (the actions the jet actually flew, not re-planned ones),
and compare the k-step predicted latent against `model.encode(obs_{t+k})` — the same target
`agent.update` regresses onto at `agent.py:327`. Reported per depth:

  * `predL2`   — ||z_hat_k − encode(obs_{t+k})||_2, the model's error.
  * `persL2`   — ||encode(obs_t) − encode(obs_{t+k})||_2, the **persistence baseline** ("predict no
                 change"). MANDATORY: the latent is SimNorm'd (`simnorm_dim=8`, 64 groups of 8, each
                 a softmax) so ||z||_2 is pinned in [2.83, 8.00] and an absolute L2 has no scale.
                 A model that does not beat persistence at depth k has learned nothing usable there.
  * `ratio`    — per-sample `predL2 / persL2`, median. Scale-free. <1 = beats persistence.
  * `cos`      — cosine similarity between prediction and target. Scale-free.
  * `randL2`   — a depth-independent ceiling: ||encode(obs_t) − encode(obs_s)|| for an unrelated s
                 drawn from the same pooled cell. This is "no information at all".
  * `predMSE`  — the per-element MSE, i.e. literally the quantity inside `F.mse_loss` at
                 `agent.py:327`, so the numbers connect to the loss the trainer actually minimises.
  * reward head — |two_hot_inv(model.reward(z_hat_k, a_{t+k})) − r_{t+k}|, median and mean, because
                 `consistency_coef` (20.0) and `reward_coef` (0.1) consume the SAME rollout and the
                 reward head is what a deeper rollout would actually have to carry. Two baselines:
                 `rMAEfresh` (same step, same action, but from the *freshly encoded* latent — the
                 reachable floor at that step) and `rMAEstale` (from `encode(obs_t)`, the reward
                 analogue of persistence). Plus `|r − median(r)|` as the trivial constant predictor.
  * `rho^k` and the EFFECTIVE weight given the checkpoint's own H, so the reader can see exactly
    which depths are trained at what weight and where the weight is zero.

WHAT THIS EXPLICITLY DOES NOT MEASURE

  * It does not measure planning quality. `cfg.mpc` is False in every shipped checkpoint, so no
    training run has ever rolled the model at collection time; this is a read of the LEARNED MODEL,
    not of a control loop.
  * The trajectory is rolled GREEDY, so the state distribution is the deployed policy's own. The
    buffer the model was trained on was collected stochastically (and with expert overrides). Model
    error on the greedy path is therefore not exactly training error.
  * It does not measure what raising `rho` would do to learning. Nothing here is a training run.
    It bounds only what a deeper rollout has to work with today.
  * `encode(obs_{t+k})` is a MOVING target during training. Here both encoder and dynamics are
    frozen, so this is the converged-encoder reading, which is the one that matters for D6b.

PRE-REGISTERED INTERPRETATION (handoff §6, D6a) — BOTH readings get reported either way

  * If error at depth 10-30 is ALREADY LOW (median ratio well under 1 and cos high out to 30):
    `rho = 0.5` is not capping anything worth having ⇒ the D6 lever DIES FOR FREE. Report FAIL.
  * If it DEGRADES SHARPLY past ~10 steps: `rho` IS capping a real capability — *and* raising rho
    would import exactly that error into the consistency/reward/value targets at full weight on the
    first update. The second half is the launch RISK and is the whole reason this box exists.
  * Either way, note the arithmetic a skeptic already put on record: `rho=1.0` is not one variable.
    The same constant weights `update_pi` (`agent.py:402-405`; mean weight over H+1=4 latents goes
    0.469 → 1.000 = a ~2.1× effective actor-LR increase, and LR perturbation of a warm peak is a
    recorded oscillator), the consistency loss (coef 20.0) and the value loss (0.1) simultaneously.

One control step = 41.7 ms (`action_repeat` 5 at 120 Hz ⇒ 24 Hz). Depth 30 = 1.25 s; depth 10 =
0.417 s; depth 3 = 0.125 s. Both units are printed in the table.

D6 applies: every checkpoint in `--checkpoints` runs in the SAME invocation, same env build, same
seeds, same depths — the first is the RUN (c96), the rest are the controls (c95, c87). Runs on CPU.
"""

from __future__ import annotations

import argparse
import builtins
import math

import numpy as np
import torch
from astrum_rl import _paths  # noqa: F401  (sys.path shim, matches the other harnesses)
from astrum_rl.tdmpc2 import functional as fn
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

_bprint = builtins.print


def print(*a, **kw):  # noqa: A001
    """Shadow `builtins.print` with flush=True.

    JSBSim writes to fd 1 from native code, bypassing Python's block buffer. On the first
    full run that reordering CLOBBERED two lines mid-write (c87/chase-asp30's `cos` line was
    truncated and its `reward MAE` line lost entirely). Forcing a flush per line fixes it.
    """
    kw.setdefault("flush", True)
    _bprint(*a, **kw)


CELL_DEFAULTS = {
    # (separation_m, max_time_s, me_dalt_ft) — copied from _bfm_verdict_tmp._cell_defaults
    "chase": (700.0, 60.0, 0.0),
    "headon": (3000.0, 180.0, 2000.0),
}
DEPTH_MARKS = (1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D6a: latent rollout error vs depth.")
    p.add_argument("--checkpoints", required=True, help="comma list; RUN first, D6 controls after")
    p.add_argument("--cells", default="chase", help="comma list of chase|headon")
    p.add_argument("--aspects", default="30,55", help="chase cell: start aspects, deg")
    p.add_argument("--laterals", default="1400", help="headon cell: lateral offset, METRES")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=30, help="deepest latent rollout, in steps")
    p.add_argument("--stride", type=int, default=3, help="stride between start indices")
    p.add_argument("--max-starts", type=int, default=2400, help="cap on start indices per cell")
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--separation", type=float, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--me-dalt-ft", type=float, default=None)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument(
        "--outcome-mode",
        default="auto",
        choices=("auto", "on", "off"),
        help="auto = take it from the checkpoint's own provenance. The reward-head numbers are "
        "meaningless if the env's reward differs from the one the head was trained on.",
    )
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--chunk", type=int, default=512, help="batch chunk for the rollout math")
    p.add_argument(
        "--no-fly",
        action="store_true",
        help="print the per-checkpoint D6b weight arithmetic only; fly nothing. Auditable "
        "companion to the measured run (the arithmetic is derivation, not measurement).",
    )
    return p.parse_args(argv)


def _p50(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def _roll_greedy(agent, env, episodes: int, seed0: int) -> list[dict]:
    """Fly `episodes` greedy; return per-episode obs/action/reward arrays.

    `obs` has T+1 rows, `action`/`reward` have T — the same alignment `buffer.add_episode` uses,
    so `action[i]`/`reward[i]` is the transition from `obs[i]` to `obs[i+1]`.
    """
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0 = True
        obs_l, act_l, rew_l = [np.asarray(obs, dtype=np.float32)], [], []
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, r, terminated, truncated, _info = env.step(a)
            obs_l.append(np.asarray(obs, dtype=np.float32))
            act_l.append(np.asarray(a, dtype=np.float32))
            rew_l.append(float(r))
            if terminated or truncated:
                break
        out.append(
            {
                "obs": np.stack(obs_l),
                "action": np.stack(act_l),
                "reward": np.asarray(rew_l, dtype=np.float32),
            }
        )
    return out


@torch.no_grad()
def _depth_stats(agent, eps: list[dict], K: int, stride: int, max_starts: int, chunk: int) -> dict:
    """Roll the model forward from many start indices and accumulate per-depth errors.

    Returns dict of depth -> dict of stat name -> np.ndarray over samples, plus scalars.
    """
    model, dev = agent.model, agent.device
    cfg = agent.cfg
    acc = {k: {n: [] for n in ("pred", "pers", "cos", "mse")} for k in range(1, K + 1)}
    racc = {k: {n: [] for n in ("err", "fresh", "stale", "true")} for k in range(0, K + 1)}
    rand_pool, all_rewards = [], []
    n_starts_total = 0
    # The first full run read median||z|| = nan on c96/headon ⇒ at least one non-finite row
    # reached the encoder. Count it explicitly rather than letting a nan hide in a median.
    n_bad_obs = n_bad_z = n_obs_rows = 0

    # Per-cell budget split evenly across episodes so no single long episode dominates.
    per_ep_cap = max(1, max_starts // max(1, len(eps)))

    for e in eps:
        obs = torch.as_tensor(e["obs"], device=dev)
        act = torch.as_tensor(e["action"], device=dev)
        rew = torch.as_tensor(e["reward"], device=dev)
        T = act.shape[0]
        all_rewards.append(e["reward"])
        # depth K needs action index t+K and obs index t+K+1 ⇒ t <= T-1-K
        last = T - 1 - K
        if last < 0:
            continue
        Z = model.encode(obs)  # (T+1, L)
        n_obs_rows += int(obs.shape[0])
        n_bad_obs += int((~torch.isfinite(obs)).any(dim=-1).sum())
        n_bad_z += int((~torch.isfinite(Z)).any(dim=-1).sum())
        rand_pool.append(Z.cpu().numpy())
        starts = torch.arange(0, last + 1, stride, device=dev)
        if starts.numel() > per_ep_cap:
            sel = torch.linspace(0, starts.numel() - 1, per_ep_cap).round().long()
            starts = starts[sel.to(dev)]
        n_starts_total += int(starts.numel())

        for lo in range(0, starts.numel(), chunk):
            s = starts[lo : lo + chunk]
            z0 = Z[s]  # (N, L) depth 0 = freshly encoded
            z = z0
            for k in range(0, K + 1):
                a = act[s + k]
                r_true = rew[s + k]
                r_pred = fn.two_hot_inv(model.reward(z, a), cfg)
                r_fresh = fn.two_hot_inv(model.reward(Z[s + k], a), cfg)
                r_stale = fn.two_hot_inv(model.reward(z0, a), cfg)
                racc[k]["err"].append((r_pred - r_true).abs().cpu().numpy())
                racc[k]["fresh"].append((r_fresh - r_true).abs().cpu().numpy())
                racc[k]["stale"].append((r_stale - r_true).abs().cpu().numpy())
                racc[k]["true"].append(r_true.cpu().numpy())
                if k == K:
                    break
                z = model.next(z, a)
                tgt = Z[s + k + 1]
                d = z - tgt
                acc[k + 1]["pred"].append(d.norm(dim=-1).cpu().numpy())
                acc[k + 1]["pers"].append((z0 - tgt).norm(dim=-1).cpu().numpy())
                acc[k + 1]["cos"].append(
                    torch.nn.functional.cosine_similarity(z, tgt, dim=-1).cpu().numpy()
                )
                acc[k + 1]["mse"].append(d.pow(2).mean(dim=-1).cpu().numpy())

    out = {
        "n_starts": n_starts_total,
        "n_bad_obs": n_bad_obs,
        "n_bad_z": n_bad_z,
        "n_obs_rows": n_obs_rows,
    }
    for k in range(1, K + 1):
        out[k] = {n: np.concatenate(v) if v else np.array([]) for n, v in acc[k].items()}
    out["rew"] = {
        k: {n: np.concatenate(v) if v else np.array([]) for n, v in racc[k].items()}
        for k in range(0, K + 1)
    }
    # Unrelated-pair latent distance: the "no information" ceiling on the same pooled cell.
    if rand_pool:
        P = np.concatenate(rand_pool)
        P = P[np.isfinite(P).all(axis=-1)]  # drop non-finite rows so a nan cannot hide in a median
        rng = np.random.default_rng(12345)
        i = rng.integers(0, P.shape[0], size=min(4000, P.shape[0]))
        j = rng.integers(0, P.shape[0], size=i.size)
        out["randL2"] = float(np.median(np.linalg.norm(P[i] - P[j], axis=-1)))
        num = (P[i] * P[j]).sum(-1)
        den = np.linalg.norm(P[i], axis=-1) * np.linalg.norm(P[j], axis=-1)
        # SimNorm latents are all-positive (64 softmax groups), so cosine is bounded far above 0.
        # This is the cos of two UNRELATED states: the ceiling any cos number must be read against.
        out["randCos"] = float(np.median(num / np.maximum(den, 1e-9)))
        out["zn"] = float(np.median(np.linalg.norm(P, axis=-1)))
    else:
        out["randL2"] = float("nan")
        out["randCos"] = float("nan")
        out["zn"] = float("nan")
    out["all_rewards"] = np.concatenate(all_rewards) if all_rewards else np.array([])
    return out


def _print_table(st: dict, K: int, H: int, rho: float, ar: int, hz: float) -> None:
    ms = 1000.0 * ar / hz
    rmed = float(np.median(st["all_rewards"])) if st["all_rewards"].size else float("nan")
    triv = (
        float(np.median(np.abs(st["all_rewards"] - rmed)))
        if st["all_rewards"].size
        else float("nan")
    )
    print(
        f"   n_starts={st['n_starts']}  median||z||={st['zn']:.3f} "
        f"(SimNorm pins it to [{math.sqrt(512 / 8) * (1 / math.sqrt(8)):.2f}, "
        f"{math.sqrt(512 / 8):.2f}])  |  NO-INFORMATION CEILING from unrelated state pairs: "
        f"||dz||={st['randL2']:.3f}, cos={st['randCos']:.4f}  |  trivial reward predictor "
        f"(const median) MAE={triv:.4f}  |  non-finite rows: obs {st['n_bad_obs']}/"
        f"{st['n_obs_rows']}, latent {st['n_bad_z']}/{st['n_obs_rows']}"
    )
    print(
        "   depth  ms      rho^k   w_cons  w_rew  | predL2 p50   p90  persL2 p50  ratio p50   p90 "
        " P(beat)  cos p50   pred/rand | predMSE p50 | rMAE p50  rMAEfresh  rMAEstale  rMAE mean"
    )
    for k in range(0, K + 1):
        # consistency at latent depth d carries rho^(d-1) (agent.py:327: rho[t] pairs with zs[t+1]);
        # the reward/value CE at a latent of depth d carries rho^d (agent.py:334/336 use zs[:-1]).
        w_cons = rho ** (k - 1) if 1 <= k <= H else 0.0
        w_rew = rho**k if k <= H - 1 else 0.0
        r = st["rew"][k]
        if k == 0:
            lat = (
                f"{'-':>11} {'-':>5} {'-':>11} {'-':>10} {'-':>5} {'-':>7} {'-':>8} {'-':>11} "
                f"| {'-':>11} |"
            )
        else:
            a = st[k]
            ratio = a["pred"] / np.maximum(a["pers"], 1e-9)
            lat = (
                f"{_p50(a['pred']):11.4f} {float(np.percentile(a['pred'], 90)):5.2f} "
                f"{_p50(a['pers']):11.4f} {_p50(ratio):10.3f} "
                f"{float(np.percentile(ratio, 90)):5.2f} {float((ratio < 1.0).mean()):7.3f} "
                f"{_p50(a['cos']):8.4f} {_p50(a['pred']) / max(st['randL2'], 1e-9):11.3f} "
                f"| {_p50(a['mse']):11.3e} |"
            )
        print(
            f"   {k:5d} {k * ms:6.1f}  {rho**k:7.5f} {w_cons:7.4f} {w_rew:6.4f} | {lat}"
            f" {_p50(r['err']):9.4f} {_p50(r['fresh']):10.4f} {_p50(r['stale']):10.4f}"
            f" {float(np.mean(r['err'])):10.4f}"
        )


def _summary(st: dict, K: int, H: int, rho: float, ar: int, hz: float) -> None:
    """The pre-registered read, computed rather than eyeballed."""
    xs = [k for k in DEPTH_MARKS if k <= K]
    ratios = {k: _p50(st[k]["pred"] / np.maximum(st[k]["pers"], 1e-9)) for k in xs}
    coss = {k: _p50(st[k]["cos"]) for k in xs}
    cross = next(
        (
            k
            for k in range(1, K + 1)
            if _p50(st[k]["pred"] / np.maximum(st[k]["pers"], 1e-9)) >= 1.0
        ),
        None,
    )
    print("   SUMMARY  ratio(pred/persist) p50: " + "  ".join(f"d{k}={ratios[k]:.3f}" for k in xs))
    print(
        "   SUMMARY  cos p50:                "
        + "  ".join(f"d{k}={coss[k]:.4f}" for k in xs)
        + f"  | unrelated-pair cos={st['randCos']:.4f}"
    )
    print(
        "   SUMMARY  predL2/randpairL2 p50:   "
        + "  ".join(f"d{k}={_p50(st[k]['pred']) / max(st['randL2'], 1e-9):.3f}" for k in xs)
    )
    print(
        f"   SUMMARY  first depth where median ratio >= 1.0 (stops beating persistence): "
        f"{cross if cross is not None else 'never within depth ' + str(K)}"
    )
    rmed = float(np.median(st["all_rewards"])) if st["all_rewards"].size else float("nan")
    triv = (
        float(np.median(np.abs(st["all_rewards"] - rmed)))
        if st["all_rewards"].size
        else float("nan")
    )
    print(
        "   SUMMARY  reward MAE p50: "
        + "  ".join(f"d{k}={_p50(st['rew'][k]['err']):.4f}" for k in [0] + xs)
        + f"  | trivial const={triv:.4f}"
    )
    print(
        f"   SUMMARY  trained depths at H={H}, rho={rho}: consistency d1..d{H} at weights "
        + ", ".join(f"{rho ** (d - 1):.3f}" for d in range(1, H + 1))
        + f"; every depth > {H} carries weight EXACTLY 0 (H truncates the rollout, not rho). "
        f"d{H} = {H * ar / hz * 1000:.0f} ms."
    )


def _weight_arithmetic(cfg) -> None:
    """Exactly what the proposed D6b edit (`rho: 0.5 -> 1.0`) changes at THIS checkpoint's H.

    Pure arithmetic off `agent.py:317` (consistency / reward / value) and `agent.py:402`
    (`update_pi`) — no measurement. Printed so the reader does not have to take the
    "caps depth at ~10 steps" framing on trust when H is 3.
    """
    H, rho = int(cfg.horizon), float(cfg.rho)
    cons = [rho**t for t in range(H)]  # rho[t] pairs with the depth-(t+1) latent
    pi_w = [rho**t for t in range(H + 1)]
    cons1 = [1.0] * H
    pi1 = [1.0] * (H + 1)
    print(
        f"   D6b ARITHMETIC at H={H} (no measurement, just `pow(rho, arange)`):\n"
        f"     deepest latent the loss EVER touches = depth {H} = {H * cfg.action_repeat / 120.0 * 1000:.0f} ms."
        f" Depths {H + 1}+ carry weight 0 at rho=0.5 AND at rho=1.0 — H truncates the rollout, not rho,\n"
        f"     so `rho` cannot cap depth at ~10 steps here: 0.5^k is only ever evaluated at k=0..{H - 1}"
        f" ({', '.join(f'{w:.3f}' for w in cons)}).\n"
        f"     rho 0.5 -> 1.0 changes, at H={H}: consistency/reward/value total rho weight "
        f"{sum(cons):.3f} -> {sum(cons1):.3f} (x{sum(cons1) / sum(cons):.3f}), i.e. an effective "
        f"consistency_coef {cfg.consistency_coef} -> {cfg.consistency_coef * sum(cons1) / sum(cons):.2f}, "
        f"reward_coef {cfg.reward_coef} -> {cfg.reward_coef * sum(cons1) / sum(cons):.4f}, "
        f"value_coef {cfg.value_coef} -> {cfg.value_coef * sum(cons1) / sum(cons):.4f};\n"
        f"     termination_coef {cfg.termination_coef} is NOT rho-weighted, so its share of `total` "
        f"falls by the same x{sum(cons1) / sum(cons):.3f};\n"
        f"     update_pi mean weight over H+1={H + 1} latents {float(np.mean(pi_w)):.3f} -> "
        f"{float(np.mean(pi1)):.3f} (x{float(np.mean(pi1)) / float(np.mean(pi_w)):.3f}). Caveat on the "
        f"'~2.1x actor LR' reading: `pi_optim` is Adam and BOTH optimizer states are restored on "
        f"resume (agent.py:467-470),\n"
        f"     so a UNIFORM gradient rescale is absorbed as v adapts (beta2=0.999 => ~1e3 updates); "
        f"what persists is the RELATIVE reweighting across depths plus the termination share, and "
        f"grad_clip_norm={cfg.grad_clip_norm} absorbs part of the transient."
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    K = int(args.max_depth)

    from astrum_rl.envs import make_tdmpc_env

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in CELL_DEFAULTS:
            raise SystemExit(f"unknown cell {c!r}")

    print(
        f"LATENT DEPTH (D6a) | depths 1..{K} ({K * args.action_repeat / args.rate_hz:.3f} s at "
        f"{args.rate_hz / args.action_repeat:.0f} Hz; 1 step = "
        f"{1000 * args.action_repeat / args.rate_hz:.1f} ms) | cells {','.join(cells)} | "
        f"{args.episodes} eps/cell | jitter {args.jitter:g} | stride {args.stride} | "
        f"max_starts {args.max_starts} | greedy (mu), STORED actions"
    )
    print(
        "Targets are `model.encode(obs_{t+k})` — the same regression target as agent.py:327. "
        "persL2 = predict-no-change. ratio<1 means the model beats persistence at that depth."
    )
    print(
        "D6: the FIRST checkpoint is the run; the rest are controls, same invocation/seeds/depths."
    )

    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        prov = sd.get("provenance", {}) or {}
        if args.outcome_mode == "auto":
            om = bool(prov.get("outcome_mode", False))
            om_src = "provenance"
        else:
            om = args.outcome_mode == "on"
            om_src = "flag"
        tag = "RUN" if ci == 0 else "CONTROL"
        print(
            f"\n{'=' * 108}\n[{tag}] {ck} (collected={int(sd.get('collected', -1)):,}, "
            f"H={cfg.horizon}, rho={cfg.rho}, latent_dim={cfg.latent_dim}, "
            f"simnorm_dim={cfg.simnorm_dim}, obs_dim={cfg.obs_dim}, "
            f"outcome_mode={om} [{om_src}])\n{'=' * 108}"
        )
        _weight_arithmetic(cfg)
        if args.no_fly:
            continue
        for cell in cells:
            sep, mts, dalt = CELL_DEFAULTS[cell]
            sep = args.separation if args.separation is not None else sep
            mts = args.max_time_s if args.max_time_s is not None else mts
            dalt = args.me_dalt_ft if args.me_dalt_ft is not None else dalt
            if cell == "chase":
                geoms = [
                    (f"asp{float(a):.0f}", float(a)) for a in args.aspects.split(",") if a.strip()
                ]
            else:
                geoms = [
                    (
                        f"lat{float(x):.0f}m",
                        math.degrees(math.asin(min(1.0, float(x) / sep))),
                    )
                    for x in args.laterals.split(",")
                    if x.strip()
                ]
            for label, aspect in geoms:
                kw = dict(
                    posture="offensive",
                    aspect_deg=aspect,
                    separation_m=sep,
                    action_repeat=args.action_repeat,
                    frame_stack=args.frame_stack,
                    max_time_s=mts,
                    rate_hz=args.rate_hz,
                    outcome_mode=om,
                )
                if args.jitter > 0:
                    kw["jitter"] = args.jitter
                if dalt:
                    kw["me_dalt_ft"] = dalt
                if cell == "headon":
                    kw["opp_heading_deg"] = 180.0
                env = make_tdmpc_env(3, **kw)
                eps = _roll_greedy(agent, env, args.episodes, args.seed + int(round(aspect)) * 1000)
                env.close()
                lens = [int(e["action"].shape[0]) for e in eps]
                print(
                    f"\n-- {cell}/{label} (bearing {aspect:.1f}deg, sep {sep:.0f} m, mt {mts:.0f} s,"
                    f" dalt {dalt:.0f} ft) : ep lens {lens} --"
                )
                st = _depth_stats(agent, eps, K, args.stride, args.max_starts, args.chunk)
                if st["n_starts"] == 0:
                    print(
                        "   NO usable start index (every episode shorter than the depth) — skipped"
                    )
                    continue
                _print_table(st, K, cfg.horizon, cfg.rho, args.action_repeat, args.rate_hz)
                _summary(st, K, cfg.horizon, cfg.rho, args.action_repeat, args.rate_hz)


if __name__ == "__main__":
    main()
