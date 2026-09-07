"""Run the A1 BFM detector over a CHECKPOINT's own flight — the policy-side half of a verdict.

    cd controllers/rl && uv run python _bfm_verdict_tmp.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --cell chase --aspects 0,30,55 --episodes 8 --jitter 0.5

Why this exists. `bfm_detect.py` (A1) is a pure function of a per-tick trace, and `_diag_stick_ceiling.py`
(A2) drives it with SCRIPTED sticks — that pairing measured the demonstration library and found it
hollow. Nothing drives it with a **learned policy**, so every score delta this program has produced is
still unattributable: a run that KILLS more is indistinguishable from a run that LEARNED BFM. The
handoff's Phase-B judging rule is "the standard greedy verdict **plus** the A1 detector's D5 ratio";
`_verdict_tmp.py` supplies the first half and this supplies the second.

WHAT IT DOES AND DOES NOT SHARE WITH `_verdict_tmp.py`

Shared on purpose: the env constructor, `eval_mode=True` greedy acting, the `_prev_mean.zero_()` reset
between episodes, and ground-truth health read straight off `base._states`. The outcome columns are
printed next to the BFM block deliberately — a D5 ratio is only interpretable beside what the policy
actually scored, and printing them from two separate runs invites comparing different seeds.

Sampling cadence matches A2 exactly — one `Sample` per POLICY step at `dt = action_repeat/rate_hz`
(24 Hz for the trainer's `ar5 @ 120 Hz`), taken from the state the action was CHOSEN in, before the
step. That matters twice: the detector's thresholds are in seconds (a 0.5 s hold is 12 samples here,
so the hold tests are well resolved), and a policy trace sampled at a different rate is not comparable
to A2's scripted-stick numbers.

D6, MANDATORY, and the reason `--checkpoints` is a LIST. Any BFM "improvement" that also shows up in
c87 is a detector artifact, not a learning result. Pass the historical checkpoints in the same
invocation so they run against the same env build, same seeds and same detector thresholds — a control
measured in a different session is not a control. Note the one thing this cannot hold fixed: c87 and
c93 predate B1, so their obs column 29 (`gun_quality`) is out of distribution here exactly as
`_verdict_naivelay_tmp.py` documents for c86. Read them as "does the DETECTOR fire on old policies",
not as a fair re-score of those policies.

Runs on CPU by default so it never contends with a GPU training run.
"""

from __future__ import annotations

import argparse
import math
import statistics

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A1 BFM detector over a checkpoint's greedy flight.")
    p.add_argument(
        "--checkpoints",
        required=True,
        help="comma list. Put the RUN FIRST and its D6 controls after (c95, c87) — same env build, "
        "same seeds, same thresholds, one invocation.",
    )
    p.add_argument(
        "--cell",
        default="chase",
        choices=("chase", "headon"),
        help="chase = the standard verdict geometry (co-heading, sep 700 m, --aspects). "
        "headon = the c94/c96 training geometry (reciprocal headings, sep 3000 m, --laterals).",
    )
    p.add_argument("--aspects", default="0,30,55", help="chase cell: nominal start aspects, deg")
    p.add_argument(
        "--laterals",
        default="1400",
        help="headon cell: lateral offset in METRES. With `opp_heading_deg=180` both jets sit on "
        "reciprocal headings at any bearing, so bearing IS lateral offset: lateral = sep·sin(aspect).",
    )
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--separation", type=float, default=None, help="default 700 chase / 3000 headon")
    p.add_argument("--max-time-s", type=float, default=None, help="default 60 chase / 180 headon")
    p.add_argument("--me-dalt-ft", type=float, default=None, help="default 0 chase / 2000 headon")
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in a verdict ([[display-killpct-is-exploration]])")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--modes",
        default="greedy",
        help="greedy (policy prior) and/or mpc (MPPI). Judge on greedy: it is the out-of-band read.",
    )
    return p.parse_args(argv)


def _cell_defaults(args) -> None:
    """Fill the geometry defaults for the chosen cell without clobbering explicit flags."""
    if args.cell == "chase":
        d = dict(separation=700.0, max_time_s=60.0, me_dalt_ft=0.0)
    else:
        d = dict(separation=3000.0, max_time_s=180.0, me_dalt_ft=2000.0)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def _cells(args) -> list[tuple[str, float]]:
    """`(label, aspect_deg)` per cell. The headon label carries the METRES, not the bearing."""
    if args.cell == "chase":
        return [(f"asp{float(a):.0f}", float(a)) for a in args.aspects.split(",") if a.strip()]
    out = []
    for x in args.laterals.split(","):
        if not x.strip():
            continue
        lat = float(x)
        out.append((f"lat{lat:.0f}m", math.degrees(math.asin(min(1.0, lat / args.separation)))))
    return out


def _roll(agent, env, base, C, B, args, episodes: int, seed0: int):
    """Greedy episodes, returning `(outcome_rows, traces)`. One `Sample` per policy step."""
    from astrum_dogfight.geometry import relative

    dt = args.action_repeat / args.rate_hz
    rows, traces = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0, t = True, 0.0
        event = None
        tr = []
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            # Sample the state the action was CHOSEN in, before the step — the action and the
            # geometry it responds to must be the same tick or every entry gate is off by one step.
            me = base._states[base.learner_id]
            bd = base._states[base.opponent_id]
            tr.append(B.make_sample(t, me, bd, a, relative(me, bd)))
            obs, _r, terminated, truncated, info = env.step(a)
            t += dt
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rows.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "event": event,
                "dealt": C.HP_MAX - float(opp.health),
                "taken": C.HP_MAX - float(me.health),
            }
        )
        traces.append(tr)
    return rows, traces


def _outcome_line(rows: list[dict]) -> str:
    n = len(rows)
    return (
        f"score: {n} eps | kills {sum(r['outcome'] == 'kill' for r in rows)} "
        f"deaths {sum(r['outcome'] == 'death' for r in rows)} "
        f"decks {sum(r['outcome'] == 'deck' for r in rows)} "
        f"t/out {sum(r['outcome'] == 'timeout' for r in rows)} | "
        f"dealt {statistics.fmean(r['dealt'] for r in rows):.1f} "
        f"taken {statistics.fmean(r['taken'] for r in rows):.1f}"
    )


def main() -> None:
    args = parse_args()
    _cell_defaults(args)
    _paths.require_astrum_env()
    from astrum_dogfight import bfm_detect as B
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    cells = _cells(args)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(
        f"BFM VERDICT | cell {args.cell} | sep {args.separation:.0f}m "
        f"mt{args.max_time_s:.0f}s jitter{args.jitter:g} perch+{args.me_dalt_ft:.0f}ft "
        f"ar{args.action_repeat} | {args.episodes} eps/cell | LETHAL_RANGE {B.LETHAL_RANGE_M:.0f}m "
        f"(D5 uses tLeth, NOT tHit)"
    )
    print(
        "D6: the FIRST checkpoint is the run; the rest are controls. Any lift that also appears in a "
        "control is a detector artifact, not a learning result."
    )

    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        tag = "RUN" if ci == 0 else "CONTROL"
        print(
            f"\n{'=' * 92}\n[{tag}] {ck} (stage_idx={int(sd.get('stage_idx', -1))}, "
            f"collected={int(sd.get('collected', -1)):,})\n{'=' * 92}"
        )
        for mode in modes:
            cfg = TDMPC2Config(**sd["cfg"])
            cfg.mpc = mode == "mpc"
            agent = TDMPC2(cfg, device)
            agent.load_state_dict(sd)
            agent.model.eval()
            pooled_rows, pooled_traces = [], []
            for label, aspect in cells:
                kw = dict(
                    posture="offensive",
                    aspect_deg=aspect,
                    separation_m=args.separation,
                    action_repeat=args.action_repeat,
                    frame_stack=args.frame_stack,
                    max_time_s=args.max_time_s,
                    rate_hz=args.rate_hz,
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
                base = env.unwrapped
                rows, traces = _roll(
                    agent, env, base, C, B, args, args.episodes, args.seed + int(round(aspect)) * 1000
                )
                env.close()
                pooled_rows.extend(rows)
                pooled_traces.extend(traces)
                print(f"\n-- {mode} / {label} (bearing {aspect:.1f}°) --")
                print(f"   {_outcome_line(rows)}")
                for ln in B.format_summary(B.summarize(traces)).splitlines():
                    print(f"   {ln}")
            if len(cells) > 1:
                print(f"\n-- {mode} / POOLED across {len(cells)} cells --")
                print(f"   {_outcome_line(pooled_rows)}")
                for ln in B.format_summary(B.summarize(pooled_traces)).splitlines():
                    print(f"   {ln}")


if __name__ == "__main__":
    main()
