"""Independent HONEST verdict of a curriculum-29 milestone vs a FRESHLY-built native Tier-3 bot.

    cd controllers/rl && uv run python _verdict_tmp.py --checkpoint runs/curriculum-29/run.pt

Deliberately NOT the training loop's gate and NOT self-play: it rebuilds the scripted Tier-3
opponent from scratch via `make_tdmpc_env(tier=3, posture="offensive", aspect_deg=A)` — the same
native bot the honest frontier gate probes — and rolls deterministic greedy (eval_mode=True)
episodes across the aspect hemisphere, reading GROUND-TRUTH health straight off the env. Matches
the c29 training env exactly: separation 700 m, action_repeat 5, frame_stack 1, max_time_s 60,
rate_hz 120. Default jitter 0 pins each nominal aspect for a clean canonical read (pass --jitter
0.5 to reproduce the gate's jittered geometry).

Per aspect it reports kills / deaths / decks / trades / timeouts, mean HP dealt & taken, mean NET
(dealt − taken), and net-ahead% — the honest `_stage_win` net criterion: a KILL, OR ended alive
with no losing terminal AND net HP ≥ threshold. This is the number that decides "T3 beaten"; do
NOT trust the in-run net% (in-memory deque, resets on restart) for the final call. Runs on CPU by
default so it never contends with a GPU training run.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

# The c29 ladder's per-rung net_threshold (HP), keyed by the rung's aspect_deg_max. For the honest
# net-ahead% we score each nominal aspect at the threshold of the rung whose frontier it sits in;
# aspects below the first ceiling use the easiest (rung-0) threshold.
_ASPECT_THRESHOLD = {60.0: 5.0, 65.0: 5.0, 70.0: 4.0, 75.0: 4.0, 80.0: 3.0, 85.0: 3.0, 90.0: 3.0}
_LOSE_EVENTS = {"killed", "death", "trade", "hard_deck", "left_bubble"}


def _threshold_for(aspect: float) -> float:
    for hi in sorted(_ASPECT_THRESHOLD):
        if aspect <= hi:
            return _ASPECT_THRESHOLD[hi]
    return _ASPECT_THRESHOLD[90.0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Honest native-T3 verdict across the aspect hemisphere.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=24, help="episodes per (aspect, mode)")
    p.add_argument(
        "--aspects",
        default="0,45,60,75,80,85,90",
        help="comma list of nominal start aspects in deg (0=dead-ahead/offensive, 90=beam)",
    )
    p.add_argument(
        "--modes",
        default="greedy,mpc",
        help="comma list of acting modes to sweep: 'greedy' (policy prior) and/or 'mpc' (MPPI)",
    )
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument(
        "--me-dalt-ft",
        type=float,
        default=0.0,
        help="LEARNER altitude-advantage in feet above the T3 (entry energy 'perch'); 0 = legacy co-alt",
    )
    p.add_argument(
        "--opp-heading",
        type=float,
        default=None,
        help="opponent ABSOLUTE start heading deg; 180 = head-on nose-to-nose merge (default None = co-heading chase)",
    )
    p.add_argument(
        "--opp-tier",
        type=int,
        default=0,
        help="opponent AI tier, independent of geometry (curriculum-53 merge-t123 lever): "
        "1=Tier1 straight NON-shooter, 2=Tier2 breaker NON-shooter, 0/3=Tier3 the full BFM shooter. "
        "Use 1 to measure vs the actual rung-0 training opponent (default 0 = full scripted T3, the north-star).",
    )
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _roll(agent, env, base, C, episodes: int, seed0: int) -> list[dict]:
    rows: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0 = True
        event = None
        terminated = truncated = False
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, _r, terminated, truncated, info = env.step(a)
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        dealt = C.HP_MAX - float(opp.health)
        taken = C.HP_MAX - float(me.health)
        ammo_left = float(me.ammo)
        rows.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "event": event,
                "dealt": dealt,
                "taken": taken,
                "net": dealt - taken,
                "me_alive": bool(me.alive),
                # reward-v10 ammo-discipline signal: how many rounds it burned this episode, and
                # (on kills) how many it had left — the c29 champion emptied its whole mag in the
                # first pass, so a HIGHER left@kill / LOWER spend means the burst-ramp is biting.
                "ammo_spent": float(C.AMMO_MAX) - ammo_left,
                "ammo_left": ammo_left,
            }
        )
    return rows


def _net_ahead(r: dict, threshold: float) -> bool:
    if r["event"] == "kill":
        return True
    if r["event"] in _LOSE_EVENTS:
        return False
    return r["net"] >= threshold


def main() -> None:
    args = parse_args(argv=None)
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stage_idx = int(sd.get("stage_idx", -1))
    collected = int(sd.get("collected", -1))
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(
        f"VERDICT {args.checkpoint} (recorded stage_idx={stage_idx}, collected={collected:,}) "
        f"vs native T3 | sep {args.separation:.0f}m ar{args.action_repeat} "
        f"mt{args.max_time_s:.0f}s jitter{args.jitter:g}"
        f"{f' perch+{args.me_dalt_ft:.0f}ft' if args.me_dalt_ft else ''} | {args.episodes} eps/cell"
    )

    for mode in modes:
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = mode == "mpc"
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        print(f"\n=== mode: {mode} (mpc={cfg.mpc}) ===")
        print(
            f"{'aspect':>6} {'N':>3} {'kill':>4} {'death':>5} {'deck':>4} {'trade':>5} "
            f"{'t/out':>5} {'dealt':>6} {'taken':>6} {'NET':>7} {'net-ahead%':>10} "
            f"{'spent':>6} {'left@kill':>9}"
        )
        overall: list[dict] = []
        for aspect in aspects:
            env = make_tdmpc_env(
                3,
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
                **({"jitter": args.jitter} if args.jitter > 0 else {}),
                **({"opp_heading_deg": args.opp_heading} if args.opp_heading is not None else {}),
                **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
                **({"me_dalt_ft": args.me_dalt_ft} if args.me_dalt_ft else {}),
            )
            base = env.unwrapped
            rows = _roll(agent, env, base, C, args.episodes, args.seed + int(aspect) * 1000)
            env.close()
            overall.extend(rows)
            thr = _threshold_for(aspect)
            n = len(rows)
            kill = sum(r["outcome"] == "kill" for r in rows)
            death = sum(r["outcome"] == "death" for r in rows)
            deck = sum(r["outcome"] == "deck" for r in rows)
            trade = sum(r["event"] == "trade" for r in rows)
            tout = sum(r["outcome"] == "timeout" for r in rows)
            dealt = statistics.fmean(r["dealt"] for r in rows)
            taken = statistics.fmean(r["taken"] for r in rows)
            net = statistics.fmean(r["net"] for r in rows)
            ahead = 100.0 * statistics.fmean(_net_ahead(r, thr) for r in rows)
            spent = statistics.fmean(r["ammo_spent"] for r in rows)
            kill_rows = [r for r in rows if r["outcome"] == "kill"]
            left_at_kill = statistics.fmean(r["ammo_left"] for r in kill_rows) if kill_rows else float("nan")
            print(
                f"{aspect:>6.0f} {n:>3} {kill:>4} {death:>5} {deck:>4} {trade:>5} "
                f"{tout:>5} {dealt:>6.1f} {taken:>6.1f} {net:>+7.1f} {ahead:>9.0f}% "
                f"{spent:>6.0f} {left_at_kill:>9.0f}"
            )
        # per-mode rollup across all aspects
        n = len(overall)
        kill = sum(r["outcome"] == "kill" for r in overall)
        loss = sum(r["outcome"] in ("death", "deck") for r in overall)
        net = statistics.fmean(r["net"] for r in overall)
        ahead = 100.0 * statistics.fmean(
            _net_ahead(r, _threshold_for(90.0)) for r in overall
        )
        print(
            f"  ALL: {n} eps | kills {kill} losses {loss} | mean NET {net:+.1f} | "
            f"net-ahead% (thr3) {ahead:.0f}%"
        )


if __name__ == "__main__":
    main()
