"""Greedy evaluation of a trained TD-MPC2 checkpoint against a scripted tier.

    cd controllers/rl
    uv run python -m astrum_rl.tdmpc2.eval_policy \
        --checkpoint runs/curriculum-01/run_end_tier2.pt --tier 2 --episodes 5

Loads a checkpoint (the embedded `cfg` fixes the model + whether it plans with MPC),
runs `--episodes` deterministic (`eval_mode=True`) rollouts against the tier's scripted
opponent, and reports the *actual* per-episode outcome — terminal event, both jets' final
HP, damage dealt/taken, sim seconds — plus an aggregate win/draw/loss breakdown. Unlike the
training log (which prints only a 20-episode mean return of the *exploring* policy), this
reads ground-truth health straight off the env so "what did it actually learn" is measured,
not inferred. Runs on CPU by default so it never contends with a GPU training run.

Outcome labels: **kill** (bandit destroyed, we live) · **death** (we're destroyed) ·
**trade** (both die same tick) · **deck** (we bust the hard deck) · **bubble** (we left the
arena) · **timeout** (clock ran out — a draw). Win = kill; loss = death/deck; draw = the rest.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics

import numpy as np
import torch

from .. import _paths
from .agent import TDMPC2, resolve_device
from .config import TDMPC2Config


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a TD-MPC2 checkpoint vs a scripted tier.")
    p.add_argument("--checkpoint", required=True, help="path to a saved run.pt")
    p.add_argument("--tier", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument(
        "--separation",
        type=float,
        default=0.0,
        help="opponent spawn separation in m (0 = tier default; set to match a curriculum stage)",
    )
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument(
        "--posture",
        default="offensive",
        choices=("offensive", "defensive"),
        help="tier-3 only: 'offensive' = bot on the learner's six (defensive fight, survive-gated); "
        "'defensive' = bot ahead / learner on its tail (offensive chase). Ignored for tier 1/2.",
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=-1.0,
        help="tier-3 offensive posture only: bot bearing from the learner's nose in deg "
        "(180=dead six, 90=beam, 0=dead ahead). <0 = scenario default (180). Ignored otherwise.",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="per-episode start-geometry jitter ∈ [0,1] (match training to measure generalisation; "
        "0 = fixed geometry for a canonical acceptance number)",
    )
    p.add_argument("--no-mpc", action="store_true", help="force policy-prior acting (override cfg)")
    p.add_argument("--mpc", action="store_true", help="force MPPI-planner acting (override cfg)")
    p.add_argument(
        "--device", default="cpu", help="'cpu' (default; won't fight a GPU run) | 'cuda'"
    )
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--telemetry-out",
        default="",
        help="write per-control-step telemetry of the FIRST episode to this CSV path",
    )
    return p.parse_args(argv)


def _classify(event: str | None, terminated: bool, truncated: bool) -> str:
    """Map the terminal (info['event'], terminated, truncated) to an outcome label."""
    if event in ("kill", "death", "trade", "hard_deck", "left_bubble"):
        return {"hard_deck": "deck", "left_bubble": "bubble"}.get(event, event)
    if event == "killed":  # reward.py uses "killed" for the learner's own death
        return "death"
    if event == "conversion":  # reward-v17: the merge-converter M reached its rear-quarter hand-off
        return "convert"
    if terminated:  # terminal with no recognized event tag — shouldn't happen, flag it
        return "terminated?"
    if truncated:
        return "timeout"
    return "unknown"


def _telemetry_row(step: int, action: np.ndarray, base, sim_time_s: float) -> dict:
    """One per-control-step telemetry sample: agent command + learner kinematics + range.

    `action` is the raw `[-1,1]^5` agent action ([ail, elev, rud, throttle, trigger]);
    trigger fires the gun when its value >= 0 (native >= 0.5). Kinematics come straight
    off the env's post-step `AState` for the learner; `range_m` is the slant range to the
    opponent. Angles are reported in degrees / deg·s⁻¹ for readability.
    """
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    return {
        "step": step,
        "t_s": round(sim_time_s, 3),
        "airspeed_mps": round(me.airspeed, 2),
        "alt_m": round(me.alt, 1),
        "nz_g": round(me.nz, 3),
        "phi_deg": round(math.degrees(me.phi), 2),
        "theta_deg": round(math.degrees(me.theta), 2),
        "psi_deg": round(math.degrees(me.psi), 2),
        "alpha_deg": round(math.degrees(me.alpha), 2),
        "beta_deg": round(math.degrees(me.beta), 2),
        "p_dps": round(math.degrees(me.p), 2),
        "q_dps": round(math.degrees(me.q), 2),
        "r_dps": round(math.degrees(me.r), 2),
        "ail_cmd": round(float(a[0]), 4),
        "elev_cmd": round(float(a[1]), 4),
        "rud_cmd": round(float(a[2]), 4),
        "thr_cmd": round(float(a[3]), 4),
        "trig_cmd": round(float(a[4]), 4),
        "firing": int(a[4] >= 0.0),
        "ammo": int(me.ammo),
        "range_m": round(float(np.linalg.norm(opp.pos - me.pos)), 1),
    }


def run(argv=None) -> dict:
    args = parse_args(argv)
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from ..envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    if args.no_mpc:
        cfg.mpc = False
    elif args.mpc:
        cfg.mpc = True
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    env = make_tdmpc_env(
        args.tier,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        max_time_s=args.max_time_s,
        rate_hz=args.rate_hz,
        **({"separation_m": args.separation} if args.separation > 0 else {}),
        **({"jitter": args.jitter} if args.jitter > 0 else {}),
        **(
            {
                "posture": args.posture,
                **(
                    {"aspect_deg": args.aspect}
                    if args.posture == "offensive" and args.aspect >= 0.0
                    else {}
                ),
            }
            if args.tier == 3
            else {}
        ),
    )
    base = env.unwrapped  # the DogfightGymEnv (carries _states + ids)

    print(
        f"eval: {args.checkpoint} | tier {args.tier}"
        f"{'/' + args.posture if args.tier == 3 else ''}"
        f"{f'/aspect{args.aspect:.0f}' if args.tier == 3 and args.posture == 'offensive' and args.aspect >= 0 else ''}"
        f" | mpc {cfg.mpc} | device {device} | "
        f"episodes {args.episodes} | action_repeat {args.action_repeat}"
    )
    print(
        f"{'ep':>3} {'outcome':>10} {'return':>9} {'ctl_steps':>9} {'t_s':>6} "
        f"{'me_hp':>6} {'opp_hp':>6} {'dmg_dealt':>9} {'dmg_taken':>9}"
    )

    rows: list[dict] = []
    tel_rows: list[dict] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        agent._prev_mean.zero_()  # clear the MPPI warm-start between episodes
        t0 = True
        ep_ret, ep_len = 0.0, 0
        event = None
        terminated = truncated = False
        capture = bool(args.telemetry_out) and ep == 0
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            if info.get("event"):
                event = info["event"]  # last recognized terminal/bubble tag wins
            if capture:
                tel_rows.append(_telemetry_row(ep_len, a, base, float(info.get("sim_time_s", 0.0))))
            if terminated or truncated:
                break

        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        me_hp = float(me.health)
        opp_hp = float(opp.health)
        outcome = _classify(event, terminated, truncated)
        row = {
            "outcome": outcome,
            "return": ep_ret,
            "ctl_steps": ep_len,
            "sim_time_s": float(info.get("sim_time_s", float("nan"))),
            "me_hp": me_hp,
            "opp_hp": opp_hp,
            "dmg_dealt": C.HP_MAX - opp_hp,
            "dmg_taken": C.HP_MAX - me_hp,
            "me_alive": bool(me.alive),
            "opp_alive": bool(opp.alive),
        }
        rows.append(row)
        print(
            f"{ep:>3} {outcome:>10} {ep_ret:>9.2f} {ep_len:>9d} {row['sim_time_s']:>6.1f} "
            f"{me_hp:>6.1f} {opp_hp:>6.1f} {row['dmg_dealt']:>9.1f} {row['dmg_taken']:>9.1f}"
        )

    env.close()

    if tel_rows:
        with open(args.telemetry_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tel_rows[0].keys()))
            w.writeheader()
            w.writerows(tel_rows)
        print(f"telemetry: wrote {len(tel_rows)} control steps → {args.telemetry_out}")

    # ---- aggregate ----
    wins = sum(r["outcome"] == "kill" for r in rows)
    losses = sum(r["outcome"] in ("death", "deck") for r in rows)
    draws = len(rows) - wins - losses
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    returns = [r["return"] for r in rows]
    summary = {
        "episodes": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "outcomes": counts,
        "return_mean": statistics.fmean(returns) if returns else float("nan"),
        "return_median": statistics.median(returns) if returns else float("nan"),
        "dmg_dealt_mean": statistics.fmean(r["dmg_dealt"] for r in rows) if rows else float("nan"),
        "dmg_taken_mean": statistics.fmean(r["dmg_taken"] for r in rows) if rows else float("nan"),
    }
    print(
        f"\nsummary: {summary['episodes']} eps | "
        f"W {wins} / D {draws} / L {losses} | outcomes {counts} | "
        f"return mean {summary['return_mean']:.2f} median {summary['return_median']:.2f} | "
        f"dmg dealt {summary['dmg_dealt_mean']:.1f} taken {summary['dmg_taken_mean']:.1f}"
    )
    return summary


if __name__ == "__main__":
    run()
