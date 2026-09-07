"""Per-step FINISH diagnostic: why does the greedy policy stall ~12-18 HP short of the
kill at aspect >=50 deg (deals 82-88 HP, times out, 400+ rounds unspent)?

    cd controllers/rl && uv run python _diag_finish.py --checkpoint runs/curriculum-41/_snap_4450k.pt --aspects 45,55,60

Rolls deterministic greedy episodes vs the native T3 (same env as _verdict_tmp.py) and, for the
TERMINAL regime (after the bandit first drops below --finish-hp), measures whether the agent:
  (a) STILL gets gun solutions but declines to FIRE      -> fire-discipline over-suppression
  (b) LOSES the bandit (range opens, no solution)        -> re-acquisition / merge geometry
  (c) FIRES in-solution rounds that do NOT connect        -> terminal aim precision
Distinguishes by: late max gun_quality, late rounds fired, late in-solution rounds vs late HP dealt,
late mean range, and time-in-solution while the bandit is low.
"""

from __future__ import annotations

import argparse
import statistics
from math import degrees, sin

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.eval_policy import _classify


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-step finish diagnostic vs native T3.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--aspects", default="45,55,60")
    p.add_argument("--finish-hp", type=float, default=25.0, help="bandit HP that starts the terminal regime")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _roll_ep(agent, env, base, C, seed, finish_hp, dt_step):
    from astrum_dogfight.geometry import relative

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    event = None
    # NOTE: base._states[...] returns a FRESH snapshot object each access (not a live
    # reference), so we MUST re-fetch me/opp every step, not cache them before the loop.
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]

    min_bhp = C.HP_MAX
    # terminal-regime accumulators (after bandit first < finish_hp)
    in_term = False
    term_rounds = 0
    term_rounds_insol = 0
    term_hp_start = None
    term_hp_end = None
    term_maxq = 0.0
    term_q_samples = []
    term_rng_samples = []
    term_insol_ticks = 0
    term_ticks = 0
    step_i = 0
    prev_ammo = float(me.ammo)
    prev_bhp = float(opp.health)

    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, terminated, truncated, info = env.step(a)
        step_i += 1
        me = base._states[base.learner_id]  # re-fetch: fresh snapshot each step
        opp = base._states[base.opponent_id]
        bhp = float(opp.health)
        ammo = float(me.ammo)
        rounds = max(0, int(prev_ammo) - int(ammo))
        rg = relative(me, opp)
        insol = bool(rg.in_range and (rg.rng * sin(rg.lead_error) < C.WEZ_HIT_M))
        min_bhp = min(min_bhp, bhp)

        if not in_term and prev_bhp >= finish_hp and bhp < finish_hp:
            in_term = True
            term_hp_start = bhp
        if in_term and opp.alive:
            term_ticks += 1
            term_rounds += rounds
            if insol:
                term_rounds_insol += rounds
                term_insol_ticks += 1
            term_maxq = max(term_maxq, rg.gun_quality)
            term_q_samples.append(rg.gun_quality)
            term_rng_samples.append(rg.rng)
            term_hp_end = bhp

        if info.get("event"):
            event = info["event"]
        prev_ammo = ammo
        prev_bhp = bhp
        if terminated or truncated:
            break

    dealt = C.HP_MAX - float(opp.health)
    taken = C.HP_MAX - float(me.health)
    term_hp_dealt = (term_hp_start - term_hp_end) if (term_hp_start is not None and term_hp_end is not None) else 0.0
    return {
        "outcome": _classify(event, terminated, truncated),
        "event": event,
        "dealt": dealt,
        "taken": taken,
        "min_bhp": min_bhp,
        "ammo_spent": float(C.AMMO_MAX) - float(me.ammo),
        "reached_term": in_term,
        "term_time_s": term_ticks * dt_step,
        "term_rounds": term_rounds,
        "term_rounds_insol": term_rounds_insol,
        "term_hp_dealt": term_hp_dealt,
        "term_maxq": term_maxq,
        "term_meanq": statistics.fmean(term_q_samples) if term_q_samples else 0.0,
        "term_mean_rng": statistics.fmean(term_rng_samples) if term_rng_samples else float("nan"),
        "term_insol_frac": (term_insol_ticks / term_ticks) if term_ticks else 0.0,
    }


def main():
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env
    from astrum_rl.tdmpc2.config import TDMPC2Config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    dt_step = args.action_repeat / args.rate_hz
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    print(
        f"FINISH-DIAG {args.checkpoint} (stage_idx={sd.get('stage_idx')}, collected={sd.get('collected'):,}) "
        f"| finish_hp<{args.finish_hp:g} | {args.episodes} eps/aspect | dt_step={dt_step*1000:.0f}ms"
    )
    hdr = (
        f"{'asp':>4} {'kill':>4} {'t/o':>3} {'dealt':>6} {'min_bhp':>7} {'spent':>6} | "
        f"{'reachTerm':>9} {'termT_s':>7} {'termRnd':>7} {'inSol':>6} {'termHP':>6} "
        f"{'maxQ':>5} {'meanQ':>5} {'mRng_m':>6} {'inSolFrac':>9}"
    )
    print(hdr)
    for aspect in aspects:
        env = make_tdmpc_env(
            3, posture="offensive", aspect_deg=aspect, separation_m=args.separation,
            action_repeat=args.action_repeat, frame_stack=args.frame_stack,
            max_time_s=args.max_time_s, rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
        )
        base = env.unwrapped
        rows = [
            _roll_ep(agent, env, base, C, args.seed + int(aspect) * 1000 + ep, args.finish_hp, dt_step)
            for ep in range(args.episodes)
        ]
        env.close()
        n = len(rows)
        kill = sum(r["outcome"] == "kill" for r in rows)
        tout = sum(r["outcome"] == "timeout" for r in rows)
        dealt = statistics.fmean(r["dealt"] for r in rows)
        min_bhp = statistics.fmean(r["min_bhp"] for r in rows)
        spent = statistics.fmean(r["ammo_spent"] for r in rows)
        reached = sum(r["reached_term"] for r in rows)
        term = [r for r in rows if r["reached_term"]]
        def m(k):
            return statistics.fmean(r[k] for r in term) if term else float("nan")
        print(
            f"{aspect:>4.0f} {kill:>4} {tout:>3} {dealt:>6.1f} {min_bhp:>7.1f} {spent:>6.0f} | "
            f"{reached:>4}/{n:<4} {m('term_time_s'):>7.1f} {m('term_rounds'):>7.0f} "
            f"{m('term_rounds_insol'):>6.0f} {m('term_hp_dealt'):>6.1f} "
            f"{m('term_maxq'):>5.2f} {m('term_meanq'):>5.2f} {m('term_mean_rng'):>6.0f} {m('term_insol_frac'):>9.2f}"
        )


if __name__ == "__main__":
    main()
