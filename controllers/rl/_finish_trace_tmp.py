"""TEMP diagnostic: trace ONE greedy champion episode at OFFENSE (aspect 60) to see WHY it
chips ~46 HP then coasts to a timeout without finishing the kill. Logs per 0.5 s: the opponent
T3's FSM state (does it break DEFENSIVE after the first pass?), range/ata/aspect, gun_quality &
in_range (is it in a firing solution?), cumulative rounds fired (did it stop shooting?), and both
HPs. Distinguishes incentive vs skill: does it lose the solution and fail to re-acquire, stop
firing while in solution, or simply never get a second solution vs a defending bandit?

    cd controllers/rl && uv run python -u _finish_trace_tmp.py \
        --checkpoint runs/curriculum-29/run_stage05_t3_700m.pt --aspect 60
"""

from __future__ import annotations

import argparse
from math import degrees

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspect", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=60)  # log cadence in sim ticks (60 = 0.5 s)
    args = ap.parse_args()

    _paths.require_astrum_env()
    from astrum_dogfight.geometry import relative

    from astrum_rl.envs import make_tdmpc_env

    device = resolve_device("cpu")
    torch.set_num_threads(8)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    env = make_tdmpc_env(
        3, posture="offensive", aspect_deg=args.aspect, separation_m=700.0,
        action_repeat=5, frame_stack=1, max_time_s=60.0, rate_hz=120.0,
        **({"jitter": args.jitter} if args.jitter > 0 else {}),
    )
    base = env.unwrapped

    obs, _ = env.reset(seed=args.seed)
    agent._prev_mean.zero_()
    ammo0 = float(base._states[base.learner_id].ammo)
    print(f"aspect {args.aspect:g} jitter {args.jitter:g} seed {args.seed}  ammo0={ammo0:.0f}")
    print(
        f"{'t':>5} {'oppfsm':>11} {'rng':>6} {'ata':>5} {'asp':>5} {'gq':>4} {'inR':>3} "
        f"{'fired':>5} {'meHP':>5} {'opHP':>5} | {'ail':>5} {'elev':>5} {'thr':>5} {'trig':>4}"
    )
    t0 = True
    step = 0
    last_ophp = 100.0
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rg = relative(me, opp)
        fired = ammo0 - float(me.ammo)
        drop = last_ophp - float(opp.health)
        if step % args.every == 0 or drop > 0.5:  # periodic + whenever the bandit takes a hit
            fsm = getattr(base.opponent, "state", "?")
            print(
                f"{step * 5 / 120.0:5.1f} {str(fsm):>11} {rg.rng:6.0f} {degrees(rg.ata):5.1f} "
                f"{degrees(rg.aspect):5.1f} {rg.gun_quality:4.2f} {int(rg.in_range):>3} "
                f"{fired:5.0f} {me.health:5.0f} {opp.health:5.0f} | "
                f"{a[0]:+5.2f} {a[1]:+5.2f} {a[3]:+5.2f} {a[4]:+4.1f}"
            )
        last_ophp = float(opp.health)
        obs, _r, terminated, truncated, info = env.step(a)
        step += 1
        if terminated or truncated:
            me = base._states[base.learner_id]
            opp = base._states[base.opponent_id]
            print(
                f"END t={step * 5 / 120.0:.1f}s event={info.get('event')} "
                f"fired={ammo0 - float(me.ammo):.0f} meHP={me.health:.0f} opHP={opp.health:.0f}"
            )
            break
    env.close()


if __name__ == "__main__":
    main()
