"""Why doesn't our best tracker CLOSE the slow-tail into WEZ? Per-step trace of the baseline through
the neutral head-on, focused on the SLOW-TAIL window (aspect<70 AND rng<1300 AND bandit slow).

The baseline energy-merge probe showed ep1: a moment behind an 89 m/s T3 with closure~0 / lead 5° —
the crossing NULLED — but at 750 m (out of gun range), transient, 0 rounds fired. This traces what
the policy actually COMMANDS in that window (throttle? pull? fire?) so we can see WHY it never drives
that matched slow-tail from 750 m into the ~250 m WEZ and holds it. Prints one row per step while in
the slow-tail window (else a compact '...' skip), per episode.

  cd controllers/rl && PYTHONPATH="vendor:../../crates/astrum-env/python" uv run python \
      _tail_trace_tmp.py --checkpoint runs/BASELINE_merge_survivor_2480k.pt --eps 1,3,5 --max-time-s 180
"""
import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import relative
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

DEG = 180.0 / math.pi


def load_agent(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def trace_ep(agent, env, base, seed, rate_hz, action_repeat):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    step = 0
    dt = action_repeat / rate_hz
    printed_hdr = False
    in_win_prev = False
    event = None
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        asp = rel.aspect * DEG
        in_win = bd.alive and asp < 75.0 and rel.rng < 1500.0 and bd.airspeed < 175.0
        if in_win:
            if not printed_hdr:
                print(f"    {'t':>5} {'rng':>5} {'asp':>4} {'ata':>4} {'lead':>5} {'clo':>5} "
                      f"{'myV':>4} {'bdV':>4} {'myAGL':>5} {'thr':>5} {'ail':>5} {'elev':>5} {'fire':>4}")
                printed_hdr = True
            print(f"    {step*dt:5.1f} {rel.rng:5.0f} {asp:4.0f} {rel.ata*DEG:4.0f} "
                  f"{rel.lead_error*DEG:5.0f} {rel.closure:5.0f} {me.airspeed:4.0f} {bd.airspeed:4.0f} "
                  f"{me.alt:5.0f} {a[3]:+5.2f} {a[0]:+5.2f} {a[1]:+5.2f} {'FIRE' if a[4]>=0.5 else '':>4}")
        elif in_win_prev:
            print("    ...")
        in_win_prev = in_win
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        step += 1
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    print(f"  => event={event} dealt={C.HP_MAX-bd.health:.0f} taken={C.HP_MAX-me.health:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eps", default="1,3,5", help="comma list of episode indices (seed=2000+idx)")
    ap.add_argument("--aspect", type=float, default=0.0)
    ap.add_argument("--opp-heading", type=float, default=180.0)
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    args = ap.parse_args()

    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(args.checkpoint, device)
    ar, rate = 5, 120.0
    env = make_tdmpc_env(3, posture="offensive", aspect_deg=args.aspect, separation_m=args.sep,
                         action_repeat=ar, frame_stack=1, max_time_s=args.max_time_s,
                         rate_hz=rate, jitter=args.jitter, opp_heading_deg=args.opp_heading)
    base = env.unwrapped
    print(f"TAIL TRACE | {args.checkpoint} | head-on | slow-tail window: asp<70 rng<1300 bdV<150")
    for idx in [int(x) for x in args.eps.split(",") if x.strip()]:
        print(f"== ep {idx} (seed {2000+idx}) ==")
        trace_ep(agent, env, base, 2000 + idx, rate, ar)
    env.close()


if __name__ == "__main__":
    main()
