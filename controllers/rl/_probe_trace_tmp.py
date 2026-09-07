"""Per-step trace built on the ENERGY-MERGE PROBE's EXACT loop (so it reproduces the probe's episodes,
unlike _tail_trace_tmp.py which diverges — reset(seed=) doesn't reseed the jitter/T3 RNG stream, so
trajectory depends on episode ORDER and harness identity). Runs the probe seeds 2000..2000+(--ep) in
order and dumps per-step rows for the LAST one, focused on the engaged tail (rng<1600 while bandit ahead).

The question this answers: at the near-solution c76 reaches (ep6: T3 slow ~92 m/s, aspect ~10°, lead 3°,
rng ~494m, firing) does it (a) OVERSHOOT — closure flips negative, aspect blows open, flies through — or
(b) STANDOFF — parks at ~500-660m firing, closure hovers ~0, never drives into the ~250m WEZ? Columns:
  t rng asp ata lead clo myV bdV myAGL thr ail elev fire  (thr=a[3] ail=a[0] elev=a[1] fire=a[4])

  cd controllers/rl && PYTHONPATH="vendor:../../crates/astrum-env/python" uv run python \
      _probe_trace_tmp.py --checkpoint runs/curriculum-76/_peak_c76_4730k.pt --ep 6
"""
import argparse
import math

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


def run_ep(agent, env, base, seed, dt, do_trace):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    step = 0
    event = None
    printed = False
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        asp = rel.aspect * DEG
        if do_trace and bd.alive and rel.rng < 1600.0 and asp < 90.0:
            if not printed:
                print(f"    {'t':>5} {'rng':>5} {'asp':>4} {'ata':>4} {'lead':>5} {'clo':>5} "
                      f"{'myV':>4} {'bdV':>4} {'myAGL':>5} {'thr':>5} {'ail':>5} {'elev':>5} {'fire':>4}")
                printed = True
            print(f"    {step*dt:5.1f} {rel.rng:5.0f} {asp:4.0f} {rel.ata*DEG:4.0f} "
                  f"{rel.lead_error*DEG:5.0f} {rel.closure:5.0f} {me.airspeed:4.0f} {bd.airspeed:4.0f} "
                  f"{me.alt:5.0f} {a[3]:+5.2f} {a[0]:+5.2f} {a[1]:+5.2f} {'FIRE' if a[4]>=0.5 else '':>4}")
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        step += 1
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    print(f"  => ep seed {seed} event={event} dealt={C.HP_MAX-bd.health:.0f} taken={C.HP_MAX-me.health:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ep", type=int, default=6, help="target episode index (probe seed 2000+ep); replays 0..ep in order")
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
    dt = ar / rate
    env = make_tdmpc_env(3, posture="offensive", aspect_deg=args.aspect, separation_m=args.sep,
                         action_repeat=ar, frame_stack=1, max_time_s=args.max_time_s,
                         rate_hz=rate, jitter=args.jitter, opp_heading_deg=args.opp_heading)
    base = env.unwrapped
    print(f"PROBE-TRACE | {args.checkpoint} | head-on | replay 0..{args.ep}, trace ep {args.ep}")
    for idx in range(args.ep + 1):
        run_ep(agent, env, base, 2000 + idx, dt, do_trace=(idx == args.ep))
    env.close()


if __name__ == "__main__":
    main()
