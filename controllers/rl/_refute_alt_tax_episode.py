"""REFUTATION probe: sum the altitude-coupled reward terms over WHOLE greedy episodes
(the actual training distribution), not an 8 s window from spawn.

Reports per episode: event, duration, min AGL, seconds spent below the SPIRAL band
(2000 m) and below the deck cushion (800 m), and the episode SUM of deck_penalty,
spiral_penalty, energy_reward, rel_energy_reward, plus hittable-tick % and s<500 m.
"""
import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative, sigma_track  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DEG = 180.0 / math.pi
KEYS = ("deck_penalty", "spiral_penalty", "energy_reward", "rel_energy_reward")


def load_agent(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--aspect", type=float, default=0.0)
    ap.add_argument("--opp-heading", type=float, default=180.0)
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--me-dalt-ft", type=float, default=2000.0)
    ap.add_argument("--lateral-m", type=float, default=1400.0)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--block-ft", type=float, default=15000.0)
    a = ap.parse_args()

    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(a.checkpoint, device)

    kw = dict(
        posture="offensive",
        aspect_deg=a.aspect,
        separation_m=a.sep,
        action_repeat=5,
        frame_stack=1,
        max_time_s=a.max_time_s,
        rate_hz=120.0,
        opp_heading_deg=a.opp_heading,
        jitter=a.jitter,
        me_dalt_ft=a.me_dalt_ft,
        alt_ft=a.block_ft,
        outcome_mode=True,
    )
    if a.lateral_m:
        kw["lateral_m_max"] = a.lateral_m
    env = make_tdmpc_env(3, **kw)
    base = env
    while hasattr(base, "env"):
        base = base.env
    dt = 5.0 / 120.0

    print(f"WHOLE-EPISODE altitude tax | {a.checkpoint} | greedy jitter {a.jitter} "
          f"| block {a.block_ft:.0f} ft + perch {a.me_dalt_ft:.0f} ft | lateral {a.lateral_m:.0f} m | {a.max_time_s:.0f} s")
    hdr = (f"{'ep':>3} {'event':>10} {'T':>6} {'minAGL':>7} {'s<2000':>7} {'s<800':>6} "
           f"{'deck':>8} {'spiral':>8} {'energy':>8} {'relE':>8} {'RET':>9} {'|R|':>9} {'hit%':>6} {'s<500':>6} {'dealt':>5}")
    print(hdr)
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        agent._prev_mean.zero_()
        t0 = True
        tot = {k: 0.0 for k in KEYS}
        n = nlow2 = nlow08 = nhit = nclose = 0
        min_agl = 1e9
        event = None
        while True:
            act = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            me = base._states[base.learner_id]
            bd = base._states[base.opponent_id]
            rel = relative(me, bd)
            agl = me.alt  # ground = 0
            min_agl = min(min_agl, agl)
            n += 1
            if agl < C.SPIRAL_ALT_BAND_AGL:
                nlow2 += 1
            if agl < C.DECK_AGL + C.DECK_MARGIN:
                nlow08 += 1
            if rel.in_range and rel.lead_error <= sigma_track(rel.rng):
                nhit += 1
            if rel.rng < 500.0:
                nclose += 1
            obs, _r, term, trunc, info = env.step(act)
            tot["TOTAL"] = tot.get("TOTAL", 0.0) + float(_r)
            tot["absTOTAL"] = tot.get("absTOTAL", 0.0) + abs(float(_r))
            for k in KEYS:
                tot[k] += float(info.get(k, 0.0))
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
        bd = base._states[base.opponent_id]
        print(f"{ep:3d} {str(event):>10} {n * dt:6.1f} {min_agl:7.0f} {nlow2 * dt:7.1f} "
              f"{nlow08 * dt:6.1f} {tot['deck_penalty']:+8.3f} {tot['spiral_penalty']:+8.3f} "
              f"{tot['energy_reward']:+8.3f} {tot['rel_energy_reward']:+8.3f} "
              f"{tot['TOTAL']:+9.2f} {tot['absTOTAL']:+9.2f} "
              f"{100.0 * nhit / max(1, n):6.1f} {nclose * dt:6.1f} {C.HP_MAX - bd.health:5.0f}")


if __name__ == "__main__":
    main()
