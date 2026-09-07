"""Trace the T3 OPPONENT through a nose-to-nose merge to find exploitable habits.

We fly our BASELINE greedy as the learner (a real fight) and log what the SCRIPTED
T3 does, to answer: (1) does it break the SAME direction every time (predictable →
we can pre-plan a lead turn)? (2) does it fight FLAT or use the VERTICAL (if flat,
a yo-yo/egg in the vertical is a free edge it can't counter)? (3) does it keep its
ENERGY or zoom-bleed to slow like we do? (4) how hard/fast does it TURN? (5) does it
ever get a real GUN SOLUTION on us (dangerous) or is it also stuck at a draw?

Run in-container:
  cd /workspace/controllers/rl && PYTHONPATH=...:vendor:... python _t3_trace_tmp.py
"""
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

CKPT = "runs/BASELINE_merge_survivor_2480k.pt"
DEG = 180.0 / math.pi


def make_env(jitter):
    kw = dict(posture="offensive", aspect_deg=0.0, opp_heading_deg=180.0,
              separation_m=1500.0, action_repeat=5, frame_stack=1,
              max_time_s=120.0, rate_hz=120.0)
    if jitter > 0:
        kw["jitter"] = jitter
    return make_tdmpc_env(3, **kw)


def load_agent(device):
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def run_ep(agent, env, base, seed, trace=False):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    step = 0
    ar, hz = 5, 120.0
    dt = ar / hz
    prev_psi = None
    prev_alt = None
    # trackers
    first_break_dir = 0        # sign of T3 bank at first |bank|>25°
    t_first_break = 0.0
    max_pitch = 0.0            # |theta| — vertical usage
    max_climb = 0.0            # |d alt/dt| — vertical usage
    max_turnrate = 0.0         # deg/s (horizontal heading change)
    turnrates = []
    min_spd, max_alt, min_alt = 1e9, -1e9, 1e9
    t3_max_gunq = 0.0          # T3's gun_quality ON US
    t3_solution_steps = 0      # steps T3 had a physical hit (rng·sin(lead)<8m in range)
    if trace:
        print(f"{'t':>5} {'ourPsi':>6} {'t3Psi':>6} {'t3bnk':>6} {'t3pit':>6} "
              f"{'trn/s':>6} {'t3alt':>6} {'t3spd':>5} {'clm/s':>6} "
              f"{'t3ata':>6} {'t3gq':>5} {'t3HP':>4}")
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        t3_on_us = relative(bd, me)          # T3 looking at us
        t_s = step * dt
        # T3 horizontal turn rate (deg/s) from heading change
        psi_d = bd.psi * DEG
        turnrate = 0.0
        climb = 0.0
        if prev_psi is not None:
            turnrate = wrap180(psi_d - prev_psi) / dt
            climb = (bd.alt - prev_alt) / dt
        prev_psi, prev_alt = psi_d, bd.alt
        # first break
        if first_break_dir == 0 and abs(bd.phi) * DEG > 25.0:
            first_break_dir = 1 if bd.phi > 0 else -1
            t_first_break = t_s
        max_pitch = max(max_pitch, abs(bd.theta) * DEG)
        max_climb = max(max_climb, abs(climb))
        if t_s > 0.5:  # skip the first spin-up step's bogus rate
            max_turnrate = max(max_turnrate, abs(turnrate))
            turnrates.append(abs(turnrate))
        min_spd = min(min_spd, bd.airspeed)
        max_alt, min_alt = max(max_alt, bd.alt), min(min_alt, bd.alt)
        t3_max_gunq = max(t3_max_gunq, t3_on_us.gun_quality)
        if t3_on_us.in_range and (t3_on_us.rng * math.sin(t3_on_us.lead_error) < C.WEZ_HIT_M):
            t3_solution_steps += 1
        if trace and step % 48 == 0:
            print(f"{t_s:5.1f} {me.psi*DEG:6.0f} {bd.psi*DEG:6.0f} {bd.phi*DEG:6.0f} "
                  f"{bd.theta*DEG:6.0f} {turnrate:6.1f} {bd.alt:6.0f} {bd.airspeed:5.0f} "
                  f"{climb:6.1f} {t3_on_us.ata*DEG:6.0f} {t3_on_us.gun_quality:5.2f} {bd.health:4.0f}")
        obs, _r, terminated, truncated, info = env.step(a)
        step += 1
        if terminated or truncated:
            break
    turnrates.sort()
    # sustained turn rate = 90th percentile (ignore transient spikes)
    sustained = turnrates[int(0.9 * len(turnrates))] if turnrates else 0.0
    return dict(first_break_dir=first_break_dir, t_first_break=t_first_break,
                max_pitch=max_pitch, max_climb=max_climb, max_turnrate=max_turnrate,
                sustained_turnrate=sustained, min_spd=min_spd, alt_range=max_alt - min_alt,
                min_alt=min_alt, max_alt=max_alt, t3_max_gunq=t3_max_gunq,
                t3_solution_steps=t3_solution_steps)


def main():
    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(device)

    print("=" * 96)
    print("CANONICAL T3 TRACE (jitter 0) — what the scripted T3 does through the merge")
    print("  t3bnk=T3 bank (sign=turn dir) | t3pit=pitch | trn/s=horiz turn rate | clm/s=climb rate")
    print("  t3ata=T3 nose off US (low=T3 pointing at us) | t3gq=T3 gun quality ON US")
    print("=" * 96)
    env = make_env(0.0)
    base = env.unwrapped
    s = run_ep(agent, env, base, seed=0, trace=True)
    env.close()
    print(f"  -> first break: {'RIGHT' if s['first_break_dir']>0 else 'LEFT'} @ {s['t_first_break']:.1f}s "
          f"| max pitch={s['max_pitch']:.0f}° max climb={s['max_climb']:.0f}m/s "
          f"| turn rate: peak={s['max_turnrate']:.1f}°/s sustained={s['sustained_turnrate']:.1f}°/s "
          f"| min spd={s['min_spd']:.0f}m/s alt band={s['alt_range']:.0f}m "
          f"| T3 max gun_q ON US={s['t3_max_gunq']:.2f} solution steps={s['t3_solution_steps']}")

    print()
    print("=" * 96)
    print("JITTERED SET (jitter 0.5) — is the T3 break PREDICTABLE? does it use the VERTICAL? keep ENERGY?")
    print("=" * 96)
    print(f"{'ep':>3} {'break':>6} {'t_brk':>6} {'maxPit':>7} {'maxClm':>7} {'peakTR':>7} "
          f"{'sustTR':>7} {'minSpd':>7} {'altBand':>8} {'t3gqUS':>7} {'solSt':>6}")
    env = make_env(0.5)
    base = env.unwrapped
    agg = []
    for ep in range(10):
        s = run_ep(agent, env, base, seed=1000 + ep, trace=False)
        agg.append(s)
        print(f"{ep:>3} {('RIGHT' if s['first_break_dir']>0 else 'LEFT'):>6} {s['t_first_break']:6.1f} "
              f"{s['max_pitch']:7.0f} {s['max_climb']:7.0f} {s['max_turnrate']:7.1f} "
              f"{s['sustained_turnrate']:7.1f} {s['min_spd']:7.0f} {s['alt_range']:8.0f} "
              f"{s['t3_max_gunq']:7.2f} {s['t3_solution_steps']:6d}")
    env.close()
    n = len(agg)
    n_right = sum(a['first_break_dir'] > 0 for a in agg)
    print(f"\nAGG over {n} eps:")
    print(f"  BREAK DIRECTION: {n_right} RIGHT / {n - n_right} LEFT  "
          f"({'PREDICTABLE' if n_right in (0, n) else 'mixed/reactive'})")
    print(f"  VERTICAL use:   mean max pitch={np.mean([a['max_pitch'] for a in agg]):.0f}°  "
          f"mean max climb={np.mean([a['max_climb'] for a in agg]):.0f} m/s  "
          f"({'stays FLAT (exploit the vertical!)' if np.mean([a['max_pitch'] for a in agg]) < 25 else 'uses the vertical'})")
    print(f"  TURN perf:      mean peak={np.mean([a['max_turnrate'] for a in agg]):.1f}°/s  "
          f"mean sustained={np.mean([a['sustained_turnrate'] for a in agg]):.1f}°/s")
    print(f"  ENERGY:         mean min speed={np.mean([a['min_spd'] for a in agg]):.0f} m/s  "
          f"mean alt band={np.mean([a['alt_range'] for a in agg]):.0f} m  "
          f"({'T3 also zoom-bleeds' if np.mean([a['min_spd'] for a in agg]) < 140 else 'T3 KEEPS energy'})")
    print(f"  DANGER to us:   mean T3 max gun_q on us={np.mean([a['t3_max_gunq'] for a in agg]):.2f}  "
          f"total solution steps={sum(a['t3_solution_steps'] for a in agg)}  "
          f"({'T3 also has NO shot (mutual draw)' if sum(a['t3_solution_steps'] for a in agg) == 0 else 'T3 CAN shoot us'})")


if __name__ == "__main__":
    main()
