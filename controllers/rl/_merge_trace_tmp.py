"""Trace the deliverable greedy through a NOSE-TO-NOSE merge to characterize the failure mode.

Nose-to-nose = tier3 offensive, aspect_deg=0 (bandit dead ahead) + opp_heading_deg=180
(bandit pointed back at us) → head-on closure at ~2x speed, neither jet with an edge.

Run in-container:
  cd /workspace/controllers/rl && PYTHONPATH=...:vendor:...python _merge_trace_tmp.py
Emits: a canonical 2s-cadence trace (jitter 0) + per-episode summary stats over a jittered set.
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

CKPT = "runs/curriculum-63/_peak_2480k_coexist.pt"
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
    cfg.mpc = False  # greedy prior = these checkpoints' real mode
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def run_ep(agent, env, base, seed, trace=False):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    event = None
    step = 0
    ar, hz = 5, 120.0
    dt = ar / hz
    # summary trackers
    min_rng, t_min_rng = 1e9, 0.0
    min_aspect, min_ata = 1e9, 1e9
    max_bank_post = 0.0
    merged = False
    fire_cmds = 0
    peak_gq = 0.0
    # crossing-velocity ceiling trackers: the smallest lead_error reached while close
    # (rng<250), and the physical hit-gate miss rng·sin(lead) there (must be <WEZ_HIT_M=8 to hit).
    min_lead_close = 1e9
    missm_at_minlead = 1e9
    if trace:
        print(f"{'t':>5} {'rng':>6} {'asp°':>5} {'ata°':>5} {'lead°':>6} {'miss_m':>7} {'gunq':>5} {'bank°':>6} "
              f"{'pit°':>5} {'alt':>5} {'spd':>4} {'clos':>6} {'thr':>4} {'fire':>4} "
              f"{'myHP':>5} {'opHP':>5}")
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        t_s = step * dt
        if rel.rng < min_rng:
            min_rng, t_min_rng = rel.rng, t_s
        if t_s > t_min_rng and rel.rng > min_rng + 50:
            merged = True  # range opened back up after closest approach
        min_aspect = min(min_aspect, rel.aspect * DEG)
        min_ata = min(min_ata, rel.ata * DEG)
        peak_gq = max(peak_gq, rel.gun_quality)
        if rel.rng < 250.0 and (rel.lead_error * DEG) < min_lead_close:
            min_lead_close = rel.lead_error * DEG
            missm_at_minlead = rel.rng * math.sin(rel.lead_error)
        if merged:
            max_bank_post = max(max_bank_post, abs(me.phi) * DEG)
        # a[4] is AGENT space: map_tdmpc_action gives native = 0.5(a+1) and the env fires at
        # native >= 0.5, i.e. agent >= 0. Testing >= 0.5 here undercounts every trigger-down figure.
        if a[4] >= 0.0:
            fire_cmds += 1
        if trace and step % 48 == 0:  # every 2 s
            _missm = rel.rng * math.sin(rel.lead_error)
            print(f"{t_s:5.1f} {rel.rng:6.0f} {rel.aspect*DEG:5.0f} {rel.ata*DEG:5.0f} "
                  f"{rel.lead_error*DEG:6.0f} {_missm:7.0f} {rel.gun_quality:5.2f} "
                  f"{me.phi*DEG:6.0f} {me.theta*DEG:5.0f} {me.alt:5.0f} {me.airspeed:4.0f} "
                  f"{rel.closure:6.0f} {a[3]:4.2f} {a[4]:4.1f} {me.health:5.0f} {bd.health:5.0f}")
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        step += 1
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    return dict(event=event, dealt=C.HP_MAX - bd.health, taken=C.HP_MAX - me.health,
                min_rng=min_rng, t_min_rng=t_min_rng, min_aspect=min_aspect,
                min_ata=min_ata, max_bank_post=max_bank_post, fire_cmds=fire_cmds,
                peak_gq=peak_gq, min_lead_close=min_lead_close,
                missm_at_minlead=missm_at_minlead, end_alt=me.alt, t_end=step * dt)


def main():
    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(device)

    print("=" * 88)
    print("CANONICAL TRACE (jitter 0) — nose-to-nose merge, deliverable greedy")
    print("  asp°=my angle off HIS tail (low=behind him) | ata°=my nose off HIM (low=pointing at him)")
    print("=" * 88)
    env = make_env(0.0)
    base = env.unwrapped
    s = run_ep(agent, env, base, seed=0, trace=True)
    env.close()
    print(f"  -> event={s['event']} dealt={s['dealt']:.0f} taken={s['taken']:.0f} "
          f"| closest={s['min_rng']:.0f}m @ {s['t_min_rng']:.1f}s "
          f"| best asp={s['min_aspect']:.0f}° best ata={s['min_ata']:.0f}° "
          f"| min lead(close)={s['min_lead_close']:.0f}° miss@minlead={s['missm_at_minlead']:.0f}m (hit<8m) "
          f"| max bank post-merge={s['max_bank_post']:.0f}° | fire_cmds={s['fire_cmds']} "
          f"| peak gun_q={s['peak_gq']:.2f} | end_alt={s['end_alt']:.0f}m t_end={s['t_end']:.0f}s")

    print()
    print("=" * 88)
    print("JITTERED SET (jitter 0.5) — per-episode summary")
    print("=" * 88)
    print(f"{'ep':>3} {'event':>8} {'dealt':>5} {'taken':>5} {'closest_m':>9} {'t_merge':>7} "
          f"{'best_asp':>8} {'best_ata':>8} {'bank_post':>9} {'fireN':>5} {'gunq':>4} {'end_alt':>7}")
    env = make_env(0.5)
    base = env.unwrapped
    agg = []
    for ep in range(8):
        s = run_ep(agent, env, base, seed=1000 + ep, trace=False)
        agg.append(s)
        print(f"{ep:>3} {s['event'] or 'none':>8} {s['dealt']:5.0f} {s['taken']:5.0f} "
              f"{s['min_rng']:9.0f} {s['t_min_rng']:7.1f} {s['min_aspect']:8.0f} "
              f"{s['min_ata']:8.0f} {s['max_bank_post']:9.0f} {s['fire_cmds']:5d} "
              f"{s['peak_gq']:4.2f} {s['end_alt']:7.0f}")
    env.close()
    n = len(agg)
    print(f"\nAGG over {n} jittered eps: "
          f"mean best_asp={np.mean([a['min_aspect'] for a in agg]):.0f}° "
          f"mean best_ata={np.mean([a['min_ata'] for a in agg]):.0f}° "
          f"mean min_lead(close)={np.mean([a['min_lead_close'] for a in agg]):.0f}° "
          f"mean miss@minlead={np.mean([a['missm_at_minlead'] for a in agg]):.0f}m (hit<8m) "
          f"mean closest={np.mean([a['min_rng'] for a in agg]):.0f}m "
          f"mean bank_post={np.mean([a['max_bank_post'] for a in agg]):.0f}° "
          f"total fire_cmds={sum(a['fire_cmds'] for a in agg)} "
          f"kills={sum(a['event']=='kill' for a in agg)} "
          f"losses={sum(a['event'] in ('death','killed','hard_deck','trade') for a in agg)}")


if __name__ == "__main__":
    main()
