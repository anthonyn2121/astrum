"""reward-v22 LEADING-INDICATOR probe on the CHASE hot-pass band (co-heading, aggressive T3).

The reward-v22 hypothesis: K_LEAD (a lead-error PBRS, Φ_L=(1+cosθ)/2) fills the gradient hole the
peaked gun_q Gaussian (LEAD_SIGMA=3°) leaves at a 30-57° co-speed crossing, PULLING the policy to
NULL the crossing (match his velocity) so lead_error → small → gun_q climbs → bullets finally hit.
A kill-count verdict lags this by a long way; the earliest signal that reward-v22 is WORKING is the
in-zone lead_error dropping across successive snapshots. So this probe reports, on greedy jitter-0.5
episodes, the metrics INSIDE the exact zone reward-v22 shapes:
    in-zone  ==  aspect <= CONVERT_ASPECT_RAD (50°)  AND  rng <= CONVERT_RANGE_MAX (1000 m)
Per episode + aggregate: min in-zone lead°, peak in-zone gun_q, min in-zone physical miss
rng·sin(lead) (must be < WEZ_HIT_M=8 m to hit), best aspect/ata, dealt/taken, fire count, event.

MEASURED BASELINE (c69/_auto_002 = c72's warm start; aspect55 co-head, 12ep j0.5, 2026-09-01):
  3 kills / 0 loss | dealt 86.3 | minLead(in-zone) 20° | miss 106m | peakGQ 0.00.
  9/12 eps convert to the six (bestAsp 2-3°) + dwell ~300 in-zone steps + deal 52-96 HP but CANNOT
  finish (20° residual lead ⇒ bullets miss); the 3 kills are ~55° gun-snapshots, not tracking kills.
WIN SIGNAL for c72: minLead(in-zone) falling 20°→<10°, peakGQ climbing off 0, miss→<8m, and those
  9 near-miss timeouts tipping into KILLS = the crossing being nulled into a tracking shot.

Run in-container (CPU, so it never contends with the GPU trainer):
  cd controllers/rl && uv run python _lead_probe_tmp.py --checkpoint runs/curriculum-72/run.pt \
      --aspect 55 --episodes 12 --jitter 0.5
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


def make_env(aspect, sep, jitter, opp_heading):
    kw = dict(posture="offensive", aspect_deg=float(aspect), separation_m=float(sep),
              action_repeat=5, frame_stack=1, max_time_s=120.0, rate_hz=120.0)
    if opp_heading is not None:
        kw["opp_heading_deg"] = float(opp_heading)
    if jitter > 0:
        kw["jitter"] = jitter
    return make_tdmpc_env(3, **kw)


def load_agent(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # greedy prior = these checkpoints' real mode
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def run_ep(agent, env, base, seed):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0 = True
    event = None
    step = 0
    min_aspect, min_ata = 1e9, 1e9
    fire_cmds = 0
    # reward-v22 IN-ZONE trackers (aspect<=CONVERT_ASPECT_RAD AND rng<=CONVERT_RANGE_MAX):
    inzone_steps = 0
    min_lead_inzone = 1e9        # deg — the crossing-null indicator
    miss_at_minlead = 1e9        # m   — rng·sin(lead) at that point (hit < WEZ_HIT_M=8 m)
    peak_gq_inzone = 0.0
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        min_aspect = min(min_aspect, rel.aspect * DEG)
        min_ata = min(min_ata, rel.ata * DEG)
        in_zone = rel.aspect <= C.CONVERT_ASPECT_RAD and rel.rng <= C.CONVERT_RANGE_MAX
        if in_zone:
            inzone_steps += 1
            peak_gq_inzone = max(peak_gq_inzone, rel.gun_quality)
            lead_deg = rel.lead_error * DEG
            if lead_deg < min_lead_inzone:
                min_lead_inzone = lead_deg
                miss_at_minlead = rel.rng * math.sin(rel.lead_error)
        # a[4] is AGENT space: map_tdmpc_action gives native = 0.5(a+1) and the env fires at
        # native >= 0.5, i.e. agent >= 0. Testing >= 0.5 here undercounts every trigger-down figure.
        if a[4] >= 0.0:
            fire_cmds += 1
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        step += 1
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    return dict(event=event, dealt=C.HP_MAX - bd.health, taken=C.HP_MAX - me.health,
                min_aspect=min_aspect, min_ata=min_ata, fire_cmds=fire_cmds,
                inzone_steps=inzone_steps, min_lead_inzone=min_lead_inzone,
                miss_at_minlead=miss_at_minlead, peak_gq_inzone=peak_gq_inzone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspect", type=float, default=55.0, help="nominal entry aspect (hot-pass band)")
    ap.add_argument("--sep", type=float, default=700.0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--opp-heading", type=lambda s: None if str(s).lower() == "none" else float(s),
                    default=None, help="None = co-heading chase (default); 180 = head-on merge")
    args = ap.parse_args()

    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(args.checkpoint, device)

    oh = "co-head" if args.opp_heading is None else f"{args.opp_heading:.0f}°"
    print("=" * 100)
    print(f"reward-v22 LEAD PROBE | {args.checkpoint} | aspect {args.aspect:.0f}° | opp {oh} | "
          f"sep {args.sep:.0f}m | greedy jitter {args.jitter} | in-zone = asp<=50° & rng<=1000m")
    print("=" * 100)
    print(f"{'ep':>3} {'event':>8} {'dealt':>5} {'taken':>5} {'bestAsp':>7} {'bestAta':>7} "
          f"{'zSteps':>6} {'minLead°':>8} {'miss_m':>7} {'peakGQ':>6} {'fireN':>5}")
    env = make_env(args.aspect, args.sep, args.jitter, args.opp_heading)
    base = env.unwrapped
    agg = []
    for ep in range(args.episodes):
        s = run_ep(agent, env, base, seed=1000 + ep)
        agg.append(s)
        ml = s["min_lead_inzone"]
        mm = s["miss_at_minlead"]
        print(f"{ep:>3} {s['event'] or 'none':>8} {s['dealt']:5.0f} {s['taken']:5.0f} "
              f"{s['min_aspect']:7.0f} {s['min_ata']:7.0f} {s['inzone_steps']:6d} "
              f"{(ml if ml < 1e8 else float('nan')):8.0f} {(mm if mm < 1e8 else float('nan')):7.0f} "
              f"{s['peak_gq_inzone']:6.2f} {s['fire_cmds']:5d}")
    env.close()

    inz = [a for a in agg if a["inzone_steps"] > 0]
    n = len(agg)
    print("-" * 100)
    if inz:
        print(f"AGG over {n} eps ({len(inz)} reached the zone): "
              f"mean bestAsp={np.mean([a['min_aspect'] for a in agg]):.0f}° "
              f"mean bestAta={np.mean([a['min_ata'] for a in agg]):.0f}° "
              f"mean minLead(in-zone)={np.mean([a['min_lead_inzone'] for a in inz]):.0f}° "
              f"mean miss@minLead={np.mean([a['miss_at_minlead'] for a in inz]):.0f}m (hit<8m) "
              f"mean peakGQ(in-zone)={np.mean([a['peak_gq_inzone'] for a in inz]):.2f}")
    else:
        print(f"AGG over {n} eps: NONE reached the reward-v22 zone (asp<=50° & rng<=1000m)")
    print(f"total fire_cmds={sum(a['fire_cmds'] for a in agg)} "
          f"kills={sum(a['event'] == 'kill' for a in agg)} "
          f"losses={sum(a['event'] in ('death', 'killed', 'hard_deck', 'trade') for a in agg)} "
          f"mean dealt={np.mean([a['dealt'] for a in agg]):.1f} "
          f"mean taken={np.mean([a['taken'] for a in agg]):.1f}")


if __name__ == "__main__":
    main()
