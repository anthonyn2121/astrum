"""curriculum-75 ENERGY-FIGHT probe: the long-range NEUTRAL HEAD-ON merge vs the aggressive T3.

The reframe (user option B): the symmetric co-speed merge KILL is a geometric no-gun-solution IN ONE
PASS ([[merge-crossing-velocity-ceiling]], [[curriculum-74-lead-null-disproof]]). Real pilots SURVIVE
the merge, OUT-ENERGY the adversary over MULTIPLE turns, and finish from the rear quarter once he is
SLOW — a slow bandit is velocity-matchable, so the crossing (and thus the lead) NULLS and the tracking
gun solution finally hits. The buffed aggressive T3 ([[aggressive-t3-break]]) bleeds energy hard in its
85°-bank AoA-limit break, which is the exploit. reward-v24 (K_REL_ENERGY 20→50) makes out-energizing a
first-class objective. This probe reports, on greedy jitter-0.5 head-on episodes (aspect 0, opp_hdg 180,
LONG 180 s), the signals that distinguish a real energy-fight kill from the old crossing draw:

  • event / dealt / taken            — did it survive and finish (vs deck/trade/timeout-draw)?
  • endRelEs / peakRelEs (m)         — specific-energy ADVANTAGE over the T3 (Es_me − Es_bd); >0 = out-energized
  • bandMinV (m/s)                   — did the T3 go SLOW (we made him bleed)? low = the exploit worked
  • bestGQ + its closure/lead/rng    — is the best gun solution a LOW-closure TRACKING shot (crossing nulled)
                                        or a high-closure crossing snapshot (the old draw)?
  • minAsp / minLead(in-zone)        — best rear-quarter position + residual lead there
  • passes                           — # of close passes (rng dipped < 500 m) = how multi-turn the fight was

WIN SIGNAL: survives (low deck), endRelEs > 0 (out-energized the T3), bandMinV low, and a KILL whose
bestGQ moment has LOW closure (|clo| < ~100) + low lead — a tracking shot on a slow T3, NOT a crossing.

Run in-container (CPU):
  cd controllers/rl && PYTHONPATH="vendor:../../crates/astrum-env/python" uv run python \
      _energy_merge_probe_tmp.py --checkpoint runs/curriculum-75/run.pt --episodes 8 --jitter 0.5
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
G = 9.80665


def _es(a, ground_alt=0.0):
    return max(0.0, a.alt - ground_alt) + a.airspeed * a.airspeed / (2.0 * G)


def make_env(aspect, sep, jitter, opp_heading, max_time_s, me_dalt_ft=0.0):
    kw = dict(posture="offensive", aspect_deg=float(aspect), separation_m=float(sep),
              action_repeat=5, frame_stack=1, max_time_s=float(max_time_s), rate_hz=120.0,
              opp_heading_deg=float(opp_heading))
    if jitter > 0:
        kw["jitter"] = jitter
    if me_dalt_ft:
        kw["me_dalt_ft"] = float(me_dalt_ft)  # curriculum-79 perch: learner +Nft above the co-alt T3
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
    min_aspect, min_ata = 1e9, 1e9
    fire_cmds = 0
    peak_rel_es = -1e9
    band_min_v = 1e9
    inzone_min_lead = 1e9
    # best gun-quality moment (in or out of zone) — the shot the policy is actually taking:
    best_gq = -1.0
    best_gq_clo = best_gq_lead = best_gq_rng = float("nan")
    passes, prev_close = 0, False
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        min_aspect = min(min_aspect, rel.aspect * DEG)
        min_ata = min(min_ata, rel.ata * DEG)
        peak_rel_es = max(peak_rel_es, _es(me) - _es(bd))
        band_min_v = min(band_min_v, bd.airspeed)
        if rel.gun_quality > best_gq:
            best_gq = rel.gun_quality
            best_gq_clo, best_gq_lead, best_gq_rng = rel.closure, rel.lead_error * DEG, rel.rng
        in_zone = rel.aspect <= C.CONVERT_ASPECT_RAD and rel.rng <= C.CONVERT_RANGE_MAX
        if in_zone:
            inzone_min_lead = min(inzone_min_lead, rel.lead_error * DEG)
        close = rel.rng < 500.0
        if close and not prev_close:
            passes += 1
        prev_close = close
        # a[4] is AGENT space: map_tdmpc_action gives native = 0.5(a+1) and the env fires at
        # native >= 0.5, i.e. agent >= 0. Testing >= 0.5 here undercounts every trigger-down figure.
        if a[4] >= 0.0:
            fire_cmds += 1
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if terminated or truncated:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    return dict(event=event, dealt=C.HP_MAX - bd.health, taken=C.HP_MAX - me.health,
                end_rel_es=_es(me) - _es(bd), peak_rel_es=peak_rel_es, band_min_v=band_min_v,
                min_aspect=min_aspect, min_ata=min_ata, inzone_min_lead=inzone_min_lead,
                best_gq=best_gq, best_gq_clo=best_gq_clo, best_gq_lead=best_gq_lead,
                best_gq_rng=best_gq_rng, fire_cmds=fire_cmds, passes=passes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspect", type=float, default=0.0, help="head-on merge = 0")
    ap.add_argument("--opp-heading", type=float, default=180.0, help="head-on = 180")
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--me-dalt-ft", type=float, default=0.0,
                    help="LEARNER altitude-advantage 'perch' in feet above the T3 (curriculum-79); 0 = co-alt")
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--jitter", type=float, default=0.5)
    args = ap.parse_args()

    device = resolve_device("cpu")
    torch.set_num_threads(2)
    agent = load_agent(args.checkpoint, device)

    print("=" * 118)
    print(f"ENERGY-MERGE PROBE | {args.checkpoint} | head-on aspect {args.aspect:.0f}° opp {args.opp_heading:.0f}° | "
          f"sep {args.sep:.0f}m | {args.max_time_s:.0f}s | greedy jitter {args.jitter}"
          f"{f' | perch +{args.me_dalt_ft:.0f}ft' if args.me_dalt_ft else ''}")
    print("=" * 118)
    print(f"{'ep':>3} {'event':>9} {'dealt':>5} {'taken':>5} {'endRelEs':>8} {'pkRelEs':>7} {'bandMinV':>8} "
          f"{'minAsp':>6} {'zLead':>6} {'bestGQ':>6} {'gqClo':>6} {'gqLead':>6} {'gqRng':>6} {'pass':>4} {'fire':>5}")
    env = make_env(args.aspect, args.sep, args.jitter, args.opp_heading, args.max_time_s, args.me_dalt_ft)
    base = env.unwrapped
    agg = []
    for ep in range(args.episodes):
        s = run_ep(agent, env, base, seed=2000 + ep)
        agg.append(s)
        zl = s["inzone_min_lead"]
        print(f"{ep:>3} {s['event'] or 'none':>9} {s['dealt']:5.0f} {s['taken']:5.0f} "
              f"{s['end_rel_es']:8.0f} {s['peak_rel_es']:7.0f} {s['band_min_v']:8.0f} "
              f"{s['min_aspect']:6.0f} {(zl if zl < 1e8 else float('nan')):6.0f} "
              f"{s['best_gq']:6.2f} {s['best_gq_clo']:6.0f} {s['best_gq_lead']:6.0f} {s['best_gq_rng']:6.0f} "
              f"{s['passes']:4d} {s['fire_cmds']:5d}")
    env.close()

    n = len(agg)
    kills = sum(a["event"] == "kill" for a in agg)
    losses = sum(a["event"] in ("death", "killed", "hard_deck", "trade") for a in agg)
    print("-" * 118)
    print(f"AGG over {n} eps: kills={kills} losses={losses} "
          f"meanDealt={np.mean([a['dealt'] for a in agg]):.1f} meanTaken={np.mean([a['taken'] for a in agg]):.1f} "
          f"meanEndRelEs={np.mean([a['end_rel_es'] for a in agg]):.0f}m "
          f"(>0 in {sum(a['end_rel_es'] > 0 for a in agg)}/{n}) "
          f"meanBandMinV={np.mean([a['band_min_v'] for a in agg]):.0f}m/s "
          f"meanPeakGQ={np.mean([a['best_gq'] for a in agg]):.2f} meanPasses={np.mean([a['passes'] for a in agg]):.1f}")


if __name__ == "__main__":
    main()
