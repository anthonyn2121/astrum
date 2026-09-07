"""Measure whether the greedy policy GAINS ENOUGH ENERGY TO REACH + HOLD THE WEZ
against the aggressive scripted Tier-3 — the skill the user asked for
("we just need to find a way to gain enough energy to put us into WEZ").

The chase-t3-killgate ladder trains from a co-altitude / co-speed start and (until
curriculum-69) with the energy reward OFF (K_ENERGY/corner-floor/rel-Es live behind
outcome_mode). Symptom: at 120 s it decks or zoom-stalls (222→80 m/s) instead of keeping
the energy needed to close into and hold a gun solution. Kills are a non-holdable peak, so
kills alone don't show whether energy management is improving. This probe reports the
UNDERLYING energy→WEZ metrics so we can see progress even before the finish converts:

  WEZ REACH/HOLD:
    wez%   = fraction of alive steps with a real gun solution (gun_quality >= --gq-wez)
    peakGQ = best gun_quality reached in the episode (did it ever get a firing solution?)
    inrng% = fraction of alive steps inside gun range (RelGeom.in_range)
    minRng = closest range achieved (m)
  ENERGY:
    vMean/vMin = my airspeed (m/s) mean / minimum over the episode
    stall% = fraction of alive steps below V_CORNER_FLOOR (bled below sustainable maneuver speed)
    dEs    = my specific-energy change over the episode (Es_end - Es_start, m); <0 = bled energy
    relEs  = (Es_me - Es_opp) at the LAST alive step (m); >0 = ended with MORE energy than the T3

Run in-container (greedy, jitter 0.5, aggressive T3):
  cd /workspace/controllers/rl && uv run python _wez_energy_probe_tmp.py \
      --checkpoints runs/curriculum-68/run_stage00_t3_700m.pt=c68kill,runs/curriculum-69/_auto_005.pt=c69 \
      --aspects 45,55,60 --episodes 16 --jitter 0.5 --max-time-s 120

Throwaway diagnostic (untracked), like _verdict_tmp.py / _yoyo_probe_tmp.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

_G = 9.80665


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe energy→WEZ behaviour of a greedy policy vs the aggressive T3.")
    p.add_argument("--checkpoints", required=True, help="comma list of PATH[=LABEL] to A/B")
    p.add_argument("--aspects", default="45,55,60", help="comma list of start aspects (deg)")
    p.add_argument("--episodes", type=int, default=16, help="episodes per aspect")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-tier", type=int, default=0, help="0 = full native (aggressive) Tier-3")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-time-s", type=float, default=120.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--gq-wez", type=float, default=0.30, help="gun_quality above which a step counts as a real WEZ solution")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load(path: str, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # greedy policy-prior mean — matches the verdict harnesses
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent, int(sd.get("stage_idx", -1)), int(sd.get("collected", -1))


def _es(alt: float, v: float) -> float:
    return max(0.0, alt) + v * v / (2.0 * _G)


def _roll(agent, env, base, C, seed: int, args) -> dict:
    from astrum_dogfight.geometry import relative

    from astrum_rl.tdmpc2.eval_policy import _classify

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    rng_s, gq_s, inr_s, clo_s = [], [], [], []
    v_s, es_me_s, es_opp_s = [], [], []
    event = None
    t0 = True
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        me = base._states.get(base.learner_id)
        opp = base._states.get(base.opponent_id)
        if me is not None and opp is not None and me.alive and opp.alive:
            rg = relative(me, opp)
            rng_s.append(rg.rng)
            gq_s.append(rg.gun_quality)
            inr_s.append(1.0 if rg.in_range else 0.0)
            clo_s.append(rg.closure)
            v_s.append(me.airspeed)
            es_me_s.append(_es(me.alt, me.airspeed))
            es_opp_s.append(_es(opp.alt, opp.airspeed))
        if terminated or truncated:
            break
    me = base._states.get(base.learner_id)
    opp = base._states.get(base.opponent_id)
    outcome = _classify(event, terminated, truncated)
    dealt = (C.HP_MAX - float(opp.health)) if opp is not None else 0.0

    n = len(rng_s)
    res = dict(outcome=outcome, dealt=dealt, steps=n,
               wez=float("nan"), peak_gq=float("nan"), inrng=float("nan"), min_rng=float("nan"),
               v_mean=float("nan"), v_min=float("nan"), stall=float("nan"),
               d_es=float("nan"), rel_es=float("nan"))
    if n < 3:
        return res
    gq = np.asarray(gq_s); rng = np.asarray(rng_s); v = np.asarray(v_s)
    es_me = np.asarray(es_me_s); es_opp = np.asarray(es_opp_s)
    res["wez"] = float(np.mean(gq >= args.gq_wez))
    res["peak_gq"] = float(gq.max())
    res["inrng"] = float(np.mean(inr_s))
    res["min_rng"] = float(rng.min())
    res["v_mean"] = float(v.mean())
    res["v_min"] = float(v.min())
    res["stall"] = float(np.mean(v < C.V_CORNER_FLOOR))
    res["d_es"] = float(es_me[-1] - es_me[0])
    res["rel_es"] = float(es_me[-1] - es_opp[-1])
    return res


def run_ckpt(path: str, label: str, args, device) -> dict:
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env

    agent, stage, coll = _load(path, device)
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    eps = []
    for aspect in aspects:
        env = make_tdmpc_env(
            3, posture="offensive", aspect_deg=aspect, separation_m=args.separation,
            action_repeat=args.action_repeat, frame_stack=1,
            max_time_s=args.max_time_s, rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
            **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
        )
        base = env.unwrapped
        for ep in range(args.episodes):
            eps.append(_roll(agent, env, base, C, args.seed + int(aspect) * 1000 + ep, args))
        env.close()

    N = len(eps)
    kills = sum(1 for e in eps if e["outcome"] == "kill")
    decks = sum(1 for e in eps if e["outcome"] == "deck")
    deaths = sum(1 for e in eps if e["outcome"] == "death")

    def m(key):
        vals = [e[key] for e in eps if e[key] == e[key]]
        return float(np.mean(vals)) if vals else float("nan")

    return dict(
        label=label, stage=stage, coll=coll, N=N, kills=kills, decks=decks, deaths=deaths,
        dealt=m("dealt"), wez=m("wez"), peak_gq=m("peak_gq"), inrng=m("inrng"),
        min_rng=m("min_rng"), v_mean=m("v_mean"), v_min=m("v_min"),
        stall=m("stall"), d_es=m("d_es"), rel_es=m("rel_es"),
    )


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    specs = []
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        path, _, label = tok.partition("=")
        specs.append((path, label or path.split("/")[-1]))

    print(f"WEZ/ENERGY PROBE | vs T{args.opp_tier or 3} (aggressive) | sep {args.separation:.0f}m "
          f"aspects {args.aspects} | {args.episodes} eps/aspect jitter{args.jitter:g} mt{args.max_time_s:g}s")
    print(f"  wez := gun_quality >= {args.gq_wez:g};  stall := airspeed < V_CORNER_FLOOR;  "
          f"dEs = Es_end-Es_start (m);  relEs = Es_me-Es_opp at last alive step (m)\n")

    rows = [run_ckpt(p, l, args, device) for p, l in specs]

    hdr = (f"{'policy':>8} {'coll':>8} {'kill':>5} {'deck':>4} {'dealt':>6} | "
           f"{'wez%':>5} {'peakGQ':>6} {'inrng%':>6} {'minRng':>6} | "
           f"{'vMean':>5} {'vMin':>5} {'stall%':>6} {'dEs':>7} {'relEs':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def pc(x):
            return f"{100 * x:4.0f}%" if x == x else "  — "
        print(f"{r['label']:>8} {r['coll']:>8} {r['kills']:>3}/{r['N']:<2} {r['decks']:>4} {r['dealt']:>6.1f} | "
              f"{pc(r['wez'])} {r['peak_gq']:>6.2f} {pc(r['inrng'])} {r['min_rng']:>5.0f}m | "
              f"{r['v_mean']:>4.0f} {r['v_min']:>4.0f} {pc(r['stall'])} {r['d_es']:>+6.0f}m {r['rel_es']:>+6.0f}m")
    print("\n  wez%/inrng% = share of alive steps with a gun solution / inside gun range;  peakGQ = best gun_quality.\n"
          "  vMin low + stall% high = zoom-stall (bled out of the WEZ);  relEs>0 = ended with more energy than the T3.")


if __name__ == "__main__":
    main()
