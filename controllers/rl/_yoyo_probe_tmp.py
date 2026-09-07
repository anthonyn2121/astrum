"""Measure whether the greedy policy actually flies HIGH / LOW YOYOS against the
(now aggressive) scripted Tier-3 — i.e. does it use the VERTICAL to manage closure
and convert, or is it still a flat pure-pursuit turner?

A yoyo is out-of-plane BFM in a turning fight:
  - HIGH yoyo: too much closure / about to overshoot → roll+pull UP out of plane
    (nose-high, climb ABOVE the bandit) to bleed closure and hold the rear quarter,
    then roll down and convert to a tracking/gun solution.
  - LOW yoyo: falling behind a tighter-turning bandit / low closure → roll+pull nose
    DOWN (dive BELOW his plane) to cut across the circle and rebuild closure/lead,
    then pull back up into the solution.

So the signatures live in (my_alt − bandit_alt) SWINGS that happen while BANKED and
ENGAGED, and a good yoyo is one that is FOLLOWED by an improved gun geometry
(ata ↓, gun_quality ↑, or damage dealt) within a few seconds. This probe:

  1. rolls the greedy (mpc=False) policy vs the aggressive native T3, jitter 0.5,
  2. logs the per-step geometry + attitude time-series,
  3. detects vertical excursions (swing detector, prominence-gated), classifies each
     as high/low yoyo, and scores whether it CONVERTED,
  4. reports flat-turn vs climbing-turn vs diving-turn fractions and vertical amplitude,
  5. prints an A/B table across the given checkpoints (baseline vs c67).

Run in-container:
  cd /workspace/controllers/rl && uv run python _yoyo_probe_tmp.py \
      --checkpoints runs/BASELINE_merge_survivor_2480k.pt=BASE,runs/curriculum-67/_yoyo_snap.pt=c67 \
      --aspects 45,55 --episodes 12 --jitter 0.5 --max-time-s 60

Throwaway diagnostic (untracked), like _verdict_tmp.py / _replay_export_tmp.py.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect + score high/low yoyos of a greedy policy vs the aggressive T3.")
    p.add_argument("--checkpoints", required=True,
                   help="comma list of PATH[=LABEL] to A/B (e.g. base.pt=BASE,run.pt=c67)")
    p.add_argument("--aspects", default="45,55", help="comma list of start aspects (deg)")
    p.add_argument("--episodes", type=int, default=12, help="episodes per aspect")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-tier", type=int, default=0, help="0 = full native (aggressive) Tier-3")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # BFM-detector knobs
    p.add_argument("--engage-rng", type=float, default=1400.0, help="max range (m) to count as 'in the fight'")
    p.add_argument("--bank-deg", type=float, default=35.0, help="|phi| above which a turn is 'hard'")
    p.add_argument("--prom-m", type=float, default=60.0, help="min vertical swing (m) for a yoyo excursion")
    p.add_argument("--conv-s", type=float, default=3.0, help="seconds after an apex to look for conversion")
    p.add_argument("--pitch-deg", type=float, default=5.0, help="|theta| deadband for flat/climb/dive split")
    return p.parse_args(argv)


def _load(path: str, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # greedy policy-prior mean — matches the verdict harnesses
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent, int(sd.get("stage_idx", -1)), int(sd.get("collected", -1))


def _swings(y: np.ndarray, prom: float):
    """Swing detector: return [(idx, kind)] extrema whose retrace exceeds `prom`.

    Walks the (smoothed) series tracking the running extremum since the last confirmed
    turning point; when the series retraces by >= prom from that extremum, the extremum
    is confirmed (kind 'max'/'min') and tracking flips direction. Prominence-gated so
    small ripples don't count as yoyos.
    """
    n = len(y)
    if n < 3:
        return []
    ext = []
    ext_idx = 0
    ext_val = y[0]
    up = None  # current search direction: True=looking for a max, False=for a min, None=unknown
    for i in range(1, n):
        v = y[i]
        if up is None:
            if v > ext_val + 1e-6:
                up = True
            elif v < ext_val - 1e-6:
                up = False
            if (up is True and v > ext_val) or (up is False and v < ext_val) or up is None:
                ext_val, ext_idx = v, i
            continue
        if up:  # tracking a maximum
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif v <= ext_val - prom:  # retraced enough → confirm the max
                ext.append((ext_idx, "max"))
                up = False
                ext_val, ext_idx = v, i
        else:  # tracking a minimum
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif v >= ext_val + prom:  # retraced enough → confirm the min
                ext.append((ext_idx, "min"))
                up = True
                ext_val, ext_idx = v, i
    return ext


def _smooth(y: np.ndarray, w: int = 7) -> np.ndarray:
    if len(y) < w or w < 2:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def _roll(agent, env, base, C, seed: int, dt: float, args) -> dict:
    from astrum_dogfight.geometry import relative

    from astrum_rl.tdmpc2.eval_policy import _classify

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    rng_s, ata_s, asp_s, clo_s, gq_s = [], [], [], [], []
    myh_s, theta_s, phi_s = [], [], []  # my_alt-opp_alt, pitch, bank
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
            ata_s.append(math.degrees(rg.ata))
            asp_s.append(math.degrees(rg.aspect))
            clo_s.append(rg.closure)
            gq_s.append(rg.gun_quality)
            myh_s.append(me.alt - opp.alt)
            theta_s.append(math.degrees(me.theta))
            phi_s.append(abs(math.degrees(me.phi)))
        if terminated or truncated:
            break
    me = base._states.get(base.learner_id)
    opp = base._states.get(base.opponent_id)
    outcome = _classify(event, terminated, truncated)
    dealt = (C.HP_MAX - float(opp.health)) if opp is not None else 0.0

    rng_a = np.asarray(rng_s); ata_a = np.asarray(ata_s); phi_a = np.asarray(phi_s)
    theta_a = np.asarray(theta_s); gq_a = np.asarray(gq_s); myh_a = np.asarray(myh_s)
    n = len(rng_a)
    res = dict(outcome=outcome, dealt=dealt, steps=n,
               hi=0, lo=0, hi_conv=0, lo_conv=0,
               flat=0.0, climb=0.0, dive=0.0, engaged=0, vamp=0.0, mean_ata=float("nan"))
    if n < 5:
        return res

    engaged = (rng_a < args.engage_rng) & (phi_a > args.bank_deg)
    eng_i = np.where(engaged)[0]
    res["engaged"] = int(engaged.sum())
    if engaged.sum() >= 3:
        th_e = theta_a[engaged]
        pdb = args.pitch_deg
        res["climb"] = float(np.mean(th_e > pdb))
        res["dive"] = float(np.mean(th_e < -pdb))
        res["flat"] = float(np.mean(np.abs(th_e) <= pdb))
        res["vamp"] = float(np.std(myh_a[engaged]))
        res["mean_ata"] = float(np.mean(ata_a[engaged]))

    # Detect yoyo excursions on my_height, but only credit those that occur while
    # engaged+banked (a genuine turning-fight vertical, not a cruise climb).
    conv_steps = max(1, int(args.conv_s / dt))
    for idx, kind in _swings(_smooth(myh_a), args.prom_m):
        if idx >= n or not engaged[idx]:
            continue
        j = min(n, idx + conv_steps)
        ata_after = ata_a[idx:j]
        gq_after = gq_a[idx:j]
        converted = (ata_after.min() <= ata_a[idx] - 8.0) or (gq_after.max() >= gq_a[idx] + 0.05)
        if kind == "max":  # apex ABOVE bandit → high yoyo
            res["hi"] += 1
            res["hi_conv"] += int(converted)
        else:              # nadir BELOW bandit → low yoyo
            res["lo"] += 1
            res["lo_conv"] += int(converted)
    return res


def run_ckpt(path: str, label: str, args, device, dt: float) -> dict:
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
            eps.append(_roll(agent, env, base, C, args.seed + int(aspect) * 1000 + ep, dt, args))
        env.close()

    N = len(eps)
    hi = sum(e["hi"] for e in eps); lo = sum(e["lo"] for e in eps)
    hi_c = sum(e["hi_conv"] for e in eps); lo_c = sum(e["lo_conv"] for e in eps)
    eng = [e for e in eps if e["engaged"] >= 3]
    kills = sum(1 for e in eps if e["outcome"] == "kill")
    return dict(
        label=label, stage=stage, coll=coll, N=N, kills=kills,
        dealt=float(np.mean([e["dealt"] for e in eps])),
        hi_ep=hi / N, lo_ep=lo / N,
        hi_conv=(hi_c / hi if hi else float("nan")),
        lo_conv=(lo_c / lo if lo else float("nan")),
        flat=float(np.mean([e["flat"] for e in eng])) if eng else float("nan"),
        climb=float(np.mean([e["climb"] for e in eng])) if eng else float("nan"),
        dive=float(np.mean([e["dive"] for e in eng])) if eng else float("nan"),
        vamp=float(np.mean([e["vamp"] for e in eng])) if eng else float("nan"),
        mean_ata=float(np.nanmean([e["mean_ata"] for e in eng])) if eng else float("nan"),
    )


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    specs = []
    for tok in args.checkpoints.split(","):
        tok = tok.strip()
        if not tok:
            continue
        path, _, label = tok.partition("=")
        specs.append((path, label or path.split("/")[-1]))

    print(f"YOYO PROBE | vs T{args.opp_tier or 3} (aggressive) | sep {args.separation:.0f}m "
          f"aspects {args.aspects} | {args.episodes} eps/aspect jitter{args.jitter:g} mt{args.max_time_s:g}s")
    print(f"  engaged := rng<{args.engage_rng:.0f}m & |bank|>{args.bank_deg:.0f}°;  "
          f"yoyo swing >= {args.prom_m:.0f}m;  convert := ata−8° or gun_q+0.05 within {args.conv_s:g}s\n")

    rows = [run_ckpt(p, l, args, device, dt) for p, l in specs]

    hdr = (f"{'policy':>6} {'stage':>5} {'kills':>6} {'dealt':>6} | "
           f"{'hi/ep':>6} {'hi-cv':>6} {'lo/ep':>6} {'lo-cv':>6} | "
           f"{'climb%':>7} {'dive%':>6} {'flat%':>6} {'vamp':>6} {'⟨ata⟩':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def pc(x):
            return f"{100 * x:5.0f}%" if x == x else "   —  "
        print(f"{r['label']:>6} {r['stage']:>5} {r['kills']:>4}/{r['N']:<2} {r['dealt']:>6.1f} | "
              f"{r['hi_ep']:>6.2f} {pc(r['hi_conv'])} {r['lo_ep']:>6.2f} {pc(r['lo_conv'])} | "
              f"{pc(r['climb'])} {pc(r['dive'])} {pc(r['flat'])} {r['vamp']:>4.0f}m {r['mean_ata']:>5.0f}°")
    print("\n  hi/lo per ep = vertical yoyo excursions detected/episode;  *-cv = fraction that CONVERTED "
          "(ata↓ or gun_q↑).\n  climb/dive/flat% = share of hard-turn steps pulling up / nose-down / flat "
          "(flat = pure-pursuit).  vamp = vertical swing amplitude (m).")


if __name__ == "__main__":
    main()
