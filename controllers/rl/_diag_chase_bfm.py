"""Chase-BFM failure autopsy: WHY does a checkpoint get 0 gun-kills vs the CURRENT
aggressive T3 in a stern chase?

`_verdict_tmp.py` reports the OUTCOME (kills/decks/net) but not the mechanism. This probe
rolls a checkpoint (default greedy) vs the *native* aggressive T3 at chase aspects and, at
every step, reads the full relative geometry (`geometry.relative`) plus two derived BFM
signals the reward never sees:

  * LOS rate  omega = |d(los_hat)/dt|  (deg/s) — the angular speed of the line of sight.
    A hard-turning bandit drives omega up; a gun tracking solution needs omega ~small.
  * Specific energy Es = alt + V^2/2g (m) for BOTH jets — the energy fight.

It detects a trigger pull by ammo decrement and logs, per FIRE event: lead_error, range,
aspect/ata, closure, omega, and whether the shot was BALLISTICALLY HITTABLE
(lead_error <= sigma_track(rng), the honest 20mm tolerance) vs merely "near" on the loose
3deg reward Gaussian. Then it classifies each episode into the BFM failure mode:

  OVERSHOOT      — flew out front / across: min-range reached with high closure, then range
                   rebounds while aspect stays high (flight-path or wingline overshoot).
  NO_CONVERT     — never got the nose to the six: min aspect / min ata stayed high, never
                   entered a rear-quarter WEZ (can't complete the turn vs the hard break).
  LOS_SPRAY      — got into WEZ and fired, but at high LOS rate / lead_error >> tolerance:
                   an un-hittable tracking pass (the crossing-velocity aim wall).
  ENERGY_DOWN    — bled below the T3 in Es (defensive, can't sustain the conversion).
  CLEAN_KILL     — killed it.

Run:  cd controllers/rl && uv run python _diag_chase_bfm.py \
        --checkpoint runs/BASELINE_merge_survivor_2480k.pt \
        --aspects 0,30,45,55 --episodes 8 --jitter 0.5 --device cuda
"""
from __future__ import annotations

import argparse
import statistics
from math import degrees

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

G = 9.80665


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chase-BFM failure autopsy vs native aggressive T3.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--aspects", default="0,30,45,55")
    p.add_argument("--mode", default="greedy", choices=["greedy", "mpc"])
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _es(s) -> float:
    return float(s.alt + s.airspeed * s.airspeed / (2.0 * G))


def _roll_episode(agent, env, base, C, geom, seed: int) -> dict:
    from astrum_dogfight.geometry import sigma_track

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    dt = args_action_repeat / args_rate_hz  # wall/sim seconds per env.step

    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    ammo_prev = int(me.ammo)
    los_prev = None

    min_rng = 1e9
    geom_at_minrng = None
    min_aspect = 1e9
    min_ata = 1e9
    fires: list[dict] = []
    es_me_start, es_opp_start = _es(me), _es(opp)
    es_me_min = es_me_start
    es_gap_at_minrng = 0.0
    rebounded_after_min = 0.0  # how far range grew back after the closest approach
    aspect_at_minrng = 0.0
    closure_at_minrng = 0.0
    reached_min_step = 0
    step = 0
    event = None

    t0 = True
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        step += 1
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rg = geom.relative(me, opp)

        los = opp.pos - me.pos
        n = float(np.linalg.norm(los))
        los_hat = los / n if n > 1e-6 else None
        omega = 0.0
        if los_hat is not None and los_prev is not None:
            c = float(np.clip(np.dot(los_hat, los_prev), -1.0, 1.0))
            omega = degrees(np.arccos(c)) / dt  # deg/s
        los_prev = los_hat

        es_me_min = min(es_me_min, _es(me))
        min_aspect = min(min_aspect, degrees(rg.aspect))
        min_ata = min(min_ata, degrees(rg.ata))

        if rg.rng < min_rng:
            min_rng = rg.rng
            geom_at_minrng = rg
            aspect_at_minrng = degrees(rg.aspect)
            closure_at_minrng = rg.closure
            es_gap_at_minrng = _es(me) - _es(opp)
            reached_min_step = step
        else:
            rebounded_after_min = max(rebounded_after_min, rg.rng - min_rng)

        ammo_now = int(me.ammo)
        if ammo_now < ammo_prev:
            tol = sigma_track(rg.rng)
            fires.append(
                {
                    "lead_deg": degrees(rg.lead_error),
                    "rng": rg.rng,
                    "aspect_deg": degrees(rg.aspect),
                    "ata_deg": degrees(rg.ata),
                    "closure": rg.closure,
                    "omega": omega,
                    "gq": rg.gun_quality,
                    "gq_honest": rg.gun_quality_honest,
                    "hittable": rg.in_range and rg.lead_error <= tol,
                    "rounds": ammo_prev - ammo_now,
                }
            )
        ammo_prev = ammo_now

        if terminated or truncated:
            break

    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    dealt = C.HP_MAX - float(opp.health)
    taken = C.HP_MAX - float(me.health)
    killed = (not opp.alive) or event == "kill"
    died = (not me.alive) or event in ("killed", "death", "trade", "hard_deck")

    n_fire = sum(f["rounds"] for f in fires)
    n_hittable = sum(f["rounds"] for f in fires if f["hittable"])
    lead_at_fire = statistics.fmean(f["lead_deg"] for f in fires) if fires else float("nan")
    rng_at_fire = statistics.fmean(f["rng"] for f in fires) if fires else float("nan")
    omega_at_fire = statistics.fmean(f["omega"] for f in fires) if fires else float("nan")

    # ----- BFM failure classification -----
    if killed:
        cls = "CLEAN_KILL"
    elif es_gap_at_minrng < -1500.0 or (es_me_min - es_opp_start) < -2500.0:
        cls = "ENERGY_DOWN"
    elif min_aspect > 60.0 and min_rng > C.R_EFF * 0.8:
        cls = "NO_CONVERT"
    elif rebounded_after_min > 400.0 and closure_at_minrng > 80.0:
        cls = "OVERSHOOT"
    elif n_fire > 0 and n_hittable == 0:
        cls = "LOS_SPRAY"
    elif n_fire == 0:
        cls = "NO_SHOT"
    else:
        cls = "OTHER"

    return {
        "event": event,
        "killed": killed,
        "died": died,
        "dealt": dealt,
        "taken": taken,
        "min_rng": min_rng,
        "min_aspect": min_aspect,
        "min_ata": min_ata,
        "aspect_at_minrng": aspect_at_minrng,
        "closure_at_minrng": closure_at_minrng,
        "es_gap_at_minrng": es_gap_at_minrng,
        "rebound": rebounded_after_min,
        "n_fire": n_fire,
        "n_hittable": n_hittable,
        "lead_at_fire": lead_at_fire,
        "rng_at_fire": rng_at_fire,
        "omega_at_fire": omega_at_fire,
        "cls": cls,
        "fires": fires,
    }


args_action_repeat = 5
args_rate_hz = 120.0


def main() -> None:
    global args_action_repeat, args_rate_hz
    args = parse_args()
    args_action_repeat = args.action_repeat
    args_rate_hz = args.rate_hz
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight import geometry as geom

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = args.mode == "mpc"
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    print(
        f"CHASE-BFM AUTOPSY {args.checkpoint} | mode={args.mode} | vs native aggressive T3 | "
        f"sep {args.separation:.0f}m jitter{args.jitter:g} | {args.episodes} eps/aspect"
    )
    print(
        f"combat: R_EFF={C.R_EFF:.0f}m  LEAD_SIGMA={degrees(C.LEAD_SIGMA):.1f}deg  "
        f"HIT_RADIUS={C.HIT_RADIUS_M:.0f}m  hittable=lead<=sigma_track(rng)"
    )
    hdr = (
        f"{'aspect':>6} {'kill':>4} {'die':>3} {'dealt':>5} {'minRng':>6} {'asp@min':>7} "
        f"{'clo@min':>7} {'minAsp':>6} {'minAta':>6} {'fires':>5} {'hit':>4} "
        f"{'lead°@f':>7} {'rng@f':>6} {'ω@f°/s':>7} {'dEs@min':>7}   classes"
    )
    all_fires: list[dict] = []
    for aspect in aspects:
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=aspect,
            separation_m=args.separation,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
        )
        base = env.unwrapped
        rows = [
            _roll_episode(agent, env, base, C, geom, seed=args.seed + int(aspect) * 1000 + ep)
            for ep in range(args.episodes)
        ]
        env.close()
        for r in rows:
            all_fires.extend(r["fires"])
        if aspect == aspects[0]:
            print(hdr)

        def m(key):
            return statistics.fmean(r[key] for r in rows)

        kills = sum(r["killed"] for r in rows)
        dies = sum(r["died"] for r in rows)
        clsc: dict[str, int] = {}
        for r in rows:
            clsc[r["cls"]] = clsc.get(r["cls"], 0) + 1
        cls_str = " ".join(f"{k}:{v}" for k, v in sorted(clsc.items(), key=lambda kv: -kv[1]))
        print(
            f"{aspect:>6.0f} {kills:>4} {dies:>3} {m('dealt'):>5.0f} {m('min_rng'):>6.0f} "
            f"{m('aspect_at_minrng'):>7.0f} {m('closure_at_minrng'):>7.0f} {m('min_aspect'):>6.0f} "
            f"{m('min_ata'):>6.0f} {m('n_fire'):>5.0f} {m('n_hittable'):>4.0f} "
            f"{m('lead_at_fire'):>7.1f} {m('rng_at_fire'):>6.0f} {m('omega_at_fire'):>7.1f} "
            f"{m('es_gap_at_minrng'):>+7.0f}   {cls_str}"
        )

    # ----- (b) range-bucketed hittability: does the shot become hittable once it presses in? -----
    print("\nFIRE HITTABILITY vs RANGE (all aspects pooled) — hittable = lead<=sigma_track(rng)")
    print(
        f"{'range band':>12} {'rounds':>7} {'hittable':>8} {'hit%':>5} "
        f"{'mean lead°':>10} {'σ_track°':>8} {'mean ω°/s':>9} {'mean clo':>9}"
    )
    from astrum_dogfight.geometry import sigma_track

    bands = [(0, 150), (150, 250), (250, 400), (400, 600), (600, 900), (900, 1200)]
    for lo, hi in bands:
        rec = [f for f in all_fires if lo <= f["rng"] < hi]
        rounds = sum(f["rounds"] for f in rec)
        hit = sum(f["rounds"] for f in rec if f["hittable"])
        if rounds == 0:
            print(f"{f'{lo}-{hi}m':>12} {0:>7} {0:>8} {'--':>5} {'--':>10} "
                  f"{degrees(sigma_track((lo + hi) / 2)):>8.2f} {'--':>9} {'--':>9}")
            continue
        lead = statistics.fmean(f["lead_deg"] for f in rec)
        om = statistics.fmean(f["omega"] for f in rec)
        clo = statistics.fmean(f["closure"] for f in rec)
        print(
            f"{f'{lo}-{hi}m':>12} {rounds:>7} {hit:>8} {100.0 * hit / rounds:>4.0f}% "
            f"{lead:>10.1f} {degrees(sigma_track((lo + hi) / 2)):>8.2f} {om:>9.1f} {clo:>9.1f}"
        )


if __name__ == "__main__":
    main()
