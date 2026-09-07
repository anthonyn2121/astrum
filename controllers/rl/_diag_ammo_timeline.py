"""Is the long-range mag-dump the CAUSE of the 55/60 collapse, or a SYMPTOM of having no
solution there? (read-only forensics; throwaway probe)

curriculum-86 (the corrected gun lay, [[curriculum-86-corrected-aim]]) tripled chase kills but
read **0.0 HP dealt at aspects 55/60/75**, where c84 still chipped 25.5/12.2. The autopsy
(`_diag_aim_counterfactual.py`) showed 64% of c86's rounds leave the barrel beyond `R_EFF`
(1200 m) at P(hit) 0.000 with negative closure. Two readings of that fact imply OPPOSITE next
runs, and the reward ambiguity is not resolvable from round counts alone:

  H1  AMMO STARVATION.  It empties the magazine early at un-hittable range, and LATER reaches a
      real firing solution it can no longer shoot at. The dump is the CAUSE; a fire-discipline
      tax (reward-v27) recovers 55/60/75 by keeping rounds for the solution.
  H2  NO-SOLUTION SYMPTOM.  At 55/60 it never reaches a hittable geometry at all. The
      long-range firing is incidental noise, taxing it buys nothing, and the real wall is
      conversion/positioning — a different edit entirely.

The discriminator is **what geometry the jet reaches AFTER the magazine runs dry**. This probe
flies a checkpoint greedy and, per episode, splits the timeline at the dry tick and counts the
env-steps on each side that sat in a genuinely shootable geometry — `in_range` and predicted
miss `lay_miss_m` inside the sim's own 8 m hit sphere (`HIT_RADIUS_M`), the same scalar
`reward.py`'s `in_solution` gate uses. Many post-dry hittable steps ⇒ H1. Near zero ⇒ H2.

Two secondary readings, both needed to SIZE the edit rather than just justify it:
  * `rds>Reff` per aspect — how much of the magazine an out-of-range tax would actually touch,
    **including in the 20/20-killing aspect-30 cell**. If the winning cell also spends heavily
    out of range, a blunt tax puts the one working capability at risk and has to be gated.
  * `dry@` — the fraction of the episode already elapsed when the jet goes dry. A dump that
    happens at 95% of the clock costs nothing no matter how wasteful it looks.

Run:  cd controllers/rl && uv run python _diag_ammo_timeline.py \
        --checkpoint runs/curriculum-86/c86_corrected_lay_5000k.pt \
        --aspects 0,30,45,55,60,75 --episodes 8 --jitter 0.5 --device cuda
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ammo-timeline autopsy: cause vs symptom of the dump.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--aspects", default="0,30,45,55,60,75")
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


def _roll_episode(agent, env, base, C, geom, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()

    me = base._states[base.learner_id]
    ammo_prev = int(me.ammo)
    ammo0 = ammo_prev

    dry_step = None  # env-step index at which the magazine hit 0
    step = 0
    event = None

    # rounds, split by whether the barrel was inside the reward's effective range
    rds_in, rds_out = 0, 0
    # shootable-geometry step counts, split at the dry tick
    pre_hit, post_hit = 0, 0
    # a looser "nearly there" band, to catch a policy that is close but not inside 8 m
    pre_near, post_near = 0, 0
    post_min_rng = 1e9
    post_best_miss = 1e9

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

        ammo_now = int(me.ammo)
        if ammo_now < ammo_prev:
            fired = ammo_prev - ammo_now
            if rg.in_range:
                rds_in += fired
            else:
                rds_out += fired
        if dry_step is None and ammo_now <= 0:
            dry_step = step
        ammo_prev = ammo_now

        # The same scalar reward.py's `in_solution` gate reads, against the sim's real hit sphere.
        shootable = rg.in_range and rg.lay_feasible and rg.lay_miss_m <= C.HIT_RADIUS_M
        near = rg.in_range and rg.lay_feasible and rg.lay_miss_m <= C.GUN_FIRE_MISS_M * 3.0
        # `dry_step` is set on the tick the mag empties, so `>` counts only DISARMED steps.
        if dry_step is not None and step > dry_step:
            post_hit += int(shootable)
            post_near += int(near)
            post_min_rng = min(post_min_rng, rg.rng)
            post_best_miss = min(post_best_miss, rg.lay_miss_m)
        else:
            pre_hit += int(shootable)
            pre_near += int(near)

        if terminated or truncated:
            break

    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    killed = (not opp.alive) or event == "kill"
    died = (not me.alive) or event in ("killed", "death", "trade", "hard_deck")
    return {
        "event": event,
        "killed": killed,
        "died": died,
        "dealt": C.HP_MAX - float(opp.health),
        "steps": step,
        "spent": ammo0 - int(me.ammo),
        "rds_in": rds_in,
        "rds_out": rds_out,
        "dry": dry_step is not None,
        # Fraction of the episode elapsed at the dry tick; NaN when it never went dry so the
        # mean over episodes reports "when it went dry, it was this far in" and is not diluted
        # toward 1.0 by the episodes that kept ammo.
        "dry_frac": (dry_step / step) if dry_step is not None else float("nan"),
        "post_steps": (step - dry_step) if dry_step is not None else 0,
        "pre_hit": pre_hit,
        "post_hit": post_hit,
        "pre_near": pre_near,
        "post_near": post_near,
        "post_min_rng": post_min_rng if post_min_rng < 1e9 else float("nan"),
        "post_best_miss": post_best_miss if post_best_miss < 1e9 else float("nan"),
    }


def main() -> None:
    args = parse_args()
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
        f"AMMO TIMELINE {args.checkpoint} | mode={args.mode} | vs native aggressive T3 | "
        f"sep {args.separation:.0f}m jitter{args.jitter:g} | {args.episodes} eps/aspect | "
        f"collected={sd.get('collected', '?')}"
    )
    print(
        f"combat: R_EFF={C.R_EFF:.0f}m  HIT_RADIUS={C.HIT_RADIUS_M:.0f}m  AMMO={C.AMMO_MAX:.0f}  "
        f"shootable = in_range AND lay_miss_m<={C.HIT_RADIUS_M:.0f}m (== reward's in_solution gate)"
    )
    print(
        "\nH1 (dump CAUSES the collapse) needs post-dry SHOOTABLE steps > 0 at 55/60.\n"
        "H2 (dump is a SYMPTOM) predicts post-dry shootable ~= 0 AND pre-dry shootable ~= 0 there.\n"
    )
    hdr = (
        f"{'aspect':>6} {'kill':>4} {'die':>3} {'dealt':>6} {'rds':>4} {'rds>Reff':>8} "
        f"{'%out':>5} {'dry':>4} {'dry@':>5} {'postN':>6} {'preShoot':>8} {'postShoot':>9} "
        f"{'preNear':>7} {'postNear':>8} {'postMinR':>8} {'postMiss':>8}"
    )
    print(hdr)
    totals: list[dict] = []
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
        totals.extend(rows)

        def m(key):
            v = [r[key] for r in rows if r[key] == r[key]]
            return statistics.fmean(v) if v else float("nan")

        spent = m("spent")
        out = m("rds_out")
        print(
            f"{aspect:>6.0f} {sum(r['killed'] for r in rows):>4} {sum(r['died'] for r in rows):>3} "
            f"{m('dealt'):>6.1f} {spent:>4.0f} {out:>8.0f} "
            f"{(100.0 * out / spent if spent else float('nan')):>4.0f}% "
            f"{sum(r['dry'] for r in rows):>2}/{len(rows):<1} {m('dry_frac'):>5.2f} "
            f"{m('post_steps'):>6.0f} {m('pre_hit'):>8.1f} {m('post_hit'):>9.1f} "
            f"{m('pre_near'):>7.1f} {m('post_near'):>8.1f} {m('post_min_rng'):>8.0f} "
            f"{m('post_best_miss'):>8.1f}"
        )

    print("\n=== POOLED ===")
    tot_spent = sum(r["spent"] for r in totals)
    tot_out = sum(r["rds_out"] for r in totals)
    print(
        f"episodes {len(totals)} | rounds {tot_spent} | out-of-range {tot_out} "
        f"({100.0 * tot_out / tot_spent if tot_spent else float('nan'):.0f}%) | "
        f"went dry {sum(r['dry'] for r in totals)}/{len(totals)}"
    )
    print(
        f"shootable steps: pre-dry {sum(r['pre_hit'] for r in totals)} | "
        f"post-dry {sum(r['post_hit'] for r in totals)}  <-- the H1/H2 discriminator"
    )
    print(
        f"near-solution steps (<= {C.GUN_FIRE_MISS_M * 3.0:.0f}m): "
        f"pre-dry {sum(r['pre_near'] for r in totals)} | post-dry {sum(r['post_near'] for r in totals)}"
    )
    print(
        "\nRead: `rds>Reff` in the KILLING cell (aspect 30) sizes the collateral risk of a blunt "
        "out-of-range tax — if that cell also spends most of its mag beyond R_EFF, the tax must be "
        "gated (e.g. on closure<0 or on a per-episode budget) rather than applied flat."
    )


if __name__ == "__main__":
    main()
