"""Throwaway trajectory logger: fly the GREEDY policy from one initial condition and print the
per-step flight path + a phase summary, so we can EXPLAIN what the agent actually does (merge,
overshoot, dive, deck, fire discipline) rather than only the aggregate kill/deck counts.

Run inside the container, e.g.:
  PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
    uv run python _traj_tmp.py --checkpoint runs/curriculum-46/_snap_2244k.pt \
      --aspect 45 --seed 1 --episodes 1
  ... --aspect 0 --opp-heading 135 --opp-tier 2 --separation 1500 ...
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--aspect", type=float, default=45.0)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-heading", type=float, default=None)
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--sample-s", type=float, default=3.0, help="print a row every N sim seconds")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main():
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight.geometry import relative
    from astrum_rl.envs import make_tdmpc_env

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    dt = args.action_repeat / args.rate_hz  # sim seconds per agent step
    sample_every = max(1, int(round(args.sample_s / dt)))
    hdg = "co-heading(None)" if args.opp_heading is None else f"{args.opp_heading:.0f}deg"
    tier = args.opp_tier if args.opp_tier > 0 else 3
    print(
        f"=== TRAJECTORY {args.checkpoint} | aspect {args.aspect:.0f} | opp_hdg {hdg} | "
        f"opp_tier {tier} | sep {args.separation:.0f}m | jitter {args.jitter:g} | mt {args.max_time_s:.0f}s ==="
    )

    for ep in range(args.episodes):
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=args.aspect,
            separation_m=args.separation,
            action_repeat=args.action_repeat,
            frame_stack=1,
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
            **({"opp_heading_deg": args.opp_heading} if args.opp_heading is not None else {}),
            **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
        )
        base = env.unwrapped
        lnr, opp = base.learner_id, base.opponent_id
        obs, _ = env.reset(seed=args.seed + ep)
        agent._prev_mean.zero_()

        # trackers
        min_alt = 1e9
        min_alt_t = 0.0
        min_rng = 1e9
        min_rng_t = 0.0
        max_ata = 0.0
        max_ata_t = 0.0
        first_fire_t = None
        fire_steps = 0
        merge_t = None  # time closure flips +→- (first pass complete)
        prev_clos = None

        print(
            f"\n-- episode seed {args.seed + ep} --\n"
            f"{'t':>5} {'alt_m':>6} {'pit°':>5} {'bnk°':>5} {'rng_m':>6} "
            f"{'ata°':>5} {'clos':>6} {'kt':>4} {'meHP':>4} {'opHP':>4} {'fire':>4}"
        )
        t0 = True
        event = None
        step = 0
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            firing = bool(a[4] >= 0.0)
            if firing:
                fire_steps += 1
            obs, _r, term, trunc, info = env.step(a)
            step += 1
            t = step * dt
            me = base._states[lnr]
            bd = base._states[opp]
            rg = relative(me, bd)
            alt = float(me.alt)
            pit = math.degrees(float(me.theta))
            bnk = math.degrees(float(me.phi))
            rng = float(rg.rng)
            ata = math.degrees(float(rg.ata))
            clos = float(rg.closure)

            if firing and first_fire_t is None:
                first_fire_t = t
            if alt < min_alt:
                min_alt, min_alt_t = alt, t
            if rng < min_rng:
                min_rng, min_rng_t = rng, t
            if ata > max_ata:
                max_ata, max_ata_t = ata, t
            if prev_clos is not None and prev_clos > 0 >= clos and merge_t is None and rng < args.separation:
                merge_t = t
            prev_clos = clos

            if info.get("event"):
                event = info["event"]
            if step % sample_every == 0 or term or trunc:
                print(
                    f"{t:>5.1f} {alt:>6.0f} {pit:>5.0f} {bnk:>5.0f} {rng:>6.0f} "
                    f"{ata:>5.0f} {clos:>+6.0f} {float(me.airspeed):>4.0f} "
                    f"{float(me.health):>4.0f} {float(bd.health):>4.0f} {'Y' if firing else '.':>4}"
                )
            if term or trunc:
                break

        me = base._states[lnr]
        bd = base._states[opp]
        dealt = C.HP_MAX - float(bd.health)
        taken = C.HP_MAX - float(me.health)
        outcome = _classify(event, term, trunc)
        env.close()
        print(
            f"   PHASE SUMMARY: outcome={outcome} event={event} t_end={step * dt:.1f}s\n"
            f"     merge(closest pass) @ {min_rng:.0f}m t={min_rng_t:.1f}s"
            + (f" (closure flip @ {merge_t:.1f}s)" if merge_t is not None else " (never flipped/co-speed)")
            + f"\n     max overshoot ATA {max_ata:.0f}° @ t={max_ata_t:.1f}s\n"
            f"     MIN ALT {min_alt:.0f} m @ t={min_alt_t:.1f}s   (spawn ~4572 m, hard deck 300 m)\n"
            f"     fire: first @ {first_fire_t}  rounds_fired_steps={fire_steps}  ammo_spent={C.AMMO_MAX - float(me.ammo):.0f}\n"
            f"     dealt={dealt:.0f} HP  taken={taken:.0f} HP  final_alt={float(me.alt):.0f} m  final_rng={float(relative(me, bd).rng):.0f} m"
        )


if __name__ == "__main__":
    main()
