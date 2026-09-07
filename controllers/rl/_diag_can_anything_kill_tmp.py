"""Is the spawn-55/60 chase entry WINNABLE at all — by any controller, learned or hand-flown?

~20 documented ceilings have been spent on this one bucket without anyone checking that the target
is reachable. `_diag_counterfactual_pull_tmp.py` round 3 found the reason it fails, and it is not
the reason every lever assumed:

    spawn 30 (kills 4/4, 100 HP)     spawn 55/60 (0 kills, 18-38 HP)
    dAGL me  +58..+128 m             dAGL me  +3101..+3263 m   (bandit +3525..+3780)
    V med    223-225 m/s             V med    136-155 m/s
    α med    0.9-2.4 deg             α med    13.6-14.4 deg,  α>=14 on ~50% of ticks
    nz  me/op  2.5-3.1 / 5.3-5.4     nz  me/op  1.7-2.1 / 2.2-2.3   <-- BOTH out of lift
    rng min  391-427 m               rng min  188-251 m        <-- CLOSER where it LOSES

So range is not the wall (it gets nearer in the buckets it loses) and neither is aim: both jets
zoom-climb ~3.4 km into a mutual α-saturated energy death where nobody can point. The kill bucket
keeps ~14 deg of α in reserve — a 6 g lead correction available on demand — and the failing bucket
has none.

That makes reachability the question. Four whole-episode arms, same seeds, same scenario:

  BASE     the c87 champion, greedy. The control.
  PURSUIT  the env's own PursuitExpert (via info["expert_action"]) — the scripted reference.
  COACH    `bfm.BFMCoach`, the 12-primitive doctrine tree. Never once run on the CHASE; it was
           built and measured for the head-on merge ([[bfm-library-coach]]).
  ENERGY   a deliberately crude ENERGY FIGHT: while α is saturated or slow, refuse to follow him
           up — unload to a nose-low attitude and rebuild speed; only when the lift margin is back
           does it point and pull. This is the maneuver the diagnosis implies, hand-flown, and it
           exists to answer one question: does preserving lift margin actually convert here?

If any arm kills, RL has a proven target and the next lever is worth 5M steps. If NOTHING kills,
the honest conclusion is that this entry vs the current aggressive T3 is a draw by construction —
which is worth more than another lever, because it retires the bucket.

    cd controllers/rl && uv run python -u _diag_can_anything_kill_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,55,60 --episodes 4
"""

import argparse
from math import degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

# The ENERGY arm's two thresholds. A_KEEP is set from MEASUREMENT, not from C.ALPHA_REF: the
# normalizer claims a 30 deg limiter but the airframe never exceeded 16.7 deg under a pegged
# elevator, so ~16 deg is the real ceiling and 11 deg leaves a usable reserve.
A_KEEP = radians(11.0)
V_KEEP = 190.0     # m/s; above corner (V_CORNER_MPS=175) with margin to spend in a pull
THETA_LOW = radians(-12.0)   # the recovery attitude: nose down, cash altitude back into speed


def _energy_stick(me, rel):
    """Crude energy fight in AGENT [-1,1] space: never spend lift margin you do not have."""
    saturated = me.alpha > A_KEEP or me.airspeed < V_KEEP
    if saturated:
        # Unload and rebuild: wings level (a pull while banked just scrubs speed) and nose below
        # the horizon. Deliberately does NOT track him — the whole point is to stop following him up.
        ail = bfm._roll_to(me.phi, 0.0)
        elev = bfm._pitch_to(me.theta, THETA_LOW)
        thr = 1.0
    else:
        ail, _ = bfm._point_at(rel)
        elev = -1.0 if rel.ata > radians(2.0) else -0.2
        thr = 1.0
    ail, elev = bfm._deck_guard(me, ail, elev)
    fire = 1.0 if (rel.in_range and rel.aspect < radians(60.0)
                   and rel.gun_quality >= 0.30) else -1.0
    return np.array([ail, elev, 0.0, thr, fire], dtype=np.float32)


NCOL = 10


def roll(agent, env, base, seed, dt, arm, low, high):
    obs, info = env.reset(seed=seed)
    agent._prev_mean.zero_()
    coach = bfm.BFMCoach() if arm == "COACH" else None
    t0, event = True, None
    rows = []
    while True:
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        if arm == "BASE":
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        elif arm == "PURSUIT":
            # Already inverted into AGENT space by the env's action wrapper.
            a = np.asarray(info["expert_action"], dtype=np.float32)
        elif arm == "COACH":
            a = coach.act(me, rg)
        else:
            a = _energy_stick(me, rg)
        t0 = False
        rows.append((rg.rng, me.airspeed, degrees(me.alpha), me.nz, degrees(me.theta),
                     me.alt - base.ground_alt, rg.lay_miss_m, degrees(rg.aspect),
                     # Trigger test is uniform across arms: every arm emits AGENT space, whose
                     # affine map sends [-1,1] -> native [0,1] with the sim gating at >= 0.5, so the
                     # agent-space threshold is 0.0 (this is the exact trap bfm.py documents).
                     degrees(rg.ata), float(a[4] >= 0.0)))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    op = base._states[base.opponent_id]
    return dict(event=event, M=np.asarray(rows).reshape(-1, NCOL),
                dealt=C.HP_MAX - float(op.health), taken=C.HP_MAX - float(me.health),
                spent=int(C.AMMO_MAX - me.ammo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="30,55,60")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--arms", default="BASE,PURSUIT,COACH,ENERGY")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    dt = 5.0 / 120.0
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"KILLABLE {args.checkpoint} (collected={sd.get('collected', 0):,}) | whole-episode arms "
          f"{'/'.join(arms)} | chase sep {args.separation:.0f}m {args.max_time_s:.0f}s "
          f"jitter {args.jitter:g} | {args.episodes} eps/aspect | ENERGY: unload below α "
          f"{degrees(A_KEEP):.0f}°/V {V_KEEP:.0f} to θ {degrees(THETA_LOW):.0f}°", flush=True)
    hdr = (f"KA| {'spawn':>5} {'arm':<8} {'kills':>5} {'events':<34} {'dealt':>5} {'taken':>5} "
           f"{'rds':>4} {'epS':>4} {'rngMin':>6} {'Vmed':>4} {'αmed':>4} {'αsat%':>5} {'nzMed':>5} "
           f"{'θmed':>4} {'dAGL':>6} {'tHit':>5}")
    print(hdr, flush=True)
    print("KA| " + "-" * (len(hdr) - 4), flush=True)

    for asp in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=asp, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        low = np.asarray(base.action_space.low, dtype=np.float32)
        high = np.asarray(base.action_space.high, dtype=np.float32)
        for arm in arms:
            rows = [roll(agent, env, base, 5000 + int(asp) * 100 + i, dt, arm, low, high)
                    for i in range(args.episodes)]
            M = np.concatenate([r["M"] for r in rows], axis=0)
            ev = {}
            for r in rows:
                ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
            n = len(rows)
            print(f"KA| {asp:>5.0f} {arm:<8} {sum(r['event'] == 'kill' for r in rows):>5} "
                  f"{' '.join(f'{k}×{v}' for k, v in sorted(ev.items())):<34} "
                  f"{np.mean([r['dealt'] for r in rows]):>5.0f} "
                  f"{np.mean([r['taken'] for r in rows]):>5.0f} "
                  f"{np.mean([r['spent'] for r in rows]):>4.0f} "
                  # Episode LENGTH: without it a short winning episode looks "flat and fast" purely
                  # because it ended before the mutual zoom-climb had time to happen.
                  f"{np.mean([len(r['M']) for r in rows]) * dt:>4.0f} "
                  f"{np.mean([r['M'][:, 0].min() for r in rows]):>6.0f} "
                  f"{np.median(M[:, 1]):>4.0f} {np.median(M[:, 2]):>4.1f} "
                  f"{100.0 * (M[:, 2] >= 14.0).mean():>5.0f} {np.median(M[:, 3]):>5.1f} "
                  f"{np.median(M[:, 4]):>+4.0f} "
                  f"{np.mean([r['M'][-1, 5] - r['M'][0, 5] for r in rows]):>+6.0f} "
                  f"{(M[:, 6] <= C.HIT_RADIUS_M).sum() * dt / n:>5.1f}", flush=True)
        env.close()
        print("KA|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
