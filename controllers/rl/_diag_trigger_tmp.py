"""Chase TRIGGER-TIME autopsy: at spawn aspect 55/60 the wall is ROUNDS, not aim — which one?

The c87 autopsy (runs/curriculum-87/autopsy_5000k.log) pins the chase failure to trigger volume,
not marksmanship:

    spawn  kills  rounds/ep  hittable  HP dealt  HP/round  dEs@min
       30    4/6        196        93        94      0.48      +46
       55    0/6         44        23        42      0.95      -528
       60    0/6         52        18        33      0.63      -578

At 55/60 it is TWICE as efficient per round as the bucket that kills, fires at 299 m (centre of the
250-600 m band where pooled hit% is 47-56%) and touches minAta 1° / minAsp 7°. It just fires ~7
rounds an episode instead of ~33. Two mutually exclusive causes, and every candidate lever depends
on which it is:

  OPPORTUNITY   — it is almost never in a usable firing solution, and shoots whenever it is.
                  ⇒ the lever is upstream: energy/geometry, so the pass lasts longer or repeats.
  FIRE DISCIPLINE — it sits in a usable solution and keeps the trigger up.
                  ⇒ the lever is the trigger itself, and is far cheaper.

The discriminator is the CONDITIONAL P(fire | hittable), which no existing instrument prints for the
chase. `_diag_headon_alt_tmp.py` computes it, but only for the head-on merge env (opp_heading 180).

Reported per spawn aspect:
  * t_hit    — seconds per episode with an HONEST solution: lay_miss <= HIT_RADIUS_M (8 m sphere),
               the sim's own ballistic test, not the ~3-4x looser reward/fire gate.
  * t_trig   — seconds with the trigger commanded down.
  * P(fire|hit) — THE number. Low ⇒ fire discipline. High ⇒ opportunity.
  * P(hit|inR)  — how often being in range is worth anything at all.
  * and a profile of the HITTABLE-BUT-NOT-FIRING ticks, because if that pool is large, its median
    geometry says exactly which gate is refusing.

    cd controllers/rl && uv run python -u _diag_trigger_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 30,55,60 --episodes 6
"""

import argparse
from math import degrees, hypot

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative, sigma_track  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DEG = 180.0 / np.pi
NCOL = 9


def roll(agent, env, base, seed, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    rows = []          # one row per control step, only while inside R_EFF
    n_steps = 0
    trig_all = 0       # trigger commanded, anywhere (incl. out of range)
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rg = relative(me, opp)
        # a[4] is AGENT space: map_tdmpc_action gives native = 0.5(a+1) and the env fires at
        # native >= 0.5, i.e. agent >= 0. Testing >= 0.5 here undercounts every trigger-down figure.
        fired = float(a[4] >= 0.0)
        trig_all += int(fired)
        n_steps += 1
        if rg.rng < C.R_EFF:
            rows.append((
                rg.rng,
                rg.lay_miss_m,                                   # honest: vs the 8 m hit sphere
                degrees(rg.lead_error),
                degrees(sigma_track(rg.rng)),                     # the autopsy's tolerance
                degrees(rg.ata),
                degrees(rg.aspect),
                rg.closure,
                rg.los_rate * DEG,
                fired,
            ))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    return dict(
        event=event, t_s=n_steps * dt, trig_all=trig_all * dt,
        dealt=C.HP_MAX - float(opp.health), taken=C.HP_MAX - float(me.health),
        spent=int(C.AMMO_MAX - me.ammo) if hasattr(C, "AMMO_MAX") else 0,
        M=np.asarray(rows).reshape(-1, NCOL),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="30,55,60")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
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

    print(f"TRIG {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy | vs native T3 | "
          f"chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g} | "
          f"{args.episodes} eps/aspect | honest hit = lay_miss <= {C.HIT_RADIUS_M:.0f} m", flush=True)
    hdr = (f"TRIG| {'spawn':>5} {'kill':>4} {'dealt':>5} {'tInR':>5} {'tHit':>5} {'tTrig':>6} "
           f"{'tBoth':>5} {'P(fire|hit)':>11} {'P(hit|inR)':>10} {'tSig':>5} {'P(f|sig)':>8}  events")
    print(hdr, flush=True)
    print("TRIG| " + "-" * (len(hdr) - 6), flush=True)

    for a in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=a, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        rows = [roll(agent, env, base, 5000 + int(a) * 100 + i, dt) for i in range(args.episodes)]
        env.close()
        eps = len(rows)
        M = np.concatenate([r["M"] for r in rows], axis=0) if eps else np.zeros((0, NCOL))
        rng, miss, lead, sig, ata, asp, clo, los, fired = (M[:, i] for i in range(NCOL))
        hit = miss <= C.HIT_RADIUS_M                # honest ballistic test
        sigok = lead <= sig                          # the autopsy's looser tolerance
        f = fired > 0.5
        kills = sum(r["event"] == "kill" for r in rows)
        ev = {}
        for r in rows:
            ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
        per_ep = dt / max(1, eps)
        print(f"TRIG| {a:>5.0f} {kills:>4} {np.mean([r['dealt'] for r in rows]):>5.0f} "
              f"{len(M) * per_ep:>5.1f} {hit.sum() * per_ep:>5.1f} "
              f"{np.mean([r['trig_all'] for r in rows]):>6.1f} {(hit & f).sum() * per_ep:>5.1f} "
              f"{(100.0 * f[hit].mean() if hit.any() else float('nan')):>10.0f}% "
              f"{(100.0 * hit.mean() if len(M) else float('nan')):>9.0f}% "
              f"{sigok.sum() * per_ep:>5.1f} "
              f"{(100.0 * f[sigok].mean() if sigok.any() else float('nan')):>7.0f}%  "
              + " ".join(f"{k}×{v}" for k, v in sorted(ev.items())), flush=True)

        # The pool that decides it: HITTABLE and NOT firing. If this is big, its median geometry names
        # the gate that is refusing; if it is empty, there is nothing to unlock and the wall is upstream.
        for label, sel in (("HIT & no-fire", hit & ~f), ("HIT & fire", hit & f),
                           ("inR & no-fire", ~hit & ~f)):
            if not sel.any():
                print(f"TRIG|      {label:>13}: 0 ticks", flush=True)
                continue
            print(f"TRIG|      {label:>13}: {int(sel.sum()):>5} ticks ({sel.sum() * per_ep:>4.1f} s/ep) "
                  f"rng {np.median(rng[sel]):>4.0f} ata {np.median(ata[sel]):>3.0f}° "
                  f"asp {np.median(asp[sel]):>3.0f}° clo {np.median(clo[sel]):>+4.0f} "
                  f"ω {np.median(los[sel]):>3.0f}°/s miss {np.median(miss[sel]):>5.0f} m", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
