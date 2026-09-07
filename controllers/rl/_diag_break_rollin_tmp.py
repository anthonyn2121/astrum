"""Does the T3's DEFENSIVE break PULL BEFORE IT ROLLS — turning a level break into a zoom climb?

`_diag_who_climbs_tmp.py` found that the scripted T3 sits in `defensive` for 98.2% of a spawn-55
episode and climbs there at +63 m/s / γ +28.1°, committing to the climb at 0.1-0.2 s in 12/12
episodes — 0.5-1.2 s BEFORE the learner. So the mutual zoom that makes the fight un-shootable is
built by the OPPONENT, and the learner's climb is a reaction to it.

That should be impossible. `_break` (opponent.py:423) rolls to BREAK_BANK_RAD = 85° and pulls to the
AoA limit, and `_diag_stalemate_tmp.py` measured the T3 actually holding 88° of bank. At 88° the lift
vector is 2° off horizontal: vertical lift is ~0.03·L, so a max-g break should hold altitude at best
and descend in practice. It cannot climb at γ +28°.

Both facts are true only if the PULL LANDS BEFORE THE ROLL DOES. Max Nz demanded while the wings are
still level is a pure vertical pull: the nose rockets up, and only then does the jet roll to 88° —
by which point the velocity vector is already climbing steeply and the "level break" is a climbing
spiral. Every other pull in the file guards against exactly this:

    lead_pursuit  :341  aligned = max(cos(bank_err), 0.0)   "pull hardest once rolled onto the target"
    Tier2 break   :214  -0.25 while rolling, then :221 -0.9 once |φ| >= 0.9 * BREAK_BANK_RAD
    Tier3 _break  :434  _corner_limited_pull(-1.0, ...)     <-- ungated, full pull from tick 0

So this measures the roll-in itself. For each episode it logs, per tick from the moment DEFENSIVE is
entered, the elevator the T3 COMMANDED (captured by wrapping `opponent.act`) against |φ|, θ and γ.

The discriminator is the ORDER, and the two outcomes imply opposite fixes:
  * If θ/γ rise while |φ| is still small, the pull leads the roll => a BUG in `_break`, fixable by
    porting the `aligned` gate its two sibling laws already have, and the whole spawn-55/60 wall is
    an artifact of a degenerate opponent rather than a skill the learner lacks.
  * If |φ| reaches ~85° first and it STILL climbs, the break is innocent and the climb comes from
    somewhere else (thrust vector at high α, or the FLCS's Nz frame), which needs a different fix.

`t_rel` is measured from DEFENSIVE entry, not from episode start, because the transition happens at
0.1-0.2 s and the interesting window is the ~2 s of roll-in after it.

    cd controllers/rl && uv run python -u _diag_break_rollin_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 55 --episodes 2
"""

import argparse
from math import asin, degrees

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402


def _gamma(s):
    v = np.asarray(s.vel, dtype=np.float64)
    sp = float(np.linalg.norm(v))
    return degrees(asin(float(np.clip(v[2] / sp, -1.0, 1.0)))) if sp > 1e-6 else 0.0


def roll(agent, env, base, seed):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    # Capture what the T3 actually COMMANDS. The FSM state alone cannot show the roll/pull ordering.
    opp = base.opponent
    real_act = opp.act
    box = {}

    def spy(s, target):
        a = real_act(s, target)
        box["a"] = [float(x) for x in a]
        box["st"] = getattr(opp, "state", "?")
        return a

    opp.act = spy
    t0, event = True, None
    rows = []
    try:
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, _r, term, trunc, info = env.step(a)
            op = base._states[base.opponent_id]
            cmd = box.get("a", [0.0] * 5)
            rows.append((box.get("st", "?"), abs(degrees(op.phi)), degrees(op.theta), _gamma(op),
                         degrees(op.alpha), op.airspeed, op.alt - base.ground_alt,
                         cmd[0], cmd[1]))
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
    finally:
        opp.act = real_act
    return dict(event=event, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="55")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--window-s", type=float, default=3.0)
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

    print(f"ROLLIN {args.checkpoint} | T3 DEFENSIVE roll-in | BREAK_BANK {degrees(C.BREAK_BANK_RAD):.0f}° "
          f"ALPHA_SOFT {degrees(C.ALPHA_SOFT_RAD):.0f}° V_ENERGY_LOW {C.V_ENERGY_LOW:.0f} | "
          f"elev<0 = pull (max -1)", flush=True)

    for asp in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=asp, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        for i in range(args.episodes):
            r = roll(agent, env, base, 5000 + int(asp) * 100 + i)
            rows = r["rows"]
            d0 = next((k for k, x in enumerate(rows) if x[0] == "defensive"), None)
            print(f"ROLLIN| ==== spawn {asp:.0f}° ep{i} [{r['event']}] DEFENSIVE entered at "
                  f"{'never' if d0 is None else f'{d0 * dt:.2f}s'} ====", flush=True)
            if d0 is None:
                continue
            print(f"ROLLIN|  {'t_rel':>6} {'state':<10} {'|φ|':>5} {'θ':>5} {'γ':>5} {'α':>5} "
                  f"{'V':>4} {'AGL':>5} {'ail':>5} {'elev':>5}", flush=True)
            end = min(len(rows), d0 + int(round(args.window_s / dt)))
            for k in range(d0, end):
                st, phi, th, gam, al, v, agl, ail, elev = rows[k]
                print(f"ROLLIN|  {(k - d0) * dt:>6.2f} {st:<10} {phi:>5.0f} {th:>+5.0f} {gam:>+5.0f} "
                      f"{al:>5.1f} {v:>4.0f} {agl:>5.0f} {ail:>+5.2f} {elev:>+5.2f}", flush=True)
            # The summary claim: at the instant the roll FINALLY reaches 85%, how much climb is
            # already banked? If γ is large there, the pull won the race.
            rolled = next((k for k in range(d0, len(rows))
                           if rows[k][1] >= 0.85 * degrees(C.BREAK_BANK_RAD)), None)
            if rolled is not None:
                st, phi, th, gam, al, v, agl, ail, elev = rows[rolled]
                print(f"ROLLIN|  -> |φ| first reached {0.85 * degrees(C.BREAK_BANK_RAD):.0f}° at "
                      f"t_rel {(rolled - d0) * dt:.2f}s, by which point θ {th:+.0f}° γ {gam:+.0f}° "
                      f"and it had already gained {agl - rows[d0][6]:+.0f} m", flush=True)
            else:
                print(f"ROLLIN|  -> |φ| NEVER reached {0.85 * degrees(C.BREAK_BANK_RAD):.0f}°"
                      f" (max {max(x[1] for x in rows[d0:]):.0f}°)", flush=True)
            print("ROLLIN|", flush=True)
        env.close()
    print(flush=True)


if __name__ == "__main__":
    main()
