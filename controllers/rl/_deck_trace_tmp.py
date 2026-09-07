"""TEMP diagnostic: trace ONE T3-vs-T3 merge episode to see WHY the learner-side T3
decks at aspect 75. Flies the native Tier3Opponent as the learner (AR=1 = 120 Hz, dt=1/120
= identical config to the native opponent), logs the trajectory + FSM state + commanded
action every 0.25 s. Answers: genuine max-performance BFM into the ground, or a spawn/geometry
artifact (e.g. spawning low)?

    cd controllers/rl && uv run python -u _deck_trace_tmp.py --aspect 75
"""

from __future__ import annotations

import argparse
from math import degrees

import numpy as np

from astrum_rl import _paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", type=float, default=75.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=30)  # log cadence in sim ticks (30 = 0.25 s)
    args = ap.parse_args()

    _paths.require_astrum_env()
    from astrum_dogfight.geometry import relative
    from astrum_dogfight.opponent import Tier3Opponent

    from astrum_rl.envs import invert_tdmpc_action, make_tdmpc_env

    env = make_tdmpc_env(
        3,
        posture="offensive",
        aspect_deg=args.aspect,
        separation_m=700.0,
        action_repeat=1,
        frame_stack=1,
        max_time_s=60.0,
        rate_hz=120.0,
        jitter=args.jitter,
    )
    base = env.unwrapped
    low = np.asarray(base.action_space.low, dtype=np.float32)
    high = np.asarray(base.action_space.high, dtype=np.float32)
    expert = Tier3Opponent(dt=1.0 / 120.0)

    env.reset(seed=args.seed)
    expert.reset(base._states[base.learner_id])
    me0 = base._states[base.learner_id]
    op0 = base._states[base.opponent_id]
    print(f"aspect {args.aspect:g} jitter {args.jitter:g} seed {args.seed}")
    print(f"SPAWN  learner alt={me0.alt:7.1f}  bandit alt={op0.alt:7.1f}  sep={relative(me0, op0).rng:6.0f}m")
    print(
        f"{'t':>5} {'state':>10} {'alt':>7} {'the':>5} {'phi':>6} {'spd':>5} "
        f"{'rng':>6} {'ata':>5} | {'ail':>5} {'elev':>5} {'thr':>4} {'fire':>4}"
    )
    step = 0
    while True:
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        native = expert.act(me, opp)
        a = invert_tdmpc_action(np.asarray(native, dtype=np.float32), low, high)
        if step % args.every == 0:
            rg = relative(me, opp)
            print(
                f"{step / 120.0:5.2f} {expert.state:>10} {me.alt:7.1f} {degrees(me.theta):5.1f} "
                f"{degrees(me.phi):6.1f} {me.airspeed:5.1f} {rg.rng:6.0f} {degrees(rg.ata):5.1f} | "
                f"{native[0]:+5.2f} {native[1]:+5.2f} {native[3]:4.2f} {native[4]:4.1f}"
            )
        _obs, _r, terminated, truncated, info = env.step(a)
        step += 1
        if terminated or truncated:
            me = base._states[base.learner_id]
            opp = base._states[base.opponent_id]
            print(
                f"END t={step / 120.0:.2f}s  event={info.get('event')}  "
                f"term={terminated} trunc={truncated}  "
                f"learner alt={me.alt:.1f} hp={me.health:.0f}  bandit alt={opp.alt:.1f} hp={opp.health:.0f}"
            )
            break
    env.close()


if __name__ == "__main__":
    main()
