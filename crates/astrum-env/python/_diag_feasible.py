"""Feasibility probe: can the BEST available tracker (scripted PursuitExpert) convert a
gun KILL on each scripted target within one magazine? If even perfect lead-pursuit caps at
a partial-damage chip-draw, the combat model (hit geometry / ammo / HP) is the bottleneck —
no reward will teach the RL policy to do something the physics won't allow.

Run from controllers/rl so astrum_rl puts astrum_env + astrum_dogfight on the path:
    uv run python /workspace/crates/astrum-env/python/_diag_feasible.py
"""
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401  (path wiring: puts astrum_env.so + astrum_dogfight on sys.path)

import numpy as np

from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier1_env, make_tier2_env, make_tier3_env


def fly_expert(env, max_steps=7300):
    obs, info = env.reset()
    me0 = env._states[env.learner_id]
    opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo)
    hp0 = float(opp0.health)
    event = None
    min_rng = 1e9
    for _ in range(max_steps):
        obs, rew, term, trunc, info = env.step(info["expert_action"])
        me = env._states[env.learner_id]
        opp = env._states[env.opponent_id]
        rng = float(np.linalg.norm(opp.pos - me.pos))
        min_rng = min(min_rng, rng)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = env._states[env.learner_id]
    opp = env._states[env.opponent_id]
    return {
        "event": event,
        "dmg_dealt": hp0 - float(opp.health),
        "dmg_taken": C.HP_MAX - float(me.health),
        "rounds_fired": ammo0 - int(me.ammo),
        "min_rng": min_rng,
    }


def run(label, make, seeds=4, max_time_s=60.0, jitter=0.5, **kw):
    print(f"\n=== {label}  (max_time={max_time_s:.0f}s jitter={jitter}) ===")
    kills = 0
    for s in range(seeds):
        env = make(max_time_s=max_time_s, jitter=jitter, jitter_seed=1000 + s, **kw)
        # With jitter, reset(seed) reseeds the per-episode geometry RNG → a distinct engagement
        # per seed (measures feasibility over the jittered distribution the RL policy trains on).
        env.reset(seed=1000 + s)
        r = fly_expert(env)
        est_hits = r["dmg_dealt"] / 2.0  # 2 HP per hit
        hitrate = (est_hits / r["rounds_fired"] * 100.0) if r["rounds_fired"] else 0.0
        killed = r["event"] == "kill"
        kills += killed
        print(
            f"  seed {s}: event={r['event']:<11} dmg_dealt={r['dmg_dealt']:5.0f} "
            f"(~{est_hits:.0f} hits) rounds={r['rounds_fired']:3d} hit%={hitrate:4.1f} "
            f"min_rng={r['min_rng']:5.0f}m dmg_taken={r['dmg_taken']:.0f}"
        )
        env.close()
    print(f"  --> {kills}/{seeds} kills")


print(f"combat model: HP_MAX={C.HP_MAX}  AMMO_MAX={C.AMMO_MAX}  R_EFF={C.R_EFF}  "
      f"MUZZLE={C.MUZZLE_MPS}  LEAD_SIGMA={np.degrees(C.LEAD_SIGMA):.1f}deg  "
      f"FIRE_LEAD={np.degrees(C.FIRE_LEAD_RAD):.1f}deg  GUN_FIRE_MISS_M={C.GUN_FIRE_MISS_M}")

# Decisive check for the T2 "reach" rung: is a maneuvering-T2 kill feasible between the
# confirmed-feasible 400m and the confirmed-infeasible 700m? (60s only — the clock the
# acceptance eval uses; 120s probes are too slow and don't match eval.)
run("T2 @ 400m (point-blank breaker)", make_tier2_env, separation_m=400.0, seeds=3)
run("T2 @ 500m", make_tier2_env, separation_m=500.0, seeds=3)
run("T2 @ 550m", make_tier2_env, separation_m=550.0, seeds=3)
run("T2 @ 700m", make_tier2_env, separation_m=700.0, seeds=3)
