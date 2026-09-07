"""T3 OFFENSE feasibility probe (curriculum-16 → -17 pivot).

curriculum-16 mastered the SURVIVE-gated T3 ladder but produced a pure passive evader:
eval shows 0.0 HP dealt to the T3 bot at EVERY range, even with the MPC planner and even at
600m. Before building a DAMAGE-gated T3 offense curriculum, answer the prerequisite question
this script exists to answer: *can the best scripted tracker (PursuitExpert) deal meaningful
damage to the aggressive, evading Tier-3 bot at all* — and how much per range? If even perfect
lead-pursuit chip-draws to ~0 HP, offense-vs-T3 is a physics/opponent-difficulty wall and no
gate will teach it; if the expert lands real damage, the RL policy can be seeded/gated to do
the same. The per-range damage ceiling here calibrates the c17 damage-gate thresholds (the
lesson from T2@700m: never gate above the feasible ceiling).

Run from controllers/rl so astrum_rl puts astrum_env + astrum_dogfight on the path:
    uv run python /workspace/crates/astrum-env/python/_diag_t3_offense.py
"""
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401  (path wiring: puts astrum_env.so + astrum_dogfight on sys.path)

import numpy as np

from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env


def fly_expert(env, max_steps=7300):
    env.reset()
    me0 = env._states[env.learner_id]
    opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo)
    hp0 = float(opp0.health)
    event = None
    min_rng = 1e9
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
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


def run(sep, seeds=6, max_time_s=60.0, jitter=0.5):
    print(f"\n=== T3 @ {sep:.0f}m  (max_time={max_time_s:.0f}s jitter={jitter}) ===")
    kills = 0
    dmgs = []
    for s in range(seeds):
        env = make_tier3_env(
            separation_m=float(sep), max_time_s=max_time_s, jitter=jitter, jitter_seed=2000 + s
        )
        env.reset(seed=2000 + s)
        r = fly_expert(env)
        dmgs.append(r["dmg_dealt"])
        est_hits = r["dmg_dealt"] / 2.0  # 2 HP per hit
        hitrate = (est_hits / r["rounds_fired"] * 100.0) if r["rounds_fired"] else 0.0
        killed = r["event"] == "kill"
        kills += killed
        print(
            f"  seed {s}: event={str(r['event']):<11} dmg_dealt={r['dmg_dealt']:5.0f} "
            f"(~{est_hits:.0f} hits) rounds={r['rounds_fired']:3d} hit%={hitrate:4.1f} "
            f"min_rng={r['min_rng']:5.0f}m dmg_taken={r['dmg_taken']:.0f}"
        )
        env.close()
    dmgs = np.array(dmgs)
    print(
        f"  --> {kills}/{seeds} kills | dmg dealt mean {dmgs.mean():.1f} "
        f"median {np.median(dmgs):.1f} max {dmgs.max():.0f} | ≥15HP {(dmgs>=15).mean()*100:.0f}%"
    )


print(
    f"combat model: HP_MAX={C.HP_MAX}  AMMO_MAX={C.AMMO_MAX}  R_EFF={C.R_EFF}  "
    f"MUZZLE={C.MUZZLE_MPS}  FIRE_LEAD={np.degrees(C.FIRE_LEAD_RAD):.1f}deg"
)
# Can the expert damage the T3 bot at each rung of the (proposed) c17 offense ladder?
for sep in (400, 600, 900, 1400, 2200):
    run(sep)
