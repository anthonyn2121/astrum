"""T3 SPRAY-AND-TRACK damage-ceiling probe (curriculum-17 calibration).

Both prior probes flew the PursuitExpert *including its own trigger*, which only fires
inside a conservative 4° lead cone — a gate that essentially never opens on the
continuously-jinking T3 tactician (0 rounds fired in 23/24 chase episodes). But the ENV
hit model is NOT cone-gated: `fire_guns`+`resolve_hits` spawn real ballistic rounds and
score a hit whenever a bullet passes within hit_radius_m=8.0 m of the target (2 HP each).
An RL policy's trigger is a LEARNED action, free to spray far more liberally than the
expert's gate. So the question a damage-gated offense curriculum hinges on is: how much HP
can a liberal-firing tracker actually put on the jinking T3 from an offensive start?

This probe flies the expert's STICK (aileron/elevator/rudder/throttle — its maneuvering is
a decent tracker) but REPLACES the trigger with a forced-fire rule, from posture="defensive"
(bot ahead, learner on its tail = offensive start). Two rules bracket the ceiling:
  * "quality": fire when gun_quality >= Q (in-range AND roughly on-aim) — the disciplined
    spray an RL fire-override would learn.
  * "inrange": fire whenever range <= R_EFF regardless of aim — the absolute spray ceiling
    (wastes ammo, but shows the max hits geometry allows).
The per-range HP dealt here calibrates whether a T3 damage gate is feasible and at what bar
(the T2@700m lesson: never gate above the feasible ceiling).

Run: uv run python /workspace/crates/astrum-env/python/_diag_t3_spray.py
"""
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401  (path wiring)

import numpy as np

from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import relative


def fly(env, rule="quality", q=0.05, max_steps=7300):
    # ⚠ B1 re-scaled what `q` MEANS. `gun_quality` is now `p_hit(rng) · track` (the measured curve),
    # not the old linear `(R_EFF − rng)/R_EFF · track`, so on a perfect lay this default bar of 0.05
    # selects "inside ~734 m" where it used to select "inside ~1140 m". At the 700 m case this probe
    # is usually run at, q = 0.060 — only 0.010 above the bar — so a slightly worse lay now holds
    # fire where it previously sprayed. Any re-run is therefore NOT comparable to the pre-B1 numbers
    # quoted in train_parallel.py's damage-gate rationale; pass an explicit `--q` and say which.
    # (The [[curriculum-92-lateral-separation]] lesson: a silently re-scaled window fakes a result.)
    env.reset()
    me0 = env._states[env.learner_id]
    opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo)
    hp0 = float(opp0.health)
    event = None
    min_rng = 1e9
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
    for _ in range(max_steps):
        a = np.array(info["expert_action"], dtype=np.float32).copy()
        # Replace the expert's cone-gated trigger with a liberal forced-fire rule.
        me = env._states[env.learner_id]
        opp = env._states[env.opponent_id]
        rg = relative(me, opp)
        if rule == "quality":
            a[-1] = 1.0 if rg.gun_quality >= q else 0.0
        else:  # "inrange"
            a[-1] = 1.0 if rg.in_range else 0.0
        obs, rew, term, trunc, info = env.step(a)
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


def run(sep, rule, seeds=6, max_time_s=60.0, jitter=0.5):
    print(f"\n=== T3 SPRAY[{rule}] @ {sep:.0f}m  (posture=defensive max_time={max_time_s:.0f}s jitter={jitter}) ===")
    kills = 0
    dmgs = []
    for s in range(seeds):
        env = make_tier3_env(
            posture="defensive", separation_m=float(sep),
            max_time_s=max_time_s, jitter=jitter, jitter_seed=4000 + s,
        )
        env.reset(seed=4000 + s)
        r = fly(env, rule=rule)
        dmgs.append(r["dmg_dealt"])
        est_hits = r["dmg_dealt"] / 2.0
        hitrate = (est_hits / r["rounds_fired"] * 100.0) if r["rounds_fired"] else 0.0
        kills += r["event"] == "kill"
        print(
            f"  seed {s}: event={str(r['event']):<11} dmg_dealt={r['dmg_dealt']:5.0f} "
            f"(~{est_hits:.0f} hits) rounds={r['rounds_fired']:4d} hit%={hitrate:4.1f} "
            f"min_rng={r['min_rng']:5.0f}m dmg_taken={r['dmg_taken']:.0f}"
        )
        env.close()
    dmgs = np.array(dmgs)
    print(
        f"  --> {kills}/{seeds} kills | dmg dealt mean {dmgs.mean():.1f} "
        f"median {np.median(dmgs):.1f} max {dmgs.max():.0f} | ≥15HP {(dmgs>=15).mean()*100:.0f}% "
        f"| ≥30HP {(dmgs>=30).mean()*100:.0f}%"
    )


print(
    f"combat model: HP_MAX={C.HP_MAX} AMMO_MAX={C.AMMO_MAX} R_EFF={C.R_EFF} "
    f"MUZZLE={C.MUZZLE_MPS} hit_radius=8.0m dmg/hit=2.0"
)
for sep in (400, 700):
    run(sep, "quality")
    run(sep, "inrange")
