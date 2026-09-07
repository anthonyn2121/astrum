"""T3 OFFENSE-FROM-AN-OFFENSIVE-START feasibility probe (curriculum-17 pivot).

`_diag_t3_offense.py` flew the PursuitExpert vs the T3 bot in the DEFAULT posture
("offensive" = the *bot* sits on the LEARNER's six) and got 0 HP / 0 kills at every
range with the expert decking itself — but that geometry is a DEFENSIVE fight for the
learner, where a pure lead-pursuit tracker is definitionally the wrong tool (it hard-
turns into its own six and decks). That result does NOT establish that the T3 *AI* is
un-gunnable; it only shows pursuit can't attack a target on its own six.

This probe answers the real prerequisite for a geometry curriculum: with the learner
starting in an OFFENSIVE position (`posture="defensive"` → tier1-style placement, bot
AHEAD, learner on its tail) against the SAME Tier3Opponent AI, can the best scripted
tracker deal meaningful damage / kill it? This is the exact mirror of `_diag_feasible.py`
(which chip-damaged the T2 breaker ~40-50 HP) but against the smarter T3 tactician.

  * If the expert lands real damage/kills here → offense-vs-T3 IS feasible from an
    offensive start → a geometry curriculum (offense-first at posture="defensive",
    gate on kills, then slide to posture="offensive" bot-on-six defense) is viable and
    the per-range damage ceiling calibrates its gate thresholds.
  * If even a perfect tracker chip-draws to ~0 from behind → the T3 AI's evasion beats
    the 4° fire cone outright → gunning T3 needs self-play / a reversal, not a gate.

Run from anywhere (path-wires astrum_env + astrum_dogfight via astrum_rl):
    uv run python /workspace/crates/astrum-env/python/_diag_t3_chase.py
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
    print(f"\n=== T3 CHASE @ {sep:.0f}m  (posture=defensive max_time={max_time_s:.0f}s jitter={jitter}) ===")
    kills = 0
    dmgs = []
    for s in range(seeds):
        env = make_tier3_env(
            posture="defensive",  # bot AHEAD, learner on its tail = OFFENSIVE start
            separation_m=float(sep),
            max_time_s=max_time_s,
            jitter=jitter,
            jitter_seed=3000 + s,
        )
        env.reset(seed=3000 + s)
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
        f"median {np.median(dmgs):.1f} max {dmgs.max():.0f} | ≥15HP {(dmgs>=15).mean()*100:.0f}% "
        f"| ≥30HP {(dmgs>=30).mean()*100:.0f}%"
    )


print(
    f"combat model: HP_MAX={C.HP_MAX}  AMMO_MAX={C.AMMO_MAX}  R_EFF={C.R_EFF}  "
    f"MUZZLE={C.MUZZLE_MPS}  FIRE_LEAD={np.degrees(C.FIRE_LEAD_RAD):.1f}deg"
)
# Can the expert damage the T3 AI from an OFFENSIVE start (chasing) at each candidate
# offense-first rung? Mirror of _diag_feasible.py's T1/T2 chase ranges.
for sep in (400, 700, 1200, 2000):
    run(sep)
