"""Aspect-parametrized SPRAY-AND-TRACK feasibility probe (curriculum-20 calibration).

curriculum-19 confirmed: the -40 damage-timeout penalty CONVERTS reachable offense to kills
(aspect 0 went draws→kills, ret +120) but CANNOT teach an unreachable skill (aspect 30 stalled
at 0-1% damage, ret pinned at the -40 penalty floor — the policy eats the penalty passively
rather than attempt a turn-to-guns it never discovers). The exploration gap: the guided-fire
override supplies the TRIGGER but never the MANEUVER, and the c18/c19 reversal rungs
(posture="offensive") have NO expert seed (the c17 edit scoped the PursuitExpert to
posture=="defensive"). This probe answers whether re-enabling the expert seed on the LOW-aspect
offensive rungs is viable — i.e. at each aspect, flying the expert's STICK + a liberal
forced-fire trigger (the validated `_diag_t3_spray` combo), from posture="offensive" at the
given `aspect_deg`:
  * does the expert DECK (the failure mode that scoped it off the bot-on-six rungs)?
  * does expert-stick + spray deal >= the rung's damage bar (12 HP @ aspect30, 10 @ 60, 8 @ 90)?
If it doesn't deck and clears the bar, seeding those rungs will inject the turn-to-guns demos the
policy is failing to explore. If it decks (expected once the bot is abeam/behind), that aspect
stays a true reversal rung with no seed.

Run: uv run python /workspace/crates/astrum-env/python/_diag_t3_aspect_spray.py
"""
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401  (path wiring)

import numpy as np

from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import relative


def fly(env, q=0.05, max_steps=7300):
    env.reset()
    me0 = env._states[env.learner_id]
    opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo)
    hp0 = float(opp0.health)
    event = None
    min_rng = 1e9
    min_alt = 1e9
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
    for _ in range(max_steps):
        a = np.array(info["expert_action"], dtype=np.float32).copy()
        me = env._states[env.learner_id]
        opp = env._states[env.opponent_id]
        rg = relative(me, opp)
        a[-1] = 1.0 if rg.gun_quality >= q else 0.0  # forced-fire trigger override
        obs, rew, term, trunc, info = env.step(a)
        me = env._states[env.learner_id]
        opp = env._states[env.opponent_id]
        rng = float(np.linalg.norm(opp.pos - me.pos))
        min_rng = min(min_rng, rng)
        min_alt = min(min_alt, float(me.alt))
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
        "min_alt": min_alt,
    }


def run(aspect, sep, bar, seeds=6, max_time_s=60.0, jitter=0.5):
    print(
        f"\n=== T3 SPRAY @ aspect {aspect:.0f}deg / {sep:.0f}m  (posture=offensive "
        f"bar>={bar:.0f}HP max_time={max_time_s:.0f}s jitter={jitter}) ==="
    )
    kills = decks = 0
    dmgs = []
    for s in range(seeds):
        env = make_tier3_env(
            posture="offensive", aspect_deg=float(aspect), separation_m=float(sep),
            max_time_s=max_time_s, jitter=jitter, jitter_seed=5000 + s,
        )
        env.reset(seed=5000 + s)
        r = fly(env)
        dmgs.append(r["dmg_dealt"])
        kills += r["event"] == "kill"
        decks += r["event"] == "hard_deck"
        print(
            f"  seed {s}: event={str(r['event']):<11} dmg_dealt={r['dmg_dealt']:5.0f} "
            f"rounds={r['rounds_fired']:4d} min_rng={r['min_rng']:5.0f}m "
            f"min_alt={r['min_alt']:6.0f}m dmg_taken={r['dmg_taken']:.0f}"
        )
        env.close()
    dmgs = np.array(dmgs)
    print(
        f"  --> {kills}/{seeds} kills | {decks}/{seeds} DECKS | dmg mean {dmgs.mean():.1f} "
        f"median {np.median(dmgs):.1f} max {dmgs.max():.0f} | >=bar({bar:.0f}HP) "
        f"{(dmgs >= bar).mean() * 100:.0f}%"
    )


print(
    f"combat model: HP_MAX={C.HP_MAX} R_EFF={C.R_EFF} DECK_AGL={C.DECK_AGL}m "
    f"hit_radius=8.0m dmg/hit=2.0"
)
# (aspect, sep, rung damage bar) — the c18/c19 reversal ladder's low-aspect offensive rungs.
for aspect, sep, bar in ((30, 700, 12), (60, 700, 10), (90, 700, 8)):
    run(aspect, sep, bar)
