"""c20-v2 calibration: expert-stick + spray(q=0.05) vs the T1 DUMMY at 700m, across aspect
60/90/120, to set REACHABLE damage bars (kill is infeasible even for the expert @900m).
Confirms whether the beam merge at the warm-start's 700m range chips a gate-able damage bar."""
import sys
sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import relative


def fly(env, q=0.05, max_steps=7300):
    env.reset()
    me0 = env._states[env.learner_id]; opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo); hp0 = float(opp0.health)
    event = None; min_rng = 1e9; min_alt = 1e9
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
    for _ in range(max_steps):
        a = np.array(info["expert_action"], dtype=np.float32).copy()
        me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
        a[-1] = 1.0 if relative(me, opp).gun_quality >= q else 0.0
        obs, rew, term, trunc, info = env.step(a)
        me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
        min_rng = min(min_rng, float(np.linalg.norm(opp.pos - me.pos)))
        min_alt = min(min_alt, float(me.alt))
        if info.get("event"): event = info["event"]
        if term or trunc: break
    me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
    return {"event": event, "dmg_dealt": hp0 - float(opp.health),
            "rounds": ammo0 - int(me.ammo), "min_rng": min_rng, "min_alt": min_alt}


def run(aspect, sep, seeds=6, max_time_s=60.0, jitter=0.5):
    print(f"\n=== T1-DUMMY SPRAY @ aspect {aspect:.0f} / {sep:.0f}m (offensive jit={jitter}) ===")
    kills = decks = 0; dmgs = []
    for s in range(seeds):
        env = make_tier3_env(posture="offensive", opponent_tier=1, aspect_deg=float(aspect),
                             separation_m=float(sep), max_time_s=max_time_s, jitter=jitter,
                             jitter_seed=6000 + s)
        env.reset(seed=6000 + s)
        r = fly(env); dmgs.append(r["dmg_dealt"])
        kills += r["event"] == "kill"; decks += r["event"] == "hard_deck"
        print(f"  seed {s}: {str(r['event']):<11} dmg={r['dmg_dealt']:5.0f} rounds={r['rounds']:4d} "
              f"min_rng={r['min_rng']:5.0f}m min_alt={r['min_alt']:6.0f}m")
        env.close()
    dmgs = np.array(dmgs)
    print(f"  --> {kills}/{seeds} kills | {decks}/{seeds} DECKS | dmg mean {dmgs.mean():.1f} "
          f"median {np.median(dmgs):.1f} max {dmgs.max():.0f}")


print(f"combat: HP_MAX={C.HP_MAX} DECK_AGL={C.DECK_AGL}m hit_radius=8.0m dmg/hit=2.0")
for aspect in (60, 90, 120):
    run(aspect, 700)
