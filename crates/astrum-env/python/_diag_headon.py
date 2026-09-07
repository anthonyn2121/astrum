"""curriculum-21 feasibility: is a HEAD-ON merge learnable where the co-heading offset wasn't?
Bot placed dead-ahead (aspect 0) pointing BACK at the learner (opp_heading_deg=180) => they
close head-on at ~2x speed, pass, and turn to fight — no overtake wall. Fly the range-keeper
expert stick + spray(q>=0.05) vs T3 and T2 and measure damage dealt/taken. If the turning
fight yields meaningful damage (unlike the ~0 forward-offset overtake), a head-on curriculum
vs the maneuvering bot is viable; the warm-start's tail-chase gun skill should exceed the stick."""
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
            "dmg_taken": C.HP_MAX - float(me.health),
            "rounds": ammo0 - int(me.ammo), "min_rng": min_rng, "min_alt": min_alt}


def run(opp_tier, aspect, sep, seeds=6, max_time_s=60.0, jitter=0.5):
    lbl = {2: "T2-break", 3: "T3-full"}[opp_tier]
    print(f"\n=== HEAD-ON vs {lbl} @ aspect {aspect:.0f} / {sep:.0f}m opp_hdg=180 ===")
    kills = decks = deaths = 0; dd = []; dt = []
    for s in range(seeds):
        env = make_tier3_env(posture="offensive", opponent_tier=opp_tier, aspect_deg=float(aspect),
                             opp_heading_deg=180.0, separation_m=float(sep), max_time_s=max_time_s,
                             jitter=jitter, jitter_seed=8000 + s)
        env.reset(seed=8000 + s)
        r = fly(env); dd.append(r["dmg_dealt"]); dt.append(r["dmg_taken"])
        kills += r["event"] == "kill"; decks += r["event"] == "hard_deck"
        deaths += r["event"] in ("killed", "death")
        print(f"  seed {s}: {str(r['event']):<11} dmg_dealt={r['dmg_dealt']:5.0f} "
              f"dmg_taken={r['dmg_taken']:5.0f} rounds={r['rounds']:4d} "
              f"min_rng={r['min_rng']:5.0f}m min_alt={r['min_alt']:6.0f}m")
        env.close()
    dd = np.array(dd); dt = np.array(dt)
    print(f"  --> {kills}/{seeds} kills | {deaths}/{seeds} deaths | {decks}/{seeds} DECKS | "
          f"dmg_dealt mean {dd.mean():.1f} med {np.median(dd):.1f} max {dd.max():.0f} | "
          f"dmg_taken mean {dt.mean():.1f}")


print(f"combat: HP_MAX={C.HP_MAX} DECK_AGL={C.DECK_AGL}m hit_radius=8.0m dmg/hit=2.0")
for opp_tier in (2, 3):
    for aspect in (0, 20):
        run(opp_tier, aspect, 1500)
