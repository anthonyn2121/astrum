"""Is the forward-offset merge (aspect 60/90) closeable, or a co-speed physics wall?
Fly the expert's AIM (lead_pursuit heading) but FIREWALL throttle to 1.0 (override the
range-keeper's taper) + spray q>=0.05. If min_rng drops well below the expert's ~200m
plateau and damage rises, the forward-offset merge is an OVERTAKE skill the RL policy
(which controls throttle) can learn. If it still plateaus, it's a turn-circle wall."""
import sys
sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import relative


def fly(env, q=0.05, max_steps=7300, firewall=True):
    env.reset()
    me0 = env._states[env.learner_id]; opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo); hp0 = float(opp0.health)
    event = None; min_rng = 1e9; min_alt = 1e9
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
    for _ in range(max_steps):
        a = np.array(info["expert_action"], dtype=np.float32).copy()
        if firewall:
            a[3] = 1.0  # override the range-keeper: firewall throttle to force overtake
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


def run(aspect, sep, firewall, seeds=6, max_time_s=60.0, jitter=0.5):
    tag = "FIREWALL" if firewall else "range-keep"
    print(f"\n=== T1-DUMMY {tag} @ aspect {aspect:.0f} / {sep:.0f}m ===")
    kills = decks = 0; dmgs = []
    for s in range(seeds):
        env = make_tier3_env(posture="offensive", opponent_tier=1, aspect_deg=float(aspect),
                             separation_m=float(sep), max_time_s=max_time_s, jitter=jitter,
                             jitter_seed=7000 + s)
        env.reset(seed=7000 + s)
        r = fly(env, firewall=firewall); dmgs.append(r["dmg_dealt"])
        kills += r["event"] == "kill"; decks += r["event"] == "hard_deck"
        print(f"  seed {s}: {str(r['event']):<11} dmg={r['dmg_dealt']:5.0f} rounds={r['rounds']:4d} "
              f"min_rng={r['min_rng']:5.0f}m min_alt={r['min_alt']:6.0f}m")
        env.close()
    dmgs = np.array(dmgs)
    print(f"  --> {kills}/{seeds} kills | {decks}/{seeds} DECKS | dmg mean {dmgs.mean():.1f} "
          f"median {np.median(dmgs):.1f} max {dmgs.max():.0f}")


for aspect in (60, 90):
    run(aspect, 700, firewall=True)
