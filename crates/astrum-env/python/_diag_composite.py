"""Composite merge-entry teacher: compose the two half-skills the bracket probes revealed.
  - FIREWALL throttle (=1.0) while aspect is HIGH (not yet behind him) OR range is large:
    an aggressive corner-cut/overtake to erase the lateral offset and close (firewall probe
    proved this closes 200m->30m from a forward offset).
  - Hand off to the range-keeper's OWN throttle once aspect<50deg and rng<500m: settle into
    the guns-trail and stop overshooting (range-keeper proved it tracks+guns once behind).
Aim is always the expert's lead_pursuit; only throttle is overridden. If this converts a merge
into damage/kills (vs dummy AND T3), it is a viable expert SEED for the c21 merge rungs. If it
also gives ~0, scripted conversion is impossible across the board -> self-play is the only path."""
import math, sys
sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import relative

ASPECT_HANDOFF = math.radians(50.0)  # I'm "getting behind him" below this
RNG_HANDOFF = 500.0


def fly(env, q=0.05, max_steps=7300):
    env.reset()
    me0 = env._states[env.learner_id]; opp0 = env._states[env.opponent_id]
    ammo0 = int(me0.ammo); hp0 = float(opp0.health)
    event = None; min_rng = 1e9; min_alt = 1e9; fw_frac = 0; n = 0
    info = {"expert_action": np.zeros(C.ACTION_DIM, np.float32)}
    for _ in range(max_steps):
        a = np.array(info["expert_action"], dtype=np.float32).copy()  # expert lead-pursuit stick
        me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
        rg = relative(me, opp)
        firewall = (rg.aspect > ASPECT_HANDOFF) or (rg.rng > RNG_HANDOFF)
        if firewall:
            a[3] = 1.0
            fw_frac += 1
        n += 1
        a[-1] = 1.0 if rg.gun_quality >= q else 0.0
        obs, rew, term, trunc, info = env.step(a)
        me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
        min_rng = min(min_rng, float(np.linalg.norm(opp.pos - me.pos)))
        min_alt = min(min_alt, float(me.alt))
        if info.get("event"): event = info["event"]
        if term or trunc: break
    me = env._states[env.learner_id]; opp = env._states[env.opponent_id]
    return {"event": event, "dmg_dealt": hp0 - float(opp.health),
            "dmg_taken": C.HP_MAX - float(me.health), "rounds": ammo0 - int(me.ammo),
            "min_rng": min_rng, "min_alt": min_alt, "fw_frac": fw_frac / max(1, n)}


def run(opp_tier, aspect, sep, headon=False, seeds=6, max_time_s=60.0, jitter=0.5):
    lbl = {1: "T1-dummy", 2: "T2-break", 3: "T3-full"}[opp_tier]
    tag = "HEAD-ON" if headon else "co-hdg"
    print(f"\n=== COMPOSITE vs {lbl} @ aspect {aspect:.0f} / {sep:.0f}m {tag} ===")
    kills = decks = deaths = 0; dd = []
    for s in range(seeds):
        kw = dict(posture="offensive", opponent_tier=opp_tier, aspect_deg=float(aspect),
                  separation_m=float(sep), max_time_s=max_time_s, jitter=jitter, jitter_seed=9000 + s)
        if headon:
            kw["opp_heading_deg"] = 180.0
        env = make_tier3_env(**kw)
        env.reset(seed=9000 + s)
        r = fly(env); dd.append(r["dmg_dealt"])
        kills += r["event"] == "kill"; decks += r["event"] == "hard_deck"
        deaths += r["event"] in ("killed", "death")
        print(f"  seed {s}: {str(r['event']):<11} dmg={r['dmg_dealt']:5.0f} taken={r['dmg_taken']:4.0f} "
              f"rounds={r['rounds']:4d} min_rng={r['min_rng']:5.0f}m fw={r['fw_frac']*100:.0f}%")
        env.close()
    dd = np.array(dd)
    print(f"  --> {kills}/{seeds} kills | {deaths}/{seeds} deaths | {decks}/{seeds} DECKS | "
          f"dmg mean {dd.mean():.1f} med {np.median(dd):.1f} max {dd.max():.0f}")


print(f"combat: HP_MAX={C.HP_MAX} DECK_AGL={C.DECK_AGL}m hit_radius=8.0m dmg/hit=2.0")
run(1, 60, 700); run(1, 90, 700)          # forward-offset dummy (the 0-dmg wall)
run(3, 45, 900); run(3, 90, 900)          # co-heading merge vs full T3
run(3, 0, 1500, headon=True)              # head-on turning fight vs T3
