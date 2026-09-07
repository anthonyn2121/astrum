"""ADVERSARIAL verification probe for the Tier-3 opponent's break / energy states.

Independently re-measures: break attitude (phi/theta), ground-course vs total
flight-path turn rate, nz/alpha, airspeed, altitude, specific energy, break-sign
flips, and the FSM state occupancy — for (a) a pinned dead-six chase and (b) a
neutral head-on merge. Learner is flown by the built-in PursuitExpert with the
trigger FORCED OFF so the bot survives and we observe its full behaviour.
"""
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401
import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight.env import DogfightGymEnv
from astrum_dogfight.geometry import relative
from astrum_dogfight.opponent import Tier3Opponent
from astrum_dogfight.scenario import tier3_scenario

G = 9.80665


def course(v):
    if np.hypot(v[0], v[1]) > 1.0:
        return np.arctan2(v[0], v[1])
    return 0.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run(mode, sep, aspect, dalt_ft, secs, seed, log_every=1.0, fire=False):
    kw = dict(
        separation_m=sep,
        aspect_deg=aspect,
        max_time_s=secs,
        rate_hz=120.0,
        seed=seed,
        me_dalt_ft=dalt_ft,
    )
    if mode == "headon":
        kw["opp_heading_deg"] = 180.0
    env = DogfightGymEnv(
        tier3_scenario(**kw), opponent=Tier3Opponent(dt=1.0 / 120.0), ground_alt=0.0
    )
    env.reset()
    bot = env.opponent
    dt = 1.0 / 120.0
    prev_sign = None
    flips = 0
    st_ticks = {}
    entries = {}
    def_ticks = 0
    prev_v = None
    prev_c = None
    bot_min_alt = 1e9
    bot_min_v = 1e9
    bot_max_v = 0.0
    es0 = None
    es_last = None
    alpha_max = -9e9
    phi_abs_def = []
    def_course_total = 0.0
    hittable_low_v = 0
    flip_hist = {}
    phi_sat = 0
    rounds_bot0 = int(env._states[env.opponent_id].ammo)
    hp_me0 = float(env._states[env.learner_id].health)
    nsteps = int(secs / dt)
    print(
        f"\n=== mode={mode} sep={sep} aspect={aspect} dalt={dalt_ft}ft secs={secs} seed={seed} ==="
    )
    print(
        "t     state            V    alt   phi  theta gamma  nz  alpha  gcRate totRate   Es   rng  sign lrnAta  lrnV lrnAlt  dEs"
    )
    next_log = 0.0
    for i in range(nsteps):
        me = env._states[env.learner_id]
        opp = env._states[env.opponent_id]
        # replicate the opponent's own sign computation for THIS tick
        sign = 1.0 if relative(opp, me).ata_az >= 0.0 else -1.0
        st = bot.state  # state BEFORE this tick's transition (post-previous act)
        a = np.array(env._expert_action(), dtype=np.float32).copy()
        if not fire:
            a[-1] = 0.0
        obs, r, term, trunc, info = env.step(a)
        opp2 = env._states[env.opponent_id]
        me2 = env._states[env.learner_id]
        st2 = bot.state  # state used for THIS tick's control law
        st_ticks[st2] = st_ticks.get(st2, 0) + 1
        v = opp2.vel
        c = course(v)
        gc_rate = 0.0 if prev_c is None else np.degrees(wrap(c - prev_c)) / dt
        if prev_v is not None:
            u1, u2 = prev_v / max(np.linalg.norm(prev_v), 1e-9), v / max(np.linalg.norm(v), 1e-9)
            tot_rate = np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))) / dt
        else:
            tot_rate = 0.0
        gamma = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
        if st2 == "defensive" and prev_c is not None:
            def_course_total += abs(np.degrees(wrap(c - prev_c)))
        if st2 != st:
            entries[st2] = entries.get(st2, 0) + 1
        prev_v, prev_c = v.copy(), c
        es = opp2.alt + opp2.airspeed**2 / (2 * G)
        if es0 is None:
            es0 = es
        es_last = es
        bot_min_alt = min(bot_min_alt, opp2.alt)
        bot_min_v = min(bot_min_v, opp2.airspeed)
        bot_max_v = max(bot_max_v, opp2.airspeed)
        alpha_max = max(alpha_max, np.degrees(opp2.alpha))
        if st2 == "defensive":
            def_ticks += 1
            phi_abs_def.append(abs(np.degrees(opp2.phi)))
            if abs(np.degrees(opp2.phi)) >= 0.9 * 85.0:
                phi_sat += 1
            if prev_sign is not None and sign != prev_sign:
                flips += 1
                w = int((i + 1) * dt // 10) * 10
                flip_hist[w] = flip_hist.get(w, 0) + 1
            prev_sign = sign
        else:
            prev_sign = None
        rg = relative(me2, opp2)
        if opp2.airspeed < 120.0 and rg.rng < 900.0:
            hittable_low_v += 1
        t = (i + 1) * dt
        if t >= next_log - 1e-9:
            next_log += log_every
            print(
                f"{t:5.1f} {st2:16s} {opp2.airspeed:5.0f} {opp2.alt:6.0f} "
                f"{np.degrees(opp2.phi):5.0f} {np.degrees(opp2.theta):5.0f} {gamma:5.0f} "
                f"{opp2.nz:5.1f} {np.degrees(opp2.alpha):5.1f} "
                f"{gc_rate:7.1f} {tot_rate:7.1f} {es:6.0f} {rg.rng:5.0f} {sign:+.0f} "
                f"{np.degrees(relative(me2, opp2).ata):5.0f}"
                f" {me2.airspeed:5.0f} {me2.alt:6.0f} "
                f"{(me2.alt + me2.airspeed**2/(2*G)) - es:+6.0f}"
            )
        if term or trunc:
            print(f"  [ended t={t:.1f} event={info.get('event')}]")
            break
    tot = sum(st_ticks.values())
    shares = ", ".join(f"{k}={100*v/tot:.0f}%" for k, v in sorted(st_ticks.items()))
    print(f"  state share: {shares}")
    print(f"  state ENTRIES: {dict(sorted(entries.items()))}")
    print(
        f"  breakSignFlips={flips} defSecs={def_ticks*dt:.1f} "
        f"flipHz={flips/max(def_ticks*dt,1e-9):.2f} "
        f"meanAbsPhiDef={np.mean(phi_abs_def) if phi_abs_def else float('nan'):.1f} "
        f"defCourseSwept={def_course_total:.0f}deg "
        f"fracPhiGE76={100*phi_sat/max(def_ticks,1):.0f}% flipHist={dict(sorted(flip_hist.items()))}"
    )
    print(
        f"  botV[{bot_min_v:.0f},{bot_max_v:.0f}] botMinAlt={bot_min_alt:.0f} "
        f"Es {es0:.0f}->{es_last:.0f} ({es_last-es0:+.0f}) alphaMax={alpha_max:.1f} "
        f"botRoundsFired={rounds_bot0-int(env._states[env.opponent_id].ammo)} "
        f"lrnHpLost={hp_me0-float(env._states[env.learner_id].health):.0f} "
        f"lowVclose_s={hittable_low_v*dt:.1f}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0]
    sep = float(args[1])
    aspect = float(args[2])
    dalt = float(args[3])
    secs = float(args[4])
    seed = int(args[5]) if len(args) > 5 else 11
    log_every = float(args[6]) if len(args) > 6 else 1.0
    run(mode, sep, aspect, dalt, secs, seed, log_every)
