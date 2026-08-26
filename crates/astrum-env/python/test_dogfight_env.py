#!/usr/bin/env python3
"""Tests for the `astrum_dogfight` Gym adapter (design doc §8.2–§8.5).

Three parts, none needing gymnasium/SB3:
  1. Geometry — fabricated `AState`s exercise ATA / aspect / closure / Φ.
  2. Reward — fabricated transitions exercise the reward table + boundaries.
  3. Live rollout — a real two-f16 Tier-1 episode: obs shape/finiteness, reward
     finiteness, and that the scripted opponent actually flies straight & level.

Run via:  bash crates/astrum-env/python/run_gym_test.sh
"""

from math import degrees

import numpy as np
from astrum_dogfight import constants as C
from astrum_dogfight import reward as R
from astrum_dogfight.env import make_tier1_env, make_tier2_env, make_tier3_env
from astrum_dogfight.geometry import AState, build_observation, potential, relative
from astrum_dogfight.opponent import Tier3Opponent, wez_fire


def _course_deg(s) -> float:
    """Ground course (deg, 0=N, +E) from an AState's ENU velocity."""
    return degrees(np.arctan2(s.vel[0], s.vel[1]))


def _sweep_deg(course_deg, ref_deg) -> float:
    """Shortest-arc |Δheading| in degrees."""
    return abs((course_deg - ref_deg + 180.0) % 360.0 - 180.0)


def _state(id, *, pos, psi=0.0, vel=None, alive=True, hp=100.0, ammo=511, alt=None):
    pos = np.asarray(pos, float)
    if vel is None:
        vel = np.array([np.sin(psi), np.cos(psi), 0.0]) * 180.0  # along heading, ENU
    return AState(
        id=id,
        team=0,
        alive=alive,
        pos=pos,
        vel=np.asarray(vel, float),
        phi=0.0,
        theta=0.0,
        psi=psi,
        p=0.0,
        q=0.0,
        r=0.0,
        alpha=0.0,
        beta=0.0,
        nz=1.0,
        airspeed=180.0,
        alt=(pos[2] if alt is None else alt),
        health=hp,
        ammo=ammo,
    )


def test_geometry():
    # Me at origin facing north; bandit 400 m north, also facing north (flying away).
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    bandit = _state(2, pos=[0, 400, 4500], psi=0.0)
    rg = relative(me, bandit)
    assert abs(rg.rng - 400.0) < 1e-6
    assert degrees(rg.ata) < 1.0, f"nose-on bandit should give ~0 ATA, got {degrees(rg.ata):.1f}°"
    assert degrees(rg.aspect) < 1.0, f"at his six → ~0 aspect, got {degrees(rg.aspect):.1f}°"
    assert rg.in_range and rg.range_factor > 0.0, "400 m < R_eff should be in range"
    assert potential(me, bandit) > 0.5, "nose-on + rear-hemisphere + in-WEZ ⇒ high Φ"

    # Bandit directly behind me → ATA ≈ 180°.
    behind = _state(2, pos=[0, -800, 4500], psi=0.0)
    assert degrees(relative(me, behind).ata) > 179.0

    # Closure: me faster than a bandit ahead → closing (+).
    fast = _state(1, pos=[0, 0, 4500], psi=0.0, vel=[0, 250, 0])
    slow_ahead = _state(2, pos=[0, 1000, 4500], psi=0.0, vel=[0, 100, 0])
    assert relative(fast, slow_ahead).closure > 0.0

    obs = build_observation(me, bandit)
    assert obs.shape == (C.OBS_DIM,) and np.all(np.isfinite(obs))
    print(f"  geometry OK — obs dim {C.OBS_DIM}, Φ(nose-on)={potential(me, bandit):.3f}")


def test_reward():
    me = _state(1, pos=[0, 0, 4500])
    bandit = _state(2, pos=[0, 1000, 4500])

    # Damage dealt: 2 HP off the bandit ⇒ +1.0 (minus tiny time/shaping).
    hit = _state(2, pos=[0, 1000, 4500], hp=98.0)
    out = R.compute(me, bandit, me, hit)
    assert 0.9 < out.reward < 1.1 and not out.terminated, out

    # Kill: bandit → dead ⇒ +100, terminated.
    dead_bandit = _state(2, pos=[0, 1000, 4500], alive=False, hp=0.0)
    out = R.compute(me, bandit, me, dead_bandit)
    assert out.terminated and out.reward > 90.0 and out.info["event"] == "kill", out

    # Death: learner → dead ⇒ −100, terminated.
    dead_me = _state(1, pos=[0, 0, 4500], alive=False, hp=0.0)
    out = R.compute(me, bandit, dead_me, bandit)
    assert out.terminated and out.reward < -90.0 and out.info["event"] == "killed", out

    # Hard deck: below 300 m AGL ⇒ −200, terminated, wins precedence over everything.
    low = _state(1, pos=[0, 0, 100], alt=100.0)
    out = R.compute(me, bandit, low, bandit)
    assert out.terminated and out.reward < -190.0 and out.info["event"] == "hard_deck", out

    # Bubble exit: horizontal range > R_bubble ⇒ −50, truncated.
    far = _state(1, pos=[C.R_BUBBLE + 100.0, 0, 4500])
    out = R.compute(me, bandit, far, bandit)
    assert out.truncated and not out.terminated and out.reward < -49.0, out
    print("  reward OK — damage/kill/death/deck/bubble table verified")


def test_live_rollout():
    env = make_tier1_env(max_time_s=30.0)
    obs, info = env.reset()
    assert obs.shape == (C.OBS_DIM,) and np.all(np.isfinite(obs)), "reset obs must be finite"

    opp0 = env._states[env.opponent_id]
    hdg0, alt0 = opp0.psi, opp0.alt

    # Learner holds a mild pursuit; opponent (Tier-1) should hold straight & level.
    action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)
    total = 0.0
    steps = 240  # ~2 s at 120 Hz
    for _ in range(steps):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), "obs stayed finite"
        assert np.isfinite(rew), f"reward finite, got {rew}"
        total += rew
        if terminated or truncated:
            break

    opp = env._states[env.opponent_id]
    dh = degrees(abs(((opp.psi - hdg0) + np.pi) % (2 * np.pi) - np.pi))
    da = abs(opp.alt - alt0)
    assert dh < 15.0, f"Tier-1 opponent should hold heading, drifted {dh:.1f}°"
    assert da < 500.0, f"Tier-1 opponent should hold altitude, drifted {da:.0f} m"
    assert env._states[env.learner_id].alive, "learner survives a 2 s neutral rollout"
    print(
        f"  live rollout OK — {steps} steps, Σreward={total:.3f}, "
        f"opp drift Δhdg={dh:.2f}° Δalt={da:.0f} m"
    )


def test_tier2_break():
    # Opponent starts on the learner's tail-side, so it detects a rear-hemisphere
    # attacker at once and breaks. A hard break must swing its heading hard.
    env = make_tier2_env(maneuver="break", max_time_s=40.0)
    obs, _ = env.reset()
    hdg0 = _course_deg(env._states[env.opponent_id])
    action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)  # learner: level, holding
    triggered = False
    max_sweep = 0.0
    for _ in range(1600):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        triggered = triggered or env.opponent._active is not None
        opp = env._states[env.opponent_id]
        max_sweep = max(max_sweep, _sweep_deg(_course_deg(opp), hdg0))
        if terminated or truncated:
            break

    opp = env._states[env.opponent_id]
    assert triggered, "Tier-2 opponent should detect the tail-side attacker and commit a break"
    assert max_sweep > 60.0, f"a hard break should swing heading, got only {max_sweep:.0f}°"
    assert opp.alive and abs(opp.nz) < 12.0, "FLCS keeps the break inside the envelope"
    print(f"  tier2 break OK — maneuver triggered, heading swept {max_sweep:.0f}°")


def test_tier2_split_s():
    # Same tail-side trigger, but the Split-S reverses heading and dumps altitude
    # (mirrors crates/astrum-sim/tests/split_s.rs: ~180° reversal, ~2 km lost).
    env = make_tier2_env(maneuver="split_s", max_time_s=40.0)
    obs, _ = env.reset()
    opp0 = env._states[env.opponent_id]
    hdg0, alt0 = _course_deg(opp0), opp0.alt
    action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)
    max_reversal = 0.0
    min_alt = alt0
    for _ in range(1800):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        opp = env._states[env.opponent_id]
        max_reversal = max(max_reversal, _sweep_deg(_course_deg(opp), hdg0))
        min_alt = min(min_alt, opp.alt)
        if terminated or truncated:
            break

    assert max_reversal > 120.0, f"a Split-S reverses heading, got {max_reversal:.0f}°"
    assert alt0 - min_alt > 300.0, f"a Split-S trades altitude, lost only {alt0 - min_alt:.0f} m"
    print(
        f"  tier2 split-s OK — heading reversed {max_reversal:.0f}°, alt lost {alt0 - min_alt:.0f} m"
    )


def test_tier3_fire_gate():
    # The WEZ trigger honors effective range, the lead cone, and ammo.
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    on_six = _state(2, pos=[0, 500, 4500], psi=0.0)  # 500 m ahead, on the nose
    assert wez_fire(me, on_six), "in range + on the nose ⇒ fire"
    far = _state(2, pos=[0, 2000, 4500], psi=0.0)  # 2 km > R_eff
    assert not wez_fire(me, far), "out of gun range ⇒ hold fire"
    off_nose = _state(2, pos=[500, 0, 4500], psi=0.0)  # in range but 90° off boresight
    assert not wez_fire(me, off_nose), "no lead solution ⇒ hold fire"
    dry = _state(1, pos=[0, 0, 4500], psi=0.0, ammo=0)
    assert not wez_fire(dry, on_six), "no ammo ⇒ never fire"
    print("  tier3 fire-gate OK — WEZ trigger honors range, lead cone, ammo")


def test_tier3_control_laws():
    # Every state's control law emits a finite, in-bounds raw action.
    opp = Tier3Opponent()
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    tgt = _state(2, pos=[0, 600, 4500], psi=0.0)

    def bounded(a4):
        assert len(a4) == 4 and all(np.isfinite(x) for x in a4)
        assert all(-1.0 <= a4[i] <= 1.0 for i in range(3)) and 0.0 <= a4[3] <= 1.0

    bounded(opp._pursue(me, tgt))
    bounded(opp._break(me, tgt))
    bounded(opp._high_yoyo(me, tgt))
    bounded(opp._energy_recover(me))
    full = opp.act(me, tgt)
    assert len(full) == 5 and full[4] in (0.0, 1.0)
    print("  tier3 control-laws OK — pursue/break/yoyo/energy emit finite, bounded actions")


def test_tier3_offensive():
    # Bot on the learner's six, learner flies straight: bot goes offensive and guns
    # it — a *real* end-to-end shot (fire_guns → bullets → resolve_hits → damage).
    env = make_tier3_env(posture="offensive", separation_m=600.0, max_time_s=25.0)
    obs, _ = env.reset()
    action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)  # learner: level, not firing
    saw_offensive = False
    min_hp = 100.0
    killed = False
    for _ in range(3000):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        saw_offensive = saw_offensive or env.opponent.state == "offensive"
        min_hp = min(min_hp, env._states[env.learner_id].health)
        if info.get("event") == "killed":
            killed = True
        if terminated or truncated:
            break

    assert saw_offensive, "Tier-3 bot on the six should enter OFFENSIVE"
    assert min_hp < 100.0, f"the bot's guns should draw blood, learner HP stayed {min_hp:.0f}"
    print(f"  tier3 offensive OK — entered OFFENSIVE, learner HP → {min_hp:.0f}, killed={killed}")


def test_tier3_defensive():
    # Learner starts on the bot's six pointing at it: the FSM should read the threat
    # and break (heading swings hard) while staying inside the envelope.
    env = make_tier3_env(posture="defensive", max_time_s=25.0)
    obs, _ = env.reset()
    hdg0 = _course_deg(env._states[env.opponent_id])
    action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)  # learner: level, not firing
    saw_defensive = False
    max_sweep = 0.0
    for _ in range(2000):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        saw_defensive = saw_defensive or env.opponent.state == "defensive"
        opp = env._states[env.opponent_id]
        max_sweep = max(max_sweep, _sweep_deg(_course_deg(opp), hdg0))
        if terminated or truncated:
            break

    opp = env._states[env.opponent_id]
    assert saw_defensive, "Tier-3 bot facing a nose-on threat should enter DEFENSIVE"
    assert max_sweep > 45.0, f"a defensive break should swing heading, got {max_sweep:.0f}°"
    assert opp.alive and abs(opp.nz) < 12.0, "FLCS keeps the break inside the envelope"
    print(f"  tier3 defensive OK — entered DEFENSIVE, heading swept {max_sweep:.0f}°")


def main():
    test_geometry()
    test_reward()
    test_live_rollout()
    test_tier2_break()
    test_tier2_split_s()
    test_tier3_fire_gate()
    test_tier3_control_laws()
    test_tier3_offensive()
    test_tier3_defensive()
    print(
        "\nOK — astrum_dogfight Tier-1/2/3 adapter: geometry, reward, rollout, "
        "break, split-s, fire-gate, FSM offense+defense all pass."
    )


if __name__ == "__main__":
    main()
