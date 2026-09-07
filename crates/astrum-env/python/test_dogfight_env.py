#!/usr/bin/env python3
"""Tests for the `astrum_dogfight` Gym adapter (design doc §8.2–§8.5).

Three parts, none needing gymnasium/SB3:
  1. Geometry — fabricated `AState`s exercise ATA / aspect / closure / Φ.
  2. Reward — fabricated transitions exercise the reward table + boundaries.
  3. Live rollout — a real two-f16 Tier-1 episode: obs shape/finiteness, reward
     finiteness, and that the scripted opponent actually flies straight & level.

Run via:  bash crates/astrum-env/python/run_gym_test.sh
"""

from dataclasses import replace
from math import cos, degrees, radians, sin

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


def _state(id, *, pos, psi=0.0, vel=None, alive=True, hp=100.0, ammo=511, alt=None, airspeed=180.0):
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
        airspeed=airspeed,
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


def test_obs_append_arrival_geometry():
    """curriculum-87: the 10 appended obs features, and that they are SIGNED and in the right slots.

    The append exists because c86 kills 8/8 arriving at ~18°/s LOS rate and 0/8 arriving at ~59°/s
    and could not tell those two arrivals apart ([[obs-blind-to-opponent-turn]]). Signedness is the
    whole point of half these columns — an unsigned rate or lead says "something is wrong" without
    saying which way to correct — so each is checked against its mirror case, not just for
    magnitude. The slot assertions pin the layout: the features are appended BEHIND the three WEZ
    entries specifically so `gun_quality` keeps absolute index 29, and a silent reorder here would
    corrupt both that index and every warm start through `_widen_resume_obs`.
    """
    from math import hypot

    from astrum_dogfight.geometry import flight_path_angle, specific_energy

    # (a) Co-speed line-astern chase: nothing rotates, nothing is off-boresight.
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    bandit = _state(2, pos=[0, 400, 4500], psi=0.0)
    rg = relative(me, bandit)
    assert abs(rg.los_rate_az) < 1e-9 and abs(rg.los_rate_el) < 1e-9, "co-moving LOS cannot rotate"
    assert abs(rg.lead_az) < 1e-6 and abs(rg.lead_el) < 1e-6
    assert degrees(rg.alpha_req) < 0.5, "a co-speed trail shot needs no off-velocity angle"
    assert specific_energy(me) == specific_energy(bandit)

    # (b) Pure horizontal crossing, bandit tracking left→right. v_perp is exactly his velocity
    #     component across the LOS, so the rate is analytic: 180/400 rad/s.
    east = _state(2, pos=[0, 400, 4500], psi=radians(90.0))
    rg_e = relative(me, east)
    assert abs(rg_e.los_rate_az - 180.0 / 400.0) < 1e-6, "LOS rate must equal |v_perp|/r"
    assert abs(rg_e.los_rate_el) < 1e-9, "a level crossing has no vertical LOS rate"
    assert rg_e.lead_az > 0.0, "must lead RIGHT at a left-to-right crosser"
    assert degrees(rg_e.alpha_req) > 5.0, "a crossing shot demands real off-velocity angle"

    # (c) The mirror. Every signed quantity flips; the unsigned total does not.
    west = _state(2, pos=[0, 400, 4500], psi=radians(-90.0))
    rg_w = relative(me, west)
    assert abs(rg_w.los_rate_az + rg_e.los_rate_az) < 1e-6, "LOS rate must be signed"
    assert abs(rg_w.lead_az + rg_e.lead_az) < 1e-9, "lead azimuth must be signed"
    assert abs(rg_w.lead_error - rg_e.lead_error) < 1e-9, "the unsigned total is mirror-invariant"

    # (d) Vertical crossing → the rate appears in elevation, not azimuth.
    up = _state(2, pos=[0, 400, 4500], vel=[0.0, 0.0, 180.0])
    rg_u = relative(me, up)
    assert abs(rg_u.los_rate_el - 180.0 / 400.0) < 1e-6
    assert abs(rg_u.los_rate_az) < 1e-9
    assert rg_u.lead_el > 0.0, "must lead UP at a climbing crosser"

    # (e) Slot layout + normalization, against a bandit that is banked, turning and lower on energy.
    hot = replace(east, phi=radians(60.0), q=0.4, r=0.3, alt=3500.0, airspeed=150.0)
    hot = replace(hot, pos=np.array([0.0, 400.0, 3500.0]))
    rg_h = relative(me, hot)
    obs = build_observation(me, hot)
    assert obs.shape == (C.OBS_DIM,) and np.all(np.isfinite(obs))
    assert abs(obs[29] - rg_h.gun_quality) < 1e-6, "gun_quality must stay at absolute index 29"
    tail = [
        rg_h.los_rate_az / C.LOS_RATE_REF,
        rg_h.los_rate_el / C.LOS_RATE_REF,
        sin(rg_h.lead_az),
        sin(rg_h.lead_el),
        min(rg_h.alpha_req / C.ALPHA_REF, 2.0),
        min(rg_h.lay_miss_m / C.MISS_REF, 2.0),
        sin(hot.phi),
        cos(hot.phi),
        hypot(hot.q, hot.r) / C.RATE_REF,
        (specific_energy(me) - specific_energy(hot)) / C.ES_REF,
    ]
    if C.C97_VERTICAL_OBS:
        # curriculum-97: the signed vertical axis (gated so the paired 41-obs control also passes).
        tail += [
            (hot.vel[2] - me.vel[2]) / C.V_REF,
            sin(flight_path_angle(me)),
            sin(flight_path_angle(hot)),
            hot.q / C.RATE_REF,
        ]
    assert len(tail) == C.OBS_DIM - 31, f"the c87 append is 10 wide, +4 when c97 is on ({C.OBS_DIM})"
    assert np.allclose(obs[31:], np.asarray(tail, np.float32), atol=1e-6), "obs tail slot mismatch"
    assert obs[40] > 0.0, "I am higher and faster ⇒ positive relative energy"

    # curriculum-97: the vertical features must be SIGNED, which is the whole point of adding them.
    # The `hypot` slot (39) is what they exist to supplement: it destroys the sign of the pitch
    # rate, so a bandit pulling UP and one pushing DOWN are currently indistinguishable there.
    pull_up = replace(hot, q=+0.3, r=0.0)
    push_dn = replace(hot, q=-0.3, r=0.0)
    o_up, o_dn = build_observation(me, pull_up), build_observation(me, push_dn)
    assert abs(o_up[39] - o_dn[39]) < 1e-9, "idx 39 is |q,r| — unsigned by construction"
    # γ is the FLIGHT PATH, not the attitude: a jet with its nose 20° up but sinking must read
    # negative. This is the distinction F4 turned on — a fixed 85° bank predicted a descent and the
    # T3 climbed, because θ and φ do not determine γ. (`flight_path_angle` itself is checked either
    # way; only the obs slots 41-44 are gated, since the 41-obs null control does not carry them.)
    sinking = _state(3, pos=[0, 400, 4000], vel=[0.0, 100.0, -40.0])
    sinking = replace(sinking, theta=radians(20.0))
    assert flight_path_angle(sinking) < 0.0, "γ follows the velocity vector, not θ"
    climbing = replace(sinking, vel=np.array([0.0, 100.0, 40.0]))
    if C.C97_VERTICAL_OBS:
        assert o_up[44] > 0.0 > o_dn[44], "idx 44 must carry the SIGN idx 39 throws away"
        assert build_observation(me, sinking)[43] < 0.0, "a sinking bandit must read γ < 0"
        assert build_observation(me, climbing)[43] > 0.0, "a climbing bandit must read γ > 0"
        assert build_observation(me, climbing)[41] > build_observation(me, sinking)[41], (
            "relative vertical rate must order a climber above a sinker"
        )
    # This geometry (1000 m below me, crossing) needs a 78° off-velocity angle — no shot exists.
    # Both achievability features must SATURATE rather than run away: unclipped they reach 6.0 and
    # ~30, which would swamp the encoder's input scale to express only "impossible".
    assert degrees(rg_h.alpha_req) > 60.0 and obs[35] == 2.0, "alpha_req must clip, not run away"
    assert obs[36] == 2.0, "a hopeless predicted miss must clip too"
    assert np.all(np.abs(obs[31:]) <= 2.0), "every appended feature must stay O(1) for the encoder"


def test_total_los_rate_is_not_the_hypot_of_its_body_components():
    """`RelGeom.los_rate` is `|v_perp|/rng`, and `hypot(los_rate_az, los_rate_el)` is NOT that.

    Four diagnostics in this program reported the hypot as "the LOS rate", so every historical
    LOS-rate figure — including the 18.6°/s-vs-59°/s arrival measurement that set `LOS_RATE_REF`
    itself — is a LOWER BOUND. The mechanism: `los_rate_az/el` project `v_perp` onto body starboard
    and body up, which span the plane perpendicular to the NOSE, not to the LOS. Any component of
    `v_perp` along the nose axis is dropped, so the two agree only when the bandit is dead ahead and
    diverge as ata grows — worst exactly at the high-ata merge geometries under study.
    """
    from math import hypot

    # Dead ahead: the LOS *is* the nose axis, so nothing is dropped and the two must agree.
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    ahead = _state(2, pos=[0, 400, 4500], psi=radians(90.0))
    rg = relative(me, ahead)
    assert abs(rg.los_rate - 180.0 / 400.0) < 1e-6, "|v_perp|/rng, analytically"
    assert abs(rg.los_rate - hypot(rg.los_rate_az, rg.los_rate_el)) < 1e-9

    # Bandit 60° off the nose AND crossing (a co-heading bandit has no relative velocity at all, so
    # every rate would be trivially 0). Now v_perp has a nose-axis component and the hypot loses it.
    off = _state(
        2,
        pos=[400.0 * sin(radians(60.0)), 400.0 * cos(radians(60.0)), 4500],
        psi=radians(90.0),
    )
    rg_off = relative(me, off)
    assert abs(degrees(rg_off.ata) - 60.0) < 1e-6, "the premise: the bandit is 60° off the nose"
    # For a coplanar crossing the understatement factor is EXACTLY cos(ata), which names the
    # mechanism rather than merely bounding it: v_perp ⟂ LOS forces its nose and starboard
    # components into the ratio -tan(ata), so |v_perp| = |v_perp·starboard| / cos(ata). At the 60°
    # here the hypot reports HALF the true rate; at the merge geometries this program studies, less.
    assert abs(
        hypot(rg_off.los_rate_az, rg_off.los_rate_el) - rg_off.los_rate * cos(rg_off.ata)
    ) < 1e-9, (
        f"hypot {degrees(hypot(rg_off.los_rate_az, rg_off.los_rate_el)):.1f} vs true "
        f"{degrees(rg_off.los_rate):.1f} deg/s"
    )
    # ...and the true rate is still exactly |v_perp|/rng, computed here from the raw states.
    r = np.asarray(off.pos, float) - np.asarray(me.pos, float)
    v = np.asarray(off.vel, float) - np.asarray(me.vel, float)
    r_hat = r / np.linalg.norm(r)
    v_perp = v - float(np.dot(v, r_hat)) * r_hat
    assert abs(rg_off.los_rate - np.linalg.norm(v_perp) / np.linalg.norm(r)) < 1e-9


def test_corrected_lay_and_scripted_invariance():
    """curriculum-86: `RelGeom.lead_error` is the VELOCITY-INHERITANCE-CORRECTED gun lay, and
    `naive_lead_error` still reproduces the OLD (pre-curriculum-86) nose→aim-point angle bit-for-bit.

    The second half is the benchmark guard. `fire_guns` launches rounds at `ac_vel + 1030·boresight`,
    so the direction the nose must point is the solution of `1030·d = s·u − v_own`, not `u`
    ([[gun-lay-velocity-inheritance]] — measured worth on c84's own rounds: +938.8 hits of 1369).
    Correcting it is a LEARNER-side fix; the scripted T3/expert are the fixed yardstick the whole
    curriculum-01→85 history is measured against, so their fire gates stay pinned to the naive angle
    and must keep reading exactly what they read before. This test fails the moment that drifts.
    """
    from astrum_dogfight.geometry import body_axes_enu, lead_point
    from astrum_dogfight.opponent import PursuitExpert

    rng_ = np.random.default_rng(86)
    max_gap = 0.0
    for _ in range(300):
        pa = rng_.normal(0.0, 600.0, 3) + np.array([0.0, 0.0, 4500.0])
        pb = rng_.normal(0.0, 600.0, 3) + np.array([0.0, 0.0, 4500.0])
        if np.linalg.norm(pb - pa) < 30.0:
            continue
        va = rng_.normal(0.0, 150.0, 3)
        vb = rng_.normal(0.0, 150.0, 3)
        a = _state(1, pos=pa, vel=va)
        b = _state(2, pos=pb, vel=vb)
        a.phi, a.theta, a.psi = rng_.uniform(-np.pi, np.pi, 3)
        rg = relative(a, b)

        # (1) `naive_lead_error` == the exact pre-curriculum-86 formula (rng/1030 tof, from the nose).
        nose, _, _ = body_axes_enu(a.phi, a.theta, a.psi)
        old_dir = lead_point(a, b) - a.pos
        old_dir = old_dir / np.linalg.norm(old_dir)
        old = np.arccos(np.clip(float(np.dot(old_dir, nose)), -1.0, 1.0))
        assert abs(rg.naive_lead_error - old) < 1e-12, (rg.naive_lead_error, old)

        # (2) The scripted bots gate on THAT angle, so their triggers are byte-identical to before.
        assert wez_fire(a, b) == bool(rg.in_range and old < C.FIRE_LEAD_RAD)
        assert PursuitExpert._fire(a, b, rg) == bool(rg.in_range and rg.rng * sin(old) <= C.GUN_FIRE_MISS_M)

        # (3) The corrected lay actually points where a round GOES: `v_own + 1030·d` must lie along
        #     the LOS to the intercept point. Check by re-deriving the miss from the launch vector.
        if rg.lay_feasible:
            aim = b.pos + b.vel * rg.lay_tof
            want = (aim - a.pos) / np.linalg.norm(aim - a.pos)
            d = (want * (np.linalg.norm(aim - a.pos) / rg.lay_tof) - a.vel) / C.MUZZLE_MPS
            assert abs(np.linalg.norm(d) - 1.0) < 1e-9, np.linalg.norm(d)
            corrected = np.arccos(np.clip(float(np.dot(d, nose)), -1.0, 1.0))
            assert abs(rg.lead_error - corrected) < 1e-9, (rg.lead_error, corrected)
            assert abs(rg.lay_miss_m - C.MUZZLE_MPS * sin(rg.lead_error) * rg.lay_tof) < 1e-9
            max_gap = max(max_gap, abs(degrees(rg.lead_error - rg.naive_lead_error)))

    # (4) The two lays are NOT the same thing: over random geometry they diverge by many degrees.
    assert max_gap > 5.0, f"corrected vs naive lay differ by only {max_gap:.2f}° — is the fix wired in?"
    print(f"  corrected lay OK — naive gate bit-identical (scripted T3 pinned); "
          f"corrected−naive lay gap up to {max_gap:.1f}°")


def test_reward():
    """reward-v2 table: damage, terminal magnitudes/precedence, dense tracking, fire
    discipline. Values are hand-verified against constants.py in the comments."""
    me = _state(1, pos=[0, 0, 4500])
    bandit = _state(2, pos=[0, 1000, 4500])

    # Damage dealt: 2 HP off the bandit ⇒ +1.0, plus a small nose-on/in-WEZ tracking
    # bonus (~+0.012 at 1 km on the nose) minus the −0.001 time penalty ⇒ ~1.01.
    hit = _state(2, pos=[0, 1000, 4500], hp=98.0)
    out = R.compute(me, bandit, me, hit)
    assert 0.9 < out.reward < 1.1 and not out.terminated, out

    # Kill: bandit → dead ⇒ +R_KILL (150) plus the +50 from the fatal 100-HP burst,
    # ~+200 total, terminated, event "kill".
    dead_bandit = _state(2, pos=[0, 1000, 4500], alive=False, hp=0.0)
    out = R.compute(me, bandit, me, dead_bandit)
    assert out.terminated and out.reward > 90.0 and out.info["event"] == "kill", out

    # Death (softened): learner → dead ⇒ R_DEATH (−60) + the −50 from the 100-HP
    # damage taken ≈ −110 — a real penalty, but clearly above the −200 deck (so the
    # agent isn't taught that committing offense is as bad as flying into the ground).
    dead_me = _state(1, pos=[0, 0, 4500], alive=False, hp=0.0)
    out = R.compute(me, bandit, dead_me, bandit)
    assert out.terminated and -190.0 < out.reward < -90.0, out
    assert out.info["event"] == "killed", out

    # Trade (mutual kill same tick): its own R_TRADE (−25); the ±50 damage terms
    # cancel (dealt 100, took 100) ⇒ ~−25. Must be NET NEGATIVE so a kamikaze trade is
    # never preferred to a clean kill, and worse than a draw — but softer than a death.
    out = R.compute(me, bandit, dead_me, dead_bandit)
    assert out.terminated and -30.0 < out.reward < -20.0, out
    assert out.info["event"] == "trade", out

    # Hard deck: below 300 m AGL ⇒ −200, terminated, wins precedence over everything.
    low = _state(1, pos=[0, 0, 100], alt=100.0)
    out = R.compute(me, bandit, low, bandit)
    assert out.terminated and out.reward < -190.0 and out.info["event"] == "hard_deck", out

    # Bubble exit: horizontal range > R_bubble ⇒ −50, truncated.
    far = _state(1, pos=[C.R_BUBBLE + 100.0, 0, 4500])
    out = R.compute(me, bandit, far, bandit)
    assert out.truncated and not out.terminated and out.reward < -49.0, out

    # Dense tracking (non-telescoping): nose-on and inside TRACK_RANGE earns a positive
    # standing reward every tick (beats the −0.001 time floor); nose-on but beyond
    # TRACK_RANGE earns nothing but the time penalty (negative) — so pursuit only pays
    # once you actually close. No damage, no fire, no terminal in either case.
    near = _state(2, pos=[0, 600, 4500])  # 600 m, on the nose, inside R_eff
    out = R.compute(me, near, me, near)
    assert not out.terminated and not out.truncated and out.reward > 0.0, out
    assert out.info.get("gun_quality", 0.0) > 0.0, out
    beyond = _state(2, pos=[0, 4000, 4500])  # 4 km — past TRACK_RANGE and R_eff
    out = R.compute(me, beyond, me, beyond)
    assert out.reward < 0.0 and out.info.get("gun_quality", 0.0) == 0.0, out

    # Fire discipline (reward-v3): rounds fired OUT of a gun solution (bandit at 4 km >
    # R_eff) are mildly penalised (−0.005 each) and net slightly negative; the SAME burst
    # fired IN solution (600 m, on the nose) is now strongly rewarded (+0.05 each) and nets
    # clearly positive — v3 makes pulling the trigger in the WEZ genuinely pay (v2's +0.005
    # was too weak, so the policy learned to never fire even from inside gun range).
    me0 = _state(1, pos=[0, 0, 4500], ammo=511)
    me_fired = _state(1, pos=[0, 0, 4500], ammo=501)  # 10 rounds gone
    out_waste = R.compute(me0, beyond, me_fired, beyond)  # reward-v13: 10·(−0.02) − 0.001 time ≈ −0.201
    assert out_waste.info.get("rounds_fired") == 10 and -0.22 < out_waste.reward < -0.18, out_waste
    out_wez = R.compute(me0, near, me_fired, near)  # 10·(+0.05) + tracking(≈0.023) − 0.001 ≈ +0.52
    assert out_wez.info.get("rounds_fired") == 10 and out_wez.reward > 0.4, out_wez
    assert out_wez.reward > out_waste.reward, (out_wez, out_waste)
    print("  reward OK — damage/kill/death/trade/deck/bubble + tracking + fire discipline verified")


def test_fire_burst_ramp():
    """reward-v10: the out-of-solution WASTE cost ramps with the current burst length; an
    in-solution round RESETS it; a no-fire tick DECAYS it; and at waste_streak=0 the cost is
    exactly R_FIRE_WASTE (v9-identical), so a short exploratory burst stays cheap. The burst is
    the *only* dense term here: at 4 km the bandit is past TRACK_RANGE and R_EFF (tracking = 0,
    gun_quality = 0) and the geometry is static (no closing/aim shaping), so stripping the −0.001
    time floor leaves the fire reward alone."""
    me0 = _state(1, pos=[0, 0, 4500], ammo=511)
    me_fired = _state(1, pos=[0, 0, 4500], ammo=501)  # 10 rounds gone
    beyond = _state(2, pos=[0, 4000, 4500])  # 4 km — out of solution ⇒ waste
    near = _state(2, pos=[0, 600, 4500])     # 600 m on the nose — in solution
    base = 10 * C.R_FIRE_WASTE

    # (1) Fresh burst (streak 0) is v9-identical: 10 wasted rounds at the flat base cost.
    out0 = R.compute(me0, beyond, me_fired, beyond, waste_streak=0.0)
    assert abs((out0.reward - C.TIME_PENALTY) - base) < 1e-9, out0
    assert abs(out0.info["waste_streak"] - 10.0) < 1e-9, out0.info

    # (2) Deep in a sustained burst the SAME 10 rounds cost strictly more (ramped by streak).
    deep = 200.0
    ramp = min(1.0 + deep / C.WASTE_RAMP_TAU, C.WASTE_RAMP_MAX)
    out1 = R.compute(me0, beyond, me_fired, beyond, waste_streak=deep)
    fire1 = out1.reward - C.TIME_PENALTY
    assert fire1 < base - 1e-6, "a sustained burst must cost more than a fresh one"
    assert abs(fire1 - base * ramp) < 1e-6, (fire1, base * ramp)
    assert abs(out1.info["waste_streak"] - (deep + 10.0)) < 1e-9, out1.info

    # (3) The multiplier saturates at WASTE_RAMP_MAX (never unbounded).
    huge = R.compute(me0, beyond, me_fired, beyond, waste_streak=1e6)
    assert abs((huge.reward - C.TIME_PENALTY) - base * C.WASTE_RAMP_MAX) < 1e-6, huge

    # (4) An IN-solution round mid-burst is rewarded AND resets the streak to 0.
    inb = R.compute(me0, near, me_fired, near, waste_streak=deep)
    assert inb.info["waste_streak"] == 0.0, inb.info
    assert inb.info["rounds_fired"] == 10 and inb.reward > 0.4, inb

    # (5) A no-fire tick decays the streak by WASTE_STREAK_DECAY (a pause/release forgives).
    idle = R.compute(me0, beyond, me0, beyond, waste_streak=deep)  # ammo unchanged ⇒ no rounds
    assert "rounds_fired" not in idle.info
    assert abs(idle.info["waste_streak"] - (deep - C.WASTE_STREAK_DECAY)) < 1e-9, idle.info

    # (6) The streak is carried on a TERMINAL return too (info is the same dict).
    dead_bandit = _state(2, pos=[0, 4000, 4500], alive=False, hp=0.0)
    term = R.compute(me0, beyond, me_fired, dead_bandit, waste_streak=deep)
    assert term.terminated and "waste_streak" in term.info, term
    print("  fire burst-ramp OK — ramp/cap/reset/decay + backward-compat at streak 0 verified")


def test_p_hit_range_is_the_measured_curve():
    """B1: `range_factor` is the MEASURED per-round p_hit, not a linear `R_EFF` ramp.

    The old form paid 0.417 at 700 m and 0.250 at 900 m for shots the sim's ballistics land 6% and
    0.2% of the time — which is how a policy learned to fire at a median 1519 m. Pinning the knots
    (not just the shape) matters because `K_AIM = 60` multiplies differences in this curve: it is the
    single largest reward coefficient in the file, so a silent re-fit here is a silent reward rescale.
    """
    from astrum_dogfight.geometry import p_hit_range

    # The knots ARE the measurement (constants.py carries the conditions: 5 g bandit, computable lay).
    for rng, want in zip(C.PHIT_RANGE_M, C.PHIT_PER_ROUND):
        assert abs(p_hit_range(rng) - want) < 1e-12, (rng, p_hit_range(rng), want)

    # Clamped outside the measured span: point-blank is certain, past 1200 m is hopeless.
    assert p_hit_range(0.0) == 1.0 and p_hit_range(50.0) == 1.0
    assert p_hit_range(1200.0) == 0.0 and p_hit_range(5000.0) == 0.0

    # Monotone non-increasing — a gun cannot get better with range, and an interpolation bug that
    # put a bump in here would create a reward gradient pointing AWAY from the target.
    sweep = [p_hit_range(r) for r in range(0, 2001, 25)]
    assert all(b <= a + 1e-12 for a, b in zip(sweep, sweep[1:])), sweep

    # It is what `relative` actually reports, and it is materially LOWER than the old linear ramp in
    # the standoff band (this is the whole point of B1 — assert it so a revert is loud, not silent).
    me = _state(1, pos=[0, 0, 4500])
    for rng, old_linear in ((700.0, 0.417), (900.0, 0.250)):
        rg = relative(me, _state(2, pos=[0, rng, 4500]))
        assert abs(rg.range_factor - p_hit_range(rng)) < 1e-12, rg
        assert rg.range_factor < old_linear / 4.0, (rng, rg.range_factor, old_linear)
    # ...and HIGHER inside ~460 m: B1 is a REALLOCATION of the aim payment toward the lethal
    # envelope, not a uniform tightening. Flagged here so the amplification is on the record.
    rg_close = relative(me, _state(2, pos=[0, 400, 4500]))
    assert rg_close.range_factor > (C.R_EFF - 400.0) / C.R_EFF, rg_close.range_factor
    print("  p_hit range curve OK — knots/clamps/monotone + reallocation direction verified")


def test_lethal_range_gates_the_wez_bonus():
    """B1: the per-round "this round connects" bonus is gated on the LETHAL envelope, not `R_EFF`.

    `in_range` runs to 1200 m, so before this an 1100 m burst with a perfect COMPUTABLE lay collected
    the full bonus at a measured p_hit of 0.000 — the reward paid for exactly the standoff spray the
    policy had learned. Geometry is static and co-speed in every case here (Δq = 0 ⇒ no aim PBRS), so
    the fire term is what moves.
    """
    me0 = _state(1, pos=[0, 0, 4500], ammo=511)
    me_fired = _state(1, pos=[0, 0, 4500], ammo=501)  # 10 rounds gone

    # 600 m exactly = the edge, and it is INCLUSIVE: the bonus is paid and the burst is forgiven.
    edge = _state(2, pos=[0, C.LETHAL_RANGE_M, 4500])
    rg_edge = relative(me0, edge)
    assert rg_edge.lay_feasible and rg_edge.lay_miss_m < C.WEZ_HIT_M, rg_edge
    out_edge = R.compute(me0, edge, me_fired, edge, waste_streak=200.0)
    assert out_edge.reward > 0.4 and out_edge.info["waste_streak"] == 0.0, out_edge

    # Just outside it, and far outside it, the SAME perfect lay is now waste — and feeds the ramp.
    for rng in (700.0, 1100.0):
        tgt = _state(2, pos=[0, rng, 4500])
        rg = relative(me0, tgt)
        assert rg.in_range and rg.lay_feasible and rg.lay_miss_m < C.WEZ_HIT_M, (rng, rg)
        out = R.compute(me0, tgt, me_fired, tgt)
        assert out.info["rounds_fired"] == 10, out
        assert out.reward < 0.0, (rng, out.reward)
        assert out.info["waste_streak"] == 10.0, out.info
    print("  lethal-range fire gate OK — 600 m inclusive pays, 700/1100 m perfect lays now waste")


def test_r_eff_is_the_opponent_benchmark_not_the_lethal_envelope():
    """The invariant B1 must not break: `R_EFF` is BENCHMARK surface, `LETHAL_RANGE_M` is learner surface.

    `in_range` is the only live reader of `R_EFF`, and the scripted T3's own trigger fires on it
    (`opponent.py:wez_fire`). Collapsing `R_EFF` to 600 m — a tempting "cleanup" while making the WEZ
    honest — would stop the benchmark shooting in the 600-1200 m band and void every score measured
    against it, the same class of silent benchmark change as the T3 ungated-pull defect. Three
    incidental guards used to catch that by accident; making `range_factor` honest removed all three,
    so it is pinned explicitly here instead.
    """
    from astrum_dogfight import bfm_detect

    assert C.R_EFF == 1200.0, "R_EFF is the scripted T3's trigger range — moving it voids the benchmark"
    assert C.LETHAL_RANGE_M == 600.0
    assert C.LETHAL_RANGE_M < C.R_EFF, "the honest envelope must stay strictly inside the benchmark's"

    # The benchmark still shoots between the two: this is the band the reward no longer pays for.
    me = _state(1, pos=[0, 0, 4500])
    for rng in (700.0, 900.0, 1100.0):
        assert wez_fire(me, _state(2, pos=[0, rng, 4500])), f"T3 must still fire at {rng:.0f} m"
    assert not wez_fire(me, _state(2, pos=[0, 1300.0, 4500])), "past R_EFF the T3 must hold fire"

    # One source of truth: the reward's `in_solution` gate and the D5 `lethal` tick read the SAME
    # number, so the BFM instrument can never disagree with the reward about what a killing shot is.
    assert bfm_detect.LETHAL_RANGE_M == C.LETHAL_RANGE_M
    print("  R_EFF/LETHAL_RANGE separation OK — benchmark range pinned, one lethal constant shared")


def test_spiral_penalty():
    """reward-v16 / v16b ANTI-SPIRAL: the nose-low DESCENDING-SPIRAL attitude (nose-low × banked) is taxed
    per tick so the policy reverses in the VERTICAL (a nose-high egg) instead of corkscrewing at the deck —
    but only DOWN LOW (reward-v16b altitude gate), where a spiral is actually deadly. A wings-level dive
    (bank→0) OR a nose-HIGH banked turn (the egg; nose-low→0) pays NOTHING, a real gun solution GATES it
    OFF, and — the v16b fix — the SAME corkscrew attitude at fighting altitude pays NOTHING so aggressive
    converting turns up high stay free (the c58 passivity cure). Not a climb bonus; only a LOW corkscrew."""
    # Bandit far (4 km) ⇒ gun_quality 0 ⇒ the gun gate is OPEN for every case below.
    beyond = _state(2, pos=[0, 4000, 4500])

    # alt_frac ramps 0→1 as AGL falls from SPIRAL_ALT_BAND_AGL (2000) to the deck-cushion top
    # (DECK_AGL + DECK_MARGIN = 800). Pick test altitudes ON that ramp deliberately (ground_alt=0 ⇒ AGL=alt).
    ALT_MID = 1400.0   # agl 1400 ⇒ alt_frac 0.5 (mid-band); >800 ⇒ deck cushion stays 0 (no confound)
    ALT_LOW = 800.0    # agl 800 ⇒ alt_frac 1.0 (full penalty) AND deck_frac exactly 0 (no confound)
    ALT_HIGH = 4500.0  # fighting altitude, above the band ⇒ alt_frac 0 ⇒ penalty FREED (v16b)
    alt_floor = C.DECK_AGL + C.DECK_MARGIN
    mid_level = _state(1, pos=[0, 0, ALT_MID])  # phi=0, theta=0 (wings level, nose level), in-band

    # (1) Nose-low AND banked (the descending spiral) DOWN LOW ⇒ a negative spiral penalty.
    spiral = replace(mid_level, theta=-0.6, phi=1.0)  # ~34° nose-down, ~57° bank
    out_spiral = R.compute(mid_level, beyond, spiral, beyond)
    assert out_spiral.info.get("gun_quality", 0.0) == 0.0, "far bandit ⇒ gate open"
    assert out_spiral.info.get("spiral_penalty", 0.0) < 0.0, out_spiral.info
    # magnitude = K_SPIRAL · nose_low · bank_frac · alt_frac (all fractions in (0,1] here).
    nose_low = min(1.0, 0.6 / C.SPIRAL_PITCH_RAD)
    bank_frac = min(1.0, (1.0 - C.SPIRAL_BANK_MIN_RAD) / (C.SPIRAL_BANK_MAX_RAD - C.SPIRAL_BANK_MIN_RAD))
    alt_frac = (C.SPIRAL_ALT_BAND_AGL - ALT_MID) / (C.SPIRAL_ALT_BAND_AGL - alt_floor)
    assert abs(out_spiral.info["spiral_penalty"] - C.K_SPIRAL * nose_low * bank_frac * alt_frac) < 1e-9, out_spiral.info

    # (2) A wings-LEVEL dive (bank→0) pays NOTHING — only the corkscrew, not the dive itself, is taxed.
    dive = replace(mid_level, theta=-0.6, phi=0.0)
    out_dive = R.compute(mid_level, beyond, dive, beyond)
    assert "spiral_penalty" not in out_dive.info, out_dive.info

    # (3) The vertical EGG (nose-HIGH + banked) pays NOTHING — this is exactly the maneuver we want.
    egg = replace(mid_level, theta=+0.6, phi=1.0)
    out_egg = R.compute(mid_level, beyond, egg, beyond)
    assert "spiral_penalty" not in out_egg.info, out_egg.info

    # (4) A deeper/steeper spiral DOWN LOW is penalised MORE, up to the K_SPIRAL cap. Placed at ALT_LOW
    #     where alt_frac saturates to 1, so it reaches the true cap and never exceeds |K_SPIRAL|.
    floor_level = _state(1, pos=[0, 0, ALT_LOW])
    steep = replace(floor_level, theta=-1.2, phi=1.4)  # near-saturating nose_low & bank
    out_steep = R.compute(floor_level, beyond, steep, beyond)
    assert out_steep.info["spiral_penalty"] <= out_spiral.info["spiral_penalty"], (out_steep.info, out_spiral.info)
    assert out_steep.info["spiral_penalty"] >= C.K_SPIRAL - 1e-9, out_steep.info
    assert abs(out_steep.info["spiral_penalty"] - C.K_SPIRAL) < 1e-9, out_steep.info  # full alt_frac ⇒ hits the cap

    # (5) reward-v16b ALTITUDE GATE: the SAME nose-low + banked corkscrew at FIGHTING ALTITUDE pays
    #     NOTHING — an aggressive converting turn up high is legitimate BFM, not a death spiral (this is
    #     the fix for c58's global passivity, where the v16 penalty taxed the converting turn everywhere).
    high_level = _state(1, pos=[0, 0, ALT_HIGH])
    high_spiral = replace(high_level, theta=-0.6, phi=1.0)
    out_high = R.compute(high_level, beyond, high_spiral, beyond)
    assert out_high.info.get("gun_quality", 0.0) == 0.0, "far bandit ⇒ gun gate open (only the alt gate exempts it)"
    assert "spiral_penalty" not in out_high.info, out_high.info

    # (6) GATED OFF in a real gun solution: the SAME nose-low + banked attitude DOWN LOW, but the bandit
    # lies DEAD ALONG the diving nose vector in close (a committed diving guns pass on a lower target). The
    # nose for (θ=−0.6, ψ=0) is [0, cosθ, sinθ] ≈ [0, 0.825, −0.565]; put a ~stationary bandit 400 m out
    # along it ⇒ high gun_quality ≥ SPIRAL_SOLUTION_Q ⇒ NO spiral penalty (never tax a real diving shot).
    # The velocity must be set ALONG that nose: `_state`'s default is level-along-heading, which for
    # θ=−0.6 would leave the velocity vector 34° off the nose, and since curriculum-86 `lead_error` is the
    # velocity-inheritance-corrected lay ([[gun-lay-velocity-inheritance]]) a 34° off-velocity shot has a
    # large — and physically real — lay correction, so gun_quality would collapse to 0.11. That is the
    # correct answer for that (unflyable) state; a committed diving pass flies down its own nose.
    n_e, n_n, n_u = 0.0, np.cos(-0.6), np.sin(-0.6)
    me_in_solution = replace(mid_level, theta=-0.6, phi=1.0, vel=np.array([n_e, n_n, n_u]) * 180.0)
    below = _state(2, pos=[400 * n_e, 400 * n_n, ALT_MID + 400 * n_u], vel=[0.0, 0.0, 0.0])
    out_gated = R.compute(me_in_solution, below, me_in_solution, below)
    assert out_gated.info.get("gun_quality", 0.0) >= C.SPIRAL_SOLUTION_Q, out_gated.info
    assert "spiral_penalty" not in out_gated.info, out_gated.info
    print("  spiral penalty OK — low nose-low×bank taxed; dive / egg / high-alt corkscrew / gun-solution exempt")


def test_aspect_conversion_reward():
    """reward-v13: the stern-conversion PBRS pays to REDUCE the bandit aspect (fly to his six),
    costs the same to fall off it (symmetric → telescoping → non-farmable), and cannot outbid the
    kill. This is the merge→WEZ gradient the c35 policy lacked (it closed range nose-off at aspect
    ≥70° and sprayed empty air)."""
    bandit = _state(2, pos=[0, 0, 4500], psi=0.0)  # faces north; his tail points south
    abeam = _state(1, pos=[2000, 0, 4500])   # due EAST of him → high aspect (~90°, abeam the merge)
    stern = _state(1, pos=[0, -2000, 4500])  # due SOUTH of him (behind) → low aspect (~0°, at his six)
    a_abeam = degrees(relative(abeam, bandit).aspect)
    a_stern = degrees(relative(stern, bandit).aspect)
    assert a_stern < a_abeam, (a_stern, a_abeam)  # the stern IS a lower aspect than abeam

    # Converting abeam → stern (aspect DROPS) PAYS a positive aspect_reward.
    conv = R.compute(abeam, bandit, stern, bandit)
    assert conv.info["aspect_reward"] > 0.0, conv.info

    # Falling off (stern → abeam, aspect RISES) costs exactly the reverse ⇒ any wobble that
    # returns to the start sums to ~0 (telescoping / policy-invariant / non-farmable).
    fall = R.compute(stern, bandit, abeam, bandit)
    assert fall.info["aspect_reward"] < 0.0, fall.info
    assert abs(conv.info["aspect_reward"] + fall.info["aspect_reward"]) < 1e-9, (conv.info, fall.info)

    # Bounded by K_ASPECT: a full 180°→0° swing caps here.
    assert 0.0 < conv.info["aspect_reward"] <= C.K_ASPECT + 1e-9, conv.info

    # …and the FINISH must still outbid parking on the six. curriculum-93 raised the cap 80 → 144
    # (see PURSUIT_GATE_HI), so "convert then let the clock run" now scores K_ASPECT + R_TIMEOUT =
    # +94 where it used to score +30 — worth pinning, because the same conversion is paid in BOTH
    # branches, so what actually decides the policy is the MARGIN between them. That margin is
    # R_KILL − R_TIMEOUT and is invariant to K_ASPECT by construction; it must stay both positive
    # and large next to the shaping term, or the PBRS starts drowning the outcome it exists to serve.
    kill_margin = C.R_KILL - C.R_TIMEOUT
    assert kill_margin > 0.0, (C.R_KILL, C.R_TIMEOUT)
    assert C.K_ASPECT <= kill_margin, (C.K_ASPECT, kill_margin)

    # Static geometry (no aspect change) ⇒ exactly zero (backward-compatible with the other tests).
    flat = R.compute(stern, bandit, stern, bandit)
    assert abs(flat.info["aspect_reward"]) < 1e-12, flat.info
    print("  aspect OK — stern-conversion PBRS pays to reduce aspect, telescopes, bounded by K_ASPECT")


def test_aspect_pbrs_slope_is_live_over_the_whole_aspect_range():
    """curriculum-93: the stern-conversion PBRS must have a NONZERO, CONSTANT gradient all the way to
    180°, and that gradient must be the historical one.

    Ψ(A) = −min(A, PURSUIT_GATE_HI)/PURSUIT_GATE_HI was clipped flat above 100°, so across the entire
    100–180° band dΨ/dA was exactly 0.000/deg — while the measured merge failure state sits at aspect
    147–158° (curriculum-91/92), i.e. wholly inside the flat region. Un-clipping to 180° fixes that, but
    only if K_ASPECT is scaled by the same factor: the slope is K_ASPECT/PURSUIT_GATE_HI per radian, so
    widening alone would have quietly taxed the chase finish's own gradient to 0.55×. This pins BOTH
    halves of that coupling, so the two constants can never drift apart silently."""
    # 1. The slope is the historical 80/rad(100°) — the un-clip is a strict no-op below 100° aspect.
    slope = C.K_ASPECT / C.PURSUIT_GATE_HI          # reward units per radian of aspect
    assert abs(slope - 80.0 / radians(100.0)) < 1e-6, (slope, C.K_ASPECT, C.PURSUIT_GATE_HI)

    # 2. No clip survives anywhere in the achievable range: aspect is bounded by 180°.
    assert C.PURSUIT_GATE_HI >= radians(180.0) - 1e-12, degrees(C.PURSUIT_GATE_HI)

    # 3. Ψ is strictly monotone across the previously-flat band — measured on the reward itself, not on a
    #    re-implementation of Ψ, so a change to reward.py's formula also trips this.
    bandit = _state(2, pos=[0, 0, 4500], psi=0.0)  # faces north
    def at(deg_off):
        """Learner placed at bearing `deg_off` around the bandit ⇒ bandit aspect ≈ deg_off."""
        r = 2000.0
        return _state(1, pos=[r * sin(radians(deg_off)), -r * cos(radians(deg_off)), 4500])

    prev_pay = None
    for lo, hi in ((100.0, 120.0), (120.0, 140.0), (140.0, 160.0), (160.0, 175.0)):
        # Converting from the HIGHER aspect down to the LOWER one must PAY, everywhere above 100°.
        pay = R.compute(at(hi), bandit, at(lo), bandit).info["aspect_reward"]
        assert pay > 0.0, (lo, hi, pay)
        if prev_pay is not None:  # equal-width steps pay equally ⇒ the gradient is CONSTANT, not curved
            assert abs(pay - prev_pay * (hi - lo) / 20.0) < 1e-6 * max(1.0, pay), (lo, hi, pay, prev_pay)
        prev_pay = pay if hi - lo == 20.0 else prev_pay
    print("  aspect PBRS OK — un-clipped to 180°, slope preserved at the historical 80/rad(100°)")


def test_conversion_terminal():
    """reward-v17 (c60 two-policy) + reward-v18 (settled hand-off): the merge-converter M's
    SURVIVE-CONVERT terminal. It fires — and ONLY fires — when convert_bonus>0 AND M sits in a genuine,
    FINISHABLE rear-quarter firing position behind a LIVE bandit: aspect ≤ CONVERT_ASPECT_RAD (his six),
    nose-on ata ≤ CONVERT_ATA_RAD (a head-on fly-through where I'm swept behind but flying AWAY does NOT
    count), the jet SETTLED / roughly wings-level |phi| ≤ CONVERT_BANK_RAD (reward-v18 — rejects the
    ~100° knife-edge merge-slice that M clipped the box with and F then decked), range ∈ [MIN, MAX]
    (F's band). Strict no-op for every legacy run (convert_bonus=0), yields to kill/death/deck
    precedence, and SUPPRESSES M's WEZ fire bonus (M saves ammo for the frozen finisher F)."""
    BONUS = 150.0
    bandit = _state(2, pos=[0, 0, 4500], psi=0.0)  # faces north; tail points south
    # Genuine conversion: 500 m due SOUTH of him (his six ⇒ low aspect), nose pointing NORTH at
    # him (ata ~0), WINGS LEVEL (phi=0), inside [MIN, MAX] ⇒ the hand-off geometry.
    behind = _state(1, pos=[0, -500, 4500], psi=0.0)
    rg = relative(behind, bandit)
    assert rg.aspect <= C.CONVERT_ASPECT_RAD and rg.ata <= C.CONVERT_ATA_RAD, rg
    assert abs(behind.phi) <= C.CONVERT_BANK_RAD, behind.phi  # wings level ⇒ settled
    assert C.CONVERT_RANGE_MIN <= rg.rng <= C.CONVERT_RANGE_MAX, rg.rng

    # convert_bonus=0 (legacy) ⇒ NO conversion (episode continues); convert_bonus>0 ⇒ terminal, and
    # in this static no-fire geometry the ONLY reward delta is exactly +BONUS.
    noconv = R.compute(behind, bandit, behind, bandit)
    conv = R.compute(behind, bandit, behind, bandit, convert_bonus=BONUS)
    assert not noconv.terminated and noconv.info.get("event") != "conversion", noconv.info
    assert conv.terminated and conv.info["event"] == "conversion", conv.info
    assert abs((conv.reward - noconv.reward) - BONUS) < 1e-9, (conv.reward, noconv.reward)

    # Head-on FLY-THROUGH: same rear-hemisphere position but facing AWAY (south) ⇒ aspect still low,
    # but ata ~180 ⇒ the ata gate (not the aspect gate) excludes it. This is the false-positive guard.
    away = _state(1, pos=[0, -500, 4500], psi=np.pi)
    rg_away = relative(away, bandit)
    assert rg_away.aspect <= C.CONVERT_ASPECT_RAD, rg_away.aspect  # aspect alone would have passed
    assert rg_away.ata > C.CONVERT_ATA_RAD, rg_away.ata            # ata is what rejects it
    ft = R.compute(away, bandit, away, bandit, convert_bonus=BONUS)
    assert not ft.terminated and ft.info.get("event") != "conversion", ft.info

    # reward-v18 KNIFE-EDGE guard: the genuine conversion geometry (aspect/ata/range dead-on) but the
    # jet BANKED ~100° is REJECTED — this is the exact c60 seam failure (M clipped the box mid-merge at
    # ~100° of bank, F inherited an unflyable knife-edge and spiralled into the deck). Bank doesn't
    # enter the LOS geometry, so rg is identical to the genuine case; only the settledness gate differs.
    knife = replace(behind, phi=np.radians(100.0))
    rg_knife = relative(knife, bandit)
    assert rg_knife.aspect <= C.CONVERT_ASPECT_RAD and rg_knife.ata <= C.CONVERT_ATA_RAD, rg_knife
    assert C.CONVERT_RANGE_MIN <= rg_knife.rng <= C.CONVERT_RANGE_MAX, rg_knife.rng
    assert abs(knife.phi) > C.CONVERT_BANK_RAD, knife.phi         # bank is what rejects it
    ke = R.compute(knife, bandit, knife, bandit, convert_bonus=BONUS)
    assert not ke.terminated and ke.info.get("event") != "conversion", ke.info

    # Out of F's prosecution band by RANGE: too far (1500 m > MAX) and too close (100 m < MIN, the
    # merge singularity) both excluded even though aspect+ata are dead-on.
    for y in (-1500.0, -100.0):
        m = _state(1, pos=[0, y, 4500], psi=0.0)
        rg_m = relative(m, bandit)
        assert rg_m.aspect <= C.CONVERT_ASPECT_RAD and rg_m.ata <= C.CONVERT_ATA_RAD, rg_m
        assert not (C.CONVERT_RANGE_MIN <= rg_m.rng <= C.CONVERT_RANGE_MAX), rg_m.rng
        out = R.compute(m, bandit, m, bandit, convert_bonus=BONUS)
        assert not out.terminated and out.info.get("event") != "conversion", (y, out.info)

    # Fire-bonus SUPPRESSION: 100 m co-heading on the nose = an in-solution burst (rng<MIN so the
    # conversion terminal can't mask the fire block). convert_bonus=0 pays the WEZ bonus and forgives
    # the burst; convert_bonus>0 pays NO bonus, treats the rounds as waste ⇒ strictly lower reward.
    shooter0 = _state(1, pos=[0, 0, 4500], psi=0.0, ammo=511)
    shooter1 = _state(1, pos=[0, 0, 4500], psi=0.0, ammo=501)  # 10 rounds gone
    target = _state(2, pos=[0, 100, 4500], psi=0.0)            # 100 m ahead, co-heading ⇒ in solution
    paid = R.compute(shooter0, target, shooter1, target)
    supp = R.compute(shooter0, target, shooter1, target, convert_bonus=BONUS)
    assert paid.info.get("rounds_fired") == 10 and supp.info.get("rounds_fired") == 10, (paid.info, supp.info)
    assert paid.info["waste_streak"] == 0.0, paid.info      # in-solution round forgiven
    assert supp.info["waste_streak"] == 10.0, supp.info     # M's rounds counted as waste
    assert supp.reward < paid.reward, (supp.reward, paid.reward)
    assert supp.info.get("event") != "conversion", supp.info  # rng<MIN ⇒ no conversion here

    # Precedence: kill / death / hard-deck all outrank the conversion terminal.
    dead_bandit = _state(2, pos=[0, 0, 4500], psi=0.0, alive=False, hp=0.0)
    kill = R.compute(behind, bandit, behind, dead_bandit, convert_bonus=BONUS)
    assert kill.info["event"] == "kill", kill.info
    dead_me = _state(1, pos=[0, -500, 4500], psi=0.0, alive=False, hp=0.0)
    death = R.compute(behind, bandit, dead_me, bandit, convert_bonus=BONUS)
    assert death.info["event"] == "killed", death.info
    low = _state(1, pos=[0, -500, 100], psi=0.0, alt=100.0)  # below the hard deck, convert geometry
    low_bandit = _state(2, pos=[0, 0, 100], psi=0.0, alt=100.0)
    deck = R.compute(low, low_bandit, low, low_bandit, convert_bonus=BONUS)
    assert deck.info["event"] == "hard_deck", deck.info
    print("  conversion OK — v17/v18 terminal fires only settled+in-band (aspect+ata+bank+range), rejects knife-edge, suppresses fire, yields to kill/death/deck")


def test_closing_reward():
    """reward-v6: potential-based closing-range shaping F = K_CLOSE·(Φ(rng′)−Φ(rng)) with
    Φ(r) = −clip(r, R_CLOSE_MIN, TRACK_RANGE)/TRACK_RANGE, discount 1.0. Verifies the three
    properties that make it a *non-farmable pull into gun range*: closing pays, opening costs,
    and — critically — holding range is exactly zero (a γ<1 PBRS on a negative potential would
    instead pay to sit far). The isolated `info["close_reward"]` sub-total is checked directly
    so the tracking term doesn't confound the sign/magnitude."""
    me = _state(1, pos=[0, 0, 4500])
    at = lambda d: _state(2, pos=[0, d, 4500])  # bandit d metres north (on the nose) → range d
    k = C.K_CLOSE / C.TRACK_RANGE

    # Closing 900 → 700 m: +K_CLOSE·(900−700)/TRACK_RANGE.
    out = R.compute(me, at(900), me, at(700))
    assert abs(out.info["close_reward"] - k * 200.0) < 1e-6, out.info
    assert out.info["close_reward"] > 0.0

    # Opening 700 → 900 m: exact negative of the closing case (telescoping is symmetric).
    out_open = R.compute(me, at(700), me, at(900))
    assert abs(out_open.info["close_reward"] + k * 200.0) < 1e-6, out_open.info
    assert out_open.info["close_reward"] < 0.0

    # Holding range: EXACTLY zero closing reward (the no-standing-bias property). A γ<1 PBRS
    # would leak a positive per-tick bonus here and re-create a stay-far farm.
    out_hold = R.compute(me, at(700), me, at(700))
    assert out_hold.info["close_reward"] == 0.0, out_hold.info

    # curriculum-82: with R_CLOSE_MIN LOWERED to 150 m, closing THROUGH the old 250 m clip into the
    # gun-kill control zone now PAYS a real gradient (250 → 180 m, both above the floor) — this is the
    # pull that lets the policy cross the 250-400 m overshoot valley instead of stalling at 250 m.
    out_zone = R.compute(me, at(250), me, at(180))
    assert abs(out_zone.info["close_reward"] - k * 70.0) < 1e-6, out_zone.info
    assert out_zone.info["close_reward"] > 0.0

    # Below the new R_CLOSE_MIN clip there is still no further pull (anti-ram floor): both operands
    # clip to R_CLOSE_MIN, so a dive from just inside the floor to point-blank earns nothing.
    out_ram = R.compute(me, at(C.R_CLOSE_MIN - 20.0), me, at(C.R_CLOSE_MIN - 70.0))
    assert out_ram.info["close_reward"] == 0.0, out_ram.info
    print("  closing-reward OK — closes pay (incl. into the control zone), opens cost, "
          "holding is zero, sub-R_CLOSE_MIN clipped")


def test_close_crossing_gate():
    """reward-v23: in the control-position gun zone (aspect ≤ CONVERT_ASPECT_RAD ∧ rng ≤
    CONVERT_RANGE_MAX) the POSITIVE closing bribe is paid ONLY at LOW crossing (|closure| ≤
    CONVERT_CLOSURE_MAX) — so the high-overtake crossing (which fights reward-v22's lead-null) no
    longer earns K_CLOSE, but EASING into range still does. Diagnosis: c73 converted to the six but
    the in-zone lead GREW because K_CLOSE bribed the overtake that sustains the crossing. Verifies:
    (1) in-zone + HIGH closure + outcome_mode → bribe suppressed to 0; (2) in-zone + LOW closure →
    full raw bribe paid; (3) OUT of zone + high closure → paid (the approach is unchanged); (4)
    outcome_mode OFF → paid (legacy no-op); (5) OPENING in-zone at high closure is STILL charged
    (the negative part is never gated → non-farmable)."""
    # me on his six (bandit flies north/away ⇒ aspect≈0), closing 700→500 m (raw_close>0, in-zone).
    fast = _state(1, pos=[0, 0, 4500], vel=[0.0, 300.0, 0.0])  # closure = 300−180 = 120 > CONVERT_CLOSURE_MAX(90)
    slow = _state(1, pos=[0, 0, 4500], vel=[0.0, 180.0, 0.0])  # closure = 180−180 = 0   ≤ CONVERT_CLOSURE_MAX(90)
    b_far, b_near = _state(2, pos=[0, 700, 4500]), _state(2, pos=[0, 500, 4500])  # default vel (0,180,0)
    k = C.K_CLOSE / C.TRACK_RANGE

    # (1) in-zone, HIGH closure, outcome_mode ON → the overtake bribe is suppressed to exactly 0.
    assert R.compute(fast, b_far, fast, b_near, outcome_mode=True).info["close_reward"] == 0.0

    # (2) in-zone, LOW closure, outcome_mode ON → the full raw closing bribe is paid (ease-in).
    out_slow = R.compute(slow, b_far, slow, b_near, outcome_mode=True)
    assert abs(out_slow.info["close_reward"] - k * 200.0) < 1e-6, out_slow.info
    assert out_slow.info["close_reward"] > 0.0

    # (3) OUT of zone (rng > CONVERT_RANGE_MAX), HIGH closure → paid: the long-range approach is
    # unchanged (closing fast to the merge is still rewarded).
    of, on = _state(2, pos=[0, 1500, 4500]), _state(2, pos=[0, 1300, 4500])
    assert R.compute(fast, of, fast, on, outcome_mode=True).info["close_reward"] > 0.0

    # (4) outcome_mode OFF (legacy dense): the gate is a strict no-op → the bribe is paid.
    assert R.compute(fast, b_far, fast, b_near).info["close_reward"] > 0.0

    # (5) OPENING in-zone at high closure (range 500→700) is STILL charged in full — the negative
    # part is never gated, so no fast-close(=0)/open(<0) cycle can farm positive.
    out_open = R.compute(fast, b_near, fast, b_far, outcome_mode=True)
    assert abs(out_open.info["close_reward"] + k * 200.0) < 1e-6, out_open.info
    assert out_open.info["close_reward"] < 0.0
    print("  close-crossing-gate OK — high-overtake bribe suppressed in-zone, ease-in paid, "
          "approach/legacy unchanged, opening always charged")


def test_aim_reward():
    """reward-v8: potential-based AIM shaping F = K_AIM·(q′ − q) on gun-solution quality q,
    discount 1.0. Same non-farmable-PBRS properties as the closing term — sharpening the firing
    solution pays, losing it costs, holding it is EXACTLY zero, and it is identically zero out of
    gun range (so it cannot revive a standoff farm). Checked on info["aim_reward"] in isolation."""
    from astrum_dogfight.geometry import relative

    me = _state(1, pos=[0, 0, 4500])
    at = lambda d: _state(2, pos=[0, d, 4500])  # bandit d m on the nose, flying away → clean q

    # Sharpening the solution: bandit 800 → 400 m on the nose raises range_factor → q up.
    q_lo = relative(me, at(800)).gun_quality
    q_hi = relative(me, at(400)).gun_quality
    assert q_hi > q_lo > 0.0, (q_lo, q_hi)
    out = R.compute(me, at(800), me, at(400))
    assert abs(out.info["aim_reward"] - C.K_AIM * (q_hi - q_lo)) < 1e-6, out.info
    assert out.info["aim_reward"] > 0.0

    # Losing the solution (400 → 800 m): exact negative — telescoping is symmetric.
    out_lose = R.compute(me, at(400), me, at(800))
    assert abs(out_lose.info["aim_reward"] + C.K_AIM * (q_hi - q_lo)) < 1e-6, out_lose.info
    assert out_lose.info["aim_reward"] < 0.0

    # Holding the solution: EXACTLY zero (non-farmable — holding a solution nets no shaping).
    out_hold = R.compute(me, at(400), me, at(400))
    assert out_hold.info["aim_reward"] == 0.0, out_hold.info

    # Out of gun range (both beyond R_EFF): gun_quality ≡ 0 → aim_reward ≡ 0, no standoff farm.
    out_far = R.compute(me, at(3000), me, at(2500))
    assert out_far.info["aim_reward"] == 0.0, out_far.info
    print("  aim-reward OK — sharpen pays, lose costs, holding is zero, out-of-range is zero")


def test_lead_reward():
    """reward-v22: lead-error PBRS F = K_LEAD·(Φ_L(lead′) − Φ_L(lead)), Φ_L(θ) = (1+cos θ)/2, that fills
    the gradient hole the peaked gun_q Gaussian (LEAD_SIGMA=3°) leaves at a 30-57° crossing — the reason
    the policy AND the perfect BFM coach reach aspect 4° yet dump the mag for 0 damage. Verified: (1)
    outcome_mode-gated (no-op for legacy runs); (2) in the rear-quarter gun zone, sharpening the lead pays
    and it is telescoping; (3) ONE-SIDED — losing lead is ALWAYS charged (even out of zone) but sharpening
    is paid ONLY in-zone (non-farmable, like K_CLOSE); (4) holding lead is exactly zero. The bandit is held
    co-heading 400 m on the LOS (aspect ≡ 0 ⇒ in zone, independent of MY heading), and only MY nose yaws to
    move the lead error, so the term is isolated from aspect/closing/aim confounds."""
    from astrum_dogfight.geometry import relative

    me_on = _state(1, pos=[0, 0, 4500], psi=0.0)            # nose north, straight at the bandit ⇒ lead≈0
    me_off = _state(1, pos=[0, 0, 4500], psi=radians(30.0))  # nose yawed 30° ⇒ lead≈30°
    away = _state(2, pos=[0, 400, 4500], psi=0.0)            # 400 m N, co-heading away ⇒ aspect 0, in range
    phi_on = (1.0 + cos(relative(me_on, away).lead_error)) * 0.5
    phi_off = (1.0 + cos(relative(me_off, away).lead_error)) * 0.5
    assert phi_on > phi_off > 0.0, (phi_on, phi_off)

    # (1) outcome_mode OFF (default): the term is a strict no-op — the key is absent.
    assert "lead_reward" not in R.compute(me_off, away, me_on, away).info

    # (2)+(3a) sharpening the lead IN ZONE pays: K_LEAD·(Φ_on − Φ_off) > 0.
    out = R.compute(me_off, away, me_on, away, outcome_mode=True)
    assert abs(out.info["lead_reward"] - C.K_LEAD * (phi_on - phi_off)) < 1e-6, out.info
    assert out.info["lead_reward"] > 0.0

    # (3b) losing the lead is the exact negative (telescoping, always charged in zone).
    out_lose = R.compute(me_on, away, me_off, away, outcome_mode=True)
    assert abs(out_lose.info["lead_reward"] + C.K_LEAD * (phi_on - phi_off)) < 1e-6, out_lose.info
    assert out_lose.info["lead_reward"] < 0.0

    # (4) holding the lead: exactly zero (non-farmable — no standing bias).
    assert R.compute(me_on, away, me_on, away, outcome_mode=True).info["lead_reward"] == 0.0

    # (3c) ONE-SIDED zone gate: a head-on bandit (aspect ≈ 180° ⇒ OUT of zone) — sharpening the lead is
    # NOT paid (raw>0 but out of zone ⇒ 0), yet losing it is STILL charged (so no in/out wobble farms).
    head_on = _state(2, pos=[0, 400, 4500], psi=radians(180.0))  # nose toward me ⇒ aspect ≈ 180°
    assert relative(me_on, head_on).aspect > C.CONVERT_ASPECT_RAD
    assert R.compute(me_off, head_on, me_on, head_on, outcome_mode=True).info["lead_reward"] == 0.0
    assert R.compute(me_on, head_on, me_off, head_on, outcome_mode=True).info["lead_reward"] < 0.0
    print("  lead-reward OK — outcome-gated, sharpen pays in-zone, lose always costs, hold zero, one-sided")


def test_sigma_track_honest_gun_quality():
    """G1 (curriculum-78): the range-dependent HONEST gun-solution quality. `sigma_track(rng)`
    tightens the tracking Gaussian with range (clamped at LEAD_SIGMA close-in), so the REWARD-only
    `gun_quality_honest` collapses the FALSE long-range merge "solution" (3°@494m scores ~0.36 on
    the loose 3° Gaussian yet lands a 26 m miss ≫ the 8 m hit sphere) while staying IDENTICAL to the
    observed `gun_quality` at the close chase-finish. Verifies: (1) sigma_track monotone-tightens and
    clamps to [FLOOR, CAP]; (2) close-in (≤~180 m) honest == observed (finish preserved); (3) at
    494m/3° honest ≪ observed (still COMPUTED as telemetry); (4) curriculum-80 P2 (REVERTED): the
    OBSERVATION carries the LOOSE gun_quality (obs idx 29) again — the honest σ flattened the cold
    aim gradient (0 kills / sprayed for ~0 HP); (5) the reward K_SOLUTION/K_AIM terms use the LOOSE
    quality in BOTH modes (identical), while outcome_mode still emits gun_quality_honest telemetry."""
    from math import atan

    from astrum_dogfight.geometry import (
        GUN_QUALITY_OBS_INDEX,
        build_observation,
        relative,
        sigma_track,
    )

    # (1) sigma_track: tightens with range, clamped to [FLOOR, CAP]; explicit 494 m ≈ 1.39°.
    assert abs(sigma_track(150.0) - C.LEAD_SIGMA_CAP) < 1e-12, "close-in ⇒ clamped at the 3° cap"
    assert sigma_track(500.0) < sigma_track(300.0) < C.LEAD_SIGMA_CAP, "tightens with range"
    assert sigma_track(1e7) >= C.LEAD_SIGMA_FLOOR - 1e-12, "never below the floor"
    expect494 = atan((C.HIT_RADIUS_M + C.DISP_MARGIN * C.DISP_SIGMA * 494.0) / 494.0)
    assert abs(sigma_track(494.0) - expect494) < 1e-9, (sigma_track(494.0), expect494)
    assert 1.2 < degrees(sigma_track(494.0)) < 1.6, degrees(sigma_track(494.0))

    # (2) CLOSE finish preserved: at 150 m on the nose the honest quality == the observed quality.
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    near = _state(2, pos=[0, 150, 4500], psi=0.0)
    rg_near = relative(me, near)
    assert abs(rg_near.gun_quality_honest - rg_near.gun_quality) < 1e-9, rg_near

    # (3) FALSE merge solution collapses: yaw my nose 3° off a co-heading bandit 494 m on the LOS.
    # (The aim point stays due north — bandit flies away up the LOS — so the NAIVE lead is exactly 3°.)
    me3 = _state(1, pos=[0, 0, 4500], psi=radians(3.0))
    tgt = _state(2, pos=[0, 494, 4500], psi=0.0)
    rg3 = relative(me3, tgt)
    assert abs(degrees(rg3.naive_lead_error) - 3.0) < 1e-6, degrees(rg3.naive_lead_error)
    # curriculum-86: `lead_error` is now the velocity-inheritance-corrected lay, and it is STRICTLY WORSE
    # here — my 180 m/s velocity is itself yawed 3° off the LOS, so the rounds carry a 9.4 m/s lateral
    # component the nose must cancel: the required boresight sits 3.52° off, not 3.00°, and the predicted
    # miss is 30.4 m rather than the naive 25.9 m. See [[gun-lay-velocity-inheritance]].
    assert degrees(rg3.lead_error) > degrees(rg3.naive_lead_error), rg3
    assert abs(degrees(rg3.lead_error) - 3.52) < 0.01, degrees(rg3.lead_error)
    assert rg3.lay_feasible and abs(rg3.lay_miss_m - 30.4) < 0.1, rg3
    assert rg3.lay_miss_m > 494.0 * sin(rg3.naive_lead_error), "corrected miss exceeds the naive estimate"
    assert rg3.gun_quality > 0.25, ("loose 3.5° still reads as a near-solution", rg3.gun_quality)
    assert rg3.gun_quality_honest < 0.10, ("honest kills the 30 m-miss shot", rg3.gun_quality_honest)

    # (4) curriculum-80 P2 (REVERTED): the OBSERVATION carries the LOOSE gun_quality (obs idx 29) —
    #     the honest σ was flat at range and killed the cold aim gradient (see geometry.py note).
    obs = build_observation(me3, tgt)
    assert abs(float(obs[GUN_QUALITY_OBS_INDEX]) - rg3.gun_quality) < 1e-5, obs[GUN_QUALITY_OBS_INDEX]
    assert abs(float(obs[GUN_QUALITY_OBS_INDEX]) - rg3.gun_quality_honest) > 0.2, "obs is the LOOSE q again"

    # (5a) K_SOLUTION: BOTH legacy and outcome_mode use the LOOSE q (curriculum-80 P2 revert) ⇒ the
    #      track_reward is IDENTICAL; outcome_mode still emits gun_quality_honest as telemetry only.
    leg = R.compute(me3, tgt, me3, tgt)
    oc = R.compute(me3, tgt, me3, tgt, outcome_mode=True)
    assert abs(leg.info["gun_quality"] - rg3.gun_quality) < 1e-6, leg.info
    assert "gun_quality_honest" not in leg.info, "legacy emits no honest telemetry"
    assert abs(oc.info["gun_quality_honest"] - rg3.gun_quality_honest) < 1e-6, oc.info
    assert abs(leg.info["track_reward"] - oc.info["track_reward"]) < 1e-6, (
        leg.info["track_reward"], oc.info["track_reward"])

    # (5b) K_AIM (the aim PBRS): BOTH modes telescope the LOOSE q now (curriculum-80 P2 revert) ⇒
    #      identical aim_reward. (gun_quality_honest stays computed for telemetry / a future annealed σ.)
    at3 = lambda d: _state(2, pos=[0, d, 4500], psi=0.0)
    rg800, rg400 = relative(me3, at3(800.0)), relative(me3, at3(400.0))
    leg_aim = R.compute(me3, at3(800.0), me3, at3(400.0))
    oc_aim = R.compute(me3, at3(800.0), me3, at3(400.0), outcome_mode=True)
    assert abs(leg_aim.info["aim_reward"] - C.K_AIM * (rg400.gun_quality - rg800.gun_quality)) < 1e-6
    assert abs(oc_aim.info["aim_reward"] - leg_aim.info["aim_reward"]) < 1e-6, "loose in both modes"
    print("  sigma-track/honest-gun OK — sigma_track still tightens with range (telemetry), close "
          "finish preserved; curriculum-80 P2: obs+reward reverted to LOOSE q (cold aim gradient)")


def test_band_reward():
    """E1 (curriculum-78): the ONE-SIDED corner-speed-BAND airspeed PBRS. Φ_band(V) =
    exp(−½(min(0, V−V_BAND_CTR)/V_BAND_SIGMA)²) is FLAT (=1) at/above corner speed and falls off only
    below it, so regaining corner speed PAYS, bleeding below it COSTS, and carrying/zooming for energy
    is NEVER taxed (the deliberate reverse of the disproven reward-v26 overspeed tax). Verifies: (1)
    outcome_mode OFF ⇒ strict no-op (key absent); (2) FLAT above corner (a 200→250 accel pays 0); (3)
    a below-corner recovery pays > 0 and equals K_BAND·(Φ−Φ); (4) a below-corner bleed is the exact
    negative (telescoping / non-farmable); (5) a climb from below to above corner never over-pays past
    the flat top. Bandit held far (4 km, off the gun) so only the airspeed term moves band_reward."""
    beyond = _state(2, pos=[0, 4000, 4500])  # far ⇒ no gun/conversion terms confound the airspeed PBRS

    def band(v_prev, v_new, **kw):
        me_prev = _state(1, pos=[0, 0, 4500], airspeed=v_prev)
        me_new = _state(1, pos=[0, 0, 4500], airspeed=v_new)
        return R.compute(me_prev, beyond, me_new, beyond, **kw)

    def _phi(v):
        d = min(0.0, v - C.V_BAND_CTR) / C.V_BAND_SIGMA
        return np.exp(-0.5 * d * d)

    # (1) outcome_mode OFF (default) ⇒ strict no-op.
    assert "band_reward" not in band(120.0, 150.0).info

    # (2) FLAT above corner: accelerating 200→250 m/s (both ≥ V_BAND_CTR=175) pays EXACTLY 0 — carrying
    #     / zooming for energy is never taxed (no overspeed tax).
    assert abs(band(200.0, 250.0, outcome_mode=True).info["band_reward"]) < 1e-12
    assert abs(band(175.0, 175.0, outcome_mode=True).info["band_reward"]) < 1e-12  # at the knee

    # (3) RECOVERY below corner (120→150) pays > 0 and matches K_BAND·(Φ(150) − Φ(120)).
    up = band(120.0, 150.0, outcome_mode=True).info["band_reward"]
    assert abs(up - C.K_BAND * (_phi(150.0) - _phi(120.0))) < 1e-9, up
    assert up > 0.0

    # (4) BLEED below corner (150→120) is the exact negative — telescoping, non-farmable.
    down = band(150.0, 120.0, outcome_mode=True).info["band_reward"]
    assert abs(up + down) < 1e-9, (up, down)
    assert down < 0.0

    # (5) A climb from below to above corner (150→250) pays only up to the flat top (Φ caps at 1) —
    #     K_BAND·(1 − Φ(150)), never more, so there is no "coast to exactly corner" over-reward.
    thru = band(150.0, 250.0, outcome_mode=True).info["band_reward"]
    assert abs(thru - C.K_BAND * (1.0 - _phi(150.0))) < 1e-9, thru
    print("  band-reward OK — E1 outcome-gated, flat above corner (no overspeed tax), pays recovery, "
          "telescoping symmetric, capped at the flat top")


def test_overspeed_penalty():
    """reward-v26 RETIRED (curriculum-77): the CONVERSION OVERSPEED TAX (tax in-zone airspeed above a
    ceiling to force a bleed-to-corner) was COUNTERPRODUCTIVE — it made the learner timid in the tracking
    band so it never pressured the T3 into bleeding (c77 bandMinV ~171 vs c76's 92) and it disengaged
    (left_bubble) instead of pressing; c77 was strictly worse than c76's reward-v25 peak. Term reverted
    (K_OVERSPEED=0), so the reward no longer emits `overspeed_penalty`. Kept as a no-op marker so the
    retirement is visible in the suite (the real wall is the 8 m gun hit-radius vs the crossing lead)."""
    away = _state(2, pos=[0, 400, 4500], psi=0.0)                   # 400 m N, co-heading ⇒ aspect 0, in range
    me_hot = _state(1, pos=[0, 0, 4500], psi=0.0, airspeed=250.0)   # hot in-zone would have been taxed by v26
    assert C.K_OVERSPEED == 0.0, "reward-v26 is retired: K_OVERSPEED must be 0.0"
    assert "overspeed_penalty" not in R.compute(me_hot, away, me_hot, away, outcome_mode=True).info
    print("  overspeed-penalty RETIRED — reward-v26 reverted (no overspeed_penalty; K_OVERSPEED=0)")


def test_wound_amplifier():
    """reward-v15 RETIRED (2026-08-29): the wounded-bandit re-attack amplifier destabilized the policy
    (c42 regressed the proven 45° kill; the reward already favours the kill ~+260 vs ~−48, so the ≥55°
    wall is an execution/exploration gap, not a reward-shaping gap). Reverted to reward-v14b — the reward
    no longer emits `wound_factor`. Kept as a no-op marker so the retirement is visible in the suite."""
    out = R.compute(
        _state(1, pos=[0, 0, 4500]), _state(2, pos=[0, 700, 4500], hp=10.0),
        _state(1, pos=[0, 0, 4500]), _state(2, pos=[0, 700, 4500], hp=10.0),
    )
    assert "wound_factor" not in out.info, out.info  # v15 term is gone
    print("  wound-amplifier RETIRED — reward is reward-v14b (no wound_factor)")


def test_pursuit_gate():
    """curriculum-81 (WARM-ADAPT): the reward-v11 aspect gate is a NO-OP again (PURSUIT_FLOOR 0.3 → 1.0,
    the v12 setting). c80's cold-from-scratch reframe (which re-armed the gate to 0.3 to damp the head-on
    merge rush) is disproven; the chosen path warm-adapts from BASELINE on the chase-t3-killgate ladder,
    which requires MATCHING BASELINE's reward — and BASELINE (c63) trained with f_pursuit ≡ 1. So the gate
    is neutralised at every aspect: the dense nose-on tracking term and the closing PBRS are UNDAMPED
    (the CHASE — pursuit turn back onto a fleeing/high-aspect bandit — pays fully, the v12 raison d'être).
    Verifies: (1) PURSUIT_FLOOR==1.0; (2) f_pursuit ≡ 1.0 across the whole hemisphere (the no-op); (3) the
    K_TRACK component carries the f_pursuit(≡1) factor and K_SOLUTION is ungated; (4) the closing PBRS is
    SYMMETRIC — a clean telescoping potential, so a close-then-open cycle nets exactly zero.
    The PURSUIT_GATE_* machinery stays in place (documented) so the gate can be re-armed without a code
    change if a raw beam/merge rush re-emerges."""
    from math import radians

    from astrum_dogfight.geometry import relative

    assert C.PURSUIT_FLOOR == 1.0, "curriculum-81 warm-adapt: the gate is a no-op again — PURSUIT_FLOOR must be 1.0"

    def fp(aspect_rad: float) -> float:
        if aspect_rad <= C.PURSUIT_GATE_LO:
            return 1.0
        if aspect_rad >= C.PURSUIT_GATE_HI:
            return C.PURSUIT_FLOOR
        frac = (aspect_rad - C.PURSUIT_GATE_LO) / (C.PURSUIT_GATE_HI - C.PURSUIT_GATE_LO)
        return 1.0 - (1.0 - C.PURSUIT_FLOOR) * frac

    # Bandit always due north of me (LOS = +y) so its heading psi EQUALS the aspect angle; my nose
    # is on it (psi=0 → f_nose ≈ 1), isolating the gate on the nose-on tracking term.
    me = _state(1, pos=[0, 0, 4500], psi=0.0)
    at = lambda d, aspect: _state(2, pos=[0, d, 4500], psi=radians(aspect))

    # (1)/(2) f_pursuit ≡ 1.0 at every aspect (the v12 no-op) — matches the generic interp formula (which
    # collapses to 1.0 when the floor is 1.0), and there is no damping anywhere in the hemisphere.
    for aspect in (0.0, 50.0, 75.0, 90.0, 150.0):
        b = at(800, aspect)
        rg = relative(me, b)
        out = R.compute(me, b, me, b)
        assert abs(fp(rg.aspect) - 1.0) < 1e-12, ("floor==1 ⇒ fp≡1", aspect)
        assert abs(out.info["f_pursuit"] - 1.0) < 1e-9, (aspect, out.info)

    # (3) Nose-on tracking carries the f_pursuit(≡1) factor; the K_SOLUTION term is NOT gated.
    for aspect in (0.0, 90.0, 150.0):
        b = at(800, aspect)
        rg = relative(me, b)
        f_nose = (1.0 + np.cos(rg.ata)) * 0.5
        f_range = max(0.0, (C.TRACK_RANGE - rg.rng) / C.TRACK_RANGE)
        expect = C.K_TRACK * f_nose * f_range * 1.0 + C.K_SOLUTION * rg.gun_quality
        out = R.compute(me, b, me, b)
        assert abs(out.info["track_reward"] - expect) < 1e-6, (aspect, out.info, expect)

    # (4) Closing PBRS is SYMMETRIC (undamped): close-toward and opening are equal-and-opposite, so a
    # close-then-open cycle nets exactly zero — the telescoping-potential invariant with no aspect gate.
    k = C.K_CLOSE / C.TRACK_RANGE
    out_close = R.compute(me, at(900, 90.0), me, at(700, 90.0))  # closing 900→700
    assert abs(out_close.info["close_reward"] - k * 200.0) < 1e-6, out_close.info
    out_open = R.compute(me, at(700, 90.0), me, at(900, 90.0))  # opening 700→900
    assert abs(out_open.info["close_reward"] + k * 200.0) < 1e-6, out_open.info
    assert abs(out_close.info["close_reward"] + out_open.info["close_reward"]) < 1e-9, "symmetric: cycle nets 0"
    print("  pursuit-gate OK — curriculum-81 warm-adapt: gate is a no-op (floor 1.0), chase undamped everywhere")


def test_timeout_penalty():
    """reward-v7 + reward-v13: a clock timeout (no kill/death, inside the bubble) is tagged
    "timeout" and, on a kill rung (timeout_penalty<0), penalised on the FINAL step so an unfinished
    engagement is EV-negative. reward-v13 FLATTENS the reward-v12 engagement split — the c35 policy
    exploited it by spraying to clear PASSIVE_DRAW_HP for the milder engaged draw instead of committing
    to the kill — so PASSIVE_TIMEOUT_MULT=1.0 now makes EVERY no-kill draw carry the SAME flat penalty
    regardless of chip damage (a draw is a failure, the kill is the goal; the dense damage term already
    credits a real chase on the way in). The passive_draw FLAG is still surfaced (damage < threshold),
    only its extra multiplier is gone. A survive rung (timeout_penalty=0) tags the timeout but is NOT
    penalised. Uses a short 2 s clock so the level/no-fire learner runs it out; the engaged branch
    pre-seeds the episode HP-dealt to confirm the penalty no longer changes with engagement."""
    penalty = -40.0
    action = np.array([0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)  # level, no fire

    def _run_to_end(env, *, seed_damage=0.0):
        env.reset()
        env._damage_dealt = float(seed_damage)  # pre-seed HP-dealt (no fire ⇒ unchanged)
        r = 0.0
        term = trunc = False
        info: dict = {}
        for _ in range(4000):
            _, r, term, trunc, info = env.step(action)
            if term or trunc:
                break
        return r, term, trunc, info

    # reward-v13: MULT=1.0 ⇒ the passive draw carries exactly the flat base penalty (no doubling).
    assert C.PASSIVE_TIMEOUT_MULT == 1.0, "reward-v13 flattens the engagement split"

    # Kill rung, PASSIVE draw (0 HP dealt): tagged passive_draw=True, penalised the flat `penalty`.
    env = make_tier1_env(separation_m=1500.0, max_time_s=2.0, timeout_penalty=penalty)
    r, term, trunc, info = _run_to_end(env)
    assert trunc and not term, f"a passive 2 s rollout must clock-timeout, not terminate: {info}"
    assert info.get("event") == "timeout" and info.get("passive_draw") is True, info
    assert r <= penalty + 1.0, f"passive draw {r:.3f} must carry the flat {penalty} penalty"

    # Kill rung, ENGAGED draw (pre-seed ≥ PASSIVE_DRAW_HP): tagged passive_draw=False but the penalty
    # is now the SAME flat value (reward-v13 removed the discount that the c35 policy sprayed to earn).
    env_e = make_tier1_env(separation_m=1500.0, max_time_s=2.0, timeout_penalty=penalty)
    r_e, _, trunc_e, info_e = _run_to_end(env_e, seed_damage=C.PASSIVE_DRAW_HP + 5.0)
    assert trunc_e and info_e.get("event") == "timeout" and info_e.get("passive_draw") is False, info_e
    assert r_e <= penalty + 1.0, f"engaged draw {r_e:.3f} must carry the flat {penalty} penalty"
    assert abs(r_e - r) < 1.0, (
        f"reward-v13: engaged {r_e:.3f} and passive {r:.3f} draws must carry the SAME flat penalty"
    )

    # Survive rung: same timeout, tagged, but NOT penalized (with or without engagement).
    env0 = make_tier1_env(separation_m=1500.0, max_time_s=2.0, timeout_penalty=0.0)
    r0, term0, trunc0, info0 = _run_to_end(env0)
    assert trunc0 and not term0 and info0.get("event") == "timeout", info0
    assert r0 > penalty + 1.0, f"an unpenalized timeout reward {r0:.3f} must not carry the penalty"
    print(f"  timeout-penalty OK — reward-v13 flat draw {penalty:+.0f} (passive == engaged) / survive 0")


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
    prev_used = 0.0
    for _ in range(steps):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), "obs stayed finite"
        assert np.isfinite(rew), f"reward finite, got {rew}"
        # reward-v4 cap invariant: credited tracking is monotonic and never past budget.
        used = info["track_used"]
        assert prev_used - 1e-9 <= used <= C.TRACK_BUDGET + 1e-9, f"track_used {used} out of [0, {C.TRACK_BUDGET}]"
        prev_used = used
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


def test_track_budget_cap():
    """reward-v4 (§8.3): cumulative dense tracking is clamped at `C.TRACK_BUDGET` per
    episode, so parking nose-on can never out-earn a kill. Shrink the budget so a brief
    close pursuit saturates it, then verify `track_used` is monotonic and bounded, that
    it actually reaches the cap, and that the step reward is trimmed once the cap binds
    (a saturated step earns strictly less than it would from its raw tracking payoff)."""
    saved = C.TRACK_BUDGET
    C.TRACK_BUDGET = 0.05  # tiny — a nose-on 1.2 km start blows past it in a few steps
    try:
        env = make_tier1_env(separation_m=1200.0, max_time_s=30.0)
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.9, 0.0], dtype=np.float32)  # hold a mild pursuit
        prev_used, saturated, trimmed = 0.0, False, False
        for _ in range(240):
            _, rew, term, trunc, info = env.step(action)
            used = info["track_used"]
            assert prev_used - 1e-9 <= used <= C.TRACK_BUDGET + 1e-9, f"track_used {used} out of range"
            raw_track = info.get("track_reward", 0.0)
            # Once the budget is spent, this step's tracking is fully trimmed: the reward
            # must sit at or below what remained creditable this step (< the raw payoff).
            if used >= C.TRACK_BUDGET - 1e-9:
                saturated = True
                if raw_track > 1e-6 and (used - prev_used) < raw_track - 1e-9:
                    trimmed = True
            prev_used = used
            if term or trunc:
                break
        assert saturated, "a close nose-on pursuit should saturate the shrunken budget"
        assert trimmed, "tracking reward must be trimmed once the budget binds"
    finally:
        C.TRACK_BUDGET = saved
    print(f"  track-budget cap OK — cumulative tracking clamped at {C.TRACK_BUDGET}/episode")


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


def test_tier3_break_roll_gate():
    """The break must ROLL BEFORE IT PULLS.

    Holding max aft stick while rolling turns a level break into a zoom climb: near wings-level the
    whole max-g demand becomes pitch. Measured on the ungated law, that put the bandit into a
    γ +38° climbing spiral to 8261 m at the 16° AoA ceiling, bleeding 218 → 138 m/s, and left both
    jets α-saturated at ~150 m/s where neither can point a gun.
    """
    opp = Tier3Opponent()
    tgt = _state(2, pos=[600, 0, 4500], psi=radians(180.0))  # off to the east ⇒ commands +85° bank

    # Wings level and 85° away from the commanded bank ⇒ cos(bank_err) ≈ 0 ⇒ essentially no pull.
    level = _state(1, pos=[0, 0, 4500], psi=0.0)
    ail_level, elev_level, _, _ = opp._break(level, tgt)
    assert abs(elev_level) < 0.15, f"wings-level break must not pull hard, got elev {elev_level:+.2f}"
    assert ail_level > 0.5, f"...but it must be rolling hard, got ail {ail_level:+.2f}"

    # Rolled onto the commanded bank ⇒ cos(bank_err) ≈ 1 ⇒ full AoA-limit pull.
    rolled = replace(level, phi=C.BREAK_BANK_RAD)
    _, elev_rolled, _, _ = opp._break(rolled, tgt)
    assert elev_rolled < -0.9, f"a settled break must pull to the limit, got elev {elev_rolled:+.2f}"
    assert elev_rolled < elev_level, "the pull must grow with roll alignment, not shrink"

    # Mid-reversal (banked the WRONG way, >90° of bank error) must not pull either — max(cos, 0).
    wrong_way = replace(level, phi=-C.BREAK_BANK_RAD)
    _, elev_wrong, _, _ = opp._break(wrong_way, tgt)
    assert elev_wrong == 0.0, f"a >90° bank error must command zero pull, got {elev_wrong:+.2f}"
    print(
        f"  tier3 break roll-gate OK — pull {elev_level:+.2f} wings-level → "
        f"{elev_rolled:+.2f} rolled, {elev_wrong:+.2f} mid-reversal"
    )


def test_tier3_break_sign_latches():
    """The break's turn direction latches, and reverses only past a real deadband.

    The ungated law recomputed `sign` from `ata_az` every tick, so it chattered whenever the
    attacker sat near the nose — and each chatter was a full 85° → 0 → −85° roll reversal.
    """
    opp = Tier3Opponent()
    me = _state(1, pos=[0, 0, 4500], psi=0.0)

    east = _state(2, pos=[600, 0, 4500], psi=radians(180.0))
    opp._break(me, east)
    latched = opp._break_sign
    assert latched != 0.0, "the break must commit to a direction"

    # A threat a hair the other side of the nose is INSIDE the deadband ⇒ hold the latched turn.
    eps = C.BREAK_REVERSE_DEADBAND_RAD * 0.5
    barely = _state(2, pos=[-600.0 * sin(eps), 600.0 * cos(eps), 4500], psi=radians(180.0))
    opp._break(me, barely)
    assert opp._break_sign == latched, "a threat inside the deadband must not reverse the break"

    # Well past it ⇒ reverse, because a genuine cross-nose reversal is legitimate BFM.
    other = _state(2, pos=[-600, 0, 4500], psi=radians(180.0))
    opp._break(me, other)
    assert opp._break_sign == -latched, "a threat well across the nose must reverse the break"

    # `reset` must clear the latch, or direction leaks between episodes.
    opp.reset(me)
    assert opp._break_sign == 0.0, "reset must unlatch the break direction"
    print(
        f"  tier3 break sign-latch OK — held through {degrees(eps):.1f}°, reversed past "
        f"{degrees(C.BREAK_REVERSE_DEADBAND_RAD):.0f}°, cleared on reset"
    )


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


def _fly_expert(env, max_steps: int):
    """Drive `env` with the pursuit-expert action it surfaces in info; return the event."""
    obs, info = env.reset()
    assert "expert_action" in info and len(info["expert_action"]) == C.ACTION_DIM
    event = None
    for _ in range(max_steps):
        obs, rew, term, trunc, info = env.step(info["expert_action"])
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    return event


def test_pursuit_expert_kills_tier1_at_range():
    # The exploration guide (train_parallel's DAgger seed) has to actually press to guns and
    # gun a straight-and-level Tier-1 from a LONG-range start — exactly where the RL prior
    # stalls (it arrests at standoff, dealing a couple of HP). Validated capability (diag,
    # 2026-08-27): a clean kill from 1100 m and 1600 m. Assert that reliable range here; the
    # longer 2000 m start is covered by the closing-boundary test below (the straight dummy can
    # simply outrun the pursuit to the bubble edge, so a kill there is NOT guaranteed).
    for sep in (1100.0, 1600.0):
        env = make_tier1_env(separation_m=sep, max_time_s=60.0)
        event = _fly_expert(env, max_steps=7300)  # 60 s @ 120 Hz
        opp = env._states[env.opponent_id]
        assert event == "kill" and not opp.alive, (
            f"PursuitExpert should close from {int(sep)} m and gun the Tier-1 dummy, got {event!r}"
        )
    print("  pursuit-expert OK — closes from 1100 m and 1600 m and guns the Tier-1 dummy")


def test_pursuit_expert_closes_and_damages_from_2000m():
    # Honest boundary of the seed: from 2000 m the expert may not finish the kill before the
    # straight-and-level dummy runs both jets out to the engagement-bubble edge, but it MUST
    # still demonstrate the long-range closing behavior the seed exists for — press inside the
    # gun's effective range and put real rounds on the bandit. (A kill is asserted only at the
    # ≤1600 m ranges above; here we guard the closing skill, not the finish.)
    env = make_tier1_env(separation_m=2000.0, max_time_s=60.0)
    event = _fly_expert(env, max_steps=7300)
    opp = env._states[env.opponent_id]
    dmg = C.HP_MAX - opp.health
    assert dmg >= 25.0, f"expert should close from 2000 m and deal real damage, dealt only {dmg:.0f} HP"
    print(f"  pursuit-expert OK — closes from 2000 m and deals {dmg:.0f} HP (event {event!r})")


def test_scenario_jitter_zero_is_legacy():
    # jitter=0 (the default) must reproduce the fixed legacy geometry byte-for-byte so every
    # existing scenario/test is unchanged; jitter>0 with no rng is inert (safe fallback).
    from astrum_dogfight.scenario import tier1_scenario, tier3_scenario

    assert tier1_scenario(separation_m=800.0, jitter=0.0) == tier1_scenario(separation_m=800.0)
    assert tier3_scenario(separation_m=700.0, jitter=0.0) == tier3_scenario(separation_m=700.0)
    assert tier1_scenario(separation_m=800.0, jitter=0.6, rng=None) == tier1_scenario(
        separation_m=800.0
    )
    print("  scenario jitter OK — jitter=0 / rng=None reproduce the legacy fixed geometry")


def test_scenario_jitter_varies_and_is_reproducible():
    # A seeded RNG makes each episode's start differ (generalisation pressure) yet a run stays
    # reproducible: the same seed replays the identical sequence of jittered scenarios.
    from astrum_dogfight.scenario import tier1_scenario

    rng = np.random.default_rng(0)
    a = tier1_scenario(separation_m=800.0, jitter=0.6, rng=rng)
    b = tier1_scenario(separation_m=800.0, jitter=0.6, rng=rng)
    assert a != b, "consecutive jittered draws should differ"
    rng2 = np.random.default_rng(0)
    assert tier1_scenario(separation_m=800.0, jitter=0.6, rng=rng2) == a, "seed must reproduce"
    print("  scenario jitter OK — varies per draw, reproducible per seed")


def test_block_altitude_band_samples_within_range_and_stays_legacy_by_default():
    # curriculum-91: `alt_ft_max` turns the fixed BLOCK altitude into a per-episode band, because the
    # 15,000 ft default is only ~63 s of altitude against a 180 s merge clock (a head-on energy fight
    # bleeds ~68 m/s) and 11/12 greedy episodes ended hard_deck at a mean 86 s. A BAND rather than a
    # fixed raise because obs feature 2 is (agl − DECK_AGL)/ALT_REF: 15,000 ft reads 0.81, 30,000 ft
    # reads 1.58, so a straight jump lands past 1.0 in a range no checkpoint has trained on.
    import re

    from astrum_dogfight.scenario import tier3_scenario

    def alts(toml):
        return [float(m) for m in re.findall(r"alt_ft\s*=\s*([\d.]+)", toml)]

    # Default (None) is byte-for-byte legacy, and the band is inert without an rng.
    assert tier3_scenario(separation_m=700.0, alt_ft_max=None) == tier3_scenario(separation_m=700.0)
    assert tier3_scenario(separation_m=700.0, alt_ft_max=30000.0) == tier3_scenario(separation_m=700.0)

    rng = np.random.default_rng(0)
    draws = [
        alts(tier3_scenario(separation_m=3000.0, jitter=0.5, rng=rng, alt_ft=15000.0, alt_ft_max=30000.0))
        for _ in range(40)
    ]
    # Both jets share the block altitude, so every altitude in a draw sits inside the band (the
    # opponent's ±240·jitter ft altitude jitter is the only spread, hence the small tolerance).
    flat = [a for d in draws for a in d]
    assert min(flat) >= 15000.0 - 200.0, min(flat)
    assert max(flat) <= 30000.0 + 200.0, max(flat)
    # It must actually SPAN the band, not cluster at one end — that is the whole anti-OOD point.
    lows = [d[0] for d in draws]
    assert max(lows) - min(lows) > 8000.0, "band should cover a wide spread of block altitudes"
    # Reversed endpoints are sorted, not silently empty.
    rng2 = np.random.default_rng(1)
    rev = alts(tier3_scenario(separation_m=3000.0, jitter=0.5, rng=rng2, alt_ft=30000.0, alt_ft_max=15000.0))
    assert 15000.0 - 200.0 <= rev[0] <= 30000.0 + 200.0
    print("  block-altitude band OK — spans [lo,hi] under rng, legacy-identical without one")


def test_block_altitude_band_composes_with_the_perch():
    # The perch (me_dalt_ft) is an EDGE over the bandit and must ride on top of whatever block
    # altitude the band drew — the two knobs are independent, not alternatives.
    import re

    from astrum_dogfight.scenario import tier3_scenario

    rng = np.random.default_rng(7)
    toml = tier3_scenario(
        separation_m=3000.0, jitter=0.5, rng=rng, alt_ft=15000.0, alt_ft_max=30000.0, me_dalt_ft=2000.0
    )
    learner, opponent = (float(m) for m in re.findall(r"alt_ft\s*=\s*([\d.]+)", toml))
    assert learner - opponent > 2000.0 - 300.0, (learner, opponent)
    print("  block-altitude band OK — perch still stacks on the sampled block altitude")


def _merge_lateral_m(toml):
    """Recover the bandit's cross-range offset, in metres, from a head-on merge scenario TOML.

    The learner sits at the origin heading north and the bandit is placed at bearing `aspect_deg`,
    so with `aspect_deg=0` the whole east displacement IS the lateral offset.
    """
    import re

    from astrum_dogfight.scenario import _M_PER_DEG_LON, _ORIGIN_LON

    # The lookbehind is load-bearing: a bare `lon_deg` also matches `origin_lon_deg` in [world].
    lons = [float(m) for m in re.findall(r"(?<!origin_)lon_deg\s*=\s*(-?[\d.]+)", toml)]
    learner, opponent = lons  # the learner is written first, always at the origin longitude
    assert learner == _ORIGIN_LON, learner
    return (opponent - _ORIGIN_LON) * _M_PER_DEG_LON


def test_merge_lateral_band_is_signed_spans_zero_and_stays_legacy_by_default():
    # curriculum-92: `lateral_m_max` is the merge's LATERAL SEPARATION — the turning room a head-on
    # merge never had. The only cross-range term was the jitter wobble, ±0.09·jitter·separation =
    # ±135 m at jitter 0.5 / 3000 m = 2.6° off the nose; against a ~634 m F-16 turn radius that is
    # 0.21 radii, i.e. a collision course in which no lead turn and no one-circle/two-circle
    # decision exist. Measured consequence (c90@5M): 14.2 s/ep inside 500 m at ata 128°, aspect
    # 147°, closure ≈0, and 0.0% of 4100 ticks hittable OR feasible.
    from astrum_dogfight.scenario import tier3_scenario

    kw = dict(separation_m=3000.0, aspect_deg=0.0, opp_heading_deg=180.0)
    # Default is byte-for-byte legacy, and the band is inert without an rng (the deterministic
    # eval/probe path must stay repeatable — it probes a fixed offset via aspect_deg instead).
    assert tier3_scenario(**kw, lateral_m_max=0.0) == tier3_scenario(**kw)
    assert tier3_scenario(**kw, lateral_m_max=1200.0) == tier3_scenario(**kw)

    rng = np.random.default_rng(0)
    draws = [
        _merge_lateral_m(tier3_scenario(**kw, jitter=0.5, rng=rng, lateral_m_max=1200.0))
        for _ in range(60)
    ]
    # Bounded by the band half-width plus the pre-existing wobble (±0.09·0.5·3000 = ±135 m). The
    # ceiling matters: past √((THREAT_RANGE·sin THREAT_CONE_RAD)² − perch²) = 1659 m at a 2000 ft
    # perch the T3 stops going defensive and the benchmark has silently moved.
    assert max(abs(d) for d in draws) <= 1200.0 + 135.0 + 1.0, max(draws)
    # SIGNED and spanning zero. Every rung in c01→c91 spawned the bandit to the RIGHT, so a
    # one-sided band would let the policy hard-code "break right" and score a fake win; and the
    # band must still cover today's familiar ±135 m (the band-not-step pattern that worked for the
    # aspect band, and that the fixed --me-dalt-ft/--merge-alt-ft jumps skipped — both backfired).
    assert min(draws) < -600.0 and max(draws) > 600.0, (min(draws), max(draws))
    assert any(abs(d) < 200.0 for d in draws), "band must still replay the legacy collision course"
    # It must SPAN, not cluster at the edges — a bimodal ±L would be a step change in disguise.
    mid = sum(1 for d in draws if 200.0 <= abs(d) <= 1000.0)
    assert mid >= len(draws) // 4, f"only {mid}/{len(draws)} draws in the interior of the band"


def test_merge_lateral_band_is_cross_range_only_and_preserves_the_rng_stream():
    # Two properties that make this a genuinely single-variable experiment.
    from astrum_dogfight.scenario import tier3_scenario

    import re

    kw = dict(separation_m=3000.0, aspect_deg=0.0, opp_heading_deg=180.0, jitter=0.5)

    def lat_lon(toml):
        lats = [float(m) for m in re.findall(r"lat_deg\s*=\s*(-?[\d.]+)", toml)]
        return lats[1]

    # (1) The offset rides on the EAST axis only, so the along-track closing distance — and
    # therefore the run-in time to the pass — is untouched. That is what makes cross-range the
    # single changed quantity rather than "offset AND a shorter merge".
    a = tier3_scenario(**kw, rng=np.random.default_rng(3))
    b = tier3_scenario(**kw, rng=np.random.default_rng(3), lateral_m_max=1200.0)
    assert lat_lon(a) == lat_lon(b), "north displacement must not move with the lateral band"
    assert abs(_merge_lateral_m(a) - _merge_lateral_m(b)) > 1.0, "east displacement must move"

    # (2) The band draws LAST, after every pre-existing jitter draw, so with the knob OFF the rng
    # stream is identical — seed-for-seed a c92 episode differs from the c90 episode only by the
    # offset. Altitudes and airspeeds are drawn before it and must therefore match exactly.
    for field in ("alt_ft", "airspeed_kts"):
        va = re.findall(rf"{field}\s*=\s*([\d.]+)", a)
        vb = re.findall(rf"{field}\s*=\s*([\d.]+)", b)
        assert va == vb, f"{field} changed ⇒ the lateral draw perturbed an earlier rng draw"


def test_env_jitter_reset_varies_and_seed_reproduces():
    # The env-level contract the training gate relies on: with jitter, each reset() rebuilds a
    # perturbed engagement (so a deterministic policy can't memorise one start), an explicit
    # reset(seed=) reproduces a specific engagement (for controlled eval), and the jitter=0
    # factory path is the unchanged fixed no-op geometry.
    env = make_tier1_env(separation_m=800.0, max_time_s=5.0, jitter=0.6, jitter_seed=3)
    o1, _ = env.reset()
    o2, _ = env.reset()
    assert not np.allclose(o1, o2), "per-episode start geometry should vary under jitter"
    a, _ = env.reset(seed=11)
    b, _ = env.reset(seed=11)
    assert np.allclose(a, b), "reset(seed=) must reproduce the same engagement"
    env.close()

    env0 = make_tier1_env(separation_m=800.0, max_time_s=5.0)  # jitter=0 default
    c, _ = env0.reset()
    d, _ = env0.reset()
    assert np.allclose(c, d), "no-jitter reset must be the fixed legacy geometry"
    env0.close()
    print("  env jitter OK — reset() varies, reset(seed=) reproduces, jitter=0 stays fixed")


def main():
    test_geometry()
    test_obs_append_arrival_geometry()
    test_corrected_lay_and_scripted_invariance()
    test_reward()
    test_fire_burst_ramp()
    test_p_hit_range_is_the_measured_curve()
    test_lethal_range_gates_the_wez_bonus()
    test_r_eff_is_the_opponent_benchmark_not_the_lethal_envelope()
    test_spiral_penalty()
    test_aspect_conversion_reward()
    test_conversion_terminal()
    test_closing_reward()
    test_close_crossing_gate()
    test_aim_reward()
    test_lead_reward()
    test_sigma_track_honest_gun_quality()
    test_band_reward()
    test_overspeed_penalty()
    test_wound_amplifier()
    test_pursuit_gate()
    test_timeout_penalty()
    test_live_rollout()
    test_track_budget_cap()
    test_tier2_break()
    test_tier2_split_s()
    test_tier3_fire_gate()
    test_tier3_control_laws()
    test_tier3_break_roll_gate()
    test_tier3_break_sign_latches()
    test_tier3_offensive()
    test_tier3_defensive()
    test_pursuit_expert_kills_tier1_at_range()
    test_pursuit_expert_closes_and_damages_from_2000m()
    test_scenario_jitter_zero_is_legacy()
    test_scenario_jitter_varies_and_is_reproducible()
    test_env_jitter_reset_varies_and_seed_reproduces()
    print(
        "\nOK — astrum_dogfight Tier-1/2/3 adapter: geometry, reward, rollout, "
        "break, split-s, fire-gate, FSM offense+defense, pursuit-expert, jitter all pass."
    )


if __name__ == "__main__":
    main()
