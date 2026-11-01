"""Tests for the `astrum_dogfight` Gym adapter (design doc §8.2–§8.5).

Fabricated `AState`s exercise geometry / reward / the fire gate with no live
sim; live rollouts exercise the scripted opponents, seeding and the agreement
between the Python features and the sim's own exports.

    PYTHONPATH="$(bash scripts/build-astrum-env.sh):crates/astrum-env/python" \
        pytest -q crates/astrum-env/python
"""

from math import atan2, cos, degrees, radians, sin

import astrum_env
import numpy as np
import pytest
from astrum_dogfight import constants as C
from astrum_dogfight import reward as R
from astrum_dogfight.env import make_tier1_env, make_tier2_env, make_tier3_env
from astrum_dogfight.geometry import (
    AState,
    GunSolution,
    body_bearing,
    build_observation,
    potential,
    relative,
)
from astrum_dogfight.opponent import Tier3Opponent, wez_fire
from astrum_dogfight.scenario import tier1_scenario

# Hand-built body axes (rows: nose, starboard, down) in grid ENU.
LEVEL_NORTH = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
LEVEL_SOUTH = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
_P = radians(30.0)
PITCHED_UP_30_NORTH = [[0, cos(_P), sin(_P)], [1, 0, 0], [0, sin(_P), -cos(_P)]]
ROLLED_RIGHT_90_NORTH = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
CRUISE = np.array([0.0, 180.0, 0.0])
NO_GUN = None
LEARNER = np.array([0.0, 0.0, 0.0, 0.9, 0.0])


def _state(axes=LEVEL_NORTH, *, pos=(0.0, 0.0, 4500.0), vel=CRUISE, **cols):
    e, n, u = pos
    ve, vn, vu = vel
    cols = {"tas": float(np.linalg.norm(vel)), "nz": 1.0, "alt_hae_m": u, **cols}
    return AState.fabricate(
        axes, pos_e=e, pos_n=n, pos_u=u, vel_e=ve, vel_n=vn, vel_u=vu, vel_ned_n=vn,
        vel_ned_e=ve, vel_ned_d=-vu, **cols,
    )  # fmt: skip


def _gun(*, aim_error, in_range=True, hit=False):
    return GunSolution(
        miss_m=0.0, tof_s=0.5, hit=hit, aim_dir=np.array([0.0, 1.0, 0.0]),
        aim_error_rad=aim_error, in_range=in_range,
    )  # fmt: skip


def _sweep_deg(a: float, b: float) -> float:
    return abs(degrees(astrum_env.wrap_pi(a - b)))


# --- geometry -------------------------------------------------------------


def test_geometry_level():
    me = _state(pos=(0, 0, 4500))
    bandit = _state(pos=(0, 400, 4500))
    rg = relative(me, bandit, _gun(aim_error=0.0))
    assert abs(rg.rng - 400.0) < 1e-9
    assert degrees(rg.ata) < 1e-6, "nose-on bandit ⇒ 0 ATA"
    assert degrees(rg.aspect) < 1e-6, "at his six ⇒ 0 aspect"
    assert rg.in_range and rg.gun_quality == pytest.approx(1.0)
    assert potential(me, bandit, rg) > 0.8

    behind = _state(pos=(0, -800, 4500))
    assert degrees(relative(me, behind, NO_GUN).ata) == pytest.approx(180.0)

    fast = _state(vel=(0, 250, 0))
    slow_ahead = _state(pos=(0, 1000, 4500), vel=(0, 100, 0))
    assert relative(fast, slow_ahead, NO_GUN).closure == pytest.approx(150.0)

    head_on = _state(LEVEL_SOUTH, pos=(0, 1000, 4500), vel=(0, -180, 0))
    assert abs(degrees(relative(me, head_on, NO_GUN).hca)) == pytest.approx(180.0)
    assert degrees(relative(me, head_on, NO_GUN).aspect) == pytest.approx(180.0)

    obs = build_observation(me, bandit, rg)
    assert obs.shape == (C.OBS_DIM,) and np.all(np.isfinite(obs))


def test_geometry_pitched():
    me = _state(PITCHED_UP_30_NORTH)
    level_ahead = _state(pos=(0, 1000, 4500))
    rg = relative(me, level_ahead, NO_GUN)
    assert degrees(rg.ata) == pytest.approx(30.0)
    assert degrees(rg.ata_el) == pytest.approx(-30.0), "level target sits below a raised nose"
    assert degrees(rg.ata_az) == pytest.approx(0.0, abs=1e-9)
    along_nose = _state(pos=(0, 1000 * cos(_P), 4500 + 1000 * sin(_P)))
    assert degrees(relative(me, along_nose, NO_GUN).ata) == pytest.approx(0.0, abs=1e-6)
    off, bank_err, _ = body_bearing(me, [0.0, 1.0, 0.0])
    assert degrees(off) == pytest.approx(30.0)
    assert abs(degrees(bank_err)) == pytest.approx(180.0), "below the nose ⇒ roll inverted"


def test_geometry_rolled():
    me = _state(ROLLED_RIGHT_90_NORTH)
    east = _state(pos=(1000, 0, 4500))
    rg = relative(me, east, NO_GUN)
    assert degrees(rg.ata) == pytest.approx(90.0)
    assert degrees(rg.ata_el) == pytest.approx(90.0), "rolled right, the east target is overhead"
    _, bank_err, _ = body_bearing(me, [1.0, 0.0, 0.0])
    assert degrees(bank_err) == pytest.approx(0.0, abs=1e-9), "lift vector already on it"
    _, bank_err_west, _ = body_bearing(me, [-1.0, 0.0, 0.0])
    assert abs(degrees(bank_err_west)) == pytest.approx(180.0)
    obs = build_observation(me, east, rg)
    assert obs[2] == pytest.approx(0.0) and obs[3] == pytest.approx(1.0), (
        "fabricated row has phi=0; sin/cos(phi) come from the row, not the axes"
    )


def test_python_features_agree_with_sim():
    """Attitude: body axes from `body_to_grid` vs the Euler angles in the obs row
    (nose·up = sinθ, starboard·up = −sinφ·cosθ, nose bearing = ψ). Velocity: the
    grid columns vs local NED (the local-level tilt is negligible near the origin).
    Gun: `aim_error_rad` vs the angle between the nose and `aim_dir`."""
    env = make_tier3_env(posture="offensive", max_time_s=20.0)
    env.reset(seed=1)
    roll_pull = np.array([0.6, -0.4, 0.0, 1.0, 0.0])
    checked = 0
    for k in range(400):
        env.step(roll_pull)
        if k % 50 != 49:
            continue
        me = env.aircraft(env.learner_id)
        nose, starboard, _ = me.axes
        assert np.allclose(me.axes @ me.axes.T, np.eye(3), atol=1e-9), "orthonormal axes"
        assert nose[2] == pytest.approx(sin(me.theta), abs=2e-3)
        assert starboard[2] == pytest.approx(-sin(me.phi) * cos(me.theta), abs=2e-3)
        if abs(cos(me.theta)) > 0.2:
            assert _sweep_deg(atan2(nose[0], nose[1]), me.psi) < 0.5
        ned = np.array([me.vel_ned_e, me.vel_ned_n, -me.vel_ned_d])
        assert np.allclose(me.vel, ned, atol=0.05 * np.linalg.norm(ned))
        checked += 1
    me = env.aircraft(env.learner_id)
    assert abs(me.phi) > radians(5) and abs(me.theta) > radians(2), "attitude exercised"
    sol = env.sim.gun_solution(env.learner_id, env.opponent_id)
    angle = np.arccos(np.clip(np.dot(me.nose, sol["aim_dir"]), -1, 1))
    assert angle == pytest.approx(sol["aim_error_rad"], abs=radians(1.0))
    assert checked == 8


# --- reward ---------------------------------------------------------------


def _reward(prev_me, prev_bandit, me, bandit):
    return R.compute(
        prev_me, prev_bandit, me, bandit,
        relative(prev_me, prev_bandit, NO_GUN), relative(me, bandit, NO_GUN),
    )  # fmt: skip


def test_reward():
    me = _state()
    bandit = _state(pos=(0, 1000, 4500))

    out = _reward(me, bandit, me, _state(pos=(0, 1000, 4500), hit_points=98.0))
    assert 0.9 < out.reward < 1.1 and not out.terminated, out

    dead_bandit = _state(pos=(0, 1000, 4500), alive=0.0, hit_points=0.0)
    out = _reward(me, bandit, me, dead_bandit)
    assert out.terminated and out.reward > 90.0 and out.info["event"] == "kill", out

    dead_me = _state(alive=0.0, hit_points=0.0)
    out = _reward(me, bandit, dead_me, bandit)
    assert out.terminated and out.reward < -90.0 and out.info["event"] == "killed", out

    low = _state(pos=(0, 0, C.DECK_HAE_M - 1.0))
    out = _reward(me, bandit, low, bandit)
    assert out.terminated and out.reward < -190.0 and out.info["event"] == "hard_deck", out
    assert not R.below_hard_deck(_state(pos=(0, 0, C.DECK_HAE_M + 1.0)))

    far = _state(pos=(C.R_BUBBLE + 100.0, 0, 4500))
    out = _reward(me, bandit, far, bandit)
    assert out.truncated and not out.terminated and out.reward < -49.0, out
    high = _state(pos=(0, 0, C.CEILING_HAE_M + 1.0))
    assert R.outside_bubble(high)


# --- fire gate / Tier-3 laws (fabricated) ----------------------------------


def test_wez_fire_gate():
    me = _state()
    tgt = _state(pos=(0, 500, 4500))
    assert wez_fire(me, tgt, _gun(aim_error=radians(1.0)))
    assert not wez_fire(me, tgt, _gun(aim_error=radians(1.0), in_range=False))
    assert not wez_fire(me, tgt, _gun(aim_error=radians(10.0)))
    assert wez_fire(me, tgt, _gun(aim_error=radians(10.0), hit=True)), "a predicted hit fires"
    assert not wez_fire(_state(ammo=0.0), tgt, _gun(aim_error=0.0)), "no ammo"
    assert not wez_fire(me, _state(alive=0.0), _gun(aim_error=0.0)), "dead target"
    assert not wez_fire(me, tgt, NO_GUN), "unarmed"


@pytest.mark.parametrize("dt", [1.0 / 120.0, 1.0 / 50.0, 0.1])
def test_tier3_dwell_is_time_based(dt):
    me = _state()
    tgt = _state(pos=(0, 1500, 4500))  # ahead, nose-on, within OFF_RANGE
    opp = Tier3Opponent()
    opp.reset(me, dt)
    ticks = 0
    while opp.state == "neutral":
        opp.act(me, tgt, NO_GUN)
        ticks += 1
        assert ticks < 10_000
    assert opp.state == "offensive"
    assert ticks * dt == pytest.approx(C.FSM_MIN_DWELL_S, abs=dt + 1e-9)


@pytest.mark.parametrize(
    "tgt",
    [
        _state(pos=(0, 600, 4500)),  # ahead: pursuit
        _state(pos=(0, -600, 4500)),  # nose-on threat behind: break
        _state(pos=(800, 300, 5200)),  # off-axis, above
    ],
)
@pytest.mark.parametrize("axes", [LEVEL_NORTH, PITCHED_UP_30_NORTH, ROLLED_RIGHT_90_NORTH])
def test_tier3_actions_bounded(tgt, axes):
    opp = Tier3Opponent()
    me = _state(axes, phi=radians(20.0), theta=radians(10.0), p=0.5)
    opp.reset(me, 1.0 / 120.0)
    for _ in range(120):
        a = opp.act(me, tgt, _gun(aim_error=radians(2.0)))
        assert len(a) == 5 and all(np.isfinite(x) for x in a)
        assert all(-1.0 <= x <= 1.0 for x in a[:3]) and 0.0 <= a[3] <= 1.0
        assert a[4] in (0.0, 1.0)


# --- live rollouts ----------------------------------------------------------


def _rollout(env, steps, *, seed, action=LEARNER):
    env.reset(seed=seed)
    for _ in range(steps):
        obs, rew, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(rew)
        yield env, info
        if terminated or truncated:
            break


def test_seeded_rollouts_are_identical():
    def run(seed):
        env = make_tier3_env(posture="offensive", max_time_s=10.0)
        out = []
        for e, _ in _rollout(env, 600, seed=seed):
            me, opp = e.aircraft(e.learner_id), e.aircraft(e.opponent_id)
            out.append(np.concatenate([me.row, opp.row]))
        return np.array(out)

    a, b = run(42), run(42)
    assert a.shape == b.shape and np.array_equal(a, b), "same seed ⇒ identical rollout"


def test_reset_changes_scenario_in_place():
    env = make_tier1_env(max_time_s=5.0)
    env.reset(seed=0)
    near = env.aircraft(env.opponent_id).pos - env.aircraft(env.learner_id).pos
    _, info = env.reset(seed=0, options={"scenario_toml": tier1_scenario(separation_m=3500.0)})
    far = env.aircraft(env.opponent_id).pos - env.aircraft(env.learner_id).pos
    assert np.linalg.norm(near) == pytest.approx(2000.0, rel=0.02)
    assert np.linalg.norm(far) == pytest.approx(3500.0, rel=0.02)
    assert isinstance(info["sim_seed"], int)


def test_tier1_holds_straight_and_level():
    env = make_tier1_env(max_time_s=30.0)
    env.reset(seed=0)
    opp0 = env.aircraft(env.opponent_id)
    hdg0, alt0, tas0 = opp0.course_local(), opp0.alt_hae_m, opp0.tas
    for _ in _rollout(env, 1200, seed=0):
        pass
    opp = env.aircraft(env.opponent_id)
    assert _sweep_deg(opp.course_local(), hdg0) < 3.0
    assert abs(opp.alt_hae_m - alt0) < 60.0
    assert abs(opp.tas - tas0) < 10.0
    assert env.aircraft(env.learner_id).alive


def test_tier2_break():
    env = make_tier2_env(maneuver="break", max_time_s=40.0)
    env.reset(seed=0)
    hdg0 = env.aircraft(env.opponent_id).course_local()
    max_sweep = 0.0
    for e, _ in _rollout(env, 1600, seed=0):
        opp = e.aircraft(e.opponent_id)
        max_sweep = max(max_sweep, _sweep_deg(opp.course_local(), hdg0))
    opp = env.aircraft(env.opponent_id)
    assert max_sweep > 60.0, f"a hard break swings the course, got {max_sweep:.0f}°"
    assert opp.alive and abs(opp.nz) < 12.0


def test_tier2_split_s():
    env = make_tier2_env(maneuver="split_s", max_time_s=40.0)
    env.reset(seed=0)
    opp0 = env.aircraft(env.opponent_id)
    hdg0, alt0 = opp0.course_local(), opp0.alt_hae_m
    max_reversal, min_alt = 0.0, alt0
    for e, _ in _rollout(env, 1800, seed=0):
        opp = e.aircraft(e.opponent_id)
        max_reversal = max(max_reversal, _sweep_deg(opp.course_local(), hdg0))
        min_alt = min(min_alt, opp.alt_hae_m)
    assert max_reversal > 120.0, f"a Split-S reverses course, got {max_reversal:.0f}°"
    assert alt0 - min_alt > 300.0, f"a Split-S trades altitude, lost {alt0 - min_alt:.0f} m"


def test_tier3_offensive_guns_learner():
    env = make_tier3_env(posture="offensive", separation_m=600.0, max_time_s=25.0)
    saw_offensive, min_hp = False, 100.0
    for e, _ in _rollout(env, 3000, seed=0):
        saw_offensive = saw_offensive or e.opponent.state == "offensive"
        min_hp = min(min_hp, e.aircraft(e.learner_id).hit_points)
    assert saw_offensive
    assert min_hp < 100.0, f"the bot's guns should draw blood, learner HP stayed {min_hp:.0f}"
    assert env.aircraft(env.opponent_id).ammo < astrum_env.GUN_AMMO


def test_tier3_defensive_breaks():
    env = make_tier3_env(posture="defensive", max_time_s=25.0)
    env.reset(seed=0)
    hdg0 = env.aircraft(env.opponent_id).course_local()
    saw_defensive, max_sweep = False, 0.0
    for e, _ in _rollout(env, 2000, seed=0):
        saw_defensive = saw_defensive or e.opponent.state == "defensive"
        max_sweep = max(max_sweep, _sweep_deg(e.aircraft(e.opponent_id).course_local(), hdg0))
    opp = env.aircraft(env.opponent_id)
    assert saw_defensive
    assert max_sweep > 45.0, f"a defensive break swings the course, got {max_sweep:.0f}°"
    assert opp.alive and abs(opp.nz) < 12.0
