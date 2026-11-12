import math

from astrum_client.hold import HoldLaw
from astrum_client.runtime import AttitudeLimits, Reference
from astrum_client.wire import RawActuators

BANK, PITCH = math.radians(25.0), math.radians(10.0)
LIMITS = AttitudeLimits(25.0, 10.0)


def law(heading_deg=0.0, alt=1500.0, speed=50.0):
    return HoldLaw(Reference(math.radians(heading_deg), alt, speed), LIMITS)


def step(hold, make_input, state) -> RawActuators:
    action = hold.step(make_input(state))
    assert not action.trigger
    return action.control


def test_level_on_reference_is_near_trim(make_state, make_input):
    u = step(law(), make_input, make_state())
    assert abs(u.aileron) < 1e-9 and abs(u.elevator) < 1e-9
    assert u.throttle == HoldLaw.THROTTLE_TRIM


def test_heading_right_rolls_right_and_climb_pitches_up(make_state, make_input):
    u = step(law(heading_deg=20.0, alt=1600.0), make_input, make_state())
    assert u.aileron > 0.0  # aileron+ rolls right
    assert u.elevator < 0.0  # elevator+ pitches nose down, so climb is negative


def test_bank_and_pitch_limits_hold_for_huge_errors(make_state, make_input):
    # 179 deg right, 5 km low: at the bank/pitch limits (no rates) the law must
    # stop pushing further, and slightly past them it must push back.
    hold = law(heading_deg=179.0, alt=6500.0)
    at_limit = step(hold, make_input, make_state(roll=BANK, pitch=PITCH))
    assert abs(at_limit.aileron) < 1e-9 and abs(at_limit.elevator) < 1e-9
    past = step(hold, make_input, make_state(roll=BANK + 0.05, pitch=PITCH + 0.05, sim_time_s=0.0))
    assert past.aileron < 0.0 and past.elevator > 0.0
    for u in (at_limit, past):
        assert -1.0 <= u.aileron <= 1.0 and -1.0 <= u.elevator <= 1.0
        assert 0.0 <= u.throttle <= 1.0


def test_integrators_do_not_wind_up(make_state, make_input):
    # Ten minutes 179 deg off heading, 5 km low and 40 m/s slow: bank, pitch and
    # throttle are saturated throughout, so no integrator may accumulate.
    hold = law(heading_deg=179.0, alt=6500.0, speed=90.0)
    dt = 1.0 / 120.0
    for k in range(600 * 120):
        hold.step(make_input(make_state(sim_time_s=k * dt, roll=BANK, pitch=PITCH)))
    assert hold.i_hdg == 0.0 and hold.i_alt == 0.0 and hold.i_spd == 0.0
    # Back on the reference, the commands are immediately near trim.
    hold.ref = Reference(0.0, 1500.0, 50.0)
    u = step(hold, make_input, make_state(sim_time_s=600.0))
    assert abs(u.aileron) < 1e-9 and abs(u.elevator) < 1e-9
    assert u.throttle == HoldLaw.THROTTLE_TRIM


def test_integrator_removes_steady_offset(make_state, make_input):
    # A persistent small airspeed deficit is integrated into extra throttle.
    hold = law(speed=52.0)
    first = step(hold, make_input, make_state(sim_time_s=0.0))
    for k in range(1, 1200):
        last = step(hold, make_input, make_state(sim_time_s=k / 120.0))
    assert last.throttle > first.throttle


def test_heading_integrator_removes_steady_offset(make_state, make_input):
    # A standing 2 deg heading error (the bank shortfall propeller torque leaves on
    # the c172x) keeps adding right aileron instead of settling at the P value.
    hold = law(heading_deg=2.0)
    first = step(hold, make_input, make_state(sim_time_s=0.0))
    for k in range(1, 1200):
        last = step(hold, make_input, make_state(sim_time_s=k / 120.0))
    assert last.aileron > first.aileron > 0.0
