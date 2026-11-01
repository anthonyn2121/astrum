import math

import numpy as np
import pytest
from astrum_client.wire import RawActuators
from astrum_optimal.runtime import Reference, parse_args

DT = 1.0 / 120.0
BANK, PITCH = math.radians(25.0), math.radians(10.0)


class NoFleet:
    def latest(self, aircraft_id):
        return None


FLEET = NoFleet()


def make_law(name, *extra):
    law, args = parse_args(["--law", name, "--alt", "1500", "--speed", "50", *extra])
    law.configure(DT)
    return law, Reference(math.radians(args.heading), args.alt, args.speed)


@pytest.fixture(params=["lqr", "mpc"])
def law_ref(request):
    return make_law(request.param)


def in_range(u: RawActuators) -> bool:
    return (
        -1.0 <= u.aileron <= 1.0
        and -1.0 <= u.elevator <= 1.0
        and -1.0 <= u.rudder <= 1.0
        and 0.0 <= u.throttle <= 1.0
    )


def test_on_reference_is_trim(law_ref, make_state):
    law, ref = law_ref
    u = law.control(make_state(), ref, FLEET)
    assert abs(u.aileron) < 1e-12 and abs(u.elevator) < 1e-12
    assert u.throttle == pytest.approx(law.model.throttle_trim)


def test_signs(law_ref, make_state):
    law, ref = law_ref
    # Reference heading 20 deg right of the nose -> roll right (aileron+).
    right = Reference(math.radians(20.0), ref.alt_m, ref.airspeed_mps)
    assert law.control(make_state(), right, FLEET).aileron > 0.0
    # 100 m below the reference -> nose up (elevator+ pitches DOWN, so negative).
    assert law.control(make_state(alt_m=1400.0, t=0.0), ref, FLEET).elevator < 0.0
    # Slow -> more throttle.
    assert law.control(make_state(airspeed=45.0, t=0.0), ref, FLEET).throttle > 0.6
    # Rolling right while level -> aileron opposes it.
    assert law.control(make_state(p=0.5, t=0.0), ref, FLEET).aileron < 0.0


def test_small_errors_are_exactly_the_linear_gain(law_ref, make_state):
    law, ref = law_ref
    s = make_state(yaw=-0.02, roll=0.01, p=0.01, alt_m=1499.0, pitch=0.001, q=0.002)
    u = law.control(s, ref, FLEET)
    x_lat = np.array([s.yaw - ref.heading_rad, s.roll, s.p])
    x_lon = np.array([0.0, s.alt_m - ref.alt_m, s.pitch, s.q])
    assert u.aileron == pytest.approx(-float(law.k_lat @ x_lat), rel=1e-9)
    assert u.elevator == pytest.approx(-float(law.k_lon @ x_lon), rel=1e-9)


def test_limits_hold_for_huge_errors(law_ref, make_state):
    law, _ = law_ref
    huge = Reference(math.radians(179.0), 6500.0, 250.0)  # 179 deg right, 5 km low, 200 m/s slow
    # At the bank/pitch limit with no rates, the law stops pushing ...
    at = law.control(make_state(roll=BANK, pitch=PITCH, t=0.0), huge, FLEET)
    assert abs(at.aileron) < 1e-9 and abs(at.elevator) < 1e-9
    # ... beyond it, it pushes back (unlimited gains would ask for >100 deg).
    past = law.control(make_state(roll=BANK + 0.05, pitch=PITCH + 0.05, t=0.0), huge, FLEET)
    assert past.aileron < 0.0 and past.elevator > 0.0
    # Every channel saturated, whatever the state.
    wild = make_state(roll=-1.5, pitch=-1.0, p=-5.0, q=-5.0, airspeed=10.0, t=0.0)
    for s in (make_state(t=0.0), wild):
        assert in_range(law.control(s, huge, FLEET))


def test_tighter_bank_limit_is_honoured(make_state):
    law, ref = make_law("lqr", "--max-bank-deg", "10")
    right = Reference(math.radians(90.0), ref.alt_m, ref.airspeed_mps)
    lim = math.radians(10.0)
    assert abs(law.control(make_state(roll=lim), right, FLEET).aileron) < 1e-9
    assert law.control(make_state(roll=lim + 0.01, t=0.0), right, FLEET).aileron < 0.0


def test_integrators_do_not_wind_up(law_ref, make_state):
    law, ref = law_ref
    far = Reference(math.radians(179.0), 6500.0, 250.0)  # bank, pitch and throttle saturate
    for k in range(600 * 120):
        law.control(make_state(t=k * DT, roll=BANK, pitch=PITCH), far, FLEET)
    assert law.i_bank == 0.0 and law.i_pitch == 0.0 and law.i_throttle == 0.0
    # Back on the reference: immediately trim, no wound-up integrator to unwind.
    u = law.control(make_state(t=600.0), ref, FLEET)
    assert abs(u.aileron) < 1e-12 and abs(u.elevator) < 1e-12
    assert u.throttle == pytest.approx(law.model.throttle_trim)


def test_integrators_are_bounded(law_ref, make_state):
    law, ref = law_ref
    # A small persistent error that never saturates the output still cannot
    # integrate past the limit.
    low = Reference(ref.heading_rad + 0.02, 1505.0, 50.5)
    for k in range(3600 * 12):
        law.control(make_state(t=k * DT), low, FLEET)
    assert abs(law.i_bank) <= BANK
    assert abs(law.i_pitch) <= PITCH
    assert abs(law.i_throttle) <= 1.0


def test_heading_integrator_removes_steady_offset(law_ref, make_state):
    law, ref = law_ref
    right = Reference(ref.heading_rad + math.radians(2.0), ref.alt_m, ref.airspeed_mps)
    first = law.control(make_state(t=0.0), right, FLEET)
    for k in range(1, 1200):
        last = law.control(make_state(t=k * DT), right, FLEET)
    assert last.aileron > first.aileron > 0.0
