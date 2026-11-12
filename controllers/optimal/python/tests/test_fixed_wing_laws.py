import math

import numpy as np
import pytest
from astrum_client.runtime import AttitudeLimits, Reference, Stop
from astrum_client.wire import RawActuators
from astrum_optimal.laws.lqr import LqrLaw
from astrum_optimal.laws.mpc import MpcLaw

DT = 1.0 / 120.0
BANK, PITCH = math.radians(25.0), math.radians(10.0)
REF = Reference(0.0, 1500.0, 50.0)
LIMITS = AttitudeLimits(25.0, 10.0)


def make_law(cls, ref=REF, limits=LIMITS):
    law = cls(ref, limits)
    law.configure(DT)
    return law


@pytest.fixture(params=[LqrLaw, MpcLaw], ids=["lqr", "mpc"])
def law_cls(request):
    return request.param


@pytest.fixture
def control(make_input):
    """``control(law, state, ref=None)``: one step's raw actuators (``ref`` replaces
    the law's reference first)."""

    def run(law, state, ref=None):
        if ref is not None:
            law.ref = ref
        action = law.step(make_input(state))
        assert isinstance(action.control, RawActuators) and not action.trigger
        return action.control

    return run


def in_range(u: RawActuators) -> bool:
    return (
        -1.0 <= u.aileron <= 1.0
        and -1.0 <= u.elevator <= 1.0
        and -1.0 <= u.rudder <= 1.0
        and 0.0 <= u.throttle <= 1.0
    )


def test_on_reference_is_trim(law_cls, make_state, control):
    law = make_law(law_cls)
    u = control(law, make_state())
    assert abs(u.aileron) < 1e-12 and abs(u.elevator) < 1e-12
    assert u.throttle == pytest.approx(law.model.throttle_trim)


def test_signs(law_cls, make_state, control):
    # Reference heading 20 deg right of the nose -> roll right (aileron+).
    right = Reference(math.radians(20.0), REF.alt_hae_m, REF.tas_mps)
    assert control(make_law(law_cls, right), make_state()).aileron > 0.0
    # 100 m below the reference -> nose up (elevator+ pitches DOWN, so negative).
    law = make_law(law_cls)
    assert control(law, make_state(alt_hae_m=1400.0, sim_time_s=0.0)).elevator < 0.0
    # Slow -> more throttle.
    assert control(make_law(law_cls), make_state(tas_mps=45.0)).throttle > 0.6
    # Rolling right while level -> aileron opposes it.
    assert control(make_law(law_cls), make_state(p=0.5)).aileron < 0.0


def test_small_errors_are_exactly_the_linear_gain(law_cls, make_state, control):
    law = make_law(law_cls)
    s = make_state(yaw=-0.02, roll=0.01, p=0.01, alt_hae_m=1499.0, pitch=0.001, q=0.002)
    u = control(law, s)
    x_lat = np.array([s.yaw - REF.heading_rad, s.roll, s.p])
    x_lon = np.array([0.0, s.alt_hae_m - REF.alt_hae_m, s.pitch, s.q])
    assert u.aileron == pytest.approx(-float(law.k_lat @ x_lat), rel=1e-9)
    assert u.elevator == pytest.approx(-float(law.k_lon @ x_lon), rel=1e-9)


def test_limits_hold_for_huge_errors(law_cls, make_state, control):
    huge = Reference(math.radians(179.0), 6500.0, 250.0)  # 179 deg right, 5 km low, 200 m/s slow
    law = make_law(law_cls)
    # At the bank/pitch limit with no rates, the law stops pushing ...
    at = control(law, make_state(roll=BANK, pitch=PITCH, sim_time_s=0.0), huge)
    assert abs(at.aileron) < 1e-9 and abs(at.elevator) < 1e-9
    # ... beyond it, it pushes back (unlimited gains would ask for >100 deg).
    past = control(law, make_state(roll=BANK + 0.05, pitch=PITCH + 0.05, sim_time_s=0.0))
    assert past.aileron < 0.0 and past.elevator > 0.0
    # Every channel saturated, whatever the state.
    wild = make_state(roll=-1.5, pitch=-1.0, p=-5.0, q=-5.0, tas_mps=10.0, sim_time_s=0.0)
    for s in (make_state(sim_time_s=0.0), wild):
        assert in_range(control(law, s))


def test_tighter_bank_limit_is_honoured(make_state, control):
    right = Reference(math.radians(90.0), REF.alt_hae_m, REF.tas_mps)
    law = make_law(LqrLaw, right, AttitudeLimits(10.0, 10.0))
    lim = math.radians(10.0)
    assert abs(control(law, make_state(roll=lim)).aileron) < 1e-9
    assert control(law, make_state(roll=lim + 0.01, sim_time_s=0.0)).aileron < 0.0


def test_integrators_do_not_wind_up(law_cls, make_state, control):
    far = Reference(math.radians(179.0), 6500.0, 250.0)  # bank, pitch and throttle saturate
    law = make_law(law_cls, far)
    for k in range(600 * 120):
        control(law, make_state(sim_time_s=k * DT, roll=BANK, pitch=PITCH))
    assert law.i_bank == 0.0 and law.i_pitch == 0.0 and law.i_throttle == 0.0
    # Back on the reference: immediately trim, no wound-up integrator to unwind.
    u = control(law, make_state(sim_time_s=600.0), REF)
    assert abs(u.aileron) < 1e-12 and abs(u.elevator) < 1e-12
    assert u.throttle == pytest.approx(law.model.throttle_trim)


def test_integrators_are_bounded(law_cls, make_state, control):
    # A small persistent error that never saturates the output still cannot
    # integrate past the limit.
    law = make_law(law_cls, Reference(REF.heading_rad + 0.02, 1505.0, 50.5))
    for k in range(3600 * 12):
        control(law, make_state(sim_time_s=k * DT))
    assert abs(law.i_bank) <= BANK
    assert abs(law.i_pitch) <= PITCH
    assert abs(law.i_throttle) <= 1.0


def test_heading_integrator_removes_steady_offset(law_cls, make_state, control):
    law = make_law(law_cls, Reference(REF.heading_rad + math.radians(2.0), 1500.0, 50.0))
    first = control(law, make_state(sim_time_s=0.0))
    for k in range(1, 1200):
        last = control(law, make_state(sim_time_s=k * DT))
    assert last.aileron > first.aileron > 0.0


def test_gains_are_designed_for_the_configured_period():
    fast, slow = LqrLaw(REF, LIMITS), LqrLaw(REF, LIMITS)
    fast.configure(1.0 / 120.0)
    slow.configure(1.0 / 30.0)
    assert not np.allclose(fast.k_lat, slow.k_lat)


def test_mpc_horizon_shorter_than_one_period_refuses():
    law = MpcLaw(REF, LIMITS, horizon_s=0.01)
    with pytest.raises(Stop):
        law.configure(1.0 / 30.0)
