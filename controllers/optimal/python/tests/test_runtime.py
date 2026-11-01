import pytest
from astrum_client.loop import RefuseToFly
from astrum_client.wire import VehicleKind
from astrum_optimal.runtime import PeriodCheck, law_names, parse_args


def test_every_law_is_selectable_and_fixed_wing_only():
    assert law_names() == ["figure8", "formation", "lqr", "mpc"]
    for name in law_names():
        extra = ["--id", "2"] if name == "formation" else []
        law, _ = parse_args(["--law", name, *extra])
        assert law.vehicle_kinds == {VehicleKind.FIXED_WING}


def test_unknown_law_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--law", "pid"])


def feed(check, make_state, ticks, rate_hz=120.0):
    for tick in ticks:
        check(make_state(tick=tick, t=tick / rate_hz))


def test_period_check_accepts_the_sim_clock(make_state):
    check = PeriodCheck(1.0 / 120.0)
    feed(check, make_state, [0, 1])
    assert check.checked


def test_period_check_accepts_a_decimated_control_rate(make_state):
    # control_rate_hz = 30 at 120 Hz physics: a sample every 4th tick.
    check = PeriodCheck(1.0 / 30.0)
    feed(check, make_state, [0, 4])
    assert check.checked


def test_period_check_tolerates_a_dropped_sample(make_state):
    check = PeriodCheck(1.0 / 120.0)
    feed(check, make_state, [5, 7, 8])
    assert check.checked


def test_period_check_refuses_a_mismatched_rate(make_state):
    with pytest.raises(RefuseToFly):  # a 60 Hz sim under a 120 Hz law
        feed(PeriodCheck(1.0 / 120.0), make_state, [0, 600], rate_hz=60.0)
    with pytest.raises(RefuseToFly):  # 50 Hz does not divide 120 Hz
        feed(PeriodCheck(1.0 / 50.0), make_state, [0, 1])
    with pytest.raises(RefuseToFly):  # every tick arrives, but the law expects 30 Hz
        feed(PeriodCheck(1.0 / 30.0), make_state, [0, 1])
    with pytest.raises(RefuseToFly):  # a 30 Hz stream flown by a 120 Hz law
        feed(PeriodCheck(1.0 / 120.0), make_state, range(0, 40, 4))
