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


def test_period_check_accepts_the_sim_clock(make_state):
    check = PeriodCheck(1.0 / 120.0)
    check(make_state(tick=0, t=0.0))  # tick 0 carries no rate information
    check(make_state(tick=1234567, t=1234567 / 120.0))
    assert check.checked


def test_period_check_refuses_a_mismatched_rate(make_state):
    check = PeriodCheck(1.0 / 120.0)
    with pytest.raises(RefuseToFly):
        check(make_state(tick=600, t=10.0))  # a 60 Hz sim
