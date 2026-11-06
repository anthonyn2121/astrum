"""run_main: law-table checks, flag parsing / usage errors, and the wiring of
session, subscriptions, publisher and the one startup deadline (transport faked)."""

import pytest
from astrum_client import runtime, transport
from astrum_client.loop import Outcome, OutcomeKind
from astrum_client.runtime import AttitudeLimits, Law, Reference, check_laws, run_main
from astrum_client.wire import Action, AutopilotTargets, EndReason, VehicleKind


class Hold(Law):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})
    made: list = []

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--gain", type=float, default=1.0, help="hold gain")

    @classmethod
    def from_args(cls, args):
        if args.gain < 0.0:
            raise ValueError("--gain must be >= 0")
        law = cls()
        law.args = args
        cls.made.append(law)
        return law

    def step(self, inp):
        return Action(AutopilotTargets(0.0, 1000.0, 50.0))


class Flock(Hold):
    needs_fleet = True


class Empty(Law):
    pass


LAWS = {"hold": Hold, "flock": Flock}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("ASTRUM_LAW", "ASTRUM_TARGET_ID", "ASTRUM_STARTUP_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    Hold.made.clear()


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_transport(monkeypatch):
    """Records what run_main opens and passes to the loop; the loop returns ``outcome``."""
    seen = {"outcome": Outcome(OutcomeKind.RUN_ENDED, EndReason.MAX_TIME)}

    def open_session(connect, deadline):
        seen["connect"], seen["session_deadline"] = connect, deadline
        return FakeSession()

    class Telemetry:
        decode_errors = 0

        def __init__(self, session, aircraft_id):
            seen["telemetry_id"] = aircraft_id

    class Events:
        decode_errors = 0

        def __init__(self, session, tick_ends):
            seen["events"], seen["tick_ends"] = True, tick_ends

    class Publisher:
        def __init__(self, session, target_id):
            seen["publisher_id"] = target_id

    def loop(telemetry, events, sink, law, options):
        seen["options"], seen["law"] = options, law
        return seen["outcome"]

    monkeypatch.setattr(transport, "open_session", open_session)
    monkeypatch.setattr(transport, "TelemetrySubscriber", Telemetry)
    monkeypatch.setattr(transport, "EventSubscriber", Events)
    monkeypatch.setattr(transport, "CommandPublisher", Publisher)
    monkeypatch.setattr(runtime, "run_control_loop", loop)
    return seen


def main(*argv):
    return run_main(LAWS, list(argv), prog="test", description="test controller")


def test_a_law_without_vehicle_kinds_is_rejected():
    with pytest.raises(ValueError, match="vehicle_kinds"):
        check_laws({"hold": Hold, "empty": Empty})
    with pytest.raises(ValueError, match="vehicle_kinds"):
        run_main({"empty": Empty}, ["--id", "1"], prog="test", description="")
    with pytest.raises(ValueError):
        check_laws({})
    with pytest.raises(TypeError):
        check_laws({"hold": object})


def test_one_startup_deadline_for_session_and_first_sample(fake_transport):
    assert main("--id", "3", "--startup-timeout", "7") == 0
    options = fake_transport["options"]
    assert options.aircraft_id == 3
    assert options.startup_deadline == fake_transport["session_deadline"]
    assert fake_transport["telemetry_id"] == 3 and fake_transport["publisher_id"] == 3
    assert fake_transport["events"] and fake_transport["tick_ends"] is False


def test_default_law_is_the_first_and_reference_limits_are_parsed(fake_transport):
    assert main("--id", "1", "--heading", "90", "--alt", "2000", "--speed", "60") == 0
    args = fake_transport["law"].args
    assert isinstance(fake_transport["law"], Hold) and not isinstance(fake_transport["law"], Flock)
    assert args.reference == Reference(pytest.approx(1.5707963267948966), 2000.0, 60.0)
    assert args.limits == AttitudeLimits(25.0, 10.0)


def test_fleet_law_subscribes_to_every_aircraft(fake_transport, monkeypatch):
    monkeypatch.setenv("ASTRUM_LAW", "flock")
    assert main("--id", "2") == 0
    assert isinstance(fake_transport["law"], Flock)
    assert fake_transport["telemetry_id"] is None and fake_transport["publisher_id"] == 2
    assert fake_transport["tick_ends"] is True


@pytest.mark.parametrize(
    ("kind", "code"),
    [(OutcomeKind.REFUSED, 2), (OutcomeKind.LAW_FAULT, 3), (OutcomeKind.TELEMETRY_LOST, 1)],
)
def test_exit_code_is_the_outcomes(fake_transport, kind, code):
    fake_transport["outcome"] = Outcome(kind)
    assert main("--id", "1") == code


@pytest.mark.parametrize(
    "argv",
    [
        (),  # --id is required
        ("--id", "1", "--law", "pid"),
        ("--id", "1", "--gain", "-1"),  # from_args ValueError
        ("--id", "1", "--heading", "nan"),
        ("--id", "1", "--max-bank-deg", "120"),
    ],
)
def test_usage_errors_exit_2(fake_transport, argv):
    assert main(*argv) == 2
    assert "options" not in fake_transport


def test_help_lists_the_selected_laws_flags(capsys):
    assert main("--law", "hold", "--help") == 0
    out = capsys.readouterr().out
    assert "--gain" in out and "--max-bank-deg" in out and "--rate-hz" not in out


def test_unreachable_sim_exits_1(monkeypatch):
    def unreachable(connect, deadline):
        raise transport.SimUnreachable("no sim")

    monkeypatch.setattr(transport, "open_session", unreachable)
    assert main("--id", "1") == 1
