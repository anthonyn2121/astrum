import logging

import pytest
from astrum_client.loop import (
    EXIT_NO_TELEMETRY,
    EXIT_REFUSED,
    EXIT_TELEMETRY_STOPPED,
    RefuseToFly,
    run_control_loop,
)
from astrum_client.wire import AutopilotTargets, RawActuators, VehicleKind

LOG = logging.getLogger("test")
FIXED_WING = {VehicleKind.FIXED_WING}


class ScriptedSource:
    """Hands out a scripted sequence of samples, then times out."""

    def __init__(self, states):
        self.states = list(states)
        self.timeouts = []

    def wait_newer(self, aircraft_id, after, timeout_s):
        self.timeouts.append(timeout_s)
        if not self.states:
            return None
        return after + 1, self.states.pop(0)


class RecordingSink:
    def __init__(self):
        self.sent = []

    def send(self, control, telemetry_tick, trigger=False):
        self.sent.append((control, telemetry_tick))
        return len(self.sent)


def run(states, step, kinds=FIXED_WING):
    src, sink = ScriptedSource(states), RecordingSink()
    code = run_control_loop(
        src,
        sink,
        1,
        step,
        accepted_kinds=kinds,
        startup_timeout_s=60.0,
        telemetry_timeout_s=10.0,
        log=LOG,
    )
    return code, src, sink


def test_no_telemetry_exits_1_after_startup_timeout():
    code, src, sink = run([], lambda s: RawActuators())
    assert code == EXIT_NO_TELEMETRY
    assert src.timeouts == [60.0] and sink.sent == []


def test_one_command_per_tick_then_exit_0_when_telemetry_stops(make_state):
    calls = []

    def step(s):
        calls.append(s.tick)
        return RawActuators(throttle=0.5)

    code, src, sink = run([make_state(tick=t) for t in (1, 2, 3)], step)
    assert code == EXIT_TELEMETRY_STOPPED
    assert calls == [1, 2, 3]
    assert [tick for _, tick in sink.sent] == [1, 2, 3]
    assert src.timeouts == [60.0, 10.0, 10.0, 10.0]


def test_repeated_tick_resends_last_control_without_rerunning_law(make_state):
    calls = []

    def step(s):
        calls.append(s.tick)
        return RawActuators(aileron=0.1 * s.tick)

    states = [make_state(tick=1), make_state(tick=1), make_state(tick=2)]
    code, _, sink = run(states, step)
    assert code == EXIT_TELEMETRY_STOPPED
    assert calls == [1, 2]
    assert sink.sent == [
        (RawActuators(aileron=0.1), 1),
        (RawActuators(aileron=0.1), 1),
        (RawActuators(aileron=0.2), 2),
    ]


def test_outputs_are_saturated(make_state):
    code, _, sink = run([make_state()], lambda s: RawActuators(5.0, -5.0, 2.0, 3.0))
    assert sink.sent[0][0] == RawActuators(1.0, -1.0, 1.0, 1.0)


def test_non_finite_output_is_a_bug_not_a_command(make_state):
    with pytest.raises(ValueError):
        run([make_state()], lambda s: RawActuators(aileron=float("nan")))
    with pytest.raises(ValueError):
        run([make_state()], lambda s: AutopilotTargets(float("inf"), 1000.0, 50.0))


@pytest.mark.parametrize("kind", [VehicleKind.MULTIROTOR, 7])
def test_wrong_or_unknown_vehicle_kind_is_refused(make_state, kind):
    code, _, sink = run([make_state(vehicle_kind=kind)], lambda s: RawActuators())
    assert code == EXIT_REFUSED and sink.sent == []


def test_step_can_refuse(make_state):
    def step(s):
        raise RefuseToFly("inconsistent sim")

    code, _, sink = run([make_state()], step)
    assert code == EXIT_REFUSED and sink.sent == []
