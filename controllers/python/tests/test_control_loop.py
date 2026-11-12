"""run_control_loop against fake telemetry / events / sink and a fake clock
(controllers/README.md, "The control loop"). Case names follow the C++ port."""

import logging
import math
from dataclasses import dataclass, field

import pytest
from astrum_client.loop import (
    WAIT_SLICE_S,
    LoopOptions,
    Outcome,
    OutcomeKind,
    Stop,
    exit_code,
    run_control_loop,
)
from astrum_client.runtime import Law
from astrum_client.wire import (
    Action,
    AutopilotTargets,
    CommandStatus,
    EndReason,
    Event,
    Hit,
    MultirotorRaw,
    RawActuators,
    Refusal,
    RunEnded,
    VehicleKind,
)

LOG = logging.getLogger("test.loop")
EPOCH = 0xE90C_0000_0000_0001
OTHER_EPOCH = 0x0BAD_0000_0000_0002


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@dataclass
class Sample:
    """One scripted wait: a state (None = the wait times out), the events that
    arrive with it, and whether its TickEnd arrives."""

    state: object = None
    events: list = field(default_factory=list)
    tick_end: bool = True
    at: float | None = None  # clock time the sample arrives at


class FakeEvents:
    def __init__(self, clock: Clock):
        self.clock = clock
        self.queue: list[Event] = []
        self.tick_end: int | None = None
        self.waits: list[tuple[int, float]] = []  # (tick, wait length)

    def wait_tick_end(self, tick, deadline):
        self.waits.append((tick, deadline - self.clock.now))
        if self.tick_end is not None and self.tick_end >= tick:
            return True
        self.clock.now = max(self.clock.now, deadline)
        return False

    def drain(self):
        out, self.queue = self.queue, []
        return out


class FakeTelemetry:
    """Hands out the scripted samples; once the script is empty every wait runs
    to its deadline (the clock jumps there) and returns None."""

    def __init__(self, clock: Clock, events: FakeEvents, script, fleet=None):
        self.clock, self.events = clock, events
        self.script = [s if isinstance(s, Sample) else Sample(s) for s in script]
        self.fleet = fleet or {}
        self.deadlines: list[float] = []
        self.arrival = 0

    def wait_newer(self, aircraft_id, after, deadline):
        assert aircraft_id == 1
        self.deadlines.append(deadline)
        item = self.script.pop(0) if self.script else Sample()
        self.events.queue.extend(item.events)
        if item.state is None:
            self.clock.now = max(self.clock.now, deadline)
            return None
        if item.at is not None:
            self.clock.now = item.at
        if item.tick_end:
            self.events.tick_end = item.state.tick
        self.arrival += 1
        return self.arrival, item.state

    def snapshot(self):
        return dict(self.fleet)


class RecordingSink:
    epoch = EPOCH

    def __init__(self):
        self.sent: list[tuple[Action, int]] = []

    def send(self, action, telemetry_tick):
        self.sent.append((action, telemetry_tick))
        return len(self.sent)


class ScriptedLaw(Law):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})

    def __init__(self, fn=None, kinds=None, needs_fleet=False):
        self.fn = fn or (lambda inp: Action(RawActuators(throttle=0.5)))
        if kinds is not None:
            self.vehicle_kinds = frozenset(kinds)
        self.needs_fleet = needs_fleet
        self.inputs = []
        self.periods = []

    def configure(self, control_period_s):
        self.periods.append(control_period_s)

    def step(self, inp):
        self.inputs.append(inp)
        return self.fn(inp)

    @property
    def ticks(self):
        return [inp.state.tick for inp in self.inputs]


@dataclass
class Run:
    outcome: Outcome
    clock: Clock
    telemetry: FakeTelemetry
    events: FakeEvents
    sink: RecordingSink
    law: ScriptedLaw

    @property
    def sent_ticks(self):
        return [tick for _, tick in self.sink.sent]


def run(script, law=None, *, start=0.0, startup_s=60.0, fleet=None) -> Run:
    clock = Clock(start)
    events = FakeEvents(clock)
    telemetry = FakeTelemetry(clock, events, script, fleet)
    sink, law = RecordingSink(), law or ScriptedLaw()
    options = LoopOptions(
        aircraft_id=1,
        startup_deadline=start + startup_s,
        telemetry_timeout_s=10.0,
        clock=clock,
        logger=LOG,
    )
    outcome = run_control_loop(telemetry, events, sink, law, options)
    return Run(outcome, clock, telemetry, events, sink, law)


def ended(reason, tick=1, message=""):
    return Event(tick, tick / 120.0, RunEnded(reason, message))


def test_one_command_per_tick_then_telemetry_lost(make_state):
    r = run([make_state(tick=t) for t in (1, 2, 3)])
    assert r.law.ticks == [1, 2, 3]
    assert r.sent_ticks == [1, 2, 3]
    assert r.outcome.kind is OutcomeKind.TELEMETRY_LOST and exit_code(r.outcome) == 1
    # Silence for telemetry_timeout_s after the last good sample, waited in slices.
    assert r.clock.now == pytest.approx(10.0)
    waits = r.telemetry.deadlines[3:]
    assert all(b - a <= WAIT_SLICE_S + 1e-9 for a, b in zip([0.0, *waits], waits, strict=False))


def test_no_telemetry(make_state):
    r = run([])
    assert r.outcome.kind is OutcomeKind.NO_TELEMETRY and exit_code(r.outcome) == 1
    assert r.sink.sent == [] and r.law.inputs == []
    assert r.clock.now == pytest.approx(60.0)


def test_single_startup_deadline(make_state):
    # The session took 9.5 s of a 12 s startup budget: the first sample gets only
    # what is left (the absolute deadline), not a fresh timeout.
    r = run([], start=9.5, startup_s=2.5)
    assert r.outcome.kind is OutcomeKind.NO_TELEMETRY
    assert r.clock.now == pytest.approx(12.0)
    assert max(r.telemetry.deadlines) == pytest.approx(12.0)
    assert len(r.telemetry.deadlines) == pytest.approx(25, abs=1)  # 100 ms slices


def test_repeated_tick_resends_without_rerunning_the_law(make_state):
    law = ScriptedLaw(lambda inp: Action(RawActuators(aileron=0.1 * inp.state.tick)))
    r = run([make_state(tick=1), make_state(tick=1), make_state(tick=2)], law)
    assert law.ticks == [1, 2]
    assert r.sink.sent == [
        (Action(RawActuators(aileron=0.1)), 1),
        (Action(RawActuators(aileron=0.1)), 1),
        (Action(RawActuators(aileron=0.2)), 2),
    ]


def test_outputs_are_saturated(make_state):
    r = run([make_state()], ScriptedLaw(lambda inp: Action(RawActuators(5.0, -5.0, 2.0, 3.0))))
    assert r.sink.sent[0][0].control == RawActuators(1.0, -1.0, 1.0, 1.0)


def test_multirotor_outputs_are_saturated(make_state):
    law = ScriptedLaw(
        lambda inp: Action(MultirotorRaw(2.0, -2.0, 0.5, -1.0)), kinds={VehicleKind.MULTIROTOR}
    )
    r = run([make_state(vehicle_kind=VehicleKind.MULTIROTOR)], law)
    assert r.sink.sent[0][0].control == MultirotorRaw(1.0, -1.0, 0.5, 0.0)


@pytest.mark.parametrize(
    "control",
    [
        RawActuators(aileron=math.nan),
        MultirotorRaw(collective=math.inf),
        AutopilotTargets(math.inf, 1000.0, 50.0),
    ],
    ids=lambda c: type(c).__name__,
)
def test_non_finite_output_is_a_law_fault(make_state, control):
    r = run([make_state()], ScriptedLaw(lambda inp: Action(control)))
    assert r.outcome.kind is OutcomeKind.LAW_FAULT and exit_code(r.outcome) == 3
    assert r.sink.sent == []


@pytest.mark.parametrize(
    ("kind", "control"),
    [
        (VehicleKind.MULTIROTOR, RawActuators(throttle=0.5)),
        (VehicleKind.FIXED_WING, MultirotorRaw(collective=0.5)),
    ],
    ids=["raw-on-multirotor", "multirotor-on-fixed-wing"],
)
def test_wrong_vehicle_class_raw_control_is_a_law_fault(make_state, kind, control):
    law = ScriptedLaw(lambda inp: Action(control), kinds=set(VehicleKind))
    r = run([make_state(vehicle_kind=kind)], law)
    assert r.outcome.kind is OutcomeKind.LAW_FAULT and exit_code(r.outcome) == 3
    assert r.sink.sent == []


def test_autopilot_targets_fly_either_vehicle_class(make_state):
    law = ScriptedLaw(lambda inp: Action(AutopilotTargets(0.0, 100.0, 0.0)), kinds=set(VehicleKind))
    for kind in VehicleKind:
        r = run([make_state(tick=1, vehicle_kind=kind)], law)
        assert r.sent_ticks == [1] and r.outcome.kind is OutcomeKind.TELEMETRY_LOST


def test_law_exception_is_a_law_fault_with_traceback(make_state, caplog):
    def boom(inp):
        raise ZeroDivisionError("division by zero")

    with caplog.at_level(logging.ERROR, logger=LOG.name):
        r = run([make_state()], ScriptedLaw(boom))
    assert r.outcome.kind is OutcomeKind.LAW_FAULT and exit_code(r.outcome) == 3
    assert any(rec.exc_info and rec.exc_info[0] is ZeroDivisionError for rec in caplog.records)


@pytest.mark.parametrize("kind", [VehicleKind.MULTIROTOR, 7])
def test_wrong_or_unknown_vehicle_kind_is_refused(make_state, kind):
    r = run([make_state(vehicle_kind=kind)])
    assert r.outcome.kind is OutcomeKind.REFUSED and exit_code(r.outcome) == 2
    assert r.sink.sent == [] and r.law.inputs == []


def test_stop_is_refused(make_state):
    def refuse(inp):
        raise Stop("inconsistent sim")

    r = run([make_state()], ScriptedLaw(refuse))
    assert r.outcome.kind is OutcomeKind.REFUSED and exit_code(r.outcome) == 2
    assert "inconsistent sim" in r.outcome.detail and r.sink.sent == []


def test_stop_in_configure_is_refused(make_state):
    class Short(ScriptedLaw):
        def configure(self, control_period_s):
            raise Stop("horizon shorter than one period")

    r = run([make_state()], Short())
    assert r.outcome.kind is OutcomeKind.REFUSED and r.sink.sent == []


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (EndReason.MAX_TIME, 0),
        (EndReason.ENGAGEMENT_DECIDED, 0),
        (EndReason.STOPPED, 0),
        (EndReason.FATAL, 1),
    ],
)
def test_run_ended_ends_the_loop_with_its_reason(make_state, reason, code):
    r = run([make_state(tick=1), Sample(make_state(tick=2), [ended(reason, 2, "why")])])
    assert r.outcome == Outcome(OutcomeKind.RUN_ENDED, reason, "why")
    assert exit_code(r.outcome) == code
    assert r.law.ticks == [1]  # the tick that carried RunEnded is not flown


def test_run_ended_while_waiting_for_telemetry(make_state):
    r = run([make_state(tick=1), Sample(None, [ended(EndReason.MAX_TIME, 2)])])
    assert r.outcome.kind is OutcomeKind.RUN_ENDED and exit_code(r.outcome) == 0
    assert r.clock.now == pytest.approx(WAIT_SLICE_S)  # one slice, not the telemetry timeout


def test_not_owner_of_our_epoch_is_refused_other_epochs_ignored(make_state):
    def refused(epoch, refusals):
        return CommandStatus(OTHER_EPOCH, 9, refusals, Refusal.NOT_OWNER, epoch)

    r = run(
        [
            # Before we have sent anything, a notOwner naming our epoch is not ours.
            make_state(tick=1, command=refused(EPOCH, 1)),
            make_state(tick=2, command=refused(OTHER_EPOCH, 2)),
            make_state(tick=3, command=refused(EPOCH, 3)),
            make_state(tick=4),
        ]
    )
    assert r.sent_ticks == [1, 2]
    assert r.outcome.kind is OutcomeKind.REFUSED and exit_code(r.outcome) == 2
    assert "flown by another controller" in r.outcome.detail


def test_refusal_of_our_epoch_is_logged_once_per_new_refusal(make_state, caplog):
    def at(t, tick, refusals=0, why=Refusal.NONE, epoch=EPOCH):
        return Sample(
            make_state(tick=tick, command=CommandStatus(EPOCH, 1, refusals, why, epoch)), at=t
        )

    script = [
        at(0.0, 1),
        at(0.0, 2, 1, Refusal.STALE),
        at(0.0, 3, 1, Refusal.STALE),  # the same refusal again: not new
        at(0.5, 4, 2, Refusal.STALE),  # new, but within 1 s of the last warning: counted
        at(2.0, 5, 3, Refusal.OUT_OF_ORDER),
        at(3.0, 6, 4, Refusal.STALE, OTHER_EPOCH),  # not ours
    ]
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        r = run(script)
    assert r.sent_ticks == [1, 2, 3, 4, 5, 6]
    refusals = [rec.getMessage() for rec in caplog.records if "refused our command" in rec.message]
    assert len(refusals) == 2
    assert "stale" in refusals[0] and "0 more" in refusals[0]
    assert "out-of-order" in refusals[1] and "1 more" in refusals[1]


def test_events_of_earlier_ticks_are_delivered_and_later_ones_held(make_state):
    e4, e5, e6 = (Event(t, t / 120.0, Hit(2, 1, 1.0, 99.0, (0.0, 0.0, 0.0))) for t in (4, 5, 6))
    r = run(
        [
            Sample(make_state(tick=5), [e4, e5, e6]),
            make_state(tick=5),  # resend: no step, nothing delivered
            make_state(tick=6),
            make_state(tick=7),
            make_state(tick=8),
        ]
    )
    # An event of the sample's own tick is delivered one step later.
    assert [list(inp.events) for inp in r.law.inputs] == [[e4], [e5], [e6], []]


def test_only_a_fleet_law_waits_for_tick_end(make_state):
    solo = run([Sample(make_state(tick=1), tick_end=False), make_state(tick=2)])
    assert solo.events.waits == []
    assert solo.law.ticks == [1, 2]
    # A fleet law waits, bounded by tick_end_wait_s, and steps anyway without it.
    flock = run(
        [Sample(make_state(tick=1), tick_end=False), make_state(tick=2)],
        ScriptedLaw(needs_fleet=True),
    )
    assert flock.events.waits == [(1, pytest.approx(0.05)), (2, pytest.approx(0.05))]
    assert flock.law.ticks == [1, 2]


@pytest.mark.parametrize("period", [0.0, -1.0 / 120.0, math.nan, math.inf])
def test_missing_control_period_is_refused(make_state, period):
    r = run([make_state(control_period_s=period)])
    assert r.outcome.kind is OutcomeKind.REFUSED and exit_code(r.outcome) == 2
    assert r.law.periods == [] and r.sink.sent == []


def test_configure_once_with_the_telemetry_period(make_state):
    period = 1.0 / 30.0
    r = run([make_state(tick=t, control_period_s=period) for t in (4, 8, 12)])
    assert r.law.periods == [period]
    assert [inp.control_period_s for inp in r.law.inputs] == [period] * 3


def test_trigger_reaches_the_sink(make_state):
    law = ScriptedLaw(lambda inp: Action(AutopilotTargets(0.0, 1000.0, 50.0), trigger=True))
    r = run([make_state(tick=1), make_state(tick=1)], law)
    assert [a.trigger for a, _ in r.sink.sent] == [True, True]


def test_fleet_is_a_snapshot_only_for_fleet_laws(make_state):
    leader = make_state(aircraft_id=2)
    fleet = {2: leader}
    solo = run([make_state()], fleet=fleet)
    assert solo.law.inputs[0].fleet == {}
    flock = run([make_state()], ScriptedLaw(needs_fleet=True), fleet=fleet)
    assert flock.law.inputs[0].fleet == {2: leader}


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (Outcome(OutcomeKind.RUN_ENDED, EndReason.MAX_TIME), 0),
        (Outcome(OutcomeKind.RUN_ENDED, EndReason.FATAL), 1),
        (Outcome(OutcomeKind.RUN_ENDED, 9), 1),
        (Outcome(OutcomeKind.TELEMETRY_LOST), 1),
        (Outcome(OutcomeKind.NO_TELEMETRY), 1),
        (Outcome(OutcomeKind.REFUSED), 2),
        (Outcome(OutcomeKind.LAW_FAULT), 3),
    ],
)
def test_exit_codes(outcome, code):
    assert exit_code(outcome) == code
