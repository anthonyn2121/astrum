"""The control loop every ASTRUM Python controller runs (controllers/README.md,
"The control loop", steps 1-8), against three interfaces so it is tested
without Zenoh: :class:`TelemetrySource`, :class:`EventSource`, :class:`CommandSink`.

All deadlines are values of ``LoopOptions.clock`` (default :func:`time.monotonic`,
the clock the Zenoh transport waits on).
"""

from __future__ import annotations

import enum
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from astrum_client.wire import (
    Action,
    AutopilotTargets,
    EndReason,
    Event,
    MultirotorRaw,
    RawActuators,
    Refusal,
    RunEnded,
    VehicleKind,
    VehicleState,
    enum_name,
    is_finite,
)

if TYPE_CHECKING:
    from astrum_client.runtime import Law

# At most one refusal warning per this long (loop clock); the rest are counted.
REFUSAL_LOG_PERIOD_S = 1.0

# Longest single telemetry wait: RunEnded is checked between slices.
WAIT_SLICE_S = 0.1


class Stop(Exception):
    """Raised by a law (``step`` or ``configure``) that refuses to fly this aircraft."""


@dataclass(frozen=True)
class StepInput:
    """What a law's ``step`` sees for one telemetry tick."""

    state: VehicleState  # this aircraft's newest sample
    fleet: Mapping[int, VehicleState]  # every aircraft (empty unless the law needs_fleet)
    events: Sequence[Event]  # not yet delivered, of earlier ticks (tick < state.tick)
    control_period_s: float  # the law's dt (Telemetry.controlPeriodS)


class TelemetrySource(Protocol):
    def wait_newer(
        self, aircraft_id: int, after: int, deadline: float
    ) -> tuple[int, VehicleState] | None:
        """The newest sample of ``aircraft_id`` with an arrival number > ``after``
        as ``(arrival, state)``, waiting until ``deadline``; None on timeout."""
        ...

    def snapshot(self) -> dict[int, VehicleState]:
        """The newest sample of every aircraft."""
        ...


class EventSource(Protocol):
    def wait_tick_end(self, tick: int, deadline: float) -> bool:
        """Wait until a ``TickEnd`` >= ``tick`` has arrived (or ``deadline``)."""
        ...

    def drain(self) -> list[Event]:
        """Every event received since the last drain, oldest first."""
        ...


class CommandSink(Protocol):
    epoch: int

    def send(self, action: Action, telemetry_tick: int) -> int:
        """Publish ``action`` answering ``telemetry_tick``; returns its seq."""
        ...


class OutcomeKind(enum.Enum):
    RUN_ENDED = "run-ended"
    TELEMETRY_LOST = "telemetry-lost"
    NO_TELEMETRY = "no-telemetry"
    REFUSED = "refused"
    LAW_FAULT = "law-fault"


@dataclass(frozen=True)
class Outcome:
    """How a control loop ended. ``end_reason`` is set for ``RUN_ENDED`` only."""

    kind: OutcomeKind
    end_reason: EndReason | int | None = None
    detail: str = ""

    def __str__(self) -> str:
        text = self.kind.value
        if self.end_reason is not None:
            text += f"({enum_name(self.end_reason)})"
        return f"{text}: {self.detail}" if self.detail else text


_CLEAN_ENDS = (
    EndReason.MAX_TIME,
    EndReason.ENGAGEMENT_DECIDED,
    EndReason.STOPPED,
    EndReason.RACE_COMPLETE,
    EndReason.CRASHED,
)


def exit_code(outcome: Outcome) -> int:
    """The process exit code for ``outcome`` (controllers/README.md table). A
    ``RunEnded`` reason this schema does not know counts as a failure."""
    if outcome.kind is OutcomeKind.RUN_ENDED:
        return 0 if outcome.end_reason in _CLEAN_ENDS else 1
    if outcome.kind in (OutcomeKind.TELEMETRY_LOST, OutcomeKind.NO_TELEMETRY):
        return 1
    if outcome.kind is OutcomeKind.REFUSED:
        return 2
    return 3


@dataclass
class LoopOptions:
    aircraft_id: int
    startup_deadline: float  # clock value by which the first sample must arrive
    telemetry_timeout_s: float = 10.0
    tick_end_wait_s: float = 0.05
    clock: Callable[[], float] = time.monotonic
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("astrum"))
    status_every: int = 120


def _run_ended(events: Sequence[Event]) -> Outcome | None:
    for ev in events:
        if isinstance(ev.detail, RunEnded):
            return Outcome(OutcomeKind.RUN_ENDED, ev.detail.reason, ev.detail.message)
    return None


def _checked(action, vehicle_kind) -> Action | str:
    """The action with its raw channels clamped, or why it cannot be sent."""
    if not isinstance(action, Action):
        return f"step returned {action!r}, not an Action"
    control = action.control
    if not isinstance(control, RawActuators | MultirotorRaw | AutopilotTargets):
        return f"step returned control {control!r}, not a Control"
    if not is_finite(control):
        return f"non-finite control {control}"
    if isinstance(control, RawActuators) and vehicle_kind == VehicleKind.MULTIROTOR:
        return f"fixed-wing raw control {control} for a multirotor"
    if isinstance(control, MultirotorRaw) and vehicle_kind == VehicleKind.FIXED_WING:
        return f"multirotor raw control {control} for a fixed-wing aircraft"
    if isinstance(control, RawActuators | MultirotorRaw):
        control = control.saturated()
    return Action(control, bool(action.trigger))


def run_control_loop(
    telemetry: TelemetrySource,
    events: EventSource,
    sink: CommandSink,
    law: Law,
    options: LoopOptions,
) -> Outcome:
    """Fly ``options.aircraft_id`` with ``law`` until the run ends, telemetry
    stops, or the aircraft is refused; one command per telemetry tick."""
    o = options
    log, clock, aid = o.logger, o.clock, o.aircraft_id

    def finish(outcome: Outcome) -> Outcome:
        level = logging.INFO if exit_code(outcome) == 0 else logging.ERROR
        log.log(level, "aircraft %d: %s", aid, outcome)
        return outcome

    arrival = 0
    last_good: float | None = None  # clock time of the last good sample
    pending: list[Event] = []  # drained, not yet delivered to the law
    last_tick: int | None = None
    last_action: Action | None = None
    configured = sent = False
    seen_refusals: int | None = None
    last_refusal_log: float | None = None  # clock time of the last refusal warning
    refusals_suppressed = 0
    last_mode = None
    steps = 0
    while True:
        # 1. The next sample, in slices, checking for RunEnded between them.
        if last_good is None:
            deadline = o.startup_deadline
        else:
            deadline = last_good + o.telemetry_timeout_s
        while True:
            got = telemetry.wait_newer(aid, arrival, min(deadline, clock() + WAIT_SLICE_S))
            if got is not None:
                break
            pending.extend(events.drain())
            if (ended := _run_ended(pending)) is not None:
                return finish(ended)
            if clock() >= deadline:
                if last_good is None:
                    return finish(
                        Outcome(
                            OutcomeKind.NO_TELEMETRY,
                            detail="no telemetry before the startup deadline; is the sim running?",
                        )
                    )
                return finish(
                    Outcome(
                        OutcomeKind.TELEMETRY_LOST,
                        detail=f"no telemetry for {o.telemetry_timeout_s:g} s",
                    )
                )
        arrival, s = got
        last_good = clock()

        # 2. Drain the events; a fleet law first waits (bounded) for the tick's TickEnd.
        if law.needs_fleet and not events.wait_tick_end(s.tick, clock() + o.tick_end_wait_s):
            log.debug("aircraft %d: no TickEnd for tick %d; stepping anyway", aid, s.tick)
        pending.extend(events.drain())
        if (ended := _run_ended(pending)) is not None:
            return finish(ended)

        # 3. Vehicle kind.
        if s.vehicle_kind not in law.vehicle_kinds:
            flies = ", ".join(sorted(enum_name(k) for k in law.vehicle_kinds))
            return finish(
                Outcome(
                    OutcomeKind.REFUSED,
                    detail=f"vehicle kind {enum_name(s.vehicle_kind)} (this law flies {flies})",
                )
            )

        # 4. Ownership and refusals of our epoch.
        st = s.command
        if sent and st.last_refused_epoch == sink.epoch:
            if st.last_refusal == Refusal.NOT_OWNER:
                return finish(
                    Outcome(
                        OutcomeKind.REFUSED,
                        detail=f"aircraft {aid} is flown by another controller "
                        f"(epoch {st.owner_epoch})",
                    )
                )
            if st.refusals != seen_refusals:
                now = clock()
                if last_refusal_log is None or now - last_refusal_log >= REFUSAL_LOG_PERIOD_S:
                    log.warning(
                        "aircraft %d: the sim refused our command (%s; %d refused so far, "
                        "%d more refusals not logged)",
                        aid,
                        enum_name(st.last_refusal),
                        st.refusals,
                        refusals_suppressed,
                    )
                    last_refusal_log, refusals_suppressed = now, 0
                else:
                    refusals_suppressed += 1
        seen_refusals = st.refusals

        if s.control_mode != last_mode:
            log.info(
                "aircraft %d: control mode %s at t=%.2f s",
                aid,
                enum_name(s.control_mode),
                s.sim_time_s,
            )
            last_mode = s.control_mode

        # 5. The tick we already answered: resend without stepping.
        if s.tick == last_tick and last_action is not None:
            sink.send(last_action, s.tick)
            continue

        # 6. Configure once, for the telemetry's control period.
        if not configured:
            period = s.control_period_s
            if not (math.isfinite(period) and period > 0.0):
                return finish(
                    Outcome(
                        OutcomeKind.REFUSED,
                        detail=f"telemetry carries no usable control period ({period!r} s)",
                    )
                )
            try:
                law.configure(period)
            except Stop as e:
                return finish(Outcome(OutcomeKind.REFUSED, detail=f"the law refused: {e}"))
            except Exception as e:
                log.exception("aircraft %d: law.configure raised", aid)
                return finish(Outcome(OutcomeKind.LAW_FAULT, detail=f"configure raised {e!r}"))
            configured = True

        # 7. One law step.
        due = [ev for ev in pending if ev.tick < s.tick]
        pending = [ev for ev in pending if ev.tick >= s.tick]
        fleet = telemetry.snapshot() if law.needs_fleet else {}
        try:
            action = law.step(StepInput(s, fleet, due, s.control_period_s))
        except Stop as e:
            return finish(Outcome(OutcomeKind.REFUSED, detail=f"the law refused: {e}"))
        except Exception as e:
            log.exception("aircraft %d: law.step raised", aid)
            return finish(Outcome(OutcomeKind.LAW_FAULT, detail=f"step raised {e!r}"))

        # 8. Check, clamp, send.
        checked = _checked(action, s.vehicle_kind)
        if isinstance(checked, str):
            return finish(Outcome(OutcomeKind.LAW_FAULT, detail=checked))
        sink.send(checked, s.tick)
        sent = True
        last_tick, last_action = s.tick, checked

        steps += 1
        if steps % o.status_every == 0:
            log.info(
                "t=%.1fs hdg=%.1fdeg alt=%.0fm tas=%.1fm/s %s",
                s.sim_time_s,
                math.degrees(s.yaw),
                s.alt_hae_m,
                s.tas_mps,
                checked.control,
            )
