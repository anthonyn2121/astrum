"""The control loop every ASTRUM Python controller runs: freshest telemetry in,
one command out per telemetry sample, with startup/staleness timeouts.

Exit codes: 0 = telemetry stopped (sim finished or went away), 1 = no telemetry
within the startup timeout, 2 = refused to fly (wrong vehicle kind or an
inconsistent sim).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection
from typing import Protocol

from astrum_client.wire import Control, RawActuators, VehicleState, enum_name

EXIT_TELEMETRY_STOPPED = 0
EXIT_NO_TELEMETRY = 1
EXIT_REFUSED = 2


class RefuseToFly(Exception):
    """Raised by a step function that must not command this aircraft."""


class TelemetrySource(Protocol):
    def wait_newer(
        self, aircraft_id: int, after: int, timeout_s: float
    ) -> tuple[int, VehicleState] | None: ...


class CommandSink(Protocol):
    def send(self, control: Control, telemetry_tick: int, trigger: bool = False) -> int: ...


def run_control_loop(
    source: TelemetrySource,
    sink: CommandSink,
    aircraft_id: int,
    step: Callable[[VehicleState], Control],
    *,
    accepted_kinds: Collection,
    startup_timeout_s: float,
    telemetry_timeout_s: float,
    log: logging.Logger,
    status_every: int = 120,
) -> int:
    """Run until telemetry stops; returns the process exit code.

    ``step`` is called exactly once per telemetry tick with the newest sample
    (older unprocessed samples are superseded, never queued). Each command
    answers its sample's tick, which keeps a raw-command lease alive and lets a
    lockstep sim advance. When the sim republishes a tick it is still waiting
    on, the previous control is resent (new seq, same telemetryTick) without
    re-running ``step``.
    """
    arrival = 0
    last_tick: int | None = None
    last_control: Control | None = None
    last_mode = None
    processed = 0
    while True:
        timeout = startup_timeout_s if arrival == 0 else telemetry_timeout_s
        got = source.wait_newer(aircraft_id, arrival, timeout)
        if got is None:
            if arrival == 0:
                log.error(
                    "no telemetry for aircraft %d within %.1f s; is the sim running?",
                    aircraft_id,
                    timeout,
                )
                return EXIT_NO_TELEMETRY
            log.info("telemetry stopped for %.1f s; exiting", timeout)
            return EXIT_TELEMETRY_STOPPED
        arrival, s = got

        if s.vehicle_kind not in accepted_kinds:
            log.error(
                "refusing to fly aircraft %d: vehicle kind %s (this controller flies %s)",
                aircraft_id,
                enum_name(s.vehicle_kind),
                ", ".join(sorted(enum_name(k) for k in accepted_kinds)) or "nothing",
            )
            return EXIT_REFUSED
        if s.control_mode != last_mode:
            log.info("control mode: %s at t=%.2f s", enum_name(s.control_mode), s.t)
            last_mode = s.control_mode

        if s.tick == last_tick and last_control is not None:
            sink.send(last_control, s.tick)
            continue
        try:
            control = step(s)
        except RefuseToFly as e:
            log.error("refusing to fly aircraft %d: %s", aircraft_id, e)
            return EXIT_REFUSED
        # Every channel is saturated to its valid range before it goes on the
        # wire; a non-finite output is a law bug and raises.
        control = control.saturated() if isinstance(control, RawActuators) else control.checked()
        sink.send(control, s.tick)
        last_tick, last_control = s.tick, control

        processed += 1
        if processed % status_every == 0:
            log.info(
                "t=%.1fs hdg=%.1fdeg alt=%.0fm tas=%.1fm/s %s",
                s.t,
                math.degrees(s.yaw),
                s.alt_m,
                s.airspeed,
                control,
            )
