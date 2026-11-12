"""Zenoh transport per the ASTRUM contract (controllers/README.md, "Transport").

* :func:`open_session`: the process's one session, ``mode=client`` of the sim
  (a Zenoh router), multicast scouting off, Zenoh's low-latency transport;
  waits for a sim that is still starting until the startup deadline.
* :class:`TelemetrySubscriber`: latest-value telemetry for one aircraft or the
  whole fleet; the Zenoh callback only stores the newest sample per aircraft.
* :class:`EventSubscriber`: FIFO of ``astrum/events`` plus, for a fleet law
  (``tick_ends``), the newest ``astrum/tick`` (:class:`~astrum_client.wire.TickEnd`).
* :class:`CommandPublisher`: ``astrum/command/<id>`` with DROP / express /
  INTERACTIVE_HIGH, a random non-zero ``epoch`` and ``seq`` = 1, 2, ...; holds
  the presence token ``astrum/alive/controller/<id>`` while it lives.

One session carries any number of subscribers and publishers. Every payload is
untrusted: a malformed one is logged (rate-limited), counted in
``decode_errors`` and skipped; nothing raises into a Zenoh thread. Deadlines
are :func:`time.monotonic` values.

Every subscriber callback runs directly on Zenoh's receive thread
(``Callback(indirect=False)``), in the order the samples arrived. zenoh-python's
default runs each subscriber's callbacks on its own Python thread, which loses the
order between subscribers — and the control loop relies on a tick's telemetry being
stored before its ``TickEnd`` (controllers/README.md, step 2). The callbacks only
decode and store under a lock, so they never stall the receive thread.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import deque

import zenoh

from astrum_client.wire import (
    Action,
    Command,
    DecodeError,
    Event,
    TickEnd,
    VehicleState,
    decode_event,
    decode_telemetry,
    decode_tick_end,
    encode_command,
)

TELEMETRY_PREFIX = "astrum/telemetry/"
COMMAND_PREFIX = "astrum/command/"
ALIVE_PREFIX = "astrum/alive/controller/"
TELEMETRY_ALL = "astrum/telemetry/*"
EVENTS = "astrum/events"
TICK = "astrum/tick"

log = logging.getLogger("astrum.transport")


def telemetry_key(aircraft_id: int) -> str:
    return f"{TELEMETRY_PREFIX}{aircraft_id}"


def command_key(aircraft_id: int) -> str:
    return f"{COMMAND_PREFIX}{aircraft_id}"


def alive_key(aircraft_id: int) -> str:
    return f"{ALIVE_PREFIX}{aircraft_id}"


def id_of(key: str, prefix: str) -> int | None:
    """The aircraft id of ``<prefix><id>``, or None if the key isn't one."""
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix) :]
    if not rest.isdigit() or len(rest) > 10:
        return None
    v = int(rest)
    return v if v <= 0xFFFFFFFF else None


def session_config(connect: str, timeout_s: float) -> zenoh.Config:
    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps([connect]))
    conf.insert_json5("connect/timeout_ms", str(max(1, int(timeout_s * 1000))))
    conf.insert_json5("scouting/multicast/enabled", "false")
    conf.insert_json5("transport/unicast/lowlatency", "true")
    conf.insert_json5("transport/unicast/qos/enabled", "false")
    return conf


class SimUnreachable(RuntimeError):
    """No sim accepted the connection before the startup deadline."""


def open_session(connect: str, deadline: float) -> zenoh.Session:
    """The process's one Zenoh session, connected to the sim; waits for it until
    ``deadline`` (a :func:`time.monotonic` value)."""
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise SimUnreachable(f"no sim reachable at {connect}: the startup deadline has passed")
    try:
        return zenoh.open(session_config(connect, remaining))
    except zenoh.ZError as e:
        raise SimUnreachable(f"no sim reachable at {connect} within {remaining:.3g} s: {e}") from e


def _count_error(owner, what: str, err: Exception) -> None:
    """Count one bad payload on ``owner.decode_errors`` and log it (rate-limited)."""
    with owner._cond:
        owner.decode_errors += 1
        n = owner.decode_errors
    if n <= 10 or n % 1000 == 0:
        log.warning("skipping bad %s payload (%d so far): %s", what, n, err)


def _guarded(name: str, fn) -> zenoh.handlers.Callback:
    """A Zenoh callback, run in arrival order on the receive thread (see the module
    docstring), that never lets an exception escape into Zenoh."""

    def callback(sample: zenoh.Sample) -> None:
        try:
            fn(sample)
        except Exception:  # pragma: no cover - defensive; offer() handles bad input
            log.exception("%s callback failed", name)

    return zenoh.handlers.Callback(callback, indirect=False)


class LatestTelemetry:
    """Latest decoded telemetry per aircraft, fed by :meth:`offer` (thread-safe).

    Bad payloads (framing/decoding failures, or a body whose aircraftId differs
    from the key's id) are counted in :attr:`decode_errors` and skipped; the
    previous good sample stays current.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._latest: dict[int, tuple[int, VehicleState]] = {}
        self._arrivals = 0
        self.decode_errors = 0

    def offer(self, key_id: int | None, payload: bytes) -> bool:
        """Decode and store one sample; returns whether it was accepted."""
        try:
            if key_id is None:
                raise DecodeError("telemetry on a key without an aircraft id")
            state = decode_telemetry(payload)
            if state.aircraft_id != key_id:
                raise DecodeError(
                    f"aircraftId {state.aircraft_id} published on the key of aircraft {key_id}"
                )
        except DecodeError as e:
            _count_error(self, "telemetry", e)
            return False
        with self._cond:
            self._arrivals += 1
            self._latest[key_id] = (self._arrivals, state)
            self._cond.notify_all()
        return True

    def latest(self, aircraft_id: int) -> VehicleState | None:
        """The newest good sample of ``aircraft_id`` (any age), or None."""
        with self._cond:
            entry = self._latest.get(aircraft_id)
        return entry[1] if entry else None

    def snapshot(self) -> dict[int, VehicleState]:
        """The newest good sample of every aircraft seen."""
        with self._cond:
            return {aid: state for aid, (_, state) in self._latest.items()}

    def wait_newer(
        self, aircraft_id: int, after: int, deadline: float
    ) -> tuple[int, VehicleState] | None:
        """Block until a sample of ``aircraft_id`` with an arrival number greater
        than ``after`` (0 = any) is stored, or until ``deadline``; returns
        ``(arrival, state)`` or None. Samples in between are superseded, never queued."""
        with self._cond:
            while True:
                entry = self._latest.get(aircraft_id)
                if entry is not None and entry[0] > after:
                    return entry
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(remaining)


class TelemetrySubscriber(LatestTelemetry):
    """Latest-value telemetry of ``aircraft_id``, or of every aircraft when it is None."""

    def __init__(self, session: zenoh.Session, aircraft_id: int | None):
        super().__init__()
        key = TELEMETRY_ALL if aircraft_id is None else telemetry_key(aircraft_id)
        self._sub = session.declare_subscriber(
            key,
            _guarded(
                "telemetry",
                lambda s: self.offer(
                    id_of(str(s.key_expr), TELEMETRY_PREFIX), s.payload.to_bytes()
                ),
            ),
        )


class EventQueue:
    """FIFO of decoded events plus the newest :class:`TickEnd`, fed by
    :meth:`offer_event` / :meth:`offer_tick_end` (thread-safe).

    Bad payloads (framing/decoding failures, or an event kind this schema
    predates) are counted in :attr:`decode_errors` and skipped.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._events: deque[Event] = deque()
        self._tick_end: TickEnd | None = None
        self.decode_errors = 0

    def offer_event(self, payload: bytes) -> bool:
        """Decode and enqueue one event; returns whether it was accepted."""
        try:
            ev = decode_event(payload)
        except DecodeError as e:
            _count_error(self, "event", e)
            return False
        with self._cond:
            self._events.append(ev)
            self._cond.notify_all()
        return True

    def offer_tick_end(self, payload: bytes) -> bool:
        """Decode one ``TickEnd`` and keep it if it is the newest; returns whether
        it decoded."""
        try:
            te = decode_tick_end(payload)
        except DecodeError as e:
            _count_error(self, "tick-end", e)
            return False
        with self._cond:
            if self._tick_end is None or te.tick >= self._tick_end.tick:
                self._tick_end = te
            self._cond.notify_all()
        return True

    def tick_end(self) -> TickEnd | None:
        """The newest ``TickEnd`` received, or None."""
        with self._cond:
            return self._tick_end

    def wait_tick_end(self, tick: int, deadline: float) -> bool:
        """Block until a ``TickEnd`` with ``tick`` or later has arrived, or until
        ``deadline``; returns whether it did."""
        with self._cond:
            while self._tick_end is None or self._tick_end.tick < tick:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(remaining)
            return True

    def drain(self) -> list[Event]:
        """Every queued event, oldest first, emptying the queue."""
        with self._cond:
            out = list(self._events)
            self._events.clear()
        return out


class EventSubscriber(EventQueue):
    """Every event the sim publishes from now on, in order, plus — when
    ``tick_ends`` is set (a ``needs_fleet`` law) — its ``TickEnd``s. Without
    ``tick_ends`` nothing subscribes to ``astrum/tick`` and :meth:`wait_tick_end`
    only sees what :meth:`offer_tick_end` was handed. Subscribe before the run
    starts: events are not replayed."""

    def __init__(self, session: zenoh.Session, tick_ends: bool):
        super().__init__()
        self._events_sub = session.declare_subscriber(
            EVENTS, _guarded("event", lambda s: self.offer_event(s.payload.to_bytes()))
        )
        self._tick_sub = (
            session.declare_subscriber(
                TICK, _guarded("tick-end", lambda s: self.offer_tick_end(s.payload.to_bytes()))
            )
            if tick_ends
            else None
        )


def new_epoch() -> int:
    """A random non-zero 64-bit controller-instance id."""
    while True:
        epoch = secrets.randbits(64)
        if epoch:
            return epoch


class CommandPublisher:
    """Publishes commands for one aircraft with the contract QoS under one random
    non-zero :attr:`epoch`, numbering them ``seq`` = 1, 2, ..., and announces this
    process as the aircraft's controller (liveliness) while it lives."""

    def __init__(self, session: zenoh.Session, target_id: int):
        self.target_id = target_id
        self.epoch = new_epoch()
        self.seq = 0
        self._pub = session.declare_publisher(
            command_key(target_id),
            congestion_control=zenoh.CongestionControl.DROP,
            priority=zenoh.Priority.INTERACTIVE_HIGH,
            express=True,
        )
        self._token = session.liveliness().declare_token(alive_key(target_id))

    def send(self, action: Action, telemetry_tick: int) -> int:
        """Publish ``action`` answering ``telemetry_tick``; returns its ``seq``."""
        self.seq += 1
        cmd = Command(
            self.target_id, self.seq, telemetry_tick, action.trigger, self.epoch, action.control
        )
        self._pub.put(encode_command(cmd))
        return self.seq
