"""Zenoh transport per the ASTRUM contract: one client session per process,
latest-value telemetry, and command publishers with the contract QoS.

* Session: ``mode=client`` of the sim, which is a Zenoh router (the hub every
  process connects to, so it routes between them); multicast scouting off.
  ``open`` waits up to the startup timeout for a sim that is still starting.
* Telemetry: a callback overwrites the latest decoded sample per aircraft, so a
  slow control law always acts on the freshest state; nothing is ever queued.
* Commands: declared publisher, CongestionControl DROP, express, priority
  INTERACTIVE_HIGH; every command carries a per-process ``seq`` (1, 2, ...) and
  the ``telemetryTick`` it answers.
"""

from __future__ import annotations

import json
import logging
import threading
import time

import zenoh

from astrum_client.wire import (
    Command,
    Control,
    DecodeError,
    VehicleState,
    decode_telemetry,
    encode_command,
)

TELEMETRY_PREFIX = "astrum/telemetry/"
COMMAND_PREFIX = "astrum/command/"
TELEMETRY_ALL = "astrum/telemetry/*"

log = logging.getLogger("astrum.transport")


def telemetry_key(aircraft_id: int) -> str:
    return f"{TELEMETRY_PREFIX}{aircraft_id}"


def command_key(aircraft_id: int) -> str:
    return f"{COMMAND_PREFIX}{aircraft_id}"


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
    conf.insert_json5("connect/timeout_ms", str(int(timeout_s * 1000)))
    conf.insert_json5("scouting/multicast/enabled", "false")
    return conf


class SimUnreachable(RuntimeError):
    """No sim accepted the connection within the timeout."""


def open_session(connect: str, timeout_s: float) -> zenoh.Session:
    """The process's one Zenoh session, connected to the sim (waits up to
    ``timeout_s`` for it)."""
    try:
        return zenoh.open(session_config(connect, timeout_s))
    except zenoh.ZError as e:
        raise SimUnreachable(f"no sim reachable at {connect} within {timeout_s:g} s: {e}") from e


class LatestTelemetry:
    """Latest decoded telemetry per aircraft, fed by :meth:`offer`.

    Bad payloads (framing/decoding failures, or a body whose aircraftId differs
    from the key's id) are logged, counted in :attr:`decode_errors` and skipped;
    the previous good sample stays current.
    """

    def __init__(self, schema):
        self._schema = schema
        self._cond = threading.Condition()
        self._latest: dict[int, tuple[int, float, VehicleState]] = {}
        self._arrivals = 0
        self.decode_errors = 0

    def offer(self, key_id: int | None, payload: bytes) -> bool:
        """Decode and store one sample; returns whether it was accepted."""
        try:
            if key_id is None:
                raise DecodeError("telemetry on a key without an aircraft id")
            state = decode_telemetry(self._schema, payload)
            if state.aircraft_id != key_id:
                raise DecodeError(
                    f"aircraftId {state.aircraft_id} published on the key of aircraft {key_id}"
                )
        except DecodeError as e:
            with self._cond:
                self.decode_errors += 1
                n = self.decode_errors
            if n <= 10 or n % 1000 == 0:
                log.warning("skipping bad telemetry payload (%d so far): %s", n, e)
            return False
        with self._cond:
            self._arrivals += 1
            self._latest[key_id] = (self._arrivals, time.monotonic(), state)
            self._cond.notify_all()
        return True

    def latest(self, aircraft_id: int) -> VehicleState | None:
        """The newest good sample of ``aircraft_id`` (any age), or None."""
        with self._cond:
            entry = self._latest.get(aircraft_id)
        return entry[2] if entry else None

    def wait_newer(
        self, aircraft_id: int, after: int, timeout_s: float
    ) -> tuple[int, VehicleState] | None:
        """Block until a sample of ``aircraft_id`` arrives with an arrival number
        greater than ``after`` (0 = any); returns ``(arrival, state)`` or None on
        timeout. Samples that arrived in between are superseded, never queued."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                entry = self._latest.get(aircraft_id)
                if entry is not None and entry[0] > after:
                    return entry[0], entry[2]
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._cond.wait(remaining)


def subscribe_telemetry(
    session: zenoh.Session, schema, aircraft_id: int | None
) -> tuple[LatestTelemetry, zenoh.Subscriber]:
    """Latest-value telemetry for one aircraft, or the whole fleet when
    ``aircraft_id`` is None. Keep the returned subscriber alive."""
    store = LatestTelemetry(schema)
    key = TELEMETRY_ALL if aircraft_id is None else telemetry_key(aircraft_id)

    def on_sample(sample: zenoh.Sample) -> None:
        # Runs on a Zenoh thread: never let an exception escape into it.
        try:
            store.offer(id_of(str(sample.key_expr), TELEMETRY_PREFIX), sample.payload.to_bytes())
        except Exception:  # pragma: no cover - defensive; offer() handles bad input
            log.exception("telemetry callback failed")

    sub = session.declare_subscriber(key, on_sample)
    return store, sub


class CommandPublisher:
    """Publishes commands for one aircraft with the contract QoS, numbering
    them ``seq`` = 1, 2, ... for the life of the process."""

    def __init__(self, session: zenoh.Session, schema, target_id: int):
        self._schema = schema
        self.target_id = target_id
        self.seq = 0
        self._pub = session.declare_publisher(
            command_key(target_id),
            congestion_control=zenoh.CongestionControl.DROP,
            priority=zenoh.Priority.INTERACTIVE_HIGH,
            express=True,
        )

    def send(self, control: Control, telemetry_tick: int, trigger: bool = False) -> int:
        self.seq += 1
        cmd = Command(self.target_id, self.seq, telemetry_tick, trigger, control)
        self._pub.put(encode_command(self._schema, cmd))
        return self.seq
