import os
import random
import struct
import threading

import pytest
from astrum_client.transport import EventQueue, LatestTelemetry, id_of
from astrum_client.wire import (
    AutopilotTargets,
    Command,
    ControllerDown,
    ControllerUp,
    ControlMode,
    DecodeError,
    EndReason,
    Event,
    Hit,
    Kill,
    ModeChange,
    ModeReason,
    RawActuators,
    RunEnded,
    TriggerPressed,
    TriggerReleased,
    VehicleKind,
    decode_command,
    decode_event,
    decode_telemetry,
    encode_command,
    encode_event,
    encode_telemetry,
)

EVENTS = [
    Event(1, 1 / 120, TriggerPressed(2, 300)),
    Event(2**40, 0.5, TriggerReleased(2, 41, 259)),
    Event(1234, 10.28125, Hit(2, 1, 2.5, 97.5, (1234.5, -6789.25, 4572.125))),
    Event(4000, 33.3125, Kill(1, 2, (-10.75, 20.5, 1499.0))),
    Event(17, 0.140625, Kill(0xFFFFFFFF, None, (0.0, 0.0, 0.0))),
    Event(600, 5.0, ModeChange(3, ControlMode.RAW, ControlMode.FAILSAFE, ModeReason.LEASE_EXPIRED)),
    Event(0, 0.0, ControllerUp(5)),
    Event(999, 8.3125, ControllerDown(7)),
    Event(7200, 60.0, RunEnded(EndReason.FATAL, "aircraft 3: non-finite state")),
    Event(7200, 60.0, RunEnded(EndReason.MAX_TIME, "")),
]


def test_telemetry_round_trip_keeps_every_field(schema, make_state):
    s = make_state(
        aircraft_id=9,
        tick=2**40 + 3,
        p=0.25,
        q=-0.5,
        r=0.125,
        vehicle_kind=VehicleKind.MULTIROTOR,
        control_mode=ControlMode.FAILSAFE,
        team=3,
        hit_points=12.5,
        alive=False,
        ammo=17,
    )
    assert decode_telemetry(schema, encode_telemetry(schema, s)) == s


def test_command_round_trip(schema):
    for control in (RawActuators(0.25, -0.5, 0.125, 0.75), AutopilotTargets(1.5, 1500.0, 120.0)):
        cmd = Command(target_id=4, seq=11, telemetry_tick=99, trigger=True, control=control)
        assert decode_command(schema, encode_command(schema, cmd)) == cmd


def test_unknown_enum_values_decode_as_raw_ordinals(schema, make_state):
    payload = bytearray(encode_telemetry(schema, make_state()))
    # Patch the two enum fields in the Telemetry data section to values this
    # schema does not define, as a newer sender might.
    ref = encode_telemetry(schema, make_state(vehicle_kind=VehicleKind.MULTIROTOR))
    diff = [i for i, (a, b) in enumerate(zip(payload, ref, strict=True)) if a != b]
    assert len(diff) == 1
    payload[diff[0]] = 7
    s = decode_telemetry(schema, bytes(payload))
    assert s.vehicle_kind == 7 and not isinstance(s.vehicle_kind, VehicleKind)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00" * 4,
        b"garbage!garbage!",
        b"\xff\xff\xff\xff" + b"\x00" * 60,
        struct.pack("<II", 0, 1000) + b"\x00" * 16,  # segment larger than the payload
    ],
)
def test_garbage_raises_decode_error(schema, payload):
    with pytest.raises(DecodeError):
        decode_telemetry(schema, payload)
    with pytest.raises(DecodeError):
        decode_command(schema, payload)


def test_truncated_and_trailing_bytes_rejected(schema, make_state):
    good = encode_telemetry(schema, make_state())
    for bad in (good[:-8], good[:-1], good + b"\x00" * 8, good + b"x"):
        with pytest.raises(DecodeError):
            decode_telemetry(schema, bad)


def test_random_payloads_never_escape_as_other_exceptions(schema):
    rng = random.Random(1234)
    for _ in range(2000):
        n = rng.choice([0, 3, 8, 16, 64, 200, 312])
        payload = bytes(rng.getrandbits(8) for _ in range(n))
        try:
            decode_telemetry(schema, payload)
        except DecodeError:
            pass


def test_latest_telemetry_skips_garbage_and_keeps_last_good(schema, make_state):
    store = LatestTelemetry(schema)
    good = make_state(aircraft_id=1, tick=5)
    assert store.offer(1, encode_telemetry(schema, good))
    assert not store.offer(1, b"not capnp at all")
    assert not store.offer(1, os.urandom(64))
    # Valid message on the wrong aircraft's key.
    assert not store.offer(2, encode_telemetry(schema, make_state(aircraft_id=1, tick=6)))
    assert not store.offer(None, encode_telemetry(schema, good))
    assert store.decode_errors == 4
    assert store.latest(1) == good
    assert store.latest(2) is None


def test_latest_telemetry_is_latest_value_not_fifo(schema, make_state):
    store = LatestTelemetry(schema)
    for tick in (1, 2, 3):
        store.offer(1, encode_telemetry(schema, make_state(tick=tick)))
    arrival, s = store.wait_newer(1, 0, 0.1)
    assert s.tick == 3
    # Nothing newer since: times out instead of replaying queued samples.
    assert store.wait_newer(1, arrival, 0.05) is None


def test_wait_newer_wakes_on_arrival(schema, make_state):
    store = LatestTelemetry(schema)
    payload = encode_telemetry(schema, make_state(tick=42))
    threading.Timer(0.05, store.offer, args=(1, payload)).start()
    got = store.wait_newer(1, 0, 5.0)
    assert got is not None and got[1].tick == 42


def test_id_of():
    assert id_of("astrum/telemetry/7", "astrum/telemetry/") == 7
    assert id_of("astrum/telemetry/x", "astrum/telemetry/") is None
    assert id_of("astrum/telemetry/-1", "astrum/telemetry/") is None
    assert id_of("astrum/telemetry/99999999999", "astrum/telemetry/") is None
    assert id_of("astrum/command/7", "astrum/telemetry/") is None


@pytest.mark.parametrize("ev", EVENTS, ids=lambda e: e.kind)
def test_event_round_trip_and_dict_shape(schema, ev):
    assert decode_event(schema, encode_event(schema, ev)) == ev
    assert Event.from_dict(ev.as_dict()) == ev


def test_event_dict_is_the_shared_flat_shape():
    hit = Event(1234, 10.25, Hit(2, 1, 2.0, 98.0, (1.0, 2.0, 3.0))).as_dict()
    assert hit == {
        "tick": 1234,
        "sim_time_s": 10.25,
        "kind": "hit",
        "shooter_id": 2,
        "target_id": 1,
        "damage": 2.0,
        "hit_points": 98.0,
        "pos_enu": [1.0, 2.0, 3.0],
    }
    mc = Event(5, 0.0, ModeChange(3, ControlMode.AUTOPILOT, 9, ModeReason.CONTROLLER_LOST))
    assert mc.as_dict() == {
        "tick": 5,
        "sim_time_s": 0.0,
        "kind": "mode-change",
        "aircraft_id": 3,
        "from": "autopilot",
        "to": "unknown(9)",
        "reason": "controller-lost",
    }
    assert Event(0, 0.0, Kill(1, None, (0.0, 0.0, 0.0))).as_dict()["killer_id"] is None
    assert Event(0, 0.0, RunEnded(EndReason.ENGAGEMENT_DECIDED, "")).as_dict()["reason"] == (
        "engagement-decided"
    )
    with pytest.raises(ValueError):
        Event.from_dict({"tick": 0, "sim_time_s": 0.0, "kind": "teleport"})


def _patch_single_diff(a: bytes, b: bytes, value: int) -> bytes:
    """``a`` with the one byte where it differs from ``b`` set to ``value``."""
    diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    assert len(diff) == 1
    out = bytearray(a)
    out[diff[0]] = value
    return bytes(out)


def test_unknown_event_enum_values_decode_as_raw_ordinals(schema):
    def run_ended(reason):
        return encode_event(schema, Event(1, 0.0, RunEnded(reason, "")))

    payload = _patch_single_diff(run_ended(EndReason.FATAL), run_ended(EndReason.STOPPED), 9)
    ev = decode_event(schema, payload)
    assert ev.detail.reason == 9 and not isinstance(ev.detail.reason, EndReason)
    assert ev.as_dict()["reason"] == "unknown(9)"


def test_event_kind_unknown_to_this_schema_is_a_decode_error(schema):
    up = encode_event(schema, Event(1, 0.0, ControllerUp(4)))
    down = encode_event(schema, Event(1, 0.0, ControllerDown(4)))
    with pytest.raises(DecodeError):
        decode_event(schema, _patch_single_diff(up, down, 10))  # discriminant past runEnded


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00" * 4,
        b"garbage!garbage!",
        b"\xff\xff\xff\xff" + b"\x00" * 60,
        struct.pack("<II", 0, 1000) + b"\x00" * 16,
    ],
)
def test_event_garbage_raises_decode_error(schema, payload):
    with pytest.raises(DecodeError):
        decode_event(schema, payload)


def test_event_truncated_and_trailing_bytes_rejected(schema):
    good = encode_event(schema, EVENTS[-2])  # run-ended: has a Text pointer
    for bad in (good[:-8], good[:-1], good + b"\x00" * 8, good + b"x"):
        with pytest.raises(DecodeError):
            decode_event(schema, bad)


def test_random_event_payloads_never_escape_as_other_exceptions(schema):
    rng = random.Random(4321)
    for _ in range(2000):
        n = rng.choice([0, 8, 16, 48, 64, 128])
        payload = bytes(rng.getrandbits(8) for _ in range(n))
        try:
            decode_event(schema, payload)
        except DecodeError:
            pass


def test_event_queue_is_fifo_and_skips_garbage(schema):
    q = EventQueue(schema)
    for ev in EVENTS[:3]:
        assert q.offer(encode_event(schema, ev))
    assert not q.offer(b"not capnp at all")
    assert q.offer(encode_event(schema, EVENTS[3]))
    assert q.decode_errors == 1
    assert q.wait(0.1) == EVENTS[0]
    assert q.drain() == EVENTS[1:4]
    assert q.drain() == []
    assert q.wait(0.05) is None


def test_event_queue_wait_wakes_on_arrival(schema):
    q = EventQueue(schema)
    threading.Timer(0.05, q.offer, args=(encode_event(schema, EVENTS[2]),)).start()
    assert q.wait(5.0) == EVENTS[2]
