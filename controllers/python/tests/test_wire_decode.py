import os
import random
import struct
import threading

import pytest
from astrum_client.transport import LatestTelemetry, id_of
from astrum_client.wire import (
    AutopilotTargets,
    Command,
    ControlMode,
    DecodeError,
    RawActuators,
    VehicleKind,
    decode_command,
    decode_telemetry,
    encode_command,
    encode_telemetry,
)


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
