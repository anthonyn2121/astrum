import os
import random
import struct
import threading
import time

import pytest
from astrum_client.transport import EventQueue, LatestTelemetry, id_of, new_epoch
from astrum_client.wire import (
    AutopilotTargets,
    Command,
    CommandStatus,
    ControllerDown,
    ControllerUp,
    ControlMode,
    Crashed,
    DecodeError,
    EndReason,
    Event,
    GateMissed,
    GatePassed,
    Hit,
    Kill,
    ModeChange,
    ModeReason,
    MultirotorRaw,
    RaceCourse,
    RaceGate,
    RaceOutcome,
    RaceState,
    RawActuators,
    Refusal,
    RunEnded,
    TickEnd,
    TriggerPressed,
    TriggerReleased,
    VehicleKind,
    decode_command,
    decode_event,
    decode_race_course,
    decode_race_state,
    decode_telemetry,
    decode_tick_end,
    encode_command,
    encode_event,
    encode_race_course,
    encode_race_state,
    encode_telemetry,
    encode_tick_end,
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
    Event(3600, 30.0, RunEnded(EndReason.RACE_COMPLETE, "")),
    Event(960, 8.0, GatePassed(1, 3, -0.125, 0.0625)),
    Event(1200, 10.0, GateMissed(1, 4)),
    Event(1500, 12.5, Crashed(1, RaceOutcome.CRASHED_GATE, 2)),
]


def test_telemetry_round_trip_keeps_every_field(make_state):
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
        control_period_s=1.0 / 30.0,
        command=CommandStatus(2**64 - 1, 41, 2**32 - 1, Refusal.OUT_OF_ORDER, 7),
    )
    assert decode_telemetry(encode_telemetry(s)) == s


@pytest.mark.parametrize(
    "control",
    [
        RawActuators(0.25, -0.5, 0.125, 0.75),
        MultirotorRaw(0.25, -0.5, 0.125, 0.4375),
        AutopilotTargets(1.5, 1500.0, 120.0),
    ],
    ids=lambda c: type(c).__name__,
)
def test_command_round_trip(control):
    cmd = Command(
        target_id=4, seq=11, telemetry_tick=99, trigger=True, epoch=2**64 - 1, control=control
    )
    assert decode_command(encode_command(cmd)) == cmd


@pytest.mark.parametrize(
    "make",
    [RawActuators, MultirotorRaw, AutopilotTargets],
    ids=lambda t: t.__name__,
)
def test_command_encodes_numpy_scalars(make):
    np = pytest.importorskip("numpy")
    values = (0.25, -0.5, 0.125, 0.75)[: 3 if make is AutopilotTargets else 4]
    scalars = [np.float64(values[0]), np.float32(values[1])] + [np.float64(v) for v in values[2:]]
    cmd = Command(
        target_id=4,
        seq=11,
        telemetry_tick=99,
        trigger=np.bool_(True),
        epoch=7,
        control=make(*scalars),
    )
    got = decode_command(encode_command(cmd))
    assert got.trigger is True
    assert got.control == make(*values)


def test_tick_end_round_trip_and_garbage():
    te = TickEnd(2**40 + 1, 1234.5)
    assert decode_tick_end(encode_tick_end(te)) == te
    for bad in (b"", b"garbage!garbage!", encode_tick_end(te) + b"\x00" * 8):
        with pytest.raises(DecodeError):
            decode_tick_end(bad)


def test_epochs_are_random_and_non_zero():
    epochs = {new_epoch() for _ in range(64)}
    assert 0 not in epochs and len(epochs) == 64
    assert all(0 < e < 2**64 for e in epochs)


def test_unknown_enum_values_decode_as_raw_ordinals(make_state):
    payload = bytearray(encode_telemetry(make_state()))
    # Patch the two enum fields in the Telemetry data section to values this
    # schema does not define, as a newer sender might.
    ref = encode_telemetry(make_state(vehicle_kind=VehicleKind.MULTIROTOR))
    diff = [i for i, (a, b) in enumerate(zip(payload, ref, strict=True)) if a != b]
    assert len(diff) == 1
    payload[diff[0]] = 7
    s = decode_telemetry(bytes(payload))
    assert s.vehicle_kind == 7 and not isinstance(s.vehicle_kind, VehicleKind)


def test_unknown_refusal_decodes_as_raw_ordinal(make_state):
    def payload(refusal):
        return encode_telemetry(make_state(command=CommandStatus(last_refusal=refusal)))

    patched = _patch_single_diff(payload(Refusal.NONE), payload(Refusal.STALE), 42)
    s = decode_telemetry(patched)
    assert s.command.last_refusal == 42 and not isinstance(s.command.last_refusal, Refusal)


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
def test_garbage_raises_decode_error(payload):
    with pytest.raises(DecodeError):
        decode_telemetry(payload)
    with pytest.raises(DecodeError):
        decode_command(payload)


def test_truncated_and_trailing_bytes_rejected(make_state):
    good = encode_telemetry(make_state())
    for bad in (good[:-8], good[:-1], good + b"\x00" * 8, good + b"x"):
        with pytest.raises(DecodeError):
            decode_telemetry(bad)


def test_random_payloads_never_escape_as_other_exceptions():
    rng = random.Random(1234)
    for _ in range(2000):
        n = rng.choice([0, 3, 8, 16, 64, 200, 312])
        payload = bytes(rng.getrandbits(8) for _ in range(n))
        try:
            decode_telemetry(payload)
        except DecodeError:
            pass


def test_latest_telemetry_skips_garbage_and_keeps_last_good(make_state):
    store = LatestTelemetry()
    good = make_state(aircraft_id=1, tick=5)
    assert store.offer(1, encode_telemetry(good))
    assert not store.offer(1, b"not capnp at all")
    assert not store.offer(1, os.urandom(64))
    # Valid message on the wrong aircraft's key.
    assert not store.offer(2, encode_telemetry(make_state(aircraft_id=1, tick=6)))
    assert not store.offer(None, encode_telemetry(good))
    assert store.decode_errors == 4
    assert store.latest(1) == good
    assert store.latest(2) is None
    assert store.snapshot() == {1: good}


def test_latest_telemetry_is_latest_value_not_fifo(make_state):
    store = LatestTelemetry()
    for tick in (1, 2, 3):
        store.offer(1, encode_telemetry(make_state(tick=tick)))
    arrival, s = store.wait_newer(1, 0, time.monotonic() + 0.1)
    assert s.tick == 3
    # Nothing newer since: times out instead of replaying queued samples.
    assert store.wait_newer(1, arrival, time.monotonic() + 0.05) is None


def test_wait_newer_wakes_on_arrival(make_state):
    store = LatestTelemetry()
    payload = encode_telemetry(make_state(tick=42))
    threading.Timer(0.05, store.offer, args=(1, payload)).start()
    got = store.wait_newer(1, 0, time.monotonic() + 5.0)
    assert got is not None and got[1].tick == 42


def test_id_of():
    assert id_of("astrum/telemetry/7", "astrum/telemetry/") == 7
    assert id_of("astrum/telemetry/x", "astrum/telemetry/") is None
    assert id_of("astrum/telemetry/-1", "astrum/telemetry/") is None
    assert id_of("astrum/telemetry/99999999999", "astrum/telemetry/") is None
    assert id_of("astrum/command/7", "astrum/telemetry/") is None


@pytest.mark.parametrize("ev", EVENTS, ids=lambda e: e.kind)
def test_event_round_trip_and_dict_shape(ev):
    assert decode_event(encode_event(ev)) == ev
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


def test_race_messages_round_trip():
    gate = RaceGate(2, (1.0, -2.5, 1.5), 0.5, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    for n in (0, 1, 16):
        course = RaceCourse(2**64 - 1, 1.016, 0.0762, 1.1684, (gate,) * n)
        assert decode_race_course(encode_race_course(course)) == course
    for outcome in RaceOutcome:
        state = RaceState(3, 4, 4, 1, 9, 6.5, 7.0, outcome != RaceOutcome.RUNNING, outcome)
        assert decode_race_state(encode_race_state(state)) == state
    for bad in (b"", b"garbage!garbage!", encode_race_state(state)[:-8]):
        with pytest.raises(DecodeError):
            decode_race_course(bad)
        with pytest.raises(DecodeError):
            decode_race_state(bad)


def test_unknown_race_outcome_decodes_as_a_raw_ordinal():
    def state(outcome):
        return encode_race_state(RaceState(1, 0, 0, 0, 3, 0.0, 0.0, False, outcome))

    ev = decode_race_state(
        _patch_single_diff(state(RaceOutcome.COMPLETE), state(RaceOutcome.TIMED_OUT), 9)
    )
    assert ev.outcome == 9 and not isinstance(ev.outcome, RaceOutcome)


def _patch_single_diff(a: bytes, b: bytes, value: int) -> bytes:
    """``a`` with the one byte where it differs from ``b`` set to ``value``."""
    diff = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    assert len(diff) == 1
    out = bytearray(a)
    out[diff[0]] = value
    return bytes(out)


def test_unknown_event_enum_values_decode_as_raw_ordinals():
    def run_ended(reason):
        return encode_event(Event(1, 0.0, RunEnded(reason, "")))

    payload = _patch_single_diff(run_ended(EndReason.FATAL), run_ended(EndReason.STOPPED), 9)
    ev = decode_event(payload)
    assert ev.detail.reason == 9 and not isinstance(ev.detail.reason, EndReason)
    assert ev.as_dict()["reason"] == "unknown(9)"


def test_event_kind_unknown_to_this_schema_is_a_decode_error():
    up = encode_event(Event(1, 0.0, ControllerUp(4)))
    down = encode_event(Event(1, 0.0, ControllerDown(4)))
    with pytest.raises(DecodeError):
        decode_event(_patch_single_diff(up, down, 11))  # discriminant past crashed


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
def test_event_garbage_raises_decode_error(payload):
    with pytest.raises(DecodeError):
        decode_event(payload)


def test_event_truncated_and_trailing_bytes_rejected():
    good = encode_event(EVENTS[-2])  # run-ended: has a Text pointer
    for bad in (good[:-8], good[:-1], good + b"\x00" * 8, good + b"x"):
        with pytest.raises(DecodeError):
            decode_event(bad)


def test_random_event_payloads_never_escape_as_other_exceptions():
    rng = random.Random(4321)
    for _ in range(2000):
        n = rng.choice([0, 8, 16, 48, 64, 128])
        payload = bytes(rng.getrandbits(8) for _ in range(n))
        try:
            decode_event(payload)
        except DecodeError:
            pass


def test_event_queue_is_fifo_and_skips_garbage():
    q = EventQueue()
    for ev in EVENTS[:3]:
        assert q.offer_event(encode_event(ev))
    assert not q.offer_event(b"not capnp at all")
    assert q.offer_event(encode_event(EVENTS[3]))
    assert not q.offer_tick_end(b"not capnp either")
    assert q.decode_errors == 2
    assert q.drain() == EVENTS[:4]
    assert q.drain() == []


def test_event_queue_keeps_the_newest_tick_end():
    q = EventQueue()
    assert not q.wait_tick_end(1, time.monotonic() + 0.01)
    assert q.offer_tick_end(encode_tick_end(TickEnd(5, 0.5)))
    assert q.offer_tick_end(encode_tick_end(TickEnd(4, 0.4)))  # late: superseded
    assert q.tick_end() == TickEnd(5, 0.5)
    assert q.wait_tick_end(5, time.monotonic()) and q.wait_tick_end(3, time.monotonic())
    assert not q.wait_tick_end(6, time.monotonic() + 0.01)


def test_wait_tick_end_wakes_on_arrival():
    q = EventQueue()
    threading.Timer(0.05, q.offer_tick_end, args=(encode_tick_end(TickEnd(7, 0.7)),)).start()
    assert q.wait_tick_end(7, time.monotonic() + 5.0)
