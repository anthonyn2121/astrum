"""ASTRUM wire format (schema v3) for Python: decode untrusted ``Telemetry``,
``Event`` and ``TickEnd`` payloads into plain dataclasses and encode ``Command``
messages. The names match controllers/README.md ("Names") and the Rust mirror.

The schema ships inside this package (``astrum_client/astrum.capnp``) and is
loaded once, lazily, on the first codec call; every codec function takes only
the payload or value.

Every decode validates the Cap'n Proto framing (segment table, exact length,
no trailing bytes) and reads every field inside one ``try``, so a malformed
payload surfaces as a single :class:`DecodeError` the caller can count and skip.
"""

from __future__ import annotations

import enum
import functools
import importlib.resources
import math
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_args

import capnp

from astrum_client.mathutil import clamp

# Generous bounds for one telemetry/command/event message; anything bigger is hostile.
_TRAVERSAL_LIMIT_WORDS = 1 << 16
_NESTING_LIMIT = 16


@functools.cache
def _schema():
    """The packaged ``astrum.capnp``, loaded on first use. Its ``/capnp/c++.capnp``
    import resolves against pycapnp's own install (``site-packages/capnp/``)."""
    capnp_root = os.path.dirname(os.path.dirname(capnp.__file__))
    resource = importlib.resources.files("astrum_client").joinpath("astrum.capnp")
    with importlib.resources.as_file(resource) as path:
        return capnp.load(str(path), imports=[capnp_root])


class VehicleKind(enum.IntEnum):
    """``astrum.capnp`` ``VehicleKind`` ordinals. A decoded state holds a member,
    or the raw ``int`` ordinal when the sender used a value this schema predates."""

    FIXED_WING = 0
    MULTIROTOR = 1


class ControlMode(enum.IntEnum):
    """``astrum.capnp`` ``ControlMode`` ordinals (unknown values stay raw ``int``)."""

    UNCOMMANDED = 0
    RAW = 1
    AUTOPILOT = 2
    FAILSAFE = 3


class ModeReason(enum.IntEnum):
    """``astrum.capnp`` ``ModeReason`` ordinals (unknown values stay raw ``int``)."""

    COMMAND = 0
    LEASE_EXPIRED = 1
    CONTROLLER_LOST = 2


class EndReason(enum.IntEnum):
    """``astrum.capnp`` ``EndReason`` ordinals (unknown values stay raw ``int``)."""

    MAX_TIME = 0
    ENGAGEMENT_DECIDED = 1
    FATAL = 2
    STOPPED = 3


class Refusal(enum.IntEnum):
    """``astrum.capnp`` ``Refusal`` ordinals: why the sim refused a command
    (unknown values stay raw ``int``)."""

    NONE = 0
    FROM_THE_FUTURE = 1
    STALE = 2
    OUT_OF_ORDER = 3
    NOT_OWNER = 4
    MISSING_EPOCH = 5
    NON_FINITE = 6
    WRONG_VEHICLE_KIND = 7


def _known(enum_cls, value):
    """Map a pycapnp enum reader (or raw ordinal) to ``enum_cls``, keeping unknown
    ordinals as plain ``int``."""
    raw = int(getattr(value, "raw", value))
    try:
        return enum_cls(raw)
    except ValueError:
        return raw


def enum_name(value) -> str:
    """Human-readable name of a decoded enum value (``unknown(7)`` for raw ints)."""
    if isinstance(value, enum.Enum):
        return value.name.lower().replace("_", "-")
    return f"unknown({value})"


def enum_member(enum_cls, name: str):
    """The member of ``enum_cls`` whose :func:`enum_name` is ``name``."""
    for m in enum_cls:
        if enum_name(m) == name:
            return m
    raise ValueError(f"{name!r} is not a {enum_cls.__name__} name")


@dataclass(frozen=True)
class CommandStatus:
    """The sim's command bookkeeping for one aircraft (``Telemetry.command``)."""

    owner_epoch: int = 0  # epoch of the controller flying it; 0 = unowned
    last_seq: int = 0  # seq of the owner's last applied command; 0 = none
    refusals: int = 0  # commands refused for this aircraft so far (u32, wraps)
    last_refusal: Refusal | int = Refusal.NONE
    last_refused_epoch: int = 0  # the epoch that sent the last refused command


@dataclass(frozen=True)
class VehicleState:
    """One aircraft's telemetry sample, every schema-v3 field, SI units.

    Angles rad, distances m, speeds m/s; ``alt_hae_m`` is height above the WGS84
    ellipsoid; ``e/n/u`` + ``vel_enu_*`` are the scenario's world grid,
    ``vel_ned_*`` the local level at the aircraft; ``p/q/r`` are the body rates
    the sim measured (not estimates).
    """

    aircraft_id: int
    tick: int
    sim_time_s: float  # tick * dt
    lat_deg: float
    lon_deg: float
    alt_hae_m: float
    e: float
    n: float
    u: float
    vel_ned_n: float
    vel_ned_e: float
    vel_ned_d: float
    vel_enu_e: float
    vel_enu_n: float
    vel_enu_u: float
    roll: float  # phi, +right wing down
    pitch: float  # theta, +nose up
    yaw: float  # psi, true heading, 0 = North, +East
    p: float
    q: float
    r: float
    tas_mps: float  # true airspeed
    alpha: float
    beta: float
    nz_g: float
    mach: float
    fault_flags: int
    vehicle_kind: VehicleKind | int
    control_mode: ControlMode | int
    team: int
    hit_points: float
    hit_points_max: float
    alive: bool
    ammo: int
    control_period_s: float  # sim time between this aircraft's samples: a law's dt
    command: CommandStatus


@dataclass(frozen=True)
class RawActuators:
    """Raw fixed-wing actuator setpoints (aileron/elevator/rudder in [-1, 1],
    throttle in [0, 1]). Aileron+ rolls right, elevator+ pitches nose DOWN."""

    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def saturated(self) -> RawActuators:
        """Clamped to the valid ranges (NaN passes through: it is a law bug)."""
        return RawActuators(
            clamp(self.aileron, -1.0, 1.0),
            clamp(self.elevator, -1.0, 1.0),
            clamp(self.rudder, -1.0, 1.0),
            clamp(self.throttle, 0.0, 1.0),
        )


@dataclass(frozen=True)
class MultirotorRaw:
    """Raw multirotor demands: normalized body-rate demands in [-1, 1] (+ rolls
    right, + pitches nose UP, + yaws nose right) and collective thrust in [0, 1]."""

    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    collective: float = 0.0

    def saturated(self) -> MultirotorRaw:
        """Clamped to the valid ranges (NaN passes through: it is a law bug)."""
        return MultirotorRaw(
            clamp(self.roll_rate, -1.0, 1.0),
            clamp(self.pitch_rate, -1.0, 1.0),
            clamp(self.yaw_rate, -1.0, 1.0),
            clamp(self.collective, 0.0, 1.0),
        )


@dataclass(frozen=True)
class AutopilotTargets:
    """Targets for the sim's inner loop: true heading, altitude HAE, true airspeed
    (ignored by multirotors)."""

    heading_rad: float
    alt_hae_m: float
    tas_mps: float


Control = RawActuators | MultirotorRaw | AutopilotTargets


def is_finite(control: Control) -> bool:
    """Whether every channel of ``control`` is finite."""
    return all(math.isfinite(v) for v in vars(control).values())


@dataclass(frozen=True)
class Action:
    """What a law returns for one step: the control and the gun trigger."""

    control: Control
    trigger: bool = False


@dataclass(frozen=True)
class Command:
    """A decoded/encodable ``Command`` (the wire message)."""

    target_id: int
    seq: int
    telemetry_tick: int
    trigger: bool
    epoch: int
    control: Control


@dataclass(frozen=True)
class TickEnd:
    """The end of a tick's publications on ``astrum/tick``."""

    tick: int
    sim_time_s: float


Vec3 = tuple[float, float, float]


def _kebab(camel: str) -> str:
    """Schema union member name -> event kind name (``triggerPressed`` -> ``trigger-pressed``)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", camel).lower()


@dataclass(frozen=True)
class TriggerPressed:
    WIRE: ClassVar[str] = "triggerPressed"
    aircraft_id: int
    ammo: int  # rounds left when pressed


@dataclass(frozen=True)
class TriggerReleased:
    WIRE: ClassVar[str] = "triggerReleased"
    aircraft_id: int
    rounds_fired: int  # since the matching TriggerPressed
    ammo: int


@dataclass(frozen=True)
class Hit:
    """A round struck ``target_id``, leaving it ``hit_points``; ``pos_enu`` is where (m)."""

    WIRE: ClassVar[str] = "hit"
    shooter_id: int
    target_id: int
    damage: float
    hit_points: float
    pos_enu: Vec3


@dataclass(frozen=True)
class Kill:
    """``victim_id`` was destroyed at ``pos_enu``; ``killer_id`` is None when unknown."""

    WIRE: ClassVar[str] = "kill"
    victim_id: int
    killer_id: int | None
    pos_enu: Vec3


@dataclass(frozen=True)
class ModeChange:
    """``aircraft_id``'s control mode went ``from_mode`` -> ``to_mode`` (JSON ``from``/``to``)."""

    WIRE: ClassVar[str] = "modeChange"
    aircraft_id: int
    from_mode: ControlMode | int
    to_mode: ControlMode | int
    reason: ModeReason | int


@dataclass(frozen=True)
class ControllerUp:
    WIRE: ClassVar[str] = "controllerUp"
    aircraft_id: int


@dataclass(frozen=True)
class ControllerDown:
    WIRE: ClassVar[str] = "controllerDown"
    aircraft_id: int


@dataclass(frozen=True)
class RunEnded:
    WIRE: ClassVar[str] = "runEnded"
    reason: EndReason | int
    message: str  # the fatal error; empty otherwise


EventDetail = (
    TriggerPressed
    | TriggerReleased
    | Hit
    | Kill
    | ModeChange
    | ControllerUp
    | ControllerDown
    | RunEnded
)

_DETAIL_BY_KIND = {_kebab(c.WIRE): c for c in get_args(EventDetail)}


@dataclass(frozen=True)
class Event:
    """Something that happened during a run: ``tick`` is the ``Telemetry.tick``
    that first shows it, ``sim_time_s`` = tick * dt, ``detail`` the kind's fields."""

    tick: int
    sim_time_s: float
    detail: EventDetail

    @property
    def kind(self) -> str:
        """``trigger-pressed``, ``hit``, ``kill``, ``mode-change``, ``run-ended``, ..."""
        return _kebab(self.detail.WIRE)

    def as_dict(self) -> dict[str, Any]:
        """The flat JSON shape shared with the RL env and the web export
        (``{"tick", "sim_time_s", "kind", <kind's fields>}``, enums as names,
        ``pos_enu`` as ``[e, n, u]``)."""
        out: dict[str, Any] = {"tick": self.tick, "sim_time_s": self.sim_time_s, "kind": self.kind}
        d = self.detail
        if isinstance(d, ModeChange):
            out["aircraft_id"] = d.aircraft_id
            out["from"] = enum_name(d.from_mode)
            out["to"] = enum_name(d.to_mode)
            out["reason"] = enum_name(d.reason)
        elif isinstance(d, RunEnded):
            out["reason"] = enum_name(d.reason)
            out["message"] = d.message
        else:
            for k, v in vars(d).items():
                out[k] = list(v) if k == "pos_enu" else v
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Event:
        """Inverse of :meth:`as_dict` for kinds and enum names this schema knows
        (anything else raises ``ValueError``)."""
        detail_cls = _DETAIL_BY_KIND.get(d["kind"])
        if detail_cls is None:
            raise ValueError(f"unknown event kind {d['kind']!r}")
        if detail_cls is ModeChange:
            detail = ModeChange(
                d["aircraft_id"],
                enum_member(ControlMode, d["from"]),
                enum_member(ControlMode, d["to"]),
                enum_member(ModeReason, d["reason"]),
            )
        elif detail_cls is RunEnded:
            detail = RunEnded(enum_member(EndReason, d["reason"]), d["message"])
        else:
            kw = {f.name: d[f.name] for f in fields(detail_cls)}
            if "pos_enu" in kw:
                kw["pos_enu"] = tuple(float(x) for x in kw["pos_enu"])
            detail = detail_cls(**kw)
        return cls(d["tick"], float(d["sim_time_s"]), detail)


class DecodeError(ValueError):
    """A payload that is not a well-formed message of the expected type."""


def _check_framing(payload: bytes) -> None:
    """Validate the unpacked-stream segment table and require the payload to be
    exactly one message (pycapnp would silently ignore trailing bytes)."""
    n = len(payload)
    if n < 8:
        raise DecodeError(f"payload too short ({n} bytes)")
    (segments_minus_one,) = struct.unpack_from("<I", payload, 0)
    segments = segments_minus_one + 1
    if segments > 512:
        raise DecodeError(f"implausible segment count {segments}")
    header_words = (segments + 2) // 2  # 4-byte count + 4 bytes per segment, padded to 8
    if n < header_words * 8:
        raise DecodeError("payload ends inside the segment table")
    sizes = struct.unpack_from(f"<{segments}I", payload, 4)
    expected = (header_words + sum(sizes)) * 8
    if n != expected:
        raise DecodeError(f"payload is {n} bytes, message framing says {expected}")


def _reader(struct_name: str, payload: bytes):
    _check_framing(payload)
    return getattr(_schema(), struct_name).from_bytes(
        payload, traversal_limit_in_words=_TRAVERSAL_LIMIT_WORDS, nesting_limit=_NESTING_LIMIT
    )


def decode_telemetry(payload: bytes) -> VehicleState:
    """Decode one untrusted ``Telemetry`` payload; raises :class:`DecodeError`."""
    try:
        with _reader("Telemetry", payload) as t:
            geo, enu, ned, venu = t.geo, t.enu, t.velNed, t.velEnu
            att, rates, aero, combat, cmd = t.attitude, t.bodyRates, t.aero, t.combat, t.command
            return VehicleState(
                aircraft_id=t.aircraftId,
                tick=t.tick,
                sim_time_s=t.simTimeS,
                lat_deg=geo.latDeg,
                lon_deg=geo.lonDeg,
                alt_hae_m=geo.altHaeM,
                e=enu.e,
                n=enu.n,
                u=enu.u,
                vel_ned_n=ned.n,
                vel_ned_e=ned.e,
                vel_ned_d=ned.d,
                vel_enu_e=venu.e,
                vel_enu_n=venu.n,
                vel_enu_u=venu.u,
                roll=att.phiRad,
                pitch=att.thetaRad,
                yaw=att.psiRad,
                p=rates.pRadS,
                q=rates.qRadS,
                r=rates.rRadS,
                tas_mps=t.tasMps,
                alpha=aero.alphaRad,
                beta=aero.betaRad,
                nz_g=aero.nzG,
                mach=aero.mach,
                fault_flags=t.faultFlags,
                vehicle_kind=_known(VehicleKind, t.vehicleKind),
                control_mode=_known(ControlMode, t.controlMode),
                team=combat.team,
                hit_points=combat.hitPoints,
                hit_points_max=combat.hitPointsMax,
                alive=combat.alive,
                ammo=combat.ammo,
                control_period_s=t.controlPeriodS,
                command=CommandStatus(
                    owner_epoch=cmd.ownerEpoch,
                    last_seq=cmd.lastSeq,
                    refusals=cmd.refusals,
                    last_refusal=_known(Refusal, cmd.lastRefusal),
                    last_refused_epoch=cmd.lastRefusedEpoch,
                ),
            )
    except DecodeError:
        raise
    except Exception as e:  # pycapnp raises KjException and friends on bad input
        raise DecodeError(f"bad Telemetry payload: {e}") from e


def decode_command(payload: bytes) -> Command:
    """Decode one untrusted ``Command`` payload; raises :class:`DecodeError`."""
    try:
        with _reader("Command", payload) as c:
            which = c.control.which()
            if which == "raw":
                r = c.control.raw
                control = RawActuators(r.aileron, r.elevator, r.rudder, r.throttle)
            elif which == "multirotor":
                m = c.control.multirotor
                control = MultirotorRaw(m.rollRate, m.pitchRate, m.yawRate, m.collective)
            elif which == "autopilot":
                a = c.control.autopilot
                control = AutopilotTargets(a.headingRad, a.altHaeM, a.tasMps)
            else:
                raise DecodeError(f"unknown Command.control variant {which!r}")
            return Command(c.targetId, c.seq, c.telemetryTick, c.trigger, c.epoch, control)
    except DecodeError:
        raise
    except Exception as e:
        raise DecodeError(f"bad Command payload: {e}") from e


def decode_tick_end(payload: bytes) -> TickEnd:
    """Decode one untrusted ``TickEnd`` payload; raises :class:`DecodeError`."""
    try:
        with _reader("TickEnd", payload) as t:
            return TickEnd(t.tick, t.simTimeS)
    except DecodeError:
        raise
    except Exception as e:
        raise DecodeError(f"bad TickEnd payload: {e}") from e


def _enu(v) -> Vec3:
    return (v.e, v.n, v.u)


def _decode_detail(k) -> EventDetail:
    which = k.which()
    if which == "triggerPressed":
        w = k.triggerPressed
        return TriggerPressed(w.aircraftId, w.ammo)
    if which == "triggerReleased":
        w = k.triggerReleased
        return TriggerReleased(w.aircraftId, w.roundsFired, w.ammo)
    if which == "hit":
        w = k.hit
        return Hit(w.shooterId, w.targetId, w.damage, w.hitPoints, _enu(w.posEnu))
    if which == "kill":
        w = k.kill
        killer = w.killer.aircraftId if w.killer.which() == "aircraftId" else None
        return Kill(w.victimId, killer, _enu(w.posEnu))
    if which == "modeChange":
        w = k.modeChange
        return ModeChange(
            w.aircraftId,
            _known(ControlMode, getattr(w, "from")),  # `from` is a Python keyword
            _known(ControlMode, w.to),
            _known(ModeReason, w.reason),
        )
    if which == "controllerUp":
        return ControllerUp(k.controllerUp)
    if which == "controllerDown":
        return ControllerDown(k.controllerDown)
    if which == "runEnded":
        w = k.runEnded
        return RunEnded(_known(EndReason, w.reason), w.message)
    raise DecodeError(f"unknown Event.kind variant {which!r}")


def decode_event(payload: bytes) -> Event:
    """Decode one untrusted ``Event`` payload; raises :class:`DecodeError` (also
    for a kind this schema predates: the caller skips it)."""
    try:
        with _reader("Event", payload) as e:
            return Event(e.tick, e.simTimeS, _decode_detail(e.kind))
    except DecodeError:
        raise
    except Exception as e:
        raise DecodeError(f"bad Event payload: {e}") from e


def encode_command(cmd: Command) -> bytes:
    """Encode a :class:`Command` exactly as given (callers saturate first).

    Control values and the trigger are coerced to built-in ``float``/``bool``: pycapnp
    rejects other numeric scalars (e.g. ``numpy.float64``) a law may return."""
    msg = _schema().Command.new_message()
    msg.targetId = cmd.target_id
    msg.seq = cmd.seq
    msg.telemetryTick = cmd.telemetry_tick
    msg.trigger = bool(cmd.trigger)
    msg.epoch = cmd.epoch
    c = cmd.control
    if isinstance(c, RawActuators):
        raw = msg.control.init("raw")
        raw.aileron, raw.elevator, raw.rudder, raw.throttle = (
            float(c.aileron),
            float(c.elevator),
            float(c.rudder),
            float(c.throttle),
        )
    elif isinstance(c, MultirotorRaw):
        mr = msg.control.init("multirotor")
        mr.rollRate, mr.pitchRate, mr.yawRate, mr.collective = (
            float(c.roll_rate),
            float(c.pitch_rate),
            float(c.yaw_rate),
            float(c.collective),
        )
    elif isinstance(c, AutopilotTargets):
        ap = msg.control.init("autopilot")
        ap.headingRad, ap.altHaeM, ap.tasMps = (
            float(c.heading_rad),
            float(c.alt_hae_m),
            float(c.tas_mps),
        )
    else:
        raise TypeError(f"not a Control: {c!r}")
    return msg.to_bytes()


def encode_telemetry(s: VehicleState) -> bytes:
    """Encode a :class:`VehicleState` as ``Telemetry`` (fixtures and tests; the sim
    is the only real telemetry publisher)."""
    m = _schema().Telemetry.new_message()
    m.aircraftId = s.aircraft_id
    m.tick = s.tick
    m.simTimeS = s.sim_time_s
    m.geo.latDeg, m.geo.lonDeg, m.geo.altHaeM = s.lat_deg, s.lon_deg, s.alt_hae_m
    m.enu.e, m.enu.n, m.enu.u = s.e, s.n, s.u
    m.velNed.n, m.velNed.e, m.velNed.d = s.vel_ned_n, s.vel_ned_e, s.vel_ned_d
    m.velEnu.e, m.velEnu.n, m.velEnu.u = s.vel_enu_e, s.vel_enu_n, s.vel_enu_u
    m.attitude.phiRad, m.attitude.thetaRad, m.attitude.psiRad = s.roll, s.pitch, s.yaw
    m.bodyRates.pRadS, m.bodyRates.qRadS, m.bodyRates.rRadS = s.p, s.q, s.r
    m.tasMps = s.tas_mps
    m.aero.alphaRad, m.aero.betaRad, m.aero.nzG, m.aero.mach = s.alpha, s.beta, s.nz_g, s.mach
    m.faultFlags = s.fault_flags
    m.vehicleKind = int(s.vehicle_kind)
    m.controlMode = int(s.control_mode)
    m.combat.team = s.team
    m.combat.hitPoints = s.hit_points
    m.combat.hitPointsMax = s.hit_points_max
    m.combat.alive = s.alive
    m.combat.ammo = s.ammo
    m.controlPeriodS = s.control_period_s
    c = s.command
    m.command.ownerEpoch = c.owner_epoch
    m.command.lastSeq = c.last_seq
    m.command.refusals = c.refusals
    m.command.lastRefusal = int(c.last_refusal)
    m.command.lastRefusedEpoch = c.last_refused_epoch
    return m.to_bytes()


def encode_tick_end(t: TickEnd) -> bytes:
    """Encode a :class:`TickEnd` (fixtures and tests; the sim is its only publisher)."""
    m = _schema().TickEnd.new_message()
    m.tick = t.tick
    m.simTimeS = t.sim_time_s
    return m.to_bytes()


def encode_event(ev: Event) -> bytes:
    """Encode an :class:`Event` (fixtures and tests; the sim is the only real
    event publisher). Enum fields must be members, not raw ordinals."""
    m = _schema().Event.new_message()
    m.tick = ev.tick
    m.simTimeS = ev.sim_time_s
    d = ev.detail
    if isinstance(d, ControllerUp | ControllerDown):
        setattr(m.kind, d.WIRE, d.aircraft_id)
        return m.to_bytes()
    w = m.kind.init(d.WIRE)
    if isinstance(d, TriggerPressed):
        w.aircraftId, w.ammo = d.aircraft_id, d.ammo
    elif isinstance(d, TriggerReleased):
        w.aircraftId, w.roundsFired, w.ammo = d.aircraft_id, d.rounds_fired, d.ammo
    elif isinstance(d, Hit):
        w.shooterId, w.targetId, w.damage, w.hitPoints = (
            d.shooter_id,
            d.target_id,
            d.damage,
            d.hit_points,
        )
        w.posEnu.e, w.posEnu.n, w.posEnu.u = d.pos_enu
    elif isinstance(d, Kill):
        w.victimId = d.victim_id
        if d.killer_id is None:
            w.killer.unknown = None
        else:
            w.killer.aircraftId = d.killer_id
        w.posEnu.e, w.posEnu.n, w.posEnu.u = d.pos_enu
    elif isinstance(d, ModeChange):
        w.aircraftId = d.aircraft_id
        setattr(w, "from", ControlMode(d.from_mode).value)
        w.to = ControlMode(d.to_mode).value
        w.reason = ModeReason(d.reason).value
    elif isinstance(d, RunEnded):
        w.reason = EndReason(d.reason).value
        w.message = d.message
    else:
        raise TypeError(f"not an EventDetail: {d!r}")
    return m.to_bytes()
