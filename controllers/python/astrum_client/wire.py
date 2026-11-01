"""ASTRUM wire format (schema v2) for Python: load ``astrum.capnp``, decode
untrusted ``Telemetry`` payloads into a plain :class:`VehicleState` and
``Event`` payloads into an :class:`Event`, and encode ``Command`` messages.

Every decode validates the Cap'n Proto framing (segment table, exact length,
no trailing bytes) and reads every field inside one ``try``, so a malformed
payload surfaces as a single :class:`DecodeError` the caller can count and skip.
"""

from __future__ import annotations

import enum
import math
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, get_args

import capnp

from astrum_client.mathutil import clamp

# pycapnp resolves `import "/capnp/c++.capnp"` against these directories.
CAPNP_IMPORTS = ["/usr/include"]

# Generous bounds for one telemetry/command/event message; anything bigger is hostile.
_TRAVERSAL_LIMIT_WORDS = 1 << 16
_NESTING_LIMIT = 16


def default_schema_path() -> str:
    """``crates/astrum-proto/schema/astrum.capnp`` of the checkout this package lives in."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "crates", "astrum-proto", "schema", "astrum.capnp")
    )


def load_schema(path: str | None = None):
    """Load the shared schema at runtime (no codegen step)."""
    return capnp.load(path or default_schema_path(), imports=CAPNP_IMPORTS)


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


@dataclass(frozen=True)
class VehicleState:
    """One aircraft's telemetry sample, every schema-v2 field, SI units.

    Angles rad, distances m, speeds m/s; ``alt_m`` is height above the WGS84
    ellipsoid; ``e/n/u`` + ``vel_enu_*`` are the scenario's world grid,
    ``vel_ned_*`` the local level at the aircraft; ``p/q/r`` are the body rates
    the sim measured (not estimates).
    """

    aircraft_id: int
    tick: int
    t: float  # sim time (s) = tick * dt
    lat_deg: float
    lon_deg: float
    alt_m: float  # HAE
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
    airspeed: float  # true airspeed
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


@dataclass(frozen=True)
class RawActuators:
    """Raw actuator setpoints (aileron/elevator/rudder in [-1, 1], throttle in [0, 1])."""

    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def saturated(self) -> RawActuators:
        """Clamped to the valid ranges; a non-finite channel is a law bug and raises."""
        vals = (self.aileron, self.elevator, self.rudder, self.throttle)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite raw actuator command {self}")
        return RawActuators(
            clamp(self.aileron, -1.0, 1.0),
            clamp(self.elevator, -1.0, 1.0),
            clamp(self.rudder, -1.0, 1.0),
            clamp(self.throttle, 0.0, 1.0),
        )


@dataclass(frozen=True)
class AutopilotTargets:
    """Targets for the sim's inner loop: true heading, altitude HAE, true airspeed."""

    heading_rad: float
    alt_hae_m: float
    tas_mps: float

    def checked(self) -> AutopilotTargets:
        vals = (self.heading_rad, self.alt_hae_m, self.tas_mps)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"non-finite autopilot command {self}")
        return self


Control = RawActuators | AutopilotTargets


@dataclass(frozen=True)
class Command:
    """A decoded/encodable ``Command``."""

    target_id: int
    seq: int
    telemetry_tick: int
    trigger: bool
    control: Control


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


def _member(enum_cls, name: str):
    """The member of ``enum_cls`` whose :func:`enum_name` is ``name``."""
    for m in enum_cls:
        if enum_name(m) == name:
            return m
    raise ValueError(f"{name!r} is not a {enum_cls.__name__} name")


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
                _member(ControlMode, d["from"]),
                _member(ControlMode, d["to"]),
                _member(ModeReason, d["reason"]),
            )
        elif detail_cls is RunEnded:
            detail = RunEnded(_member(EndReason, d["reason"]), d["message"])
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


def _reader(struct_schema, payload: bytes):
    _check_framing(payload)
    return struct_schema.from_bytes(
        payload, traversal_limit_in_words=_TRAVERSAL_LIMIT_WORDS, nesting_limit=_NESTING_LIMIT
    )


def decode_telemetry(schema, payload: bytes) -> VehicleState:
    """Decode one untrusted ``Telemetry`` payload; raises :class:`DecodeError`."""
    try:
        with _reader(schema.Telemetry, payload) as t:
            geo, enu, ned, venu = t.geo, t.enu, t.velNed, t.velEnu
            att, rates, aero, combat = t.attitude, t.bodyRates, t.aero, t.combat
            return VehicleState(
                aircraft_id=t.aircraftId,
                tick=t.tick,
                t=t.simTimeS,
                lat_deg=geo.latDeg,
                lon_deg=geo.lonDeg,
                alt_m=geo.altHaeM,
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
                airspeed=t.tasMps,
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
            )
    except DecodeError:
        raise
    except Exception as e:  # pycapnp raises KjException and friends on bad input
        raise DecodeError(f"bad Telemetry payload: {e}") from e


def decode_command(schema, payload: bytes) -> Command:
    """Decode one untrusted ``Command`` payload; raises :class:`DecodeError`."""
    try:
        with _reader(schema.Command, payload) as c:
            which = c.control.which()
            if which == "raw":
                r = c.control.raw
                control = RawActuators(r.aileron, r.elevator, r.rudder, r.throttle)
            elif which == "autopilot":
                a = c.control.autopilot
                control = AutopilotTargets(a.headingRad, a.altHaeM, a.tasMps)
            else:
                raise DecodeError(f"unknown Command.control variant {which!r}")
            return Command(c.targetId, c.seq, c.telemetryTick, c.trigger, control)
    except DecodeError:
        raise
    except Exception as e:
        raise DecodeError(f"bad Command payload: {e}") from e


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


def decode_event(schema, payload: bytes) -> Event:
    """Decode one untrusted ``Event`` payload; raises :class:`DecodeError` (also
    for a kind this schema predates: the caller skips it)."""
    try:
        with _reader(schema.Event, payload) as e:
            return Event(e.tick, e.simTimeS, _decode_detail(e.kind))
    except DecodeError:
        raise
    except Exception as e:
        raise DecodeError(f"bad Event payload: {e}") from e


def encode_command(schema, cmd: Command) -> bytes:
    """Encode a :class:`Command` exactly as given (callers saturate first)."""
    msg = schema.Command.new_message()
    msg.targetId = cmd.target_id
    msg.seq = cmd.seq
    msg.telemetryTick = cmd.telemetry_tick
    msg.trigger = cmd.trigger
    c = cmd.control
    if isinstance(c, RawActuators):
        raw = msg.control.init("raw")
        raw.aileron, raw.elevator, raw.rudder, raw.throttle = (
            c.aileron,
            c.elevator,
            c.rudder,
            c.throttle,
        )
    elif isinstance(c, AutopilotTargets):
        ap = msg.control.init("autopilot")
        ap.headingRad, ap.altHaeM, ap.tasMps = c.heading_rad, c.alt_hae_m, c.tas_mps
    else:
        raise TypeError(f"not a Control: {c!r}")
    return msg.to_bytes()


def encode_telemetry(schema, s: VehicleState) -> bytes:
    """Encode a :class:`VehicleState` as ``Telemetry`` (fixtures and tests; the sim
    is the only real telemetry publisher)."""
    m = schema.Telemetry.new_message()
    m.aircraftId = s.aircraft_id
    m.tick = s.tick
    m.simTimeS = s.t
    m.geo.latDeg, m.geo.lonDeg, m.geo.altHaeM = s.lat_deg, s.lon_deg, s.alt_m
    m.enu.e, m.enu.n, m.enu.u = s.e, s.n, s.u
    m.velNed.n, m.velNed.e, m.velNed.d = s.vel_ned_n, s.vel_ned_e, s.vel_ned_d
    m.velEnu.e, m.velEnu.n, m.velEnu.u = s.vel_enu_e, s.vel_enu_n, s.vel_enu_u
    m.attitude.phiRad, m.attitude.thetaRad, m.attitude.psiRad = s.roll, s.pitch, s.yaw
    m.bodyRates.pRadS, m.bodyRates.qRadS, m.bodyRates.rRadS = s.p, s.q, s.r
    m.tasMps = s.airspeed
    m.aero.alphaRad, m.aero.betaRad, m.aero.nzG, m.aero.mach = s.alpha, s.beta, s.nz_g, s.mach
    m.faultFlags = s.fault_flags
    m.vehicleKind = int(s.vehicle_kind)
    m.controlMode = int(s.control_mode)
    m.combat.team = s.team
    m.combat.hitPoints = s.hit_points
    m.combat.hitPointsMax = s.hit_points_max
    m.combat.alive = s.alive
    m.combat.ammo = s.ammo
    return m.to_bytes()


def encode_event(schema, ev: Event) -> bytes:
    """Encode an :class:`Event` (fixtures and tests; the sim is the only real
    event publisher). Enum fields must be members, not raw ordinals."""
    m = schema.Event.new_message()
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
