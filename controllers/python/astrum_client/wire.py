"""ASTRUM wire format (schema v2) for Python: load ``astrum.capnp``, decode
untrusted ``Telemetry`` payloads into a plain :class:`VehicleState`, and encode
``Command`` messages.

Every decode validates the Cap'n Proto framing (segment table, exact length,
no trailing bytes) and reads every field inside one ``try``, so a malformed
payload surfaces as a single :class:`DecodeError` the caller can count and skip.
"""

from __future__ import annotations

import enum
import math
import os
import struct
from dataclasses import dataclass

import capnp

from astrum_client.mathutil import clamp

# pycapnp resolves `import "/capnp/c++.capnp"` against these directories.
CAPNP_IMPORTS = ["/usr/include"]

# Generous bounds for one telemetry/command message; anything bigger is hostile.
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
