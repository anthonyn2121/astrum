"""astrum_control — the clean runtime for ASTRUM control-law development (Python).

This module is the plumbing between the wire and *your* control law. It owns
everything you shouldn't have to think about while developing an LQR or MPC
controller:

  * opens the one peer-mode Zenoh session and connects to the sim,
  * subscribes to a single aircraft's telemetry (``astrum/telemetry/<id>``) —
    or, for a law that sets ``needs_world = True``, to the whole fleet
    (``astrum/telemetry/*``) so it can watch its neighbours (e.g. a formation
    follower tracking the leader),
  * decodes each Cap'n Proto ``Telemetry`` frame into a plain :class:`VehicleState`
    (SI units; body rates estimated by finite-differencing attitude, since the
    stream carries none),
  * calls your ``Law.control(state, ref)`` once per frame,
  * clamps and encodes the returned :class:`Command`, and publishes it
    (``astrum/command/<id>``).

You never touch Zenoh or Cap'n Proto. Write a :class:`Law` (see ``laws/lqr.py``,
``laws/mpc.py``) that maps a :class:`VehicleState` + :class:`Reference` to a
:class:`Command`, and run it with ``python run.py --law lqr``.

Wire compatibility is by construction: the schema is loaded at runtime from the
same ``astrum.capnp`` the sim compiles against.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import capnp
import zenoh

# --- CLI / env helpers (precedence: CLI arg > $ENV > default, as everywhere) --


def env_str(key, fallback):
    return os.environ.get(key, fallback)


def env_f64(key, fallback):
    v = os.environ.get(key)
    return float(v) if v is not None else fallback


def default_schema_path():
    """The bundled astrum.capnp, resolved relative to this file."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "crates",
        "astrum-proto",
        "schema",
        "astrum.capnp",
    )


def add_common_args(p):
    """Register the flags every control law shares. Each falls back to its old
    ``ASTRUM_*`` env var, so precedence is CLI arg > env > default — consistent
    with the sim and the example controllers, and with the compose wiring."""
    p.add_argument(
        "--connect",
        default=env_str("ASTRUM_ZENOH_CONNECT", "tcp/127.0.0.1:7447"),
        help="sim Zenoh endpoint (telemetry + commands)",
    )
    p.add_argument(
        "--id",
        type=int,
        default=int(env_f64("ASTRUM_TARGET_ID", 1)),
        help="aircraft id to subscribe to / command",
    )
    p.add_argument(
        "--heading",
        type=float,
        default=env_f64("ASTRUM_TARGET_HEADING_DEG", 0.0),
        help="reference heading in degrees (0 = due north)",
    )
    p.add_argument(
        "--alt",
        type=float,
        default=env_f64("ASTRUM_TARGET_ALT_M", 1500.0),
        help="reference MSL altitude (m)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=env_f64("ASTRUM_TARGET_SPD_MPS", 50.0),
        help="reference airspeed (m/s)",
    )
    p.add_argument(
        "--schema",
        default=env_str("ASTRUM_SCHEMA", default_schema_path()),
        help="path to astrum.capnp",
    )
    return p


# --- The data your law sees and returns --------------------------------------


@dataclass
class VehicleState:
    """One aircraft's state for a single frame, in SI units — everything a
    control law needs, decoded from ``Telemetry``.

    Body rates ``p``/``q``/``r`` are **estimated** by finite-differencing
    attitude between frames (the telemetry stream carries no rates); they are
    0.0 on the first frame. Angles are radians, distances metres, speeds m/s.
    """

    t: float  # sim time (s)
    aircraft_id: int
    lat_deg: float  # geodetic latitude
    lon_deg: float  # geodetic longitude
    alt_m: float  # MSL altitude (m)
    e: float  # ENU east (m)
    n: float  # ENU north (m)
    u: float  # ENU up (m)
    v_n: float  # NED velocity north (m/s)
    v_e: float  # NED velocity east (m/s)
    v_d: float  # NED velocity down (m/s)
    roll: float  # phi (rad)
    pitch: float  # theta (rad)
    yaw: float  # psi / true heading (rad)
    airspeed: float  # true airspeed (m/s)
    p: float  # roll rate estimate (rad/s)
    q: float  # pitch rate estimate (rad/s)
    r: float  # yaw rate estimate (rad/s)
    vehicle_kind: str  # "fixed-wing" | "multirotor"


@dataclass
class Reference:
    """The commanded setpoint (from the CLI) your law should track."""

    heading_rad: float
    alt_m: float
    airspeed_mps: float


class Command:
    """A control output. Build one with :meth:`raw` (you own the control law) or
    :meth:`autopilot` (hand a target to the sim's inner loop). The runtime clamps
    the values to their valid ranges and encodes them to the wire."""

    RAW = "raw"
    AUTOPILOT = "autopilot"

    def __init__(self, kind, values):
        self.kind = kind
        self.values = values

    @classmethod
    def raw(cls, aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.0):
        """Raw actuator setpoints: aileron/elevator/rudder in [-1, 1], throttle
        in [0, 1] (the runtime clamps). The path for a law you own (LQR/MPC)."""
        return cls(cls.RAW, (aileron, elevator, rudder, throttle))

    @classmethod
    def autopilot(cls, heading_rad, alt_m, airspeed_mps):
        """High-level target handed to the sim's built-in inner loop."""
        return cls(cls.AUTOPILOT, (heading_rad, alt_m, airspeed_mps))


class Law:
    """Base class for a control law — subclass it in ``laws/<name>.py`` and
    expose the subclass as ``Law`` there.

    ``__init__`` receives the parsed CLI ``argparse.Namespace`` (so a law can
    read its own flags); ``control`` is called once per telemetry frame.
    """

    def __init__(self, args):
        self.args = args

    def control(self, s: VehicleState, ref: Reference) -> Command:  # pragma: no cover
        raise NotImplementedError("your Law must implement control(state, ref)")


# --- The runtime: wire in, law, wire out -------------------------------------


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Runtime:
    """Drives one aircraft: telemetry in → your law → command out, forever."""

    def __init__(self, args, law: Law):
        self.args = args
        self.law = law
        self.target_id = args.id
        self.ref = Reference(math.radians(args.heading), args.alt, args.speed)
        self._astrum = capnp.load(args.schema)
        # (t, roll, pitch, yaw) of each aircraft's previous frame, keyed by id,
        # for body-rate estimation (per-entity so a world-view law estimates
        # rates for every aircraft it sees, not just its own).
        self._prev = {}
        # A law that sets ``needs_world = True`` (e.g. formation) is driven off a
        # fleet-wide subscription: ``self.world`` holds the latest VehicleState
        # per aircraft id, and the law reads its neighbours (the leader) from it.
        self.needs_world = getattr(law, "needs_world", False)
        self.world = {}
        if self.needs_world:
            law.world = self.world

    def _decode(self, payload: bytes) -> VehicleState:
        with self._astrum.Telemetry.from_bytes(payload) as tel:
            att = tel.attitude
            roll, pitch, yaw = att.phiRad, att.thetaRad, att.psiRad
            t = tel.simTimeS
            aid = tel.aircraftId

            # Estimate body rates from attitude differences (no rates on the
            # wire), keyed per aircraft so a world-view law tracks every entity.
            p = q = r = 0.0
            prev = self._prev.get(aid)
            if prev is not None:
                pt, proll, ppitch, pyaw = prev
                dt = t - pt
                if dt > 0.0:
                    p = (roll - proll) / dt
                    q = (pitch - ppitch) / dt
                    r = _wrap_pi(yaw - pyaw) / dt
            self._prev[aid] = (t, roll, pitch, yaw)

            kind = "multirotor" if str(tel.vehicleKind) == "multirotor" else "fixed-wing"
            return VehicleState(
                t=t,
                aircraft_id=aid,
                lat_deg=tel.geo.latDeg,
                lon_deg=tel.geo.lonDeg,
                alt_m=tel.geo.altM,
                e=tel.enu.x,
                n=tel.enu.y,
                u=tel.enu.z,
                v_n=tel.velNed.x,
                v_e=tel.velNed.y,
                v_d=tel.velNed.z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                airspeed=tel.airspeedMps,
                p=p,
                q=q,
                r=r,
                vehicle_kind=kind,
            )

    def _encode(self, cmd: Command) -> bytes:
        msg = self._astrum.Command.new_message()
        msg.targetId = self.target_id
        if cmd.kind == Command.AUTOPILOT:
            hdg, alt, spd = cmd.values
            ap = msg.control.init("autopilot")
            ap.headingRad = hdg
            ap.altMslM = alt
            ap.airspeedMps = spd
        else:
            aileron, elevator, rudder, throttle = cmd.values
            raw = msg.control.init("raw")
            raw.aileron = _clamp(aileron, -1.0, 1.0)
            raw.elevator = _clamp(elevator, -1.0, 1.0)
            raw.rudder = _clamp(rudder, -1.0, 1.0)
            raw.throttle = _clamp(throttle, 0.0, 1.0)
        return msg.to_bytes()

    def run(self):
        import sys

        connect = self.args.connect
        command_key = f"astrum/command/{self.target_id}"
        # A world-view law needs every aircraft's telemetry (to see the leader),
        # so it subscribes fleet-wide and acts only on its own frames; a normal
        # law subscribes to just its own aircraft.
        telemetry_key = (
            "astrum/telemetry/*" if self.needs_world else f"astrum/telemetry/{self.target_id}"
        )

        # Peer session onto the sim's single Zenoh endpoint. Multicast scouting is
        # off (Docker bridge networks don't pass it), so peers connect explicitly;
        # peer mode makes open() non-fatal if the sim isn't listening yet (it
        # spends ~10 s loading JSBSim before it binds) — Zenoh retries in the
        # background. listen/endpoints=[] keeps this peer purely connect-out.
        conf = zenoh.Config()
        conf.insert_json5("mode", '"peer"')
        conf.insert_json5("connect/endpoints", f'["{connect}"]')
        conf.insert_json5("listen/endpoints", "[]")
        conf.insert_json5("scouting/multicast/enabled", "false")

        with zenoh.open(conf) as session:
            pub = session.declare_publisher(command_key)
            print(
                f"[optimal.py] law={type(self.law).__name__} target={self.target_id} "
                f"connect={connect} ref: hdg={math.degrees(self.ref.heading_rad):.0f}deg "
                f"alt={self.ref.alt_m:.0f}m spd={self.ref.airspeed_mps:.0f}m/s",
                file=sys.stderr,
                flush=True,
            )

            # The default subscriber handler is a FIFO channel; iterating blocks
            # for the next telemetry sample, so there's no busy-poll.
            sub = session.declare_subscriber(telemetry_key)
            n = 0
            for sample in sub:
                s = self._decode(bytes(sample.payload))
                if self.needs_world:
                    # Keep the fleet snapshot fresh, but only run our own law when
                    # our aircraft's frame arrives (that sets the control cadence).
                    self.world[s.aircraft_id] = s
                    if s.aircraft_id != self.target_id:
                        continue
                cmd = self.law.control(s, self.ref)
                pub.put(self._encode(cmd))
                n += 1
                if n % 120 == 0:
                    print(
                        f"[optimal.py] t={s.t:.1f}s hdg={math.degrees(s.yaw):.1f}deg "
                        f"alt={s.alt_m:.0f}m spd={s.airspeed:.1f}m/s",
                        file=sys.stderr,
                        flush=True,
                    )
