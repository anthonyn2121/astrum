#!/usr/bin/env python3
"""ASTRUM example flight controller (Python).

A minimal out-of-sim controller that talks to ``astrum-sim`` purely over Zenoh,
using the shared Cap'n Proto schema (``astrum.capnp``) for the wire format. It is
the Python twin of ``controllers/cpp`` — same key expressions, same env config,
same hold-law gains, same framing — proving a controller in any language plugs
into ASTRUM with nothing but eclipse-zenoh + pycapnp.

  * SUB  astrum/telemetry/<id> — telemetry for one aircraft
  * PUB  astrum/command/<id>   — commands (target id + raw actuators | autopilot)

Both share one Zenoh session (one TCP endpoint) instead of ZMQ's two ports.

Two modes (env ``ASTRUM_CONTROL_MODE``):
  * "raw" (default) — closes a heading/altitude/airspeed hold loop *here*, in
    the controller, sending raw ``fcs/*-cmd-norm`` deflections each tick. This
    is ASTRUM's intended usage: the user owns the control law.
  * "autopilot" — sends one high-level target (reasserted over the first second
    of telemetry to clear the startup race) and then just monitors, letting the
    sim's built-in inner loop fly it.

Config is via CLI flags (run with ``--help``); each falls back to its old
ASTRUM_* env var, then a default, so precedence is CLI arg > env > default and
the compose wiring keeps working:
  --connect  ASTRUM_ZENOH_CONNECT      (default tcp/127.0.0.1:7447; tcp/sim:7447 under compose)
  --id       ASTRUM_TARGET_ID          (default 1)
  --heading  ASTRUM_TARGET_HEADING_DEG (default 0    — due north, degrees)
  --alt      ASTRUM_TARGET_ALT_M       (default 1500)
  --speed    ASTRUM_TARGET_SPD_MPS     (default 50)
  --mode     ASTRUM_CONTROL_MODE       (default raw)
  --schema   ASTRUM_SCHEMA             (default: astrum.capnp relative to this file)
"""

import argparse
import math
import os
import sys

import capnp
import zenoh


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
        "crates",
        "astrum-proto",
        "schema",
        "astrum.capnp",
    )


def parse_args():
    """CLI flags, each falling back to its old ASTRUM_* env var, then a default
    (so precedence is CLI arg > env > default, and compose keeps working)."""
    p = argparse.ArgumentParser(
        description="ASTRUM example flight controller (Python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="target heading in degrees (0 = due north)",
    )
    p.add_argument(
        "--alt",
        type=float,
        default=env_f64("ASTRUM_TARGET_ALT_M", 1500.0),
        help="target MSL altitude (m)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=env_f64("ASTRUM_TARGET_SPD_MPS", 50.0),
        help="target airspeed (m/s)",
    )
    p.add_argument(
        "--mode",
        choices=["raw", "autopilot"],
        default=env_str("ASTRUM_CONTROL_MODE", "raw"),
        help="control mode",
    )
    p.add_argument(
        "--schema",
        default=env_str("ASTRUM_SCHEMA", default_schema_path()),
        help="path to astrum.capnp",
    )
    return p.parse_args()


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class HoldLaw:
    """Raw-actuator heading + altitude + airspeed hold loop.

    Same control signs verified against the c172x: aileron+ rolls right,
    elevator+ pitches down. Roll/pitch rate damping uses attitude differences
    between telemetry frames (the stream carries no body rates) — identical to
    the C++ controller's HoldLaw.
    """

    def __init__(self, target_hdg, target_alt, target_spd):
        self.target_hdg = target_hdg
        self.target_alt = target_alt
        self.target_spd = target_spd
        self.have_prev = False
        self.prev_phi = 0.0
        self.prev_theta = 0.0
        self.prev_t = 0.0

    def compute(self, phi, theta, psi, alt, airspeed, now):
        p_est = 0.0
        q_est = 0.0
        if self.have_prev and now > self.prev_t:
            dt = now - self.prev_t
            p_est = (phi - self.prev_phi) / dt
            q_est = (theta - self.prev_theta) / dt

        max_roll = math.radians(25.0)
        max_pitch = math.radians(10.0)

        hdg_err = wrap_pi(self.target_hdg - psi)
        roll_target = clamp(1.2 * hdg_err, -max_roll, max_roll)
        aileron = clamp(1.5 * (roll_target - phi) - 0.3 * p_est, -1.0, 1.0)

        pitch_target = clamp(0.01 * (self.target_alt - alt), -max_pitch, max_pitch)
        elevator = clamp(-2.0 * (pitch_target - theta) + 0.5 * q_est, -1.0, 1.0)

        throttle = clamp(0.6 + 0.02 * (self.target_spd - airspeed), 0.0, 1.0)

        self.prev_phi = phi
        self.prev_theta = theta
        self.prev_t = now
        self.have_prev = True
        return aileron, elevator, throttle


def main():
    args = parse_args()
    connect = args.connect
    target_id = args.id
    target_hdg = math.radians(args.heading)
    target_alt = args.alt
    target_spd = args.speed
    mode = args.mode

    telemetry_key = f"astrum/telemetry/{target_id}"
    command_key = f"astrum/command/{target_id}"

    # Load the shared schema at runtime (no codegen step) — same source of truth
    # the Rust sim and the C++ controller compile against.
    astrum = capnp.load(args.schema)

    # Peer session onto the sim's single Zenoh endpoint. Multicast scouting is
    # off — Docker bridge networks don't pass it, so peers connect explicitly.
    # Peer mode makes open non-fatal when the sim isn't listening yet (it spends
    # ~10s loading JSBSim before it binds): open returns immediately and Zenoh
    # retries the connection in the background, so a controller started alongside
    # the sim just attaches once it comes up — mirroring ZMQ's lazy connect. A
    # client would instead abort open() on a cold start. listen/endpoints=[] keeps
    # this peer purely connect-out (the link to the sim carries pub/sub both ways).
    conf = zenoh.Config()
    conf.insert_json5("mode", '"peer"')
    conf.insert_json5("connect/endpoints", f'["{connect}"]')
    conf.insert_json5("listen/endpoints", "[]")
    conf.insert_json5("scouting/multicast/enabled", "false")

    with zenoh.open(conf) as session:
        pub = session.declare_publisher(command_key)

        print(
            f"[example.py] target={target_id} mode={mode} connect={connect} "
            f"telemetry={telemetry_key} command={command_key}",
            file=sys.stderr,
            flush=True,
        )

        law = HoldLaw(target_hdg, target_alt, target_spd)
        n = 0

        # The default subscriber handler is a FIFO channel; iterating blocks for
        # the next telemetry sample, so there's no busy-poll.
        sub = session.declare_subscriber(telemetry_key)
        for sample in sub:
            payload = bytes(sample.payload)

            with astrum.Telemetry.from_bytes(payload) as tel:
                att = tel.attitude
                phi = att.phiRad
                theta = att.thetaRad
                psi = att.psiRad
                alt = tel.geo.altM
                airspeed = tel.airspeedMps
                now = tel.simTimeS

            if mode != "raw":
                n += 1
                # Autopilot mode: reassert the one high-level target over the first
                # second of telemetry, then just monitor. Receiving telemetry proves
                # the bidirectional peer link is up, so the command now reaches the
                # sim; resending briefly guarantees delivery despite the startup race
                # — Zenoh drops a put with no matching subscriber (unlike ZMQ PUSH,
                # which queues it), so a single shot at t=0 would be lost while the
                # sim spends ~10 s loading JSBSim before it subscribes.
                if n <= 120:
                    msg = astrum.Command.new_message()
                    msg.targetId = target_id
                    ap = msg.control.init("autopilot")
                    ap.headingRad = target_hdg
                    ap.altMslM = target_alt
                    ap.airspeedMps = target_spd
                    pub.put(msg.to_bytes())
                if n % 120 == 0:
                    print(
                        f"[example.py] t={now:.1f}s hdg={math.degrees(psi):.1f}deg alt={alt:.0f}m",
                        file=sys.stderr,
                        flush=True,
                    )
                continue

            aileron, elevator, throttle = law.compute(phi, theta, psi, alt, airspeed, now)

            msg = astrum.Command.new_message()
            msg.targetId = target_id
            raw = msg.control.init("raw")
            raw.aileron = aileron
            raw.elevator = elevator
            raw.rudder = 0.0
            raw.throttle = throttle
            pub.put(msg.to_bytes())

            n += 1
            if n % 120 == 0:
                print(
                    f"[example.py] t={now:.1f}s hdg={math.degrees(psi):.1f}deg "
                    f"alt={alt:.0f}m ail={aileron:.3f} elev={elevator:.3f} thr={throttle:.3f}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    main()
