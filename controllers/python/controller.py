#!/usr/bin/env python3
"""FORGE example flight controller (Python).

A minimal out-of-sim controller that talks to ``forge-sim`` purely over ZMQ,
using the shared Cap'n Proto schema (``forge.capnp``) for the wire format. It is
the Python twin of ``controllers/cpp`` — same endpoints, same env config, same
hold-law gains, same framing — proving a controller in any language plugs into
FORGE with nothing but pyzmq + pycapnp.

  * SUB  tcp://sim:5555 — telemetry for one aircraft (topic = 4-byte BE id)
  * PUSH tcp://sim:5556 — commands (target id + raw actuators | autopilot)

Two modes (env ``FORGE_CONTROL_MODE``):
  * "raw" (default) — closes a heading/altitude/airspeed hold loop *here*, in
    the controller, sending raw ``fcs/*-cmd-norm`` deflections each tick. This
    is FORGE's intended usage: the user owns the control law.
  * "autopilot" — sends a single high-level target command and then just
    monitors, letting the sim's built-in inner loop fly it.

Config (env), matching the C++ controller so the compose wiring is shared:
  FORGE_TELEMETRY_CONNECT  (default tcp://127.0.0.1:5555)
  FORGE_COMMAND_CONNECT    (default tcp://127.0.0.1:5556)
  FORGE_TARGET_ID          (default 1)
  FORGE_TARGET_HEADING_DEG (default 0    — due north)
  FORGE_TARGET_ALT_M       (default 1500)
  FORGE_TARGET_SPD_MPS     (default 50)
  FORGE_CONTROL_MODE       (default raw)
  FORGE_SCHEMA             (default: forge.capnp relative to this file)
"""

import math
import os
import sys
import time

import capnp
import zmq


def env_str(key, fallback):
    return os.environ.get(key, fallback)


def env_f64(key, fallback):
    v = os.environ.get(key)
    return float(v) if v is not None else fallback


def topic_for(aircraft_id):
    """4-byte big-endian aircraft id — matches forge-net's topic frame."""
    return int(aircraft_id).to_bytes(4, "big")


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
    tlm_endpoint = env_str("FORGE_TELEMETRY_CONNECT", "tcp://127.0.0.1:5555")
    cmd_endpoint = env_str("FORGE_COMMAND_CONNECT", "tcp://127.0.0.1:5556")
    target_id = int(env_f64("FORGE_TARGET_ID", 1))
    target_hdg = math.radians(env_f64("FORGE_TARGET_HEADING_DEG", 0.0))
    target_alt = env_f64("FORGE_TARGET_ALT_M", 1500.0)
    target_spd = env_f64("FORGE_TARGET_SPD_MPS", 50.0)
    mode = env_str("FORGE_CONTROL_MODE", "raw")

    # Load the shared schema at runtime (no codegen step) — same source of truth
    # the Rust sim and the C++ controller compile against.
    default_schema = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "crates", "forge-proto", "schema", "forge.capnp",
    )
    schema_path = env_str("FORGE_SCHEMA", default_schema)
    forge = capnp.load(schema_path)

    ctx = zmq.Context.instance()

    tlm = ctx.socket(zmq.SUB)
    tlm.connect(tlm_endpoint)
    tlm.setsockopt(zmq.SUBSCRIBE, topic_for(target_id))

    cmd = ctx.socket(zmq.PUSH)
    cmd.connect(cmd_endpoint)

    print(
        f"[controller.py] target={target_id} mode={mode} "
        f"telemetry={tlm_endpoint} command={cmd_endpoint}",
        file=sys.stderr,
        flush=True,
    )

    # Autopilot mode: send one high-level target, then just monitor.
    if mode == "autopilot":
        msg = forge.Command.new_message()
        msg.targetId = target_id
        ap = msg.control.init("autopilot")
        ap.headingRad = target_hdg
        ap.altMslM = target_alt
        ap.airspeedMps = target_spd
        # PUSH connect is async; give it a moment to establish before the send.
        time.sleep(0.2)
        cmd.send(msg.to_bytes())
        print(
            f"[controller.py] sent autopilot target hdg={target_hdg:.3f}rad "
            f"alt={target_alt}m spd={target_spd}m/s",
            file=sys.stderr,
            flush=True,
        )

    law = HoldLaw(target_hdg, target_alt, target_spd)
    n = 0

    while True:
        # PUB frames arrive as [topic, payload]; the payload is the last frame.
        frames = tlm.recv_multipart()
        payload = frames[-1]

        with forge.Telemetry.from_bytes(payload) as tel:
            att = tel.attitude
            phi = att.phiRad
            theta = att.thetaRad
            psi = att.psiRad
            alt = tel.geo.altM
            airspeed = tel.airspeedMps
            now = tel.simTimeS

        if mode != "raw":
            n += 1
            if n % 120 == 0:
                print(
                    f"[controller.py] t={now:.1f}s hdg={math.degrees(psi):.1f}deg "
                    f"alt={alt:.0f}m",
                    file=sys.stderr,
                    flush=True,
                )
            continue

        aileron, elevator, throttle = law.compute(phi, theta, psi, alt, airspeed, now)

        msg = forge.Command.new_message()
        msg.targetId = target_id
        raw = msg.control.init("raw")
        raw.aileron = aileron
        raw.elevator = elevator
        raw.rudder = 0.0
        raw.throttle = throttle
        cmd.send(msg.to_bytes())

        n += 1
        if n % 120 == 0:
            print(
                f"[controller.py] t={now:.1f}s hdg={math.degrees(psi):.1f}deg "
                f"alt={alt:.0f}m ail={aileron:.3f} elev={elevator:.3f} thr={throttle:.3f}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
