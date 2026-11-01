#!/usr/bin/env python3
"""ASTRUM example flight controller (Python).

A minimal out-of-sim controller that talks to ``astrum-sim`` purely over Zenoh
with the shared Cap'n Proto schema — the Python twin of ``controllers/cpp``.

  * SUB  astrum/telemetry/<id>  latest-value telemetry for one aircraft
  * PUB  astrum/command/<id>    one command per telemetry tick (seq + telemetryTick)

Modes (``--mode`` / ``ASTRUM_CONTROL_MODE``):
  * ``raw`` (default): closes a heading/altitude/airspeed hold loop here
    (:class:`astrum_client.hold.HoldLaw`) and sends raw actuators every tick.
    Raw commands are a lease: if they stop, the sim's failsafe takes over.
    Fixed-wing only; any other vehicle kind is refused (exit 2).
  * ``autopilot``: sends the target to the sim's inner loop every tick (which
    also keeps the lease fresh). Accepts fixed-wing and multirotor aircraft.

Exits 1 if no telemetry arrives within ``--startup-timeout``, 0 once telemetry
stops for ``--telemetry-timeout``. Run with ``--help`` for every flag; each falls
back to its ``ASTRUM_*`` env var (CLI arg > env > default).
"""

import argparse
import logging
import math
import sys

from astrum_client import cli, transport
from astrum_client.hold import HoldLaw
from astrum_client.loop import run_control_loop
from astrum_client.wire import AutopilotTargets, VehicleKind, load_schema


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ASTRUM example flight controller (Python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_connection_args(p)
    cli.add_reference_args(p)
    cli.add_limit_args(p)
    p.add_argument(
        "--mode",
        choices=["raw", "autopilot"],
        default=cli.env_default("ASTRUM_CONTROL_MODE", "raw"),
        help="control mode [$ASTRUM_CONTROL_MODE]",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
    log = logging.getLogger("example.py")

    schema = load_schema(args.schema)
    heading = math.radians(args.heading)
    if args.mode == "raw":
        law = HoldLaw(
            heading,
            args.alt,
            args.speed,
            math.radians(args.max_bank_deg),
            math.radians(args.max_pitch_deg),
        )
        step = law.compute
        kinds = {VehicleKind.FIXED_WING}
    else:
        targets = AutopilotTargets(heading, args.alt, args.speed)

        def step(_s):
            return targets

        kinds = {VehicleKind.FIXED_WING, VehicleKind.MULTIROTOR}

    try:
        session = transport.open_session(args.connect, args.startup_timeout)
    except transport.SimUnreachable as e:
        log.error("%s", e)
        return 1
    with session:
        telemetry, _sub = transport.subscribe_telemetry(session, schema, args.id)
        publisher = transport.CommandPublisher(session, schema, args.id)
        log.info(
            "aircraft=%d mode=%s connect=%s ref: hdg=%.0fdeg alt=%.0fm tas=%.0fm/s",
            args.id,
            args.mode,
            args.connect,
            args.heading,
            args.alt,
            args.speed,
        )
        code = run_control_loop(
            telemetry,
            publisher,
            args.id,
            step,
            accepted_kinds=kinds,
            startup_timeout_s=args.startup_timeout,
            telemetry_timeout_s=args.telemetry_timeout,
            log=log,
        )
        if telemetry.decode_errors:
            log.warning("skipped %d bad telemetry payloads", telemetry.decode_errors)
        return code


if __name__ == "__main__":
    sys.exit(main())
