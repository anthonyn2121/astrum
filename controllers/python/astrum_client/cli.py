"""Command-line helpers shared by the ASTRUM Python controllers.

Every flag falls back to an ``ASTRUM_*`` environment variable, then a default
(precedence: CLI arg > env > default), so scripts and run files can configure them.
"""

from __future__ import annotations

import argparse
import os

from astrum_client.wire import default_schema_path


def env_default(key: str, fallback: str) -> str:
    """``$key`` if set, else ``fallback`` — as a *string*, so argparse runs it
    through the flag's ``type`` exactly like a command-line value (env values
    get the same validation and error messages as flags)."""
    return os.environ.get(key, fallback)


def positive_float(text: str) -> float:
    v = float(text)
    if not v > 0.0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return v


def non_negative_float(text: str) -> float:
    v = float(text)
    if not v >= 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return v


def add_connection_args(p: argparse.ArgumentParser) -> None:
    """Session, aircraft and liveness flags every controller shares."""
    p.add_argument(
        "--connect",
        default=env_default("ASTRUM_ZENOH_CONNECT", "tcp/127.0.0.1:7447"),
        help="sim Zenoh endpoint (telemetry + commands) [$ASTRUM_ZENOH_CONNECT]",
    )
    p.add_argument(
        "--id",
        type=int,
        default=env_default("ASTRUM_TARGET_ID", "1"),
        help="aircraft id to subscribe to / command [$ASTRUM_TARGET_ID]",
    )
    p.add_argument(
        "--schema",
        default=env_default("ASTRUM_SCHEMA", default_schema_path()),
        help="path to astrum.capnp [$ASTRUM_SCHEMA]",
    )
    p.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=env_default("ASTRUM_STARTUP_TIMEOUT_S", "60"),
        help="exit 1 if no telemetry for --id arrives within this many seconds "
        "[$ASTRUM_STARTUP_TIMEOUT_S]",
    )
    p.add_argument(
        "--telemetry-timeout",
        type=positive_float,
        default=env_default("ASTRUM_TELEMETRY_TIMEOUT_S", "10"),
        help="exit 0 once telemetry has stopped for this many seconds "
        "[$ASTRUM_TELEMETRY_TIMEOUT_S]",
    )


def add_reference_args(p: argparse.ArgumentParser) -> None:
    """The heading / altitude / airspeed reference."""
    p.add_argument(
        "--heading",
        type=float,
        default=env_default("ASTRUM_TARGET_HEADING_DEG", "0"),
        help="target true heading, degrees (0 = north) [$ASTRUM_TARGET_HEADING_DEG]",
    )
    p.add_argument(
        "--alt",
        type=float,
        default=env_default("ASTRUM_TARGET_ALT_M", "1500"),
        help="target altitude, m above the WGS84 ellipsoid [$ASTRUM_TARGET_ALT_M]",
    )
    p.add_argument(
        "--speed",
        # >= 0: a multirotor's autopilot ignores the airspeed target (0 is the
        # natural value there); a raw fixed-wing law needs a sensible speed.
        type=non_negative_float,
        default=env_default("ASTRUM_TARGET_SPD_MPS", "50"),
        help="target true airspeed, m/s (ignored by multirotors) [$ASTRUM_TARGET_SPD_MPS]",
    )


def add_limit_args(p: argparse.ArgumentParser) -> None:
    """Attitude limits for raw control laws. Defaults suit the c172x trainer;
    the f16 (formation demo airframe) tolerates 60 / 24."""
    p.add_argument(
        "--max-bank-deg",
        type=positive_float,
        default=env_default("ASTRUM_MAX_BANK_DEG", "25"),
        help="bank-angle limit, degrees [$ASTRUM_MAX_BANK_DEG]",
    )
    p.add_argument(
        "--max-pitch-deg",
        type=positive_float,
        default=env_default("ASTRUM_MAX_PITCH_DEG", "10"),
        help="pitch-angle limit, degrees [$ASTRUM_MAX_PITCH_DEG]",
    )
