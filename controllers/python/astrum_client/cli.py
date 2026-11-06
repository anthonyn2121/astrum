"""Command-line flags shared by the ASTRUM Python controllers (controllers/README.md,
"Flags").

Every flag falls back to an ``ASTRUM_*`` environment variable, then a default
(precedence: CLI arg > env > default). Env values are passed through the flag's
``type`` exactly like command-line values, so they get the same validation.
Numbers must be finite.
"""

from __future__ import annotations

import argparse
import math
import os

MAX_AIRCRAFT_ID = 0xFFFFFFFF


def env_default(key: str, fallback: str) -> str:
    """``$key`` if set, else ``fallback`` — as a *string*, so argparse runs it
    through the flag's ``type`` exactly like a command-line value."""
    return os.environ.get(key, fallback)


def finite_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"must be finite, got {text}")
    return v


def positive_float(text: str) -> float:
    v = finite_float(text)
    if not v > 0.0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return v


def non_negative_float(text: str) -> float:
    v = finite_float(text)
    if not v >= 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return v


def attitude_limit_deg(text: str) -> float:
    v = finite_float(text)
    if not 0.0 < v < 90.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 90) degrees, got {text}")
    return v


def aircraft_id(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an aircraft id: {text!r}") from None
    if not 1 <= v <= MAX_AIRCRAFT_ID:
        raise argparse.ArgumentTypeError(f"must be in 1..{MAX_AIRCRAFT_ID}, got {text}")
    return v


def add_connection_args(p: argparse.ArgumentParser) -> None:
    """Session, aircraft and liveness flags every controller shares."""
    p.add_argument(
        "--connect",
        default=env_default("ASTRUM_ZENOH_CONNECT", "tcp/127.0.0.1:7447"),
        help="sim Zenoh endpoint [$ASTRUM_ZENOH_CONNECT]",
    )
    target = os.environ.get("ASTRUM_TARGET_ID")
    p.add_argument(
        "--id",
        type=aircraft_id,
        default=target,
        required=target is None,
        help=f"aircraft to fly, 1..{MAX_AIRCRAFT_ID} (required) [$ASTRUM_TARGET_ID]",
    )
    p.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=env_default("ASTRUM_STARTUP_TIMEOUT_S", "60"),
        help="one deadline (s) for reaching the sim and its first sample; exit 1 "
        "[$ASTRUM_STARTUP_TIMEOUT_S]",
    )
    p.add_argument(
        "--telemetry-timeout",
        type=positive_float,
        default=env_default("ASTRUM_TELEMETRY_TIMEOUT_S", "10"),
        help="telemetry silence (s, without RunEnded) that ends the run; exit 1 "
        "[$ASTRUM_TELEMETRY_TIMEOUT_S]",
    )


def add_reference_args(p: argparse.ArgumentParser) -> None:
    """The heading / altitude / airspeed reference."""
    p.add_argument(
        "--heading",
        type=finite_float,
        default=env_default("ASTRUM_TARGET_HEADING_DEG", "0"),
        help="reference true heading, degrees (0 = north) [$ASTRUM_TARGET_HEADING_DEG]",
    )
    p.add_argument(
        "--alt",
        type=finite_float,
        default=env_default("ASTRUM_TARGET_ALT_M", "1500"),
        help="reference altitude, m above the WGS84 ellipsoid [$ASTRUM_TARGET_ALT_M]",
    )
    p.add_argument(
        "--speed",
        # >= 0: a multirotor's autopilot ignores the airspeed target (0 is the
        # natural value there); a raw fixed-wing law needs a sensible speed.
        type=non_negative_float,
        default=env_default("ASTRUM_TARGET_SPD_MPS", "50"),
        help="reference true airspeed, m/s (ignored by multirotors) [$ASTRUM_TARGET_SPD_MPS]",
    )


def add_limit_args(p: argparse.ArgumentParser) -> None:
    """Attitude limits for raw control laws. Defaults suit the c172x trainer;
    the f16 (formation demo airframe) tolerates 60 / 24."""
    p.add_argument(
        "--max-bank-deg",
        type=attitude_limit_deg,
        default=env_default("ASTRUM_MAX_BANK_DEG", "25"),
        help="bank-angle limit, degrees, in (0, 90) [$ASTRUM_MAX_BANK_DEG]",
    )
    p.add_argument(
        "--max-pitch-deg",
        type=attitude_limit_deg,
        default=env_default("ASTRUM_MAX_PITCH_DEG", "10"),
        help="pitch-angle limit, degrees, in (0, 90) [$ASTRUM_MAX_PITCH_DEG]",
    )
