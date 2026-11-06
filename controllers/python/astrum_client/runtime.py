"""The law runtime (controllers/README.md, "Writing a controller"): a :class:`Law`
plus a table handed to :func:`run_main`::

    raise SystemExit(run_main({"hold": Hold}, prog="hold.py", description="..."))

:func:`run_main` parses the shared flags (:mod:`astrum_client.cli`) plus the
selected law's own, opens the one Zenoh session, subscribes, publishes, runs
:func:`astrum_client.loop.run_control_loop` and returns the exit code.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from astrum_client import cli, transport
from astrum_client.loop import LoopOptions, StepInput, Stop, exit_code, run_control_loop
from astrum_client.wire import Action, VehicleKind, enum_name

__all__ = ["AttitudeLimits", "Law", "Reference", "StepInput", "Stop", "check_laws", "run_main"]


@dataclass(frozen=True)
class Reference:
    """The commanded setpoint from ``--heading`` / ``--alt`` / ``--speed``."""

    heading_rad: float  # true heading
    alt_hae_m: float
    tas_mps: float  # true airspeed


@dataclass(frozen=True)
class AttitudeLimits:
    """``--max-bank-deg`` / ``--max-pitch-deg``, each in (0, 90) degrees."""

    max_bank_deg: float
    max_pitch_deg: float


class Law:
    """Base class of a control law.

    * ``vehicle_kinds``: the vehicle classes it can fly (required, non-empty);
      any other kind, including one this schema doesn't know, is refused.
    * ``needs_fleet``: subscribe to every aircraft (``StepInput.fleet``).
    * :meth:`add_args` / :meth:`from_args`: law-specific flags and construction;
      a ``ValueError`` from :meth:`from_args` is a usage error (exit 2). The
      parsed ``args`` carry ``reference`` and ``limits``.
    * :meth:`configure` designs gains for the control period, once, before the
      first :meth:`step`; :meth:`step` runs once per telemetry tick and may
      ``raise Stop(reason)`` to refuse.
    """

    vehicle_kinds: ClassVar[frozenset[VehicleKind]] = frozenset()
    needs_fleet: ClassVar[bool] = False

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register law-specific flags. Default: none."""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Law:
        """Construct the law from the parsed flags. Default: no parameters."""
        return cls()

    def configure(self, control_period_s: float) -> None:
        """Design gains for the control period (s). Default: nothing to design."""

    def step(self, inp: StepInput) -> Action:
        raise NotImplementedError(f"{type(self).__name__} must implement step(inp)")


def check_laws(laws: Mapping[str, type[Law]]) -> None:
    """Reject a malformed law table (a programming error, not a usage error)."""
    if not laws:
        raise ValueError("the law table is empty")
    for name, law in laws.items():
        if not (isinstance(law, type) and issubclass(law, Law)):
            raise TypeError(f"law {name!r} is {law!r}, not a Law subclass")
        if not law.vehicle_kinds:
            raise ValueError(f"law {name!r} ({law.__name__}) declares no vehicle_kinds")


def _parse(
    laws: Mapping[str, type[Law]], argv: Sequence[str], prog: str, description: str
) -> tuple[Law, argparse.Namespace]:
    """Parse the shared flags plus the selected law's and construct the law;
    usage errors exit 2 through argparse."""
    names = list(laws)
    law_default = cli.env_default("ASTRUM_LAW", names[0])
    # Pre-parse only --law so the law's flags are registered before the real
    # parser handles -h/--help.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--law", default=law_default)
    known, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--law", choices=names, default=law_default, help="law to run [$ASTRUM_LAW]")
    cli.add_connection_args(p)
    cli.add_reference_args(p)
    cli.add_limit_args(p)
    if known.law not in laws:
        p.error(f"unknown --law {known.law!r} (choose from {', '.join(names)})")
    law_cls = laws[known.law]
    law_cls.add_args(p)
    args = p.parse_args(argv)
    args.reference = Reference(math.radians(args.heading), args.alt, args.speed)
    args.limits = AttitudeLimits(args.max_bank_deg, args.max_pitch_deg)
    try:
        law = law_cls.from_args(args)
    except (ValueError, argparse.ArgumentTypeError) as e:
        p.error(f"--law {args.law}: {e}")
    return law, args


def run_main(
    laws: Mapping[str, type[Law]],
    argv: Sequence[str] | None = None,
    *,
    prog: str,
    description: str,
) -> int:
    """Run the law selected by ``--law`` from ``laws`` (default: the first) on
    ``--id``; returns the process exit code (controllers/README.md, "Outcomes")."""
    check_laws(laws)
    try:
        law, args = _parse(laws, sys.argv[1:] if argv is None else list(argv), prog, description)
    except SystemExit as e:  # usage error (2) or --help (0)
        return e.code if isinstance(e.code, int) else 2

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
    log = logging.getLogger(f"astrum.{prog.removesuffix('.py')}")
    deadline = time.monotonic() + args.startup_timeout
    log.info(
        "aircraft=%d law=%s kinds=%s connect=%s ref: hdg=%.0fdeg alt=%.0fm tas=%.0fm/s",
        args.id,
        args.law,
        ",".join(sorted(enum_name(k) for k in law.vehicle_kinds)),
        args.connect,
        args.heading,
        args.alt,
        args.speed,
    )
    try:
        session = transport.open_session(args.connect, deadline)
    except transport.SimUnreachable as e:
        log.error("%s", e)
        return 1
    with session:
        telemetry = transport.TelemetrySubscriber(session, None if law.needs_fleet else args.id)
        events = transport.EventSubscriber(session, law.needs_fleet)
        publisher = transport.CommandPublisher(session, args.id)
        outcome = run_control_loop(
            telemetry,
            events,
            publisher,
            law,
            LoopOptions(
                aircraft_id=args.id,
                startup_deadline=deadline,
                telemetry_timeout_s=args.telemetry_timeout,
                logger=log,
            ),
        )
        for what, n in (("telemetry", telemetry.decode_errors), ("event", events.decode_errors)):
            if n:
                log.warning("skipped %d bad %s payloads", n, what)
        return exit_code(outcome)
