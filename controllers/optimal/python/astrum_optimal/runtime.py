"""astrum_optimal runtime: the clean API for control-law development.

You write a :class:`Law` in ``astrum_optimal/laws/<name>.py`` mapping a decoded
:class:`~astrum_client.wire.VehicleState` + :class:`Reference` to a control
(:class:`~astrum_client.wire.RawActuators` or
:class:`~astrum_client.wire.AutopilotTargets`); the runtime (built on
:mod:`astrum_client`) owns the Zenoh session, latest-value telemetry, command
QoS/seq/telemetryTick, saturation, the vehicle-kind check and the timeouts::

    python controllers/optimal/python/run.py --law lqr --id 1
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import pkgutil
import sys
from dataclasses import dataclass
from typing import Protocol

from astrum_client import cli, transport
from astrum_client.loop import RefuseToFly, run_control_loop
from astrum_client.wire import Control, VehicleState, load_schema


@dataclass(frozen=True)
class Reference:
    """The commanded setpoint (from the CLI) a law tracks."""

    heading_rad: float  # true heading
    alt_m: float  # HAE
    airspeed_mps: float  # true airspeed


class Fleet(Protocol):
    """Read access to the newest telemetry of any aircraft the runtime sees."""

    def latest(self, aircraft_id: int) -> VehicleState | None: ...


class Law:
    """Base class for a control law; expose the subclass as ``Law`` in its module.

    * ``vehicle_kinds``: the :class:`~astrum_client.wire.VehicleKind` values the
      law was written for; the runtime refuses (exit 2) any other kind,
      including values unknown to this schema.
    * ``needs_fleet``: subscribe to every aircraft's telemetry (the ``fleet``
      argument of :meth:`control` then sees neighbours, e.g. a leader).
    * :meth:`configure` is called once with the control period before the first
      :meth:`control`; :meth:`control` runs once per telemetry tick.
    """

    vehicle_kinds: frozenset = frozenset()
    needs_fleet: bool = False

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def configure(self, control_period_s: float) -> None:
        """Design gains for the control period (s). Default: nothing to design."""

    def control(self, s: VehicleState, ref: Reference, fleet: Fleet) -> Control:
        raise NotImplementedError("a Law must implement control(state, ref, fleet)")


def law_names() -> list[str]:
    from astrum_optimal import laws

    return sorted(m.name for m in pkgutil.iter_modules(laws.__path__) if not m.name.startswith("_"))


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    """Flags every law shares (each falls back to its ``ASTRUM_*`` env var)."""
    cli.add_connection_args(p)
    cli.add_reference_args(p)
    cli.add_limit_args(p)
    p.add_argument(
        "--rate-hz",
        type=cli.positive_float,
        default=cli.env_default("ASTRUM_RATE_HZ", "120"),
        help="control rate the laws are discretized for: the rate this aircraft's "
        "telemetry arrives at (its control_rate_hz, default sim.rate_hz). Checked "
        "against the telemetry; a mismatch exits 2 [$ASTRUM_RATE_HZ]",
    )


class PeriodCheck:
    """Verifies the law's control period against the telemetry it flies on: the
    sim step (``simTimeS / tick``) must divide it, and samples must arrive one
    period apart (the aircraft's ``control_rate_hz``). A dropped sample only
    lengthens a gap, so the check passes on the first gap of exactly one period
    and refuses a shorter gap, or :attr:`MAX_LONG_GAPS` longer ones in a row."""

    MAX_LONG_GAPS = 8

    def __init__(self, period_s: float):
        self.period_s = period_s
        self.checked = False
        self._step_checked = False
        self._prev_t: float | None = None
        self._long_gaps = 0

    def __call__(self, s: VehicleState) -> None:
        if self.checked:
            return
        if not self._step_checked and s.tick > 0:
            step = s.t / s.tick
            n = self.period_s / step
            if n < 1.0 - 1e-6 or not math.isclose(n, round(n), rel_tol=1e-6):
                raise RefuseToFly(
                    f"the sim steps at {1.0 / step:.6g} Hz, which --rate-hz "
                    f"{1.0 / self.period_s:.6g} does not divide; pass the aircraft's "
                    "control_rate_hz (default sim.rate_hz)"
                )
            self._step_checked = True
        if self._prev_t is not None and s.t > self._prev_t:
            gap = s.t - self._prev_t
            if math.isclose(gap, self.period_s, rel_tol=1e-6):
                self.checked = self._step_checked
            elif gap < self.period_s:
                raise RefuseToFly(
                    f"telemetry arrives at {1.0 / gap:.6g} Hz but the law was discretized "
                    f"for --rate-hz {1.0 / self.period_s:.6g}"
                )
            else:
                self._long_gaps += 1
                if self._long_gaps >= self.MAX_LONG_GAPS:
                    raise RefuseToFly(
                        f"telemetry arrives every {gap:.6g} s ({1.0 / gap:.6g} Hz) but the "
                        f"law was discretized for --rate-hz {1.0 / self.period_s:.6g}; pass "
                        "the aircraft's control_rate_hz"
                    )
        self._prev_t = s.t


def parse_args(argv: list[str]) -> tuple[Law, argparse.Namespace]:
    """Parse the shared flags plus the selected law's; returns the constructed
    (not yet configured) law and the arguments."""
    names = law_names()
    law_default = cli.env_default("ASTRUM_LAW", "lqr")

    # Pre-parse only --law so the law's module can register its flags before
    # the real parser handles -h/--help.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--law", default=law_default)
    known, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(
        prog="run.py",
        description="ASTRUM optimal-control controller (Python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--law", choices=names, default=law_default, help="control law [$ASTRUM_LAW]")
    add_runtime_args(p)
    if known.law not in names:
        p.error(f"unknown --law {known.law!r} (choose from {', '.join(names)})")
    law_mod = importlib.import_module(f"astrum_optimal.laws.{known.law}")
    if hasattr(law_mod, "add_args"):
        law_mod.add_args(p)
    args = p.parse_args(argv)
    return law_mod.Law(args), args


def main(argv: list[str] | None = None) -> int:
    law, args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stderr)
    log = logging.getLogger(f"run.py:{args.law}")

    period = 1.0 / args.rate_hz
    law.configure(period)
    ref = Reference(math.radians(args.heading), args.alt, args.speed)
    schema = load_schema(args.schema)
    check_period = PeriodCheck(period)

    try:
        session = transport.open_session(args.connect, args.startup_timeout)
    except transport.SimUnreachable as e:
        log.error("%s", e)
        return 1
    with session:
        selection = None if law.needs_fleet else args.id
        telemetry, _sub = transport.subscribe_telemetry(session, schema, selection)
        publisher = transport.CommandPublisher(session, schema, args.id)
        log.info(
            "aircraft=%d connect=%s rate=%gHz ref: hdg=%.0fdeg alt=%.0fm tas=%.0fm/s",
            args.id,
            args.connect,
            args.rate_hz,
            args.heading,
            args.alt,
            args.speed,
        )

        def step(s: VehicleState) -> Control:
            check_period(s)
            return law.control(s, ref, telemetry)

        code = run_control_loop(
            telemetry,
            publisher,
            args.id,
            step,
            accepted_kinds=law.vehicle_kinds,
            startup_timeout_s=args.startup_timeout,
            telemetry_timeout_s=args.telemetry_timeout,
            log=log,
        )
        if telemetry.decode_errors:
            log.warning("skipped %d bad telemetry payloads", telemetry.decode_errors)
        return code
