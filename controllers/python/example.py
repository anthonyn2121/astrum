#!/usr/bin/env python3
"""ASTRUM example flight controller (Python) — the twin of ``controllers/cpp``.

Laws (``--law`` / ``ASTRUM_LAW``):
  * ``hold`` (default): closes a heading / altitude / airspeed hold loop here
    (:class:`astrum_client.hold.HoldLaw`) and sends raw actuators every tick.
    Fixed-wing only.
  * ``autopilot``: hands the reference to the sim's inner loop every tick.
    Fixed-wing or multirotor.

The SDK contract (flags, control loop, exit codes) is controllers/README.md;
``--help`` lists every flag.
"""

import sys

from astrum_client.hold import HoldLaw
from astrum_client.runtime import Law, StepInput, run_main
from astrum_client.wire import Action, AutopilotTargets, VehicleKind


class AutopilotLaw(Law):
    """The reference as targets for the sim's autopilot."""

    vehicle_kinds = frozenset({VehicleKind.FIXED_WING, VehicleKind.MULTIROTOR})

    @classmethod
    def from_args(cls, args):
        return cls(args.reference)

    def __init__(self, ref):
        self.targets = AutopilotTargets(ref.heading_rad, ref.alt_hae_m, ref.tas_mps)

    def step(self, inp: StepInput) -> Action:
        return Action(self.targets)


LAWS = {"hold": HoldLaw, "autopilot": AutopilotLaw}

if __name__ == "__main__":
    sys.exit(
        run_main(LAWS, prog="example.py", description="ASTRUM example flight controller (Python).")
    )
