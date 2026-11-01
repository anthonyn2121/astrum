"""figure8 — leader guidance: fly a horizontal figure-eight (Gerono lemniscate).

The leader of a formation. It emits high-level autopilot targets
(heading/alt/airspeed) and the sim's fixed-wing inner loop closes to the
control surfaces. The ground track, in the scenario's ENU grid, is::

    E(tau) = center_e + Rx * sin(tau)
    N(tau) = center_n + Ry * sin(tau) * cos(tau)

— width ``2*Rx`` along East, two lobes of height ``Ry`` along North. Each tick
it commands the path tangent plus a cross-track pull back onto the curve, at
``--alt`` and ``--speed``. Pure guidance: no MPC, no neighbours — the followers
(``formation``) key off *its* broadcast telemetry.

**Carrot pacing.** The path parameter ``tau`` advances by the arc length the
aircraft actually flew (``ground_speed * dt / |dP/dtau|``), so the reference
point stays abreast of the aircraft whatever speed it really makes good
(airspeed tracking error, wind, speed loss in hard turns) instead of running
away from or lagging behind it as a clock-driven carrot would.

**Heading.** The tangent + pull-back is a desired ground *track*; it is sent as
the autopilot's true heading (grid convergence applied), which assumes no wind
and coordinated flight — see :mod:`astrum_optimal.laws._guidance`.

Size the pattern for the vehicle: the tightest curvature is at the lobe ends
(``tau = ±pi/2``), radius ``Ry^2 / Rx``, which sets the peak bank
~atan(V^2 / (g*R)). The defaults (Rx=8 km, Ry=4 km, so R=2 km) take an F-16 at
120 m/s (the formation demo's speed) to ~36 deg of bank, inside the fighter
autopilot's limit; for the c172x grow the pattern so that radius stays above its
minimum turn radius V^2/(g*tan(limit)).
"""

from __future__ import annotations

import math

from astrum_client.mathutil import integration_step
from astrum_client.wire import AutopilotTargets, VehicleKind, VehicleState

from astrum_optimal.laws._guidance import true_heading_for_grid_track
from astrum_optimal.runtime import Fleet, Reference
from astrum_optimal.runtime import Law as BaseLaw


def add_args(p) -> None:
    p.add_argument("--fig-rx", type=float, default=8000.0, help="half-width along East (m)")
    p.add_argument("--fig-ry", type=float, default=4000.0, help="lobe height along North (m)")
    p.add_argument("--fig-center-e", type=float, default=0.0, help="pattern centre, ENU East (m)")
    p.add_argument("--fig-center-n", type=float, default=0.0, help="pattern centre, ENU North (m)")
    p.add_argument(
        "--fig-xtrack",
        type=float,
        default=0.05,
        help="cross-track gain (1/s): commanded pull-back speed per metre off the path",
    )


class Law(BaseLaw):
    """Speed-paced lemniscate follower (fixed-wing only)."""

    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})

    def __init__(self, args):
        super().__init__(args)
        self.rx = args.fig_rx
        self.ry = args.fig_ry
        self.ce = args.fig_center_e
        self.cn = args.fig_center_n
        self.kxt = args.fig_xtrack
        self.tau = 0.0  # path parameter; advanced by arc length flown
        self._prev_t: float | None = None

    def _tangent(self) -> tuple[float, float]:
        """dP/dtau at the current tau."""
        return self.rx * math.cos(self.tau), self.ry * math.cos(2.0 * self.tau)

    def control(self, s: VehicleState, ref: Reference, fleet: Fleet) -> AutopilotTargets:
        gs = math.hypot(s.vel_enu_e, s.vel_enu_n)
        dt = integration_step(self._prev_t, s.t)
        self._prev_t = s.t
        speed_dpdtau = math.hypot(*self._tangent()) or 1.0
        self.tau += gs * dt / speed_dpdtau

        sin, cos = math.sin(self.tau), math.cos(self.tau)
        ref_e = self.ce + self.rx * sin
        ref_n = self.cn + self.ry * sin * cos
        de, dn = self._tangent()
        norm = math.hypot(de, dn) or 1.0

        # Desired ground velocity: ground speed along the tangent plus a pull
        # back onto the path. Airspeed stays the operator's --speed.
        v_e = gs * de / norm + self.kxt * (ref_e - s.e)
        v_n = gs * dn / norm + self.kxt * (ref_n - s.n)
        heading = true_heading_for_grid_track(s, math.atan2(v_e, v_n))
        return AutopilotTargets(heading, ref.alt_m, ref.airspeed_mps)
