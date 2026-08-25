"""figure8 — leader guidance: fly a horizontal figure-eight (lemniscate).

The leader of a formation. It emits high-level autopilot targets
(heading/alt/airspeed); the sim's fixed-wing inner loop closes to the control
surfaces. The ground track is a Gerono lemniscate centred at
``(--fig-center-e, --fig-center-n)`` in the ENU grid::

    E(tau) = center_e + Rx * sin(tau)
    N(tau) = center_n + Ry * sin(tau) * cos(tau)

— width ``2*Rx`` along East, two lobes of height ``Ry`` along North. Each frame
it commands heading along the path tangent plus a cross-track term that pulls the
aircraft back onto the curve, and holds ``--alt`` / ``--speed``. This is pure
guidance: no MPC, no neighbours — the followers (``laws/formation.py``) key off
*its* broadcast telemetry.

**Speed-synced carrot.** The path parameter ``tau`` is not advanced by wall clock
but by *arc length flown*: each frame ``tau`` steps by ``ground_speed * dt /
|dP/dtau|`` so the reference point travels along the curve at exactly the
aircraft's own ground speed. This matters because the sim's P-only throttle loop
holds a speed offset from the command (a ~150 m/s command can settle ~20 m/s
high), so a fixed-period carrot would out- or under-run the jet and the pattern
would smear. Pacing by the measured speed keeps the carrot glued to the aircraft
whatever it actually flies.

Size the pattern for the vehicle. The tightest curvature (the lobe ends, radius
~Ry) sets the peak bank, ~atan(V^2 / (g*R)). The defaults (Rx=8 km, Ry=4 km)
drive an F-16 flying ~120 m/s to ~40 deg of bank at the lobes — aggressive, but
comfortably inside the f16 inner loop's 60 deg limit (which also feeds the turn's
load factor forward to hold altitude). Scale Rx/Ry together to keep the two-lobe
shape; for a gentler airframe (the c172x's 25 deg limit) grow the pattern so the
lobe radius stays above the min turn radius V^2/(g*tan(limit)).
"""

import math

from astrum_control import Command, Reference, VehicleState
from astrum_control import Law as BaseLaw


def add_args(p):
    """figure-eight geometry + guidance flags."""
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
    """Speed-synced lemniscate follower. See module docstring."""

    def __init__(self, args):
        super().__init__(args)
        self.rx = args.fig_rx
        self.ry = args.fig_ry
        self.ce = args.fig_center_e
        self.cn = args.fig_center_n
        self.kxt = args.fig_xtrack
        self.speed = args.speed
        self.alt = args.alt
        self.tau = 0.0  # path parameter; advanced by arc length flown
        self.prev_t = None

    def control(self, s: VehicleState, ref: Reference) -> Command:
        # Ground speed (no wind => ~= airspeed) both paces the carrot and scales
        # the tangential feed-forward, so the reference tracks the *actual* speed
        # the throttle loop settles at rather than the commanded one.
        gs = math.hypot(s.v_e, s.v_n)
        sin, cos = math.sin(self.tau), math.cos(self.tau)

        # Advance tau so the reference point moves along the curve at ground
        # speed: d(arc)/dt = gs and d(arc)/dtau = |dP/dtau|, so dtau = gs*dt/|dP/dtau|.
        de = self.rx * cos  # dP/dtau
        dn = self.ry * (cos * cos - sin * sin)  # = ry * cos(2*tau)
        speed_dpdtau = math.hypot(de, dn) or 1.0
        if self.prev_t is not None:
            dt = s.t - self.prev_t
            if dt > 0.0:
                self.tau += gs * dt / speed_dpdtau
        self.prev_t = s.t

        # Reference point on the lemniscate and its unit tangent (recomputed at
        # the freshly advanced tau).
        sin, cos = math.sin(self.tau), math.cos(self.tau)
        ref_e = self.ce + self.rx * sin
        ref_n = self.cn + self.ry * sin * cos
        de = self.rx * cos
        dn = self.ry * (cos * cos - sin * sin)
        norm = math.hypot(de, dn) or 1.0
        te, tn = de / norm, dn / norm

        # Commanded velocity = ground speed along the tangent + a pull back onto
        # the path; heading is its direction (ENU: 0 = North, +East clockwise).
        # Airspeed target stays the operator's --speed (the throttle loop's job).
        v_e = gs * te + self.kxt * (ref_e - s.e)
        v_n = gs * tn + self.kxt * (ref_n - s.n)
        heading = math.atan2(v_e, v_n)
        return Command.autopilot(heading, self.alt, self.speed)
