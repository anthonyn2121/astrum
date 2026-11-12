"""mpc — receding-horizon control on the linear fixed-wing model.

Each channel predicts its error-state over ``--horizon-s`` seconds with the
model of :mod:`astrum_optimal.laws._fixed_wing`, discretized at the *control
period* (one prediction step per sim tick, so the plan's step is exactly the
time between re-plans), minimizes ``sum x'Qx + u'Ru``, and applies the first
move. Unconstrained, that first move is a constant linear gain
(:class:`~astrum_optimal.linear.ChannelMPC`), computed once in
:meth:`MpcLaw.design`; as the horizon grows it converges to the LQR gain.

Bank/pitch limits, actuator saturation and anti-windup are the same as
``lqr`` (see ``_fixed_wing``). To enforce bounds *inside* the horizon instead,
replace :meth:`ChannelMPC.move` with a QP (the constraint hook).
"""

from __future__ import annotations

from astrum_client import cli
from astrum_client.runtime import AttitudeLimits, Reference, Stop

from astrum_optimal.laws._fixed_wing import (
    Q_LAT,
    Q_LON,
    Q_SPD,
    R_LAT,
    R_LON,
    R_SPD,
    LimitedFixedWingLaw,
)
from astrum_optimal.linear import ChannelMPC, discretize_zoh


def horizon_steps(horizon_s: float, dt: float) -> int:
    """Prediction steps of ``horizon_s`` at control period ``dt``; a horizon
    shorter than one period refuses to fly."""
    n = round(horizon_s / dt)
    if n < 1:
        raise Stop(f"--horizon-s {horizon_s:g} is shorter than one control period ({dt:g} s)")
    return n


class MpcLaw(LimitedFixedWingLaw):
    """Limited unconstrained MPC (fixed-wing only)."""

    @classmethod
    def add_args(cls, p) -> None:
        p.add_argument(
            "--horizon-s",
            type=cli.positive_float,
            default=cli.env_default("ASTRUM_MPC_HORIZON_S", "6"),
            help="prediction horizon (s); steps = horizon-s / control period "
            "[$ASTRUM_MPC_HORIZON_S]",
        )

    @classmethod
    def from_args(cls, args):
        return cls(args.reference, args.limits, args.horizon_s)

    def __init__(self, ref: Reference, limits: AttitudeLimits, horizon_s: float = 6.0, model=None):
        super().__init__(ref, limits, model)
        self.horizon_s = horizon_s

    def design(self, dt: float, v: float) -> None:
        m = self.model
        n = horizon_steps(self.horizon_s, dt)
        self.k_lat = ChannelMPC(*discretize_zoh(*m.lateral(v), dt), Q_LAT, R_LAT, n).K[0]
        self.k_lon = ChannelMPC(*discretize_zoh(*m.longitudinal(v), dt), Q_LON, R_LON, n).K[0]
        self.k_spd = ChannelMPC(*discretize_zoh(*m.speed(), dt), Q_SPD, R_SPD, n).K[0]
