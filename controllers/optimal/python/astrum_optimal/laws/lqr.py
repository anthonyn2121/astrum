"""lqr — limited discrete LQR on the linear fixed-wing model.

Each channel's gain is ``dlqr`` of the model in
:mod:`astrum_optimal.laws._fixed_wing`, discretized (ZOH) at the control period;
that module also documents the model, the bank/pitch limiting and the
anti-windup integrators. Replace the placeholder airframe constants and the
``Q``/``R`` weights with your own design.
"""

from __future__ import annotations

from astrum_optimal.laws._fixed_wing import (
    Q_LAT,
    Q_LON,
    Q_SPD,
    R_LAT,
    R_LON,
    R_SPD,
    LimitedFixedWingLaw,
)
from astrum_optimal.linear import discretize_zoh, dlqr


class Law(LimitedFixedWingLaw):
    """Limited LQR state feedback (fixed-wing only)."""

    def design(self, dt: float, v: float) -> None:
        m = self.model
        self.k_lat = dlqr(*discretize_zoh(*m.lateral(v), dt), Q_LAT, R_LAT)[0][0]
        self.k_lon = dlqr(*discretize_zoh(*m.longitudinal(v), dt), Q_LON, R_LON)[0][0]
        self.k_spd = dlqr(*discretize_zoh(*m.speed(), dt), Q_SPD, R_SPD)[0][0]
