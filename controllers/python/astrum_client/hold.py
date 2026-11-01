"""The example raw-actuator hold law: heading, altitude and airspeed hold for a
fixed-wing aircraft (the Python twin of the C++ example's ``HoldLaw``).

Signs (verified on the c172x): aileron+ rolls right, elevator+ pitches nose
down, throttle+ speeds up. Roll/pitch damping uses the body rates the sim
publishes. Bank and pitch targets are limited, every actuator is saturated,
and the heading/altitude/airspeed integrators use conditional integration (they
freeze while their output is saturated in the error's direction) so they cannot
wind up during a large manoeuvre. The heading integral removes the steady error
a pure-P bank command leaves on an airframe that needs a standing aileron.
"""

from __future__ import annotations

from astrum_client.mathutil import clamp, integrate_conditionally, integration_step, wrap_pi
from astrum_client.wire import RawActuators, VehicleState


class HoldLaw:
    K_HDG = 1.2  # bank target per rad of heading error
    KI_HDG = 0.05  # bank target per rad*s of heading error
    K_ROLL = 1.5  # aileron per rad of bank error
    K_P = 0.3  # aileron per rad/s of roll rate
    K_ALT = 0.01  # pitch target per m of altitude error
    KI_ALT = 0.0005  # pitch target per m*s of altitude error
    K_PITCH = 2.0  # elevator per rad of pitch error
    K_Q = 0.5  # elevator per rad/s of pitch rate
    THROTTLE_TRIM = 0.6
    K_SPD = 0.02  # throttle per m/s of airspeed error
    KI_SPD = 0.005  # throttle per m of airspeed error integral

    def __init__(
        self,
        heading_rad: float,
        alt_m: float,
        airspeed_mps: float,
        max_bank_rad: float,
        max_pitch_rad: float,
    ):
        self.heading_rad = heading_rad
        self.alt_m = alt_m
        self.airspeed_mps = airspeed_mps
        self.max_bank = max_bank_rad
        self.max_pitch = max_pitch_rad
        self.i_hdg = 0.0  # bank-target contribution (rad)
        self.i_alt = 0.0  # pitch-target contribution (rad)
        self.i_spd = 0.0  # throttle contribution
        self._prev_t: float | None = None

    def compute(self, s: VehicleState) -> RawActuators:
        dt = integration_step(self._prev_t, s.t)
        self._prev_t = s.t

        hdg_err = wrap_pi(self.heading_rad - s.yaw)
        roll_unsat = self.K_HDG * hdg_err + self.i_hdg
        roll_target = clamp(roll_unsat, -self.max_bank, self.max_bank)
        aileron = self.K_ROLL * (roll_target - s.roll) - self.K_P * s.p

        alt_err = self.alt_m - s.alt_m
        pitch_unsat = self.K_ALT * alt_err + self.i_alt
        pitch_target = clamp(pitch_unsat, -self.max_pitch, self.max_pitch)
        elevator = -self.K_PITCH * (pitch_target - s.pitch) + self.K_Q * s.q

        spd_err = self.airspeed_mps - s.airspeed
        throttle_unsat = self.THROTTLE_TRIM + self.K_SPD * spd_err + self.i_spd

        self.i_hdg = integrate_conditionally(
            self.i_hdg,
            self.KI_HDG * hdg_err,
            dt,
            roll_unsat,
            -self.max_bank,
            self.max_bank,
            self.max_bank,
        )
        self.i_alt = integrate_conditionally(
            self.i_alt,
            self.KI_ALT * alt_err,
            dt,
            pitch_unsat,
            -self.max_pitch,
            self.max_pitch,
            self.max_pitch,
        )
        self.i_spd = integrate_conditionally(
            self.i_spd, self.KI_SPD * spd_err, dt, throttle_unsat, 0.0, 1.0, 1.0
        )
        return RawActuators(aileron, elevator, 0.0, throttle_unsat).saturated()
