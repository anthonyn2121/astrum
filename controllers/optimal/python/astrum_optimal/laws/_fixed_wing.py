"""Shared fixed-wing machinery for the raw-actuator laws (``lqr``, ``mpc``).

**Linear model** about straight, level flight at the reference true airspeed
``V``. Errors are *measured minus reference*; small angles; rudder holds
coordination (no sideslip). Three decoupled channels:

    lateral       x = [psi_err, phi, p]               u = aileron
        psi_err' = (g/V) * phi           (coordinated turn: psi' = (g/V) tan(phi))
        phi'     = p
        p'       = -a_p p + b_p aileron  (aileron+ rolls right)

    longitudinal  x = [int(h_err), h_err, theta, q]   u = elevator
        h_err'   = V * theta             (climb rate; the integral absorbs trim alpha)
        theta'   = q
        q'       = -a_q q - b_q elevator (elevator+ pitches nose DOWN)

    speed         x = [int(v_err), v_err]             u = throttle - trim
        v_err'   = -a_v v_err + b_v (throttle - trim)

The continuous model is discretized exactly (zero-order hold) at the control
period — the sim step, since a command answers every telemetry tick. The
airframe constants below are placeholders for the c172x trainer; replace them
with your own linearization (JSBSim trim + finite differences).

**Limits.** A linear gain ``u = -K x`` has no notion of a bank or pitch limit
(90 deg of heading error would ask for ~100 deg of bank). Each attitude channel
is therefore split at its attitude state: the outer states define an attitude
reference ``att_ref = -(K_outer . x_outer) / K_att``, limited to
``±max-bank`` / ``±max-pitch``, and the inner loop applies
``u = -K_att (att - att_ref) - K_rate rate``. Unsaturated, this is *exactly*
``u = -K x``; saturated, it holds the limit attitude. Every actuator is then
saturated, and the heading/altitude/airspeed integrators use conditional
integration (frozen while their output is saturated in the error's direction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astrum_client.mathutil import clamp, integrate_conditionally, integration_step, wrap_pi
from astrum_client.wire import RawActuators, VehicleKind, VehicleState

from astrum_optimal.runtime import Fleet, Law, Reference

G = 9.80665
# Heading integral (rad bank per rad*s of heading error), outside the LQR/MPC
# design: it trims out the standing aileron a real airframe needs, which the
# placeholder model does not contain. Matches the C++ OuterLoopGains default.
KI_BANK = 0.05


@dataclass(frozen=True)
class FixedWingModel:
    """First-order rate dynamics per axis (placeholder c172x-like values)."""

    roll_damping: float = 2.0  # a_p (1/s)
    roll_authority: float = 4.0  # b_p (rad/s^2 per unit aileron)
    pitch_damping: float = 2.0  # a_q (1/s)
    pitch_authority: float = 4.0  # b_q (rad/s^2 per unit elevator, nose down)
    speed_damping: float = 0.05  # a_v (1/s)
    speed_authority: float = 2.0  # b_v (m/s^2 per unit throttle)
    throttle_trim: float = 0.6

    def lateral(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        A = np.array([[0.0, G / v, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -self.roll_damping]])
        B = np.array([[0.0], [0.0], [self.roll_authority]])
        return A, B

    def longitudinal(self, v: float) -> tuple[np.ndarray, np.ndarray]:
        A = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, v, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, -self.pitch_damping],
            ]
        )
        B = np.array([[0.0], [0.0], [0.0], [-self.pitch_authority]])
        return A, B

    def speed(self) -> tuple[np.ndarray, np.ndarray]:
        A = np.array([[0.0, 1.0], [0.0, -self.speed_damping]])
        B = np.array([[0.0], [self.speed_authority]])
        return A, B


# Design weights (per channel state order above; R on the single input). Tuned
# so the placeholder model lands near the hand-tuned example hold law.
Q_LAT = np.diag([4.0, 1.0, 0.0])
R_LAT = np.array([[1.0]])
Q_LON = np.diag([1e-6, 4e-4, 1.0, 0.0])
R_LON = np.array([[0.25]])
Q_SPD = np.diag([2.5e-5, 4e-4])
R_SPD = np.array([[1.0]])


class LimitedFixedWingLaw(Law):
    """Limited three-channel state feedback; subclasses set the gains in
    :meth:`design` (``self.k_lat`` [psi, phi, p], ``self.k_lon`` [int_h, h, theta, q],
    ``self.k_spd`` [int_v, v])."""

    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})

    def __init__(self, args, model: FixedWingModel | None = None):
        super().__init__(args)
        self.model = model or FixedWingModel()
        self.max_bank = math.radians(args.max_bank_deg)
        self.max_pitch = math.radians(args.max_pitch_deg)
        self.i_bank = 0.0  # heading integrator, as its bank-reference contribution (rad)
        self.i_pitch = 0.0  # altitude integrator, as its pitch-reference contribution (rad)
        self.i_throttle = 0.0  # airspeed integrator, as its throttle contribution
        self._prev_t: float | None = None
        self.k_lat = self.k_lon = self.k_spd = None

    def configure(self, control_period_s: float) -> None:
        self.design(control_period_s, self.args.speed)

    def design(self, dt: float, v: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def control(self, s: VehicleState, ref: Reference, fleet: Fleet) -> RawActuators:
        dt = integration_step(self._prev_t, s.t)
        self._prev_t = s.t

        # Lateral: heading PI -> limited bank reference -> aileron. The heading
        # integral removes the steady error a pure-P bank reference leaves when the
        # airframe needs a standing aileron (torque, asymmetry).
        k_psi, k_phi, k_p = self.k_lat
        psi_err = wrap_pi(s.yaw - ref.heading_rad)
        phi_unsat = self.i_bank - k_psi * psi_err / k_phi
        phi_ref = clamp(phi_unsat, -self.max_bank, self.max_bank)
        aileron = -(k_phi * (s.roll - phi_ref) + k_p * s.p)

        # Longitudinal: altitude PI -> limited pitch reference -> elevator.
        k_ih, k_h, k_th, k_q = self.k_lon
        h_err = s.alt_m - ref.alt_m
        theta_unsat = self.i_pitch - k_h * h_err / k_th
        theta_ref = clamp(theta_unsat, -self.max_pitch, self.max_pitch)
        elevator = -(k_th * (s.pitch - theta_ref) + k_q * s.q)

        # Speed: airspeed PI -> throttle about trim.
        k_iv, k_v = self.k_spd
        v_err = s.airspeed - ref.airspeed_mps
        throttle_unsat = self.model.throttle_trim + self.i_throttle - k_v * v_err

        p_lim = self.max_pitch
        b_lim = self.max_bank
        self.i_bank = integrate_conditionally(
            self.i_bank, -KI_BANK * psi_err, dt, phi_unsat, -b_lim, b_lim, b_lim
        )
        self.i_pitch = integrate_conditionally(
            self.i_pitch, -k_ih * h_err / k_th, dt, theta_unsat, -p_lim, p_lim, p_lim
        )
        self.i_throttle = integrate_conditionally(
            self.i_throttle, -k_iv * v_err, dt, throttle_unsat, 0.0, 1.0, 1.0
        )
        return RawActuators(aileron, elevator, 0.0, throttle_unsat).saturated()
