"""LQR control law (skeleton).

Full-state feedback ``u = -K (x - x_ref)`` on a linearized model of the
aircraft. This file is where *your* control-theory work goes; the runtime
(``astrum_control``) hands you a decoded :class:`VehicleState` each frame and
publishes whatever :class:`Command` you return.

The channels are kept decoupled for clarity (each is a small SISO loop):

    lateral      x_lat = [heading_err, roll,  roll_rate ]   -> aileron
    longitudinal x_lon = [alt_err,     pitch, pitch_rate]   -> elevator
    speed        x_spd = [airspeed_err]                     -> throttle

**What to do:** replace the hand-tuned placeholder gains below with the LQR
solution of your own model. Linearize the aircraft about trim to get each
channel's ``(A, B)``, choose weights ``(Q, R)``, and call :func:`dlqr` to get
``K``. The placeholder gains reproduce the proven example controller's hold law
so ``--law lqr`` flies out of the box while you develop the real thing.

Note LQR is *unconstrained*: it can't respect a bank-angle or actuator limit
except by weighting. When you need hard limits (max bank, elevator saturation),
that's the motivation for ``--law mpc``.
"""

import numpy as np
from astrum_control import Command, Reference, VehicleState
from astrum_control import Law as BaseLaw


def dlqr(A, B, Q, R):
    """Discrete-time LQR gain ``K`` minimizing sum(x'Qx + u'Ru) for x[k+1]=Ax+Bu.

    Solves the discrete algebraic Riccati equation, then
    ``K = (R + B'PB)^-1 B'PA``. Requires scipy (imported lazily so the
    placeholder gains below work even without it).
    """
    from scipy.linalg import solve_discrete_are

    A, B, Q, R = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R))
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class Law(BaseLaw):
    """LQR full-state feedback. See module docstring."""

    # --- PLACEHOLDER GAINS -------------------------------------------------
    # Hand-tuned to match the example controller's proven hold law so the loop
    # flies today. Replace each with `dlqr(A, B, Q, R)` for your linearized
    # model — e.g. for the lateral channel:
    #
    #     dt = 1.0 / 120.0                       # sim rate
    #     A_lat = np.array([[1, -V*dt, 0],       # heading_err, roll, roll_rate
    #                       [0,  1,    dt],
    #                       [0,  0,    1 - a*dt]])
    #     B_lat = np.array([[0], [0], [b*dt]])   # aileron -> roll accel
    #     K_lat = dlqr(A_lat, B_lat, np.diag([Q_hdg, Q_roll, Q_p]), [[R_ail]])
    #
    # u = -K @ x, with x measured relative to the reference.
    K_lat = np.array([-1.8, 1.5, 0.3])  # [heading_err, roll, roll_rate] -> aileron
    K_lon = np.array([0.02, -2.0, -0.5])  # [alt_err, pitch, pitch_rate]  -> elevator
    K_spd = np.array([-0.02])  # [airspeed_err]                -> (throttle - trim)
    THROTTLE_TRIM = 0.6

    def control(self, s: VehicleState, ref: Reference) -> Command:
        # Errors relative to the reference (heading wrapped to [-pi, pi]).
        heading_err = _wrap_pi(ref.heading_rad - s.yaw)
        alt_err = ref.alt_m - s.alt_m
        airspeed_err = ref.airspeed_mps - s.airspeed

        # State vectors, then u = -K x per channel. Signs of K already fold in
        # the error convention above (positive heading_err -> roll right, etc.).
        x_lat = np.array([heading_err, s.roll, s.p])
        x_lon = np.array([alt_err, s.pitch, s.q])
        x_spd = np.array([airspeed_err])

        aileron = -float(self.K_lat @ x_lat)
        elevator = -float(self.K_lon @ x_lon)
        throttle = self.THROTTLE_TRIM - float(self.K_spd @ x_spd)
        rudder = 0.0  # coordinated-turn assumption; not modeled here

        return Command.raw(aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle)
