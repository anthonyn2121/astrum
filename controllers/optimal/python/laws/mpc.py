"""MPC control law (skeleton).

Receding-horizon (model-predictive) control. Each frame it predicts the
error-state ``N`` steps ahead with a linear model, minimizes a quadratic cost
over the input sequence, and applies the first move — then re-plans next frame.

Like ``laws/lqr.py`` the channels are decoupled for clarity:

    lateral      x_lat = [heading_err, roll,  roll_rate ]   -> aileron
    longitudinal x_lon = [alt_err,     pitch, pitch_rate]   -> elevator
    speed        x_spd = [airspeed_err]                     -> throttle

Each channel is regulated to zero error. For the **unconstrained** quadratic
problem the optimal first move is a constant linear gain on the current
error-state (computed once in :class:`ChannelMPC`), so the hot path is a single
matrix–vector product. That gain approaches the LQR gain as ``N -> inf`` — the
point of MPC is what you add next:

Because heading and altitude respond over *seconds*, the model is discretized on
a coarse step (``--horizon-s / --horizon``, ~0.1 s) so a modest step count still
looks seconds ahead — the horizon must outlast the mode you're regulating, or the
optimizer sees the control do nothing and won't use it. This model step is
independent of the 120 Hz control rate: the loop re-solves and applies the first
move on every telemetry frame regardless.

**What to do:** (1) replace the placeholder per-channel ``(A, B)`` below with
*your* linearized model, tune ``(Q, R)`` and the horizon; (2) add the hard limits
LQR can't express — bank/pitch/actuator bounds — at the marked hook in
:meth:`ChannelMPC.move`, which then needs a QP solver (e.g. ``osqp`` or ``cvxpy``)
instead of the closed form. The shipped model flies ``--law mpc`` (its gains land
near the proven LQR law) so you have a working loop to grow.
"""

import numpy as np

from astrum_control import Command, Law as BaseLaw, Reference, VehicleState


def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class ChannelMPC:
    """Unconstrained condensed MPC for one linear channel x[k+1]=Ax+Bu.

    Regulates the error-state to zero over an ``N``-step horizon with stage cost
    ``x'Qx + u'Ru``. Precomputes the first-move feedback gain ``K`` such that the
    optimal first input is ``u0 = -K @ x0``.
    """

    def __init__(self, A, B, Q, R, N):
        A, B, Q, R = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R))
        n, m = B.shape
        # Condensed prediction: X = Sx x0 + Su U, over k = 1..N.
        Sx = np.zeros((N * n, n))
        Su = np.zeros((N * n, N * m))
        Apow = np.eye(n)
        for i in range(N):
            Apow = Apow @ A
            Sx[i * n:(i + 1) * n, :] = Apow
            for j in range(i + 1):
                # block (i,j) = A^(i-j) B
                blk = np.linalg.matrix_power(A, i - j) @ B
                Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = blk
        Qbar = np.kron(np.eye(N), Q)
        Rbar = np.kron(np.eye(N), R)
        # U* = -(Su'Qbar Su + Rbar)^-1 Su'Qbar Sx x0 ; take the first m rows.
        H = Su.T @ Qbar @ Su + Rbar
        Kfull = np.linalg.solve(H, Su.T @ Qbar @ Sx)
        self.K = Kfull[:m, :]
        self.m = m

    def move(self, x0):
        """First optimal input for current error-state ``x0`` (a length-n vector)."""
        # ---- CONSTRAINT HOOK -------------------------------------------------
        # Unconstrained closed form. To enforce hard limits (|u| <= umax, bank
        # <= phi_max, ...), replace this with a QP over the horizon subject to
        # those bounds (osqp / cvxpy) and return the first solved input.
        u = -self.K @ np.atleast_1d(np.asarray(x0, dtype=float))
        return float(u[0]) if self.m == 1 else u


def add_args(p):
    """MPC-specific flags."""
    p.add_argument("--horizon", type=int, default=30, help="MPC prediction horizon (steps)")
    p.add_argument("--horizon-s", type=float, default=3.0,
                   help="total look-ahead (s); model step = horizon-s / horizon")


class Law(BaseLaw):
    """Per-channel receding-horizon control. See module docstring."""

    THROTTLE_TRIM = 0.6

    def __init__(self, args):
        super().__init__(args)
        N = args.horizon
        dt = args.horizon_s / N  # coarse model step (~0.1s), NOT the control rate

        # --- PLACEHOLDER MODELS (Euler-discretized). Replace with your own. ---
        # Continuous-time constants (all PLACEHOLDER — swap in your linearization):
        g_over_v = 9.81 / 50.0  # turn-rate per unit bank at ~50 m/s
        v = 50.0                # airspeed for the climb-rate coupling
        damp = 2.0              # angular-rate damping (roll/pitch)
        b = 4.0                 # actuator -> angular acceleration

        # Lateral: heading changes with bank; roll is a damped 2nd-order response
        # to aileron. State [heading_err, roll, roll_rate], input aileron.
        A_lat = np.array([[1.0, -g_over_v * dt, 0.0],
                          [0.0, 1.0, dt],
                          [0.0, 0.0, 1.0 - damp * dt]])
        B_lat = np.array([[0.0], [0.0], [b * dt]])
        self.lat = ChannelMPC(A_lat, B_lat, np.diag([6.0, 1.0, 0.1]), [[1.0]], N)

        # Longitudinal: altitude changes with pitch; pitch is a damped response
        # to elevator (elevator+ pitches DOWN, hence -B). State [alt_err, pitch,
        # pitch_rate], input elevator.
        A_lon = np.array([[1.0, -v * dt, 0.0],
                          [0.0, 1.0, dt],
                          [0.0, 0.0, 1.0 - damp * dt]])
        B_lon = np.array([[0.0], [0.0], [-b * dt]])
        self.lon = ChannelMPC(A_lon, B_lon, np.diag([0.02, 1.0, 0.05]), [[16.0]], N)

        # Speed: single integrator; more throttle raises airspeed (reduces error).
        A_spd = np.array([[1.0]])
        B_spd = np.array([[-b * dt]])
        self.spd = ChannelMPC(A_spd, B_spd, [[1.0]], [[500.0]], N)

    def control(self, s: VehicleState, ref: Reference) -> Command:
        heading_err = _wrap_pi(ref.heading_rad - s.yaw)
        alt_err = ref.alt_m - s.alt_m
        airspeed_err = ref.airspeed_mps - s.airspeed

        aileron = self.lat.move([heading_err, s.roll, s.p])
        elevator = self.lon.move([alt_err, s.pitch, s.q])
        throttle = self.THROTTLE_TRIM + self.spd.move([airspeed_err])
        rudder = 0.0  # coordinated-turn assumption; not modeled here

        return Command.raw(aileron=aileron, elevator=elevator, rudder=rudder, throttle=throttle)
