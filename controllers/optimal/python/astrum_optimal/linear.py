"""Linear-systems tools for control-law design: exact ZOH discretization,
discrete LQR, and the unconstrained condensed MPC first-move gain."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve, solve_discrete_are


def _mat(m) -> np.ndarray:
    return np.atleast_2d(np.asarray(m, dtype=float))


def discretize_zoh(A, B, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization of ``x' = A x + B u`` at step ``dt``."""
    A, B = _mat(A), _mat(B)
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def dlqr(A, B, Q, R) -> tuple[np.ndarray, np.ndarray]:
    """Infinite-horizon discrete LQR for ``x[k+1] = A x + B u`` minimizing
    ``sum x'Qx + u'Ru``: returns ``(K, P)`` with ``u = -K x`` and ``P`` the DARE
    solution."""
    A, B, Q, R = _mat(A), _mat(B), _mat(Q), _mat(R)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class ChannelMPC:
    """Unconstrained condensed MPC for one linear channel ``x[k+1] = A x + B u``.

    Minimizes ``sum_{k=1..N} x_k' Q x_k + sum_{k=0..N-1} u_k' R u_k`` (with
    ``terminal`` replacing ``Q`` on ``x_N`` when given) over the input sequence.
    The model step of ``(A, B)`` must be the control period — the receding
    horizon re-plans and applies the first move every period. Without
    constraints the optimal first move is linear, ``u0 = -K x0``; ``K`` is
    precomputed here, and with ``terminal`` = the DARE solution it equals the
    LQR gain for any ``N``.
    """

    def __init__(self, A, B, Q, R, N: int, terminal=None):
        if N < 1:
            raise ValueError(f"MPC horizon must be >= 1 step, got {N}")
        A, B, Q, R = _mat(A), _mat(B), _mat(Q), _mat(R)
        n, m = B.shape
        # Condensed prediction over k = 1..N: X = Sx x0 + Su U.
        powers = [np.eye(n)]
        for _ in range(N):
            powers.append(powers[-1] @ A)
        Sx = np.vstack(powers[1:])
        # Su is block lower-triangular Toeplitz: block (i, j) = A^(i-j) B for j <= i.
        AkB = np.stack([p @ B for p in powers[:N]])  # (N, n, m)
        Su4 = np.zeros((N, n, N, m))
        for lag in range(N):
            rows = np.arange(lag, N)
            Su4[rows, :, rows - lag, :] = AkB[lag]
        Su = Su4.reshape(N * n, N * m)
        # Qbar = blockdiag(Q, ..., Q, terminal), applied block-wise.
        QSu = np.einsum("ab,kbc->kac", Q, Su4.reshape(N, n, N * m))
        if terminal is not None:
            QSu[-1] = _mat(terminal) @ Su4[-1].reshape(n, N * m)
        QSu = QSu.reshape(N * n, N * m)
        H = Su.T @ QSu + np.kron(np.eye(N), R)
        # H is symmetric positive definite (R > 0): Cholesky, which is also far
        # faster than numpy's threaded LU on these sizes.
        Kfull = solve(H, QSu.T @ Sx, assume_a="pos")
        self.K = Kfull[:m, :]
        self.N = N

    def move(self, x0) -> np.ndarray:
        """First optimal input for the current error-state ``x0``.

        Constraint hook: to enforce bounds *inside* the horizon (rather than the
        laws' output saturation), replace this closed form with a QP over the
        horizon (e.g. osqp) and return its first input."""
        return -self.K @ np.asarray(x0, dtype=float)
