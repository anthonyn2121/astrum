import numpy as np
import pytest
from astrum_optimal.laws._fixed_wing import FixedWingModel
from astrum_optimal.linear import ChannelMPC, discretize_zoh, dlqr
from scipy.linalg import solve_discrete_are

DT = 1.0 / 120.0


def channels():
    m = FixedWingModel()
    yield discretize_zoh(*m.lateral(50.0), DT), np.diag([4.0, 1.0, 0.0]), np.array([[1.0]])
    yield discretize_zoh(*m.speed(), DT), np.diag([1e-3, 1e-2]), np.array([[1.0]])
    yield discretize_zoh([[0, 1], [0, 0]], [[0], [1]], 0.1), np.diag([0.8, 3.0]), np.array([[2.0]])


def test_zoh_of_double_integrator_is_exact():
    A, B = discretize_zoh([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], 0.2)
    np.testing.assert_allclose(A, [[1.0, 0.2], [0.0, 1.0]], atol=1e-14)
    np.testing.assert_allclose(B, [[0.02], [0.2]], atol=1e-14)


@pytest.mark.parametrize("ch", list(channels()))
def test_dlqr_gain_satisfies_dare_optimality(ch):
    (A, B), Q, R = ch
    K, P = dlqr(A, B, Q, R)
    np.testing.assert_allclose(P, solve_discrete_are(A, B, Q, R))
    # Riccati fixed point, and the closed loop is stable.
    P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    np.testing.assert_allclose(P_next, P, rtol=1e-8, atol=1e-10)
    assert max(abs(np.linalg.eigvals(A - B @ K))) < 1.0


@pytest.mark.parametrize("ch", list(channels()))
def test_unconstrained_mpc_with_dare_terminal_equals_lqr(ch):
    (A, B), Q, R = ch
    K, P = dlqr(A, B, Q, R)
    for n in (1, 5, 40):
        np.testing.assert_allclose(ChannelMPC(A, B, Q, R, n, terminal=P).K, K, rtol=1e-6)


def test_unconstrained_mpc_converges_to_lqr_with_horizon():
    (A, B), Q, R = next(iter(channels()))
    K, _ = dlqr(A, B, Q, R)
    errs = [np.abs(ChannelMPC(A, B, Q, R, n).K - K).max() for n in (60, 240, 960)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-2 * np.abs(K).max()


def test_mpc_first_move_is_minus_k_x():
    (A, B), Q, R = next(iter(channels()))
    mpc = ChannelMPC(A, B, Q, R, 30)
    x = np.array([0.3, -0.1, 0.05])
    np.testing.assert_allclose(mpc.move(x), -mpc.K @ x)
