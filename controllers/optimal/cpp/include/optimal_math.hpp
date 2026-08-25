// Shared control-math helpers for the optimal-control laws (Eigen, header-only).
//
// These are the building blocks you call from a law once you've linearized the
// aircraft into per-channel (A, B) models. `inline` keeps them header-only and
// silences unused-function warnings when a given law doesn't call one yet.

#ifndef ASTRUM_OPTIMAL_MATH_HPP
#define ASTRUM_OPTIMAL_MATH_HPP

#include <Eigen/Dense>

namespace astrum {

// Discrete-time LQR gain K minimizing sum(x'Qx + u'Ru) for x[k+1] = A x + B u.
//
// Eigen has no algebraic-Riccati solver (scipy's solve_discrete_are, used by the
// Python side, has no C++ analogue here), so we iterate the discrete Riccati
// recursion to a fixed point, then K = (R + B'PB)^-1 B'PA. Converges for a
// stabilizable (A, B) with Q >= 0, R > 0.
inline Eigen::MatrixXd dlqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                            const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                            int iters = 1000, double tol = 1e-12) {
  Eigen::MatrixXd P = Q;
  for (int i = 0; i < iters; ++i) {
    const Eigen::MatrixXd Pn =
        A.transpose() * P * A -
        A.transpose() * P * B * (R + B.transpose() * P * B).ldlt().solve(B.transpose() * P * A) +
        Q;
    if ((Pn - P).cwiseAbs().maxCoeff() < tol) {
      P = Pn;
      break;
    }
    P = Pn;
  }
  return (R + B.transpose() * P * B).ldlt().solve(B.transpose() * P * A);
}

}  // namespace astrum

#endif  // ASTRUM_OPTIMAL_MATH_HPP
