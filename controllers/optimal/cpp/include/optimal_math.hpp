// Control math for the optimal-control laws (Eigen, header-only).

#ifndef ASTRUM_OPTIMAL_MATH_HPP
#define ASTRUM_OPTIMAL_MATH_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace astrum {

// Discrete LTI model x[k+1] = A x[k] + B u[k].
struct LinearModel {
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
};

// One attitude axis about trim, x = [angle error, angular rate], u = actuator:
//     d(angle)/dt = rate,   d(rate)/dt = -damping * rate + gain * u
// discretized EXACTLY (zero-order hold) at the control period dt, so the model step is
// the step the law actually runs at. With l = exp(-damping dt):
//     A = [1, (1 - l)/damping; 0, l],  B = gain * [(dt - (1 - l)/damping)/damping; (1 - l)/damping]
// `gain` carries the actuator polarity (elevator+ pitches nose down -> negative gain).
inline LinearModel attitude_axis_model(double damping, double gain, double dt) {
  const double l = std::exp(-damping * dt);
  const double s = (1.0 - l) / damping;
  LinearModel m{Eigen::MatrixXd(2, 2), Eigen::MatrixXd(2, 1)};
  m.A << 1.0, s, 0.0, l;
  m.B << gain * (dt - s) / damping, gain * s;
  return m;
}

struct DlqrResult {
  Eigen::MatrixXd K;  // u = -K x
  Eigen::MatrixXd P;  // Riccati solution (cost-to-go x'Px)
  bool converged = false;
  int iterations = 0;
};

// Discrete-time LQR minimizing sum(x'Qx + u'Ru) for x[k+1] = A x + B u, by iterating the
// Riccati recursion to a fixed point (relative change <= tol). `converged` is false when
// the recursion has not settled within max_iters or diverged (e.g. (A, B) not
// stabilizable) — K is then meaningless and callers must not use it.
inline DlqrResult dlqr(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q,
                       const Eigen::MatrixXd& R, int max_iters = 100000, double tol = 1e-11) {
  DlqrResult out;
  Eigen::MatrixXd P = Q;
  for (int i = 1; i <= max_iters; ++i) {
    const Eigen::MatrixXd BtP = B.transpose() * P;
    const Eigen::MatrixXd Pn =
        A.transpose() * P * A - (BtP * A).transpose() * (R + BtP * B).ldlt().solve(BtP * A) + Q;
    out.iterations = i;
    if (!Pn.allFinite()) return out;
    const double change = (Pn - P).cwiseAbs().maxCoeff();
    P = Pn;
    if (change <= tol * std::max(1.0, P.cwiseAbs().maxCoeff())) {
      out.converged = true;
      break;
    }
  }
  const Eigen::MatrixXd BtP = B.transpose() * P;
  out.K = (R + BtP * B).ldlt().solve(BtP * A);
  out.P = P;
  return out;
}

// Minimize 0.5 u'Hu + f'u subject to lo <= u <= hi (H symmetric positive definite) by
// accelerated projected gradient (FISTA), starting from `u` (clamped). Returns when an
// iterate moves less than tol (infinity norm) or after max_iters. `lipschitz` is the
// largest eigenvalue of H.
inline Eigen::VectorXd solve_box_qp(const Eigen::MatrixXd& H, const Eigen::VectorXd& f, double lo,
                                    double hi, Eigen::VectorXd u, double lipschitz,
                                    int max_iters = 500, double tol = 1e-10) {
  u = u.cwiseMax(lo).cwiseMin(hi);
  Eigen::VectorXd y = u;
  double t = 1.0;
  for (int i = 0; i < max_iters; ++i) {
    const Eigen::VectorXd next = (y - (H * y + f) / lipschitz).cwiseMax(lo).cwiseMin(hi);
    const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
    y = next + ((t - 1.0) / t_next) * (next - u);
    const double moved = (next - u).cwiseAbs().maxCoeff();
    u = next;
    t = t_next;
    if (moved < tol) break;
  }
  return u;
}

}  // namespace astrum

#endif  // ASTRUM_OPTIMAL_MATH_HPP
