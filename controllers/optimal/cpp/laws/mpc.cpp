// MPC control law.
//
// Same cascade as laws/lqr.cpp — astrum::OuterLoops supplies saturated bank / pitch /
// throttle targets — but each attitude axis is flown by a receding-horizon controller
// that plans WITH the actuator limits:
//
//   minimize   sum_{k=1..N-1} x_k' Q x_k + x_N' P x_N + sum_{k=0..N-1} u_k' R u_k
//   subject to x_{k+1} = A x_k + B u_k,  -1 <= u_k <= 1
//
// with x = [angle - target, rate] (include/attitude_axes.hpp). The prediction step IS
// the control period: (A, B) is the exact zero-order-hold discretization at
// dt = 1 / --rate-hz, the horizon is N = round(--horizon-s / dt) steps of dt, and each
// sample re-plans and applies u_0. The terminal weight P is the axis's discrete Riccati
// solution, so when no limit is active the first move equals the LQR law's exactly; when
// the plan saturates, the box-constrained QP is solved by accelerated projected gradient,
// warm-started from the previous plan shifted by one step.
//
// What to do: replace the axis models / weights in attitude_axes.hpp, tune --horizon-s
// (the QP has N variables, so long horizons cost CPU when a limit is active),
// and add state constraints (e.g. a rate limit) as further QP rows if you need them.

#include <Eigen/Eigenvalues>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "attitude_axes.hpp"
#include "laws.hpp"

namespace astrum {
namespace {

// Prediction steps of dt covering horizon_s seconds.
int horizon_steps(double horizon_s, double dt) {
  const double steps = std::round(horizon_s / dt);
  if (!(steps >= 1.0) || steps > 1e5) {
    throw std::invalid_argument("--horizon-s must cover at least one control period (got " +
                                std::to_string(horizon_s) + " s)");
  }
  return static_cast<int>(steps);
}

class AxisMpc {
 public:
  AxisMpc(const AxisDesign& axis, double dt, int horizon) {
    const LinearModel m = axis.model(dt);
    const DlqrResult terminal = dlqr(m.A, m.B, axis.Q, axis.R);
    if (!terminal.converged) {
      throw std::runtime_error(std::string("dlqr (MPC terminal cost) for the ") + axis.name +
                               " axis did not converge after " +
                               std::to_string(terminal.iterations) + " iterations");
    }

    // Condensed prediction over k = 1..N: x_k = A^k x0 + sum_{j<k} A^(k-1-j) B u_j.
    const int n = 2;
    const int N = horizon;
    Eigen::MatrixXd Sx(n * N, n);
    Eigen::MatrixXd Su = Eigen::MatrixXd::Zero(n * N, N);
    Eigen::MatrixXd Ak = m.A;
    for (int k = 0; k < N; ++k) {
      Sx.middleRows(k * n, n) = Ak;  // A^(k+1)
      Ak = m.A * Ak;
      if (k > 0) Su.block(k * n, 0, n, k) = m.A * Su.block((k - 1) * n, 0, n, k);
      Su.block(k * n, k, n, 1) = m.B;
    }

    // H = Su' Qbar Su + Rbar, F = Su' Qbar Sx with Qbar = diag(Q, ..., Q, P).
    Eigen::MatrixXd QSu(n * N, N);
    Eigen::MatrixXd QSx(n * N, n);
    for (int k = 0; k < N; ++k) {
      const Eigen::MatrixXd& W = k + 1 == N ? terminal.P : Eigen::MatrixXd(axis.Q);
      QSu.middleRows(k * n, n) = W * Su.middleRows(k * n, n);
      QSx.middleRows(k * n, n) = W * Sx.middleRows(k * n, n);
    }
    h_ = Su.transpose() * QSu;
    h_.diagonal().array() += axis.R(0, 0);
    f_ = Su.transpose() * QSx;
    unconstrained_ = h_.ldlt().solve(f_);  // U* = -unconstrained_ x0
    lipschitz_ = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(h_).eigenvalues().maxCoeff();
    plan_ = Eigen::VectorXd::Zero(N);
  }

  // First input of the optimal plan for error-state x0, within [-1, 1].
  double move(const Eigen::Vector2d& x0) {
    Eigen::VectorXd u = -unconstrained_ * x0;
    if (u.cwiseAbs().maxCoeff() > 1.0) {
      // Warm start: last plan shifted one step (the horizon moved by exactly dt).
      Eigen::VectorXd warm(plan_.size());
      warm.head(plan_.size() - 1) = plan_.tail(plan_.size() - 1);
      warm(plan_.size() - 1) = plan_(plan_.size() - 1);
      u = solve_box_qp(h_, f_ * x0, -1.0, 1.0, warm, lipschitz_, 500, 1e-9);
    }
    plan_ = u;
    return u(0);
  }

 private:
  Eigen::MatrixXd h_;
  Eigen::MatrixXd f_;
  Eigen::MatrixXd unconstrained_;
  double lipschitz_ = 1.0;
  Eigen::VectorXd plan_;
};

class MpcLaw : public Law {
 public:
  explicit MpcLaw(const LawConfig& config)
      : outer_(config.limits),
        roll_(roll_axis(), config.dt, horizon_steps(config.horizon_s, config.dt)),
        pitch_(pitch_axis(), config.dt, horizon_steps(config.horizon_s, config.dt)) {}

  Command control(const VehicleState& s, const Reference& ref) override {
    const AttitudeTargets target = outer_.update(s, ref);
    const double aileron = roll_.move(Eigen::Vector2d(s.roll - target.bank_rad, s.p));
    const double elevator = pitch_.move(Eigen::Vector2d(s.pitch - target.pitch_rad, s.q));
    return Command::raw(aileron, elevator, /*rudder=*/0.0, target.throttle);
  }

 private:
  OuterLoops outer_;
  AxisMpc roll_;
  AxisMpc pitch_;
};

}  // namespace

std::unique_ptr<Law> make_mpc_law(const LawConfig& config) {
  return std::make_unique<MpcLaw>(config);
}

}  // namespace astrum
