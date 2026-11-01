#include "optimal_math.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace astrum {
namespace {

Eigen::MatrixXd scalar(double v) { return Eigen::MatrixXd::Constant(1, 1, v); }

TEST(Dlqr, MatchesTheScalarClosedForm) {
  const double a = 1.2, b = 0.5, q = 2.0, r = 0.3;
  // Scalar DARE: b^2 P^2 + (r - a^2 r - q b^2) P - q r = 0, positive root.
  const double c1 = r - a * a * r - q * b * b;
  const double p = (-c1 + std::sqrt(c1 * c1 + 4.0 * b * b * q * r)) / (2.0 * b * b);
  const double k = a * b * p / (r + b * b * p);

  const DlqrResult res = dlqr(scalar(a), scalar(b), scalar(q), scalar(r));
  ASSERT_TRUE(res.converged);
  EXPECT_NEAR(res.P(0, 0), p, 1e-9 * p);
  EXPECT_NEAR(res.K(0, 0), k, 1e-9 * k);
  EXPECT_LT(std::abs(a - b * res.K(0, 0)), 1.0);  // closed loop is stable
}

TEST(Dlqr, ReportsDivergenceForAnUnstabilizableSystem) {
  // Unstable mode the input cannot reach: P grows without bound.
  const DlqrResult res = dlqr(scalar(2.0), scalar(0.0), scalar(1.0), scalar(1.0));
  EXPECT_FALSE(res.converged);
}

TEST(Dlqr, ReportsNonConvergenceWhenOutOfIterations) {
  const LinearModel m = attitude_axis_model(2.0, 4.0, 1.0 / 120.0);
  const Eigen::MatrixXd Q = Eigen::Vector2d(2.25, 0.0).asDiagonal();
  EXPECT_FALSE(dlqr(m.A, m.B, Q, scalar(1.0), /*max_iters=*/5).converged);
  const DlqrResult full = dlqr(m.A, m.B, Q, scalar(1.0));
  EXPECT_TRUE(full.converged);
  EXPECT_GT(full.iterations, 5);
}

TEST(AttitudeAxisModel, IsAnExactZeroOrderHold) {
  // Exact discretization composes: two half steps equal one full step.
  const double dt = 0.1;
  const LinearModel full = attitude_axis_model(2.0, 4.0, dt);
  const LinearModel half = attitude_axis_model(2.0, 4.0, dt / 2.0);
  EXPECT_TRUE(full.A.isApprox(half.A * half.A, 1e-12));
  EXPECT_TRUE(full.B.isApprox(half.A * half.B + half.B, 1e-12));
  // Positive actuator with positive gain accelerates the angle positively.
  EXPECT_GT(full.B(0, 0), 0.0);
  EXPECT_GT(full.B(1, 0), 0.0);
}

TEST(BoxQp, ReturnsTheUnconstrainedOptimumWhenFeasible) {
  const Eigen::Matrix2d H{{2.0, 1.0}, {1.0, 2.0}};
  const Eigen::Vector2d f(-0.3, 0.0);
  const Eigen::Vector2d expected = H.ldlt().solve(-f);
  const Eigen::VectorXd u = solve_box_qp(H, f, -1.0, 1.0, Eigen::Vector2d::Zero(), 3.0);
  EXPECT_TRUE(u.isApprox(expected, 1e-8));
}

TEST(BoxQp, ClampsActiveBoundsAndReoptimizesTheRest) {
  // Unconstrained optimum (4, -2); with u in [-1, 1], u0 = 1 is active and
  // u1 = argmin (u1^2 + u1) = -0.5.
  const Eigen::Matrix2d H{{2.0, 1.0}, {1.0, 2.0}};
  const Eigen::Vector2d f(-6.0, 0.0);
  const Eigen::VectorXd u = solve_box_qp(H, f, -1.0, 1.0, Eigen::Vector2d::Zero(), 3.0);
  EXPECT_NEAR(u(0), 1.0, 1e-9);
  EXPECT_NEAR(u(1), -0.5, 1e-8);
}

}  // namespace
}  // namespace astrum
