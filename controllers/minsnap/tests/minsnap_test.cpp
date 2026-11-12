#include "minsnap.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace astrum::minsnap {
namespace {

// Five knots in 3-D with pinned start / end velocity, one pinned interior velocity.
Problem waypoint_problem() {
  Problem p;
  p.cost_order = 4;
  p.channels = 3;
  p.durations = {1.0, 1.5, 0.8, 2.0};
  const double pos[5][3] = {{0, 0, 1}, {2, 1, 1.5}, {3, 3, 2}, {1, 5, 2}, {-1, 6, 1}};
  for (std::size_t k = 0; k < 5; ++k) {
    p.constraints.push_back({k, 0, {pos[k][0], pos[k][1], pos[k][2]}});
  }
  p.constraints.push_back({0, 1, {0.0, 0.0, 0.0}});
  p.constraints.push_back({0, 2, {0.1, -0.2, 0.3}});
  p.constraints.push_back({2, 1, {0.5, 2.0, 0.0}});
  p.constraints.push_back({4, 1, {0.0, 1.0, -1.0}});
  return p;
}

TEST(MinSnap, MeetsEveryPinToNineDigits) {
  const Problem p = waypoint_problem();
  const Trajectory t = solve(p);
  const auto& times = t.knot_times();
  for (const Constraint& c : p.constraints) {
    // Both sides of an interior knot: just inside the previous and the next segment.
    for (double side : {-1.0, 1.0}) {
      if ((c.knot == 0 && side < 0) || (c.knot == p.durations.size() && side > 0)) continue;
      const double eps = 1e-12;
      const double at = times[c.knot] + side * eps;
      for (std::size_t ch = 0; ch < 3; ++ch) {
        EXPECT_NEAR(t.value(at, c.derivative, ch), c.value[ch], 1e-9)
            << "knot " << c.knot << " d" << c.derivative << " ch" << ch << " side " << side;
      }
    }
  }
}

TEST(MinSnap, DerivativesUpToTheThirdAreContinuousAtEveryJoin) {
  const Trajectory t = solve(waypoint_problem());
  const auto& times = t.knot_times();
  for (std::size_t k = 1; k + 1 < times.size(); ++k) {
    for (int d = 0; d <= 3; ++d) {
      for (std::size_t ch = 0; ch < 3; ++ch) {
        const double before = t.value(times[k] - 1e-12, d, ch);
        const double after = t.value(times[k] + 1e-12, d, ch);
        EXPECT_NEAR(before, after, 1e-8) << "knot " << k << " d" << d;
      }
    }
  }
}

TEST(MinSnap, RestToRestSingleSegmentIsTheClassicSepticAndScalesWithTime) {
  // p, v, a, j pinned at both ends: x(tau) = 35 tau^4 - 84 tau^5 + 70 tau^6 - 20 tau^7.
  for (double T : {0.5, 2.0}) {
    Problem p;
    p.cost_order = 4;
    p.channels = 1;
    p.durations = {T};
    for (std::size_t knot : {0u, 1u}) {
      for (int d = 0; d <= 3; ++d)
        p.constraints.push_back({knot, d, {d == 0 ? static_cast<double>(knot) : 0.0}});
    }
    const Trajectory t = solve(p);
    for (double tau : {0.1, 0.37, 0.5, 0.8}) {
      const double want = 35 * std::pow(tau, 4) - 84 * std::pow(tau, 5) + 70 * std::pow(tau, 6) -
                          20 * std::pow(tau, 7);
      EXPECT_NEAR(t.value(tau * T, 0, 0), want, 1e-10) << "T=" << T << " tau=" << tau;
    }
    EXPECT_GT(t.cost(), 0.0);
  }
}

TEST(MinSnap, TemporalScalingStretchesTheSameShape) {
  // Section V-A: scaling every duration by a leaves the path unchanged in scaled time
  // (all derivative pins here are zero, so they are scale invariant).
  Problem p;
  p.cost_order = 4;
  p.channels = 2;
  p.durations = {1.0, 2.0, 1.5};
  const double pos[4][2] = {{0, 0}, {1, 2}, {3, 2}, {4, 0}};
  for (std::size_t k = 0; k < 4; ++k) p.constraints.push_back({k, 0, {pos[k][0], pos[k][1]}});
  for (std::size_t k : {0u, 3u}) p.constraints.push_back({k, 1, {0.0, 0.0}});
  const Trajectory base = solve(p);
  const double a = 2.5;
  for (double& d : p.durations) d *= a;
  const Trajectory slow = solve(p);
  for (double t : {0.3, 1.7, 3.1, 4.4}) {
    EXPECT_NEAR(slow.value(t * a, 0, 0), base.value(t, 0, 0), 1e-9);
    EXPECT_NEAR(slow.value(t * a, 0, 1), base.value(t, 0, 1), 1e-9);
    EXPECT_NEAR(slow.value(t * a, 2, 0) * a * a, base.value(t, 2, 0), 1e-8);
  }
}

TEST(MinSnap, NoOtherTrajectoryWithTheSamePinsHasLessSnap) {
  // Perturb the optimum within the constraint set: any feasible variation costs more.
  // Reuse the solver with one extra knot position pinned to the optimum's value plus a
  // delta — a feasible but non-optimal trajectory for the original problem.
  Problem p = waypoint_problem();
  const Trajectory best = solve(p);
  Problem q = p;
  q.durations.insert(q.durations.begin() + 1, q.durations[0] * 0.5);
  q.durations[0] *= 0.5;
  // Shift constraint knots after the split.
  for (Constraint& c : q.constraints) {
    if (c.knot >= 1) ++c.knot;
  }
  const double mid = q.durations[0];  // the new knot
  q.constraints.push_back(
      {1, 0, {best.value(mid, 0, 0) + 0.05, best.value(mid, 0, 1), best.value(mid, 0, 2)}});
  EXPECT_GT(solve(q).cost(), best.cost());
  q.constraints.back().value = {best.value(mid, 0, 0), best.value(mid, 0, 1),
                                best.value(mid, 0, 2)};
  EXPECT_NEAR(solve(q).cost(), best.cost(), 1e-6 * best.cost());
}

TEST(MinSnap, YawProblemMinimisesAcceleration) {
  Problem p;
  p.cost_order = 2;
  p.channels = 1;
  p.durations = {2.0};
  p.constraints = {{0, 0, {0.0}}, {1, 0, {1.0}}, {0, 1, {0.0}}, {1, 1, {0.0}}};
  const Trajectory t = solve(p);
  // Rest to rest, min acceleration: the smoothstep 3 tau^2 - 2 tau^3 (the septic's extra
  // freedom is spent on lower-order terms being zero).
  for (double tau : {0.25, 0.5, 0.75}) {
    EXPECT_NEAR(t.value(tau * 2.0, 0, 0), 3 * tau * tau - 2 * tau * tau * tau, 1e-10);
  }
}

TEST(MinSnap, SegmentTimeReallocationLowersTheCostAndKeepsTheTotal) {
  Problem p;
  p.cost_order = 4;
  p.channels = 2;
  p.durations = {1.0, 1.0, 1.0, 1.0};  // equal times for very unequal legs
  const double pos[5][2] = {{0, 0}, {0.5, 0}, {1.0, 3}, {6.0, 3}, {6.5, 3}};
  for (std::size_t k = 0; k < 5; ++k) p.constraints.push_back({k, 0, {pos[k][0], pos[k][1]}});
  for (std::size_t k : {0u, 4u}) p.constraints.push_back({k, 1, {0.0, 0.0}});
  const double before = solve(p).cost();
  const std::vector<double> t = reallocate_durations(p, 10);
  Problem q = p;
  q.durations = t;
  EXPECT_LT(solve(q).cost(), before);
  double sum = 0.0;
  for (double v : t) sum += v;
  EXPECT_NEAR(sum, 4.0, 1e-9);
  for (double v : t) EXPECT_GE(v, 0.2 - 1e-12);
  EXPECT_EQ(reallocate_durations(p, 0), p.durations);
}

TEST(MinSnap, RejectsMalformedProblems) {
  Problem p = waypoint_problem();
  p.durations[1] = 0.0;
  EXPECT_THROW(solve(p), std::invalid_argument);
  p = waypoint_problem();
  p.constraints.erase(p.constraints.begin() + 2);  // knot 2 loses its position
  EXPECT_THROW(solve(p), std::invalid_argument);
  p = waypoint_problem();
  p.constraints.push_back({1, 1, {0.0}});  // wrong channel count
  EXPECT_THROW(solve(p), std::invalid_argument);
  p = waypoint_problem();
  p.constraints.push_back({9, 0, {0, 0, 0}});
  EXPECT_THROW(solve(p), std::invalid_argument);
  // Contradictory duplicate pin: singular KKT system.
  p = waypoint_problem();
  p.constraints.push_back({2, 1, {0.5, 2.0, 0.0}});
  EXPECT_THROW(solve(p), std::runtime_error);
}

TEST(MinSnap, EvaluatesTheFourthDerivativeAndClampsTime) {
  Problem p;
  p.cost_order = 4;
  p.channels = 1;
  p.durations = {1.0};
  p.constraints = {{0, 0, {0.0}}, {1, 0, {1.0}}};
  for (int d = 1; d <= 3; ++d) {
    p.constraints.push_back({0, d, {0.0}});
    p.constraints.push_back({1, d, {0.0}});
  }
  const Trajectory t = solve(p);
  // x'''' of 35 t^4 - 84 t^5 + 70 t^6 - 20 t^7 at t = 0: 24 * 35.
  EXPECT_NEAR(t.value(0.0, 4, 0), 840.0, 1e-7);
  EXPECT_NEAR(t.value(-3.0, 0, 0), t.value(0.0, 0, 0), 1e-15);
  EXPECT_NEAR(t.value(9.0, 0, 0), 1.0, 1e-12);
}

}  // namespace
}  // namespace astrum::minsnap
