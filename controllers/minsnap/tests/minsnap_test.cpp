#include "minsnap.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "minsnap_reference.hpp"

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

// Legs of 0.05 s next to 2.5 s ones: the snap cost matrices differ by ~1e10, the case the
// equilibration exists for. Start velocity and acceleration pinned, interior velocity pinned.
Problem short_segment_problem() {
  Problem p;
  p.cost_order = 4;
  p.channels = 2;
  p.durations = {0.05, 2.5, 0.1, 1.0, 0.25};
  const double pos[6][2] = {{0, 0}, {0.05, 0.01}, {4, 2}, {4.1, 2.2}, {6, 5}, {6.2, 5.1}};
  for (std::size_t k = 0; k < 6; ++k) p.constraints.push_back({k, 0, {pos[k][0], pos[k][1]}});
  p.constraints.push_back({0, 1, {1.0, 0.2}});
  p.constraints.push_back({0, 2, {0.0, -0.5}});
  p.constraints.push_back({2, 1, {2.0, 1.5}});
  p.constraints.push_back({5, 1, {0.0, 0.0}});
  return p;
}

// The planner's position problem: per gate an entry, a centre (velocity pinned along the
// normal) and an exit knot, with short legs through the gate and long ones between gates.
Problem gate_chain_problem(int gates) {
  Problem p;
  p.cost_order = 4;
  p.channels = 3;
  p.constraints.push_back({0, 0, {-4.0, 0.0, 1.5}});
  p.constraints.push_back({0, 1, {0.0, 0.0, 0.0}});
  p.constraints.push_back({0, 2, {0.0, 0.0, 0.0}});
  std::size_t knot = 0;
  for (int g = 0; g < gates; ++g) {
    const double a = 0.4 * g;
    const double n[3] = {std::cos(a), std::sin(a), 0.0};
    const double c[3] = {6.0 * g + 2.0, 3.0 * std::sin(1.1 * g), 2.0 + 0.5 * std::cos(1.3 * g)};
    const double speed = 2.5 + 0.5 * (g % 3);
    const double legs[3] = {1.2 + 0.4 * (g % 2), 0.15 + 0.05 * (g % 3), 0.15 + 0.03 * (g % 4)};
    p.durations.push_back(legs[0]);
    p.durations.push_back(legs[1]);
    p.durations.push_back(legs[2]);
    const double offset[3] = {-0.6, 0.0, 0.6};
    for (int s = 0; s < 3; ++s) {
      ++knot;
      p.constraints.push_back(
          {knot, 0, {c[0] + offset[s] * n[0], c[1] + offset[s] * n[1], c[2] + offset[s] * n[2]}});
    }
    p.constraints.push_back({knot - 1, 1, {speed * n[0], speed * n[1], speed * n[2]}});
  }
  return p;
}

// The yaw problem: minimum acceleration, headings at the knots, zero rate at some of them.
Problem yaw_chain_problem() {
  Problem p;
  p.cost_order = 2;
  p.channels = 1;
  p.durations = {0.8, 1.3, 0.6, 0.9, 1.5, 0.7};
  const double heading[7] = {0.0, 0.3, 0.9, 1.1, 0.2, -0.8, -0.6};
  for (std::size_t k = 0; k < 7; ++k) p.constraints.push_back({k, 0, {heading[k]}});
  for (std::size_t k : {0u, 2u, 6u}) p.constraints.push_back({k, 1, {0.0}});
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

Problem fixture(const std::string& name) {
  if (name == "waypoint") return waypoint_problem();
  if (name == "short") return short_segment_problem();
  if (name == "chain4") return gate_chain_problem(4);
  if (name == "chain8") return gate_chain_problem(8);
  if (name == "yaw") return yaw_chain_problem();
  throw std::logic_error("unknown fixture " + name);
}

TEST(MinSnap, MatchesTheDenseLuSolverItReplaced) {
  for (const Reference& ref : references()) {
    const Problem p = fixture(ref.name);
    const Trajectory t = solve(p);
    ASSERT_EQ(ref.coefficients.size(), t.segments() * t.channels() * kCoefficients) << ref.name;
    EXPECT_NEAR(t.cost(), ref.cost, 1e-9 * ref.cost) << ref.name;
    for (std::size_t seg = 0; seg < t.segments(); ++seg) {
      for (std::size_t ch = 0; ch < t.channels(); ++ch) {
        for (std::size_t k = 0; k < kCoefficients; ++k) {
          const double want = ref.coefficients[(seg * t.channels() + ch) * kCoefficients + k];
          EXPECT_NEAR(t.coefficients(seg, ch)[k], want, 1e-9 * std::max(1.0, std::fabs(want)))
              << ref.name << " segment " << seg << " channel " << ch << " c" << k;
        }
      }
    }
  }
}

TEST(MinSnap, BasisRowsAndCoefficientsReproduceTheDerivatives) {
  const Trajectory t = solve(waypoint_problem());
  for (std::size_t seg : {0u, 2u, 3u}) {
    for (double tau : {0.0, 0.3, 0.97}) {
      for (int d = 0; d <= kMaxDerivative; ++d) {
        const auto row = basis_row(tau, d);
        double v = 0.0;
        for (std::size_t k = 0; k < kCoefficients; ++k) v += row[k] * t.coefficients(seg, 1)[k];
        const double T = t.durations()[seg];
        const double at = t.knot_times()[seg] + tau * T;
        EXPECT_NEAR(v / std::pow(T, d), t.value(at, d, 1), 1e-9 * (1.0 + std::fabs(v)))
            << "segment " << seg << " tau " << tau << " d" << d;
      }
    }
  }
}

// An objective of the solution: its value and its partials at fixed everything else, the
// inputs of SolvedProblem::adjoint.
struct Objective {
  double value = 0.0;
  std::vector<double> d_coefficients;  // [(segment * channels + channel) * 8 + k]
  std::vector<double> d_durations;     // explicit, at fixed coefficients
};
using ObjectiveFn = std::function<Objective(const Trajectory&)>;

// Sum over segments, channels and fixed tau samples of the squared acceleration
// (T^-2 b_2(tau) . c)^2: explicit in T as well as in c.
Objective squared_acceleration(const Trajectory& t) {
  Objective o;
  o.d_coefficients.assign(t.segments() * t.channels() * kCoefficients, 0.0);
  o.d_durations.assign(t.segments(), 0.0);
  for (std::size_t seg = 0; seg < t.segments(); ++seg) {
    const double T = t.durations()[seg];
    for (std::size_t ch = 0; ch < t.channels(); ++ch) {
      for (double tau : {0.0, 0.1, 0.35, 0.5, 0.8, 1.0}) {
        const auto row = basis_row(tau, 2);
        double a = 0.0;
        for (std::size_t k = 0; k < kCoefficients; ++k) a += row[k] * t.coefficients(seg, ch)[k];
        a /= T * T;
        o.value += a * a;
        o.d_durations[seg] += -4.0 * a * a / T;
        for (std::size_t k = 0; k < kCoefficients; ++k) {
          o.d_coefficients[(seg * t.channels() + ch) * kCoefficients + k] +=
              2.0 * a * row[k] / (T * T);
        }
      }
    }
  }
  return o;
}

// Central finite differences of `value(problem)` against an analytic gradient, for every
// duration and every constraint value.
void expect_gradient_matches_fd(const Problem& p, const SolvedProblem::Gradient& g,
                                const std::function<double(const Problem&)>& value) {
  double biggest = 0.0;
  for (double v : g.d_durations) biggest = std::max(biggest, std::fabs(v));
  for (const auto& row : g.d_constraints) {
    for (double v : row) biggest = std::max(biggest, std::fabs(v));
  }
  ASSERT_GT(biggest, 0.0);
  auto check = [&](double analytic, double fd, const std::string& what) {
    EXPECT_NEAR(analytic, fd, 1e-5 * std::max(std::fabs(fd), 1e-3 * biggest)) << what;
  };
  ASSERT_EQ(g.d_durations.size(), p.durations.size());
  for (std::size_t i = 0; i < p.durations.size(); ++i) {
    Problem hi = p, lo = p;
    const double h = 1e-5 * p.durations[i];
    hi.durations[i] += h;
    lo.durations[i] -= h;
    check(g.d_durations[i], (value(hi) - value(lo)) / (2.0 * h), "duration " + std::to_string(i));
  }
  ASSERT_EQ(g.d_constraints.size(), p.constraints.size());
  for (std::size_t i = 0; i < p.constraints.size(); ++i) {
    ASSERT_EQ(g.d_constraints[i].size(), p.channels);
    for (std::size_t ch = 0; ch < p.channels; ++ch) {
      Problem hi = p, lo = p;
      const double h = 1e-5 * (1.0 + std::fabs(p.constraints[i].value[ch]));
      hi.constraints[i].value[ch] += h;
      lo.constraints[i].value[ch] -= h;
      check(g.d_constraints[i][ch], (value(hi) - value(lo)) / (2.0 * h),
            "constraint " + std::to_string(i) + " d" + std::to_string(p.constraints[i].derivative) +
                " channel " + std::to_string(ch));
    }
  }
}

TEST(MinSnapAdjoint, CostGradientMatchesFiniteDifferences) {
  for (const char* name : {"waypoint", "short", "chain4", "yaw"}) {
    SCOPED_TRACE(name);
    const Problem p = fixture(name);
    expect_gradient_matches_fd(p, SolvedProblem(p).cost_gradient(),
                               [](const Problem& q) { return solve(q).cost(); });
  }
}

TEST(MinSnapAdjoint, SampledValueFunctionalMatchesFiniteDifferences) {
  for (const char* name : {"waypoint", "short", "chain4", "yaw"}) {
    SCOPED_TRACE(name);
    const Problem p = fixture(name);
    const SolvedProblem solved(p);
    const Objective o = squared_acceleration(solved.trajectory());
    expect_gradient_matches_fd(
        p, solved.adjoint(o.d_coefficients, o.d_durations),
        [](const Problem& q) { return squared_acceleration(solve(q)).value; });
  }
}

TEST(MinSnapAdjoint, RejectsGradientsOfTheWrongSize) {
  const SolvedProblem solved(waypoint_problem());
  EXPECT_THROW(solved.adjoint({1.0}, {1.0, 1.0, 1.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(solved.adjoint(std::vector<double>(4 * 3 * kCoefficients, 0.0), {1.0}),
               std::invalid_argument);
}

TEST(MinSnapTiming, PrintsSolveTimesForGateHorizons) {
  // Not asserted (CI machines vary): the plan's targets are 24 gates < 5 ms and 64 < 20 ms
  // in a Release build.
  using Clock = std::chrono::steady_clock;
  for (int gates : {8, 24, 64}) {
    const Problem p = gate_chain_problem(gates);
    std::vector<double> solve_ms, gradient_ms;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = Clock::now();
      const SolvedProblem solved(p);
      const auto t1 = Clock::now();
      const SolvedProblem::Gradient g = solved.cost_gradient();
      const auto t2 = Clock::now();
      ASSERT_TRUE(std::isfinite(g.d_durations.front()));
      solve_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      gradient_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    std::sort(solve_ms.begin(), solve_ms.end());
    std::sort(gradient_ms.begin(), gradient_ms.end());
    std::printf(
        "[ TIMING ] %2d gates (%3zu segments): solve %.2f ms, cost gradient %.2f ms (median of "
        "5)\n",
        gates, p.durations.size(), solve_ms[2], gradient_ms[2]);
  }
}

}  // namespace
}  // namespace astrum::minsnap
