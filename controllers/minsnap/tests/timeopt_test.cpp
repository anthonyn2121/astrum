#include "timeopt.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>

#include "flatness.hpp"

namespace astrum::minsnap {
namespace {

constexpr double kHalfPi = std::numbers::pi / 2.0;

Limits loose_limits() {
  Limits l;
  l.v_max = 100.0;
  l.f_max = 100.0;
  l.f_min = 0.0;
  l.tilt_max_rad = 3.0;
  l.rate_max = 1000.0;
  l.yaw_rate_max = 1000.0;
  return l;
}

// A straight rest-to-rest line of `length` metres along x in `segments` equal-length
// pieces (velocity and acceleration pinned to zero at both ends), yaw 0 throughout.
TimeOptModel line_model(std::size_t segments, double length) {
  TimeOptModel m;
  m.position.cost_order = 4;
  m.position.channels = 3;
  m.position.durations.assign(segments, 1.0);
  for (std::size_t k = 0; k <= segments; ++k) {
    m.position.constraints.push_back(
        {k, 0, {length * static_cast<double>(k) / static_cast<double>(segments), 0.0, 1.0}});
  }
  for (int d = 1; d <= 2; ++d) {
    m.position.constraints.push_back({0, d, {0.0, 0.0, 0.0}});
    m.position.constraints.push_back({segments, d, {0.0, 0.0, 0.0}});
  }
  m.yaw.cost_order = 2;
  m.yaw.channels = 1;
  m.yaw.durations = {1.0};
  m.yaw.constraints = {{0, 0, {0.0}}, {1, 0, {0.0}}};
  m.yaw_spans = {{0, segments}};
  return m;
}

// Rest at A = (0,0,1); a gate at B = (8,0,1) crossed along +x; a gate at C = (8,8,1)
// crossed along +y (a 90 degree turn); rest at E = (8,16,1). The heading follows the
// travel direction: 0 at A and B, 90 degrees at C and E. `merged_yaw` spans the legs B-C
// and C-E with one yaw segment, so the yaw samples sit at an offset into a longer segment.
TimeOptModel turn_model(bool merged_yaw = false) {
  TimeOptModel m;
  m.position.cost_order = 4;
  m.position.channels = 3;
  m.position.durations = {1.0, 1.0, 1.0};
  const double p[4][3] = {{0, 0, 1}, {8, 0, 1}, {8, 8, 1}, {8, 16, 1}};
  for (std::size_t k = 0; k < 4; ++k)
    m.position.constraints.push_back({k, 0, {p[k][0], p[k][1], p[k][2]}});
  for (int d = 1; d <= 2; ++d) {
    m.position.constraints.push_back({0, d, {0.0, 0.0, 0.0}});
    m.position.constraints.push_back({3, d, {0.0, 0.0, 0.0}});
  }
  m.gate_pins.push_back({m.position.constraints.size(), {1.0, 0.0, 0.0}});
  m.position.constraints.push_back({1, 1, {1.0, 0.0, 0.0}});
  m.gate_pins.push_back({m.position.constraints.size(), {0.0, 1.0, 0.0}});
  m.position.constraints.push_back({2, 1, {0.0, 1.0, 0.0}});

  m.yaw.cost_order = 2;
  m.yaw.channels = 1;
  if (merged_yaw) {
    m.yaw.durations = {1.0, 1.0};
    m.yaw.constraints = {{0, 0, {0.0}}, {1, 0, {0.0}}, {2, 0, {kHalfPi}},
                         {0, 1, {0.0}}, {1, 1, {0.0}}, {2, 1, {0.0}}};
    m.yaw_spans = {{0, 1}, {1, 3}};
  } else {
    m.yaw.durations = {1.0, 1.0, 1.0};
    m.yaw.constraints = {{0, 0, {0.0}}, {1, 0, {0.0}}, {2, 0, {kHalfPi}}, {3, 0, {kHalfPi}},
                         {0, 1, {0.0}}, {1, 1, {0.0}}, {2, 1, {0.0}},     {3, 1, {0.0}}};
    m.yaw_spans = {{0, 1}, {1, 2}, {2, 3}};
  }
  return m;
}

struct Peaks {
  double speed = 0.0, thrust_max = 0.0, thrust_min = 1e30, tilt = 0.0, rate = 0.0, yaw_rate = 0.0;
};

// The candidate's trajectories on a fine grid, measured independently of the optimiser.
Peaks peaks_of(const TimeOptModel& model, const std::vector<double>& durations,
               const std::vector<double>& speeds) {
  Problem position = model.position;
  position.durations = durations;
  for (std::size_t g = 0; g < speeds.size(); ++g) {
    auto& v = position.constraints[model.gate_pins[g].constraint].value;
    for (std::size_t ch = 0; ch < 3; ++ch) v[ch] = speeds[g] * model.gate_pins[g].direction[ch];
  }
  Problem yaw = model.yaw;
  for (std::size_t q = 0; q < model.yaw_spans.size(); ++q) {
    yaw.durations[q] = 0.0;
    for (std::size_t i = model.yaw_spans[q].first; i < model.yaw_spans[q].second; ++i) {
      yaw.durations[q] += durations[i];
    }
  }
  const Trajectory p = solve(position), y = solve(yaw);
  Peaks peaks;
  for (double t = 0.0; t <= p.total_time(); t += 0.005) {
    const auto v = p.value(t, 1), a = p.value(t, 2), j = p.value(t, 3);
    peaks.speed = std::max(peaks.speed, norm({v[0], v[1], v[2]}));
    const Vec3 thrust = Vec3{a[0], a[1], a[2]} + kUnitZ * kGravity;
    peaks.thrust_max = std::max(peaks.thrust_max, norm(thrust));
    peaks.thrust_min = std::min(peaks.thrust_min, norm(thrust));
    peaks.tilt = std::max(peaks.tilt, std::acos(thrust.z / norm(thrust)));
    peaks.yaw_rate = std::max(peaks.yaw_rate, std::fabs(y.value(t, 1, 0)));
    if (const auto flat = flat_state(
            {{a[0], a[1], a[2]}, {j[0], j[1], j[2]}, y.value(t, 0, 0), y.value(t, 1, 0)})) {
      peaks.rate = std::max(peaks.rate, std::hypot(flat->omega_body.x, flat->omega_body.y));
    }
  }
  return peaks;
}

double sum(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x;
  return s;
}

TEST(TimeOpt, StraightLineApproachesTheDoubleIntegratorBound) {
  for (std::size_t segments : {std::size_t{1}, std::size_t{2}}) {
    const double length = 10.0;
    Limits limits = loose_limits();
    limits.f_max = 15.0;
    const TimeOptModel model = line_model(segments, length);
    // Just feasible at the start, as the heuristic plan the optimiser improves on is.
    const std::vector<double> d0(segments, 3.4 / static_cast<double>(segments));
    const TimeOptResult r = optimise_times(model, d0, {}, limits, TimeOptParams{});

    // Bang-bang rest to rest under |a| <= a_max takes 2 sqrt(L / a_max) with unbounded
    // jerk; the thrust limit |a + g e_z| <= margin f_max leaves a_max horizontally.
    const double thrust = limits.margin * limits.f_max;
    const double a_max = std::sqrt(thrust * thrust - kGravity * kGravity);
    const double bound = 2.0 * std::sqrt(length / a_max);
    std::cout << "[ TimeOpt ] line, " << segments << " segment(s): " << r.total_time << " s, bound "
              << bound << " s, ratio " << r.total_time / bound << " (from " << sum(d0) << " s, "
              << r.iterations << " iterations)\n";
    EXPECT_TRUE(r.improved);
    EXPECT_GE(r.total_time, bound);
    EXPECT_LE(r.total_time, 1.35 * bound);
    const Peaks p = peaks_of(model, r.durations, r.gate_speeds);
    EXPECT_LE(p.thrust_max, limits.f_max);
    EXPECT_GE(p.thrust_max, 0.95 * thrust);  // the thrust limit is what binds
  }
}

TEST(TimeOpt, NinetyDegreeTurnIsFasterAndWithinTheLimits) {
  Limits limits = loose_limits();
  limits.v_max = 8.0;
  limits.f_max = 16.0;
  limits.f_min = 3.0;
  limits.tilt_max_rad = 1.0;
  limits.rate_max = 2.5;
  limits.yaw_rate_max = 2.0;
  for (bool merged : {false, true}) {
    const TimeOptModel model = turn_model(merged);
    const std::vector<double> d0 = {4.0, 4.0, 4.0};
    const TimeOptResult r = optimise_times(model, d0, {2.0, 2.0}, limits, TimeOptParams{});
    std::cout << "[ TimeOpt ] turn" << (merged ? " (merged yaw)" : "") << ": " << r.total_time
              << " s from " << sum(d0) << " s, gate speeds " << r.gate_speeds[0] << ", "
              << r.gate_speeds[1] << " m/s, violation " << r.max_violation << "\n";
    EXPECT_TRUE(r.improved);
    EXPECT_LT(r.total_time, 0.6 * sum(d0));
    EXPECT_LE(r.max_violation, 5e-3);
    const Peaks p = peaks_of(model, r.durations, r.gate_speeds);
    EXPECT_LE(p.speed, limits.v_max * 1.005);
    EXPECT_LE(p.thrust_max, limits.f_max * 1.005);
    EXPECT_GE(p.thrust_min, limits.f_min * 0.995);
    EXPECT_LE(p.tilt, limits.tilt_max_rad * 1.005);
    EXPECT_LE(p.rate, limits.rate_max * 1.005);
    EXPECT_LE(p.yaw_rate, limits.yaw_rate_max * 1.005);
  }
}

TEST(TimeOpt, TheBodyRateLimitBindsWhenItIsTheTightOne) {
  Limits limits = loose_limits();
  limits.f_max = 30.0;
  limits.rate_max = 0.7;
  const TimeOptModel model = turn_model();
  const TimeOptResult r =
      optimise_times(model, {8.0, 8.0, 8.0}, {1.5, 1.5}, limits, TimeOptParams{});
  const Peaks p = peaks_of(model, r.durations, r.gate_speeds);
  std::cout << "[ TimeOpt ] rate-limited turn: " << r.total_time << " s, peak rate " << p.rate
            << " of " << limits.rate_max << " rad/s, peak thrust " << p.thrust_max << "\n";
  EXPECT_TRUE(r.improved);
  EXPECT_LE(p.rate, limits.rate_max * 1.005);
  EXPECT_GE(p.rate, 0.9 * limits.margin * limits.rate_max);
  EXPECT_LT(p.thrust_max, 0.8 * limits.f_max);  // and nothing else is close
}

TEST(TimeOpt, NeverWorseThanTheStartingPoint) {
  Limits limits = loose_limits();
  limits.f_max = 16.0;
  limits.rate_max = 1.5;
  const TimeOptModel model = turn_model();
  const std::vector<double> d0 = {6.0, 6.0, 6.0};
  const std::vector<double> s0 = {2.0, 2.0};
  for (int iterations : {0, 1, 5, 40, 150}) {
    TimeOptParams params;
    params.iterations = iterations;
    const TimeOptResult r = optimise_times(model, d0, s0, limits, params);
    EXPECT_LE(r.total_time, sum(d0) + 1e-12);
    EXPECT_EQ(r.improved, r.total_time < sum(d0));
    EXPECT_LE(r.max_violation, 5e-3);
  }
  // Restarting from an optimum does not lose anything either.
  const TimeOptResult once = optimise_times(model, d0, s0, limits, TimeOptParams{});
  const TimeOptResult twice =
      optimise_times(model, once.durations, once.gate_speeds, limits, TimeOptParams{});
  EXPECT_LE(twice.total_time, once.total_time);
}

TEST(TimeOpt, AnInfeasibleStartBecomesFeasibleEvenIfThatTakesLonger) {
  Limits limits = loose_limits();
  limits.rate_max = 0.2;  // the start flies far above this
  const TimeOptModel model = turn_model();
  const std::vector<double> d0 = {2.0, 2.0, 2.0};
  const TimeOptResult r = optimise_times(model, d0, {2.0, 2.0}, limits, TimeOptParams{});
  const Peaks p0 = peaks_of(model, d0, {2.0, 2.0});
  const Peaks p = peaks_of(model, r.durations, r.gate_speeds);
  EXPECT_GT(p0.rate, limits.rate_max);
  EXPECT_TRUE(r.improved);
  EXPECT_LE(r.max_violation, 5e-3);
  EXPECT_LE(p.rate, limits.rate_max * 1.006);
  EXPECT_GT(r.total_time, sum(d0));  // feasibility first: the lap gets longer
  // With no iterations to spend the start is returned as it is.
  TimeOptParams none;
  none.iterations = 0;
  const TimeOptResult same = optimise_times(model, d0, {2.0, 2.0}, limits, none);
  EXPECT_FALSE(same.improved);
  EXPECT_EQ(same.durations, d0);
  EXPECT_GT(same.max_violation, 0.5);
}

TEST(TimeOpt, TheDenseGridStaysInsideTheTrueLimits) {
  Limits limits = loose_limits();
  limits.v_max = 6.0;
  limits.f_max = 14.0;
  limits.f_min = 5.0;
  limits.tilt_max_rad = 0.9;
  limits.rate_max = 1.2;
  limits.yaw_rate_max = 0.8;
  const TimeOptModel model = turn_model();
  const TimeOptResult r =
      optimise_times(model, {10.0, 10.0, 10.0}, {1.5, 1.5}, limits, TimeOptParams{});
  const Peaks p = peaks_of(model, r.durations, r.gate_speeds);
  EXPECT_TRUE(r.improved);
  EXPECT_LE(p.speed, limits.v_max);
  EXPECT_LE(p.thrust_max, limits.f_max);
  EXPECT_GE(p.thrust_min, limits.f_min);
  EXPECT_LE(p.tilt, limits.tilt_max_rad);
  EXPECT_LE(p.rate, limits.rate_max);
  EXPECT_LE(p.yaw_rate, limits.yaw_rate_max);
}

TEST(TimeOpt, IsBitwiseDeterministic) {
  Limits limits = loose_limits();
  limits.f_max = 16.0;
  limits.rate_max = 1.5;
  const TimeOptModel model = turn_model(true);
  const TimeOptResult a =
      optimise_times(model, {6.0, 6.0, 6.0}, {2.0, 2.0}, limits, TimeOptParams{});
  const TimeOptResult b =
      optimise_times(model, {6.0, 6.0, 6.0}, {2.0, 2.0}, limits, TimeOptParams{});
  EXPECT_EQ(a.durations, b.durations);
  EXPECT_EQ(a.gate_speeds, b.gate_speeds);
  EXPECT_EQ(a.total_time, b.total_time);
  EXPECT_EQ(a.iterations, b.iterations);
}

// One constraint at a time tight enough to be violated by the starting layout, so the
// penalty of each is exercised, with the gradient checked against central differences.
TEST(TimeOpt, GradientMatchesCentralDifferences) {
  struct Case {
    const char* name;
    void (*tighten)(Limits&);
  };
  const Case cases[] = {
      {"speed", [](Limits& l) { l.v_max = 2.2; }},
      {"thrust max", [](Limits& l) { l.f_max = 10.2; }},
      {"thrust min", [](Limits& l) { l.f_min = 9.6; }},
      {"tilt", [](Limits& l) { l.tilt_max_rad = 0.08; }},
      {"body rate", [](Limits& l) { l.rate_max = 0.05; }},
      {"yaw rate", [](Limits& l) { l.yaw_rate_max = 0.3; }},
  };
  for (bool merged : {false, true}) {
    for (const Case& c : cases) {
      for (double transient : {0.0, 1.5}) {
        Limits limits = loose_limits();
        c.tighten(limits);
        TimeOptParams params;
        params.start_transient_s = transient;
        const TimeOptModel model = turn_model(merged);
        const std::vector<double> d0 = {2.5, 3.0, 2.0};
        const std::vector<double> s0 = {2.5, 2.0};
        const TimeObjective objective(model, limits, params, d0, s0);
        std::vector<double> z = objective.variables(d0, s0);
        const double mu = 10.0;
        const auto base = objective.evaluate(z, mu);
        if (transient > 0.0 && std::string(c.name) != "speed" &&
            std::string(c.name) != "yaw rate") {
          // Only the first 1.5 s are exempt; the rest of the plan still sees the limit.
          EXPECT_GT(base.penalty, 0.0) << c.name;
        } else {
          ASSERT_GT(base.penalty, 0.0) << c.name;
        }
        ASSERT_EQ(base.gradient.size(), z.size());
        for (std::size_t i = 0; i < z.size(); ++i) {
          // Relative steps (the yaw KKT solve carries ~1e-12 of noise in F: smaller ones
          // lose the gradient in it).
          const bool is_duration = i < model.position.durations.size();
          const double h = is_duration ? 1e-4 : 1e-4 * z[i];
          std::vector<double> up = z, down = z;
          up[i] += h;
          down[i] -= h;
          const double fd =
              (objective.evaluate(up, mu).objective - objective.evaluate(down, mu).objective) /
              (2.0 * h);
          EXPECT_NEAR(base.gradient[i], fd, 1e-5 * std::max(1.0, std::fabs(fd)))
              << c.name << (merged ? " merged" : "") << " transient " << transient << " var " << i;
        }
      }
    }
  }
}

TEST(TimeOpt, TheDenseGridIsSkippedWhenTheSamplesAlreadyViolateALimit) {
  // A starting point well past the speed limit: the samples say so, whatever the grid does.
  Limits limits = loose_limits();
  limits.v_max = 1.0;
  const TimeOptModel model = turn_model();
  const TimeOptParams params;
  const std::vector<double> d0 = {2.5, 3.0, 2.0};
  const std::vector<double> s0 = {0.9, 0.9};
  const TimeObjective objective(model, limits, params, d0, s0);
  const std::vector<double> z = objective.variables(d0, s0);
  const auto walked = objective.evaluate(z, 1.0, true);
  ASSERT_TRUE(walked.dense_excess);
  ASSERT_GT(*walked.dense_excess, 0.1);
  // With the cutoff below that, the sampled excess stands in (the samples and the grid
  // are different points of the same trajectory, so they agree only closely).
  const auto skipped = objective.evaluate(z, 1.0, true, 0.01);
  ASSERT_TRUE(skipped.dense_excess);
  EXPECT_GT(*skipped.dense_excess, 0.01);
  EXPECT_NEAR(*skipped.dense_excess, *walked.dense_excess, 0.05 * *walked.dense_excess);
  // A cutoff above the true excess still walks the grid.
  EXPECT_DOUBLE_EQ(*objective.evaluate(z, 1.0, true, 10.0).dense_excess, *walked.dense_excess);
  // Same objective and gradient either way.
  EXPECT_DOUBLE_EQ(skipped.objective, walked.objective);
}

TEST(TimeOpt, RejectsInconsistentModels) {
  const TimeOptModel model = turn_model();
  const Limits limits = loose_limits();
  EXPECT_THROW(optimise_times(model, {1.0, 1.0}, {1.0, 1.0}, limits, TimeOptParams{}),
               std::invalid_argument);
  EXPECT_THROW(optimise_times(model, {1.0, 1.0, 1.0}, {1.0}, limits, TimeOptParams{}),
               std::invalid_argument);
  TimeOptModel gap = model;
  gap.yaw_spans = {{0, 1}, {2, 3}, {3, 3}};
  EXPECT_THROW(optimise_times(gap, {1.0, 1.0, 1.0}, {1.0, 1.0}, limits, TimeOptParams{}),
               std::invalid_argument);
  Limits bad = limits;
  bad.f_max = 0.0;
  EXPECT_THROW(optimise_times(model, {1.0, 1.0, 1.0}, {1.0, 1.0}, bad, TimeOptParams{}),
               std::invalid_argument);
}

}  // namespace
}  // namespace astrum::minsnap
