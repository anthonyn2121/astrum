#include "planner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "flatness.hpp"
#include "support.hpp"

namespace astrum::minsnap {
namespace {

using testing::make_course;
using testing::make_gate;

constexpr double kPi = 3.14159265358979323846;

Course slalom() {
  return make_course({make_gate({0.0, 6.0, 2.0}, 0.0), make_gate({4.0, 12.0, 2.5}, 40.0),
                      make_gate({10.0, 14.0, 1.6}, 90.0), make_gate({16.0, 10.0, 2.2}, 150.0)});
}

StartState start_state(const Vec3& p, const Vec3& v, double flat_yaw) {
  StartState s;
  s.position = p;
  s.velocity = v;
  s.yaw = flat_yaw;
  return s;
}

StartState at_rest(const Vec3& p, double heading_rad) {
  return start_state(p, {0, 0, 0}, kPi / 2 - heading_rad);
}

PlanParams heuristic() {
  PlanParams p;
  p.planner = PlannerKind::Heuristic;
  return p;
}

// The worst use of every limit of `params` along the plan, as a fraction of the limit
// (1 = at the limit; the thrust floor as limit / thrust).
struct Use {
  double speed = 0, thrust = 0, floor = 0, tilt = 0, rate = 0, yaw_rate = 0;
};

Use use_of(const Plan& plan, const PlanParams& params, double from_s = 0.0) {
  Use u;
  for (double t = from_s; t <= plan.duration(); t += 0.01) {
    const FlatReference r = plan.at(t);
    const Vec3 thrust = r.acceleration + kUnitZ * kGravity;
    u.speed = std::max(u.speed, norm(r.velocity) / params.max_speed_mps);
    u.thrust = std::max(u.thrust, norm(thrust) / params.max_thrust_mps2);
    u.floor = std::max(u.floor, params.min_thrust_mps2 / norm(thrust));
    u.tilt = std::max(u.tilt, std::acos(thrust.z / norm(thrust)) / params.max_tilt_rad);
    u.yaw_rate = std::max(u.yaw_rate, std::fabs(r.yaw_rate) / params.max_yaw_rate);
    if (const auto flat = flat_state({r.acceleration, r.jerk, r.yaw, r.yaw_rate})) {
      u.rate = std::max(u.rate,
                        std::hypot(flat->omega_body.x, flat->omega_body.y) / params.max_rate_rad_s);
    }
  }
  return u;
}

TEST(Planner, PassesEveryGateCentreSquareOnInOrder) {
  const Course c = slalom();
  const Plan plan = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  ASSERT_EQ(plan.gate_count(), 4u);
  EXPECT_EQ(plan.first_gate(), 0u);
  double previous = 0.0;
  for (std::size_t g = 0; g < 4; ++g) {
    const double t = plan.gate_times()[g];
    EXPECT_GT(t, previous);
    previous = t;
    const FlatReference r = plan.at(t);
    EXPECT_NEAR(norm(r.position - c.gates[g].center), 0.0, 1e-9) << "gate " << g;
    // Velocity along the normal (square crossing), the yaw along the normal too, no spin.
    EXPECT_NEAR(norm(cross(unit_or(r.velocity, kUnitZ), c.gates[g].normal)), 0.0, 1e-9)
        << "gate " << g;
    EXPECT_GT(dot(r.velocity, c.gates[g].normal), 0.2);
    EXPECT_NEAR(std::remainder(r.yaw - heading_of(c.gates[g].normal), 2 * kPi), 0.0, 1e-9)
        << "gate " << g;
    EXPECT_NEAR(r.yaw_rate, 0.0, 1e-9);
  }
}

TEST(Planner, RespectsItsOwnLimitsAndKeepsClear) {
  const Course c = slalom();
  PlanParams params;
  const Plan plan = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), params);
  EXPECT_LE(plan.peak_speed(), params.max_speed_mps * 1.01);
  EXPECT_LE(plan.peak_accel(), params.max_accel_mps2 * 1.01);
  EXPECT_LE(plan.peak_rate(), params.max_rate_rad_s * 1.01);
  EXPECT_GE(plan.min_margin(), 0.0);
  // A square crossing needs lateral clearance: the sampled path stays within the opening
  // minus the square-on reach at the crossing itself.
  for (std::size_t g = 0; g < 4; ++g) {
    const Vec3 at = plan.at(plan.gate_times()[g]).position;
    EXPECT_NEAR(dot(at - c.gates[g].center, c.gates[g].right), 0.0, 1e-9);
  }
}

TEST(Planner, StartsFromTheGivenStateAndHoldsAtTheEnd) {
  const Course c = slalom();
  const StartState s = start_state({0.2, -0.1, 2.05}, {0.0, 1.0, 0.0}, kPi / 2);
  const Plan plan = plan_race(c, 0, s, PlanParams{});
  const FlatReference r0 = plan.at(0.0);
  EXPECT_NEAR(norm(r0.position - s.position), 0.0, 1e-9);
  EXPECT_NEAR(norm(r0.velocity - s.velocity), 0.0, 1e-9);
  EXPECT_NEAR(r0.yaw, s.yaw, 1e-9);
  // Before 0 and after the end the reference is the end state at rest.
  EXPECT_EQ(plan.at(-5.0).position, r0.position);
  const FlatReference end = plan.at(plan.duration() + 10.0);
  EXPECT_EQ(end.velocity, Vec3{});
  EXPECT_EQ(end.acceleration, Vec3{});
  EXPECT_NEAR(norm(end.position - plan.at(plan.duration()).position), 0.0, 1e-12);
}

TEST(Planner, TimeScaleSlowsThePlanDown) {
  const Course c = slalom();
  PlanParams fast;
  fast.max_rate_rad_s = fast.max_accel_mps2 = fast.max_speed_mps = 100.0;  // nothing stretches it
  fast.max_yaw_rate = 100.0;
  PlanParams slow = fast;
  slow.time_scale = 2.0;
  const Plan a = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), fast);
  const Plan b = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), slow);
  EXPECT_NEAR(b.duration(), 2.0 * a.duration(), 1e-9);
  // Same path, half the speed (section V-A): the gate speeds are scaled with the times.
  EXPECT_NEAR(norm(b.at(2.0 * a.gate_times()[1]).velocity) * 2.0,
              norm(a.at(a.gate_times()[1]).velocity), 1e-6);
  EXPECT_LT(b.peak_accel(), a.peak_accel() * 0.3);
}

TEST(Planner, LimitsStretchTheTimeline) {
  const Course c = slalom();
  PlanParams loose;
  loose.max_rate_rad_s = loose.max_accel_mps2 = loose.max_speed_mps = 100.0;
  loose.max_yaw_rate = 100.0;
  loose.planner = PlannerKind::Heuristic;
  PlanParams tight = loose;
  tight.max_accel_mps2 = 2.0;
  const Plan a = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), loose);
  const Plan b = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), tight);
  EXPECT_GT(b.duration(), a.duration());
  EXPECT_LE(b.peak_accel(), 2.0 * 1.01);
  EXPECT_GT(a.peak_accel(), 2.0);
}

TEST(Planner, ContinuesAnotherPlanWithItsAccelerationAndYawRate) {
  const Course c = slalom();
  const Plan first = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  const double t = first.gate_times()[0] + 1.0;  // mid-turn, away from any pin
  const FlatReference r = first.at(t);
  StartState s = start_state(r.position, r.velocity, r.yaw);
  s.acceleration = r.acceleration;
  s.yaw_rate = r.yaw_rate;
  const Plan next = plan_race(c, 1, s, PlanParams{});
  const FlatReference n = next.at(0.0);
  EXPECT_NEAR(norm(n.position - r.position), 0.0, 1e-9);
  EXPECT_NEAR(norm(n.velocity - r.velocity), 0.0, 1e-9);
  EXPECT_NEAR(norm(n.acceleration - r.acceleration), 0.0, 1e-8);
  EXPECT_NEAR(n.yaw_rate, r.yaw_rate, 1e-9);
}

TEST(Planner, CoversAtMostMaxGatesGates) {
  const Course c = slalom();
  PlanParams params;
  params.max_gates = 2;
  const Plan plan = plan_race(c, 1, at_rest({4, 11, 2.5}, 0.7), params);
  EXPECT_EQ(plan.first_gate(), 1u);
  EXPECT_EQ(plan.gate_count(), 2u);  // gates 1 and 2 of 1..3
}

TEST(Planner, SegmentTimeOptimisationKeepsTheCrossings) {
  const Course c = slalom();
  PlanParams params;
  params.realloc_iterations = 3;
  const Plan plan = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), params);
  for (std::size_t g = 0; g < 4; ++g) {
    EXPECT_NEAR(norm(plan.at(plan.gate_times()[g]).position - c.gates[g].center), 0.0, 1e-9)
        << "gate " << g;
  }
  EXPECT_GE(plan.min_margin(), 0.0);
}

TEST(Planner, YawRateLimitSlowsSharpTurns) {
  // Two gates, the second facing back the way we came: a 180-degree heading change.
  const Course c = make_course({make_gate({0, 5, 2}, 0.0), make_gate({0, 10, 2}, 180.0)});
  PlanParams params;
  const Plan plan = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), params);
  const double leg = plan.gate_times()[1] - plan.gate_times()[0];
  EXPECT_GE(leg, 1.5 * kPi / params.max_yaw_rate * 0.999 -
                     2.0);  // the exit/entry segments are outside the span
}

TEST(Planner, PlansFromAMidCourseGateAndRoutesAroundAFrame) {
  // Gate 0 is a blocker facing north; the drone is south of it and gate 2 lies straight
  // behind its left upright, so the direct line to gate 2's entry runs through the frame.
  const double x = -(testing::kOpening / 2.0 + testing::kBar / 2.0);
  const Course c = make_course({make_gate({0.0, 0.0, 2.0}, 0.0), make_gate({0.0, 3.0, 5.0}, 0.0),
                                make_gate({x, 6.0, 2.0}, 0.0)});
  const Vec3 start{x, -4.0, 2.0};
  PlanParams params;
  const Hazards h{&c, params.hull_radius_m, params.floor_m};
  ASSERT_FALSE(segment_clear(start, {x, 5.0, 2.0}, h));
  const Plan plan = plan_race(c, 2, start_state(start, {0, 0, 0}, kPi / 2), params);
  EXPECT_EQ(plan.first_gate(), 2u);
  ASSERT_EQ(plan.gate_count(), 1u);
  EXPECT_GT(plan.waypoints().size(), 4u);  // start, route point(s), entry, centre, exit
  EXPECT_GE(plan.min_margin(), 0.0);
  EXPECT_NEAR(norm(plan.at(plan.gate_times()[0]).position - c.gates[2].center), 0.0, 1e-9);
}

TEST(Planner, RepairsATrajectoryThatCutsACorner) {
  // A tight dog-leg round a frame: the min-snap spline, left alone, bulges into it.
  const Course c = make_course({make_gate({0.0, 8.0, 2.0}, 90.0), make_gate({3.0, 8.0, 2.0}, 0.0),
                                make_gate({3.0, 14.0, 2.0}, 0.0)});
  PlanParams off = heuristic();
  off.refine_iterations = 0;
  const Plan raw = plan_race(c, 0, at_rest({-6.0, 8.0, 2.0}, kPi / 2), off);
  const Plan plan = plan_race(c, 0, at_rest({-6.0, 8.0, 2.0}, kPi / 2), heuristic());
  EXPECT_LT(raw.min_margin(), 0.0);  // without the repair pass the spline hits the frame
  EXPECT_GE(plan.min_margin(), 0.0);
  EXPECT_GT(plan.waypoints().size(), raw.waypoints().size());  // a corridor keyframe was added
  // The time-optimal plan keeps clear of the frame too, with or without the first repair.
  const Plan optimal = plan_race(c, 0, at_rest({-6.0, 8.0, 2.0}, kPi / 2), PlanParams{});
  EXPECT_GE(optimal.min_margin(), 0.0);
  EXPECT_TRUE(optimal.info().improved);
}

TEST(Planner, TimeOptimalPlanIsFasterAndRespectsEveryLimit) {
  const Course c = slalom();
  const PlanParams params;
  const Plan fast = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), params);
  const Plan cruise = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), heuristic());
  EXPECT_EQ(fast.info().planner, PlannerKind::TimeOptimal);
  EXPECT_TRUE(fast.info().improved);
  EXPECT_FALSE(fast.info().warm_start);
  EXPECT_GT(fast.info().iterations, 0);
  EXPECT_GT(fast.info().compute_s, 0.0);
  EXPECT_DOUBLE_EQ(fast.info().heuristic_duration_s, cruise.duration());
  EXPECT_LT(fast.duration(), 0.85 * cruise.duration());
  // The true limits hold on a dense grid (the optimiser's own check is 0.5 % loose).
  const Use u = use_of(fast, params);
  EXPECT_LE(u.speed, 1.006);
  EXPECT_LE(u.thrust, 1.006);
  EXPECT_LE(u.floor, 1.006);
  EXPECT_LE(u.tilt, 1.006);
  EXPECT_LE(u.rate, 1.006);
  EXPECT_LE(u.yaw_rate, 1.006);
  // ... and it does use them: the lap is limited by something.
  EXPECT_GT(std::max({u.speed, u.thrust, u.tilt, u.rate, u.yaw_rate}), 0.85);
  // Every crossing is still square on at the gate centre.
  EXPECT_GE(fast.min_margin(), 0.0);
  for (std::size_t g = 0; g < 4; ++g) {
    const FlatReference r = fast.at(fast.gate_times()[g]);
    EXPECT_NEAR(norm(r.position - c.gates[g].center), 0.0, 1e-9) << "gate " << g;
    EXPECT_NEAR(norm(cross(unit_or(r.velocity, kUnitZ), c.gates[g].normal)), 0.0, 1e-9)
        << "gate " << g;
    EXPECT_GT(dot(r.velocity, c.gates[g].normal), 0.2);
  }
}

TEST(Planner, TheHeuristicPlannerRunsNoOptimiser) {
  const Plan plan = plan_race(slalom(), 0, at_rest({0, 0, 2}, 0.0), heuristic());
  EXPECT_EQ(plan.info().planner, PlannerKind::Heuristic);
  EXPECT_FALSE(plan.info().improved);
  EXPECT_EQ(plan.info().iterations, 0);
  EXPECT_DOUBLE_EQ(plan.info().heuristic_duration_s, plan.duration());
}

TEST(Planner, ZeroIterationsFlyTheStartingPoint) {
  // No optimisation: the plan is the best starting point the planner has, and never slower
  // than the heuristic plan.
  PlanParams params;
  params.opt_iterations = 0;
  const Plan plan = plan_race(slalom(), 0, at_rest({0, 0, 2}, 0.0), params);
  EXPECT_EQ(plan.info().iterations, 0);
  EXPECT_LE(plan.duration(), plan.info().heuristic_duration_s * (1.0 + 1e-9));
  EXPECT_GE(plan.min_margin(), 0.0);
}

TEST(Planner, TimeOptimalPlanIsDeterministic) {
  const Course c = slalom();
  const Plan a = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  const Plan b = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  EXPECT_EQ(a.position().durations(), b.position().durations());  // bitwise: no clock decides
  EXPECT_EQ(a.info().iterations, b.info().iterations);
}

TEST(Planner, TimeScaleStretchesTheTimeOptimalPlanToo) {
  const Course c = slalom();
  PlanParams slow;
  slow.time_scale = 2.0;
  const Plan a = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  const Plan b = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), slow);
  EXPECT_NEAR(b.duration(), 2.0 * a.duration(), 1e-9 * a.duration());
  EXPECT_NEAR(b.time_scale(), 2.0, 1e-12);
  EXPECT_LT(use_of(b, slow).rate, 0.5 * use_of(a, PlanParams{}).rate);
}

TEST(Planner, ALimitMarginKeepsAwayFromTheLimits) {
  const Course c = slalom();
  PlanParams tight;
  tight.limit_margin = 0.7;
  const Plan a = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  const Plan b = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), tight);
  EXPECT_GT(b.duration(), a.duration());
  // Fractions of the true limits: the default plan uses them to the full, the other stays
  // measurably below (its penalty is on 70 % of each limit, a soft one: the plan overshoots
  // 70 % - this one by about a quarter - but never the limit itself).
  const Use ua = use_of(a, PlanParams{}), ub = use_of(b, PlanParams{});
  const auto worst = [](const Use& u) {
    return std::max({u.speed, u.thrust, u.tilt, u.rate, u.yaw_rate});
  };
  EXPECT_GT(worst(ua), 0.95);
  EXPECT_LT(worst(ub), worst(ua) - 0.05);
}

TEST(Planner, AContinuationWarmStartsFromThePlanItContinues) {
  const Course c = slalom();
  const Plan first = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), PlanParams{});
  const double t = first.gate_times()[0] + 1.0;
  const FlatReference r = first.at(t);
  StartState s = start_state(r.position, r.velocity, r.yaw);
  s.acceleration = r.acceleration;
  s.jerk = r.jerk;
  s.yaw_rate = r.yaw_rate;

  PlanParams params;
  const Plan warm = plan_race(c, 1, s, params, PlanSeed{&first, t});
  EXPECT_TRUE(warm.info().warm_start);
  EXPECT_TRUE(warm.info().improved);
  EXPECT_LE(warm.info().iterations, params.replan_iterations);
  // It is the same lap seen from here: what is left of the first plan, give or take what
  // the optimiser can still gain with its small budget.
  EXPECT_NEAR(warm.duration(), first.duration() - t, 0.1 * (first.duration() - t));
  // The pinned state carries over: position, velocity, acceleration, jerk, yaw rate.
  const FlatReference n = warm.at(0.0);
  EXPECT_NEAR(norm(n.position - r.position), 0.0, 1e-9);
  EXPECT_NEAR(norm(n.velocity - r.velocity), 0.0, 1e-9);
  EXPECT_NEAR(norm(n.acceleration - r.acceleration), 0.0, 1e-8);
  EXPECT_NEAR(norm(n.jerk - r.jerk), 0.0, 1e-6);
  EXPECT_NEAR(n.yaw_rate, r.yaw_rate, 1e-9);
  // Within the limits from its first instant (the start is not exempt).
  const Use u = use_of(warm, params);
  EXPECT_LE(std::max({u.speed, u.thrust, u.floor, u.tilt, u.rate, u.yaw_rate}), 1.006);
  // The same budget without the seed starts from the heuristic plan and gets less done.
  const Plan cold = plan_race(c, 1, s, params);
  EXPECT_FALSE(cold.info().warm_start);
  EXPECT_LE(warm.duration(), cold.duration() * 1.02);
}

TEST(Planner, ASeedWithoutACommonGateIsIgnored) {
  const Course c = slalom();
  PlanParams params;
  params.max_gates = 2;  // gates 0 and 1, the seed flies gate 3
  const Plan other = plan_race(c, 3, at_rest({16, 9, 2.2}, 2.6), params);
  const Plan plan = plan_race(c, 0, at_rest({0, 0, 2}, 0.0), params, PlanSeed{&other, 1.0});
  EXPECT_FALSE(plan.info().warm_start);
  EXPECT_TRUE(plan.info().improved);
}

TEST(Planner, TurnSpeedBoundFollowsTheTurn) {
  const PlanParams params;
  const double ceiling = params.limit_margin * params.max_speed_mps;
  constexpr double kDeg = kPi / 180.0;
  // Straight through: only the speed limit.
  EXPECT_DOUBLE_EQ(turn_speed_bound(0.0, 20.0, params), ceiling);
  EXPECT_DOUBLE_EQ(turn_speed_bound(2.0 * kDeg, 20.0, params), ceiling);
  // Sharper turns are slower, for any reach; more room is faster.
  double previous = ceiling;
  for (double turn : {20.0, 45.0, 90.0, 120.0, 150.0}) {
    const double v = turn_speed_bound(turn * kDeg, 20.0, params);
    EXPECT_LE(v, previous) << turn;
    previous = v;
  }
  EXPECT_LT(turn_speed_bound(90.0 * kDeg, 10.0, params),
            turn_speed_bound(90.0 * kDeg, 20.0, params));
  EXPECT_LE(turn_speed_bound(90.0 * kDeg, 20.0, params),
            turn_speed_bound(90.0 * kDeg, 80.0, params));
  // The numbers, by hand: a 90 degree turn in 20 m of reach is an arc of radius R = 10 m;
  // lateral acceleration g tan(45 deg) = 9.8 m/s^2 allows sqrt(98) = 9.9 m/s, rolling into
  // it (phi = atan(v^2 / (g R)) at 0.675 rad/s) within 10 m allows about 9.3 m/s.
  EXPECT_NEAR(turn_speed_bound(90.0 * kDeg, 20.0, params), 9.3, 0.4);
  // 120 degrees: R = 5.8 m, sqrt(9.8 * 5.8) = 7.5 m/s, rolling in allows about 7.3.
  EXPECT_NEAR(turn_speed_bound(120.0 * kDeg, 20.0, params), 7.3, 0.4);
  // 45 degrees: R = 24 m (15 m/s), nothing below the speed limit.
  EXPECT_DOUBLE_EQ(turn_speed_bound(45.0 * kDeg, 20.0, params), ceiling);
  // Never below the floor, however tight.
  EXPECT_GE(turn_speed_bound(3.0, 0.5, params), 0.5);
  // A tighter limit margin or tilt lowers it.
  PlanParams tight = params;
  tight.limit_margin = 0.7;
  EXPECT_LT(turn_speed_bound(90.0 * kDeg, 20.0, tight),
            turn_speed_bound(90.0 * kDeg, 20.0, params));
}

// The start the law gives a plan from rest: level, so no acceleration, no jerk.
StartState law_start() {
  StartState s = at_rest({0, 0, 2}, 0.0);
  s.acceleration = Vec3{};
  s.jerk = Vec3{};
  return s;
}

// Speed through the gate with index `g` of a plan.
double crossing_speed(const Plan& plan, std::size_t g) {
  return norm(plan.at(plan.gate_times()[g]).velocity);
}

TEST(Planner, GatesAreCrossedFasterThanCruiseSpeedWhereTheCourseAllows) {
  // Six gates in a straight line, 30 m apart: nothing but the speed limit slows the lap.
  std::vector<GatePose> line;
  for (int i = 0; i < 6; ++i) line.push_back(testing::make_gate({0.0, 10.0 + 30.0 * i, 2.0}, 0.0));
  const Course straight = make_course(line);
  const Plan fast = plan_race(straight, 0, law_start(), PlanParams{});
  const Plan cruise = plan_race(straight, 0, law_start(), heuristic());
  for (std::size_t g = 1; g + 1 < 6; ++g) {
    EXPECT_GT(crossing_speed(fast, g), 9.0) << "gate " << g;
    EXPECT_LT(crossing_speed(cruise, g), 6.1) << "gate " << g;
  }
  EXPECT_LT(fast.duration(), 0.8 * cruise.duration());
}

TEST(Planner, ASharpGateIsCrossedSlowerThanAStraightOne) {
  // The same first four gates; then the course goes on straight, or turns 90 degrees at gate 3.
  std::vector<GatePose> line, bend;
  for (int i = 0; i < 4; ++i) {
    line.push_back(testing::make_gate({0.0, 10.0 + 30.0 * i, 2.0}, 0.0));
    bend.push_back(line.back());
  }
  bend[3] = testing::make_gate({0.0, 100.0, 2.0}, 45.0);
  line.push_back(testing::make_gate({0.0, 130.0, 2.0}, 0.0));
  line.push_back(testing::make_gate({0.0, 160.0, 2.0}, 0.0));
  bend.push_back(testing::make_gate({30.0, 100.0, 2.0}, 90.0));
  bend.push_back(testing::make_gate({60.0, 100.0, 2.0}, 90.0));
  const Plan a = plan_race(make_course(line), 0, law_start(), PlanParams{});
  const Plan b = plan_race(make_course(bend), 0, law_start(), PlanParams{});
  EXPECT_GT(crossing_speed(a, 3), 9.0);
  EXPECT_LT(crossing_speed(b, 3), 0.85 * crossing_speed(a, 3));
}

TEST(Planner, ThrowsWithoutAGateOrARoute) {
  const Course c = slalom();
  EXPECT_THROW(plan_race(c, 4, at_rest({0, 0, 2}, 0.0), PlanParams{}), PlanError);
  PlanParams high;
  high.floor_m = 50.0;  // the whole box is below the floor
  EXPECT_THROW(plan_race(c, 0, at_rest({0, 0, 2}, 0.0), high), PlanError);
}

TEST(Planner, SkipsTheEntryPointWhenAlreadyAtTheGate) {
  const Course c = slalom();
  // 0.2 m upstream of gate 0's plane: an entry point 1 m back would send it backwards.
  const Plan plan =
      plan_race(c, 0, start_state({0.0, 5.8, 2.0}, {0, 1.0, 0}, kPi / 2), PlanParams{});
  EXPECT_NEAR(plan.at(0.0).position.y, 5.8, 1e-12);
  for (double t = 0.0; t < plan.gate_times()[0]; t += 0.02) {
    EXPECT_GE(plan.at(t).position.y, 5.8 - 0.6)
        << t;  // the spline may swing back a little, not to an entry point 1 m upstream
  }
}

}  // namespace
}  // namespace astrum::minsnap
