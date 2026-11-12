#include "planner.hpp"

#include <gtest/gtest.h>

#include <cmath>

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
  PlanParams off;
  off.refine_iterations = 0;
  const Plan raw = plan_race(c, 0, at_rest({-6.0, 8.0, 2.0}, kPi / 2), off);
  const Plan plan = plan_race(c, 0, at_rest({-6.0, 8.0, 2.0}, kPi / 2), PlanParams{});
  EXPECT_LT(raw.min_margin(), 0.0);  // without the repair pass the spline hits the frame
  EXPECT_GE(plan.min_margin(), 0.0);
  EXPECT_GT(plan.waypoints().size(), raw.waypoints().size());  // a corridor keyframe was added
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
