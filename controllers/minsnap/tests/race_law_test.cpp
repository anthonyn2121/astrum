#include "race_law.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <variant>

#include "support.hpp"

namespace astrum::minsnap {
namespace {

using testing::make_course;
using testing::make_gate;
using testing::wire_course;

Course slalom() {
  return make_course({make_gate({0.0, 6.0, 2.0}, 0.0), make_gate({4.0, 12.0, 2.5}, 40.0),
                      make_gate({10.0, 14.0, 1.6}, 90.0)});
}

VehicleState level_state(const Vec3& p, double t, double heading_rad = 0.0) {
  VehicleState s;
  s.e = p.x;
  s.n = p.y;
  s.u = p.z;
  s.yaw = heading_rad;
  s.sim_time_s = t;
  s.control_period_s = 0.01;
  s.vehicle_kind = VehicleKind::Multirotor;
  return s;
}

MultirotorRaw raw_of(const LawResult& r) {
  return std::get<MultirotorRaw>(std::get<Action>(r).control);
}

StepInput input(const VehicleState& s, const std::optional<Course>& course = std::nullopt,
                std::uint32_t pending = 0, bool finished = false, std::uint32_t gates = 3) {
  StepInput in;
  in.state = s;
  in.control_period_s = s.control_period_s;
  if (course) in.race_course = wire_course(*course);
  RaceState race;
  race.pending_gate = pending;
  race.gate_count = gates;
  race.finished = finished;
  in.race_state = race;
  return in;
}

TEST(RaceLaw, IdentifiesItself) {
  const RaceLaw law{RaceLawParams{}};
  EXPECT_EQ(law.vehicle_kinds(), std::vector<VehicleKind>{VehicleKind::Multirotor});
  EXPECT_TRUE(law.wants_race());
  EXPECT_FALSE(law.needs_fleet());
}

TEST(RaceLaw, CourseFromWireSortsGatesAndKeepsTheHardware) {
  const Course c = slalom();
  RaceCourse wire = wire_course(c);
  std::swap(wire.gates[0], wire.gates[2]);
  const Course back = course_from_wire(wire);
  ASSERT_EQ(back.gates.size(), 3u);
  EXPECT_DOUBLE_EQ(back.opening_m, c.opening_m);
  EXPECT_DOUBLE_EQ(back.bar_m, c.bar_m);
  EXPECT_DOUBLE_EQ(back.top_bar_m, c.top_bar_m);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(back.gates[i].center, c.gates[i].center);
    EXPECT_EQ(back.gates[i].normal, c.gates[i].normal);
    EXPECT_EQ(back.gates[i].right, c.gates[i].right);
  }
}

TEST(RaceLaw, ValidatesItsParameters) {
  EXPECT_FALSE(RaceLawParams{}.validate());
  RaceLawParams p;
  p.plan.cruise_mps = 0.0;
  EXPECT_NE(p.validate()->find("--cruise-mps"), std::string::npos);
  p = {};
  p.gains.kp = -1.0;
  EXPECT_NE(p.validate()->find("--kp"), std::string::npos);
  p = {};
  p.demands.hover_collective = 1.0;
  EXPECT_NE(p.validate()->find("--hover-collective"), std::string::npos);
  p = {};
  p.plan.cross_radius_m = p.plan.hull_radius_m + 0.1;
  EXPECT_NE(p.validate()->find("--cross-radius-m"), std::string::npos);
  p = {};
  p.plan.cruise_mps = std::nan("");
  EXPECT_TRUE(p.validate());
  p = {};
  p.gains.max_tilt_rad = 2.0;
  EXPECT_TRUE(p.validate());
  // The F450 limits and the optimiser's settings.
  p = {};
  p.plan.max_thrust_mps2 = 9.0;
  EXPECT_NE(p.validate()->find("--max-thrust-mps2"), std::string::npos);
  p = {};
  p.plan.min_thrust_mps2 = 10.0;
  EXPECT_NE(p.validate()->find("--min-thrust-mps2"), std::string::npos);
  p = {};
  p.plan.min_thrust_mps2 = 0.0;
  EXPECT_FALSE(p.validate());  // no floor is allowed
  p = {};
  p.plan.limit_margin = 1.2;
  EXPECT_NE(p.validate()->find("--limit-margin"), std::string::npos);
  p.plan.limit_margin = 0.0;
  EXPECT_NE(p.validate()->find("--limit-margin"), std::string::npos);
  p = {};
  p.plan.snap_weight = -1.0;
  EXPECT_NE(p.validate()->find("--snap-weight"), std::string::npos);
  p = {};
  p.plan.opt_iterations = -1;
  EXPECT_TRUE(p.validate());
  p = {};
  p.plan.replan_iterations = -1;
  EXPECT_TRUE(p.validate());
  // The plan may not tilt further than the tracker allows.
  p = {};
  p.plan.max_tilt_rad = p.gains.max_tilt_rad + 0.1;
  EXPECT_NE(p.validate()->find("--max-tilt-deg"), std::string::npos);
  // The collective clamp: above the hover collective, at most full, and high enough for the
  // thrust the plan may ask for.
  p = {};
  p.demands.max_collective = p.demands.hover_collective;
  EXPECT_NE(p.validate()->find("--max-collective"), std::string::npos);
  p.demands.max_collective = 1.1;
  EXPECT_NE(p.validate()->find("--max-collective"), std::string::npos);
  p = {};
  p.demands.max_collective = 0.5;  // 1.7 g needs about 0.55
  EXPECT_NE(p.validate()->find("--max-thrust-mps2"), std::string::npos);
}

TEST(RaceLaw, DefaultsAreTheMeasuredAirframe) {
  const RaceLawParams p;
  EXPECT_FALSE(p.validate());
  EXPECT_EQ(p.plan.planner, PlannerKind::TimeOptimal);
  EXPECT_DOUBLE_EQ(p.demands.hover_collective, 0.408);
  EXPECT_DOUBLE_EQ(p.plan.max_rate_rad_s, 0.75);
  EXPECT_DOUBLE_EQ(p.plan.max_yaw_rate, 0.75);
  EXPECT_DOUBLE_EQ(p.plan.max_speed_mps, 12.0);
  EXPECT_NEAR(p.plan.max_thrust_mps2, 1.7 * 9.80665, 0.05);
  EXPECT_NEAR(p.plan.min_thrust_mps2, 0.3 * 9.80665, 0.05);
  EXPECT_NEAR(p.plan.max_tilt_rad, 50.0 * 3.14159265358979 / 180.0, 1e-12);
  // The tracker may tilt a little further than the plan, never less.
  EXPECT_GE(p.gains.max_tilt_rad, p.plan.max_tilt_rad);
}

TEST(RaceLaw, HoversUntilTheCourseArrives) {
  RaceLaw law{RaceLawParams{}};
  const MultirotorRaw raw = raw_of(law.step(input(level_state({1, 2, 3}, 0.0))));
  EXPECT_NEAR(raw.collective, RaceLawParams{}.demands.hover_collective, 1e-12);
  EXPECT_NEAR(raw.roll_rate, 0.0, 1e-12);
  EXPECT_NEAR(raw.pitch_rate, 0.0, 1e-12);
  EXPECT_NEAR(raw.yaw_rate, 0.0, 1e-12);
  EXPECT_EQ(law.plan(), nullptr);
}

TEST(RaceLaw, PlansOnceOnTheCourseAndFliesIt) {
  RaceLaw law{RaceLawParams{}};
  const Course c = slalom();
  law.step(input(level_state({0, 0, 2}, 0.0), c));
  ASSERT_NE(law.plan(), nullptr);
  EXPECT_EQ(law.plans_made(), 1u);
  EXPECT_EQ(law.plan()->gate_count(), 3u);
  // Following the reference closely never triggers another plan.
  for (double t = 0.01; t < 2.0; t += 0.01) {
    const FlatReference ref = law.plan()->at(t);
    law.step(input(level_state(ref.position, t), std::nullopt, 0));
  }
  EXPECT_EQ(law.plans_made(), 1u);
}

TEST(RaceLaw, LogsOneLineForEveryPlan) {
  std::vector<std::string> lines;
  RaceLaw law{RaceLawParams{}, [&](LogLevel level, std::string_view m) {
                EXPECT_EQ(level, LogLevel::Info);
                lines.emplace_back(m);
              }};
  law.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("plan 1: gates 0..2, time-optimal"), std::string::npos) << lines[0];
  EXPECT_NE(lines[0].find("iterations"), std::string::npos) << lines[0];
  EXPECT_NE(lines[0].find("planned in"), std::string::npos) << lines[0];
  // A re-plan from the old one says so.
  law.step(input(level_state({3.0, 0.5, 2}, 1.0), std::nullopt, 0));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_NE(lines[1].find("plan 2"), std::string::npos) << lines[1];
}

TEST(RaceLaw, AFirstPlanStartsAtTheAttitudeTheAircraftHas) {
  // Level: no acceleration to start with. Rolled 0.2 rad: the thrust it already points
  // sideways gives the plan's start its acceleration, so the attitude does not jump.
  RaceLaw level{RaceLawParams{}};
  level.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  EXPECT_NEAR(norm(level.plan()->at(0.0).acceleration), 0.0, 1e-9);
  EXPECT_NEAR(norm(level.plan()->at(0.0).jerk), 0.0, 1e-9);

  RaceLaw rolled{RaceLawParams{}};
  VehicleState s = level_state({0, 0, 2}, 0.0);
  s.roll = 0.2;
  rolled.step(input(s, slalom()));
  const Vec3 a = rolled.plan()->at(0.0).acceleration;
  EXPECT_NEAR(std::hypot(a.x, a.y), 9.80665 * std::tan(0.2), 1e-6);
  EXPECT_NEAR(a.z, 0.0, 1e-9);
}

TEST(RaceLaw, ANewCourseReplansFromTheCurrentState) {
  RaceLaw law{RaceLawParams{}};
  const Course c = slalom();
  law.step(input(level_state({0, 0, 2}, 0.0), c));
  // The course is republished about once a second: the same one is not a reason to plan again.
  law.step(input(level_state({0, 0, 2}, 0.01), c));
  EXPECT_EQ(law.plans_made(), 1u);
  // A different one is.
  Course other = c;
  other.gates[1].center.z += 0.5;
  law.step(input(level_state({0, 0, 2}, 0.02), other));
  EXPECT_EQ(law.plans_made(), 2u);
}

TEST(RaceLaw, ReplansWhenFarFromTheReference) {
  RaceLaw law{RaceLawParams{}};
  law.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  // 3 m off the line a second later: beyond --replan-error-m.
  law.step(input(level_state({3.0, 0.5, 2}, 1.0), std::nullopt, 0));
  EXPECT_EQ(law.plans_made(), 2u);
  // And the new plan starts from where the aircraft is.
  EXPECT_NEAR(law.plan()->at(0.0).position.x, 3.0, 1e-9);
}

TEST(RaceLaw, ReplansWhenAGateIsOverdue) {
  RaceLaw law{RaceLawParams{}};
  law.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  const double due = law.plan()->gate_times()[0];
  // On the reference but the race still waits for gate 0 well after the planned time.
  const double t = due + 2.0;
  law.step(input(level_state(law.plan()->at(t).position, t), std::nullopt, 0));
  EXPECT_EQ(law.plans_made(), 2u);
  // The same moment with the gate scored: no re-plan.
  RaceLaw ok{RaceLawParams{}};
  ok.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  ok.step(input(level_state(ok.plan()->at(t).position, t), std::nullopt, 1));
  EXPECT_EQ(ok.plans_made(), 1u);
}

TEST(RaceLaw, PlansFromThePendingGate) {
  RaceLaw law{RaceLawParams{}};
  law.step(input(level_state({3.0, 10.0, 2.4}, 0.0, 0.7), slalom(), 1));
  ASSERT_NE(law.plan(), nullptr);
  EXPECT_EQ(law.plan()->first_gate(), 1u);
  EXPECT_EQ(law.plan()->gate_count(), 2u);
}

TEST(RaceLaw, ExtendsItsPlanAlongALongCourse) {
  // Twelve gates in a gentle zig-zag, planned four at a time.
  std::vector<GatePose> gates;
  for (int i = 0; i < 12; ++i)
    gates.push_back(make_gate({2.0 * (i % 2), 10.0 * (i + 1), 2.0}, 0.0));
  const Course c = make_course(gates);
  RaceLawParams params;
  params.plan.max_gates = 4;
  RaceLaw law{params};
  law.step(input(level_state({0, 0, 2}, 0.0), c, 0, false, 12));
  ASSERT_NE(law.plan(), nullptr);
  EXPECT_EQ(law.plan()->gate_count(), 4u);

  // Follow the reference exactly; the race scores each gate at its planned time.
  double plan_start = 0.0;
  std::size_t seen = law.plans_made();
  for (double t = 0.01; t < 400.0; t += 0.01) {
    const Plan& plan = *law.plan();
    const FlatReference ref = plan.at(t - plan_start);
    std::uint32_t pending = static_cast<std::uint32_t>(plan.first_gate());
    for (double due : plan.gate_times()) pending += (due <= t - plan_start) ? 1 : 0;
    if (pending >= 12) break;
    law.step(input(level_state(ref.position, t), std::nullopt, pending, false, 12));
    if (law.plans_made() != seen) {
      seen = law.plans_made();
      plan_start = t;
    }
  }
  const Plan& last = *law.plan();
  EXPECT_EQ(last.first_gate() + last.gate_count(), 12u);  // the plan reached the final gate
  EXPECT_GE(law.plans_made(), 3u);
  EXPECT_LE(law.plans_made(), 6u);      // extensions only: never a re-plan for tracking error
  EXPECT_TRUE(last.info().warm_start);  // an extension continues the plan it replaces
}

TEST(RaceLaw, HoldsWhenTheRaceIsFinished) {
  RaceLaw law{RaceLawParams{}};
  const Course c = slalom();
  law.step(input(level_state({0, 0, 2}, 0.0), c));
  const std::size_t plans = law.plans_made();
  const MultirotorRaw raw =
      raw_of(law.step(input(level_state({5, 5, 2}, 4.0), std::nullopt, 3, true)));
  EXPECT_EQ(law.plans_made(), plans);
  EXPECT_NEAR(raw.collective, RaceLawParams{}.demands.hover_collective, 1e-12);
  EXPECT_NEAR(raw.pitch_rate, 0.0, 1e-12);
}

TEST(RaceLaw, StopsWhenTheCourseCannotBePlanned) {
  RaceLawParams params;
  params.plan.floor_m = 50.0;  // no route above the floor
  RaceLaw law{params};
  const LawResult r = law.step(input(level_state({0, 0, 2}, 0.0), slalom()));
  ASSERT_TRUE(std::holds_alternative<Stop>(r));
  EXPECT_NE(std::get<Stop>(r).reason.find("cannot plan"), std::string::npos);
}

TEST(RaceLaw, KeepsFlyingTheOldPlanWhenAReplanFails) {
  RaceLawParams params;
  RaceLaw law{params};
  const Course c = slalom();
  law.step(input(level_state({0, 0, 2}, 0.0), c));
  // From the middle of the sky below the floor no route exists any more; the old plan flies on.
  const LawResult r = law.step(input(level_state({3.0, 0.5, -20.0}, 1.0), std::nullopt, 0));
  EXPECT_TRUE(std::holds_alternative<Action>(r));
  EXPECT_EQ(law.plans_made(), 1u);
}

}  // namespace
}  // namespace astrum::minsnap
