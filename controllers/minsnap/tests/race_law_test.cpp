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
  EXPECT_LE(law.plans_made(), 6u);  // extensions only: never a re-plan for tracking error
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
