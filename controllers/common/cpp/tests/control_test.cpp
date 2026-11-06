#include "astrum/control.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

#include "astrum/math.hpp"

namespace astrum {
namespace {

VehicleState level(double t, double yaw, double alt, double tas) {
  VehicleState s;
  s.sim_time_s = t;
  s.yaw = yaw;
  s.alt_hae_m = alt;
  s.tas_mps = tas;
  return s;
}

TEST(SaturatedPi, IntegratesSmallErrorsOverTime) {
  SaturatedPi pi(0.0, 0.5, -1.0, 1.0);
  EXPECT_DOUBLE_EQ(pi.update(0.2, 1.0), 0.1);
  EXPECT_DOUBLE_EQ(pi.update(0.2, 1.0), 0.2);
}

TEST(SaturatedPi, DoesNotIntegrateWithoutElapsedTime) {
  SaturatedPi pi(1.0, 1.0, -10.0, 10.0);
  pi.update(1.0, 0.0);
  pi.update(1.0, -1.0);
  EXPECT_EQ(pi.integral(), 0.0);
}

TEST(SaturatedPi, StopsIntegratingWhileSaturatedAndRecoversImmediately) {
  SaturatedPi pi(1.0, 1.0, -1.0, 1.0);
  // A long, large positive error saturates the output high: no wind-up.
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(pi.update(5.0, 0.1), 1.0);
  EXPECT_LE(pi.integral(), 1.0);
  // As soon as the error reverses, the output leaves the upper limit.
  EXPECT_LT(pi.update(-0.5, 0.1), 0.6);
}

TEST(SaturatedPi, IntegratesBackOutOfSaturation) {
  SaturatedPi pi(0.0, 1.0, -1.0, 1.0);
  for (int i = 0; i < 10; ++i) pi.update(0.5, 1.0);
  EXPECT_DOUBLE_EQ(pi.integral(), 1.0);  // stopped exactly at the limit
  // Saturated high, but this error pulls the output down: it must integrate.
  EXPECT_DOUBLE_EQ(pi.update(-0.1, 1.0), 0.9);
}

TEST(OuterLoops, RightHeadingErrorBanksRight) {
  OuterLoops loops(AttitudeLimits{});
  const Reference ref{deg_to_rad(10.0), 1000.0, 50.0};
  EXPECT_GT(loops.update(level(0.0, 0.0, 1000.0, 50.0), ref).bank_rad, 0.0);
  OuterLoops left(AttitudeLimits{});
  EXPECT_LT(left.update(level(0.0, deg_to_rad(20.0), 1000.0, 50.0), ref).bank_rad, 0.0);
}

TEST(OuterLoops, HeadingErrorWrapsTheShortWayRound) {
  OuterLoops loops(AttitudeLimits{});
  // From 350 deg to 10 deg is a 20 deg right turn, not a 340 deg left one.
  const Reference ref{deg_to_rad(10.0), 1000.0, 50.0};
  EXPECT_GT(loops.update(level(0.0, deg_to_rad(350.0), 1000.0, 50.0), ref).bank_rad, 0.0);
}

TEST(OuterLoops, HugeErrorsRespectTheAttitudeLimits) {
  const AttitudeLimits limits{25.0, 10.0};
  OuterLoops loops(limits);
  const Reference ref{deg_to_rad(90.0), 1500.0, 50.0};
  // 90 deg heading error and a 500 m climb, held for a minute of sim time.
  AttitudeTargets out;
  for (int i = 0; i < 7200; ++i) out = loops.update(level(i / 120.0, 0.0, 1000.0, 50.0), ref);
  EXPECT_DOUBLE_EQ(out.bank_rad, deg_to_rad(25.0));
  EXPECT_DOUBLE_EQ(out.pitch_rad, deg_to_rad(10.0));

  OuterLoops down(limits);
  const Reference low{deg_to_rad(-90.0), 500.0, 50.0};
  out = down.update(level(0.0, 0.0, 1000.0, 50.0), low);
  EXPECT_DOUBLE_EQ(out.bank_rad, deg_to_rad(-25.0));
  EXPECT_DOUBLE_EQ(out.pitch_rad, deg_to_rad(-10.0));
}

TEST(OuterLoops, AltitudeIntegratorDoesNotWindUpDuringALongClimb) {
  OuterLoops loops(AttitudeLimits{25.0, 10.0});
  const Reference ref{0.0, 1500.0, 50.0};
  // Pitch saturated for 10 minutes far below the target...
  for (int i = 0; i < 72000; ++i) loops.update(level(i / 120.0, 0.0, 500.0, 50.0), ref);
  // ...then the aircraft overshoots by 50 m: the pitch command reverses at once.
  const double t = 72000 / 120.0;
  EXPECT_LT(loops.update(level(t, 0.0, 1550.0, 50.0), ref).pitch_rad, 0.0);
}

TEST(OuterLoops, ThrottleRisesWhenSlowAndStaysInRange) {
  OuterLoops loops(AttitudeLimits{});
  const Reference ref{0.0, 1000.0, 50.0};
  const double slow = loops.update(level(0.0, 0.0, 1000.0, 40.0), ref).throttle;
  EXPECT_GT(slow, 0.6);
  OuterLoops very_slow(AttitudeLimits{});
  AttitudeTargets out;
  for (int i = 0; i < 12000; ++i) out = very_slow.update(level(i / 120.0, 0.0, 1000.0, 0.0), ref);
  EXPECT_EQ(out.throttle, 1.0);
  OuterLoops fast(AttitudeLimits{});
  EXPECT_EQ(fast.update(level(0.0, 0.0, 1000.0, 200.0), ref).throttle, 0.0);
}

TEST(OuterLoops, AirspeedIntegratorRemovesSteadyError) {
  OuterLoops loops(AttitudeLimits{});
  const Reference ref{0.0, 1000.0, 50.0};
  const double first = loops.update(level(0.0, 0.0, 1000.0, 49.0), ref).throttle;
  double last = first;
  for (int i = 1; i <= 1200; ++i)
    last = loops.update(level(i / 120.0, 0.0, 1000.0, 49.0), ref).throttle;
  EXPECT_NEAR(last - first, 0.005 * 1.0 * 10.0, 1e-9);  // ki * error * 10 s
}

TEST(OuterLoops, HeadingIntegratorRemovesSteadyError) {
  // A standing 2 deg heading error (e.g. the bank shortfall propeller torque leaves)
  // must keep growing the bank target, not settle at the proportional value.
  OuterLoops loops(AttitudeLimits{});
  const Reference ref{deg_to_rad(2.0), 1000.0, 50.0};
  const double first = loops.update(level(0.0, 0.0, 1000.0, 50.0), ref).bank_rad;
  double last = first;
  for (int i = 1; i <= 1200; ++i)
    last = loops.update(level(i / 120.0, 0.0, 1000.0, 50.0), ref).bank_rad;
  EXPECT_NEAR(last - first, 0.05 * deg_to_rad(2.0) * 10.0, 1e-9);  // ki * error * 10 s
}

TEST(OuterLoops, RejectsNonsenseLimits) {
  EXPECT_THROW(OuterLoops(AttitudeLimits{0.0, 10.0}), std::invalid_argument);
  EXPECT_THROW(OuterLoops(AttitudeLimits{25.0, 90.0}), std::invalid_argument);
  EXPECT_THROW(OuterLoops(AttitudeLimits{-5.0, 10.0}), std::invalid_argument);
}

}  // namespace
}  // namespace astrum
