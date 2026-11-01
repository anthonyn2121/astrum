#include "hold_law.hpp"

#include <gtest/gtest.h>

#include "astrum/math.hpp"

namespace astrum {
namespace {

const AttitudeLimits kLimits{25.0, 10.0};

VehicleState trimmed(double yaw_deg, double alt, double tas) {
  VehicleState s;
  s.yaw = deg_to_rad(yaw_deg);
  s.alt_m = alt;
  s.airspeed = tas;
  return s;
}

TEST(HoldLaw, RightHeadingErrorRollsRight) {
  HoldLaw law(kLimits, Reference{deg_to_rad(30.0), 1000.0, 50.0});
  EXPECT_GT(law.control(trimmed(0.0, 1000.0, 50.0)).aileron, 0.0);
}

TEST(HoldLaw, BelowTargetAltitudePitchesNoseUp) {
  HoldLaw law(kLimits, Reference{0.0, 1100.0, 50.0});
  EXPECT_LT(law.control(trimmed(0.0, 1000.0, 50.0)).elevator, 0.0);  // elevator+ = nose down
}

TEST(HoldLaw, SlowAddsThrottle) {
  HoldLaw law(kLimits, Reference{0.0, 1000.0, 50.0});
  EXPECT_GT(law.control(trimmed(0.0, 1000.0, 45.0)).throttle, 0.6);
}

TEST(HoldLaw, NinetyDegreeTurnStopsRollingAtTheBankLimit) {
  HoldLaw law(kLimits, Reference{deg_to_rad(90.0), 1000.0, 50.0});
  VehicleState s = trimmed(0.0, 1000.0, 50.0);
  s.roll = deg_to_rad(25.0);
  EXPECT_NEAR(law.control(s).aileron, 0.0, 1e-12);
  s.roll = deg_to_rad(35.0);
  EXPECT_LT(law.control(s).aileron, 0.0);
}

TEST(HoldLaw, LargeClimbStopsPitchingAtThePitchLimit) {
  HoldLaw law(kLimits, Reference{0.0, 1500.0, 50.0});
  VehicleState s = trimmed(0.0, 1000.0, 50.0);
  s.pitch = deg_to_rad(10.0);
  EXPECT_NEAR(law.control(s).elevator, 0.0, 1e-12);
  s.pitch = deg_to_rad(15.0);
  EXPECT_GT(law.control(s).elevator, 0.0);
}

TEST(HoldLaw, OutputsStayInActuatorRanges) {
  HoldLaw law(kLimits, Reference{deg_to_rad(180.0), 5000.0, 300.0});
  VehicleState s = trimmed(1.0, 0.0, 0.0);
  s.roll = -1.5;
  s.p = -5.0;
  s.q = -5.0;
  const RawActuators a = law.control(s);
  EXPECT_LE(std::abs(a.aileron), 1.0);
  EXPECT_LE(std::abs(a.elevator), 1.0);
  EXPECT_GE(a.throttle, 0.0);
  EXPECT_LE(a.throttle, 1.0);
}

}  // namespace
}  // namespace astrum
