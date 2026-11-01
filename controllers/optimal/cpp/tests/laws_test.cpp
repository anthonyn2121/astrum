#include "laws.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <variant>

namespace astrum {
namespace {

using Factory = std::unique_ptr<Law> (*)(const LawConfig&);

struct LawCase {
  std::string name;
  Factory make;
};

void PrintTo(const LawCase& c, std::ostream* os) { *os << c.name; }

class LawTest : public ::testing::TestWithParam<LawCase> {
 protected:
  std::unique_ptr<Law> make(double rate_hz = 120.0) const {
    LawConfig config;
    config.dt = 1.0 / rate_hz;
    config.limits = AttitudeLimits{25.0, 10.0};
    return GetParam().make(config);
  }
};

VehicleState level(double yaw_deg, double alt, double tas) {
  VehicleState s;
  s.yaw = deg_to_rad(yaw_deg);
  s.alt_m = alt;
  s.airspeed = tas;
  return s;
}

RawActuators raw(const Command& c) { return std::get<RawActuators>(c.control); }

TEST_P(LawTest, FliesFixedWingOnly) { EXPECT_EQ(make()->vehicle_kind(), VehicleKind::FixedWing); }

TEST_P(LawTest, RightHeadingErrorRollsRight) {
  EXPECT_GT(
      raw(make()->control(level(0.0, 1000.0, 50.0), {deg_to_rad(20.0), 1000.0, 50.0})).aileron,
      0.0);
  EXPECT_LT(
      raw(make()->control(level(0.0, 1000.0, 50.0), {deg_to_rad(-20.0), 1000.0, 50.0})).aileron,
      0.0);
}

TEST_P(LawTest, BelowTargetAltitudePitchesNoseUp) {
  // elevator+ pitches nose down, so a climb needs negative elevator.
  EXPECT_LT(raw(make()->control(level(0.0, 1000.0, 50.0), {0.0, 1050.0, 50.0})).elevator, 0.0);
  EXPECT_GT(raw(make()->control(level(0.0, 1000.0, 50.0), {0.0, 950.0, 50.0})).elevator, 0.0);
}

TEST_P(LawTest, SlowAddsThrottle) {
  EXPECT_GT(raw(make()->control(level(0.0, 1000.0, 45.0), {0.0, 1000.0, 50.0})).throttle, 0.6);
}

TEST_P(LawTest, NinetyDegreeTurnHoldsTheBankLimit) {
  const Reference ref{deg_to_rad(90.0), 1000.0, 50.0};
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = deg_to_rad(25.0);  // at the limit: no further roll command
  EXPECT_NEAR(raw(make()->control(s, ref)).aileron, 0.0, 1e-9);
  s.roll = deg_to_rad(40.0);  // past it: roll back
  EXPECT_LT(raw(make()->control(s, ref)).aileron, 0.0);
}

TEST_P(LawTest, FiveHundredMetreClimbHoldsThePitchLimit) {
  const Reference ref{0.0, 1500.0, 50.0};
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.pitch = deg_to_rad(10.0);
  EXPECT_NEAR(raw(make()->control(s, ref)).elevator, 0.0, 1e-9);
  s.pitch = deg_to_rad(20.0);
  EXPECT_GT(raw(make()->control(s, ref)).elevator, 0.0);
}

TEST_P(LawTest, OutputsStayInActuatorRanges) {
  VehicleState s = level(0.0, 0.0, 0.0);
  s.roll = -1.4;
  s.pitch = 0.8;
  s.p = -3.0;
  s.q = 3.0;
  const RawActuators a = raw(make()->control(s, {deg_to_rad(179.0), 9000.0, 300.0}));
  EXPECT_LE(std::abs(a.aileron), 1.0);
  EXPECT_LE(std::abs(a.elevator), 1.0);
  EXPECT_GE(a.throttle, 0.0);
  EXPECT_LE(a.throttle, 1.0);
  EXPECT_EQ(a.aileron, 1.0);  // far left of the bank target, rolling left: full right
}

TEST_P(LawTest, OutputIsConsistentAcrossControlRates) {
  // The laws are discretized at the control period, so the same state gives nearly the
  // same command at 60 Hz and 120 Hz.
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = 0.05;
  s.p = 0.02;
  const Reference ref{deg_to_rad(5.0), 1000.0, 50.0};
  const double at120 = raw(make(120.0)->control(s, ref)).aileron;
  const double at60 = raw(make(60.0)->control(s, ref)).aileron;
  EXPECT_NEAR(at60, at120, 0.05 * std::abs(at120));
}

INSTANTIATE_TEST_SUITE_P(Laws, LawTest,
                         ::testing::Values(LawCase{"lqr", make_lqr_law},
                                           LawCase{"mpc", make_mpc_law}),
                         [](const auto& param_info) { return param_info.param.name; });

TEST(Mpc, MatchesLqrWhileNoLimitIsActive) {
  LawConfig config;
  const auto lqr = make_lqr_law(config);
  const auto mpc = make_mpc_law(config);
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = 0.03;
  s.p = -0.01;
  s.pitch = 0.02;
  s.q = 0.01;
  const Reference ref{deg_to_rad(3.0), 1002.0, 50.0};
  const RawActuators a = raw(lqr->control(s, ref));
  const RawActuators b = raw(mpc->control(s, ref));
  EXPECT_NEAR(a.aileron, b.aileron, 1e-6);
  EXPECT_NEAR(a.elevator, b.elevator, 1e-6);
}

TEST(Mpc, RejectsAHorizonShorterThanOneControlPeriod) {
  LawConfig config;
  config.horizon_s = 0.4 * config.dt;
  EXPECT_THROW(make_mpc_law(config), std::invalid_argument);
  config.horizon_s = -1.0;
  EXPECT_THROW(make_mpc_law(config), std::invalid_argument);
}

}  // namespace
}  // namespace astrum
