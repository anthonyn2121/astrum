#include "laws.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

#include "astrum/math.hpp"

namespace astrum {
namespace {

using Factory = std::function<std::unique_ptr<Law>(const LawContext&)>;

struct LawCase {
  std::string name;
  Factory make;
};

void PrintTo(const LawCase& c, std::ostream* os) { *os << c.name; }

class LawTest : public ::testing::TestWithParam<LawCase> {
 protected:
  // A law for `ref`, configured for a `rate_hz` control rate.
  std::unique_ptr<Law> make(const Reference& ref, double rate_hz = 120.0) const {
    std::unique_ptr<Law> law = GetParam().make(LawContext{ref, AttitudeLimits{25.0, 10.0}});
    law->configure(1.0 / rate_hz);
    return law;
  }
};

VehicleState level(double yaw_deg, double alt, double tas) {
  VehicleState s;
  s.yaw = deg_to_rad(yaw_deg);
  s.alt_hae_m = alt;
  s.tas_mps = tas;
  return s;
}

RawActuators raw(Law& law, const VehicleState& s) {
  StepInput input;
  input.state = s;
  const LawResult result = law.step(input);
  return std::get<RawActuators>(std::get<Action>(result).control);
}

RawActuators raw(const std::unique_ptr<Law>& law, const VehicleState& s) { return raw(*law, s); }

TEST_P(LawTest, FliesFixedWingOnly) {
  EXPECT_EQ(make({})->vehicle_kinds(), std::vector<VehicleKind>{VehicleKind::FixedWing});
}

TEST_P(LawTest, StepBeforeConfigureIsALogicError) {
  const std::unique_ptr<Law> law = GetParam().make(LawContext{});
  EXPECT_THROW(raw(law, level(0.0, 1000.0, 50.0)), std::logic_error);
}

TEST_P(LawTest, RightHeadingErrorRollsRight) {
  EXPECT_GT(raw(make({deg_to_rad(20.0), 1000.0, 50.0}), level(0.0, 1000.0, 50.0)).aileron, 0.0);
  EXPECT_LT(raw(make({deg_to_rad(-20.0), 1000.0, 50.0}), level(0.0, 1000.0, 50.0)).aileron, 0.0);
}

TEST_P(LawTest, BelowTargetAltitudePitchesNoseUp) {
  // elevator+ pitches nose down, so a climb needs negative elevator.
  EXPECT_LT(raw(make({0.0, 1050.0, 50.0}), level(0.0, 1000.0, 50.0)).elevator, 0.0);
  EXPECT_GT(raw(make({0.0, 950.0, 50.0}), level(0.0, 1000.0, 50.0)).elevator, 0.0);
}

TEST_P(LawTest, SlowAddsThrottle) {
  EXPECT_GT(raw(make({0.0, 1000.0, 50.0}), level(0.0, 1000.0, 45.0)).throttle, 0.6);
}

TEST_P(LawTest, NinetyDegreeTurnHoldsTheBankLimit) {
  const Reference ref{deg_to_rad(90.0), 1000.0, 50.0};
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = deg_to_rad(25.0);  // at the limit: no further roll command
  EXPECT_NEAR(raw(make(ref), s).aileron, 0.0, 1e-9);
  s.roll = deg_to_rad(40.0);  // past it: roll back
  EXPECT_LT(raw(make(ref), s).aileron, 0.0);
}

TEST_P(LawTest, FiveHundredMetreClimbHoldsThePitchLimit) {
  const Reference ref{0.0, 1500.0, 50.0};
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.pitch = deg_to_rad(10.0);
  EXPECT_NEAR(raw(make(ref), s).elevator, 0.0, 1e-9);
  s.pitch = deg_to_rad(20.0);
  EXPECT_GT(raw(make(ref), s).elevator, 0.0);
}

TEST_P(LawTest, OutputsStayInActuatorRanges) {
  VehicleState s = level(0.0, 0.0, 0.0);
  s.roll = -1.4;
  s.pitch = 0.8;
  s.p = -3.0;
  s.q = 3.0;
  const RawActuators a = raw(make({deg_to_rad(179.0), 9000.0, 300.0}), s);
  EXPECT_LE(std::abs(a.aileron), 1.0);
  EXPECT_LE(std::abs(a.elevator), 1.0);
  EXPECT_GE(a.throttle, 0.0);
  EXPECT_LE(a.throttle, 1.0);
  EXPECT_EQ(a.aileron, 1.0);  // far left of the bank target, rolling left: full right
}

TEST_P(LawTest, OutputIsConsistentAcrossControlRates) {
  // The laws are discretized at the configured control period, so the same state gives
  // nearly the same command at 60 Hz and 120 Hz.
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = 0.05;
  s.p = 0.02;
  const Reference ref{deg_to_rad(5.0), 1000.0, 50.0};
  const double at120 = raw(make(ref, 120.0), s).aileron;
  const double at60 = raw(make(ref, 60.0), s).aileron;
  EXPECT_NEAR(at60, at120, 0.05 * std::abs(at120));
}

INSTANTIATE_TEST_SUITE_P(
    Laws, LawTest,
    ::testing::Values(LawCase{"lqr", make_lqr_law},
                      LawCase{"mpc", [](const LawContext& c) { return make_mpc_law(c, 0.5); }}),
    [](const auto& param_info) { return param_info.param.name; });

TEST(Mpc, MatchesLqrWhileNoLimitIsActive) {
  const Reference ref{deg_to_rad(3.0), 1002.0, 50.0};
  const auto lqr = make_lqr_law(LawContext{ref, {}});
  const auto mpc = make_mpc_law(LawContext{ref, {}}, 0.5);
  lqr->configure(1.0 / 120.0);
  mpc->configure(1.0 / 120.0);
  VehicleState s = level(0.0, 1000.0, 50.0);
  s.roll = 0.03;
  s.p = -0.01;
  s.pitch = 0.02;
  s.q = 0.01;
  const RawActuators a = raw(lqr, s);
  const RawActuators b = raw(mpc, s);
  EXPECT_NEAR(a.aileron, b.aileron, 1e-6);
  EXPECT_NEAR(a.elevator, b.elevator, 1e-6);
}

TEST(Mpc, RejectsAHorizonShorterThanOneControlPeriod) {
  const double dt = 1.0 / 120.0;
  EXPECT_THROW(make_mpc_law(LawContext{}, 0.4 * dt)->configure(dt), std::invalid_argument);
  EXPECT_THROW(make_mpc_law(LawContext{}, -1.0), std::invalid_argument);
  EXPECT_THROW(make_mpc_law(LawContext{}, std::nan("")), std::invalid_argument);
}

TEST(Mpc, EntryRegistersTheHorizonFlag) {
  const LawEntry entry = mpc_law_entry();
  ASSERT_TRUE(entry.add_options);
  Options options("prog", "test");
  entry.add_options(options);
  const char* argv[] = {"prog", "--horizon-s", "0.001"};
  ASSERT_FALSE(options.parse(3, argv));
  // 0.001 s is less than one 120 Hz period: the parsed value reached the factory.
  EXPECT_THROW(entry.make(LawContext{})->configure(1.0 / 120.0), std::invalid_argument);
}

TEST(Lqr, EntryHasNoFlags) {
  const LawEntry entry = lqr_law_entry();
  EXPECT_EQ(entry.name, "lqr");
  EXPECT_FALSE(entry.add_options);
}

}  // namespace
}  // namespace astrum
