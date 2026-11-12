#include "astrum/options.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "astrum/cli.hpp"
#include "astrum/math.hpp"

namespace astrum {
namespace {

struct Parsed {
  std::optional<int> exit;
  double alt = 1500.0;
  std::uint32_t id = 1;
  std::string mode = "raw";
};

Parsed parse(std::vector<const char*> args) {
  Parsed p;
  Options options("prog", "test");
  options.add("--alt", "m", "altitude", "ASTRUM_TEST_ALT_M", &p.alt)
      .add("--id", "n", "id", "ASTRUM_TEST_ID", &p.id)
      .add("--mode", "m", "mode", nullptr, &p.mode);
  args.insert(args.begin(), "prog");
  p.exit = options.parse(static_cast<int>(args.size()), args.data());
  return p;
}

class OptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    unsetenv("ASTRUM_TEST_ALT_M");
    unsetenv("ASTRUM_TEST_ID");
  }
  void TearDown() override { SetUp(); }
};

TEST_F(OptionsTest, DefaultsApplyWithoutEnvOrFlags) {
  const Parsed p = parse({});
  EXPECT_FALSE(p.exit);
  EXPECT_EQ(p.alt, 1500.0);
  EXPECT_EQ(p.id, 1U);
}

TEST_F(OptionsTest, FlagBeatsEnvBeatsDefault) {
  setenv("ASTRUM_TEST_ALT_M", "900", 1);
  setenv("ASTRUM_TEST_ID", "3", 1);
  const Parsed env_only = parse({});
  EXPECT_EQ(env_only.alt, 900.0);
  EXPECT_EQ(env_only.id, 3U);
  const Parsed flags = parse({"--alt", "1200.5", "--id=4", "--mode", "autopilot"});
  EXPECT_FALSE(flags.exit);
  EXPECT_EQ(flags.alt, 1200.5);
  EXPECT_EQ(flags.id, 4U);
  EXPECT_EQ(flags.mode, "autopilot");
}

TEST_F(OptionsTest, MalformedNumbersAreUsageErrors) {
  EXPECT_EQ(parse({"--alt", "12abc"}).exit, 2);
  EXPECT_EQ(parse({"--alt", "inf"}).exit, 2);
  EXPECT_EQ(parse({"--id", "-1"}).exit, 2);
  EXPECT_EQ(parse({"--id", "4294967296"}).exit, 2);
  setenv("ASTRUM_TEST_ALT_M", "high", 1);
  EXPECT_EQ(parse({}).exit, 2);
}

TEST_F(OptionsTest, UnknownFlagsAndMissingValuesAreUsageErrors) {
  EXPECT_EQ(parse({"--altitude", "5"}).exit, 2);
  EXPECT_EQ(parse({"--alt"}).exit, 2);
}

TEST_F(OptionsTest, HelpExitsZero) { EXPECT_EQ(parse({"--help"}).exit, 0); }

TEST_F(OptionsTest, HelpWinsOverAnInvalidEnvironment) {
  setenv("ASTRUM_TEST_ALT_M", "high", 1);
  EXPECT_EQ(parse({"--help"}).exit, 0);
}

TEST_F(OptionsTest, RequiredOptionMustBeGivenByFlagOrEnv) {
  const auto run = [](std::vector<const char*> args) {
    std::uint32_t id = 0;
    Options options("prog", "test");
    options.add("--id", "n", "id", "ASTRUM_TEST_ID", &id).required();
    args.insert(args.begin(), "prog");
    return options.parse(static_cast<int>(args.size()), args.data());
  };
  EXPECT_EQ(run({}), 2);
  EXPECT_FALSE(run({"--id", "3"}));
  setenv("ASTRUM_TEST_ID", "3", 1);
  EXPECT_FALSE(run({}));
}

TEST_F(OptionsTest, AddingAFlagTwiceIsAProgrammingError) {
  double a = 0.0;
  Options options("prog", "test");
  options.add("--a", "x", "a", nullptr, &a);
  EXPECT_THROW(options.add("--a", "x", "a", nullptr, &a), std::logic_error);
}

TEST(Cli, ConnectionOptionsRequireAnIdAndPositiveTimeouts) {
  ConnectionOptions c;
  EXPECT_TRUE(validate(c));  // id 0
  c.id = 4294967295U;
  EXPECT_FALSE(validate(c));
  c.startup_timeout_s = 0.0;
  EXPECT_TRUE(validate(c));
  c.startup_timeout_s = 1.0;
  c.telemetry_timeout_s = -1.0;
  EXPECT_TRUE(validate(c));
}

TEST(Cli, IdFlagIsRequiredAndParsed) {
  unsetenv("ASTRUM_TARGET_ID");
  ConnectionOptions c;
  Options options("prog", "test");
  add_connection_options(options, c);
  const char* none[] = {"prog"};
  EXPECT_EQ(options.parse(1, none), 2);
  const char* some[] = {"prog", "--id", "7", "--connect=tcp/10.0.0.1:7447"};
  EXPECT_FALSE(options.parse(4, some));
  EXPECT_EQ(c.id, 7U);
  EXPECT_EQ(c.connect, "tcp/10.0.0.1:7447");
}

TEST(Cli, ReferenceIsWrappedRadiansAndSpeedIsNonNegative) {
  ReferenceOptions r;
  r.heading_deg = 270.0;
  r.alt_hae_m = 1200.0;
  r.tas_mps = 60.0;
  const Reference ref = r.reference();
  EXPECT_NEAR(ref.heading_rad, -kPi / 2.0, 1e-12);
  EXPECT_EQ(ref.alt_hae_m, 1200.0);
  EXPECT_EQ(ref.tas_mps, 60.0);
  EXPECT_FALSE(validate(r));
  r.tas_mps = -1.0;
  EXPECT_TRUE(validate(r));
}

TEST(Cli, AttitudeLimitsMustBeInsideZeroToNinetyDegrees) {
  EXPECT_FALSE(validate(AttitudeLimits{}));
  EXPECT_TRUE(validate(AttitudeLimits{0.0, 10.0}));
  EXPECT_TRUE(validate(AttitudeLimits{25.0, 90.0}));
}

}  // namespace
}  // namespace astrum
