#include "astrum/options.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

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

}  // namespace
}  // namespace astrum
