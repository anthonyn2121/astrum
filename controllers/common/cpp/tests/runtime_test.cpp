// run_main's flag handling: every case here fails (or prints help) before any Zenoh
// session is opened.

#include "astrum/runtime.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace astrum {
namespace {

class NullLaw : public Law {
 public:
  std::vector<VehicleKind> vehicle_kinds() const override { return {VehicleKind::FixedWing}; }
  LawResult step(const StepInput& /*input*/) override { return Action{RawActuators{}}; }
};

class RunMainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (const char* env :
         {"ASTRUM_LAW", "ASTRUM_TARGET_ID", "ASTRUM_ZENOH_CONNECT", "ASTRUM_STARTUP_TIMEOUT_S",
          "ASTRUM_TELEMETRY_TIMEOUT_S", "ASTRUM_TARGET_HEADING_DEG", "ASTRUM_TARGET_ALT_M",
          "ASTRUM_TARGET_SPD_MPS", "ASTRUM_MAX_BANK_DEG", "ASTRUM_MAX_PITCH_DEG"}) {
      unsetenv(env);
    }
  }
  void TearDown() override { SetUp(); }

  int run(std::vector<const char*> args) {
    args.insert(args.begin(), "prog");
    const LawEntry laws[] = {
        {"null", "does nothing", nullptr,
         [this](const LawContext& ctx) {
           made.push_back(ctx);
           return std::make_unique<NullLaw>();
         }},
        {"picky", "rejects its flags",
         [this](Options& o) { o.add("--gain", "k", "gain", nullptr, &gain); },
         [this](const LawContext&) -> std::unique_ptr<Law> {
           if (gain < 0.0) throw std::invalid_argument("--gain must be >= 0");
           return std::make_unique<NullLaw>();
         }},
    };
    return run_main(static_cast<int>(args.size()), const_cast<char**>(args.data()), laws, "prog",
                    "test");
  }

  double gain = 1.0;
  std::vector<LawContext> made;
};

TEST_F(RunMainTest, HelpExitsZero) { EXPECT_EQ(run({"--help"}), 0); }

TEST_F(RunMainTest, IdIsRequiredAndNonZero) {
  EXPECT_EQ(run({}), 2);
  EXPECT_EQ(run({"--id", "0"}), 2);
  EXPECT_EQ(run({"--id", "4294967296"}), 2);
  EXPECT_TRUE(made.empty());
}

TEST_F(RunMainTest, UnknownLawIsAUsageError) {
  EXPECT_EQ(run({"--id", "1", "--law", "nope"}), 2);
  setenv("ASTRUM_LAW", "nope", 1);
  EXPECT_EQ(run({"--id", "1"}), 2);
}

TEST_F(RunMainTest, OutOfRangeAndNonFiniteNumbersAreUsageErrors) {
  EXPECT_EQ(run({"--id", "1", "--max-bank-deg", "90"}), 2);
  EXPECT_EQ(run({"--id", "1", "--max-pitch-deg", "0"}), 2);
  EXPECT_EQ(run({"--id", "1", "--speed", "-1"}), 2);
  EXPECT_EQ(run({"--id", "1", "--alt", "nan"}), 2);
  EXPECT_EQ(run({"--id", "1", "--startup-timeout", "0"}), 2);
  EXPECT_EQ(run({"--id", "1", "--telemetry-timeout", "-5"}), 2);
  EXPECT_EQ(run({"--id", "1", "--rate-hz", "120"}), 2);  // the rate is not a flag
  EXPECT_TRUE(made.empty());
}

TEST_F(RunMainTest, LawFlagsAreRegisteredAndTheFactoryValidatesThem) {
  EXPECT_EQ(run({"--id", "1", "--law", "picky", "--gain", "-1"}), 2);
  EXPECT_EQ(gain, -1.0);
}

}  // namespace
}  // namespace astrum
