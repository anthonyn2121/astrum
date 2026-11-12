#include "astrum/math.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace astrum {
namespace {

TEST(WrapPi, IsIdentityInsideTheHalfOpenRange) {
  EXPECT_DOUBLE_EQ(wrap_pi(0.0), 0.0);
  EXPECT_DOUBLE_EQ(wrap_pi(1.0), 1.0);
  EXPECT_DOUBLE_EQ(wrap_pi(-1.0), -1.0);
  EXPECT_DOUBLE_EQ(wrap_pi(-kPi), -kPi);
}

TEST(WrapPi, MapsPlusPiToMinusPi) { EXPECT_DOUBLE_EQ(wrap_pi(kPi), -kPi); }

TEST(WrapPi, WrapsMultipleTurns) {
  EXPECT_NEAR(wrap_pi(0.5 + 2.0 * kPi), 0.5, 1e-12);
  EXPECT_NEAR(wrap_pi(0.5 - 6.0 * kPi), 0.5, 1e-12);
  EXPECT_NEAR(wrap_pi(deg_to_rad(270.0)), deg_to_rad(-90.0), 1e-12);
}

TEST(WrapPi, StaysInRangeForHugeAndTinyInputs) {
  for (const double a : {1e300, -1e300, 1e-300, -1e-300, -1e-17, 2.0 * kPi - 1e-16, 1e9 + 0.25}) {
    const double w = wrap_pi(a);
    EXPECT_GE(w, -kPi) << a;
    EXPECT_LT(w, kPi) << a;
  }
}

TEST(WrapPi, NonFiniteInputReturnsNaNWithoutLooping) {
  EXPECT_TRUE(std::isnan(wrap_pi(std::numeric_limits<double>::infinity())));
  EXPECT_TRUE(std::isnan(wrap_pi(-std::numeric_limits<double>::infinity())));
  EXPECT_TRUE(std::isnan(wrap_pi(std::numeric_limits<double>::quiet_NaN())));
}

}  // namespace
}  // namespace astrum
