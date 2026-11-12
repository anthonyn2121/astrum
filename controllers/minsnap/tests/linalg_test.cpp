#include "linalg.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace astrum::minsnap {
namespace {

TEST(Linalg, SolvesAGeneralSystemToMachinePrecision) {
  // A diagonally weak matrix that needs pivoting (zero leading entry).
  Matrix a(4, 4);
  const double v[4][4] = {{0, 2, 1, -1}, {3, -1, 2, 0.5}, {1, 4, -2, 1}, {-2, 1, 0.25, 3}};
  for (std::size_t r = 0; r < 4; ++r) {
    for (std::size_t c = 0; c < 4; ++c) a(r, c) = v[r][c];
  }
  const std::vector<double> x_true{1.5, -2.0, 0.75, 3.0};
  const std::vector<double> b = a.multiply(x_true);
  const std::vector<double> x = LuFactor(a).solve(b);
  for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(x[i], x_true[i], 1e-12);
}

TEST(Linalg, OneFactorisationSolvesManyRightHandSides) {
  Matrix a(2, 2);
  a(0, 0) = 4;
  a(0, 1) = 1;
  a(1, 0) = 2;
  a(1, 1) = 3;
  const LuFactor lu(a);
  for (double s : {1.0, -3.0, 0.5}) {
    const std::vector<double> x = lu.solve({s * 5.0, s * 5.0});
    EXPECT_NEAR(x[0], s, 1e-13);
    EXPECT_NEAR(x[1], s, 1e-13);
  }
}

TEST(Linalg, SingularAndMalformedInputsThrow) {
  Matrix singular(2, 2);
  singular(0, 0) = 1;
  singular(0, 1) = 2;
  singular(1, 0) = 2;
  singular(1, 1) = 4;
  EXPECT_THROW(LuFactor{singular}, std::runtime_error);
  EXPECT_THROW(LuFactor{Matrix(2, 3)}, std::invalid_argument);
  Matrix id(2, 2);
  id(0, 0) = id(1, 1) = 1;
  EXPECT_THROW(LuFactor(id).solve({1.0}), std::invalid_argument);
}

}  // namespace
}  // namespace astrum::minsnap
