// Dense linear algebra for the min-snap solver: row-major matrix and an LU
// factorisation with partial pivoting plus one step of iterative refinement. The KKT
// systems are ~8M x 8M (M trajectory segments), a few hundred unknowns, so a plain
// dense solve is the right size of tool.

#ifndef ASTRUM_MINSNAP_LINALG_HPP
#define ASTRUM_MINSNAP_LINALG_HPP

#include <cstddef>
#include <vector>

namespace astrum::minsnap {

class Matrix {
 public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), a_(rows * cols, 0.0) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  double& operator()(std::size_t r, std::size_t c) { return a_[r * cols_ + c]; }
  double operator()(std::size_t r, std::size_t c) const { return a_[r * cols_ + c]; }

  // y = A x.
  std::vector<double> multiply(const std::vector<double>& x) const;

 private:
  std::size_t rows_ = 0, cols_ = 0;
  std::vector<double> a_;
};

// Factorises a (row / column equilibrated) square matrix once; solve() may then be called for any
// number of right-hand sides. The constructor throws std::runtime_error when the matrix is
// numerically singular.
class LuFactor {
 public:
  explicit LuFactor(Matrix a);

  // Solves A x = b, refined once against the original matrix.
  std::vector<double> solve(const std::vector<double>& b) const;

 private:
  std::vector<double> solve_factored(std::vector<double> b) const;

  Matrix original_;
  Matrix lu_;
  std::vector<std::size_t> pivot_;
  std::vector<double> row_scale_, col_scale_;
};

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_LINALG_HPP
