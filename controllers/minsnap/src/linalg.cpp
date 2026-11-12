#include "linalg.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace astrum::minsnap {

std::vector<double> Matrix::multiply(const std::vector<double>& x) const {
  if (x.size() != cols_) throw std::invalid_argument("Matrix::multiply: size mismatch");
  std::vector<double> y(rows_, 0.0);
  for (std::size_t r = 0; r < rows_; ++r) {
    double s = 0.0;
    for (std::size_t c = 0; c < cols_; ++c) s += (*this)(r, c) * x[c];
    y[r] = s;
  }
  return y;
}

LuFactor::LuFactor(Matrix a)
    : original_(a),
      lu_(std::move(a)),
      pivot_(original_.rows()),
      row_scale_(original_.rows(), 1.0),
      col_scale_(original_.rows(), 1.0) {
  const std::size_t n = lu_.rows();
  if (lu_.cols() != n) throw std::invalid_argument("LuFactor: matrix is not square");

  // Ruiz equilibration: the KKT systems mix blocks whose scales differ by many orders of
  // magnitude (a 0.25 s segment's snap cost is ~1e10 times a 2.5 s segment's), which no
  // single pivot threshold can judge. Scale rows and columns until every entry is O(1);
  // the solution is unchanged (x = C y with (R A C) y = R b).
  for (int sweep = 0; sweep < 12; ++sweep) {
    std::vector<double> r(n, 0.0), c(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double v = std::fabs(lu_(i, j));
        r[i] = std::fmax(r[i], v);
        c[j] = std::fmax(c[j], v);
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      r[i] = r[i] > 0.0 ? 1.0 / std::sqrt(r[i]) : 1.0;
      c[i] = c[i] > 0.0 ? 1.0 / std::sqrt(c[i]) : 1.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) lu_(i, j) *= r[i] * c[j];
      row_scale_[i] *= r[i];
      col_scale_[i] *= c[i];
    }
  }

  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t r = k + 1; r < n; ++r) {
      if (std::fabs(lu_(r, k)) > std::fabs(lu_(p, k))) p = r;
    }
    if (!(std::fabs(lu_(p, k)) > 1e-12)) {
      throw std::runtime_error("LuFactor: singular matrix (pivot " + std::to_string(k) + ")");
    }
    pivot_[k] = p;
    if (p != k) {
      for (std::size_t c = 0; c < n; ++c) std::swap(lu_(k, c), lu_(p, c));
    }
    for (std::size_t r = k + 1; r < n; ++r) {
      const double f = lu_(r, k) / lu_(k, k);
      lu_(r, k) = f;
      if (f == 0.0) continue;
      for (std::size_t c = k + 1; c < n; ++c) lu_(r, c) -= f * lu_(k, c);
    }
  }
}

std::vector<double> LuFactor::solve_factored(std::vector<double> b) const {
  const std::size_t n = lu_.rows();
  for (std::size_t i = 0; i < n; ++i) b[i] *= row_scale_[i];
  for (std::size_t k = 0; k < n; ++k) std::swap(b[k], b[pivot_[k]]);
  for (std::size_t r = 1; r < n; ++r) {
    double s = b[r];
    for (std::size_t c = 0; c < r; ++c) s -= lu_(r, c) * b[c];
    b[r] = s;
  }
  for (std::size_t r = n; r-- > 0;) {
    double s = b[r];
    for (std::size_t c = r + 1; c < n; ++c) s -= lu_(r, c) * b[c];
    b[r] = s / lu_(r, r);
  }
  for (std::size_t i = 0; i < n; ++i) b[i] *= col_scale_[i];
  return b;
}

std::vector<double> LuFactor::solve(const std::vector<double>& b) const {
  if (b.size() != lu_.rows()) throw std::invalid_argument("LuFactor::solve: size mismatch");
  std::vector<double> x = solve_factored(b);
  const std::vector<double> ax = original_.multiply(x);
  std::vector<double> residual(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) residual[i] = b[i] - ax[i];
  const std::vector<double> dx = solve_factored(std::move(residual));
  for (std::size_t i = 0; i < x.size(); ++i) x[i] += dx[i];
  return x;
}

}  // namespace astrum::minsnap
