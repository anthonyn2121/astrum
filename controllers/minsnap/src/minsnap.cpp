#include "minsnap.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "linalg.hpp"

namespace astrum::minsnap {
namespace {

// Falling factorial i (i-1) ... (i-d+1), i.e. i! / (i-d)!.
double falling(int i, int d) {
  double f = 1.0;
  for (int j = 0; j < d; ++j) f *= static_cast<double>(i - j);
  return f;
}

// Row of d/dtau^d of the monomial basis at tau.
std::array<double, kCoefficients> derivative_row(int d, double tau) {
  std::array<double, kCoefficients> row{};
  for (int k = d; k < kCoefficients; ++k)
    row[static_cast<std::size_t>(k)] = falling(k, d) * std::pow(tau, k - d);
  return row;
}

struct Row {
  std::vector<double> a;    // dense coefficients over all segments
  std::vector<double> rhs;  // one entry per channel
};

void validate(const Problem& p) {
  if (p.durations.empty()) throw std::invalid_argument("minsnap: no segments");
  if (p.channels == 0) throw std::invalid_argument("minsnap: no channels");
  if (p.cost_order < 1 || p.cost_order >= kCoefficients) {
    throw std::invalid_argument("minsnap: cost_order must be in [1, 7]");
  }
  for (double t : p.durations) {
    if (!(t > 0.0) || !std::isfinite(t))
      throw std::invalid_argument("minsnap: duration must be > 0");
  }
  const std::size_t knots = p.durations.size() + 1;
  std::vector<bool> has_position(knots, false);
  for (const Constraint& c : p.constraints) {
    if (c.knot >= knots) throw std::invalid_argument("minsnap: constraint knot out of range");
    if (c.derivative < 0 || c.derivative > kMaxDerivative) {
      throw std::invalid_argument("minsnap: constraint derivative out of range");
    }
    if (c.value.size() != p.channels) {
      throw std::invalid_argument("minsnap: constraint value has the wrong channel count");
    }
    if (c.derivative == 0) has_position[c.knot] = true;
  }
  for (std::size_t k = 0; k < knots; ++k) {
    if (!has_position[k])
      throw std::invalid_argument("minsnap: knot " + std::to_string(k) + " has no position");
  }
}

}  // namespace

double Trajectory::value(double t, int derivative, std::size_t channel) const {
  if (segments() == 0) throw std::logic_error("Trajectory::value on an empty trajectory");
  if (derivative < 0 || derivative > kMaxDerivative) return 0.0;
  t = std::clamp(t, 0.0, total_time());
  // Last knot start <= t, so that t == total_time falls in the last segment.
  std::size_t i = static_cast<std::size_t>(std::upper_bound(starts_.begin(), starts_.end() - 1, t) -
                                           starts_.begin());
  i = i == 0 ? 0 : i - 1;
  const double tau = std::clamp((t - starts_[i]) / durations_[i], 0.0, 1.0);
  const auto& c = coefficients_[i * channels_ + channel];
  double v = 0.0;
  for (int k = derivative; k < kCoefficients; ++k) {
    v += c[static_cast<std::size_t>(k)] * falling(k, derivative) * std::pow(tau, k - derivative);
  }
  return v / std::pow(durations_[i], derivative);
}

std::vector<double> Trajectory::value(double t, int derivative) const {
  std::vector<double> v(channels_);
  for (std::size_t ch = 0; ch < channels_; ++ch) v[ch] = value(t, derivative, ch);
  return v;
}

Trajectory solve(const Problem& problem) {
  validate(problem);
  const std::size_t m = problem.durations.size();
  const std::size_t n = m * kCoefficients;
  const std::size_t ch = problem.channels;
  const int k = problem.cost_order;

  std::vector<Row> rows;
  auto add_segment = [&](Row& row, std::size_t seg, int d, double tau, double scale) {
    const auto basis = derivative_row(d, tau);
    for (std::size_t i = 0; i < kCoefficients; ++i)
      row.a[seg * kCoefficients + i] += scale * basis[i];
  };
  auto new_row = [&]() {
    Row r;
    r.a.assign(n, 0.0);
    r.rhs.assign(ch, 0.0);
    return r;
  };

  // Pins: a segment-local row per side of the knot, scaled to nondimensional time
  // (d^d x / dt^d = T^-d d^d x / dtau^d).
  std::vector<std::vector<bool>> pinned(m + 1, std::vector<bool>(4, false));
  for (const Constraint& c : problem.constraints) {
    if (c.derivative <= 3) pinned[c.knot][static_cast<std::size_t>(c.derivative)] = true;
    if (c.knot > 0) {
      Row r = new_row();
      add_segment(r, c.knot - 1, c.derivative, 1.0, 1.0);
      const double s = std::pow(problem.durations[c.knot - 1], c.derivative);
      for (std::size_t i = 0; i < ch; ++i) r.rhs[i] = c.value[i] * s;
      rows.push_back(std::move(r));
    }
    if (c.knot < m) {
      Row r = new_row();
      add_segment(r, c.knot, c.derivative, 0.0, 1.0);
      const double s = std::pow(problem.durations[c.knot], c.derivative);
      for (std::size_t i = 0; i < ch; ++i) r.rhs[i] = c.value[i] * s;
      rows.push_back(std::move(r));
    }
  }
  // Continuity of derivatives 1..3 at interior knots that do not pin them.
  for (std::size_t j = 1; j < m; ++j) {
    for (int d = 1; d <= 3; ++d) {
      if (pinned[j][static_cast<std::size_t>(d)]) continue;
      Row r = new_row();
      add_segment(r, j - 1, d, 1.0, 1.0);
      add_segment(r, j, d, 0.0, -std::pow(problem.durations[j - 1] / problem.durations[j], d));
      rows.push_back(std::move(r));
    }
  }
  if (rows.size() > n) {
    throw std::invalid_argument("minsnap: " + std::to_string(rows.size()) + " constraints for " +
                                std::to_string(n) + " unknowns");
  }

  // Cost matrix: integral over tau in [0,1] of (T^-k p^(k)(tau))^2 * T.
  Matrix q(n, n);
  for (std::size_t seg = 0; seg < m; ++seg) {
    const double s = std::pow(problem.durations[seg], 1 - 2 * k);
    for (int i = k; i < kCoefficients; ++i) {
      for (int j = k; j < kCoefficients; ++j) {
        q(seg * kCoefficients + static_cast<std::size_t>(i),
          seg * kCoefficients + static_cast<std::size_t>(j)) =
            s * falling(i, k) * falling(j, k) / static_cast<double>(i + j - 2 * k + 1);
      }
    }
  }

  const std::size_t total = n + rows.size();
  Matrix kkt(total, total);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) kkt(i, j) = 2.0 * q(i, j);
  }
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      kkt(n + r, c) = rows[r].a[c];
      kkt(c, n + r) = rows[r].a[c];
    }
  }
  const LuFactor lu(std::move(kkt));

  Trajectory t;
  t.channels_ = ch;
  t.durations_ = problem.durations;
  t.starts_.assign(m + 1, 0.0);
  for (std::size_t i = 0; i < m; ++i) t.starts_[i + 1] = t.starts_[i] + problem.durations[i];
  t.coefficients_.assign(m * ch, {});
  for (std::size_t c = 0; c < ch; ++c) {
    std::vector<double> rhs(total, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r) rhs[n + r] = rows[r].rhs[c];
    const std::vector<double> x = lu.solve(rhs);
    for (std::size_t seg = 0; seg < m; ++seg) {
      for (std::size_t i = 0; i < kCoefficients; ++i)
        t.coefficients_[seg * ch + c][i] = x[seg * kCoefficients + i];
    }
    const std::vector<double> qx =
        q.multiply(std::vector<double>(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n)));
    for (std::size_t i = 0; i < n; ++i) t.cost_ += x[i] * qx[i];
  }
  return t;
}

std::vector<double> reallocate_durations(const Problem& problem, int iterations) {
  std::vector<double> t = problem.durations;
  const std::size_t m = t.size();
  if (m < 2 || iterations <= 0) return t;
  std::vector<double> floor(m);
  for (std::size_t i = 0; i < m; ++i) floor[i] = 0.2 * t[i];

  auto cost_of = [&](const std::vector<double>& durations) {
    Problem p = problem;
    p.durations = durations;
    return solve(p).cost();
  };
  double mean = 0.0;
  for (double v : t) mean += v / static_cast<double>(m);
  const double h = 1e-3 * mean;

  double f = cost_of(t);
  for (int it = 0; it < iterations; ++it) {
    // Directional derivative along g_i: +1 on segment i, -1/(m-1) on the others, so the
    // total time never changes.
    std::vector<double> descent(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      std::vector<double> shifted = t;
      for (std::size_t j = 0; j < m; ++j) {
        shifted[j] += h * (j == i ? 1.0 : -1.0 / static_cast<double>(m - 1));
      }
      const double grad = (cost_of(shifted) - f) / h;
      for (std::size_t j = 0; j < m; ++j) {
        descent[j] -= grad * (j == i ? 1.0 : -1.0 / static_cast<double>(m - 1));
      }
    }
    double biggest = 0.0;
    for (double d : descent) biggest = std::max(biggest, std::fabs(d));
    if (!(biggest > 0.0)) break;
    double alpha = 0.25 * mean / biggest;
    bool improved = false;
    for (int back = 0; back < 20; ++back, alpha *= 0.5) {
      std::vector<double> next = t;
      bool ok = true;
      for (std::size_t j = 0; j < m; ++j) {
        next[j] += alpha * descent[j];
        ok = ok && next[j] >= floor[j];
      }
      if (!ok) continue;
      const double fn = cost_of(next);
      if (fn < f) {
        t = std::move(next);
        f = fn;
        improved = true;
        break;
      }
    }
    if (!improved) break;
  }
  return t;
}

}  // namespace astrum::minsnap
