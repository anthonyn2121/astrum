#include "minsnap.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace astrum::minsnap {
namespace {

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Block = Eigen::Matrix<double, kCoefficients, kCoefficients>;
using LuFactor = Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>>;

Index idx(std::size_t i) { return static_cast<Index>(i); }

// Falling factorial i (i-1) ... (i-d+1), i.e. i! / (i-d)!.
double falling(int i, int d) {
  double f = 1.0;
  for (int j = 0; j < d; ++j) f *= static_cast<double>(i - j);
  return f;
}

// Cost matrix of one segment of unit duration: the integral over tau in [0,1] of
// (p^(k)(tau))^2. A segment of duration T scales it by T^(1 - 2k) (T^-k per derivative
// factor, T from dt = T dtau).
Block unit_cost_block(int k) {
  Block q = Block::Zero();
  for (int i = k; i < kCoefficients; ++i) {
    for (int j = k; j < kCoefficients; ++j) {
      q(i, j) = falling(i, k) * falling(j, k) / static_cast<double>(i + j - 2 * k + 1);
    }
  }
  return q;
}

// One constraint row of the KKT system. A pin is one side of a knot: the end (tau = 1) of
// the segment before it or the start (tau = 0) of the one after it. A continuity row ties
// derivative `derivative` at the start of segment `segment` to the end of the segment before.
struct RowSpec {
  bool pin = true;
  std::size_t constraint = 0;  // pins only: index into Problem::constraints
  std::size_t segment = 0;
  int derivative = 0;
  bool at_end = false;  // pins only
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

std::array<double, kCoefficients> basis_row(double tau, int derivative) {
  std::array<double, kCoefficients> row{};
  for (int k = derivative; k < kCoefficients; ++k)
    row[static_cast<std::size_t>(k)] = falling(k, derivative) * std::pow(tau, k - derivative);
  return row;
}

// The assembled KKT system  [2Q A^T; A 0] [c; lambda] = [0; b]  and its factorisation.
struct SolvedProblem::Impl {
  using Factor = LuFactor;

  explicit Impl(const Problem& problem);
  ~Impl();

  // K^-1 rhs for the unscaled system, refined once against the matrix.
  MatrixXd solve(const MatrixXd& rhs) const;

  Problem problem;
  std::size_t unknowns = 0;
  std::vector<RowSpec> rows;
  Eigen::SparseMatrix<double> kkt;     // D K D
  VectorXd scale;                      // D
  std::vector<std::uint32_t> pattern;  // the rows' structure: what the symbolic analysis is for
  std::unique_ptr<Factor> lu;
  MatrixXd solution;  // [c; lambda], one column per channel
};

namespace {

// The symbolic analysis (the column ordering) of a KKT matrix depends only on the structure
// of its rows, which an optimiser re-solving the same problem with other durations and
// values keeps: a finished system hands its factor object back, and the next one with the
// same structure only factorises.
struct FactorCache {
  using Entry = std::pair<std::vector<std::uint32_t>, std::unique_ptr<LuFactor>>;
  static constexpr std::size_t kEntries = 6;
  std::vector<Entry> entries;
};

FactorCache& factor_cache() {
  thread_local FactorCache cache;
  return cache;
}

}  // namespace

SolvedProblem::Impl::~Impl() {
  if (!lu) return;
  FactorCache& cache = factor_cache();
  if (cache.entries.size() >= FactorCache::kEntries) cache.entries.erase(cache.entries.begin());
  cache.entries.emplace_back(std::move(pattern), std::move(lu));
}

SolvedProblem::Impl::Impl(const Problem& p) : problem(p) {
  validate(problem);
  const std::size_t m = problem.durations.size();
  const std::size_t ch = problem.channels;
  const int order = problem.cost_order;
  const std::vector<double>& dur = problem.durations;
  unknowns = m * kCoefficients;

  // Pins: a segment-local row per side of the knot, scaled to nondimensional time
  // (d^d x / dt^d = T^-d d^d x / dtau^d).
  std::vector<std::vector<bool>> pinned(m + 1, std::vector<bool>(4, false));
  for (std::size_t ci = 0; ci < problem.constraints.size(); ++ci) {
    const Constraint& c = problem.constraints[ci];
    if (c.derivative <= 3) pinned[c.knot][static_cast<std::size_t>(c.derivative)] = true;
    if (c.knot > 0) rows.push_back({true, ci, c.knot - 1, c.derivative, true});
    if (c.knot < m) rows.push_back({true, ci, c.knot, c.derivative, false});
  }
  // Continuity of derivatives 1..3 at interior knots that do not pin them.
  for (std::size_t j = 1; j < m; ++j) {
    for (int d = 1; d <= 3; ++d) {
      if (!pinned[j][static_cast<std::size_t>(d)]) rows.push_back({false, 0, j, d, false});
    }
  }
  if (rows.size() > unknowns) {
    throw std::invalid_argument("minsnap: " + std::to_string(rows.size()) + " constraints for " +
                                std::to_string(unknowns) + " unknowns");
  }

  const std::size_t total = unknowns + rows.size();
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(m * 64 + rows.size() * 32);
  const Block q0 = unit_cost_block(order);
  for (std::size_t seg = 0; seg < m; ++seg) {
    const double s = std::pow(dur[seg], 1 - 2 * order);
    for (int i = order; i < kCoefficients; ++i) {
      for (int j = order; j < kCoefficients; ++j) {
        entries.emplace_back(idx(seg) * kCoefficients + i, idx(seg) * kCoefficients + j,
                             2.0 * s * q0(i, j));
      }
    }
  }
  MatrixXd rhs = MatrixXd::Zero(idx(total), idx(ch));
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const RowSpec& row = rows[r];
    const Index at = idx(unknowns + r);
    auto add = [&](std::size_t seg, double tau, double scale_factor) {
      const auto basis = basis_row(tau, row.derivative);
      for (int i = row.derivative; i < kCoefficients; ++i) {
        const double v = scale_factor * basis[static_cast<std::size_t>(i)];
        entries.emplace_back(at, idx(seg) * kCoefficients + i, v);
        entries.emplace_back(idx(seg) * kCoefficients + i, at, v);
      }
    };
    if (row.pin) {
      add(row.segment, row.at_end ? 1.0 : 0.0, 1.0);
      const Constraint& c = problem.constraints[row.constraint];
      const double s = std::pow(dur[row.segment], row.derivative);
      for (std::size_t i = 0; i < ch; ++i) rhs(at, idx(i)) = c.value[i] * s;
    } else {
      add(row.segment - 1, 1.0, 1.0);
      add(row.segment, 0.0, -std::pow(dur[row.segment - 1] / dur[row.segment], row.derivative));
    }
  }
  kkt.resize(idx(total), idx(total));
  kkt.setFromTriplets(entries.begin(), entries.end());

  // Ruiz equilibration: the KKT systems mix blocks whose scales differ by many orders of
  // magnitude (a 0.25 s segment's snap cost is ~1e10 times a 2.5 s segment's), which no
  // single pivot threshold can judge. Scale rows and columns symmetrically until every
  // entry is O(1); the solution is unchanged (x = D y with (D K D) y = D b).
  scale = VectorXd::Ones(idx(total));
  for (int sweep = 0; sweep < 12; ++sweep) {
    VectorXd big = VectorXd::Zero(idx(total));
    for (Index col = 0; col < kkt.outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(kkt, col); it; ++it) {
        big[it.row()] = std::max(big[it.row()], std::fabs(it.value()));
      }
    }
    for (Index i = 0; i < big.size(); ++i) big[i] = big[i] > 0.0 ? 1.0 / std::sqrt(big[i]) : 1.0;
    for (Index col = 0; col < kkt.outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(kkt, col); it; ++it) {
        it.valueRef() *= big[it.row()] * big[col];
      }
    }
    scale.array() *= big.array();
  }

  pattern.reserve(rows.size() + 1);
  pattern.push_back(static_cast<std::uint32_t>(m));
  for (const RowSpec& row : rows) {
    pattern.push_back(static_cast<std::uint32_t>(row.segment) << 8 |
                      static_cast<std::uint32_t>(row.derivative) << 2 | (row.pin ? 2u : 0u) |
                      (row.at_end ? 1u : 0u));
  }
  {
    FactorCache& cache = factor_cache();
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
      if (it->first == pattern) {
        lu = std::move(it->second);
        cache.entries.erase(it);
        break;
      }
    }
  }
  if (!lu) {
    lu = std::make_unique<Factor>();
    lu->analyzePattern(kkt);
  }
  lu->factorize(kkt);
  if (lu->info() != Eigen::Success) throw std::runtime_error("minsnap: singular KKT system");
  solution = solve(rhs);
  if (!solution.allFinite()) throw std::runtime_error("minsnap: singular KKT system");
}

MatrixXd SolvedProblem::Impl::solve(const MatrixXd& rhs) const {
  const MatrixXd b = scale.asDiagonal() * rhs;
  MatrixXd y = lu->solve(b);
  y += lu->solve(MatrixXd(b - kkt * y));
  return scale.asDiagonal() * y;
}

SolvedProblem::SolvedProblem(const Problem& problem) : impl_(std::make_unique<Impl>(problem)) {
  const Problem& p = impl_->problem;
  const std::size_t m = p.durations.size();
  const std::size_t ch = p.channels;
  trajectory_.channels_ = ch;
  trajectory_.durations_ = p.durations;
  trajectory_.starts_.assign(m + 1, 0.0);
  for (std::size_t i = 0; i < m; ++i)
    trajectory_.starts_[i + 1] = trajectory_.starts_[i] + p.durations[i];
  trajectory_.coefficients_.assign(m * ch, {});
  const Block q0 = unit_cost_block(p.cost_order);
  for (std::size_t seg = 0; seg < m; ++seg) {
    const double s = std::pow(p.durations[seg], 1 - 2 * p.cost_order);
    for (std::size_t c = 0; c < ch; ++c) {
      const Eigen::Matrix<double, kCoefficients, 1> x =
          impl_->solution.block<kCoefficients, 1>(idx(seg) * kCoefficients, idx(c));
      for (std::size_t i = 0; i < kCoefficients; ++i)
        trajectory_.coefficients_[seg * ch + c][i] = x[idx(i)];
      trajectory_.cost_ += s * x.dot(q0 * x);
    }
  }
}

SolvedProblem::~SolvedProblem() = default;
SolvedProblem::SolvedProblem(SolvedProblem&&) noexcept = default;
SolvedProblem& SolvedProblem::operator=(SolvedProblem&&) noexcept = default;

SolvedProblem::Gradient SolvedProblem::adjoint(const std::vector<double>& dphi_dc,
                                               const std::vector<double>& dphi_dt_explicit) const {
  const Problem& p = impl_->problem;
  const std::size_t m = p.durations.size();
  const std::size_t ch = p.channels;
  const std::size_t n = impl_->unknowns;
  if (dphi_dc.size() != m * ch * kCoefficients || dphi_dt_explicit.size() != m) {
    throw std::invalid_argument("SolvedProblem::adjoint: gradient size mismatch");
  }
  const std::vector<double>& dur = p.durations;

  // K is symmetric, so one solve with [dPhi/dc; 0] gives the multipliers y of the sensitivity
  // equation: dPhi/dtheta = explicit + y^T (db/dtheta - dK/dtheta [c; lambda]).
  MatrixXd seed = MatrixXd::Zero(idx(n + impl_->rows.size()), idx(ch));
  for (std::size_t seg = 0; seg < m; ++seg) {
    for (std::size_t c = 0; c < ch; ++c) {
      for (std::size_t i = 0; i < kCoefficients; ++i) {
        seed(idx(seg) * kCoefficients + idx(i), idx(c)) =
            dphi_dc[(seg * ch + c) * kCoefficients + i];
      }
    }
  }
  const MatrixXd y = impl_->solve(seed);
  const MatrixXd& u = impl_->solution;

  Gradient g;
  g.d_durations = dphi_dt_explicit;
  g.d_constraints.assign(p.constraints.size(), std::vector<double>(ch, 0.0));

  // dK/dT_s on the cost block: 2 Q_s scales as T_s^(1 - 2k).
  const Block q0 = unit_cost_block(p.cost_order);
  for (std::size_t seg = 0; seg < m; ++seg) {
    const double factor = 2.0 * (1 - 2 * p.cost_order) * std::pow(dur[seg], -2 * p.cost_order);
    for (std::size_t c = 0; c < ch; ++c) {
      const Eigen::Matrix<double, kCoefficients, 1> ys =
          y.block<kCoefficients, 1>(idx(seg) * kCoefficients, idx(c));
      const Eigen::Matrix<double, kCoefficients, 1> us =
          u.block<kCoefficients, 1>(idx(seg) * kCoefficients, idx(c));
      g.d_durations[seg] -= factor * ys.dot(q0 * us);
    }
  }

  for (std::size_t r = 0; r < impl_->rows.size(); ++r) {
    const RowSpec& row = impl_->rows[r];
    const Index at = idx(n + r);
    const int d = row.derivative;
    if (row.pin) {
      // b = value * T^d on this side's segment.
      const Constraint& cons = p.constraints[row.constraint];
      for (std::size_t c = 0; c < ch; ++c) {
        g.d_constraints[row.constraint][c] += y(at, idx(c)) * std::pow(dur[row.segment], d);
        if (d > 0) {
          g.d_durations[row.segment] +=
              y(at, idx(c)) * cons.value[c] * d * std::pow(dur[row.segment], d - 1);
        }
      }
      continue;
    }
    // Continuity: the row holds w = -(T_{j-1} / T_j)^d on c_{j,d} * d!.
    const std::size_t j = row.segment;
    const double w = -std::pow(dur[j - 1] / dur[j], d);
    const Index coef = idx(j) * kCoefficients + d;
    const double dw_prev = d * w / dur[j - 1];
    const double dw_next = -d * w / dur[j];
    for (std::size_t c = 0; c < ch; ++c) {
      const double term =
          falling(d, d) * (y(at, idx(c)) * u(coef, idx(c)) + y(coef, idx(c)) * u(at, idx(c)));
      g.d_durations[j - 1] -= dw_prev * term;
      g.d_durations[j] -= dw_next * term;
    }
  }
  return g;
}

SolvedProblem::Gradient SolvedProblem::cost_gradient() const {
  const Problem& p = impl_->problem;
  const std::size_t m = p.durations.size();
  const std::size_t ch = p.channels;
  const Block q0 = unit_cost_block(p.cost_order);
  std::vector<double> dphi_dc(m * ch * kCoefficients);
  std::vector<double> dphi_dt(m, 0.0);
  for (std::size_t seg = 0; seg < m; ++seg) {
    const double s = std::pow(p.durations[seg], 1 - 2 * p.cost_order);
    for (std::size_t c = 0; c < ch; ++c) {
      const Eigen::Matrix<double, kCoefficients, 1> x =
          impl_->solution.block<kCoefficients, 1>(idx(seg) * kCoefficients, idx(c));
      const Eigen::Matrix<double, kCoefficients, 1> qx = 2.0 * s * (q0 * x);
      for (std::size_t i = 0; i < kCoefficients; ++i) {
        dphi_dc[(seg * ch + c) * kCoefficients + i] = qx[idx(i)];
      }
      dphi_dt[seg] += (1 - 2 * p.cost_order) / p.durations[seg] * s * x.dot(q0 * x);
    }
  }
  return adjoint(dphi_dc, dphi_dt);
}

Trajectory solve(const Problem& problem) { return SolvedProblem(problem).trajectory(); }

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
