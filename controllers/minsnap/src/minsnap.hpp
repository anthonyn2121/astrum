// Piecewise-polynomial minimum-derivative trajectories (Mellinger & Kumar, "Minimum
// Snap Trajectory Generation and Control for Quadrotors", ICRA 2011, section V).
//
// A trajectory through M + 1 knots is M degree-7 polynomial segments per channel. It
// minimises the integral of the squared `cost_order`-th derivative (4 = snap for the
// position channels, 2 for yaw) subject to equality constraints on the derivatives at
// the knots; derivatives 1..3 are kept continuous at every interior knot unless a
// constraint pins them. Because the cost is a quadratic form and the constraints are
// linear, the optimum is the solution of one KKT system (closed form, no QP solver).
//
// Each segment is stored in nondimensional time tau = (t - t_i) / T_i in [0, 1]
// (section V-A), which keeps the 8 x 8 blocks well conditioned whatever the segment
// duration is. All channels share the constraint pattern and the durations, so the
// KKT matrix (block banded, assembled sparse) is factorised once and each channel is one
// right-hand side. `SolvedProblem` keeps that factorisation so a time optimiser can also ask for
// the derivatives of any objective of the solution with respect to the durations and the
// pinned values (adjoint, one extra solve per objective).

#ifndef ASTRUM_MINSNAP_MINSNAP_HPP
#define ASTRUM_MINSNAP_MINSNAP_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace astrum::minsnap {

inline constexpr int kCoefficients = 8;  // degree 7
inline constexpr int kMaxDerivative = 7;

// Pins derivative `derivative` of every channel at knot `knot` to `value[channel]`.
// Every knot needs a derivative-0 (position) pin; derivatives 1..3 may be pinned too
// (left unpinned they are free at the two ends and continuous in between).
struct Constraint {
  std::size_t knot = 0;
  int derivative = 0;
  std::vector<double> value;
};

struct Problem {
  int cost_order = 4;             // k in "minimise the integral of (d^k x / dt^k)^2"
  std::vector<double> durations;  // T_i > 0, one per segment
  std::size_t channels = 0;
  std::vector<Constraint> constraints;
};

// d-th derivative (0..7) of the monomial basis tau^k, k = 0..7, at tau: for a segment of
// duration T with coefficients c, the d-th time derivative of the channel is
// T^-d * (basis_row(tau, d) . c).
std::array<double, kCoefficients> basis_row(double tau, int derivative);

class Trajectory {
 public:
  std::size_t segments() const { return durations_.size(); }
  std::size_t channels() const { return channels_; }
  double total_time() const { return starts_.empty() ? 0.0 : starts_.back(); }
  const std::vector<double>& durations() const { return durations_; }
  // Start time of every knot (size segments + 1, first 0).
  const std::vector<double>& knot_times() const { return starts_; }
  // The minimised integral, summed over channels.
  double cost() const { return cost_; }

  // Coefficients c_0..c_7 of tau^k of one segment of one channel (tau in [0, 1]).
  const std::array<double, kCoefficients>& coefficients(std::size_t segment,
                                                        std::size_t channel) const {
    return coefficients_[segment * channels_ + channel];
  }

  // d-th derivative (0..7) of one channel at time t, with t clamped to [0, total_time].
  double value(double t, int derivative, std::size_t channel) const;
  // d-th derivative of every channel.
  std::vector<double> value(double t, int derivative) const;

 private:
  friend class SolvedProblem;

  std::size_t channels_ = 0;
  std::vector<double> durations_;
  std::vector<double> starts_;
  std::vector<std::array<double, kCoefficients>> coefficients_;  // [segment * channels + channel]
  double cost_ = 0.0;
};

// A solved problem whose factorisation is kept for adjoint solves. The constructor throws
// std::invalid_argument for a malformed problem (no position pin at a knot, a non-positive
// duration, more constraints than unknowns) and std::runtime_error when the constraints
// are inconsistent (singular KKT system).
class SolvedProblem {
 public:
  // Derivatives of a scalar objective Phi with respect to every duration and every
  // constraint value; d_constraints[i][channel] belongs to Problem::constraints[i].value.
  struct Gradient {
    std::vector<double> d_durations;
    std::vector<std::vector<double>> d_constraints;
  };

  explicit SolvedProblem(const Problem& problem);
  ~SolvedProblem();
  SolvedProblem(SolvedProblem&&) noexcept;
  SolvedProblem& operator=(SolvedProblem&&) noexcept;

  const Trajectory& trajectory() const { return trajectory_; }

  // Total derivatives of Phi(c, T) given dPhi/dc (size segments * channels * 8, index
  // [(segment * channels + channel) * 8 + k], the nondimensional coefficients) and the
  // explicit dPhi/dT_i at fixed c (size segments). Throws std::invalid_argument on a size
  // mismatch.
  Gradient adjoint(const std::vector<double>& dphi_dc,
                   const std::vector<double>& dphi_dt_explicit) const;

  // Gradient of the minimised cost (summed over channels).
  Gradient cost_gradient() const;

 private:
  struct Impl;

  Trajectory trajectory_;
  std::unique_ptr<Impl> impl_;
};

// Solves the problem (same exceptions as SolvedProblem).
Trajectory solve(const Problem& problem);

// Section V-C: moves time between segments, total fixed, by projected gradient descent
// with backtracking on the optimal cost f(T). `iterations` = 0 returns the durations
// unchanged. No segment is allowed to shrink below 20 % of its initial duration.
std::vector<double> reallocate_durations(const Problem& problem, int iterations);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_MINSNAP_HPP
