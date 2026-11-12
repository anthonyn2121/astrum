// Time-optimal minimum snap: choose the segment durations T_i and the speeds the
// aircraft is pinned to at the gate centres so that the lap takes as little time as the
// airframe's limits allow, with the shape of every segment still the minimum-snap
// solution for those times (Richter, Bry & Roy, "Polynomial Trajectory Planning for
// Aggressive Quadrotor Flight in Dense Indoor Environments", ISRR 2013, with the limits
// evaluated through the differential flatness of flatness.hpp).
//
// Variables z = (log T_i for every segment, log s_g for every gate): relative changes of
// the durations and of the crossing speeds. The inner
// problems (position and yaw QPs, closed form in minsnap.hpp) are solved for each z and
// differentiated by their KKT adjoint, so the gradient of
//
//   F(z) = sum_i T_i + w_snap J_snap(z) / J_snap(z0) + mu sum_samples sum_c max(0, x_c)^2
//
// is analytic. x_c are the constraints, each normalised by its limit and taken at
// `margin` of it (x_c <= 0 satisfies the constraint): speed, thrust (upper and lower),
// tilt, roll / pitch body rate and yaw rate. They are sampled at fixed nondimensional
// times of every segment, so a sample moves with its segment. The penalty weight mu
// grows by continuation; each round is a projected L-BFGS with Armijo backtracking on
// the bounds T_i >= min_duration, s_g in [min_gate_speed, v_max].
//
// The search is deterministic (an iteration budget, no clock) and the result is never
// worse than the starting point: every evaluated point is checked on a dense time grid
// against the true limits, and the fastest one that passes is returned. A starting point
// that does not pass is itself the answer only if no evaluated point passes or is
// less infeasible; the result then may be slower (feasibility first, speed second).

#ifndef ASTRUM_MINSNAP_TIMEOPT_HPP
#define ASTRUM_MINSNAP_TIMEOPT_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "minsnap.hpp"

namespace astrum::minsnap {

// True limits of the airframe; the optimiser keeps `margin` of each in its penalty.
struct Limits {
  double v_max = 0.0;         // m/s
  double f_max = 0.0;         // m/s^2, |r'' + g z_W| (thrust per unit mass)
  double f_min = 0.0;         // m/s^2, the same quantity from below (> 0: never free fall)
  double tilt_max_rad = 0.0;  // angle between z_B and z_W
  double rate_max = 0.0;      // rad/s, roll / pitch body rate |omega_xy|
  double yaw_rate_max = 0.0;  // rad/s
  double margin = 0.9;
};

struct TimeOptParams {
  int iterations = 150;  // L-BFGS iterations over all continuation rounds
  double snap_weight = 1e-3;
  int samples_per_segment = 12;  // at least; see max_sample_dt
  double max_sample_dt = 0.15;   // s at the initial durations: long segments get more samples
  double mu0 = 1.0, mu_growth = 10.0, mu_max = 1e6;
  int lbfgs_memory = 8;
  double min_duration = 0.05;   // s
  double min_gate_speed = 0.5;  // m/s
  double grid_step_s = 0.05;    // dense feasibility grid
  // A result is feasible when no true limit is exceeded by more than this fraction.
  double feasibility_tol = 5e-3;
  // The first seconds of a plan that starts in flight are dictated by the state it starts
  // from: thrust, tilt and body rate are not constrained there (speed and yaw rate are).
  double start_transient_s = 0.0;
};

// The pin of a gate centre: the velocity constraint `position.constraints[constraint]`
// is `speed * direction` (the gate normal), the speed being the gate's variable.
struct GatePin {
  std::size_t constraint = 0;
  std::array<double, 3> direction{};
};

// The two inner problems for any candidate. `position` (3 channels, cost order 4) and
// `yaw` (1 channel, cost order 2) are templates: the optimiser replaces their durations
// and the gate pins' values and keeps everything else.
struct TimeOptModel {
  Problem position;
  Problem yaw;
  // Position segments [first, last) that each yaw segment spans; contiguous and covering
  // every position segment. A yaw duration is the sum of the position durations in it.
  std::vector<std::pair<std::size_t, std::size_t>> yaw_spans;
  std::vector<GatePin> gate_pins;  // one per gate variable
};

struct TimeOptResult {
  std::vector<double> durations;
  std::vector<double> gate_speeds;
  double total_time = 0.0;
  bool improved = false;       // faster than the starting point
  int iterations = 0;          // L-BFGS iterations taken
  double max_violation = 0.0;  // worst limit excess on the dense grid, fraction of the limit
};

// Minimises the lap time from (durations0, gate_speeds0). Throws std::invalid_argument
// for an inconsistent model and whatever minsnap::solve throws when the starting point
// itself cannot be solved; a candidate that cannot be solved is simply rejected.
TimeOptResult optimise_times(const TimeOptModel& model, const std::vector<double>& durations0,
                             const std::vector<double>& gate_speeds0, const Limits& limits,
                             const TimeOptParams& params);

// The objective F(z) above with its gradient (exposed for the gradient test).
class TimeObjective {
 public:
  // Fixes the sample counts (from durations0) and the snap normalisation J_snap(z0).
  TimeObjective(const TimeOptModel& model, const Limits& limits, const TimeOptParams& params,
                const std::vector<double>& durations0, const std::vector<double>& gate_speeds0);

  struct Value {
    double objective = 0.0;
    double time = 0.0;
    double snap = 0.0;     // J_snap(z) / J_snap(z0)
    double penalty = 0.0;  // sum of max(0, x_c)^2 over the samples (before mu)
    std::vector<double> gradient;
    // Worst true-limit excess on the dense grid; only when `dense` was requested.
    std::optional<double> dense_excess;
  };

  // `dense`: also the dense-grid check. When the samples alone already exceed a true
  // limit by more than `dense_cutoff` (a fraction of the limit), the grid is not walked
  // and `dense_excess` is that sampled excess.
  Value evaluate(const std::vector<double>& z, double mu, bool dense = false,
                 double dense_cutoff = std::numeric_limits<double>::infinity()) const;

  std::size_t segments() const { return segments_; }
  std::size_t gates() const { return gates_; }
  // z = (log T_i, log s_g), and back.
  std::vector<double> variables(const std::vector<double>& durations,
                                const std::vector<double>& gate_speeds) const;
  std::vector<double> durations(const std::vector<double>& z) const;
  std::vector<double> gate_speeds(const std::vector<double>& z) const;
  // The box the variables live in.
  std::vector<double> lower_bounds() const;
  std::vector<double> upper_bounds() const;

 private:
  // The position problem of the template with the durations and gate pins of z.
  Problem position_problem(const std::vector<double>& z) const;

  TimeOptModel model_;
  Limits limits_;
  TimeOptParams params_;
  std::size_t segments_ = 0, gates_ = 0;
  std::vector<int> samples_;         // per position segment
  std::vector<std::size_t> yaw_of_;  // yaw segment of every position segment
  double snap_ref_ = 1.0;
};

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_TIMEOPT_HPP
