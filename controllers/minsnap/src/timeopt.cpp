#include "timeopt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "flatness.hpp"
#include "vec3.hpp"

namespace astrum::minsnap {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMinThrust = 1e-6;  // |r'' + g z_W| below this has no direction
constexpr double kMinSine = 1e-6;    // |z_B x x_C| below this is the flatness singularity
constexpr double kMaxStep = 0.5;     // largest change of any variable in one step
constexpr double kStationary = 1e-7;
constexpr int kMaxBacktracks = 30;

// ---- Forward-mode derivatives of the flatness expressions --------------------------------
// The body rate of flatness.cpp as a function of (r'', r''', psi): seven partials.

constexpr std::size_t kPartials = 7;  // a (3), jerk (3), yaw

struct Dual {
  double v = 0.0;
  std::array<double, kPartials> g{};
};

Dual variable(double value, std::size_t index) {
  Dual d;
  d.v = value;
  d.g[index] = 1.0;
  return d;
}

Dual constant(double value) {
  Dual d;
  d.v = value;
  return d;
}

// f(a) with f(a.v) = value and f'(a.v) = slope.
Dual chain(const Dual& a, double value, double slope) {
  Dual d;
  d.v = value;
  for (std::size_t i = 0; i < kPartials; ++i) d.g[i] = slope * a.g[i];
  return d;
}

Dual operator+(const Dual& a, const Dual& b) {
  Dual d;
  d.v = a.v + b.v;
  for (std::size_t i = 0; i < kPartials; ++i) d.g[i] = a.g[i] + b.g[i];
  return d;
}

Dual operator-(const Dual& a, const Dual& b) {
  Dual d;
  d.v = a.v - b.v;
  for (std::size_t i = 0; i < kPartials; ++i) d.g[i] = a.g[i] - b.g[i];
  return d;
}

Dual operator-(const Dual& a) { return chain(a, -a.v, -1.0); }

Dual operator*(const Dual& a, const Dual& b) {
  Dual d;
  d.v = a.v * b.v;
  for (std::size_t i = 0; i < kPartials; ++i) d.g[i] = a.g[i] * b.v + a.v * b.g[i];
  return d;
}

Dual operator/(const Dual& a, const Dual& b) {
  const double inv = 1.0 / b.v;
  Dual d;
  d.v = a.v * inv;
  for (std::size_t i = 0; i < kPartials; ++i) d.g[i] = (a.g[i] - d.v * b.g[i]) * inv;
  return d;
}

Dual sqrt(const Dual& a) {
  const double r = std::sqrt(a.v);
  return chain(a, r, 0.5 / r);
}
Dual sin(const Dual& a) { return chain(a, std::sin(a.v), std::cos(a.v)); }
Dual cos(const Dual& a) { return chain(a, std::cos(a.v), -std::sin(a.v)); }
Dual acos(const Dual& a) {
  // Only evaluated where the tilt constraint is active, i.e. far from straight up.
  const double c = std::clamp(a.v, -1.0 + 1e-12, 1.0 - 1e-12);
  return chain(a, std::acos(c), -1.0 / std::sqrt(1.0 - c * c));
}

using DVec = std::array<Dual, 3>;

Dual dot(const DVec& a, const DVec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
DVec cross(const DVec& a, const DVec& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
DVec operator*(const DVec& a, const Dual& s) { return {a[0] * s, a[1] * s, a[2] * s}; }
DVec operator/(const DVec& a, const Dual& s) { return {a[0] / s, a[1] / s, a[2] / s}; }
DVec operator-(const DVec& a, const DVec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// ---- The constraints at one instant -----------------------------------------------------

enum Constraint { kSpeed, kThrustMax, kThrustMin, kTilt, kRate, kYawRate, kConstraints };

struct Kinematics {
  Vec3 velocity, acceleration, jerk;
  double yaw = 0.0, yaw_rate = 0.0;
};

// The limit each constraint is normalised by: x_c = value / scale - 1 (the thrust floor
// 1 - value / scale), so x_c <= 0 satisfies it and x_c is a fraction of the limit.
struct Scales {
  double scale[kConstraints] = {};
};

Scales scales_of(const Limits& l, double margin) {
  Scales s;
  s.scale[kSpeed] = margin * l.v_max;
  s.scale[kThrustMax] = margin * l.f_max;
  s.scale[kThrustMin] = l.f_min / margin;
  s.scale[kTilt] = margin * l.tilt_max_rad;
  s.scale[kRate] = margin * l.rate_max;
  s.scale[kYawRate] = margin * l.yaw_rate_max;
  return s;
}

using Excess = std::array<double, kConstraints>;

// x_c for every constraint (-1: not evaluated or free). `transient`: thrust, tilt and
// body rate are not constrained (the start of a plan that begins in flight).
Excess excess_of(const Kinematics& k, const Scales& s, bool transient) {
  Excess x;
  x.fill(-1.0);
  x[kSpeed] = norm(k.velocity) / s.scale[kSpeed] - 1.0;
  x[kYawRate] = std::fabs(k.yaw_rate) / s.scale[kYawRate] - 1.0;
  if (transient) return x;
  const Vec3 t = k.acceleration + kUnitZ * kGravity;
  const double f = norm(t);
  x[kThrustMax] = f / s.scale[kThrustMax] - 1.0;
  if (s.scale[kThrustMin] > 0.0) x[kThrustMin] = 1.0 - f / s.scale[kThrustMin];
  if (f > kMinThrust) {
    x[kTilt] = std::acos(std::clamp(t.z / f, -1.0, 1.0)) / s.scale[kTilt] - 1.0;
  }
  if (const auto flat = flat_state({k.acceleration, k.jerk, k.yaw, k.yaw_rate})) {
    x[kRate] = std::hypot(flat->omega_body.x, flat->omega_body.y) / s.scale[kRate] - 1.0;
  }
  return x;
}

// d(penalty)/d(kinematics), weighted.
struct Bars {
  Vec3 velocity, acceleration, jerk;
  double yaw = 0.0, yaw_rate = 0.0;
};

// Adds `weight * 2 max(0, x_c) * dx_c/d(kinematics)` of every active constraint to `bars`.
void accumulate(const Kinematics& k, const Scales& s, const Excess& x, double weight, Bars& bars) {
  if (x[kSpeed] > 0.0) {
    bars.velocity += k.velocity * (weight * 2.0 * x[kSpeed] / (norm(k.velocity) * s.scale[kSpeed]));
  }
  if (x[kYawRate] > 0.0) {
    bars.yaw_rate +=
        weight * 2.0 * x[kYawRate] * (k.yaw_rate > 0.0 ? 1.0 : -1.0) / s.scale[kYawRate];
  }
  if (!(x[kThrustMax] > 0.0 || x[kThrustMin] > 0.0 || x[kTilt] > 0.0 || x[kRate] > 0.0)) return;

  // The same quantities as excess_of / flat_state, with their derivatives.
  const DVec t{variable(k.acceleration.x, 0), variable(k.acceleration.y, 1),
               variable(k.acceleration.z + kGravity, 2)};
  const DVec jerk{variable(k.jerk.x, 3), variable(k.jerk.y, 4), variable(k.jerk.z, 5)};
  const Dual yaw = variable(k.yaw, 6);
  const Dual thrust = sqrt(dot(t, t));
  const DVec z_b = t / thrust;
  auto add = [&](const Dual& q, double coefficient) {
    bars.acceleration += Vec3{q.g[0], q.g[1], q.g[2]} * coefficient;
    bars.jerk += Vec3{q.g[3], q.g[4], q.g[5]} * coefficient;
    bars.yaw += q.g[6] * coefficient;
  };
  if (x[kThrustMax] > 0.0) add(thrust, weight * 2.0 * x[kThrustMax] / s.scale[kThrustMax]);
  if (x[kThrustMin] > 0.0) add(thrust, -weight * 2.0 * x[kThrustMin] / s.scale[kThrustMin]);
  if (x[kTilt] > 0.0) add(acos(z_b[2]), weight * 2.0 * x[kTilt] / s.scale[kTilt]);
  if (x[kRate] > 0.0) {
    const DVec x_c{cos(yaw), sin(yaw), constant(0.0)};
    const DVec u = cross(z_b, x_c);
    const DVec y_b = u / sqrt(dot(u, u));
    const DVec x_b = cross(y_b, z_b);
    const DVec z_dot = (jerk - z_b * dot(z_b, jerk)) / thrust;
    const Dual wx = -dot(z_dot, y_b);
    const Dual wy = dot(z_dot, x_b);
    add(sqrt(wx * wx + wy * wy), weight * 2.0 * x[kRate] / s.scale[kRate]);
  }
}

double hinge_sum(const Excess& x) {
  double p = 0.0;
  for (double v : x) p += v > 0.0 ? v * v : 0.0;
  return p;
}

// Worst x_c at the true limits over the dense grid.
double dense_excess_of(const Trajectory& position, const Trajectory& yaw, const Limits& limits,
                       double step, double transient_s) {
  const Scales s = scales_of(limits, 1.0);
  double worst = -kInf;
  const double end = position.total_time();
  for (double t = 0.0;; t += step) {
    const double at = std::min(t, end);
    Kinematics k;
    k.velocity = {position.value(at, 1, 0), position.value(at, 1, 1), position.value(at, 1, 2)};
    k.acceleration = {position.value(at, 2, 0), position.value(at, 2, 1), position.value(at, 2, 2)};
    k.jerk = {position.value(at, 3, 0), position.value(at, 3, 1), position.value(at, 3, 2)};
    k.yaw = yaw.value(at, 0, 0);
    k.yaw_rate = yaw.value(at, 1, 0);
    for (double v : excess_of(k, s, at < transient_s)) worst = std::max(worst, v);
    if (t >= end) break;
  }
  return worst;
}

}  // namespace

TimeObjective::TimeObjective(const TimeOptModel& model, const Limits& limits,
                             const TimeOptParams& params, const std::vector<double>& durations0,
                             const std::vector<double>& gate_speeds0)
    : model_(model),
      limits_(limits),
      params_(params),
      segments_(model.position.durations.size()),
      gates_(model.gate_pins.size()) {
  const Problem& pos = model.position;
  if (pos.channels != 3 || model.yaw.channels != 1) {
    throw std::invalid_argument("timeopt: the position problem needs 3 channels, the yaw 1");
  }
  if (durations0.size() != segments_ || gate_speeds0.size() != gates_) {
    throw std::invalid_argument("timeopt: the starting point does not match the model");
  }
  if (model.yaw.durations.size() != model.yaw_spans.size() || model.yaw_spans.empty()) {
    throw std::invalid_argument("timeopt: one yaw span per yaw segment");
  }
  if (!(limits.v_max > 0.0 && limits.f_max > 0.0 && limits.tilt_max_rad > 0.0 &&
        limits.rate_max > 0.0 && limits.yaw_rate_max > 0.0 && limits.f_min >= 0.0 &&
        limits.margin > 0.0 && limits.margin <= 1.0)) {
    throw std::invalid_argument("timeopt: invalid limits");
  }
  if (params.samples_per_segment < 1 || !(params.max_sample_dt > 0.0) ||
      !(params.min_duration > 0.0) || !(params.grid_step_s > 0.0)) {
    throw std::invalid_argument("timeopt: invalid parameters");
  }
  yaw_of_.assign(segments_, 0);
  std::size_t next = 0;
  for (std::size_t q = 0; q < model.yaw_spans.size(); ++q) {
    const auto [first, last] = model.yaw_spans[q];
    if (first != next || last <= first || last > segments_) {
      throw std::invalid_argument("timeopt: yaw spans must tile the position segments");
    }
    for (std::size_t i = first; i < last; ++i) yaw_of_[i] = q;
    next = last;
  }
  if (next != segments_) throw std::invalid_argument("timeopt: yaw spans must tile the segments");
  for (const GatePin& pin : model.gate_pins) {
    if (pin.constraint >= pos.constraints.size() ||
        pos.constraints[pin.constraint].value.size() != 3 ||
        pos.constraints[pin.constraint].derivative != 1) {
      throw std::invalid_argument("timeopt: a gate pin must name a 3-channel velocity constraint");
    }
  }
  samples_.resize(segments_);
  for (std::size_t i = 0; i < segments_; ++i) {
    samples_[i] = std::max(params.samples_per_segment,
                           static_cast<int>(std::ceil(durations0[i] / params.max_sample_dt)));
  }
  snap_ref_ = solve(position_problem(variables(durations0, gate_speeds0))).cost();
  if (!(snap_ref_ > 0.0)) snap_ref_ = 1.0;
}

std::vector<double> TimeObjective::variables(const std::vector<double>& durations,
                                             const std::vector<double>& gate_speeds) const {
  std::vector<double> z;
  z.reserve(durations.size() + gate_speeds.size());
  for (double t : durations) z.push_back(std::log(t));
  for (double s : gate_speeds) z.push_back(std::log(s));
  return z;
}

std::vector<double> TimeObjective::durations(const std::vector<double>& z) const {
  std::vector<double> t(segments_);
  for (std::size_t i = 0; i < segments_; ++i) t[i] = std::exp(z[i]);
  return t;
}

std::vector<double> TimeObjective::gate_speeds(const std::vector<double>& z) const {
  std::vector<double> s(gates_);
  for (std::size_t g = 0; g < gates_; ++g) s[g] = std::exp(z[segments_ + g]);
  return s;
}

std::vector<double> TimeObjective::lower_bounds() const {
  std::vector<double> lo(segments_ + gates_);
  for (std::size_t i = 0; i < segments_; ++i) lo[i] = std::log(params_.min_duration);
  for (std::size_t g = 0; g < gates_; ++g) lo[segments_ + g] = std::log(params_.min_gate_speed);
  return lo;
}

std::vector<double> TimeObjective::upper_bounds() const {
  std::vector<double> hi(segments_ + gates_, kInf);
  for (std::size_t g = 0; g < gates_; ++g) hi[segments_ + g] = std::log(limits_.v_max);
  return hi;
}

Problem TimeObjective::position_problem(const std::vector<double>& z) const {
  Problem p = model_.position;
  p.durations = durations(z);
  for (std::size_t g = 0; g < gates_; ++g) {
    const GatePin& pin = model_.gate_pins[g];
    const double speed = std::exp(z[segments_ + g]);
    std::vector<double>& value = p.constraints[pin.constraint].value;
    for (std::size_t ch = 0; ch < 3; ++ch) value[ch] = speed * pin.direction[ch];
  }
  return p;
}

TimeObjective::Value TimeObjective::evaluate(const std::vector<double>& z, double mu, bool dense,
                                             double dense_cutoff) const {
  const Problem position = position_problem(z);
  const std::vector<double>& t_pos = position.durations;
  Problem yaw = model_.yaw;
  for (std::size_t q = 0; q < model_.yaw_spans.size(); ++q) {
    const auto [first, last] = model_.yaw_spans[q];
    yaw.durations[q] = 0.0;
    for (std::size_t i = first; i < last; ++i) yaw.durations[q] += t_pos[i];
  }
  const SolvedProblem position_solved(position);
  const SolvedProblem yaw_solved(yaw);
  const Trajectory& pt = position_solved.trajectory();
  const Trajectory& yt = yaw_solved.trajectory();
  const std::size_t yaw_segments = yaw.durations.size();
  constexpr std::size_t kK = kCoefficients;

  const Scales scales = scales_of(limits_, limits_.margin);
  // Penalty derivatives: w.r.t. the coefficients (nondimensional, [(segment * channels +
  // channel) * 8 + k]), the durations at fixed coefficients, and the position durations
  // through the yaw sample's time offset into its segment.
  std::vector<double> d_coef(segments_ * 3 * kK, 0.0), d_t_explicit(segments_, 0.0);
  std::vector<double> d_t_offset(segments_, 0.0);
  std::vector<double> d_coef_yaw(yaw_segments * kK, 0.0), d_t_explicit_yaw(yaw_segments, 0.0);

  double penalty = 0.0;
  double worst_sample =
      -kInf;                // the worst constraint at the samples, as a fraction of the true limit
  double seg_start = 0.0;   // lap time at the start of the position segment
  double span_start = 0.0;  // lap time at the start of its yaw segment
  for (std::size_t i = 0; i < segments_; ++i) {
    const std::size_t q = yaw_of_[i];
    if (i == model_.yaw_spans[q].first) span_start = seg_start;
    const double ti = t_pos[i], tq = yaw.durations[q];
    const int n = samples_[i];
    const int count = i + 1 == segments_ ? n + 1 : n;  // interior knots belong to the next segment
    const auto& cy = yt.coefficients(q, 0);
    for (int s = 0; s < count; ++s) {
      const double tau = static_cast<double>(s) / n;
      std::array<std::array<double, kK>, 4> rows;  // rows[d] = basis_row(tau, d); d = 0 unused
      std::array<std::array<double, 3>, 4> raw{};  // raw[d][channel] = rows[d] . c
      for (int d = 1; d <= 3; ++d) {
        rows[static_cast<std::size_t>(d)] = basis_row(tau, d);
        for (std::size_t ch = 0; ch < 3; ++ch) {
          const auto& c = pt.coefficients(i, ch);
          double v = 0.0;
          for (std::size_t k = 0; k < kK; ++k) v += rows[static_cast<std::size_t>(d)][k] * c[k];
          raw[static_cast<std::size_t>(d)][ch] = v;
        }
      }
      auto kin = [&](int d) {
        const double scale = std::pow(ti, -d);
        return Vec3{raw[static_cast<std::size_t>(d)][0] * scale,
                    raw[static_cast<std::size_t>(d)][1] * scale,
                    raw[static_cast<std::size_t>(d)][2] * scale};
      };
      // The yaw sample sits at the same lap time, i.e. u into its (longer) yaw segment.
      const double u = seg_start - span_start + tau * ti;
      const double tau_y = std::clamp(u / tq, 0.0, 1.0);
      const auto y0 = basis_row(tau_y, 0), y1 = basis_row(tau_y, 1), y2 = basis_row(tau_y, 2);
      double c0 = 0.0, c1 = 0.0, c2 = 0.0;  // psi, tau_y-derivatives of psi
      for (std::size_t k = 0; k < kK; ++k) {
        c0 += y0[k] * cy[k];
        c1 += y1[k] * cy[k];
        c2 += y2[k] * cy[k];
      }
      Kinematics k;
      k.velocity = kin(1);
      k.acceleration = kin(2);
      k.jerk = kin(3);
      k.yaw = c0;
      k.yaw_rate = c1 / tq;

      const Excess x = excess_of(k, scales, seg_start + tau * ti < params_.start_transient_s);
      for (int c = 0; c < kConstraints; ++c) {
        const double e = c == kThrustMin
                             ? 1.0 - (1.0 - x[static_cast<std::size_t>(c)]) / limits_.margin
                             : (1.0 + x[static_cast<std::size_t>(c)]) * limits_.margin - 1.0;
        worst_sample = std::max(worst_sample, e);
      }
      const double hinge = hinge_sum(x);
      if (!(hinge > 0.0)) continue;
      penalty += hinge;
      Bars bars;
      accumulate(k, scales, x, mu, bars);

      // value_d = ti^-d rows[d] . c, so d value_d / d c = ti^-d rows[d] and
      // d value_d / d ti = -d value_d / ti at fixed coefficients.
      const Vec3 bar[4] = {{}, bars.velocity, bars.acceleration, bars.jerk};
      const Vec3 val[4] = {{}, k.velocity, k.acceleration, k.jerk};
      for (std::size_t ch = 0; ch < 3; ++ch) {
        for (int d = 1; d <= 3; ++d) {
          const double coef = bar[d][static_cast<int>(ch)] * std::pow(ti, -d);
          for (std::size_t kk = 0; kk < kK; ++kk) {
            d_coef[(i * 3 + ch) * kK + kk] += coef * rows[static_cast<std::size_t>(d)][kk];
          }
          d_t_explicit[i] -= d * bar[d][static_cast<int>(ch)] * val[d][static_cast<int>(ch)] / ti;
        }
      }
      // psi = y0 . c; psi' = (y1 . c) / tq with tau_y = u / tq: the yaw duration moves
      // both tq and tau_y, and u itself moves with the position durations before it.
      for (std::size_t kk = 0; kk < kK; ++kk) {
        d_coef_yaw[q * kK + kk] += bars.yaw * y0[kk] + bars.yaw_rate * y1[kk] / tq;
      }
      d_t_explicit_yaw[q] +=
          bars.yaw * (-tau_y * c1 / tq) - bars.yaw_rate * (c1 + tau_y * c2) / (tq * tq);
      const double d_u = bars.yaw * c1 / tq + bars.yaw_rate * c2 / (tq * tq);
      for (std::size_t l = model_.yaw_spans[q].first; l < i; ++l) d_t_offset[l] += d_u;
      d_t_offset[i] += d_u * tau;
    }
    seg_start += ti;
  }

  Value value;
  value.time = seg_start;
  value.snap = pt.cost() / snap_ref_;
  value.penalty = penalty;
  value.objective = value.time + params_.snap_weight * value.snap + mu * penalty;
  if (dense) {
    // A sample past the true limit already settles that the trajectory violates it by at
    // least that much; the dense grid is walked only when that cannot decide the question.
    value.dense_excess =
        worst_sample > dense_cutoff
            ? worst_sample
            : dense_excess_of(pt, yt, limits_, params_.grid_step_s, params_.start_transient_s);
  }

  const double w_snap = params_.snap_weight / snap_ref_;
  const SolvedProblem::Gradient snap = position_solved.cost_gradient();
  const bool active = penalty > 0.0;
  SolvedProblem::Gradient pen, yaw_grad;
  if (active) {
    pen = position_solved.adjoint(d_coef, d_t_explicit);
    yaw_grad = yaw_solved.adjoint(d_coef_yaw, d_t_explicit_yaw);
  }
  value.gradient.assign(segments_ + gates_, 0.0);
  for (std::size_t i = 0; i < segments_; ++i) {
    double d_t = 1.0 + w_snap * snap.d_durations[i];
    if (active) d_t += pen.d_durations[i] + d_t_offset[i] + yaw_grad.d_durations[yaw_of_[i]];
    value.gradient[i] = t_pos[i] * d_t;  // dF / d log T_i
  }
  for (std::size_t g = 0; g < gates_; ++g) {
    const GatePin& pin = model_.gate_pins[g];
    double d_speed = 0.0;
    for (std::size_t ch = 0; ch < 3; ++ch) {
      double d_value = w_snap * snap.d_constraints[pin.constraint][ch];
      if (active) d_value += pen.d_constraints[pin.constraint][ch];
      d_speed += d_value * pin.direction[ch];
    }
    value.gradient[segments_ + g] += d_speed * std::exp(z[segments_ + g]);  // dF / d log s
  }
  return value;
}

namespace {

// Limited-memory BFGS inverse-Hessian product restricted to the free variables.
class History {
 public:
  explicit History(std::size_t memory) : memory_(memory) {}

  bool empty() const { return s_.empty(); }
  void clear() {
    s_.clear();
    y_.clear();
    rho_.clear();
  }

  void push(std::vector<double> s, std::vector<double> y) {
    if (s_.size() == memory_) {
      s_.erase(s_.begin());
      y_.erase(y_.begin());
      rho_.erase(rho_.begin());
    }
    rho_.push_back(1.0 / dot(s, y));
    s_.push_back(std::move(s));
    y_.push_back(std::move(y));
  }

  // -H g (two-loop recursion) with every non-free component held at zero.
  std::vector<double> direction(const std::vector<double>& g, const std::vector<bool>& free) const {
    std::vector<double> q = masked(g, free);
    std::vector<double> alpha(s_.size());
    for (std::size_t i = s_.size(); i-- > 0;) {
      alpha[i] = rho_[i] * dot(s_[i], q);
      for (std::size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * y_[i][j];
      q = masked(q, free);
    }
    if (!s_.empty()) {
      const double gamma = dot(s_.back(), y_.back()) / dot(y_.back(), y_.back());
      for (double& v : q) v *= gamma;
    }
    for (std::size_t i = 0; i < s_.size(); ++i) {
      const double beta = rho_[i] * dot(y_[i], q);
      for (std::size_t j = 0; j < q.size(); ++j) q[j] += (alpha[i] - beta) * s_[i][j];
      q = masked(q, free);
    }
    for (double& v : q) v = -v;
    return q;
  }

  static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
  }

 private:
  static std::vector<double> masked(std::vector<double> v, const std::vector<bool>& free) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!free[i]) v[i] = 0.0;
    }
    return v;
  }

  std::size_t memory_;
  std::vector<std::vector<double>> s_, y_;
  std::vector<double> rho_;
};

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return m;
}

}  // namespace

TimeOptResult optimise_times(const TimeOptModel& model, const std::vector<double>& durations0,
                             const std::vector<double>& gate_speeds0, const Limits& limits,
                             const TimeOptParams& params) {
  if (params.iterations < 0 || !(params.mu0 > 0.0) || !(params.mu_growth > 1.0) ||
      !(params.mu_max >= params.mu0) || params.lbfgs_memory < 1 || !(params.snap_weight >= 0.0) ||
      !(params.feasibility_tol >= 0.0) || !(params.min_gate_speed > 0.0)) {
    throw std::invalid_argument("timeopt: invalid parameters");
  }
  const TimeObjective objective(model, limits, params, durations0, gate_speeds0);
  const std::size_t m = objective.segments();
  const std::size_t n = m + objective.gates();

  const std::vector<double> z0 = objective.variables(durations0, gate_speeds0);
  const auto start = objective.evaluate(z0, params.mu0, true);
  std::vector<double> best_z = z0;
  double best_time = start.time;
  double best_excess = *start.dense_excess;
  int iterations = 0;

  const std::vector<double> lo = objective.lower_bounds(), hi = objective.upper_bounds();
  auto lap_time = [&](const std::vector<double>& x) {
    double total = 0.0;
    for (double t : objective.durations(x)) total += t;
    return total;
  };
  auto project = [&](std::vector<double> x) {
    for (std::size_t i = 0; i < n; ++i) x[i] = std::clamp(x[i], lo[i], hi[i]);
    return x;
  };

  // Evaluates F at x and records x when it is the best point seen so far: feasibility
  // first, then speed. Once a feasible point is known, only a faster feasible one replaces
  // it (and the dense check only runs for points that could); until then, any point that is
  // feasible or less infeasible than the best does.
  auto evaluate = [&](const std::vector<double>& x,
                      double mu) -> std::optional<TimeObjective::Value> {
    const bool best_feasible = best_excess <= params.feasibility_tol;
    const bool dense = !best_feasible || lap_time(x) < best_time;
    const double cutoff = best_feasible ? params.feasibility_tol : best_excess;
    try {
      TimeObjective::Value v = objective.evaluate(x, mu, dense, cutoff);
      if (!std::isfinite(v.objective) || !std::isfinite(max_abs(v.gradient))) return std::nullopt;
      if (dense) {
        const double excess = *v.dense_excess;
        const bool better = best_feasible
                                ? excess <= params.feasibility_tol && v.time < best_time
                                : excess <= params.feasibility_tol || excess < best_excess;
        if (better) {
          best_z = x;
          best_time = v.time;
          best_excess = excess;
        }
      }
      return v;
    } catch (const std::runtime_error&) {
      return std::nullopt;  // an unsolvable candidate is just a bad step
    }
  };

  int rounds = 1;
  for (double mu = params.mu0; mu < params.mu_max; mu *= params.mu_growth) ++rounds;

  std::vector<double> x = project(z0);
  History history(static_cast<std::size_t>(params.lbfgs_memory));
  double mu = params.mu0;
  for (int round = 0; round < rounds && iterations < params.iterations;
       ++round, mu *= params.mu_growth) {
    mu = std::min(mu, params.mu_max);
    const int budget = (params.iterations - iterations + (rounds - round) - 1) / (rounds - round);
    history.clear();
    auto current = evaluate(x, mu);
    if (!current) break;
    double f = current->objective;
    std::vector<double> g = current->gradient;
    for (int attempt = 0; attempt < budget; ++attempt) {
      std::vector<bool> free(n);
      std::vector<double> g_free(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        free[i] = !((x[i] <= lo[i] && g[i] > 0.0) || (x[i] >= hi[i] && g[i] < 0.0));
        if (free[i]) g_free[i] = g[i];
      }
      if (max_abs(g_free) < kStationary) break;
      std::vector<double> d = history.direction(g, free);
      if (!(History::dot(g_free, d) < 0.0)) {
        history.clear();
        d = history.direction(g, free);
      }
      const double reach = max_abs(d);
      double alpha =
          history.empty() ? std::min(1.0, 0.05 / reach) : std::min(1.0, kMaxStep / reach);

      bool accepted = false;
      std::vector<double> x_new;
      std::optional<TimeObjective::Value> trial;
      for (int back = 0; back < kMaxBacktracks; ++back, alpha *= 0.5) {
        std::vector<double> candidate = x;
        for (std::size_t i = 0; i < n; ++i) candidate[i] += alpha * d[i];
        candidate = project(std::move(candidate));
        double slope = 0.0;
        for (std::size_t i = 0; i < n; ++i) slope += g[i] * (candidate[i] - x[i]);
        if (!(slope < 0.0)) break;
        trial = evaluate(candidate, mu);
        if (trial && trial->objective <= f + 1e-4 * slope) {
          x_new = std::move(candidate);
          accepted = true;
          break;
        }
      }
      if (!accepted) {
        if (history.empty()) break;  // steepest descent found nothing: stationary to precision
        history.clear();
        continue;
      }
      std::vector<double> s(n), y(n);
      for (std::size_t i = 0; i < n; ++i) {
        s[i] = x_new[i] - x[i];
        y[i] = trial->gradient[i] - g[i];
      }
      if (History::dot(s, y) > 1e-10 * std::sqrt(History::dot(s, s) * History::dot(y, y))) {
        history.push(std::move(s), std::move(y));
      }
      x = std::move(x_new);
      f = trial->objective;
      g = std::move(trial->gradient);
      ++iterations;
    }
  }

  TimeOptResult result;
  result.iterations = iterations;
  result.max_violation = best_excess;
  if (best_z == z0) {
    result.durations = durations0;
    result.gate_speeds = gate_speeds0;
  } else {
    result.durations = objective.durations(best_z);
    result.gate_speeds = objective.gate_speeds(best_z);
    result.improved = true;
  }
  for (double t : result.durations) result.total_time += t;
  return result;
}

}  // namespace astrum::minsnap
