#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "astrum/math.hpp"
#include "flatness.hpp"

namespace astrum::minsnap {
namespace {

constexpr double kMinDuration = 0.05;
constexpr double kMinKnotSpacing = 0.05;   // route points closer than this are merged
constexpr double kSampleStep = 0.05;       // s, trajectory sampling for peaks and clearance
constexpr double kYawProfileFactor = 1.5;  // peak / mean yaw rate of a min-acceleration turn
constexpr int kStretchIterations = 12;
constexpr double kMovingStart = 0.5;  // m/s: a start faster than this is mid-flight
constexpr double kStartTransient =
    1.0;                             // s of a mid-flight start that do not count towards the limits
constexpr double kMaxStretch = 6.0;  // limit speed / acceleration stretching to this factor overall

enum class Kind { Start, Route, Entry, Gate, Exit };

struct Knot {
  Vec3 p;
  Kind kind = Kind::Route;
  std::size_t gate = 0;  // course index, Entry / Gate / Exit only
};

bool has_heading(const Knot& k) { return k.kind != Kind::Route; }
bool in_group(const Knot& k) {
  return k.kind == Kind::Entry || k.kind == Kind::Gate || k.kind == Kind::Exit;
}

// Everything the solver needs about the waypoints: the durations (consistent with the
// speeds pinned at the gates; `lambda` scales both together) and those pinned speeds.
struct Layout {
  std::vector<Knot> knots;
  std::vector<double> base;       // segment durations at lambda = 1
  std::vector<double> pin_speed;  // per knot: speed along the gate normal pinned there; 0 = free
};

std::vector<Knot> build_knots(const Course& course, std::size_t first_gate, const StartState& start,
                              const PlanParams& params, const Hazards& hazards) {
  std::vector<Knot> knots{{start.position, Kind::Start, 0}};
  auto append_route = [&](const Vec3& to, Kind kind, std::size_t gate) {
    // A leg may begin (the aircraft is where it is, say halfway through a gate) or end
    // closer to a frame than the hull radius allows; relax the radius by that deficit so
    // the leg is searched for what it can still do rather than reported as boxed in. (The
    // floor is never relaxed: a leg that cannot get above it has no route.)
    const double clearance =
        std::min(nearest_frame_distance(course, knots.back().p, course.gates.size()),
                 nearest_frame_distance(course, to, course.gates.size()));
    const double deficit = std::max(0.0, hazards.hull_radius_m - clearance);
    const Hazards relaxed{hazards.course,
                          std::max(0.0, hazards.hull_radius_m - deficit - kMinKnotSpacing),
                          hazards.floor_m};
    const auto path =
        astar_path(knots.back().p, to, deficit > 0.0 ? relaxed : hazards, params.astar);
    if (!path) {
      throw PlanError("no collision-free route to gate " + std::to_string(gate));
    }
    for (std::size_t i = 1; i + 1 < path->size(); ++i) {
      if (norm((*path)[i] - knots.back().p) > kMinKnotSpacing &&
          norm(to - (*path)[i]) > kMinKnotSpacing) {
        knots.push_back({(*path)[i], Kind::Route, 0});
      }
    }
    knots.push_back({to, kind, gate});
  };
  const std::size_t end =
      std::min(course.gates.size(), first_gate + static_cast<std::size_t>(params.max_gates));
  for (std::size_t g = first_gate; g < end; ++g) {
    const GatePose& gate = course.gates[g];
    // Already in front of the opening, closer than half the approach distance: an entry
    // point would pull the drone backwards, so go straight for the centre.
    const Vec3 rel = knots.back().p - gate.center;
    const bool close = dot(rel, gate.normal) > -0.5 * params.approach_m &&
                       std::fabs(dot(rel, gate.right)) < course.opening_m &&
                       std::fabs(dot(rel, gate.up)) < course.opening_m;
    if (!close) append_route(gate.center - gate.normal * params.approach_m, Kind::Entry, g);
    append_route(gate.center, Kind::Gate, g);
    knots.push_back({gate.center + gate.normal * params.approach_m, Kind::Exit, g});
  }
  return knots;
}

// Flat-output yaw of every knot that has one (nan for route points), unwrapped so each
// turn takes the short way round.
std::vector<double> knot_headings(const Course& course, const std::vector<Knot>& knots,
                                  double start_yaw) {
  std::vector<double> yaw(knots.size(), std::numeric_limits<double>::quiet_NaN());
  double prev = start_yaw;
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (!has_heading(knots[i])) continue;
    if (knots[i].kind == Kind::Start) {
      yaw[i] = start_yaw;
    } else {
      const double target = heading_of(course.gates[knots[i].gate].normal);
      yaw[i] = prev + astrum::wrap_pi(target - prev);
    }
    prev = yaw[i];
  }
  return yaw;
}

std::vector<std::size_t> heading_knots(const std::vector<Knot>& knots) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (has_heading(knots[i])) idx.push_back(i);
  }
  return idx;
}

// Segment times and pinned gate speeds.
//
// A *group* is a gate's entry, centre and exit points; a *leg* is what lies between two
// groups (or the start and the first group). A leg takes its length over the cruise speed
// — or longer, when the heading change between the two gates would need a yaw rate above
// the limit — and slows the gates on either side of a yaw-limited leg to the speed that
// leg actually flies (otherwise the spline would have to brake and re-accelerate between
// pinned speeds). Segments inside a group take (length / the gate's speed).
Layout allocate(const std::vector<Knot>& knots, const std::vector<double>& yaw,
                const Course& course, double start_speed, const PlanParams& params) {
  const std::size_t m = knots.size() - 1;
  const double pass = std::min(params.cruise_mps, 0.9 * params.max_speed_mps);
  std::vector<double> len(m);
  for (std::size_t i = 0; i < m; ++i) len[i] = norm(knots[i + 1].p - knots[i].p);
  auto group_segment = [&](std::size_t i) {
    return in_group(knots[i]) && in_group(knots[i + 1]) && knots[i].gate == knots[i + 1].gate;
  };

  Layout out;
  out.knots = knots;
  out.base.assign(m, 0.0);
  out.pin_speed.assign(knots.size(), 0.0);
  const double never = std::numeric_limits<double>::infinity();
  std::vector<double> cap(course.gates.size(), never);

  for (std::size_t i = 0; i < m;) {
    if (group_segment(i)) {
      ++i;
      continue;
    }
    std::size_t j = i;
    while (j < m && !group_segment(j)) ++j;  // the leg is segments [i, j), knots i .. j
    double base_total = 0.0, length = 0.0;
    std::vector<double> t(j - i);
    for (std::size_t k = i; k < j; ++k) {
      const double ve = k == 0 ? start_speed : pass;
      const double mean = std::max(0.5 * (ve + pass), 0.25 * pass);
      t[k - i] =
          std::max({len[k] / mean, std::fabs(pass - ve) / params.max_accel_mps2, kMinDuration});
      base_total += t[k - i];
      length += len[k];
    }
    double leg_time = base_total;
    if (i > 0 && j < knots.size()) {
      const double turn = std::fabs(yaw[j] - yaw[i]);
      leg_time = std::max(base_total, kYawProfileFactor * turn / params.max_yaw_rate);
      if (leg_time > base_total * 1.001) {
        const double flown = length / leg_time;
        cap[knots[i].gate] = std::min(cap[knots[i].gate], flown);
        cap[knots[j].gate] = std::min(cap[knots[j].gate], flown);
      }
    }
    for (std::size_t k = i; k < j; ++k)
      out.base[k] = std::max(t[k - i] * leg_time / base_total, kMinDuration);
    i = j;
  }

  auto gate_speed = [&](std::size_t gate) { return std::max(std::min(pass, cap[gate]), 0.5); };
  for (std::size_t i = 0; i < m; ++i) {
    if (group_segment(i)) out.base[i] = std::max(len[i] / gate_speed(knots[i].gate), kMinDuration);
  }
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (knots[i].kind == Kind::Gate || (i + 1 == knots.size() && knots[i].kind == Kind::Exit)) {
      out.pin_speed[i] = gate_speed(knots[i].gate);
    }
  }
  return out;
}

minsnap::Problem position_problem(const Course& course, const Layout& layout, double lambda,
                                  const StartState& start) {
  minsnap::Problem p;
  p.cost_order = 4;
  p.channels = 3;
  for (double t : layout.base) p.durations.push_back(t * lambda);
  for (std::size_t i = 0; i < layout.knots.size(); ++i) {
    const Vec3& pos = layout.knots[i].p;
    p.constraints.push_back({i, 0, {pos.x, pos.y, pos.z}});
    if (i == 0) {
      p.constraints.push_back({i, 1, {start.velocity.x, start.velocity.y, start.velocity.z}});
      if (start.acceleration) {
        const Vec3& a = *start.acceleration;
        p.constraints.push_back({i, 2, {a.x, a.y, a.z}});
      }
    } else if (layout.pin_speed[i] > 0.0) {
      // Temporal scaling (section V-A): all times x lambda divides every speed by lambda.
      const Vec3 v = course.gates[layout.knots[i].gate].normal * (layout.pin_speed[i] / lambda);
      p.constraints.push_back({i, 1, {v.x, v.y, v.z}});
    }
  }
  return p;
}

minsnap::Problem yaw_problem(const Layout& layout, const std::vector<std::size_t>& hk,
                             double start_rate, const std::vector<double>& yaw, double lambda) {
  minsnap::Problem p;
  p.cost_order = 2;
  p.channels = 1;
  for (std::size_t a = 0; a + 1 < hk.size(); ++a) {
    double span = 0.0;
    for (std::size_t i = hk[a]; i < hk[a + 1]; ++i) span += layout.base[i] * lambda;
    p.durations.push_back(span);
  }
  for (std::size_t a = 0; a < hk.size(); ++a) {
    p.constraints.push_back({a, 0, {yaw[hk[a]]}});
    const Kind kind = layout.knots[hk[a]].kind;
    if (kind == Kind::Start) p.constraints.push_back({a, 1, {start_rate}});
    if (kind == Kind::Gate) p.constraints.push_back({a, 1, {0.0}});
  }
  return p;
}

// minsnap::solve, with a numerically singular system reported as a planning failure.
minsnap::Trajectory solve_or_throw(const minsnap::Problem& problem, const char* what) {
  try {
    return minsnap::solve(problem);
  } catch (const std::runtime_error& e) {
    throw PlanError(std::string("cannot solve the ") + what + " trajectory: " + e.what());
  }
}

struct Peaks {
  double speed = 0.0, accel = 0.0;
  double rate = 0.0;  // largest roll / pitch body rate the reference asks for
};

// `skip_s`: the accelerations and rates of the first seconds are dictated by the state the
// plan starts from (a pinned velocity no time scaling can change), so they do not count
// towards the limits that stretch the plan.
Peaks sample_peaks(const minsnap::Trajectory& position, const minsnap::Trajectory& yaw,
                   double skip_s) {
  Peaks p;
  for (double t = 0.0; t <= position.total_time() + 1e-9; t += kSampleStep) {
    const auto v = position.value(t, 1);
    const auto a = position.value(t, 2);
    const auto j = position.value(t, 3);
    p.speed = std::max(p.speed, norm({v[0], v[1], v[2]}));
    if (t < skip_s) continue;
    p.accel = std::max(p.accel, norm({a[0], a[1], a[2]}));
    if (const auto flat = flat_state(
            {{a[0], a[1], a[2]}, {j[0], j[1], j[2]}, yaw.value(t, 0, 0), yaw.value(t, 1, 0)})) {
      p.rate = std::max(p.rate, std::hypot(flat->omega_body.x, flat->omega_body.y));
    }
  }
  return p;
}

struct Solved {
  minsnap::Trajectory position, yaw;
  Peaks peaks;
};

// Solves the position and yaw trajectories at time scale `lambda`, raising `lambda`
// (all times up, all pinned speeds down, section V-A) until the speed, acceleration and
// body-rate limits hold or lambda reaches `lambda_max`.
Solved solve_stretched(const Course& course, const Layout& layout, double& lambda,
                       double lambda_max, const std::vector<double>& yaw, const StartState& start,
                       const PlanParams& params) {
  const auto hk = heading_knots(layout.knots);
  // The pinned start velocity is not the plan's to slow down: allow at least that speed.
  const double speed_limit = std::max(params.max_speed_mps, 1.05 * norm(start.velocity));
  for (int it = 0;; ++it) {
    Solved s{solve_or_throw(position_problem(course, layout, lambda, start), "position"),
             solve_or_throw(yaw_problem(layout, hk, start.yaw_rate, yaw, lambda), "yaw"),
             {}};
    s.peaks = sample_peaks(s.position, s.yaw,
                           norm(start.velocity) > kMovingStart ? kStartTransient : 0.0);
    // Under temporal scaling speeds go as 1/lambda, accelerations as 1/lambda^2 and (jerk
    // over thrust, near hover) body rates as 1/lambda^3.
    const double excess =
        std::max({std::sqrt(s.peaks.accel / params.max_accel_mps2), s.peaks.speed / speed_limit,
                  std::cbrt(s.peaks.rate / params.max_rate_rad_s)});
    if (excess <= 1.001 || it >= kStretchIterations || lambda >= lambda_max) return s;
    lambda = std::min(lambda * excess * 1.02, lambda_max);
  }
}

// The clearance at `p` as trajectory_margin would measure it at t = 0 (hull radius for
// every frame; the crossing radius is only for a gate being flown through).
double initial_margin(const Course& course, const Vec3& p, const PlanParams& params) {
  double worst = p.z - params.floor_m;
  for (std::size_t g = 0; g < course.gates.size(); ++g)
    worst = std::min(worst, frame_distance(course, g, p) - params.hull_radius_m);
  return worst;
}

// Worst clearance of the trajectory (frames beyond the hull radius, floor), and the
// segment it occurs in.
struct Margin {
  double value = std::numeric_limits<double>::infinity();
  std::size_t segment = 0;
};

Margin trajectory_margin(const Course& course, const std::vector<Knot>& knots,
                         const minsnap::Trajectory& traj, const PlanParams& params) {
  // Plan time at which each gate's entry / exit points are reached, for the smaller
  // crossing radius while flying through it.
  const auto& times = traj.knot_times();
  const double never = std::numeric_limits<double>::infinity();
  std::vector<double> open(course.gates.size(), never), close(course.gates.size(), -never);
  std::vector<bool> has_entry(course.gates.size(), false);
  for (std::size_t i = 0; i < knots.size(); ++i) {
    const Knot& k = knots[i];
    if (k.kind == Kind::Entry) {
      open[k.gate] = times[i];
      has_entry[k.gate] = true;
    }
    if (k.kind == Kind::Gate && !has_entry[k.gate]) open[k.gate] = 0.0;
    if (k.kind == Kind::Gate || k.kind == Kind::Exit)
      close[k.gate] = std::max(close[k.gate], times[i]);
  }
  Margin m;
  std::size_t seg = 0;
  for (double t = 0.0; t <= traj.total_time() + 1e-9; t += kSampleStep) {
    while (seg + 2 < times.size() && t >= times[seg + 1]) ++seg;
    const auto v = traj.value(t, 0);
    const Vec3 p{v[0], v[1], v[2]};
    double worst = p.z - params.floor_m;
    for (std::size_t g = 0; g < course.gates.size(); ++g) {
      const double r =
          (t >= open[g] && t <= close[g]) ? params.cross_radius_m : params.hull_radius_m;
      worst = std::min(worst, frame_distance(course, g, p) - r);
    }
    if (worst < m.value) m = {worst, seg};
  }
  return m;
}

}  // namespace

FlatReference Plan::at(double t) const {
  FlatReference r;
  const double end = duration();
  const double tc = std::clamp(t, 0.0, end);
  const bool holding = t > end;
  auto vec = [&](int d) {
    const auto v = position_.value(tc, d);
    return Vec3{v[0], v[1], v[2]};
  };
  r.position = vec(0);
  if (!holding) {
    r.velocity = vec(1);
    r.acceleration = vec(2);
    r.jerk = vec(3);
    r.yaw_rate = yaw_.value(tc, 1, 0);
  }
  r.yaw = yaw_.value(tc, 0, 0);
  return r;
}

Plan plan_race(const Course& course, std::size_t first_gate, const StartState& start,
               const PlanParams& params) {
  if (first_gate >= course.gates.size()) throw PlanError("no gate left to fly");
  const Hazards hazards{&course, params.hull_radius_m, params.floor_m};
  std::vector<Knot> knots = build_knots(course, first_gate, start, params, hazards);
  const std::vector<double> yaw = knot_headings(course, knots, start.yaw);
  Layout layout = allocate(knots, yaw, course, norm(start.velocity), params);
  if (params.realloc_iterations > 0) {
    layout.base = minsnap::reallocate_durations(position_problem(course, layout, 1.0, start),
                                                params.realloc_iterations);
  }

  const double start_margin = initial_margin(course, start.position, params);
  Solved solved;
  double lambda = params.time_scale;
  const double lambda_max = params.time_scale * kMaxStretch;
  for (int pass = 0;; ++pass) {
    const std::vector<double> headings = knot_headings(course, layout.knots, start.yaw);
    solved = solve_stretched(course, layout, lambda, lambda_max, headings, start, params);
    const Margin margin = trajectory_margin(course, layout.knots, solved.position, params);
    // Starting inside the hull radius of a frame is not the spline's doing: repair only
    // what is worse than the start.
    const double tolerated = std::min(0.0, start_margin);
    if (margin.value >= tolerated - 1e-6 || pass >= params.refine_iterations) break;
    const std::size_t i = margin.segment;
    const Knot a = layout.knots[i];
    const Knot b = layout.knots[i + 1];
    if (norm(b.p - a.p) < 4.0 * kMinKnotSpacing) break;
    // Pull the offending segment back onto the (clear) polyline: a corridor keyframe at
    // its midpoint, taking half of the segment's time.
    layout.knots.insert(layout.knots.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                        Knot{(a.p + b.p) * 0.5, Kind::Route, 0});
    layout.pin_speed.insert(layout.pin_speed.begin() + static_cast<std::ptrdiff_t>(i) + 1, 0.0);
    layout.base[i] *= 0.5;
    layout.base.insert(layout.base.begin() + static_cast<std::ptrdiff_t>(i), layout.base[i]);
  }

  Plan plan;
  plan.first_gate_ = first_gate;
  plan.time_scale_ = lambda;
  plan.position_ = std::move(solved.position);
  plan.yaw_ = std::move(solved.yaw);
  plan.peak_speed_ = solved.peaks.speed;
  plan.peak_accel_ = solved.peaks.accel;
  plan.peak_rate_ = solved.peaks.rate;
  plan.min_margin_ = trajectory_margin(course, layout.knots, plan.position_, params).value;
  for (std::size_t i = 0; i < layout.knots.size(); ++i) {
    plan.waypoints_.push_back(layout.knots[i].p);
    if (layout.knots[i].kind == Kind::Gate)
      plan.gate_times_.push_back(plan.position_.knot_times()[i]);
  }
  return plan;
}

}  // namespace astrum::minsnap
