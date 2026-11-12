#include "planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "astrum/math.hpp"
#include "flatness.hpp"
#include "timeopt.hpp"

namespace astrum::minsnap {
namespace {

constexpr double kMinDuration = 0.05;
constexpr double kMinGateSpeed = 0.5;      // m/s
constexpr double kMinKnotSpacing = 0.05;   // route points closer than this are merged
constexpr double kSampleStep = 0.05;       // s, trajectory sampling for peaks and clearance
constexpr double kYawProfileFactor = 1.5;  // peak / mean yaw rate of a min-acceleration turn
constexpr int kStretchIterations = 12;
constexpr double kMovingStart = 0.5;  // m/s: a start faster than this is mid-flight
constexpr double kStartTransient =
    1.0;                             // s of a mid-flight start that do not count towards the limits
constexpr double kMaxStretch = 6.0;  // limit speed / acceleration stretching to this factor overall
constexpr double kPenaltyStart = 3.0;     // optimiser: weight of the limit penalty at first ...
constexpr double kPenaltyGrowth = 100.0;  // ... and the factor it grows by each round
// A continuation of a plan that respected the limits, searched for only a few iterations, is
// taken if it exceeds one by no more than this fraction (the optimiser's own tolerance is
// 0.5 %): the alternative is two more searches, and a re-plan has little time.
constexpr double kSeededTolerance = 0.02;
constexpr double kSeedAccel =
    3.0;  // m/s^2, speed gained or shed between gates in the turn-aware start
// The turn-aware start is tried at gate speeds of 1, kSpeedBackoff, kSpeedBackoff^2 ... times
// the estimate (kBackoffs + 1 of them) and the shortest lap that fits is kept.
constexpr double kSpeedBackoff = 0.75;
constexpr int kBackoffs = 2;
constexpr int kOptimiserRounds = 2;  // time optimisations of one plan (the second after a repair)

using Kind = KnotKind;

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
  // Just through the previous gate, closer to its plane than the exit point: that point is
  // still ahead and stays a waypoint, so the plan does not have to turn the aircraft round
  // the exit's heading from a standing start next to the frame.
  if (first_gate > 0) {
    const GatePose& through = course.gates[first_gate - 1];
    const Vec3 rel = start.position - through.center;
    const double along = dot(rel, through.normal);
    if (along > -0.2 && along < params.approach_m - 0.2 &&
        std::fabs(dot(rel, through.right)) < 0.5 * course.opening_m &&
        std::fabs(dot(rel, through.up)) < 0.5 * course.opening_m) {
      append_route(through.center + through.normal * params.approach_m, Kind::Exit, first_gate - 1);
    }
  }
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
  const double pass = std::min(params.cruise_mps, params.limit_margin * params.max_speed_mps);
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
      leg_time = std::max(base_total,
                          kYawProfileFactor * turn / (params.limit_margin * params.max_yaw_rate));
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

  auto gate_speed = [&](std::size_t gate) {
    return std::max(std::min(pass, cap[gate]), kMinGateSpeed);
  };
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

// The position problem at time scale `lambda`. `pins`, when given, receives the gate-speed
// pins in the form timeopt.hpp wants them.
minsnap::Problem position_problem(const Course& course, const Layout& layout, double lambda,
                                  const StartState& start, std::vector<GatePin>* pins = nullptr) {
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
      if (start.jerk) {
        const Vec3& j = *start.jerk;
        p.constraints.push_back({i, 3, {j.x, j.y, j.z}});
      }
    } else if (layout.pin_speed[i] > 0.0) {
      // Temporal scaling (section V-A): all times x lambda divides every speed by lambda.
      const Vec3& normal = course.gates[layout.knots[i].gate].normal;
      const Vec3 v = normal * (layout.pin_speed[i] / lambda);
      if (pins) pins->push_back({p.constraints.size(), {normal.x, normal.y, normal.z}});
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

// The position and yaw trajectories at time scale `lambda`, without their peaks.
Solved solve_trajectories(const Course& course, const Layout& layout, double lambda,
                          const std::vector<double>& yaw, const StartState& start) {
  const auto hk = heading_knots(layout.knots);
  return {solve_or_throw(position_problem(course, layout, lambda, start), "position"),
          solve_or_throw(yaw_problem(layout, hk, start.yaw_rate, yaw, lambda), "yaw"),
          {}};
}

// Solves the position and yaw trajectories at time scale `lambda`.
Solved solve_at(const Course& course, const Layout& layout, double lambda,
                const std::vector<double>& yaw, const StartState& start) {
  Solved s = solve_trajectories(course, layout, lambda, yaw, start);
  s.peaks =
      sample_peaks(s.position, s.yaw, norm(start.velocity) > kMovingStart ? kStartTransient : 0.0);
  return s;
}

// Solves at time scale `lambda`, raising `lambda` (all times up, all pinned speeds down,
// section V-A) until the speed, acceleration and body-rate limits hold (at `limit_margin`
// of the true limits) or lambda reaches `lambda_max`.
Solved solve_stretched(const Course& course, const Layout& layout, double& lambda,
                       double lambda_max, const std::vector<double>& yaw, const StartState& start,
                       const PlanParams& params) {
  // The pinned start velocity is not the plan's to slow down: allow at least that speed.
  const double speed_limit =
      std::max(params.limit_margin * params.max_speed_mps, 1.05 * norm(start.velocity));
  const double rate_limit = params.limit_margin * params.max_rate_rad_s;
  for (int it = 0;; ++it) {
    Solved s = solve_at(course, layout, lambda, yaw, start);
    // Under temporal scaling speeds go as 1/lambda, accelerations as 1/lambda^2 and (jerk
    // over thrust, near hover) body rates as 1/lambda^3.
    const double excess =
        std::max({std::sqrt(s.peaks.accel / params.max_accel_mps2), s.peaks.speed / speed_limit,
                  std::cbrt(s.peaks.rate / rate_limit)});
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

// The durations and pinned gate speeds of `layout` at time scale `factor` x its own (all
// times up, all pinned speeds down: section V-A).
void scale_layout(Layout& layout, double factor) {
  for (double& t : layout.base) t *= factor;
  for (double& v : layout.pin_speed) v /= factor;
}

// Pulls segment `i` back onto the (clear) polyline: a corridor keyframe at its midpoint,
// taking half of the segment's time. False when the segment is too short to split.
bool add_corridor_knot(Layout& layout, std::size_t i) {
  const Knot a = layout.knots[i];
  const Knot b = layout.knots[i + 1];
  if (norm(b.p - a.p) < 4.0 * kMinKnotSpacing) return false;
  layout.knots.insert(layout.knots.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                      Knot{(a.p + b.p) * 0.5, Kind::Route, 0});
  layout.pin_speed.insert(layout.pin_speed.begin() + static_cast<std::ptrdiff_t>(i) + 1, 0.0);
  layout.base[i] *= 0.5;
  layout.base.insert(layout.base.begin() + static_cast<std::ptrdiff_t>(i), layout.base[i]);
  return true;
}

// A plan stage: the waypoints and times (at the factor `lambda` the limits asked for,
// before PlanParams::time_scale) and the trajectory solved from them.
struct Stage {
  Layout layout;
  double lambda = 1.0;
  Solved solved;
  Margin margin;
};

// The heuristic plan: stretched to the limits, then to `time_scale`, with corridor
// keyframes where the trajectory comes too close to a frame. Starting inside the hull
// radius of a frame is not the spline's doing: only what is worse than `tolerated` is
// repaired.
Stage heuristic_stage(const Course& course, Layout layout, const StartState& start,
                      double tolerated, const PlanParams& params) {
  Stage stage;
  double lambda = 1.0;
  for (int pass = 0;; ++pass) {
    const std::vector<double> headings = knot_headings(course, layout.knots, start.yaw);
    stage.solved = solve_stretched(course, layout, lambda, kMaxStretch, headings, start, params);
    if (params.time_scale != 1.0) {
      stage.solved = solve_at(course, layout, lambda * params.time_scale, headings, start);
    }
    stage.margin = trajectory_margin(course, layout.knots, stage.solved.position, params);
    if (stage.margin.value >= tolerated - 1e-6 || pass >= params.refine_iterations) break;
    if (!add_corridor_knot(layout, stage.margin.segment)) break;
  }
  stage.layout = std::move(layout);
  stage.lambda = lambda;
  return stage;
}

Limits limits_of(const PlanParams& p) {
  return {p.max_speed_mps,  p.max_thrust_mps2, p.min_thrust_mps2, p.max_tilt_rad,
          p.max_rate_rad_s, p.max_yaw_rate,    p.limit_margin};
}

// The gate speeds the layout pins, in knot order (the order of TimeOptModel::gate_pins).
std::vector<double> pinned_speeds(const Layout& layout) {
  std::vector<double> speeds;
  for (std::size_t i = 1; i < layout.knots.size(); ++i) {
    if (layout.pin_speed[i] > 0.0) speeds.push_back(layout.pin_speed[i]);
  }
  return speeds;
}

void set_pinned_speeds(Layout& layout, const std::vector<double>& speeds) {
  std::size_t next = 0;
  for (std::size_t i = 1; i < layout.knots.size(); ++i) {
    if (layout.pin_speed[i] > 0.0) layout.pin_speed[i] = speeds[next++];
  }
}

TimeOptModel timeopt_model(const Course& course, const Layout& layout, const StartState& start,
                           const std::vector<double>& headings) {
  TimeOptModel model;
  model.position = position_problem(course, layout, 1.0, start, &model.gate_pins);
  const auto hk = heading_knots(layout.knots);
  model.yaw = yaw_problem(layout, hk, start.yaw_rate, headings, 1.0);
  for (std::size_t a = 0; a + 1 < hk.size(); ++a) model.yaw_spans.push_back({hk[a], hk[a + 1]});
  return model;
}

// The optimiser from the layout's times and gate speeds; nullopt when that starting
// point cannot be solved.
std::optional<TimeOptResult> run_optimiser(const Course& course, const Layout& layout,
                                           const StartState& start, const Limits& limits,
                                           const TimeOptParams& params) {
  const std::vector<double> headings = knot_headings(course, layout.knots, start.yaw);
  try {
    return optimise_times(timeopt_model(course, layout, start, headings), layout.base,
                          pinned_speeds(layout), limits, params);
  } catch (const std::runtime_error&) {
    return std::nullopt;
  }
}

// Replaces the times and gate speeds of `layout` (heuristic, time scale 1) with those the
// previous plan has for the same waypoints: between two gate waypoints (or the start) the
// plan takes what the previous plan has left, shared out as the layout shares it. False
// when nothing matched.
bool seed_from(const Plan& prev, double offset_s, const PlanParams& params, Layout& layout) {
  const auto& tags = prev.tags();
  const auto& times = prev.position().knot_times();
  auto find = [&](const Knot& k) -> std::optional<std::size_t> {
    for (std::size_t i = 0; i < tags.size(); ++i) {
      if (tags[i].kind == k.kind && tags[i].gate == k.gate) return i;
    }
    return std::nullopt;
  };
  const double scale = params.time_scale;  // the previous plan's times carry it
  bool matched = false;
  std::size_t a = 0;
  for (std::size_t b = 1; b < layout.knots.size(); ++b) {
    if (layout.knots[b].kind == Kind::Route) continue;
    const auto tb = find(layout.knots[b]);
    const auto ta = a == 0 ? std::optional<std::size_t>{} : find(layout.knots[a]);
    if (tb && (a == 0 || ta)) {
      const double from = a == 0 ? offset_s : times[*ta];
      const double total = (times[*tb] - from) / scale;
      double now = 0.0;
      for (std::size_t i = a; i < b; ++i) now += layout.base[i];
      if (total >= kMinDuration * static_cast<double>(b - a)) {
        for (std::size_t i = a; i < b; ++i) layout.base[i] *= total / now;
        matched = true;
      }
    }
    if (layout.pin_speed[b] > 0.0 && tb) {
      const auto v = prev.position().value(times[*tb], 1);
      layout.pin_speed[b] =
          std::clamp(norm({v[0], v[1], v[2]}) * scale, kMinGateSpeed, params.max_speed_mps);
    }
    a = b;
  }
  return matched;
}

// The time-optimal plan: what the optimiser makes of the heuristic stage (or of the
// previous plan's remaining times), cleared of the frames. Nullopt when it finds nothing
// faster or cannot clear the frames, so that the heuristic plan is flown.
struct Optimised {
  Layout layout;
  Solved solved;
  Margin margin;
};

// The layout's waypoints with the times and gate speeds a lap that carries speed through the
// gates would start from: every gate is crossed at turn_speed_bound() of the turn the course
// makes there, the approach segments of a gate take their length over that speed, and a leg
// between two gates is a trapezoid speed profile between the speeds at its ends (acceleration
// `max_accel_mps2`, peak speed `limit_margin * max_speed_mps`), at least as long as the yaw
// rate needs. `base` holds real durations and speeds (time scale 1). This is a starting
// point, not a plan: it may break limits, which the optimiser then removes.
Layout turn_aware_layout(const Course& course, const Layout& base, const StartState& start,
                         const PlanParams& params, double speed_factor) {
  Layout out = base;
  const std::size_t n = out.knots.size();
  const double start_speed = std::max(norm(start.velocity), kMinGateSpeed);
  std::vector<std::size_t> gates;  // knot indices of the gate centres
  for (std::size_t i = 0; i < n; ++i) {
    if (out.knots[i].kind == Kind::Gate) gates.push_back(i);
  }
  std::vector<double> speed(n, 0.0);  // per knot of a gate in the layout: that gate's speed
  std::vector<double> gate_speed(gates.size());
  for (std::size_t q = 0; q < gates.size(); ++q) {
    const Vec3& at = out.knots[gates[q]].p;
    const Vec3 from = q > 0 ? out.knots[gates[q - 1]].p : out.knots[0].p;
    const bool last = q + 1 == gates.size();
    const Vec3 to = last ? at + (at - from) : out.knots[gates[q + 1]].p;
    const Vec3 in = unit_or(at - from, kUnitZ), leave = unit_or(to - at, in);
    const double reach = last ? norm(at - from) : std::min(norm(at - from), norm(to - at));
    gate_speed[q] =
        speed_factor *
        turn_speed_bound(std::acos(std::clamp(dot(in, leave), -1.0, 1.0)), reach, params);
  }
  // Speed can only be gained and shed at a modest rate (kSeedAccel: well below what the
  // tilt allows, as the thrust vector needs time to tilt at the body-rate limit): no gate is
  // faster than the one before it plus what the leg in between gives, nor than the one
  // after it plus what that leg can shed, nor than the start plus the first leg.
  const auto gain = [](double v, double length) {
    return std::sqrt(v * v + 2.0 * kSeedAccel * length);
  };
  for (std::size_t q = 0; q < gates.size(); ++q) {
    const Vec3& from = q > 0 ? out.knots[gates[q - 1]].p : out.knots[0].p;
    gate_speed[q] = std::min(gate_speed[q], gain(q > 0 ? gate_speed[q - 1] : start_speed,
                                                 norm(out.knots[gates[q]].p - from)));
  }
  for (std::size_t q = gates.size(); q-- > 1;) {
    gate_speed[q - 1] =
        std::min(gate_speed[q - 1],
                 gain(gate_speed[q], norm(out.knots[gates[q]].p - out.knots[gates[q - 1]].p)));
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!in_group(out.knots[i])) continue;
    speed[i] = start_speed;  // the exit of the gate just flown through
    for (std::size_t q = 0; q < gates.size(); ++q) {
      if (out.knots[gates[q]].gate == out.knots[i].gate) speed[i] = gate_speed[q];
    }
    if (out.pin_speed[i] > 0.0) out.pin_speed[i] = speed[i];
  }
  speed[0] = start_speed;

  const std::vector<double> yaw = knot_headings(course, out.knots, start.yaw);
  const double peak = params.limit_margin * params.max_speed_mps;
  const double accel = params.max_accel_mps2;
  std::size_t a = 0;
  for (std::size_t b = 1; b < n; ++b) {
    if (out.knots[b].kind == Kind::Route) continue;
    double length = 0.0;
    for (std::size_t i = a; i < b; ++i) length += norm(out.knots[i + 1].p - out.knots[i].p);
    const bool group = a > 0 && in_group(out.knots[a]) && in_group(out.knots[b]) &&
                       out.knots[a].gate == out.knots[b].gate;
    double time;
    if (group) {
      time = length / speed[a];
    } else {
      const double va = speed[a], vb = speed[b], vp = std::max({peak, va, vb});
      const double ramp = (vp * vp - va * va) / (2.0 * accel) + (vp * vp - vb * vb) / (2.0 * accel);
      if (ramp <= length) {
        time = (2.0 * vp - va - vb) / accel + (length - ramp) / vp;
      } else {
        const double top = std::sqrt((2.0 * accel * length + va * va + vb * vb) / 2.0);
        time = top > std::max(va, vb) ? (2.0 * top - va - vb) / accel : 2.0 * length / (va + vb);
      }
      if (a > 0) {
        time = std::max(time, kYawProfileFactor * std::fabs(yaw[b] - yaw[a]) /
                                  (params.limit_margin * params.max_yaw_rate));
      }
    }
    for (std::size_t i = a; i < b; ++i) {
      const double share = length > 0.0 ? norm(out.knots[i + 1].p - out.knots[i].p) / length
                                        : 1.0 / static_cast<double>(b - a);
      out.base[i] = std::max(time * share, kMinDuration);
    }
    a = b;
  }
  return out;
}

// Per segment, the factor its time must grow by to bring the sampled |r''|, speed and roll /
// pitch body rate within their limits (those of the heuristic stretch: `max_accel_mps2`,
// speed and rate at `limit_margin`), the largest over the segment.
std::vector<double> segment_excess(const Solved& s, const StartState& start,
                                   const PlanParams& params) {
  const double speed_limit =
      std::max(params.limit_margin * params.max_speed_mps, 1.05 * norm(start.velocity));
  const double rate_limit = params.limit_margin * params.max_rate_rad_s;
  const auto& times = s.position.knot_times();
  std::vector<double> worst(s.position.segments(), 0.0);
  std::size_t seg = 0;
  for (double t = 0.0; t <= s.position.total_time() + 1e-9; t += kSampleStep) {
    while (seg + 1 < worst.size() && t >= times[seg + 1]) ++seg;
    const auto v = s.position.value(t, 1);
    const auto a = s.position.value(t, 2);
    const auto j = s.position.value(t, 3);
    // The factor the segment's time must grow by, as for temporal scaling: speeds go as
    // 1 / T, accelerations as 1 / T^2 and body rates (jerk over thrust) as 1 / T^3.
    double e = std::max(std::sqrt(norm({a[0], a[1], a[2]}) / params.max_accel_mps2),
                        norm({v[0], v[1], v[2]}) / speed_limit);
    if (const auto flat = flat_state(
            {{a[0], a[1], a[2]}, {j[0], j[1], j[2]}, s.yaw.value(t, 0, 0), s.yaw.value(t, 1, 0)})) {
      e = std::max(e, std::cbrt(std::hypot(flat->omega_body.x, flat->omega_body.y) / rate_limit));
    }
    worst[seg] = std::max(worst[seg], e);
  }
  return worst;
}

// The turn-aware layout of `base` with each of its legs (a run of segments that are not
// inside a gate's own group; the group segments and the pinned speeds stay) lengthened by
// its own factor until the sampled peaks over it and the group segments beside it respect
// the limits, for gate speeds at 1, kSpeedBackoff, kSpeedBackoff^2 of their estimate: of
// those that fit (no leg needs more than kMaxStretch), the one with the shortest lap, since
// where the estimate is too optimistic, lengthening the legs costs more than lower speeds.
// Nullopt when none fits.
std::optional<Layout> stretched_turn_aware_layout(const Course& course, const Layout& base,
                                                  const StartState& start,
                                                  const PlanParams& params) {
  const std::vector<double> headings = knot_headings(course, base.knots, start.yaw);
  const std::size_t segments = base.base.size();
  std::optional<Layout> best;
  double best_time = 0.0;
  double speed_factor = 1.0;
  try {
    for (int attempt = 0; attempt <= kBackoffs; ++attempt, speed_factor *= kSpeedBackoff) {
      const Layout aware = turn_aware_layout(course, base, start, params, speed_factor);
      std::vector<bool> group(segments);
      for (std::size_t i = 0; i < segments; ++i) {
        group[i] = in_group(aware.knots[i]) && in_group(aware.knots[i + 1]) &&
                   aware.knots[i].gate == aware.knots[i + 1].gate;
      }
      std::vector<double> factor(segments, 1.0);  // per leg segment; a leg's segments share one
      bool fits = false, hopeless = false;
      for (int it = 0; it < kStretchIterations && !fits && !hopeless; ++it) {
        Layout trial = aware;
        for (std::size_t i = 0; i < segments; ++i) trial.base[i] *= factor[i];
        const std::vector<double> excess =
            segment_excess(solve_trajectories(course, trial, 1.0, headings, start), start, params);
        fits = true;
        for (std::size_t first = 0; first < segments;) {
          if (group[first]) {
            ++first;
            continue;
          }
          std::size_t last = first;
          while (last < segments && !group[last]) ++last;  // the leg is [first, last)
          double e = 0.0;
          for (std::size_t i = first > 0 ? first - 1 : first; i < std::min(last + 1, segments);
               ++i) {
            e = std::max(e, excess[i]);
          }
          if (e > 1.001) {
            fits = false;
            for (std::size_t i = first; i < last; ++i) {
              factor[i] *= std::min(e * 1.02, 2.0);
              if (factor[i] > kMaxStretch) hopeless = true;
            }
          }
          first = last;
        }
        if (fits) {
          double time = 0.0;
          for (double t : trial.base) time += t;
          if (!best || time < best_time) {
            best = std::move(trial);
            best_time = time;
          }
        }
      }
    }
  } catch (const PlanError&) {
    return std::nullopt;
  }
  return best;
}

std::optional<Optimised> optimise_stage(const Course& course, const Stage& heuristic,
                                        const StartState& start, const PlanParams& params,
                                        const PlanSeed& seed, double tolerated, PlanInfo& info) {
  const Limits limits = limits_of(params);
  TimeOptParams topt;
  topt.iterations = seed.previous ? params.replan_iterations : params.opt_iterations;
  topt.snap_weight = params.snap_weight;
  topt.mu0 = kPenaltyStart;
  topt.mu_growth = kPenaltyGrowth;
  const auto feasible = [&](const TimeOptResult& r) {
    return r.max_violation <= topt.feasibility_tol;
  };

  Layout from_heuristic = heuristic.layout;
  scale_layout(from_heuristic, heuristic.lambda);
  double heuristic_time = 0.0;
  for (double t : from_heuristic.base) heuristic_time += t;
  // A start that carries speed through the gates (turn_aware_layout, with its legs lengthened
  // until it respects the limits).
  std::optional<Layout> aware = stretched_turn_aware_layout(course, from_heuristic, start, params);
  double aware_time = 0.0;
  if (aware) {
    for (double t : aware->base) aware_time += t;
  }
  const bool aware_ok = aware.has_value() && aware_time < heuristic_time;

  // Where the search may start, best first: the previous plan's remaining times (what it
  // does not cover from the turn-aware layout), that layout, the heuristic plan. A start is
  // taken when the result respects the limits and is something to fly: the seeded plan as
  // it is, the turn-aware one only if it beats the heuristic plan, the heuristic one only if
  // the optimiser improved it.
  struct Start {
    Layout layout;
    bool seeded;
    bool heuristic;
  };
  std::vector<Start> starts;
  if (seed.previous) {
    Layout seeded = aware_ok ? *aware : from_heuristic;
    if (seed_from(*seed.previous, seed.time_s, params, seeded)) {
      starts.push_back({std::move(seeded), true, false});
    }
  }
  if (aware_ok) starts.push_back({std::move(*aware), false, false});
  starts.push_back({from_heuristic, false, true});

  Layout layout;
  std::optional<TimeOptResult> result;
  for (Start& candidate : starts) {
    // A re-plan has little time: what is tried after its seeded start fails gets half.
    if (seed.previous && !candidate.seeded) topt.iterations = std::max(1, topt.iterations / 2);
    result = run_optimiser(course, candidate.layout, start, limits, topt);
    if (!result) continue;
    info.iterations += result->iterations;
    const bool wanted =
        candidate.seeded ||
        (candidate.heuristic ? result->improved : result->total_time < heuristic_time);
    if (wanted &&
        (candidate.seeded ? result->max_violation <= kSeededTolerance : feasible(*result))) {
      layout = std::move(candidate.layout);
      info.warm_start = candidate.seeded;
      break;
    }
    result.reset();
  }
  if (!result) return std::nullopt;

  try {
    for (int round = 0; round < kOptimiserRounds; ++round) {
      if (round > 0) {
        result = run_optimiser(course, layout, start, limits, topt);
        if (!result || !feasible(*result)) return std::nullopt;
        info.iterations += result->iterations;
      }
      layout.base = result->durations;
      set_pinned_speeds(layout, result->gate_speeds);

      Layout scaled = layout;
      scale_layout(scaled, params.time_scale);
      std::vector<double> headings = knot_headings(course, layout.knots, start.yaw);
      Solved solved = solve_at(course, scaled, 1.0, headings, start);
      Margin margin = trajectory_margin(course, layout.knots, solved.position, params);
      if (margin.value >= tolerated - 1e-6)
        return Optimised{std::move(layout), std::move(solved), margin};
      if (round + 1 == kOptimiserRounds) return std::nullopt;
      // Too close to a frame: keyframes as for the heuristic plan, then optimise again.
      for (int pass = 0; margin.value < tolerated - 1e-6; ++pass) {
        if (pass >= params.refine_iterations || !add_corridor_knot(layout, margin.segment)) {
          return std::nullopt;
        }
        scaled = layout;
        scale_layout(scaled, params.time_scale);
        headings = knot_headings(course, layout.knots, start.yaw);
        solved = solve_at(course, scaled, 1.0, headings, start);
        margin = trajectory_margin(course, layout.knots, solved.position, params);
      }
    }
  } catch (const PlanError&) {
    return std::nullopt;
  }
  return std::nullopt;
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

double turn_speed_bound(double turn_rad, double reach_m, const PlanParams& params) {
  const double m = params.limit_margin;
  const double ceiling = m * params.max_speed_mps;
  const double turn = std::clamp(std::fabs(turn_rad), 0.0, 0.95 * 3.14159265358979323846);
  if (turn < 1e-3) return ceiling;
  const double half_reach = 0.5 * reach_m;
  const double radius = half_reach / std::tan(0.5 * turn);
  const double lateral = std::min(
      kGravity * std::tan(m * params.max_tilt_rad),
      std::sqrt(std::max(0.0, std::pow(m * params.max_thrust_mps2, 2) - kGravity * kGravity)));
  double bound = std::min(ceiling, std::sqrt(lateral * radius));
  // v * atan(v^2 / (g R)) / w grows with v: the largest v that rolls in within half the reach.
  const double rate = m * params.max_rate_rad_s;
  const auto roll_in = [&](double v) { return v * std::atan(v * v / (kGravity * radius)) / rate; };
  if (roll_in(bound) > half_reach) {
    double lo = 0.0, hi = bound;
    for (int i = 0; i < 40; ++i) {
      const double mid = 0.5 * (lo + hi);
      (roll_in(mid) > half_reach ? hi : lo) = mid;
    }
    bound = lo;
  }
  return std::max(bound, kMinGateSpeed);
}

const char* to_string(PlannerKind kind) {
  return kind == PlannerKind::TimeOptimal ? "time-optimal" : "heuristic";
}

std::optional<PlannerKind> planner_from_string(std::string_view name) {
  for (const PlannerKind kind : {PlannerKind::TimeOptimal, PlannerKind::Heuristic}) {
    if (name == to_string(kind)) return kind;
  }
  return std::nullopt;
}

// Builds a Plan from the planner's internals.
struct PlanAccess {
  static Plan make(std::size_t first_gate, const Layout& layout, double time_scale, Solved solved,
                   double min_margin) {
    Plan plan;
    plan.first_gate_ = first_gate;
    plan.time_scale_ = time_scale;
    plan.position_ = std::move(solved.position);
    plan.yaw_ = std::move(solved.yaw);
    plan.peak_speed_ = solved.peaks.speed;
    plan.peak_accel_ = solved.peaks.accel;
    plan.peak_rate_ = solved.peaks.rate;
    plan.min_margin_ = min_margin;
    for (std::size_t i = 0; i < layout.knots.size(); ++i) {
      plan.waypoints_.push_back(layout.knots[i].p);
      plan.tags_.push_back({layout.knots[i].kind, layout.knots[i].gate});
      if (layout.knots[i].kind == Kind::Gate)
        plan.gate_times_.push_back(plan.position_.knot_times()[i]);
    }
    return plan;
  }
  static PlanInfo& info(Plan& plan) { return plan.info_; }
};

Plan plan_race(const Course& course, std::size_t first_gate, const StartState& start,
               const PlanParams& params, const PlanSeed& seed) {
  const auto begin = std::chrono::steady_clock::now();
  if (first_gate >= course.gates.size()) throw PlanError("no gate left to fly");
  const Hazards hazards{&course, params.hull_radius_m, params.floor_m};
  std::vector<Knot> knots = build_knots(course, first_gate, start, params, hazards);
  const std::vector<double> yaw = knot_headings(course, knots, start.yaw);
  Layout layout = allocate(knots, yaw, course, norm(start.velocity), params);
  if (params.realloc_iterations > 0) {
    layout.base = minsnap::reallocate_durations(position_problem(course, layout, 1.0, start),
                                                params.realloc_iterations);
  }

  const double tolerated = std::min(0.0, initial_margin(course, start.position, params));
  Stage heuristic = heuristic_stage(course, std::move(layout), start, tolerated, params);

  PlanInfo info;
  info.planner = params.planner;
  info.heuristic_duration_s = heuristic.solved.position.total_time();
  std::optional<Optimised> optimised;
  if (params.planner == PlannerKind::TimeOptimal) {
    optimised = optimise_stage(course, heuristic, start, params, seed, tolerated, info);
  }
  info.improved = optimised.has_value();

  Plan plan = optimised ? PlanAccess::make(first_gate, optimised->layout, params.time_scale,
                                           std::move(optimised->solved), optimised->margin.value)
                        : PlanAccess::make(first_gate, heuristic.layout,
                                           heuristic.lambda * params.time_scale,
                                           std::move(heuristic.solved), heuristic.margin.value);
  info.compute_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  PlanAccess::info(plan) = info;
  return plan;
}

}  // namespace astrum::minsnap
