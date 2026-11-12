#include "race_law.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "frames.hpp"

namespace astrum::minsnap {
namespace {

Vec3 vec(const Enu& v) { return {v.e, v.n, v.u}; }

std::optional<std::string> positive(const char* flag, double v) {
  if (!(v > 0.0) || !std::isfinite(v)) return std::string(flag) + " must be > 0";
  return std::nullopt;
}

}  // namespace

std::optional<std::string> RaceLawParams::validate() const {
  struct Check {
    const char* flag;
    double value;
  };
  for (const Check& c : {Check{"--cruise-mps", plan.cruise_mps},
                         {"--time-scale", plan.time_scale},
                         {"--max-speed-mps", plan.max_speed_mps},
                         {"--max-accel-mps2", plan.max_accel_mps2},
                         {"--max-rate", plan.max_rate_rad_s},
                         {"--max-yaw-rate", plan.max_yaw_rate},
                         {"--approach-m", plan.approach_m},
                         {"--hull-radius-m", plan.hull_radius_m},
                         {"--cross-radius-m", plan.cross_radius_m},
                         {"--cell-m", plan.astar.cell_m},
                         {"--kp", gains.kp},
                         {"--kv", gains.kv},
                         {"--k-att", gains.k_att_xy},
                         {"--k-yaw", gains.k_att_z},
                         {"--hover-collective", demands.hover_collective},
                         {"--thrust-exponent", demands.thrust_exponent},
                         {"--rate-gain", demands.rate_gain},
                         {"--replan-error-m", replan_error_m},
                         {"--replan-slack-s", replan_slack_s}}) {
    if (auto e = positive(c.flag, c.value)) return e;
  }
  if (!(demands.hover_collective < 1.0)) return "--hover-collective must be < 1";
  if (!(gains.ki >= 0.0) || !(gains.integral_limit >= 0.0))
    return "--ki and --integral-limit must be >= 0";
  if (!(gains.max_tilt_rad > 0.0 && gains.max_tilt_rad < 1.5))
    return "--max-tilt-deg must be in (0, 85.9)";
  if (plan.cross_radius_m > plan.hull_radius_m)
    return "--cross-radius-m must not exceed --hull-radius-m";
  if (plan.max_gates < 2) return "--horizon-gates must be >= 2";
  if (plan.realloc_iterations < 0 || plan.refine_iterations < 0)
    return "iteration counts must be >= 0";
  return std::nullopt;
}

Course course_from_wire(const RaceCourse& wire) {
  Course c;
  c.opening_m = wire.opening_m;
  c.bar_m = wire.bar_m;
  c.top_bar_m = wire.top_bar_m;
  std::vector<RaceGate> sorted = wire.gates;
  std::sort(sorted.begin(), sorted.end(),
            [](const RaceGate& a, const RaceGate& b) { return a.index < b.index; });
  for (const RaceGate& g : sorted)
    c.gates.push_back({vec(g.center), vec(g.normal), vec(g.right), vec(g.up)});
  return c;
}

RaceLaw::RaceLaw(const RaceLawParams& params) : params_(params), tracker_(params.gains) {}

StartState RaceLaw::measured_start(const VehicleState& state) const {
  return {{state.e, state.n, state.u},
          {state.vel_enu_e, state.vel_enu_n, state.vel_enu_u},
          flat_yaw_from_heading(state.yaw),
          std::nullopt,
          0.0};
}

void RaceLaw::make_plan(const VehicleState& state, const StartState& start) {
  const std::size_t pending = race_ ? race_->pending_gate : 0;
  plan_ = plan_race(*course_, pending, start, params_.plan);
  plan_start_s_ = state.sim_time_s;
  last_plan_s_ = state.sim_time_s;
  ++plans_made_;
  tracker_.reset();
}

bool RaceLaw::window_ends_soon() const {
  // A plan covers at most --horizon-gates gates (planning cost grows with the cube of
  // that); when the aircraft is within a quarter of it from the plan's last gate, the
  // plan is extended.
  const std::size_t end = plan_->first_gate() + plan_->gate_count();
  const std::size_t pending = race_ ? static_cast<std::size_t>(race_->pending_gate) : 0;
  const std::size_t lookahead = std::max<std::size_t>(1, plan_->gate_count() / 4);
  return end < course_->gates.size() && pending + lookahead >= end;
}

bool RaceLaw::needs_replan(const VehicleState& state, const FlatReference& ref) const {
  if (state.sim_time_s - last_plan_s_ < params_.replan_min_interval_s) return false;
  const Vec3 error = Vec3{state.e, state.n, state.u} - ref.position;
  if (norm(error) > params_.replan_error_m) return true;
  // The gate the race is waiting for was due some time ago: it was missed (or flown
  // past); start over from where the aircraft is.
  const std::size_t k = race_ ? static_cast<std::size_t>(race_->pending_gate) : 0;
  if (k < plan_->first_gate()) return true;
  const std::size_t rel = k - plan_->first_gate();
  const double t = state.sim_time_s - plan_start_s_;
  return rel < plan_->gate_count() && t > plan_->gate_times()[rel] + params_.replan_slack_s;
}

LawResult RaceLaw::step(const StepInput& input) {
  const VehicleState& s = input.state;
  // The course is republished about once a second: only a different one restarts planning.
  if (input.race_course && (!wire_course_ || !(*wire_course_ == *input.race_course))) {
    wire_course_ = *input.race_course;
    course_ = course_from_wire(*wire_course_);
    plan_.reset();
  }
  if (input.race_state) race_ = *input.race_state;

  const Estimate est{{s.e, s.n, s.u},
                     {s.vel_enu_e, s.vel_enu_n, s.vel_enu_u},
                     body_to_world(s.roll, s.pitch, s.yaw)};

  const bool racing = course_ && !course_->gates.empty() && !(race_ && race_->finished) &&
                      (!race_ || race_->pending_gate < course_->gates.size());
  FlatReference ref;
  if (racing) {
    hold_position_.reset();
    try {
      if (!plan_) make_plan(s, measured_start(s));
      ref = plan_->at(s.sim_time_s - plan_start_s_);
      const bool lost = needs_replan(s, ref);
      const bool extend = !lost && window_ends_soon() &&
                          s.sim_time_s - last_plan_s_ >= params_.replan_min_interval_s;
      if (lost || extend) {
        try {
          // Extending continues from the reference (the aircraft is on it); a re-plan
          // after a miss or a large error starts from where the aircraft actually is.
          make_plan(s, lost ? measured_start(s)
                            : StartState{ref.position, ref.velocity, ref.yaw, ref.acceleration,
                                         ref.yaw_rate});
          ref = plan_->at(0.0);
        } catch (const PlanError&) {
          // Keep flying the plan we have (its tracking error is what asked for a new one);
          // try again after the usual interval.
          last_plan_s_ = s.sim_time_s;
        }
      }
    } catch (const PlanError& e) {
      return Stop{std::string("cannot plan the course: ") + e.what()};
    }
  } else {
    if (!hold_position_) {
      hold_position_ = Vec3{s.e, s.n, s.u};
      hold_yaw_ = flat_yaw_from_heading(s.yaw);
    }
    ref.position = *hold_position_;
    ref.yaw = hold_yaw_;
  }

  const double dt = input.control_period_s > 0.0 ? input.control_period_s : 0.01;
  const Setpoint setpoint = tracker_.update(est, ref, dt);
  return Action{to_demands(setpoint, params_.demands)};
}

}  // namespace astrum::minsnap
