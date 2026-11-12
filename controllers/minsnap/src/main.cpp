// astrum-minsnap-controller: flies a multirotor through a racing course (crates/astrum-race)
// with a minimum-snap trajectory and a geometric tracking controller — see race_law.hpp.
// Run with --help for every flag and its ASTRUM_* environment fallback.

#include <memory>
#include <stdexcept>
#include <string>

#include "astrum/log.hpp"
#include "astrum/math.hpp"
#include "astrum/runtime.hpp"
#include "race_law.hpp"

namespace {

using astrum::minsnap::RaceLaw;
using astrum::minsnap::RaceLawParams;

// The law's parameters live behind a shared pointer: the option table writes into them
// when the flags are parsed, before the entry's `make` reads them.
struct Flags {
  RaceLawParams params;
  std::string planner = astrum::minsnap::to_string(RaceLawParams{}.plan.planner);
  double max_tilt_deg = astrum::rad_to_deg(RaceLawParams{}.plan.max_tilt_rad);
};

void add_options(astrum::Options& o, Flags& f) {
  RaceLawParams& p = f.params;
  o.add("--planner", "name",
        "time-optimal (fastest lap within the limits) or heuristic (cruise speed, stretched to "
        "the limits)",
        "ASTRUM_MINSNAP_PLANNER", &f.planner)
      .add("--cruise-mps", "m/s", "heuristic planner: speed through the gates",
           "ASTRUM_MINSNAP_CRUISE_MPS", &p.plan.cruise_mps)
      .add("--time-scale", "x", "stretch all segment times by this factor (> 1 = slower, safer)",
           "ASTRUM_MINSNAP_TIME_SCALE", &p.plan.time_scale)
      .add("--max-speed-mps", "m/s", "speed limit", "ASTRUM_MINSNAP_MAX_SPEED_MPS",
           &p.plan.max_speed_mps)
      .add("--max-thrust-mps2", "m/s^2", "thrust limit per unit mass, |acceleration + gravity|",
           "ASTRUM_MINSNAP_MAX_THRUST_MPS2", &p.plan.max_thrust_mps2)
      .add("--min-thrust-mps2", "m/s^2", "thrust floor per unit mass (0 = none)",
           "ASTRUM_MINSNAP_MIN_THRUST_MPS2", &p.plan.min_thrust_mps2)
      .add("--max-tilt-deg", "deg",
           "largest tilt of the thrust from vertical the plan asks for (the tracker may tilt "
           "5 deg further)",
           "ASTRUM_MINSNAP_MAX_TILT_DEG", &f.max_tilt_deg)
      .add("--max-rate", "rad/s", "roll / pitch body rate limit (the demand saturates at 0.75)",
           "ASTRUM_MINSNAP_MAX_RATE", &p.plan.max_rate_rad_s)
      .add("--max-yaw-rate", "rad/s", "yaw rate limit (heuristic: mean rate between two headings)",
           "ASTRUM_MINSNAP_MAX_YAW_RATE", &p.plan.max_yaw_rate)
      .add("--limit-margin", "0..1", "fraction of each limit the plan is allowed to use",
           "ASTRUM_MINSNAP_LIMIT_MARGIN", &p.plan.limit_margin)
      .add("--max-accel-mps2", "m/s^2", "heuristic planner: peak acceleration |r''|",
           "ASTRUM_MINSNAP_MAX_ACCEL_MPS2", &p.plan.max_accel_mps2)
      .add("--opt-iterations", "n", "time-optimal planner: optimiser iterations of a first plan",
           "ASTRUM_MINSNAP_OPT_ITERATIONS", &p.plan.opt_iterations)
      .add("--replan-iterations", "n",
           "time-optimal planner: optimiser iterations of an extension or re-plan",
           "ASTRUM_MINSNAP_REPLAN_ITERATIONS", &p.plan.replan_iterations)
      .add("--snap-weight", "w", "time-optimal planner: weight of the snap cost against the time",
           "ASTRUM_MINSNAP_SNAP_WEIGHT", &p.plan.snap_weight)
      .add("--approach-m", "m", "entry / exit waypoint distance from the gate plane",
           "ASTRUM_MINSNAP_APPROACH_M", &p.plan.approach_m)
      .add("--hull-radius-m", "m",
           "clearance kept to frames not being crossed (F450 worst crab: 0.353)",
           "ASTRUM_MINSNAP_HULL_RADIUS_M", &p.plan.hull_radius_m)
      .add("--cross-radius-m", "m",
           "clearance kept to the frame being crossed (F450 square on: 0.285)",
           "ASTRUM_MINSNAP_CROSS_RADIUS_M", &p.plan.cross_radius_m)
      .add("--floor-m", "m", "lowest centre-of-mass height (ENU z)", "ASTRUM_MINSNAP_FLOOR_M",
           &p.plan.floor_m)
      .add("--cell-m", "m", "A* grid cell size", "ASTRUM_MINSNAP_CELL_M", &p.plan.astar.cell_m)
      .add("--horizon-gates", "n",
           "gates covered by one plan (extended as the aircraft nears its last)",
           "ASTRUM_MINSNAP_HORIZON_GATES", &p.plan.max_gates)
      .add("--realloc-iters", "n", "segment-time optimisation iterations (0 = off)",
           "ASTRUM_MINSNAP_REALLOC_ITERS", &p.plan.realloc_iterations)
      .add("--refine-iters", "n", "corridor repair passes", "ASTRUM_MINSNAP_REFINE_ITERS",
           &p.plan.refine_iterations)
      .add("--kp", "1/s^2", "position gain", "ASTRUM_MINSNAP_KP", &p.gains.kp)
      .add("--kv", "1/s", "velocity gain", "ASTRUM_MINSNAP_KV", &p.gains.kv)
      .add("--ki", "1/s^3", "position integral gain", "ASTRUM_MINSNAP_KI", &p.gains.ki)
      .add("--integral-limit", "m/s^2", "largest acceleration the integral term may command",
           "ASTRUM_MINSNAP_INTEGRAL_LIMIT", &p.gains.integral_limit)
      .add("--k-att", "1/s", "roll / pitch attitude error -> body rate gain",
           "ASTRUM_MINSNAP_K_ATT", &p.gains.k_att_xy)
      .add("--k-yaw", "1/s", "yaw attitude error -> body rate gain", "ASTRUM_MINSNAP_K_YAW",
           &p.gains.k_att_z)
      .add("--max-collective", "0..1",
           "collective is never commanded above this (the F450 rate loop is unstable near 0.7)",
           "ASTRUM_MINSNAP_MAX_COLLECTIVE", &p.demands.max_collective)
      .add("--hover-collective", "0..1", "collective that holds altitude",
           "ASTRUM_MINSNAP_HOVER_COLLECTIVE", &p.demands.hover_collective)
      .add("--thrust-exponent", "n", "thrust ~ collective^n around hover",
           "ASTRUM_MINSNAP_THRUST_EXPONENT", &p.demands.thrust_exponent)
      .add("--rate-gain", "rad/s", "body rate of a full-scale rate demand",
           "ASTRUM_MINSNAP_RATE_GAIN", &p.demands.rate_gain)
      .add("--replan-error-m", "m", "position error that forces a re-plan",
           "ASTRUM_MINSNAP_REPLAN_ERROR_M", &p.replan_error_m)
      .add("--replan-slack-s", "s", "grace after a gate's planned time before re-planning",
           "ASTRUM_MINSNAP_REPLAN_SLACK_S", &p.replan_slack_s);
}

}  // namespace

int main(int argc, char** argv) {
  auto flags = std::make_shared<Flags>();
  const astrum::LawEntry laws[] = {
      {"minsnap", "minimum-snap trajectory through the gates + geometric tracking (multirotor)",
       [flags](astrum::Options& o) { add_options(o, *flags); },
       [flags](const astrum::LawContext&) -> std::unique_ptr<astrum::Law> {
         RaceLawParams& p = flags->params;
         const auto planner = astrum::minsnap::planner_from_string(flags->planner);
         if (!planner) throw std::invalid_argument("--planner must be time-optimal or heuristic");
         p.plan.planner = *planner;
         p.plan.max_tilt_rad = astrum::deg_to_rad(flags->max_tilt_deg);
         p.gains.max_tilt_rad = p.plan.max_tilt_rad + astrum::minsnap::kTrackerTiltHeadroomRad;
         if (const auto error = p.validate()) throw std::invalid_argument(*error);
         return std::make_unique<RaceLaw>(p, astrum::stderr_logger("astrum-minsnap-controller"));
       }},
  };
  return astrum::run_main(
      argc, argv, laws, "astrum-minsnap-controller",
      "ASTRUM min-snap race controller: flies a multirotor through the gates of a race "
      "course over Zenoh.");
}
