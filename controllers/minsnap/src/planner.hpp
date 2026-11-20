// The race planner: gate poses -> a smooth, time-parameterised flat reference.
//
//   1. Waypoints: the drone's position, then for every remaining gate an entry point
//      (approach_m upstream of the opening, on its normal), the gate centre and an exit
//      point (approach_m downstream). The legs between an exit point and the next entry
//      point are searched with A* when a straight line is blocked (astar.hpp); a clear
//      leg stays a straight segment.
//   2. Times: each leg between two gates takes its length over the cruise speed, or
//      longer when the heading change between the gates would need a yaw rate above the
//      limit (the gates beside such a leg are then flown at the speed the leg flies);
//      segments around a gate take their length over that gate's speed. All times are
//      redistributed by the section V-C gradient descent when asked, and then scaled by a
//      factor, found by iterating, until the solved trajectory respects the speed,
//      acceleration and body-rate limits (at `limit_margin` of them). Scaling time by a
//      divides the pinned gate speeds by a too (section V-A), so the shape is kept. That
//      is the `heuristic` plan.
//   3. Trajectory: degree-7 minimum-snap splines for the position (the velocity at
//      every gate centre is pinned along the gate normal, so each crossing is square)
//      and a minimum-acceleration spline for the yaw, which points along the normal at
//      every gate.
//   4. `time-optimal` (the default) takes the heuristic plan as the starting point of
//      timeopt.hpp, which moves the segment times and the pinned gate speeds to the
//      fastest lap the airframe's limits (speed, thrust, tilt, body rate, yaw rate) allow.
//      A plan that continues another one (an extension, a re-plan) starts the search from
//      that plan's remaining times.
//   5. The sampled trajectory is checked against the hull-inflated frames and the
//      ground; a violating segment gets an extra waypoint at its midpoint and the
//      trajectory is re-solved (the paper's corridor constraint, as an intermediate
//      point); after the time optimisation the times are optimised once more when
//      waypoints were added. If the optimised plan cannot be made clear of the frames,
//      the heuristic plan is flown.

#ifndef ASTRUM_MINSNAP_PLANNER_HPP
#define ASTRUM_MINSNAP_PLANNER_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "astar.hpp"
#include "course.hpp"
#include "minsnap.hpp"
#include "tracking.hpp"
#include "vec3.hpp"

namespace astrum::minsnap {

enum class PlannerKind {
  TimeOptimal,  // the heuristic plan, then timeopt.hpp (fastest lap within the limits)
  Heuristic     // times from cruise speed and temporal scaling to the limits
};

// "time-optimal" / "heuristic".
const char* to_string(PlannerKind kind);

// The planner named `name` ("time-optimal" / "heuristic").
std::optional<PlannerKind> planner_from_string(std::string_view name);

struct PlanParams {
  PlannerKind planner = PlannerKind::TimeOptimal;
  double cruise_mps = 6.0;  // heuristic: target speed through the gates
  // The plan that respects the limits is stretched by this factor (> 1 slows it down:
  // section V-A temporal scaling).
  double time_scale = 1.0;
  // The F450's limits (controllers/minsnap/README.md, "Limits"). The time-optimal plan
  // flies at `limit_margin` of each (as does the heuristic one, for the speed, body rate
  // and yaw rate).
  double max_speed_mps = 12.0;
  double max_thrust_mps2 = 16.7;             // |r'' + g z_W| (1.7 g)
  double min_thrust_mps2 = 2.9;              // the same from below (0.3 g)
  double max_tilt_rad = 0.8726646259971648;  // 50 deg between the thrust and vertical
  double max_rate_rad_s = 0.75;              // roll / pitch body rate (the FCS demand saturates)
  double max_yaw_rate = 0.75;  // rad/s; the heuristic plan's mean rate between two gates' headings
  double limit_margin = 0.9;
  double max_accel_mps2 = 8.0;  // heuristic only: the plan is stretched until |r''| stays below
  // Time optimisation (timeopt.hpp): L-BFGS iterations for a first plan and for a plan
  // that continues another (an extension, a re-plan), and the weight of the snap cost
  // that regularises it.
  int opt_iterations = 130;
  int replan_iterations = 20;
  double snap_weight = 1e-3;
  double approach_m = 1.0;        // entry / exit point distance from the gate plane
  double hull_radius_m = 0.353;   // clearance to a frame away from its own crossing
  double cross_radius_m = 0.285;  // clearance to the frame being crossed (square on)
  double floor_m = 0.3;           // lowest allowed height of the centre of mass
  int max_gates = 16;             // gates covered by one plan
  int realloc_iterations = 0;     // heuristic: section V-C segment-time optimisation
  int refine_iterations = 4;      // corridor repair passes
  AStarParams astar;
};

struct StartState {
  Vec3 position;
  Vec3 velocity;
  double yaw = 0.0;  // flat-output yaw (0 = east, ccw)
  // Set when the plan continues another one (its reference at this instant): the
  // acceleration, jerk and yaw rate are then pinned too, so neither the attitude nor the
  // body-rate feed-forward jumps.
  std::optional<Vec3> acceleration;
  std::optional<Vec3> jerk;
  double yaw_rate = 0.0;
};

class PlanError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// What a plan was made of and how long it took.
struct PlanInfo {
  PlannerKind planner = PlannerKind::Heuristic;  // the planner that was asked for
  // The optimiser's trajectory is the one flown. False for a time-optimal plan when the
  // optimiser found nothing faster than the heuristic plan, or its result could not be
  // made clear of the frames: the heuristic trajectory is flown instead.
  bool improved = false;
  bool warm_start = false;            // the search started from a previous plan's remaining times
  int iterations = 0;                 // L-BFGS iterations of the optimiser, all rounds
  double heuristic_duration_s = 0.0;  // the heuristic plan's duration
  double compute_s = 0.0;             // wall-clock time spent planning (informational)
};

// The role of a waypoint in the plan.
enum class KnotKind { Start, Route, Entry, Gate, Exit };

struct KnotTag {
  KnotKind kind = KnotKind::Route;
  std::size_t gate = 0;  // course index; Entry / Gate / Exit only
};

class Plan;

// An upper estimate of the speed the aircraft can carry through a gate where the course turns
// by `turn_rad` (the angle between the direction it arrives in and the direction it leaves
// in), with `reach_m` of path (the shorter of the two adjacent legs) to do it in; used to
// start the time optimisation from fast crossings instead of one flat speed.
//
// The path is taken to be a circular arc, tangent to both legs at half the reach from the
// corner: radius R = (reach / 2) / tan(turn / 2). With the margined limits
//   * lateral acceleration a = min(g tan(tilt_max), sqrt(f_max^2 - g^2)) bounds v^2 / R <= a;
//   * rolling into the turn (the thrust vector tilts by phi = atan(v^2 / (g R)) at no more
//     than the body-rate limit w) takes phi / w seconds, flown at v within half the reach:
//     v phi / w <= reach / 2 (solved by bisection);
//   * the speed limit.
// A straight crossing (turn 0) is bounded by the speed limit alone; the result is never below
// `min_gate_speed` (0.5 m/s).
double turn_speed_bound(double turn_rad, double reach_m, const PlanParams& params);

// Where a plan that continues another one starts its search: the previous plan, and how
// far into it the new plan starts (its remaining times are the starting point).
struct PlanSeed {
  const Plan* previous = nullptr;
  double time_s = 0.0;
};

class Plan {
 public:
  // The reference at `t` seconds into the plan; after the end it holds the final
  // position and yaw at rest.
  FlatReference at(double t) const;
  double duration() const { return position_.total_time(); }

  // Index of the first gate the plan flies, and the plan time at which each gate from
  // there on is crossed (gate centre).
  std::size_t first_gate() const { return first_gate_; }
  const std::vector<double>& gate_times() const { return gate_times_; }
  std::size_t gate_count() const { return gate_times_.size(); }

  // Waypoints in flight order (start, route points, entry / centre / exit per gate).
  const std::vector<Vec3>& waypoints() const { return waypoints_; }
  const std::vector<KnotTag>& tags() const { return tags_; }
  const minsnap::Trajectory& position() const { return position_; }
  const minsnap::Trajectory& yaw() const { return yaw_; }

  // What the final trajectory does, sampled at 20 Hz.
  double peak_speed() const { return peak_speed_; }
  double peak_accel() const { return peak_accel_; }
  double peak_rate() const { return peak_rate_; }
  // The factor all planned times were finally scaled by (>= PlanParams::time_scale).
  double time_scale() const { return time_scale_; }
  // Smallest clearance of the sampled trajectory to a frame (beyond the hull radius) or
  // the floor; negative when the repair passes could not clear it.
  double min_margin() const { return min_margin_; }
  const PlanInfo& info() const { return info_; }

 private:
  friend Plan plan_race(const Course&, std::size_t, const StartState&, const PlanParams&,
                        const PlanSeed&);
  friend struct PlanAccess;

  std::size_t first_gate_ = 0;
  std::vector<double> gate_times_;
  std::vector<Vec3> waypoints_;
  std::vector<KnotTag> tags_;
  minsnap::Trajectory position_;
  minsnap::Trajectory yaw_;
  double time_scale_ = 1.0;
  double peak_speed_ = 0.0, peak_accel_ = 0.0, peak_rate_ = 0.0, min_margin_ = 0.0;
  PlanInfo info_;
};

// Plans from `start` through gates first_gate ... end of the course. Throws PlanError
// when no route exists or the course has no gate left to fly.
Plan plan_race(const Course& course, std::size_t first_gate, const StartState& start,
               const PlanParams& params, const PlanSeed& seed = {});

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_PLANNER_HPP
