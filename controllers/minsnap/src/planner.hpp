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
//      scaled by `time_scale`, optionally redistributed by the section V-C gradient
//      descent, and then by a further factor, found by iterating, until the solved
//      trajectory respects the speed, acceleration and body-rate limits. Scaling time
//      by a divides the pinned gate speeds by a too (section V-A), so the shape is kept.
//   3. Trajectory: degree-7 minimum-snap splines for the position (the velocity at
//      every gate centre is pinned along the gate normal, so each crossing is square)
//      and a minimum-acceleration spline for the yaw, which points along the normal at
//      every gate.
//   4. The sampled trajectory is checked against the hull-inflated frames and the
//      ground; a violating segment gets an extra waypoint at its midpoint and the
//      trajectory is re-solved (the paper's corridor constraint, as an intermediate point).

#ifndef ASTRUM_MINSNAP_PLANNER_HPP
#define ASTRUM_MINSNAP_PLANNER_HPP

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "astar.hpp"
#include "course.hpp"
#include "minsnap.hpp"
#include "tracking.hpp"
#include "vec3.hpp"

namespace astrum::minsnap {

struct PlanParams {
  double cruise_mps = 6.0;      // target speed through the gates
  double time_scale = 1.0;      // > 1 slows the whole plan down (section V-A temporal scaling)
  double max_speed_mps = 10.0;  // the solved trajectory is stretched until it stays below
  double max_accel_mps2 = 8.0;  // ... and |r''| stays below
  double max_rate_rad_s =
      0.6;  // ... and the roll / pitch body rate it asks for (the F450 demand saturates at 0.75)
  double max_yaw_rate = 0.8;      // rad/s, mean rate allowed between two gates' headings
  double approach_m = 1.0;        // entry / exit point distance from the gate plane
  double hull_radius_m = 0.353;   // clearance to a frame away from its own crossing
  double cross_radius_m = 0.285;  // clearance to the frame being crossed (square on)
  double floor_m = 0.3;           // lowest allowed height of the centre of mass
  int max_gates = 16;             // gates covered by one plan (the solve grows with the cube of it)
  int realloc_iterations = 0;     // section V-C segment-time optimisation
  int refine_iterations = 4;      // corridor repair passes
  AStarParams astar;
};

struct StartState {
  Vec3 position;
  Vec3 velocity;
  double yaw = 0.0;  // flat-output yaw (0 = east, ccw)
  // Set when the plan continues another one (its reference at this instant): the
  // acceleration and yaw rate are then pinned too, so the attitude does not jump.
  std::optional<Vec3> acceleration;
  double yaw_rate = 0.0;
};

class PlanError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
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

 private:
  friend Plan plan_race(const Course&, std::size_t, const StartState&, const PlanParams&);

  std::size_t first_gate_ = 0;
  std::vector<double> gate_times_;
  std::vector<Vec3> waypoints_;
  minsnap::Trajectory position_;
  minsnap::Trajectory yaw_;
  double time_scale_ = 1.0;
  double peak_speed_ = 0.0, peak_accel_ = 0.0, peak_rate_ = 0.0, min_margin_ = 0.0;
};

// Plans from `start` through gates first_gate ... end of the course. Throws PlanError
// when no route exists or the course has no gate left to fly.
Plan plan_race(const Course& course, std::size_t first_gate, const StartState& start,
               const PlanParams& params);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_PLANNER_HPP
