// The min-snap race law (`--law minsnap`): flies a multirotor through a racing course.
//
// It reads the course (RaceCourse: gate poses and hardware dimensions) and the race
// progress (RaceState) off the wire, plans a minimum-snap trajectory through the gates
// from the aircraft's current state (planner.hpp), tracks it with the geometric
// controller of Mellinger & Kumar (tracking.hpp) and answers with MultirotorRaw
// demands. It re-plans from the current state when the aircraft is far from the
// reference or has not passed the gate it should have by now (a missed gate); until the
// course arrives, and after the race finishes, it holds position. A plan covers at most
// `--horizon-gates` gates and is extended from its own reference as the aircraft nears the
// last of them, so the cost of planning does not grow with the length of the course.

#ifndef ASTRUM_MINSNAP_RACE_LAW_HPP
#define ASTRUM_MINSNAP_RACE_LAW_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "astrum/loop.hpp"
#include "astrum/types.hpp"
#include "course.hpp"
#include "planner.hpp"
#include "tracking.hpp"

namespace astrum::minsnap {

struct RaceLawParams {
  PlanParams plan;
  TrackingGains gains;
  DemandMap demands;
  double replan_error_m = 1.5;  // position error that forces a re-plan
  double replan_slack_s = 1.0;  // grace after a gate's planned crossing time
  double replan_min_interval_s = 0.5;

  // The first problem with the values, or nullopt.
  std::optional<std::string> validate() const;
};

// The planner's view of a wire course (gates sorted by index).
Course course_from_wire(const RaceCourse& wire);

class RaceLaw : public Law {
 public:
  explicit RaceLaw(const RaceLawParams& params);

  std::vector<VehicleKind> vehicle_kinds() const override { return {VehicleKind::Multirotor}; }
  bool wants_race() const override { return true; }
  LawResult step(const StepInput& input) override;

  // For tests and logs: the plan being flown and how many plans were made.
  const Plan* plan() const { return plan_ ? &*plan_ : nullptr; }
  std::size_t plans_made() const { return plans_made_; }

 private:
  StartState measured_start(const VehicleState& state) const;
  void make_plan(const VehicleState& state, const StartState& start);
  bool window_ends_soon() const;
  bool needs_replan(const VehicleState& state, const FlatReference& ref) const;

  RaceLawParams params_;
  Tracker tracker_;
  std::optional<RaceCourse> wire_course_;
  std::optional<Course> course_;
  std::optional<RaceState> race_;
  std::optional<Plan> plan_;
  double plan_start_s_ = 0.0;
  double last_plan_s_ = 0.0;
  std::size_t plans_made_ = 0;
  std::optional<Vec3> hold_position_;
  double hold_yaw_ = 0.0;
};

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_RACE_LAW_HPP
