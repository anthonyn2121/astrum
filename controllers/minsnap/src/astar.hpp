// A* over free space between two waypoints.
//
// The planner's waypoints are gate centres plus entry/exit points on each gate's
// normal; the legs between a gate's exit point and the next gate's entry point only
// need a search when a straight line is blocked by some frame. The obstacles are every
// gate's hardware inflated by the airframe's hull radius, and the ground (a minimum
// height for the centre of mass). The search is a 26-connected grid A* on a box around
// the leg; cells are blocked conservatively (by the cell's half diagonal) so that every
// edge between free cell centres is itself clear, and the resulting path is then
// shortened to the fewest straight segments that an exact test still finds clear.

#ifndef ASTRUM_MINSNAP_ASTAR_HPP
#define ASTRUM_MINSNAP_ASTAR_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include "course.hpp"
#include "vec3.hpp"

namespace astrum::minsnap {

struct Hazards {
  const Course* course = nullptr;
  double hull_radius_m = 0.0;  // the centre of mass keeps this far from every frame
  double floor_m = 0.0;        // lowest allowed height of the centre of mass (ENU z)

  // Signed clearance of a centre-of-mass position: distance to the nearest inflated
  // frame, or height above the floor, whichever is smaller. Negative means collision.
  double margin(const Vec3& p) const;
};

struct AStarParams {
  double cell_m = 0.4;
  double box_margin_m = 3.0;  // search box = the two endpoints' bounding box grown by this
  std::size_t max_cells = 6'000'000;
};

// True if every point of the segment a-b (sampled every `step_m`) has margin >= 0.
bool segment_clear(const Vec3& a, const Vec3& b, const Hazards& hazards, double step_m = 0.05);

// A collision-free polyline from `start` to `goal` (both included), or nullopt when
// the search box holds no such path. A clear straight line returns {start, goal}.
std::optional<std::vector<Vec3>> astar_path(const Vec3& start, const Vec3& goal,
                                            const Hazards& hazards, const AStarParams& params);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_ASTAR_HPP
