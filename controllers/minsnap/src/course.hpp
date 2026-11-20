// The racing course as the planner sees it: gate poses plus the hardware dimensions,
// both taken from the wire (RaceCourse) — nothing about a gate is hard-coded here.
//
// A gate is three boxes of square stock in its own frame (x right, y up, z along the
// flight normal), as built by crates/astrum-race: two uprights as tall as the opening
// flanking it, and a top bar over both. `opening_m` is the clear width and height,
// `bar_m` the stock thickness, `top_bar_m` the top bar's length. The centre is the
// middle of the opening; below the opening is open air down to the ground.

#ifndef ASTRUM_MINSNAP_COURSE_HPP
#define ASTRUM_MINSNAP_COURSE_HPP

#include <cstddef>
#include <vector>

#include "vec3.hpp"

namespace astrum::minsnap {

struct GatePose {
  Vec3 center;  // world ENU, middle of the opening
  Vec3 normal;  // unit, direction of flight through the opening
  Vec3 right;   // unit, starboard of the flight direction
  Vec3 up;      // unit
};

struct Course {
  double opening_m = 0.0;
  double bar_m = 0.0;
  double top_bar_m = 0.0;
  std::vector<GatePose> gates;  // flight order
};

// Euclidean distance from `p` to the hardware of gate `gate` (0 inside the stock).
double frame_distance(const Course& course, std::size_t gate, const Vec3& p);

// Smallest distance from `p` to the hardware of any gate except `skip` (pass
// course.gates.size() to skip none); infinity for an empty course.
double nearest_frame_distance(const Course& course, const Vec3& p, std::size_t skip);

// Yaw (radians, 0 = east, counter-clockwise) of a direction's horizontal part.
double heading_of(const Vec3& direction);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_COURSE_HPP
