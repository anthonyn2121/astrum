// Shared fixtures for the min-snap tests: a gate at a pose, a course from gate poses,
// the same course as the wire message the law receives.

#ifndef ASTRUM_MINSNAP_TEST_SUPPORT_HPP
#define ASTRUM_MINSNAP_TEST_SUPPORT_HPP

#include <cmath>
#include <vector>

#include "astrum/types.hpp"
#include "course.hpp"

namespace astrum::minsnap::testing {

// The gate hardware of crates/astrum-race at the shipped size: 40" opening, 3" stock.
inline constexpr double kOpening = 40.0 * 0.0254;
inline constexpr double kBar = 3.0 * 0.0254;
inline constexpr double kTopBar = kOpening + 2.0 * kBar;

// A gate at `center` whose normal points along compass heading `heading_deg`
// (0 = north, 90 = east): right = normal rotated clockwise, as in gate.rs.
inline GatePose make_gate(const Vec3& center, double heading_deg) {
  const double psi = heading_deg * 3.14159265358979323846 / 180.0;
  return {center,
          {std::sin(psi), std::cos(psi), 0.0},
          {std::cos(psi), -std::sin(psi), 0.0},
          {0.0, 0.0, 1.0}};
}

inline Course make_course(const std::vector<GatePose>& gates) {
  Course c;
  c.opening_m = kOpening;
  c.bar_m = kBar;
  c.top_bar_m = kTopBar;
  c.gates = gates;
  return c;
}

inline Enu enu(const Vec3& v) { return {v.x, v.y, v.z}; }

inline RaceCourse wire_course(const Course& c) {
  RaceCourse w;
  w.seed = 1;
  w.opening_m = c.opening_m;
  w.bar_m = c.bar_m;
  w.top_bar_m = c.top_bar_m;
  for (std::size_t i = 0; i < c.gates.size(); ++i) {
    const GatePose& g = c.gates[i];
    RaceGate r;
    r.index = static_cast<std::uint32_t>(i);
    r.center = enu(g.center);
    r.yaw_rad = std::atan2(g.normal.x, g.normal.y);
    r.normal = enu(g.normal);
    r.right = enu(g.right);
    r.up = enu(g.up);
    w.gates.push_back(r);
  }
  return w;
}

}  // namespace astrum::minsnap::testing

#endif  // ASTRUM_MINSNAP_TEST_SUPPORT_HPP
