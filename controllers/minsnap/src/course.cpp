#include "course.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace astrum::minsnap {
namespace {

// Distance from a point (relative to the box centre) to an axis-aligned box.
double box_distance(const Vec3& d, const Vec3& half) {
  const double ox = std::max(std::fabs(d.x) - half.x, 0.0);
  const double oy = std::max(std::fabs(d.y) - half.y, 0.0);
  const double oz = std::max(std::fabs(d.z) - half.z, 0.0);
  return std::sqrt(ox * ox + oy * oy + oz * oz);
}

}  // namespace

double frame_distance(const Course& course, std::size_t gate, const Vec3& p) {
  const GatePose& g = course.gates[gate];
  const Vec3 rel = p - g.center;
  const Vec3 local{dot(rel, g.right), dot(rel, g.up), dot(rel, g.normal)};
  const double h = course.opening_m / 2.0;
  const double t = course.bar_m / 2.0;
  const Vec3 upright_half{t, h, t};
  const Vec3 top_half{course.top_bar_m / 2.0, t, t};
  return std::min({box_distance(local - Vec3{-(h + t), 0.0, 0.0}, upright_half),
                   box_distance(local - Vec3{h + t, 0.0, 0.0}, upright_half),
                   box_distance(local - Vec3{0.0, h + t, 0.0}, top_half)});
}

double nearest_frame_distance(const Course& course, const Vec3& p, std::size_t skip) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < course.gates.size(); ++i) {
    if (i != skip) best = std::min(best, frame_distance(course, i, p));
  }
  return best;
}

double heading_of(const Vec3& direction) { return std::atan2(direction.y, direction.x); }

}  // namespace astrum::minsnap
