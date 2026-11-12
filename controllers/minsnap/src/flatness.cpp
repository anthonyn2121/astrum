#include "flatness.hpp"

#include <cmath>

namespace astrum::minsnap {
namespace {

constexpr double kMinThrust = 1e-6;
constexpr double kMinSine = 1e-6;  // |z_B x x_C| below this is the singularity

}  // namespace

std::optional<FlatState> flat_state(const FlatOutputs& flat, double gravity) {
  const Vec3 t = flat.acceleration + kUnitZ * gravity;
  const double thrust = norm(t);
  if (!(thrust > kMinThrust)) return std::nullopt;
  const Vec3 z_b = t / thrust;
  const Vec3 x_c{std::cos(flat.yaw), std::sin(flat.yaw), 0.0};
  const Vec3 y_c{-std::sin(flat.yaw), std::cos(flat.yaw), 0.0};
  const Vec3 u = cross(z_b, x_c);
  const double un = norm(u);
  if (!(un > kMinSine)) return std::nullopt;
  const Vec3 y_b = u / un;
  const Vec3 x_b = cross(y_b, z_b);

  const Vec3 z_dot = (flat.jerk - z_b * dot(z_b, flat.jerk)) / thrust;
  FlatState s;
  s.rotation = Mat3::from_columns(x_b, y_b, z_b);
  s.thrust = thrust;
  s.omega_body = {-dot(z_dot, y_b), dot(z_dot, x_b),
                  -dot(cross(z_dot, x_c) + cross(z_b, y_c) * flat.yaw_rate, x_b) / un};
  return s;
}

Mat3 desired_rotation(const Vec3& z_b, double yaw, const Mat3& reference) {
  const Vec3 x_c{std::cos(yaw), std::sin(yaw), 0.0};
  Vec3 y_b = cross(z_b, x_c);
  const double n = norm(y_b);
  // Exactly at the singularity any y_B perpendicular to z_B is consistent with the yaw;
  // fall back to the actual body y axis projected off z_B.
  y_b = n > kMinSine ? y_b / n
                     : unit_or(reference.col[1] - z_b * dot(z_b, reference.col[1]), Vec3{0, 1, 0});
  const Mat3 a = Mat3::from_columns(cross(y_b, z_b), y_b, z_b);
  const Mat3 b = Mat3::from_columns(cross(-y_b, z_b), -y_b, z_b);
  // trace(R_d^T R) = 1 + 2 cos(angle between them): larger is closer.
  return (a.transposed() * reference).trace() >= (b.transposed() * reference).trace() ? a : b;
}

}  // namespace astrum::minsnap
