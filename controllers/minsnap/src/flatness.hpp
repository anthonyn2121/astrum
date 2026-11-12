// Differential flatness of the quadrotor (Mellinger & Kumar, section III): the flat
// outputs are the centre-of-mass position and the yaw angle, and everything else — the
// attitude, the thrust and the body angular velocity — is an algebraic function of them
// and their derivatives.
//
// The world z axis points UP (ENU), so per unit mass
//   t   = r'' + g z_W,   z_B = t / |t|,   u1/m = |t|,
//   x_C = (cos psi, sin psi, 0),   y_B = z_B x x_C / |z_B x x_C|,   x_B = y_B x z_B.
// The angular velocity follows from the time derivative of the same expressions; its
// body z component (the paper states it only for the Euler angles) is derived here from
// d y_B / dt, which keeps y_B perpendicular to x_C.

#ifndef ASTRUM_MINSNAP_FLATNESS_HPP
#define ASTRUM_MINSNAP_FLATNESS_HPP

#include <optional>

#include "vec3.hpp"

namespace astrum::minsnap {

inline constexpr double kGravity = 9.80665;  // m/s^2

struct FlatOutputs {
  Vec3 acceleration;      // r''
  Vec3 jerk;              // r'''
  double yaw = 0.0;       // psi, rad, 0 = east, counter-clockwise from above
  double yaw_rate = 0.0;  // psi'
};

struct FlatState {
  Mat3 rotation;        // body (FLU) -> world (ENU); columns x_B, y_B, z_B
  double thrust = 0.0;  // u1 / m = |r'' + g z_W|, m/s^2
  Vec3 omega_body;      // angular velocity of the body in the world, body components
};

// The state that produces these flat outputs. nullopt at the singularity: no thrust
// (free fall, |t| ~ 0) or z_B parallel to x_C (|z_B x x_C| ~ 0).
std::optional<FlatState> flat_state(const FlatOutputs& flat, double gravity = kGravity);

// Desired body axes for a given thrust direction and yaw (the construction of
// section IV): x_B, y_B, z_B for z_B = `z_b` and heading `yaw`. `reference` is the
// actual attitude: of the two solutions (y_B, x_B) and (-y_B, -x_B), the one closer to
// it is used, which is the paper's practical fix near the z_B || x_C singularity.
Mat3 desired_rotation(const Vec3& z_b, double yaw, const Mat3& reference);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_FLATNESS_HPP
