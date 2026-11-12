// Conversions between the sim's telemetry / command conventions and this controller's
// FLU body frame (vec3.hpp).
//
// Telemetry: Euler angles (roll phi, pitch theta, yaw psi; psi 0 = north, + east) of a
// FRD body in the local-level NED frame, body rates (p, q, r) about FRD axes, position
// and velocity in world ENU. MultirotorRaw demands: +roll_rate rolls right (+p),
// +pitch_rate pitches the nose UP (+q), +yaw_rate yaws the nose right (+r), each
// normalised by the airframe's demand scale (~0.75 rad/s per unit).

#ifndef ASTRUM_MINSNAP_FRAMES_HPP
#define ASTRUM_MINSNAP_FRAMES_HPP

#include "vec3.hpp"

namespace astrum::minsnap {

// Body (FLU) -> world (ENU) rotation from the telemetry Euler angles.
Mat3 body_to_world(double roll, double pitch, double yaw);

// Telemetry body rates (FRD) as the angular velocity in FLU components.
inline Vec3 omega_flu(double p, double q, double r) { return {p, -q, -r}; }

// A desired FLU angular velocity as the (roll, pitch, yaw) rate demands in rad/s with
// the sim's sign conventions.
inline Vec3 rate_demands(const Vec3& omega_flu_body) {
  return {omega_flu_body.x, -omega_flu_body.y, -omega_flu_body.z};
}

// A telemetry yaw (0 = north, + east) as the flat-output yaw (0 = east, counter-clockwise
// from above).
double flat_yaw_from_heading(double psi);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_FRAMES_HPP
