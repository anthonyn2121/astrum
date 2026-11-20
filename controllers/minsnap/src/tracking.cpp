#include "tracking.hpp"

#include <algorithm>
#include <cmath>

#include "flatness.hpp"
#include "frames.hpp"

namespace astrum::minsnap {

Setpoint Tracker::update(const Estimate& state, const FlatReference& ref, double dt) {
  const Vec3 e_p = state.position - ref.position;
  const Vec3 e_v = state.velocity - ref.velocity;
  integral_ += e_p * dt;
  if (gains_.ki > 0.0) {
    const double limit = gains_.integral_limit / gains_.ki;
    for (int i = 0; i < 3; ++i) integral_[i] = std::clamp(integral_[i], -limit, limit);
  }

  Setpoint c;
  c.force = -gains_.kp * e_p - gains_.kv * e_v - gains_.ki * integral_ + kUnitZ * kGravity +
            ref.acceleration;
  // Never ask for more tilt than max_tilt: scale the horizontal part down to the cone.
  const double horizontal = std::hypot(c.force.x, c.force.y);
  const double max_horizontal = std::max(c.force.z, 0.0) * std::tan(gains_.max_tilt_rad);
  if (horizontal > max_horizontal && horizontal > 0.0) {
    const double s = max_horizontal / horizontal;
    c.force.x *= s;
    c.force.y *= s;
  }
  const Vec3 z_des = unit_or(c.force, kUnitZ);
  const Vec3 z_b = state.rotation.col[2];
  c.thrust = std::max(dot(c.force, z_b), 0.0);

  c.desired = desired_rotation(z_des, ref.yaw, state.rotation);
  c.attitude_error = vee_skew(c.desired.transposed() * state.rotation);

  // Feed-forward body rate of the reference, re-expressed in the actual body frame.
  Vec3 omega_ff;
  if (const auto flat = flat_state({ref.acceleration, ref.jerk, ref.yaw, ref.yaw_rate})) {
    omega_ff = flat->omega_body;
  }
  const Vec3 ff = state.rotation.transposed() * (c.desired * omega_ff);
  c.omega = {ff.x - gains_.k_att_xy * c.attitude_error.x,
             ff.y - gains_.k_att_xy * c.attitude_error.y,
             ff.z - gains_.k_att_z * c.attitude_error.z};
  return c;
}

double collective_for(double thrust, const DemandMap& map) {
  if (!(thrust > 0.0) || !std::isfinite(thrust)) return 0.0;
  return std::clamp(map.hover_collective * std::pow(thrust / kGravity, 1.0 / map.thrust_exponent),
                    0.0, map.max_collective);
}

MultirotorRaw to_demands(const Setpoint& command, const DemandMap& map) {
  const Vec3 rates = rate_demands(command.omega) / map.rate_gain;
  auto unit = [](double v) { return std::isfinite(v) ? std::clamp(v, -1.0, 1.0) : 0.0; };
  MultirotorRaw raw;
  raw.roll_rate = unit(rates.x);
  raw.pitch_rate = unit(rates.y);
  raw.yaw_rate = unit(rates.z);
  raw.collective = collective_for(command.thrust, map);
  if (!std::isfinite(command.thrust)) raw.collective = map.hover_collective;
  return raw;
}

}  // namespace astrum::minsnap
