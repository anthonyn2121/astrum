// Geometric tracking of a flat reference (Mellinger & Kumar, section IV) and the mapping
// of its output onto the sim's multirotor demands.
//
// Position loop, per unit mass:
//   F_des = -Kp e_p - Kv e_v - Ki int(e_p) + g z_W + r_T''      u1/m = F_des . z_B
// Attitude: R_des has z_B,des = F_des/|F_des| and the reference yaw (with the paper's
// choice of the solution nearest the actual attitude near the singularity), and
//   e_R = 1/2 (R_des^T R - R^T R_des)^vee.
// The paper turns e_R and the angular-velocity error into body moments. The sim's F450
// accepts normalised body-RATE demands and closes the rate loop itself, so the same
// error is turned into a rate command instead:
//   omega_cmd = R^T R_des omega_ref - K_R e_R.
// (Rate-demand = omega_cmd / rate_gain; thrust u1 -> collective by the motor model.)

#ifndef ASTRUM_MINSNAP_TRACKING_HPP
#define ASTRUM_MINSNAP_TRACKING_HPP

#include "astrum/types.hpp"
#include "vec3.hpp"

namespace astrum::minsnap {

struct TrackingGains {
  double kp = 9.0;              // position, 1/s^2
  double kv = 5.5;              // velocity, 1/s
  double ki = 1.5;              // position integral, 1/s^3
  double integral_limit = 2.0;  // |Ki int e_p| per axis, m/s^2 (anti-windup)
  double k_att_xy = 8.0;        // roll / pitch attitude error -> rate, 1/s
  double k_att_z = 3.0;         // yaw attitude error -> rate, 1/s
  double max_tilt_rad =
      1.0471975511965976;  // 60 deg: F_des is never tilted further than this from vertical
};

// How a thrust and a body rate become MultirotorRaw demands.
struct DemandMap {
  double hover_collective = 0.44;  // collective that holds altitude
  double thrust_exponent = 1.8;    // thrust ~ collective^exponent around hover
  double rate_gain = 0.75;         // rad/s of body rate per unit of demand
};

struct Estimate {
  Vec3 position, velocity;  // ENU
  Mat3 rotation;            // body (FLU) -> world
};

struct FlatReference {
  Vec3 position, velocity, acceleration, jerk;
  double yaw = 0.0;  // flat-output yaw (0 = east, ccw)
  double yaw_rate = 0.0;
};

struct Setpoint {
  Vec3 force;           // F_des per unit mass, m/s^2
  double thrust = 0.0;  // u1 / m, m/s^2
  Mat3 desired;         // R_des
  Vec3 attitude_error;  // e_R
  Vec3 omega;           // commanded body rate, FLU components, rad/s
};

class Tracker {
 public:
  explicit Tracker(const TrackingGains& gains) : gains_(gains) {}

  // One control step of `dt` seconds.
  Setpoint update(const Estimate& state, const FlatReference& ref, double dt);
  void reset() { integral_ = Vec3{}; }
  const Vec3& integral() const { return integral_; }

 private:
  TrackingGains gains_;
  Vec3 integral_;
};

// Normalised collective for a thrust of `thrust` m/s^2 (per unit mass), clamped to [0, 1].
double collective_for(double thrust, const DemandMap& map);

// The sim demands for a tracking command.
MultirotorRaw to_demands(const Setpoint& command, const DemandMap& map);

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_TRACKING_HPP
