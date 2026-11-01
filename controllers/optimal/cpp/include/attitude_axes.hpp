// The inner-loop design shared by laws/lqr.cpp and laws/mpc.cpp: one linear model and
// one set of LQR weights per attitude axis.
//
// Both laws are cascades. astrum::OuterLoops supplies saturated attitude targets from
// the heading / altitude / airspeed errors using the linearized outer dynamics
//     heading:  d(psi)/dt = (g / V) tan(phi) ~ (g / V) phi      (coordinated turn)
//     altitude: d(h)/dt   = V sin(gamma)     ~ V (theta - alpha) (flight-path angle)
// and the inner loop regulates x = [angle - target, body rate] (p ~ d(phi)/dt and
// q ~ d(theta)/dt near wings-level) with the actuator, using attitude_axis_model():
//     d(rate)/dt = -damping * rate + gain * actuator
//
// PLACEHOLDER constants: replace them with your aircraft's identified roll / pitch
// dynamics. With these, the LQR gains land near the hand-tuned hold law
// (aileron ~ 1.4 (phi_t - phi) - 0.5 p, elevator ~ -1.9 (theta_t - theta) + 0.6 q).

#ifndef ASTRUM_ATTITUDE_AXES_HPP
#define ASTRUM_ATTITUDE_AXES_HPP

#include <Eigen/Dense>

#include "optimal_math.hpp"

namespace astrum {

struct AxisDesign {
  const char* name;
  double damping;  // 1/s
  double gain;     // rad/s^2 per unit actuator (sign = actuator polarity)
  Eigen::Matrix2d Q;
  Eigen::Matrix<double, 1, 1> R;

  LinearModel model(double dt) const { return attitude_axis_model(damping, gain, dt); }
};

// Aileron+ rolls right: positive gain.
inline AxisDesign roll_axis() {
  return AxisDesign{"roll", 2.0, 4.0, Eigen::Vector2d(2.25, 0.0).asDiagonal(),
                    Eigen::Matrix<double, 1, 1>::Constant(1.0)};
}

// Elevator+ pitches nose DOWN: negative gain.
inline AxisDesign pitch_axis() {
  return AxisDesign{"pitch", 2.0, -4.0, Eigen::Vector2d(4.0, 0.0).asDiagonal(),
                    Eigen::Matrix<double, 1, 1>::Constant(1.0)};
}

}  // namespace astrum

#endif  // ASTRUM_ATTITUDE_AXES_HPP
