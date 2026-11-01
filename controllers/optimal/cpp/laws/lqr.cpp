// LQR control law.
//
// Cascade (see include/attitude_axes.hpp for the models):
//   outer  astrum::OuterLoops: heading PI -> bank target (|bank| <= --max-bank-deg),
//          altitude PI -> pitch target (|pitch| <= --max-pitch-deg), airspeed PI ->
//          throttle; integrators with anti-windup.
//   inner  discrete LQR per attitude axis, u = -K x with x = [angle - target, rate]:
//          roll  [phi - bank_t,    p] -> aileron
//          pitch [theta - pitch_t, q] -> elevator
//          K = dlqr(A, B, Q, R) of the axis model discretized at the control period
//          (--rate-hz), computed once at startup; a Riccati recursion that does not
//          converge aborts startup instead of flying an unconverged gain.
// Every output is saturated to its actuator range. LQR itself cannot honour hard limits
// beyond that clamp; --law mpc plans with the actuator bounds.
//
// What to do: identify your aircraft's roll / pitch dynamics, put them in
// attitude_axes.hpp, and tune Q / R there.

#include <memory>
#include <stdexcept>
#include <string>

#include "attitude_axes.hpp"
#include "laws.hpp"

namespace astrum {
namespace {

Eigen::RowVector2d synthesize(const AxisDesign& axis, double dt) {
  const LinearModel m = axis.model(dt);
  const DlqrResult r = dlqr(m.A, m.B, axis.Q, axis.R);
  if (!r.converged) {
    throw std::runtime_error(std::string("dlqr for the ") + axis.name +
                             " axis did not converge after " + std::to_string(r.iterations) +
                             " iterations");
  }
  return r.K;
}

class LqrLaw : public Law {
 public:
  explicit LqrLaw(const LawConfig& config)
      : outer_(config.limits),
        k_roll_(synthesize(roll_axis(), config.dt)),
        k_pitch_(synthesize(pitch_axis(), config.dt)) {}

  Command control(const VehicleState& s, const Reference& ref) override {
    const AttitudeTargets target = outer_.update(s, ref);
    const Eigen::Vector2d x_roll(s.roll - target.bank_rad, s.p);
    const Eigen::Vector2d x_pitch(s.pitch - target.pitch_rad, s.q);
    const double aileron = clamp(-k_roll_.dot(x_roll), -1.0, 1.0);
    const double elevator = clamp(-k_pitch_.dot(x_pitch), -1.0, 1.0);
    return Command::raw(aileron, elevator, /*rudder=*/0.0, target.throttle);
  }

 private:
  OuterLoops outer_;
  Eigen::RowVector2d k_roll_;
  Eigen::RowVector2d k_pitch_;
};

}  // namespace

std::unique_ptr<Law> make_lqr_law(const LawConfig& config) {
  return std::make_unique<LqrLaw>(config);
}

}  // namespace astrum
