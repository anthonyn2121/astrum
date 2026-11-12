// LQR control law (`--law lqr`).
//
// Cascade (see include/attitude_axes.hpp for the models):
//   outer  astrum::OuterLoops: heading PI -> bank target (|bank| <= --max-bank-deg),
//          altitude PI -> pitch target (|pitch| <= --max-pitch-deg), airspeed PI ->
//          throttle; integrators with anti-windup.
//   inner  discrete LQR per attitude axis, u = -K x with x = [angle - target, rate]:
//          roll  [phi - bank_t,    p] -> aileron
//          pitch [theta - pitch_t, q] -> elevator
//          K = dlqr(A, B, Q, R) of the axis model discretized at the control period
//          (the telemetry's controlPeriodS), computed once in configure(); a Riccati
//          recursion that does not converge fails the run (LawFault) instead of flying
//          an unconverged gain.
// Every output is saturated to its actuator range. LQR itself cannot honour hard limits
// beyond that clamp; --law mpc plans with the actuator bounds.
//
// What to do: identify your aircraft's roll / pitch dynamics, put them in
// attitude_axes.hpp, and tune Q / R there.

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "astrum/control.hpp"
#include "astrum/math.hpp"
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
  explicit LqrLaw(const LawContext& context)
      : outer_(context.limits), reference_(context.reference) {}

  std::vector<VehicleKind> vehicle_kinds() const override { return {VehicleKind::FixedWing}; }

  void configure(double control_period_s) override {
    k_roll_ = synthesize(roll_axis(), control_period_s);
    k_pitch_ = synthesize(pitch_axis(), control_period_s);
    configured_ = true;
  }

  LawResult step(const StepInput& input) override {
    if (!configured_) throw std::logic_error("lqr: step before configure");
    const VehicleState& s = input.state;
    const AttitudeTargets target = outer_.update(s, reference_);
    const Eigen::Vector2d x_roll(s.roll - target.bank_rad, s.p);
    const Eigen::Vector2d x_pitch(s.pitch - target.pitch_rad, s.q);
    const double aileron = clamp(-k_roll_.dot(x_roll), -1.0, 1.0);
    const double elevator = clamp(-k_pitch_.dot(x_pitch), -1.0, 1.0);
    return Action{RawActuators{aileron, elevator, /*rudder=*/0.0, target.throttle}};
  }

 private:
  OuterLoops outer_;
  Reference reference_;
  bool configured_ = false;
  Eigen::RowVector2d k_roll_ = Eigen::RowVector2d::Zero();
  Eigen::RowVector2d k_pitch_ = Eigen::RowVector2d::Zero();
};

}  // namespace

std::unique_ptr<Law> make_lqr_law(const LawContext& context) {
  return std::make_unique<LqrLaw>(context);
}

LawEntry lqr_law_entry() {
  return LawEntry{"lqr", "discrete LQR attitude loops under saturated outer loops (fixed-wing)",
                  nullptr, make_lqr_law};
}

}  // namespace astrum
