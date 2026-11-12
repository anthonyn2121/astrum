#include "hold_law.hpp"

#include "astrum/math.hpp"

namespace astrum {

namespace {
constexpr double kRollGain = 1.5;      // aileron / rad bank error
constexpr double kRollDamping = 0.3;   // aileron / (rad/s) roll rate
constexpr double kPitchGain = 2.0;     // elevator / rad pitch error (elevator+ = nose down)
constexpr double kPitchDamping = 0.5;  // elevator / (rad/s) pitch rate
}  // namespace

HoldLaw::HoldLaw(const AttitudeLimits& limits, const Reference& reference)
    : outer_(limits), reference_(reference) {}

LawResult HoldLaw::step(const StepInput& input) {
  const VehicleState& s = input.state;
  const AttitudeTargets target = outer_.update(s, reference_);
  RawActuators out;
  out.aileron = clamp(kRollGain * (target.bank_rad - s.roll) - kRollDamping * s.p, -1.0, 1.0);
  out.elevator = clamp(-kPitchGain * (target.pitch_rad - s.pitch) + kPitchDamping * s.q, -1.0, 1.0);
  out.rudder = 0.0;
  out.throttle = target.throttle;
  return Action{out};
}

}  // namespace astrum
