// The example's raw hold law: heading / altitude / airspeed hold for a fixed-wing.
//
// Outer loops (shared astrum::OuterLoops): heading PI -> bank target within
// --max-bank-deg, altitude PI -> pitch target within --max-pitch-deg, airspeed PI ->
// throttle, every integrator with anti-windup. Inner loops (here): hand-tuned PD on
// the attitude error damped by the telemetry body rates p / q. Signs as measured in
// the sim: aileron+ rolls right, elevator+ pitches nose down.

#ifndef ASTRUM_EXAMPLE_HOLD_LAW_HPP
#define ASTRUM_EXAMPLE_HOLD_LAW_HPP

#include "astrum/control.hpp"
#include "astrum/types.hpp"

namespace astrum {

class HoldLaw {
 public:
  HoldLaw(const AttitudeLimits& limits, const Reference& reference);

  // One control step; outputs are within their actuator ranges.
  RawActuators control(const VehicleState& s);

 private:
  OuterLoops outer_;
  Reference reference_;
};

}  // namespace astrum

#endif  // ASTRUM_EXAMPLE_HOLD_LAW_HPP
