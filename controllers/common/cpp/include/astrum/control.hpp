// Control building blocks shared by the hold example and the LQR/MPC laws.
//
// Every fixed-wing law here is a cascade: OuterLoops turns the heading / altitude /
// airspeed errors into SATURATED attitude and throttle targets (bank within
// --max-bank-deg, pitch within --max-pitch-deg, throttle within [0, 1], each loop
// with anti-windup integral action), and a law-specific inner loop drives
// aileron / elevator to hold that attitude. Keeping the limits in the outer loop is
// what stops a 90 deg heading error or a 500 m altitude step from commanding an
// unflyable attitude, whatever the inner loop's gains.

#ifndef ASTRUM_CONTROL_HPP
#define ASTRUM_CONTROL_HPP

#include <optional>

#include "astrum/types.hpp"

namespace astrum {

// PI controller with an output clamp and conditional-integration anti-windup: the
// integral only advances when doing so would not push an already-saturated output
// further into its limit.
class SaturatedPi {
 public:
  // output = clamp(bias + kp * e + integral, lo, hi), d(integral)/dt = ki * e.
  SaturatedPi(double kp, double ki, double lo, double hi, double bias = 0.0);

  // Advance by `dt` seconds (<= 0: no integration) and return the clamped output.
  double update(double error, double dt);

  double integral() const { return integral_; }

 private:
  double kp_, ki_, lo_, hi_, bias_;
  double integral_ = 0.0;
};

// Outer-loop gains. The proportional terms are the hold law verified on the c172x;
// heading follows psi_dot = (g / V) tan(phi), so bank = 1.2 * heading_err gives a
// heading time constant of V / (1.2 g) ~ 4 s at 50 m/s until the bank limit takes over.
struct OuterLoopGains {
  double bank_per_heading_err = 1.2;  // rad bank / rad heading error
  // Removes the steady heading error a pure-P bank command leaves when the airframe
  // needs a small standing bank/aileron (propeller torque, asymmetry).
  double bank_integral_rate = 0.05;       // rad bank / (rad heading error s)
  double pitch_per_alt_err = 0.01;        // rad pitch / m
  double pitch_integral_rate = 0.0005;    // rad pitch / (m s)
  double throttle_trim = 0.6;             //
  double throttle_per_tas_err = 0.02;     // throttle / (m/s)
  double throttle_integral_rate = 0.005;  // throttle / (m/s s)
};

struct AttitudeTargets {
  double bank_rad = 0.0;   // within +-max bank
  double pitch_rad = 0.0;  // within +-max pitch
  double throttle = 0.0;   // within [0, 1]
};

class OuterLoops {
 public:
  // Throws std::invalid_argument unless both limits are in (0, 90) degrees.
  explicit OuterLoops(const AttitudeLimits& limits, const OuterLoopGains& gains = {});

  // Integrators advance by the sim-time delta since the previous call (0 on the first
  // call and on a repeated or earlier sample).
  AttitudeTargets update(const VehicleState& s, const Reference& ref);

  double max_bank_rad() const { return max_bank_rad_; }
  double max_pitch_rad() const { return max_pitch_rad_; }

 private:
  double max_bank_rad_;
  double max_pitch_rad_;
  SaturatedPi heading_;
  SaturatedPi altitude_;
  SaturatedPi airspeed_;
  std::optional<double> prev_sim_time_s_;
};

}  // namespace astrum

#endif  // ASTRUM_CONTROL_HPP
