#include "astrum/control.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "astrum/math.hpp"

namespace astrum {

SaturatedPi::SaturatedPi(double kp, double ki, double lo, double hi, double bias)
    : kp_(kp), ki_(ki), lo_(lo), hi_(hi), bias_(bias) {}

double SaturatedPi::update(double error, double dt) {
  if (dt > 0.0) {
    const double unsaturated = bias_ + kp_ * error + integral_;
    const double step = ki_ * error * dt;
    const bool winds_up = (unsaturated >= hi_ && step > 0.0) || (unsaturated <= lo_ && step < 0.0);
    if (!winds_up && std::isfinite(step)) integral_ += step;
  }
  return clamp(bias_ + kp_ * error + integral_, lo_, hi_);
}

void add_attitude_limit_options(Options& options, AttitudeLimits& limits) {
  options
      .add("--max-bank-deg", "deg", "bank command limit (f16: 60)", "ASTRUM_MAX_BANK_DEG",
           &limits.max_bank_deg)
      .add("--max-pitch-deg", "deg", "pitch command limit (f16: 24)", "ASTRUM_MAX_PITCH_DEG",
           &limits.max_pitch_deg);
}

Reference ReferenceOptions::reference() const {
  return Reference{wrap_pi(deg_to_rad(heading_deg)), alt_m, speed_mps};
}

void add_reference_options(Options& options, ReferenceOptions& reference) {
  options
      .add("--heading", "deg", "target true heading (0 = north)", "ASTRUM_TARGET_HEADING_DEG",
           &reference.heading_deg)
      .add("--alt", "m", "target altitude, HAE", "ASTRUM_TARGET_ALT_M", &reference.alt_m)
      .add("--speed", "m/s", "target true airspeed", "ASTRUM_TARGET_SPD_MPS", &reference.speed_mps);
}

namespace {

double limit_rad(double deg, const char* name) {
  if (!(deg > 0.0 && deg < 90.0)) {
    throw std::invalid_argument(std::string(name) + " must be in (0, 90) degrees, got " +
                                std::to_string(deg));
  }
  return deg_to_rad(deg);
}

}  // namespace

OuterLoops::OuterLoops(const AttitudeLimits& limits, const OuterLoopGains& gains)
    : max_bank_rad_(limit_rad(limits.max_bank_deg, "--max-bank-deg")),
      max_pitch_rad_(limit_rad(limits.max_pitch_deg, "--max-pitch-deg")),
      heading_(gains.bank_per_heading_err, gains.bank_integral_rate, -max_bank_rad_, max_bank_rad_),
      altitude_(gains.pitch_per_alt_err, gains.pitch_integral_rate, -max_pitch_rad_,
                max_pitch_rad_),
      airspeed_(gains.throttle_per_tas_err, gains.throttle_integral_rate, 0.0, 1.0,
                gains.throttle_trim) {}

AttitudeTargets OuterLoops::update(const VehicleState& s, const Reference& ref) {
  const double dt = prev_t_ && s.t > *prev_t_ ? s.t - *prev_t_ : 0.0;
  prev_t_ = s.t;

  AttitudeTargets out;
  out.bank_rad = heading_.update(wrap_pi(ref.heading_rad - s.yaw), dt);
  out.pitch_rad = altitude_.update(ref.alt_m - s.alt_m, dt);
  out.throttle = airspeed_.update(ref.airspeed_mps - s.airspeed, dt);
  return out;
}

}  // namespace astrum
