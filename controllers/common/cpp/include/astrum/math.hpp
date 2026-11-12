// Small numeric helpers shared by every ASTRUM C++ controller.

#ifndef ASTRUM_MATH_HPP
#define ASTRUM_MATH_HPP

#include <algorithm>
#include <cmath>

namespace astrum {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 2.0 * kPi;
inline constexpr double kGravity = 9.80665;  // standard gravity (m/s^2)

constexpr double deg_to_rad(double deg) { return deg * (kPi / 180.0); }
constexpr double rad_to_deg(double rad) { return rad * (180.0 / kPi); }

// Wrap an angle to [-pi, pi) — the same convention as the Python controllers'
// `(a + pi) % (2 pi) - pi`. Constant time; a non-finite input yields NaN (it never
// loops on +-inf).
inline double wrap_pi(double a) {
  double w = std::fmod(a + kPi, kTwoPi);  // (-2pi, 2pi), NaN for non-finite a
  if (w < 0.0) w += kTwoPi;               // [0, 2pi]
  if (w >= kTwoPi) w = 0.0;               // a tiny negative w can round up to 2pi
  return w - kPi;
}

// Clamp x to [lo, hi]. NaN stays NaN (callers that must never emit NaN check it).
constexpr double clamp(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

}  // namespace astrum

#endif  // ASTRUM_MATH_HPP
