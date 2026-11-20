// Small 3-vector / 3x3 rotation types for the min-snap controller (no Eigen: the
// controller needs nothing else from it, and the SDK does not provide one).
//
// Frames: world is ENU (x east, y north, z up). The body frame used throughout this
// controller is FLU — x forward, y left, z up (the thrust axis) — which is the frame
// of Mellinger & Kumar (ICRA 2011). The sim's telemetry is FRD; frames.hpp converts.

#ifndef ASTRUM_MINSNAP_VEC3_HPP
#define ASTRUM_MINSNAP_VEC3_HPP

#include <array>
#include <cmath>

namespace astrum::minsnap {

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;

  constexpr Vec3() = default;
  constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  constexpr double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  constexpr double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

  constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  constexpr Vec3 operator-() const { return {-x, -y, -z}; }
  constexpr Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  constexpr Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
  Vec3& operator+=(const Vec3& o) { return *this = *this + o; }
  Vec3& operator-=(const Vec3& o) { return *this = *this - o; }
  Vec3& operator*=(double s) { return *this = *this * s; }
  bool operator==(const Vec3&) const = default;
};

constexpr Vec3 operator*(double s, const Vec3& v) { return v * s; }
constexpr double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(const Vec3& v) { return std::sqrt(dot(v, v)); }
// v / |v|, or `fallback` when |v| is too small to define a direction.
inline Vec3 unit_or(const Vec3& v, const Vec3& fallback) {
  const double n = norm(v);
  return n > 1e-12 ? v / n : fallback;
}

inline constexpr Vec3 kUnitZ{0.0, 0.0, 1.0};

// A rotation / orthogonal matrix stored by columns: col[0] = image of x, ...
struct Mat3 {
  std::array<Vec3, 3> col{Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};

  static Mat3 from_columns(const Vec3& x, const Vec3& y, const Vec3& z) { return Mat3{{x, y, z}}; }

  double operator()(int row, int c) const { return col[static_cast<std::size_t>(c)][row]; }
  Vec3 operator*(const Vec3& v) const { return col[0] * v.x + col[1] * v.y + col[2] * v.z; }
  Mat3 transposed() const {
    Mat3 t;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) t.col[static_cast<std::size_t>(r)][c] = (*this)(r, c);
    }
    return t;
  }
  Mat3 operator*(const Mat3& o) const {
    return from_columns(*this * o.col[0], *this * o.col[1], *this * o.col[2]);
  }
  double trace() const { return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2); }
};

// The vee map of the skew part of M: (M - M^T)/2 as a vector (so(3) -> R^3).
inline Vec3 vee_skew(const Mat3& m) {
  return {0.5 * (m(2, 1) - m(1, 2)), 0.5 * (m(0, 2) - m(2, 0)), 0.5 * (m(1, 0) - m(0, 1))};
}

}  // namespace astrum::minsnap

#endif  // ASTRUM_MINSNAP_VEC3_HPP
