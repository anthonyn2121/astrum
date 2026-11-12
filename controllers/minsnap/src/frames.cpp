#include "frames.hpp"

#include <cmath>

namespace astrum::minsnap {

Mat3 body_to_world(double roll, double pitch, double yaw) {
  const double sp = std::sin(roll), cp = std::cos(roll);
  const double st = std::sin(pitch), ct = std::cos(pitch);
  const double ss = std::sin(yaw), cs = std::cos(yaw);
  // Columns of Rz(psi) Ry(theta) Rx(phi) (FRD -> NED) read out in ENU, with y and z
  // negated (FRD -> FLU): x_B forward, y_B left, z_B up.
  const Vec3 forward{ct * ss, ct * cs, st};
  const Vec3 right{sp * st * ss + cp * cs, sp * st * cs - cp * ss, -sp * ct};
  const Vec3 up{sp * cs - cp * st * ss, -(cp * st * cs + sp * ss), cp * ct};
  return Mat3::from_columns(forward, -right, up);
}

double flat_yaw_from_heading(double psi) { return 1.5707963267948966 - psi; }

}  // namespace astrum::minsnap
