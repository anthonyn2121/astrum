// Relative geometry between two aircraft samples (store samples, ENU grid),
// with the dogfight env's definitions (astrum_dogfight/geometry.py):
//
//  * range   — |LOS|, LOS = bandit − me (m)
//  * ata     — antenna train angle: my nose to the LOS (rad); 0 = nose-on
//  * aspect  — the bandit's nose to the LOS (rad); 0 = I sit on his six
//  * closure — −(v_bandit − v_me) · LOS_hat (m/s); positive = closing
//  * relAlt  — bandit HAE − my HAE (m)
//
// The nose comes from the telemetry Euler angles, which are relative to the
// local level; the env uses the sim's body-to-grid rotation instead, so ATA and
// aspect here differ from the env's by the local level's tilt away from the
// grid (negligible within a few km of the scenario origin).

// Unit nose vector [e, n, u] of a sample (3-2-1 Euler about local NED).
export function noseEnu(s, out = [0, 0, 0]) {
  const ct = Math.cos(s.theta_rad);
  out[0] = ct * Math.sin(s.psi_rad);
  out[1] = ct * Math.cos(s.psi_rad);
  out[2] = Math.sin(s.theta_rad);
  return out;
}

const _nose = [0, 0, 0];

// Angle (rad, [0, π]) between unit vectors a and b; atan2 keeps it accurate
// near 0 and π, where acos of the dot product loses precision.
function angleBetween(a, b) {
  const cx = a[1] * b[2] - a[2] * b[1];
  const cy = a[2] * b[0] - a[0] * b[2];
  const cz = a[0] * b[1] - a[1] * b[0];
  return Math.atan2(Math.hypot(cx, cy, cz), a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

// Geometry of `bandit` seen from `me` (two store samples). ATA, aspect and
// closure are NaN when the two positions coincide (no line of sight).
export function relativeGeometry(me, bandit) {
  const le = bandit.enu_e - me.enu_e;
  const ln = bandit.enu_n - me.enu_n;
  const lu = bandit.enu_u - me.enu_u;
  const range = Math.hypot(le, ln, lu);
  const relAlt = bandit.alt_hae_m - me.alt_hae_m;
  if (!(range > 1e-6)) return { range: 0, ata: NaN, aspect: NaN, closure: NaN, relAlt };
  const los = [le / range, ln / range, lu / range];
  const ata = angleBetween(noseEnu(me, _nose), los);
  const aspect = angleBetween(noseEnu(bandit, _nose), los);
  const closure = -((bandit.vel_e_mps - me.vel_e_mps) * los[0]
    + (bandit.vel_n_mps - me.vel_n_mps) * los[1]
    + (bandit.vel_u_mps - me.vel_u_mps) * los[2]);
  return { range, ata, aspect, closure, relAlt };
}
