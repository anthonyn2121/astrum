#include "flatness.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "frames.hpp"
#include "minsnap.hpp"

namespace astrum::minsnap {
namespace {

constexpr double kPi = 3.14159265358979323846;

void expect_vec(const Vec3& a, const Vec3& b, double tol) {
  EXPECT_NEAR(a.x, b.x, tol);
  EXPECT_NEAR(a.y, b.y, tol);
  EXPECT_NEAR(a.z, b.z, tol);
}

void expect_rotation(const Mat3& r, double tol) {
  const Mat3 i = r.transposed() * r;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) EXPECT_NEAR(i(a, b), a == b ? 1.0 : 0.0, tol);
  }
  EXPECT_NEAR(dot(cross(r.col[0], r.col[1]), r.col[2]), 1.0, tol);
}

// Rotation by `angle` about a unit axis (Rodrigues).
Mat3 rotation_about(const Vec3& axis, double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  Mat3 r;
  for (int i = 0; i < 3; ++i) {
    Vec3 e;
    e[i] = 1.0;
    r.col[static_cast<std::size_t>(i)] =
        e * c + cross(axis, e) * s + axis * (dot(axis, e) * (1.0 - c));
  }
  return r;
}

TEST(Frames, BodyToWorldMatchesHandChecks) {
  // Level, nose north: forward = N, left = W, up = up.
  Mat3 r = body_to_world(0, 0, 0);
  expect_vec(r.col[0], {0, 1, 0}, 1e-12);
  expect_vec(r.col[1], {-1, 0, 0}, 1e-12);
  expect_vec(r.col[2], {0, 0, 1}, 1e-12);
  // Heading east: forward = E, left = N.
  r = body_to_world(0, 0, kPi / 2);
  expect_vec(r.col[0], {1, 0, 0}, 1e-12);
  expect_vec(r.col[1], {0, 1, 0}, 1e-12);
  // Right wing down 90 deg, nose north: the thrust axis points to the RIGHT (east).
  r = body_to_world(kPi / 2, 0, 0);
  expect_vec(r.col[2], {1, 0, 0}, 1e-12);
  expect_vec(r.col[0], {0, 1, 0}, 1e-12);
  // Nose up 90 deg, heading north: forward = up, the thrust axis points aft (south).
  r = body_to_world(0, kPi / 2, 0);
  expect_vec(r.col[0], {0, 0, 1}, 1e-12);
  expect_vec(r.col[2], {0, -1, 0}, 1e-12);
  expect_rotation(body_to_world(0.4, -0.7, 2.1), 1e-12);
}

TEST(Frames, RateDemandsFollowTheSimSigns) {
  expect_vec(omega_flu(1.0, 2.0, 3.0), {1.0, -2.0, -3.0}, 0.0);
  // +p right wing down, +q nose up, +r nose right.
  expect_vec(rate_demands({0.3, 0.2, 0.1}), {0.3, -0.2, -0.1}, 0.0);
  // omega_flu and rate_demands are inverse for the telemetry -> demand round trip.
  const Vec3 w = omega_flu(0.1, -0.2, 0.3);
  expect_vec(rate_demands(w), {0.1, -0.2, 0.3}, 1e-15);
  EXPECT_NEAR(flat_yaw_from_heading(0.0), kPi / 2, 1e-15);  // north = +y = 90 deg from east
  EXPECT_NEAR(flat_yaw_from_heading(kPi / 2), 0.0, 1e-15);  // east
}

TEST(Flatness, HoverHasUprightLevelZeroRates) {
  const auto s = flat_state({{0, 0, 0}, {0, 0, 0}, 0.7, 0.0});
  ASSERT_TRUE(s);
  EXPECT_NEAR(s->thrust, kGravity, 1e-12);
  expect_vec(s->rotation.col[2], {0, 0, 1}, 1e-12);
  expect_vec(s->rotation.col[0], {std::cos(0.7), std::sin(0.7), 0}, 1e-12);
  expect_vec(s->omega_body, {0, 0, 0}, 1e-12);
}

TEST(Flatness, YawRateIsTheBodyZRateInHover) {
  const auto s = flat_state({{0, 0, 0}, {0, 0, 0}, 0.0, 0.4});
  ASSERT_TRUE(s);
  expect_vec(s->omega_body, {0, 0, 0.4}, 1e-12);
}

TEST(Flatness, AccelerationTiltsThrustTowardIt) {
  // Accelerating east at g: thrust vector 45 deg toward east, |t| = sqrt(2) g.
  const auto s = flat_state({{kGravity, 0, 0}, {0, 0, 0}, 0.0, 0.0});
  ASSERT_TRUE(s);
  EXPECT_NEAR(s->thrust, std::sqrt(2.0) * kGravity, 1e-12);
  expect_vec(s->rotation.col[2], {std::sqrt(0.5), 0, std::sqrt(0.5)}, 1e-12);
  // Facing east and leaning east is a nose-down pitch: forward axis points below the horizon.
  EXPECT_LT(s->rotation.col[0].z, 0.0);
  expect_rotation(s->rotation, 1e-12);
}

TEST(Flatness, SingularityIsReported) {
  // t along +x with yaw 0: z_B parallel to x_C.
  EXPECT_FALSE(flat_state({{5.0, 0, -kGravity}, {0, 0, 0}, 0.0, 0.0}));
  // Free fall: no thrust direction.
  EXPECT_FALSE(flat_state({{0, 0, -kGravity}, {1, 0, 0}, 0.0, 0.0}));
}

// A smooth trajectory with real tilts: through several keyframes, yaw turning too.
struct Flight {
  Trajectory pos, yaw;
};

Flight smooth_flight() {
  Problem p;
  p.cost_order = 4;
  p.channels = 3;
  p.durations = {1.2, 1.4, 1.0, 1.6};
  const double pts[5][3] = {{0, 0, 1}, {3, 1, 1.5}, {5, 4, 2}, {2, 6, 1.2}, {0, 8, 1.5}};
  for (std::size_t k = 0; k < 5; ++k)
    p.constraints.push_back({k, 0, {pts[k][0], pts[k][1], pts[k][2]}});
  for (std::size_t k : {0u, 4u}) {
    p.constraints.push_back({k, 1, {0, 0, 0}});
    p.constraints.push_back({k, 2, {0, 0, 0}});
  }
  Problem y;
  y.cost_order = 2;
  y.channels = 1;
  y.durations = {2.6, 3.0};
  const double yaw[3] = {0.0, 0.8, 1.9};
  for (std::size_t k = 0; k < 3; ++k) y.constraints.push_back({k, 0, {yaw[k]}});
  y.constraints.push_back({0, 1, {0.0}});
  return {solve(p), solve(y)};
}

FlatOutputs outputs_at(const Flight& f, double t) {
  const auto a = f.pos.value(t, 2), j = f.pos.value(t, 3);
  return {{a[0], a[1], a[2]}, {j[0], j[1], j[2]}, f.yaw.value(t, 0, 0), f.yaw.value(t, 1, 0)};
}

TEST(Flatness, AngularVelocityIsTheRotationRateOfTheAttitude) {
  const Flight f = smooth_flight();
  const double h = 1e-5;
  for (double t : {0.4, 1.3, 2.2, 3.1, 4.0, 5.0}) {
    const auto lo = flat_state(outputs_at(f, t - h));
    const auto mid = flat_state(outputs_at(f, t));
    const auto hi = flat_state(outputs_at(f, t + h));
    ASSERT_TRUE(lo && mid && hi);
    expect_rotation(mid->rotation, 1e-12);
    // omega_body = vee(R^T dR/dt).
    const Mat3 rt = mid->rotation.transposed();
    Mat3 dr;
    for (int c = 0; c < 3; ++c)
      dr.col[static_cast<std::size_t>(c)] = (hi->rotation.col[static_cast<std::size_t>(c)] -
                                             lo->rotation.col[static_cast<std::size_t>(c)]) /
                                            (2 * h);
    expect_vec(vee_skew(rt * dr), mid->omega_body, 2e-5);
  }
}

TEST(Flatness, RoundTripIntegratingTheRigidBodyRecoversTheTrajectory) {
  const Flight f = smooth_flight();
  const double total = f.pos.total_time();
  // Start from the reference state, then integrate the dynamics with only the
  // flat-derived inputs (thrust and body rates): R' = R [omega]x, r'' = u1/m z_B - g z_W.
  const auto s0 = flat_state(outputs_at(f, 0.0));
  ASSERT_TRUE(s0);
  Mat3 rot = s0->rotation;
  Vec3 p{f.pos.value(0.0, 0, 0), f.pos.value(0.0, 0, 1), f.pos.value(0.0, 0, 2)};
  Vec3 v{0, 0, 0};
  const double dt = 1e-3;
  double worst_pos = 0.0, worst_att = 0.0;
  for (double t = 0.0; t < total - dt; t += dt) {
    // Midpoint rule: the inputs at the middle of the step drive both the attitude and
    // the translation.
    const auto s = flat_state(outputs_at(f, t + 0.5 * dt));
    ASSERT_TRUE(s);
    const Vec3 w = s->omega_body;
    const double wn = norm(w);
    const Vec3 axis = wn > 1e-12 ? w / wn : kUnitZ;
    const Mat3 rot_mid = rot * rotation_about(axis, 0.5 * wn * dt);
    const Vec3 a = rot_mid.col[2] * s->thrust - kUnitZ * kGravity;
    p += v * dt + a * (0.5 * dt * dt);
    v += a * dt;
    rot = rot * rotation_about(axis, wn * dt);
    const Vec3 want{f.pos.value(t + dt, 0, 0), f.pos.value(t + dt, 0, 1),
                    f.pos.value(t + dt, 0, 2)};
    worst_pos = std::fmax(worst_pos, norm(p - want));
    const auto next = flat_state(outputs_at(f, t + dt));
    ASSERT_TRUE(next);
    worst_att = std::fmax(worst_att, norm(vee_skew(next->rotation.transposed() * rot)));
  }
  EXPECT_LT(worst_pos, 5e-3);
  EXPECT_LT(worst_att, 5e-3);
}

TEST(Flatness, DesiredRotationHasTheThrustAxisAndYaw) {
  const Mat3 level;
  const Vec3 z = unit_or({0.3, -0.2, 1.0}, kUnitZ);
  const Mat3 r = desired_rotation(z, 0.9, level);
  expect_vec(r.col[2], z, 1e-12);
  expect_rotation(r, 1e-12);
  // Forward axis has the requested heading: its horizontal part is along x_C.
  EXPECT_GT(dot(r.col[0], {std::cos(0.9), std::sin(0.9), 0}), 0.0);
  EXPECT_NEAR(r.col[1].x * std::cos(0.9) + r.col[1].y * std::sin(0.9), 0.0, 1e-12);
}

TEST(Flatness, DesiredRotationGuardsTheSingularity) {
  // Thrust exactly along x_C: any y_B perpendicular to z_B is valid; the result must
  // still be a proper rotation, with the nearest-to-actual choice.
  const Vec3 z{1.0, 0.0, 0.0};
  const Mat3 r = desired_rotation(z, 0.0, Mat3{});
  expect_rotation(r, 1e-12);
  expect_vec(r.col[2], z, 1e-12);
  // Near the singularity the two solutions differ by a half turn about z_B: the one
  // nearer the actual attitude wins, so the choice follows the actual y axis.
  const Vec3 z2 = unit_or({1.0, 1e-3, 0.0}, kUnitZ);
  const Mat3 actual =
      Mat3::from_columns({0, -1, 0}, {0, 0, -1}, {1, 0, 0});  // thrust east, left axis down
  const Mat3 flipped = Mat3::from_columns({0, 1, 0}, {0, 0, 1}, {1, 0, 0});  // half turn about z_B
  const Mat3 d1 = desired_rotation(z2, 0.0, actual);
  const Mat3 d2 = desired_rotation(z2, 0.0, flipped);
  EXPECT_GT(dot(d1.col[1], actual.col[1]), 0.0);
  EXPECT_GT(dot(d2.col[1], flipped.col[1]), 0.0);
  EXPECT_LT(dot(d1.col[1], d2.col[1]), 0.0);
}

}  // namespace
}  // namespace astrum::minsnap
