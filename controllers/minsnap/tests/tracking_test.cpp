#include "tracking.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "flatness.hpp"
#include "frames.hpp"

namespace astrum::minsnap {
namespace {

Estimate level_at(const Vec3& p, double yaw = 0.0) {
  return {p,
          {0, 0, 0},
          Mat3::from_columns({std::cos(yaw), std::sin(yaw), 0}, {-std::sin(yaw), std::cos(yaw), 0},
                             kUnitZ)};
}

FlatReference hold(const Vec3& p, double yaw = 0.0) {
  FlatReference r;
  r.position = p;
  r.yaw = yaw;
  return r;
}

Mat3 about_x(double a) {
  return Mat3::from_columns({1, 0, 0}, {0, std::cos(a), std::sin(a)},
                            {0, -std::sin(a), std::cos(a)});
}

TEST(Tracking, OnTheReferenceItHoversWithNoRateDemand) {
  Tracker t(TrackingGains{});
  const Setpoint s = t.update(level_at({1, 2, 3}), hold({1, 2, 3}), 0.01);
  EXPECT_NEAR(s.thrust, kGravity, 1e-12);
  EXPECT_NEAR(norm(s.attitude_error), 0.0, 1e-12);
  EXPECT_NEAR(norm(s.omega), 0.0, 1e-12);
  const MultirotorRaw raw = to_demands(s, DemandMap{});
  EXPECT_NEAR(raw.collective, 0.44, 1e-12);
  EXPECT_NEAR(raw.roll_rate, 0.0, 1e-12);
  EXPECT_NEAR(raw.pitch_rate, 0.0, 1e-12);
  EXPECT_NEAR(raw.yaw_rate, 0.0, 1e-12);
}

TEST(Tracking, AttitudeErrorIsTheSineOfASmallRotation) {
  Tracker t(TrackingGains{});
  // Body rolled 0.1 rad about its forward axis relative to the (level) desired attitude.
  Estimate e = level_at({0, 0, 0});
  e.rotation = about_x(0.1);
  // Aim the force at the actual z so that the desired z is z_W (hover reference).
  const Setpoint s = t.update(e, hold({0, 0, 0}), 0.01);
  EXPECT_NEAR(s.attitude_error.x, std::sin(0.1), 1e-12);
  EXPECT_NEAR(s.attitude_error.y, 0.0, 1e-12);
  EXPECT_NEAR(s.attitude_error.z, 0.0, 1e-12);
  // Rolled one way, the rate command rolls it back: -k_att * e_R about x.
  EXPECT_NEAR(s.omega.x, -TrackingGains{}.k_att_xy * std::sin(0.1), 1e-12);
}

TEST(Tracking, DemandSignsFollowTheSimConventions) {
  const DemandMap map;
  Tracker t(TrackingGains{});
  // Rolled right (right wing down, +phi): the demand must roll back left (negative).
  Estimate e = level_at({0, 0, 0});
  e.rotation = body_to_world(0.2, 0.0, 0.0);
  Setpoint s = t.update(e, hold({0, 0, 0}, flat_yaw_from_heading(0.0)), 0.01);
  EXPECT_LT(to_demands(s, map).roll_rate, 0.0);
  // Nose up (+theta): the demand pitches back down (negative pitch demand).
  e.rotation = body_to_world(0.0, 0.2, 0.0);
  s = t.update(e, hold({0, 0, 0}, flat_yaw_from_heading(0.0)), 0.01);
  EXPECT_LT(to_demands(s, map).pitch_rate, 0.0);
  // Heading 0.3 rad east of north when north is wanted: yaw left = negative demand.
  e.rotation = body_to_world(0.0, 0.0, 0.3);
  s = t.update(e, hold({0, 0, 0}, flat_yaw_from_heading(0.0)), 0.01);
  EXPECT_LT(to_demands(s, map).yaw_rate, 0.0);
  // And the mirror: heading west of north, yaw right = positive.
  e.rotation = body_to_world(0.0, 0.0, -0.3);
  s = t.update(e, hold({0, 0, 0}, flat_yaw_from_heading(0.0)), 0.01);
  EXPECT_GT(to_demands(s, map).yaw_rate, 0.0);
}

TEST(Tracking, PositionErrorsPushTheThrustTheRightWay) {
  Tracker t(TrackingGains{});
  // Below the reference: more thrust. Above: less.
  EXPECT_GT(t.update(level_at({0, 0, -1}), hold({0, 0, 0}), 0.01).thrust, kGravity);
  Tracker t2(TrackingGains{});
  EXPECT_LT(t2.update(level_at({0, 0, 1}), hold({0, 0, 0}), 0.01).thrust, kGravity);
  // East of the reference: the force points back west.
  Tracker t3(TrackingGains{});
  EXPECT_LT(t3.update(level_at({0.5, 0, 0}), hold({0, 0, 0}), 0.01).force.x, 0.0);
  // Moving away faster than the reference: the velocity term brakes too.
  Tracker t4(TrackingGains{});
  Estimate moving = level_at({0, 0, 0});
  moving.velocity = {0, 2, 0};
  EXPECT_LT(t4.update(moving, hold({0, 0, 0}), 0.01).force.y, 0.0);
}

TEST(Tracking, AccelerationFeedForwardTiltsTheDesiredAttitude) {
  Tracker t(TrackingGains{});
  FlatReference r = hold({0, 0, 0});
  r.acceleration = {2.0, 0, 0};
  const Setpoint s = t.update(level_at({0, 0, 0}), r, 0.01);
  EXPECT_GT(s.desired.col[2].x, 0.0);
  EXPECT_NEAR(s.thrust, kGravity, 1e-9);  // thrust projected on the (level) actual z axis
  EXPECT_GT(norm(s.force), kGravity);
}

TEST(Tracking, TiltLimitCapsTheHorizontalForce) {
  TrackingGains g;
  g.max_tilt_rad = 0.5;
  Tracker t(g);
  FlatReference r = hold({0, 0, 0});
  r.acceleration = {50.0, 0, 0};
  const Setpoint s = t.update(level_at({0, 0, 0}), r, 0.01);
  EXPECT_NEAR(std::atan2(std::hypot(s.force.x, s.force.y), s.force.z), 0.5, 1e-12);
}

TEST(Tracking, IntegralActsAndIsLimited) {
  TrackingGains g;
  g.ki = 1.0;
  g.integral_limit = 0.5;
  Tracker t(g);
  Setpoint s;
  for (int i = 0; i < 2000; ++i) s = t.update(level_at({0, 0, -0.5}), hold({0, 0, 0}), 0.01);
  // Persistent undershoot: the integral term adds up but never beyond integral_limit.
  EXPECT_NEAR(t.integral().z, -0.5 / g.ki * 1.0, 1e-9);
  const double proportional = g.kp * 0.5;
  EXPECT_NEAR(s.force.z, kGravity + proportional + 0.5, 1e-9);
  t.reset();
  EXPECT_EQ(t.integral(), Vec3{});
}

TEST(Tracking, SingularityDoesNotProduceNaN) {
  // Force exactly along x_C (a 90-degree tilt demanded of a hovering aircraft).
  TrackingGains g;
  g.max_tilt_rad = 1.5;
  Tracker big(g);
  FlatReference r = hold({0, 0, 0});
  r.acceleration = {1e6, 0, -kGravity + 1e-9};
  const Setpoint s = big.update(level_at({0, 0, 0}), r, 0.01);
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(std::isfinite(s.omega[i]));
  const MultirotorRaw raw = to_demands(s, DemandMap{});
  for (double v : {raw.roll_rate, raw.pitch_rate, raw.yaw_rate, raw.collective})
    EXPECT_TRUE(std::isfinite(v));
}

TEST(Tracking, DemandsAreClampedAndNeverNaN) {
  Setpoint s;
  s.omega = {100.0, -100.0, 5.0};
  s.thrust = 1e9;
  MultirotorRaw raw = to_demands(s, DemandMap{});
  EXPECT_EQ(raw.roll_rate, 1.0);
  EXPECT_EQ(raw.pitch_rate, 1.0);
  EXPECT_EQ(raw.yaw_rate, -1.0);
  EXPECT_EQ(raw.collective, 1.0);
  s.thrust = -3.0;
  EXPECT_EQ(to_demands(s, DemandMap{}).collective, 0.0);
  s.omega = {std::nan(""), 0, 0};
  s.thrust = std::nan("");
  raw = to_demands(s, DemandMap{});
  EXPECT_EQ(raw.roll_rate, 0.0);
  EXPECT_EQ(raw.collective, DemandMap{}.hover_collective);
}

TEST(Tracking, CollectiveFollowsTheThrustModel) {
  const DemandMap m;
  EXPECT_NEAR(collective_for(kGravity, m), m.hover_collective, 1e-15);
  EXPECT_NEAR(collective_for(kGravity * 2.0, m),
              m.hover_collective * std::pow(2.0, 1.0 / m.thrust_exponent), 1e-12);
  EXPECT_EQ(collective_for(0.0, m), 0.0);
}

}  // namespace
}  // namespace astrum::minsnap
