// In-process closed loop: the law flies a point-mass-plus-attitude model of the F450
// through a course. The model takes exactly what the sim takes — normalised body-rate
// demands (first-order rate loop, 0.75 rad/s per unit) and a collective (thrust a power
// law of it) — and reports exactly what the sim reports (FRD Euler angles and rates,
// ENU position and velocity), so the whole chain is exercised: telemetry -> frames ->
// tracking -> demands -> vehicle. It is a model: flying the real F450 in astrum-race is
// a separate check.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "flatness.hpp"
#include "frames.hpp"
#include "race_law.hpp"
#include "support.hpp"

namespace astrum::minsnap {
namespace {

using testing::make_course;
using testing::make_gate;
using testing::wire_course;

constexpr double kDt = 0.01;

Mat3 exp_map(const Vec3& w, double dt) {
  const double n = norm(w) * dt;
  if (n < 1e-12) return Mat3{};
  const Vec3 axis = w / norm(w);
  const double c = std::cos(n), s = std::sin(n);
  Mat3 r;
  for (int i = 0; i < 3; ++i) {
    Vec3 e;
    e[i] = 1.0;
    r.col[static_cast<std::size_t>(i)] =
        e * c + cross(axis, e) * s + axis * (dot(axis, e) * (1.0 - c));
  }
  return r;
}

struct Quad {
  Vec3 p, v;
  Mat3 rot;                         // FLU -> ENU
  Vec3 omega;                       // body rate, FLU
  double hover_collective = 0.408;  // the F450's, measured (the law's default)
  double exponent = 1.8;
  double rate_tau_s = 0.04;
  double drag = 0.007;  // m/s^2 per (m/s)^2, ~ the F450's flat-plate drag

  void step(const MultirotorRaw& cmd, double dt) {
    const Vec3 target{0.75 * cmd.roll_rate, -0.75 * cmd.pitch_rate, -0.75 * cmd.yaw_rate};
    omega += (target - omega) * (1.0 - std::exp(-dt / rate_tau_s));
    const double thrust =
        kGravity * std::pow(std::max(cmd.collective, 0.0) / hover_collective, exponent);
    const Vec3 a = rot.col[2] * thrust - kUnitZ * kGravity - v * (norm(v) * drag);
    p += v * dt + a * (0.5 * dt * dt);
    v += a * dt;
    rot = rot * exp_map(omega, dt);
  }

  // What the sim's telemetry would say.
  VehicleState telemetry(double t) const {
    // FRD axes in NED: x_f = forward, y_s = right (-left), z_d = down (-up); NED = (N, E, -U).
    auto ned = [](const Vec3& enu) { return Vec3{enu.y, enu.x, -enu.z}; };
    const Vec3 xf = ned(rot.col[0]), ys = ned(-rot.col[1]), zd = ned(-rot.col[2]);
    VehicleState s;
    s.aircraft_id = 1;
    s.sim_time_s = t;
    s.control_period_s = kDt;
    s.e = p.x;
    s.n = p.y;
    s.u = p.z;
    s.vel_enu_e = v.x;
    s.vel_enu_n = v.y;
    s.vel_enu_u = v.z;
    s.roll = std::atan2(ys.z, zd.z);
    s.pitch = -std::asin(std::clamp(xf.z, -1.0, 1.0));
    s.yaw = std::atan2(xf.y, xf.x);
    s.p = omega.x;
    s.q = -omega.y;
    s.r = -omega.z;
    s.vehicle_kind = VehicleKind::Multirotor;
    s.alive = true;
    return s;
  }
};

Quad hovering_at(const Vec3& p, double heading_rad) {
  Quad q;
  q.p = p;
  q.rot = body_to_world(0.0, 0.0, heading_rad);
  return q;
}

struct Crossing {
  std::size_t gate;
  double right_m, up_m, time_s, speed_mps;
};

struct Flight {
  std::vector<Crossing> crossings;
  double time_s = 0.0;
  bool finished = false;
  std::size_t plans = 0;
  double peak_tilt_rad = 0.0;
  std::size_t out_of_order = 0;
};

// Called after every step with the clock and the vehicle: lets a test disturb the flight.
using Disturbance = std::function<void(double t, Quad& quad)>;

// Flies `law` until every gate is passed or `limit_s`. Crossings of the pending gate's
// plane are scored the way the race environment does: whichever the offset, the gate
// is consumed and the offset recorded for the test to judge.
Flight fly(RaceLaw& law, Quad quad, const Course& course, double limit_s,
           const Disturbance& disturb = nullptr) {
  Flight f;
  RaceState race;
  race.gate_count = static_cast<std::uint32_t>(course.gates.size());
  std::vector<double> side(course.gates.size());
  auto plane = [&](std::size_t g, const Vec3& p) {
    return dot(p - course.gates[g].center, course.gates[g].normal);
  };
  for (std::size_t g = 0; g < course.gates.size(); ++g) side[g] = plane(g, quad.p);

  bool first = true;
  for (double t = 0.0; t < limit_s; t += kDt) {
    StepInput in;
    in.state = quad.telemetry(t);
    in.control_period_s = kDt;
    if (first) {
      in.race_course = wire_course(course);
      first = false;
    }
    race.sim_time_s = t;
    in.race_state = race;
    const LawResult r = law.step(in);
    const auto* action = std::get_if<Action>(&r);
    if (action == nullptr) {
      ADD_FAILURE() << "law stopped: " << std::get<Stop>(r).reason;
      break;
    }
    const Vec3 before = quad.p;
    quad.step(std::get<MultirotorRaw>(action->control), kDt);
    if (disturb) disturb(t, quad);
    f.peak_tilt_rad =
        std::max(f.peak_tilt_rad, std::acos(std::clamp(quad.rot.col[2].z, -1.0, 1.0)));

    for (std::size_t g = 0; g < course.gates.size(); ++g) {
      const double now = plane(g, quad.p);
      if (side[g] < 0.0 && now >= 0.0) {
        // Crossing: interpolate to the plane and read the offset in the gate frame.
        const double a = side[g] / (side[g] - now);
        const Vec3 rel = before + (quad.p - before) * a - course.gates[g].center;
        if (g == race.pending_gate) {
          f.crossings.push_back(
              {g, dot(rel, course.gates[g].right), dot(rel, course.gates[g].up), t, norm(quad.v)});
          ++race.passed;
          ++race.pending_gate;
        } else {
          ++f.out_of_order;
        }
      }
      side[g] = now;
    }
    if (race.pending_gate == course.gates.size()) {
      f.finished = true;
      f.time_s = t;
      break;
    }
  }
  f.plans = law.plans_made();
  return f;
}

Course slalom() {
  return make_course({make_gate({0.0, 6.0, 2.0}, 0.0), make_gate({4.0, 12.0, 2.5}, 40.0),
                      make_gate({10.0, 14.0, 1.6}, 90.0), make_gate({16.0, 10.0, 2.2}, 150.0)});
}

// A climbing, descending 3-D course with a hairpin.
Course roller_coaster() {
  return make_course({make_gate({0.0, 5.0, 2.0}, 0.0), make_gate({0.0, 11.0, 3.4}, 0.0),
                      make_gate({4.0, 16.0, 2.4}, 60.0), make_gate({9.0, 17.0, 1.2}, 120.0),
                      make_gate({10.0, 11.0, 1.8}, 180.0)});
}

// Every gate passed, within `reach` of its centre in both directions (the opening is
// +-0.508 m and the F450 reaches 0.285 m square on, so 0.223 m is the edge of a crash).
void expect_clean(const Flight& f, const Course& course, double reach) {
  ASSERT_TRUE(f.finished);
  ASSERT_EQ(f.crossings.size(), course.gates.size());
  for (const Crossing& c : f.crossings) {
    EXPECT_LT(std::fabs(c.right_m), reach) << "gate " << c.gate;
    EXPECT_LT(std::fabs(c.up_m), reach) << "gate " << c.gate;
  }
  EXPECT_EQ(f.out_of_order, 0u);
}

TEST(ClosedLoop, FliesAMultiGateCourseThroughTheCentres) {
  const Course course = slalom();
  RaceLaw law(RaceLawParams{});
  const Flight f = fly(law, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 60.0);
  expect_clean(f, course, 0.05);
  EXPECT_EQ(f.plans, 1u);  // one plan was enough: no re-planning
  for (const Crossing& c : f.crossings) EXPECT_GT(c.speed_mps, 0.5) << "gate " << c.gate;
  EXPECT_LT(f.peak_tilt_rad, 0.6);
}

TEST(ClosedLoop, FliesAClimbingAndDescendingCourse) {
  const Course course = roller_coaster();
  RaceLaw law(RaceLawParams{});
  const Flight f = fly(law, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  expect_clean(f, course, 0.08);
}

TEST(ClosedLoop, RecoversFromAnInitialOffsetAndVelocity) {
  const Course course = slalom();
  RaceLaw law(RaceLawParams{});
  Quad quad = hovering_at({0.3, -0.4, 2.1}, 0.3);
  quad.v = {0.5, 0.8, 0.1};
  const Flight f = fly(law, quad, course, 60.0);
  expect_clean(f, course, 0.1);
}

TEST(ClosedLoop, HoldsTheLineWithAWrongHoverCollective) {
  // The airframe needs 0.44 where the controller assumes 0.408: the integral term and the
  // position loop take up the 14 % thrust error. The heuristic plan (6 m/s through the
  // gates) leaves them the time to. (This was 0.48 against 0.44, 17 %, before the hover
  // collective default became the measured 0.408; the integral limit, 2 m/s^2, is 20 % of
  // gravity, so errors much beyond 15 % are not this test's.)
  const Course course = slalom();
  RaceLawParams params;
  params.plan.planner = PlannerKind::Heuristic;
  RaceLaw law(params);
  Quad quad = hovering_at({0.0, 0.0, 2.0}, 0.0);
  quad.hover_collective = 0.44;
  const Flight f = fly(law, quad, course, 60.0);
  expect_clean(f, course, 0.15);
}

TEST(ClosedLoop, TimeOptimalPlanHoldsTheLineWithAWrongHoverCollective) {
  // The time-optimal plan reaches the first gate sooner, so the integrator has less time:
  // a 10 % thrust error (0.43 where 0.408 is assumed) is held to the same 0.15 m.
  const Course course = slalom();
  RaceLaw law{RaceLawParams{}};
  Quad quad = hovering_at({0.0, 0.0, 2.0}, 0.0);
  quad.hover_collective = 0.43;
  const Flight f = fly(law, quad, course, 60.0);
  expect_clean(f, course, 0.15);
}

TEST(ClosedLoop, ReplansAfterAShove) {
  const Course course = slalom();
  RaceLaw law(RaceLawParams{});
  const Disturbance shove = [](double t, Quad& q) {
    if (t > 1.0 && t < 1.0 + kDt) q.p += Vec3{3.0, -1.0, 0.0};  // carried 3 m off the line
  };
  const Flight f = fly(law, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0, shove);
  EXPECT_TRUE(f.finished);
  EXPECT_GE(f.plans, 2u);
  // The first gate is scored on whatever the shove left; the rest are flown cleanly.
  ASSERT_EQ(f.crossings.size(), course.gates.size());
  for (std::size_t g = 1; g < f.crossings.size(); ++g) {
    EXPECT_LT(std::fabs(f.crossings[g].right_m), 0.15) << "gate " << g;
    EXPECT_LT(std::fabs(f.crossings[g].up_m), 0.15) << "gate " << g;
  }
}

TEST(ClosedLoop, TighterLimitsFlySlowerAndStillCleanly) {
  // (The defaults are the airframe's true limits, so "looser" no longer exists: this is the
  // same check the other way round.)
  const Course course = slalom();
  RaceLawParams tight;
  tight.plan.max_rate_rad_s = 0.5;
  tight.plan.max_speed_mps = 8.0;
  tight.plan.max_tilt_rad = 0.6;
  tight.plan.max_yaw_rate = 0.5;
  RaceLaw a{RaceLawParams{}}, b(tight);
  const Flight defaults = fly(a, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  const Flight slow = fly(b, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  expect_clean(defaults, course, 0.05);
  expect_clean(slow, course, 0.05);
  EXPECT_GT(slow.time_s, defaults.time_s);
  EXPECT_LT(slow.peak_tilt_rad, 0.6 + 0.1);
}

TEST(ClosedLoop, TimeOptimalPlannerFliesFasterThanTheHeuristicOne) {
  const Course course = slalom();
  RaceLawParams heuristic;
  heuristic.plan.planner = PlannerKind::Heuristic;
  RaceLaw a{RaceLawParams{}}, b(heuristic);
  const Flight optimal = fly(a, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  const Flight cruise = fly(b, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  expect_clean(optimal, course, 0.05);
  expect_clean(cruise, course, 0.05);
  EXPECT_LT(optimal.time_s, 0.85 * cruise.time_s);
  // Within what the planner was told: the thrust tilts no further than the tracker's limit.
  EXPECT_LT(optimal.peak_tilt_rad, RaceLawParams{}.gains.max_tilt_rad);
  EXPECT_EQ(optimal.plans, 1u);
}

TEST(ClosedLoop, TimeScaleTradesSpeedForMargin) {
  const Course course = slalom();
  RaceLawParams slow;
  slow.plan.time_scale = 3.0;
  RaceLaw a{RaceLawParams{}}, b(slow);
  const Flight base = fly(a, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 90.0);
  const Flight stretched = fly(b, hovering_at({0.0, 0.0, 2.0}, 0.0), course, 120.0);
  expect_clean(stretched, course, 0.05);
  EXPECT_GT(stretched.time_s, 1.3 * base.time_s);
  EXPECT_LT(stretched.peak_tilt_rad, base.peak_tilt_rad);
}

}  // namespace
}  // namespace astrum::minsnap
