#include "astrum/wire.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace astrum {
namespace {

VehicleState sample_state() {
  VehicleState s;
  s.t = 12.5;
  s.tick = 1500;
  s.aircraft_id = 4;
  s.lat_deg = 37.6;
  s.lon_deg = -122.4;
  s.alt_m = 1524.0;
  s.e = 10.0;
  s.n = -20.0;
  s.u = 1500.0;
  s.vel_ned_n = 30.0;
  s.vel_ned_e = 40.0;
  s.vel_ned_d = -1.0;
  s.vel_enu_e = 40.5;
  s.vel_enu_n = 29.5;
  s.vel_enu_u = 1.0;
  s.roll = 0.1;
  s.pitch = 0.05;
  s.yaw = 1.2;
  s.p = 0.01;
  s.q = -0.02;
  s.r = 0.03;
  s.airspeed = 55.0;
  s.alpha = 0.04;
  s.beta = -0.001;
  s.nz_g = 1.02;
  s.mach = 0.16;
  s.fault_flags = 1;
  s.vehicle_kind = VehicleKind::Multirotor;
  s.control_mode = ControlMode::Failsafe;
  s.team = 1;
  s.hit_points = 75.0F;
  s.hit_points_max = 100.0F;
  s.alive = true;
  s.ammo = 200;
  return s;
}

TEST(Wire, GarbageBytesAreAnErrorNotAThrow) {
  std::mt19937 rng(7);
  for (int i = 0; i < 500; ++i) {
    std::vector<std::uint8_t> bytes(8 * (1 + i % 16));
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    const auto telemetry = decode_telemetry(bytes);
    const auto command = decode_command(bytes);
    // Reaching here means no exception escaped; each result is a value XOR an error.
    EXPECT_EQ(telemetry.value.has_value(), telemetry.error.empty());
    EXPECT_EQ(command.value.has_value(), command.error.empty());
  }
  // A segment table claiming a huge segment is the classic hostile header.
  const std::vector<std::uint8_t> hostile = {0, 0, 0, 0, 0xff, 0xff, 0xff, 0x7f};
  const auto decoded = decode_telemetry(hostile);
  EXPECT_FALSE(decoded);
  EXPECT_FALSE(decoded.error.empty());
}

TEST(Wire, RejectsEmptyAndPartialWordPayloads) {
  EXPECT_FALSE(decode_telemetry({}));
  const std::vector<std::uint8_t> bytes = encode_telemetry(sample_state());
  const std::vector<std::uint8_t> partial(bytes.begin(), bytes.end() - 3);
  EXPECT_FALSE(decode_telemetry(partial));
}

TEST(Wire, RejectsTruncatedMessage) {
  const std::vector<std::uint8_t> bytes = encode_telemetry(sample_state());
  const std::vector<std::uint8_t> truncated(bytes.begin(), bytes.end() - 8);
  EXPECT_FALSE(decode_telemetry(truncated));
}

TEST(Wire, RejectsTrailingBytes) {
  std::vector<std::uint8_t> bytes = encode_telemetry(sample_state());
  bytes.resize(bytes.size() + 8, 0);
  const auto decoded = decode_telemetry(bytes);
  EXPECT_FALSE(decoded);
  EXPECT_NE(decoded.error.find("trailing"), std::string::npos);
}

TEST(Wire, TelemetryRoundTripsEveryField) {
  const VehicleState s = sample_state();
  const auto decoded = decode_telemetry(encode_telemetry(s));
  ASSERT_TRUE(decoded) << decoded.error;
  const VehicleState& d = *decoded.value;
  EXPECT_EQ(d.t, s.t);
  EXPECT_EQ(d.tick, s.tick);
  EXPECT_EQ(d.aircraft_id, s.aircraft_id);
  EXPECT_EQ(d.lat_deg, s.lat_deg);
  EXPECT_EQ(d.lon_deg, s.lon_deg);
  EXPECT_EQ(d.alt_m, s.alt_m);
  EXPECT_EQ(d.e, s.e);
  EXPECT_EQ(d.n, s.n);
  EXPECT_EQ(d.u, s.u);
  EXPECT_EQ(d.vel_ned_n, s.vel_ned_n);
  EXPECT_EQ(d.vel_ned_e, s.vel_ned_e);
  EXPECT_EQ(d.vel_ned_d, s.vel_ned_d);
  EXPECT_EQ(d.vel_enu_e, s.vel_enu_e);
  EXPECT_EQ(d.vel_enu_n, s.vel_enu_n);
  EXPECT_EQ(d.vel_enu_u, s.vel_enu_u);
  EXPECT_EQ(d.roll, s.roll);
  EXPECT_EQ(d.pitch, s.pitch);
  EXPECT_EQ(d.yaw, s.yaw);
  EXPECT_EQ(d.p, s.p);
  EXPECT_EQ(d.q, s.q);
  EXPECT_EQ(d.r, s.r);
  EXPECT_EQ(d.airspeed, s.airspeed);
  EXPECT_EQ(d.alpha, s.alpha);
  EXPECT_EQ(d.beta, s.beta);
  EXPECT_EQ(d.nz_g, s.nz_g);
  EXPECT_EQ(d.mach, s.mach);
  EXPECT_EQ(d.fault_flags, s.fault_flags);
  EXPECT_EQ(d.vehicle_kind, s.vehicle_kind);
  EXPECT_EQ(d.control_mode, s.control_mode);
  EXPECT_EQ(d.team, s.team);
  EXPECT_EQ(d.hit_points, s.hit_points);
  EXPECT_EQ(d.hit_points_max, s.hit_points_max);
  EXPECT_EQ(d.alive, s.alive);
  EXPECT_EQ(d.ammo, s.ammo);
}

TEST(Wire, DecodesFromAnUnalignedBuffer) {
  const std::vector<std::uint8_t> bytes = encode_telemetry(sample_state());
  std::vector<std::uint8_t> shifted(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), shifted.begin() + 1);
  const auto decoded = decode_telemetry(std::span(shifted).subspan(1));
  ASSERT_TRUE(decoded) << decoded.error;
  EXPECT_EQ(decoded.value->yaw, 1.2);
}

TEST(Wire, UnknownEnumValuesSurviveAsUnknown) {
  VehicleState s = sample_state();
  s.vehicle_kind = static_cast<VehicleKind>(7);
  s.control_mode = static_cast<ControlMode>(9);
  const auto decoded = decode_telemetry(encode_telemetry(s));
  ASSERT_TRUE(decoded) << decoded.error;
  EXPECT_EQ(to_string(decoded.value->vehicle_kind), "unknown(7)");
  EXPECT_EQ(to_string(decoded.value->control_mode), "unknown(9)");
}

TEST(Wire, CommandRoundTripsBothControlVariants) {
  CommandMessage raw{3, 42, 1000, Command::raw(0.25, -0.5, 0.125, 0.75)};
  raw.command.trigger = true;
  const auto r = decode_command(encode_command(raw));
  ASSERT_TRUE(r) << r.error;
  EXPECT_EQ(r.value->target_id, 3U);
  EXPECT_EQ(r.value->seq, 42U);
  EXPECT_EQ(r.value->telemetry_tick, 1000U);
  EXPECT_TRUE(r.value->command.trigger);
  const auto* act = std::get_if<RawActuators>(&r.value->command.control);
  ASSERT_NE(act, nullptr);
  EXPECT_EQ(act->aileron, 0.25);
  EXPECT_EQ(act->elevator, -0.5);
  EXPECT_EQ(act->rudder, 0.125);
  EXPECT_EQ(act->throttle, 0.75);

  const CommandMessage ap{9, 7, 5, Command::autopilot(1.5, 1500.0, 120.0)};
  const auto a = decode_command(encode_command(ap));
  ASSERT_TRUE(a) << a.error;
  EXPECT_FALSE(a.value->command.trigger);
  const auto* targets = std::get_if<AutopilotTargets>(&a.value->command.control);
  ASSERT_NE(targets, nullptr);
  EXPECT_EQ(targets->heading_rad, 1.5);
  EXPECT_EQ(targets->alt_m, 1500.0);
  EXPECT_EQ(targets->tas_mps, 120.0);
}

}  // namespace
}  // namespace astrum
