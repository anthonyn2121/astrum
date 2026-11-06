#include "astrum/wire.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace astrum {
namespace {

// Every field distinct and non-default, so a swapped or dropped field cannot round-trip.
VehicleState sample_state() {
  VehicleState s;
  s.sim_time_s = 12.5;
  s.tick = 1500;
  s.aircraft_id = 4;
  s.lat_deg = 37.6;
  s.lon_deg = -122.4;
  s.alt_hae_m = 1524.0;
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
  s.tas_mps = 55.0;
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
  s.control_period_s = 1.0 / 30.0;
  s.command = CommandStatus{0xAAAAAAAAAAAAAAAAULL, 41, 4294967295U, Refusal::NotOwner, 7};
  return s;
}

TEST(Wire, GarbageBytesAreAnErrorNotAThrow) {
  std::mt19937 rng(7);
  for (int i = 0; i < 500; ++i) {
    std::vector<std::uint8_t> bytes(8 * (1 + i % 16));
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    const auto telemetry = decode_telemetry(bytes);
    const auto command = decode_command(bytes);
    const auto event = decode_event(bytes);
    const auto tick_end = decode_tick_end(bytes);
    // Reaching here means no exception escaped; each result is a value XOR an error.
    EXPECT_EQ(telemetry.value.has_value(), telemetry.error.empty());
    EXPECT_EQ(command.value.has_value(), command.error.empty());
    EXPECT_EQ(event.value.has_value(), event.error.empty());
    EXPECT_EQ(tick_end.value.has_value(), tick_end.error.empty());
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
  EXPECT_TRUE(*decoded.value == s);
  EXPECT_EQ(decoded.value->control_period_s, 1.0 / 30.0);
  EXPECT_EQ(decoded.value->command.owner_epoch, 0xAAAAAAAAAAAAAAAAULL);
  EXPECT_EQ(decoded.value->command.refusals, 4294967295U);
  EXPECT_EQ(decoded.value->command.last_refusal, Refusal::NotOwner);
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
  s.command.last_refusal = static_cast<Refusal>(11);
  const auto decoded = decode_telemetry(encode_telemetry(s));
  ASSERT_TRUE(decoded) << decoded.error;
  EXPECT_EQ(to_string(decoded.value->vehicle_kind), "unknown(7)");
  EXPECT_EQ(to_string(decoded.value->control_mode), "unknown(9)");
  EXPECT_EQ(to_string(decoded.value->command.last_refusal), "unknown(11)");

  const Event e{3, 0.5,
                ModeChange{1, ControlMode::Raw, ControlMode::Failsafe, static_cast<ModeReason>(6)}};
  const auto ev = decode_event(encode_event(e));
  ASSERT_TRUE(ev) << ev.error;
  EXPECT_EQ(to_string(std::get<ModeChange>(ev.value->detail).reason), "unknown(6)");
  const auto end =
      decode_event(encode_event(Event{3, 0.5, RunEnded{static_cast<EndReason>(9), ""}}));
  ASSERT_TRUE(end) << end.error;
  EXPECT_EQ(to_string(std::get<RunEnded>(end.value->detail).reason), "unknown(9)");
}

TEST(Wire, EnumNamesAreKebabCase) {
  EXPECT_EQ(to_string(VehicleKind::FixedWing), "fixed-wing");
  EXPECT_EQ(to_string(Refusal::FromTheFuture), "from-the-future");
  EXPECT_EQ(to_string(Refusal::WrongVehicleKind), "wrong-vehicle-kind");
  EXPECT_EQ(to_string(ModeReason::LeaseExpired), "lease-expired");
  EXPECT_EQ(to_string(EndReason::EngagementDecided), "engagement-decided");
}

TEST(Wire, CommandRoundTripsEveryControlVariant) {
  const Command raw{
      3, 42, 1000, true, 0xFFFFFFFFFFFFFFFFULL, RawActuators{0.25, -0.5, 0.125, 0.75}};
  const Command multirotor{4, 9, 33, false, 5, MultirotorRaw{0.25, -0.5, 0.125, 0.4375}};
  const Command autopilot{9, 7, 5, false, 1, AutopilotTargets{1.5, 1500.0, 120.0}};
  for (const Command& c : {raw, multirotor, autopilot}) {
    const auto decoded = decode_command(encode_command(c));
    ASSERT_TRUE(decoded) << decoded.error;
    EXPECT_TRUE(*decoded.value == c) << "control index " << c.control.index();
  }
}

TEST(Wire, TickEndRoundTrips) {
  const TickEnd t{1234567890123ULL, 10288065.6875};
  const auto decoded = decode_tick_end(encode_tick_end(t));
  ASSERT_TRUE(decoded) << decoded.error;
  EXPECT_TRUE(*decoded.value == t);
}

TEST(Wire, EveryEventKindRoundTrips) {
  const std::vector<Event> events = {
      {120, 1.0, TriggerPressed{2, 300}},
      {121, 1.0083, TriggerReleased{2, 41, 259}},
      {1234, 10.28125, Hit{2, 1, 2.5F, 97.5F, Enu{1234.5, -6789.25, 4572.125}}},
      {4000, 33.3125, Kill{1, 2U, Enu{-10.75, 20.5, 1499.0}}},
      {17, 0.140625, Kill{4294967295U, std::nullopt, Enu{}}},
      {600, 5.0, ModeChange{3, ControlMode::Raw, ControlMode::Failsafe, ModeReason::LeaseExpired}},
      {0, 0.0, ControllerUp{5}},
      {999, 8.3125, ControllerDown{7}},
      {7200, 60.0, RunEnded{EndReason::Fatal, "aircraft 3: non-finite state"}},
  };
  for (const Event& e : events) {
    const auto decoded = decode_event(encode_event(e));
    ASSERT_TRUE(decoded) << decoded.error;
    EXPECT_TRUE(*decoded.value == e) << "event kind " << e.detail.index();
  }
}

}  // namespace
}  // namespace astrum
