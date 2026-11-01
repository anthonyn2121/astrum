#include "astrum/client.hpp"

#include <gtest/gtest.h>

namespace astrum {
namespace {

VehicleState of_kind(VehicleKind kind) {
  VehicleState s;
  s.aircraft_id = 2;
  s.vehicle_kind = kind;
  return s;
}

TEST(CheckVehicleKind, AcceptsTheLawsKind) {
  EXPECT_FALSE(check_vehicle_kind(of_kind(VehicleKind::FixedWing), VehicleKind::FixedWing, "law"));
}

TEST(CheckVehicleKind, RefusesOtherAndUnknownKindsWithExitTwo) {
  for (const VehicleKind kind : {VehicleKind::Multirotor, static_cast<VehicleKind>(5)}) {
    const auto stop = check_vehicle_kind(of_kind(kind), VehicleKind::FixedWing, "law");
    ASSERT_TRUE(stop);
    EXPECT_EQ(stop->exit_code, 2);
    EXPECT_NE(stop->reason.find(to_string(kind)), std::string::npos);
  }
}

}  // namespace
}  // namespace astrum
