// Plain-data controller types: what a law reads (VehicleState) and returns (Command).
//
// These mirror the wire schema v2 (crates/astrum-proto/schema/astrum.capnp) field for
// field, flattened and in SI units, so control code never touches Cap'n Proto. The field
// names match the Python client (controllers/python/astrum_client/wire.py, VehicleState).

#ifndef ASTRUM_TYPES_HPP
#define ASTRUM_TYPES_HPP

#include <cstdint>
#include <string>
#include <variant>

namespace astrum {

// Flight-vehicle class. Values outside the named enumerants are kinds this build does
// not know (a newer schema); a raw law must refuse them.
enum class VehicleKind : std::uint16_t { FixedWing = 0, Multirotor = 1 };

// Who flies the aircraft this tick (Telemetry.controlMode). Unnamed values: newer schema.
enum class ControlMode : std::uint16_t { Uncommanded = 0, Raw = 1, Autopilot = 2, Failsafe = 3 };

// "fixed-wing" / "multirotor" / "unknown(<n>)".
std::string to_string(VehicleKind kind);
// "uncommanded" / "raw" / "autopilot" / "failsafe" / "unknown(<n>)".
std::string to_string(ControlMode mode);

// One aircraft's state after sim step `tick`. Angles rad, distances m, speeds m/s.
struct VehicleState {
  double t = 0.0;                    // sim time (s) = tick * dt
  std::uint64_t tick = 0;            // sim step index (echoed as Command.telemetryTick)
  std::uint32_t aircraft_id = 0;     //
  double lat_deg = 0.0;              // geodetic latitude
  double lon_deg = 0.0;              // geodetic longitude
  double alt_m = 0.0;                // height above the WGS84 ellipsoid (HAE)
  double e = 0.0, n = 0.0, u = 0.0;  // world-grid ENU position
  double vel_ned_n = 0.0, vel_ned_e = 0.0, vel_ned_d = 0.0;  // local-level NED velocity
  double vel_enu_e = 0.0, vel_enu_n = 0.0, vel_enu_u = 0.0;  // world-grid ENU velocity
  double roll = 0.0;                                         // phi, +right wing down
  double pitch = 0.0;                                        // theta, +nose up
  double yaw = 0.0;                                          // psi, true heading (0 = North, +East)
  double p = 0.0, q = 0.0, r = 0.0;  // body rates (rad/s), x nose / y starboard / z down
  double airspeed = 0.0;             // true airspeed
  double alpha = 0.0;                // angle of attack
  double beta = 0.0;                 // sideslip
  double nz_g = 0.0;                 // normal load factor (~ +1 in level flight)
  double mach = 0.0;                 //
  std::uint8_t fault_flags = 0;      // 0 = nominal
  VehicleKind vehicle_kind = VehicleKind::FixedWing;
  ControlMode control_mode = ControlMode::Uncommanded;
  std::uint8_t team = 0;
  float hit_points = 0.0F;
  float hit_points_max = 0.0F;
  bool alive = false;
  std::uint16_t ammo = 0;
};

// The setpoint a law tracks (from the CLI): true heading, HAE altitude, TAS.
struct Reference {
  double heading_rad = 0.0;
  double alt_m = 0.0;
  double airspeed_mps = 0.0;
};

// Raw actuator setpoints; the client clamps them to aileron/elevator/rudder [-1, 1] and
// throttle [0, 1] before sending. Signs (measured in the sim): aileron+ rolls right,
// elevator+ pitches nose DOWN.
struct RawActuators {
  double aileron = 0.0;
  double elevator = 0.0;
  double rudder = 0.0;
  double throttle = 0.0;
};

// Targets for the sim's built-in inner loop (tas_mps is ignored by multirotors).
struct AutopilotTargets {
  double heading_rad = 0.0;
  double alt_m = 0.0;
  double tas_mps = 0.0;
};

// What a controller sends for one telemetry sample.
struct Command {
  std::variant<RawActuators, AutopilotTargets> control;
  bool trigger = false;  // gun trigger (armed aircraft only)

  static Command raw(double aileron, double elevator, double rudder, double throttle) {
    Command c;
    c.control = RawActuators{aileron, elevator, rudder, throttle};
    return c;
  }
  static Command autopilot(double heading_rad, double alt_m, double tas_mps) {
    Command c;
    c.control = AutopilotTargets{heading_rad, alt_m, tas_mps};
    return c;
  }
};

// A Command as it travels on the wire, with the transport fields the client fills in.
struct CommandMessage {
  std::uint32_t target_id = 0;
  std::uint64_t seq = 0;             // per-sender, strictly increasing from 1
  std::uint64_t telemetry_tick = 0;  // Telemetry.tick the command answers
  Command command;
};

}  // namespace astrum

#endif  // ASTRUM_TYPES_HPP
