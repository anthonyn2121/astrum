// Plain-data ASTRUM types: what a law reads (VehicleState, Event) and returns (Action),
// and the wire messages (Command, TickEnd) — no Cap'n Proto.
//
// They mirror the wire schema v3 (crates/astrum-proto/schema/astrum.capnp) field for
// field, flattened and in SI units (angles in radians). The names are the SDK contract
// shared with the Python client and the Rust mirror (controllers/README.md, "Names").
// Enum values outside the named enumerants come from a newer schema: they are kept as
// is (to_string gives "unknown(<n>)") and never mapped onto a known value.

#ifndef ASTRUM_TYPES_HPP
#define ASTRUM_TYPES_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace astrum {

// Flight-vehicle class; raw control is class-specific (RawActuators: fixed-wing,
// MultirotorRaw: multirotor).
enum class VehicleKind : std::uint16_t { FixedWing = 0, Multirotor = 1 };

// Who flies the aircraft this tick (Telemetry.controlMode).
enum class ControlMode : std::uint16_t { Uncommanded = 0, Raw = 1, Autopilot = 2, Failsafe = 3 };

// Why the sim refused a command (CommandStatus.lastRefusal).
enum class Refusal : std::uint16_t {
  None = 0,
  FromTheFuture = 1,
  Stale = 2,
  OutOfOrder = 3,
  NotOwner = 4,
  MissingEpoch = 5,
  NonFinite = 6,
  WrongVehicleKind = 7,
};

// Why an aircraft's control mode changed (ModeChange.reason).
enum class ModeReason : std::uint16_t { Command = 0, LeaseExpired = 1, ControllerLost = 2 };

// Why a run ended (RunEnded.reason).
enum class EndReason : std::uint16_t {
  MaxTime = 0,
  EngagementDecided = 1,
  Fatal = 2,
  Stopped = 3,
  RaceComplete = 4,
  Crashed = 5,
};

// How a race ended for an aircraft (RaceState.outcome, Crashed.outcome).
enum class RaceOutcome : std::uint16_t {
  Running = 0,
  Complete = 1,
  CrashedGround = 2,
  CrashedGate = 3,
  TimedOut = 4,
  SpawnFailed = 5,
};

// Kebab-case names as in values.json ("fixed-wing", "not-owner", "max-time", ...);
// "unknown(<n>)" for a value this build does not know.
std::string to_string(VehicleKind kind);
std::string to_string(ControlMode mode);
std::string to_string(Refusal refusal);
std::string to_string(ModeReason reason);
std::string to_string(EndReason reason);
std::string to_string(RaceOutcome outcome);

// The sim's command bookkeeping for one aircraft (Telemetry.command).
struct CommandStatus {
  std::uint64_t owner_epoch = 0;  // epoch of the controller flying it; 0 = unowned
  std::uint64_t last_seq = 0;     // the owner's last applied seq; 0 = none
  std::uint32_t refusals = 0;     // commands refused so far (wraps)
  Refusal last_refusal = Refusal::None;
  std::uint64_t last_refused_epoch = 0;  // the epoch that sent the last refused command

  bool operator==(const CommandStatus&) const = default;
};

// One aircraft's state after sim step `tick` (one decoded Telemetry).
struct VehicleState {
  std::uint32_t aircraft_id = 0;     //
  std::uint64_t tick = 0;            // sim step index (echoed as Command.telemetry_tick)
  double sim_time_s = 0.0;           // tick * dt
  double lat_deg = 0.0;              // geodetic latitude
  double lon_deg = 0.0;              // geodetic longitude
  double alt_hae_m = 0.0;            // height above the WGS84 ellipsoid
  double e = 0.0, n = 0.0, u = 0.0;  // world-grid ENU position (m)
  double vel_ned_n = 0.0, vel_ned_e = 0.0, vel_ned_d = 0.0;  // local-level NED velocity
  double vel_enu_e = 0.0, vel_enu_n = 0.0, vel_enu_u = 0.0;  // world-grid ENU velocity
  double roll = 0.0;                                         // phi, +right wing down
  double pitch = 0.0;                                        // theta, +nose up
  double yaw = 0.0;                                          // psi, true heading (0 = North, +East)
  double p = 0.0, q = 0.0, r = 0.0;  // body rates (rad/s), x nose / y starboard / z down
  double tas_mps = 0.0;              // true airspeed
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
  double control_period_s = 0.0;  // sim time between this aircraft's samples: the law's dt
  CommandStatus command;          // ownership and command refusals

  bool operator==(const VehicleState&) const = default;
};

// The setpoint a law tracks (from --heading / --alt / --speed): true heading, HAE
// altitude, true airspeed.
struct Reference {
  double heading_rad = 0.0;
  double alt_hae_m = 0.0;
  double tas_mps = 0.0;

  bool operator==(const Reference&) const = default;
};

// Attitude command limits in degrees (--max-bank-deg / --max-pitch-deg), each in (0, 90).
// Defaults suit the trainer (c172x); the F-16 formation demo flies 60 / 24.
struct AttitudeLimits {
  double max_bank_deg = 25.0;
  double max_pitch_deg = 10.0;

  bool operator==(const AttitudeLimits&) const = default;
};

// Raw fixed-wing actuator setpoints: aileron / elevator / rudder in [-1, 1], throttle in
// [0, 1] (the loop clamps them). Signs: aileron+ rolls right, elevator+ pitches nose DOWN.
struct RawActuators {
  double aileron = 0.0;
  double elevator = 0.0;
  double rudder = 0.0;
  double throttle = 0.0;

  bool operator==(const RawActuators&) const = default;
};

// Raw multirotor demands: normalized body rates in [-1, 1] (+ rolls right, + pitches
// nose UP, + yaws nose right) and collective thrust in [0, 1].
struct MultirotorRaw {
  double roll_rate = 0.0;
  double pitch_rate = 0.0;
  double yaw_rate = 0.0;
  double collective = 0.0;

  bool operator==(const MultirotorRaw&) const = default;
};

// Targets for the sim's built-in inner loop (tas_mps is ignored by multirotors).
struct AutopilotTargets {
  double heading_rad = 0.0;
  double alt_hae_m = 0.0;
  double tas_mps = 0.0;

  bool operator==(const AutopilotTargets&) const = default;
};

using Control = std::variant<RawActuators, MultirotorRaw, AutopilotTargets>;

// What a law returns for one telemetry sample.
struct Action {
  Control control;
  bool trigger = false;  // gun trigger (armed aircraft only)

  bool operator==(const Action&) const = default;
};

// The wire command (astrum/command/<target_id>).
struct Command {
  std::uint32_t target_id = 0;
  std::uint64_t seq = 0;             // strictly increasing within one epoch, from 1
  std::uint64_t telemetry_tick = 0;  // Telemetry.tick the command answers
  bool trigger = false;
  std::uint64_t epoch = 0;  // random non-zero id of the sending publisher
  Control control;

  bool operator==(const Command&) const = default;
};

// World-grid vector (m).
struct Enu {
  double e = 0.0, n = 0.0, u = 0.0;

  bool operator==(const Enu&) const = default;
};

struct TriggerPressed {
  std::uint32_t aircraft_id = 0;
  std::uint16_t ammo = 0;  // rounds left when pressed

  bool operator==(const TriggerPressed&) const = default;
};

struct TriggerReleased {
  std::uint32_t aircraft_id = 0;
  std::uint16_t rounds_fired = 0;  // since the matching TriggerPressed
  std::uint16_t ammo = 0;          // rounds left

  bool operator==(const TriggerReleased&) const = default;
};

// A round struck an aircraft.
struct Hit {
  std::uint32_t shooter_id = 0;
  std::uint32_t target_id = 0;
  float damage = 0.0F;
  float hit_points = 0.0F;  // the target's, after this hit
  Enu pos_enu;              // where the round struck

  bool operator==(const Hit&) const = default;
};

// An aircraft was destroyed.
struct Kill {
  std::uint32_t victim_id = 0;
  std::optional<std::uint32_t> killer_id;  // nullopt: no hit recorded
  Enu pos_enu;                             // the victim's position

  bool operator==(const Kill&) const = default;
};

struct ModeChange {
  std::uint32_t aircraft_id = 0;
  ControlMode from = ControlMode::Uncommanded;
  ControlMode to = ControlMode::Uncommanded;
  ModeReason reason = ModeReason::Command;

  bool operator==(const ModeChange&) const = default;
};

// An aircraft's controller connected (its presence token appeared).
struct ControllerUp {
  std::uint32_t aircraft_id = 0;

  bool operator==(const ControllerUp&) const = default;
};

// An aircraft's controller went away.
struct ControllerDown {
  std::uint32_t aircraft_id = 0;

  bool operator==(const ControllerDown&) const = default;
};

struct RunEnded {
  EndReason reason = EndReason::MaxTime;
  std::string message;  // the fatal error; empty otherwise

  bool operator==(const RunEnded&) const = default;
};

// Race runs: the aircraft's CG crossed gate `gate`'s opening at (offset_x_m along the
// gate's `right`, offset_y_m along its `up`) from the gate centre.
struct GatePassed {
  std::uint32_t aircraft_id = 0;
  std::uint32_t gate = 0;
  double offset_x_m = 0.0;
  double offset_y_m = 0.0;

  bool operator==(const GatePassed&) const = default;
};

// Race runs: the aircraft crossed gate `gate`'s plane outside its opening (or skipped it).
struct GateMissed {
  std::uint32_t aircraft_id = 0;
  std::uint32_t gate = 0;

  bool operator==(const GateMissed&) const = default;
};

// Race runs: the aircraft crashed; `gate` is the gate it hit or was flying to.
struct Crashed {
  std::uint32_t aircraft_id = 0;
  RaceOutcome outcome = RaceOutcome::Running;
  std::uint32_t gate = 0;

  bool operator==(const Crashed&) const = default;
};

using EventDetail =
    std::variant<TriggerPressed, TriggerReleased, Hit, Kill, ModeChange, ControllerUp,
                 ControllerDown, RunEnded, GatePassed, GateMissed, Crashed>;

// Something that happened (astrum/events).
struct Event {
  std::uint64_t tick = 0;  // the Telemetry.tick that first shows it
  double sim_time_s = 0.0;
  EventDetail detail;

  bool operator==(const Event&) const = default;
};

// The end of a tick's publications (astrum/tick): every Telemetry and Event of `tick`
// that reached this subscriber arrived before it.
struct TickEnd {
  std::uint64_t tick = 0;
  double sim_time_s = 0.0;

  bool operator==(const TickEnd&) const = default;
};

// One race gate (RaceCourse.gates). The gate-local basis is LEFT-handed (x = right,
// y = up, z = normal), so use the explicit vectors rather than deriving them from yaw.
struct RaceGate {
  std::uint32_t index = 0;  // 0-based order in the course
  Enu center;               // world grid (m)
  double yaw_rad = 0.0;     // heading of `normal`, 0 = North, +East
  Enu normal;               // unit vector through the opening in the flight direction
  Enu right;                // unit vector, starboard of the flight direction
  Enu up;                   // unit vector, up

  bool operator==(const RaceGate&) const = default;
};

// A race course (astrum/race/course/<id>).
struct RaceCourse {
  std::uint64_t seed = 0;
  double opening_m = 0.0;  // square clear opening side
  double bar_m = 0.0;      // frame stock width
  double top_bar_m = 0.0;  // outer frame side (opening_m + 2 bar_m)
  std::vector<RaceGate> gates;

  bool operator==(const RaceCourse&) const = default;
};

// One aircraft's race progress (astrum/race/state/<id>).
struct RaceState {
  std::uint32_t aircraft_id = 0;
  std::uint32_t pending_gate = 0;  // next gate to pass; gate_count once finished
  std::uint32_t passed = 0;
  std::uint32_t missed = 0;
  std::uint32_t gate_count = 0;
  double last_pass_s = 0.0;  // sim time of the last gate passed; 0 if none
  double sim_time_s = 0.0;
  bool finished = false;  // no more gates will be scored (outcome != Running)
  RaceOutcome outcome = RaceOutcome::Running;

  bool operator==(const RaceState&) const = default;
};

}  // namespace astrum

#endif  // ASTRUM_TYPES_HPP
