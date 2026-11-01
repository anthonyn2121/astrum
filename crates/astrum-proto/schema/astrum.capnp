@0x8ce66e85b6cf0006;
#
# ASTRUM wire schema (v2) — the single source of truth shared by the Rust sim,
# the C++/Python controllers, the bag tools and the web bridge.
#
# Transport (Zenoh, one session per process):
#   * Telemetry : sim --put--> astrum/telemetry/<aircraftId>   (one per aircraft per tick)
#   * Command   : controller --put--> astrum/command/<targetId> (the key's id MUST equal
#                 targetId; the sim rejects mismatches)
#   * Scenario  : the sim answers `get astrum/scenario` with its scenario TOML (UTF-8 text)
# Both streams are latest-value state: publish with CongestionControl::DROP, express,
# priority INTERACTIVE_HIGH, and read with a latest-value (ring) handler — a stale
# sample is worse than a dropped one.
#
# Framing: one unpacked flat message per Zenoh payload (Rust write_message /
# C++ messageToFlatArray / pycapnp to_bytes).
#
# Frames and units: SI throughout, angles in radians. `Enu` is the scenario's world
# grid (x=East, y=North, z=Up about the scenario origin); `Ned` is the local-level
# frame at the aircraft (x=North, y=East, z=Down). Altitudes are heights above the
# WGS84 ellipsoid (HAE). Evolution is additive only: never renumber or retype a field.

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("astrum::wire");

# World-grid vector (metres or m/s).
struct Enu {
  e @0 :Float64;
  n @1 :Float64;
  u @2 :Float64;
}

# Local-level vector at the aircraft (metres or m/s).
struct Ned {
  n @0 :Float64;
  e @1 :Float64;
  d @2 :Float64;
}

# WGS84 position. Latitude is GEODETIC (not geocentric).
struct Geodetic {
  latDeg  @0 :Float64;
  lonDeg  @1 :Float64;
  altHaeM @2 :Float64;   # height above the ellipsoid
}

# Aerospace 3-2-1 Euler attitude of the body relative to local-level NED.
struct Euler {
  phiRad   @0 :Float64;   # roll, +right wing down
  thetaRad @1 :Float64;   # pitch, +nose up
  psiRad   @2 :Float64;   # true heading, 0 = North, +East
}

# Body-axis angular rates relative to the Earth (x nose, y starboard, z down).
struct BodyRates {
  pRadS @0 :Float64;
  qRadS @1 :Float64;
  rRadS @2 :Float64;
}

struct Aero {
  alphaRad @0 :Float64;   # angle of attack
  betaRad  @1 :Float64;   # sideslip
  nzG      @2 :Float64;   # normal load factor (body -z), ~ +1 in level flight
  mach     @3 :Float64;
}

struct Combat {
  team         @0 :UInt8;
  hitPoints    @1 :Float32;
  hitPointsMax @2 :Float32;
  alive        @3 :Bool;
  ammo         @4 :UInt16;    # rounds left; 0 for unarmed aircraft
}

# Flight-vehicle class. The four raw actuator channels mean different things per
# class, so a raw control law MUST check this and refuse an unknown value.
enum VehicleKind {
  fixedWing  @0;
  multirotor @1;
}

# Who is flying the aircraft this tick.
enum ControlMode {
  uncommanded @0;   # no command received yet: initial (trimmed) controls held
  raw         @1;   # a controller's raw actuators, within the command lease
  autopilot   @2;   # the sim's inner loop flying commanded targets
  failsafe    @3;   # raw lease expired: the sim holds heading/altitude/speed
}

# One aircraft's authoritative state after sim step `tick`.
struct Telemetry {
  aircraftId  @0  :UInt32;
  simTimeS    @1  :Float64;      # tick * dt
  geo         @2  :Geodetic;
  enu         @3  :Enu;          # world-grid position (m)
  velNed      @4  :Ned;          # velocity relative to the Earth, local level (m/s)
  attitude    @5  :Euler;
  tasMps      @6  :Float64;      # true airspeed
  faultFlags  @7  :UInt8;        # 0 = nominal (reserved for fault injection)
  vehicleKind @8  :VehicleKind;
  tick        @9  :UInt64;       # sim step index; echo it in Command.telemetryTick
  bodyRates   @10 :BodyRates;
  aero        @11 :Aero;
  velEnu      @12 :Enu;          # velocity relative to the Earth, world grid (m/s)
  combat      @13 :Combat;
  controlMode @14 :ControlMode;
}

# Raw actuator setpoints — the controller owns the control law.
struct RawActuators {
  aileron  @0 :Float64;   # [-1, 1]
  elevator @1 :Float64;   # [-1, 1]
  rudder   @2 :Float64;   # [-1, 1]
  throttle @3 :Float64;   # [0, 1]
}

# Targets for the sim's built-in inner loop (vehicle-class specific).
struct AutopilotTargets {
  headingRad @0 :Float64;   # true heading
  altHaeM    @1 :Float64;
  tasMps     @2 :Float64;   # ignored by multirotors
}

# A command for one aircraft. Raw actuators are a lease: the sim falls back to its
# failsafe hold if no fresh raw command arrives within the scenario's lease, so a
# raw controller must keep sending every tick it receives telemetry.
struct Command {
  targetId @0 :UInt32;
  control :union {
    raw       @1 :RawActuators;
    autopilot @2 :AutopilotTargets;
  }
  seq           @3 :UInt64;   # per-sender, strictly increasing; older seqs are ignored
  telemetryTick @4 :UInt64;   # Telemetry.tick this command was computed from
  trigger       @5 :Bool;     # gun trigger (armed aircraft only)
}
