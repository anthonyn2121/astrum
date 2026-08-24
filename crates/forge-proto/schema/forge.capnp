@0x8ce66e85b6cf0006;
#
# FORGE wire schema — the single source of truth shared by the Rust sim, the
# C++/Python flight controllers, and the bag record/replay tools.
#
#  * Telemetry : sim  --PUB-->  controllers / replay / bag-record  (topic = aircraftId)
#  * Command   : controllers  --PUSH-->  sim (PULL)                (targetId selects aircraft)
#  * BagFrame  : one recorded telemetry sample + wall-clock stamp; a bag is a
#               capnp message stream of these, replayable back onto the PUB socket.

struct Vec3 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

# Geodetic position (WGS84). Latitude is GEODETIC (not geocentric).
struct Geodetic {
  latDeg @0 :Float64;
  lonDeg @1 :Float64;
  altM   @2 :Float64;   # metres above the ellipsoid
}

# Body attitude as Euler angles (radians).
struct Euler {
  phiRad   @0 :Float64;   # roll
  thetaRad @1 :Float64;   # pitch
  psiRad   @2 :Float64;   # yaw / true heading
}

# One aircraft's authoritative state for a single sim step.
struct Telemetry {
  aircraftId  @0 :UInt32;
  simTimeS    @1 :Float64;
  geo         @2 :Geodetic;   # geodetic lat/lon/alt
  enu         @3 :Vec3;       # local world grid: x=E, y=N, z=U (metres)
  velNed      @4 :Vec3;       # velocity: x=North, y=East, z=Down (m/s)
  attitude    @5 :Euler;
  airspeedMps @6 :Float64;    # true airspeed
  health      @7 :UInt8;      # 0 = nominal (reserved for future fault flags)
}

# Raw actuator setpoints — the controller writes the control law directly.
struct RawActuators {
  aileron  @0 :Float64;   # [-1, 1]
  elevator @1 :Float64;   # [-1, 1]
  rudder   @2 :Float64;   # [-1, 1]
  throttle @3 :Float64;   # [0, 1]
}

# High-level autopilot targets — handled by in-sim inner loops.
struct AutopilotTargets {
  headingRad  @0 :Float64;
  altMslM     @1 :Float64;
  airspeedMps @2 :Float64;
}

# A command aimed at one aircraft. Layered: raw actuators or autopilot targets.
struct Command {
  targetId @0 :UInt32;
  control :union {
    raw       @1 :RawActuators;
    autopilot @2 :AutopilotTargets;
  }
}

# One entry in a telemetry bag: the sample plus the wall-clock instant it was
# recorded, so replay can pace frames back out in ~real-time.
struct BagFrame {
  recvWallNs @0 :UInt64;      # nanoseconds since the recorder's epoch
  telemetry  @1 :Telemetry;
}
