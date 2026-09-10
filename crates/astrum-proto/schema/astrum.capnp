@0x8ce66e85b6cf0006;
#
# ASTRUM wire schema — the single source of truth shared by the Rust sim, the
# C++/Python flight controllers, and the bag record/replay tools.
#
#  * Telemetry : sim  --PUB-->  controllers / replay / bag-record  (topic = aircraftId)
#  * Command   : controllers  --PUSH-->  sim (PULL)                (targetId selects aircraft)
#  * BagFrame  : one recorded telemetry sample + wall-clock stamp; a bag is a
#               capnp message stream of these, replayable back onto the PUB socket.
#  * RaceCourse / RaceState : the drone-racing environment's task state (see the
#               "Drone racing" section at the foot of this file).

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

# Flight-vehicle class. Carried in telemetry so a subscriber (e.g. the 3-D replay
# renderer) can pick the right model without any out-of-band scenario knowledge.
# Mirrors `astrum_sim::components::VehicleKind`.
enum VehicleKind {
  fixedWing  @0;
  multirotor @1;
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
  vehicleKind @8 :VehicleKind;   # additive (ordinal 8); pre-existing readers ignore it
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

# === Drone racing (astrum-race) ==============================================
#
# The racing environment reuses Telemetry (the drone's state) and Command (raw
# actuators) unchanged — a quadrotor's four normalized channels are the same four
# channels. What it cannot borrow is the *task* state: which gate is next, where
# the gates are, and how the run is going.
#
# These are separate top-level messages rather than fields on Telemetry, because
# Telemetry means "one aircraft's authoritative state" and is consumed by the bag
# recorder and the 3-D replay — course progress is neither, and would sit dead in
# every combat recording. They ride their own key space (astrum/race/**), so
# nothing above changes and no existing reader is affected.

# One gate's pose. The centre is the middle of the OPENING (not of the hardware),
# on the same world ENU grid Telemetry.enu reports, so a controller can subtract
# the two directly with no frame conversion.
struct RaceGate {
  index  @0 :UInt32;     # position in the course order; 0 is flown first
  center @1 :Vec3;       # centre of the opening: x=E, y=N, z=U (metres)
  yawRad @2 :Float64;    # heading of the gate NORMAL (the direction of flight
                         # through it): 0 = North, + = East
}

# The whole track. Static for a run, but published repeatedly at a low rate
# rather than once: Zenoh pub/sub has no late-joiner replay, so a controller
# started after the sim (the normal case — the sim spends ~10 s loading JSBSim)
# would otherwise never see the course at all.
struct RaceCourse {
  droneId  @0 :UInt32;   # the drone this course is set for (also in the key)
  seed     @1 :UInt64;   # the seed that drew it — a course's identity
  openingM @2 :Float64;  # side of each gate's square opening (m)
  barM     @3 :Float64;  # frame stock width/depth (m); hardware is outside the opening
  gates    @4 :List(RaceGate);
}

# Why a run ended, or that it hasn't. `running` is ordinal 0, which is also what an
# unset field reads as, so a RaceState written before this field existed decodes as
# "still going" rather than as some spurious outcome.
enum RaceOutcome {
  running       @0;  # still flying
  complete      @1;  # every gate passed, in order — the only way to win
  crashedGround @2;  # the airframe reached the ground plane
  crashedGate   @3;  # the airframe struck a gate's frame
  timedOut      @4;  # the simulated-time limit expired with gates still to fly
  spawnFailed   @5;  # the drone never launched; nothing was ever flown
}

# How the run is going. Published every tick alongside the drone's Telemetry, and
# deliberately carrying no vehicle state — that is Telemetry's job.
#
# Every progress figure is a CUMULATIVE COUNTER, not an event. That is what makes
# it safe for a subscriber to skip messages: a controller reading at 10 Hz off a
# 120 Hz stream still sees the true totals, where a "gate passed!" event stream
# would lose them.
struct RaceState {
  droneId     @0 :UInt32;
  simTimeS    @1 :Float64;
  pendingGate @2 :UInt32;   # index of the gate being flown at; == gateCount when done
  passed      @3 :UInt32;   # gates flown cleanly through, in order
  missed      @4 :UInt32;   # plane crossings that went round the outside of a gate
  gateCount   @5 :UInt32;   # so a state message is readable before the course arrives
  lastPassS   @6 :Float64;  # sim time the last gate was passed (0 if none yet)
  finished    @7 :Bool;     # the run is over, for whatever reason `outcome` names —
                            # a controller reads this to know when to stop flying, and
                            # `outcome` to know whether it won
  outcome     @8 :RaceOutcome;
}
