@0x8ce66e85b6cf0006;
#
# ASTRUM wire schema (v3) — the single source of truth shared by the Rust sim,
# the C++/Python controllers, the bag tools and the web bridge.
#
# Transport (Zenoh, one session per process):
#   * Telemetry : sim --put--> astrum/telemetry/<aircraftId>   (one per aircraft per
#                 control tick, aircraft id ascending)
#   * Command   : controller --put--> astrum/command/<targetId> (the key's id MUST equal
#                 targetId; the sim rejects mismatches)
#   * Event     : sim --put--> astrum/events (one per event, in the order they happened)
#   * TickEnd   : sim --put--> astrum/tick, after the tick's telemetry and events
#   * Race      : a race run (astrum-race) additionally publishes, per aircraft,
#                 astrum/race/course/<aircraftId> (RaceCourse, ~1 Hz and on the first
#                 tick) and astrum/race/state/<aircraftId> (RaceState, every control
#                 tick), both latest-value (CongestionControl::DROP); its race Events
#                 (gatePassed, gateMissed, crashed) travel on astrum/events.
#   * Scenario  : the sim answers `get astrum/scenario` with its scenario TOML (UTF-8 text)
#   * Presence, control and status (astrum/alive/controller/<id>, astrum/control/*,
#     astrum/status) carry no Cap'n Proto: a liveliness token and JSON / text replies.
#
# Per tick the sim publishes, in this order: every due Telemetry, then the tick's
# Events, then one TickEnd. Zenoh delivers a link's messages of one priority in send
# order and sends a higher priority first, so a subscriber that has received TickEnd N
# has received every Telemetry and Event of tick N that reached it. While a lockstep
# tick (or a pause) waits, the sim re-sends every due Telemetry of the tick and its
# TickEnd about every 100 ms, so a controller that subscribed late still sees the whole
# tick — its own sample and every other aircraft's.
#
# QoS. Telemetry and commands are latest-value state: CongestionControl::DROP, express,
# priority INTERACTIVE_HIGH, read with a latest-value handler — a stale sample is worse
# than a dropped one. Events and TickEnd must not be lost: CongestionControl::BLOCK,
# priority DATA (events read with a FIFO; TickEnd needs only the newest). Controllers use
# Zenoh's low-latency transport (the sim offers it): it has ONE lane per link, so
# priorities reorder nothing on it — messages arrive in the order the sim sent them.
#
# Ownership. Every Command carries the `epoch` of the controller instance that sent it.
# The first epoch whose command the sim applies owns the aircraft; commands from any
# other epoch are refused (Refusal.notOwner) until the owner lets go — its presence
# token disappears or its command lease expires (the aircraft enters failsafe). Within
# the owner's epoch `seq` must strictly increase. Telemetry.command reports the owner
# and the last refusal, so a controller learns it is not flying the aircraft.
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

# Flight-vehicle class. Raw control is class-specific (Command.control.raw is
# fixed-wing only, .multirotor multirotor only), so a raw control law MUST check this
# and refuse an unknown value.
enum VehicleKind {
  fixedWing  @0;
  multirotor @1;
}

# Who is flying the aircraft this tick.
enum ControlMode {
  uncommanded @0;   # no command received yet: initial (trimmed) controls held
  raw         @1;   # a controller's raw commands (raw or multirotor), within the lease
  autopilot   @2;   # the sim's inner loop flying commanded targets
  failsafe    @3;   # lease expired or controller gone: the sim holds heading/altitude/speed
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
  # Sim time between this aircraft's telemetry samples (1 / its control_rate_hz,
  # default 1 / sim.rate_hz); in lockstep, a command is awaited every period. The dt
  # a discretized control law runs at.
  controlPeriodS @15 :Float64;
  command        @16 :CommandStatus;   # ownership and command refusals
}

# Raw fixed-wing actuator setpoints — the controller owns the control law. Signs (as
# measured in the sim): aileron+ rolls right, elevator+ pitches nose DOWN.
struct RawActuators {
  aileron  @0 :Float64;   # [-1, 1]
  elevator @1 :Float64;   # [-1, 1]
  rudder   @2 :Float64;   # [-1, 1]
  throttle @3 :Float64;   # [0, 1]
}

# Raw multirotor demands: normalized body-rate demands and collective thrust, the
# inputs of the multirotor's own rate loops. The controller owns the attitude law.
struct MultirotorRaw {
  rollRate   @0 :Float64;   # [-1, 1], + rolls right
  pitchRate  @1 :Float64;   # [-1, 1], + pitches nose UP
  yawRate    @2 :Float64;   # [-1, 1], + yaws nose right
  collective @3 :Float64;   # [0, 1]
}

# Targets for the sim's built-in inner loop (vehicle-class specific).
struct AutopilotTargets {
  headingRad @0 :Float64;   # true heading
  altHaeM    @1 :Float64;
  tasMps     @2 :Float64;   # ignored by multirotors
}

# A command for one aircraft. Raw commands are a lease: the sim falls back to its
# failsafe hold if no fresh command arrives within the scenario's lease, so a raw
# controller must keep sending every tick it receives telemetry.
struct Command {
  targetId @0 :UInt32;
  control :union {
    raw        @1 :RawActuators;       # fixed-wing aircraft only
    autopilot  @2 :AutopilotTargets;
    multirotor @7 :MultirotorRaw;      # multirotor aircraft only
  }
  seq           @3 :UInt64;   # strictly increasing within one epoch
  telemetryTick @4 :UInt64;   # Telemetry.tick this command was computed from
  trigger       @5 :Bool;     # gun trigger (armed aircraft only)
  # Random non-zero id of the controller instance (one per command publisher) that
  # sent this command; 0 is refused. See "Ownership" above.
  epoch         @6 :UInt64;
}

# Why the sim refused a command.
enum Refusal {
  none             @0;   # no command refused yet
  fromTheFuture    @1;   # telemetryTick is ahead of the sim (sender bug or sim restart)
  stale            @2;   # telemetryTick is older than the command lease
  outOfOrder       @3;   # seq not above the owner's last applied seq
  notOwner         @4;   # another controller instance (epoch) owns the aircraft
  missingEpoch     @5;   # epoch 0
  nonFinite        @6;   # a NaN / infinite autopilot target
  wrongVehicleKind @7;   # raw on a multirotor, or multirotor on a fixed-wing
}

# The sim's command bookkeeping for one aircraft (Telemetry.command).
struct CommandStatus {
  ownerEpoch       @0 :UInt64;    # epoch of the controller flying it; 0 = unowned
  lastSeq          @1 :UInt64;    # seq of the owner's last applied command; 0 = none
  refusals         @2 :UInt32;    # commands refused for this aircraft so far (wraps)
  lastRefusal      @3 :Refusal;   # why the most recent one was refused
  lastRefusedEpoch @4 :UInt64;    # the epoch that sent it
}

# The end of a tick's publications on astrum/tick (see the ordering note above).
struct TickEnd {
  tick     @0 :UInt64;    # the Telemetry.tick just published
  simTimeS @1 :Float64;   # tick * dt
}

# Why an aircraft's controlMode changed.
enum ModeReason {
  command        @0;   # a controller's command (raw or autopilot)
  leaseExpired   @1;   # no fresh command within the command lease
  controllerLost @2;   # the controller's presence token went away
}

# Why a run ended.
enum EndReason {
  maxTime           @0;   # sim.max_time_s reached
  engagementDecided @1;   # at most one team still has a living aircraft
  fatal             @2;   # a fatal error; RunEnded.message says which
  stopped           @3;   # SIGINT / SIGTERM
  raceComplete      @4;   # a race's aircraft passed every gate
  crashed           @5;   # a race's aircraft crashed (Crashed event says how)
}

struct TriggerPressed {
  aircraftId @0 :UInt32;
  ammo       @1 :UInt16;   # rounds left when pressed
}

struct TriggerReleased {
  aircraftId  @0 :UInt32;
  roundsFired @1 :UInt16;   # since the matching TriggerPressed
  ammo        @2 :UInt16;   # rounds left
}

# A round struck an aircraft.
struct Hit {
  shooterId @0 :UInt32;
  targetId  @1 :UInt32;
  damage    @2 :Float32;
  hitPoints @3 :Float32;   # the target's, after this hit
  posEnu    @4 :Enu;       # where the round struck (world grid, m)
}

# An aircraft was destroyed (its hit points reached zero).
struct Kill {
  victimId @0 :UInt32;
  killer :union {
    unknown    @1 :Void;     # no hit recorded (e.g. destroyed by the sim's caller)
    aircraftId @2 :UInt32;   # the shooter of the last round that hit it
  }
  posEnu   @3 :Enu;         # the victim's position (world grid, m)
}

struct ModeChange {
  aircraftId @0 :UInt32;
  from       @1 :ControlMode;
  to         @2 :ControlMode;
  reason     @3 :ModeReason;
}

struct RunEnded {
  reason  @0 :EndReason;
  message @1 :Text;   # the fatal error; empty otherwise
}

# How a race ended for an aircraft (or `running` while it has not).
enum RaceOutcome {
  running       @0;
  complete      @1;   # every gate passed
  crashedGround @2;   # hit the ground (or the hull touched it)
  crashedGate   @3;   # hit a gate frame or bar
  timedOut      @4;   # the race's time limit ran out
  spawnFailed   @5;   # the start pose could not be set up
}

# The aircraft's centre of gravity crossed a gate's opening.
struct GatePassed {
  aircraftId @0 :UInt32;
  gate       @1 :UInt32;    # RaceGate.index
  offsetXM   @2 :Float64;   # CG position in the gate frame: along `right` ...
  offsetYM   @3 :Float64;   # ... and along `up`, from the gate centre (m)
}

# The aircraft crossed a gate's plane outside its opening (or skipped it).
struct GateMissed {
  aircraftId @0 :UInt32;
  gate       @1 :UInt32;
}

# The aircraft crashed. `gate` is the gate it hit (crashedGate) or the one it was
# flying to; 0 otherwise.
struct Crashed {
  aircraftId @0 :UInt32;
  outcome    @1 :RaceOutcome;
  gate       @2 :UInt32;
}

# Something that happened, as opposed to the per-tick state in Telemetry. A
# subscriber must ignore (skip) a kind it does not know.
struct Event {
  tick     @0 :UInt64;    # the Telemetry.tick that first shows it (while paused: the current tick)
  simTimeS @1 :Float64;   # tick * dt
  kind :union {
    triggerPressed  @2 :TriggerPressed;
    triggerReleased @3 :TriggerReleased;
    hit             @4 :Hit;
    kill            @5 :Kill;
    modeChange      @6 :ModeChange;
    controllerUp    @7 :UInt32;   # aircraft id whose controller connected (presence)
    controllerDown  @8 :UInt32;   # aircraft id whose controller went away
    runEnded        @9 :RunEnded;
    gatePassed      @10 :GatePassed;   # race runs only
    gateMissed      @11 :GateMissed;
    crashed         @12 :Crashed;
  }
}

# One race gate. The gate-local basis is LEFT-handed (x = right = starboard of the
# flight direction, y = up, z = normal = the flight direction), so a consumer must
# use the explicit basis vectors below instead of deriving them from yawRad.
struct RaceGate {
  index  @0 :UInt32;    # 0-based order in the course
  center @1 :Enu;       # world grid (m)
  yawRad @2 :Float64;   # heading of `normal`, 0 = North, +East
  normal @3 :Enu;       # unit vector through the opening in the flight direction
  right  @4 :Enu;       # unit vector, starboard of the flight direction
  up     @5 :Enu;       # unit vector, up
}

# A race course (astrum/race/course/<aircraftId>, ~1 Hz and on the first tick).
struct RaceCourse {
  seed     @0 :UInt64;    # the seed the gates came from
  openingM @1 :Float64;   # square clear opening side
  barM     @2 :Float64;   # frame stock width
  topBarM  @3 :Float64;   # outer frame side (openingM + 2 barM)
  gates    @4 :List(RaceGate);
}

# One aircraft's race progress (astrum/race/state/<aircraftId>, every control tick).
struct RaceState {
  aircraftId  @0 :UInt32;
  pendingGate @1 :UInt32;    # index of the next gate to pass; gateCount once finished
  passed      @2 :UInt32;
  missed      @3 :UInt32;
  gateCount   @4 :UInt32;
  lastPassS   @5 :Float64;   # sim time of the last gate passed; 0 if none
  simTimeS    @6 :Float64;
  finished    @7 :Bool;      # no more gates will be scored (outcome != running)
  outcome     @8 :RaceOutcome;
}
