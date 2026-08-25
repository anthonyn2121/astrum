// astrum_control — the clean runtime for ASTRUM control-law development (C++).
//
// This header is the plumbing between the wire and *your* control law. The
// runtime owns everything you shouldn't have to think about while developing an
// LQR or MPC controller:
//
//   * opens the one peer-mode Zenoh session and connects to the sim,
//   * subscribes to a single aircraft's telemetry (astrum/telemetry/<id>),
//   * decodes each Cap'n Proto Telemetry frame into a plain VehicleState (SI
//     units; body rates estimated by finite-differencing attitude, since the
//     stream carries none),
//   * calls your Law::control(state, ref) once per frame,
//   * clamps and encodes the returned Command, and publishes it
//     (astrum/command/<id>).
//
// You never touch Zenoh or Cap'n Proto — and this header pulls in neither, so a
// law only ever includes "astrum_control.hpp" (plus whatever math it wants, e.g.
// Eigen). Write a Law (see laws/lqr.cpp, laws/mpc.cpp) that maps a VehicleState +
// Reference to a Command, register it in the factory (laws.hpp), and select it at
// runtime with `--law <name>`.

#ifndef ASTRUM_CONTROL_HPP
#define ASTRUM_CONTROL_HPP

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

namespace astrum {

constexpr double kPi = 3.14159265358979323846;

// Wrap an angle to [-pi, pi]. Handy for heading errors; provided here so laws
// don't each re-roll it.
inline double wrap_pi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// --- The data your law sees --------------------------------------------------

// One aircraft's state for a single frame, in SI units — everything a control
// law needs, decoded from Telemetry. Body rates p/q/r are ESTIMATED by
// finite-differencing attitude between frames (the stream carries no rates);
// they are 0 on the first frame. Angles radians, distances metres, speeds m/s.
struct VehicleState {
  double t = 0.0;  // sim time (s)
  std::uint32_t aircraft_id = 0;
  double lat_deg = 0.0;      // geodetic latitude
  double lon_deg = 0.0;      // geodetic longitude
  double alt_m = 0.0;        // MSL altitude (m)
  double e = 0.0;            // ENU east (m)
  double n = 0.0;            // ENU north (m)
  double u = 0.0;            // ENU up (m)
  double v_n = 0.0;          // NED velocity north (m/s)
  double v_e = 0.0;          // NED velocity east (m/s)
  double v_d = 0.0;          // NED velocity down (m/s)
  double roll = 0.0;         // phi (rad)
  double pitch = 0.0;        // theta (rad)
  double yaw = 0.0;          // psi / true heading (rad)
  double airspeed = 0.0;     // true airspeed (m/s)
  double p = 0.0;            // roll rate estimate (rad/s)
  double q = 0.0;            // pitch rate estimate (rad/s)
  double r = 0.0;            // yaw rate estimate (rad/s)
  std::string vehicle_kind;  // "fixed-wing" | "multirotor"
};

// The commanded setpoint (from the CLI) your law should track.
struct Reference {
  double heading_rad = 0.0;
  double alt_m = 0.0;
  double airspeed_mps = 0.0;
};

// A control output. Build one with Command::raw (you own the control law) or
// Command::autopilot (hand a target to the sim's inner loop). The runtime clamps
// the values to their valid ranges and encodes them to the wire.
struct Command {
  enum class Kind { Raw, Autopilot };
  Kind kind = Kind::Raw;

  // raw actuator setpoints (aileron/elevator/rudder in [-1,1], throttle [0,1])
  double aileron = 0.0, elevator = 0.0, rudder = 0.0, throttle = 0.0;
  // high-level autopilot target
  double heading_rad = 0.0, alt_m = 0.0, airspeed_mps = 0.0;

  static Command raw(double aileron, double elevator, double rudder, double throttle) {
    Command c;
    c.kind = Kind::Raw;
    c.aileron = aileron;
    c.elevator = elevator;
    c.rudder = rudder;
    c.throttle = throttle;
    return c;
  }
  static Command autopilot(double heading_rad, double alt_m, double airspeed_mps) {
    Command c;
    c.kind = Kind::Autopilot;
    c.heading_rad = heading_rad;
    c.alt_m = alt_m;
    c.airspeed_mps = airspeed_mps;
    return c;
  }
};

// --- CLI (precedence: CLI arg > $ENV > default, as everywhere in ASTRUM) ------

// Parsed configuration shared by every law, plus a couple of law-specific knobs
// (horizon/rate_hz are read only by the MPC law). Populated by Cli::parse.
struct Cli {
  std::string connect = "tcp/127.0.0.1:7447";  // ASTRUM_ZENOH_CONNECT
  std::uint32_t id = 1;                        // ASTRUM_TARGET_ID
  double heading_deg = 0.0;                    // ASTRUM_TARGET_HEADING_DEG
  double alt_m = 1500.0;                       // ASTRUM_TARGET_ALT_M
  double speed_mps = 50.0;                     // ASTRUM_TARGET_SPD_MPS
  std::string law = "lqr";                     // ASTRUM_LAW
  int horizon = 30;                            // MPC prediction horizon (steps)
  double horizon_s = 3.0;  // MPC look-ahead (s); model step = horizon_s / horizon

  // Seed from env (or built-in default), then override from argv. Returns false
  // and sets *exit_code if the program should stop now (--help -> 0, error -> 2).
  static bool parse(int argc, char** argv, Cli& out, int& exit_code);

  Reference reference() const { return Reference{heading_deg * kPi / 180.0, alt_m, speed_mps}; }
};

// --- Your law ----------------------------------------------------------------

// Base class for a control law — subclass it in laws/<name>.cpp and hand the
// factory (laws.hpp) a function that builds it. control() is called once per
// telemetry frame; the ctor gets the parsed Cli so a law can read its own knobs.
class Law {
 public:
  virtual ~Law() = default;
  virtual Command control(const VehicleState& s, const Reference& ref) = 0;
};

// --- The runtime: wire in, law, wire out -------------------------------------

// Drives one aircraft: telemetry in -> your law -> command out, forever. Owns
// the Zenoh session and Cap'n Proto codec (both hidden in runtime.cpp) and the
// attitude history used to estimate body rates.
class Runtime {
 public:
  Runtime(const Cli& cli, std::unique_ptr<Law> law);
  void run();

 private:
  // Decode one Telemetry payload into a VehicleState, updating the rate estimate.
  VehicleState decode(const std::uint8_t* data, std::size_t len);

  Cli cli_;
  std::unique_ptr<Law> law_;
  Reference ref_;

  // (t, phi, theta, psi) of the previous frame, for body-rate estimation.
  bool have_prev_ = false;
  double prev_t_ = 0.0, prev_phi_ = 0.0, prev_theta_ = 0.0, prev_psi_ = 0.0;
};

}  // namespace astrum

#endif  // ASTRUM_CONTROL_HPP
