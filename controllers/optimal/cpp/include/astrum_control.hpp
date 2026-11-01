// astrum_control — the law API of the ASTRUM optimal-control controller (C++).
//
// A law maps a VehicleState + Reference to a Command once per telemetry sample, and
// nothing else. The runtime (src/runtime.cpp, on the shared astrum::Client) owns the
// wire: the Zenoh session, latest-value telemetry, decoding, the vehicle-kind check,
// seq / telemetryTick, the command lease, clamping, and the startup / telemetry
// timeouts. This header pulls in no Zenoh or Cap'n Proto.
//
// Write a Law (see laws/lqr.cpp, laws/mpc.cpp), declare its factory in laws.hpp, add it
// to make_law() in src/runtime.cpp, and select it with `--law <name>`.

#ifndef ASTRUM_OPTIMAL_CONTROL_HPP
#define ASTRUM_OPTIMAL_CONTROL_HPP

#include "astrum/control.hpp"  // OuterLoops, AttitudeLimits — the saturated outer loops
#include "astrum/math.hpp"     // wrap_pi, clamp, deg_to_rad
#include "astrum/types.hpp"    // VehicleState, Reference, Command

namespace astrum {

// Construction-time knobs every law receives (from the CLI).
struct LawConfig {
  double dt = 1.0 / 120.0;  // control period = 1 / --rate-hz (the aircraft's control_rate_hz)
  AttitudeLimits limits;    // --max-bank-deg / --max-pitch-deg
  double horizon_s = 0.5;   // --horizon-s: MPC look-ahead, round(horizon_s / dt) steps of dt
};

class Law {
 public:
  virtual ~Law() = default;

  // The vehicle class the law's actuator mapping is written for; the runtime refuses
  // telemetry of any other kind (exit 2).
  virtual VehicleKind vehicle_kind() const { return VehicleKind::FixedWing; }

  // Called once per fresh telemetry sample (s.tick strictly new).
  virtual Command control(const VehicleState& s, const Reference& ref) = 0;
};

}  // namespace astrum

#endif  // ASTRUM_OPTIMAL_CONTROL_HPP
