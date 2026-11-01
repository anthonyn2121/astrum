// ASTRUM controller client: the one Zenoh session of a controller process.
//
// Implements the transport contract for a single aircraft:
//   * session: Zenoh client of the sim (a router, the hub every process connects
//     to), multicast scouting off; open waits up to --startup-timeout for the sim;
//   * telemetry `astrum/telemetry/<id>`: LATEST-VALUE — the Zenoh callback only
//     overwrites one slot (never decodes, never throws), and decoding plus the control
//     law run on the caller's thread, so a slow law skips stale samples instead of
//     queueing them or stalling Zenoh's receive thread;
//   * session: client of the sim, asking for Zenoh's low-latency transport;
//   * commands `astrum/command/<id>`: declared publisher with CongestionControl DROP,
//     express, priority INTERACTIVE_HIGH; one command per telemetry sample (the raw
//     lease), each with seq (strictly increasing from 1) and telemetryTick;
//   * untrusted input: a payload that fails to decode is counted, logged, and skipped;
//   * timeouts: no telemetry within --startup-timeout -> exit 1; telemetry stops for
//     --telemetry-timeout after it started -> exit 0.

#ifndef ASTRUM_CLIENT_HPP
#define ASTRUM_CLIENT_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "astrum/options.hpp"
#include "astrum/types.hpp"

namespace astrum {

struct ClientConfig {
  std::string connect = "tcp/127.0.0.1:7447";
  std::uint32_t id = 1;
  double startup_timeout_s = 60.0;
  double telemetry_timeout_s = 10.0;
};

// --connect, --id, --startup-timeout, --telemetry-timeout (env ASTRUM_ZENOH_CONNECT,
// ASTRUM_TARGET_ID, ASTRUM_STARTUP_TIMEOUT_S, ASTRUM_TELEMETRY_TIMEOUT_S).
void add_client_options(Options& options, ClientConfig& config);

// A controller's verdict for one sample: a command to send, or stop with an exit code.
struct Stop {
  int exit_code = 0;
  std::string reason;
};
using Step = std::variant<Command, Stop>;
using Controller = std::function<Step(const VehicleState&)>;

// A raw control law is written for one vehicle class: Stop (exit 2) unless the telemetry's
// vehicle kind is `expected` (an unknown kind from a newer schema is refused too).
std::optional<Stop> check_vehicle_kind(const VehicleState& state, VehicleKind expected,
                                       const std::string& law);

class Client {
 public:
  // Opens the session and declares the subscriber + publisher. Throws std::runtime_error
  // (zenoh::ZException) if that fails and std::invalid_argument for non-positive timeouts.
  // `tag` prefixes log lines, e.g. "[example]".
  Client(ClientConfig config, std::string tag);
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  // Drive `controller` once per fresh telemetry sample until it stops or a timeout
  // fires; returns the process exit code. A sample whose tick repeats the previous one
  // (the lockstep sim republishing while it waits) is answered by resending the last
  // command with a new seq, without running the controller again. Raw commands are
  // clamped to their ranges; a non-finite command is never sent — it stops the run with
  // exit code 1 (a law bug; the sim's lease failsafe then takes the aircraft).
  int run(const Controller& controller);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace astrum

#endif  // ASTRUM_CLIENT_HPP
