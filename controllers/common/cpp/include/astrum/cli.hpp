// The shared controller flags (controllers/README.md, "Flags") as Options helpers, and
// their validation. Part of astrum::cli; astrum::runtime's run_main adds all of them.

#ifndef ASTRUM_CLI_HPP
#define ASTRUM_CLI_HPP

#include <cstdint>
#include <optional>
#include <string>

#include "astrum/options.hpp"
#include "astrum/types.hpp"

namespace astrum {

// Where to fly and when to give up.
struct ConnectionOptions {
  std::string connect = "tcp/127.0.0.1:7447";
  std::uint32_t id = 0;  // required, 1 ... 4294967295
  double startup_timeout_s = 60.0;
  double telemetry_timeout_s = 10.0;
};

// --id (required), --connect, --startup-timeout, --telemetry-timeout (env
// ASTRUM_TARGET_ID, ASTRUM_ZENOH_CONNECT, ASTRUM_STARTUP_TIMEOUT_S,
// ASTRUM_TELEMETRY_TIMEOUT_S).
void add_connection_options(Options& options, ConnectionOptions& connection);

// The setpoint in CLI units: true heading in degrees, altitude m HAE, TAS m/s.
struct ReferenceOptions {
  double heading_deg = 0.0;
  double alt_hae_m = 1500.0;
  double tas_mps = 50.0;

  // heading wrapped to [-pi, pi) and converted to radians.
  Reference reference() const;
};

// --heading / --alt / --speed (env ASTRUM_TARGET_HEADING_DEG / _ALT_M / _SPD_MPS).
void add_reference_options(Options& options, ReferenceOptions& reference);

// --max-bank-deg / --max-pitch-deg (env ASTRUM_MAX_BANK_DEG / ASTRUM_MAX_PITCH_DEG).
void add_attitude_limit_options(Options& options, AttitudeLimits& limits);

// The usage error for a value outside its range (id 0, timeouts <= 0, speed < 0, a limit
// outside (0, 90) degrees), or nullopt when every value is valid.
std::optional<std::string> validate(const ConnectionOptions& connection);
std::optional<std::string> validate(const ReferenceOptions& reference);
std::optional<std::string> validate(const AttitudeLimits& limits);

}  // namespace astrum

#endif  // ASTRUM_CLI_HPP
