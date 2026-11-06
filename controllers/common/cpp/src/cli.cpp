#include "astrum/cli.hpp"

#include <sstream>

#include "astrum/math.hpp"

namespace astrum {
namespace {

std::string number(double v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

}  // namespace

void add_connection_options(Options& options, ConnectionOptions& connection) {
  options
      .add("--id", "n", "aircraft id to fly, 1 ... 4294967295", "ASTRUM_TARGET_ID", &connection.id)
      .required()
      .add("--connect", "endpoint", "sim Zenoh endpoint", "ASTRUM_ZENOH_CONNECT",
           &connection.connect)
      .add("--startup-timeout", "s",
           "one deadline for reaching the sim and its first sample (exit 1)",
           "ASTRUM_STARTUP_TIMEOUT_S", &connection.startup_timeout_s)
      .add("--telemetry-timeout", "s",
           "telemetry silence (without RunEnded) that ends the run (exit 1)",
           "ASTRUM_TELEMETRY_TIMEOUT_S", &connection.telemetry_timeout_s);
}

Reference ReferenceOptions::reference() const {
  return Reference{wrap_pi(deg_to_rad(heading_deg)), alt_hae_m, tas_mps};
}

void add_reference_options(Options& options, ReferenceOptions& reference) {
  options
      .add("--heading", "deg", "reference true heading (0 = north)", "ASTRUM_TARGET_HEADING_DEG",
           &reference.heading_deg)
      .add("--alt", "m", "reference altitude, HAE", "ASTRUM_TARGET_ALT_M", &reference.alt_hae_m)
      .add("--speed", "m/s", "reference true airspeed (>= 0)", "ASTRUM_TARGET_SPD_MPS",
           &reference.tas_mps);
}

void add_attitude_limit_options(Options& options, AttitudeLimits& limits) {
  options
      .add("--max-bank-deg", "deg", "bank command limit, in (0, 90) (f16: 60)",
           "ASTRUM_MAX_BANK_DEG", &limits.max_bank_deg)
      .add("--max-pitch-deg", "deg", "pitch command limit, in (0, 90) (f16: 24)",
           "ASTRUM_MAX_PITCH_DEG", &limits.max_pitch_deg);
}

std::optional<std::string> validate(const ConnectionOptions& connection) {
  if (connection.id == 0) return "--id must be in 1 ... 4294967295, got 0";
  if (!(connection.startup_timeout_s > 0.0)) {
    return "--startup-timeout must be > 0, got " + number(connection.startup_timeout_s);
  }
  if (!(connection.telemetry_timeout_s > 0.0)) {
    return "--telemetry-timeout must be > 0, got " + number(connection.telemetry_timeout_s);
  }
  return std::nullopt;
}

std::optional<std::string> validate(const ReferenceOptions& reference) {
  if (!(reference.tas_mps >= 0.0)) return "--speed must be >= 0, got " + number(reference.tas_mps);
  return std::nullopt;
}

std::optional<std::string> validate(const AttitudeLimits& limits) {
  if (!(limits.max_bank_deg > 0.0 && limits.max_bank_deg < 90.0)) {
    return "--max-bank-deg must be in (0, 90), got " + number(limits.max_bank_deg);
  }
  if (!(limits.max_pitch_deg > 0.0 && limits.max_pitch_deg < 90.0)) {
    return "--max-pitch-deg must be in (0, 90), got " + number(limits.max_pitch_deg);
  }
  return std::nullopt;
}

}  // namespace astrum
