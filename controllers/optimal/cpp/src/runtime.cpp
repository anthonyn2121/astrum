// astrum-optimal-controller: runs a control law (`--law lqr|mpc`) against one aircraft.
//
// The wire is the shared astrum::Client (controllers/common/cpp): latest-value
// telemetry, exception-safe decoding, contract QoS, seq + telemetryTick on every
// command, one command per sample, startup / telemetry timeouts. This file only parses
// the CLI, builds the law, refuses vehicle kinds the law was not written for, checks
// that the sim runs at the --rate-hz the law was discretized for, and logs.

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "astrum/client.hpp"
#include "astrum/options.hpp"
#include "astrum_control.hpp"
#include "laws.hpp"

namespace astrum {
namespace {

constexpr const char* kTag = "[optimal]";

std::unique_ptr<Law> make_law(const std::string& name, const LawConfig& config) {
  if (name == "lqr") return make_lqr_law(config);
  if (name == "mpc") return make_mpc_law(config);
  return nullptr;
}

// Stop (exit 2) if the sim's tick period (simTimeS / tick) is not the law's dt.
std::optional<Stop> check_rate(const VehicleState& s, double rate_hz) {
  if (s.tick == 0) return std::nullopt;
  const double sim_rate_hz = static_cast<double>(s.tick) / s.t;
  if (std::abs(sim_rate_hz - rate_hz) <= 1e-6 * rate_hz) return std::nullopt;
  return Stop{2, "the sim runs at " + std::to_string(sim_rate_hz) +
                     " Hz but the law was discretized for --rate-hz " + std::to_string(rate_hz) +
                     "; pass --rate-hz to match the scenario's sim.rate_hz"};
}

int run(int argc, char** argv) {
  ClientConfig client;
  ReferenceOptions reference;
  LawConfig law_config;
  std::string law_name = "lqr";
  double rate_hz = 120.0;

  Options options("astrum-optimal-controller",
                  "ASTRUM optimal-control controller (C++): runs an LQR or MPC law against "
                  "one aircraft over Zenoh.");
  options.add("--law", "lqr|mpc", "control law", "ASTRUM_LAW", &law_name);
  add_client_options(options, client);
  add_reference_options(options, reference);
  add_attitude_limit_options(options, law_config.limits);
  options
      .add("--rate-hz", "hz", "control period the laws are discretized for (= sim.rate_hz)",
           "ASTRUM_RATE_HZ", &rate_hz)
      .add("--horizon-s", "s", "MPC look-ahead (mpc only; steps = horizon-s * rate-hz)", nullptr,
           &law_config.horizon_s);
  if (const auto exit_code = options.parse(argc, argv)) return *exit_code;
  if (!(rate_hz > 0.0)) throw std::invalid_argument("--rate-hz must be > 0");
  law_config.dt = 1.0 / rate_hz;

  std::unique_ptr<Law> law = make_law(law_name, law_config);
  if (!law) throw std::invalid_argument("unknown --law '" + law_name + "' (known: lqr, mpc)");
  const Reference ref = reference.reference();

  Client zenoh_client(client, kTag);
  std::cerr << kTag << " law=" << law_name << " rate=" << rate_hz
            << "Hz ref: hdg=" << reference.heading_deg << "deg alt=" << ref.alt_m
            << "m tas=" << ref.airspeed_mps << "m/s limits: bank=" << law_config.limits.max_bank_deg
            << "deg pitch=" << law_config.limits.max_pitch_deg << "deg\n";

  std::uint64_t samples = 0;
  bool rate_checked = false;
  return zenoh_client.run([&](const VehicleState& s) -> Step {
    if (auto stop = check_vehicle_kind(s, law->vehicle_kind(), "law " + law_name)) return *stop;
    if (!rate_checked && s.tick > 0) {
      if (auto stop = check_rate(s, rate_hz)) return *stop;
      rate_checked = true;
    }
    const Command command = law->control(s, ref);
    if (samples++ % 120 == 0) {
      std::cerr << kTag << " t=" << s.t << "s hdg=" << rad_to_deg(s.yaw) << "deg alt=" << s.alt_m
                << "m tas=" << s.airspeed << "m/s mode=" << to_string(s.control_mode) << "\n";
    }
    return command;
  });
}

}  // namespace
}  // namespace astrum

int main(int argc, char** argv) {
  try {
    return astrum::run(argc, argv);
  } catch (const std::invalid_argument& e) {
    std::cerr << astrum::kTag << " " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << astrum::kTag << " fatal: " << e.what() << "\n";
    return 1;
  }
}
