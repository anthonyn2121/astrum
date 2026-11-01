// astrum-optimal-controller: runs a control law (`--law lqr|mpc`) against one aircraft.
//
// The wire is the shared astrum::Client (controllers/common/cpp): latest-value
// telemetry, exception-safe decoding, contract QoS, seq + telemetryTick on every
// command, one command per sample, startup / telemetry timeouts. This file only parses
// the CLI, builds the law, refuses vehicle kinds the law was not written for, checks
// that telemetry arrives at the --rate-hz the law was discretized for, and logs.

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
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

// Stops (exit 2) unless telemetry arrives at the law's control rate: the sim step
// (simTimeS / tick) must divide the law's period, and samples must arrive one period
// apart (the aircraft's control_rate_hz). A dropped sample only lengthens a gap, so the
// check passes on the first gap of exactly one period and fails on a shorter gap or on
// kMaxLongGaps longer ones in a row.
class RateCheck {
 public:
  explicit RateCheck(double rate_hz) : rate_hz_(rate_hz), period_(1.0 / rate_hz) {}

  std::optional<Stop> operator()(const VehicleState& s) {
    if (done_) return std::nullopt;
    if (!step_checked_ && s.tick > 0) {
      const double step = s.t / static_cast<double>(s.tick);
      const double n = period_ / step;
      if (n < 1.0 - 1e-6 || std::abs(n - std::round(n)) > 1e-6 * n) {
        return fail("the sim steps at " + std::to_string(1.0 / step) +
                    " Hz, which --rate-hz does not divide");
      }
      step_checked_ = true;
    }
    if (prev_t_ && s.t > *prev_t_) {
      const double gap = s.t - *prev_t_;
      if (std::abs(gap - period_) <= 1e-6 * period_) {
        done_ = step_checked_;
      } else if (gap < period_) {
        return fail("telemetry arrives at " + std::to_string(1.0 / gap) + " Hz");
      } else if (++long_gaps_ >= kMaxLongGaps) {
        return fail("telemetry arrives every " + std::to_string(gap) + " s");
      }
    }
    prev_t_ = s.t;
    return std::nullopt;
  }

 private:
  static constexpr int kMaxLongGaps = 8;

  Stop fail(const std::string& why) const {
    return Stop{2, why + " but the law was discretized for --rate-hz " + std::to_string(rate_hz_) +
                       "; pass the aircraft's control_rate_hz (default sim.rate_hz)"};
  }

  double rate_hz_;
  double period_;
  bool step_checked_ = false;
  bool done_ = false;
  int long_gaps_ = 0;
  std::optional<double> prev_t_;
};

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
      .add("--rate-hz", "hz",
           "control rate the laws are discretized for (= the aircraft's control_rate_hz, "
           "default sim.rate_hz)",
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
  RateCheck check_rate(rate_hz);
  return zenoh_client.run([&](const VehicleState& s) -> Step {
    if (auto stop = check_vehicle_kind(s, law->vehicle_kind(), "law " + law_name)) return *stop;
    if (auto stop = check_rate(s)) return *stop;
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
