// ASTRUM example flight controller (C++).
//
// A minimal out-of-sim controller for one aircraft, built on the shared client
// (controllers/common/cpp): it reads astrum/telemetry/<id> as latest-value state and
// answers every sample on astrum/command/<id>. Two modes (--mode):
//   * raw (default): the hold law in hold_law.cpp flies the aircraft with raw
//     actuators — ASTRUM's intended usage, you own the control law. Fixed-wing only;
//     any other vehicle kind is refused (exit 2).
//   * autopilot: sends the heading / altitude / airspeed target every sample (the
//     command is a lease) and the sim's built-in inner loop flies it.
// Run with --help for every flag and its ASTRUM_* environment fallback.

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "astrum/client.hpp"
#include "astrum/control.hpp"
#include "astrum/math.hpp"
#include "astrum/options.hpp"
#include "hold_law.hpp"

namespace {

constexpr const char* kTag = "[example]";

void log_status(const astrum::VehicleState& s) {
  std::cerr << kTag << " t=" << s.t << "s hdg=" << astrum::rad_to_deg(s.yaw)
            << "deg alt=" << s.alt_m << "m tas=" << s.airspeed
            << "m/s mode=" << astrum::to_string(s.control_mode) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  astrum::ClientConfig client;
  astrum::ReferenceOptions reference;
  astrum::AttitudeLimits limits;
  std::string mode = "raw";

  astrum::Options options("astrum-controller",
                          "ASTRUM example flight controller (C++): raw hold law or sim autopilot "
                          "targets for one aircraft over Zenoh.");
  astrum::add_client_options(options, client);
  astrum::add_reference_options(options, reference);
  options.add("--mode", "raw|autopilot", "who closes the loop", "ASTRUM_CONTROL_MODE", &mode);
  astrum::add_attitude_limit_options(options, limits);
  if (const auto exit_code = options.parse(argc, argv)) return *exit_code;
  if (mode != "raw" && mode != "autopilot") {
    std::cerr << kTag << " --mode must be raw or autopilot, got '" << mode << "'\n";
    return 2;
  }

  try {
    const astrum::Reference ref = reference.reference();
    astrum::HoldLaw law(limits, ref);
    astrum::Client zenoh_client(client, kTag);
    std::cerr << kTag << " mode=" << mode << " target hdg=" << reference.heading_deg
              << "deg alt=" << ref.alt_m << "m tas=" << ref.airspeed_mps << "m/s\n";

    std::uint64_t samples = 0;
    return zenoh_client.run([&](const astrum::VehicleState& s) -> astrum::Step {
      const bool report = samples++ % 120 == 0;
      if (mode == "autopilot") {
        if (s.vehicle_kind != astrum::VehicleKind::FixedWing &&
            s.vehicle_kind != astrum::VehicleKind::Multirotor) {
          return astrum::Stop{2, "autopilot targets are undefined for vehicle kind " +
                                     astrum::to_string(s.vehicle_kind)};
        }
        if (report) log_status(s);
        return astrum::Command::autopilot(ref.heading_rad, ref.alt_m, ref.airspeed_mps);
      }
      if (auto stop = astrum::check_vehicle_kind(s, astrum::VehicleKind::FixedWing, "hold law")) {
        return *stop;
      }
      const astrum::RawActuators act = law.control(s);
      if (report) {
        log_status(s);
        std::cerr << kTag << "   ail=" << act.aileron << " elev=" << act.elevator
                  << " thr=" << act.throttle << "\n";
      }
      astrum::Command command;
      command.control = act;
      return command;
    });
  } catch (const std::invalid_argument& e) {
    std::cerr << kTag << " " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << kTag << " fatal: " << e.what() << "\n";
    return 1;
  }
}
