// astrum_control runtime (C++) — the wire plumbing behind the clean law API.
//
// This is the only translation unit that touches Zenoh or Cap'n Proto. It:
//   * parses the CLI (Cli::parse), selecting a law with `--law`,
//   * opens one peer-mode Zenoh session to the sim,
//   * decodes each Telemetry frame into a VehicleState (estimating body rates),
//   * calls the law's control(), and clamps + encodes + publishes the Command.
//
// Laws (laws/*.cpp) never include any of this — only "astrum_control.hpp".

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <zenoh.hxx>

#include "astrum.capnp.h"
#include "astrum_control.hpp"
#include "laws.hpp"

namespace astrum {
namespace {

std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : fallback;
}

double env_f64(const char* key, double fallback) {
  const char* v = std::getenv(key);
  return v ? std::strtod(v, nullptr) : fallback;
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n\n"
            << "ASTRUM optimal-control controller (C++). Runs a control law (LQR/MPC)\n"
            << "against one aircraft over Zenoh. Each option falls back to its ASTRUM_*\n"
            << "env var, then a default (precedence: CLI arg > env > default).\n\n"
            << "Options:\n"
            << "  --law <lqr|mpc>       control law to run       [env ASTRUM_LAW]\n"
            << "  --connect <endpoint>  sim Zenoh endpoint        [env ASTRUM_ZENOH_CONNECT]\n"
            << "  --id <n>              aircraft id               [env ASTRUM_TARGET_ID]\n"
            << "  --heading <deg>       reference heading, deg    [env ASTRUM_TARGET_HEADING_DEG]\n"
            << "  --alt <m>             reference MSL altitude, m [env ASTRUM_TARGET_ALT_M]\n"
            << "  --speed <mps>         reference airspeed, m/s   [env ASTRUM_TARGET_SPD_MPS]\n"
            << "  --horizon <steps>     MPC prediction horizon    (mpc only, default 30)\n"
            << "  --horizon-s <s>       MPC look-ahead, seconds   (mpc only, default 3)\n"
            << "  -h, --help            show this help and exit\n";
}

// Serialise a Command message to an owned Zenoh payload (Zenoh copies it out of
// the temporary string, so the capnp buffer can be released immediately).
zenoh::Bytes command_bytes(capnp::MallocMessageBuilder& message) {
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  kj::ArrayPtr<kj::byte> bytes = flat.asBytes();
  return zenoh::Bytes(std::string(reinterpret_cast<const char*>(bytes.begin()), bytes.size()));
}

std::unique_ptr<Law> make_law(const Cli& cli) {
  if (cli.law == "lqr") return make_lqr_law(cli);
  if (cli.law == "mpc") return make_mpc_law(cli);
  return nullptr;
}

}  // namespace

// --- Cli::parse --------------------------------------------------------------

bool Cli::parse(int argc, char** argv, Cli& out, int& exit_code) {
  // Seed every knob from its env var (or built-in default), then let CLI flags
  // override — precedence CLI arg > env > default, so compose keeps working.
  out.connect = env_or("ASTRUM_ZENOH_CONNECT", "tcp/127.0.0.1:7447");
  out.id = static_cast<std::uint32_t>(
      std::strtoul(env_or("ASTRUM_TARGET_ID", "1").c_str(), nullptr, 10));
  out.heading_deg = env_f64("ASTRUM_TARGET_HEADING_DEG", 0.0);
  out.alt_m = env_f64("ASTRUM_TARGET_ALT_M", 1500.0);
  out.speed_mps = env_f64("ASTRUM_TARGET_SPD_MPS", 50.0);
  out.law = env_or("ASTRUM_LAW", "lqr");

  // Hand-rolled parse (no argument-parsing dependency, matching the example
  // controller's minimal-deps ethos). Accepts "--flag value" and "--flag=value".
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      exit_code = 0;
      return false;
    }

    std::string key = arg;
    std::string inline_val;
    bool has_inline = false;
    if (arg.rfind("--", 0) == 0) {
      if (const auto eq = arg.find('='); eq != std::string::npos) {
        key = arg.substr(0, eq);
        inline_val = arg.substr(eq + 1);
        has_inline = true;
      }
    }
    auto value = [&]() -> std::string {
      if (has_inline) return inline_val;
      if (i + 1 < argc) return argv[++i];
      std::cerr << "[optimal] missing value for " << key << "\n";
      std::exit(2);
    };

    if (key == "--law") {
      out.law = value();
    } else if (key == "--connect") {
      out.connect = value();
    } else if (key == "--id") {
      out.id = static_cast<std::uint32_t>(std::strtoul(value().c_str(), nullptr, 10));
    } else if (key == "--heading") {
      out.heading_deg = std::strtod(value().c_str(), nullptr);
    } else if (key == "--alt") {
      out.alt_m = std::strtod(value().c_str(), nullptr);
    } else if (key == "--speed") {
      out.speed_mps = std::strtod(value().c_str(), nullptr);
    } else if (key == "--horizon") {
      out.horizon = static_cast<int>(std::strtol(value().c_str(), nullptr, 10));
    } else if (key == "--horizon-s") {
      out.horizon_s = std::strtod(value().c_str(), nullptr);
    } else {
      std::cerr << "[optimal] unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      exit_code = 2;
      return false;
    }
  }

  exit_code = 0;
  return true;
}

// --- Runtime -----------------------------------------------------------------

Runtime::Runtime(const Cli& cli, std::unique_ptr<Law> law)
    : cli_(cli), law_(std::move(law)), ref_(cli.reference()) {}

VehicleState Runtime::decode(const std::uint8_t* data, std::size_t len) {
  // capnp needs word-aligned input; the sim always writes whole words.
  auto words = kj::heapArray<capnp::word>(len / sizeof(capnp::word));
  std::memcpy(words.begin(), data, len);
  capnp::FlatArrayMessageReader reader(words.asPtr());
  Telemetry::Reader tel = reader.getRoot<Telemetry>();

  auto att = tel.getAttitude();
  VehicleState s;
  s.t = tel.getSimTimeS();
  s.aircraft_id = tel.getAircraftId();
  s.lat_deg = tel.getGeo().getLatDeg();
  s.lon_deg = tel.getGeo().getLonDeg();
  s.alt_m = tel.getGeo().getAltM();
  s.e = tel.getEnu().getX();
  s.n = tel.getEnu().getY();
  s.u = tel.getEnu().getZ();
  s.v_n = tel.getVelNed().getX();
  s.v_e = tel.getVelNed().getY();
  s.v_d = tel.getVelNed().getZ();
  s.roll = att.getPhiRad();
  s.pitch = att.getThetaRad();
  s.yaw = att.getPsiRad();
  s.airspeed = tel.getAirspeedMps();
  s.vehicle_kind = tel.getVehicleKind() == VehicleKind::MULTIROTOR ? "multirotor" : "fixed-wing";

  // Estimate body rates from attitude differences (no rates on the wire); 0 on
  // the first frame. Yaw difference wrapped to [-pi, pi] before dividing.
  if (have_prev_ && s.t > prev_t_) {
    const double dt = s.t - prev_t_;
    s.p = (s.roll - prev_phi_) / dt;
    s.q = (s.pitch - prev_theta_) / dt;
    s.r = wrap_pi(s.yaw - prev_psi_) / dt;
  }
  prev_t_ = s.t;
  prev_phi_ = s.roll;
  prev_theta_ = s.pitch;
  prev_psi_ = s.yaw;
  have_prev_ = true;
  return s;
}

void Runtime::run() {
  using namespace zenoh;

  const std::string telemetry_key = "astrum/telemetry/" + std::to_string(cli_.id);
  const std::string command_key = "astrum/command/" + std::to_string(cli_.id);

  // Peer session onto the sim's single Zenoh endpoint (multicast off — Docker
  // bridge networks don't pass it, so peers connect by explicit endpoint). Peer
  // mode makes open() non-fatal when the sim isn't listening yet (it spends ~10s
  // loading JSBSim before it binds): Zenoh retries in the background.
  // listen/endpoints=[] keeps this peer purely connect-out.
  Config config = Config::create_default();
  config.insert_json5("mode", "\"peer\"");
  config.insert_json5("connect/endpoints", "[\"" + cli_.connect + "\"]");
  config.insert_json5("listen/endpoints", "[]");
  config.insert_json5("scouting/multicast/enabled", "false");
  auto session = Session::open(std::move(config));

  auto publisher = session.declare_publisher(KeyExpr(command_key));

  std::cerr << "[optimal] law=" << cli_.law << " target=" << cli_.id << " connect=" << cli_.connect
            << " ref: hdg=" << cli_.heading_deg << "deg alt=" << ref_.alt_m
            << "m spd=" << ref_.airspeed_mps << "m/s\n";

  std::atomic<std::uint64_t> n{0};

  // Telemetry callback: decode -> law -> encode + publish. Zenoh serialises a
  // subscriber's callbacks, so the runtime/law state is single-owner (no locks).
  auto on_sample = [&](const Sample& sample) {
    auto payload = sample.get_payload().as_vector();
    const std::size_t nbytes = payload.size();
    if (nbytes % sizeof(capnp::word) != 0) return;

    const VehicleState s = decode(payload.data(), nbytes);
    const Command cmd = law_->control(s, ref_);

    capnp::MallocMessageBuilder message;
    ::Command::Builder c = message.initRoot<::Command>();
    c.setTargetId(cli_.id);
    if (cmd.kind == astrum::Command::Kind::Autopilot) {
      auto ap = c.getControl().initAutopilot();
      ap.setHeadingRad(cmd.heading_rad);
      ap.setAltMslM(cmd.alt_m);
      ap.setAirspeedMps(cmd.airspeed_mps);
    } else {
      auto raw = c.getControl().initRaw();
      raw.setAileron(clampd(cmd.aileron, -1.0, 1.0));
      raw.setElevator(clampd(cmd.elevator, -1.0, 1.0));
      raw.setRudder(clampd(cmd.rudder, -1.0, 1.0));
      raw.setThrottle(clampd(cmd.throttle, 0.0, 1.0));
    }
    publisher.put(command_bytes(message));

    if (++n % 120 == 0) {
      std::cerr << "[optimal] t=" << s.t << "s hdg=" << s.yaw * 180.0 / kPi << "deg alt=" << s.alt_m
                << "m spd=" << s.airspeed << "m/s\n";
    }
  };

  auto subscriber = session.declare_subscriber(KeyExpr(telemetry_key), on_sample, []() {});

  // The callback drives everything; park the main thread. The demo tears the
  // controller down when the bounded sim exits.
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

}  // namespace astrum

int main(int argc, char** argv) {
  astrum::Cli cli;
  int exit_code = 0;
  if (!astrum::Cli::parse(argc, argv, cli, exit_code)) {
    return exit_code;
  }

  auto law = astrum::make_law(cli);
  if (!law) {
    std::cerr << "[optimal] unknown --law '" << cli.law << "' (known: lqr, mpc)\n";
    return 2;
  }

  astrum::Runtime(cli, std::move(law)).run();
  return 0;
}
