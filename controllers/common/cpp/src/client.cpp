#include "astrum/client.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <zenoh.hxx>

#include "astrum/math.hpp"
#include "astrum/wire.hpp"

namespace astrum {
namespace {

using Clock = std::chrono::steady_clock;

// The sim is a Zenoh router (the hub every process connects to); a client lets
// the hub route for it and waits up to `timeout_s` for the hub at open.
zenoh::Config session_config(const std::string& connect, double timeout_s) {
  zenoh::Config config = zenoh::Config::create_default();
  config.insert_json5("mode", "\"client\"");
  config.insert_json5("scouting/multicast/enabled", "false");
  // Low-latency transport to the sim (no batching, no QoS lanes): telemetry and
  // commands are small, latest-value samples, and every round trip counts in
  // lockstep. A sim that doesn't offer it falls back to the default transport.
  config.insert_json5("transport/unicast/lowlatency", "true");
  config.insert_json5("transport/unicast/qos/enabled", "false");
  config.insert_json5("connect/endpoints", "[\"" + connect + "\"]");
  config.insert_json5("connect/timeout_ms",
                      std::to_string(static_cast<long long>(timeout_s * 1000.0)));
  return config;
}

zenoh::Session open_session(const std::string& connect, double timeout_s) {
  try {
    return zenoh::Session::open(session_config(connect, timeout_s));
  } catch (const std::exception& e) {
    throw std::runtime_error("no sim reachable at " + connect + " within " +
                             std::to_string(timeout_s) + " s: " + e.what());
  }
}

zenoh::Session::PublisherOptions command_qos() {
  zenoh::Session::PublisherOptions options;
  options.congestion_control = Z_CONGESTION_CONTROL_DROP;
  options.priority = Z_PRIORITY_INTERACTIVE_HIGH;
  options.is_express = true;
  return options;
}

Clock::duration seconds(double s) {
  return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s));
}

bool finite(const Command& command) {
  if (const auto* raw = std::get_if<RawActuators>(&command.control)) {
    return std::isfinite(raw->aileron) && std::isfinite(raw->elevator) &&
           std::isfinite(raw->rudder) && std::isfinite(raw->throttle);
  }
  const auto& ap = std::get<AutopilotTargets>(command.control);
  return std::isfinite(ap.heading_rad) && std::isfinite(ap.alt_m) && std::isfinite(ap.tas_mps);
}

Command clamped(Command command) {
  if (auto* raw = std::get_if<RawActuators>(&command.control)) {
    raw->aileron = clamp(raw->aileron, -1.0, 1.0);
    raw->elevator = clamp(raw->elevator, -1.0, 1.0);
    raw->rudder = clamp(raw->rudder, -1.0, 1.0);
    raw->throttle = clamp(raw->throttle, 0.0, 1.0);
  }
  return command;
}

const ClientConfig& validated(const ClientConfig& config) {
  if (!(config.startup_timeout_s > 0.0) || !(config.telemetry_timeout_s > 0.0)) {
    throw std::invalid_argument("--startup-timeout and --telemetry-timeout must be > 0");
  }
  return config;
}

}  // namespace

std::optional<Stop> check_vehicle_kind(const VehicleState& state, VehicleKind expected,
                                       const std::string& law) {
  if (state.vehicle_kind == expected) return std::nullopt;
  return Stop{2, law + " is written for " + to_string(expected) + " aircraft but aircraft " +
                     std::to_string(state.aircraft_id) + " is " + to_string(state.vehicle_kind) +
                     "; refusing to fly it"};
}

void add_client_options(Options& options, ClientConfig& config) {
  options
      .add("--connect", "endpoint", "sim Zenoh endpoint", "ASTRUM_ZENOH_CONNECT", &config.connect)
      .add("--id", "n", "aircraft id to fly", "ASTRUM_TARGET_ID", &config.id)
      .add("--startup-timeout", "s", "exit 1 if no telemetry arrives within this time",
           "ASTRUM_STARTUP_TIMEOUT_S", &config.startup_timeout_s)
      .add("--telemetry-timeout", "s", "exit 0 once telemetry has stopped for this long",
           "ASTRUM_TELEMETRY_TIMEOUT_S", &config.telemetry_timeout_s);
}

struct Client::Impl {
  Impl(const ClientConfig& cfg, std::string log_tag)
      : config(validated(cfg)),
        tag(std::move(log_tag)),
        telemetry_key("astrum/telemetry/" + std::to_string(config.id)),
        command_key("astrum/command/" + std::to_string(config.id)),
        session(open_session(config.connect, config.startup_timeout_s)),
        publisher(session.declare_publisher(zenoh::KeyExpr(command_key), command_qos())),
        subscriber(session.declare_subscriber(
            zenoh::KeyExpr(telemetry_key), [this](zenoh::Sample& s) { on_sample(s); }, [] {})) {}

  // Zenoh RX thread: overwrite the latest-value slot. Must never throw into zenoh-c.
  void on_sample(zenoh::Sample& sample) noexcept {
    try {
      const zenoh::Bytes& payload = sample.get_payload();
      {
        std::lock_guard<std::mutex> lock(mutex);
        slot.resize(payload.size());
        payload.reader().read(slot.data(), slot.size());
        fresh = true;
      }
      ready.notify_one();
    } catch (...) {
      // Allocation failure: drop this sample; the next one overwrites the slot anyway.
    }
  }

  // Swap the newest unread payload into `out`; false if none arrives before `deadline`.
  bool take(std::vector<std::uint8_t>& out, Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex);
    if (!ready.wait_until(lock, deadline, [this] { return fresh; })) return false;
    out.swap(slot);
    fresh = false;
    return true;
  }

  void send(const Command& command, std::uint64_t telemetry_tick) {
    CommandMessage message;
    message.target_id = config.id;
    message.seq = ++seq;
    message.telemetry_tick = telemetry_tick;
    message.command = command;
    publisher.put(zenoh::Bytes(encode_command(message)));
  }

  ClientConfig config;
  std::string tag;
  std::string telemetry_key;
  std::string command_key;

  // Latest-value slot, written by on_sample, consumed by take(). Declared before the
  // Zenoh objects so the subscriber is undeclared before these are destroyed.
  std::mutex mutex;
  std::condition_variable ready;
  std::vector<std::uint8_t> slot;
  bool fresh = false;

  std::uint64_t seq = 0;

  zenoh::Session session;
  zenoh::Publisher publisher;
  zenoh::Subscriber<void> subscriber;
};

Client::Client(ClientConfig config, std::string tag)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(tag))) {}

Client::~Client() = default;

int Client::run(const Controller& controller) {
  Impl& z = *impl_;
  std::cerr << z.tag << " connect=" << z.config.connect << " telemetry=" << z.telemetry_key
            << " command=" << z.command_key << "\n";

  const Clock::time_point start = Clock::now();
  std::optional<Clock::time_point> last_good;  // receipt of the last valid sample
  std::optional<std::uint64_t> last_tick;
  std::optional<Command> last_command;
  std::uint64_t decode_errors = 0;
  std::vector<std::uint8_t> payload;

  while (true) {
    const Clock::time_point deadline = last_good
                                           ? *last_good + seconds(z.config.telemetry_timeout_s)
                                           : start + seconds(z.config.startup_timeout_s);
    if (!z.take(payload, deadline)) {
      if (!last_good) {
        std::cerr << z.tag << " no telemetry on " << z.telemetry_key << " within "
                  << z.config.startup_timeout_s << " s (is the sim running at " << z.config.connect
                  << "?)\n";
        return 1;
      }
      std::cerr << z.tag << " telemetry on " << z.telemetry_key << " stopped for "
                << z.config.telemetry_timeout_s << " s; exiting\n";
      return 0;
    }

    const Decoded<VehicleState> decoded = decode_telemetry(payload);
    if (!decoded) {
      if (decode_errors++ % 100 == 0) {
        std::cerr << z.tag << " dropped undecodable telemetry (" << decode_errors
                  << " so far): " << decoded.error << "\n";
      }
      continue;
    }
    const VehicleState& state = *decoded.value;
    if (state.aircraft_id != z.config.id) {
      std::cerr << z.tag << " dropped telemetry for aircraft " << state.aircraft_id << " on "
                << z.telemetry_key << "\n";
      continue;
    }
    if (!last_good) std::cerr << z.tag << " telemetry up at tick " << state.tick << "\n";
    last_good = Clock::now();

    if (last_tick == state.tick && last_command) {
      z.send(*last_command, state.tick);  // lockstep republish: answer it again
      continue;
    }

    const Step step = controller(state);
    if (const auto* stop = std::get_if<Stop>(&step)) {
      std::cerr << z.tag << " " << stop->reason << "\n";
      return stop->exit_code;
    }
    const Command& command = std::get<Command>(step);
    if (!finite(command)) {
      std::cerr << z.tag << " controller produced a non-finite command at tick " << state.tick
                << "; refusing to send it\n";
      return 1;
    }
    last_command = clamped(command);
    last_tick = state.tick;
    z.send(*last_command, state.tick);
  }
}

}  // namespace astrum
