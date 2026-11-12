#include "astrum/transport.hpp"

#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <zenoh.hxx>

#include "astrum/wire.hpp"

namespace astrum {
namespace {

constexpr std::string_view kTelemetryPrefix = "astrum/telemetry/";

// Report the first undecodable payload and then every 100th.
bool report_error(std::uint64_t count) { return count % 100 == 1; }

void log_to(const Logger& log, LogLevel level, const std::string& message) {
  if (log) log(level, message);
}

zenoh::Session::PublisherOptions command_qos() {
  zenoh::Session::PublisherOptions options;
  options.congestion_control = Z_CONGESTION_CONTROL_DROP;
  options.priority = Z_PRIORITY_INTERACTIVE_HIGH;
  options.is_express = true;
  return options;
}

std::uint64_t random_epoch() {
  std::random_device device;
  std::uint64_t epoch = 0;
  while (epoch == 0) {
    epoch = (static_cast<std::uint64_t>(device()) << 32) ^ static_cast<std::uint64_t>(device());
  }
  return epoch;
}

}  // namespace

// --- Session ---------------------------------------------------------------------------

struct Session::Impl {
  zenoh::Session session;
};

Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Session Session::open(const std::string& connect, SteadyClock::time_point deadline) {
  const auto remaining =
      std::chrono::ceil<std::chrono::milliseconds>(deadline - SteadyClock::now()).count();
  if (remaining <= 0) {
    throw std::runtime_error("the startup deadline passed before connecting to " + connect);
  }
  zenoh::Config config = zenoh::Config::create_default();
  // The sim is a Zenoh router (the hub every process connects to); a client lets the
  // hub route for it.
  config.insert_json5("mode", "\"client\"");
  config.insert_json5("scouting/multicast/enabled", "false");
  // Low-latency transport to the sim (no batching, one lane per link): telemetry and
  // commands are small latest-value samples, and every round trip counts in lockstep.
  config.insert_json5("transport/unicast/lowlatency", "true");
  config.insert_json5("transport/unicast/qos/enabled", "false");
  config.insert_json5("connect/endpoints", "[\"" + connect + "\"]");
  config.insert_json5("connect/timeout_ms", std::to_string(remaining));
  try {
    return Session(std::make_unique<Impl>(Impl{zenoh::Session::open(std::move(config))}));
  } catch (const std::exception& e) {
    throw std::runtime_error("no sim reachable at " + connect + " within " +
                             std::to_string(remaining) + " ms: " + e.what());
  }
}

// --- TelemetrySubscriber ---------------------------------------------------------------

struct TelemetrySubscriber::Impl {
  struct Slot {
    std::vector<std::uint8_t> payload;  // newest raw sample
    std::uint64_t version = 0;
    std::uint64_t decoded_version = 0;  // the version `decoded` holds (0: none)
    std::optional<VehicleState> decoded;
  };

  Impl(zenoh::Session& session, const std::string& key, Logger logger)
      : log(std::move(logger)),
        subscriber(session.declare_subscriber(
            zenoh::KeyExpr(key), [this](zenoh::Sample& s) { on_sample(s); }, [] {})) {}

  // Zenoh RX thread: overwrite the aircraft's slot. Must never throw into zenoh-c.
  void on_sample(zenoh::Sample& sample) noexcept {
    try {
      const std::string_view key = sample.get_keyexpr().as_string_view();
      std::uint32_t id = 0;
      const std::string_view digits =
          key.starts_with(kTelemetryPrefix) ? key.substr(kTelemetryPrefix.size()) : "";
      const auto [end, ec] = std::from_chars(digits.data(), digits.data() + digits.size(), id);
      if (digits.empty() || ec != std::errc() || end != digits.data() + digits.size()) {
        bad_keys.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      const zenoh::Bytes& payload = sample.get_payload();
      {
        std::lock_guard<std::mutex> lock(mutex);
        Slot& slot = slots[id];
        slot.payload.resize(payload.size());
        payload.reader().read(slot.payload.data(), slot.payload.size());
        slot.version = ++last_version;
      }
      arrived.notify_all();
    } catch (...) {
      // Allocation failure: drop this sample; the next one overwrites the slot anyway.
    }
  }

  // Decode `payload` (a sample of `id`) on the caller's thread; nullopt (counted, logged)
  // when it is malformed or names another aircraft.
  std::optional<VehicleState> decode(std::uint32_t id, const std::vector<std::uint8_t>& payload) {
    Decoded<VehicleState> d = decode_telemetry(payload);
    std::string error = std::move(d.error);
    if (d && d.value->aircraft_id != id) {
      error = "aircraftId " + std::to_string(d.value->aircraft_id) + " on the key of aircraft " +
              std::to_string(id);
      d.value.reset();
    }
    if (d) return std::move(d.value);
    const std::uint64_t count = ++errors;
    if (report_error(count)) {
      log_to(log, LogLevel::Warning,
             "dropped undecodable telemetry for aircraft " + std::to_string(id) + " (" +
                 std::to_string(count) + " so far): " + error);
    }
    return std::nullopt;
  }

  // The decoded newest sample of `id` (decoding at most once per version). Called with
  // `lock` held; releases it while decoding.
  std::optional<VehicleState> decoded_newest(std::uint32_t id, std::unique_lock<std::mutex>& lock) {
    Slot& slot = slots.at(id);
    if (slot.decoded_version != slot.version) {
      const std::uint64_t version = slot.version;
      const std::vector<std::uint8_t> payload = slot.payload;
      lock.unlock();
      std::optional<VehicleState> state = decode(id, payload);
      lock.lock();
      Slot& same = slots.at(id);  // the map may have rehashed while unlocked
      if (same.decoded_version < version) {
        same.decoded_version = version;
        same.decoded = std::move(state);
      }
      return same.decoded_version == version ? same.decoded : std::nullopt;
    }
    return slot.decoded;
  }

  Logger log;
  std::atomic<std::uint64_t> errors{0};
  std::atomic<std::uint64_t> bad_keys{0};

  // Declared before the subscriber, so that it is undeclared before these are destroyed.
  std::mutex mutex;
  std::condition_variable arrived;
  std::unordered_map<std::uint32_t, Slot> slots;
  std::uint64_t last_version = 0;

  zenoh::Subscriber<void> subscriber;
};

TelemetrySubscriber::TelemetrySubscriber(Session& session, std::uint32_t aircraft_id, Logger log)
    : impl_(std::make_unique<Impl>(session.impl_->session,
                                   std::string(kTelemetryPrefix) + std::to_string(aircraft_id),
                                   std::move(log))) {}

TelemetrySubscriber::TelemetrySubscriber(Session& session, AllAircraft /*fleet*/, Logger log)
    : impl_(std::make_unique<Impl>(session.impl_->session, std::string(kTelemetryPrefix) + "*",
                                   std::move(log))) {}

TelemetrySubscriber::~TelemetrySubscriber() = default;

std::optional<TelemetrySample> TelemetrySubscriber::wait_newer(std::uint32_t aircraft_id,
                                                               std::uint64_t after,
                                                               SteadyClock::time_point deadline) {
  Impl& z = *impl_;
  std::unique_lock<std::mutex> lock(z.mutex);
  while (true) {
    const bool fresh = z.arrived.wait_until(lock, deadline, [&] {
      const auto it = z.slots.find(aircraft_id);
      return it != z.slots.end() && it->second.version > after;
    });
    if (!fresh) return std::nullopt;
    const std::uint64_t version = z.slots.at(aircraft_id).version;
    std::optional<VehicleState> state = z.decoded_newest(aircraft_id, lock);
    if (state) return TelemetrySample{version, std::move(*state)};
    after = version;  // undecodable: wait for the next one
  }
}

std::map<std::uint32_t, VehicleState> TelemetrySubscriber::snapshot() {
  Impl& z = *impl_;
  std::unique_lock<std::mutex> lock(z.mutex);
  std::vector<std::uint32_t> ids;
  ids.reserve(z.slots.size());
  for (const auto& [id, slot] : z.slots) ids.push_back(id);
  std::map<std::uint32_t, VehicleState> fleet;
  for (const std::uint32_t id : ids) {
    if (std::optional<VehicleState> state = z.decoded_newest(id, lock)) {
      fleet.emplace(id, std::move(*state));
    }
  }
  return fleet;
}

std::uint64_t TelemetrySubscriber::decode_errors() const {
  return impl_->errors.load() + impl_->bad_keys.load();
}

// --- CommandPublisher ------------------------------------------------------------------

struct CommandPublisher::Impl {
  Impl(zenoh::Session& session, std::uint32_t aircraft_id)
      : id(aircraft_id),
        epoch(random_epoch()),
        publisher(session.declare_publisher(
            zenoh::KeyExpr("astrum/command/" + std::to_string(aircraft_id)), command_qos())),
        // Presence: the sim learns the moment this publisher (or its process) goes away.
        token(session.liveliness_declare_token(
            zenoh::KeyExpr("astrum/alive/controller/" + std::to_string(aircraft_id)))) {}

  std::uint32_t id;
  std::uint64_t epoch;
  std::uint64_t seq = 0;
  zenoh::Publisher publisher;
  zenoh::LivelinessToken token;
};

CommandPublisher::CommandPublisher(Session& session, std::uint32_t aircraft_id)
    : impl_(std::make_unique<Impl>(session.impl_->session, aircraft_id)) {}

CommandPublisher::~CommandPublisher() = default;

std::uint64_t CommandPublisher::epoch() const { return impl_->epoch; }

std::uint64_t CommandPublisher::send(const Action& action, std::uint64_t telemetry_tick) {
  Impl& z = *impl_;
  Command command;
  command.target_id = z.id;
  command.seq = ++z.seq;
  command.telemetry_tick = telemetry_tick;
  command.trigger = action.trigger;
  command.epoch = z.epoch;
  command.control = action.control;
  z.publisher.put(zenoh::Bytes(encode_command(command)));
  return command.seq;
}

// --- EventSubscriber -------------------------------------------------------------------

struct EventSubscriber::Impl {
  Impl(zenoh::Session& session, bool tick_ends, Logger logger)
      : log(std::move(logger)),
        events(session.declare_subscriber(
            zenoh::KeyExpr("astrum/events"), [this](zenoh::Sample& s) { on_event(s); }, [] {})) {
    if (tick_ends) {
      ticks.emplace(session.declare_subscriber(
          zenoh::KeyExpr("astrum/tick"), [this](zenoh::Sample& s) { on_tick(s); }, [] {}));
    }
  }

  // Zenoh RX thread: queue the raw event (decoded by drain()). Never throws.
  void on_event(zenoh::Sample& sample) noexcept {
    try {
      const zenoh::Bytes& payload = sample.get_payload();
      std::vector<std::uint8_t> bytes(payload.size());
      payload.reader().read(bytes.data(), bytes.size());
      std::lock_guard<std::mutex> lock(mutex);
      queue.push_back(std::move(bytes));
    } catch (...) {
      lost.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Zenoh RX thread: keep the newest TickEnd (a two-field, noexcept decode).
  void on_tick(zenoh::Sample& sample) noexcept {
    try {
      const zenoh::Bytes& payload = sample.get_payload();
      std::vector<std::uint8_t> bytes(payload.size());
      payload.reader().read(bytes.data(), bytes.size());
      const Decoded<TickEnd> decoded = decode_tick_end(bytes);
      if (!decoded) {
        bad_ticks.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        newest_tick_end = *decoded.value;
      }
      tick_arrived.notify_all();
    } catch (...) {
      lost.fetch_add(1, std::memory_order_relaxed);
    }
  }

  Logger log;
  std::atomic<std::uint64_t> errors{0};
  std::atomic<std::uint64_t> bad_ticks{0};
  std::atomic<std::uint64_t> lost{0};

  mutable std::mutex mutex;
  std::condition_variable tick_arrived;
  std::deque<std::vector<std::uint8_t>> queue;
  std::optional<TickEnd> newest_tick_end;

  zenoh::Subscriber<void> events;
  std::optional<zenoh::Subscriber<void>> ticks;  // only with tick_ends
};

EventSubscriber::EventSubscriber(Session& session, bool tick_ends, Logger log)
    : impl_(std::make_unique<Impl>(session.impl_->session, tick_ends, std::move(log))) {}

EventSubscriber::~EventSubscriber() = default;

bool EventSubscriber::wait_tick_end(std::uint64_t tick, SteadyClock::time_point deadline) {
  Impl& z = *impl_;
  if (!z.ticks) {
    throw std::logic_error("EventSubscriber::wait_tick_end without the astrum/tick subscriber");
  }
  std::unique_lock<std::mutex> lock(z.mutex);
  return z.tick_arrived.wait_until(
      lock, deadline, [&] { return z.newest_tick_end && z.newest_tick_end->tick >= tick; });
}

std::vector<Event> EventSubscriber::drain() {
  Impl& z = *impl_;
  std::deque<std::vector<std::uint8_t>> raw;
  {
    std::lock_guard<std::mutex> lock(z.mutex);
    raw.swap(z.queue);
  }
  std::vector<Event> out;
  out.reserve(raw.size());
  for (const std::vector<std::uint8_t>& bytes : raw) {
    Decoded<Event> decoded = decode_event(bytes);
    if (decoded) {
      out.push_back(std::move(*decoded.value));
      continue;
    }
    const std::uint64_t count = ++z.errors;
    if (report_error(count)) {
      log_to(z.log, LogLevel::Warning,
             "skipped an event (" + std::to_string(count) + " so far): " + decoded.error);
    }
  }
  return out;
}

std::optional<TickEnd> EventSubscriber::tick_end() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->newest_tick_end;
}

std::uint64_t EventSubscriber::decode_errors() const {
  return impl_->errors.load() + impl_->bad_ticks.load() + impl_->lost.load();
}

}  // namespace astrum
