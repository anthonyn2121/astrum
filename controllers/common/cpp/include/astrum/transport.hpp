// The Zenoh transport of a controller process (part of astrum::client):
// controllers/README.md, "Transport".
//
//   Session              the process's ONE Zenoh session: client of the sim (a router),
//                        multicast scouting off, Zenoh's low-latency transport; open waits
//                        for the sim until the startup deadline
//   TelemetrySubscriber  astrum/telemetry/<id> (or every aircraft): latest-value — the
//                        Zenoh callback only stores the newest raw payload per aircraft;
//                        decoding happens on the caller's thread
//   CommandPublisher     astrum/command/<id>: DROP / express / INTERACTIVE_HIGH, a random
//                        non-zero epoch, seq 1, 2, ...; holds the presence token
//                        astrum/alive/controller/<id> while it lives
//   EventSubscriber      astrum/events as a FIFO (unknown kinds skipped), plus — only
//                        when asked for (a needs_fleet law) — the newest astrum/tick
//                        (TickEnd)
//   RaceSubscriber       astrum/race/course/<id> and astrum/race/state/<id> (a race run's
//                        gates and progress): latest-value, decoded on the caller's thread
//
// One Session carries any number of subscribers and publishers (several aircraft per
// process); it must outlive them. Every piece is non-copyable and non-movable (the Zenoh
// callbacks hold its address).

#ifndef ASTRUM_TRANSPORT_HPP
#define ASTRUM_TRANSPORT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "astrum/log.hpp"
#include "astrum/loop.hpp"
#include "astrum/types.hpp"

namespace astrum {

class Session {
 public:
  // Connect to the sim at `connect` (a Zenoh endpoint such as tcp/127.0.0.1:7447),
  // waiting at most until `deadline`. Throws std::runtime_error if no sim is reachable
  // by then.
  static Session open(const std::string& connect, SteadyClock::time_point deadline);

  ~Session();
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

 private:
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class TelemetrySubscriber;
  friend class CommandPublisher;
  friend class EventSubscriber;
  friend class RaceSubscriber;
};

// Which aircraft a TelemetrySubscriber follows: one id, or every aircraft.
struct AllAircraft {};
inline constexpr AllAircraft kFleet{};

class TelemetrySubscriber : public TelemetrySource {
 public:
  TelemetrySubscriber(Session& session, std::uint32_t aircraft_id,
                      Logger log = stderr_logger("astrum"));
  TelemetrySubscriber(Session& session, AllAircraft fleet, Logger log = stderr_logger("astrum"));
  ~TelemetrySubscriber() override;
  TelemetrySubscriber(const TelemetrySubscriber&) = delete;
  TelemetrySubscriber& operator=(const TelemetrySubscriber&) = delete;

  std::optional<TelemetrySample> wait_newer(std::uint32_t aircraft_id, std::uint64_t after,
                                            SteadyClock::time_point deadline) override;
  std::map<std::uint32_t, VehicleState> snapshot() override;

  // Samples that failed to decode (or whose aircraftId differs from their key) so far.
  std::uint64_t decode_errors() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CommandPublisher : public CommandSink {
 public:
  CommandPublisher(Session& session, std::uint32_t aircraft_id);
  ~CommandPublisher() override;
  CommandPublisher(const CommandPublisher&) = delete;
  CommandPublisher& operator=(const CommandPublisher&) = delete;

  std::uint64_t epoch() const override;
  std::uint64_t send(const Action& action, std::uint64_t telemetry_tick) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class EventSubscriber : public EventSource {
 public:
  // `tick_ends`: also subscribe to astrum/tick (run_main sets it for a needs_fleet law).
  EventSubscriber(Session& session, bool tick_ends, Logger log = stderr_logger("astrum"));
  ~EventSubscriber() override;
  EventSubscriber(const EventSubscriber&) = delete;
  EventSubscriber& operator=(const EventSubscriber&) = delete;

  // Throws std::logic_error unless constructed with `tick_ends`.
  bool wait_tick_end(std::uint64_t tick, SteadyClock::time_point deadline) override;
  std::vector<Event> drain() override;

  // The newest TickEnd received, if any.
  std::optional<TickEnd> tick_end() const;
  // Event / TickEnd payloads skipped so far (malformed, or a kind this build lacks).
  std::uint64_t decode_errors() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class RaceSubscriber : public RaceSource {
 public:
  RaceSubscriber(Session& session, std::uint32_t aircraft_id, Logger log = stderr_logger("astrum"));
  ~RaceSubscriber() override;
  RaceSubscriber(const RaceSubscriber&) = delete;
  RaceSubscriber& operator=(const RaceSubscriber&) = delete;

  std::optional<RaceCourse> take_course() override;
  std::optional<RaceState> take_state() override;

  // Race payloads skipped so far (malformed, or naming another aircraft).
  std::uint64_t decode_errors() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace astrum

#endif  // ASTRUM_TRANSPORT_HPP
