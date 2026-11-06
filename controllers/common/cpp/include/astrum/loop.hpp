// The control loop (part of astrum::client): controllers/README.md, "The control loop".
//
// run_control_loop drives one Law for one aircraft on the caller's thread against three
// interfaces — TelemetrySource, EventSource, CommandSink — so it runs on the Zenoh
// transport (astrum/transport.hpp) in a controller and on scripted fakes in tests. Its
// result is an Outcome; exit_code() maps it to the process exit code.

#ifndef ASTRUM_LOOP_HPP
#define ASTRUM_LOOP_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "astrum/log.hpp"
#include "astrum/types.hpp"

namespace astrum {

using SteadyClock = std::chrono::steady_clock;

// One decoded telemetry sample. `version` orders the samples a source has received
// (strictly increasing, never 0), so "newer than the last one" is `version > last`.
struct TelemetrySample {
  std::uint64_t version = 0;
  VehicleState state;
};

class TelemetrySource {
 public:
  virtual ~TelemetrySource() = default;
  // The newest sample of `aircraft_id` with version > `after` (0: any), or nullopt if
  // none arrives before `deadline`. Undecodable samples are skipped (and counted).
  virtual std::optional<TelemetrySample> wait_newer(std::uint32_t aircraft_id, std::uint64_t after,
                                                    SteadyClock::time_point deadline) = 0;
  // The newest sample of every aircraft the source sees, by aircraft id.
  virtual std::map<std::uint32_t, VehicleState> snapshot() = 0;
};

class EventSource {
 public:
  virtual ~EventSource() = default;
  // True once a TickEnd with tick >= `tick` has arrived; false at `deadline` otherwise.
  // The loop calls it for a needs_fleet() law only.
  virtual bool wait_tick_end(std::uint64_t tick, SteadyClock::time_point deadline) = 0;
  // Every event received since the previous call, oldest first.
  virtual std::vector<Event> drain() = 0;
};

class CommandSink {
 public:
  virtual ~CommandSink() = default;
  // The non-zero epoch every command of this sink carries.
  virtual std::uint64_t epoch() const = 0;
  // Send `action` answering Telemetry.tick `telemetry_tick` with the next seq; returns it.
  virtual std::uint64_t send(const Action& action, std::uint64_t telemetry_tick) = 0;
};

// What a law sees for one new telemetry tick.
struct StepInput {
  VehicleState state;                           // this aircraft, the newest sample
  std::map<std::uint32_t, VehicleState> fleet;  // every aircraft (empty unless needs_fleet)
  std::vector<Event> events;      // not delivered yet, of ticks < state.tick, oldest first
  double control_period_s = 0.0;  // state.control_period_s: the law's dt
};

// A law's refusal to fly (Outcome Refused, exit 2).
struct Stop {
  std::string reason;
};

using LawResult = std::variant<Action, Stop>;

// A control law. The runtime (astrum/runtime.hpp) builds it from the flags; the loop
// calls configure once, then step once per new telemetry tick.
class Law {
 public:
  virtual ~Law() = default;
  // The vehicle classes the law can fly (non-empty); any other kind is refused.
  virtual std::vector<VehicleKind> vehicle_kinds() const = 0;
  // True: subscribe fleet-wide, wait for each tick's TickEnd and fill StepInput::fleet.
  virtual bool needs_fleet() const { return false; }
  // Gain design for the control period, once, before the first step. Throw
  // std::invalid_argument if the law cannot fly at this period (Outcome Refused); any
  // other exception is a LawFault.
  virtual void configure(double control_period_s) { (void)control_period_s; }
  // One control step. An exception is a LawFault.
  virtual LawResult step(const StepInput& input) = 0;
};

enum class OutcomeKind { RunEnded, TelemetryLost, NoTelemetry, Refused, LawFault };

// "run-ended" / "telemetry-lost" / "no-telemetry" / "refused" / "law-fault".
std::string_view to_string(OutcomeKind kind);

// Why the loop ended. `end_reason` is set for RunEnded only; `detail` is the RunEnded
// message or a human-readable explanation.
struct Outcome {
  OutcomeKind kind = OutcomeKind::NoTelemetry;
  std::optional<EndReason> end_reason;
  std::string detail;
};

// The process exit code (controllers/README.md, "Outcomes and exit codes"): RunEnded
// maxTime / engagementDecided / stopped 0; RunEnded fatal (or an unknown reason),
// TelemetryLost, NoTelemetry 1; Refused 2; LawFault 3.
int exit_code(const Outcome& outcome);

struct LoopOptions {
  std::uint32_t aircraft_id = 0;
  // The first sample must arrive by then (the same deadline the session open used).
  SteadyClock::time_point startup_deadline;
  // Telemetry silence after the last good sample that ends the run (TelemetryLost).
  double telemetry_timeout_s = 10.0;
  // The longest wait for a sample's TickEnd (needs_fleet laws only) before stepping anyway.
  double tick_end_wait_s = 0.05;
  std::function<SteadyClock::time_point()> clock = SteadyClock::now;
  Logger log = stderr_logger("astrum");
  // Log a status line every this many steps (0: never).
  std::uint64_t status_every = 120;
};

// Run `law` until an Outcome (controllers/README.md, steps 1-8). Exceptions from the
// sources and the sink propagate; the law's are caught (LawFault).
Outcome run_control_loop(TelemetrySource& telemetry, EventSource& events, CommandSink& sink,
                         Law& law, const LoopOptions& options);

}  // namespace astrum

#endif  // ASTRUM_LOOP_HPP
