#include "astrum/loop.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "astrum/math.hpp"

namespace astrum {
namespace {

// The longest single wait for telemetry: RunEnded is checked between slices.
constexpr auto kSlice = std::chrono::milliseconds(100);
// At most one refusal report per this interval.
constexpr auto kRefusalReportInterval = std::chrono::seconds(1);

SteadyClock::duration seconds(double s) {
  // Bounded so that a huge (but finite, validated > 0) timeout cannot overflow.
  return std::chrono::duration_cast<SteadyClock::duration>(
      std::chrono::duration<double>(std::min(s, 1e9)));
}

bool finite(const Control& control) {
  return std::visit(
      [](const auto& c) {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, RawActuators>) {
          return std::isfinite(c.aileron) && std::isfinite(c.elevator) && std::isfinite(c.rudder) &&
                 std::isfinite(c.throttle);
        } else if constexpr (std::is_same_v<T, MultirotorRaw>) {
          return std::isfinite(c.roll_rate) && std::isfinite(c.pitch_rate) &&
                 std::isfinite(c.yaw_rate) && std::isfinite(c.collective);
        } else {
          return std::isfinite(c.heading_rad) && std::isfinite(c.alt_hae_m) &&
                 std::isfinite(c.tas_mps);
        }
      },
      control);
}

// Raw channels to their ranges; autopilot targets pass unchanged.
void clamp_channels(Control& control) {
  if (auto* raw = std::get_if<RawActuators>(&control)) {
    raw->aileron = clamp(raw->aileron, -1.0, 1.0);
    raw->elevator = clamp(raw->elevator, -1.0, 1.0);
    raw->rudder = clamp(raw->rudder, -1.0, 1.0);
    raw->throttle = clamp(raw->throttle, 0.0, 1.0);
  } else if (auto* mr = std::get_if<MultirotorRaw>(&control)) {
    mr->roll_rate = clamp(mr->roll_rate, -1.0, 1.0);
    mr->pitch_rate = clamp(mr->pitch_rate, -1.0, 1.0);
    mr->yaw_rate = clamp(mr->yaw_rate, -1.0, 1.0);
    mr->collective = clamp(mr->collective, 0.0, 1.0);
  }
}

std::string kinds_text(const std::vector<VehicleKind>& kinds) {
  std::string out;
  for (const VehicleKind kind : kinds) {
    if (!out.empty()) out += " / ";
    out += to_string(kind);
  }
  return out.empty() ? "no" : out;
}

template <class... Args>
std::string cat(const Args&... args) {
  std::ostringstream os;
  (os << ... << args);
  return os.str();
}

Outcome make(OutcomeKind kind, std::string detail) {
  return Outcome{kind, std::nullopt, std::move(detail)};
}

class Loop {
 public:
  Loop(TelemetrySource& telemetry, EventSource& events, CommandSink& sink, Law& law,
       const LoopOptions& options)
      : telemetry_(telemetry),
        events_(events),
        sink_(sink),
        law_(law),
        options_(options),
        id_(options.aircraft_id),
        kinds_(law.vehicle_kinds()),
        fleet_(law.needs_fleet()) {}

  Outcome run() {
    std::uint64_t version = 0;
    std::optional<SteadyClock::time_point> last_good;
    while (true) {
      // 1. The next sample, in slices, watching for RunEnded.
      const SteadyClock::time_point deadline =
          last_good ? *last_good + seconds(options_.telemetry_timeout_s)
                    : options_.startup_deadline;
      std::optional<TelemetrySample> sample;
      do {
        if (auto end = pull_events()) return *end;
        sample = telemetry_.wait_newer(id_, version, std::min(deadline, now() + kSlice));
      } while (!sample && now() < deadline);
      if (!sample) {
        if (!last_good) {
          return make(OutcomeKind::NoTelemetry,
                      cat("no telemetry for aircraft ", id_, " before the startup deadline"));
        }
        return make(OutcomeKind::TelemetryLost, cat("telemetry for aircraft ", id_, " stopped for ",
                                                    options_.telemetry_timeout_s, " s"));
      }
      version = sample->version;
      const VehicleState& s = sample->state;
      if (!last_good) {
        log(LogLevel::Info, cat("telemetry up for aircraft ", id_, " at tick ", s.tick,
                                " (control period ", s.control_period_s, " s)"));
      }
      last_good = now();

      if (auto outcome = answer(s)) return *outcome;
    }
  }

 private:
  // Steps 2-8 for one sample; an Outcome ends the loop.
  std::optional<Outcome> answer(const VehicleState& s) {
    // 2. A fleet law waits (bounded) for the tick's TickEnd, so that the snapshot holds this
    //    tick for every aircraft; other laws never wait. Then drain the events.
    if (fleet_ && !events_.wait_tick_end(s.tick, now() + seconds(options_.tick_end_wait_s))) {
      log(LogLevel::Debug,
          cat("no TickEnd for tick ", s.tick, " within ", options_.tick_end_wait_s, " s"));
    }
    if (auto end = pull_events()) return end;

    // 3. Vehicle kind.
    if (std::find(kinds_.begin(), kinds_.end(), s.vehicle_kind) == kinds_.end()) {
      return make(OutcomeKind::Refused,
                  cat("the law flies ", kinds_text(kinds_), " aircraft but aircraft ", id_, " is ",
                      to_string(s.vehicle_kind), "; refusing to fly it"));
    }

    // 4. Ownership and refusals of our commands.
    if (auto refused = check_command_status(s)) return refused;

    // 5. A repeated tick (lockstep wait, pause): answer it again without stepping.
    if (answered_tick_ == s.tick && last_action_) {
      sink_.send(*last_action_, s.tick);
      return std::nullopt;
    }

    // 6. Configure once from the telemetry's control period.
    if (!configured_period_) {
      if (!(std::isfinite(s.control_period_s) && s.control_period_s > 0.0)) {
        return make(OutcomeKind::Refused,
                    cat("aircraft ", id_, " reports no control period (control_period_s = ",
                        s.control_period_s, "); refusing to fly it"));
      }
      try {
        law_.configure(s.control_period_s);
      } catch (const std::invalid_argument& e) {
        return make(OutcomeKind::Refused, cat("the law cannot fly at a control period of ",
                                              s.control_period_s, " s: ", e.what()));
      } catch (const std::exception& e) {
        return make(OutcomeKind::LawFault, cat("the law's configure failed: ", e.what()));
      } catch (...) {
        return make(OutcomeKind::LawFault, "the law's configure threw a non-std exception");
      }
      configured_period_ = s.control_period_s;
    } else if (s.control_period_s != *configured_period_ && !period_warned_) {
      period_warned_ = true;
      log(LogLevel::Warning,
          cat("aircraft ", id_, " control period changed to ", s.control_period_s,
              " s; the law stays configured for ", *configured_period_, " s"));
    }

    // 7. Step the law.
    StepInput input;
    input.state = s;
    if (fleet_) input.fleet = telemetry_.snapshot();
    input.events = take_events(s.tick);
    input.control_period_s = s.control_period_s;
    LawResult result;
    try {
      result = law_.step(input);
    } catch (const std::exception& e) {
      return make(OutcomeKind::LawFault, cat("the law threw at tick ", s.tick, ": ", e.what()));
    } catch (...) {
      return make(OutcomeKind::LawFault, cat("the law threw a non-std exception at tick ", s.tick));
    }
    if (const auto* stop = std::get_if<Stop>(&result)) {
      return make(OutcomeKind::Refused, cat("the law refused to fly: ", stop->reason));
    }

    // 8. Check, saturate, send.
    Action action = std::get<Action>(std::move(result));
    if (!finite(action.control)) {
      return make(OutcomeKind::LawFault,
                  cat("the law produced a non-finite command at tick ", s.tick));
    }
    // Raw control of the other vehicle class: the sim would refuse it (wrongVehicleKind).
    const bool raw_for_multirotor = std::holds_alternative<RawActuators>(action.control) &&
                                    s.vehicle_kind != VehicleKind::FixedWing;
    const bool multirotor_for_fixed_wing = std::holds_alternative<MultirotorRaw>(action.control) &&
                                           s.vehicle_kind != VehicleKind::Multirotor;
    if (raw_for_multirotor || multirotor_for_fixed_wing) {
      return make(
          OutcomeKind::LawFault,
          cat("the law produced ", raw_for_multirotor ? "fixed-wing RawActuators" : "MultirotorRaw",
              " for aircraft ", id_, ", which is ", to_string(s.vehicle_kind), ", at tick ",
              s.tick));
    }
    clamp_channels(action.control);
    sink_.send(action, s.tick);
    sent_ = true;
    last_action_ = std::move(action);
    answered_tick_ = s.tick;
    if (options_.status_every != 0 && steps_++ % options_.status_every == 0) status(s);
    return std::nullopt;
  }

  std::optional<Outcome> check_command_status(const VehicleState& s) {
    const CommandStatus& c = s.command;
    const bool new_refusal = refusals_seen_ && c.refusals != *refusals_seen_;
    refusals_seen_ = c.refusals;
    if (!sent_) return std::nullopt;
    const std::uint64_t epoch = sink_.epoch();
    if (c.last_refused_epoch == epoch && c.last_refusal == Refusal::NotOwner) {
      return make(OutcomeKind::Refused,
                  cat("aircraft ", id_, " is flown by another controller (epoch ", c.owner_epoch,
                      "); refusing to fight it for the aircraft"));
    }
    if (new_refusal && c.last_refused_epoch == epoch) {
      ++unreported_refusals_;
      const SteadyClock::time_point t = now();
      if (!last_refusal_report_ || t - *last_refusal_report_ >= kRefusalReportInterval) {
        log(LogLevel::Warning,
            cat("aircraft ", id_, " refused our command: ", to_string(c.last_refusal), " (",
                unreported_refusals_, " new refusal(s) since the last report, ", c.refusals,
                " for this aircraft in total)"));
        last_refusal_report_ = t;
        unreported_refusals_ = 0;
      }
    }
    if (!owner_logged_ && c.owner_epoch == epoch) {
      owner_logged_ = true;
      log(LogLevel::Info, cat("flying aircraft ", id_, " (epoch ", epoch, ")"));
    }
    return std::nullopt;
  }

  // Move newly arrived events into the pending queue; a RunEnded among them ends the loop.
  std::optional<Outcome> pull_events() {
    for (Event& e : events_.drain()) {
      if (const auto* end = std::get_if<RunEnded>(&e.detail)) {
        return Outcome{OutcomeKind::RunEnded, end->reason, end->message};
      }
      pending_.push_back(std::move(e));
    }
    return std::nullopt;
  }

  // The pending events of ticks before `tick`, oldest first; the rest stay pending (a tick's
  // events arrive after its telemetry, so they are complete only once a later tick's sample
  // is here).
  std::vector<Event> take_events(std::uint64_t tick) {
    std::vector<Event> due;
    std::deque<Event> later;
    for (Event& e : pending_) {
      if (e.tick < tick) {
        due.push_back(std::move(e));
      } else {
        later.push_back(std::move(e));
      }
    }
    pending_.swap(later);
    return due;
  }

  void status(const VehicleState& s) {
    const std::uint64_t epoch = sink_.epoch();
    const char* owner = s.command.owner_epoch == epoch ? "us"
                        : s.command.owner_epoch == 0   ? "none"
                                                       : "other";
    log(LogLevel::Info,
        cat("tick=", s.tick, " t=", s.sim_time_s, "s hdg=", rad_to_deg(s.yaw),
            "deg alt=", s.alt_hae_m, "m tas=", s.tas_mps, "m/s mode=", to_string(s.control_mode),
            " owner=", owner, " refusals=", s.command.refusals));
  }

  SteadyClock::time_point now() const { return options_.clock(); }

  void log(LogLevel level, const std::string& message) const {
    if (options_.log) options_.log(level, message);
  }

  TelemetrySource& telemetry_;
  EventSource& events_;
  CommandSink& sink_;
  Law& law_;
  const LoopOptions& options_;
  const std::uint32_t id_;
  const std::vector<VehicleKind> kinds_;
  const bool fleet_;

  std::deque<Event> pending_;
  std::optional<std::uint64_t> answered_tick_;
  std::optional<Action> last_action_;
  std::optional<double> configured_period_;
  bool period_warned_ = false;
  bool sent_ = false;
  bool owner_logged_ = false;
  std::uint64_t steps_ = 0;
  std::optional<std::uint32_t> refusals_seen_;
  std::optional<SteadyClock::time_point> last_refusal_report_;
  std::uint64_t unreported_refusals_ = 0;
};

}  // namespace

std::string_view to_string(OutcomeKind kind) {
  switch (kind) {
    case OutcomeKind::RunEnded:
      return "run-ended";
    case OutcomeKind::TelemetryLost:
      return "telemetry-lost";
    case OutcomeKind::NoTelemetry:
      return "no-telemetry";
    case OutcomeKind::Refused:
      return "refused";
    case OutcomeKind::LawFault:
      return "law-fault";
  }
  return "unknown";
}

int exit_code(const Outcome& outcome) {
  switch (outcome.kind) {
    case OutcomeKind::RunEnded:
      if (outcome.end_reason == EndReason::MaxTime ||
          outcome.end_reason == EndReason::EngagementDecided ||
          outcome.end_reason == EndReason::Stopped) {
        return 0;
      }
      return 1;
    case OutcomeKind::TelemetryLost:
    case OutcomeKind::NoTelemetry:
      return 1;
    case OutcomeKind::Refused:
      return 2;
    case OutcomeKind::LawFault:
      return 3;
  }
  return 1;
}

Outcome run_control_loop(TelemetrySource& telemetry, EventSource& events, CommandSink& sink,
                         Law& law, const LoopOptions& options) {
  if (!options.clock) throw std::invalid_argument("LoopOptions::clock is empty");
  return Loop(telemetry, events, sink, law, options).run();
}

}  // namespace astrum
