#include "astrum/runtime.hpp"

#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include "astrum/log.hpp"
#include "astrum/transport.hpp"

namespace astrum {
namespace {

std::string law_names(std::span<const LawEntry> laws) {
  std::string out;
  for (const LawEntry& law : laws) {
    if (!out.empty()) out += "|";
    out += law.name;
  }
  return out;
}

std::string help_summary(std::string_view summary, std::span<const LawEntry> laws) {
  std::ostringstream os;
  os << summary << "\n\nLaws (--law):";
  for (const LawEntry& law : laws) os << "\n  " << law.name << "  " << law.summary;
  os << "\n\nThe control rate is not a flag: it comes from the telemetry (controlPeriodS).";
  return os.str();
}

SteadyClock::duration seconds(double s) {
  return std::chrono::duration_cast<SteadyClock::duration>(std::chrono::duration<double>(s));
}

}  // namespace

int run_main(int argc, char** argv, std::span<const LawEntry> laws, std::string_view program,
             std::string_view summary) {
  const Logger log = stderr_logger(std::string(program));
  if (laws.empty()) {
    log(LogLevel::Error, "no laws to run");
    return 2;
  }

  std::string law_name = laws.front().name;
  ConnectionOptions connection;
  ReferenceOptions reference;
  AttitudeLimits limits;
  Options options(std::string(program), help_summary(summary, laws));
  options.add("--law", law_names(laws), "law to run", "ASTRUM_LAW", &law_name);
  add_connection_options(options, connection);
  add_reference_options(options, reference);
  add_attitude_limit_options(options, limits);
  for (const LawEntry& law : laws) {
    if (law.add_options) law.add_options(options);
  }
  if (const auto exit = options.parse(argc, argv)) return *exit;

  const LawEntry* entry = nullptr;
  for (const LawEntry& law : laws) {
    if (law.name == law_name) entry = &law;
  }
  std::optional<std::string> usage_error;
  if (entry == nullptr) {
    usage_error = "unknown --law '" + law_name + "' (known: " + law_names(laws) + ")";
  } else if (auto e = validate(connection)) {
    usage_error = e;
  } else if (auto r = validate(reference)) {
    usage_error = r;
  } else if (auto l = validate(limits)) {
    usage_error = l;
  }
  if (usage_error) {
    log(LogLevel::Error, *usage_error);
    return 2;
  }

  const LawContext context{reference.reference(), limits};
  std::unique_ptr<Law> law;
  try {
    law = entry->make(context);
  } catch (const std::exception& e) {
    log(LogLevel::Error, "--law " + law_name + ": " + e.what());
    return 2;
  }
  if (!law) {
    log(LogLevel::Error, "--law " + law_name + ": the law factory returned nothing");
    return 2;
  }

  // One deadline for reaching the sim and for its first sample.
  const SteadyClock::time_point startup_deadline =
      SteadyClock::now() + seconds(connection.startup_timeout_s);
  try {
    std::optional<Session> session;
    try {
      session.emplace(Session::open(connection.connect, startup_deadline));
    } catch (const std::exception& e) {
      log(LogLevel::Error, std::string(e.what()) + " (is the sim running?)");
      return 1;
    }
    // Events first, so nothing published once the publisher's presence is seen is missed.
    // Only a fleet law waits for TickEnd, so only it subscribes to astrum/tick.
    EventSubscriber events(*session, law->needs_fleet(), log);
    std::optional<TelemetrySubscriber> telemetry;
    if (law->needs_fleet()) {
      telemetry.emplace(*session, kFleet, log);
    } else {
      telemetry.emplace(*session, connection.id, log);
    }
    CommandPublisher publisher(*session, connection.id);

    std::ostringstream start;
    start << "law=" << law_name << " id=" << connection.id << " connect=" << connection.connect
          << " epoch=" << publisher.epoch() << " ref: hdg=" << reference.heading_deg
          << "deg alt=" << context.reference.alt_hae_m << "m tas=" << context.reference.tas_mps
          << "m/s limits: bank=" << limits.max_bank_deg << "deg pitch=" << limits.max_pitch_deg
          << "deg";
    log(LogLevel::Info, start.str());

    LoopOptions loop;
    loop.aircraft_id = connection.id;
    loop.startup_deadline = startup_deadline;
    loop.telemetry_timeout_s = connection.telemetry_timeout_s;
    loop.log = log;
    const Outcome outcome = run_control_loop(*telemetry, events, publisher, *law, loop);
    const int code = exit_code(outcome);
    std::string line = std::string(to_string(outcome.kind));
    if (outcome.end_reason) line += " (" + to_string(*outcome.end_reason) + ")";
    if (!outcome.detail.empty()) line += ": " + outcome.detail;
    line += "; exit " + std::to_string(code);
    log(code == 0 ? LogLevel::Info : LogLevel::Error, line);
    return code;
  } catch (const std::exception& e) {
    log(LogLevel::Error, std::string("fatal: ") + e.what());
    return 1;
  }
}

}  // namespace astrum
