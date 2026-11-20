// The control loop against scripted fakes (controllers/README.md, "The control loop").
// A fake clock makes every timeout deterministic: a wait that finds nothing jumps the
// clock to its deadline.

#include "astrum/loop.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace astrum {
namespace {

using std::chrono::milliseconds;
using std::chrono::seconds;
using Duration = SteadyClock::duration;
using TimePoint = SteadyClock::time_point;

constexpr std::uint64_t kEpoch = 77;
constexpr double kPeriod = 1.0 / 120.0;

struct FakeClock {
  TimePoint now = TimePoint(seconds(1000));
};

VehicleState make_state(std::uint64_t tick, VehicleKind kind = VehicleKind::FixedWing,
                        double period = kPeriod) {
  VehicleState s;
  s.aircraft_id = 1;
  s.tick = tick;
  s.sim_time_s = static_cast<double>(tick) * period;
  s.vehicle_kind = kind;
  s.control_period_s = period;
  return s;
}

// Hands out a scripted sequence of samples (each after an optional clock advance), then
// times out.
class FakeTelemetry : public TelemetrySource {
 public:
  explicit FakeTelemetry(FakeClock& clock) : clock_(clock) {}

  void push(VehicleState s, Duration after = Duration::zero()) {
    script.emplace_back(after, std::move(s));
  }

  std::optional<TelemetrySample> wait_newer(std::uint32_t aircraft_id, std::uint64_t after,
                                            TimePoint deadline) override {
    EXPECT_EQ(aircraft_id, 1U);
    EXPECT_EQ(after, version);
    waits.push_back(deadline - clock_.now);
    deadlines.push_back(deadline);
    if (!script.empty() && clock_.now + script.front().first <= deadline) {
      clock_.now += script.front().first;
      VehicleState s = std::move(script.front().second);
      script.pop_front();
      return TelemetrySample{++version, std::move(s)};
    }
    // The next sample's delay counts down across the slices that miss it.
    if (!script.empty() && deadline > clock_.now) script.front().first -= deadline - clock_.now;
    clock_.now = std::max(clock_.now, deadline);
    return std::nullopt;
  }

  std::map<std::uint32_t, VehicleState> snapshot() override {
    ++snapshots;
    return fleet;
  }

  std::deque<std::pair<Duration, VehicleState>> script;
  std::uint64_t version = 0;
  std::vector<Duration> waits;
  std::vector<TimePoint> deadlines;
  std::map<std::uint32_t, VehicleState> fleet;
  int snapshots = 0;

 private:
  FakeClock& clock_;
};

// drain() returns the scripted batches in order (one per call), then nothing.
class FakeEvents : public EventSource {
 public:
  explicit FakeEvents(FakeClock& clock) : clock_(clock) {}

  bool wait_tick_end(std::uint64_t tick, TimePoint deadline) override {
    tick_waits.push_back(deadline - clock_.now);
    waited_ticks.push_back(tick);
    if (tick_end_arrives) return true;
    clock_.now = std::max(clock_.now, deadline);
    return false;
  }

  std::vector<Event> drain() override {
    ++drains;
    if (batches.empty()) return {};
    std::vector<Event> batch = std::move(batches.front());
    batches.pop_front();
    return batch;
  }

  bool tick_end_arrives = true;
  std::deque<std::vector<Event>> batches;
  std::vector<Duration> tick_waits;
  std::vector<std::uint64_t> waited_ticks;
  int drains = 0;

 private:
  FakeClock& clock_;
};

// take_*() hand out each scripted message once, in order, then nothing.
class FakeRace : public RaceSource {
 public:
  std::optional<RaceCourse> take_course() override { return pop(courses); }
  std::optional<RaceState> take_state() override { return pop(states); }

  std::deque<std::optional<RaceCourse>> courses;
  std::deque<std::optional<RaceState>> states;

 private:
  template <class T>
  static std::optional<T> pop(std::deque<std::optional<T>>& q) {
    if (q.empty()) return std::nullopt;
    const std::optional<T> out = q.front();
    q.pop_front();
    return out;
  }
};

class RecordingSink : public CommandSink {
 public:
  std::uint64_t epoch() const override { return kEpoch; }
  std::uint64_t send(const Action& action, std::uint64_t telemetry_tick) override {
    sent.emplace_back(action, telemetry_tick);
    return sent.size();
  }
  std::vector<std::pair<Action, std::uint64_t>> sent;
};

class ScriptedLaw : public Law {
 public:
  std::vector<VehicleKind> vehicle_kinds() const override { return kinds; }
  bool needs_fleet() const override { return fleet; }
  void configure(double control_period_s) override {
    configured.push_back(control_period_s);
    if (on_configure) on_configure();
  }
  LawResult step(const StepInput& input) override {
    inputs.push_back(input);
    return on_step ? on_step(input) : LawResult{Action{RawActuators{}}};
  }

  std::vector<VehicleKind> kinds = {VehicleKind::FixedWing};
  bool fleet = false;
  std::function<LawResult(const StepInput&)> on_step;
  std::function<void()> on_configure;
  std::vector<double> configured;
  std::vector<StepInput> inputs;
};

struct Harness {
  FakeClock clock;
  FakeTelemetry telemetry{clock};
  FakeEvents events{clock};
  RecordingSink sink;
  ScriptedLaw law;
  std::vector<std::pair<LogLevel, std::string>> logs;
  TimePoint start = clock.now;
  LoopOptions options = make_options();

  LoopOptions make_options() {
    LoopOptions o;
    o.aircraft_id = 1;
    o.startup_deadline = clock.now + seconds(60);
    o.telemetry_timeout_s = 10.0;
    o.clock = [this] { return clock.now; };
    o.log = [this](LogLevel level, std::string_view m) { logs.emplace_back(level, m); };
    return o;
  }

  Outcome run() { return run_control_loop(telemetry, events, sink, law, options); }

  std::vector<std::uint64_t> sent_ticks() const {
    std::vector<std::uint64_t> ticks;
    for (const auto& [action, tick] : sink.sent) ticks.push_back(tick);
    return ticks;
  }
  std::vector<std::uint64_t> stepped_ticks() const {
    std::vector<std::uint64_t> ticks;
    for (const StepInput& in : law.inputs) ticks.push_back(in.state.tick);
    return ticks;
  }
  int warnings_containing(const std::string& text) const {
    int n = 0;
    for (const auto& [level, m] : logs) {
      if (level == LogLevel::Warning && m.find(text) != std::string::npos) ++n;
    }
    return n;
  }
};

Event event(std::uint64_t tick, EventDetail detail) {
  return Event{tick, static_cast<double>(tick) * kPeriod, std::move(detail)};
}

const RawActuators& raw(const Action& a) { return std::get<RawActuators>(a.control); }

// --- The Python test_control_loop.py cases ---------------------------------------------

TEST(ControlLoop, NoTelemetryEndsAtTheStartupDeadline) {
  Harness h;
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::NoTelemetry);
  EXPECT_EQ(exit_code(out), 1);
  EXPECT_TRUE(h.sink.sent.empty());
  EXPECT_EQ(h.clock.now, h.start + seconds(60));
  EXPECT_EQ(h.telemetry.deadlines.back(), h.start + seconds(60));
  // In slices of at most 100 ms, checking the events between them.
  EXPECT_EQ(h.telemetry.waits.size(), 600U);
  for (const Duration w : h.telemetry.waits) EXPECT_LE(w, milliseconds(100));
  EXPECT_GE(h.events.drains, 600);
}

TEST(ControlLoop, OneCommandPerTickThenTelemetryLost) {
  Harness h;
  for (const std::uint64_t t : {1, 2, 3}) h.telemetry.push(make_state(t), milliseconds(8));
  h.law.on_step = [](const StepInput&) { return LawResult{Action{RawActuators{0, 0, 0, 0.5}}}; };
  const TimePoint last_sample = h.start + milliseconds(24);
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::TelemetryLost);
  EXPECT_EQ(exit_code(out), 1);
  EXPECT_EQ(h.stepped_ticks(), (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(h.sent_ticks(), (std::vector<std::uint64_t>{1, 2, 3}));
  // The telemetry timeout runs from the last good sample.
  EXPECT_EQ(h.telemetry.deadlines.back(), last_sample + seconds(10));
  EXPECT_EQ(h.clock.now, last_sample + seconds(10));
}

TEST(ControlLoop, RepeatedTickResendsTheLastActionWithoutSteppingTheLaw) {
  Harness h;
  for (const std::uint64_t t : {1, 1, 2}) h.telemetry.push(make_state(t));
  h.law.on_step = [](const StepInput& in) {
    return LawResult{Action{RawActuators{0.1 * static_cast<double>(in.state.tick), 0, 0, 0}}};
  };
  EXPECT_EQ(h.run().kind, OutcomeKind::TelemetryLost);
  EXPECT_EQ(h.stepped_ticks(), (std::vector<std::uint64_t>{1, 2}));
  ASSERT_EQ(h.sink.sent.size(), 3U);
  EXPECT_EQ(h.sent_ticks(), (std::vector<std::uint64_t>{1, 1, 2}));
  EXPECT_EQ(raw(h.sink.sent[0].first).aileron, 0.1);
  EXPECT_EQ(raw(h.sink.sent[1].first).aileron, 0.1);
  EXPECT_EQ(raw(h.sink.sent[2].first).aileron, 0.2);
}

TEST(ControlLoop, RawAndMultirotorOutputsAreSaturatedAutopilotTargetsAreNot) {
  Harness h;
  h.law.kinds = {VehicleKind::FixedWing, VehicleKind::Multirotor};
  h.telemetry.push(make_state(1));
  h.telemetry.push(make_state(2, VehicleKind::Multirotor));
  h.telemetry.push(make_state(3));
  h.law.on_step = [](const StepInput& in) -> LawResult {
    switch (in.state.tick) {
      case 1:
        return Action{RawActuators{5.0, -5.0, 2.0, 3.0}};
      case 2:
        return Action{MultirotorRaw{5.0, -5.0, -2.0, -1.0}};
      default:
        return Action{AutopilotTargets{10.0, 1e6, 900.0}};
    }
  };
  h.run();
  ASSERT_EQ(h.sink.sent.size(), 3U);
  EXPECT_TRUE((h.sink.sent[0].first.control == Control{RawActuators{1.0, -1.0, 1.0, 1.0}}));
  EXPECT_TRUE((h.sink.sent[1].first.control == Control{MultirotorRaw{1.0, -1.0, -1.0, 0.0}}));
  EXPECT_TRUE((h.sink.sent[2].first.control == Control{AutopilotTargets{10.0, 1e6, 900.0}}));
}

TEST(ControlLoop, WrongVehicleClassRawControlIsALawFault) {
  const std::pair<VehicleKind, Control> cases[] = {
      {VehicleKind::Multirotor, RawActuators{0.0, 0.0, 0.0, 0.5}},
      {VehicleKind::FixedWing, MultirotorRaw{0.0, 0.0, 0.0, 0.5}},
  };
  for (const auto& [kind, control] : cases) {
    Harness h;
    h.law.kinds = {VehicleKind::FixedWing, VehicleKind::Multirotor};
    h.telemetry.push(make_state(1, kind));
    h.law.on_step = [control](const StepInput&) { return LawResult{Action{control}}; };
    const Outcome out = h.run();
    EXPECT_EQ(out.kind, OutcomeKind::LawFault);
    EXPECT_EQ(exit_code(out), 3);
    EXPECT_TRUE(h.sink.sent.empty());
  }
  // Autopilot targets are valid for both classes.
  for (const VehicleKind kind : {VehicleKind::FixedWing, VehicleKind::Multirotor}) {
    Harness h;
    h.law.kinds = {kind};
    h.telemetry.push(make_state(1, kind));
    h.law.on_step = [](const StepInput&) {
      return LawResult{Action{AutopilotTargets{0.0, 1000.0, 50.0}}};
    };
    EXPECT_EQ(h.run().kind, OutcomeKind::TelemetryLost);
    EXPECT_EQ(h.sink.sent.size(), 1U);
  }
}

TEST(ControlLoop, NonFiniteOutputIsALawFaultNotACommand) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const Control& bad :
       {Control{RawActuators{nan, 0, 0, 0}}, Control{MultirotorRaw{0, 0, 0, nan}},
        Control{AutopilotTargets{inf, 1000.0, 50.0}}}) {
    Harness h;
    h.telemetry.push(make_state(1));
    h.law.on_step = [bad](const StepInput&) { return LawResult{Action{bad}}; };
    const Outcome out = h.run();
    EXPECT_EQ(out.kind, OutcomeKind::LawFault);
    EXPECT_EQ(exit_code(out), 3);
    EXPECT_TRUE(h.sink.sent.empty());
  }
}

TEST(ControlLoop, WrongOrUnknownVehicleKindIsRefused) {
  for (const VehicleKind kind : {VehicleKind::Multirotor, static_cast<VehicleKind>(7)}) {
    Harness h;
    h.telemetry.push(make_state(1, kind));
    const Outcome out = h.run();
    EXPECT_EQ(out.kind, OutcomeKind::Refused);
    EXPECT_EQ(exit_code(out), 2);
    EXPECT_NE(out.detail.find(to_string(kind)), std::string::npos) << out.detail;
    EXPECT_TRUE(h.sink.sent.empty());
    EXPECT_TRUE(h.law.inputs.empty());
  }
}

TEST(ControlLoop, LawStopIsRefused) {
  Harness h;
  h.telemetry.push(make_state(1));
  h.law.on_step = [](const StepInput&) { return LawResult{Stop{"inconsistent sim"}}; };
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::Refused);
  EXPECT_EQ(exit_code(out), 2);
  EXPECT_NE(out.detail.find("inconsistent sim"), std::string::npos);
  EXPECT_TRUE(h.sink.sent.empty());
}

// --- Beyond the Python cases -----------------------------------------------------------

TEST(ControlLoop, AnExceptionInTheLawIsALawFault) {
  Harness h;
  h.telemetry.push(make_state(1));
  h.law.on_step = [](const StepInput&) -> LawResult { throw std::runtime_error("boom"); };
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::LawFault);
  EXPECT_EQ(exit_code(out), 3);
  EXPECT_NE(out.detail.find("boom"), std::string::npos);
}

TEST(ControlLoop, RunEndedWhileWaitingEndsTheRun) {
  for (const auto& [reason, code] :
       {std::pair{EndReason::MaxTime, 0}, std::pair{EndReason::Fatal, 1},
        std::pair{EndReason::EngagementDecided, 0}, std::pair{EndReason::Stopped, 0}}) {
    Harness h;
    // Three empty 100 ms slices, then the sim ends the run.
    h.events.batches = {{}, {}, {}, {event(5, RunEnded{reason, "why"})}};
    const Outcome out = h.run();
    EXPECT_EQ(out.kind, OutcomeKind::RunEnded);
    EXPECT_EQ(out.end_reason, reason);
    EXPECT_EQ(out.detail, "why");
    EXPECT_EQ(exit_code(out), code);
    EXPECT_EQ(h.clock.now, h.start + milliseconds(300));
  }
}

TEST(ControlLoop, RunEndedAmongATicksEventsEndsTheRunBeforeStepping) {
  Harness h;
  h.telemetry.push(make_state(5));
  // drain 1: before the first wait; drain 2: after the sample arrived.
  h.events.batches = {{}, {event(5, RunEnded{EndReason::MaxTime, ""})}};
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::RunEnded);
  EXPECT_EQ(exit_code(out), 0);
  EXPECT_TRUE(h.law.inputs.empty());
  EXPECT_TRUE(h.sink.sent.empty());
}

TEST(ControlLoop, NotOwnerRefusalOfOurEpochIsRefused) {
  Harness h;
  h.telemetry.push(make_state(1));
  VehicleState refused = make_state(2);
  refused.command = CommandStatus{5, 3, 1, Refusal::NotOwner, kEpoch};
  h.telemetry.push(refused);
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::Refused);
  EXPECT_EQ(exit_code(out), 2);
  EXPECT_NE(out.detail.find("another controller"), std::string::npos) << out.detail;
  EXPECT_EQ(h.sent_ticks(), (std::vector<std::uint64_t>{1}));
}

TEST(ControlLoop, AnotherEpochsRefusalIsIgnored) {
  Harness h;
  h.telemetry.push(make_state(1));
  for (const std::uint64_t t : {2, 3}) {
    VehicleState s = make_state(t);
    s.command = CommandStatus{kEpoch, t - 1, static_cast<std::uint32_t>(t), Refusal::NotOwner, 99};
    h.telemetry.push(s);
  }
  EXPECT_EQ(h.run().kind, OutcomeKind::TelemetryLost);
  EXPECT_EQ(h.sent_ticks(), (std::vector<std::uint64_t>{1, 2, 3}));
  EXPECT_EQ(h.warnings_containing("refused"), 0);
}

TEST(ControlLoop, RefusalsOfOurEpochAreLoggedOncePerNewRefusalRateLimited) {
  Harness h;
  const auto with_refusals = [](std::uint64_t tick, std::uint32_t refusals) {
    VehicleState s = make_state(tick);
    s.command = CommandStatus{0, 0, refusals, Refusal::Stale, kEpoch};
    return s;
  };
  h.telemetry.push(make_state(1));
  h.telemetry.push(with_refusals(2, 1));                      // new: logged
  h.telemetry.push(with_refusals(3, 1));                      // the same refusal: not logged
  h.telemetry.push(with_refusals(4, 2));                      // new, within 1 s: held back
  h.telemetry.push(with_refusals(5, 3), milliseconds(1500));  // new, later: logged
  EXPECT_EQ(h.run().kind, OutcomeKind::TelemetryLost);
  EXPECT_EQ(h.warnings_containing("refused our command: stale"), 2);
  EXPECT_EQ(h.warnings_containing("2 new refusal(s)"), 1);
  EXPECT_EQ(h.sink.sent.size(), 5U);
}

TEST(ControlLoop, EventsOfEarlierTicksAreDeliveredAndLaterOnesHeld) {
  Harness h;
  h.events.batches = {{event(1, ControllerUp{1}), event(2, Hit{2, 1, 1.0F, 99.0F, Enu{}}),
                       event(3, ControllerDown{2})}};
  h.telemetry.push(make_state(2));
  h.telemetry.push(make_state(3));
  h.telemetry.push(make_state(4));
  h.run();
  ASSERT_EQ(h.law.inputs.size(), 3U);
  // Step at tick 2: only tick 1's event; tick 2's waits a step.
  ASSERT_EQ(h.law.inputs[0].events.size(), 1U);
  EXPECT_TRUE(std::holds_alternative<ControllerUp>(h.law.inputs[0].events[0].detail));
  // Step at tick 3: the event of the tick stepped before.
  ASSERT_EQ(h.law.inputs[1].events.size(), 1U);
  EXPECT_TRUE(std::holds_alternative<Hit>(h.law.inputs[1].events[0].detail));
  ASSERT_EQ(h.law.inputs[2].events.size(), 1U);
  EXPECT_TRUE(std::holds_alternative<ControllerDown>(h.law.inputs[2].events[0].detail));
}

TEST(ControlLoop, OnlyAFleetLawWaitsForTickEnd) {
  Harness solo;
  solo.telemetry.push(make_state(1));
  solo.telemetry.push(make_state(2));
  solo.run();
  EXPECT_TRUE(solo.events.waited_ticks.empty());
  EXPECT_EQ(solo.sent_ticks(), (std::vector<std::uint64_t>{1, 2}));

  Harness fleet;
  fleet.law.fleet = true;
  fleet.events.tick_end_arrives = false;
  fleet.options.tick_end_wait_s = 0.05;
  fleet.telemetry.push(make_state(1));
  fleet.telemetry.push(make_state(2));
  fleet.run();
  EXPECT_EQ(fleet.events.waited_ticks, (std::vector<std::uint64_t>{1, 2}));
  for (const Duration w : fleet.events.tick_waits) EXPECT_EQ(w, milliseconds(50));  // bounded
  EXPECT_EQ(fleet.sent_ticks(), (std::vector<std::uint64_t>{1, 2}));  // stepped anyway
}

TEST(ControlLoop, NoControlPeriodIsRefused) {
  for (const double period : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    Harness h;
    h.telemetry.push(make_state(1, VehicleKind::FixedWing, period));
    const Outcome out = h.run();
    EXPECT_EQ(out.kind, OutcomeKind::Refused);
    EXPECT_EQ(exit_code(out), 2);
    EXPECT_TRUE(h.law.configured.empty());
    EXPECT_TRUE(h.sink.sent.empty());
  }
}

TEST(ControlLoop, ConfigureIsCalledOnceWithTheTelemetrysPeriod) {
  Harness h;
  for (const std::uint64_t t : {1, 2, 3}) {
    h.telemetry.push(make_state(t, VehicleKind::FixedWing, 1.0 / 30.0));
  }
  h.run();
  EXPECT_EQ(h.law.configured, (std::vector<double>{1.0 / 30.0}));
  ASSERT_EQ(h.law.inputs.size(), 3U);
  for (const StepInput& in : h.law.inputs) EXPECT_EQ(in.control_period_s, 1.0 / 30.0);
}

TEST(ControlLoop, ConfigureFailuresAreRefusedOrLawFaults) {
  Harness bad_period;
  bad_period.telemetry.push(make_state(1));
  bad_period.law.on_configure = [] { throw std::invalid_argument("horizon too short"); };
  EXPECT_EQ(bad_period.run().kind, OutcomeKind::Refused);

  Harness diverged;
  diverged.telemetry.push(make_state(1));
  diverged.law.on_configure = [] { throw std::runtime_error("dlqr diverged"); };
  EXPECT_EQ(diverged.run().kind, OutcomeKind::LawFault);
}

TEST(ControlLoop, TriggerInTheActionReachesTheSink) {
  Harness h;
  h.telemetry.push(make_state(1));
  h.telemetry.push(make_state(1));  // a lockstep resend keeps it
  h.law.on_step = [](const StepInput&) { return LawResult{Action{RawActuators{}, true}}; };
  h.run();
  ASSERT_EQ(h.sink.sent.size(), 2U);
  EXPECT_TRUE(h.sink.sent[0].first.trigger);
  EXPECT_TRUE(h.sink.sent[1].first.trigger);
}

TEST(ControlLoop, OneStartupDeadlineSharedWithTheSessionOpen) {
  Harness h;
  h.clock.now += seconds(45);  // opening the session took 45 of the 60 s
  const Outcome out = h.run();
  EXPECT_EQ(out.kind, OutcomeKind::NoTelemetry);
  EXPECT_EQ(h.clock.now, h.start + seconds(60));
  EXPECT_EQ(h.telemetry.waits.size(), 150U);  // 15 s left, in 100 ms slices
}

TEST(ControlLoop, FleetSnapshotOnlyForLawsThatNeedIt) {
  Harness solo;
  solo.telemetry.fleet = {{1, make_state(1)}, {2, make_state(1)}};
  solo.telemetry.push(make_state(1));
  solo.run();
  EXPECT_EQ(solo.telemetry.snapshots, 0);
  EXPECT_TRUE(solo.law.inputs.at(0).fleet.empty());

  Harness fleet;
  fleet.law.fleet = true;
  fleet.telemetry.fleet = {{1, make_state(1)}, {2, make_state(1)}};
  fleet.telemetry.push(make_state(1));
  fleet.run();
  EXPECT_EQ(fleet.telemetry.snapshots, 1);
  EXPECT_EQ(fleet.law.inputs.at(0).fleet.size(), 2U);
}

TEST(ControlLoop, RaceMessagesReachTheLawOnTheStepTheyArrivedAndNotOnRepeats) {
  Harness h;
  FakeRace race;
  race.courses = {RaceCourse{5, 1.0, 0.1, 1.2, {}}, std::nullopt, std::nullopt};
  race.states = {std::nullopt, RaceState{1, 1, 1, 0, 3, 0.5, 0.5, false, RaceOutcome::Running}};
  h.options.race = &race;
  h.telemetry.push(make_state(1));
  h.telemetry.push(make_state(1));  // a lockstep resend does not step, nor take
  h.telemetry.push(make_state(2));
  h.telemetry.push(make_state(3));
  h.run();
  ASSERT_EQ(h.law.inputs.size(), 3U);
  ASSERT_TRUE(h.law.inputs[0].race_course);
  EXPECT_EQ(h.law.inputs[0].race_course->seed, 5U);
  EXPECT_FALSE(h.law.inputs[0].race_state);
  EXPECT_FALSE(h.law.inputs[1].race_course);
  ASSERT_TRUE(h.law.inputs[1].race_state);
  EXPECT_EQ(h.law.inputs[1].race_state->gate_count, 3U);
  EXPECT_FALSE(h.law.inputs[2].race_course);
  EXPECT_FALSE(h.law.inputs[2].race_state);

  Harness plain;  // no race source: nothing is filled
  plain.telemetry.push(make_state(1));
  plain.run();
  EXPECT_FALSE(plain.law.inputs.at(0).race_course);
  EXPECT_FALSE(plain.law.wants_race());
}

TEST(Outcome, ExitCodes) {
  EXPECT_EQ(exit_code({OutcomeKind::RunEnded, EndReason::MaxTime, ""}), 0);
  EXPECT_EQ(exit_code({OutcomeKind::RunEnded, EndReason::RaceComplete, ""}), 0);
  EXPECT_EQ(exit_code({OutcomeKind::RunEnded, EndReason::Crashed, ""}), 0);
  EXPECT_EQ(exit_code({OutcomeKind::RunEnded, EndReason::Fatal, ""}), 1);
  EXPECT_EQ(exit_code({OutcomeKind::RunEnded, static_cast<EndReason>(9), ""}), 1);
  EXPECT_EQ(exit_code({OutcomeKind::TelemetryLost, std::nullopt, ""}), 1);
  EXPECT_EQ(exit_code({OutcomeKind::NoTelemetry, std::nullopt, ""}), 1);
  EXPECT_EQ(exit_code({OutcomeKind::Refused, std::nullopt, ""}), 2);
  EXPECT_EQ(exit_code({OutcomeKind::LawFault, std::nullopt, ""}), 3);
}

}  // namespace
}  // namespace astrum
