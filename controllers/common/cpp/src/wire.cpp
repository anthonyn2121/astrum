#include "astrum/wire.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include <kj/io.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "astrum.capnp.h"

namespace astrum {
namespace {

// Read one flat message from `bytes` and convert its root with `convert`. The payload is
// read in place when word-aligned and copied once otherwise. Every failure (framing,
// bounds, traversal limits, trailing bytes, unknown union members) becomes
// Decoded::error.
template <class T, class Root, class Convert>
Decoded<T> decode_flat(std::span<const std::uint8_t> bytes, Convert convert) noexcept {
  Decoded<T> out;
  constexpr std::size_t kWord = sizeof(capnp::word);
  if (bytes.empty() || bytes.size() % kWord != 0) {
    out.error = "payload of " + std::to_string(bytes.size()) +
                " bytes is not a whole number of 8-byte words";
    return out;
  }
  try {
    const std::size_t n_words = bytes.size() / kWord;
    kj::Array<capnp::word> copy;
    kj::ArrayPtr<const capnp::word> words;
    if (reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(capnp::word) == 0) {
      words = kj::arrayPtr(reinterpret_cast<const capnp::word*>(bytes.data()), n_words);
    } else {
      copy = kj::heapArray<capnp::word>(n_words);
      std::memcpy(copy.begin(), bytes.data(), bytes.size());
      words = copy.asPtr();
    }
    capnp::FlatArrayMessageReader reader(words);
    if (reader.getEnd() != words.end()) {
      out.error = "trailing bytes after the message";
      return out;
    }
    out.value = convert(reader.getRoot<Root>());
  } catch (const kj::Exception& e) {
    out.value.reset();
    out.error = e.getDescription().cStr();
  } catch (const std::exception& e) {
    out.value.reset();
    out.error = e.what();
  } catch (...) {
    out.value.reset();
    out.error = "unknown decode failure";
  }
  return out;
}

std::vector<std::uint8_t> to_bytes(capnp::MessageBuilder& message) {
  std::vector<std::uint8_t> bytes(capnp::computeSerializedSizeInWords(message) *
                                  sizeof(capnp::word));
  kj::ArrayOutputStream stream(kj::arrayPtr(bytes.data(), bytes.size()));
  capnp::writeMessage(stream, message);
  return bytes;
}

// Enum values travel as their raw 16-bit number in both directions, so a value from a
// newer schema survives decoding unchanged.
template <class To, class From>
To enum_cast(From value) {
  return static_cast<To>(static_cast<std::uint16_t>(value));
}

std::string unknown(std::uint16_t value) { return "unknown(" + std::to_string(value) + ")"; }

VehicleState to_state(wire::Telemetry::Reader t) {
  VehicleState s;
  s.aircraft_id = t.getAircraftId();
  s.tick = t.getTick();
  s.sim_time_s = t.getSimTimeS();
  const auto geo = t.getGeo();
  s.lat_deg = geo.getLatDeg();
  s.lon_deg = geo.getLonDeg();
  s.alt_hae_m = geo.getAltHaeM();
  const auto enu = t.getEnu();
  s.e = enu.getE();
  s.n = enu.getN();
  s.u = enu.getU();
  const auto vned = t.getVelNed();
  s.vel_ned_n = vned.getN();
  s.vel_ned_e = vned.getE();
  s.vel_ned_d = vned.getD();
  const auto venu = t.getVelEnu();
  s.vel_enu_e = venu.getE();
  s.vel_enu_n = venu.getN();
  s.vel_enu_u = venu.getU();
  const auto att = t.getAttitude();
  s.roll = att.getPhiRad();
  s.pitch = att.getThetaRad();
  s.yaw = att.getPsiRad();
  const auto rates = t.getBodyRates();
  s.p = rates.getPRadS();
  s.q = rates.getQRadS();
  s.r = rates.getRRadS();
  s.tas_mps = t.getTasMps();
  const auto aero = t.getAero();
  s.alpha = aero.getAlphaRad();
  s.beta = aero.getBetaRad();
  s.nz_g = aero.getNzG();
  s.mach = aero.getMach();
  s.fault_flags = t.getFaultFlags();
  s.vehicle_kind = enum_cast<VehicleKind>(t.getVehicleKind());
  s.control_mode = enum_cast<ControlMode>(t.getControlMode());
  const auto combat = t.getCombat();
  s.team = combat.getTeam();
  s.hit_points = combat.getHitPoints();
  s.hit_points_max = combat.getHitPointsMax();
  s.alive = combat.getAlive();
  s.ammo = combat.getAmmo();
  s.control_period_s = t.getControlPeriodS();
  const auto status = t.getCommand();
  s.command.owner_epoch = status.getOwnerEpoch();
  s.command.last_seq = status.getLastSeq();
  s.command.refusals = status.getRefusals();
  s.command.last_refusal = enum_cast<Refusal>(status.getLastRefusal());
  s.command.last_refused_epoch = status.getLastRefusedEpoch();
  return s;
}

Command to_command(wire::Command::Reader c) {
  Command m;
  m.target_id = c.getTargetId();
  m.seq = c.getSeq();
  m.telemetry_tick = c.getTelemetryTick();
  m.trigger = c.getTrigger();
  m.epoch = c.getEpoch();
  const auto control = c.getControl();
  switch (control.which()) {
    case wire::Command::Control::RAW: {
      const auto raw = control.getRaw();
      m.control =
          RawActuators{raw.getAileron(), raw.getElevator(), raw.getRudder(), raw.getThrottle()};
      break;
    }
    case wire::Command::Control::AUTOPILOT: {
      const auto ap = control.getAutopilot();
      m.control = AutopilotTargets{ap.getHeadingRad(), ap.getAltHaeM(), ap.getTasMps()};
      break;
    }
    case wire::Command::Control::MULTIROTOR: {
      const auto mr = control.getMultirotor();
      m.control =
          MultirotorRaw{mr.getRollRate(), mr.getPitchRate(), mr.getYawRate(), mr.getCollective()};
      break;
    }
    default:
      throw std::runtime_error("unknown command control variant " +
                               std::to_string(static_cast<unsigned>(control.which())));
  }
  return m;
}

Enu to_enu(wire::Enu::Reader v) { return Enu{v.getE(), v.getN(), v.getU()}; }

Event to_event(wire::Event::Reader e) {
  Event out;
  out.tick = e.getTick();
  out.sim_time_s = e.getSimTimeS();
  const auto kind = e.getKind();
  switch (kind.which()) {
    case wire::Event::Kind::TRIGGER_PRESSED: {
      const auto v = kind.getTriggerPressed();
      out.detail = TriggerPressed{v.getAircraftId(), v.getAmmo()};
      break;
    }
    case wire::Event::Kind::TRIGGER_RELEASED: {
      const auto v = kind.getTriggerReleased();
      out.detail = TriggerReleased{v.getAircraftId(), v.getRoundsFired(), v.getAmmo()};
      break;
    }
    case wire::Event::Kind::HIT: {
      const auto v = kind.getHit();
      out.detail = Hit{v.getShooterId(), v.getTargetId(), v.getDamage(), v.getHitPoints(),
                       to_enu(v.getPosEnu())};
      break;
    }
    case wire::Event::Kind::KILL: {
      const auto v = kind.getKill();
      Kill kill;
      kill.victim_id = v.getVictimId();
      const auto killer = v.getKiller();
      switch (killer.which()) {
        case wire::Kill::Killer::UNKNOWN:
          break;
        case wire::Kill::Killer::AIRCRAFT_ID:
          kill.killer_id = killer.getAircraftId();
          break;
        default:
          throw std::runtime_error("unknown Kill.killer variant " +
                                   std::to_string(static_cast<unsigned>(killer.which())));
      }
      kill.pos_enu = to_enu(v.getPosEnu());
      out.detail = kill;
      break;
    }
    case wire::Event::Kind::MODE_CHANGE: {
      const auto v = kind.getModeChange();
      out.detail =
          ModeChange{v.getAircraftId(), enum_cast<ControlMode>(v.getFrom()),
                     enum_cast<ControlMode>(v.getTo()), enum_cast<ModeReason>(v.getReason())};
      break;
    }
    case wire::Event::Kind::CONTROLLER_UP:
      out.detail = ControllerUp{kind.getControllerUp()};
      break;
    case wire::Event::Kind::CONTROLLER_DOWN:
      out.detail = ControllerDown{kind.getControllerDown()};
      break;
    case wire::Event::Kind::RUN_ENDED: {
      const auto v = kind.getRunEnded();
      const auto message = v.getMessage();
      out.detail = RunEnded{enum_cast<EndReason>(v.getReason()),
                            std::string(message.begin(), message.size())};
      break;
    }
    case wire::Event::Kind::GATE_PASSED: {
      const auto v = kind.getGatePassed();
      out.detail = GatePassed{v.getAircraftId(), v.getGate(), v.getOffsetXM(), v.getOffsetYM()};
      break;
    }
    case wire::Event::Kind::GATE_MISSED: {
      const auto v = kind.getGateMissed();
      out.detail = GateMissed{v.getAircraftId(), v.getGate()};
      break;
    }
    case wire::Event::Kind::CRASHED: {
      const auto v = kind.getCrashed();
      out.detail = Crashed{v.getAircraftId(), enum_cast<RaceOutcome>(v.getOutcome()), v.getGate()};
      break;
    }
    default:
      throw std::runtime_error("unknown event kind " +
                               std::to_string(static_cast<unsigned>(kind.which())));
  }
  return out;
}

TickEnd to_tick_end(wire::TickEnd::Reader t) { return TickEnd{t.getTick(), t.getSimTimeS()}; }

RaceCourse to_race_course(wire::RaceCourse::Reader c) {
  RaceCourse out;
  out.seed = c.getSeed();
  out.opening_m = c.getOpeningM();
  out.bar_m = c.getBarM();
  out.top_bar_m = c.getTopBarM();
  const auto gates = c.getGates();
  out.gates.reserve(gates.size());
  for (const auto g : gates) {
    out.gates.push_back(RaceGate{g.getIndex(), to_enu(g.getCenter()), g.getYawRad(),
                                 to_enu(g.getNormal()), to_enu(g.getRight()), to_enu(g.getUp())});
  }
  return out;
}

RaceState to_race_state(wire::RaceState::Reader s) {
  return RaceState{s.getAircraftId(), s.getPendingGate(), s.getPassed(),
                   s.getMissed(),     s.getGateCount(),   s.getLastPassS(),
                   s.getSimTimeS(),   s.getFinished(),    enum_cast<RaceOutcome>(s.getOutcome())};
}

void set_enu(wire::Enu::Builder b, const Enu& v) {
  b.setE(v.e);
  b.setN(v.n);
  b.setU(v.u);
}

}  // namespace

std::string to_string(VehicleKind kind) {
  switch (kind) {
    case VehicleKind::FixedWing:
      return "fixed-wing";
    case VehicleKind::Multirotor:
      return "multirotor";
  }
  return unknown(static_cast<std::uint16_t>(kind));
}

std::string to_string(ControlMode mode) {
  switch (mode) {
    case ControlMode::Uncommanded:
      return "uncommanded";
    case ControlMode::Raw:
      return "raw";
    case ControlMode::Autopilot:
      return "autopilot";
    case ControlMode::Failsafe:
      return "failsafe";
  }
  return unknown(static_cast<std::uint16_t>(mode));
}

std::string to_string(Refusal refusal) {
  switch (refusal) {
    case Refusal::None:
      return "none";
    case Refusal::FromTheFuture:
      return "from-the-future";
    case Refusal::Stale:
      return "stale";
    case Refusal::OutOfOrder:
      return "out-of-order";
    case Refusal::NotOwner:
      return "not-owner";
    case Refusal::MissingEpoch:
      return "missing-epoch";
    case Refusal::NonFinite:
      return "non-finite";
    case Refusal::WrongVehicleKind:
      return "wrong-vehicle-kind";
  }
  return unknown(static_cast<std::uint16_t>(refusal));
}

std::string to_string(ModeReason reason) {
  switch (reason) {
    case ModeReason::Command:
      return "command";
    case ModeReason::LeaseExpired:
      return "lease-expired";
    case ModeReason::ControllerLost:
      return "controller-lost";
  }
  return unknown(static_cast<std::uint16_t>(reason));
}

std::string to_string(EndReason reason) {
  switch (reason) {
    case EndReason::MaxTime:
      return "max-time";
    case EndReason::EngagementDecided:
      return "engagement-decided";
    case EndReason::Fatal:
      return "fatal";
    case EndReason::Stopped:
      return "stopped";
    case EndReason::RaceComplete:
      return "race-complete";
    case EndReason::Crashed:
      return "crashed";
  }
  return unknown(static_cast<std::uint16_t>(reason));
}

std::string to_string(RaceOutcome outcome) {
  switch (outcome) {
    case RaceOutcome::Running:
      return "running";
    case RaceOutcome::Complete:
      return "complete";
    case RaceOutcome::CrashedGround:
      return "crashed-ground";
    case RaceOutcome::CrashedGate:
      return "crashed-gate";
    case RaceOutcome::TimedOut:
      return "timed-out";
    case RaceOutcome::SpawnFailed:
      return "spawn-failed";
  }
  return unknown(static_cast<std::uint16_t>(outcome));
}

Decoded<VehicleState> decode_telemetry(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<VehicleState, wire::Telemetry>(bytes, to_state);
}

Decoded<Command> decode_command(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<Command, wire::Command>(bytes, to_command);
}

Decoded<Event> decode_event(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<Event, wire::Event>(bytes, to_event);
}

Decoded<TickEnd> decode_tick_end(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<TickEnd, wire::TickEnd>(bytes, to_tick_end);
}

Decoded<RaceCourse> decode_race_course(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<RaceCourse, wire::RaceCourse>(bytes, to_race_course);
}

Decoded<RaceState> decode_race_state(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<RaceState, wire::RaceState>(bytes, to_race_state);
}

std::vector<std::uint8_t> encode_telemetry(const VehicleState& s) {
  capnp::MallocMessageBuilder message;
  auto t = message.initRoot<wire::Telemetry>();
  t.setAircraftId(s.aircraft_id);
  t.setSimTimeS(s.sim_time_s);
  t.setTick(s.tick);
  auto geo = t.initGeo();
  geo.setLatDeg(s.lat_deg);
  geo.setLonDeg(s.lon_deg);
  geo.setAltHaeM(s.alt_hae_m);
  auto enu = t.initEnu();
  enu.setE(s.e);
  enu.setN(s.n);
  enu.setU(s.u);
  auto vned = t.initVelNed();
  vned.setN(s.vel_ned_n);
  vned.setE(s.vel_ned_e);
  vned.setD(s.vel_ned_d);
  auto venu = t.initVelEnu();
  venu.setE(s.vel_enu_e);
  venu.setN(s.vel_enu_n);
  venu.setU(s.vel_enu_u);
  auto att = t.initAttitude();
  att.setPhiRad(s.roll);
  att.setThetaRad(s.pitch);
  att.setPsiRad(s.yaw);
  auto rates = t.initBodyRates();
  rates.setPRadS(s.p);
  rates.setQRadS(s.q);
  rates.setRRadS(s.r);
  t.setTasMps(s.tas_mps);
  auto aero = t.initAero();
  aero.setAlphaRad(s.alpha);
  aero.setBetaRad(s.beta);
  aero.setNzG(s.nz_g);
  aero.setMach(s.mach);
  t.setFaultFlags(s.fault_flags);
  t.setVehicleKind(enum_cast<wire::VehicleKind>(s.vehicle_kind));
  t.setControlMode(enum_cast<wire::ControlMode>(s.control_mode));
  auto combat = t.initCombat();
  combat.setTeam(s.team);
  combat.setHitPoints(s.hit_points);
  combat.setHitPointsMax(s.hit_points_max);
  combat.setAlive(s.alive);
  combat.setAmmo(s.ammo);
  t.setControlPeriodS(s.control_period_s);
  auto status = t.initCommand();
  status.setOwnerEpoch(s.command.owner_epoch);
  status.setLastSeq(s.command.last_seq);
  status.setRefusals(s.command.refusals);
  status.setLastRefusal(enum_cast<wire::Refusal>(s.command.last_refusal));
  status.setLastRefusedEpoch(s.command.last_refused_epoch);
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_command(const Command& m) {
  capnp::MallocMessageBuilder message;
  auto c = message.initRoot<wire::Command>();
  c.setTargetId(m.target_id);
  c.setSeq(m.seq);
  c.setTelemetryTick(m.telemetry_tick);
  c.setTrigger(m.trigger);
  c.setEpoch(m.epoch);
  auto control = c.getControl();
  std::visit(
      [&control](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, RawActuators>) {
          auto r = control.initRaw();
          r.setAileron(v.aileron);
          r.setElevator(v.elevator);
          r.setRudder(v.rudder);
          r.setThrottle(v.throttle);
        } else if constexpr (std::is_same_v<T, MultirotorRaw>) {
          auto r = control.initMultirotor();
          r.setRollRate(v.roll_rate);
          r.setPitchRate(v.pitch_rate);
          r.setYawRate(v.yaw_rate);
          r.setCollective(v.collective);
        } else {
          auto a = control.initAutopilot();
          a.setHeadingRad(v.heading_rad);
          a.setAltHaeM(v.alt_hae_m);
          a.setTasMps(v.tas_mps);
        }
      },
      m.control);
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_event(const Event& e) {
  capnp::MallocMessageBuilder message;
  auto ev = message.initRoot<wire::Event>();
  ev.setTick(e.tick);
  ev.setSimTimeS(e.sim_time_s);
  auto kind = ev.getKind();
  std::visit(
      [&kind](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, TriggerPressed>) {
          auto b = kind.initTriggerPressed();
          b.setAircraftId(v.aircraft_id);
          b.setAmmo(v.ammo);
        } else if constexpr (std::is_same_v<T, TriggerReleased>) {
          auto b = kind.initTriggerReleased();
          b.setAircraftId(v.aircraft_id);
          b.setRoundsFired(v.rounds_fired);
          b.setAmmo(v.ammo);
        } else if constexpr (std::is_same_v<T, Hit>) {
          auto b = kind.initHit();
          b.setShooterId(v.shooter_id);
          b.setTargetId(v.target_id);
          b.setDamage(v.damage);
          b.setHitPoints(v.hit_points);
          set_enu(b.initPosEnu(), v.pos_enu);
        } else if constexpr (std::is_same_v<T, Kill>) {
          auto b = kind.initKill();
          b.setVictimId(v.victim_id);
          if (v.killer_id) {
            b.getKiller().setAircraftId(*v.killer_id);
          } else {
            b.getKiller().setUnknown();
          }
          set_enu(b.initPosEnu(), v.pos_enu);
        } else if constexpr (std::is_same_v<T, ModeChange>) {
          auto b = kind.initModeChange();
          b.setAircraftId(v.aircraft_id);
          b.setFrom(enum_cast<wire::ControlMode>(v.from));
          b.setTo(enum_cast<wire::ControlMode>(v.to));
          b.setReason(enum_cast<wire::ModeReason>(v.reason));
        } else if constexpr (std::is_same_v<T, ControllerUp>) {
          kind.setControllerUp(v.aircraft_id);
        } else if constexpr (std::is_same_v<T, ControllerDown>) {
          kind.setControllerDown(v.aircraft_id);
        } else if constexpr (std::is_same_v<T, RunEnded>) {
          auto b = kind.initRunEnded();
          b.setReason(enum_cast<wire::EndReason>(v.reason));
          b.setMessage(capnp::Text::Reader(v.message.data(), v.message.size()));
        } else if constexpr (std::is_same_v<T, GatePassed>) {
          auto b = kind.initGatePassed();
          b.setAircraftId(v.aircraft_id);
          b.setGate(v.gate);
          b.setOffsetXM(v.offset_x_m);
          b.setOffsetYM(v.offset_y_m);
        } else if constexpr (std::is_same_v<T, GateMissed>) {
          auto b = kind.initGateMissed();
          b.setAircraftId(v.aircraft_id);
          b.setGate(v.gate);
        } else {
          static_assert(std::is_same_v<T, Crashed>);
          auto b = kind.initCrashed();
          b.setAircraftId(v.aircraft_id);
          b.setOutcome(enum_cast<wire::RaceOutcome>(v.outcome));
          b.setGate(v.gate);
        }
      },
      e.detail);
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_tick_end(const TickEnd& tick_end) {
  capnp::MallocMessageBuilder message;
  auto t = message.initRoot<wire::TickEnd>();
  t.setTick(tick_end.tick);
  t.setSimTimeS(tick_end.sim_time_s);
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_race_course(const RaceCourse& c) {
  capnp::MallocMessageBuilder message;
  auto b = message.initRoot<wire::RaceCourse>();
  b.setSeed(c.seed);
  b.setOpeningM(c.opening_m);
  b.setBarM(c.bar_m);
  b.setTopBarM(c.top_bar_m);
  auto gates = b.initGates(static_cast<unsigned>(c.gates.size()));
  for (std::size_t i = 0; i < c.gates.size(); ++i) {
    const RaceGate& g = c.gates[i];
    auto w = gates[static_cast<unsigned>(i)];
    w.setIndex(g.index);
    set_enu(w.initCenter(), g.center);
    w.setYawRad(g.yaw_rad);
    set_enu(w.initNormal(), g.normal);
    set_enu(w.initRight(), g.right);
    set_enu(w.initUp(), g.up);
  }
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_race_state(const RaceState& s) {
  capnp::MallocMessageBuilder message;
  auto b = message.initRoot<wire::RaceState>();
  b.setAircraftId(s.aircraft_id);
  b.setPendingGate(s.pending_gate);
  b.setPassed(s.passed);
  b.setMissed(s.missed);
  b.setGateCount(s.gate_count);
  b.setLastPassS(s.last_pass_s);
  b.setSimTimeS(s.sim_time_s);
  b.setFinished(s.finished);
  b.setOutcome(enum_cast<wire::RaceOutcome>(s.outcome));
  return to_bytes(message);
}

}  // namespace astrum
