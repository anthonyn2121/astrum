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

#include "astrum.capnp.h"

namespace astrum {
namespace {

// Read one flat message from `bytes` and convert its root with `convert`. The payload is
// read in place when word-aligned and copied once otherwise. Every failure (framing,
// bounds, traversal limits, trailing bytes) becomes Decoded::error.
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

VehicleState to_state(wire::Telemetry::Reader t) {
  VehicleState s;
  s.t = t.getSimTimeS();
  s.tick = t.getTick();
  s.aircraft_id = t.getAircraftId();
  const auto geo = t.getGeo();
  s.lat_deg = geo.getLatDeg();
  s.lon_deg = geo.getLonDeg();
  s.alt_m = geo.getAltHaeM();
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
  s.airspeed = t.getTasMps();
  const auto aero = t.getAero();
  s.alpha = aero.getAlphaRad();
  s.beta = aero.getBetaRad();
  s.nz_g = aero.getNzG();
  s.mach = aero.getMach();
  s.fault_flags = t.getFaultFlags();
  s.vehicle_kind = static_cast<VehicleKind>(static_cast<std::uint16_t>(t.getVehicleKind()));
  s.control_mode = static_cast<ControlMode>(static_cast<std::uint16_t>(t.getControlMode()));
  const auto combat = t.getCombat();
  s.team = combat.getTeam();
  s.hit_points = combat.getHitPoints();
  s.hit_points_max = combat.getHitPointsMax();
  s.alive = combat.getAlive();
  s.ammo = combat.getAmmo();
  return s;
}

CommandMessage to_message(wire::Command::Reader c) {
  CommandMessage m;
  m.target_id = c.getTargetId();
  m.seq = c.getSeq();
  m.telemetry_tick = c.getTelemetryTick();
  m.command.trigger = c.getTrigger();
  const auto control = c.getControl();
  switch (control.which()) {
    case wire::Command::Control::RAW: {
      const auto raw = control.getRaw();
      m.command.control =
          RawActuators{raw.getAileron(), raw.getElevator(), raw.getRudder(), raw.getThrottle()};
      break;
    }
    case wire::Command::Control::AUTOPILOT: {
      const auto ap = control.getAutopilot();
      m.command.control = AutopilotTargets{ap.getHeadingRad(), ap.getAltHaeM(), ap.getTasMps()};
      break;
    }
    default:
      throw std::runtime_error("unknown command control variant " +
                               std::to_string(static_cast<unsigned>(control.which())));
  }
  return m;
}

}  // namespace

std::string to_string(VehicleKind kind) {
  switch (kind) {
    case VehicleKind::FixedWing:
      return "fixed-wing";
    case VehicleKind::Multirotor:
      return "multirotor";
  }
  return "unknown(" + std::to_string(static_cast<std::uint16_t>(kind)) + ")";
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
  return "unknown(" + std::to_string(static_cast<std::uint16_t>(mode)) + ")";
}

Decoded<VehicleState> decode_telemetry(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<VehicleState, wire::Telemetry>(bytes, to_state);
}

Decoded<CommandMessage> decode_command(std::span<const std::uint8_t> bytes) noexcept {
  return decode_flat<CommandMessage, wire::Command>(bytes, to_message);
}

std::vector<std::uint8_t> encode_telemetry(const VehicleState& s) {
  capnp::MallocMessageBuilder message;
  auto t = message.initRoot<wire::Telemetry>();
  t.setAircraftId(s.aircraft_id);
  t.setSimTimeS(s.t);
  t.setTick(s.tick);
  auto geo = t.initGeo();
  geo.setLatDeg(s.lat_deg);
  geo.setLonDeg(s.lon_deg);
  geo.setAltHaeM(s.alt_m);
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
  t.setTasMps(s.airspeed);
  auto aero = t.initAero();
  aero.setAlphaRad(s.alpha);
  aero.setBetaRad(s.beta);
  aero.setNzG(s.nz_g);
  aero.setMach(s.mach);
  t.setFaultFlags(s.fault_flags);
  t.setVehicleKind(static_cast<wire::VehicleKind>(static_cast<std::uint16_t>(s.vehicle_kind)));
  t.setControlMode(static_cast<wire::ControlMode>(static_cast<std::uint16_t>(s.control_mode)));
  auto combat = t.initCombat();
  combat.setTeam(s.team);
  combat.setHitPoints(s.hit_points);
  combat.setHitPointsMax(s.hit_points_max);
  combat.setAlive(s.alive);
  combat.setAmmo(s.ammo);
  return to_bytes(message);
}

std::vector<std::uint8_t> encode_command(const CommandMessage& m) {
  capnp::MallocMessageBuilder message;
  auto c = message.initRoot<wire::Command>();
  c.setTargetId(m.target_id);
  c.setSeq(m.seq);
  c.setTelemetryTick(m.telemetry_tick);
  c.setTrigger(m.command.trigger);
  auto control = c.getControl();
  if (const auto* raw = std::get_if<RawActuators>(&m.command.control)) {
    auto r = control.initRaw();
    r.setAileron(raw->aileron);
    r.setElevator(raw->elevator);
    r.setRudder(raw->rudder);
    r.setThrottle(raw->throttle);
  } else {
    const auto& ap = std::get<AutopilotTargets>(m.command.control);
    auto a = control.initAutopilot();
    a.setHeadingRad(ap.heading_rad);
    a.setAltHaeM(ap.alt_m);
    a.setTasMps(ap.tas_mps);
  }
  return to_bytes(message);
}

}  // namespace astrum
