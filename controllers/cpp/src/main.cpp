// ASTRUM example flight controller (C++).
//
// A minimal out-of-sim controller that talks to `astrum-sim` purely over ZMQ,
// using the shared Cap'n Proto schema (`astrum.capnp`) for the wire format:
//
//   * SUB  tcp://sim:5555 — telemetry for one aircraft (topic = 4-byte BE id)
//   * PUSH tcp://sim:5556 — commands (target id + raw actuators | autopilot)
//
// Two modes (env `ASTRUM_CONTROL_MODE`):
//   * "raw" (default) — closes a heading/altitude/airspeed hold loop *here*,
//     in the controller, sending raw `fcs/*-cmd-norm` deflections each tick.
//     This is ASTRUM's intended usage: the user owns the control law.
//   * "autopilot" — sends a single high-level target command and then just
//     monitors, letting the sim's built-in inner loop fly it.
//
// Config (env):
//   ASTRUM_TELEMETRY_CONNECT  (default tcp://127.0.0.1:5555)
//   ASTRUM_COMMAND_CONNECT    (default tcp://127.0.0.1:5556)
//   ASTRUM_TARGET_ID          (default 1)
//   ASTRUM_TARGET_HEADING_DEG (default 0    — due north)
//   ASTRUM_TARGET_ALT_M       (default 1500)
//   ASTRUM_TARGET_SPD_MPS     (default 50)
//   ASTRUM_CONTROL_MODE       (default raw)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include <zmq.hpp>

#include "astrum.capnp.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string env_or(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : fallback;
}

double env_f64(const char* key, double fallback) {
  const char* v = std::getenv(key);
  return v ? std::strtod(v, nullptr) : fallback;
}

// 4-byte big-endian aircraft id — matches astrum-net's topic frame.
std::string topic_for(uint32_t id) {
  std::string t(4, '\0');
  t[0] = static_cast<char>((id >> 24) & 0xFF);
  t[1] = static_cast<char>((id >> 16) & 0xFF);
  t[2] = static_cast<char>((id >> 8) & 0xFF);
  t[3] = static_cast<char>(id & 0xFF);
  return t;
}

double wrap_pi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

double clampd(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

// Serialise a Command and PUSH it.
void send_command(zmq::socket_t& cmd_sock, capnp::MallocMessageBuilder& message) {
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  kj::ArrayPtr<kj::byte> bytes = flat.asBytes();
  cmd_sock.send(zmq::buffer(bytes.begin(), bytes.size()), zmq::send_flags::none);
}

// ---- Raw-actuator hold loop (heading + altitude + airspeed) ----------------
// Same control signs verified against the c172x: aileron+ rolls right,
// elevator+ pitches down. Rate damping uses attitude differences between
// telemetry frames (the stream carries no body rates).
struct HoldLaw {
  double target_hdg, target_alt, target_spd;
  bool have_prev = false;
  double prev_phi = 0, prev_theta = 0, prev_t = 0;

  void compute(const Telemetry::Reader& t, double& aileron, double& elevator,
               double& throttle) {
    auto att = t.getAttitude();
    const double phi = att.getPhiRad();
    const double theta = att.getThetaRad();
    const double psi = att.getPsiRad();
    const double alt = t.getGeo().getAltM();
    const double airspeed = t.getAirspeedMps();
    const double now = t.getSimTimeS();

    double p_est = 0, q_est = 0;
    if (have_prev && now > prev_t) {
      const double dt = now - prev_t;
      p_est = (phi - prev_phi) / dt;
      q_est = (theta - prev_theta) / dt;
    }

    const double max_roll = 25.0 * kPi / 180.0;
    const double max_pitch = 10.0 * kPi / 180.0;

    const double hdg_err = wrap_pi(target_hdg - psi);
    const double roll_target = clampd(1.2 * hdg_err, -max_roll, max_roll);
    aileron = clampd(1.5 * (roll_target - phi) - 0.3 * p_est, -1.0, 1.0);

    const double pitch_target = clampd(0.01 * (target_alt - alt), -max_pitch, max_pitch);
    elevator = clampd(-2.0 * (pitch_target - theta) + 0.5 * q_est, -1.0, 1.0);

    throttle = clampd(0.6 + 0.02 * (target_spd - airspeed), 0.0, 1.0);

    prev_phi = phi;
    prev_theta = theta;
    prev_t = now;
    have_prev = true;
  }
};

}  // namespace

int main() {
  const std::string tlm_endpoint = env_or("ASTRUM_TELEMETRY_CONNECT", "tcp://127.0.0.1:5555");
  const std::string cmd_endpoint = env_or("ASTRUM_COMMAND_CONNECT", "tcp://127.0.0.1:5556");
  const uint32_t target_id = static_cast<uint32_t>(std::strtoul(
      env_or("ASTRUM_TARGET_ID", "1").c_str(), nullptr, 10));
  const double target_hdg = env_f64("ASTRUM_TARGET_HEADING_DEG", 0.0) * kPi / 180.0;
  const double target_alt = env_f64("ASTRUM_TARGET_ALT_M", 1500.0);
  const double target_spd = env_f64("ASTRUM_TARGET_SPD_MPS", 50.0);
  const std::string mode = env_or("ASTRUM_CONTROL_MODE", "raw");

  zmq::context_t ctx(1);

  zmq::socket_t tlm(ctx, zmq::socket_type::sub);
  tlm.connect(tlm_endpoint);
  tlm.set(zmq::sockopt::subscribe, topic_for(target_id));

  zmq::socket_t cmd(ctx, zmq::socket_type::push);
  cmd.connect(cmd_endpoint);

  std::cerr << "[controller] target=" << target_id << " mode=" << mode
            << " telemetry=" << tlm_endpoint << " command=" << cmd_endpoint << "\n";

  // Autopilot mode: one high-level target command, then monitor.
  if (mode == "autopilot") {
    capnp::MallocMessageBuilder message;
    Command::Builder c = message.initRoot<Command>();
    c.setTargetId(target_id);
    auto ap = c.getControl().initAutopilot();
    ap.setHeadingRad(target_hdg);
    ap.setAltMslM(target_alt);
    ap.setAirspeedMps(target_spd);
    // PUSH connect is async; give it a moment to establish before the send.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    send_command(cmd, message);
    std::cerr << "[controller] sent autopilot target hdg=" << target_hdg
              << "rad alt=" << target_alt << "m spd=" << target_spd << "m/s\n";
  }

  HoldLaw law{target_hdg, target_alt, target_spd};
  uint64_t n = 0;

  while (true) {
    zmq::message_t topic;
    auto res = tlm.recv(topic, zmq::recv_flags::none);
    if (!res) continue;
    zmq::message_t payload;
    if (topic.more()) {
      if (!tlm.recv(payload, zmq::recv_flags::none)) continue;
    } else {
      payload = std::move(topic); // single-frame fallback
    }

    // Copy into a word-aligned buffer for the Cap'n Proto reader.
    const size_t nbytes = payload.size();
    if (nbytes % sizeof(capnp::word) != 0) continue;
    auto words = kj::heapArray<capnp::word>(nbytes / sizeof(capnp::word));
    std::memcpy(words.begin(), payload.data(), nbytes);
    capnp::FlatArrayMessageReader reader(words.asPtr());
    Telemetry::Reader tel = reader.getRoot<Telemetry>();

    if (mode != "raw") {
      // Autopilot mode just reports tracking progress.
      if (++n % 120 == 0) {
        const double psi = tel.getAttitude().getPsiRad();
        std::cerr << "[controller] t=" << tel.getSimTimeS() << "s hdg="
                  << psi * 180.0 / kPi << "deg alt=" << tel.getGeo().getAltM() << "m\n";
      }
      continue;
    }

    double aileron, elevator, throttle;
    law.compute(tel, aileron, elevator, throttle);

    capnp::MallocMessageBuilder message;
    Command::Builder c = message.initRoot<Command>();
    c.setTargetId(target_id);
    auto raw = c.getControl().initRaw();
    raw.setAileron(aileron);
    raw.setElevator(elevator);
    raw.setRudder(0.0);
    raw.setThrottle(throttle);
    send_command(cmd, message);

    if (++n % 120 == 0) {
      const double psi = tel.getAttitude().getPsiRad();
      std::cerr << "[controller] t=" << tel.getSimTimeS() << "s hdg="
                << psi * 180.0 / kPi << "deg alt=" << tel.getGeo().getAltM()
                << "m ail=" << aileron << " elev=" << elevator << " thr=" << throttle << "\n";
    }
  }

  return 0;
}
