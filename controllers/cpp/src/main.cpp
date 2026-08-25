// ASTRUM example flight controller (C++).
//
// A minimal out-of-sim controller that talks to `astrum-sim` purely over Zenoh,
// using the shared Cap'n Proto schema (`astrum.capnp`) for the wire format:
//
//   * SUB  astrum/telemetry/<id> — telemetry for one aircraft
//   * PUB  astrum/command/<id>   — commands (target id + raw actuators | autopilot)
//
// Both share one Zenoh session (one TCP endpoint) instead of ZMQ's two ports.
// Telemetry drives the loop: a subscriber callback fires on each frame, and in
// raw mode computes and publishes the next command inline (Zenoh invokes a
// subscriber's callbacks serially, so the hold-law state needs no locking).
//
// Two modes (env `ASTRUM_CONTROL_MODE`):
//   * "raw" (default) — closes a heading/altitude/airspeed hold loop *here*,
//     in the controller, sending raw `fcs/*-cmd-norm` deflections each tick.
//     This is ASTRUM's intended usage: the user owns the control law.
//   * "autopilot" — sends one high-level target (reasserted over the first second
//     of telemetry to clear the startup race) and then just monitors, letting the
//     sim's built-in inner loop fly it.
//
// Config (env):
//   ASTRUM_ZENOH_CONNECT      (default tcp/127.0.0.1:7447; tcp/sim:7447 under compose)
//   ASTRUM_TARGET_ID          (default 1)
//   ASTRUM_TARGET_HEADING_DEG (default 0    — due north)
//   ASTRUM_TARGET_ALT_M       (default 1500)
//   ASTRUM_TARGET_SPD_MPS     (default 50)
//   ASTRUM_CONTROL_MODE       (default raw)

#include <algorithm>
#include <atomic>
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
#include <zenoh.hxx>

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

double wrap_pi(double a) {
  while (a > kPi) a -= 2.0 * kPi;
  while (a < -kPi) a += 2.0 * kPi;
  return a;
}

double clampd(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

// Serialise a Command to an owned Zenoh payload (Zenoh copies it out of the
// temporary string, so the capnp buffer can be released immediately).
zenoh::Bytes command_bytes(capnp::MallocMessageBuilder& message) {
  kj::Array<capnp::word> flat = capnp::messageToFlatArray(message);
  kj::ArrayPtr<kj::byte> bytes = flat.asBytes();
  return zenoh::Bytes(std::string(reinterpret_cast<const char*>(bytes.begin()), bytes.size()));
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
  using namespace zenoh;

  const std::string connect = env_or("ASTRUM_ZENOH_CONNECT", "tcp/127.0.0.1:7447");
  const uint32_t target_id = static_cast<uint32_t>(
      std::strtoul(env_or("ASTRUM_TARGET_ID", "1").c_str(), nullptr, 10));
  const double target_hdg = env_f64("ASTRUM_TARGET_HEADING_DEG", 0.0) * kPi / 180.0;
  const double target_alt = env_f64("ASTRUM_TARGET_ALT_M", 1500.0);
  const double target_spd = env_f64("ASTRUM_TARGET_SPD_MPS", 50.0);
  const std::string mode = env_or("ASTRUM_CONTROL_MODE", "raw");

  const std::string telemetry_key = "astrum/telemetry/" + std::to_string(target_id);
  const std::string command_key = "astrum/command/" + std::to_string(target_id);

  // Peer session onto the sim's single Zenoh endpoint (multicast off — Docker
  // bridge networks don't pass it, so peers connect by explicit endpoint).
  // Peer mode makes open() non-fatal when the sim isn't listening yet (it spends
  // ~10s loading JSBSim before it binds): open returns immediately and Zenoh
  // retries the connection in the background, so a controller started alongside
  // the sim just attaches once it comes up — mirroring ZMQ's lazy connect. A
  // client would instead abort open() on a cold start. listen/endpoints=[] keeps
  // this peer purely connect-out (the link to the sim carries pub/sub both ways).
  Config config = Config::create_default();
  config.insert_json5("mode", "\"peer\"");
  config.insert_json5("connect/endpoints", "[\"" + connect + "\"]");
  config.insert_json5("listen/endpoints", "[]");
  config.insert_json5("scouting/multicast/enabled", "false");
  auto session = Session::open(std::move(config));

  auto publisher = session.declare_publisher(KeyExpr(command_key));

  std::cerr << "[controller] target=" << target_id << " mode=" << mode
            << " connect=" << connect << " telemetry=" << telemetry_key
            << " command=" << command_key << "\n";

  if (mode == "autopilot") {
    std::cerr << "[controller] autopilot target hdg=" << target_hdg << "rad alt="
              << target_alt << "m spd=" << target_spd
              << "m/s (asserted once telemetry confirms the link)\n";
  }

  HoldLaw law{target_hdg, target_alt, target_spd};
  std::atomic<uint64_t> n{0};

  // Telemetry callback: monitor (autopilot mode) or fly the raw hold loop.
  // Zenoh serialises a subscriber's callbacks, so HoldLaw state is single-owner.
  auto on_sample = [&](const Sample& sample) {
    auto payload = sample.get_payload().as_vector();
    const size_t nbytes = payload.size();
    if (nbytes % sizeof(capnp::word) != 0) return;
    auto words = kj::heapArray<capnp::word>(nbytes / sizeof(capnp::word));
    std::memcpy(words.begin(), payload.data(), nbytes);
    capnp::FlatArrayMessageReader reader(words.asPtr());
    Telemetry::Reader tel = reader.getRoot<Telemetry>();

    if (mode != "raw") {
      const uint64_t count = ++n;
      // Autopilot mode: reassert the one high-level target over the first second
      // of telemetry, then monitor. Receiving telemetry proves the bidirectional
      // peer link is up, so the command now reaches the sim; resending briefly
      // guarantees delivery despite the startup race — Zenoh drops a put with no
      // matching subscriber (unlike ZMQ PUSH, which queues it), so a single shot
      // at t=0 would be lost while the sim spends ~10s loading JSBSim.
      if (count <= 120) {
        capnp::MallocMessageBuilder message;
        Command::Builder c = message.initRoot<Command>();
        c.setTargetId(target_id);
        auto ap = c.getControl().initAutopilot();
        ap.setHeadingRad(target_hdg);
        ap.setAltMslM(target_alt);
        ap.setAirspeedMps(target_spd);
        publisher.put(command_bytes(message));
      }
      if (count % 120 == 0) {
        const double psi = tel.getAttitude().getPsiRad();
        std::cerr << "[controller] t=" << tel.getSimTimeS() << "s hdg="
                  << psi * 180.0 / kPi << "deg alt=" << tel.getGeo().getAltM() << "m\n";
      }
      return;
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
    publisher.put(command_bytes(message));

    if (++n % 120 == 0) {
      const double psi = tel.getAttitude().getPsiRad();
      std::cerr << "[controller] t=" << tel.getSimTimeS() << "s hdg="
                << psi * 180.0 / kPi << "deg alt=" << tel.getGeo().getAltM()
                << "m ail=" << aileron << " elev=" << elevator << " thr=" << throttle << "\n";
    }
  };

  auto subscriber = session.declare_subscriber(KeyExpr(telemetry_key), on_sample, []() {});

  // The callback drives everything; park the main thread. The demo tears the
  // controller down when the bounded sim exits.
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return 0;
}
