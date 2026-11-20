// Cross-language wire fixtures for schema v3 (crates/astrum-proto/tests/fixtures).
//
//   astrum-wire-fixture write  <dir>   encode every kind in <dir>/values.json into
//                                      <dir>/<kind>.cpp.bin
//   astrum-wire-fixture verify <dir>   decode every <dir>/<kind>.<lang>.bin present and
//                                      compare it exactly to values.json
//
// A kind is a values.json key: telemetry, command_* (Command), tick_end, race_course,
// race_state, event_* (Event, in the shared event JSON shape: enums as kebab-case names,
// pos_enu [e, n, u], killer_id null = unknown); lang in {rust, cpp, python}. `verify` turns each
// decoded value back into that JSON shape and diffs it against values.json, so a field that is
// dropped, renamed or mis-decoded shows up by its path. It runs as the ctest
// `wire_fixtures`; exit status 0 = every present file matches.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "astrum/types.hpp"
#include "astrum/wire.hpp"

namespace {

namespace fs = std::filesystem;
using nlohmann::json;
using namespace astrum;

constexpr const char* kLangs[] = {"rust", "cpp", "python"};

// --- values.json -> types ---------------------------------------------------------------

template <class T>
T unsigned_at(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_number_unsigned() || v.get<std::uint64_t>() > std::numeric_limits<T>::max()) {
    throw std::runtime_error(std::string(key) + " must be an unsigned integer in range");
  }
  return static_cast<T>(v.get<std::uint64_t>());
}

double number_at(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_number()) throw std::runtime_error(std::string(key) + " must be a number");
  return v.get<double>();
}

bool bool_at(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_boolean()) throw std::runtime_error(std::string(key) + " must be a boolean");
  return v.get<bool>();
}

std::string string_at(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_string()) throw std::runtime_error(std::string(key) + " must be a string");
  return v.get<std::string>();
}

// The enumerant whose to_string() is `name` (values 0 .. count-1).
template <class E>
E enum_at(const json& j, const char* key, std::uint16_t count) {
  const std::string name = string_at(j, key);
  for (std::uint16_t v = 0; v < count; ++v) {
    if (to_string(static_cast<E>(v)) == name) return static_cast<E>(v);
  }
  throw std::runtime_error(std::string("bad ") + key + " '" + name + "'");
}

Enu enu_object(const json& j) {
  return Enu{number_at(j, "e"), number_at(j, "n"), number_at(j, "u")};
}

Enu enu_array(const json& j) {
  if (!j.is_array() || j.size() != 3) throw std::runtime_error("pos_enu must be [e, n, u]");
  for (const json& v : j) {
    if (!v.is_number()) throw std::runtime_error("pos_enu must hold numbers");
  }
  return Enu{j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
}

VehicleState telemetry_from(const json& v) {
  VehicleState s;
  s.aircraft_id = unsigned_at<std::uint32_t>(v, "aircraft_id");
  s.tick = unsigned_at<std::uint64_t>(v, "tick");
  s.sim_time_s = number_at(v, "sim_time_s");
  s.lat_deg = number_at(v, "lat_deg");
  s.lon_deg = number_at(v, "lon_deg");
  s.alt_hae_m = number_at(v, "alt_hae_m");
  const Enu pos = enu_object(v.at("pos_enu"));
  s.e = pos.e;
  s.n = pos.n;
  s.u = pos.u;
  const Enu venu = enu_object(v.at("vel_enu"));
  s.vel_enu_e = venu.e;
  s.vel_enu_n = venu.n;
  s.vel_enu_u = venu.u;
  const json& vned = v.at("vel_ned");
  s.vel_ned_n = number_at(vned, "n");
  s.vel_ned_e = number_at(vned, "e");
  s.vel_ned_d = number_at(vned, "d");
  const json& att = v.at("attitude");
  s.roll = number_at(att, "phi_rad");
  s.pitch = number_at(att, "theta_rad");
  s.yaw = number_at(att, "psi_rad");
  const json& rates = v.at("rates");
  s.p = number_at(rates, "p_rad_s");
  s.q = number_at(rates, "q_rad_s");
  s.r = number_at(rates, "r_rad_s");
  const json& aero = v.at("aero");
  s.alpha = number_at(aero, "alpha_rad");
  s.beta = number_at(aero, "beta_rad");
  s.nz_g = number_at(aero, "nz_g");
  s.mach = number_at(aero, "mach");
  s.tas_mps = number_at(v, "tas_mps");
  s.fault_flags = unsigned_at<std::uint8_t>(v, "fault_flags");
  s.vehicle_kind = enum_at<VehicleKind>(v, "vehicle_kind", 2);
  s.control_mode = enum_at<ControlMode>(v, "control_mode", 4);
  const json& combat = v.at("combat");
  s.team = unsigned_at<std::uint8_t>(combat, "team");
  s.hit_points = static_cast<float>(number_at(combat, "hit_points"));
  s.hit_points_max = static_cast<float>(number_at(combat, "hit_points_max"));
  s.alive = bool_at(combat, "alive");
  s.ammo = unsigned_at<std::uint16_t>(combat, "ammo");
  s.control_period_s = number_at(v, "control_period_s");
  const json& c = v.at("command");
  s.command.owner_epoch = unsigned_at<std::uint64_t>(c, "owner_epoch");
  s.command.last_seq = unsigned_at<std::uint64_t>(c, "last_seq");
  s.command.refusals = unsigned_at<std::uint32_t>(c, "refusals");
  s.command.last_refusal = enum_at<Refusal>(c, "last_refusal", 8);
  s.command.last_refused_epoch = unsigned_at<std::uint64_t>(c, "last_refused_epoch");
  return s;
}

Command command_from(const json& v) {
  Command m;
  m.target_id = unsigned_at<std::uint32_t>(v, "target_id");
  m.seq = unsigned_at<std::uint64_t>(v, "seq");
  m.telemetry_tick = unsigned_at<std::uint64_t>(v, "telemetry_tick");
  m.trigger = bool_at(v, "trigger");
  m.epoch = unsigned_at<std::uint64_t>(v, "epoch");
  const json& control = v.at("control");
  if (control.size() != 1) throw std::runtime_error("control must have exactly one variant");
  if (control.contains("raw")) {
    const json& r = control.at("raw");
    m.control = RawActuators{number_at(r, "aileron"), number_at(r, "elevator"),
                             number_at(r, "rudder"), number_at(r, "throttle")};
  } else if (control.contains("multirotor")) {
    const json& r = control.at("multirotor");
    m.control = MultirotorRaw{number_at(r, "roll_rate"), number_at(r, "pitch_rate"),
                              number_at(r, "yaw_rate"), number_at(r, "collective")};
  } else {
    const json& a = control.at("autopilot");
    m.control = AutopilotTargets{number_at(a, "heading_rad"), number_at(a, "alt_hae_m"),
                                 number_at(a, "tas_mps")};
  }
  return m;
}

TickEnd tick_end_from(const json& v) {
  return TickEnd{unsigned_at<std::uint64_t>(v, "tick"), number_at(v, "sim_time_s")};
}

RaceCourse race_course_from(const json& v) {
  RaceCourse c;
  c.seed = unsigned_at<std::uint64_t>(v, "seed");
  c.opening_m = number_at(v, "opening_m");
  c.bar_m = number_at(v, "bar_m");
  c.top_bar_m = number_at(v, "top_bar_m");
  for (const json& g : v.at("gates")) {
    c.gates.push_back(RaceGate{unsigned_at<std::uint32_t>(g, "index"), enu_object(g.at("center")),
                               number_at(g, "yaw_rad"), enu_object(g.at("normal")),
                               enu_object(g.at("right")), enu_object(g.at("up"))});
  }
  return c;
}

RaceState race_state_from(const json& v) {
  return RaceState{unsigned_at<std::uint32_t>(v, "aircraft_id"),
                   unsigned_at<std::uint32_t>(v, "pending_gate"),
                   unsigned_at<std::uint32_t>(v, "passed"),
                   unsigned_at<std::uint32_t>(v, "missed"),
                   unsigned_at<std::uint32_t>(v, "gate_count"),
                   number_at(v, "last_pass_s"),
                   number_at(v, "sim_time_s"),
                   bool_at(v, "finished"),
                   enum_at<RaceOutcome>(v, "outcome", 6)};
}

Event event_from(const json& v) {
  Event e;
  e.tick = unsigned_at<std::uint64_t>(v, "tick");
  e.sim_time_s = number_at(v, "sim_time_s");
  const std::string kind = string_at(v, "kind");
  if (kind == "trigger-pressed") {
    e.detail = TriggerPressed{unsigned_at<std::uint32_t>(v, "aircraft_id"),
                              unsigned_at<std::uint16_t>(v, "ammo")};
  } else if (kind == "trigger-released") {
    e.detail = TriggerReleased{unsigned_at<std::uint32_t>(v, "aircraft_id"),
                               unsigned_at<std::uint16_t>(v, "rounds_fired"),
                               unsigned_at<std::uint16_t>(v, "ammo")};
  } else if (kind == "hit") {
    e.detail =
        Hit{unsigned_at<std::uint32_t>(v, "shooter_id"), unsigned_at<std::uint32_t>(v, "target_id"),
            static_cast<float>(number_at(v, "damage")),
            static_cast<float>(number_at(v, "hit_points")), enu_array(v.at("pos_enu"))};
  } else if (kind == "kill") {
    Kill kill;
    kill.victim_id = unsigned_at<std::uint32_t>(v, "victim_id");
    if (!v.at("killer_id").is_null()) kill.killer_id = unsigned_at<std::uint32_t>(v, "killer_id");
    kill.pos_enu = enu_array(v.at("pos_enu"));
    e.detail = kill;
  } else if (kind == "mode-change") {
    e.detail =
        ModeChange{unsigned_at<std::uint32_t>(v, "aircraft_id"), enum_at<ControlMode>(v, "from", 4),
                   enum_at<ControlMode>(v, "to", 4), enum_at<ModeReason>(v, "reason", 3)};
  } else if (kind == "controller-up") {
    e.detail = ControllerUp{unsigned_at<std::uint32_t>(v, "aircraft_id")};
  } else if (kind == "controller-down") {
    e.detail = ControllerDown{unsigned_at<std::uint32_t>(v, "aircraft_id")};
  } else if (kind == "run-ended") {
    e.detail = RunEnded{enum_at<EndReason>(v, "reason", 6), string_at(v, "message")};
  } else if (kind == "gate-passed") {
    e.detail = GatePassed{unsigned_at<std::uint32_t>(v, "aircraft_id"),
                          unsigned_at<std::uint32_t>(v, "gate"), number_at(v, "offset_x_m"),
                          number_at(v, "offset_y_m")};
  } else if (kind == "gate-missed") {
    e.detail = GateMissed{unsigned_at<std::uint32_t>(v, "aircraft_id"),
                          unsigned_at<std::uint32_t>(v, "gate")};
  } else if (kind == "crashed") {
    e.detail =
        Crashed{unsigned_at<std::uint32_t>(v, "aircraft_id"), enum_at<RaceOutcome>(v, "outcome", 6),
                unsigned_at<std::uint32_t>(v, "gate")};
  } else {
    throw std::runtime_error("bad event kind '" + kind + "'");
  }
  return e;
}

// --- types -> values.json shape ---------------------------------------------------------

json to_json(const VehicleState& s) {
  return json{
      {"aircraft_id", s.aircraft_id},
      {"tick", s.tick},
      {"sim_time_s", s.sim_time_s},
      {"lat_deg", s.lat_deg},
      {"lon_deg", s.lon_deg},
      {"alt_hae_m", s.alt_hae_m},
      {"pos_enu", {{"e", s.e}, {"n", s.n}, {"u", s.u}}},
      {"vel_enu", {{"e", s.vel_enu_e}, {"n", s.vel_enu_n}, {"u", s.vel_enu_u}}},
      {"vel_ned", {{"n", s.vel_ned_n}, {"e", s.vel_ned_e}, {"d", s.vel_ned_d}}},
      {"attitude", {{"phi_rad", s.roll}, {"theta_rad", s.pitch}, {"psi_rad", s.yaw}}},
      {"rates", {{"p_rad_s", s.p}, {"q_rad_s", s.q}, {"r_rad_s", s.r}}},
      {"aero", {{"alpha_rad", s.alpha}, {"beta_rad", s.beta}, {"nz_g", s.nz_g}, {"mach", s.mach}}},
      {"tas_mps", s.tas_mps},
      {"fault_flags", s.fault_flags},
      {"vehicle_kind", to_string(s.vehicle_kind)},
      {"control_mode", to_string(s.control_mode)},
      {"combat",
       {{"team", s.team},
        {"hit_points", s.hit_points},
        {"hit_points_max", s.hit_points_max},
        {"alive", s.alive},
        {"ammo", s.ammo}}},
      {"control_period_s", s.control_period_s},
      {"command",
       {{"owner_epoch", s.command.owner_epoch},
        {"last_seq", s.command.last_seq},
        {"refusals", s.command.refusals},
        {"last_refusal", to_string(s.command.last_refusal)},
        {"last_refused_epoch", s.command.last_refused_epoch}}},
  };
}

json to_json(const Command& m) {
  json control = std::visit(
      [](const auto& c) -> json {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, RawActuators>) {
          return {{"raw",
                   {{"aileron", c.aileron},
                    {"elevator", c.elevator},
                    {"rudder", c.rudder},
                    {"throttle", c.throttle}}}};
        } else if constexpr (std::is_same_v<T, MultirotorRaw>) {
          return {{"multirotor",
                   {{"roll_rate", c.roll_rate},
                    {"pitch_rate", c.pitch_rate},
                    {"yaw_rate", c.yaw_rate},
                    {"collective", c.collective}}}};
        } else {
          return {{"autopilot",
                   {{"heading_rad", c.heading_rad},
                    {"alt_hae_m", c.alt_hae_m},
                    {"tas_mps", c.tas_mps}}}};
        }
      },
      m.control);
  return json{{"target_id", m.target_id}, {"seq", m.seq},     {"telemetry_tick", m.telemetry_tick},
              {"trigger", m.trigger},     {"epoch", m.epoch}, {"control", std::move(control)}};
}

json to_json(const TickEnd& t) { return json{{"tick", t.tick}, {"sim_time_s", t.sim_time_s}}; }

json enu_json(const Enu& v) { return json::array({v.e, v.n, v.u}); }

json enu_object_json(const Enu& v) { return json{{"e", v.e}, {"n", v.n}, {"u", v.u}}; }

json to_json(const RaceCourse& c) {
  json gates = json::array();
  for (const RaceGate& g : c.gates) {
    gates.push_back(json{{"index", g.index},
                         {"center", enu_object_json(g.center)},
                         {"yaw_rad", g.yaw_rad},
                         {"normal", enu_object_json(g.normal)},
                         {"right", enu_object_json(g.right)},
                         {"up", enu_object_json(g.up)}});
  }
  return json{{"seed", c.seed},
              {"opening_m", c.opening_m},
              {"bar_m", c.bar_m},
              {"top_bar_m", c.top_bar_m},
              {"gates", std::move(gates)}};
}

json to_json(const RaceState& s) {
  return json{{"aircraft_id", s.aircraft_id},
              {"pending_gate", s.pending_gate},
              {"passed", s.passed},
              {"missed", s.missed},
              {"gate_count", s.gate_count},
              {"last_pass_s", s.last_pass_s},
              {"sim_time_s", s.sim_time_s},
              {"finished", s.finished},
              {"outcome", to_string(s.outcome)}};
}

json to_json(const Event& e) {
  json out{{"tick", e.tick}, {"sim_time_s", e.sim_time_s}};
  std::visit(
      [&out](const auto& d) {
        using T = std::decay_t<decltype(d)>;
        if constexpr (std::is_same_v<T, TriggerPressed>) {
          out["kind"] = "trigger-pressed";
          out["aircraft_id"] = d.aircraft_id;
          out["ammo"] = d.ammo;
        } else if constexpr (std::is_same_v<T, TriggerReleased>) {
          out["kind"] = "trigger-released";
          out["aircraft_id"] = d.aircraft_id;
          out["rounds_fired"] = d.rounds_fired;
          out["ammo"] = d.ammo;
        } else if constexpr (std::is_same_v<T, Hit>) {
          out["kind"] = "hit";
          out["shooter_id"] = d.shooter_id;
          out["target_id"] = d.target_id;
          out["damage"] = d.damage;
          out["hit_points"] = d.hit_points;
          out["pos_enu"] = enu_json(d.pos_enu);
        } else if constexpr (std::is_same_v<T, Kill>) {
          out["kind"] = "kill";
          out["victim_id"] = d.victim_id;
          out["killer_id"] = d.killer_id ? json(*d.killer_id) : json(nullptr);
          out["pos_enu"] = enu_json(d.pos_enu);
        } else if constexpr (std::is_same_v<T, ModeChange>) {
          out["kind"] = "mode-change";
          out["aircraft_id"] = d.aircraft_id;
          out["from"] = to_string(d.from);
          out["to"] = to_string(d.to);
          out["reason"] = to_string(d.reason);
        } else if constexpr (std::is_same_v<T, ControllerUp>) {
          out["kind"] = "controller-up";
          out["aircraft_id"] = d.aircraft_id;
        } else if constexpr (std::is_same_v<T, ControllerDown>) {
          out["kind"] = "controller-down";
          out["aircraft_id"] = d.aircraft_id;
        } else if constexpr (std::is_same_v<T, RunEnded>) {
          out["kind"] = "run-ended";
          out["reason"] = to_string(d.reason);
          out["message"] = d.message;
        } else if constexpr (std::is_same_v<T, GatePassed>) {
          out["kind"] = "gate-passed";
          out["aircraft_id"] = d.aircraft_id;
          out["gate"] = d.gate;
          out["offset_x_m"] = d.offset_x_m;
          out["offset_y_m"] = d.offset_y_m;
        } else if constexpr (std::is_same_v<T, GateMissed>) {
          out["kind"] = "gate-missed";
          out["aircraft_id"] = d.aircraft_id;
          out["gate"] = d.gate;
        } else {
          static_assert(std::is_same_v<T, Crashed>);
          out["kind"] = "crashed";
          out["aircraft_id"] = d.aircraft_id;
          out["outcome"] = to_string(d.outcome);
          out["gate"] = d.gate;
        }
      },
      e.detail);
  return out;
}

// --- kinds ------------------------------------------------------------------------------

enum class Type { Telemetry, Command, TickEnd, RaceCourse, RaceState, Event };

Type type_of(const std::string& kind) {
  if (kind == "telemetry") return Type::Telemetry;
  if (kind == "tick_end") return Type::TickEnd;
  if (kind == "race_course") return Type::RaceCourse;
  if (kind == "race_state") return Type::RaceState;
  if (kind.rfind("command_", 0) == 0) return Type::Command;
  if (kind.rfind("event_", 0) == 0) return Type::Event;
  throw std::runtime_error("values.json kind '" + kind + "' has no known message type");
}

std::vector<std::string> kinds_of(const json& values) {
  std::vector<std::string> kinds;
  for (const auto& [key, value] : values.items()) {
    if (key.rfind('_', 0) == 0) continue;  // "_comment"
    type_of(key);                          // reject unknown kinds early
    kinds.push_back(key);
  }
  return kinds;
}

std::vector<std::uint8_t> encode(const std::string& kind, const json& value) {
  switch (type_of(kind)) {
    case Type::Telemetry:
      return encode_telemetry(telemetry_from(value));
    case Type::Command:
      return encode_command(command_from(value));
    case Type::TickEnd:
      return encode_tick_end(tick_end_from(value));
    case Type::RaceCourse:
      return encode_race_course(race_course_from(value));
    case Type::RaceState:
      return encode_race_state(race_state_from(value));
    case Type::Event:
      return encode_event(event_from(value));
  }
  throw std::logic_error("unreachable");
}

// The decoded payload in the values.json shape, or the decode error.
template <class T>
json decoded_json(const Decoded<T>& d) {
  if (!d) throw std::runtime_error("decode failed: " + d.error);
  return to_json(*d.value);
}

json decode(const std::string& kind, const std::vector<std::uint8_t>& bytes) {
  switch (type_of(kind)) {
    case Type::Telemetry:
      return decoded_json(decode_telemetry(bytes));
    case Type::Command:
      return decoded_json(decode_command(bytes));
    case Type::TickEnd:
      return decoded_json(decode_tick_end(bytes));
    case Type::RaceCourse:
      return decoded_json(decode_race_course(bytes));
    case Type::RaceState:
      return decoded_json(decode_race_state(bytes));
    case Type::Event:
      return decoded_json(decode_event(bytes));
  }
  throw std::logic_error("unreachable");
}

// Exact differences between `expected` and `actual`, by JSON path.
void diff(const std::string& path, const json& expected, const json& actual,
          std::vector<std::string>& out) {
  if (expected.is_object() && actual.is_object()) {
    for (const auto& [key, value] : expected.items()) {
      if (!actual.contains(key)) {
        out.push_back(path + "/" + key + ": not decoded");
      } else {
        diff(path + "/" + key, value, actual.at(key), out);
      }
    }
    for (const auto& [key, value] : actual.items()) {
      if (!expected.contains(key)) out.push_back(path + "/" + key + ": not in values.json");
    }
    return;
  }
  if (expected.is_array() && actual.is_array() && expected.size() == actual.size()) {
    for (std::size_t i = 0; i < expected.size(); ++i) {
      diff(path + "/" + std::to_string(i), expected[i], actual[i], out);
    }
    return;
  }
  if (expected != actual) {
    std::ostringstream os;
    os << path << ": expected " << expected.dump() << ", decoded " << actual.dump();
    out.push_back(os.str());
  }
}

json load_values(const fs::path& dir) {
  std::ifstream in(dir / "values.json");
  if (!in) throw std::runtime_error("cannot open " + (dir / "values.json").string());
  return json::parse(in);
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int write(const fs::path& dir) {
  const json values = load_values(dir);
  for (const std::string& kind : kinds_of(values)) {
    const std::vector<std::uint8_t> bytes = encode(kind, values.at(kind));
    const fs::path path = dir / (kind + ".cpp.bin");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + path.string());
    std::cout << "wrote " << path.string() << "\n";
  }
  return 0;
}

int verify(const fs::path& dir) {
  const json values = load_values(dir);
  int checked = 0;
  int failed = 0;
  for (const std::string& kind : kinds_of(values)) {
    for (const std::string lang : kLangs) {
      const fs::path path = dir / (kind + "." + lang + ".bin");
      if (!fs::exists(path)) {
        std::cout << "skip " << path.filename().string() << " (not present)\n";
        continue;
      }
      ++checked;
      std::vector<std::string> problems;
      try {
        diff("", values.at(kind), decode(kind, read_file(path)), problems);
      } catch (const std::exception& e) {
        problems.push_back(e.what());
      }
      std::cout << (problems.empty() ? "ok   " : "FAIL ") << path.filename().string() << "\n";
      for (const std::string& p : problems) std::cout << "       " << p << "\n";
      if (!problems.empty()) ++failed;
    }
  }
  if (checked == 0) {
    std::cout << "no fixture files in " << dir.string() << "\n";
    return 1;
  }
  std::cout << checked - failed << " of " << checked << " fixture files match\n";
  return failed == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || (std::string(argv[1]) != "write" && std::string(argv[1]) != "verify")) {
    std::cerr << "usage: " << argv[0] << " write|verify <fixtures-dir>\n";
    return 2;
  }
  try {
    const fs::path dir = argv[2];
    return std::string(argv[1]) == "write" ? write(dir) : verify(dir);
  } catch (const std::exception& e) {
    std::cerr << argv[0] << ": " << e.what() << "\n";
    return 1;
  }
}
