// Cross-language wire fixtures for schema v2 (crates/astrum-proto/tests/fixtures).
//
//   astrum-wire-fixture write  <dir>   encode <dir>/values.json into <dir>/<kind>.cpp.bin
//   astrum-wire-fixture verify <dir>   decode every <dir>/<kind>.<lang>.bin present and
//                                      compare every field exactly to values.json
//
// kind in {telemetry, command_raw, command_autopilot}, lang in {rust, cpp, python}.
// `verify` runs as the ctest `wire_fixtures`; exit status 0 = all present files match.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "astrum/types.hpp"
#include "astrum/wire.hpp"

namespace {

namespace fs = std::filesystem;
using astrum::CommandMessage;
using astrum::VehicleState;
using nlohmann::json;

constexpr const char* kKinds[] = {"telemetry", "command_raw", "command_autopilot"};
constexpr const char* kLangs[] = {"rust", "cpp", "python"};

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

astrum::VehicleKind vehicle_kind(const std::string& s) {
  if (s == "fixed-wing") return astrum::VehicleKind::FixedWing;
  if (s == "multirotor") return astrum::VehicleKind::Multirotor;
  throw std::runtime_error("bad vehicle_kind '" + s + "'");
}

astrum::ControlMode control_mode(const std::string& s) {
  if (s == "uncommanded") return astrum::ControlMode::Uncommanded;
  if (s == "raw") return astrum::ControlMode::Raw;
  if (s == "autopilot") return astrum::ControlMode::Autopilot;
  if (s == "failsafe") return astrum::ControlMode::Failsafe;
  throw std::runtime_error("bad control_mode '" + s + "'");
}

VehicleState telemetry_from(const json& v) {
  VehicleState s;
  s.aircraft_id = unsigned_at<std::uint32_t>(v, "aircraft_id");
  s.tick = unsigned_at<std::uint64_t>(v, "tick");
  s.t = number_at(v, "sim_time_s");
  s.lat_deg = number_at(v, "lat_deg");
  s.lon_deg = number_at(v, "lon_deg");
  s.alt_m = number_at(v, "alt_hae_m");
  const json& pos = v.at("pos_enu");
  s.e = number_at(pos, "e");
  s.n = number_at(pos, "n");
  s.u = number_at(pos, "u");
  const json& venu = v.at("vel_enu");
  s.vel_enu_e = number_at(venu, "e");
  s.vel_enu_n = number_at(venu, "n");
  s.vel_enu_u = number_at(venu, "u");
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
  s.airspeed = number_at(v, "tas_mps");
  s.fault_flags = unsigned_at<std::uint8_t>(v, "fault_flags");
  s.vehicle_kind = vehicle_kind(v.at("vehicle_kind").get<std::string>());
  s.control_mode = control_mode(v.at("control_mode").get<std::string>());
  const json& combat = v.at("combat");
  s.team = unsigned_at<std::uint8_t>(combat, "team");
  s.hit_points = static_cast<float>(number_at(combat, "hit_points"));
  s.hit_points_max = static_cast<float>(number_at(combat, "hit_points_max"));
  s.alive = combat.at("alive").get<bool>();
  s.ammo = unsigned_at<std::uint16_t>(combat, "ammo");
  return s;
}

CommandMessage command_from(const json& v) {
  CommandMessage m;
  m.target_id = unsigned_at<std::uint32_t>(v, "target_id");
  m.seq = unsigned_at<std::uint64_t>(v, "seq");
  m.telemetry_tick = unsigned_at<std::uint64_t>(v, "telemetry_tick");
  m.command.trigger = v.at("trigger").get<bool>();
  const json& control = v.at("control");
  if (control.contains("raw")) {
    const json& r = control.at("raw");
    m.command.control = astrum::RawActuators{number_at(r, "aileron"), number_at(r, "elevator"),
                                             number_at(r, "rudder"), number_at(r, "throttle")};
  } else {
    const json& a = control.at("autopilot");
    m.command.control = astrum::AutopilotTargets{
        number_at(a, "heading_rad"), number_at(a, "alt_hae_m"), number_at(a, "tas_mps")};
  }
  return m;
}

// Collects exact-equality mismatches as readable lines.
class Diff {
 public:
  template <class T>
  void field(const char* name, const T& expected, const T& actual) {
    if (expected == actual) return;
    std::ostringstream os;
    os.precision(17);
    os << name << ": expected " << +expected << ", decoded " << +actual;
    lines_.push_back(os.str());
  }
  void field(const char* name, astrum::VehicleKind expected, astrum::VehicleKind actual) {
    if (expected != actual) {
      lines_.push_back(std::string(name) + ": expected " + to_string(expected) + ", decoded " +
                       to_string(actual));
    }
  }
  void field(const char* name, astrum::ControlMode expected, astrum::ControlMode actual) {
    if (expected != actual) {
      lines_.push_back(std::string(name) + ": expected " + to_string(expected) + ", decoded " +
                       to_string(actual));
    }
  }
  void note(std::string line) { lines_.push_back(std::move(line)); }
  const std::vector<std::string>& lines() const { return lines_; }

 private:
  std::vector<std::string> lines_;
};

#define ASTRUM_FIELD(diff, e, a, member) (diff).field(#member, (e).member, (a).member)

void compare(Diff& d, const VehicleState& e, const VehicleState& a) {
  ASTRUM_FIELD(d, e, a, aircraft_id);
  ASTRUM_FIELD(d, e, a, tick);
  ASTRUM_FIELD(d, e, a, t);
  ASTRUM_FIELD(d, e, a, lat_deg);
  ASTRUM_FIELD(d, e, a, lon_deg);
  ASTRUM_FIELD(d, e, a, alt_m);
  ASTRUM_FIELD(d, e, a, e);
  ASTRUM_FIELD(d, e, a, n);
  ASTRUM_FIELD(d, e, a, u);
  ASTRUM_FIELD(d, e, a, vel_ned_n);
  ASTRUM_FIELD(d, e, a, vel_ned_e);
  ASTRUM_FIELD(d, e, a, vel_ned_d);
  ASTRUM_FIELD(d, e, a, vel_enu_e);
  ASTRUM_FIELD(d, e, a, vel_enu_n);
  ASTRUM_FIELD(d, e, a, vel_enu_u);
  ASTRUM_FIELD(d, e, a, roll);
  ASTRUM_FIELD(d, e, a, pitch);
  ASTRUM_FIELD(d, e, a, yaw);
  ASTRUM_FIELD(d, e, a, p);
  ASTRUM_FIELD(d, e, a, q);
  ASTRUM_FIELD(d, e, a, r);
  ASTRUM_FIELD(d, e, a, airspeed);
  ASTRUM_FIELD(d, e, a, alpha);
  ASTRUM_FIELD(d, e, a, beta);
  ASTRUM_FIELD(d, e, a, nz_g);
  ASTRUM_FIELD(d, e, a, mach);
  ASTRUM_FIELD(d, e, a, fault_flags);
  ASTRUM_FIELD(d, e, a, vehicle_kind);
  ASTRUM_FIELD(d, e, a, control_mode);
  ASTRUM_FIELD(d, e, a, team);
  ASTRUM_FIELD(d, e, a, hit_points);
  ASTRUM_FIELD(d, e, a, hit_points_max);
  ASTRUM_FIELD(d, e, a, alive);
  ASTRUM_FIELD(d, e, a, ammo);
}

void compare(Diff& d, const CommandMessage& e, const CommandMessage& a) {
  ASTRUM_FIELD(d, e, a, target_id);
  ASTRUM_FIELD(d, e, a, seq);
  ASTRUM_FIELD(d, e, a, telemetry_tick);
  ASTRUM_FIELD(d, e, a, command.trigger);
  const auto* er = std::get_if<astrum::RawActuators>(&e.command.control);
  const auto* ar = std::get_if<astrum::RawActuators>(&a.command.control);
  if ((er == nullptr) != (ar == nullptr)) {
    d.note(std::string("control: expected ") + (er ? "raw" : "autopilot") + ", decoded " +
           (ar ? "raw" : "autopilot"));
  } else if (er != nullptr) {
    ASTRUM_FIELD(d, *er, *ar, aileron);
    ASTRUM_FIELD(d, *er, *ar, elevator);
    ASTRUM_FIELD(d, *er, *ar, rudder);
    ASTRUM_FIELD(d, *er, *ar, throttle);
  } else {
    const auto& ea = std::get<astrum::AutopilotTargets>(e.command.control);
    const auto& aa = std::get<astrum::AutopilotTargets>(a.command.control);
    ASTRUM_FIELD(d, ea, aa, heading_rad);
    ASTRUM_FIELD(d, ea, aa, alt_m);
    ASTRUM_FIELD(d, ea, aa, tas_mps);
  }
}

#undef ASTRUM_FIELD

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

std::vector<std::uint8_t> encode(const std::string& kind, const json& values) {
  if (kind == "telemetry") return astrum::encode_telemetry(telemetry_from(values.at(kind)));
  return astrum::encode_command(command_from(values.at(kind)));
}

// Decode `bytes` as `kind` and compare to values.json; returns mismatch lines.
std::vector<std::string> check(const std::string& kind, const json& values,
                               const std::vector<std::uint8_t>& bytes) {
  Diff diff;
  if (kind == "telemetry") {
    const auto decoded = astrum::decode_telemetry(bytes);
    if (!decoded) return {"decode failed: " + decoded.error};
    compare(diff, telemetry_from(values.at(kind)), *decoded.value);
  } else {
    const auto decoded = astrum::decode_command(bytes);
    if (!decoded) return {"decode failed: " + decoded.error};
    compare(diff, command_from(values.at(kind)), *decoded.value);
  }
  return diff.lines();
}

int write(const fs::path& dir) {
  const json values = load_values(dir);
  for (const std::string kind : kKinds) {
    const std::vector<std::uint8_t> bytes = encode(kind, values);
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
  for (const std::string kind : kKinds) {
    for (const std::string lang : kLangs) {
      const fs::path path = dir / (kind + "." + lang + ".bin");
      if (!fs::exists(path)) {
        std::cout << "skip " << path.filename().string() << " (not present)\n";
        continue;
      }
      ++checked;
      const std::vector<std::string> problems = check(kind, values, read_file(path));
      std::cout << (problems.empty() ? "ok   " : "FAIL ") << path.filename().string() << "\n";
      for (const std::string& p : problems) std::cout << "       " << p << "\n";
      if (!problems.empty()) ++failed;
    }
  }
  if (checked == 0) {
    std::cout << "no fixture files in " << dir.string() << "\n";
    return 1;
  }
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
