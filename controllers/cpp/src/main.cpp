// astrum-controller: the example ASTRUM flight controller (C++), two laws on the shared
// runtime (controllers/README.md):
//   * hold (default): the raw hold law in hold_law.cpp flies a fixed-wing with raw
//     actuators — ASTRUM's intended usage, you own the control law;
//   * autopilot: sends the --heading / --alt / --speed reference every step (the command
//     is a lease) and the sim's built-in inner loop flies it (fixed-wing or multirotor).
// Run with --help for every flag and its ASTRUM_* environment fallback.

#include <memory>
#include <vector>

#include "astrum/runtime.hpp"
#include "hold_law.hpp"

namespace {

class AutopilotLaw : public astrum::Law {
 public:
  explicit AutopilotLaw(const astrum::Reference& reference) : reference_(reference) {}

  std::vector<astrum::VehicleKind> vehicle_kinds() const override {
    return {astrum::VehicleKind::FixedWing, astrum::VehicleKind::Multirotor};
  }

  astrum::LawResult step(const astrum::StepInput& /*input*/) override {
    return astrum::Action{
        astrum::AutopilotTargets{reference_.heading_rad, reference_.alt_hae_m, reference_.tas_mps}};
  }

 private:
  astrum::Reference reference_;
};

}  // namespace

int main(int argc, char** argv) {
  const astrum::LawEntry laws[] = {
      {"hold", "raw hold law: heading / altitude / airspeed with raw actuators (fixed-wing)",
       nullptr,
       [](const astrum::LawContext& ctx) {
         return std::make_unique<astrum::HoldLaw>(ctx.limits, ctx.reference);
       }},
      {"autopilot", "the reference as sim autopilot targets (fixed-wing, multirotor)", nullptr,
       [](const astrum::LawContext& ctx) { return std::make_unique<AutopilotLaw>(ctx.reference); }},
  };
  return astrum::run_main(argc, argv, laws, "astrum-controller",
                          "ASTRUM example flight controller (C++): a raw hold law or sim "
                          "autopilot targets for one aircraft over Zenoh.");
}
