// The smallest ASTRUM controller: hold the --heading / --alt / --speed reference with the
// sim's autopilot (controllers/README.md, "Writing a controller").

#include <memory>
#include <vector>

#include "astrum/runtime.hpp"

class Hold : public astrum::Law {
 public:
  explicit Hold(const astrum::Reference& ref) : ref_(ref) {}

  std::vector<astrum::VehicleKind> vehicle_kinds() const override {
    return {astrum::VehicleKind::FixedWing, astrum::VehicleKind::Multirotor};
  }

  astrum::LawResult step(const astrum::StepInput& /*in*/) override {
    return astrum::Action{astrum::AutopilotTargets{ref_.heading_rad, ref_.alt_hae_m, ref_.tas_mps}};
  }

 private:
  astrum::Reference ref_;
};

int main(int argc, char** argv) {
  const astrum::LawEntry laws[] = {
      {"hold", "hold the reference with the sim autopilot", nullptr,
       [](const astrum::LawContext& ctx) { return std::make_unique<Hold>(ctx.reference); }}};
  return astrum::run_main(argc, argv, laws, "hold-controller",
                          "Out-of-tree example: the reference as sim autopilot targets.");
}
