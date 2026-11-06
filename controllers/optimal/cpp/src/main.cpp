// astrum-optimal-controller: the LQR / MPC laws (laws/) on the shared runtime
// (astrum/runtime.hpp, controllers/README.md). The runtime owns the flags, the Zenoh
// transport and the control loop; the laws design their gains for the telemetry's
// control period.

#include "astrum/runtime.hpp"
#include "laws.hpp"

int main(int argc, char** argv) {
  const astrum::LawEntry laws[] = {astrum::lqr_law_entry(), astrum::mpc_law_entry()};
  return astrum::run_main(argc, argv, laws, "astrum-optimal-controller",
                          "ASTRUM optimal-control controller (C++): an LQR or MPC law against "
                          "one aircraft over Zenoh.");
}
