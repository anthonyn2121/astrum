// Minimal smoke test: proves the JSBSim headers are found and libJSBSim links.
//
// Compiled during `make verify` with:
//     g++ -std=c++17 docker/checks/jsbsim_smoke.cpp \
//         $(pkg-config --cflags --libs JSBSim) -o /tmp/jsbsim_smoke
//
// Constructing an FGFDMExec exercises the property-tree setup, which is enough
// to confirm the shared library is wired up correctly. Loading aircraft data is
// validated separately via the JSBSim CLI in scripts/verify-toolchain.sh.
#include <FGFDMExec.h>

#include <iostream>

int main() {
  JSBSim::FGFDMExec fdm;
  std::cout << "[jsbsim-smoke] libJSBSim linked and FGFDMExec constructed OK\n";
  return 0;
}
