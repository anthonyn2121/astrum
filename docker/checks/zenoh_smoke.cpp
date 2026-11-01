// Minimal smoke test: proves the zenoh-cpp headers are found and libzenohc
// links and runs (config + key-expression construction go through the C lib).
//
// Compiled during `make verify` with:
//     g++ -std=c++17 -DZENOHCXX_ZENOHC docker/checks/zenoh_smoke.cpp \
//         -o /tmp/zenoh_smoke -lzenohc -pthread
#include <iostream>
#include <zenoh.hxx>

int main() {
  zenoh::Config config = zenoh::Config::create_default();
  zenoh::KeyExpr ke("astrum/telemetry/1");
  std::cout << "[zenoh-smoke] zenoh-cpp OK: key=" << ke.as_string_view() << "\n";
  return 0;
}
