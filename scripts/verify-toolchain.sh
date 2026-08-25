#!/usr/bin/env bash
#
# Verify the ASTRUM toolchain image end-to-end. Run inside the dev container via
# `make verify`. Hard checks (compilers, pkg-config libs, python bindings, and a
# C++ program that links libJSBSim) abort on failure; the JSBSim CLI functional
# check is best-effort and only prints its output.
set -euo pipefail

hr() { printf '%s\n' "============================================================"; }

hr
echo " ASTRUM toolchain verification"
hr

echo "-- compilers & build tools --"
rustc --version
cargo --version
g++ --version | head -1
cmake --version | head -1
capnp --version
python --version

echo
echo "-- libraries via pkg-config --"
echo "libzmq : $(pkg-config --modversion libzmq)"
echo "capnp  : $(pkg-config --modversion capnp)"
echo "JSBSim : $(pkg-config --modversion JSBSim)"

echo
echo "-- python bindings --"
python - <<'PY'
import zmq, capnp  # noqa: F401
print("pyzmq -> libzmq", ".".join(map(str, zmq.zmq_version_info())))
print("pycapnp import OK")
PY

echo
echo "-- C++ links libJSBSim (compile + run smoke) --"
g++ -std=c++17 docker/checks/jsbsim_smoke.cpp $(pkg-config --cflags --libs JSBSim) -o /tmp/jsbsim_smoke
/tmp/jsbsim_smoke

echo
echo "-- JSBSim CLI + aircraft data (best effort) --"
JSBSim --version 2>&1 | head -1 || true
echo "(loading c172x catalog from JSBSIM_ROOT=${JSBSIM_ROOT})"
JSBSim --root="${JSBSIM_ROOT}" --aircraft=c172x --catalog 2>&1 | head -15 || \
    echo "(JSBSim catalog check returned non-zero — inspect output above)"

echo
hr
echo " ALL HARD CHECKS PASSED"
hr
