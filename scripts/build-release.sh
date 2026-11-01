#!/usr/bin/env bash
#
# Compile everything the compose services run, once: the Rust release binaries
# (astrum-sim, astrum-bag-record/-play, astrum-web/-export) into the shared
# target volume and the C++ controllers into build/controllers. Run by the
# one-shot `build` compose service; every other service depends on it and execs
# the binaries directly. Run inside the dev container.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --locked -p astrum-sim -p astrum-bag -p astrum-web

cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j"$(nproc)" --target astrum-controller astrum-optimal-controller
