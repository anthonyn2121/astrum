#!/usr/bin/env bash
#
# Compile everything `astrum` runs: the Rust release binaries (astrum,
# astrum-sim, astrum-ctl, astrum-bag-record/-play, astrum-web/-export) into the
# shared target volume and the C++ controllers into build/controllers. Run by
# the `astrum` wrapper (scripts/bin/astrum) before every command — a quick no-op
# when nothing changed. Run inside the dev container.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --locked -p astrum-sim -p astrum-net -p astrum-bag -p astrum-web

cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j"$(nproc)" --target astrum-controller astrum-optimal-controller
