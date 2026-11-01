#!/usr/bin/env bash
#
# Regenerate the cross-language wire fixtures in crates/astrum-proto/tests/fixtures:
# each language encodes the canonical values (values.json) into <kind>.<lang>.bin.
# Telemetry/commands (kind ∈ {telemetry, command_raw, command_autopilot}) are
# written by Rust, C++ and Python; events (kind ∈ event_*) by Rust and Python only
# (C++ has no Event codec). The Rust test crates/astrum-proto/tests/fixtures.rs and
# the Python tests decode every file (the C++ `wire_fixtures` ctest the nine
# telemetry/command files) and compare them to values.json.
# Run inside the dev container (`make wire-fixtures`).
set -euo pipefail
cd "$(dirname "$0")/.."

fixtures=crates/astrum-proto/tests/fixtures

echo "-- rust --"
cargo run --locked -p astrum-proto --example wire_fixture

echo "-- c++ --"
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j"$(nproc)" --target astrum-wire-fixture
build/controllers/common/cpp/astrum-wire-fixture write "${fixtures}"

echo "-- python --"
python -m astrum_client.wire_fixture --dir "${fixtures}"
