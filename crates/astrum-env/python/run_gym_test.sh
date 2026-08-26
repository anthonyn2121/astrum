#!/usr/bin/env bash
# Build the astrum_env extension module and run the astrum_dogfight adapter tests.
# Run inside the dev container:  bash crates/astrum-env/python/run_gym_test.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (workspace)

# 1. Build the cdylib (see run_smoke.sh — no python3-dev needed).
cargo build -p astrum-env --release

# 2. Stage the .so under its import name.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp target/release/libastrum_env.so "$STAGE/astrum_env.so"

# 3. Run the tests with both the staged module and the pure-Python package on path.
PYTHONPATH="$STAGE:crates/astrum-env/python" python3 crates/astrum-env/python/test_dogfight_env.py
