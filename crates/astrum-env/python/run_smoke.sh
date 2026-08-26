#!/usr/bin/env bash
# Build the astrum_env extension module and run the Python smoke test against it.
# Run inside the dev container:  bash crates/astrum-env/python/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root (workspace)

# 1. Build the cdylib. `extension-module` leaves libpython symbols undefined
#    (resolved by the interpreter at import), so this needs no python3-dev.
cargo build -p astrum-env --release

# 2. PyO3 emits `libastrum_env.so`; Python imports the module name `astrum_env`,
#    so stage a copy under that name on a temp import path.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp target/release/libastrum_env.so "$STAGE/astrum_env.so"

# 3. Run the smoke test with the staged module importable.
PYTHONPATH="$STAGE" python3 crates/astrum-env/python/smoke.py
