#!/usr/bin/env bash
#
# Build the astrum_env Python extension module and stage it under its import
# name, then print the staging directory (put it on PYTHONPATH). Run inside the
# dev container:
#   PYTHONPATH="$(bash scripts/build-astrum-env.sh):crates/astrum-env/python" python ...
#
# The cdylib is built with PyO3's `extension-module` (libpython symbols are left
# undefined and resolved by the interpreter at import), so no python3-dev is
# needed. PyO3 emits `libastrum_env.so`; Python imports `astrum_env`, so the
# library is copied to `<stage>/astrum_env.so`. The stage lives in the cargo
# target dir (default target/python; override with ASTRUM_ENV_STAGE).
set -euo pipefail
cd "$(dirname "$0")/.."

stage="${ASTRUM_ENV_STAGE:-target/python}"
# Cargo's progress goes to stderr; stdout carries only the stage path.
cargo build --release --locked -p astrum-env 1>&2
mkdir -p "${stage}"
cp target/release/libastrum_env.so "${stage}/astrum_env.so"
realpath "${stage}"
