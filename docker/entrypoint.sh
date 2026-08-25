#!/usr/bin/env bash
# ASTRUM dev container entrypoint.
#
# All environment (JSBSIM_ROOT, PKG_CONFIG_PATH, LD_LIBRARY_PATH, and the
# cargo/venv/JSBSim dirs on PATH) is baked in via ENV in the Dockerfile, so this
# is intentionally thin: it just execs whatever command the container was given.
set -euo pipefail
exec "$@"
