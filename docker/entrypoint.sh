#!/usr/bin/env bash
# ASTRUM dev container entrypoint.
#
# All environment (JSBSIM_ROOT, PKG_CONFIG_PATH, LD_LIBRARY_PATH, PYTHONPATH and
# the cargo/venv/JSBSim dirs on PATH) is baked in via ENV in the Dockerfile.
#
# The container starts as root only so it can hand the named volumes (cargo
# registry, shared target dir) to the unprivileged `astrum` user — volumes
# created by an older root-running image, or by an image built for another UID,
# are re-owned once — and then drops privileges for the real command. Started
# as any other user (`--user`), it just execs the command.
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
    owner="$(id -u astrum):$(id -g astrum)"
    for dir in "${CARGO_HOME}/registry" /workspace/target; do
        if [[ -d "${dir}" && "$(stat -c %u:%g "${dir}")" != "${owner}" ]]; then
            chown -R "${owner}" "${dir}"
        fi
    done
    exec setpriv --reuid=astrum --regid=astrum --init-groups -- "$@"
fi

exec "$@"
