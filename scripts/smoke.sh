#!/usr/bin/env bash
#
# Fly every demo run file (config/demos/*.toml) with `astrum run`, failing on
# the first failing one (`make smoke`; CI runs it). Run inside the dev container.
#
# Each demo flies in lockstep, as fast as possible: the sim waits every tick for
# a command from each aircraft's controller, so a controller that never
# connects or stops answering fails the run (lockstep timeout) instead of
# passing unnoticed, and the demos' minutes of real time take seconds. Every
# aircraft in a demo's base scenario must be flown by one of its controllers.
set -euo pipefail
cd "$(dirname "$0")/.."

for demo in config/demos/*.toml; do
    # `--set` arguments putting the sim and every aircraft of the base scenario
    # in lockstep.
    lockstep="$(
        python - "${demo}" <<'PY'
import sys
import tomllib

run = tomllib.load(open(sys.argv[1], "rb"))
scenario = tomllib.load(open(run["base"], "rb")) if "base" in run else run
sets = ["sim.lockstep=true"] + [f"aircraft.{a['id']}.lockstep=true" for a in scenario["aircraft"]]
print(" ".join(f"--set {s}" for s in sets))
PY
    )"
    read -ra lockstep <<<"${lockstep}"
    astrum run "${demo}" --speed 0 "${lockstep[@]}"
done
