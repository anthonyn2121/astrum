#!/usr/bin/env bash
#
# Fly every demo run file (config/demos/*.toml), then every controller-catalog
# entry (config/controllers.toml), with `astrum run`, failing on the first
# failing run (`make smoke`; CI runs it). Run inside the dev container.
#
# Each run flies in lockstep, as fast as possible: the sim waits every tick for
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

# The catalog: each entry once with its defaults, for CATALOG_TIME_S of sim time,
# its command line rendered by `astrum controllers render` (as the web studio
# renders it). Fixed-wing laws fly scenario.demo.toml, multirotor-capable ones
# also scenario.quad.toml; py-formation follows a py-figure8 leader in
# scenario.formation.toml. Only the flown aircraft are in lockstep; the rest fly
# uncommanded, as in a web-launched run.
CATALOG_TIME_S=10
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# fly SCENARIO "AIRCRAFT ENTRY [RENDER-ARGS...]"...: one run with these
# assignments, from a temporary run file.
fly() {
    local scenario="$1" assignment line
    shift
    local run="${tmp}/run.toml"
    local -a words
    {
        echo "base = \"${scenario}\""
        echo "controllers = ["
        for assignment in "$@"; do
            read -ra words <<<"${assignment}"
            line="$(astrum controllers render "${words[1]}" --id "${words[0]}" "${words[@]:2}")"
            line="${line//\\/\\\\}"
            echo "  \"${line//\"/\\\"}\","
        done
        echo "]"
        echo "[set]"
        echo "\"sim.lockstep\" = true"
        echo "\"sim.max_time_s\" = ${CATALOG_TIME_S}"
        for assignment in "$@"; do
            echo "\"aircraft.${assignment%% *}.lockstep\" = true"
        done
    } >"${run}"
    echo "[smoke] ${scenario}: $(printf '[aircraft %s] ' "$@")"
    astrum run "${run}" --speed 0 </dev/null
}

entries="$(astrum controllers)"
while read -r entry kinds _; do
    if [[ "${entry}" == py-formation ]]; then
        continue
    fi
    if [[ ",${kinds}," == *,fixed-wing,* ]]; then
        fly config/scenario.demo.toml "1 ${entry}"
    fi
    if [[ ",${kinds}," == *,multirotor,* ]]; then
        fly config/scenario.quad.toml "1 ${entry}"
    fi
done <<<"${entries}"
fly config/scenario.formation.toml "1 py-figure8" "2 py-formation --fleet 1,2,3,4"
