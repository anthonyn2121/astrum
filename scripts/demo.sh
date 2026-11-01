#!/usr/bin/env bash
#
# Run one bounded compose demo and report whether it passed, always tearing the
# stack down. Run on the host (needs docker compose):
#   scripts/demo.sh <scenario.toml> <profile|-> <controller-service>...
#
# The sim decides the run: it exits when the scenario's max_time_s elapses and
# `--exit-code-from sim` makes its exit code the stack's. A controller that
# fails on its own (refuses the vehicle kind, loses the sim, crashes) aborts
# the stack early; that is caught by checking every controller's exit code —
# 0, or 143 (SIGTERM from the teardown after the sim exited) — so a crashed
# controller fails the demo even if the sim then shuts down cleanly.
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <scenario.toml> <profile|-> <controller-service>..." >&2
    exit 2
fi
scenario="$1"
profile="$2"
shift 2
controllers=("$@")

compose=(${COMPOSE:-docker compose})
if [[ "${profile}" != "-" ]]; then
    compose+=(--profile "${profile}")
fi

teardown() { "${compose[@]}" down --remove-orphans >/dev/null 2>&1 || true; }
trap teardown EXIT

ASTRUM_SCENARIO="${scenario}" "${compose[@]}" up --exit-code-from sim sim "${controllers[@]}"
rc=$?

for svc in "${controllers[@]}"; do
    code="$("${compose[@]}" ps --all --format '{{.ExitCode}}' "${svc}" 2>/dev/null)"
    case "${code}" in
        0 | 143) ;;
        *)
            echo "demo: controller '${svc}' exited with code '${code:-?}'" >&2
            [[ ${rc} -ne 0 ]] || rc=1
            ;;
    esac
done

if [[ ${rc} -ne 0 ]]; then
    echo "demo: FAILED (${scenario}, exit ${rc})" >&2
else
    echo "demo: passed (${scenario})"
fi
exit "${rc}"
