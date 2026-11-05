# ASTRUM — Docker-based dev environment
#
# Everything runs inside the `astrum-dev` image so nothing is installed on the
# host. `make dev` builds the image (if missing or out of date), starts the
# long-lived `dev` container and opens a shell in it; inside, `astrum …` runs
# scenarios, run files and sweeps. Containers run as the unprivileged image
# user, built with the host's UID/GID (exported below) so files written into
# the bind mount stay yours. Run `make` (or `make help`) for the target list.
# `make ci` is the full gate.
#
# The task targets (verify … ci) work on the host — each in a fresh container —
# and inside the dev container (the image sets ASTRUM_DEV=1), where they run in
# place. The container targets (dev, build, down, ps, clean) are host-only.

COMPOSE ?= docker compose

UID ?= $(shell id -u)
GID ?= $(shell id -g)
export UID GID

.DEFAULT_GOAL := help
.PHONY: help dev build down ps clean verify fmt fmt-check lint test test-rust test-cpp \
        test-python wire-fixtures deny bench smoke ci

# Run a bash script: in place inside the dev container, else in a fresh one
# (no TTY): $(IN) '<bash script>'.
ifeq ($(ASTRUM_DEV),1)
IN = bash -c
else
IN = $(COMPOSE) run --rm -T dev bash -c
endif

# First line of every container target's recipe: they drive docker, which the
# dev container doesn't have.
HOST_ONLY = $(if $(filter 1,$(ASTRUM_DEV)),$(error make $@ runs on the host, not inside the dev container))

# Hand-written sources from git (see scripts/sources.sh), NUL-separated.
CPP_FILES = bash scripts/sources.sh cpp
PY_FILES = bash scripts/sources.sh py

# The C++ controllers' single CMake project (controllers/CMakeLists.txt).
CMAKE_CONTROLLERS = cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

dev: ## Build the image if needed, start the dev container, open a shell in it (again: another shell)
	$(HOST_ONLY)
	$(COMPOSE) up -d --build dev
	$(COMPOSE) exec --user astrum dev bash

build: ## Rebuild the ASTRUM dev image (compiles JSBSim from source)
	$(HOST_ONLY)
	$(COMPOSE) build

down: ## Stop and remove the dev container (and every shell in it)
	$(HOST_ONLY)
	$(COMPOSE) down

ps: ## Show whether the dev container is running
	$(HOST_ONLY)
	$(COMPOSE) ps

verify: ## Verify the toolchain (pinned versions agree, JSBSim/zenoh link + run)
	$(IN) 'bash scripts/verify-toolchain.sh'

fmt: ## Format everything in place (rustfmt, clang-format, ruff format + fixes)
	$(IN) 'set -euo pipefail; \
	  cargo fmt --all; \
	  $(CPP_FILES) | xargs -0 -r clang-format -i; \
	  $(PY_FILES) | xargs -0 -r ruff format --no-cache; \
	  $(PY_FILES) | xargs -0 -r ruff check --no-cache --fix'

fmt-check: ## Check formatting without rewriting (rustfmt, clang-format, ruff format)
	$(IN) 'set -euo pipefail; \
	  cargo fmt --all --check; \
	  $(CPP_FILES) | xargs -0 -r clang-format --dry-run --Werror; \
	  $(PY_FILES) | xargs -0 -r ruff format --no-cache --check'

lint: ## Lint Rust (clippy -D warnings, all targets) and Python (ruff check)
	$(IN) 'set -euo pipefail; \
	  cargo clippy --workspace --all-targets --locked -- -D warnings; \
	  $(PY_FILES) | xargs -0 -r ruff check --no-cache'

test: test-rust test-cpp test-python ## Run every test suite

test-rust: ## Rust tests (cargo test --workspace)
	$(IN) 'cargo test --workspace --locked'

test-cpp: ## C++ controller tests (CMake + ctest)
	$(IN) 'set -euo pipefail; \
	  $(CMAKE_CONTROLLERS); \
	  cmake --build build/controllers -j"$$(nproc)"; \
	  ctest --test-dir build/controllers --output-on-failure'

test-python: ## Python tests: controllers + astrum-env adapter (builds the astrum_env module)
	$(IN) 'set -euo pipefail; \
	  stage="$$(bash scripts/build-astrum-env.sh)"; \
	  export PYTHONPATH="$$stage:crates/astrum-env/python:$$PYTHONPATH"; \
	  python -m pytest -q controllers/python controllers/optimal/python crates/astrum-env/python; \
	  python crates/astrum-env/python/smoke.py'

wire-fixtures: ## Regenerate the Rust/C++/Python wire fixtures (crates/astrum-proto/tests/fixtures)
	$(IN) 'bash scripts/wire-fixtures.sh'

deny: ## Dependency audit: licenses, advisories, bans, sources (cargo deny)
	$(IN) 'cargo deny --locked check'

bench: ## Headless sim throughput (µs/tick, × real time) for fixed fleets and thread counts
	$(IN) 'cargo run --release --locked -q -p astrum-sim --example throughput 2>/dev/null | grep "^|"'

smoke: ## Fly every demo (config/demos/*.toml) with `astrum run`; fails on the first failing one
	$(IN) 'set -euo pipefail; for demo in config/demos/*.toml; do astrum run "$$demo"; done'

ci: fmt-check lint test deny ## Full gate: fmt-check + lint + all tests + deny (fails on any failure)

clean: ## Remove the dev container, the volumes and the local image
	$(HOST_ONLY)
	$(COMPOSE) down -v --rmi local
