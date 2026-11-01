# ASTRUM — Docker-based dev environment
#
# Everything runs inside the `astrum-dev` image so nothing is installed on the
# host. Containers run as the unprivileged image user, built with the host's
# UID/GID (exported below) so files written into the bind mount stay yours.
# Run `make` (or `make help`) for the target list. `make ci` is the full gate.

COMPOSE ?= docker compose
SERVICE ?= sim

UID ?= $(shell id -u)
GID ?= $(shell id -g)
export UID GID

.DEFAULT_GOAL := help
.PHONY: help build up down demo demo-python demo-quad demo-formation smoke replay export \
        shell exec verify fmt fmt-check lint test test-rust test-cpp test-python \
        wire-fixtures deny bench ci logs ps clean

# One-off command in a fresh container (no deps, no TTY): $(IN) '<bash script>'.
IN = $(COMPOSE) run --rm --no-deps -T $(SERVICE) bash -c

# Hand-written sources from git (see scripts/sources.sh), NUL-separated.
CPP_FILES = bash scripts/sources.sh cpp
PY_FILES = bash scripts/sources.sh py

# The C++ controllers' single CMake project (controllers/CMakeLists.txt).
CMAKE_CONTROLLERS = cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the ASTRUM dev image (compiles JSBSim from source)
	$(COMPOSE) build

up: ## Build the binaries, then start sim + replay + C++ controller (detached)
	$(COMPOSE) up -d

down: ## Stop and remove services
	$(COMPOSE) down

demo: ## Integration demo: C++ controller flies aircraft 1 in a bounded scenario; fails if the sim or controller fails
	bash scripts/demo.sh config/scenario.demo.toml - controller

demo-python: ## Same demo, but the Python controller flies aircraft 1
	bash scripts/demo.sh config/scenario.demo.toml python controller-python

demo-quad: ## Quadrotor demo: an F450 hovers/climbs under the built-in multirotor autopilot
	bash scripts/demo.sh config/scenario.quad.toml quad controller-quad

demo-formation: ## Formation demo: a leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
	bash scripts/demo.sh config/scenario.formation.toml formation \
	  controller-lead controller-wing-left controller-wing-right controller-trail

smoke: demo demo-python demo-quad demo-formation ## Run every bounded demo; fails on the first failing one

replay: ## Start the fleet sim + the 3-D replay renderer, then open http://localhost:8080
	$(COMPOSE) up -d sim replay
	@echo "ASTRUM replay renderer → http://localhost:8080  (Ctrl-C-safe; run 'make down' to stop)"

export: ## Bake a bag to a replay JSON the browser can open offline (BAG=path OUT=path)
	$(COMPOSE) run --rm --no-deps -T \
	  -e ASTRUM_BAG=$(or $(BAG),recordings/session.mcap) \
	  -e ASTRUM_WEB_EXPORT=$(or $(OUT),recordings/session.json) \
	  $(SERVICE) cargo run --release --locked -p astrum-web --bin astrum-web-export

shell: ## Open a one-off shell in a fresh container (SERVICE=sim|controller)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash

exec: ## Exec a shell into an already-running container (SERVICE=sim|controller)
	$(COMPOSE) exec --user astrum $(SERVICE) bash

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

ci: fmt-check lint test deny ## Full gate: fmt-check + lint + all tests + deny (fails on any failure)

logs: ## Follow service logs
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

clean: ## Stop services and remove volumes + the local image
	$(COMPOSE) down -v --rmi local
