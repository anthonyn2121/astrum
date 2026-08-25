# ASTRUM — Docker-based dev environment
#
# Everything runs inside the `astrum-dev` image so nothing is installed on the host.
# Run `make` (or `make help`) for the target list.

COMPOSE ?= docker compose
SERVICE ?= sim

.DEFAULT_GOAL := help
.PHONY: help build up down demo demo-python demo-quad demo-formation replay export shell exec verify fmt lint logs ps clean

# All hand-written C++ controller sources (excludes generated capnp + build dirs).
CPP_SRC = find controllers -type f \( -name '*.cpp' -o -name '*.hpp' \) -not -path '*/build/*'

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the ASTRUM dev image (compiles JSBSim from source)
	$(COMPOSE) build

up: ## Start sim + controller services (detached)
	$(COMPOSE) up -d

down: ## Stop and remove services
	$(COMPOSE) down

demo: ## Run the integration demo: C++ controller flies aircraft 1 in a bounded scenario, then auto-tears-down
	-ASTRUM_SCENARIO=config/scenario.demo.toml $(COMPOSE) up --exit-code-from sim sim controller
	$(COMPOSE) down

demo-python: ## Same demo, but the Python controller flies aircraft 1
	-ASTRUM_SCENARIO=config/scenario.demo.toml $(COMPOSE) --profile python up --exit-code-from sim sim controller-python
	$(COMPOSE) --profile python down

demo-quad: ## Quadrotor demo: an F450 hovers/climbs under the built-in multirotor autopilot
	-ASTRUM_SCENARIO=config/scenario.quad.toml $(COMPOSE) --profile quad up --exit-code-from sim sim controller-quad
	$(COMPOSE) --profile quad down

demo-formation: ## Formation demo: a leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
	-ASTRUM_SCENARIO=config/scenario.formation.toml $(COMPOSE) --profile formation up --exit-code-from sim \
	  sim controller-lead controller-wing-left controller-wing-right controller-trail
	$(COMPOSE) --profile formation down

replay: ## Start the fleet sim + the 3-D replay renderer, then open http://localhost:8080
	$(COMPOSE) up -d sim replay
	@echo "ASTRUM replay renderer → http://localhost:8080  (Ctrl-C-safe; run 'make down' to stop)"

export: ## Bake a bag to a replay JSON the browser can open offline (BAG=path OUT=path)
	$(COMPOSE) run --rm --no-deps \
	  -e ASTRUM_BAG=$(or $(BAG),recordings/session.bag) \
	  -e ASTRUM_WEB_EXPORT=$(or $(OUT),recordings/session.json) \
	  sim bash -lc 'cargo run --release -p astrum-web --bin astrum-web-export'

shell: ## Open a one-off shell in a fresh container (SERVICE=sim|controller)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash

exec: ## Exec a shell into an already-running container (SERVICE=sim|controller)
	$(COMPOSE) exec $(SERVICE) bash

verify: ## Verify the toolchain (versions + JSBSim link + functional run)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash scripts/verify-toolchain.sh

# fmt/lint run as the host user so in-place rewrites keep file ownership (not
# root), and with ruff's cache off so no .ruff_cache lands on the bind mount.
RUN_USER = $(COMPOSE) run --rm --no-deps --user $$(id -u):$$(id -g) $(SERVICE)

fmt: ## Format the controllers in place (clang-format for C++, ruff for Python)
	$(RUN_USER) bash -lc 'clang-format -i $$($(CPP_SRC)) && ruff format controllers && ruff check --no-cache --fix controllers'

lint: ## Lint Python (ruff) and check C++ formatting (clang-format --dry-run)
	$(RUN_USER) bash -lc 'ruff check --no-cache controllers && ruff format --check controllers && clang-format --dry-run --Werror $$($(CPP_SRC))'

logs: ## Follow service logs
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

clean: ## Stop services and remove volumes + the local image
	$(COMPOSE) down -v --rmi local
