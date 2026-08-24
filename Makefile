# FORGE — Docker-based dev environment (M0)
#
# Everything runs inside the `forge-dev` image so nothing is installed on the host.
# Run `make` (or `make help`) for the target list.

COMPOSE ?= docker compose
SERVICE ?= sim

.DEFAULT_GOAL := help
.PHONY: help build up down demo demo-python demo-quad shell exec verify logs ps clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the FORGE dev image (compiles JSBSim from source)
	$(COMPOSE) build

up: ## Start sim + controller services (detached)
	$(COMPOSE) up -d

down: ## Stop and remove services
	$(COMPOSE) down

demo: ## Run the integration demo: C++ controller flies aircraft 1 in a bounded scenario, then auto-tears-down
	-FORGE_SCENARIO=config/scenario.demo.toml $(COMPOSE) up --exit-code-from sim sim controller
	$(COMPOSE) down

demo-python: ## Same demo, but the Python controller flies aircraft 1
	-FORGE_SCENARIO=config/scenario.demo.toml $(COMPOSE) --profile python up --exit-code-from sim sim controller-python
	$(COMPOSE) --profile python down

demo-quad: ## Quadrotor demo: an F450 hovers/climbs under the built-in multirotor autopilot
	-FORGE_SCENARIO=config/scenario.quad.toml $(COMPOSE) --profile quad up --exit-code-from sim sim controller-quad
	$(COMPOSE) --profile quad down

shell: ## Open a one-off shell in a fresh container (SERVICE=sim|controller)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash

exec: ## Exec a shell into an already-running container (SERVICE=sim|controller)
	$(COMPOSE) exec $(SERVICE) bash

verify: ## Verify the toolchain (versions + JSBSim link + functional run)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash scripts/verify-toolchain.sh

logs: ## Follow service logs
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

clean: ## Stop services and remove volumes + the local image
	$(COMPOSE) down -v --rmi local
