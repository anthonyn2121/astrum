# ASTRUM — Docker-based dev environment
#
# Everything runs inside the `astrum-dev` image so nothing is installed on the host.
# Run `make` (or `make help`) for the target list.

COMPOSE ?= docker compose
SERVICE ?= sim
# Extra args forwarded to the RL trainer, e.g. `make rl-train ARGS="--tier 2"`.
ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help build up down demo demo-python demo-quad demo-formation replay export shell exec verify fmt lint logs ps clean rl-env rl-smoke rl-train rl-test rl-bench rl-tdmpc2-smoke rl-tdmpc2-train rl-tdmpc2-parallel-smoke rl-tdmpc2-selfplay-smoke rl-tdmpc2-train-parallel

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

# --- RL dogfight training (controllers/rl; separate uv env, GPU) ----------------
# The dogfight env is embedded (astrum_env runs its own headless sim in-process),
# so training needs neither the `sim` service nor Zenoh — hence `--no-deps`.
rl-env: ## Build the astrum_env pyo3 .so and stage it into controllers/rl/vendor
# Stage with a same-directory `mv` (an atomic rename), NEVER a bare `cp`. Every
# running collector and diagnostic has this .so mmap'd; `cp` opens the existing
# inode O_TRUNC and rewrites its pages in place, so the mapped text segment
# changes under those processes and they take SIGBUS/SIGSEGV — silently, with no
# Python traceback, ALL of them at once. That is how a `make rl-*` launch killed
# a running 5M trainer's 6 collectors and a 6-worker search mid-flight on
# 2026-09-06 (.so rewritten 05:23:51, both dead within the minute). `mv` replaces
# the *name* with a new inode, so existing mappings keep the old one alive.
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cargo build -p astrum-env --release \
	   && mkdir -p controllers/rl/vendor \
	   && cp target/release/libastrum_env.so controllers/rl/vendor/.astrum_env.so.new \
	   && mv -f controllers/rl/vendor/.astrum_env.so.new controllers/rl/vendor/astrum_env.so'

rl-smoke: rl-env ## Smoke-test the RL env on the GPU (torch + CUDA + a Tier-1 rollout)
	$(COMPOSE) --profile rl run --rm --no-deps dogfight-train bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.smoke_gpu'

rl-train: rl-env ## Train the dogfight learner (from-scratch PPO, GPU); ARGS="--tier 2 ..."
	$(COMPOSE) --profile rl run --rm --no-deps dogfight-train bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.train $(ARGS)'

# --- TD-MPC2 (the primary from-scratch learner; PPO above is the sanity baseline) ---
rl-test: ## Run the RL unit suites (env wrappers + TD-MPC2 model/buffer/agent; CPU)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cd controllers/rl && uv run python -m pytest -q $(ARGS)'

# The dogfight env's own Python suites live next to the package (crates/astrum-env/
# python) rather than under controllers/rl, so they need the vendored .so plus that
# directory on PYTHONPATH. Run with `make env-test` after `make rl-env`.
env-test: ## Run the dogfight ENV suites (gun-lay math + env contract; CPU)
	$(COMPOSE) run --rm --no-deps --user $$(id -u):$$(id -g) \
	  -e UV_CACHE_DIR=/tmp/uvcache -e HOME=/tmp $(SERVICE) bash -lc \
	  'cd controllers/rl \
	   && PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
	      uv run --no-sync python -m pytest -q -p no:cacheprovider \
	      /workspace/crates/astrum-env/python/test_gunlay.py \
	      /workspace/crates/astrum-env/python/test_dogfight_env.py \
	      /workspace/crates/astrum-env/python/test_bfm_detect.py $(ARGS)'

# The container's JSBSim data root is a third-party asset OUTSIDE the repo, so editing an
# aircraft in place is invisible to git and would silently void every prior measurement (the
# same trap the T3 opponent fix sprang). This copies the root to a gitignored workspace path
# and adds named f16 VARIANTS there; select it with JSBSIM_ROOT=/workspace/.jsbsim-variants.
#   f16nopid  the <trigger> dropped from fcs/g-load-pid (does it gate the ki=0.025 integrator?)
#   f16kp10   kp 0.3 -> 1.0 on the same loop
# Both isolate defect D-C in docs/bfm-program-handoff.md. The copied root also carries an
# UNMODIFIED `f16`, which is the control arm: run all three from this root so the only
# difference between arms is the XML edit and not the data root itself.
jsbsim-variants: ## Copy the JSBSim root to .jsbsim-variants/ and add the f16 FCS variants (A3/D-C)
	$(COMPOSE) run --rm --no-deps --user $$(id -u):$$(id -g) -e HOME=/tmp $(SERVICE) bash -lc \
	  'set -e; R=/workspace/.jsbsim-variants; rm -rf "$$R"; mkdir -p "$$R"; \
	   cp -r /opt/jsbsim/share/jsbsim/. "$$R"/; \
	   for v in f16nopid f16kp10; do \
	     rm -rf "$$R/aircraft/$$v"; cp -r "$$R/aircraft/f16" "$$R/aircraft/$$v"; \
	     mv "$$R/aircraft/$$v/f16.xml" "$$R/aircraft/$$v/$$v.xml"; done; \
	   python3 /workspace/crates/astrum-env/python/_patch_jsbsim_variants.py'

rl-bench: rl-env ## Benchmark env throughput with a random policy (CPU); ARGS="--tier 2"
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.benchmark $(ARGS)'

rl-tdmpc2-smoke: rl-env ## End-to-end TD-MPC2 smoke on the real env (tiny model, CPU)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.tdmpc2.smoke'

rl-tdmpc2-train: rl-env ## Train the dogfight learner with TD-MPC2 (GPU); ARGS="--tier 2 ..."
	$(COMPOSE) --profile rl run --rm --no-deps dogfight-train bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.tdmpc2.train $(ARGS)'

rl-tdmpc2-parallel-smoke: rl-env ## Smoke the async multi-collector trainer (2 workers, tiny model, CPU)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.tdmpc2.smoke_parallel'

rl-tdmpc2-selfplay-smoke: rl-env ## Smoke the SELF-PLAY pipeline (pool seed → per-episode opponent swap → snapshot growth, CPU)
	$(COMPOSE) run --rm --no-deps $(SERVICE) bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.tdmpc2.smoke_parallel --selfplay'

rl-tdmpc2-train-parallel: rl-env ## TD-MPC2 with K parallel 1v1 collectors + a central GPU learner (§8 scale-out); ARGS="--workers 6 ..."
	$(COMPOSE) --profile rl run --rm --no-deps dogfight-train bash -lc \
	  'cd controllers/rl && uv run python -m astrum_rl.tdmpc2.train_parallel $(ARGS)'

logs: ## Follow service logs
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

clean: ## Stop services and remove volumes + the local image
	$(COMPOSE) down -v --rmi local
