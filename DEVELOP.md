# Developing on ASTRUM

A practical guide for getting productive in this repo. For the *why* behind the
design, see [README.md](README.md); this doc is the *how*.

## 1. The mental model (read this first)

ASTRUM is **one server + many clients** talking over Zenoh.

- **The sim** (`astrum-sim`) is the authoritative state server. It's a headless
  [Bevy](https://bevyengine.org/) ECS: one *entity* per aircraft, each owning its
  own [JSBSim](https://github.com/JSBSim-Team/jsbsim) flight-dynamics model. Every
  tick it drains the commands it received (`astrum/command/<id>`), steps physics, and
  **publishes** telemetry (`astrum/telemetry/<id>`). It also answers
  `get astrum/scenario` with its scenario TOML.
- **Controllers** (`controllers/*`) are separate processes — the flight-control
  logic. They **subscribe** to telemetry and **publish** commands. They never touch
  the sim's memory; only bytes over Zenoh.
- **The wire** is [Cap'n Proto](https://capnproto.org/), one schema (v2,
  `crates/astrum-proto/schema/astrum.capnp`) shared by Rust, C++, and Python.

```
controllers ──put astrum/command/<id>─────►  sim (subscribes astrum/command/*)
controllers ◄──put astrum/telemetry/<id>───  sim (publishes per aircraft; id in the key)
```

Both key spaces ride **one** Zenoh session on a single endpoint (`tcp/…:7447`). The sim (and
`astrum-bag-play`) **listens as a Zenoh router**; every other process (recorder, web bridge, C++
and Python controllers) opens a `mode=client` session that connects to it. The router routes
between its clients, so the recorder sees the commands a controller sends (Zenoh 1.10 routers do
not forward between two *peers*, which is why clients are not peers). A client's open waits for a
sim that is still starting, up to its connect timeout, then fails with a clear message:
controllers exit 1 after `--startup-timeout`; `astrum-bag-record` and `astrum-web` take
`--connect-timeout` (`ASTRUM_CONNECT_TIMEOUT_S`, default 60 s).

Each entity is addressed by its id *in the key* — not a socket or a port — so **one endpoint serves
the whole fleet no matter how many aircraft you add**, and any client can subscribe to a single
aircraft (`astrum/telemetry/<id>`) or the entire fleet (`astrum/telemetry/*`). The command key id
must equal `Command.targetId`; the sim refuses mismatches.

Rules every client follows (the shared C++ and Python clients implement them for you):

- QoS for both streams: CongestionControl `Drop`, express, priority `InteractiveHigh`, declared
  publishers.
- Telemetry subscriptions are **latest-value** (one slot per aircraft), never a FIFO.
- **One command per telemetry sample**, carrying `seq` (strictly increasing from 1 per process)
  and `telemetryTick` (the `Telemetry.tick` it answers). Raw commands are a **lease**: stop sending
  for `sim.command_lease_s` and the sim's failsafe hold takes over (`controlMode = failsafe`).
- A raw control law checks `vehicleKind` and refuses (exits non-zero) any kind it wasn't written
  for, including unknown values.
- Decoding is untrusted input: a bad payload is logged, counted and skipped.

## 2. The golden rule: everything runs in Docker

Nothing (Rust, JSBSim, Zenoh, Cap'n Proto) is installed on the host — it all lives
in the `astrum-dev` image. You run commands *inside* a container, as the unprivileged
`astrum` user built with your UID/GID (the Makefile exports them), so files you create in
the bind mount stay yours.

```bash
make build      # build the image once (compiles JSBSim from source — slow first time)
make verify     # sanity-check the toolchain (pinned versions agree, JSBSim/zenoh link + run)
make shell      # open a shell in a fresh container (workspace mounted, on the network)
make help       # list all make targets
```

Your source is bind-mounted at `/workspace`, so **edit files on the host in your
normal editor**, and build/run/test them in a container. One-off commands:

```bash
docker compose run --rm --no-deps sim bash -lc '<command>'
```

Published ports (7447 Zenoh, 8080 web) bind to **127.0.0.1** on the host only.

## Quick start

The fast path from a fresh checkout to a running sim. Every command below runs
**inside the dev container** (§2). Do the one-time image build, then open a container
shell and run steps 1–3 inside it:

```bash
make build      # one time — builds the astrum-dev image (compiles JSBSim; slow first run)
make shell      # open a shell in the container; run the steps below there
```

**1. Build the Rust code** (the sim + every crate in the workspace):

```bash
cargo build --workspace              # debug build of every crate
cargo build --release --workspace    # optimized build (what the demos run)
```

**2. Build the C++ code** (all controllers are one CMake project):

```bash
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release   # configure
cmake --build build/controllers -j                                     # compile
# → build/controllers/cpp/astrum-controller
#   build/controllers/optimal/cpp/astrum-optimal-controller
#   build/controllers/common/cpp/astrum-wire-fixture
```

**3. Run a scenario from a config** (a `config/*.toml` fleet definition):

```bash
cargo run -p astrum-sim                                                    # default: config/scenario.example.toml
cargo run --release -p astrum-sim -- config/scenario.demo.toml             # pick a specific scenario (positional arg)
cargo run -p astrum-sim -- --help                                          # list every flag + its env fallback
```

Every binary honours an `ASTRUM_*` env var per flag as a fallback (so `docker compose` and
`make` work unchanged). Precedence is **CLI arg > `$ENV` > default** — see `--help`.

The sim loads the fleet from the TOML, spins up (and, for fixed-wing, trims) one JSBSim
model per aircraft, then publishes telemetry over Zenoh until `max_time_s` is reached (or
you Ctrl-C). Any startup failure — bad scenario, model load, trim, network — exits
non-zero with a message. Scenario format + the bundled examples are in §5.

To see everything working together — sim + controller + the 3-D browser view —
the demos do it in one command: `make demo` and `make replay` (§7).

## 3. Repo layout — where files go

```
crates/            # Rust workspace (one crate per responsibility)
  astrum-fdm/       #   JSBSim FFI: safe `Fdm` wrapper (C shim, crate-owned lock, `Fdm: Send`)
  astrum-geo/       #   WGS84 geodetic ↔ ECEF ↔ ENU math, local-level ↔ grid rotations
  astrum-proto/     #   the Cap'n Proto schema v2 + encode/decode + wire fixtures
  astrum-net/       #   Zenoh transport (keys, QoS, LatestTelemetry, CommandSender, scenario)
  astrum-bag/       #   MCAP record (telemetry + commands) / replay (telemetry)
  astrum-web/       #   axum bridge + tile proxy (astrum-web) + MCAP→JSON exporter (astrum-web-export)
  astrum-sim/       #   the Bevy ECS app — the sim binary + the embedded env seam (env.rs)
  astrum-env/       #   PyO3 module `astrum_env` + the dogfight Gym adapter (python/)
controllers/       # flight controllers (clients); controllers/CMakeLists.txt is the one C++ project
  common/cpp/       #   shared C++ wire codec + Zenoh client + wire-fixture tool
  cpp/  python/     #   example controllers (python/astrum_client is the shared Python client)
  optimal/          #   runtime + law split (cpp/, python/astrum_optimal)
web/               # browser 3-D replay renderer (Three.js, vendored; served by astrum-web)
config/            # scenario files (*.toml) — the fleet + world origin
docker/            # Dockerfile, entrypoint, pinned Python requirements
scripts/           # build-release, demo, verify-toolchain, wire-fixtures, build-astrum-env, sources
docs/              # architecture diagram, dogfight env design + onboarding
docker-compose.yml # service presets: build, sim, replay, recorder, controllers
Makefile           # the commands you'll actually type
```

**Adding things:**
| I want to add… | Put it in… | Then… |
|---|---|---|
| A new Rust library crate | `crates/<name>/` | add it to `members` in the root `Cargo.toml` |
| A new controller | `controllers/<name>/` | C++: `add_subdirectory` in `controllers/CMakeLists.txt`; (optional) a compose service for demos |
| A new scenario | `config/<name>.toml` | run with `cargo run -p astrum-sim -- config/<name>.toml` |
| A scenario field | `crates/astrum-sim/src/scenario.rs` | see §5 |
| A schema field | `crates/astrum-proto/schema/astrum.capnp` | see §8 |
| A test | `crates/<crate>/tests/<name>.rs` | see §6 |

## 4. Build & run the sim

Inside a shell (`make shell`) or via `docker compose run`:

```bash
cargo build --workspace                                          # compile everything
cargo run -p astrum-sim                                          # run the sim (uses config/scenario.example.toml)
cargo run --release -p astrum-sim -- config/scenario.demo.toml   # pick a scenario (positional)
cargo run -p astrum-sim -- --listen tcp/0.0.0.0:7447 --threads 8 config/scenario.demo.toml
```

The sim's flags (run `astrum-sim --help` to see them, each with its `[env: …]`
fallback). Every flag falls back to the env var, then the default:
| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| *(positional)* `scenario` | `ASTRUM_SCENARIO` | `config/scenario.example.toml` | which scenario to load |
| `--listen` | `ASTRUM_ZENOH_LISTEN` | `tcp/0.0.0.0:7447` | Zenoh endpoint the sim's router listens on |
| `--threads` | `ASTRUM_THREADS` | scenario `sim.threads` | FDM stepping threads (0 = all cores) |

## 5. Scenarios

A scenario is a TOML file (v2): a world origin, sim settings and a fleet. Unknown keys are
errors; units are SI, heights are above the WGS84 ellipsoid (HAE), airspeeds are true (TAS).
Copy an existing one and edit:

```toml
[world]                      # ENU grid origin (WGS84). Keep it near your fleet.
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_hae_m = 0.0

[sim]
rate_hz    = 120.0           # fixed physics step = 1/rate_hz seconds
max_time_s = 40.0            # 0 = run forever; >0 = stop after N sim-seconds
realtime   = true            # true ≈ wall-clock (for live control); false = as fast as possible
seed       = 0               # gun dispersion + each aircraft's JSBSim RNG
threads    = 1               # FDM stepping threads (0 = all cores); identical results for any value
command_lease_s = 0.5        # raw-command lease → failsafe hold; 0 disables
lockstep   = false           # wait each tick for commands from lockstep aircraft
lockstep_timeout_s = 30.0    # fatal if a lockstep wait exceeds this

[[aircraft]]                 # one entity per block; `id` must be unique
id = 1
model = "c172x"              # known: c172x (trainer), f16 (fighter), F450 (multirotor)
# kind = "multirotor"        # required (with autopilot_profile for fixed-wing) for other models
lat_deg = 37.6188
lon_deg = -122.3756
alt_hae_m = 1524.0
heading_deg = 90.0
tas_mps = 55.0               # a multirotor initialises at hover: tas_mps = 0.0
# trim = true                # default: fixed-wing yes, multirotor no; a failed trim is fatal
# lockstep = false
# team = 0 / health = 100.0 / ammo = 511
```

Bundled examples: `scenario.example.toml` (2-ship fleet), `scenario.demo.toml`
(bounded single fixed-wing for the demo), `scenario.quad.toml` (single F450 quad),
`scenario.formation.toml` (4-ship: a figure-eight leader + 3 MPC followers),
`scenario.smoke.toml` (fast, for tests).

**Vehicle classes.** Each aircraft is a fixed-wing or a multirotor (`VehicleKind`),
which decides its control law — the four `fcs/*-cmd-norm` inputs are surface
(f16: FLCS) commands on a fixed-wing but body-rate + collective demands on a multirotor.
`kind` comes from the known-model table; set `kind` (and `autopilot_profile` for a
fixed-wing) for any other model. A fleet can mix classes freely. A multirotor also needs its
stability-augmentation loop engaged (`fcs/ScasEngage = 1`); spawn does this automatically.

**Lockstep.** For reproducible runs with external controllers set `sim.lockstep = true` and
`lockstep = true` on the aircraft they fly: the sim waits each tick for a command answering the
last published tick (republishing that telemetry every 100 ms while waiting). Without lockstep a
command lands on whichever tick it arrives.

**Adding a scenario field.** Add it to `Sim` / `World` / `AircraftSpec` in
`crates/astrum-sim/src/scenario.rs` with a serde default (or `Option`) so existing scenarios still
load — the structs are `deny_unknown_fields`, so a typo in a file is an error, not a silent
default. Check it in `Scenario::validate`, document it in `config/scenario.example.toml`, and
cover the boundary in `crates/astrum-sim/tests/scenario.rs`. The sim serves the scenario back on
`astrum/scenario` (and the recorder stores it in the bag) via `Scenario::to_toml`, so the field
is included automatically.

## 6. Tests — the convention that matters

**Tests live in each crate's `tests/` directory as integration tests — never
inline `#[cfg(test)] mod tests` in `src/`.** Rationale: `src/` stays production
code, and integration tests exercise the crate through its **public API** (the
same surface real callers use). If a test needs an item, that item must be `pub`.
Write behaviour and boundary tests, not tests of wiring or forwarding.

A minimal test — create `crates/astrum-geo/tests/my_test.rs`:

```rust
use astrum_geo::{Geodetic, LocalFrame};   // import from the crate, like a user

#[test]
fn origin_maps_to_zero() {
    let o = Geodetic::from_degrees(40.0, -105.0, 1600.0);
    let enu = LocalFrame::new(o).geodetic_to_enu(&o);
    assert!(enu.e.abs() < 1e-6 && enu.n.abs() < 1e-6 && enu.u.abs() < 1e-6);
}
```

**Format, lint, test** — each target runs in a fresh container, so run them from the host:

| Target | Rust | C++ | Python |
|---|---|---|---|
| `make fmt` | `cargo fmt --all` | `clang-format -i` | `ruff format` + `ruff check --fix` |
| `make fmt-check` | `cargo fmt --all --check` | `clang-format --dry-run --Werror` | `ruff format --check` |
| `make lint` | `cargo clippy --workspace --all-targets -- -D warnings` | — | `ruff check` |
| `make test-*` | `test-rust`: `cargo test --workspace` | `test-cpp`: CMake build + `ctest --test-dir build/controllers` | `test-python`: builds `astrum_env`, pytest over `controllers/python`, `controllers/optimal/python`, `crates/astrum-env/python`, then the env smoke |

`make test` runs all three suites, `make deny` runs `cargo deny check`, and **`make ci`** is the
full gate (fmt-check + lint + test + deny). C++ and Python sources are the git-tracked files listed
by `scripts/sources.sh` (config in `.clang-format`, `ruff.toml`).

Two things to know:
- **Shared test fixtures** go in `tests/common/mod.rs` (a subdirectory with
  `mod.rs` is *not* itself a test binary). Put `#![allow(dead_code)]` at its top,
  since each test binary uses only some of the helpers. See
  `crates/astrum-sim/tests/common/mod.rs`.
- **The sim's tests drive the real app.** `astrum-sim` is a **lib + bin**: the
  logic is in `src/lib.rs` (`build_app(scenario, mode)`), and `src/main.rs` is thin process
  wiring. Tests call `build_app` (or the embedded `SimEnv`) and pump `app.update()` — the
  same schedule the binary runs. See `crates/astrum-sim/tests/simulation.rs`.

**JSBSim and threads:** JSBSim has process-global initialisation state (a lazily-filled
unit-conversion table and a static debug level written by every constructor). `astrum-fdm`
owns a process-wide lock around it — lifecycle calls (load, IC, trim, reset, drop) exclusive,
everything else shared — so `Fdm` is `Send` and tests need **no** locking of their own; build
FDMs from as many threads as you like. `crates/astrum-fdm/tests/concurrency.rs` is the
regression test.

## 7. Running the whole stack (sim + controller)

The demos are one-command, self-terminating, and tear themselves down:

```bash
make demo           # C++ controller flies aircraft 1 in the bounded scenario
make demo-python    # same, with the Python controller
make demo-quad      # F450 quadrotor climbs + yaws under the built-in multirotor autopilot
make demo-formation # leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
make smoke          # all four; fails on the first failure (CI runs this)
```

Under compose a one-shot `build` service compiles the release binaries and the C++
controllers once; the other services wait for it, and controllers also wait for the sim's
healthcheck.

**Watch it in 3-D.** `make replay` starts the fleet sim + the `astrum-web` bridge; open
http://localhost:8080 to see every aircraft on the ENU grid (orbit/follow cameras, trails,
playback transport). It works against a *live* sim or a bag being replayed by `astrum-bag-play`
(both listen on the endpoint the bridge connects to). For an offline, service-free view,
bake an MCAP bag to JSON and drag it onto the page (or use "Open file…"):

```bash
docker compose --profile record up recorder           # record to recordings/session.mcap
make replay                                           # sim + renderer at http://localhost:8080
make export BAG=recordings/session.mcap OUT=out.json  # MCAP → astrum-replay (v2) JSON
make down                                             # stop the sim + renderer
```

The renderer is static files under `web/` (Three.js vendored in `web/vendor/`; the modules
live in `web/js/`); the bridge and exporter are the two binaries in `crates/astrum-web/`, both
projecting telemetry through the one `astrum_web::FrameJson` shape. The bridge sends one
batched snapshot of the fleet at most `--max-hz` (default 30) times per second.

**Ground map.** For **fixed-wing** fleets — which fly a real geographic route — the renderer lays a
slippy-map (XYZ / Web-Mercator) aerial layer flat under the aircraft, centred on the scenario's ENU
datum. Multirotor and idle views keep the plain ENU grid. Tiles are served by `astrum-web` at
`/tiles/{z}/{x}/{y}.jpg`: it proxies an upstream tile server and disk-caches every tile under
`tilecache/` (git-ignored). So any datum gets crisp imagery with zero manual bundling, and an area
viewed once while online works **offline** afterward; a tile that 404s (offline + uncached) is
skipped and the grid shows through. Compose passes `ASTRUM_TILE_URL`, `ASTRUM_TILE_CACHE` and
`ASTRUM_WEB_MAX_HZ` through from your shell when set:

```bash
# Default: Esri World Imagery (global aerial, no key). Point elsewhere with:
ASTRUM_TILE_URL='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}' make replay  # public-domain, US-only
ASTRUM_TILE_URL='' make replay                   # cache-only: serve tiles already cached, never fetch
ASTRUM_TILE_CACHE=/workspace/mytiles make replay # relocate the on-disk cache (default ./tilecache)
```

Imagery is © its provider (the page footer credits the Esri default); mind the tile source's terms
of service if you redistribute or use it beyond evaluation.

**To develop your own controller**, don't add a compose service — treat it as a
process you run in a dev shell against a running sim:

```bash
# 1. start just the sim (bounded scenario), detached
ASTRUM_SCENARIO=config/scenario.demo.toml docker compose up -d sim

# 2. open a shell on the same network (sim resolves by name)
make shell

# 3. build + run YOUR controller, pointing at the sim; rerun as you iterate.
#    The bundled controllers take flags (env vars still work as a fallback):
python controllers/python/example.py --connect tcp/sim:7447 --id 1
build/controllers/cpp/astrum-controller --connect tcp/sim:7447 --id 1

# 4. when done
docker compose down
```

Compose services (`controller`, `controller-python`, …) are just **canned demo
presets** — add one only when you want a controller to be a permanent part of the
topology. See `controllers/cpp/README.md` and `controllers/python/README.md` for
how a controller subscribes, decodes telemetry, and publishes commands.

**RL environment.** `bash scripts/build-astrum-env.sh` builds the `astrum_env` module and prints
its staging directory for `PYTHONPATH`; `make test-python` does that and runs its tests. See
`crates/astrum-env/README.md` and `docs/dogfight-env-onboarding.md`.

## 8. Cap'n Proto wire quick reference

The schema (`crates/astrum-proto/schema/astrum.capnp`, v2) defines two messages:
- **`Telemetry`** — sim → subscribers (`astrum/telemetry/<id>`; the id is in the key). Geodetic
  position (HAE), world-grid `enu` + `velEnu`, local-level `velNed`, attitude, `tasMps`,
  `faultFlags`, `vehicleKind`, `tick`, `bodyRates`, `aero`, `combat`, `controlMode`.
- **`Command`** — controllers → sim; `targetId` (must equal the key id), a union of **`raw`**
  actuator setpoints (`fcs/*-cmd-norm`) *or* high-level **`autopilot`** targets (heading, HAE
  altitude, TAS), plus `seq`, `telemetryTick` and `trigger`.

Recordings are MCAP files (one channel per Zenoh key, raw payload bytes); there is no
recording message in the schema.

Change the schema in **one** place; Rust regenerates on build, C++ regenerates via its
CMake step, and Python loads it at runtime (`capnp.load(...)`).

**Adding a schema field.** Changes must be **additive**: append a field with the next ordinal
(never renumber, retype or reuse one), so old and new peers stay wire-compatible. Then:

1. Rust: extend the types and encode/decode in `crates/astrum-proto/src/lib.rs`.
2. C++: extend the codec in `controllers/common/cpp` (`wire.cpp`); Python: `astrum_client/wire.py`.
3. Add the canonical value to `crates/astrum-proto/tests/fixtures/values.json` and teach the three
   fixture writers about it (`crates/astrum-proto/examples/wire_fixture.rs`,
   `controllers/common/cpp/tools/wire_fixture.cpp`, `astrum_client/wire_fixture.py`).
4. `make wire-fixtures` regenerates all nine `<kind>.<lang>.bin` files; `make test` checks that
   every language decodes every file to `values.json` exactly.

## 9. Before you commit

```bash
make fmt   # format everything
make ci    # fmt-check + lint + all tests + cargo deny
```

`make ci` must pass — that's the bar, and it's what GitHub Actions runs
(`.github/workflows/ci.yml`: build the image, `make verify`, `make ci`, `make smoke`).
