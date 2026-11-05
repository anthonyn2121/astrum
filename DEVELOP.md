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
  `get astrum/scenario` with its scenario TOML, `get astrum/status` with its run status
  (JSON), and `get astrum/control/<pause|resume|step|speed>` requests (`astrum ctl`).
- **Controllers** (`controllers/*`) are separate processes — the flight-control
  logic. They **subscribe** to telemetry and **publish** commands. They never touch
  the sim's memory; only bytes over Zenoh.
- **The wire** is [Cap'n Proto](https://capnproto.org/), one schema (v2,
  `crates/astrum-proto/schema/astrum.capnp`) shared by Rust, C++, and Python.

```
controllers ──put astrum/command/<id>─────►  sim (subscribes astrum/command/*)
controllers ◄──put astrum/telemetry/<id>───  sim (publishes per aircraft; id in the key)
subscribers ◄──put astrum/events───────────  sim (hits, kills, mode changes, ...; one per event)
controllers ──token astrum/alive/controller/<id>──►  sim (liveliness: who is connected)
astrum ctl  ──get astrum/control/*, astrum/status──►  sim (pause / step / speed / status)
```

Both key spaces ride **one** Zenoh session on a single endpoint (`tcp/…:7447`). The sim (and
`astrum play`) **listens as a Zenoh router**; every other process (recorder, web bridge, C++
and Python controllers) opens a `mode=client` session that connects to it. The router routes
between its clients, so the recorder sees the commands a controller sends (Zenoh 1.10 routers do
not forward between two *peers*, which is why clients are not peers). A client's open waits for a
sim that is still starting, up to its connect timeout, then fails with a clear message:
controllers exit 1 after `--startup-timeout`; `astrum record` and `astrum web` take
`--connect-timeout` (`ASTRUM_CONNECT_TIMEOUT_S`, default 60 s).

Each entity is addressed by its id *in the key* — not a socket or a port — so **one endpoint serves
the whole fleet no matter how many aircraft you add**, and any client can subscribe to a single
aircraft (`astrum/telemetry/<id>`) or the entire fleet (`astrum/telemetry/*`). The command key id
must equal `Command.targetId`; the sim refuses mismatches.

Rules every client follows (the shared C++ and Python clients implement them for you):

- QoS for both streams: CongestionControl `Drop`, express, priority `InteractiveHigh`, declared
  publishers; controller sessions ask for Zenoh's low-latency transport
  (`transport/unicast/lowlatency = true`, `transport/unicast/qos/enabled = false`).
- Telemetry subscriptions are **latest-value** (one slot per aircraft), never a FIFO.
- **One command per telemetry sample**, carrying `seq` (strictly increasing from 1 per process)
  and `telemetryTick` (the `Telemetry.tick` it answers). Raw commands are a **lease**: stop sending
  for `sim.command_lease_s` and the sim's failsafe hold takes over (`controlMode = failsafe`).
- **Presence**: a controller holds the liveliness token `astrum/alive/controller/<id>` while it
  runs. When it goes away (exit, crash, lost connection) a lockstep run fails at once and any
  other commanded aircraft enters the failsafe hold without waiting for its lease. A controller
  without a token still works; leases and the lockstep timeout cover it.
- A raw control law checks `vehicleKind` and refuses (exits non-zero) any kind it wasn't written
  for, including unknown values.
- Decoding is untrusted input: a bad payload is logged, counted and skipped.

## 2. The golden rule: everything runs in Docker

Nothing (Rust, JSBSim, Zenoh, Cap'n Proto) is installed on the host — it all lives
in the `astrum-dev` image. You run commands *inside* a container, as the unprivileged
`astrum` user built with your UID/GID (the Makefile exports them), so files you create in
the bind mount stay yours.

```bash
make dev        # build the astrum-dev image if missing or changed, start the long-lived dev
                # container, open a bash shell in it at /workspace (rerun: another shell, same container)
make build      # rebuild the image explicitly (compiles JSBSim from source — slow first time)
make down       # stop and remove the dev container
make clean      # down + volumes + the local image
make ps         # container status
```

Your source is bind-mounted at `/workspace`, so **edit files on the host in your
normal editor**, and build/run/test them in the container. Shells opened by `make dev`
share one container, so their processes see each other on loopback. `verify fmt fmt-check
lint test test-rust test-cpp test-python wire-fixtures deny bench smoke ci` work both on the
host (each in a fresh `docker compose run --rm -T dev` container) and inside the dev
container (run in place); `dev build down clean ps` are host-only.

The container publishes 7447 (Zenoh) and 8080 (web viewer) on **127.0.0.1** on the host only.

The dev shell's prompt is starship's Catppuccin Mocha powerline preset, with Catppuccin `ls`
colours (vivid). Its separators and icons are Nerd Font glyphs: set a
[Nerd Font](https://www.nerdfonts.com/) (e.g. JetBrainsMono Nerd Font) in your host terminal,
or they show as boxes; a [Catppuccin](https://catppuccin.com/ports) theme for the terminal
itself matches the rest. Point `STARSHIP_CONFIG` at your own `starship.toml` to change it
(the image's is `/etc/astrum/starship.toml`, the wiring `docker/shell/bashrc`).

## Quick start

The fast path from a fresh checkout to a running sim. Every command below runs
**inside the dev container** (§2). Open a container shell (the first run builds the image)
and run steps 1–3 inside it:

```bash
make dev        # on the host; run the steps below in the shell it opens
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

Every binary honours an `ASTRUM_*` env var per flag as a fallback (so scripts and
`make` can configure them). Precedence is **CLI arg > `$ENV` > default** — see `--help`.

The sim loads the fleet from the TOML, spins up (and, for fixed-wing, trims) one JSBSim
model per aircraft, then publishes telemetry over Zenoh until `max_time_s` is reached (or
you Ctrl-C). Any startup failure — bad scenario, model load, trim, network — exits
non-zero with a message. Scenario format + the bundled examples are in §5.

To see everything working together — sim + controller + the 3-D browser view —
the demos do it in one command: `astrum run config/demos/cpp.toml` and
`astrum run config/scenario.example.toml --watch` (§7).

## 3. Repo layout — where files go

```
crates/            # Rust workspace (one crate per responsibility)
  astrum-fdm/       #   JSBSim FFI: safe `Fdm` wrapper (C shim, crate-owned lock, `Fdm: Send`)
  astrum-geo/       #   WGS84 geodetic ↔ ECEF ↔ ENU math, local-level ↔ grid rotations
  astrum-proto/     #   the Cap'n Proto schema v2 + encode/decode + wire fixtures
  astrum-net/       #   Zenoh transport (keys, QoS, LatestTelemetry, CommandSender, events, presence, control, scenario)
  astrum-bag/       #   MCAP record (telemetry + commands + events) / replay (telemetry + events)
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
scripts/           # bin/astrum, build-release, verify-toolchain, wire-fixtures, build-astrum-env, sources
docs/              # architecture diagram, dogfight env design + onboarding
docker-compose.yml # the one long-lived `dev` service
Makefile           # the commands you'll actually type
```

**Adding things:**
| I want to add… | Put it in… | Then… |
|---|---|---|
| A new Rust library crate | `crates/<name>/` | add it to `members` in the root `Cargo.toml` |
| A new controller | `controllers/<name>/` | C++: `add_subdirectory` in `controllers/CMakeLists.txt`; (optional) a run file under `config/demos/` |
| A new scenario | `config/<name>.toml` | run with `astrum run config/<name>.toml` |
| A scenario field | `crates/astrum-sim/src/scenario.rs` | see §5 |
| A schema field | `crates/astrum-proto/schema/astrum.capnp` | see §8 |
| A test | `crates/<crate>/tests/<name>.rs` | see §6 |

## 4. Build & run the sim

Inside a dev shell (`make dev`):

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
| `--threads` | `ASTRUM_THREADS` | scenario `sim.threads` | FDM stepping threads (0 = all cores, capped at the fleet size) |
| `--speed` | `ASTRUM_SPEED` | scenario `sim.speed` | pace as × real time (1 = real time, 0 = as fast as possible) |
| `--record` | `ASTRUM_RECORD` | — | record every tick's telemetry + received commands to this MCAP bag, in-process |

SIGINT/SIGTERM end the run cleanly at the end of the tick (a `--record` bag is finished); a
second signal exits at once. `make bench` measures the headless tick cost for a few fleet sizes
and thread counts. To fly many variants of a scenario at once, write a sweep file and run
`astrum sweep config/sweep.<name>.toml --jobs 8 --overwrite` (README §Run files and sweeps).

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
speed      = 1.0             # × wall-clock (1 for live control / watching); 0 = as fast as possible
seed       = 0               # gun dispersion + each aircraft's JSBSim RNG
threads    = 1               # FDM stepping threads (0 = all cores, ≤ fleet); identical results for any value
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
# control_rate_hz = 30.0     # telemetry (+ lockstep commands) at this rate; divides rate_hz
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

The demos are run files under `config/demos/` (README §Run files and sweeps); each is one
command, self-terminating, and tears itself down:

```bash
astrum list                                # every scenario / run / sweep file, with its kind
astrum run config/demos/cpp.toml           # C++ controller flies aircraft 1 in the bounded scenario
astrum run config/demos/python.toml        # same, with the Python controller
astrum run config/demos/quad.toml          # F450 quadrotor climbs + yaws under the built-in multirotor autopilot
astrum run config/demos/formation.toml     # leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
make smoke                                 # every config/demos/*.toml; fails on the first failure (CI runs this)
```

`astrum` rebuilds the release binaries and the C++ controllers first when their sources
changed (a no-op otherwise). Run it from `/workspace`: paths in files are relative to it.

**Watch it in 3-D.** `--watch` (e.g. `astrum run config/scenario.example.toml --watch`)
also starts the `astrum web` bridge; open http://localhost:8080 to see every aircraft on the ENU grid (orbit/follow cameras, trails,
playback transport). It works against a *live* sim or a bag being replayed by `astrum play`
(both listen on the endpoint the bridge connects to). For an offline, service-free view,
bake an MCAP bag to JSON and drag it onto the page (or use Source ▸ "Open export file…"):

```bash
astrum run config/scenario.example.toml --watch --record recordings/session.mcap  # sim + renderer, recorded
astrum record recordings/session.mcap                 # or record an already-running sim
astrum play recordings/session.mcap                   # replay it (run `astrum web` in another shell)
astrum export recordings/session.mcap --out out.json  # MCAP → astrum-replay (v3) JSON
```

The renderer is static files under `web/` (Three.js and uPlot vendored in `web/vendor/`; the
modules live in `web/js/`: `main.js` owns playback and selection state, `store.js` the timeline,
`scene.js`/`meshes.js` the 3-D view, `ui.js` the shell and transport, `fleet.js`, `inspector.js`,
`eventlog.js` and `plots.js` the docks, `geometry.js` the relative geometry); the bridge and
exporter are the two binaries in `crates/astrum-web/`, both projecting telemetry through the one
`astrum_web::FrameJson` shape. The bridge sends one batched snapshot of the fleet at most
`--max-hz` (default 30) times per second. Telemetry carries only the vehicle class, so the bridge
also polls `astrum/scenario` every 5 s and sends each aircraft's JSBSim model
(`{"type":"models","models":[{"id","model"}…]}`) on connect and whenever it changes; exports carry
the same list as `meta.models`.

**Ground map and sky.** Only when the fleet includes an **F-16** (a scenario `model` starting with
`f16`; the bridge polls the sim's scenario and sends `{"type":"models",…}`, exports carry
`meta.models`) does the renderer switch to daytime: a sky dome with the sun, a drifting cumulus deck
(~1.8 km) and cirrus (~9 km), haze fog, and slippy-map (XYZ / Web-Mercator) aerial imagery streamed
in level-of-detail rings around the ground point below the camera — fine tiles near it, coarser
ones out to ~150 km — each tile placed at its exact WGS84 position in the scenario's ENU frame, so
the map follows the Earth's curvature and stays under the aircraft however far they fly. Every other
platform (c172x, F450, unknown) keeps the dark ENU grid with no map. Tiles are served by
`astrum web` at `/tiles/{z}/{x}/{y}.jpg`: it proxies an upstream tile server and disk-caches every
tile under `tilecache/` (git-ignored). So any datum gets crisp imagery with zero manual bundling,
and an area viewed once while online works **offline** afterward; a tile that 404s (offline +
uncached) is skipped and the coarser level or the haze shows through. Set
`ASTRUM_TILE_URL`, `ASTRUM_TILE_CACHE` or `ASTRUM_WEB_MAX_HZ` in the dev shell to override them:

```bash
# Default: Esri World Imagery (global aerial, no key). Point elsewhere with:
ASTRUM_TILE_URL='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}' astrum web  # public-domain, US-only
ASTRUM_TILE_URL='' astrum web                    # cache-only: serve tiles already cached, never fetch
ASTRUM_TILE_CACHE=/workspace/mytiles astrum web  # relocate the on-disk cache (default ./tilecache)
```

Imagery is © its provider (the page footer credits the Esri default); mind the tile source's terms
of service if you redistribute or use it beyond evaluation.

**To develop your own controller**, treat it as a process you run in a dev shell against
a running sim:

```bash
# 1. shell 1: start just the sim (bounded scenario; add --watch for the viewer)
astrum run config/scenario.demo.toml

# 2. shell 2: another `make dev` on the host (same container, same loopback)

# 3. build + run YOUR controller against the default tcp/127.0.0.1:7447; rerun as you iterate.
#    The bundled controllers take flags (env vars still work as a fallback):
python controllers/python/example.py --id 1
build/controllers/cpp/astrum-controller --id 1

# 4. when done: Ctrl-C in shell 1
```

Run files (`config/demos/*.toml`) are just **canned demo presets** — add your controller to
one's `controllers` when you want it launched with the sim. See `controllers/cpp/README.md` and `controllers/python/README.md` for
how a controller subscribes, decodes telemetry, and publishes commands.

**RL environment.** `bash scripts/build-astrum-env.sh` builds the `astrum_env` module and prints
its staging directory for `PYTHONPATH`; `make test-python` does that and runs its tests. See
`crates/astrum-env/README.md` and `docs/dogfight-env-onboarding.md`.

## 8. Cap'n Proto wire quick reference

The schema (`crates/astrum-proto/schema/astrum.capnp`, v2) defines three messages:
- **`Telemetry`** — sim → subscribers (`astrum/telemetry/<id>`; the id is in the key). Geodetic
  position (HAE), world-grid `enu` + `velEnu`, local-level `velNed`, attitude, `tasMps`,
  `faultFlags`, `vehicleKind`, `tick`, `bodyRates`, `aero`, `combat`, `controlMode`.
- **`Command`** — controllers → sim; `targetId` (must equal the key id), a union of **`raw`**
  actuator setpoints (`fcs/*-cmd-norm`) *or* high-level **`autopilot`** targets (heading, HAE
  altitude, TAS), plus `seq`, `telemetryTick` and `trigger`.
- **`Event`** — sim → subscribers (`astrum/events`); `tick`, `simTimeS` and a `kind` union
  (trigger pressed / released, hit, kill, mode change, controller up / down, run ended). See the
  README's "Events".

Recordings are MCAP files (one channel per Zenoh key, raw payload bytes); there is no
recording message in the schema.

Change the schema in **one** place; Rust regenerates on build, C++ regenerates via its
CMake step, and Python loads it at runtime (`capnp.load(...)`).

**Adding a schema field.** Changes must be **additive**: append a field with the next ordinal
(never renumber, retype or reuse one), so old and new peers stay wire-compatible. Then:

1. Rust: extend the types and encode/decode in `crates/astrum-proto/src/lib.rs`.
2. C++: extend the codec in `controllers/common/cpp` (`wire.cpp`); Python: `astrum_client/wire.py`.
   (`Event` has no C++ codec: nothing in C++ consumes it.)
3. Add the canonical value to `crates/astrum-proto/tests/fixtures/values.json` and teach the
   fixture writers about it (`crates/astrum-proto/examples/wire_fixture.rs`,
   `controllers/common/cpp/tools/wire_fixture.cpp`, `astrum_client/wire_fixture.py`). A new event
   kind is an `event_<kind>` entry plus its name in `EVENT_KINDS` in
   `crates/astrum-proto/tests/fixture/mod.rs` and `astrum_client/wire_fixture.py`.
4. `make wire-fixtures` regenerates every `<kind>.<lang>.bin` file (nine telemetry/command files
   in three languages, the events in Rust and Python); `make test` checks that every language
   decodes every file to `values.json` exactly.

## 9. Before you commit

```bash
make fmt   # format everything
make ci    # fmt-check + lint + all tests + cargo deny
```

`make ci` must pass — that's the bar, and it's what GitHub Actions runs
(`.github/workflows/ci.yml`: build the image, `make verify`, `make ci`, `make smoke`).
