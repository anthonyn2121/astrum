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
- **The wire** is [Cap'n Proto](https://capnproto.org/), one schema (v3,
  `crates/astrum-proto/schema/astrum.capnp`) shared by Rust, C++, and Python.

```
controllers ──put astrum/command/<id>─────►  sim (subscribes astrum/command/*)
controllers ◄──put astrum/telemetry/<id>───  sim (publishes per aircraft; id in the key)
subscribers ◄──put astrum/events───────────  sim (hits, kills, mode changes, ...; one per event)
subscribers ◄──put astrum/tick─────────────  sim (TickEnd: after a tick's telemetry and events)
controllers ──token astrum/alive/controller/<id>──►  sim (liveliness: who is connected)
astrum ctl  ──get astrum/control/*, astrum/status──►  sim (pause / step / speed / status)
```

Both key spaces ride **one** Zenoh session on a single endpoint (`tcp/…:7447`). The sim (and
`astrum play`) **listens as a Zenoh router**; every other process (recorder, web bridge, C++
and Python controllers) opens a `mode=client` session that connects to it. The router routes
between its clients, so the recorder sees the commands a controller sends (Zenoh 1.10 routers do
not forward between two *peers*, which is why clients are not peers). A client's open waits for a
sim that is still starting, up to its connect timeout, then fails with a clear message:
controllers exit 1 after `--startup-timeout`; `astrum record` takes `--connect-timeout`
(`ASTRUM_CONNECT_TIMEOUT_S`, default 60 s); `astrum web` never gives up: it keeps retrying in the
background and serves pages meanwhile.

Each entity is addressed by its id *in the key* — not a socket or a port — so **one endpoint serves
the whole fleet no matter how many aircraft you add**, and any client can subscribe to a single
aircraft (`astrum/telemetry/<id>`) or the entire fleet (`astrum/telemetry/*`). The command key id
must equal `Command.targetId`; the sim refuses mismatches.

Rules every client follows (the C++ and Python SDKs implement them for you — see
[`controllers/README.md`](controllers/README.md)):

- QoS for telemetry and commands: CongestionControl `Drop`, express, priority `InteractiveHigh`,
  declared publishers; events and `TickEnd`: `Block`, priority `Data`. **Every client** — the
  controller SDKs and `astrum_net::Session::connect` (recorder, web bridge, `astrum ctl`) — asks for
  Zenoh's low-latency transport (`transport/unicast/lowlatency = true`,
  `transport/unicast/qos/enabled = false`) for its streams; it has one lane per link, so priorities
  reorder nothing on it. The recorder and bridge must not use the default transport for them:
  Zenoh 1.10's transmission pipeline on a link without QoS lanes (the sim's router disables them
  for low latency) sets the congestion flag on bit `Priority::DEFAULT` but `refill` clears the bit
  of queue 0 (`zenoh-transport` `src/common/pipeline.rs`, `push_network_message` / both `refill`
  impls; 1.10.2, the latest, is the same), so after one dropped sample a client never received
  telemetry again — the bridge went blank at `--speed 0`, or after a hiccup even at 1×. (On the
  default transport ~7000 of a 20000-sample burst arrived, then none; `tests/low_latency.rs` pins
  that a slow client keeps receiving on the clients' transport.) The price: nothing is dropped on
  low latency, so **a client that falls behind slows the sim, and one that stops reading — a
  suspended recorder, web bridge, `astrum ctl` or controller — freezes it, and its control plane
  with it, until it reads again or its connection dies** (the command lease does not release it:
  measured, a stopped bridge held `--speed 0` at one tick for 43 s). There is also no
  fragmentation (a message must fit one frame: the TCP link's Zenoh MTU, ~48 KiB on loopback, ≥ 32
  KiB on any link), so a Rust client's **queries** — scenario, status, control — use a second
  session on the default transport (`Session::connect` opens both; `Session::close` closes both):
  their replies are `Block`, which the latch does not touch, and fragment, so a scenario of any
  size arrives.
- Telemetry subscriptions are **latest-value** (one slot per aircraft), never a FIFO — except the
  web bridge's `SampledTelemetry`, a bounded trail thinned by sim time.
- **One command per telemetry sample**, carrying the controller instance's random `epoch`, `seq`
  (strictly increasing from 1 within the epoch) and `telemetryTick` (the `Telemetry.tick` it
  answers). Raw commands are a **lease**: stop sending for `sim.command_lease_s` and the sim's
  failsafe hold takes over (`controlMode = failsafe`).
- **Ownership**: the first epoch whose command the sim applies owns the aircraft; other epochs are
  refused until the owner's token disappears or its lease expires. `Telemetry.command` reports the
  owner and every refusal — a controller that finds itself refused as `notOwner` stops (exit 2).
- **Presence**: a controller holds the liveliness token `astrum/alive/controller/<id>` while it
  runs. When it goes away (exit, crash, lost connection) a lockstep run fails at once and any
  other commanded aircraft enters the failsafe hold without waiting for its lease. A controller
  without a token still works; leases and the lockstep timeout cover it.
- A law declares the vehicle kinds it flies; the runtime refuses (exit 2) any other kind,
  including unknown values. A law's `dt` is `Telemetry.controlPeriodS`, never a flag.
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
  astrum-proto/     #   the Cap'n Proto schema v3 + encode/decode + wire fixtures
  astrum-net/       #   Zenoh transport (keys, QoS, LatestTelemetry, CommandSender, events, tick ends, presence, control, scenario)
  astrum-bag/       #   MCAP record (telemetry + commands + events) / replay (telemetry + events) / run metrics (astrum-bag-metrics) / per-run flight tracks for sweep overviews (`tracks`)
  astrum-web/       #   axum bridge + tile proxy (astrum-web) + MCAP→JSON exporter (astrum-web-export)
  astrum-sim/       #   the Bevy ECS app — the sim binary + the embedded env seam (env.rs) + run supervision (launch.rs), sweeps (sweep.rs, IC offsets perturb.rs), file/environment catalog (catalog.rs), controller catalog (controllers.rs), the astrum CLI
  astrum-env/       #   PyO3 module `astrum_env` + the dogfight Gym adapter (python/)
controllers/       # flight controllers (clients); README.md is the SDK contract, CMakeLists.txt the one C++ project
  common/cpp/       #   the C++ SDK: astrum::wire / control / cli / client / runtime + wire-fixture tool
  cpp/  python/     #   example controllers (python/astrum_client is the Python SDK)
  optimal/          #   LQR / MPC / figure-eight / formation laws on the SDK runtime (cpp/, python/astrum_optimal)
  examples/         #   out_of_tree/: a C++ controller built against an installed SDK
web/               # browser 3-D replay renderer (Three.js, vendored; served by astrum-web)
config/            # scenario files (*.toml) — the fleet + world origin; run files, sweeps; controllers.toml (the controller catalog)
docker/            # Dockerfile, entrypoint, pinned Python requirements
scripts/           # bin/astrum, build-release, verify-toolchain, smoke, wire-fixtures, build-astrum-env, sources
docs/              # architecture diagram, dogfight env design + onboarding
docker-compose.yml # the one long-lived `dev` service
Makefile           # the commands you'll actually type
```

**Adding things:**
| I want to add… | Put it in… | Then… |
|---|---|---|
| A new Rust library crate | `crates/<name>/` | add it to `members` in the root `Cargo.toml` |
| A new controller | `controllers/<name>/` | C++: `add_subdirectory` in `controllers/CMakeLists.txt`; one `config/controllers.toml` entry per law (the web studio launches only catalog entries); (optional) a run file under `config/demos/` |
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

Process supervision lives in the `astrum_sim::launch` library, not the CLI: `LiveRun` starts one
run (the sim, companions such as the `--watch` viewer, the controllers) and supervises it on its
own thread, routing output to the tagged terminal, per-process log files, or `LogLine`s over a
channel; `run_processes` is a sweep's process mode on top of it. There is no process-global state:
the caller owns the `Children` registry (the CLI's second Ctrl-C kills it) and each run's stop
flag. `LiveRun::stop`/`kill`/drop act on that run only; the spec's shared `stop` stops every run
sharing it. `start` (on failure), `wait` and drop block until the run's processes are reaped —
call them from a blocking thread in async code. `astrum_sim::catalog` classifies files for
`astrum list`, finds JSBSim models (`jsbsim_models`) and defines the environments (`empty`).

Sweeps are the `astrum_sim::sweep` library, the CLI only prints: `Sweep::plan` (or
`plan_scenario` from an in-memory base `Scenario`) expands the grid, draws the `[[perturb]]`
samples (`astrum_sim::perturb`: level-frame offsets via astrum-geo, rigid groups, spreads, from
the crate's one PRNG `rng::SplitMix64`, which `CombatRng` also wraps) and validates every run;
`check_trim` builds each run's trimmed aircraft with the sim's own `build_fdm`; `execute(sweep,
jobs, stop, on_event, run_one)` flies the pool, calling `on_event` with `RunEvent::Started` /
`Finished { outcome, metrics }` from the worker threads, writing `<stem>.metrics.json` and
rewriting `summary.csv` (atomically) at every start and finish. `SweepFile` serialises back to
TOML (`to_toml`).

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
# climb_deg = 0.0 / bank_deg = 0.0   # flight-path angle (+ climbing) / roll (+ right wing down) at spawn; |x| < 90
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

Three things to know:
- **Shared test fixtures** go in `tests/common/mod.rs` (a subdirectory with
  `mod.rs` is *not* itself a test binary). Put `#![allow(dead_code)]` at its top,
  since each test binary uses only some of the helpers. See
  `crates/astrum-sim/tests/common/mod.rs`.
- **The sim's tests drive the real app.** `astrum-sim` is a **lib + bin**: the
  logic is in `src/lib.rs` (`build_app(scenario, mode)`), and `src/main.rs` is thin process
  wiring. Tests call `build_app` (or the embedded `SimEnv`) and pump `app.update()` — the
  same schedule the binary runs. See `crates/astrum-sim/tests/simulation.rs`.
- **Process supervision is tested without JSBSim.** `crates/astrum-sim/tests/launch.rs` drives
  `LiveRun`, `run_processes` and `astrum run` with shell-script stand-ins for `astrum-sim`.

**JSBSim and threads:** JSBSim has process-global initialisation state (a lazily-filled
unit-conversion table and a static debug level written by every constructor). `astrum-fdm`
owns a process-wide lock around it — lifecycle calls (load, IC, trim, reset, drop) exclusive,
everything else shared — so `Fdm` is `Send` and tests need **no** locking of their own; build
FDMs from as many threads as you like. `crates/astrum-fdm/tests/concurrency.rs` is the
regression test.

**Initial attitude and trim:** `InitialConditions` carries `gamma_deg` (flight-path angle) and
`phi_deg` (bank). A non-zero `gamma_deg` is staged after `ic/vt-fps` — JSBSim scales the existing
velocity vector, so a climb angle set at zero speed is silently lost — and a level start stages
exactly the pre-climb sequence (pinned bit-for-bit by `level_start_replays_the_original_staging_sequence`).
`TrimMode::Full` trims bank back to 0, so the sim uses `TrimMode::Turn` (JSBSim `tTurn`) for a
banked start; the shim mutes `setupTurn`'s INFO line while debug output is silenced.

## 7. Running the whole stack (sim + controller)

The demos are run files under `config/demos/` (README §Run files and sweeps); each is one
command, self-terminating, and tears itself down:

```bash
astrum list                                # every scenario / run / sweep file, with its kind
astrum controllers                         # the controller catalog: id, vehicle kinds, name
astrum controllers render py-hold --id 1   # one entry's command line (--param NAME=VALUE, --fleet IDS)
astrum run config/demos/cpp.toml           # C++ controller flies aircraft 1 in the bounded scenario
astrum run config/demos/python.toml        # same, with the Python controller
astrum run config/demos/quad.toml          # F450 quadrotor climbs + yaws under the built-in multirotor autopilot
astrum run config/demos/formation.toml     # leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
make smoke                                 # every config/demos/*.toml, then every catalog entry (10 s each), in lockstep, as fast as possible; fails on the first failure (CI runs this)
```

`astrum` rebuilds the release binaries and the C++ controllers first when their sources
changed (a no-op otherwise). Run it from `/workspace`: paths in files are relative to it.

**Watch it in 3-D.** `--watch` (e.g. `astrum run config/scenario.example.toml --watch`)
also starts the `astrum web` bridge; open http://localhost:8080/view.html to see every aircraft on the ENU grid (orbit/follow cameras, trails,
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
`scene.js`/`meshes.js` the 3-D view, `environment.js` the sky / map, `geo.js` WGS84 geodetic ⇄
ECEF ⇄ ENU (the same math as `crates/astrum-geo`, to ~1e-8 m), `ui.js` the shell and transport,
`fleet.js`, `inspector.js`, `eventlog.js` and `plots.js` the docks, `geometry.js` the relative
geometry, `simcontrol.js` the live sim's control bar and `runpanel.js` a studio run's panel); the
viewer page is `view.html`. The studio's other pages are `index.html` (home, `home.js`),
`interactive.html` (`interactive.js`) and the scenario editor `sandbox.html` (`sandbox.js` +
`editstore.js`), styled by `css/studio.css` / `css/sandbox.css` over `style.css`; every page
reaches `/api` only through `js/api.js`. The bridge and
exporter are the two binaries in `crates/astrum-web/`, both projecting telemetry through the one
`astrum_web::FrameJson` shape. The bridge thins every aircraft's telemetry by sim time
(`astrum_net::SampledTelemetry`: at most `--sample-hz`, default 30, samples per sim second plus
the newest; at most `BATCH_SAMPLES` = 120 pending per aircraft, past which the trail is halved —
every other sample, the newest always, and the period doubled — so a batch always spans the whole
stretch since the last one) and sends what arrived as one
`frames` batch at most `--max-hz` (default 30) times per second, so trails stay smooth faster
than real time; `store.js` appends each aircraft's samples in order and keeps 20 min of sim time
at 30 Hz (`LIVE_CAPACITY`). Telemetry carries only the vehicle class, so the bridge
also polls `astrum/scenario` every 5 s and sends each aircraft's JSBSim model
(`{"type":"models","models":[{"id","model"}…]}`) on connect and whenever it changes, and
`{"type":"run","run":N,"endpoint"}` on connect and on every `Bridge::attach`/`detach`
(`crates/astrum-web/src/bridge.rs`); exports carry the same models list as `meta.models`. The
bridge owns the browser feeds for the server's lifetime; each run gets a fresh session and
`SampledTelemetry`, and pumps publish only while their run is current, so nothing from an old run
follows the new run's message. `/api` lives in `src/api.rs`; its guard (`api::guard`) wraps the
whole nested API service, so new routes inherit the Host, Origin and JSON-body checks. Control
values are checked by `astrum_net::ControlRequest::validate` against `astrum_net::MIN_SPEED`, the
one limit the sim and the API share.

**Studio runs** (`crates/astrum-web/src/runs.rs`): `Runs` owns the one live interactive run. A
start is checked (assignments, then each controller program exists — so "not built" names the
aircraft instead of failing in `LiveRun::start`) and launched on the blocking pool
(`LiveRun::start` with `Log::Channel`), so a client that goes away mid-request cannot leave a
half-started run. The log and monitor threads exist before the run's record is installed and act
only once it is (a failed install drops the run and its lines); the monitor `wait`s for the run,
drains its log, writes `metrics.json`, sends the bridge back to `--connect` if it still carries
the run, and only then marks it finished — so `finished` means the metrics are on disk and the
bridge is home. Each `LiveSpec` controller is a `launch::ControllerCmd` carrying its name and log
source: `astrum run` and sweeps use `launch::numbered` (`controller k` / `ctlk`), studio runs
name them by aircraft (`aircraft N's controller` / `aircraft N`). Stop requests flip the run's own stop
handle and await the monitor's `watch` channel; nothing holds a lock across an await. The log
ring's `seq` is monotonic over the server's lifetime (a reader's stale `since` never hides a new
run's lines). `astrum-web` stops the active run when it is told to shut down (`Runs::close`) and
waits for it after serving (`Runs::shutdown`). Its tests (`tests/runs.rs`) fly a shell-script
stand-in sim and the real one, which must be built first (`cargo build -p astrum-sim`; or set
`ASTRUM_SKIP_REAL_SIM=1`).

**Scene edit API** (the sandbox; sweep previews build on it). `Scene.setEditing({onSelect, onEdit,
onPoint})` turns on edit mode (and the `"map"` environment: imagery for any fleet); the scene only
*reports* what the pointer did and draws what its store says. `onSelect(ids, "replace" | "toggle" |
"add")` for clicks / Shift+clicks / Shift+boxes; `onEdit({type: "move" | "raise" | "rotate", phase:
"start" | "move" | "end" | "cancel", changes})` with `[{id, e, n, u}]` (ENU m) or `[{id,
heading_deg}]` (the TRUE heading, Shift-snapped to 5° in true terms); and `onPoint({e, n})` for a
click on empty ground. Presses on an aircraft or knob are taken in the capture phase, so
OrbitControls and the tracked-view hand-off never steal them; a perspective drag ignores pointer
rays within 3° of the drag plane and hits beyond 4× the aircraft's distance. `setSelection(ids)`
(last = primary, which gets the handles), `cancelEdit()`, `setOrigin(lat°, lon°, alt)` (the map
datum from a `[world]`, no telemetry), `setCameraMode("map")` (orthographic, north up; pans and
zooms by itself), `viewCenter()` / `setViewCenter(e, n)`, and `setGhosts(layer, [{kind, e, n, u,
heading_deg, climb_deg?, bank_deg?, color?}])` (named instanced translucent preview layers —
`"spawn"` is the sandbox's, a sweep cloud takes its own; `null` clears one; each ghost sized by its
own distance). Static initial conditions render through `editstore.js`'s
`EditStore.setScenario(world, aircraft, kindOf)`, which offers what `Scene.update` reads from a
store (`ids()`, `track(id).kind`, `sampleAt(id, t)`). Headings everywhere — telemetry `psi_rad`,
editor samples, edit events, ghosts — are true headings in the aircraft's local level; the scene
turns each onto the grid by `geo.js` `Datum.northBearing` at the aircraft's lat/lon (meridian
convergence, ≈ −sin(lat)·Δlon: 0.2° at 30 km east at 37° N), in the viewer and the sandbox alike.
A horizontal drag keeps each aircraft's `alt_hae_m`; its grid height is recomputed, so far out it
sits below the flat drag plane by the Earth's curvature (d²/2R: ~18 m at 15 km).

**Ground map and sky.** Only when the fleet includes an **F-16** (a scenario `model` starting with
`f16`; the bridge polls the sim's scenario and sends `{"type":"models",…}`, exports carry
`meta.models`) does the renderer switch to daytime: a sky dome with the sun, a drifting cumulus deck
(~1.8 km) and cirrus (~9 km), haze fog, and slippy-map (XYZ / Web-Mercator) aerial imagery streamed
in level-of-detail rings around the ground point below the camera — fine tiles near it, coarser
ones out to ~150 km — each tile placed at its exact WGS84 position in the scenario's ENU frame, so
the map follows the Earth's curvature and stays under the aircraft however far they fly. Every other
platform (c172x, F450, unknown) keeps the dark ENU grid with no map in the viewer; the sandbox
editor always shows the map (the `"map"` look: daylight, no clouds), its origin set from the
scenario's `[world]`. Tiles are served by
`astrum web` at `/tiles/{z}/{x}/{y}.jpg`: it proxies an upstream tile server and disk-caches every
tile under `tilecache/` (git-ignored). So any datum gets crisp imagery with zero manual bundling,
and an area viewed once while online works **offline** afterward; a tile that 404s (offline +
uncached) is skipped and the coarser level or the haze shows through. Set `ASTRUM_TILE_URL`,
`ASTRUM_TILE_CACHE`, `ASTRUM_WEB_MAX_HZ` or `ASTRUM_WEB_SAMPLE_HZ` in the dev shell to override them:

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
python controllers/python/example.py --id 1                 # --law hold (default) | autopilot
build/controllers/cpp/astrum-controller --id 1

# 4. when done: Ctrl-C in shell 1
```

Run files (`config/demos/*.toml`) are just **canned demo presets** — add your controller to
one's `controllers` when you want it launched with the sim. [`controllers/README.md`](controllers/README.md)
is the SDK contract (the law API, the control loop, outcomes and exit codes);
`controllers/cpp/README.md` and `controllers/python/README.md` cover each SDK.

**Controller catalog.** When a controller program gains, renames or drops a flag or law, update
its entries in `config/controllers.toml`. `crates/astrum-sim/tests/controllers.rs` pins the
shipped entry list, kinds and default renders, and `controllers/optimal/python/tests/test_catalog.py`
parses every `py-*` entry's default command line with the real program parser (and checks the
law's vehicle kinds), so a renamed Python flag fails `make test-python`.

**RL environment.** `bash scripts/build-astrum-env.sh` builds the `astrum_env` module and prints
its staging directory for `PYTHONPATH`; `make test-python` does that and runs its tests. See
`crates/astrum-env/README.md` and `docs/dogfight-env-onboarding.md`.

## 8. Cap'n Proto wire quick reference

The schema (`crates/astrum-proto/schema/astrum.capnp`, v3) defines four messages:
- **`Telemetry`** — sim → subscribers (`astrum/telemetry/<id>`; the id is in the key). Geodetic
  position (HAE), world-grid `enu` + `velEnu`, local-level `velNed`, attitude, `tasMps`,
  `faultFlags`, `vehicleKind`, `tick`, `bodyRates`, `aero`, `combat`, `controlMode`,
  `controlPeriodS`, `command` (owner epoch, last applied seq, refusals).
- **`Command`** — controllers → sim; `targetId` (must equal the key id), a union of **`raw`**
  fixed-wing actuator setpoints (`fcs/*-cmd-norm`), **`multirotor`** body-rate demands +
  collective, *or* high-level **`autopilot`** targets (heading, HAE altitude, TAS), plus `epoch`,
  `seq`, `telemetryTick` and `trigger`.
- **`Event`** — sim → subscribers (`astrum/events`); `tick`, `simTimeS` and a `kind` union
  (trigger pressed / released, hit, kill, mode change, controller up / down, run ended). See the
  README's "Events".
- **`TickEnd`** — sim → subscribers (`astrum/tick`); `tick`, `simTimeS`, sent after the tick's
  telemetry and events.

Recordings are MCAP files (one channel per Zenoh key, raw payload bytes); there is no
recording message in the schema.

Change the schema in **one** place; Rust regenerates on build, C++ regenerates via its
CMake step, and Python loads it at runtime (`capnp.load(...)` of the copy packaged with
`astrum_client`, a symlink to the schema in the checkout).

**Adding a schema field.** Changes must be **additive**: append a field with the next ordinal
(never renumber, retype or reuse one), so old and new peers stay wire-compatible. Then:

1. Rust: extend the types and encode/decode in `crates/astrum-proto/src/lib.rs`.
2. C++: extend the codec in `controllers/common/cpp` (`wire.cpp`); Python: `astrum_client/wire.py`.
3. Add the canonical value to `crates/astrum-proto/tests/fixtures/values.json` and teach the
   fixture writers about it (`crates/astrum-proto/examples/wire_fixture.rs`,
   `controllers/common/cpp/tools/wire_fixture.cpp`, `astrum_client/wire_fixture.py`). A new event
   kind is an `event_<kind>` entry plus its name in the event kind lists of all three writers.
4. `make wire-fixtures` regenerates every `<kind>.<lang>.bin` file (every kind in Rust, C++ and
   Python); `make test` checks that every language decodes every file to `values.json` exactly.

## 9. Before you commit

```bash
make fmt   # format everything
make ci    # fmt-check + lint + all tests + cargo deny
```

`make ci` must pass — that's the bar, and it's what GitHub Actions runs
(`.github/workflows/ci.yml`: build the image, `make verify`, `make ci`, `make smoke`, with the
cargo registry and target dir cached between runs via `.github/compose.ci.yml`).
