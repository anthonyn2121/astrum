# Developing on ASTRUM

A practical guide for getting productive in this repo. For the *why* behind the
design, see [README.md](README.md); this doc is the *how*.

## 1. The mental model (read this first)

ASTRUM is **one server + many clients** talking over ZMQ.

- **The sim** (`astrum-sim`) is the authoritative state server. It's a headless
  [Bevy](https://bevyengine.org/) ECS: one *entity* per aircraft, each owning its
  own [JSBSim](https://github.com/JSBSim-Team/jsbsim) flight-dynamics model. Every
  tick it steps physics, then **PUB**lishes telemetry and **PULL**s commands.
- **Controllers** (`controllers/*`) are separate processes — the flight-control
  logic. They **SUB**scribe to telemetry and **PUSH** commands. They never touch
  the sim's memory; only bytes over ZMQ.
- **The wire** is [Cap'n Proto](https://capnproto.org/), one schema
  (`crates/astrum-proto/schema/astrum.capnp`) shared by Rust, C++, and Python.

```
controllers ──PUSH commands──►  sim (binds :5556 PULL)
controllers ◄──PUB telemetry──  sim (binds :5555 PUB, topic = aircraft id)
```

## 2. The golden rule: everything runs in Docker

Nothing (Rust, JSBSim, ZMQ, Cap'n Proto) is installed on the host — it all lives
in the `astrum-dev` image. You run commands *inside* a container.

```bash
make build      # build the image once (compiles JSBSim from source — slow first time)
make verify     # sanity-check the toolchain (compilers, libs, JSBSim link + run)
make shell      # open a shell in a fresh container (workspace mounted, on the network)
make help       # list all make targets
```

Your source is bind-mounted at `/workspace`, so **edit files on the host in your
normal editor**, and build/run/test them in a container. One-off commands:

```bash
docker compose run --rm --no-deps sim bash -lc '<command>'
```

## 3. Repo layout — where files go

```
crates/            # Rust workspace (one crate per responsibility)
  astrum-fdm/       #   JSBSim FFI: safe `Fdm` wrapper over the C++ engine
  astrum-geo/       #   WGS84 geodetic ↔ ECEF ↔ ENU coordinate math
  astrum-proto/     #   the Cap'n Proto schema + encode/decode helpers
  astrum-net/       #   ZMQ transport (PUB/SUB telemetry, PUSH/PULL commands)
  astrum-bag/       #   record/replay telemetry to a file ("bag")
  astrum-web/       #   WebSocket+HTTP bridge (astrum-web) + bag→JSON exporter (astrum-web-export)
  astrum-sim/       #   the Bevy ECS app — the sim binary
controllers/       # flight controllers (clients), one dir per language
  cpp/  python/
web/               # browser 3-D replay renderer (Three.js, vendored; static app served by astrum-web)
config/            # scenario files (*.toml) — the fleet + world origin
docker/            # Dockerfile + entrypoint for the dev image
scripts/           # helper scripts (e.g. verify-toolchain.sh)
docker-compose.yml # service presets: sim, controllers, recorder, replay
Makefile           # the commands you'll actually type
```

**Adding things:**
| I want to add… | Put it in… | Then… |
|---|---|---|
| A new Rust library crate | `crates/<name>/` | add it to `members` in the root `Cargo.toml` |
| A new controller | `controllers/<name>/` | (optional) add a compose service for demos |
| A new scenario | `config/<name>.toml` | run with `ASTRUM_SCENARIO=config/<name>.toml` |
| A test | `crates/<crate>/tests/<name>.rs` | see §6 |

## 4. Build & run the sim

Inside a shell (`make shell`) or via `docker compose run`:

```bash
cargo build --workspace              # compile everything
cargo run -p astrum-sim               # run the sim (uses config/scenario.example.toml)
ASTRUM_SCENARIO=config/scenario.demo.toml cargo run --release -p astrum-sim
```

Handy environment variables the sim reads:
| Var | Default | Meaning |
|---|---|---|
| `ASTRUM_SCENARIO` | `config/scenario.example.toml` | which scenario to load |
| `ASTRUM_TELEMETRY_BIND` | `tcp://0.0.0.0:5555` | telemetry PUB endpoint |
| `ASTRUM_COMMAND_BIND` | `tcp://0.0.0.0:5556` | command PULL endpoint |

## 5. Scenarios

A scenario is a TOML file: a world origin + a fleet. Copy an existing one and edit.

```toml
[world]                      # ENU grid origin (WGS84). Keep it near your fleet.
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_m   = 0.0

[sim]
rate_hz    = 120.0           # fixed physics step = 1/rate_hz seconds
max_time_s = 40.0            # 0 = run forever; >0 = stop after N sim-seconds
realtime   = true            # true ≈ wall-clock (for live control); false = as fast as possible

[[aircraft]]                 # one entity per block; `id` must be unique
id = 1
model = "c172x"              # a JSBSim aircraft under $JSBSIM_ROOT
# kind = "multirotor"        # optional; inferred from model (F450 → multirotor, else fixed-wing)
lat_deg = 37.6188
lon_deg = -122.3756
alt_ft = 5000.0
heading_deg = 90.0
airspeed_kts = 100.0         # a multirotor initialises at hover: airspeed_kts = 0.0
```

Bundled examples: `scenario.example.toml` (2-ship fleet), `scenario.demo.toml`
(bounded single fixed-wing for the demo), `scenario.quad.toml` (single F450 quad),
`scenario.smoke.toml` (fast, for tests).

**Vehicle classes.** Each aircraft is a fixed-wing or a multirotor (`VehicleKind`),
which decides its control law — the four `fcs/*-cmd-norm` inputs are surface
deflections on a fixed-wing but body-rate + collective demands on a multirotor.
`kind` is inferred from `model`; set it explicitly for a model the inference
doesn't know. A fleet can mix classes freely. A multirotor entity also needs its
stability-augmentation loop engaged (`fcs/ScasEngage = 1`); `spawn_fleet` does this
automatically for the multirotor class — without it roll/pitch/yaw commands are inert.

## 6. Tests — the convention that matters

**Tests live in each crate's `tests/` directory as integration tests — never
inline `#[cfg(test)] mod tests` in `src/`.** Rationale: `src/` stays production
code, and integration tests exercise the crate through its **public API** (the
same surface real callers use). If a test needs an item, that item must be `pub`.

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

**Running tests / lint / format** (in a container):
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings   # must be clean
cargo fmt --all                                          # format before committing
```

Two things to know:
- **Shared test fixtures** go in `tests/common/mod.rs` (a subdirectory with
  `mod.rs` is *not* itself a test binary). Put `#![allow(dead_code)]` at its top,
  since each test binary uses only some of the helpers. See
  `crates/astrum-sim/tests/common/mod.rs`.
- **The sim's tests drive the real app.** `astrum-sim` is a **lib + bin**: the
  logic is in `src/lib.rs` (`build_app(...)`), and `src/main.rs` is thin process
  wiring. Tests call `build_app(scenario)` and pump `app.update()` — the same
  schedule the binary runs. See `crates/astrum-sim/tests/simulation.rs`.

⚠️ **JSBSim gotcha:** JSBSim has process-global init state, so constructing two
flight models from two threads at once *segfaults*. Cargo runs tests in parallel,
so any test binary that builds a `Fdm`/aircraft must serialize construction with a
`static Mutex<()>` — copy the pattern from `crates/astrum-fdm/tests/fdm.rs`.

## 7. Running the whole stack (sim + controller)

The demos are one-command, self-terminating, and tear themselves down:

```bash
make demo          # C++ controller flies aircraft 1 in the bounded scenario
make demo-python   # same, with the Python controller
make demo-quad     # F450 quadrotor climbs + yaws under the built-in multirotor autopilot
```

**Watch it in 3-D.** `make replay` starts the fleet sim + the `astrum-web` bridge; open
http://localhost:8080 to see every aircraft on the ENU grid (orbit/follow cameras, trails,
playback transport). It works against a *live* sim or a bag being replayed by `astrum-bag-play`
(both PUB on the telemetry endpoint the bridge subscribes to). For an offline, service-free view,
bake a bag to JSON and drag it onto the page (or use "Open file…"):

```bash
make replay                                          # sim + renderer at http://localhost:8080
make export BAG=recordings/session.bag OUT=out.json  # bag → astrum-replay-bag JSON
make down                                            # stop the sim + renderer
```

The renderer is static files under `web/` (Three.js vendored in `web/vendor/`; the module contract
lives in `web/js/`); the bridge and exporter are the two binaries in `crates/astrum-web/`, both
projecting telemetry through the one `astrum_web::FrameJson` shape.

**To develop your own controller**, don't add a compose service — treat it as a
process you run in a dev shell against a running sim:

```bash
# 1. start just the sim (bounded scenario), detached
ASTRUM_SCENARIO=config/scenario.demo.toml docker compose up -d sim

# 2. open a shell on the same network (sim resolves by name)
make shell

# 3. build + run YOUR controller, pointing at the sim; rerun as you iterate
ASTRUM_TELEMETRY_CONNECT=tcp://sim:5555 ASTRUM_COMMAND_CONNECT=tcp://sim:5556 \
ASTRUM_TARGET_ID=1 <build-and-run-your-controller>

# 4. when done
docker compose down
```

Compose services (`controller`, `controller-python`) are just **canned demo
presets** — add one only when you want a controller to be a permanent part of the
topology. See `controllers/cpp/README.md` and `controllers/python/README.md` for
how a controller subscribes, decodes telemetry, and pushes commands.

## 8. Cap'n Proto wire quick reference

The schema (`crates/astrum-proto/schema/astrum.capnp`) defines three messages:
- **`Telemetry`** — sim → subscribers (topic = aircraft id as 4 big-endian bytes). Carries
  `vehicleKind` (fixed-wing | multirotor) so a subscriber knows the class without the scenario.
- **`Command`** — controllers → sim; a union of **`raw`** actuator setpoints
  (`fcs/*-cmd-norm`) *or* high-level **`autopilot`** targets.
- **`BagFrame`** — one recorded telemetry sample + timestamp (for record/replay).

Change the schema in **one** place; Rust regenerates on build, C++ regenerates via
its CMake step, and Python loads it at runtime (`capnp.load(...)`). Keeping one
schema is what guarantees the three languages stay wire-compatible.

## 9. Before you commit

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
All three must pass. That's the bar.
