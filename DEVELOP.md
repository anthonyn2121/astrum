# Developing on ASTRUM

A practical guide for getting productive in this repo. For the *why* behind the
design, see [README.md](README.md); this doc is the *how*.

## 1. The mental model (read this first)

ASTRUM is **one server + many clients** talking over Zenoh.

- **The sim** (`astrum-sim`) is the authoritative state server. It's a headless
  [Bevy](https://bevyengine.org/) ECS: one *entity* per aircraft, each owning its
  own [JSBSim](https://github.com/JSBSim-Team/jsbsim) flight-dynamics model. Every
  tick it **publishes** telemetry (`astrum/telemetry/<id>`) and drains **subscribed**
  commands (`astrum/command/**`).
- **Controllers** (`controllers/*`) are separate processes — the flight-control
  logic. They **subscribe** to telemetry and **publish** commands. They never touch
  the sim's memory; only bytes over Zenoh.
- **The wire** is [Cap'n Proto](https://capnproto.org/), one schema
  (`crates/astrum-proto/schema/astrum.capnp`) shared by Rust, C++, and Python.

```
controllers ──put astrum/command/<id>─────►  sim (subscribes astrum/command/**)
controllers ◄──put astrum/telemetry/<id>───  sim (publishes per aircraft; id in the key)
```

Both key spaces ride **one** Zenoh session on a single endpoint (`tcp/…:7447`) — the sim (and bag
replay) `mode=peer` **listen** there, everyone else `mode=peer` **connects** to it. Peer mode makes
a controller's `open()` non-fatal if the sim isn't up yet: it returns at once and Zenoh retries the
connection in the background, so a controller started alongside the sim just attaches once the sim
finishes its ~10 s JSBSim load (a Zenoh `client` would instead abort `open()` on that cold start).

**`astrum-race`** (§5b) — the drone-racing environment, kept a separate app so a racing change can't
disturb the WVR combat environment — speaks the *same* wire on the same two key spaces, plus two of
its own for the racing task:

```
controllers ◄──put astrum/race/course/<id>──  astrum-race (the gates, republished ~1 Hz)
controllers ◄──put astrum/race/state/<id>───  astrum-race (gate progress, every tick)
```

It publishes `Telemetry` and takes `Command` unchanged, because a quadrotor's four normalized
channels are the same four channels — only the *task* state needed new messages (§8). It listens on
the same default endpoint as the sim, so run one or the other, or give one `--listen` a different
port. Everything else in this section is about the sim.

Each entity is addressed by its id *in the key* — not a socket or a port — so **one endpoint serves
the whole fleet no matter how many aircraft you add** (there are no ports to manage per entity), and
any client can subscribe to a single aircraft (`astrum/telemetry/<id>`) or the entire fleet
(`astrum/telemetry/*`). That makes cross-entity state sharing free: a controller can watch its
neighbours' telemetry as easily as its own, with no extra wiring. The sim is the one authoritative
hub (a star: it `listen`s, everyone else `connect`s), and telemetry is best-effort streaming, not an
on-demand query — Zenoh's mesh routing and queryable/storage are available but unused today.

## 2. The golden rule: everything runs in Docker

Nothing (Rust, JSBSim, Zenoh, Cap'n Proto) is installed on the host — it all lives
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

## Quick start

The fast path from a fresh checkout to a running sim. Every command below runs
**inside the dev container** (§2) — nothing is built on your host. Do the one-time
image build, then open a container shell and run steps 1–3 inside it:

```bash
make build      # one time — builds the astrum-dev image (compiles JSBSim; slow first run)
make shell      # open a shell in the container; run the steps below there
```

**1. Build the Rust code** (the sim + every crate in the workspace):

```bash
cargo build --workspace              # debug build of every crate
cargo build --release --workspace    # optimized build (what the demos run)
```

**2. Build the C++ code** (the example flight controller, built with CMake):

```bash
cmake -S controllers/cpp -B controllers/cpp/build -DCMAKE_BUILD_TYPE=Release   # configure
cmake --build controllers/cpp/build -j                                         # compile
# → produces the binary ./controllers/cpp/build/astrum-controller
```

**3. Run a scenario from a config** (a `config/*.toml` fleet definition):

```bash
cargo run -p astrum-sim                                                    # default: config/scenario.example.toml
cargo run --release -p astrum-sim -- config/scenario.demo.toml             # pick a specific scenario (positional arg)
cargo run -p astrum-sim -- --help                                          # list every flag + its env fallback
```

The scenario is a plain positional argument; every binary also still honours its
old `ASTRUM_*` env var as a fallback (so `docker compose` and `make` keep working
unchanged). Precedence is **CLI arg > `$ENV` > default** — see `--help`.

The sim loads the fleet from the TOML, spins up one JSBSim model per aircraft
(the first load takes ~10 s), then publishes telemetry over Zenoh until
`max_time_s` is reached (or you Ctrl-C). Scenario format + the bundled examples
are in §5.

Prefer not to keep a shell open? Run any single command in a throwaway container
instead of `make shell`:

```bash
docker compose run --rm --no-deps sim bash -lc 'cargo build --workspace'
```

To see everything working together — sim + controller + the 3-D browser view —
the demos do it in one command: `make demo` and `make replay` (§7).

## 3. Repo layout — where files go

```
crates/            # Rust workspace (one crate per responsibility)
  astrum-fdm/       #   JSBSim FFI: safe `Fdm` wrapper over the C++ engine
  astrum-geo/       #   WGS84 geodetic ↔ ECEF ↔ ENU coordinate math
  astrum-proto/     #   the Cap'n Proto schema + encode/decode helpers
  astrum-net/       #   Zenoh transport (telemetry + command key exprs, one session)
  astrum-bag/       #   record/replay telemetry to a file ("bag")
  astrum-web/       #   WebSocket+HTTP bridge (astrum-web) + bag→JSON exporter (astrum-web-export)
  astrum-sim/       #   the Bevy ECS app — the sim binary (fleet + WVR gun combat)
  astrum-race/      #   the drone-racing environment: a SEPARATE Bevy app (F450 + gates)
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
| A new scenario | `config/<name>.toml` | run with `cargo run -p astrum-sim -- config/<name>.toml` |
| A test | `crates/<crate>/tests/<name>.rs` | see §6 |

## 4. Build & run the sim

Inside a shell (`make shell`) or via `docker compose run`:

```bash
cargo build --workspace                                          # compile everything
cargo run -p astrum-sim                                          # run the sim (uses config/scenario.example.toml)
cargo run --release -p astrum-sim -- config/scenario.demo.toml   # pick a scenario (positional)
cargo run -p astrum-sim -- --listen tcp/0.0.0.0:7447 config/scenario.demo.toml
```

The sim's flags (run `astrum-sim --help` to see them, each with its `[env: …]`
fallback shown). Every flag falls back to the env var, then the default:
| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| *(positional)* `scenario` | `ASTRUM_SCENARIO` | `config/scenario.example.toml` | which scenario to load |
| `--listen` | `ASTRUM_ZENOH_LISTEN` | `tcp/0.0.0.0:7447` | Zenoh endpoint the sim listens on (telemetry + commands) |

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
`scenario.formation.toml` (4-ship: a figure-eight leader + 3 MPC followers),
`scenario.smoke.toml` (fast, for tests).

**Vehicle classes.** Each aircraft is a fixed-wing or a multirotor (`VehicleKind`),
which decides its control law — the four `fcs/*-cmd-norm` inputs are surface
deflections on a fixed-wing but body-rate + collective demands on a multirotor.
`kind` is inferred from `model`; set it explicitly for a model the inference
doesn't know. A fleet can mix classes freely. A multirotor entity also needs its
stability-augmentation loop engaged (`fcs/ScasEngage = 1`); `spawn_fleet` does this
automatically for the multirotor class — without it roll/pitch/yaw commands are inert.

## 5b. The drone-racing environment (`astrum-race`)

A **second, separate** Bevy app: a DJI F450 and a course of racing gates randomized
per seed. It is not a mode of `astrum-sim` — that crate hosts the WVR gun-combat
environment, and the two share only the task-neutral crates: `astrum-fdm` (JSBSim),
`astrum-geo` (coordinates) and `astrum-proto`/`astrum-net` (the wire). Design notes:
`docs/racing-environment-design.md`.

```bash
make race-course SEED=42 GATES=10       # print the gates a seed draws, without flying
make race SEED=42                       # fly it, publishing on tcp/…:7447
make race LISTEN=tcp/0.0.0.0:7448       # …on another endpoint (a sim already has 7447)
make race-offline SEED=42               # physics only: no transport, no port bound
cargo run -p astrum-race -- --help      # the flags, inside a shell
```

Its scenario is `config/race.toml` — a different schema from a fleet scenario:
`[world]` / `[sim]` as above, plus a single `[drone]` and a `[course]` table that is
the **randomization envelope** (gate count, leg lengths, turn and climb limits,
clearances, flight volume). Every `[course]` field has a default, so an absent table
is still a valid course. Course generation is fallible before the app is built, so an
unsatisfiable envelope is a clear error rather than an app that spins.

A run ends in exactly one way: something writes an `Outcome` onto the `RaceProgress`
resource and `stop_when_finished` exits the app. The outcomes are `Complete` (every gate,
in order — the only way to win), `CrashedGround`, `CrashedGate`, `TimedOut` and
`SpawnFailed`, and the one that ended the run goes out on `astrum/race/state/<id>` so an
off-process controller learns *why* it stopped rather than just that telemetry ceased.

Crashing is real, not cosmetic. The F450's hull is built from its own JSBSim model files
(four 9.4″ rotor discs on 0.1651 m arms over a core box from the gear feet to the rotor
plane), and it has **10.92 cm of clearance per side** in the 31″ opening — 4.08 cm at the
worst heading, where the drone presents a corner and reaches 0.35287 m rather than
0.28448 m.

The opening is 31″ and not the 25″ of the original brief. 25″ passed the airframe only
within **±13° of the gate's normal**, while a gate's normal bisects a turn of up to 120°
and so can be crossed 60° off — the geometry and the envelope contradicted each other.
27.78″ is the exact size at which heading stops mattering; 31″ is the first whole inch past
it. §2 of the design doc has the crab table and the before/after clearances.

Two numbers to know before writing a controller. The **lateral corridor is ±0.109 m inside
a ±0.394 m hole**: the scorer credits a pass on where the CG pierced the plane, so a
crossing it would credit can still be a crash — that ordering is deliberate. And the
vertical **headroom is 0.249 m**, not the 0.394 m the opening looks like it offers, because
what reaches the top bar is a blade tip at the rotor plane. §4b of the design doc has the
derivation and why the hull is five parts rather than one box.

The drone launches holding station on gate 0's approach. **Nothing flies the course
yet**, but both seams for a controller that will are in place:

* **In-process** — write the `FlightCommand` component (target altitude, heading,
  horizontal velocity); see §7 of the design doc.
* **Off-process** — subscribe to `astrum/telemetry/<id>` for the drone's state and
  `astrum/race/{course,state}/<id>` for the track and gate progress, then publish
  `Command` on `astrum/command/<id>`: `Control::Raw` to fly the four channels
  yourself, `Control::Autopilot` to drive the inner loop. Raw control here is a
  **lease** — unrefreshed for 0.5 s it lapses and the inner loop resumes station
  keeping, because a quadrotor with a frozen collective and a dead controller falls.

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

The C++ and Python **controllers** have their own formatter + linter (config at
`.clang-format` and `ruff.toml`), run from the host via make:
```bash
make fmt     # clang-format -i (C++) + ruff format + ruff check --fix (Python)
make lint    # ruff check + ruff format --check (Python) + clang-format --dry-run --Werror (C++)
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
make demo           # C++ controller flies aircraft 1 in the bounded scenario
make demo-python    # same, with the Python controller
make demo-quad      # F450 quadrotor climbs + yaws under the built-in multirotor autopilot
make demo-formation # leader flies a figure-eight; 3 F-16s MPC-track their slots (decentralized)
```

**Watch it in 3-D.** `make replay` starts the fleet sim + the `astrum-web` bridge; open
http://localhost:8080 to see every aircraft on the ENU grid (orbit/follow cameras, trails,
playback transport). It works against a *live* sim or a bag being replayed by `astrum-bag-play`
(both publish on the telemetry endpoint the bridge subscribes to). For an offline, service-free view,
bake a bag to JSON and drag it onto the page (or use "Open file…"):

```bash
make replay                                          # sim + renderer at http://localhost:8080
make export BAG=recordings/session.bag OUT=out.json  # bag → astrum-replay-bag JSON
make down                                            # stop the sim + renderer
```

The renderer is static files under `web/` (Three.js vendored in `web/vendor/`; the module contract
lives in `web/js/`); the bridge and exporter are the two binaries in `crates/astrum-web/`, both
projecting telemetry through the one `astrum_web::FrameJson` shape.

**Ground map.** For **fixed-wing** fleets — which fly a real geographic route — the renderer lays a
slippy-map (XYZ / Web-Mercator) aerial layer flat under the aircraft, centred on the scenario's ENU
datum. Multirotor and idle views keep the plain ENU grid. Tiles are served by `astrum-web` at
`/tiles/{z}/{x}/{y}.jpg`: it proxies an upstream tile server and disk-caches every tile under
`tilecache/` (git-ignored). So any datum gets crisp imagery with zero manual bundling, and an area
viewed once while online works **offline** afterward; a tile that 404s (offline + uncached) is
skipped and the grid shows through. The upstream is configurable:

```bash
# Default: Esri World Imagery (global aerial, no key). Point elsewhere with:
ASTRUM_TILE_URL='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}' make replay  # public-domain, US-only
ASTRUM_TILE_URL='' make replay        # cache-only: serve tiles already cached, never fetch
ASTRUM_TILE_CACHE=/path/to/tiles      # relocate the on-disk cache (default ./tilecache)
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
./controllers/cpp/build/astrum-controller --connect tcp/sim:7447 --id 1

# 4. when done
docker compose down
```

Compose services (`controller`, `controller-python`) are just **canned demo
presets** — add one only when you want a controller to be a permanent part of the
topology. See `controllers/cpp/README.md` and `controllers/python/README.md` for
how a controller subscribes, decodes telemetry, and publishes commands.

## 8. Cap'n Proto wire quick reference

The schema (`crates/astrum-proto/schema/astrum.capnp`) defines these messages:
- **`Telemetry`** — sim → subscribers (published on `astrum/telemetry/<id>`; the id is in the key,
  no topic frame). Carries `vehicleKind` (fixed-wing | multirotor) so a subscriber knows the class
  without the scenario.
- **`Command`** — controllers → sim; a union of **`raw`** actuator setpoints
  (`fcs/*-cmd-norm`) *or* high-level **`autopilot`** targets.
- **`BagFrame`** — one recorded telemetry sample + timestamp (for record/replay).
- **`RaceCourse`** / **`RaceState`** — `astrum-race` → subscribers, on
  `astrum/race/course/<id>` and `astrum/race/state/<id>`: the gate list (centre, normal heading,
  hardware dimensions) and the run's progress, including how it ended (`finished` + `outcome`).
  These are separate top-level messages rather than fields on `Telemetry`, which means "one
  aircraft's authoritative state" and would otherwise carry a dead course column in every combat
  recording and replay frame.

Two properties of the racing pair are worth knowing before you write a subscriber. Every progress
figure is a **cumulative counter, not an event** (`passed`, `missed`, `pendingGate`), so a subscriber
that skips messages loses nothing — which is why both channels publish with `CongestionControl::Drop`
and why a read returns the *latest* sample and discards the backlog. And the course is **republished
~1 Hz** even though it never changes, because Zenoh pub/sub has no late-joiner replay and the normal
case is a controller started after the environment. An `outcome` your build does not recognise decodes
as `running`, so an old controller against a new environment degrades rather than misreads.

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
