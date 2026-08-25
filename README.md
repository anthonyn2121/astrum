# ASTRUM

A simulation environment for **multi-agent flight** — dozens to hundreds of independent
fixed-wing aircraft and sUAS quadrotors, each with its own high-fidelity flight dynamics.

- **Bevy + Rust** ECS as the authoritative ("know-it-all") state server (headless — no renderer).
- **JSBSim** 6-DOF flight dynamics embedded via Rust FFI; **each entity owns its own FDM instance**.
- Two **vehicle classes** — fixed-wing and multirotor (quadrotor) — share one wire protocol; the
  sim applies the right control law per aircraft (chosen by `VehicleKind`, inferred from the model).
- **Geodetic ↔ Cartesian (ENU)** coordinate mapping so global lat/lon/alt state lives on a local grid.
- **ZMQ** messaging backbone: the sim broadcasts telemetry and ingests flight-control commands.
- Flight controllers live **outside** the sim (C++ first, Python later) and talk only over ZMQ.
- **Cap'n Proto** wire format, shared across Rust / C++ / Python.

Telemetry can be recorded to a **bag** and replayed to any subscriber byte-for-byte (ROS-style),
and a browser-based **3-D replay renderer** (Three.js, procedural meshes) visualises the fleet
live or from an exported bag — `make replay`, then open http://localhost:8080.

## Status

Under active construction. Milestones:

| # | Milestone | State |
|---|---|---|
| M0 | Docker toolchain environment | **done** ✓ |
| M1 | JSBSim FFI (`astrum-fdm`) | **done** ✓ |
| M2 | Coordinate mapping (`astrum-geo`) | **done** ✓ |
| M3 | Bevy ECS core (`astrum-sim`) | **done** ✓ |
| M4 | Wire schema + ZMQ transport + bag replay (`astrum-proto`, `astrum-net`, `astrum-bag`) | **done** ✓ |
| M5 | C++ example flight controller (`controllers/cpp`) | **done** ✓ |
| M6 | Integration demo (`make demo`) + docs | **done** ✓ |
| M7 | Browser 3-D replay renderer (`astrum-web` + `web/`) | **done** ✓ |

The engine is end-to-end functional: `make demo` brings up the sim and a C++ controller that
flies an aircraft to a commanded heading/altitude over ZMQ (`make demo-python` runs the equivalent
Python controller). `make demo-quad` flies ASTRUM's second vehicle class — an **F450 quadrotor** that
climbs and yaws under the built-in multirotor hover-hold autopilot. `make replay` starts the fleet
alongside a browser-based **3-D replay renderer** that visualises every aircraft in real time (or
from an exported bag) on the ENU grid.

## Architecture & data flow

`astrum-sim` is the authoritative state server: a headless Bevy ECS where **each aircraft entity
owns its own JSBSim `FGFDMExec`**. Controllers and recorders are separate processes that only ever
touch it through ZMQ + Cap'n Proto — never shared memory — so the same wire bytes drive a live
controller, a bag recorder, or a replayed session identically.

```
   ZMQ PUB  telemetry (topic = aircraftId)                    ZMQ PUSH  commands (targetId)
   ┌──────────────────────────────► SUB ┐                    ┌ PUSH ◄──────────────────────┐
   │                                     │                    │                             │
   │                          controllers / recorder / replay │                    controllers (C++/Python)
   │                                                          ▼                             │
┌──┴──────────────────────────── astrum-sim (Bevy, headless) ─┴─────────────────────────────┴──┐
│  each Update tick, chained:                                                                  │
│    ingest_commands   PULL → per-entity: raw → fcs/*-cmd-norm | autopilot → AutopilotState    │
│    autopilot_step    engaged targets → fcs/*-cmd-norm (inner loop, per VehicleKind)          │
│    step_fdm          FGFDMExec::Run() once per entity (authoritative 6-DOF integration)      │
│    sync_geo_transform JSBSim geodetic state → GeoState + ENU Transform (world grid)          │
│    publish_telemetry  encode Telemetry → PUB                                                 │
│    report_progress / stop_at_max_time / pace                                                 │
│                                                                                              │
│  entity = { Aircraft{id}, FdmInstance(FGFDMExec), VehicleKind, GeoState, Transform, AutopilotState } │
└──────────────────────────────────────────────────────────────────────────────────────────── ┘
```

The controllers write **control inputs** (`fcs/*-cmd-norm`), never state; JSBSim computes all
motion and the ECS `Transform`/`GeoState` are a downstream mirror of its output (used for the world
grid + telemetry), never fed back into the physics.

### Coordinate conventions

Aircraft state is authored and integrated in **geodetic** WGS84 (JSBSim uses *geodetic* latitude,
`position/lat-geod-deg`, not geocentric). For the world model, `astrum-geo` maps geodetic ↔ ECEF ↔
**local-tangent ENU** about the scenario's `origin_*`: **x = East, y = North, z = Up** (metres), a
right-handed frame carried in Bevy's `Transform`. Telemetry carries both the geodetic position and
the ENU vector, plus NED velocity (x = North, y = East, z = Down) and Euler attitude (roll φ, pitch
θ, yaw/heading ψ, radians). Pick an origin near the fleet so ENU stays small and well-conditioned.

## Requirements

Only **Docker** (with Compose v2+) on the host. The entire toolchain — Rust, a source
build of JSBSim 1.3.1, C++/CMake, libzmq + cppzmq, Cap'n Proto, and Python — lives inside
the `astrum-dev` image. Nothing else is installed on the host.

## Quick start

```bash
make build      # build the dev image (compiles JSBSim from source — first run is slow)
make verify     # confirm the whole toolchain works (compilers, libs, JSBSim link + run)
make demo       # integration demo: C++ controller flies aircraft 1, then auto-tears-down
make demo-python # same demo, flown by the Python controller
make demo-quad  # quadrotor demo: an F450 climbs + yaws under the built-in multirotor autopilot
make shell      # drop into a one-off dev shell (SERVICE=sim|controller)
make up         # start the sim + controller services (detached)
make down       # stop them
make help       # list all targets
```

## Running the simulation

The headless ECS state server (`astrum-sim`) spawns one aircraft entity — each owning its own
JSBSim flight-dynamics instance — per `[[aircraft]]` entry in a scenario file:

```bash
make shell                              # into a dev container
cargo run -p astrum-sim                  # uses config/scenario.example.toml
ASTRUM_SCENARIO=config/my.toml cargo run -p astrum-sim
cargo test --workspace                  # astrum-fdm / astrum-geo / astrum-sim suites
```

Scenario files (see `config/scenario.example.toml`) set the WGS84 **ENU origin**, the fixed
physics `rate_hz`, an optional `max_time_s`, and a `realtime` flag. Physics advances **exactly
one deterministic step of `1/rate_hz` per loop**; `realtime = true` throttles that to wall-clock
(≈ real-time, for live controllers) while `false` runs as fast as the host allows — the
trajectory is identical either way. Each aircraft's authoritative geodetic state is read from
JSBSim and mapped to a Cartesian ENU `Transform` on the world grid.

Once running, the sim **publishes telemetry** and **ingests commands** over ZMQ (below), and a
flight controller flies one of the aircraft.

## Flight controllers

Controllers run **outside** the sim and speak only ZMQ + Cap'n Proto. The reference example is
`controllers/cpp` (cppzmq + capnp) — see its [README](controllers/cpp/README.md). It subscribes
to one aircraft's telemetry and either closes a heading/altitude/airspeed hold loop itself
(**raw** actuators — the intended usage, you own the control law) or hands a high-level target to
the sim's built-in inner loop (**autopilot**). Both fly a 90° turn to within ~4° while holding
altitude to a few metres.

`controllers/python` ([README](controllers/python/README.md)) is a line-for-line twin using
`pyzmq` + `pycapnp` (no codegen — it `capnp.load()`s the same `astrum.capnp` at runtime), proving
any language with ZMQ + Cap'n Proto bindings plugs in. It flies the identical trace.

**Vehicle classes.** The same four `fcs/*-cmd-norm` inputs mean different things per class: on a
fixed-wing they are control-surface deflections, on a **multirotor** they are body roll/pitch/yaw
**rate** demands plus collective throttle. The wire protocol is identical, but a control law is
not portable across classes — a fixed-wing raw law would mis-fly a quad. Either send **autopilot**
targets and let the sim run the matching inner loop (`make demo-quad` flies an F450 this way), or
send **raw** actuators with a law written for that class. ASTRUM picks the class per aircraft from
`VehicleKind` (see *Extending ASTRUM*).

```bash
# whole stack over the compose network (sim + controller by service name)
docker compose up sim controller

# or standalone in one dev shell against a bounded demo scenario:
ASTRUM_SCENARIO=config/scenario.demo.toml cargo run --release -p astrum-sim &   # aircraft 1 heads east
cmake -S controllers/cpp -B controllers/cpp/build && cmake --build controllers/cpp/build -j
ASTRUM_TARGET_HEADING_DEG=0 ./controllers/cpp/build/astrum-controller           # commands it north
# ...or the Python controller, no build step:
ASTRUM_TARGET_HEADING_DEG=0 python controllers/python/controller.py
```

The controller's target aircraft and setpoints are set via `ASTRUM_TARGET_*` env vars (defaults:
id 1, heading 0°/north, 1500 m, 50 m/s); `ASTRUM_CONTROL_MODE` selects `raw` (default) or `autopilot`.

## Wire protocol, telemetry & commands

All three payloads share one Cap'n Proto schema (`crates/astrum-proto/schema/astrum.capnp`), so
Rust, C++, and Python speak the same bytes:

- **Telemetry** — the sim `PUB`s one message per aircraft per tick: a 4-byte big-endian
  aircraft-id topic frame, then a `Telemetry` struct (sim time, geodetic + ENU pose, NED
  velocity, attitude, airspeed, health). Subscribers filter by id or take the whole fleet.
- **Commands** — controllers `PUSH`, the sim `PULL`s (fan-in) and drains them each tick. A
  `Command` targets one aircraft id and carries either **raw actuators**
  (`aileron`/`elevator`/`rudder` ∈ [-1,1], `throttle` ∈ [0,1] → `fcs/*-cmd-norm`) or
  **autopilot targets** (heading / altitude / airspeed), serviced by a built-in proportional
  inner loop. A raw command always overrides and disengages the autopilot.

| Purpose | Sim (binds) | Controller (connects) |
|---|---|---|
| Telemetry (PUB → SUB) | `tcp://0.0.0.0:5555` | `tcp://sim:5555` |
| Commands (PULL ← PUSH) | `tcp://0.0.0.0:5556` | `tcp://sim:5556` |

Endpoints are overridable via `$ASTRUM_TELEMETRY_BIND` / `$ASTRUM_COMMAND_BIND` (sim) and
`$ASTRUM_TELEMETRY_CONNECT` (subscribers).

The schema (`crates/astrum-proto/schema/astrum.capnp`) defines: `Telemetry` (`aircraftId`, `simTimeS`,
`geo` = `Geodetic` lat/lon/alt, `enu` = `Vec3` E/N/U, `velNed`, `attitude` = `Euler` φ/θ/ψ,
`airspeedMps`, `health`); `Command` (`targetId` + a `control` union of `RawActuators` |
`AutopilotTargets`); and `BagFrame` (`recvWallNs` + `Telemetry`) for recordings. Both languages
generate from this one file, so they are wire-compatible by construction. Framing is Cap'n Proto's
flat serialization (Rust `serialize::write_message`/`read_message` ↔ C++
`messageToFlatArray`/`FlatArrayMessageReader`).

## Recording & replay (bag)

Recording and replay are ROS-style: the recorder is just another `SUB` client and replay is
just another `PUB`, so a replayed bag reaches any subscriber (a controller, the 3-D renderer)
byte-for-byte identically to the live sim.

```bash
# record the live stream to a bag (Ctrl-C or set ASTRUM_BAG_SECONDS to stop)
ASTRUM_BAG=recordings/session.bag cargo run -p astrum-bag --bin astrum-bag-record

# replay it — binds the telemetry endpoint, preserving recorded inter-frame timing
ASTRUM_BAG=recordings/session.bag cargo run -p astrum-bag --bin astrum-bag-play

# under compose: record the sim's stream to recordings/session.bag
docker compose --profile record up recorder
```

`ASTRUM_REPLAY_RATE` scales playback speed (default `1.0`); `ASTRUM_BAG_REPLAY_BIND` overrides
the replay endpoint (defaults to the live telemetry endpoint, so replay is a drop-in).

## 3-D replay renderer (`astrum-web` + `web/`)

A browser-based renderer visualises the fleet on the ENU grid — procedurally-generated Cessna and
F450 meshes (no external assets), tinted per vehicle class and per id, oriented from the telemetry
Euler angles, with motion trails, orbit + follow cameras, a playback transport (play/pause/scrub/
speed) and per-aircraft readouts. It's self-contained: Three.js is vendored under `web/vendor/`,
so nothing loads from the network at runtime.

There are two data paths, both feeding one shared JSON frame shape (`astrum_web::FrameJson`), so the
live stream and an offline export can't drift:

```bash
# LIVE / replay — start the fleet + the bridge, then open http://localhost:8080
make replay          # docker compose up -d sim replay
# (the bridge also serves a bag being replayed by astrum-bag-play — same endpoint)

# OFFLINE — bake a bag into a single JSON the renderer opens with "Open file…" (no services)
make export BAG=recordings/session.bag OUT=recordings/session.json
```

`astrum-web` is one synchronous process (no async runtime): a background ZMQ `SUB` fans each
telemetry sample out to every connected browser over WebSocket (`GET /telemetry`), while the same
port serves the static `web/` app. Config via env: `ASTRUM_TELEMETRY_CONNECT` (upstream PUB,
default `tcp://127.0.0.1:5555`), `ASTRUM_WEB_BIND` (default `0.0.0.0:8080`), `ASTRUM_WEB_ROOT`
(default `web`). `astrum-web-export` reads a bag and writes a `astrum-replay-bag` document
(`$ASTRUM_BAG` → `$ASTRUM_WEB_EXPORT`).

## Extending ASTRUM

**Add an aircraft** — append a `[[aircraft]]` block to your scenario TOML with a unique `id`
(the ZMQ topic / command target), a `model` present in `$JSBSIM_ROOT/aircraft`, and initial
conditions (`lat_deg`, `lon_deg`, `alt_ft`, `heading_deg`, `airspeed_kts`). `spawn_fleet` creates
one ECS entity — with its own `FGFDMExec` — per entry at startup; nothing else needs to change.
The **vehicle class** is inferred from `model` (`F450` → multirotor, else fixed-wing); set `kind =
"fixed-wing"` / `"multirotor"` explicitly for a model ASTRUM doesn't recognise. A scenario can mix
classes freely — each entity carries its own `VehicleKind` and gets the matching control law.
`config/scenario.quad.toml` is a worked single-quad example.

**Add a controller** — in any language with ZMQ + Cap'n Proto bindings (the image ships C++
cppzmq/capnp and Python `pyzmq`/`pycapnp`):

1. Generate bindings from `crates/astrum-proto/schema/astrum.capnp`.
2. `SUB` to `tcp://sim:5555`, subscribing to the 4-byte big-endian topic of the aircraft id you
   want (or subscribe to everything and filter).
3. Decode each `Telemetry`, compute your control, and `PUSH` a `Command` (same `targetId`) to
   `tcp://sim:5556` — either `RawActuators` (your own law) or `AutopilotTargets` (sim inner loop).

`controllers/cpp` is the worked reference; mirror its structure. New controllers can run as extra
`docker-compose` services on the `astrum` network or as any external process reaching the sim's
endpoints.

## Layout

```
docker/            Dockerfile, entrypoint, toolchain smoke checks
scripts/           helper scripts (toolchain verification)
docker-compose.yml sim + controller + recorder services
Makefile           dev workflow entrypoints (build / demo / up / shell / verify)
crates/            Rust workspace — astrum-fdm/geo/proto/net/bag/sim
controllers/       out-of-sim flight controllers — cpp/ (built), python/ (later)
config/            scenarios — scenario.example.toml (fleet), scenario.demo.toml (fixed-wing demo), scenario.quad.toml (quadrotor demo)
recordings/        telemetry bags written by astrum-bag-record (git-ignored)
```

JSBSim inside the image: library/headers/pkg-config under `/opt/jsbsim`, aircraft data at
`$JSBSIM_ROOT` (`/opt/jsbsim/share/jsbsim`).
