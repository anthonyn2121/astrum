# ASTRUM

A simulation environment for **multi-agent flight** — dozens to hundreds of independent
fixed-wing aircraft and sUAS quadrotors, each with its own high-fidelity flight dynamics.

- **Bevy + Rust** ECS as the authoritative ("know-it-all") state server (headless — no renderer).
- **JSBSim** 6-DOF flight dynamics embedded via Rust FFI; **each entity owns its own FDM instance**.
- Two **vehicle classes** — fixed-wing and multirotor (quadrotor) — share one wire protocol; the
  sim applies the right control law per aircraft (chosen by `VehicleKind`, inferred from the model).
- **Geodetic ↔ Cartesian (ENU)** coordinate mapping so global lat/lon/alt state lives on a local grid.
- **Zenoh** messaging backbone: the sim broadcasts telemetry and ingests flight-control commands.
- Flight controllers live **outside** the sim (C++ first, Python later) and talk only over Zenoh.
- **Cap'n Proto** wire format, shared across Rust / C++ / Python.

Telemetry can be recorded to a **bag** and replayed to any subscriber byte-for-byte (ROS-style),
and a browser-based **3-D replay renderer** (Three.js, procedural meshes) visualises the fleet
live or from an exported bag — `make replay`, then open http://localhost:8080.

## Status

Everything below is built and working; here's what lives where:

| Component | Responsibility |
|---|---|
| `astrum-fdm` | JSBSim 6-DOF flight dynamics via a thin Rust FFI shim |
| `astrum-geo` | WGS84 geodetic ↔ ECEF ↔ ENU coordinate mapping |
| `astrum-sim` | headless Bevy ECS state server — one JSBSim FDM per entity |
| `astrum-proto` + `astrum-net` | Cap'n Proto wire schema + Zenoh transport (telemetry + commands) |
| `astrum-bag` | record / replay telemetry to a bag |
| `controllers/cpp`, `controllers/python` | out-of-sim example flight controllers |
| `astrum-web` + `web/` | browser 3-D replay renderer |

The engine is end-to-end functional: `make demo` brings up the sim and a C++ controller that
flies an aircraft to a commanded heading/altitude over Zenoh (`make demo-python` runs the equivalent
Python controller). `make demo-quad` flies ASTRUM's second vehicle class — an **F450 quadrotor** that
climbs and yaws under the built-in multirotor hover-hold autopilot. `make replay` starts the fleet
alongside a browser-based **3-D replay renderer** that visualises every aircraft in real time (or
from an exported bag) on the ENU grid.

## Architecture & data flow

`astrum-sim` is the authoritative state server: a headless Bevy ECS where **each aircraft entity
owns its own JSBSim `FGFDMExec`**. Controllers and recorders are separate processes that only ever
touch it through Zenoh + Cap'n Proto — never shared memory — so the same wire bytes drive a live
controller, a bag recorder, or a replayed session identically. Telemetry and commands are two sets
of key expressions multiplexed over **one** Zenoh session (a single TCP endpoint), not separate ports.

```
   Zenoh  telemetry (put astrum/telemetry/<id>)              Zenoh  commands (put astrum/command/<id>)
   ┌──────────────────────────────► SUB ┐                    ┌ SUB ◄───────────────────────┐
   │                                     │                    │                             │
   │                          controllers / recorder / replay │                    controllers (C++/Python)
   │                                                          ▼                             │
┌──┴──────────────────────────── astrum-sim (Bevy, headless) ─┴─────────────────────────────┴──┐
│  each Update tick, chained:                                                                  │
│    ingest_commands   sub astrum/command/** → raw → fcs/*-cmd-norm | autopilot → AutopilotState│
│    autopilot_step    engaged targets → fcs/*-cmd-norm (inner loop, per VehicleKind)          │
│    step_fdm          FGFDMExec::Run() once per entity (authoritative 6-DOF integration)      │
│    sync_geo_transform JSBSim geodetic state → GeoState + ENU Transform (world grid)          │
│    publish_telemetry  encode Telemetry → put astrum/telemetry/<id>                           │
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
build of JSBSim 1.3.1, C++/CMake, Zenoh (libzenohc + zenoh-cpp), Cap'n Proto, and Python — lives
inside the `astrum-dev` image. Nothing else is installed on the host.

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
cargo run -p astrum-sim -- config/my.toml   # scenario is a positional argument
cargo run -p astrum-sim -- --help        # every flag, each with its [env: …] fallback
cargo test --workspace                  # astrum-fdm / astrum-geo / astrum-sim suites
```

Every binary takes its config as normal **command-line arguments**; each flag
also falls back to its old `ASTRUM_*` env var, then a default, so precedence is
**CLI arg > `$ENV` > default** and `docker compose` / `make` keep working
unchanged. Run any binary with `--help` to see its flags and env fallbacks.

Scenario files (see `config/scenario.example.toml`) set the WGS84 **ENU origin**, the fixed
physics `rate_hz`, an optional `max_time_s`, and a `realtime` flag. Physics advances **exactly
one deterministic step of `1/rate_hz` per loop**; `realtime = true` throttles that to wall-clock
(≈ real-time, for live controllers) while `false` runs as fast as the host allows — the
trajectory is identical either way. Each aircraft's authoritative geodetic state is read from
JSBSim and mapped to a Cartesian ENU `Transform` on the world grid.

Once running, the sim **publishes telemetry** and **ingests commands** over Zenoh (below), and a
flight controller flies one of the aircraft.

## Flight controllers

Controllers run **outside** the sim and speak only Zenoh + Cap'n Proto. The reference example is
`controllers/cpp` (zenoh-cpp + capnp) — see its [README](controllers/cpp/README.md). It subscribes
to one aircraft's telemetry and either closes a heading/altitude/airspeed hold loop itself
(**raw** actuators — the intended usage, you own the control law) or hands a high-level target to
the sim's built-in inner loop (**autopilot**). Both fly a 90° turn to within ~4° while holding
altitude to a few metres.

`controllers/python` ([README](controllers/python/README.md)) is a line-for-line twin using
`eclipse-zenoh` + `pycapnp` (no codegen — it `capnp.load()`s the same `astrum.capnp` at runtime),
proving any language with Zenoh + Cap'n Proto bindings plugs in. It flies the identical trace.

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
cargo run --release -p astrum-sim -- config/scenario.demo.toml &     # aircraft 1 heads east
cmake -S controllers/cpp -B controllers/cpp/build && cmake --build controllers/cpp/build -j
./controllers/cpp/build/astrum-controller --heading 0               # commands it north
# ...or the Python controller, no build step:
python controllers/python/controller.py --heading 0
```

The controller's target aircraft and setpoints are `--id` / `--heading` / `--alt` / `--speed`
flags (defaults: id 1, heading 0°/north, 1500 m, 50 m/s) and `--mode raw` (default) or
`--mode autopilot`; run `controller --help` for the full list. Each flag also honours its old
`ASTRUM_TARGET_*` / `ASTRUM_CONTROL_MODE` env var as a fallback, so the compose services are
unchanged.

## Wire protocol, telemetry & commands

All three payloads share one Cap'n Proto schema (`crates/astrum-proto/schema/astrum.capnp`), so
Rust, C++, and Python speak the same bytes:

- **Telemetry** — the sim publishes one `Telemetry` message per aircraft per tick on the key
  `astrum/telemetry/<id>` (sim time, geodetic + ENU pose, NED velocity, attitude, airspeed,
  health). The aircraft id lives in the key — no topic frame — so subscribers take one aircraft
  (`astrum/telemetry/<id>`) or the whole fleet (`astrum/telemetry/*`); key matching is exact, so
  `.../1` never matches `.../10`. Telemetry uses `CongestionControl::Drop` (best-effort — a dropped
  frame is non-fatal, the subscriber just renders the next state).
- **Commands** — controllers publish on `astrum/command/<id>`; the sim subscribes
  `astrum/command/**` (fan-in from every controller) and drains them each tick. A `Command`
  targets one aircraft id and carries either **raw actuators** (`aileron`/`elevator`/`rudder` ∈
  [-1,1], `throttle` ∈ [0,1] → `fcs/*-cmd-norm`) or **autopilot targets** (heading / altitude /
  airspeed), serviced by a built-in proportional inner loop. Commands use `CongestionControl::Block`
  (reliable — a control input is never silently dropped). A raw command always overrides and
  disengages the autopilot.

Both key spaces are multiplexed over **one** Zenoh session on a single TCP endpoint (port 7447),
replacing ZMQ's two ports. The sim (and bag replay) `mode=peer` **listen** on that endpoint; every
other process (`mode=peer`, connect-out only) **connects** to it:

| Purpose | Sim (listens) | Controller (connects) |
|---|---|---|
| Telemetry + commands (one Zenoh session) | `tcp/0.0.0.0:7447` | `tcp/sim:7447` |

Endpoints are overridable via `$ASTRUM_ZENOH_LISTEN` (sim / replay) and `$ASTRUM_ZENOH_CONNECT`
(controllers / recorder / web). Zenoh endpoint syntax is `proto/host:port` (a single `/`, not `://`).

**One endpoint, key-addressed entities.** Every aircraft is addressed by its id *in the key
expression*, not by a socket or a port, so the transport needs **no per-entity configuration** —
there are no ports to allocate or manage as the fleet grows. Adding aircraft to a scenario changes
nothing about the endpoints: the sim simply publishes on `astrum/telemetry/<newid>` and any
`astrum/telemetry/*` subscriber picks it up automatically. This also makes **fleet-wide /
entity-to-entity state sharing** a one-line subscribe: any process — a controller, a monitor, a
swarm coordinator — takes one aircraft (`astrum/telemetry/<id>`) or the whole fleet
(`astrum/telemetry/*`) and receives every entity's state with no new socket, port, or handshake, so
a per-aircraft controller can react to its neighbours' telemetry as easily as its own. State flows
through the sim, which stays the single authoritative hub — a **star** topology (the sim `listen`s,
everyone else `connect`s), not a peer mesh, and telemetry is best-effort streaming re-sent every
tick rather than an on-demand query. Zenoh's peer-mesh routing and queryable/storage would enable
direct entity-to-entity links or pull-the-latest-state semantics, but ASTRUM does not use them today.

The schema (`crates/astrum-proto/schema/astrum.capnp`) defines: `Telemetry` (`aircraftId`, `simTimeS`,
`geo` = `Geodetic` lat/lon/alt, `enu` = `Vec3` E/N/U, `velNed`, `attitude` = `Euler` φ/θ/ψ,
`airspeedMps`, `health`); `Command` (`targetId` + a `control` union of `RawActuators` |
`AutopilotTargets`); and `BagFrame` (`recvWallNs` + `Telemetry`) for recordings. Both languages
generate from this one file, so they are wire-compatible by construction. Framing is Cap'n Proto's
flat serialization (Rust `serialize::write_message`/`read_message` ↔ C++
`messageToFlatArray`/`FlatArrayMessageReader`).

## Recording & replay (bag)

Recording and replay are ROS-style: the recorder is just another subscriber and replay is
just another publisher, so a replayed bag reaches any subscriber (a controller, the 3-D renderer)
byte-for-byte identically to the live sim.

```bash
# record the live stream to a bag (Ctrl-C, or --seconds N, to stop)
cargo run -p astrum-bag --bin astrum-bag-record -- recordings/session.bag

# replay it — listens on the telemetry endpoint, preserving recorded inter-frame timing
cargo run -p astrum-bag --bin astrum-bag-play -- recordings/session.bag

# under compose: record the sim's stream to recordings/session.bag
docker compose --profile record up recorder
```

The bag path is a positional argument; `--rate` scales playback speed (default `1.0`), `--listen`
overrides the endpoint replay listens on (defaults to the live sim endpoint, so replay is a drop-in
for any subscriber), and `--warmup-ms` pauses after opening so subscribers can attach first. As
everywhere, each flag falls back to its old env var (`ASTRUM_BAG`, `ASTRUM_REPLAY_RATE`,
`ASTRUM_ZENOH_LISTEN`, `ASTRUM_REPLAY_WARMUP_MS`) — run with `--help` to see them.

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

`astrum-web` is one synchronous process (no async runtime): a background Zenoh subscriber fans each
telemetry sample out to every connected browser over WebSocket (`GET /telemetry`), while the same
port serves the static `web/` app. Flags: `--connect` (upstream sim/replay endpoint, default
`tcp/127.0.0.1:7447`), `--bind` (default `0.0.0.0:8080`), `--root` (default `web`) — each with its
old `ASTRUM_WEB_*` / `ASTRUM_ZENOH_CONNECT` env fallback. `astrum-web-export` takes the bag as a
positional argument and writes a `astrum-replay-bag` document to `--out` (defaults to `<bag>.json`;
env fallbacks `ASTRUM_BAG` / `ASTRUM_WEB_EXPORT`).

## Extending ASTRUM

**Add an aircraft** — append a `[[aircraft]]` block to your scenario TOML with a unique `id`
(the Zenoh telemetry key / command target), a `model` present in `$JSBSIM_ROOT/aircraft`, and initial
conditions (`lat_deg`, `lon_deg`, `alt_ft`, `heading_deg`, `airspeed_kts`). `spawn_fleet` creates
one ECS entity — with its own `FGFDMExec` — per entry at startup; nothing else needs to change.
The **vehicle class** is inferred from `model` (`F450` → multirotor, else fixed-wing); set `kind =
"fixed-wing"` / `"multirotor"` explicitly for a model ASTRUM doesn't recognise. A scenario can mix
classes freely — each entity carries its own `VehicleKind` and gets the matching control law.
`config/scenario.quad.toml` is a worked single-quad example.

**Add a controller** — in any language with Zenoh + Cap'n Proto bindings (the image ships C++
zenoh-cpp/capnp and Python `eclipse-zenoh`/`pycapnp`):

1. Generate bindings from `crates/astrum-proto/schema/astrum.capnp`.
2. Open a `peer`-mode Zenoh session connecting to `tcp/sim:7447` and subscribe to
   `astrum/telemetry/<id>` for the aircraft you want (or `astrum/telemetry/*` for the whole fleet).
3. Decode each `Telemetry`, compute your control, and `put` a `Command` (same `targetId`) on
   `astrum/command/<id>` — either `RawActuators` (your own law) or `AutopilotTargets` (sim inner loop).

`controllers/cpp` is the worked reference; mirror its structure. New controllers can run as extra
`docker-compose` services on the `astrum` network or as any external process reaching the sim's
endpoint.

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
