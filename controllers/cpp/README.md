# FORGE example flight controller (C++)

A minimal out-of-sim flight controller that talks to `forge-sim` **only over ZMQ**,
using the shared Cap'n Proto schema (`crates/forge-proto/schema/forge.capnp`) for the
wire format. It is the reference for how a real controller (C++ or, later, Python)
plugs into FORGE:

- **SUB** `tcp://sim:5555` — telemetry for one aircraft (topic = 4-byte big-endian id).
- **PUSH** `tcp://sim:5556` — commands (target id + raw actuators *or* autopilot targets).

## Modes (`FORGE_CONTROL_MODE`)

- **`raw`** (default) — the controller closes a heading / altitude / airspeed hold loop
  *itself* and sends raw `fcs/*-cmd-norm` deflections every telemetry tick. This is
  FORGE's intended usage: **you own the control law**.
- **`autopilot`** — the controller sends a single high-level target and then just
  monitors; the sim's built-in proportional inner loop flies the aircraft.

Both converge a 90° turn to within ~4° while holding altitude to a few metres (see
the traces in the repo's verification notes).

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_TELEMETRY_CONNECT` | `tcp://127.0.0.1:5555` | sim telemetry PUB endpoint |
| `FORGE_COMMAND_CONNECT`   | `tcp://127.0.0.1:5556` | sim command PULL endpoint |
| `FORGE_TARGET_ID`         | `1`    | aircraft id to subscribe to / command |
| `FORGE_TARGET_HEADING_DEG`| `0`    | target heading (0 = due north) |
| `FORGE_TARGET_ALT_M`      | `1500` | target MSL altitude (m) |
| `FORGE_TARGET_SPD_MPS`    | `50`   | target airspeed (m/s) |
| `FORGE_CONTROL_MODE`      | `raw`  | `raw` or `autopilot` |

## Build & run (inside the dev container)

```bash
cmake -S controllers/cpp -B controllers/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build controllers/cpp/build -j
FORGE_TARGET_HEADING_DEG=0 ./controllers/cpp/build/forge-controller
```

Under compose the `controller` service does exactly this and points at the `sim`
service by name (`make up`, or `docker compose up sim controller`).

## Notes

- The build uses **pkg-config** (`capnp`, `libzmq`) rather than `find_package(CapnProto)`:
  the CapnProto CMake config on this image has a spuriously fatal libatomic probe
  (8-byte atomics are lock-free compiler builtins on x86-64, so no libatomic is
  actually needed). pkg-config gives the same libs without that check.
- C++ bindings are generated from the **same** `forge.capnp` the Rust side uses, so
  the controller and sim are wire-compatible by construction. Framing is Cap'n Proto's
  flat serialization (`messageToFlatArray` / `FlatArrayMessageReader`); the zmq payload
  is copied into a word-aligned buffer before reading.
- Telemetry carries no body rates, so the raw hold loop estimates roll/pitch rate by
  finite-differencing attitude between frames for damping.
