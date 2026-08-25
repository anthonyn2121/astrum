# ASTRUM example flight controller (C++)

A minimal out-of-sim flight controller that talks to `astrum-sim` **only over Zenoh**,
using the shared Cap'n Proto schema (`crates/astrum-proto/schema/astrum.capnp`) for the
wire format. It is the reference for how a real controller (C++ or, later, Python)
plugs into ASTRUM. Telemetry and commands ride **one** `mode=peer` Zenoh session connecting
to `tcp/sim:7447` (not two ports); the id lives in the key, so there's no topic frame:

- **SUB** `astrum/telemetry/<id>` — telemetry for one aircraft.
- **PUB** `astrum/command/<id>` — commands (target id + raw actuators *or* autopilot targets).

## Modes (`--mode`)

- **`raw`** (default) — the controller closes a heading / altitude / airspeed hold loop
  *itself* and sends raw `fcs/*-cmd-norm` deflections every telemetry tick. This is
  ASTRUM's intended usage: **you own the control law**.
- **`autopilot`** — the controller sends one high-level target (reasserted over the first
  second of telemetry) and then just monitors; the sim's built-in proportional inner loop flies
  the aircraft. Resending briefly clears the Zenoh startup race: Zenoh drops a `put` with no
  matching subscriber yet (unlike ZMQ PUSH, which queued it), so a single shot at t=0 would be
  lost while the sim loads JSBSim — receiving telemetry proves the link is up.

Both converge a 90° turn to within ~4° while holding altitude to a few metres (see
the traces in the repo's verification notes).

## Configuration (flags)

Pass config as command-line flags (run `--help` for the list). Each flag falls
back to its old `ASTRUM_*` env var, then the default, so precedence is
**CLI arg > `$ENV` > default** and the compose services are unchanged.

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--connect` | `ASTRUM_ZENOH_CONNECT`      | `tcp/127.0.0.1:7447` | sim Zenoh endpoint (telemetry + commands) |
| `--id`      | `ASTRUM_TARGET_ID`         | `1`    | aircraft id to subscribe to / command |
| `--heading` | `ASTRUM_TARGET_HEADING_DEG`| `0`    | target heading in degrees (0 = due north) |
| `--alt`     | `ASTRUM_TARGET_ALT_M`      | `1500` | target MSL altitude (m) |
| `--speed`   | `ASTRUM_TARGET_SPD_MPS`    | `50`   | target airspeed (m/s) |
| `--mode`    | `ASTRUM_CONTROL_MODE`      | `raw`  | `raw` or `autopilot` |

## Build & run (inside the dev container)

```bash
cmake -S controllers/cpp -B controllers/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build controllers/cpp/build -j
./controllers/cpp/build/astrum-controller --heading 0 --id 1
```

Under compose the `controller` service does exactly this (via its env vars, which
the flags fall back to) and points at the `sim` service by name (`make up`, or
`docker compose up sim controller`).

## Notes

- Zenoh links via **zenoh-cpp**: include `<zenoh.hxx>`, compile with the `ZENOHCXX_ZENOHC`
  define (selects the zenoh-c backend) and link `-lzenohc`. Do **not** use pkg-config for
  Zenoh — the shipped `zenohc.pc` has a stale `/usr/local` prefix; rely on the default
  include/lib paths instead.
- Cap'n Proto still uses **pkg-config** (`capnp`) rather than `find_package(CapnProto)`:
  the CapnProto CMake config on this image has a spuriously fatal libatomic probe
  (8-byte atomics are lock-free compiler builtins on x86-64, so no libatomic is
  actually needed). pkg-config gives the same libs without that check.
- C++ bindings are generated from the **same** `astrum.capnp` the Rust side uses, so
  the controller and sim are wire-compatible by construction. Framing is Cap'n Proto's
  flat serialization (`messageToFlatArray` / `FlatArrayMessageReader`); the Zenoh payload
  is copied into a word-aligned buffer before reading.
- Telemetry carries no body rates, so the raw hold loop estimates roll/pitch rate by
  finite-differencing attitude between frames for damping.
- This example follows **one** aircraft (`--id`), but the transport is fleet-wide:
  subscribe `astrum/telemetry/*` instead of a single id to receive **every** aircraft's telemetry
  over the same session — useful for a controller that reacts to its neighbours. Adding aircraft
  needs no endpoint or port change; the id is just another key.
