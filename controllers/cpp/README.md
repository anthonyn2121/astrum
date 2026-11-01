# ASTRUM example flight controller (C++)

A minimal out-of-sim flight controller that talks to `astrum-sim` **only over Zenoh**, using
the shared Cap'n Proto schema (`crates/astrum-proto/schema/astrum.capnp`, C++ namespace
`astrum::wire`). It is a thin program on the shared C++ client in
[`controllers/common/cpp`](../common/cpp), which implements the transport contract:

- one `mode=client` Zenoh session to the sim (which listens as the router), multicast
  scouting off, connecting to `--connect`; opening waits for a sim that is still starting,
  up to `--startup-timeout`;
- **SUB** `astrum/telemetry/<id>` as **latest-value** state: the Zenoh callback only
  overwrites one slot, and decoding + the control law run on the main thread, so a slow
  law skips stale samples instead of queueing them or stalling Zenoh's receive thread;
- **PUB** `astrum/command/<id>` on a declared publisher with CongestionControl `DROP`,
  `express`, priority `INTERACTIVE_HIGH`; **one command per telemetry sample** (raw
  commands are a lease — stop sending and the sim's failsafe takes over), each carrying
  `seq` (strictly increasing from 1) and `telemetryTick` (the tick it answers). A repeated
  tick (the lockstep sim republishing while it waits) is answered again with a new `seq`;
- presence: the client holds the liveliness token `astrum/alive/controller/<id>`, so the
  sim learns at once when the process goes away (a lockstep run fails immediately;
  otherwise the aircraft enters the failsafe hold);
- untrusted input: a payload that fails to decode is counted, logged and skipped — it
  never throws through Zenoh;
- timeouts: no telemetry within `--startup-timeout` → exit **1**; telemetry stops for
  `--telemetry-timeout` after it started (the sim finished) → exit **0**.

## Modes (`--mode`)

- **`raw`** (default) — the controller closes the loop itself (`src/hold_law.cpp`) and
  sends raw actuators every sample. You own the control law. The law is written for
  **fixed-wing** aircraft: telemetry with any other `vehicleKind` (including a kind this
  build does not know) is refused with exit **2**.
  - Outer loops (`astrum::OuterLoops`, shared with the optimal laws): heading PI → bank
    target clamped to `±--max-bank-deg`; altitude PI → pitch target clamped to
    `±--max-pitch-deg`; airspeed PI → throttle in `[0, 1]`. Every integrator uses
    conditional integration (it stops integrating while the output is saturated in the
    error's direction), so a long turn, climb or acceleration does not wind up; the
    heading integral removes the few-degree steady error propeller torque leaves.
  - Inner loops: `aileron = 1.5 (bank_t − φ) − 0.3 p`, `elevator = −2 (θ_t − θ) + 0.5 q`,
    using the telemetry **body rates** `p`/`q`. Signs as measured in the sim: aileron+ rolls
    right, elevator+ pitches nose down.
- **`autopilot`** — sends the heading / altitude / airspeed target **every sample** (it
  refreshes the lease too); the sim's built-in inner loop flies fixed-wing and multirotor
  aircraft.

## Configuration (flags)

Run `--help` for the list. Each flag falls back to its `ASTRUM_*` env var, then the default
(precedence **CLI arg > `$ENV` > default**). Numbers must parse completely; a bad value is a
usage error (exit 2).

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--connect` | `ASTRUM_ZENOH_CONNECT` | `tcp/127.0.0.1:7447` | sim Zenoh endpoint (`tcp/sim:7447` under compose) |
| `--id` | `ASTRUM_TARGET_ID` | `1` | aircraft id to fly |
| `--heading` | `ASTRUM_TARGET_HEADING_DEG` | `0` | target true heading, degrees (0 = north) |
| `--alt` | `ASTRUM_TARGET_ALT_M` | `1500` | target altitude, m above the WGS84 ellipsoid (HAE) |
| `--speed` | `ASTRUM_TARGET_SPD_MPS` | `50` | target **true** airspeed, m/s |
| `--mode` | `ASTRUM_CONTROL_MODE` | `raw` | `raw` or `autopilot` |
| `--startup-timeout` | `ASTRUM_STARTUP_TIMEOUT_S` | `60` | exit 1 if no telemetry arrives within this many seconds |
| `--telemetry-timeout` | `ASTRUM_TELEMETRY_TIMEOUT_S` | `10` | exit 0 once telemetry has stopped this long |
| `--max-bank-deg` | `ASTRUM_MAX_BANK_DEG` | `25` | bank command limit (trainer; the F-16 formation demo uses 60) |
| `--max-pitch-deg` | `ASTRUM_MAX_PITCH_DEG` | `10` | pitch command limit (trainer; F-16: 24) |

## Build, test & run (inside the dev container)

All C++ controllers are one CMake project rooted at `controllers/`:

```bash
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j
ctest --test-dir build/controllers --output-on-failure
./build/controllers/cpp/astrum-controller --heading 90 --id 1
```

Under compose the `controller` service runs this binary against the `sim` service (`make up`,
or `make demo`). `make test-cpp` runs the build + ctest above.

## Notes

- Dependencies come from their CMake packages: `CapnProto` (schema codegen with
  `capnp_generate_cpp`, once, in `controllers/common/cpp`), `zenohc` + `zenohcxx`
  (`zenohcxx::zenohc`), `Eigen3`, `nlohmann_json` (fixture tool), `GTest`. The project enables
  the C language only because `CapnProtoConfig.cmake` probes libatomic with a C test program.
- Our sources build with `-Wall -Wextra -Wpedantic -Wshadow -Werror`; generated capnp code is
  compiled separately without `-Werror`.
- Framing is one unpacked flat Cap'n Proto message per payload; decoding reads word-aligned
  payloads in place, rejects partial words, malformed messages and trailing bytes.
- The cross-language wire fixtures (`crates/astrum-proto/tests/fixtures`) are written by
  `build/controllers/common/cpp/astrum-wire-fixture write crates/astrum-proto/tests/fixtures`
  (`<kind>.cpp.bin`); the ctest `wire_fixtures` (`... verify <dir>`) decodes every
  `<kind>.<lang>.bin` present and compares each field exactly to `values.json`.
- This example follows one aircraft (`--id`); the transport is fleet-wide (`astrum/telemetry/*`
  carries every aircraft), and adding aircraft needs no endpoint or port change.
