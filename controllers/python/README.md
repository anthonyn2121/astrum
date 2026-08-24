# FORGE example flight controller (Python)

The Python twin of [`controllers/cpp`](../cpp) — a minimal out-of-sim flight
controller that talks to `forge-sim` **only over ZMQ**, using the shared Cap'n
Proto schema (`crates/forge-proto/schema/forge.capnp`) for the wire format. It
exists to prove a controller in any language plugs into FORGE with nothing but
**pyzmq + pycapnp** (both ship in the dev image); there is **no codegen step** —
`capnp.load()` reads `forge.capnp` at runtime.

- **SUB** `tcp://sim:5555` — telemetry for one aircraft (topic = 4-byte big-endian id).
- **PUSH** `tcp://sim:5556` — commands (target id + raw actuators *or* autopilot targets).

## Modes (`FORGE_CONTROL_MODE`)

- **`raw`** (default) — the controller closes a heading / altitude / airspeed hold loop
  *itself* (identical gains to the C++ `HoldLaw`) and sends raw `fcs/*-cmd-norm`
  deflections every telemetry tick. This is FORGE's intended usage: **you own the
  control law**.
- **`autopilot`** — the controller sends a single high-level target and then just
  monitors; the sim's built-in proportional inner loop flies the aircraft.

## Configuration (environment)

Identical to the C++ controller, so the compose wiring is shared:

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_TELEMETRY_CONNECT` | `tcp://127.0.0.1:5555` | sim telemetry PUB endpoint |
| `FORGE_COMMAND_CONNECT`   | `tcp://127.0.0.1:5556` | sim command PULL endpoint |
| `FORGE_TARGET_ID`         | `1`    | aircraft id to subscribe to / command |
| `FORGE_TARGET_HEADING_DEG`| `0`    | target heading (0 = due north) |
| `FORGE_TARGET_ALT_M`      | `1500` | target MSL altitude (m) |
| `FORGE_TARGET_SPD_MPS`    | `50`   | target airspeed (m/s) |
| `FORGE_CONTROL_MODE`      | `raw`  | `raw` or `autopilot` |
| `FORGE_SCHEMA`            | `../../crates/forge-proto/schema/forge.capnp` | schema to load |

## Run (inside the dev container)

```bash
python controllers/python/controller.py
```

Or the whole self-terminating demo (bounded single-ship scenario, auto teardown):

```bash
make demo-python
```

which is `docker compose --profile python up --exit-code-from sim sim controller-python`.

## Notes

- **Wire compatibility is by construction:** the same `forge.capnp` is the source of
  truth for the Rust sim, the C++ controller, and this script. Framing is Cap'n Proto's
  flat serialization — pycapnp's `to_bytes()` / `from_bytes()` interoperate directly with
  Rust's `serialize::write_message` / `read_message`. (Verified live: this controller
  drove aircraft 1 from 90° to ~3.4° heading holding ~1500 m, the same trace as the C++
  controller.)
- Telemetry carries no body rates, so the raw hold loop estimates roll/pitch rate by
  finite-differencing attitude between frames for damping — same as the C++ `HoldLaw`.
- The telemetry topic is the aircraft id as **4 big-endian bytes** (`id.to_bytes(4, "big")`),
  matching `forge-net`'s topic frame; ZMQ delivers `[topic, payload]` and the payload is the
  last frame.
