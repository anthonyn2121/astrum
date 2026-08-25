# ASTRUM example flight controller (Python)

The Python twin of [`controllers/cpp`](../cpp) — a minimal out-of-sim flight
controller that talks to `astrum-sim` **only over Zenoh**, using the shared Cap'n
Proto schema (`crates/astrum-proto/schema/astrum.capnp`) for the wire format. It
exists to prove a controller in any language plugs into ASTRUM with nothing but
**eclipse-zenoh + pycapnp** (both ship in the dev image); there is **no codegen step** —
`capnp.load()` reads `astrum.capnp` at runtime. Telemetry and commands share **one**
`mode=peer` Zenoh session connecting to `tcp/sim:7447` (not two ports); the id is in the key:

- **SUB** `astrum/telemetry/<id>` — telemetry for one aircraft.
- **PUB** `astrum/command/<id>` — commands (target id + raw actuators *or* autopilot targets).

## Modes (`ASTRUM_CONTROL_MODE`)

- **`raw`** (default) — the controller closes a heading / altitude / airspeed hold loop
  *itself* (identical gains to the C++ `HoldLaw`) and sends raw `fcs/*-cmd-norm`
  deflections every telemetry tick. This is ASTRUM's intended usage: **you own the
  control law**.
- **`autopilot`** — the controller sends one high-level target (reasserted over the first
  second of telemetry) and then just monitors; the sim's built-in proportional inner loop flies
  the aircraft. Resending briefly clears the Zenoh startup race: Zenoh drops a `put` with no
  matching subscriber yet (unlike ZMQ PUSH, which queued it), so a single shot at t=0 would be
  lost while the sim loads JSBSim — receiving telemetry proves the link is up.

## Configuration (environment)

Identical to the C++ controller, so the compose wiring is shared:

| Variable | Default | Meaning |
|---|---|---|
| `ASTRUM_ZENOH_CONNECT`     | `tcp/127.0.0.1:7447` | sim Zenoh endpoint (telemetry + commands) |
| `ASTRUM_TARGET_ID`         | `1`    | aircraft id to subscribe to / command |
| `ASTRUM_TARGET_HEADING_DEG`| `0`    | target heading (0 = due north) |
| `ASTRUM_TARGET_ALT_M`      | `1500` | target MSL altitude (m) |
| `ASTRUM_TARGET_SPD_MPS`    | `50`   | target airspeed (m/s) |
| `ASTRUM_CONTROL_MODE`      | `raw`  | `raw` or `autopilot` |
| `ASTRUM_SCHEMA`            | `../../crates/astrum-proto/schema/astrum.capnp` | schema to load |

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

- **Wire compatibility is by construction:** the same `astrum.capnp` is the source of
  truth for the Rust sim, the C++ controller, and this script. Framing is Cap'n Proto's
  flat serialization — pycapnp's `to_bytes()` / `from_bytes()` interoperate directly with
  Rust's `serialize::write_message` / `read_message`. (Verified live: this controller
  drove aircraft 1 from 90° to ~3.4° heading holding ~1500 m, the same trace as the C++
  controller.)
- Telemetry carries no body rates, so the raw hold loop estimates roll/pitch rate by
  finite-differencing attitude between frames for damping — same as the C++ `HoldLaw`.
- The aircraft id lives in the Zenoh **key** (`astrum/telemetry/<id>`), not a topic frame:
  the subscriber declares that key and each sample's `payload` is the whole Cap'n Proto message.
  Key matching is exact, so `astrum/telemetry/1` never matches `astrum/telemetry/10`.
