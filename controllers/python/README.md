# ASTRUM example flight controller (Python)

The Python twin of [`controllers/cpp`](../cpp): a minimal out-of-sim flight
controller that talks to `astrum-sim` **only over Zenoh**, using the shared Cap'n
Proto schema (`crates/astrum-proto/schema/astrum.capnp`, schema v2). There is no
codegen step — `capnp.load()` reads the schema at runtime.

The directory is also the **`astrum_client`** package — the shared client every
Python controller (this example and [`controllers/optimal`](../optimal)) builds on:

| Module | What it does |
|---|---|
| `astrum_client.wire` | `load_schema()`, `decode_telemetry()` → `VehicleState` (every v2 field), `encode_command()`; bad payloads raise `DecodeError` (framing checked, trailing bytes rejected) |
| `astrum_client.transport` | `open_session()` (client mode, no multicast scouting, Zenoh's low-latency transport, waits up to the startup timeout for the sim), `subscribe_telemetry()` (latest-value, per aircraft or fleet-wide), `CommandPublisher` (DROP / express / INTERACTIVE_HIGH, `seq` 1, 2, … and `telemetryTick`) |
| `astrum_client.loop` | `run_control_loop()`: one command per telemetry tick, startup/staleness timeouts, vehicle-kind refusal, output saturation |
| `astrum_client.cli` | shared flags with `ASTRUM_*` env fallbacks |
| `astrum_client.mathutil` | `wrap_pi` (→ `[-pi, pi)`, NaN for non-finite input), `clamp`, anti-windup integrator step |
| `astrum_client.hold` | the example's raw `HoldLaw` |
| `astrum_client.wire_fixture` | writes / checks the cross-language wire fixtures |

The dev image puts `controllers/python` and `controllers/optimal/python` on
`PYTHONPATH`; elsewhere `pip install -e controllers/python` (+ `-e
controllers/optimal/python`). The schema is resolved relative to the checkout.

## Transport

One `mode=client` Zenoh session connecting to the sim, which listens as the Zenoh router
(`tcp/sim:7447` under compose):

- **SUB** `astrum/telemetry/<id>` — a callback keeps only the **newest** sample per
  aircraft, so a slow law always acts on fresh state (never a FIFO of stale data).
  A payload that fails to decode, or whose `aircraftId` differs from its key, is
  logged, counted and skipped.
- **PUB** `astrum/command/<id>` — one command per telemetry tick, carrying
  `seq` (strictly increasing per process) and `telemetryTick` (the tick it answers).
  If the sim republishes a tick it is waiting on (lockstep), the last command is
  resent with a new `seq` without re-running the law.

## Modes (`--mode`)

- **`raw`** (default) — closes a heading / altitude / airspeed hold loop here and
  sends raw actuators every tick. Raw commands are a **lease**: if they stop, the
  sim's failsafe takes over (`Telemetry.controlMode = failsafe`, logged). The law
  damps with the body rates the sim publishes, limits bank/pitch to
  `--max-bank-deg` / `--max-pitch-deg`, saturates every actuator, and has
  heading/altitude/airspeed integrators with conditional-integration anti-windup
  (the heading integral removes the few-degree steady error propeller torque leaves).
  Fixed-wing only: any other `vehicleKind` (including an unknown value) is refused.
- **`autopilot`** — sends the target to the sim's inner loop every tick (which also
  keeps the lease fresh). Fixed-wing or multirotor.

## Configuration (flags)

Each flag falls back to its `ASTRUM_*` env var, then the default
(**CLI arg > `$ENV` > default**). Values are SI, altitudes HAE, speeds TAS.

| Flag | Env fallback | Default | Meaning |
|---|---|---|---|
| `--connect` | `ASTRUM_ZENOH_CONNECT` | `tcp/127.0.0.1:7447` | sim Zenoh endpoint |
| `--id` | `ASTRUM_TARGET_ID` | `1` | aircraft id to subscribe to / command |
| `--heading` | `ASTRUM_TARGET_HEADING_DEG` | `0` | target true heading, degrees (0 = north) |
| `--alt` | `ASTRUM_TARGET_ALT_M` | `1500` | target altitude, m HAE |
| `--speed` | `ASTRUM_TARGET_SPD_MPS` | `50` | target true airspeed, m/s |
| `--mode` | `ASTRUM_CONTROL_MODE` | `raw` | `raw` or `autopilot` |
| `--max-bank-deg` | `ASTRUM_MAX_BANK_DEG` | `25` | bank limit (raw); f16: `60` |
| `--max-pitch-deg` | `ASTRUM_MAX_PITCH_DEG` | `10` | pitch limit (raw); f16: `24` |
| `--startup-timeout` | `ASTRUM_STARTUP_TIMEOUT_S` | `60` | exit 1 if no telemetry arrives within this many seconds |
| `--telemetry-timeout` | `ASTRUM_TELEMETRY_TIMEOUT_S` | `10` | exit 0 once telemetry has stopped this long |
| `--schema` | `ASTRUM_SCHEMA` | `crates/astrum-proto/schema/astrum.capnp` | schema to load |

Exit codes: `0` telemetry stopped, `1` no telemetry at startup, `2` refused to fly
(wrong/unknown vehicle kind).

## Run (inside the dev container)

```bash
python controllers/python/example.py --heading 0 --id 1
make demo-python     # bounded single-ship demo with auto teardown
```

## Tests and wire fixtures

```bash
python -m pytest -q controllers/python controllers/optimal/python
python -m astrum_client.wire_fixture    # writes crates/astrum-proto/tests/fixtures/*.python.bin
```

`tests/test_wire_fixtures.py` decodes every `<kind>.<lang>.bin` present
(rust / cpp / python) and compares every field exactly with `values.json`; the
Python files must exist and match what the encoder produces today.
