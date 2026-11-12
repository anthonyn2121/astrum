# ASTRUM Python controller SDK (`astrum_client`) and example controller

This directory is the **`astrum-client`** package — the Python half of the controller SDK — plus
`example.py`, the Python twin of [`controllers/cpp`](../cpp). The contract both SDKs implement
(names, the `Law` API, the transport, the control-loop algorithm, outcomes and exit codes, the
flags) is **[controllers/README.md](../README.md)**; this page only covers what is Python-specific.

## Install

The dev image puts `controllers/python` and `controllers/optimal/python` on `PYTHONPATH`.
Anywhere else:

```bash
pip install controllers/python            # astrum-client: eclipse-zenoh==1.10.0, pycapnp
pip install controllers/optimal/python    # astrum-optimal (LQR / MPC / formation laws)
```

The schema ships inside the package (`astrum_client/astrum.capnp`, a symlink to
`crates/astrum-proto/schema/astrum.capnp` that builds copy) and is loaded once, lazily, by
`astrum_client.wire`; its `/capnp/c++.capnp` import resolves from pycapnp's own install, so
nothing depends on a checkout or `/usr/include`. `eclipse-zenoh` is pinned to the Rust `zenoh`
crate's version (Zenoh is protocol-compatible only within a release; `make verify` checks it).

## Modules

| Module | What it holds |
|---|---|
| `astrum_client.wire` | `VehicleState`, `CommandStatus`, `RawActuators` / `MultirotorRaw` / `AutopilotTargets`, `Action`, `Command`, `Event` (+ detail kinds), `TickEnd`, the enums (`VehicleKind`, `ControlMode`, `ModeReason`, `EndReason`, `Refusal`); `decode_telemetry` / `decode_command` / `decode_event` / `decode_tick_end` (untrusted input → value or `DecodeError`) and the encoders |
| `astrum_client.transport` | `open_session(connect, deadline)`, `TelemetrySubscriber(session, id \| None)`, `EventSubscriber(session, tick_ends)`, `CommandPublisher(session, id)`, `SimUnreachable`; every callback runs on Zenoh's receive thread in arrival order (`Callback(indirect=False)`) |
| `astrum_client.loop` | `run_control_loop(telemetry, events, sink, law, options) -> Outcome`, `LoopOptions`, `Outcome` / `OutcomeKind`, `exit_code`, the `TelemetrySource` / `EventSource` / `CommandSink` protocols |
| `astrum_client.runtime` | `Law`, `StepInput`, `Stop`, `Reference`, `AttitudeLimits`, `run_main(laws, argv=None, *, prog, description)` |
| `astrum_client.cli` | the shared flags with `ASTRUM_*` env fallbacks and their validators |
| `astrum_client.mathutil` | `wrap_pi`, `clamp`, the anti-windup integrator step |
| `astrum_client.hold` | `HoldLaw`, the example raw-actuator hold law |
| `astrum_client.wire_fixture` | writes the cross-language wire fixtures (`python -m astrum_client.wire_fixture`) |

Deadlines are `time.monotonic()` values. `LoopOptions` takes an injectable `clock` and `logger`
(defaults `time.monotonic`, `logging.getLogger("astrum")`), so the loop runs against fakes in
[`tests/test_control_loop.py`](tests/test_control_loop.py).

## A law

```python
from astrum_client.runtime import Law, StepInput, run_main
from astrum_client.wire import Action, RawActuators, VehicleKind


class Wings(Law):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})  # required, non-empty

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--gain", type=float, default=1.5)

    @classmethod
    def from_args(cls, args):  # args.reference, args.limits are set; ValueError -> exit 2
        return cls(args.gain)

    def __init__(self, gain):
        self.gain = gain

    def configure(self, control_period_s):  # once, from Telemetry.controlPeriodS
        self.dt = control_period_s

    def step(self, inp: StepInput) -> Action:  # raise Stop("why") to refuse
        return Action(RawActuators(aileron=-self.gain * inp.state.roll, throttle=0.6))


if __name__ == "__main__":
    raise SystemExit(run_main({"wings": Wings}, prog="wings.py", description="level the wings"))
```

`inp.events` holds every event of an earlier tick (`tick < state.tick`) not delivered yet,
oldest first: an event shows in `state` first and in `events` one step later. `inp.fleet` is a
`{aircraft_id: VehicleState}` snapshot when the law sets `needs_fleet = True`; only such a law
subscribes to `astrum/tick` and waits (at most `tick_end_wait_s`) for the tick's `TickEnd`
before stepping, so every aircraft in the snapshot is at the same tick — other laws skip that
wakeup, which costs a Python controller about a fifth of its lockstep throughput. The loop
clamps raw / multirotor channels; a non-finite channel, a raw control of the other vehicle
class, or an exception is a `LawFault` (exit 3). When the sim re-publishes a tick, the loop
resends the last action without stepping.

## The example controller

```bash
python controllers/python/example.py --id 1 --law hold --heading 0
python controllers/python/example.py --id 1 --law autopilot --alt 130 --speed 0   # quad demo
astrum run config/demos/python.toml   # bounded single-ship demo with auto teardown
```

- **`hold`** (default) — `HoldLaw`: heading / altitude / airspeed hold closed here, raw actuators
  every tick (a lease: if they stop, the sim's failsafe takes over). Damps with the measured body
  rates, limits bank / pitch to `--max-bank-deg` / `--max-pitch-deg`, and its integrators use
  conditional-integration anti-windup. Fixed-wing only.
- **`autopilot`** — hands the reference to the sim's inner loop every tick. Fixed-wing or
  multirotor.

`--help` lists every flag (see [controllers/README.md](../README.md#flags)); `--id` is required.

## Tests and wire fixtures

```bash
python -m pytest -q controllers/python controllers/optimal/python
python -m astrum_client.wire_fixture    # writes crates/astrum-proto/tests/fixtures/*.python.bin
```

[`tests/test_wire_fixtures.py`](tests/test_wire_fixtures.py) decodes every `<kind>.<lang>.bin`
present (rust / cpp / python, events included) and compares it field by field with
`values.json`; the Python files must exist and match what the encoder produces today.
