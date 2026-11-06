# ASTRUM controller SDK

Out-of-sim controllers talk to `astrum-sim` only over Zenoh, using the shared Cap'n Proto
schema ([`crates/astrum-proto/schema/astrum.capnp`](../crates/astrum-proto/schema/astrum.capnp),
schema v3). Two SDKs implement **this one contract**, with the same names, the same loop
and the same exit codes:

| | C++ ([`common/cpp`](common/cpp)) | Python ([`python/astrum_client`](python/astrum_client)) |
|---|---|---|
| Wire types + codec | `astrum::wire` — `astrum/types.hpp`, `astrum/wire.hpp` | `astrum_client.wire` |
| Control blocks | `astrum::control` — `astrum/control.hpp` | `astrum_client.mathutil` |
| CLI flags | `astrum::cli` — `astrum/options.hpp`, `astrum/cli.hpp` | `astrum_client.cli` |
| Transport + loop | `astrum::client` — `astrum/transport.hpp`, `astrum/loop.hpp` (also declares `Law`, `StepInput`, `Stop`), `astrum/log.hpp` | `astrum_client.transport`, `astrum_client.loop` |
| Law runtime | `astrum::runtime` — `astrum/runtime.hpp` | `astrum_client.runtime` |

Programs built on them: [`cpp`](cpp) (`astrum-controller`: `hold`, `autopilot`),
[`optimal/cpp`](optimal/cpp) (`astrum-optimal-controller`: `lqr`, `mpc`),
[`python/example.py`](python/example.py) (`hold`, `autopilot`) and
[`optimal/python/run.py`](optimal/python/run.py) (`lqr`, `mpc`, `figure8`, `formation`).

## Writing a controller

A controller is a **law** plus a five-line `main` that hands a table of laws to the runtime:

```python
from astrum_client.runtime import Law, StepInput, run_main
from astrum_client.wire import Action, AutopilotTargets, VehicleKind

class Hold(Law):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING, VehicleKind.MULTIROTOR})

    @classmethod
    def from_args(cls, args):
        return cls(args.reference)

    def __init__(self, ref):
        self.ref = ref

    def step(self, inp: StepInput) -> Action:
        return Action(AutopilotTargets(self.ref.heading_rad, self.ref.alt_hae_m, self.ref.tas_mps))

if __name__ == "__main__":
    raise SystemExit(run_main({"hold": Hold}, prog="hold.py", description="..."))
```

```cpp
#include "astrum/runtime.hpp"

class Hold : public astrum::Law {
 public:
  explicit Hold(const astrum::Reference& ref) : ref_(ref) {}
  std::vector<astrum::VehicleKind> vehicle_kinds() const override {
    return {astrum::VehicleKind::FixedWing, astrum::VehicleKind::Multirotor};
  }
  astrum::LawResult step(const astrum::StepInput& in) override {
    return astrum::Action{astrum::AutopilotTargets{ref_.heading_rad, ref_.alt_hae_m, ref_.tas_mps}};
  }
 private:
  astrum::Reference ref_;
};

int main(int argc, char** argv) {
  const astrum::LawEntry laws[] = {{"hold", "hold the reference with the sim autopilot", nullptr,
      [](const astrum::LawContext& ctx) { return std::make_unique<Hold>(ctx.reference); }}};
  return astrum::run_main(argc, argv, laws, "hold", "...");
}
```

An out-of-tree C++ project uses `find_package(astrum CONFIG REQUIRED)` and links
`astrum::runtime` (see [`examples/out_of_tree`](examples/out_of_tree)); a Python project
`pip install`s `astrum-client` (the schema ships inside the package).

### The law

| | Python (`astrum_client.runtime.Law`) | C++ (`astrum::Law`) |
|---|---|---|
| Vehicle classes it can fly (required, non-empty; any other kind — including one this schema doesn't know — is refused) | `vehicle_kinds: frozenset[VehicleKind]` | `std::vector<VehicleKind> vehicle_kinds() const` |
| Reads other aircraft's telemetry (subscribes fleet-wide) | `needs_fleet = True` | `bool needs_fleet() const` |
| Law-specific flags | `@classmethod add_args(parser)` | `LawEntry::add_options(Options&)` |
| Construction from the flags (validate here: errors are usage errors, exit 2) | `@classmethod from_args(args)` | `LawEntry::make(const LawContext&)` |
| Gain design for the control period, once, before the first `step` | `configure(control_period_s)` | `configure(double control_period_s)` |
| One control step per new telemetry tick | `step(inp) -> Action`; `raise Stop(reason)` to refuse | `LawResult step(const StepInput&)` = `std::variant<Action, Stop>` |

`StepInput`:

| Field | Meaning |
|---|---|
| `state` | this aircraft's `VehicleState` (the newest sample; older unprocessed samples are superseded) |
| `fleet` | `{aircraft_id: VehicleState}` snapshot of every aircraft (empty unless `needs_fleet`), taken after the tick's `TickEnd`, so every aircraft is at the same tick |
| `events` | every event of an **earlier** tick (`tick < state.tick`) not delivered yet, oldest first. The sim sends a tick's events after its telemetry and before the next tick's, so this set is exact on every run; an event shows in `state` first (a kill: `alive = false`) and in `events` one step later |
| `control_period_s` | `Telemetry.controlPeriodS`: the sim time between this aircraft's samples — the law's `dt` |

`LawContext` (C++) and the parsed `args` (Python) carry the shared flags: `reference`
(`Reference{heading_rad, alt_hae_m, tas_mps}` from `--heading` / `--alt` / `--speed`) and
`limits` (`AttitudeLimits{max_bank_deg, max_pitch_deg}`). The rate is **not** a flag: it
comes from the telemetry.

### Names (shared with the Rust mirror in `astrum-proto`)

- `VehicleState` — one decoded `Telemetry`, flat: `aircraft_id`, `tick`, `sim_time_s`,
  `lat_deg`, `lon_deg`, `alt_hae_m`, `e`/`n`/`u`, `vel_ned_n/e/d`, `vel_enu_e/n/u`,
  `roll`/`pitch`/`yaw`, `p`/`q`/`r`, `tas_mps`, `alpha`, `beta`, `nz_g`, `mach`,
  `fault_flags`, `vehicle_kind`, `control_mode`, `team`, `hit_points`, `hit_points_max`,
  `alive`, `ammo`, `control_period_s`, `command` (`CommandStatus{owner_epoch, last_seq,
  refusals, last_refusal, last_refused_epoch}`).
- `Control` = `RawActuators{aileron, elevator, rudder, throttle}` (fixed-wing) |
  `MultirotorRaw{roll_rate, pitch_rate, yaw_rate, collective}` (multirotor) |
  `AutopilotTargets{heading_rad, alt_hae_m, tas_mps}`.
- `Action{control, trigger = false}` — what a law returns.
- `Command{target_id, seq, telemetry_tick, trigger, epoch, control}` — the wire message.
- `Event{tick, sim_time_s, detail}` with `detail` one of `TriggerPressed`, `TriggerReleased`,
  `Hit`, `Kill`, `ModeChange`, `ControllerUp`, `ControllerDown`, `RunEnded`; `TickEnd{tick,
  sim_time_s}`; enums `VehicleKind`, `ControlMode`, `ModeReason`, `EndReason`, `Refusal`.

Decoding treats every payload as untrusted: a malformed one is an error value (C++
`Decoded<T>`) or `DecodeError` (Python), counted and skipped — never an exception into Zenoh.

## Transport

| Piece | C++ / Python | Does |
|---|---|---|
| Session | `Session::open(connect, deadline)` / `open_session(connect, deadline)` | the process's one Zenoh session: `mode=client` of the sim (a router), multicast scouting off, Zenoh's low-latency transport; waits for the sim until the startup deadline |
| `TelemetrySubscriber(session, id \| fleet)` | latest-value telemetry for one aircraft or every aircraft: the Zenoh callback only stores the newest sample per aircraft, nothing is queued; a sample whose `aircraftId` differs from its key is a decode error |
| `CommandPublisher(session, id)` | `astrum/command/<id>` with DROP / express / INTERACTIVE_HIGH; a random non-zero `epoch` and `seq` = 1, 2, … per publisher; holds the presence token `astrum/alive/controller/<id>` while it lives |
| `EventSubscriber(session, tick_ends)` | FIFO of `astrum/events` (unknown kinds skipped), plus — only when `tick_ends` is set (the runtime sets it for a `needs_fleet` law) — the newest `astrum/tick` (`TickEnd`) |

One session can carry any number of subscribers and publishers (several aircraft per process).
Every subscriber callback runs on Zenoh's receive thread, in arrival order (zenoh-c does this
itself; the Python SDK asks for it with `Callback(indirect=False)`), so a tick's telemetry is
stored before its `TickEnd`, and a tick's events before the next tick's telemetry — the orders
steps 2 and 7 below rely on.

## The control loop

`run_control_loop(telemetry, events, sink, law, options) -> Outcome` runs on the caller's
thread against three interfaces (`TelemetrySource`, `EventSource`, `CommandSink`), so it is
tested without Zenoh. `options`: `aircraft_id`, `startup_deadline`, `telemetry_timeout_s`,
`tick_end_wait_s` (0.05), a clock, a logger, `status_every` (120). Per sample:

1. Wait for a sample of `aircraft_id` newer than the last one — until `startup_deadline`
   for the first, else until `telemetry_timeout_s` after the last good one — in slices of at
   most 100 ms, checking for `RunEnded` between slices. Timeout: `NoTelemetry` (no sample
   ever) or `TelemetryLost`.
2. Drain the events; a `RunEnded` among them ends the loop: `RunEnded(reason, message)`. For a
   `needs_fleet` law only, first wait up to `tick_end_wait_s` for `TickEnd` ≥ the sample's tick
   (none in time: step anyway), so the fleet snapshot holds that tick for every aircraft. Other
   laws never wait: subscribing to `astrum/tick` and waking for it costs a Python controller
   about a fifth of its lockstep throughput.
3. A vehicle kind outside `law.vehicle_kinds` → `Refused`.
4. Ownership: after this publisher has sent, a sample whose `command.last_refusal` is
   `notOwner` with `last_refused_epoch` = our epoch → `Refused` ("aircraft N is flown by
   another controller"). Other refusals of our epoch are logged with the reason when the count
   changes, at most once per second, with the number suppressed in between.
5. The same tick as the last answered one (the lockstep sim re-publishing while it waits, or
   a pause): resend the last action (new `seq`, same `telemetryTick`) without stepping.
6. First step only: `control_period_s` must be finite and > 0 (else `Refused`), then
   `law.configure(control_period_s)`. A refusal there — Python `Stop`, C++
   `std::invalid_argument` (e.g. an MPC horizon shorter than one period) — → `Refused`; any
   other exception → `LawFault`.
7. `law.step(StepInput)` with the drained events of earlier ticks (`tick < state.tick`; later
   ones wait for a later step). `Stop` → `Refused`; an exception, or a result that is not an
   `Action`, → `LawFault`.
8. A non-finite channel, or a raw control of the other vehicle class (`RawActuators` for a
   multirotor, `MultirotorRaw` for a fixed-wing — the sim would refuse it) → `LawFault`; raw /
   multirotor channels are clamped to their ranges (`[-1, 1]`, throttle / collective `[0, 1]`);
   send it answering the sample's tick.

## Outcomes and exit codes

| Outcome | Exit |
|---|---|
| `RunEnded` — `maxTime`, `engagementDecided`, `stopped` | 0 |
| `RunEnded(fatal)` or an end reason this build doesn't know, `TelemetryLost`, `NoTelemetry`, sim unreachable at startup | 1 |
| usage error, `Refused` (vehicle kind, another controller owns the aircraft, no control period, the law refused) | 2 |
| `LawFault` (non-finite output, raw control for the wrong vehicle class, exception in the law) | 3 |

`astrum run` fails a run when a controller exits non-zero while the sim is still running.

## Flags

Every shared flag falls back to an environment variable (CLI > env > default); numbers must be
finite. Law-specific flags are listed by `--help` (e.g. `mpc`'s `--horizon-s`, env
`ASTRUM_MPC_HORIZON_S`).

| Flag | Env | Default | |
|---|---|---|---|
| `--law` | `ASTRUM_LAW` | first in the table | law to run |
| `--id` | `ASTRUM_TARGET_ID` | **required** | aircraft to fly, 1 … 4294967295 |
| `--connect` | `ASTRUM_ZENOH_CONNECT` | `tcp/127.0.0.1:7447` | sim endpoint |
| `--startup-timeout` | `ASTRUM_STARTUP_TIMEOUT_S` | `60` | one deadline for reaching the sim and its first sample; exit 1 |
| `--telemetry-timeout` | `ASTRUM_TELEMETRY_TIMEOUT_S` | `10` | telemetry silence (without `RunEnded`) that ends the run; exit 1 |
| `--heading` | `ASTRUM_TARGET_HEADING_DEG` | `0` | reference true heading, degrees |
| `--alt` | `ASTRUM_TARGET_ALT_M` | `1500` | reference altitude, m HAE |
| `--speed` | `ASTRUM_TARGET_SPD_MPS` | `50` | reference true airspeed, m/s (≥ 0) |
| `--max-bank-deg` | `ASTRUM_MAX_BANK_DEG` | `25` | bank limit, in (0, 90) |
| `--max-pitch-deg` | `ASTRUM_MAX_PITCH_DEG` | `10` | pitch limit, in (0, 90) |

## Wire fixtures

[`crates/astrum-proto/tests/fixtures`](../crates/astrum-proto/tests/fixtures) holds every
kind encoded by Rust, C++ and Python (`<kind>.<lang>.bin`) from `values.json`; each language
decodes every file and compares it field by field. Regenerate with `scripts/wire-fixtures.sh`.
