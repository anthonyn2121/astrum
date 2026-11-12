# ASTRUM C++ controller SDK and example controller

The C++ side of the controller SDK lives in [`controllers/common/cpp`](../common/cpp); the
**contract it implements** — names, the `Law` API, `StepInput`, the transport, the control
loop, outcomes and exit codes, flags — is [`controllers/README.md`](../README.md), shared
with the Python SDK. This page covers what is C++-specific.

## Libraries (CMake targets)

| Target | Headers | Links | What |
|---|---|---|---|
| `astrum::wire` | `astrum/types.hpp`, `astrum/wire.hpp` | Cap'n Proto (privately, plus the generated code) | plain types, `decode_*` (noexcept → `Decoded<T>`) / `encode_*` for telemetry, command, event, tick end |
| `astrum::control` | `astrum/control.hpp`, `astrum/math.hpp` | `wire` | `SaturatedPi`, `OuterLoops`, angle helpers |
| `astrum::cli` | `astrum/options.hpp`, `astrum/cli.hpp` | `wire` | `Options` (CLI > env > default) and the shared-flag helpers + validation |
| `astrum::client` | `astrum/transport.hpp`, `astrum/loop.hpp`, `astrum/log.hpp` | `wire`, Zenoh (privately) | `Session`, `TelemetrySubscriber`, `CommandPublisher`, `EventSubscriber`; `run_control_loop`, `Outcome`, `exit_code`; the `Law` interface. No CLI code. |
| `astrum::runtime` | `astrum/runtime.hpp` | `client`, `cli`, `control` | `LawEntry`, `LawContext`, `run_main` |

`Law`, `StepInput`, `Stop` and `LawResult` are declared in `astrum/loop.hpp` (the loop
runs them) and come with `astrum/runtime.hpp`. Logging goes through an injected
`Logger = std::function<void(LogLevel, std::string_view)>` (`stderr_logger(tag)` by
default) — the transport and the loop never write to `std::cerr` themselves.

Several aircraft in one process share one `Session`: per aircraft, construct a
`TelemetrySubscriber` (or a fleet-wide `TelemetrySubscriber(session, kFleet)`), a
`CommandPublisher` and an `EventSubscriber` (each holds its own event FIFO), and run a
`run_control_loop` on its own thread.

Events reach a law one step late, exactly: `StepInput::events` holds the events of ticks
before `state.tick` (the sim publishes a tick's events after its telemetry). Only a
`needs_fleet()` law waits for `TickEnd` (bounded by `tick_end_wait_s`) so its fleet
snapshot is tick-aligned; `EventSubscriber(session, tick_ends)` subscribes to `astrum/tick`
only when `tick_ends` is set, and `run_main` sets it from `needs_fleet()`.

## `astrum-controller` (this directory)

`run_main` with two laws:

- **`hold`** (default) — the raw hold law in [`src/hold_law.cpp`](src/hold_law.cpp), for
  **fixed-wing** aircraft. Outer loops (`astrum::OuterLoops`, shared with the optimal laws):
  heading PI → bank target within `±--max-bank-deg`, altitude PI → pitch target within
  `±--max-pitch-deg`, airspeed PI → throttle, each with conditional-integration
  anti-windup. Inner loops: `aileron = 1.5 (bank_t − φ) − 0.3 p`,
  `elevator = −2 (θ_t − θ) + 0.5 q` (aileron+ rolls right, elevator+ pitches nose down).
- **`autopilot`** — the `--heading` / `--alt` / `--speed` reference as `AutopilotTargets`
  every step; the sim's inner loop flies fixed-wing and multirotor aircraft.

```bash
./build/controllers/cpp/astrum-controller --id 1 --heading 90            # hold
./build/controllers/cpp/astrum-controller --id 1 --law autopilot --alt 1200
```

`astrum run config/demos/cpp.toml` flies it against a fresh sim. `--help` lists every
flag with its `ASTRUM_*` environment fallback (table: [`controllers/README.md`](../README.md#flags)).

## Build, test and install (inside the dev container)

All C++ controllers are one CMake project rooted at `controllers/`:

```bash
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j
ctest --test-dir build/controllers --output-on-failure
cmake --install build/controllers --prefix /opt/astrum      # the SDK, for out-of-tree use
```

`make test-cpp` runs the build and ctest, then installs to a temporary prefix and builds
[`../examples/out_of_tree`](../examples/out_of_tree) against it (`find_package(astrum
CONFIG REQUIRED)`, `target_link_libraries(... astrum::runtime)`).

Options: `-DASTRUM_BUILD_TESTS=OFF` drops the tests and the fixture tool (and the GTest /
nlohmann_json dependencies); `-DASTRUM_BUILD_OPTIMAL=OFF` drops
`astrum-optimal-controller` (and Eigen). The SDK itself needs CapnProto, zenohc + zenohcxx
and Threads; a consumer project enables C as well as CXX because `CapnProtoConfig.cmake`
probes libatomic with a C program.

## Notes

- Hand-written sources build with `-Wall -Wextra -Wpedantic -Wshadow -Werror`; generated
  capnp code is compiled separately, without `-Werror`, and is not installed.
- The loop is tested without Zenoh against scripted `TelemetrySource` / `EventSource` /
  `CommandSink` fakes and a fake clock (`common/cpp/tests/loop_test.cpp`).
- Wire fixtures: `build/controllers/common/cpp/astrum-wire-fixture write
  crates/astrum-proto/tests/fixtures` writes every `<kind>.cpp.bin` from `values.json`; the
  ctest `wire_fixtures` (`... verify <dir>`) decodes every `<kind>.<lang>.bin` present and
  diffs it against `values.json`.
