# ASTRUM optimal-control controllers (LQR / MPC)

A clean-API scaffold for developing **your own** control laws — LQR, MPC, or
whatever else — against `astrum-sim`. Unlike the minimal
[`controllers/cpp`](../cpp) / [`controllers/python`](../python) examples (one file,
one hard-coded hold law), this splits the code in two:

- a **runtime** that owns everything you shouldn't have to think about — the one
  `mode=client` Zenoh session to the sim (the router), the Cap'n Proto codec, decoding telemetry into a
  clean state object, latest-value subscriptions, the vehicle-kind check, output
  saturation, command `seq`/`telemetryTick`, and the startup/staleness timeouts;
- **laws** — the only files you write. A law maps a `VehicleState` + `Reference`
  to a control, and nothing else. No Zenoh, no Cap'n Proto, no wire.

Both languages share the same shape, so you can prototype a law in Python and port
it to C++ (or vice versa) with the same mental model.

```
controllers/optimal/
├── python/                 # the `astrum_optimal` package (built on controllers/python's astrum_client)
│   ├── run.py              # entry point: `--law <name>` selects a law
│   ├── pyproject.toml
│   ├── tests/              # pytest: laws, MPC/LQR, formation, runtime
│   └── astrum_optimal/
│       ├── runtime.py      # Law base class, Reference, flags, main()
│       ├── linear.py       # ZOH discretization, dlqr, ChannelMPC
│       └── laws/
│           ├── _fixed_wing.py  # linear model + limited three-channel feedback (shared)
│           ├── _guidance.py    # grid-track -> true-heading helpers (shared)
│           ├── lqr.py          # LQR law   (edit me)
│           ├── mpc.py          # MPC law   (edit me)
│           ├── figure8.py      # leader guidance: fly a figure-eight
│           └── formation.py    # follower: decentralized formation-keeping MPC
└── cpp/                    # astrum-optimal-controller (built from controllers/, on controllers/common/cpp)
    ├── include/
    │   ├── astrum_control.hpp  # the law API: Law, LawConfig
    │   ├── laws.hpp            # law factories
    │   ├── attitude_axes.hpp   # per-axis models + LQR weights shared by lqr/mpc (edit me)
    │   └── optimal_math.hpp    # ZOH axis model, dlqr (reports non-convergence), box QP
    ├── src/runtime.cpp     # CLI, law factory, kind/rate checks, main
    ├── laws/
    │   ├── lqr.cpp         # LQR law   (edit me)
    │   └── mpc.cpp         # MPC law   (edit me)
    └── tests/              # gtest: dlqr, ZOH model, box QP, law signs/limits
```

## The API

You implement one method. In Python (`astrum_optimal/laws/<name>.py`):

```python
from astrum_client.wire import RawActuators, VehicleKind
from astrum_optimal.runtime import Law as BaseLaw


class Law(BaseLaw):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})  # anything else is refused

    def control(self, s, ref, fleet):
        aileron = -1.5 * s.roll  # your control law here
        return RawActuators(aileron=aileron, elevator=0.0, rudder=0.0, throttle=0.6)
```

`configure(control_period_s)` is called once before the first `control()` (design
your gains there); `fleet.latest(id)` returns any aircraft's newest state for a law
that sets `needs_fleet = True`.

In C++ (a `class MyLaw : public astrum::Law` in `laws/<name>.cpp`, its factory in
`laws.hpp`, and a line in `make_law()`):

```cpp
class MyLaw : public Law {
 public:
  explicit MyLaw(const LawConfig& config) : outer_(config.limits) {}  // config.dt = 1/--rate-hz
  VehicleKind vehicle_kind() const override { return VehicleKind::FixedWing; }  // the default
  Command control(const VehicleState& s, const Reference& ref) override {
    const AttitudeTargets t = outer_.update(s, ref);  // saturated bank/pitch/throttle targets
    return Command::raw(/*aileron=*/1.5 * (t.bank_rad - s.roll) - 0.3 * s.p, 0.0, 0.0, t.throttle);
  }

 private:
  OuterLoops outer_;
};
```

The C++ runtime is the shared client in [`controllers/common/cpp`](../common/cpp) (the
same one the example uses): latest-value telemetry, exception-safe decoding, contract
QoS, `seq`/`telemetryTick`, one command per sample, `--startup-timeout` (exit 1) /
`--telemetry-timeout` (exit 0). It also exits 2 unless telemetry arrives every
`1/--rate-hz` (the sim step must divide that period, and samples must come one period
apart), because the laws are discretized for it.

`VehicleState` is SI units, decoded from telemetry v2 — including the body rates
`p`/`q`/`r` the sim measures, `alpha`/`beta`, ENU and NED velocities, vehicle kind
and control mode. `Reference` is the CLI setpoint (true heading, altitude HAE, true
airspeed). A raw control (`RawActuators` / `Command::raw`) sends actuator
deflections (you own the loop); an autopilot control (`AutopilotTargets` /
`Command::autopilot`) hands a high-level target to the sim's built-in inner loop.
The runtime sends one command per telemetry tick, saturates every channel, and
refuses (exit 2) a vehicle kind the law does not list.

## What ships

Both `lqr` and `mpc` fly out of the box. In Python they share
`laws/_fixed_wing.py`: a linear fixed-wing model (lateral `[psi_err, phi, p]`
with heading rate `(g/V) phi`, longitudinal `[int h_err, h_err, theta, q]` with
climb rate `V theta`, speed `[int v_err, v_err]`) discretized exactly (ZOH) at the
control period `1/--rate-hz`. The gains are applied with **limits**: the outer
states set a bank/pitch reference clamped to `--max-bank-deg` / `--max-pitch-deg`
(defaults 25 / 10 for the c172x trainer; the f16 takes 60 / 24), every actuator is
saturated, and the heading/altitude/airspeed integrators use conditional-integration
anti-windup (the heading integral, `0.05` rad bank per rad·s, sits outside the
design and trims out the standing aileron a real airframe needs). Unsaturated and
with the heading integral at zero, the law is exactly `u = -K x`.

- **`lqr`** — `K = dlqr(A, B, Q, R)` per channel (scipy `solve_discrete_are`).
- **`mpc`** — `K` = the first move of an unconstrained condensed MPC over
  `--horizon-s` seconds, predicting one step per control period; with an LQR
  terminal cost it equals the LQR gain (tested). The constraint hook in
  `ChannelMPC.move` is where an in-horizon QP goes.

The airframe constants and weights are placeholders to replace with your own
linearization; the point is the API and the machinery.

The **C++** laws (`laws/*.cpp`) are cascades. The shared outer loops
(`astrum::OuterLoops`, also used by the example) turn the heading error into a bank
target (`ψ̇ = (g/V) tan φ`, so PI with bank = 1.2 × heading error + 0.05 × its
integral, clamped to `±--max-bank-deg`), the altitude error into a pitch target (PI,
clamped to `±--max-pitch-deg`) and the airspeed error into throttle (PI, `[0, 1]`),
with conditional-integration anti-windup on all three integrators. The inner loop
regulates `x = [angle − target, body rate]` per axis
(roll → aileron, pitch → elevator) on the model in `include/attitude_axes.hpp`,
`rate' = −damping·rate + gain·u`, discretized exactly (ZOH) at `dt = 1/--rate-hz`:

- **`lqr`** — `u = −K x` with `K = dlqr(A, B, Q, R)` computed at startup by the Riccati
  recursion; non-convergence aborts startup (exit 1) instead of flying an unconverged
  gain. Outputs are clamped to the actuator ranges.
- **`mpc`** — a receding-horizon QP per axis over `N = round(--horizon-s · rate)` steps
  of `dt` (C++ default `--horizon-s 0.5`, i.e. 60 steps at 120 Hz) with the DARE
  solution as terminal cost, so with no active limit its first move equals the LQR
  law's exactly (tested). When the plan would exceed `|u| ≤ 1` it solves the
  box-constrained QP (accelerated projected gradient, warm-started from the previous
  plan shifted one step). The QP has `N` variables, so long horizons cost CPU.

## Formation demo

`figure8` and `formation` compose into a 4-ship, fully **decentralized** formation
— `make demo-formation` (scenario `config/scenario.formation.toml`):

- **`figure8`** flies the *leader*: pure guidance that traces a horizontal
  figure-eight (a Gerono lemniscate). It emits heading/alt/airspeed targets and
  paces its reference point by *arc length flown*, so the reference stays abreast
  of the aircraft whatever speed it actually makes good.
- **`formation`** flies each of the 3 *followers*, independently. A follower
  watches only the leader's broadcast telemetry, places its slot (left / right /
  trail) in the leader's ground-track frame, and drives its slot error to zero with
  a per-axis receding-horizon MPC (`ChannelMPC`) on the motion *relative* to the
  slot. The slot velocity and acceleration come from the leader's own velocity and
  turn rate (no finite differences), the leader sample is used at its own sim time,
  and a sample older than `--leader-max-age-s` makes the follower hold. The
  follower's acceleration (slot feedforward + MPC correction, limited to
  `g tan(--max-bank-deg)`) is split across and along its ground track: the
  cross-track part becomes the coordinated-turn bank `atan(a/g)`, sent as the
  heading offset the sim autopilot's heading loop turns into that bank (offset =
  bank / `--ap-heading-gain`, the fighter profile's 2.5 by default); the
  along-track part leads the speed command. Speed = desired ground speed, which
  assumes **no wind**. No follower talks to another and there is no central
  optimizer.

The leader carves an 8 km × 4 km pattern (lobe-end radius `Ry²/Rx` = 2 km, so
~36° of bank at the demo's 120 m/s) inside the F-16 autopilot's limits; the followers
run with `--max-bank-deg 60` to match the f16. At 120 m/s the F-16 at 4572 m has
α margin to spare; at 100 m/s it flies a 30° turn at ~16° α, and a follower on the
inside of the turn (slower) runs out of lift and sinks. Watch
it live with `make replay` in a second shell, or record it and scrub the replay
offline (`make export`, then drag the exported JSON onto the renderer — see
DEVELOP.md §7). Because the formation is fixed-wing, the replay lays a real aerial
map under the ground track (centred on the scenario datum, ~37.62 N, −122.38 E
over the SF Bay), so the figure-eight reads against actual geography — see the
ground-map note in DEVELOP.md §7.

## Run (inside the dev container)

Python needs no build:

```bash
python controllers/optimal/python/run.py --law lqr --id 1 --heading 90
python controllers/optimal/python/run.py --law mpc --horizon-s 6
python -m pytest -q controllers/optimal/python
```

C++ is part of the one CMake project rooted at `controllers/` (see
[`controllers/cpp`](../cpp/README.md)):

```bash
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j
ctest --test-dir build/controllers --output-on-failure
./build/controllers/optimal/cpp/astrum-optimal-controller --law lqr --heading 90
./build/controllers/optimal/cpp/astrum-optimal-controller --law mpc --rate-hz 120 --horizon-s 0.5
```

C++ flags: `--law lqr|mpc` (`ASTRUM_LAW`), the client and reference flags of the
example (`--connect --id --heading --alt --speed --startup-timeout --telemetry-timeout`),
`--max-bank-deg` / `--max-pitch-deg` (25 / 10; f16: 60 / 24), `--rate-hz` (`ASTRUM_RATE_HZ`,
default 120 — the aircraft's control rate: its `control_rate_hz` in the scenario, which
defaults to `sim.rate_hz`) and `--horizon-s` (mpc only).

`--help` lists every flag with its `ASTRUM_*` env fallback (precedence: **CLI arg >
`$ENV` > default**, as everywhere in ASTRUM). Start a sim in another shell first,
e.g. `cargo run -p astrum-sim -- config/scenario.demo.toml`.

## Dependencies (image rebuild)

These controllers add libraries to the dev image, so **rebuild it** (`make build`)
before first use:

- **Python:** `numpy` (linear algebra) + `scipy` (`expm`, `solve_discrete_are`).
- **C++:** `libeigen3-dev` (header-only linear algebra), `libgtest-dev` (tests),
  `nlohmann-json3-dev` (wire-fixture tool).

Both are wired into `docker/Dockerfile`.

## Adding a law

1. **Python:** drop `astrum_optimal/laws/<name>.py` exposing a `Law(BaseLaw)` with
   `vehicle_kinds` and `control()` (and `configure()` if it designs gains);
   optionally an `add_args(parser)` hook for law-specific flags. Run it with
   `run.py --law <name>`.
2. **C++:** add `laws/<name>.cpp` with a `class : public Law`, declare its factory
   in `include/laws.hpp`, wire it into `make_law()` in `src/runtime.cpp`, and add the
   source to `astrum_laws` in `CMakeLists.txt`.
