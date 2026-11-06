# ASTRUM optimal-control laws (LQR / MPC / figure-eight / formation)

Control laws for developing **your own** controllers — LQR, MPC, or whatever else —
against `astrum-sim`, on the SDK runtime ([`controllers/README.md`](../README.md)). The
runtime owns everything you shouldn't have to think about: the one `mode=client` Zenoh
session to the sim, the Cap'n Proto codec, latest-value telemetry and the fleet snapshot,
events, command ownership, `epoch`/`seq`/`telemetryTick`, the vehicle-kind check, output
saturation and the timeouts. A **law** maps a `StepInput` (the vehicle state, the fleet, the
tick's events, the control period) to an `Action`, and nothing else — no Zenoh, no
Cap'n Proto, no wire.

Both languages share the same shape, so you can prototype a law in Python and port
it to C++ (or vice versa) with the same mental model.

```
controllers/optimal/
├── python/                 # the `astrum_optimal` law pack (on controllers/python's astrum_client)
│   ├── run.py              # entry point: the law table {lqr, mpc, figure8, formation} → run_main
│   ├── pyproject.toml
│   ├── tests/              # pytest: laws, MPC/LQR, formation, the law table
│   └── astrum_optimal/
│       ├── linear.py       # ZOH discretization, dlqr, ChannelMPC
│       └── laws/
│           ├── _fixed_wing.py  # linear model + limited three-channel feedback (shared)
│           ├── _guidance.py    # grid-track -> true-heading helpers (shared)
│           ├── lqr.py          # LqrLaw   (edit me)
│           ├── mpc.py          # MpcLaw   (edit me)
│           ├── figure8.py      # Figure8Law: leader guidance, fly a figure-eight
│           └── formation.py    # FormationLaw: follower, decentralized formation-keeping MPC
└── cpp/                    # astrum-optimal-controller (built from controllers/, on astrum::runtime)
    ├── include/
    │   ├── laws.hpp            # law factories + LawEntry for each law
    │   ├── attitude_axes.hpp   # per-axis models + LQR weights shared by lqr/mpc (edit me)
    │   └── optimal_math.hpp    # ZOH axis model, dlqr (reports non-convergence), box QP
    ├── src/main.cpp        # the law table {lqr, mpc} → astrum::run_main
    ├── laws/
    │   ├── lqr.cpp         # LQR law   (edit me)
    │   └── mpc.cpp         # MPC law   (edit me)
    └── tests/              # gtest: dlqr, ZOH model, box QP, law signs/limits
```

## The API

In Python, a law is a `astrum_client.runtime.Law` subclass:

```python
from astrum_client.runtime import Law, StepInput
from astrum_client.wire import Action, RawActuators, VehicleKind


class RollLeveller(Law):
    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})  # anything else is refused

    @classmethod
    def from_args(cls, args):  # args.reference / args.limits / your add_args flags
        return cls(gain=1.5)

    def __init__(self, gain: float):
        self.gain = gain

    def configure(self, control_period_s: float) -> None:
        self.dt = control_period_s  # design your gains here

    def step(self, inp: StepInput) -> Action:
        s = inp.state
        return Action(RawActuators(aileron=-self.gain * s.roll, throttle=0.6))
```

`configure(dt)` is called once, before the first `step`, with the telemetry's control
period. A law that sets `needs_fleet = True` reads any aircraft's state from `inp.fleet`
(a snapshot taken after the tick's `TickEnd`, so it holds the same tick for every aircraft);
`inp.events` carries the events of earlier ticks not yet delivered; `Action(..., trigger=True)`
fires the gun. Raise `Stop(reason)` to refuse the aircraft. Register the class in `run.py`'s
`LAWS` table.

In C++, a `class MyLaw : public astrum::Law` with a `LawEntry` handed to `run_main` (see
`src/main.cpp`):

```cpp
class MyLaw : public Law {
 public:
  explicit MyLaw(const LawContext& ctx) : outer_(ctx.limits), ref_(ctx.reference) {}
  std::vector<VehicleKind> vehicle_kinds() const override { return {VehicleKind::FixedWing}; }
  void configure(double dt) override { dt_ = dt; }  // design your gains here
  LawResult step(const StepInput& in) override {
    const VehicleState& s = in.state;
    const AttitudeTargets t = outer_.update(s, ref_);  // saturated bank/pitch/throttle targets
    return Action{RawActuators{1.5 * (t.bank_rad - s.roll) - 0.3 * s.p, 0.0, 0.0, t.throttle}};
  }

 private:
  OuterLoops outer_;
  Reference ref_;
  double dt_ = 0.0;
};
```

The runtime is the SDK in [`controllers/common/cpp`](../common/cpp) — the same one the
example uses: latest-value telemetry, exception-safe decoding, ownership, one command per
sample, `--startup-timeout` / `--telemetry-timeout`, and the exit codes of the
[contract](../README.md#outcomes-and-exit-codes). The control period comes from the
telemetry (`controlPeriodS`, the aircraft's `control_rate_hz`), so there is no rate flag to
get wrong.

`VehicleState` is SI units, decoded from telemetry v3 — including the body rates
`p`/`q`/`r` the sim measures, `alpha`/`beta`, ENU and NED velocities, vehicle kind,
control mode and command status. `Reference` is the CLI setpoint (true heading, altitude
HAE, true airspeed). A raw control (`RawActuators`, or `MultirotorRaw` for a multirotor)
sends actuator commands (you own the loop); an autopilot control (`AutopilotTargets`) hands a
high-level target to the sim's built-in inner loop. The runtime sends one command per
telemetry tick, saturates every channel, and refuses (exit 2) a vehicle kind the law does not
list.

## What ships

Both `lqr` and `mpc` fly out of the box. In Python they share
`laws/_fixed_wing.py`: a linear fixed-wing model (lateral `[psi_err, phi, p]`
with heading rate `(g/V) phi`, longitudinal `[int h_err, h_err, theta, q]` with
climb rate `V theta`, speed `[int v_err, v_err]`) discretized exactly (ZOH) at the
control period (the telemetry's `controlPeriodS`). The gains are applied with **limits**: the outer
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
`rate' = −damping·rate + gain·u`, discretized exactly (ZOH) at the control period `dt`:

- **`lqr`** — `u = −K x` with `K = dlqr(A, B, Q, R)` computed in `configure()` by the Riccati
  recursion; non-convergence ends the run (law fault, exit 3) instead of flying an
  unconverged gain. Outputs are clamped to the actuator ranges.
- **`mpc`** — a receding-horizon QP per axis over `N = round(--horizon-s / dt)` steps
  of `dt` (C++ default `--horizon-s 0.5`, i.e. 60 steps at 120 Hz) with the DARE
  solution as terminal cost, so with no active limit its first move equals the LQR
  law's exactly (tested). When the plan would exceed `|u| ≤ 1` it solves the
  box-constrained QP (accelerated projected gradient, warm-started from the previous
  plan shifted one step). The QP has `N` variables, so long horizons cost CPU.

## Formation demo

`figure8` and `formation` compose into a 4-ship, fully **decentralized** formation
— `astrum run config/demos/formation.toml --watch` (scenario `config/scenario.formation.toml`):

- **`figure8`** flies the *leader*: pure guidance that traces a horizontal
  figure-eight (a Gerono lemniscate). It emits heading/alt/airspeed targets and
  paces its reference point by *arc length flown*, so the reference stays abreast
  of the aircraft whatever speed it actually makes good.
- **`formation`** flies each of the 3 *followers*, independently. A follower
  reads only the leader's broadcast telemetry (from the fleet snapshot), places its slot (left / right /
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
it live with `--watch` (http://localhost:8080), or record it and scrub the replay
offline (`astrum export <bag> --out <file>`, then drag the exported JSON onto the renderer — see
DEVELOP.md §7). Because the formation is fixed-wing, the replay lays a real aerial
map under the ground track (centred on the scenario datum, ~37.62 N, −122.38 E
over the SF Bay), so the figure-eight reads against actual geography — see the
ground-map note in DEVELOP.md §7.

## Run (inside the dev container)

Python needs no build:

```bash
python controllers/optimal/python/run.py --law lqr --id 1 --heading 90
python controllers/optimal/python/run.py --law mpc --id 1 --horizon-s 6
python -m pytest -q controllers/optimal/python
```

C++ is part of the one CMake project rooted at `controllers/` (see
[`controllers/cpp`](../cpp/README.md)):

```bash
cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
cmake --build build/controllers -j
ctest --test-dir build/controllers --output-on-failure
./build/controllers/optimal/cpp/astrum-optimal-controller --law lqr --id 1 --heading 90
./build/controllers/optimal/cpp/astrum-optimal-controller --law mpc --id 1 --horizon-s 0.5
```

Flags: the [shared ones](../README.md#flags) (`--law`, `--id` (required), `--connect`,
`--heading`, `--alt`, `--speed`, `--max-bank-deg` / `--max-pitch-deg` (25 / 10; f16: 60 / 24),
`--startup-timeout`, `--telemetry-timeout`) plus each law's own: `--horizon-s` (mpc,
`ASTRUM_MPC_HORIZON_S`), and in Python `figure8`'s `--fig-*` and `formation`'s `--leader`,
`--slot*`, ….

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

1. **Python:** write a `Law` subclass (in `astrum_optimal/laws/<name>.py`, or anywhere on
   your `PYTHONPATH`) with `vehicle_kinds`, `from_args`, `step` (and `configure()` if it
   designs gains, `add_args(parser)` for its own flags), and add it to `LAWS` in `run.py` —
   or hand your own table to `astrum_client.runtime.run_main` from your own script.
2. **C++:** add `laws/<name>.cpp` with a `class : public Law` and a `LawEntry` (declared in
   `include/laws.hpp`), add the entry to the table in `src/main.cpp`, and the source to
   `CMakeLists.txt` — or build your own program against an installed SDK
   ([`controllers/examples/out_of_tree`](../examples/out_of_tree)).
