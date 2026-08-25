# ASTRUM optimal-control controllers (LQR / MPC)

A clean-API scaffold for developing **your own** control laws — LQR, MPC, or
whatever else — against `astrum-sim`. Unlike the minimal
[`controllers/cpp`](../cpp) / [`controllers/python`](../python) examples (one file,
one hard-coded hold law), this splits the code in two:

- a **runtime** that owns everything you shouldn't have to think about — the one
  `mode=peer` Zenoh session, the Cap'n Proto codec, decoding telemetry into a
  clean state object, body-rate estimation, clamping, and publishing;
- **laws** — the only files you write. A law maps a `VehicleState` + `Reference`
  to a `Command`, and nothing else. No Zenoh, no Cap'n Proto, no wire.

Both languages share the same shape, so you can prototype a law in Python and port
it to C++ (or vice versa) with the same mental model.

```
controllers/optimal/
├── python/
│   ├── astrum_control.py   # the runtime (Zenoh + Cap'n Proto + state decode)
│   ├── run.py              # entry point: `--law <name>` selects a law
│   └── laws/
│       ├── lqr.py          # LQR law   (edit me)
│       ├── mpc.py          # MPC law + reusable ChannelMPC (edit me)
│       ├── figure8.py      # leader guidance: fly a figure-eight
│       └── formation.py    # follower: decentralized formation-keeping MPC
└── cpp/
    ├── include/            # astrum_control.hpp (clean API), laws.hpp, optimal_math.hpp
    ├── src/runtime.cpp     # the runtime + main + law factory
    └── laws/
        ├── lqr.cpp         # LQR law   (edit me)
        └── mpc.cpp         # MPC law   (edit me)
```

## The API

You implement one method. In Python:

```python
from astrum_control import Command, Law as BaseLaw, Reference, VehicleState


class Law(BaseLaw):
    def control(self, s: VehicleState, ref: Reference) -> Command:
        aileron = -1.5 * s.roll  # your control law here
        return Command.raw(aileron=aileron, elevator=0.0, rudder=0.0, throttle=0.6)
```

In C++ (a `class Law : public astrum::Law` in `laws/<name>.cpp`, plus a factory
line in `laws.hpp` / `make_law()`):

```cpp
Command control(const VehicleState& s, const Reference& ref) override {
    return Command::raw(/*aileron=*/-1.5 * s.roll, 0.0, 0.0, 0.6);
}
```

`VehicleState` is SI units, decoded from telemetry (body rates `p`/`q`/`r` are
**estimated** by finite-differencing attitude — the stream carries none).
`Reference` is the CLI setpoint (heading/altitude/airspeed). `Command::raw(...)`
sends actuator deflections (you own the loop); `Command::autopilot(...)` hands a
high-level target to the sim's built-in inner loop instead.

## What ships

Both `lqr` and `mpc` are **runnable skeletons** that fly out of the box:

- **`lqr`** — full-state feedback `u = -K(x - x_ref)`, decoupled into lateral /
  longitudinal / speed channels. The shipped gains are hand-tuned to reproduce the
  proven example hold law; the file shows exactly how to replace them with the
  `dlqr(A, B, Q, R)` solution for your own linearized model.
- **`mpc`** — receding-horizon control on the same channels. The shipped model
  flies so you have a working loop; the file marks where to drop in your model and
  where the **constraint hook** goes (the reason to prefer MPC over LQR — hard
  bank/actuator limits, which need a QP solver like OSQP or cvxpy).

These are starting points, not tuned autopilots. The point is the API and the
machinery; the control theory is yours to fill in.

## Formation demo

`figure8` and `formation` compose into a 4-ship, fully **decentralized** formation
— `make demo-formation` (scenario `config/scenario.formation.toml`):

- **`figure8`** flies the *leader*: pure guidance that traces a horizontal
  figure-eight (a Gerono lemniscate). It emits heading/alt/airspeed targets and
  paces its reference point by *arc length flown*, so the ground track keeps its
  shape at whatever speed the sim's throttle loop settles at.
- **`formation`** flies each of the 3 *followers*, independently. A follower
  watches only the leader's broadcast telemetry, places its slot (left / right /
  trail) in the leader's heading frame, and drives its slot error to zero with a
  per-axis receding-horizon MPC (`ChannelMPC`, shared from `mpc.py`). No follower
  talks to another and there is no central optimizer — the only coupling is that
  one broadcast state.

In a full JSBSim run the leader holds an 18 km × 9 km pattern at ~19° bank (under
the sim's 25° limit) and the followers capture in ~5 s, then hold their slots to
~20 m (≈35 m at the tightest lobe tip). Watch it live with `make replay` in a
second shell, or record it and scrub the replay offline (`make export`, then drag
the exported JSON onto the renderer — see DEVELOP.md §7).

## Run (inside the dev container)

Python needs no build:

```bash
python controllers/optimal/python/run.py --law lqr --id 1 --heading 90
python controllers/optimal/python/run.py --law mpc --horizon 30
```

C++ builds with CMake (mirrors `controllers/cpp`):

```bash
cmake -S controllers/optimal/cpp -B controllers/optimal/cpp/build
cmake --build controllers/optimal/cpp/build -j
./controllers/optimal/cpp/build/astrum-optimal-controller --law lqr --heading 90
```

`--help` lists every flag with its `ASTRUM_*` env fallback (precedence: **CLI arg >
`$ENV` > default**, as everywhere in ASTRUM). Start a sim in another shell first,
e.g. `cargo run -p astrum-sim -- config/scenario.demo.toml`.

## Dependencies (image rebuild)

These controllers add libraries to the dev image, so **rebuild it** (`make build`)
before first use:

- **Python:** `numpy` (linear algebra) + `scipy` (`solve_discrete_are` behind
  `dlqr`).
- **C++:** `libeigen3-dev` (header-only linear algebra).

Both are wired into `docker/Dockerfile`.

## Adding a law

1. **Python:** drop `laws/<name>.py` exposing a `Law(BaseLaw)` with `control()`;
   optionally an `add_args(parser)` hook for law-specific flags. Run it with
   `run.py --law <name>`.
2. **C++:** add `laws/<name>.cpp` with a `class : public Law`, declare its factory
   in `include/laws.hpp`, wire it into `make_law()` in `src/runtime.cpp`, and list
   the source in `CMakeLists.txt`.
