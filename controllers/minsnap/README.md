# ASTRUM min-snap race controller (C++)

`astrum-minsnap-controller --law minsnap` flies a multirotor (the F450) through the gates of
an `astrum-race` scenario. It is an ordinary SDK controller ([`controllers/README.md`](../README.md)):
the runtime owns the Zenoh session, command ownership and the lease; the law reads the gate
course and the race progress off the wire (`RaceCourse` on `astrum/race/course/<id>`,
`RaceState` on `astrum/race/state/<id>`; nothing about a gate is hard-coded) and answers
every telemetry tick with `MultirotorRaw` demands (normalised body rates and a collective).

```
RaceCourse ──► A* over gate centres ──► piecewise min-snap splines ──► flatness ─┐
 (gates)         entry / exit points      position: degree 7, snap                │ reference
                 frames + hull inflated    yaw: degree 7, acceleration            │ R, omega, u1
                                                                                  ▼
telemetry ─────────────────────────────────────────────► geometric SO(3) tracking ──► rate demands
                                                                                      + collective
```

Method: Mellinger & Kumar, *Minimum Snap Trajectory Generation and Control for Quadrotors*,
ICRA 2011. Source map (`src/`, all pure and unit-tested without sockets except `main.cpp`):

| file | |
|---|---|
| `astar.*`, `course.*` | gate hardware from the wire (`opening_m`, `bar_m`, `top_bar_m`), distance to a frame, 26-connected grid A* with frames inflated by the hull radius and a ground floor |
| `minsnap.*`, `linalg.*` | polynomial basis in nondimensional segment time, snap (any order) cost matrix, constraint assembly (pinned position / velocity / ..., C³ continuity otherwise), one equilibrated LU solve of the KKT system shared by all channels, evaluation of derivatives 0–7, section V-C segment-time optimisation |
| `planner.*` | waypoints (gate centre plus an entry and an exit point on its normal, so every crossing is square), time allocation, yaw along the gate normal, speed / acceleration / body-rate limits by temporal scaling, corridor repair |
| `flatness.*` | thrust, attitude, body angular velocity from the flat outputs and their derivatives; the paper's choice of the nearer `x_B, y_B` solution at the `z_B ∥ x_C` singularity |
| `tracking.*` | `F_des = -Kp e_p - Kv e_v - Ki ∫e_p + m g z_W + m r_T''`, `u1 = F_des · z_B`, `e_R` vee map, attitude → body-rate law, demand / collective mapping |
| `frames.*` | telemetry Euler angles (FRD) ⇄ the FLU frame of the paper, the sim's rate-demand signs |
| `race_law.*`, `main.cpp` | the `Law`: plans from the pending gate, re-plans when more than `--replan-error-m` off the reference or when the pending gate is overdue (a miss), holds before the course arrives and after the race ends; plans at most `--horizon-gates` gates and extends the plan as the aircraft nears its end |

## The airframe interface

The F450's flight control takes normalised body-rate demands (about 0.75 rad/s per unit; `--rate-gain`)
and closes the rate loop itself, so the paper's moment law becomes an attitude → rate law:
`omega_cmd = R^T R_des omega_ref - K_R e_R`. The collective follows a power law around hover
(`thrust ∝ collective^--thrust-exponent`, `--hover-collective` 0.44 as measured by the sim's own
hover-hold). Because the demand saturates at 0.75 rad/s, the planner limits the body rate it asks
for (`--max-rate`, default 0.7): that, not the speed limit, is what sets the pace on a tight course.

## Parameters

Every flag also reads an `ASTRUM_MINSNAP_*` environment variable; `--help` lists them, and
`config/controllers.toml` (`cpp-minsnap`) exposes the same set to the web studio.

| | |
|---|---|
| `--cruise-mps`, `--time-scale` | speed through the gates; stretch all times (`> 1` = slower, safer) |
| `--max-speed-mps`, `--max-accel-mps2`, `--max-rate`, `--max-yaw-rate` | limits the plan is stretched to respect |
| `--approach-m` | entry / exit point distance from the gate plane |
| `--hull-radius-m`, `--cross-radius-m`, `--floor-m`, `--cell-m` | clearance to frames (worst crab 0.353 m) and to the frame being crossed (square on 0.285 m), ground height, A* cell |
| `--horizon-gates` | gates covered by one plan; the plan is extended from its own reference when the aircraft is near its last gate (bounds planning cost on long courses) |
| `--realloc-iters`, `--refine-iters` | segment-time optimisation (off by default), corridor repair passes |
| `--kp`, `--kv`, `--ki`, `--integral-limit`, `--k-att`, `--k-yaw`, `--max-tilt-deg` | tracking gains |
| `--hover-collective`, `--thrust-exponent`, `--rate-gain` | the collective / rate-demand model |
| `--replan-error-m`, `--replan-slack-s` | re-plan triggers |

## Build and test

```
cmake -S controllers -B build/controllers && cmake --build build/controllers -j
ctest --test-dir build/controllers -R 'MinSnap|Planner|Flatness|Tracking|RaceLaw|ClosedLoop|AStar|Linalg|Frames'
```

Against the real thing: with `astrum-race` (lockstep, `--speed 0`, `controllers` line
`build/controllers/minsnap/astrum-minsnap-controller --id 1 --law minsnap`) the defaults complete
the ~500 m courses (`course.length_m`) of 8, 12, 24, 28 and 32 gates with no missed gate, at most 10 cm
from a gate centre (the opening leaves 22 cm square on). Speed is set by the body-rate limit rather than the cruise setting once legs are
short (24 gates: about 150 s; 8 gates: about 90 s). Raising the limits past `--max-rate 0.75`
(the FCS demand saturation) crashes; that is the practical ceiling of this airframe interface.

`closed_loop_test.cpp` flies the law in-process on a point-mass-plus-attitude model fed the sim's
own interface (rate demands with a first-order rate loop, collective power law, FRD telemetry).
That checks the whole chain and its sign conventions against the sim's *documented* behaviour;
it does not replace flying the real F450 in `astrum-race`.
