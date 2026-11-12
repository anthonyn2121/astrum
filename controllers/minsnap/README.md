# ASTRUM min-snap race controller (C++)

`astrum-minsnap-controller --law minsnap` flies a multirotor (the F450) through the gates of
an `astrum-race` scenario. It is an ordinary SDK controller ([`controllers/README.md`](../README.md)):
the runtime owns the Zenoh session, command ownership and the lease; the law reads the gate
course and the race progress off the wire (`RaceCourse` on `astrum/race/course/<id>`,
`RaceState` on `astrum/race/state/<id>`; nothing about a gate is hard-coded) and answers
every telemetry tick with `MultirotorRaw` demands (normalised body rates and a collective).

```
RaceCourse ──► A* over gate centres ──► waypoints ──► heuristic times ──► time optimisation ─┐
 (gates)         entry / exit points     (entry, centre, exit   (cruise speed,   (fastest lap within    │
                 frames + hull inflated   of every gate)         stretched to     the F450's limits)     │
                                                                 the limits)                             │
   piecewise min-snap splines (position: degree 7, snap; yaw: degree 7, acceleration)  ◄────────────────┘
        │ flatness: R, omega, thrust
        ▼
telemetry ───────────────────────────────────────────► geometric SO(3) tracking ──► rate demands
                                                                                      + collective
```

Methods: Mellinger & Kumar, *Minimum Snap Trajectory Generation and Control for Quadrotors*,
ICRA 2011 (the trajectories and the tracker), and Richter, Bry & Roy, *Polynomial Trajectory
Planning for Aggressive Quadrotor Flight in Dense Indoor Environments*, ISRR 2013 (choosing the
segment times). Source map (`src/`, all pure and unit-tested without sockets except `main.cpp`):

| file | |
|---|---|
| `astar.*`, `course.*` | gate hardware from the wire (`opening_m`, `bar_m`, `top_bar_m`), distance to a frame, 26-connected grid A* with frames inflated by the hull radius and a ground floor |
| `minsnap.*` | polynomial basis in nondimensional segment time, snap (any order) cost matrix, constraint assembly (pinned position / velocity / ..., C³ continuity otherwise) as one sparse KKT system (Eigen `SparseLU`, Ruiz-equilibrated, one refinement) shared by all channels; evaluation of derivatives 0–7; `SolvedProblem` keeps the factorisation for **adjoint** solves (the derivatives of any objective of the trajectory with respect to the durations and the pinned values); section V-C segment-time optimisation |
| `timeopt.*` | the time optimiser: segment durations and gate crossing speeds that minimise the lap time within the airframe's limits (below) |
| `planner.*` | waypoints (gate centre plus an entry and an exit point on its normal, so every crossing is square), the heuristic times, the time optimisation and its warm start, yaw along the gate normal, corridor repair |
| `flatness.*` | thrust, attitude, body angular velocity from the flat outputs and their derivatives; the paper's choice of the nearer `x_B, y_B` solution at the `z_B ∥ x_C` singularity |
| `tracking.*` | `F_des = -Kp e_p - Kv e_v - Ki ∫e_p + m g z_W + m r_T''`, `u1 = F_des · z_B`, `e_R` vee map, attitude → body-rate law, demand / collective mapping (with the collective clamp) |
| `frames.*` | telemetry Euler angles (FRD) ⇄ the FLU frame of the paper, the sim's rate-demand signs |
| `race_law.*`, `main.cpp` | the `Law`: plans from the pending gate, re-plans when more than `--replan-error-m` off the reference or when the pending gate is overdue (a miss), holds before the course arrives and after the race ends; plans at most `--horizon-gates` gates and extends the plan as the aircraft nears its end; logs one line per plan |

## The airframe interface

The F450's flight control takes normalised body-rate demands (about 0.75 rad/s per unit; `--rate-gain`)
and closes the rate loop itself, so the paper's moment law becomes an attitude → rate law:
`omega_cmd = R^T R_des omega_ref - K_R e_R`. The collective follows a power law around hover
(`thrust ∝ collective^--thrust-exponent`, `--hover-collective` 0.408, measured). Two properties of
the real airframe decide what the planner may ask for (both measured, see "Limits"): the rate demand
saturates at 0.75 rad/s, and the roll / pitch rate loop is unstable for collectives of about
0.70–0.97 and badly damped from 0.6 — so the collective is clamped to `--max-collective` (0.62)
and the planned thrust stays well below what that allows.

## Planning

**Waypoints.** Start, A* route points around frames where a straight line is blocked, and for every
gate an entry point (`--approach-m` upstream, on the normal), the centre and an exit point. The
velocity at every gate centre is pinned along the normal (a square crossing); its magnitude is
a free variable of the time optimisation. A plan that starts just past a gate (a re-plan) keeps
that gate's exit point as a waypoint.

**Heuristic times** (`--planner heuristic`, and the starting point of the optimiser). A leg takes its
length over `--cruise-mps`, or longer when the heading change between two gates would need a yaw rate
above the limit; the whole timeline is then stretched (section V-A temporal scaling, which divides the
pinned speeds by the same factor) until speed, `|r''|` (`--max-accel-mps2`), the roll / pitch body rate
(flatness) and the yaw rate are within `--limit-margin` of their limits. This plan is feasible but slow:
a cruise speed, not the limits, sets the pace.

**Turn-aware start.** The optimiser is a local method with a small iteration budget, and from the
heuristic plan (every gate at 6 m/s) it leaves the crossing speeds where they are: a speed moves the
lap far less per step than a duration does (measured: the gradient of the objective with respect to a
gate's speed is 10–100× smaller), so the aircraft braked to 6 m/s before every gate whatever the turn,
while the legs between peaked at 10–12 m/s. The search therefore starts from a plan that carries
speed through the gates (`turn_aware_layout` in `planner.cpp`):

* each gate's crossing speed is the *upper estimate* `turn_speed_bound(turn, reach)`, with `turn` the angle
  between the direction the aircraft arrives in and the one it leaves in, and `reach` the shorter of the
  two adjacent legs. The path through the corner is taken as a circular arc, tangent to both legs at
  `reach / 2` from the corner, radius `R = (reach / 2) / tan(turn / 2)`. With every limit at `--limit-margin`:
  lateral acceleration `a = min(g tan(tilt_max), sqrt(f_max² − g²))` gives `v ≤ sqrt(a R)`; rolling into the turn
  (thrust vector tilting by `φ = atan(v² / (g R))` at no more than the body-rate limit `ω`) takes `φ / ω`
  seconds, flown at `v` within `reach / 2`, which gives `v φ / ω ≤ reach / 2` (solved by bisection); and
  `v ≤ v_max`. Straight through, only the speed limit remains; 90° with 20 m of reach gives 9.3 m/s,
  120° gives 7.3, 90° with 10 m gives less (unit-tested, `planner_test.cpp`);
* speed can only be gained or shed at a modest 3 m/s² between gates (the thrust vector needs time to tilt),
  so no gate is faster than its neighbour plus what the leg between them gives, nor than the start plus the
  first leg (a forward and a backward pass);
* a gate's own approach segments take their length over its speed, and a leg between two gates is a
  trapezoid speed profile between the speeds at its ends (`--max-accel-mps2`, peak `limit_margin × v_max`),
  at least as long as the yaw rate needs;
* that layout breaks limits (it is an estimate, not a plan), so every leg is lengthened by its own factor until
  the sampled peaks over it respect the limits as the heuristic stretch measures them (a leg between pinned
  end speeds has lower peaks the longer it takes); this is tried at 1, 0.75 and 0.56 of the estimated
  speeds and the shortest lap that fits is kept. Where lengthening costs more than slower gates (short
  legs, tight turns) that is the 0.56 or the heuristic plan itself.

The turn-aware layout is the starting point when it is faster than the heuristic plan, the heuristic plan
otherwise; a seeded start (below) uses it for the part of the course the previous plan does not cover.
If the optimiser cannot make the turn-aware start feasible the search is repeated from the heuristic
plan, so the result is never slower or less feasible than that.

**Time optimisation** (`--planner time-optimal`, the default; `timeopt.hpp`). Variables: the log of
every segment duration and the log of the crossing speed of every gate. For any value of them the position and yaw
trajectories are the closed-form minimum-snap solution (one KKT solve each), and the objective

```
F = sum_i T_i  +  w_snap J_snap / J_snap(start)  +  mu * sum over samples, constraints of max(0, x)^2
```

has an analytic gradient by the adjoint of that solve. The constraints x ≤ 0, each sampled along every
segment at a fixed fraction of it and taken at `--limit-margin` (0.9) of its limit, are

| | via flatness |
|---|---|
| speed `|v|` | |
| thrust `|r'' + g z_W|` from above and from below (never free fall) | the thrust the airframe must produce |
| tilt, the angle between the thrust and vertical | |
| roll / pitch body rate `|omega_xy|` | needs the snap's lower derivative: jerk |
| yaw rate | |

`mu` is raised by continuation (3, 300, 3·10⁴, 10⁶), each round a projected L-BFGS with Armijo
backtracking, for a fixed iteration budget (`--opt-iterations`, 130 for a first plan; `--replan-iterations`,
20 for an extension or re-plan): no clock decides anything, so the plan is a function of its inputs and
lockstep flights are bit-reproducible. A point counts only if a dense grid (0.05 s) is within the *true*
limits (0.5 % tolerance); feasibility first, speed second, so the result is never less feasible than what
it started from, and the heuristic plan is flown when nothing faster is found.

**Corridor repair.** The sampled trajectory is checked against the hull-inflated frames and the floor; a
violating segment gets an extra waypoint at its midpoint. For the optimised plan that is done after the
optimisation, and if waypoints were added the times are optimised once more; if the result still cannot
be cleared, the heuristic plan is flown (logged: "the optimiser found nothing better").

**Continuing a plan.** An extension (the aircraft nears the last gate of its plan) or a re-plan
(too far off the reference, a missed gate) starts from the plan it replaces: a continuation pins
position, velocity, acceleration, **jerk** and yaw rate of the reference, so neither the attitude nor the
body-rate feed-forward jumps; the optimiser starts from that plan's remaining times, shared out over the
new waypoints (what the old plan does not cover is stretched as a plan from rest would be), with the
smaller budget. An extension waits until the reference is past the previous gate's exit point. A plan
from the measured state pins the acceleration the attitude implies (thrust along `z_B`, weight balanced)
and zero jerk, so it starts at the attitude the aircraft has instead of asking for a tilt it cannot take
at once.

## Limits (the F450 in the real `astrum-race`)

Measured in lockstep over Zenoh with a throwaway driver (the SysId campaign, 2026-11): hover at 3–40 m
AGL, collective steps and sweeps, rate pulses and frequency sweeps, held tilts at altitude, level-flight
drag. All limits below are the *true* limits; the optimiser takes `--limit-margin` of each.

| quantity | flag | default | basis |
|---|---|---|---|
| roll / pitch body rate | `--max-rate` | 0.75 rad/s | rate demand clips at u = 1 ↔ 0.752 rad/s; gain linear 0.1–1.0 (0.7515–0.7527 rad/s per unit, roll = pitch), 21.5 ms pure delay below ~2 Hz |
| yaw rate | `--max-yaw-rate` | 0.75 rad/s | gain 0.760 rad/s per unit; 0.62–0.66 for fast demands |
| thrust per unit mass, upper | `--max-thrust-mps2` | 16.7 m/s² (1.7 g) | T/W is 5.2–5.4 at full collective, but the rate loop is only acceptably damped up to a collective of 0.6 (ζ 0.035; 0.70 neutral, 0.72–0.97 unstable); N(0.6, 6 m/s axial flow) = 1.70 g |
| thrust per unit mass, lower | `--min-thrust-mps2` | 2.9 m/s² (0.3 g) | rate gain 0.75 and ζ 0.26–0.30 down to collective 0.2 (0.27 g); applied as ≥ f_min / margin |
| tilt | `--max-tilt-deg` | 50° | collective ≤ 0.6 holds altitude to 50° at 10 m/s (56° at 5 m/s); the tracker may tilt 5° further |
| speed | `--max-speed-mps` | 12 m/s | at 12 m/s a 50° tilt still needs only collective ≈ 0.60; drag 0.0050 v² |
| hover collective | `--hover-collective` | 0.408 | 0.4074–0.4080 over 3–40 m; was 0.44 |
| collective clamp | `--max-collective` | 0.62 | edge of the stable rate-loop band |

Thrust falls with axial inflow (−14 % at 6 m/s, −32…−44 % at 10 m/s for collective 0.6…0.5); the tracker's
integral term absorbs it. The old defaults were `--max-rate 0.6`, `--max-yaw-rate 0.8`, `--max-speed-mps 10`
and no tilt, thrust or collective limit.

## Parameters

Every flag also reads an `ASTRUM_MINSNAP_*` environment variable; `--help` lists them, and
`config/controllers.toml` (`cpp-minsnap`) exposes the same set to the web studio.

| | |
|---|---|
| `--planner` | `time-optimal` (default) or `heuristic` |
| `--max-speed-mps`, `--max-thrust-mps2`, `--min-thrust-mps2`, `--max-tilt-deg`, `--max-rate`, `--max-yaw-rate` | the limits above |
| `--limit-margin` | fraction of each limit the plans may use (0.9); the heuristic plan holds to it for speed, body rate and yaw rate, the time-optimal plan penalises beyond it and is checked against the true limit |
| `--opt-iterations`, `--replan-iterations`, `--snap-weight` | optimiser budgets (130 / 20 L-BFGS iterations) and the snap-cost weight (1e-3) |
| `--cruise-mps`, `--max-accel-mps2` | heuristic planner (and the fallback starting point of the optimiser): speed through the gates, peak `|r''|`; `--max-accel-mps2` also paces the legs of the turn-aware start |
| `--time-scale` | stretch the finished plan by this factor (`> 1` = slower, safer) |
| `--approach-m` | entry / exit point distance from the gate plane |
| `--hull-radius-m`, `--cross-radius-m`, `--floor-m`, `--cell-m` | clearance to frames (worst crab 0.353 m) and to the frame being crossed (square on 0.285 m), ground height, A* cell |
| `--horizon-gates` | gates covered by one plan (16); the plan is extended from its own reference when the aircraft is near its last gate |
| `--realloc-iters`, `--refine-iters` | heuristic segment-time optimisation (off by default), corridor repair passes |
| `--kp`, `--kv`, `--ki`, `--integral-limit`, `--k-att`, `--k-yaw` | tracking gains |
| `--hover-collective`, `--max-collective`, `--thrust-exponent`, `--rate-gain` | the collective / rate-demand model |
| `--replan-error-m`, `--replan-slack-s` | re-plan triggers |

The default horizon stays 16: the cost of a plan is roughly linear in the gates it covers, and extensions
(16 gates, 20 iterations) take about 100–150 ms (below); a plan of 32 gates costs roughly twice as much,
which is too close to the 0.5 s command lease for an extension.

## Build and test

```
cmake -S controllers -B build/controllers && cmake --build build/controllers -j
ctest --test-dir build/controllers -R 'MinSnap|TimeOpt|Planner|Flatness|Tracking|RaceLaw|ClosedLoop|AStar|Frames'
```

`closed_loop_test.cpp` flies the law in-process on a point-mass-plus-attitude model fed the sim's
own interface (rate demands with a first-order rate loop, collective power law, FRD telemetry).
That checks the whole chain and its sign conventions against the sim's *documented* behaviour;
it does not replace flying the real F450 in `astrum-race`.

## Results (real `astrum-race`, lockstep, 500 m courses, Release build of the controller)

`astrum-race config/scenario.race.toml --seed N --gates G` with `lockstep = true` on the sim and the
aircraft and `speed = 0`; controller `--law minsnap` with the defaults, `--planner heuristic` against
`--planner time-optimal` on the same seeds. "Flat 6 m/s start" is the first version of the time-optimal
planner, which started its search from the heuristic plan (every gate at 6 m/s); "legacy" is the heuristic
planner with the pre-measurement defaults (`--max-rate 0.6 --max-yaw-rate 0.8 --max-speed-mps 10
--limit-margin 1 --hover-collective 0.44 --max-collective 1 --max-tilt-deg 60`). Every run of every row
completed with no missed crossing.

| gates (runs) | legacy heuristic, mean | heuristic, mean | time-optimal, flat 6 m/s start, mean | time-optimal, turn-aware start, mean | reduction vs heuristic: mean / median / min / max | worst centre offset: heuristic / time-optimal |
|---|---|---|---|---|---|---|
| 8 (seeds 1–10) | 88.7 s | 87.7 s | 65.8 s | 59.4 s | 32.3 % / 33.9 % / 22.7 % / 36.8 % | 3 cm / 8 cm |
| 12 (1–6) | 90.3 s | 88.3 s | 69.7 s | 65.9 s | 25.4 % / 25.6 % / 17.2 % / 31.9 % | 3 cm / 7 cm |
| 24 (1–8) | 141.3 s | 136.3 s | 112.3 s | 100.2 s | 26.3 % / 26.9 % / 21.0 % / 30.3 % | 2 cm / 4 cm |
| 32 (1–2) | 174.3 s | 170.5 s | 140.1 s | 123.0 s | 27.2 % / 27.2 % / 21.6 % / 32.8 % | 2 cm / 2 cm |

Against the flat-start planner the turn-aware start is faster on every run: 9.8 % (8 gates), 5.7 % (12),
10.6 % (24) and 12.1 % (32) on average, 2.0 % at the least. (The opening leaves 22 cm of centre offset
square on; the legacy defaults reached 10 cm.) Per run:

| gates | seed | heuristic | flat 6 m/s start | turn-aware start | vs heuristic | vs flat start | worst centre offset (cm) |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 89.8 s | 64.9 s | 57.4 s | -36.1 % | -11.5 % | 6 |
| 8 | 2 | 87.4 s | 63.5 s | 56.2 s | -35.7 % | -11.4 % | 5 |
| 8 | 3 | 90.2 s | 69.2 s | 65.0 s | -27.9 % | -6.0 % | 6 |
| 8 | 4 | 90.9 s | 62.6 s | 57.5 s | -36.8 % | -8.2 % | 6 |
| 8 | 5 | 87.1 s | 70.0 s | 62.7 s | -28.1 % | -10.5 % | 6 |
| 8 | 6 | 84.9 s | 68.8 s | 65.6 s | -22.7 % | -4.7 % | 5 |
| 8 | 7 | 78.4 s | 61.5 s | 50.9 s | -35.1 % | -17.3 % | 8 |
| 8 | 8 | 80.7 s | 59.9 s | 54.4 s | -32.6 % | -9.1 % | 6 |
| 8 | 9 | 91.2 s | 66.7 s | 60.5 s | -33.7 % | -9.3 % | 6 |
| 8 | 10 | 96.4 s | 70.7 s | 63.6 s | -34.0 % | -10.0 % | 6 |
| 12 | 1 | 85.7 s | 65.7 s | 60.5 s | -29.4 % | -7.9 % | 5 |
| 12 | 2 | 84.0 s | 62.4 s | 57.2 s | -31.9 % | -8.3 % | 5 |
| 12 | 3 | 92.2 s | 74.4 s | 72.4 s | -21.5 % | -2.7 % | 6 |
| 12 | 4 | 91.5 s | 70.9 s | 63.2 s | -30.9 % | -10.9 % | 7 |
| 12 | 5 | 90.4 s | 72.3 s | 70.7 s | -21.8 % | -2.2 % | 6 |
| 12 | 6 | 86.0 s | 72.7 s | 71.3 s | -17.2 % | -2.0 % | 4 |
| 24 | 1 | 141.1 s | 112.3 s | 99.9 s | -29.2 % | -11.1 % | 2 |
| 24 | 2 | 119.8 s | 103.3 s | 94.6 s | -21.0 % | -8.4 % | 3 |
| 24 | 3 | 129.8 s | 111.3 s | 98.3 s | -24.3 % | -11.7 % | 3 |
| 24 | 4 | 142.7 s | 112.8 s | 99.5 s | -30.3 % | -11.8 % | 4 |
| 24 | 5 | 144.2 s | 123.6 s | 103.5 s | -28.2 % | -16.2 % | 3 |
| 24 | 6 | 142.7 s | 114.8 s | 103.6 s | -27.4 % | -9.7 % | 3 |
| 24 | 7 | 132.1 s | 104.9 s | 101.1 s | -23.5 % | -3.6 % | 3 |
| 24 | 8 | 137.7 s | 115.8 s | 101.3 s | -26.5 % | -12.5 % | 3 |
| 32 | 1 | 190.1 s | 149.0 s | 127.8 s | -32.8 % | -14.3 % | 2 |
| 32 | 2 | 150.9 s | 131.2 s | 118.2 s | -21.6 % | -9.9 % | 2 |

A wider sweep with the same settings (time-optimal, 20 seeds of 8 gates, 12 of 12, 16 of 24, 6 of 32):
54 of 54 completed, worst centre offset 8 cm; mean lap 58.9 / 65.8 / 99.0 / 126.8 s.

**Crossing speed against the turn** (8 gates, seed 1, from the sim's per-gate log; the turn is the angle
between the legs into and out of the gate):

| gate | turn | heuristic | time-optimal | centre offset (time-optimal) |
|---|---|---|---|---|
| 0 | – | 6.0 m/s | 7.8 m/s | 2 cm |
| 1 | 13° | 6.0 m/s | 10.8 m/s | 2 cm |
| 2 | 91° | 6.0 m/s | 10.8 m/s | 6 cm |
| 3 | 71° | 6.0 m/s | 10.8 m/s | 5 cm |
| 4 | 11° | 6.0 m/s | 10.8 m/s | 1 cm |
| 5 | 80° | 6.0 m/s | 10.8 m/s | 6 cm |
| 6 | 100° | 6.0 m/s | 10.8 m/s | 6 cm |
| 7 | – | 6.0 m/s | 10.8 m/s | 1 cm |

Gate 0 is limited by the start from rest 10 m away. The rest are at the speed ceiling (`limit_margin × v_max`
= 10.8 m/s), the turns included: on this course the legs are 67–82 m, so a 90–100° corner is a wide arc
(`turn_speed_bound` is the speed ceiling there: radius 35 m) and the tracker holds the 22 cm corridor at speed, centre offsets
of at most 6 cm. Where legs are short and the turn sharp the crossing is slower, as it should be: 30 m legs and one 90°
corner (`planner_test.cpp`) cross it at less than 0.85 of the speed of the straight course.

**Planning time** (controller, Release, one run at a time, 3 repetitions; the log line of every plan
reports it; on a loaded host the same plans take up to 1.5× as long):

| gates | first plan | extensions / re-plans |
|---|---|---|
| 8 | 264 ms (max 271) | – |
| 12 | 314 ms (max 327) | – |
| 24 | 375 ms (max 429) | median 117 ms, max 131 |
| 32 | 335 ms (max 347) | median 106 ms, max 150 |

(A first plan covers at most 16 gates, so a 24- or 32-gate course is planned as 16 gates + the rest; an
8-gate course is one plan of 8. An extension covers 16 gates with 20 iterations; if its seeded start misses
the limits by more than 2 % a second start is tried with half the budget, which is the 180–200 ms case.) The
heuristic planner takes 3–8 ms.

**Where the time goes.** Fraction of the first plan's lap spent within 3 % of the planned limit (limit ×
margin), averaged over the courses of each size (the first 16 gates of 24 and 32): speed 53 % on 8 gates
(70 m legs), 25 % on 12, 0 on 24 and 32; roll / pitch body rate 0, 4, 9 and 6 %; thrust, tilt and yaw rate
never (peaks of about 11–12 m/s² of 16.7, 25–30° of 50°, 0.3–0.55 rad/s of 0.75). So the laps are paced
by the speed limit on long legs and by acceleration between gates on short ones (an unpinned leg of 15–20 m
between two gates at 6–8 m/s accelerates only as far as the thrust vector can tilt at the rate limit
allows), not by thrust or tilt, and only rarely by the body rate itself: the 24- and 32-gate courses,
whose crossing speeds settle at 5–8 m/s, gain the least. The remaining headroom is in how well the optimiser
converges (a straight 20 m course reaches 8.5 m/s at its gates after 2000 iterations and 11 m/s from the
turn-aware start in 130), not in the airframe's limits.

## Safety notes found while tuning

* An optimiser that is free to start a plan in flight with unconstrained thrust / body rate for the
  first second uses it (a rate jump at the splice that saturated the demand and crashed the aircraft):
  continuations pin the jerk and nothing is exempt.
* A plan from the measured state that leaves its start acceleration free tilts at once; the first
  second of such a flight saturated the pitch demand. It now starts at the acceleration the attitude
  implies.
* The turn-aware start's legs are stretched by their own factors, with a root law per limit (speed 1/T,
  acceleration 1/T², body rate 1/T³). One common factor for all legs, and the linear law, over-stretched the
  slow legs (a 6× lengthened start from rest took the optimiser 130 iterations to shorten, and the laps
  were 20 % *longer* than from the heuristic plan).
* The heuristic stretch fails (×6) for a plan that starts just past a gate with a hairpin ahead, because
  a pinned start velocity is not scaled with time: the optimiser's seed there is stretched as a plan
  from rest would be, and an optimiser that cannot reach feasibility returns the least infeasible point,
  never a faster infeasible one.

## Not verified

Only lockstep flights were run (the sim waits for the controller each tick); paced real-time flight,
where the 0.5 s command lease applies to every plan, was not flown: plans of 100–750 ms are the margin.
The closed-loop unit tests use a model of the airframe, not the F450.
