# Design: drone-racing environment for the DJI F450

**Status:** the course generator, the gate model, gate scoring, crash termination, the
race wire messages and events, and the scenario/CLI/recording integration are implemented
and tested; a scripted controller flies gate 0 in lockstep. A course-flying controller is
the next piece of work (§6).

**Crate:** `crates/astrum-race` — a *client* of `astrum-sim`: its binary and library build the
sim's own app and add the race to it. A race is an ordinary `astrum-sim` scenario with a
`[course]` table.

---

## 1. Why a client of `astrum-sim`

An earlier design made this a separate app with its own spawn, step, pose, pacing, Zenoh
telemetry and command code and a stabilising inner loop. That duplicated, and would have had
to keep tracking, everything the sim grew around the tick: command ownership and leases,
lockstep, pause/step, recording, the control plane, the multirotor FCS mapping. A race now
**runs on the sim**:

| Comes from `astrum-sim` (`build_app`) | Added by `astrum-race` |
| --- | --- |
| fleet spawn from the scenario, JSBSim stepping, `AircraftState` | the course and the scenario `[course]` table |
| telemetry, events, `TickEnd`, recording to MCAP | gate scoring, crash detection (the hull), the time-limit marker |
| command ownership, leases, lockstep, pause/step, speed, the control plane | `RaceCourse` / `RaceState` publication and the race events |
| the multirotor hover-hold and body-rate FCS, pacing, exit handling | the end of the run as `RaceComplete` / `Crashed` |

The cost is one direction of dependency — `astrum-race` depends on `astrum-sim` (and so
`bevy`, `astrum-fdm`, …) — so `astrum-sim` knows nothing about racing: its only concession is
`TickSet` (§5), two empty system sets. The pure math (gate, course sampler, hull, segment
geometry, RNG) is untouched by this and still depends on nothing but Bevy's maths. The web
studio and the sweep runner treat a race as an ordinary scenario with a `[course]`.

---

## 2. The gate

One gate is three pieces of 3-inch stock: two 40-inch uprights carrying a 46-inch top
bar, giving a **40-inch square opening**.

```text
           <------------ 46" ------------->         top bar (46 x 3)
          +----------------------------------+  --+
          |###                            ###|    | 3"
          |###                            ###|  --+
          |###                            ###|
          |###          40" x 40"         ###|    upright (40 x 3)
          |###          opening           ###|    40" tall
          |###                            ###|
          |###                            ###|
          +###----------------------------###+  --- sill (open below)
           <3">        <-- 40" -->
```

Everything derives from **one constant**, `OPENING_IN = 40.0` (plus the stock width
`BAR_IN = 3.0`), converted at `IN_TO_M = 0.0254`:

```
OPENING_M = OPENING_IN * IN_TO_M          UPRIGHT_M = OPENING_M
TOP_BAR_M = OPENING_M + 2 * BAR_M         (46" = 40" + 2 x 3")
```

Resizing the gate is a one-constant change: the top bar, outer frame, `frame_radius_m()`,
the sampler's sill check and the scoring all follow. Dimensions are held in inches
because that is what makes the identity exact — deriving from rounded centimetre figures
would leave the top bar short of the frame it has to span.

Two properties worth stating because they are easy to get wrong:

* **The frame is not symmetric about the opening.** There is a bar above and nothing
  below, so the lowest hardware is the *sill* at the bottom of the opening, and
  `frame_radius_m()` — the bounding sphere the course sampler clears legs against —
  is set by a **top** corner of the top bar.
* **The gate-local basis is left-handed.** Local coordinates are `x` = right, `y` =
  up, `z` = along the normal (positive downstream), and `x` is *starboard of the
  flight direction*, so the right-handed cycle is `(normal, up, right)` — a
  camera-style arrangement. Nothing takes a cross product in local coordinates;
  anything that starts to should know which it is. The world-ENU vectors of that basis
  are `Gate::normal()`, `Gate::right()` and `Gate::up()` (for yaw `ψ`:
  `(sin ψ, cos ψ, 0)`, `(cos ψ, −sin ψ, 0)`, `(0, 0, 1)`), and these — not a re-derived
  convention — are what goes on the wire.

A gate's pose is a centre (the middle of the **opening**, not of the hardware) plus a
yaw, using the sim's convention: `psi = 0` faces North, `+psi` turns East. Gates stand
upright — no bank, no tilt.

### Why 40", not the brief's 25" and not 31"

The original brief fixed the opening at 25" (0.635 m). **That size makes the course
unflyable rather than merely hard**, and it was widened to 31" on 2026-09-08 at the
user's request ("we can make the gates slightly larger so that we don't have any
collisions on turns out of the gate"). 31" was then widened to 40". The reasoning,
because the numbers are not arbitrary:

An F450 is **0.569 m across its prop tips** but **0.7057 m corner to corner** (§4b), and
it is the corner a gate sees whenever the drone is not square to the opening. Lateral
reach as a function of crab angle `ψ` off the normal is `ARM·(|sin ψ| + |cos ψ|) + r`:

| crab off the normal | half-reach | opening it needs |
|---|---|---|
| 0° (square) | 0.28448 m | 22.4" |
| 13° | 0.31739 m | **25.0"** |
| 20° | 0.33099 m | 26.1" |
| 30° | 0.34491 m | 27.2" |
| **45° (worst case)** | **0.35287 m** | **27.8"** |
| 60° | 0.34491 m | 27.2" |
| 90° | 0.28448 m | 22.4" |

Two things follow. First, a 25" opening caps the approach at **±13°** — while §3.3 gives a
gate's normal the *bisector* of a turn up to 120°, so a crossing can be up to 60° off. The
hardware and the envelope contradicted each other: a controller could track a gate's
centre perfectly and still be flown into an upright. Second, because the curve peaks at
45° and is nearly flat from 30–60°, **there is no gradual relief** — 27" would only buy
±28°. The constraint does not soften, it switches off, at 27.78".

31" cleared that threshold, but with only 4.08 cm to spare at the worst heading — less
than a controller that is merely good will hold, so a decent run still scored as a
crash. 40" is where the margin stops being the thing a controller fights. All three
sizes, every number computed from the constants (`gate.rs`, `airframe.rs`) and held as
assertions in `tests/airframe.rs` and `tests/gate.rs`:

| | 25" (brief) | 31" | **40" (built)** |
|---|---|---|---|
| opening | 0.6350 m | 0.7874 m | **1.0160 m** |
| top bar (opening + 2 x 3") | 0.7874 m | 0.9398 m | **1.1684 m** |
| outer height (opening + 3") | 0.7112 m | 0.8636 m | **1.0922 m** |
| clearance, square on | 3.30 cm | 10.92 cm | **22.35 cm** |
| clearance, worst heading (45°) | −3.54 cm (a strike) | 4.08 cm | **15.51 cm** |
| headroom under the top bar | 17.31 cm | 24.93 cm | **36.36 cm** |
| CG offset below the opening centre at which a blade clears the uprights | −46.19 cm | −53.81 cm | **−65.24 cm** |
| legal crab | ±13° | any | **any** |
| `frame_radius_m()` | 0.5568 m | 0.6645 m | **0.8262 m** |
| sill height at `alt_min_m` = 1.5 m | 1.183 m | 1.106 m | **0.992 m** |

(Clearance = half-opening − reach; headroom = half-opening − rotor-plane height −
blade radius; "below the sill" is the CG height at which a blade tip just clears the
bottom of an upright.)

The bigger bounding sphere costs the course sampler nothing measurable: 0 failures in
5000 seeds at each of the shipped envelope, the dense 12-gate/45 m one, and a crowded
16-gate/40 m one at the shipped clearances (`tests/course.rs`), with the 16/40 worst
case using 68 of its 200 backtracks. The shipped envelope defaults did not need
adjusting.

What did **not** change, and still matters to a controller:

* **The flyable corridor is narrower than the hole.** The scorer credits a pass on
  where the *CG* pierced the plane, anywhere in ±0.508 m; the props reach 0.28448 m
  either side of it, so anything past ±0.22352 m is a crash. The outer 0.28 m of each
  side of the opening is a trap, and `a_crash_beats_a_pass_on_the_same_tick` is exactly
  that case.
* **The headroom is 0.364 m, not the 0.508 m the opening looks like it offers**, because
  what reaches the top bar is a blade tip at the rotor plane, 0.144 m out and up from the
  CG.
* **Heading still costs corridor**, just far less of it: 15.51 cm at the worst heading
  against 22.35 cm square on.

---

## 3. Course generation (the Monte-Carlo part)

`course.rs`. Gates are placed one at a time by **rejection sampling with
backtracking**: draw a turn, a leg length and a target altitude; test the resulting
gate against every clearance rule; redraw on rejection.

### 3.1 The envelope

`CourseParams`, deserialized from `[course]` in the scenario TOML. Defaults:

| Parameter | Default | What it protects |
| --- | --- | --- |
| `gates` | 8 | course length |
| `leg_min_m` / `leg_max_m` | 15 / 30 | room to line up on the next gate |
| `max_turn_deg` | 120 | crossing angle (see §3.3) |
| `max_climb_deg` | 20 | flight-path angle of a leg |
| `alt_min_m` / `alt_max_m` | 1.5 / 4.0 | flight volume, sill above ground |
| `min_gate_gap_m` | 10.0 | gates not on top of each other |
| `min_leg_gate_gap_m` | 1.5 | a leg must not thread a frame it isn't flying through |
| `min_leg_leg_gap_m` | 2.0 | non-adjacent legs must not cross |
| `course_radius_m` | 100 | horizontal containment |
| `tries_per_gate` | 200 | draws before backtracking |
| `max_backtracks` | 200 | total unwinding before "infeasible" |
| `start_heading_deg` | 0 | entry direction, fixed so it is repeatable across seeds |

`validate()` rejects contradictory envelopes up front (`CourseError::Params`) so a
bad input is a clear message rather than an exhausted sampler: inverted ranges, a
`min_gate_gap_m` wider than `leg_min_m` (which would reject every candidate, since
consecutive gates are exactly one leg apart), an `alt_min_m` that would put the sill
underground, a `course_radius_m` smaller than one leg.

**Choosing the length: `length_m`.** The scenario layer (not the sampler) adds one more
`[course]` key, `length_m`, the target total length of the gate-to-gate legs (3-D). When it
is set, every envelope field the file does **not** state is derived from it and the gate
count `g` (`CourseSpec::effective_params`, applied by `RaceScenario::resolve` after any
`--gates` / web `gates` override has set `course.gates`):

| Derived field | Value |
| --- | --- |
| `leg_min_m` | `0.7 · L / (g − 1)` |
| `leg_max_m` | `1.3 · L / (g − 1)` |
| `course_radius_m` | `max(0.5 · L, 100)` |

A field the file states always wins. Changing the gate count therefore keeps the course
about `L` long and changes the leg length instead; without `length_m` the envelope is used
exactly as written. The legs are drawn uniformly over ±30% of the mean, so the realised
length has a standard deviation of ~17%/√(g−1) of `L` — over 200 seeds the worst error is
14% at 8 gates, 15% at 12 and 9% at 24 (`tests/length.rs`, which bounds it at 20%).
`CourseView.length_m` reports the realised length of the actual gates.

Infeasible combinations fail with a message naming `course.gates` and `course.length_m`
rather than exhausting the sampler: fewer than 2 or more than 64 gates; derived
`leg_min_m` below `min_gate_gap_m` (too many gates for the length: 64 gates over 500 m is
7.9 m a leg against a 10 m gap); a derived `leg_max_m` longer than the course radius
(too few: 3 gates over 500 m).

### 3.2 Two rules hold by construction, not by rejection

The **turn** is drawn inside its own limit. The **altitude** is drawn from the
intersection of the altitude band and what the climb limit permits over the leg just
drawn:

```
z ~ U( max(alt_min, z0 - L*sin(gamma_max)),  min(alt_max, z0 + L*sin(gamma_max)) )
```

Rejecting on these instead would waste samples, and — the reason it matters — would
make a **zero-width** altitude band (a deliberately flat course) look infeasible. The
RNG's `uniform(lo, hi)` returns `lo` without consuming a draw when `hi <= lo`, so a
flat course and a zero turn limit are both legal and both cost no randomness.

### 3.3 Gate orientation: the normal bisects the turn

A gate's yaw is the **bisector** of its incoming and outgoing leg headings (summed as
unit vectors, so it is wrap-safe), which is the orientation a pilot would set. The
consequence is the one that makes `max_turn_deg` mean something: a turn of `T` at a
gate is crossed within `T/2` of the normal, so the 120° default guarantees no crossing
is more than **60°** off the opening's normal. End gates have one leg and face along
it.

### 3.4 Backtracking, and the bug worth remembering

On a dead end the sampler unwinds and re-draws. Undoing only the last gate assumes
that gate is what boxed the track in — usually true, but when a *pair* of gates has
closed off the volume, re-drawing just the last one samples the same trap until the
budget is gone. So the unwind gets **deeper the longer the sampler stays stuck**
(`BACKTRACK_ESCALATION`).

The subtlety: "stuck" is measured against the **high-water mark of gates placed**, not
against the last successful placement. Counting any success as progress — including
the one that merely re-fills the gate just popped — pins the unwind at one gate
forever. That is not hypothetical: it is exactly why seed 60 of the dense
12-gate/45 m envelope wedged at 11 of 12 gates, and
`a_dead_end_deeper_than_one_gate_is_escaped` in `tests/course.rs` exists to keep it
fixed.

Generation is fallible **before the app is built** (`RaceScenario::resolve -> Result`), so an
unsatisfiable envelope is an error the caller sees, not a log line from inside a
startup system.

---

## 4. Scoring a flythrough

`systems::track_gates`. A gate counts when the drone crosses its plane **inside the
opening**, **downstream**, and gates must be taken **in order**.

The crossing is tested on the **segment** from `prev_pos_enu` to `pos_enu` (the sim keeps
the position before each step), in the
gate's local frame — not by asking whether a point is inside the frame. At racing speed
a tick covers metres while the frame is 7 cm thick, so a point-in-volume test would
step clean over the gate. Concretely: look for the sign change of local `z`, solve for
the fraction `t` where it crosses, and test the interpolated hit point against the
opening.

A crossing outside the opening but within `MISS_RADIUS_M` (3 m, a few gate-widths) is
counted as a **miss** — "went round the outside" is a pilot's miss. Further out than
that it is unrelated flying and is not scored at all. A miss does **not** advance the
course; the gate stays pending and can be re-attempted.

Each pass is announced as a `GatePassed` event carrying the CG's offset in the gate frame, each
miss as `GateMissed`.

---

## 4b. The airframe, and how a run ends

`airframe.rs`, `systems.rs::detect_crash`. A run ends in exactly one way: some system
writes a `RaceOutcome` onto `RaceProgress` via `finish()`, and the sim's own end-of-run
machinery (§5) turns that into the app's single `AppExit` and its `EndReason`.

| `RaceOutcome` | Set by | Meaning |
| --- | --- | --- |
| `Running` | — | still flying (the default) |
| `Complete` | `track_gates` | every gate passed, in order — the only way to win |
| `CrashedGround` | `detect_crash` | the hull reached the ground plane |
| `CrashedGate` | `detect_crash` | the hull struck a gate's frame |
| `TimedOut` | `time_limit` | `sim.max_time_s` expired with gates to fly |
| `SpawnFailed` | `fail_spawn` | the sim could not spawn the drone — *our* fault, not the pilot's |

`finish()` is **first-writer-wins**, which is not tidiness: it decides races between
terminators on the same tick, and the order it enforces is the one a race would judge by.
`detect_crash` runs *before* `track_gates`, so clipping an upright while the CG crosses
inside the opening is a **crash, not a pass** — which is not a corner case, because the
flyable corridor is ±0.224 m inside a ±0.508 m hole (§2).

### The hull is five parts, because the worst-heading margin is the budget

Every dimension comes from the shipped model files, not from a datasheet:

| Constant | Value | Source |
| --- | --- | --- |
| `ARM_M` | 0.1651 | `Propulsion.xml` — four thrusters at (∓0.1651, ±0.1651, +0.025) from the CG |
| `ROTOR_PLANE_M` | 0.025 | same, the `z` of those offsets |
| `PROP_DIAMETER_M` | 9.4 in | `engine/DJI_9450.xml` |
| `GEAR_SPAN_M` / `GEAR_DEPTH_M` | 0.1016 / 0.1524 | `Gear.xml` contacts |

Note `Metrics.xml`'s `wingspan 0.127` is **not** a dimension — it is a fake aero
reference (`0.127² = wingarea 0.016129`), and reading it as a span would make the drone
five times too small.

The hull is **four rotor spheres on the arm diagonals plus one core box** from the gear
feet to the rotor plane. The budget it has to be accurate inside is the **15.5 cm** at the
worst heading, not the 22.4 cm square on, and neither single-shape alternative is:

* **A box across the tip circle** presents a 0.40232 m diagonal when crabbed. In the 31"
  gate that exceeded the 0.3937 m half-opening, so the box would have handed the whole
  enlargement straight back; at 40" it fits the 0.508 m half-opening. It is still wrong
  vertically: 0.371 m of claimed reach at 45° bank / 30° pitch against the hull's real
  0.318 m above the CG (`a_single_box_overclaims_banked_reach`), which shrinks the
  vertical corridor at that attitude from 19.0 cm to 13.7 cm.
* **A sphere on the CG** needs the tip circle, `ARM·√2 + PROP_RADIUS = 0.35287 m`. That
  fits the opening, but it is round and the airframe is not: it reaches 0.353 m *below*
  the CG where the gear stops at 0.152 m (20.0 cm of error), so a drone hovering 0.30 m
  up would be a ground crash, and it cuts the headroom under a top bar from 36.4 cm to
  15.5 cm (20.8 cm of error). Both exceed the whole worst-heading margin.

(In the 25" gate both alternatives were impossible outright — the sphere and the box were
each wider than the half-opening. Enlarging the gate did not weaken the case for an honest
hull; it moved the argument from *impassable* to *wrong by centimetres against a
15.5 cm budget*. The honest hull stays because it costs nothing and cannot be wrong in
either direction, not because the course needs it to be passable.)

With the parts model the worst-case over-report is 2.5 cm (a prop modelled as a ball
rather than a disc, at compound attitudes), and 0.9 mm at the one marginal attitude. That
is what makes `banking_and_pitching_still_fit` pass at every roll angle and every sane
pitch.

### Substepping, not sweeping

One test per tick is not enough: at 20 m/s a 120 Hz step covers 0.17 m against 0.0762 m
of frame stock, so the drone would teleport through gates. The usual fix — inflating the
hull to cover the step — would add 8 cm at that speed, twice the worst-heading clearance,
which the budget cannot afford. So `detect_crash` subdivides the step instead and tests the **true** hull at each
sample, at most `SAMPLE_STEP_M = 0.15 m` apart.

That cannot tunnel: tunnelling needs a gap wider than the stock (0.0762 m) plus the
smallest hull part's own thickness (a 0.2388 m rotor), and 0.15 m is under half of that
0.315 m. What it *can* miss is a graze lasting less than 0.15 m of travel. Attitude is
held across the samples; at 120 Hz it moves a degree or two.

### The ground rules

Ground contact is **any** contact — `hull.min_z() <= 0.0`, no tolerance constant and no
impact-speed threshold. The racing floor is the ENU origin plane, which is also JSBSim's
terrain in the shipped scenario (`world.origin_alt_hae_m = 0`, which a race requires), so this fires exactly as the
gear feet touch down; the real-FDM test confirms JSBSim logs its own `GEAR_CONTACT` on
the same step. Touching down gently is still a crash — this is a race, and a drone that
put its legs on the ground is out of it. A speed threshold would be a one-line change,
but also a number to tune and a way to game the ending.

Note it is the lowest part of the *hull*, not the CG: level, the feet are 0.1524 m below
the CG; rolled onto its side, an arm reaches 0.284 m down.

Gate strikes are judged against **every** gate, not just the pending one — you can very
easily clip the gate you have already passed, or the next one round a tight corner. The
opening is a hole and nothing hangs below the sill, so flying through or underneath a gate
touches nothing.

---

## 5. The app

`build_race_app(ResolvedRace, RunMode)` is two lines of idea: build the sim's own app
with `astrum_sim::build_app`, then add the race's resources (`RaceCourse`,
`RaceProgress`) and systems. Nothing about flying, commanding or publishing the drone is
re-implemented — the spawn, the FDM step, command ownership and the lease, lockstep,
pause/step, telemetry, events, recording, pacing and the control plane are the sim's.

`astrum-sim` exposes exactly one extension point for this, `TickSet`:

| Set | Schedule | Where |
| --- | --- | --- |
| `TickSet::Spawned` | `Startup` | after `spawn_fleet`, before tick 0 is published |
| `TickSet::Stepped` | `Update` | after `end_on_elimination`, before `publish_telemetry` — i.e. after `step_fdm` and `advance_clock` — under the tick's own conditions (not paused, not fatal) |

The race's tick, `systems::tick_systems()`, is one chain in `Stepped`:

```
detect_crash → track_gates → time_limit → end_race → publish_race
```

The order is the invariant. Everything that can end the race is judged against the pose
the sim's `step_fdm` just produced (`AircraftState::{prev_pos_enu, pos_enu}` and the
attitude), `detect_crash` ahead of `track_gates` (§4b). Every terminator only *marks* an
outcome; `end_race` turns it into the sim's own end of run (`EndReason`) and `publish_race`
runs last, so the wire carries the outcome that ended the run and the events the race
pushed are stamped and published with the tick's telemetry by the sim's `publish_events`.
Because `Stepped` sits before `publish_telemetry`, a lockstep controller always sees a
tick's `RaceState` before it sees that tick's telemetry.

Startup (`Spawned`): log the course, publish `RaceCourse` and the initial `RaceState` (so a
controller already listening has the course with tick 0), and — if the sim could not spawn
the drone — mark the race `SpawnFailed` (the sim itself ends the run as `Fatal`).

How a race ends:

| Race outcome | Run ends as (`EndReason`) | By |
| --- | --- | --- |
| `Complete` | `RaceComplete` | `end_race` |
| `CrashedGround` / `CrashedGate` | `Crashed` | `end_race` |
| `TimedOut` | `MaxTime` | the sim's `stop_at_max_time`; `time_limit` marks the outcome in the same tick |
| `SpawnFailed` | `Fatal` | the sim's `exit_on_fatal` |

In `RunMode::Embedded` (tests, an RL env) nothing exits: the caller reads `RaceProgress`.
Once the race is decided the scoring systems stand down (first writer wins) but the sim
keeps stepping until the caller stops it.

---

## 6. Flying it

The race is an ordinary multirotor scenario, so a controller is an ordinary multirotor
controller: it subscribes to `astrum/telemetry/<id>` and sends `Control::Multirotor`
commands (normalized body-rate demands in [-1, 1], ~0.75 rad/s per unit with `+pitch` =
nose up, and collective in [0, 1]) — or `Control::Autopilot` for the sim's built-in
hover-hold — and reads the gates from `astrum/race/course/<id>`. Ownership epochs, the
command lease (`sim.command_lease_s`; expiry drops the drone into the sim's failsafe
hover-hold), refusals and lockstep are the sim's own and need nothing here.

An F450 with **no commands does not hover** — it falls. The sim holds the controls JSBSim
had at spawn, which is zero collective. That is what `an_unflown_drone_crashes_and_the_run_ends_as_crashed`
relies on, and why a race run without a controller ends as `Crashed` within a few seconds.

`tests/flight.rs` is the reference for what a controller has to do: a ~25-line feedback law
(altitude PI on collective, forward speed through pitch, lateral position through roll,
heading hold) running in its own thread over an in-process session, in lockstep, flies the
drone 10 m up gate 0's approach and through the opening. The same test flown twice ends in
bit-identical telemetry.

What the controller has on its plate from the first episode is a **±0.224 m lateral
corridor inside a ±0.508 m hole** (§2): heading barely matters, position does, and a pass
the scorer credits can still be a crash.

Open questions for the controller work (not decided here): **observations** — egocentric
gate geometry is the obvious basis (the pending gate's centre and normal in body axes,
range, the crossing angle the bisector implies, plus a one-gate lookahead); **reward** —
gates passed is the sparse truth, progress along the leg a dense proxy; ask what the
controller can *see* before what the reward *pays*.

---

## 7. The wire

The race adds to the sim's bus; everything else a controller or the web studio uses is the
sim's.

```
controller ◄── astrum/telemetry/<id>      the drone's state                (sim)
controller ◄── astrum/race/state/<id>     gate progress          (every control tick)
controller ◄── astrum/race/course/<id>    the track              (first tick, then ~1 Hz)
controller ◄── astrum/events              gatePassed, gateMissed, crashed, runEnded
controller ──► astrum/command/<id>        multirotor | autopilot            (sim)
```

| Message | Carries |
| --- | --- |
| `RaceCourse` | `seed`, `openingM` / `barM` / `topBarM`, and the ordered gates: `index`, `center` (ENU), `yawRad` of the **normal**, and the explicit `normal` / `right` / `up` unit vectors |
| `RaceState` | `pendingGate` (the gate count once finished), `passed`, `missed`, `gateCount`, `lastPassS`, `simTimeS`, `finished`, `outcome` |
| `GatePassed` event | `gate`, and where the CG pierced the plane: `offsetXM` along `right`, `offsetYM` along `up` |
| `GateMissed` event | `gate` |
| `Crashed` event | `outcome` (`crashedGround` / `crashedGate`) and `gate` — the gate struck, or the one it was flying to |

The basis vectors are on the wire because the gate-local basis is left-handed (§2): a
consumer that derived `right` from `yawRad` with a right-handed convention would mirror the
gate. The hardware dimensions travel with the course so a controller sizes its aim from the
message instead of hard-coding them.

`outcome` is the one thing a controller cannot work out for itself: it has the course and the
telemetry but not the airframe's hull, so without it a crashed drone looks exactly like one
whose telemetry stopped. `RaceOutcome` is the single type that `RaceProgress` holds and the
wire carries, so the two cannot drift; a subscriber's build that does not know a value
decodes it as `Unknown`.

**Counters, not only events.** Every `RaceState` figure is cumulative, so a subscriber that
skips messages loses nothing — which is what licenses `CongestionControl::Drop` on both
race channels (a slow controller can never stall a 120 Hz tick) and the latest-wins read:
`try_recv_*` drains the queue and returns the newest sample. The events are the lossless
record (they go in the bag); the state is the cheap live view.

**The course is republished ~1 Hz** even though it is static, because Zenoh pub/sub has no
late-joiner replay and the normal case is a controller started after the sim.

---

## 8. Scenario, CLI, recording

A race scenario is an ordinary `astrum-sim` scenario (`[world]`, `[sim]`, exactly one
`[[aircraft]]` — the F450) plus a `[course]` table. `RaceScenario::from_toml_str` strips
`[course]` and deserializes the rest as `astrum_sim::scenario::Scenario`, so every sim
table, default and validation rule (including `deny_unknown_fields`) is the sim's. The shipped
`config/scenario.race.toml` spells the whole envelope out with a comment per limit; the
`config/scenario.*.toml` name is what the web studio's scenario catalogue lists.

`[course]` holds the envelope fields (§3.1) plus:

| Key | Meaning |
| --- | --- |
| `seed` | course seed; default `sim.seed`; `--seed` overrides both. It is **not** the sim's RNG: `sim.seed` still seeds JSBSim's noise, so a sweep can hold the course and vary the rest |
| `start_behind_m` | the drone starts this far (default 10 m) upstream of gate 0, on its centreline, at its height, facing along it — the aircraft's lat/lon/alt/heading are **overwritten** from the course |
| `length_m` | target total leg length (m); derives the unstated leg limits and radius from `gates` (§3.1). `config/scenario.race.toml` ships `length_m = 500.0` and `sim.max_time_s = 300` so a run fits |
| `[[course.gate]]` | explicit gates (`index`, `east_m`, `north_m`, `up_m`, `yaw_rad`), used verbatim — nothing is sampled, and `--seed` is an error |
| `opening_m`, `bar_m` | the gate size the scenario was laid out for; a mismatch with this build is an error rather than a silent re-sizing |

A race also requires `world.origin_alt_hae_m = 0` (the ENU origin plane is the ground, so
gate heights are above ground and the hull's ground test agrees with JSBSim's terrain).

`RaceScenario::resolve(seed_override)` fixes the course and aims the drone;
`ResolvedRace::to_toml()` writes the **resolved** scenario — the sim's tables plus `[course]`
with the envelope, the effective `seed`, `start_behind_m`, the gate size and every gate as
`[[course.gate]]`. That text is what `astrum-race` serves on `astrum/scenario` and stores as
the bag's scenario metadata, so a recording replays the course that was flown whatever the
sampler has become since. The same API serves the web studio: `course_view(toml, seed)` →
`CourseView { seed, opening_m, bar_m, top_bar_m, gates }`, `is_race_scenario(toml)`.

```bash
make race-course SEED=42 GATES=12         # print the gates a seed draws, without flying
make race SEED=42                         # run it: listens for a controller, like astrum-sim
make race-offline SEED=42 BAG=run.mcap    # no transport, unpaced, optionally recorded
cargo run -p astrum-race -- config/scenario.race.toml --seed 7 --record run.mcap
```

`astrum-race <scenario> [--listen EP] [--seed N] [--gates N] [--dump-course] [--threads N]
[--speed X] [--record BAG] [--no-net]` — the flags are `astrum-sim`'s (with the same
environment variables) plus the course ones. The default endpoint is the sim's, so a
controller reaches either with no reconfiguration, and running both at once needs one moved.
The run loop is `astrum_sim::run_to_exit`; SIGINT/SIGTERM end it cleanly with the bag
finished.

---

## 9. Determinism

Same `(seed, params)` ⇒ identical gates, asserted by `generation_is_deterministic`. The RNG
is a seeded SplitMix64 (`rng.rs`); nothing calls `thread_rng`. The sim's lockstep makes a
networked run reproducible too: `a_scripted_controller_flies_through_gate_0_and_the_run_replays_exactly`
flies a controller thread through gate 0 twice and asserts identical final telemetry, race
state and events.

Changing the sampler's *search* (the backtracking policy, for instance) changes which courses
a seed draws, because it changes how much of the RNG stream is consumed. Determinism is per
build, not across generator changes — which is exactly what the explicit `[[course.gate]]` in
a recorded scenario is for (`the_resolved_scenario_replays_to_the_same_course`).

---

## 10. Tests

`crates/astrum-race/tests/` — integration tests only, per the repo convention (no inline
`#[cfg(test)]`); shared fixtures in `tests/common/mod.rs`: a scenario-TOML builder, and a
**rig** — the race's own `tick_systems()` plus a drone the test places by hand, no JSBSim.

| File | Covers |
| --- | --- |
| `gate.rs` | frame arithmetic from inches, the basis and its handedness, local↔world round trip, the opening, the bars, the bounding radius. The "no hardware in the opening" sweep samples the hole's **interior**: an upright's inner face and the opening's edge are the same plane, and `opening_contains` and `Bar::contains_local` are both closed, so a point exactly on it is claimed by both by construction — which way that tie falls is down to how two expressions round, so it is not asserted either way |
| `geom.rs` | hand-worked segment–point and segment–segment distances (degenerate and parallel cases included), the oriented box's support radius and basis change, box–box overlap (exact contact, a plank vs a wall, and 2000 random pairs cross-checked against a grid-sampled shared point), sphere–box at the true distance to the surface |
| `airframe.rs` | every dimension re-asserted as a literal against the model files it came from; the body-axis triad; that the drone fits with 22.35 cm to spare square on and 15.51 cm at any heading, and what the single-box and single-sphere alternatives would get wrong; and the collision cases as whole **passes** — centred, wide, high, under, banked and pitched to 45°/20°, crabbed through a full turn, and in a yawed gate's own frame |
| `course.rs` | every envelope rule re-derived **independently** from the gates, swept over 5000 seeds at the shipped, the dense 12-gate/45 m and the crowded 16-gate/40 m envelopes; envelope coverage; determinism; parameter errors; infeasibility; the deep dead end; `Course::from_gates` (verbatim, validation, `validate_clearance`); gate poses round-tripping bit-exactly through TOML |
| `length.rs` | `length_m`: the derived legs and radius, stated fields winning, the gate count re-deriving the legs, the realised length over 200 seeds at 8, 12 and 24 gates, the length surviving the resolved scenario, and the infeasible combinations' errors |
| `scoring.rs` | (rig) flythrough rules — long steps, backwards crossings, misses, ordering, under the sill, interpolated crossings — and that each pass/miss is announced as an event, with the CG's offset in the gate frame, and that the last gate ends the run as `RaceComplete` |
| `crash.rs` | (rig) termination *behaviour*: the ground threshold is the hull's lowest part, a strike scores nothing and is reported as `Crashed` + `RunEnded(Crashed)`, a crash beats a same-tick pass with the CG inside the hole (the centred pass as the control), any gate is solid, a 4 m tick cannot tunnel one — plus one real-JSBSim fall from 30 m through the sim's own step and exit |
| `app.rs` | scenario resolution — the shipped file, `[course]` marking a race, `course.seed` pinned vs `sim.seed` vs the override, explicit gates verbatim and the drone aimed at them, the resolved TOML round trip (a fixed point), the `CourseView`, a scenario for other gate sizes refused, only the F450 on a ground-level origin accepted — and on JSBSim: the spawn on the approach, and a recorded run whose bag carries the resolved gates, replays to the same course and ends `TimedOut` / `MaxTime` |
| `net.rs` | over an in-process session: the course (with each gate's explicit basis) and state reach a `RaceSubscriber` from the first tick, state time moves forward, the crash arrives as a final `finished` state, a `Crashed` event and `RunEnded(Crashed)` |
| `flight.rs` | a scripted `Multirotor` controller flies gate 0 in lockstep over an in-process session, and the run replays bit-identically |
| `spawn_failure.rs` | (its own binary: it breaks `JSBSIM_ROOT`) a drone that cannot spawn ends the race `SpawnFailed` and the run as an error |

Three conventions matter. **JSBSim is process-global and not thread-safe**, so every test
that reaches an FDM takes `fdm_guard()` as its first line — that keeps a plain `cargo test
--workspace` reliable without `--test-threads=1`. `course.rs` re-checks the sampler's rules
rather than trusting `accepts`: the sampler decides whether to keep a draw, the tests decide
whether it was right. And every collision case is written as a **pass**, never as a static
pose: the drone's widest parts are its rotors, 0.165 m fore and aft of the CG, so at the
instant the CG is in a gate's plane none of them are — a hand-picked pose in the opening
touches nothing however badly it is placed, and a verdict from one mostly answers a question
about the along-normal coordinate. (The rig makes the corollary easy to trip over: a CG path
that runs diagonally across an opening *does* put a rotor in an upright long before the CG
reaches the plane.)

The 5000-seed sweep is not decoration. It costs a fraction of a second (generation is pure
arithmetic — no FDM, no world) and it is what caught the seed-60 wedge in §3.4.

---

## Appendix: key code references

| Concern | Location |
| --- | --- |
| Gate hardware, pose, opening test | `crates/astrum-race/src/gate.rs` |
| Monte-Carlo sampler, envelope, errors, `Course::from_gates` | `crates/astrum-race/src/course.rs` |
| Seeded RNG (the sim's SplitMix64 + a range draw) | `crates/astrum-race/src/rng.rs` |
| Segment distances, the oriented box, box/sphere overlap | `crates/astrum-race/src/geom.rs` |
| F450 dimensions and the five-part `Hull` | `crates/astrum-race/src/airframe.rs` |
| The F450's model files | `.jsbsim-variants/aircraft/F450/` |
| The extension points (`TickSet`) and the app the race builds on | `crates/astrum-sim/src/lib.rs` |
| Race scenario, resolution, `CourseView` | `crates/astrum-race/src/scenario.rs` |
| `build_race_app` | `crates/astrum-race/src/lib.rs` |
| Crash detection, scoring, the end of the run, `tick_systems`, the wire publication | `crates/astrum-race/src/systems.rs` |
| `RaceProgress` (`RaceOutcome`, first-writer-wins `finish`), `RaceCourse` | `crates/astrum-race/src/components.rs` |
| The race's Zenoh publisher | `crates/astrum-race/src/net.rs` |
| `RaceCourse` / `RaceState` / race events schema and codecs | `crates/astrum-proto/schema/astrum.capnp`, `crates/astrum-proto/src/lib.rs` |
| `RacePublisher` / `RaceSubscriber`, key space | `crates/astrum-net/src/lib.rs` |
| CLI | `crates/astrum-race/src/main.rs` |
| Shipped scenario | `config/scenario.race.toml` |
