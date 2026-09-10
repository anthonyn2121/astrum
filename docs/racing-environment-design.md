# Design: drone-racing environment for the DJI F450

**Status:** the course generator, the gate model, the gate scoring, the F450's
stabilising inner loop, crash termination and the Zenoh/Cap'n Proto bridge are
implemented and tested. **Nothing flies the course yet** — that is the next piece of
work, and §7 (in-process) and §7b (off-process, over the wire) are the two seams it
plugs into.

**Crate:** `crates/astrum-race` — a *separate* headless Bevy app, not a mode of
`astrum-sim`.

---

## 1. Why a separate crate

`astrum-sim` hosts the within-visual-range gun-combat environment
(`docs/dogfight-environment-design.md`). Its combat systems, its observation vector,
its reward and its curriculum are a large, load-bearing, actively-trained-against
body of work. A racing environment shares almost nothing with it: no weapons, no
opponent, no aspect angle, a different airframe and a different task.

So `astrum-race` is its own binary and its own app, and it depends only on the crates
that are genuinely task-neutral:

| Dependency | Why |
| --- | --- |
| `astrum-fdm` | the JSBSim FFI (`Fdm`, `prop::*`, `FT_TO_M`) |
| `astrum-geo` | WGS84 ↔ ENU (`LocalFrame`, `Geodetic`, `Enu`) |
| `astrum-proto` | the Cap'n Proto wire messages (§7b) |
| `astrum-net` | the Zenoh session, publishers and subscribers (§7b) |
| `bevy` (no default features) | ECS + the headless schedule runner |

None of those four depends on `astrum-sim`, so the separation holds: nothing here can
reach the combat crate's code. The wire crates are also *not* on the RL training path
— `astrum-sim`'s embedded environment never opens a Zenoh session — so a schema change
cannot perturb a training run either.

The cost of that choice is duplication: `astrum-race` carries its own seeded RNG
(`rng.rs`) and its own segment-distance math (`geom.rs`) rather than reaching into
`astrum-sim`'s. That is the intended trade — a copy of ~200 lines of pure arithmetic
is cheaper than a shared dependency that couples a training run to a racing change.

---

## 2. The gate

One gate is three pieces of 3-inch stock: two 31-inch uprights carrying a 37-inch top
bar, giving a **31-inch square opening**.

```text
           <------------- 37" ------------->         top bar (37 x 3)
          +----------------------------------+  --+
          |###                            ###|    | 3"
          |###                            ###|  --+
          |###                            ###|
          |###          31" x 31"         ###|    upright (31 x 3)
          |###          opening           ###|    31" tall
          |###                            ###|
          |###                            ###|
          +###----------------------------###+  --- sill (open below)
           <3">        <-- 31" -->
```

Dimensions are held **in inches** (`IN_TO_M = 0.0254`) because that is what makes the
frame's defining identity exact:

```
37" = 31" + 2 x 3"     →     TOP_BAR_M == OPENING_M + 2 * BAR_M
```

Deriving from rounded centimetre figures instead would leave the top bar 2 mm short of
the frame it has to span.

Two properties worth stating because they are easy to get wrong:

* **The frame is not symmetric about the opening.** There is a bar above and nothing
  below, so the lowest hardware is the *sill* at the bottom of the opening, and
  `frame_radius_m()` — the bounding sphere the course sampler clears legs against —
  is set by a **top** corner of the top bar.
* **The gate-local basis is left-handed.** Local coordinates are `x` = right, `y` =
  up, `z` = along the normal (positive downstream), and `x` is *starboard of the
  flight direction*, so the right-handed cycle is `(normal, up, right)` — a
  camera-style arrangement. Nothing takes a cross product in local coordinates today;
  anything that starts to should know which it is.

A gate's pose is a centre (the middle of the **opening**, not of the hardware) plus a
yaw, using the sim's convention: `psi = 0` faces North, `+psi` turns East. Gates stand
upright — no bank, no tilt.

### Why 31" and not the 25" of the brief

The original brief fixed the opening at 25" (0.635 m). **That size makes the course
unflyable rather than merely hard**, and it was widened on 2026-09-08 at the user's
request ("we can make the gates slightly larger so that we don't have any collisions on
turns out of the gate"). The reasoning, because the number is not arbitrary:

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

31" is the first whole inch past that threshold whose *worst-heading* margin beats what
25" gave only when perfectly square:

| | 25" (brief) | **31" (built)** |
|---|---|---|
| clearance, square on | 3.30 cm | **10.92 cm** |
| clearance, worst heading | −3.54 cm (a strike) | **4.08 cm** |
| headroom under the top bar | 17.31 cm | **24.93 cm** |
| legal crab | ±13° | **any** |
| `frame_radius_m()` | 0.5568 m | 0.6645 m |

The bigger bounding sphere costs the course sampler nothing measurable: 0 failures in
5000 seeds at the shipped envelope, at the 12-gate/45 m dense one, and at a 16-gate/40 m
envelope well past anything shipped.

What did **not** change, and still matters to a controller:

* **The flyable corridor is much narrower than the hole.** The scorer credits a pass on
  where the *CG* pierced the plane, anywhere in ±0.3937 m; the props reach 0.28448 m
  either side of it, so anything past ±0.10922 m is a crash. Two thirds of the opening is
  a trap, and `a_crash_beats_a_pass_on_the_same_tick` is exactly that case.
* **The headroom is 0.249 m, not the 0.394 m the opening looks like it offers**, because
  what reaches the top bar is a blade tip at the rotor plane, 0.145 m out and up from the
  CG.
* **Heading still costs corridor**, just no longer all of it: 4.08 cm at the worst
  heading against 10.92 cm square on.

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
| `leg_min_m` / `leg_max_m` | 8 / 20 | room to line up on the next gate |
| `max_turn_deg` | 120 | crossing angle (see §3.3) |
| `max_climb_deg` | 20 | flight-path angle of a leg |
| `alt_min_m` / `alt_max_m` | 1.5 / 4.0 | flight volume, sill above ground |
| `min_gate_gap_m` | 6.0 | gates not on top of each other |
| `min_leg_gate_gap_m` | 1.5 | a leg must not thread a frame it isn't flying through |
| `min_leg_leg_gap_m` | 2.0 | non-adjacent legs must not cross |
| `course_radius_m` | 60 | horizontal containment |
| `tries_per_gate` | 200 | draws before backtracking |
| `max_backtracks` | 200 | total unwinding before "infeasible" |
| `start_heading_deg` | 0 | entry direction, fixed so it is repeatable across seeds |

`validate()` rejects contradictory envelopes up front (`CourseError::Params`) so a
bad input is a clear message rather than an exhausted sampler: inverted ranges, a
`min_gate_gap_m` wider than `leg_min_m` (which would reject every candidate, since
consecutive gates are exactly one leg apart), an `alt_min_m` that would put the sill
underground, a `course_radius_m` smaller than one leg.

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

Generation is fallible **before the app is built** (`build_app -> Result`), so an
unsatisfiable envelope is an error the caller sees, not a log line from inside a
startup system.

---

## 4. Scoring a flythrough

`systems::track_gates`. A gate counts when the drone crosses its plane **inside the
opening**, **downstream**, and gates must be taken **in order**.

The crossing is tested on the **segment** between the previous and current tick, in the
gate's local frame — not by asking whether a point is inside the frame. At racing speed
a tick covers metres while the frame is 7 cm thick, so a point-in-volume test would
step clean over the gate. Concretely: look for the sign change of local `z`, solve for
the fraction `t` where it crosses, and test the interpolated hit point against the
opening.

A crossing outside the opening but within `MISS_RADIUS_M` (3 m, a few gate-widths) is
counted as a **miss** — "went round the outside" is a pilot's miss. Further out than
that it is unrelated flying and is not scored at all. A miss does **not** advance the
course; the gate stays pending and can be re-attempted.

On a pass, `last_local` is reset to `None`, because the next gate's local frame is a
different basis: the next gate needs one reference tick before it can score.

---

## 4b. The airframe, and how a run ends

`airframe.rs`, `systems.rs::detect_crash`. A run ends in exactly one way: some system
writes an `Outcome` onto `RaceProgress` via `finish()`, and `stop_when_finished` turns
that into the app's single `AppExit`.

| `Outcome` | Set by | Meaning |
| --- | --- | --- |
| `Running` | — | still flying (the default) |
| `Complete` | `track_gates` | every gate passed, in order — the only way to win |
| `CrashedGround` | `detect_crash` | the hull reached the ground plane |
| `CrashedGate` | `detect_crash` | the hull struck a gate's frame |
| `TimedOut` | `enforce_time_limit` | `sim.max_time_s` expired with gates to fly |
| `SpawnFailed` | `spawn_drone` | nothing launched — *our* fault, not the pilot's |

`finish()` is **first-writer-wins**, which is not tidiness: it decides races between
terminators on the same tick, and the order it enforces is the one a race would judge by.
`detect_crash` runs *before* `track_gates`, so clipping an upright while the CG crosses
inside the opening is a **crash, not a pass** — which is not a corner case, because the
flyable corridor is ±0.109 m inside a ±0.394 m hole (§2).

### The hull is five parts, because 4.1 cm is the budget

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
feet to the rotor plane. The budget it has to be accurate inside is the **4.1 cm** at the
worst heading, not the 10.9 cm square on, and neither single-shape alternative is:

* **A box across the tip circle** presents a 0.40232 m diagonal when crabbed, against a
  0.3937 m half-opening — so a crabbed pass would still be a crash. That is precisely what
  widening the gate to 31" was for, so the box model would hand the whole enlargement
  straight back. It is wrong vertically too: 0.371 m of claimed reach at 45° bank / 30°
  pitch against the airframe's real 0.293 m, which shrinks the vertical corridor at that
  attitude from 10.1 cm to 2.3 cm.
* **A sphere on the CG** needs the tip circle, `ARM·√2 + PROP_RADIUS = 0.35287 m`. At 31"
  that does now fit the opening — with the same 4.1 cm as the real airframe's worst
  heading — so it no longer makes the course impassable. But it is round and the airframe
  is not: it reaches 0.353 m *below* the CG where the gear stops at 0.152 m, so a drone
  hovering 0.30 m up would be a ground crash, and it cuts the headroom under a top bar
  from 24.9 cm to 4.1 cm. Both errors are ~5x the budget.

(In the 25" gate both alternatives were impossible outright — the sphere and the box were
each wider than the half-opening. Enlarging the gate did not weaken the case for an honest
hull; it moved the argument from *impassable* to *wrong by five times the budget*.)

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
terrain in the shipped scenario (`world.origin_alt_m = 0`), so this fires exactly as the
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

Headless Bevy, `MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO))`,
one physics step per `App::update()`, no `FixedUpdate` — the same shape as
`astrum-sim`.

```
Startup:  spawn_course → spawn_drone                       (.chain())

Update:   ingest_commands → stabilize → step_fdm → sync_pose
          → detect_crash → track_gates → enforce_time_limit     (the terminators)
          → publish_telemetry → publish_race → report_progress
          → stop_when_finished → pace                      (.chain())
```

The order is the invariant, in three groups. `ingest_commands` lands a wire command in
time for the loop that services it, and `stabilize` sets the control inputs *this* tick
integrates. Then everything that can end the run is judged against the pose `sync_pose`
just derived, with `detect_crash` ahead of `track_gates` (§4b). Every one of those only
*marks* an outcome; publishing comes after them all, so the wire always carries the
outcome that ended the run, and `stop_when_finished` is the single place a run exits.

The three bridge systems take their transport as `Option<NonSend<..>>` and no-op when
it is absent, so `build_app_with_course` stays a pure physics app — which is what lets
every integration test but `net.rs` run without opening a session.

The drone launches `start_behind_m` (default 10 m) upstream of gate 0, on its
centreline, at its height. `pace` throttles to real time with `std::thread::sleep`,
because Bevy's own runner sleep is a no-op under `default-features = false`; it
changes wall-clock timing only, never the step count or `dt`.

---

## 6. The F450's inner loop

`control.rs`. An F450 with its motors running and **no commands does not hover** — it
falls, hits the ground, and JSBSim declares a crash and NaNs the state. So the
environment needs a stabilising loop before it needs anything else.

With `fcs/ScasEngage = 1.0` the four `fcs/*-cmd-norm` inputs are not aerodynamic
surfaces: `aileron`/`elevator`/`rudder` demand body roll/pitch/yaw **rates** (~0.75
rad/s per unit, `+cmd` → `+rate`, so `elevator+` pitches the nose *up*) and `throttle`
is **collective**. The law is cascaded:

```
velocity error (body fwd/right)  →  attitude targets (clamped to MAX_TILT)
attitude error                   →  rate demands, damped on measured p/q/r
heading error                    →  rudder, damped on r
altitude error                   →  collective (PI + D on climb rate)
```

**Why the altitude loop is PI and not PD.** With proportional gain only, the drone
settles wherever `K_ALT * error` makes up the gap between the hardcoded
`HOVER_COLLECTIVE` and the collective the airframe actually needs — a standing offset
you can shrink with gain but never remove. It showed up as a clean 1.60 m sag at zero
residual velocity, which said the true hover trim was ≈0.47, not the 0.44 assumed.
The integrator *finds* the trim instead, so the same gains hold altitude at any weight
or density altitude; `ALT_I_LIMIT` bounds the wind-up. The integral state lives on the
`InnerLoop` component, so it is per-drone and deterministic.

---

## 7. The in-process seam for a course-flying controller

An outer controller inside the app writes the `FlightCommand` component (§7b is the
same job from another process):

```rust
pub struct FlightCommand {
    pub alt_m: f64,        // ENU altitude (not above sea level)
    pub heading_rad: f64,  // 0 = North, + = East
    pub vel_enu: DVec3,    // horizontal ground-velocity demand; z ignored
}
```

`FlightCommand::hold(alt, heading)` is station keeping, which is what the drone
launches with. This is deliberately the *whole* interface: a policy, an MPC or a
hand-coded line-follower is written in terms of where to go, not motor commands. A
controller that would rather supply its own law writes `RawOverride` — the four
normalized channels, with the inner loop standing down (see §7b for the lease rule that
comes with it).

Open questions for that work, not yet decided:

1. **Observations.** Egocentric gate geometry is the obvious basis — the pending
   gate's centre and normal in body axes, range, the crossing angle the bisector
   implies, plus a one-gate lookahead so the controller can set up the next turn.
2. **Reward.** Gates passed is the sparse truth; a dense proxy would be progress along
   the leg. Note the standing lesson from the combat program: ask what the controller
   can **see** before what the reward **pays**.

Collision is **no longer** an open question: the frame and the ground are solid, and a
strike ends the run (§4b). That does change the task substantially, which is the point.
What it puts on the controller's plate from the first episode is a **±0.109 m lateral
corridor inside a ±0.394 m hole** (§2) — heading no longer has to be tracked to ±13°,
but position does, and a pass credited by the scorer can still be a crash.

---

## 7b. The wire: flying it from another process

`net.rs`. The same Zenoh + Cap'n Proto stack the fleet sim uses, so a racing controller
can be written in Rust, C++ or Python and developed against a running environment.

```
controller ◄── astrum/telemetry/<id>      the drone's state          (every tick)
controller ◄── astrum/race/state/<id>     gate progress              (every tick)
controller ◄── astrum/race/course/<id>    the track                  (~1 Hz)
controller ──► astrum/command/<id>        raw actuators | autopilot
```

### What is reused, and what needed a new message

Nothing about *flying* needed a new format. The drone's state goes out as a plain
`Telemetry` (with `vehicleKind = multirotor`), and a controller flies it with a plain
`Command`: `Control::Raw` is four normalized channels, which is exactly what a
quadrotor takes. That is why the bundled C++/Python controllers already work against
an F450 in the fleet sim (`make demo-quad`) — the wire is vehicle-agnostic.

What had no home was the **task**: where the gates are, and which one is pending.
`Telemetry` means "one aircraft's authoritative state", and it is what the bag recorder
and the 3-D replay consume; a course column would sit dead in every combat recording.
So the task state got two of its own messages on their own key space:

| Message | Carries |
| --- | --- |
| `RaceCourse` | `seed`, `openingM`/`barM`, and the ordered gate list (`index`, `center` as ENU x=E/y=N/z=U, `yawRad` of the **normal**) |
| `RaceState` | `pendingGate`, `passed`, `missed`, `gateCount`, `lastPassS`, `simTimeS`, `finished`, and `outcome` (§4b's enum) |

`seed` travels with the course because it *is* the course's identity, and the hardware
dimensions travel with it so a controller sizes its aim from the message instead of
hard-coding 0.7874 m. A completed run reports `pendingGate == gateCount`.

`outcome` is the one thing a controller cannot work out for itself: it has the course and
the telemetry, but not the airframe's hull, so without it a crashed drone looks exactly
like one whose telemetry stopped. It is a separate enum from the environment's `Outcome`,
projected by `wire_outcome` — the terminal state is physics-side truth and must not depend
on a transport being installed. An outcome a subscriber's build does not know decodes as
`Running`, which is the safe reading for an old controller against a new environment.

Both go in the existing `astrum.capnp` rather than a second schema file: one schema is
what keeps Rust (build.rs), C++ (CMake `capnp compile`) and Python (`capnp.load` at
runtime) wire-compatible with no per-language edits.

### Two properties a subscriber can rely on

**Counters, not events.** Every progress figure is cumulative, so a subscriber that
skips messages loses nothing — `passed` is not a stream of "gate passed" notifications
whose loss would desynchronise it. That is what licenses `CongestionControl::Drop` on
both channels (a slow controller can never stall a 120 Hz physics tick) and the
latest-wins read: `try_recv_*` drains the queue and returns the newest sample, because a
backlog of *state* is pure staleness.

**The course is republished ~1 Hz** even though it is static, because Zenoh pub/sub has
no late-joiner replay and the normal case is a controller started after the
environment. It is also published on the very first tick, so a controller already
listening does not wait a period.

### Raw control is a lease, not a latch

The fleet sim latches raw actuator commands indefinitely: a fixed-wing whose controller
dies keeps flying. A quadrotor whose controller dies **falls** — collective is the only
thing holding it up, and the last commanded value is frozen. So here a `Control::Raw`
command expires: unrefreshed for `RAW_TIMEOUT_S` (0.5 s ≈ 60 ticks at 120 Hz) it lapses,
the inner loop resumes, and `hold_here` pins station at whatever altitude and heading
the drone had *at that moment* — not the command it had before the override. The
altitude integrator is reset with it, since the trim it had learnt was for a different
regime. A controller running at any sane rate never notices the lease exists.

`Control::Autopilot` is accepted too, mapped onto `FlightCommand`. Be aware of what
that mapping cannot express: `AutopilotTargets` carries a **scalar** `airspeedMps`, so
the only velocity it can ask for is *forward along the commanded heading*. A racing
drone crabs — it holds a heading while translating sideways, which is how you line up on
a gate you are already turning past. There is no encoding of that; a controller that
needs it flies `Control::Raw` and owns the law.

---

## 8. Configuration

`config/race.toml` — world origin (UMore Park), `[sim]` rate/limit/realtime/seed,
`[drone]` id/model/start, and `[course]` spelling out the whole envelope with a
comment per limit. Every `[course]` field has a default, so an absent table is a
valid and reasonable course.

```bash
make race-course SEED=42 GATES=10       # print the gates a seed draws, without flying
make race SEED=42                       # fly it, publishing on tcp/…:7447
make race LISTEN=tcp/0.0.0.0:7448       # …on another endpoint (a sim already has 7447)
make race-offline SEED=42               # physics only: --no-net, no port bound
```

The endpoint is `--listen` (env `ASTRUM_RACE_LISTEN`), defaulting to the *same*
`tcp/0.0.0.0:7447` the fleet sim uses — so a controller reaches either environment with
no reconfiguration, and running both at once needs one of them moved. A failed listen is
logged, not fatal: the race still flies, silently.

---

## 9. Determinism

Same `(seed, params)` ⇒ identical gates, asserted by `generation_is_deterministic`.
The RNG is a seeded SplitMix64 (`rng.rs`); nothing calls `thread_rng`. The app's
trajectory is reproducible too — `app_honours_the_time_limit_and_repeats_exactly`
asserts the same scenario twice gives the same step count and bit-identical final
position.

Note that changing the sampler's *search* (the backtracking policy, for instance)
changes which courses a seed draws, because it changes how much of the RNG stream is
consumed. Determinism is per build, not across generator changes.

---

## 10. Tests

`crates/astrum-race/tests/` — integration tests only, per the repo convention (no
inline `#[cfg(test)]`); shared fixtures in `tests/common/mod.rs`.

| File | Covers |
| --- | --- |
| `gate.rs` | frame arithmetic from inches, the basis and its handedness, local↔world round trip, the opening, the bars, the bounding radius. The "no hardware in the opening" sweep samples the hole's **interior**: an upright's inner face and the opening's edge are the same plane, and `opening_contains` and `Bar::contains_local` are both closed, so a point exactly on it is claimed by both by construction — which way that tie falls is down to how two expressions round, so it is not asserted either way |
| `geom.rs` | hand-worked segment–point and segment–segment distances (degenerate and parallel cases included), the oriented box's support radius and basis change, box–box overlap (exact contact, a plank vs a wall, and 2000 random pairs cross-checked against a grid-sampled shared point), sphere–box at the true distance to the surface |
| `airframe.rs` | every dimension re-asserted as a literal against the model files it came from; the body-axis triad; that the drone fits with 10.92 cm to spare square on and 4.08 cm at any heading, and that neither single shape can; and the collision cases as whole **passes** — centred, wide, high, under, banked and pitched to 45°/20°, crabbed through a full turn, and in a yawed gate's own frame |
| `course.rs` | every envelope rule re-derived **independently** from the gates, swept over 5000 seeds; envelope coverage; determinism; parameter errors; infeasibility; the deep dead end |
| `scoring.rs` | flythrough rules over hand-placed positions — long steps, backwards crossings, misses, ordering, under the sill, interpolated crossings |
| `crash.rs` | termination *behaviour*: the ground threshold is the hull's lowest part, a strike scores nothing, a crash beats a same-tick pass with the CG inside the hole (with the centred pass as the control), any gate is solid, a 4 m tick cannot tunnel one (both endpoints verified clear first), a failed spawn exits at once — plus one real-JSBSim descent flown into the ground |
| `app.rs` | the real app through `update()`: spawn, the station hold, the time limit, reproducibility, and that `config/race.toml` loads and generates |
| `net.rs` | the wire through the real app on an in-process session: the course matches the generated track and arrives on tick 1, a late joiner still gets it, `RaceState` mirrors the progress resource, the outcome that ended a run travels (and a running race says so), telemetry goes out as a multirotor, a raw command takes the actuators (and is clamped, and is ignored for another id), the lease lapses and the inner loop catches the drone, autopilot maps onto `FlightCommand`, and a session-less app is unchanged |

Three conventions matter. **JSBSim is process-global and not thread-safe**, so every
test that reaches an FDM takes `fdm_guard()` as its first line — that keeps a plain
`cargo test --workspace` reliable without `--test-threads=1`. `course.rs` re-checks the
sampler's rules rather than trusting `accepts`: the sampler decides whether to keep a
draw, the tests decide whether it was right. And every collision case is written as a
**pass**, never as a static pose: the drone's widest parts are its rotors, 0.165 m fore
and aft of the CG, so at the instant the CG is in a gate's plane none of them are — a
hand-picked pose in the opening touches nothing however badly it is placed, and a verdict
from one mostly answers a question about the along-normal coordinate.

The 5000-seed sweep is not decoration. It costs a fraction of a second (generation is
pure arithmetic — no FDM, no world) and it is what caught the seed-60 wedge in §3.4.

---

## Appendix: key code references

| Concern | Location |
| --- | --- |
| Gate hardware, pose, opening test | `crates/astrum-race/src/gate.rs` |
| Monte-Carlo sampler, envelope, errors | `crates/astrum-race/src/course.rs` |
| Seeded RNG (SplitMix64) | `crates/astrum-race/src/rng.rs` |
| Segment distances, the oriented box, box/sphere overlap | `crates/astrum-race/src/geom.rs` |
| F450 dimensions and the five-part `Hull` | `crates/astrum-race/src/airframe.rs` |
| The F450's model files | `.jsbsim-variants/aircraft/F450/` |
| `FlightCommand`, `RawOverride`, the inner loop | `crates/astrum-race/src/control.rs` |
| Zenoh bridge (publish, ingest, the lease) | `crates/astrum-race/src/net.rs` |
| `RaceCourse` / `RaceState` schema | `crates/astrum-proto/schema/astrum.capnp` |
| Their codecs | `crates/astrum-proto/src/lib.rs` |
| `RacePublisher` / `RaceSubscriber`, key space | `crates/astrum-net/src/lib.rs` |
| Spawn, step, pose, scoring, crash detection, pacing, termination | `crates/astrum-race/src/systems.rs` |
| `Outcome`, `RaceProgress::finish` | `crates/astrum-race/src/components.rs` |
| Scenario TOML schema | `crates/astrum-race/src/scenario.rs` |
| App assembly | `crates/astrum-race/src/lib.rs` |
| CLI (`--dump-course`, `--seed`, `--gates`, `--listen`, `--no-net`) | `crates/astrum-race/src/main.rs` |
| Shipped scenario | `config/race.toml` |
