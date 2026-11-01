# Design: adversarial WVR gun-combat environment for the F-16

**Status:** §12 decisions locked 2026-08-25; some were later superseded by the
schema-v2 remediation and are marked as such below. The gun/damage combat system
(`astrum-sim/src/combat`: `Gun`/`Trigger`/`Bullet`/`Health`, `fire_guns` →
`step_bullets` → `resolve_hits` → `apply_damage`/`end_on_elimination`) and the
embedded RL seam (§8.1, `astrum-sim/src/env.rs` + `crates/astrum-env`) are
**implemented**. The numpy-only Gym adapter (`crates/astrum-env/python/astrum_dogfight`)
plus the **full scripted-opponent curriculum (Tiers 1–3)** are implemented and
tested — the Tier-3 FSM tactician guns the learner down end-to-end through the real
combat pipeline. **Self-play** (and adding gymnasium + SB3 to the dev image) is next.
**Date:** 2026-08-25 (implementation pointers updated for schema v2)
**Scope:** Add a nose-mounted gun (M61A1 Vulcan), realistic bullet ballistics, and a
health/damage system to the ASTRUM sim, so two (or more) F-16s can fight within visual
range. The end goal this enables is training an RL agent for WVR dogfighting.

This document began as a map of *what to build and where it plugs in*; the design
sections keep that rationale, and every statement about the code describes the
current implementation. The decisions are in §12.

> **New to the env and just want to train (or hand-code) a policy?** Start with
> [`dogfight-env-onboarding.md`](dogfight-env-onboarding.md) — the practical
> build/run/interface guide. This doc is the design rationale behind it (§8 is the
> RL seam).

---

## 1. Feasibility summary

Greenfield but well-supported. When this was written there was **no**
gun/bullet/weapon/damage code in the repo — the only adjacent primitive was an inert
`health` field on the wire. Every seam needed already existed:

- a fixed-step, deterministic ECS tick (one `App::update()` = one physics step),
- per-vehicle state in a clean ENU world frame,
- an additively-extensible command/telemetry wire,
- a headless test harness that drives the loop exactly the way an RL stepper would.

The work partitions cleanly:

1. **Sim mechanics** — guns, ballistics, hit-detection, health. Pure ECS, no wire changes,
   fully testable headless. *This is the core.*
2. **Wiring** — a fire trigger on `Command`, HP/ammo on `Telemetry`, a demo scenario + target.
3. **Visualization** — bullet tracers + HP/ammo HUD in the browser replayer.
4. **RL environment** — the seam that makes training practical (future work).

Phases 1–2 alone give a functioning, deterministic gun-combat sim.

---

## 2. How it maps onto ASTRUM

### 2.1 The tick

The sim is a headless Bevy app (`crates/astrum-sim`). It uses only
`ScheduleRunnerPlugin::run_loop(Duration::ZERO)` (no `MinimalPlugins`); **there is no `FixedUpdate`** — all physics
runs in the plain `Update` schedule, so **one `App::update()` == exactly one physics tick** at
`dt = 1/rate_hz` (default **120 Hz → 8.33 ms**, `SimParams` in `components.rs`). The sim
clock is the `SimClock` resource (`tick · dt`), independent of any aircraft.

Systems run `.chain()`-ordered in `build_app` (`astrum-sim/src/lib.rs`), combat included:

```
ingest_commands → expire_leases → autopilot_step → fire_guns → step_fdm → freeze_dead
    → advance_clock → step_bullets → resolve_hits → apply_damage → end_on_elimination
    → publish_telemetry → report_progress → stop_at_max_time   (then pace)
```

`fire_guns` runs **before** physics: rounds leave the muzzle at `t_k` from the shooter's
state at `t_k`; `step_fdm` and `step_bullets` then both advance to `t_k+1`, so every hit test
spans one consistent interval `[t_k, t_k+1]`. HP/ammo ship in the same tick's telemetry.
Combat resources (`CombatParams`, the seeded `CombatRng`, `CombatState`) are inserted in
`build_app`.

### 2.2 New ECS components

They live in `crates/astrum-sim/src/combat/components.rs` and are attached by
`initial_components` in `systems.rs`. Every fixed-wing aircraft (any model with an autopilot
profile — `c172x`, `f16`, or an explicit `autopilot_profile`) is armed, alongside its
`FixedWingGains`; multirotors are unarmed:

- `Gun { muzzle_offset_body: DVec3, rate_hz, muzzle_speed_mps, ammo: u16, dispersion_rad,
  accum }` (f64) — `accum` carries the fractional rounds-per-tick counter.
- `Health { hp: f32, hp_max: f32 }`.
- `Team(u8)` — for owner/target exclusion; generalizes 1v1 → NvN.
- `Trigger(bool)` — per-aircraft firing state, set from `Command.trigger`, read by `fire_guns`.
- `Dead` — marker inserted when `hp <= 0`.

### 2.3 Bullets are their own entities — but NOT FDM entities

Each round is a lightweight entity: `Bullet { pos, prev_pos, vel: DVec3, ballistics,
owner_id: u32, team: u8, ttl_s: f64 }` in f64 world-grid (ENU) coordinates (no Bevy
`Transform`). Integrated with plain point-mass kinematics.

> Bullets never own an `FdmInstance`: a point mass needs no 6-DOF model, and per-round
> JSBSim instances would cost orders of magnitude more. (The original "JSBSim is not
> thread-safe" constraint is gone: `astrum-fdm` serialises JSBSim's global initialisation,
> see §8.6.)

Population is bounded and trivial for Bevy: 100 rounds/s × ~3 s TTL ≈ a few hundred live
entities per shooter.

### 2.4 The `raw` control interface is FLCS NRT, not bare surfaces

Worth stating up front for the RL action design (§8): the JSBSim **f16** model runs a full
fly-by-wire FLCS, so `Control::Raw`'s `fcs/*-cmd-norm` inputs are **not** raw surface
deflections — they are normalized **Nz / roll-rate / throttle (NRT)** commands the FLCS turns
into demands, with built-in envelope protection:

- `elevator-cmd-norm` → **g-command (Nz)**, clip `[-1, 0.44]` = +9/−4 g, plus a ~30° AoA
  limiter — full aft stick pulls the *max the jet currently allows*, not a fixed deflection.
- `aileron-cmd-norm` → **roll-rate command** (~180–188°/s at full stick).
- `rudder-cmd-norm` → yaw-rate/coordination; `throttle-cmd-norm` → direct throttle.

Two consequences for the environment:

1. **The action space is already a performance interface with departure protection.** An RL
   agent commanding `[aileron, elevator, rudder, throttle]` (§8) cannot stall, spin, or over-g
   the jet — the FLCS bounds it. That removes the need to heavily penalize departures and keeps
   the action naturally normalized to available performance.
2. **Aggressive BFM is flyable through `raw` with no new control mode.** A Split-S flown purely
   via `raw` sustains ~6.9 g (lift-limited at idle, α ~12°, under the 30° limiter, so below the
   9 g cap), ~188°/s roll, reverses heading ~180°, in a ~2 km loop (`r ≈ V²/(n·g)`, so entry
   speed dominates the radius). No `<autopilot>` block exists; `fcs/fbw-override` defaults 0
   (FLCS active) and ASTRUM never sets it. The model lives in the **container** at
   `/opt/jsbsim/share/jsbsim/aircraft/f16/f16.xml`, not in the repo tree.

**Observation gotcha (§8):** `accelerations/n-pilot-z-norm` is **negative for a positive-g
pull** (~−1 level upright, ~−6.9 in a hard pull). ASTRUM therefore reports
`accelerations/Nz` instead (`Telemetry.aero.nzG`, ≈ +1 level, positive in a pull).

---

## 3. Ballistics model

Three requested effects — "shoot in front of the enemy," "bullet drop," "coriolis when
banking/turning" — resolve into **two genuine projectile forces plus one aiming problem**.
Getting the framing right makes the implementation correct *and* cheaper.

**Simulate every round as a free projectile in the ENU world frame.** ENU is ASTRUM's
canonical grid: `East=+X, North=+Y, Up=+Z`, right-handed (`astrum-geo/src/lib.rs`).
Attitude is aerospace Euler on the wire (`phi`=roll, `theta`=pitch, `psi`=yaw/true heading,
radians; body axes nose=+X, starboard=+Y, down=+Z; `ψ=0` faces North, `+ψ` yaws East). The
boresight (muzzle direction) is:

```
nose_hat_ENU = ( cosθ·sinψ,  cosθ·cosψ,  sinθ )        # (E, N, U)
```

(the same rotation the sim's `combat::geometry::body_to_grid` applies, including the local-level
tilt between the aircraft's position and the grid origin). Initial round velocity at the
trigger instant, as implemented in `fire_guns`:

```
v0 = v_aircraft_grid  +  muzzle_speed · dispersed(boresight_grid)
     (dispersion: seeded Gaussian cone, σ = dispersion_rad)
```

The rotational term `ω_body × r_muzzle` is omitted: with the 4 m muzzle offset it is
~1 m/s against a 1030 m/s muzzle speed.

Per-tick integration (semi-implicit Euler at `dt`):

```
a = (0, 0, −g)                          # gravity → bullet drop
  − (½·ρ·C_d·A / m)·|v|·v                # aerodynamic drag  (dominant term at gun range)
  − 2·(Ω_ENU × v)                        # Earth Coriolis, Ω_ENU = Ω·(0, cosφ0, sinφ0)
v += a·dt ;  x += v·dt
```

`φ0` = scenario ENU origin latitude (`LocalFrame`); `Ω = 7.292e-5 rad/s`. `ρ` is ISA density
at the round's height above the ellipsoid; rounds are culled at `h = 0` (there is no terrain
model) or when their 3 s lifetime runs out.

### 3.1 The two clarifications that shape the design

**(a) "Shoot in front" is aiming, not a bullet force.** The sim propagates rounds truthfully
wherever they are pointed; hitting a mover requires pointing ahead so round and target arrive
together (time-of-flight lead). That belongs to the **shooter** — the RL agent, or an optional
fire-control aid we compute as an observation/HUD "lead pip." We do **not** bake lead into the
bullet integrator.

**(b) The "banking/turning coriolis" you want is captured automatically by world-frame
integration — do NOT add a turn force.** Once a round leaves the muzzle it is inertial; the jet
turning underneath exerts nothing on it. What a gunnery computer compensates for is that *in
the rotating cockpit frame the path appears to curve* — because the jet rotated, not because a
force acted. Integrating in ENU with muzzle-velocity inheritance (`v_aircraft`) reproduces
this, and the apparent curve shows up correctly when viewed from the jet. Adding a fake
"turn-coriolis" acceleration would **double-count and be wrong.**
The only genuine inertial Coriolis is Earth's rotation (`−2Ω×v` above) — cheap to include, but
honestly ~centimeters at WVR ranges (TOF ~1 s), dwarfed by drag and aim error. Include it,
clearly commented as negligible, so nobody tunes against noise.

### 3.2 Drag is the term that matters

20 mm rounds go transonic→subsonic over ~1–2 km and `C_d` varies sharply through Mach 1.
- **Start:** constant `C_d` with air density `ρ` from a standard-atmosphere `ρ(altitude)`.
- **Later:** a G1/G7-style Mach-dependent drag curve. Leave the seam.

### 3.3 M61A1 reference parameters (all tunable)

| Quantity | Value |
|---|---|
| Fire rate | ~6000 rpm = **100 rounds/s** |
| Muzzle velocity | ~**1030 m/s** |
| Round mass | ~**0.10 kg** (20 mm) |
| Magazine | **511 rounds** |
| Effective range | ~1200 m |
| Dispersion | ~8 mrad |

At 120 Hz that is **0.83 rounds/tick**, so `fire_guns` accumulates a fractional counter
(`Gun.accum`) and spawns a round when it crosses 1; releasing the trigger clears the partial
round. There is no spin-up delay. **Ammo is a plain `u16` counter** — no JSBSim mass/CG coupling.

---

## 4. Hit detection & damage

At ~1030 m/s and `dt = 8.33 ms` a round moves **~8.6 m/tick** — comparable to an F-16's size —
so a naive point-in-sphere test each tick **tunnels through targets**. `resolve_hits` uses a
**swept test** in relative coordinates (both bullet and target move during the tick):

```
r(s) = (bullet_prev + s·Δbullet) − (target_prev + s·Δtarget),   s ∈ [0, 1]
hit at the first s with |r(s)| ≤ R_hit   (combat::geometry::first_contact)
```

- `R_hit` = bounding sphere, **8 m** (`CombatParams::hit_radius_m`). A capsule (fuselage)
  remains a possible upgrade.
- Only living, **non-friendly** aircraft are tested (team exclusion). A round hits the target
  it reaches **first**; ties go to the lower id. The round is then despawned.

**Damage** (`Health { hp }`): each connecting round subtracts `damage_per_hit` = 2 HP (100 HP
default, so ~50 rounds ≈ a 0.5 s tracking burst kills). On `hp <= 0`, insert a `Dead` marker.

**Dead behavior (decision §12): freeze the FDM + insert a `Dead` marker** — clean, stable
episode boundaries for RL, and the wreck stays observable. Ballistic-fall / despawn rejected
(they muddy termination and risk despawn/NaN races).

---

## 5. Wire protocol (as shipped in schema v2)

Schema `crates/astrum-proto/schema/astrum.capnp` is the single source of truth. Rust
regenerates on `cargo build` (`astrum-proto/build.rs`), C++ via the CMake `capnp compile`
step, Python via runtime `capnp.load` (no codegen). The original plan here was additive
fields on the v1 schema (`fireTrigger @3`, `hp @9`, `ammo @10`); that plan is
**superseded**: the remediation replaced v1 with **schema v2** (a breaking change, all three
languages migrated together), which carries combat as follows.

### 5.1 Fire input

`Command.control` is a `union { raw, autopilot }` — you fly with *one or the other*, so
"fire" cannot live inside the union without losing flight control. It is a **plain top-level
field**, `Command.trigger @5 :Bool`, next to `seq` and `telemetryTick`. `ingest_commands`
(`astrum-sim/src/net.rs`, via `pilot::apply_command`) sets the per-aircraft `Trigger` that
`fire_guns` reads; an expired command lease releases it. An RL action = raw actuators **+**
trigger, sent together each tick.

### 5.2 Combat telemetry

`Telemetry.combat @13 :Combat` = `{ team, hitPoints, hitPointsMax, alive, ammo }`, filled from
the `Team` / `Health` / `Dead` / `Gun` components in the one telemetry assembly
(`LastPublished`) that feeds both the Zenoh publisher and the embedded env. The old v1
`health` byte became `faultFlags @7 :UInt8` ("0 = nominal", reserved for fault injection), so
the intent of decision §12.3 — keep a separate fault channel — survives.

### 5.3 Tracers / hit-kill events (Phase 3, not built)

These are **not** `Telemetry` fields. They would be a new Zenoh key space (e.g.
`astrum/projectile/*` or `astrum/events`) with a publisher/subscriber pair in
`astrum-net`. Recordings are MCAP with one channel per Zenoh key, so the recorder would only
need to subscribe to the new keys to capture them; the web exporter would need to render them.

---

## 6. Scenario, config & termination

`AircraftSpec` (`astrum-sim/src/scenario.rs`) carries optional `team` (default 0), `health`
(default 100) and `ammo` (armed aircraft only), copied into the components by
`initial_components`.

Scenario initial conditions are **geodetic (HAE) + heading + true airspeed**; fixed-wing
aircraft are trimmed for level flight at spawn (`trim = false` skips it; a failed trim is
fatal). That is sufficient for a merge setup; the Gym adapter's `scenario.py` builds its
two-`f16` scenarios this way, and `tests/common/mod.rs` has `two_f16_opposing()`.

**Termination:** `end_on_elimination` ends a ≥ 2-team engagement when at most one team has a
living aircraft (server mode writes `AppExit`; the embedded env reports `terminated`), and
`stop_at_max_time` checks the sim clock (`SimClock`, `tick · dt`).

**Demo wiring** (not built): a `make demo-dogfight` target + compose profile with one
controller per jet, mirroring `demo-formation`.

---

## 7. Determinism

For RL the embedded env is fully reproducible: the same scenario and seed reproduce an
episode bit-for-bit, `sim.speed` only changes wall-clock pacing (never step count or `dt`), and
results are identical for any `sim.threads` value (tested). Gun dispersion and each aircraft's
JSBSim random generator draw from `sim.seed` (`CombatRng`; shooters are visited in id order so
draws don't depend on ECS iteration order). Over the wire, external controllers make a run
reproducible only in **lockstep** mode (`sim.lockstep`); otherwise a command lands on whichever
tick it arrives.

---

## 8. RL environment seam & Gym-wrapper spec

The sim *is* the environment; the integration-test pattern (`build_app` + a bare
`app.update()` loop, unpaced) is already the RL stepping loop. Two paths were
considered:

- **(a) Over-the-wire Python Gym env** — reuse the controller/Zenoh harness
  (`controllers/python/astrum_client`): connect, telemetry = obs, command = action. Fast to
  prototype; Zenoh-coupled, awkward to reset/vectorize, reproducible only in lockstep.
- **(b) Embedded env (recommended for throughput)** — expose `build_app`/step via a thin
  Rust→Python binding (PyO3), call `app.update()` directly, no transport overhead, seedable.

Path **(b)** is the one we built. §8.1–§8.6 below are the concrete spec, locked
2026-08-25. (The original "separate processes, never threads" constraint is superseded; see
§8.6.)

### 8.1 The embedded seam — implemented (two layers)

The embedded env is built and tested; the Gym adapter (§8.2–§8.5) sits on top of
it in Python. Two layers keep the FFI shim trivial and the sim logic
`cargo test`-able without Python:

| Layer | Where | Surface |
|---|---|---|
| 1. Rust seam | `astrum-sim/src/env.rs` (`SimEnv`, `Action`, `StepResult`) | multi-agent `reset(seed, scenario)` / `step(&[(id, Action)])` / `observe`, observations are the sim's own `Telemetry` samples (sorted by id); one `step` = one `app.update()` = one tick; `RunMode::Embedded` (no pacing/progress/exit). Fast reset reuses the FDMs and is bit-identical to a fresh build. Tested by `astrum-sim/tests/env.rs`. |
| 2. PyO3 binding | `crates/astrum-env` (`Env`, `BatchEnv`, `StepInfo`, `FixedWingAutopilot`, `MultirotorAutopilot`) | `(n, 5)` NumPy actions in, `(n, len(OBS_FIELDS))` NumPy observations out; `Env` is `Send` and releases the GIL; `BatchEnv` steps K envs in parallel. Built by `scripts/build-astrum-env.sh`. |

`Action = [aileron, elevator, rudder, throttle, trigger]` — the four flight axes
are **FLCS NRT commands** (g / roll-rate / throttle with envelope protection, not
bare surfaces; §2.4), and `trigger` is a **learned** fire gate (§8), never
auto-aimed. Each observation row is the aircraft's telemetry: geodetic position (HAE),
world-grid ENU position and velocity, local-level NED velocity, attitude (φ/θ/ψ), body
rates (p/q/r), **α, β, Nz, Mach**, true airspeed, team, hit points, ammo, control mode, tick
and sim time. Schema v2 carries `p/q/r/α/β/Nz` on the wire too (`bodyRates`, `aero`), so
the embedded path is chosen for throughput, reset and reproducibility rather than for
data the wire lacks.

> **Nz sign:** `aero.nzG` is JSBSim's `accelerations/Nz` (≈ **+1** level, **+** in a
> pull), *not* the `n-pilot-z-norm` of the §2.4 gotcha (which is negative in a pull).
> So own-g needs **no** sign flip in the wrapper.

The wrapper's job (below) is everything the seam deliberately omits: egocentric
relative geometry, normalization, the gun-solution feature, reward/penalty, the
episode boundaries the sim doesn't enforce, and the scripted opponent.

### 8.2 Observation vector (egocentric, normalized ~[−1, 1])

Built in the Python wrapper (`geometry.build_observation`) from the two aircraft's
observation rows. **Egocentric** (everything in *my* body frame — absolute heading ψ is
dropped; it carries no signal without a reference). Angles enter as **sin/cos pairs** (no
wrap discontinuity). **31 floats** for 1v1 (`OBS_DIM`).

**Own state (frame-invariant):**

| feature | source | normalization |
|---|---|---|
| true airspeed | `tas` | `/ V_ref` (300 m/s) |
| height above hard deck | `alt_hae_m − deck` | `/ alt_ref` (6000 m) — the recovery-room signal |
| roll sin φ, cos φ | `phi` | — |
| pitch sin θ, cos θ | `theta` | — |
| body rates p, q, r | `p/q/r` | `/ rate_ref` (roll ≈3.3 rad/s = 188°/s; §2.4) |
| angle of attack α | `alpha` | `/ α_ref` (≈0.52 rad = 30° AoA limiter) |
| sideslip β | `beta` | `/ β_ref` |
| load factor Nz | `nz` | `(nz − 1) / 8` (0 at 1 g, ~1 at the 9 g cap) |
| own hit points | `hit_points` | `/ 100` |
| own ammo | `ammo` | `/ GUN_AMMO` (511) |

**Relative geometry (bandit in my body frame):** LOS = `p_bandit − p_me` (world grid),
rotated into my body axes (nose +X, starboard +Y, down +Z; §3) with the sim's own
`Env.grid_to_body` / `body_to_grid` (which include the local-level tilt).

| feature | definition | normalization |
|---|---|---|
| range R | `‖LOS‖` | `/ R_max` (or `log R / log R_max`), clamp |
| ATA azimuth sin/cos | `atan2(los_body_y, los_body_x)` — bearing off my nose | sin/cos |
| ATA elevation sin/cos | LOS elevation in body frame | sin/cos |
| aspect angle sin/cos | angle at bandit between its tail and LOS-to-me (low = I'm at his 6) | sin/cos |
| HCA sin/cos | heading-crossing angle (my vel vs bandit vel) | sin/cos |
| closure / range-rate | `−(v_rel · LOS_hat)` | `/ V_ref` |
| relative altitude | `alt_bandit − alt_me` | `/ alt_ref` |
| bandit airspeed | `tas` | `/ V_ref` |
| bandit hit points | `hit_points` | `/ 100` |
| bandit alive | flag | 0/1 |

**Gun-solution / WEZ feature** (from the sim's own fire control, `Env.gun_solution`, which
flies the same round model as §3 — drag, gravity, velocity inheritance, muzzle offset; an
observation aid, **never** auto-fire):

| feature | definition |
|---|---|
| in-range flag | the solution's `in_range` |
| lead-angle error (as cos) | `aim_error_rad`: angle between my boresight and the direction that would hit; small = tracking solution |
| gun-solution quality ∈ [0,1] | in-range × a tracking Gaussian on the aim error (feeds Φ below) |

### 8.3 Reward function

Hierarchy: potential-based geometry shaping ≪ damage ≪ terminal kill/death, with
the floor the single worst outcome. Magnitudes below are the locked starting
point (all tunable).

| event | reward | kind |
|---|---|---|
| hard-deck bust (< deck, HAE) | **−200** | terminal |
| killed (own hp → 0) | **−100** | terminal |
| kill (bandit hp → 0) | **+100** | terminal |
| leave engagement bubble | **−50** | truncate |
| timeout (max_time) | 0 | truncate |
| damage dealt | **+0.5 / HP** (a 2-HP hit = +1.0) | dense |
| damage taken | **−0.5 / HP** | dense |
| geometry shaping | **k_geom·(γΦ′ − Φ)**, k_geom ≈ 0.05–0.1 | dense |
| per step | **−0.001** | dense |

Floor (−200) ≥ death (−100): flying into the ground is at best as bad as losing
the fight. Damage is inherently zero-sum in self-play.

**Potential-based shaping** (Ng et al. — `γΦ(s′) − Φ(s)` telescopes over an
episode, so it can't distort the optimal policy or swamp the +100 kill). Φ ∈
[0, 1] is the product of three "clean gun-kill geometry" factors, so *all* must be
good:

```
Φ = f_ATA · f_aspect · f_range
    f_ATA    = (1 + cos ATA)/2       # 1 = nose-on the bandit, 0 = pointed away
    f_aspect = (1 + cos aspect)/2    # 1 = at the bandit's 6 o'clock
    f_range  = gun-solution quality  # in-WEZ, from §8.2
```

Reward is nose-on **∧** rear-hemisphere **∧** in-WEZ — the corrected sign (we want
**low** ATA; rewarding high ATA would train the agent to point *away*).

### 8.4 Episode boundaries (wrapper-enforced)

The sim has **no** arena and **no terrain model** — the ground is the WGS84 ellipsoid
(rounds are culled at `h = 0`; an aircraft's ground contact is whatever JSBSim's gear model
does). It only ends on elimination (`terminated`) or `max_time_s` (`truncated`). The
wrapper adds two boundaries:

- **Hard deck:** height above the ellipsoid `< 300 m` (`DECK_HAE_M`) → **terminate**, −200.
- **Engagement bubble:** horizontal range from the grid origin `> R_bubble` (15 km) or
  height `> ceiling` (15 km HAE) → **truncate**, −50.

### 8.5 Scripted opponent (curriculum) — driven through the raw action interface

Self-play comes later; we validate obs + reward against a **scripted** bot first.
It is just another agent id fed to `step()`; **all its logic lives in Python**,
computing `[aileron, elevator, rudder, throttle, fire]` each tick through the raw
interface. Its level hold is the sim's own fixed-wing law, exposed by the binding as
`astrum_env.FixedWingAutopilot("fighter", dt)` (PI on bank, altitude — with climb-rate
damping — and airspeed; the same code `autopilot_step` runs), so there is no Python port
to drift. Maneuvers bypass it with open-loop raw commands, because the hold law
**cannot fire** (no trigger) and **cannot fly vertical BFM** (it fights any
inverted/vertical state). The FLCS envelope protection (§2.4) keeps the bot from
departing regardless.

**Curriculum (increasing difficulty):**

| Tier | Behavior | Validates |
|---|---|---|
| **1 — Target dummy** | straight & level, constant Mach/alt (e.g. M0.6, 15 kft) | flight-control acquisition, WEZ alignment, ammo conservation, basic intercept geometry |
| **2 — Predictable defense** | fixed high-g maneuvers on detecting an attacker in its rear hemisphere: hard level turn, break turn, Split-S | energy management, corner-speed hold, lead/pure pursuit without overshoot or stall |
| **3 — Rule-based tactician** | a 4-state FSM on standard BFM rules-of-thumb (below) + a "fire when attacker in my WEZ" trigger | non-cooperative dynamic geometry & counter-attacks before self-play's non-stationarity |

**Status (2026-08-25):** Tiers 1 & 2 implemented in
`crates/astrum-env/python/astrum_dogfight` (`Tier1Opponent`; `Tier2Opponent` with
`BreakTurn` / `SplitS`, corner-speed-limited pull, and post-maneuver cooldown for
dwell). A maneuver is an open-loop phase machine over raw `[ail,elev,rud,thr]`,
triggered when the attacker is detected aft of the 3-9 line within `R_DETECT`
(4 km). Verified against a live two-f16 rollout (`test_tier2_break`,
`test_tier2_split_s`). Both are non-shooting; the WEZ trigger arrives with Tier 3.

**Tier 3 also implemented** (`Tier3Opponent` + `wez_fire`): the 5-state FSM below with
dead-banded, dwell-gated transitions (a genuine threat preempts into DEFENSIVE at once),
lead-pursuit aiming (`geometry.lead_point` / `body_bearing` — lift-vector-on-target),
corner-limited pulls, and a state-independent WEZ gun gate (fire when a live target is in
`R_eff` with lead error < 4°). Verified live end-to-end: on the six it enters OFFENSIVE
and **guns the learner to 0 HP** through the sim's real `fire_guns`→`resolve_hits` pipeline;
facing a nose-on threat it enters DEFENSIVE and breaks ~180°. (HIGH-YO-YO / ENERGY-RECOVERY
laws are implemented and unit-checked for finiteness; the straight-line live tests exercise
NEUTRAL/OFFENSIVE/DEFENSIVE.)

**Tier-3 FSM** (deterministic, driven by geometry):

```
NEUTRAL (fly intercept / lead)
  ├─ target in front & close ─────────────▶ OFFENSIVE (lead pursuit + fire guns)
  │                                            └─ target overshoots ─▶ HIGH YO-YO (pitch up, dump speed)
  └─ target behind (I'm in his WEZ) ──────▶ DEFENSIVE (hard break, high-g)
                                               └─ airspeed bleeds low ─▶ ENERGY RECOVERY (unload g, nose down)
```

Three implementation notes:

- **Hysteresis + dwell required.** Transitions on raw ATA/aspect thresholds will
  chatter at the boundary (OFFENSIVE↔DEFENSIVE). Each transition gets a dead-band
  and a minimum dwell time.
- **Godmode state is fine.** The bot reads both aircraft's full observation rows
  (perfect state, no sensor model) — standard for a scripted baseline.
- **Corner speed ← Nz.** "Hold v_corner / don't depart" caps the FSM's commanded
  pull on `nz` / `alpha` — the same fields the policy observes (§8.2).

Start the opponent **non-shooting** (agent learns pure offense first); the WEZ
trigger is added at Tier 3. Hand-coded aiming is legitimate *for the scripted
opponent* — never for the learner.

### 8.6 Parallelism (recap) — supersedes the original "processes only" rule

The original design assumed JSBSim could only be used from one thread per process. The
root cause of the crashes behind that rule was found and fixed: JSBSim lazily fills a
process-global unit-conversion table and writes a static debug level in every constructor.
`astrum-fdm` now owns a process-wide lock (lifecycle calls exclusive, stepping shared), so
`Fdm` and `astrum_env.Env` are `Send`, `Env` releases the GIL in construction, `reset` and
`step`, and envs can run on threads. `BatchEnv` steps K envs in parallel (rayon) in one
call, the cheapest way to vectorize; process-level vec-envs (`AsyncVectorEnv` /
`SubprocVecEnv`) also work. *Within* an env the FDMs are stepped on a `sim.threads`-sized
pool with bit-identical results for any thread count, and the env is inherently
multi-agent, which is what NvN / self-play needs.

---

## 9. Visualization (browser replayer)

- **Done:** team / HP / ammo reach the browser through `FrameJson` (`team`, `hit_points`,
  `hit_points_max`, `alive`, `ammo`; `astrum-web/src/lib.rs`) and the readout panel
  (`web/js/ui.js`).
- **Later (Phase 3):** bullet tracers + hit/kill markers, driven by the projectile/events
  keys from §5.3.

---

## 10. Testing (mirrors existing conventions)

Combat and env tests live in `crates/astrum-sim/tests/` (shared builders such as
`two_f16_opposing()` in `tests/common/mod.rs`). No test-side JSBSim locking is needed:
`astrum-fdm` owns the lock.

- **Ballistics** (`ballistics.rs`, no Bevy): vacuum drop matches free fall, drag bleeds
  speed, Coriolis deflects right in the northern hemisphere, ISA density.
- **Hit test** (`combat.rs`, `combat_geometry.rs`): a round stepping past a target is
  caught by the swept test, a target moving during the tick is tested where it was, the
  first-reached target takes the hit, friendly fire does nothing, rounds below the ellipsoid
  are culled; body↔grid rotations including the local-level tilt.
- **Damage/kill and determinism** (`combat.rs`, `combat_rng.rs`): a sustained burst kills and
  decides the engagement; a seeded engagement is reproducible.
- **Fire control** (`fire_control.rs`): `gun_solution` against known geometry.
- **Wire** (`astrum-proto/tests/wire.rs`, `fixtures.rs`; `astrum-sim/tests/net_bridge.rs`):
  `trigger`/`combat` round-trip, cross-language fixtures, command delivery over Zenoh.
- **Env** (`env.rs`): seeded episodes and fast reset are bit-identical to a fresh build.

---

## 11. Phasing & rough effort

| Phase | Content | Wire change? | Size |
|---|---|---|---|
| **0** | Lock decisions (§12) | — | discussion |
| **1** | Sim mechanics: `Gun`/`Health`/`Team`/`Bullet` components; `fire_guns`/`step_bullets`/`resolve_hits`/`apply_damage`/`end_on_elimination` systems; dogfight scenario; full test suite. Validatable headless with a scripted "fire straight" — no controller/wire needed. | No | **M** (core) |
| **2** | `Command.trigger`; `Telemetry.combat`; populate/ingest; C++/Python pick-up; web HUD (HP/ammo). **Done** (as schema v2). `make demo-dogfight` + a scripted pursuit-and-fire compose controller: **not built**. | Yes | **M** |
| **3** | Tracers + hit/kill events (new keys, recorded as extra MCAP channels; `astrum-web` render). **Not built.** | Yes | **M–L** |
| **4** | RL env (embedded PyO3), obs/action/reward, vectorization, curriculum. **Done** except self-play/training. | binding | **L** |

Phases 1–2 yield a real, deterministic gun-combat sim you can fly two jets in. Phase 3 is
observability/polish. Phase 4 is the RL payoff.

---

## 12. Decisions (locked 2026-08-25)

1. **Drag fidelity** — **constant `C_d` + `ρ(altitude)`** to start; leave the Mach-dependent
   G1/G7 seam (§3.2). Enough for WVR time-of-flight (~1 s); the drag curve is tuning, not
   structure.
2. **Earth Coriolis** — **include** the `−2Ω×v` term, clearly commented as negligible (~cm at
   WVR). Do **not** add a "turn-Coriolis" force — world-frame integration already reproduces the
   apparent curve (§3.1b); a turn force would double-count.
3. **`health @7` semantics** — add a dedicated hit-point field and keep a separate
   fault-state channel. *Superseded in form by schema v2:* hit points, team, alive and ammo
   live in `Telemetry.combat`, and the fault channel is `faultFlags @7` ("0 = nominal").
4. **Projectile representation** — **bullet entities** (§2.3). Required for tracers (§9) and
   honest drop/lead/Coriolis; live population is trivial.
5. **Dead-aircraft behavior** — **freeze + insert a `Dead` marker** (§4). Stable terminal state
   for RL episodes; the wreck stays observable; no despawn/NaN races.
6. **First-pass depth** — **Phase 1 only** (sim core + full headless tests, scripted fire).
   Defer Phase 2's wire/demo: the embedded RL env (§8b) drives `app.update()` directly and
   never touches the wire. *Since overtaken:* the wire half of Phase 2 shipped with schema v2;
   the dogfight demo has not.

---

## Appendix: key code references

| Concern | Location |
|---|---|
| App build + system chain + `dt` | `astrum-sim/src/lib.rs` (`build_app`, `sim_params`) |
| Spawn / arming / trim | `astrum-sim/src/systems.rs` (`spawn_fleet`, `initial_components`) |
| FDM stepping (per-aircraft shard workers) / FDM→grid state | `astrum-sim/src/step_pool.rs` (`ShardPool`), `systems.rs` (`StepPool`, `step_fdm`, `grid_state`) |
| State components (`AircraftState`, `SimParams`, `SimClock`) | `astrum-sim/src/components.rs` |
| Command ingest / lease / failsafe | `astrum-sim/src/net.rs` (`ingest_commands`), `pilot.rs` (`apply_command`, `expire_leases`) |
| Built-in autopilot laws | `astrum-sim/src/pilot.rs` (`FixedWingGains`, `autopilot_step`) |
| Telemetry assembly / publish | `astrum-sim/src/net.rs` (`LastPublished`, `publish_telemetry`) |
| Combat chain | `astrum-sim/src/combat/` (`systems.rs`, `ballistics.rs`, `geometry.rs`, `fire_control.rs`) |
| Termination | `astrum-sim/src/systems.rs` (`stop_at_max_time`), `combat/systems.rs` (`end_on_elimination`) |
| Pacing (`sim.speed`); lockstep wait (`control_rate_hz`) | `astrum-sim/src/systems.rs` (`Pacer`, `pace`); `net.rs` (`ingest_commands`) |
| Run control (pause / step / speed), status; controller presence | `astrum-sim/src/control.rs` (`RunControl`, `serve_control`, `publish_status`); `net.rs` (`connect`, `Controllers`, `track_controllers`, `HeldCommands`); `astrum-net/src/lib.rs` (`ControlServer`, `PresenceWatcher`), `src/bin/ctl.rs` |
| In-process recording (`--record`) | `astrum-sim/src/record.rs` (`TickRecorder`, `Recording`) |
| Sweeps (`astrum-sweep`) | `astrum-sim/src/sweep.rs` (planning, in-process runs, summary), `src/bin/sweep.rs` (process mode) |
| Wire schema | `astrum-proto/schema/astrum.capnp` (`Telemetry`, `Command`, `Combat`) |
| Encode / decode | `astrum-proto/src/lib.rs` |
| Zenoh keys / transport | `astrum-net/src/lib.rs` (`keys`, `qos`, `LatestTelemetry`, `CommandSender`) |
| Geodetic↔ECEF↔ENU, local-level rotations | `astrum-geo/src/lib.rs` (`LocalFrame`) |
| Scenario structs / load / validate | `astrum-sim/src/scenario.rs` (`Scenario`, `AircraftSpec`) |
| Embedded env | `astrum-sim/src/env.rs` (`SimEnv`), `crates/astrum-env/src/lib.rs` |
| Test harness | `astrum-sim/tests/common/mod.rs` (`two_ship`, `two_f16_opposing`) |
| Controller obs/act pattern | `controllers/python/astrum_client` (subscribe, decode, send) |
