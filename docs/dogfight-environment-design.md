# Design: adversarial WVR gun-combat environment for the F-16

**Status:** §12 decisions locked 2026-08-25. The gun/damage combat system
(`astrum-sim/src/combat`: `Gun`/`Trigger`/`Bullet`/`Health`, `fire_guns` →
`step_bullets` → `resolve_hits` → `apply_damage`/`end_on_elimination`) and the
embedded RL seam (§8.1, `astrum-sim/src/env.rs` + `crates/astrum-env`) are
**implemented**. The numpy-only Gym adapter (`crates/astrum-env/python/astrum_dogfight`)
plus the **full scripted-opponent curriculum (Tiers 1–3)** are implemented and
tested — the Tier-3 FSM tactician guns the learner down end-to-end through the real
combat pipeline. **Self-play** (and adding gymnasium + SB3 to the dev image) is next.
**Date:** 2026-08-25
**Scope:** Add a nose-mounted gun (M61A1 Vulcan), realistic bullet ballistics, and a
health/damage system to the ASTRUM sim, so two (or more) F-16s can fight within visual
range. The end goal this enables is training an RL agent for WVR dogfighting.

This document is a map of *what to build and where it plugs in*, grounded in the current
codebase. It deliberately does not commit to line-by-line implementation; the "Decisions to
lock" section (§11) lists the forks to settle before coding.

> **New to the env and just want to train (or hand-code) a policy?** Start with
> [`dogfight-env-onboarding.md`](dogfight-env-onboarding.md) — the practical
> build/run/interface guide. This doc is the design rationale behind it (§8 is the
> RL seam).

---

## 1. Feasibility summary

Greenfield but well-supported. There is **no** gun/bullet/weapon/damage code anywhere in the
repo today — the only adjacent primitive is an inert `health` field on the wire. Every seam
needed already exists:

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

The sim is a headless Bevy app (`crates/astrum-sim`). It uses `MinimalPlugins` +
`ScheduleRunnerPlugin::run_loop(Duration::ZERO)`; **there is no `FixedUpdate`** — all physics
runs in the plain `Update` schedule, so **one `App::update()` == exactly one physics tick** at
`dt = 1/rate_hz` (default **120 Hz → 8.33 ms**, `SimParams` in `components.rs:81-89`).

Systems run `.chain()`-ordered in `build_app` (`astrum-sim/src/lib.rs:66-81`):

```
ingest_commands → autopilot_step → step_fdm → sync_geo_transform → publish_telemetry
    → report_progress → stop_at_max_time → pace
```

**Insertion point for combat** — after `sync_geo_transform` (ENU `Transform` + `GeoState` are
fresh) and before `publish_telemetry` (so HP/ammo ship this tick):

```
… sync_geo_transform → fire_guns → step_bullets → resolve_hits → apply_damage
    → end_on_elimination → publish_telemetry …
```

Register the systems + any new resources (seeded RNG, combat params) alongside the existing
`insert_resource(...)` calls in `build_app` (`lib.rs:60-64`).

### 2.2 New ECS components

Add to `crates/astrum-sim/src/components.rs`; attach in the `spawn_fleet` bundle
(`systems.rs:58-65`), gated on fixed-wing / model `"f16"` exactly like `FixedWingGains`
(`systems.rs:69-71`):

- `Gun { muzzle_offset_body: Vec3, rate_hz: f32, muzzle_speed_mps: f32, ammo: u16,
  dispersion_mrad: f32, spin_up_s: f32, accum: f32 }` — `accum` carries the fractional
  rounds-per-tick counter.
- `Health { hp: f32, hp_max: f32 }`.
- `Team(u8)` — for owner/target exclusion; generalizes 1v1 → NvN.
- `Trigger(bool)` — per-aircraft firing state, set by `ingest_commands`, read by `fire_guns`.
- `Dead` — marker inserted when `hp <= 0`.

### 2.3 Bullets are their own entities — but NOT FDM entities

Each round is a lightweight entity: `Bullet { vel_enu: Vec3, owner_id: u32, team: u8,
ttl_s: f32 }` + a Bevy `Transform` for its ENU position. Integrated with plain point-mass
kinematics.

> **Hard constraint:** JSBSim is process-global and **not thread-safe** (see
> `crates/astrum-sim/tests/common/mod.rs` `fdm_guard()`). Bullets must **never** own an
> `FdmInstance`. This also bounds RL parallelism (see §8).

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
   `/opt/jsbsim/share/jsbsim/aircraft/f16/f16.xml`, not in the repo tree. Reference maneuver:
   `crates/astrum-sim/tests/split_s.rs` + `examples/split_s_bag.rs` (kept uncommitted).

**Observation gotcha (§8):** `accelerations/n-pilot-z-norm` is **negative for a positive-g
pull** (~−1 level upright, ~−6.9 in a hard pull). Whatever surfaces own-g/energy in the
observation must flip the sign (or use `|n_z|`).

---

## 3. Ballistics model

Three requested effects — "shoot in front of the enemy," "bullet drop," "coriolis when
banking/turning" — resolve into **two genuine projectile forces plus one aiming problem**.
Getting the framing right makes the implementation correct *and* cheaper.

**Simulate every round as a free projectile in the ENU world frame.** ENU is ASTRUM's
canonical grid: `East=+X, North=+Y, Up=+Z`, right-handed (`astrum-geo/src/lib.rs:1-6`).
Attitude is aerospace Euler on the wire (`phi`=roll, `theta`=pitch, `psi`=yaw/true heading,
radians; body axes nose=+X, starboard=+Y, down=+Z; `ψ=0` faces North, `+ψ` yaws East). The
boresight (muzzle direction) is:

```
nose_hat_ENU = ( cosθ·sinψ,  cosθ·cosψ,  sinθ )        # (E, N, U)
```

(equals `(fE, fN, −fD)` from `web/js/meshes.js:40`, the authoritative attitude math). Initial
round velocity at the trigger instant:

```
v0 = v_aircraft_ENU  +  (ω_body × r_muzzle) rotated to ENU  +  muzzle_speed · nose_hat_ENU
     (+ optional dispersion cone, seeded)
```

Per-tick integration (semi-implicit Euler at `dt`):

```
a = (0, 0, −g)                          # gravity → bullet drop
  − (½·ρ·C_d·A / m)·|v|·v                # aerodynamic drag  (dominant term at gun range)
  − 2·(Ω_ENU × v)                        # Earth Coriolis, Ω_ENU = Ω·(0, cosφ0, sinφ0)
v += a·dt ;  x += v·dt
```

`φ0` = scenario ENU origin latitude (`LocalFrame`); `Ω = 7.292e-5 rad/s`.

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
force acted. Integrating in ENU with correct muzzle-velocity inheritance (`v_aircraft +
ω×r_muzzle`) reproduces this exactly, and the apparent curve shows up correctly when viewed
from the jet. Adding a fake "turn-coriolis" acceleration would **double-count and be wrong.**
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
(`Gun.accum`) and spawns a round when it crosses 1. Optional ~0.4 s spin-up to full rate.
Per requirement, **ammo is a plain `u16` counter** — no JSBSim mass/CG coupling.

---

## 4. Hit detection & damage

At ~1030 m/s and `dt = 8.33 ms` a round moves **~8.6 m/tick** — comparable to an F-16's size —
so a naive point-in-sphere test each tick **tunnels through targets**. Use a **swept test** in
relative coordinates (both bullet and target move during the tick):

```
r0 = bullet_start − target_start
r1 = bullet_end   − target_end
hit  ⇔  min distance of segment (r0 → r1) to the origin  ≤  R_hit
```

- Baseline `R_hit` = bounding sphere (~8 m). Upgrade to a capsule (fuselage) later for
  aspect-dependent realism.
- Resolve only bullet-vs-**opposing** entity (owner id / team exclusion).

**Damage** (`Health { hp }`): each connecting round subtracts a tunable amount so a ~0.5 s
tracking burst kills. On `hp <= 0`, insert a `Dead` marker.

**Dead behavior (decision §12): freeze the FDM + insert a `Dead` marker** — clean, stable
episode boundaries for RL, and the wreck stays observable. Ballistic-fall / despawn rejected
(they muddy termination and risk despawn/NaN races).

---

## 5. Wire-protocol changes (additive, all three languages)

Schema `crates/astrum-proto/schema/astrum.capnp` is the single source of truth. Rust
regenerates on `cargo build` (`astrum-proto/build.rs`), C++ via each CMake `capnp compile`
custom command, Python via runtime `capnp.load` (no codegen). All changes are **additive with
new field ordinals**, following the existing `vehicleKind @8` precedent.

### 5.1 Fire input

`Command.control` is today a `union { raw @1, autopilot @2 }` — you fly with *one or the
other*, so "fire" cannot live inside the union without losing flight control. Add a **plain
top-level field**:

```capnp
struct Command {
  targetId    @0 :UInt32;
  control :union { raw @1 :RawActuators;  autopilot @2 :AutopilotTargets; }
  fireTrigger @3 :Bool;        # NEW — coexists with flight control
}
```

Handle it in `ingest_commands` (`astrum-sim/src/net.rs:35-67`) by setting the per-aircraft
`Trigger` state that `fire_guns` reads. An RL action = raw actuators **+** trigger, sent
together each tick.

### 5.2 Combat telemetry

`health @7 :UInt8` **already exists end-to-end** (schema `astrum.capnp:48`,
`TelemetrySample.health` in `proto/src/lib.rs:51`, encode/decode `:75`/`:127`) but is hardcoded
`0` at `astrum-sim/src/net.rs:308` and documented "0 = nominal." **Decision (§12): keep
`health @7` as fault/damage-state flags (its "0 = nominal" meaning stands) and add a dedicated
`hp @9 :UInt16`** (0 = destroyed) — avoids inverting the documented meaning and leaves a channel
for a future damage model (engine/wing/hydraulics hits). Also add `ammo @10 :UInt16`. Populate
both from the new components in `publish_telemetry`.

### 5.3 Tracers / hit-kill events (Phase 3, optional)

These are **not** `Telemetry` fields. They would be a new Zenoh topic (e.g.
`astrum/projectile/*` or `astrum/events`) with a publisher/subscriber pair added to
`astrum-net/src/lib.rs` (mirroring `TelemetryPublisher`/`TelemetrySubscriber`, new prefix
constant + key builder). To make them **recordable/replayable**, `BagFrame`
(`astrum.capnp:76-81`, which carries only `Telemetry` today) and the bag encode/decode
(`proto/src/lib.rs`) + recorders (`crates/astrum-bag`) must be extended. Piggybacking HP/ammo
on `Telemetry` needs none of this; only tracers/events do.

---

## 6. Scenario, config & termination

Add optional fields to `AircraftSpec` (`astrum-sim/src/scenario.rs:50`), serde-defaulted so
existing scenarios are untouched: `team`, `health`, `ammo`. Carry them into the new components
in `spawn_fleet`.

Scenario initial conditions are **geodetic + heading + airspeed only** (attitude/velocity fall
out of JSBSim trim after `run_ic`); that is sufficient for a merge setup. New
`config/scenario.dogfight.toml` = two `f16` on opposing headings (mirror the `two_ship`
builder / `config/scenario.example.toml`).

**Termination gap:** `stop_at_max_time` (`systems.rs:203-220`) checks only the *first*
aircraft's clock — there is no "end on kill." For episodic RL, add an `end_on_elimination`
system (last-team-standing / out-of-ammo / timeout) that writes `AppExit`.

**Demo wiring** mirrors `demo-formation`: a `make demo-dogfight` target sets `ASTRUM_SCENARIO`
+ a `--profile dogfight` compose profile with `controller-red`/`controller-blue` services
pinned by `ASTRUM_TARGET_ID` (clone the formation controller pattern; remember to add the
target to `.PHONY`).

---

## 7. Determinism

Non-negotiable for RL and already the sim's design: `realtime:false` runs the loop as fast as
the host allows with identical trajectories (pacing only touches wall-clock, never step count
or `dt`). Any randomness (gun dispersion) must come from a **seeded** Bevy RNG resource, not
`thread_rng`; expose the seed in the scenario.

---

## 8. RL environment seam & Gym-wrapper spec

The sim *is* the environment; the integration-test pattern (`build_app` + a bare
`app.update()` loop at `realtime:false`) is already the RL stepping loop. Two paths:

- **(a) Over-the-wire Python Gym env** — reuse the controller/Zenoh harness
  (`controllers/optimal/python/astrum_control.py`): connect, telemetry = obs, command =
  action. Fast to prototype; Zenoh-coupled, awkward to reset/vectorize.
- **(b) Embedded env (recommended for throughput)** — expose `build_app`/step via a thin
  Rust→Python binding (pyo3), call `app.update()` directly, read ECS state, rebuild for
  `reset()`. Deterministic, no transport overhead, seedable.

> **Hard constraint:** JSBSim's process-global state means parallel/vectorized envs must be
> **separate processes, not threads.** Design the vec-env around multiprocessing from day one.

Path **(b)** is the one we build. §8.1–§8.6 below are the concrete spec, locked
2026-08-25.

### 8.1 The embedded seam — implemented (two layers)

The embedded env is built and tested; the Gym adapter (§8.2–§8.5) sits on top of
it in Python. Two layers keep the FFI shim trivial and the sim logic
`cargo test`-able without Python:

| Layer | Where | Surface |
|---|---|---|
| 1. Rust seam | `astrum-sim/src/env.rs` (`SimEnv`, `Action`, `AgentObs`, `StepResult`) | multi-agent `reset` / `step(&[(id, Action)])` / `observe`; one `step` = one `app.update()` = one tick. Tested by `astrum-sim/tests/env.rs`. |
| 2. pyo3 binding | `crates/astrum-env` (`DogfightEnv`, `Observation`, `StepInfo`) | `#[pyclass(unsendable)]` — pins the env to its process. Built by `run_smoke.sh`. |

`Action = [aileron, elevator, rudder, throttle, trigger]` — the four flight axes
are **FLCS NRT commands** (g / roll-rate / throttle with envelope protection, not
bare surfaces; §2.4), and `trigger` is a **learned** fire gate (§8), never
auto-aimed. `AgentObs` exposes each aircraft's raw world-frame state: ENU
position, NED velocity, attitude (φ/θ/ψ), body rates (p/q/r), **α, β, Nz**,
airspeed, altitude, health, ammo, sim-time. `p/q/r/α/β/Nz` are read straight from
JSBSim — they are **not on the wire** — which is a decisive reason the env is
embedded, not over-the-wire.

> **Nz sign:** the env reads `accelerations/Nz` (≈ **+1** level, **+** in a pull —
> verified: ~+1.95 g in a hard pull), *not* the `n-pilot-z-norm` of the §2.4
> gotcha (which is negative in a pull). So own-g needs **no** sign flip in the
> wrapper.

The wrapper's job (below) is everything the seam deliberately omits: egocentric
relative geometry, normalization, the gun-solution feature, reward/penalty, the
episode boundaries the sim doesn't enforce, and the scripted opponent.

### 8.2 Observation vector (egocentric, normalized ~[−1, 1])

Built in the Python wrapper from the two agents' `AgentObs`. **Egocentric**
(everything in *my* body frame — absolute heading ψ is dropped; it carries no
signal without a reference). Angles enter as **sin/cos pairs** (no wrap
discontinuity). ~28–32 floats for 1v1.

**Own state (frame-invariant):**

| feature | source | normalization |
|---|---|---|
| airspeed | `airspeed_mps` | `/ V_ref` (≈250 m/s) |
| altitude above hard deck | `alt_m − deck_alt` | `/ alt_ref` (≈6000 m), clamp — the recovery-room signal |
| roll sin φ, cos φ | `phi` | — |
| pitch sin θ, cos θ | `theta` | — |
| body rates p, q, r | `p/q/r` | `/ rate_ref` (roll ≈3.3 rad/s = 188°/s; §2.4) |
| angle of attack α | `alpha_rad` | `/ α_ref` (≈0.52 rad = 30° AoA limiter) |
| sideslip β | `beta_rad` | `/ β_ref` |
| load factor Nz | `nz_g` | `(nz_g − 1) / 8` (0 at 1 g, ~1 at the 9 g cap) |
| own health | `health` | `/ 100` |
| own ammo | `ammo` | `/ 511` |

**Relative geometry (bandit in my body frame):** LOS = `p_bandit − p_me` (ENU),
rotated into my body axes (nose +X, starboard +Y, down +Z; §3) via my (φ, θ, ψ).

| feature | definition | normalization |
|---|---|---|
| range R | `‖LOS‖` | `/ R_max` (or `log R / log R_max`), clamp |
| ATA azimuth sin/cos | `atan2(los_body_y, los_body_x)` — bearing off my nose | sin/cos |
| ATA elevation sin/cos | LOS elevation in body frame | sin/cos |
| aspect angle sin/cos | angle at bandit between its tail and LOS-to-me (low = I'm at his 6) | sin/cos |
| HCA sin/cos | heading-crossing angle (my vel vs bandit vel) | sin/cos |
| closure / range-rate | `−(v_rel · LOS_hat)` | `/ V_ref` |
| relative altitude | `alt_bandit − alt_me` | `/ alt_ref` |
| bandit airspeed | `airspeed_mps` | `/ V_ref` |
| bandit health | `health` | `/ 100` |
| bandit alive | flag | 0/1 |

**Gun-solution / WEZ feature** (wrapper-computed from ballistics §3; an
observation aid, **never** auto-fire):

| feature | definition |
|---|---|
| in-range flag | `R ≤ R_eff` (≈1200 m) |
| lead-angle error | angle between my boresight `nose_hat` and the TOF-lead direction to the bandit; small = tracking solution |
| gun-solution quality ∈ [0,1] | high when in-range **and** lead error small **and** aspect favorable (feeds Φ below) |

### 8.3 Reward function

Hierarchy: potential-based geometry shaping ≪ damage ≪ terminal kill/death, with
the floor the single worst outcome. Magnitudes below are the locked starting
point (all tunable).

| event | reward | kind |
|---|---|---|
| hard-deck bust (< deck AGL) | **−200** | terminal |
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

The sim has **no** arena or ground-collision handling — it only ends on
elimination (`terminated`) or `max_time_s` (`truncated`). ENU is unbounded and
there is no floor. The wrapper adds two:

- **Hard deck:** `alt AGL < 300 m` (≈1000 ft — above real terrain, leaving
  recovery room) → **terminate**, −200.
- **Engagement bubble:** horizontal range from arena center `> R_bubble`
  (≈10–20 km) or altitude `> ceiling` (≈15 km) → **truncate**, −50.

### 8.5 Scripted opponent (curriculum) — driven through the raw action interface

Self-play comes later; we validate obs + reward against a **scripted** bot first.
It is just another agent id fed to `step()`; **all its logic lives in Python**,
computing `[aileron, elevator, rudder, throttle, fire]` each tick from a small
inner loop **ported from the sim's own `autopilot_step` / `FixedWingGains::f16()`**
(bank-to-turn, pitch-for-alt/g, throttle-for-speed; ~50 lines). This one control
path — rather than exposing the in-sim autopilot — is required because that
autopilot is a heading/alt/airspeed hold that **cannot fire** (no trigger) and
**cannot fly vertical BFM** (it fights any inverted/vertical state). The FLCS
envelope protection (§2.4) keeps the bot from departing regardless.

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
(4 km). Verified against a live two-f16 rollout: the break swings heading ~150°,
the Split-S reverses ~170° and sheds ~2.1 km (matching `astrum-sim/tests/split_s.rs`).
Both are non-shooting; the WEZ trigger arrives with Tier 3.

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
- **Godmode state is fine.** The bot reads both aircraft's full `AgentObs`
  (perfect state, no sensor model) — standard for a scripted baseline.
- **Corner speed ← Nz.** "Hold v_corner / don't depart" caps the FSM's commanded
  pull on `nz_g` / `alpha_rad` — the same fields the policy observes (§8.2).

Start the opponent **non-shooting** (agent learns pure offense first); the WEZ
trigger is added at Tier 3. Hand-coded aiming is legitimate *for the scripted
opponent* — never for the learner.

### 8.6 Parallelism (recap)

JSBSim is process-global (§2.3): scale with **separate processes** (Gymnasium
`AsyncVectorEnv` / SB3 `SubprocVecEnv`), one env per worker, never threads. The
`#[pyclass(unsendable)]` binding enforces this at runtime. *Within* a process the
env is inherently multi-agent (N aircraft on one FDM-sequential app), which is
what NvN / self-play needs.

---

## 9. Visualization (browser replayer)

- **Cheap now:** HP/ammo HUD via the existing `health` field (already forwarded to `FrameJson`,
  `astrum-web/src/lib.rs`) + the new `ammo` field; surfaced in `web/js/store.js` (`NUM_FIELDS`)
  and the readout panel.
- **Later (Phase 3):** bullet tracers + hit/kill markers, driven by the projectile/events topic
  from §5.3 (needs the bag extension to replay offline).

---

## 10. Testing (mirrors existing conventions)

All in `crates/astrum-sim/tests/` (+ a `two_f16_opposing()` builder in `tests/common/mod.rs`),
each taking `fdm_guard()` first:

- **Ballistics unit test** (low-level, no Bevy — mirror `astrum-fdm/tests/fdm.rs`): fire
  horizontally, assert drop ≈ `½g·t²`, drag slows the round, Coriolis sign is correct.
- **No-tunnel test:** a fast round through a target-sized sphere registers a hit (guards the
  swept test).
- **Determinism:** run a fixed firing scenario twice; assert identical `snapshot()` + HP
  (mirror `tests/simulation.rs`).
- **Damage/kill:** scripted tracking burst drives opponent `hp → 0` and triggers termination.
- **Wire round-trip:** `fireTrigger`/`ammo` encode/decode (`proto/tests/wire.rs`) and command
  delivery (`tests/net_bridge.rs`).

Note: `astrum-sim/Cargo.toml` has no `[dev-dependencies]`; tests use only workspace crates +
std.

---

## 11. Phasing & rough effort

| Phase | Content | Wire change? | Size |
|---|---|---|---|
| **0** | Lock decisions (§12) | — | discussion |
| **1** | Sim mechanics: `Gun`/`Health`/`Team`/`Bullet` components; `fire_guns`/`step_bullets`/`resolve_hits`/`apply_damage`/`end_on_elimination` systems; dogfight scenario; full test suite. Validatable headless with a scripted "fire straight" — no controller/wire needed. | No | **M** (core) |
| **2** | `fireTrigger` on `Command`; HP/ammo on `Telemetry`; populate/ingest; C++/Python pick-up; `make demo-dogfight` + compose + scripted pursuit-and-fire controller; web HUD (HP/ammo). | Yes (additive) | **M** |
| **3** | Tracers + hit/kill events (new topic + `BagFrame` + `astrum-web` render). | Yes | **M–L** |
| **4** | RL env (embedded pyo3 recommended), obs/action/reward, multiprocess vec-env, curriculum. | binding | **L** (future) |

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
3. **`health @7` semantics** — **add a dedicated `hp @9 :UInt16`**; leave `health @7` as
   fault/damage-state flags ("0 = nominal" stands). `ammo` takes `@10`.
4. **Projectile representation** — **bullet entities** (§2.3). Required for tracers (§9) and
   honest drop/lead/Coriolis; live population is trivial.
5. **Dead-aircraft behavior** — **freeze + insert a `Dead` marker** (§4). Stable terminal state
   for RL episodes; the wreck stays observable; no despawn/NaN races.
6. **First-pass depth** — **Phase 1 only** (sim core + full headless tests, scripted fire).
   Defer Phase 2's wire/demo: the recommended embedded RL env (§8b) drives `app.update()`
   directly and never touches the wire, so Phase 2 is off the critical path to the RL goal.

---

## Appendix: key code references

| Concern | Location |
|---|---|
| App build + system chain + `dt` | `astrum-sim/src/lib.rs:32,58-82` |
| Spawn bundle / fixed-wing gating | `astrum-sim/src/systems.rs:19-81` (bundle 58-65, gate 69-71) |
| FDM step / FDM→ECS | `astrum-sim/src/systems.rs:84-88` (step), `91-117` (sync) |
| State components (`GeoState`, `SimParams`) | `astrum-sim/src/components.rs:54-89` |
| Command ingest / apply | `astrum-sim/src/net.rs:35-67` |
| Telemetry build/publish (`health` = 0) | `astrum-sim/src/net.rs:286-319` (health at 308) |
| Termination | `astrum-sim/src/systems.rs:203-220` |
| Pacing / realtime | `astrum-sim/src/systems.rs:166-200`, `lib.rs:54-59` |
| Wire schema | `astrum-proto/schema/astrum.capnp` (Telemetry 39-50, Command 52-74, BagFrame 76-81) |
| Proto mirrors / encode / decode | `astrum-proto/src/lib.rs` (`TelemetrySample` 33-53, `Control` 56-62) |
| Zenoh keys / transport | `astrum-net/src/lib.rs` (prefixes 41-43, key builders 52-61) |
| Geodetic↔ECEF↔ENU | `astrum-geo/src/lib.rs` (`LocalFrame` 96-150, `geodetic_to_enu` 142) |
| Attitude → forward vector | `web/js/meshes.js:34-55` (`attitudeToQuaternion`, nose at :40) |
| Scenario structs / load | `astrum-sim/src/scenario.rs` (`AircraftSpec` 50, `load`/`validate` 78-97) |
| Test harness | `astrum-sim/tests/common/mod.rs` (`fdm_guard` 32, `two_ship` 75, `snapshot` 135) |
| Controller obs/act pattern | `controllers/optimal/python/astrum_control.py` (decode 217-258, encode 260-276) |
