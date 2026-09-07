"""Honest gun-lay geometry — the aim point a round actually flies to.

`geometry.py` measures `lead_error` from the **nose** against a drag-free aim point
built with `tof = rng / MUZZLE_MPS`. The sim does something different
(`crates/astrum-sim/src/combat/systems.rs:75-76`):

```rust
let dir = disperse(boresight, gun.dispersion_rad, &mut rng);
let vel = ac_vel + dir * gun.muzzle_speed_mps;   // VELOCITY INHERITANCE
```

so the round departs along `own_vel + 1030·boresight`, not along the boresight, and
then decelerates hard under 20 mm drag. Two consequences the reward never modelled:

1. **Direction.** At any loaded pull the nose must sit *further off the velocity
   vector* than the target does, by `asin(V·sin α / 1030)` — 0.97° at α=5°, 2.31° at
   α=12°, 3.81° at α=20° (V=200 m/s). The 8 m kill sphere subtends only 1.83° at
   250 m, so this is a whole-sphere error.
2. **Time of flight.** `rng / 1030` is wrong in *both* directions: inheritance makes
   the round's initial groundspeed ~1180-1290 m/s on a forward shot, while drag then
   bleeds ~500 m/s² off it. Only a propagated tof is right.

This module supplies the ground truth for both: a closed-form drag-free intercept
(exact, single positive root, no iteration) and a numeric propagator that mirrors
`crates/astrum-sim/src/combat/ballistics.rs` tick for tick so any candidate lay can
be *scored* rather than argued about.

Nothing here is imported by the environment, the reward or the observation. It is a
reference implementation plus a measurement harness; see `test_gunlay.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, sin, sqrt

import numpy as np

from . import constants as C

# ---------------------------------------------------------------------------
# Sim constants, mirrored from Rust. Keep these in sync by hand — the unit test
# pins the derived quantities, and every value below names its source.
# ---------------------------------------------------------------------------

# combat/ballistics.rs
G0 = 9.806_65  # m/s²
OMEGA_EARTH = 7.292_115e-5  # rad/s
RHO0 = 1.225  # kg/m³
T0 = 288.15  # K
LAPSE = 0.0065  # K/m
R_AIR = 287.052_87  # J/(kg·K)
TROPOPAUSE = 11_000.0  # m

# combat/resources.rs — CombatParams::default()
ROUND_CD = 0.3
ROUND_AREA_M2 = np.pi * 0.010 * 0.010  # 20 mm ⇒ r = 10 mm
ROUND_MASS_KG = 0.10
DRAG_FACTOR = ROUND_CD * ROUND_AREA_M2 / ROUND_MASS_KG  # m²/kg
DAMAGE_PER_HIT = 2.0
HIT_RADIUS_M = 8.0
BULLET_TTL_S = 3.0

# combat/components.rs — Gun::m61a1()
MUZZLE_SPEED_MPS = 1030.0
MUZZLE_OFFSET_BODY = np.array([4.0, 0.0, 0.0])  # nose +X, starboard +Y, down +Z
GUN_RATE_HZ = 100.0
DISPERSION_RAD = 0.008  # 1σ half-angle of the muzzle cone

SIM_DT = 1.0 / 120.0  # scenario.rs rate_hz default
HP_FULL = 100.0
HITS_TO_KILL = int(np.ceil(HP_FULL / DAMAGE_PER_HIT))  # 50


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


# ---------------------------------------------------------------------------
# Projectile physics — a tick-for-tick mirror of combat/ballistics.rs
# ---------------------------------------------------------------------------


def air_density(alt_m: float) -> float:
    """ISA density (kg/m³). Mirrors `ballistics.rs::air_density`."""
    if alt_m <= TROPOPAUSE:
        t = T0 - LAPSE * alt_m
        if t <= 0.0:
            return 0.0
        return RHO0 * (t / T0) ** (G0 / (R_AIR * LAPSE) - 1.0)
    t_trop = T0 - LAPSE * TROPOPAUSE
    rho_trop = RHO0 * (t_trop / T0) ** (G0 / (R_AIR * LAPSE) - 1.0)
    return rho_trop * np.exp(-G0 * (alt_m - TROPOPAUSE) / (R_AIR * t_trop))


def earth_rate_enu(lat_rad: float) -> np.ndarray:
    """Earth angular rate in local ENU (x=East, y=North, z=Up)."""
    return OMEGA_EARTH * np.array([0.0, cos(lat_rad), sin(lat_rad)])


def projectile_accel(vel: np.ndarray, alt_m: float, omega: np.ndarray) -> np.ndarray:
    """Gravity + quadratic drag + Coriolis. Mirrors `ballistics.rs::acceleration`."""
    speed = float(np.linalg.norm(vel))
    drag = -0.5 * air_density(alt_m) * DRAG_FACTOR * speed * vel
    coriolis = -2.0 * np.cross(omega, vel)
    return np.array([0.0, 0.0, -G0]) + drag + coriolis


def step_projectile(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    dt: float = SIM_DT,
    alt_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One semi-implicit Euler tick: velocity first, then position with the NEW
    velocity — exactly as `ballistics.rs::step_projectile` does it.

    Density is looked up at `pos.z + alt_offset`, mirroring `step_bullets`' own
    `base_alt + pos.z`. Two consistent ways to call this: work in MSL z and leave
    `alt_offset` at 0 (the default), or work in the sim's ENU frame — which is what
    `resolve_hits` compares positions in — and pass the ENU origin's altitude."""
    new_vel = vel + projectile_accel(vel, float(pos[2]) + alt_offset, omega) * dt
    return pos + new_vel * dt, new_vel


# ---------------------------------------------------------------------------
# Closed-form drag-free intercept
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intercept:
    """A gun solution: which way the *boresight* must point, and when it arrives."""

    boresight: np.ndarray  # ENU unit vector the nose must lie along
    tof_s: float  # time of flight to the intercept
    ground_speed_mps: float  # |own_vel + 1030·boresight| — the round's real speed
    feasible: bool  # False ⇒ no forward-time solution (target outruns the round)


def intercept(
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    muzzle_speed: float = MUZZLE_SPEED_MPS,
) -> Intercept:
    """Exact drag-free intercept with velocity inheritance.

    A round launched now has velocity ``w = shooter_vel + muzzle_speed·d``, so
    intercept at time ``t`` requires

        shooter_pos + w·t == target_pos + target_vel·t

    ⇒ ``muzzle_speed·d == r/t + u`` with ``r = target_pos - shooter_pos`` and
    ``u = target_vel - shooter_vel``. Imposing ``|d| = 1`` and substituting
    ``x = 1/t`` gives a scalar quadratic

        |r|²·x² + 2(r·u)·x + (|u|² - muzzle_speed²) == 0

    whose constant term is negative whenever the *relative* closing speed is below
    the muzzle speed — i.e. always in this sim, since both aircraft are subsonic and
    ``|u| ≤ ~2·340 ≪ 1030``. So there is exactly one positive root and no branch:

        x = (-(r·u) + sqrt((r·u)² - |r|²(|u|² - muzzle_speed²))) / |r|²

    Note what this *does not* need: a time-of-flight guess. The naive
    ``tof = rng/1030`` in `geometry.py:184` exists only because it aims at a
    predicted point instead of solving for the launch direction directly.
    """
    r = np.asarray(target_pos, float) - np.asarray(shooter_pos, float)
    u = np.asarray(target_vel, float) - np.asarray(shooter_vel, float)
    rr = float(np.dot(r, r))
    if rr < 1e-12:
        return Intercept(np.zeros(3), 0.0, muzzle_speed, False)
    ru = float(np.dot(r, u))
    disc = ru * ru - rr * (float(np.dot(u, u)) - muzzle_speed * muzzle_speed)
    if disc < 0.0:
        return Intercept(np.zeros(3), 0.0, 0.0, False)
    x = (-ru + sqrt(disc)) / rr
    if x <= 0.0:
        return Intercept(np.zeros(3), 0.0, 0.0, False)
    tof = 1.0 / x
    d = (r * x + u) / muzzle_speed
    w = np.asarray(shooter_vel, float) + d * muzzle_speed
    return Intercept(_unit(d), tof, float(np.linalg.norm(w)), True)


def theta_bias(own_speed_mps: float, off_velocity_rad: float, muzzle_speed: float = MUZZLE_SPEED_MPS) -> float:
    """Closed-form boresight correction for a *static* aim point (rad).

    With the target at rest the required lay is ``1030·d = s·û - v_own``, so the
    component of ``-v_own`` perpendicular to the line of sight is ``V·sin α`` and

        θ_bias = asin(V·sin α / muzzle_speed)

    where ``α`` is the angle between the shooter's **velocity vector** and the aim
    point (not aerodynamic AoA, though in a steady pull they coincide in the pull
    plane). The correction rotates the nose *away* from the velocity vector, so the
    nose must sit at ``α + θ_bias`` off velocity. It is **range-independent**.
    """
    s = own_speed_mps * sin(off_velocity_rad) / muzzle_speed
    return asin(float(np.clip(s, -1.0, 1.0)))


def sample_path(target_path: np.ndarray, t: float) -> np.ndarray:
    """Interpolate an `(M,4)` `[t, E, N, z]` path table at time `t` (clamped at both ends)."""
    p = np.asarray(target_path, float)
    return np.array([float(np.interp(t, p[:, 0], p[:, k])) for k in (1, 2, 3)])


def intercept_path(
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    target_path: np.ndarray,
    muzzle_speed: float = MUZZLE_SPEED_MPS,
    iters: int = 8,
) -> Intercept:
    """Drag-free intercept of a target's *actual* recorded path — the clairvoyant lay.

    `intercept()` assumes the bandit holds its instantaneous velocity, which is exactly
    what a real fire-control computer must assume and exactly what an 8 g break breaks.
    Given the path the bandit really flew, the intercept condition becomes a scalar fixed
    point in the time of flight:

        t ← |path(t) − shooter_pos − shooter_vel·t| / muzzle_speed

    (the drag-free round covers `muzzle_speed·t` relative to the shooter's own inertial
    drift). It contracts quickly because `|d(path)/dt| ≪ muzzle_speed`. Comparing this
    lay against `intercept()` separates an aim-MATH error from a PREDICTION error.
    """
    sp = np.asarray(shooter_pos, float)
    sv = np.asarray(shooter_vel, float)
    t = float(np.linalg.norm(sample_path(target_path, 0.0) - sp)) / muzzle_speed
    for _ in range(iters):
        rel = sample_path(target_path, t) - sp - sv * t
        t_new = float(np.linalg.norm(rel)) / muzzle_speed
        if abs(t_new - t) < 1e-9:
            t = t_new
            break
        t = t_new
    if t <= 0.0:
        return Intercept(np.zeros(3), 0.0, 0.0, False)
    d = _unit(sample_path(target_path, t) - sp - sv * t)
    if not d.any():
        return Intercept(np.zeros(3), 0.0, 0.0, False)
    return Intercept(d, t, float(np.linalg.norm(sv + d * muzzle_speed)), True)


def _closest_approach_table(
    launch_pos: np.ndarray,
    launch_vel: np.ndarray,
    tgt_table: np.ndarray,
    omega: np.ndarray,
    dt: float = SIM_DT,
    alt_offset: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """`_closest_approach` against a pre-tabulated target position per hit test.

    The contract is positional, matching `resolve_hits`: row `k` is the target position
    the sim compares against on the round's `k`-th swept segment. Because
    `sync_geo_transform → fire_guns → step_bullets → resolve_hits` all run inside ONE
    tick, the sim's first test pits the segment `muzzle → muzzle+v·dt` against the target
    at the SHOT tick — so a caller reconstructing a real shot must put the shot-tick
    position in row 0, and row `k` is then `k·dt` later. (A synthetic caller that starts
    the table at `(k+1)·dt` is simply scoring against a one-tick-late target: ~1.3 m at
    160 m/s, which is inside the 8 m sphere but not inside the ~1 m sampling floor.)

    Returns `(miss_vector, miss_distance, tof)`; the miss vector points target → round,
    which is what `refine_lay` needs to know which way to move the nose."""
    pos = np.asarray(launch_pos, float).copy()
    vel = np.asarray(launch_vel, float).copy()
    best_d2 = float("inf")
    best_vec = pos - tgt_table[0]
    best_t = 0.0
    for k in range(len(tgt_table)):
        prev = pos
        pos, vel = step_projectile(pos, vel, omega, dt, alt_offset)
        tgt = tgt_table[k]
        seg = pos - prev
        seg2 = float(np.dot(seg, seg))
        s = 0.0 if seg2 < 1e-12 else float(np.clip(np.dot(tgt - prev, seg) / seg2, 0.0, 1.0))
        vec = (prev + seg * s) - tgt
        d2 = float(np.dot(vec, vec))
        if d2 < best_d2:
            best_d2, best_vec, best_t = d2, vec, (k + 1) * dt
    return best_vec, sqrt(best_d2), best_t


def refine_lay(
    boresight: np.ndarray,
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    tgt_table: np.ndarray,
    omega: np.ndarray,
    dt: float = SIM_DT,
    iters: int = 4,
    muzzle_speed: float = MUZZLE_SPEED_MPS,
    alt_offset: float = 0.0,
) -> np.ndarray:
    """Newton-correct a drag-free lay for drag, gravity drop and Coriolis.

    Propagate the round, take the component of the miss perpendicular to the boresight,
    and rotate the nose to cancel it: displacing the boresight by `δ` moves the impact
    point by about `muzzle_speed·tof·δ` (only the muzzle term rotates — the inherited
    aircraft velocity does not).

    That gain is an OVER-estimate of the real sensitivity, because drag bleeds most of the
    perturbation away over the time of flight, so each pass removes only ~75-80% of the
    residual rather than all of it — geometric, not quadratic. Measured miss by iteration
    (m): 250 m 0.22→0.013→0.001; 400 m/200 m-crosser 5.30→1.19→1.13 (floor); 600 m
    6.11→0.66→0.07→0.008; 900 m 28.1→6.3→2.4→1.9→0.31→0.05. Four iterations put every
    in-envelope geometry well inside the 8 m hit sphere and agree with
    `numeric_intercept`'s Gauss-Newton solve to <0.03°, for 4 propagations instead of ~18.
    Pass `iters=6` past ~800 m."""
    d = _unit(boresight)
    for _ in range(iters):
        vec, dist, tof = _closest_approach_table(
            shooter_pos,
            np.asarray(shooter_vel, float) + d * muzzle_speed,
            tgt_table,
            omega,
            dt,
            alt_offset,
        )
        if dist < 1e-3 or tof <= 0.0:
            break
        tangential = vec - float(np.dot(vec, d)) * d
        d = _unit(d - tangential / (muzzle_speed * tof))
    return d


def clamp_to_alpha_cone(
    boresight: np.ndarray, shooter_vel: np.ndarray, alpha_max_rad: float
) -> np.ndarray:
    """The closest lay to `boresight` reachable within `alpha_max_rad` of the velocity
    vector — the airframe constraint that decides whether an honest solution is even
    flyable (the nose cannot be pointed arbitrarily far off the flight path)."""
    v = _unit(shooter_vel)
    d = _unit(boresight)
    if not v.any() or not d.any():
        return d
    cos_a = float(np.clip(np.dot(v, d), -1.0, 1.0))
    if cos_a >= cos(alpha_max_rad):
        return d
    perp = _unit(d - cos_a * v)
    if not perp.any():
        return d
    return _unit(cos(alpha_max_rad) * v + sin(alpha_max_rad) * perp)


# ---------------------------------------------------------------------------
# Numeric (drag-inclusive) intercept — the reference truth
# ---------------------------------------------------------------------------


def _closest_approach(
    launch_pos: np.ndarray,
    launch_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    omega: np.ndarray,
    dt: float = SIM_DT,
    ttl_s: float = BULLET_TTL_S,
) -> tuple[np.ndarray, float, float]:
    """Propagate one round and return `(miss_vector, miss_distance, tof)` at closest
    approach to a constant-velocity target.

    `miss_vector` points target → round. Uses the same swept-segment logic as
    `systems.rs::resolve_hits` (segment start reconstructed from the post-step
    velocity), so a 1200 m/s round cannot tunnel through the 8 m sphere.
    """
    pos = np.asarray(launch_pos, float).copy()
    vel = np.asarray(launch_vel, float).copy()
    best_d2 = float("inf")
    best_vec = pos - np.asarray(target_pos, float)
    best_t = 0.0
    t = 0.0
    n = int(ttl_s / dt)
    for _ in range(n):
        prev = pos
        pos, vel = step_projectile(pos, vel, omega, dt)
        t += dt
        tgt = np.asarray(target_pos, float) + np.asarray(target_vel, float) * t
        # Swept segment `prev -> pos` vs the target point at the end of the tick.
        seg = pos - prev
        seg2 = float(np.dot(seg, seg))
        s = 0.0 if seg2 < 1e-12 else float(np.clip(np.dot(tgt - prev, seg) / seg2, 0.0, 1.0))
        vec = (prev + seg * s) - tgt
        d2 = float(np.dot(vec, vec))
        if d2 < best_d2:
            best_d2, best_vec, best_t = d2, vec, t
        elif d2 > best_d2 and t > 0.02:
            break  # receding — closest approach is behind us
    return best_vec, sqrt(best_d2), best_t


def numeric_intercept(
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    lat_rad: float = 0.0,
    muzzle_speed: float = MUZZLE_SPEED_MPS,
    dt: float = SIM_DT,
    iters: int = 6,
) -> Intercept:
    """Drag/gravity/Coriolis-inclusive intercept: the boresight that *actually* puts
    a round on the target, found by Gauss-Newton on the two tangential angles with
    the closed-form drag-free solution as the seed.

    This is the reference every candidate lay is scored against. It is far too
    expensive for the reward (≈360 propagator ticks per residual evaluation); its job
    is to say how much of the residual miss the cheap closed form leaves on the
    table.
    """
    omega = earth_rate_enu(lat_rad)
    seed = intercept(shooter_pos, shooter_vel, target_pos, target_vel, muzzle_speed)
    if not seed.feasible:
        return seed
    d = seed.boresight.copy()

    def miss_of(direction: np.ndarray) -> tuple[np.ndarray, float, float]:
        launch_vel = np.asarray(shooter_vel, float) + _unit(direction) * muzzle_speed
        return _closest_approach(shooter_pos, launch_vel, target_pos, target_vel, omega, dt)

    eps = 1e-4  # rad
    for _ in range(iters):
        # Tangent basis at d.
        helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        e1 = _unit(np.cross(d, helper))
        e2 = np.cross(d, e1)
        vec0, dist0, _ = miss_of(d)
        if dist0 < 1e-3:
            break
        # Residual = miss projected onto the tangent plane (the part aiming can fix).
        r0 = np.array([float(np.dot(vec0, e1)), float(np.dot(vec0, e2))])
        jac = np.zeros((2, 2))
        for k, e in enumerate((e1, e2)):
            vec1, _, _ = miss_of(_unit(d + eps * e))
            r1 = np.array([float(np.dot(vec1, e1)), float(np.dot(vec1, e2))])
            jac[:, k] = (r1 - r0) / eps
        try:
            step = np.linalg.solve(jac, -r0)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -0.05, 0.05)
        d = _unit(d + step[0] * e1 + step[1] * e2)

    _, _, tof = miss_of(d)
    w = np.asarray(shooter_vel, float) + d * muzzle_speed
    return Intercept(d, tof, float(np.linalg.norm(w)), True)


# ---------------------------------------------------------------------------
# Dispersion + hit probability
# ---------------------------------------------------------------------------


def disperse(direction: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Mirror of `systems.rs::disperse` — a Gaussian tangent-plane kick then
    renormalise. Note the kick is applied to the **boresight**, *before* the
    shooter's velocity is added, so the angular spread of the resulting round
    velocity is smaller than `sigma` by roughly `muzzle_speed / |round_vel|`."""
    d = _unit(direction)
    if sigma <= 0.0 or not d.any():
        return d
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(d, helper))
    e2 = np.cross(d, e1)
    a1, a2 = sigma * rng.standard_normal(2)
    return _unit(d + a1 * e1 + a2 * e2)


def air_density_vec(alt_m: np.ndarray) -> np.ndarray:
    """`air_density` over an array of altitudes (same branch structure, vectorised)."""
    alt = np.asarray(alt_m, float)
    t = np.maximum(T0 - LAPSE * alt, 1e-9)
    exponent = G0 / (R_AIR * LAPSE) - 1.0
    trop = RHO0 * (t / T0) ** exponent
    t_trop = T0 - LAPSE * TROPOPAUSE
    rho_trop = RHO0 * (t_trop / T0) ** exponent
    strat = rho_trop * np.exp(-G0 * (alt - TROPOPAUSE) / (R_AIR * t_trop))
    return np.where(alt <= TROPOPAUSE, np.where(T0 - LAPSE * alt <= 0.0, 0.0, trop), strat)


def step_projectile_vec(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    dt: float = SIM_DT,
    alt_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """`step_projectile` over `(N,3)` batches — same semi-implicit order, same physics."""
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    rho = air_density_vec(pos[:, 2] + alt_offset)[:, None]
    accel = np.array([0.0, 0.0, -G0]) - 0.5 * rho * DRAG_FACTOR * speed * vel - 2.0 * np.cross(omega, vel)
    new_vel = vel + accel * dt
    return pos + new_vel * dt, new_vel


def p_hit_vec(
    boresight: np.ndarray,
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    samples: int = 2000,
    lat_rad: float = 0.0,
    sigma: float = DISPERSION_RAD,
    seed: int = 0,
    dt: float = SIM_DT,
    horizon_s: float = 0.0,
    target_path: np.ndarray | None = None,
    tgt_table: np.ndarray | None = None,
    alt_offset: float = 0.0,
) -> tuple[float, float]:
    """Batched `p_hit`: identical model, all `samples` rounds propagated together.

    The scalar version costs ~1 s per call (360 python-level propagator ticks per
    sample), which is unaffordable when scoring every round a rollout fires under
    three candidate lays. This runs one vectorised tick loop instead.

    `horizon_s` bounds the propagation; 0 derives `2·tof + 0.05 s` from the closed-form
    solution (capped at the round's TTL), which comfortably brackets closest approach —
    a round still receding at 2× its own intercept time is not going to hit.

    `target_path`, when given, is an `(M,4)` table of `[t, E, N, z]` samples of where the
    target ACTUALLY went (t = 0 at the shot), linearly interpolated per tick and held at
    the last sample past its end. It replaces the constant-velocity assumption, which
    matters against an 8 g break: over a 0.35 s time of flight a constant-velocity
    extrapolation of a cornering bandit is itself tens of metres wrong, and without this
    a scored lay cannot be told apart from a scored *prediction*.

    `tgt_table` short-circuits both: an `(N,3)` table of target positions indexed by hit
    test, exactly as `_closest_approach_table` consumes it (row 0 = the shot tick). Prefer
    it whenever the caller already has the sim's own per-tick track — it needs no
    interpolation and no `horizon_s`, since its length IS the horizon."""
    rng = np.random.default_rng(seed)
    omega = earth_rate_enu(lat_rad)
    sp = np.asarray(shooter_pos, float)
    sv = np.asarray(shooter_vel, float)
    tp = np.asarray(target_pos, float)
    tv = np.asarray(target_vel, float)

    if tgt_table is not None:
        table = np.asarray(tgt_table, float)
    else:
        if horizon_s <= 0.0:
            sol = intercept(sp, sv, tp, tv)
            tof = sol.tof_s if sol.feasible else float(np.linalg.norm(tp - sp)) / MUZZLE_SPEED_MPS
            horizon_s = min(BULLET_TTL_S, 2.0 * tof + 0.05)
        ticks = (1 + np.arange(int(horizon_s / dt))) * dt
        if target_path is None:
            table = tp[None, :] + tv[None, :] * ticks[:, None]
        else:
            path = np.asarray(target_path, float)
            table = np.column_stack([np.interp(ticks, path[:, 0], path[:, k]) for k in (1, 2, 3)])

    d0 = _unit(boresight)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d0[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(d0, helper))
    e2 = np.cross(d0, e1)
    kick = sigma * rng.standard_normal((samples, 2))
    dirs = d0 + kick[:, 0:1] * e1 + kick[:, 1:2] * e2
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    vel = sv + dirs * MUZZLE_SPEED_MPS
    pos = np.repeat(sp[None, :], samples, axis=0)
    best = np.full(samples, np.inf)
    for k in range(len(table)):
        prev = pos
        pos, vel = step_projectile_vec(pos, vel, omega, dt, alt_offset)
        tgt = table[k]
        seg = pos - prev
        seg2 = np.einsum("ij,ij->i", seg, seg)[:, None]
        s = np.clip(np.einsum("ij,ij->i", tgt - prev, seg)[:, None] / np.maximum(seg2, 1e-12), 0.0, 1.0)
        vec = prev + seg * s - tgt
        np.minimum(best, np.einsum("ij,ij->i", vec, vec), out=best)
    miss = np.sqrt(best)
    return float(np.mean(miss <= HIT_RADIUS_M)), float(np.mean(miss))


def p_hit(
    boresight: np.ndarray,
    shooter_pos: np.ndarray,
    shooter_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    samples: int = 2000,
    lat_rad: float = 0.0,
    sigma: float = DISPERSION_RAD,
    seed: int = 0,
    dt: float = SIM_DT,
) -> tuple[float, float]:
    """Monte-Carlo `(P(hit), mean miss distance)` for firing one round along
    `boresight`. The target is propagated at constant velocity — over a ≤0.5 s time
    of flight that is a much smaller error than the lay bias being measured."""
    rng = np.random.default_rng(seed)
    omega = earth_rate_enu(lat_rad)
    d0 = _unit(boresight)
    hits = 0
    miss_sum = 0.0
    for _ in range(samples):
        d = disperse(d0, sigma, rng)
        launch_vel = np.asarray(shooter_vel, float) + d * MUZZLE_SPEED_MPS
        _, dist, _ = _closest_approach(shooter_pos, launch_vel, target_pos, target_vel, omega, dt)
        miss_sum += dist
        if dist <= HIT_RADIUS_M:
            hits += 1
    return hits / samples, miss_sum / samples


# ---------------------------------------------------------------------------
# The lays under test, expressed against `geometry.AState`
# ---------------------------------------------------------------------------


def naive_lead_point_from_parts(
    shooter_pos: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray
) -> np.ndarray:
    """What `geometry.lead_point` computes today: constant-velocity extrapolation over
    `tof = rng / 1030`, ignoring velocity inheritance and drag. Note the shooter's own
    velocity never enters — that omission *is* the bug."""
    rng = float(np.linalg.norm(np.asarray(target_pos, float) - np.asarray(shooter_pos, float)))
    tof = rng / C.MUZZLE_MPS if rng > 1e-6 else 0.0
    return np.asarray(target_pos, float) + np.asarray(target_vel, float) * tof


def naive_lead_point(me, bandit) -> np.ndarray:
    """`naive_lead_point_from_parts` against a `geometry.AState` pair."""
    return naive_lead_point_from_parts(me.pos, bandit.pos, bandit.vel)


def corrected_boresight(me, bandit) -> Intercept:
    """The closed-form lay for the `me → bandit` engagement (ENU unit vector)."""
    return intercept(me.pos, me.vel, bandit.pos, bandit.vel)


def corrected_lead_error(me, bandit) -> float:
    """Angle (rad) between `me`'s nose and the *correct* boresight — the drop-in
    replacement for `RelGeom.lead_error`. Falls back to the naive nose-to-aim-point
    angle if no forward-time solution exists (never observed in this sim)."""
    from .geometry import body_axes_enu  # local import: avoid an import cycle

    sol = corrected_boresight(me, bandit)
    nose, _, _ = body_axes_enu(me.phi, me.theta, me.psi)
    if not sol.feasible:
        aim = _unit(naive_lead_point(me, bandit) - me.pos)
        return atan2(float(np.linalg.norm(np.cross(aim, nose))), float(np.dot(aim, nose)))
    d = sol.boresight
    return atan2(float(np.linalg.norm(np.cross(d, nose))), float(np.clip(np.dot(d, nose), -1.0, 1.0)))
