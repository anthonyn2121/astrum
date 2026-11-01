"""Egocentric dogfight geometry and the observation vector (design doc §8.2).

Frames:

- **World grid ENU** (`pos_*`, `vel_e/vel_n/vel_u` columns): East=+X, North=+Y,
  Up=+Z at the scenario origin. Every relative-geometry computation (LOS, range,
  closure, aspect) happens here.
- **Body axes** (x nose, y starboard, z down): an `AState` carries its three body
  axes expressed in the grid, obtained from the sim's own `Env.body_to_grid`
  (which includes the local-level tilt away from the grid origin). Nothing here
  re-derives an attitude matrix.
- **Local level NED** (`vel_ned_*`, `psi`): only where local-level is meant —
  ground course for the autopilot heading targets.

Gun fire control is the sim's (`Env.gun_solution`: drag, gravity, velocity
inheritance, muzzle offset); this module only consumes a `GunSolution`.

`AState` is a zero-copy view of one row of the env's observation matrix plus
its body axes, so per-tick snapshotting builds no per-field Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, sin

import astrum_env
import numpy as np

from . import constants as C

COL = {name: i for i, name in enumerate(astrum_env.OBS_FIELDS)}
_POS = slice(COL["pos_e"], COL["pos_u"] + 1)
_VEL = slice(COL["vel_e"], COL["vel_u"] + 1)
_BODY_UNIT = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])


class AState:
    """One aircraft: a view of its observation row + body axes in the grid.

    `axes` is a (3, 3) array whose rows are the nose, starboard and down unit
    vectors in grid ENU.
    """

    __slots__ = ("row", "axes")

    def __init__(self, row: np.ndarray, axes: np.ndarray):
        self.row = row
        self.axes = axes

    @staticmethod
    def from_env(env, row: np.ndarray) -> AState:
        """Snapshot the aircraft of `row` (an `env.step`/`env.reset` row)."""
        i = int(row[COL["id"]])
        return AState(row, np.array([env.body_to_grid(i, e) for e in _BODY_UNIT]))

    @staticmethod
    def fabricate(axes, **cols: float) -> AState:
        """A synthetic state for tests: unspecified columns are 0, except
        `alive=1`, `hit_points=hit_points_max=100` and `ammo=GUN_AMMO`."""
        row = np.zeros(len(astrum_env.OBS_FIELDS))
        defaults = {
            "alive": 1.0,
            "hit_points": 100.0,
            "hit_points_max": 100.0,
            "ammo": float(astrum_env.GUN_AMMO),
        }
        for name, value in {**defaults, **cols}.items():
            row[COL[name]] = value
        return AState(row, np.asarray(axes, dtype=np.float64))

    def __getattr__(self, name: str) -> float:
        try:
            return float(self.row[COL[name]])
        except KeyError:
            raise AttributeError(name) from None

    @property
    def id(self) -> int:
        return int(self.row[COL["id"]])

    @property
    def alive(self) -> bool:
        return self.row[COL["alive"]] >= 0.5

    @property
    def ammo(self) -> int:
        return int(self.row[COL["ammo"]])

    @property
    def pos(self) -> np.ndarray:
        """Grid ENU position (m)."""
        return self.row[_POS]

    @property
    def vel(self) -> np.ndarray:
        """Grid ENU velocity (m/s)."""
        return self.row[_VEL]

    @property
    def nose(self) -> np.ndarray:
        return self.axes[0]

    def course_local(self) -> float:
        """Local-level ground course (rad, 0 = N, +E). Gimbal-free through the
        vertical, unlike Euler ψ (a Split-S flips ψ); falls back to ψ when slow."""
        n, e = self.row[COL["vel_ned_n"]], self.row[COL["vel_ned_e"]]
        return atan2(e, n) if hypot(n, e) > 1.0 else self.psi


@dataclass(frozen=True)
class GunSolution:
    """The sim's ballistic fire control for one shooter→target pair."""

    miss_m: float
    tof_s: float
    hit: bool
    aim_dir: np.ndarray  # grid unit vector the boresight must point along
    aim_error_rad: float
    in_range: bool

    @staticmethod
    def query(env, shooter: int, target: int) -> GunSolution | None:
        """`env.gun_solution` as a `GunSolution`; None if the shooter is unarmed."""
        d = env.gun_solution(shooter, target)
        if d is None:
            return None
        return GunSolution(
            miss_m=d["miss_m"],
            tof_s=d["tof_s"],
            hit=d["hit"],
            aim_dir=np.asarray(d["aim_dir"], dtype=np.float64),
            aim_error_rad=d["aim_error_rad"],
            in_range=d["in_range"],
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@dataclass
class RelGeom:
    """Relative geometry of `bandit` seen from `me` (design doc §8.2)."""

    rng: float  # range (m)
    ata: float  # antenna train angle: nose-to-LOS (rad); LOW = nose-on
    ata_az: float  # LOS bearing off my nose in body XY (rad, right +)
    ata_el: float  # LOS elevation in body frame (rad, up +)
    aspect: float  # angle at bandit, tail→LOS-to-me (rad); LOW = his 6 o'clock
    hca: float  # heading-crossing angle of the two flight paths (rad)
    closure: float  # +closing / −opening (m/s)
    rel_alt: float  # bandit_hae − my_hae (m)
    range_factor: float  # ∈[0,1]: 1 at point-blank → 0 at R_MAX (shaping only)
    aim_error: float  # sim fire control: boresight vs required aim (rad); π if unarmed
    in_range: bool  # sim fire control: aim solution reaches the target in time
    gun_quality: float  # ∈[0,1]: in-range × tracking Gaussian on aim_error


def relative(me: AState, bandit: AState, gun: GunSolution | None) -> RelGeom:
    """Geometry of `bandit` from `me`; `gun` = the sim's solution me→bandit."""
    los = bandit.pos - me.pos
    rng = float(np.linalg.norm(los))
    los_hat = _unit(los)

    lx, ly, lz = (float(c) for c in me.axes @ los_hat)  # LOS in my body axes
    ata = atan2(hypot(ly, lz), lx)
    ata_az = atan2(ly, lx)
    ata_el = atan2(-lz, hypot(lx, ly))

    # cos(aspect) = nose_b · LOS_hat (=+1 when I sit on his 6).
    nose_b = bandit.nose
    aspect = atan2(float(np.linalg.norm(np.cross(nose_b, los_hat))), float(np.dot(nose_b, los_hat)))

    hca = _wrap_pi(_grid_course(bandit) - _grid_course(me))
    closure = -float(np.dot(bandit.vel - me.vel, los_hat))

    if gun is None:
        aim_error, in_range, quality = np.pi, False, 0.0
    else:
        aim_error, in_range = gun.aim_error_rad, gun.in_range
        quality = exp(-0.5 * (aim_error / C.LEAD_SIGMA) ** 2) if in_range else 0.0

    return RelGeom(
        rng=rng,
        ata=ata,
        ata_az=ata_az,
        ata_el=ata_el,
        aspect=aspect,
        hca=hca,
        closure=closure,
        rel_alt=bandit.alt_hae_m - me.alt_hae_m,
        range_factor=float(np.clip(1.0 - rng / C.R_MAX, 0.0, 1.0)),
        aim_error=aim_error,
        in_range=in_range,
        gun_quality=quality,
    )


def _grid_course(s: AState) -> float:
    if hypot(s.vel[0], s.vel[1]) > 1.0:
        return atan2(s.vel[0], s.vel[1])  # atan2(E, N)
    return s.psi


def body_bearing(me: AState, direction: np.ndarray):
    """A grid `direction` expressed in `me`'s body frame → `(off, bank_err, forward)`.

    - `off` — angle between the nose and the direction (rad; 0 = on boresight).
    - `bank_err` — roll (rad, + = right) that puts the direction in the plane
      above the nose, i.e. puts the lift vector on it.
    - `forward` — nose-alignment cosine.
    """
    lx, ly, lz = (float(c) for c in me.axes @ _unit(np.asarray(direction, dtype=np.float64)))
    return atan2(hypot(ly, lz), lx), atan2(ly, -lz), lx


def aim_direction(me: AState, target: AState, gun: GunSolution | None) -> np.ndarray:
    """Where to point the nose: the sim's lead (`gun.aim_dir`) when armed, else
    the line of sight."""
    return gun.aim_dir if gun is not None else _unit(target.pos - me.pos)


def potential(me: AState, bandit: AState, rg: RelGeom) -> float:
    """Shaping potential Φ ∈ [0, 1] = nose-on ∧ rear-hemisphere ∧ close (§8.3).

    0 when either aircraft is dead (no live advantage to bank).
    """
    if not me.alive or not bandit.alive:
        return 0.0
    return (1.0 + cos(rg.ata)) * 0.5 * (1.0 + cos(rg.aspect)) * 0.5 * rg.range_factor


def build_observation(me: AState, bandit: AState, rg: RelGeom) -> np.ndarray:
    """The egocentric, normalized observation vector (design doc §8.2).

    Layout (31 floats): 14 own frame-invariant + 14 relative-geometry + 3 WEZ.
    Absolute heading ψ is deliberately dropped.
    """
    feats = [
        # --- own state (frame-invariant), 14 ---
        me.tas / C.V_REF,
        (me.alt_hae_m - C.DECK_HAE_M) / C.ALT_REF,
        sin(me.phi),
        cos(me.phi),
        sin(me.theta),
        cos(me.theta),
        me.p / C.RATE_REF,
        me.q / C.RATE_REF,
        me.r / C.RATE_REF,
        me.alpha / C.ALPHA_REF,
        me.beta / C.BETA_REF,
        (me.nz - 1.0) / C.NZ_SCALE,
        me.hit_points / C.HP_MAX,
        me.ammo / C.AMMO_MAX,
        # --- relative geometry (bandit in my body frame), 14 ---
        min(rg.rng / C.R_MAX, 2.0),
        sin(rg.ata_az),
        cos(rg.ata_az),
        sin(rg.ata_el),
        cos(rg.ata_el),
        sin(rg.aspect),
        cos(rg.aspect),
        sin(rg.hca),
        cos(rg.hca),
        rg.closure / C.V_REF,
        rg.rel_alt / C.ALT_REF,
        bandit.tas / C.V_REF,
        bandit.hit_points / C.HP_MAX,
        1.0 if bandit.alive else 0.0,
        # --- WEZ / gun-solution aids (sim fire control), 3 ---
        1.0 if rg.in_range else 0.0,
        rg.gun_quality,
        cos(rg.aim_error),
    ]
    assert len(feats) == C.OBS_DIM, f"obs dim {len(feats)} != {C.OBS_DIM}"
    return np.asarray(feats, dtype=np.float32)


def _wrap_pi(a: float) -> float:
    return astrum_env.wrap_pi(a)
