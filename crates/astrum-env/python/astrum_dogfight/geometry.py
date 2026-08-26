"""Egocentric dogfight geometry and the observation vector (design doc §8.2).

All frame math is grounded in the two authoritative sources:

- **ENU world frame** (`crates/astrum-geo`): East=+X, North=+Y, Up=+Z.
- **Body attitude DCM** (`web/js/meshes.js` `attitudeToQuaternion`): aerospace
  3-2-1 Euler (φ roll, θ pitch, ψ yaw/true-heading); body axes nose=+X,
  starboard=+Y, down=+Z. That file emits the body→NED columns; here we express
  the same axes directly in ENU via (E, N, U) = (e, n, −d).

`AState` is a plain, fabricable snapshot of one aircraft (so reward/geometry unit
tests need no live sim). Everything downstream operates on `AState`, never on the
pyo3 `Observation` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin

import numpy as np

from . import constants as C


@dataclass
class AState:
    """A plain snapshot of one aircraft, in ENU. Fabricable for tests."""

    id: int
    team: int
    alive: bool
    pos: np.ndarray  # ENU (E, N, U) metres
    vel: np.ndarray  # ENU velocity (m/s)
    phi: float
    theta: float
    psi: float
    p: float
    q: float
    r: float
    alpha: float
    beta: float
    nz: float
    airspeed: float
    alt: float  # MSL/ellipsoidal metres (== GeoState alt)
    health: float
    ammo: int

    @staticmethod
    def from_obs(o) -> "AState":
        """Convert a pyo3 `astrum_env.Observation` into an `AState`.

        NED velocity → ENU: (E, N, U) = (vel_e, vel_n, −vel_d).
        """
        return AState(
            id=int(o.id),
            team=int(o.team),
            alive=bool(o.alive),
            pos=np.array([o.pos_e, o.pos_n, o.pos_u], dtype=np.float64),
            vel=np.array([o.vel_e, o.vel_n, -o.vel_d], dtype=np.float64),
            phi=float(o.phi),
            theta=float(o.theta),
            psi=float(o.psi),
            p=float(o.p),
            q=float(o.q),
            r=float(o.r),
            alpha=float(o.alpha_rad),
            beta=float(o.beta_rad),
            nz=float(o.nz_g),
            airspeed=float(o.airspeed_mps),
            alt=float(o.alt_m),
            health=float(o.health),
            ammo=int(o.ammo),
        )


def body_axes_enu(phi: float, theta: float, psi: float):
    """Return (nose, starboard, down) unit vectors in ENU for an attitude.

    Derived from the body→NED DCM in `web/js/meshes.js:40-42`, mapped to ENU by
    (E, N, U) = (e, n, −d). Verified level (φ=θ=0, ψ=0): nose=+N, starboard=+E,
    down=−U.
    """
    cph, sph = cos(phi), sin(phi)
    cth, sth = cos(theta), sin(theta)
    cps, sps = cos(psi), sin(psi)
    nose = np.array([cth * sps, cth * cps, sth])
    starboard = np.array([sph * sth * sps + cph * cps, sph * sth * cps - cph * sps, -sph * cth])
    down = np.array([cph * sth * sps - sph * cps, cph * sth * cps + sph * sps, -cph * cth])
    return nose, starboard, down


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
    rel_alt: float  # bandit_alt − my_alt (m)
    lead_error: float  # nose vs required TOF-lead direction (rad); LOW = solution
    in_range: bool
    gun_quality: float  # ∈[0,1]: in-range × tracking Gaussian
    range_factor: float  # ∈[0,1]: 1 at point-blank → 0 at R_eff


def relative(me: AState, bandit: AState) -> RelGeom:
    los = bandit.pos - me.pos  # ENU, me → bandit
    rng = float(np.linalg.norm(los))
    los_hat = _unit(los)

    nose, starboard, down = body_axes_enu(me.phi, me.theta, me.psi)
    lx = float(np.dot(los_hat, nose))  # forward component
    ly = float(np.dot(los_hat, starboard))  # right component
    lz = float(np.dot(los_hat, down))  # down component
    ata = atan2(hypot(ly, lz), lx)  # total angle off the nose
    ata_az = atan2(ly, lx)
    ata_el = atan2(-lz, hypot(lx, ly))

    # Aspect: at the bandit, angle between its tail and the LOS to me. With
    # LOS = me→bandit and nose_b the bandit's forward, cos(aspect) = nose_b · LOS_hat
    # (=+1 when I sit on his 6, nose pointing away from me).
    nose_b, _, _ = body_axes_enu(bandit.phi, bandit.theta, bandit.psi)
    aspect = atan2(
        float(np.linalg.norm(np.cross(nose_b, los_hat))),
        float(np.dot(nose_b, los_hat)),
    )

    # HCA from flight-path courses (falls back to heading when nearly stationary).
    def course(s: AState) -> float:
        if hypot(s.vel[0], s.vel[1]) > 1.0:
            return atan2(s.vel[0], s.vel[1])  # atan2(E, N)
        return s.psi

    hca = _wrap_pi(course(bandit) - course(me))

    v_rel = bandit.vel - me.vel
    closure = -float(np.dot(v_rel, los_hat))  # + = closing
    rel_alt = bandit.alt - me.alt

    # Gun solution: lead the bandit by its travel over the round's time-of-flight
    # (drag-free approximation), then measure my boresight error to that aim point.
    tof = rng / C.MUZZLE_MPS if rng > 1e-6 else 0.0
    aim = bandit.pos + bandit.vel * tof
    aim_dir = _unit(aim - me.pos)
    lead_cos = float(np.clip(np.dot(aim_dir, nose), -1.0, 1.0))
    lead_error = atan2(float(np.linalg.norm(np.cross(aim_dir, nose))), lead_cos)
    in_range = rng <= C.R_EFF
    range_factor = float(np.clip((C.R_EFF - rng) / C.R_EFF, 0.0, 1.0))
    track = float(np.exp(-0.5 * (lead_error / C.LEAD_SIGMA) ** 2))
    gun_quality = (range_factor * track) if in_range else 0.0

    return RelGeom(
        rng=rng,
        ata=ata,
        ata_az=ata_az,
        ata_el=ata_el,
        aspect=aspect,
        hca=hca,
        closure=closure,
        rel_alt=rel_alt,
        lead_error=lead_error,
        in_range=in_range,
        gun_quality=gun_quality,
        range_factor=range_factor,
    )


def lead_point(me: AState, bandit: AState) -> np.ndarray:
    """Drag-free gun-lead aim point (ENU): where to point the nose so a round fired
    now meets the bandit, using its travel over the round's time-of-flight (§8.2)."""
    rng = float(np.linalg.norm(bandit.pos - me.pos))
    tof = rng / C.MUZZLE_MPS if rng > 1e-6 else 0.0
    return bandit.pos + bandit.vel * tof


def body_bearing(me: AState, point: np.ndarray):
    """Direction to an ENU `point` expressed in `me`'s body frame → `(off, bank_err,
    forward)`.

    - `off` — total angle between the nose and the point (rad; 0 = on boresight).
    - `bank_err` — roll (rad, + = roll right) that would bring the point into the
      vertical plane *above* the nose, i.e. put the lift vector on it.
    - `forward` — nose-alignment cosine (dot of the direction with the nose axis).
    """
    d = _unit(point - me.pos)
    nose, starboard, down = body_axes_enu(me.phi, me.theta, me.psi)
    lx = float(np.dot(d, nose))
    ly = float(np.dot(d, starboard))
    lz = float(np.dot(d, down))
    off = atan2(hypot(ly, lz), lx)
    bank_err = atan2(ly, -lz)  # 0 when the point is straight up out of the canopy
    return off, bank_err, lx


def potential(me: AState, bandit: AState, rg: RelGeom | None = None) -> float:
    """Shaping potential Φ ∈ [0, 1] = nose-on ∧ rear-hemisphere ∧ in-WEZ (§8.3).

    Product form: all three factors must be good for a clean gun-kill geometry. 0
    when either aircraft is dead (no live advantage to bank).
    """
    if not me.alive or not bandit.alive:
        return 0.0
    rg = rg or relative(me, bandit)
    f_ata = (1.0 + cos(rg.ata)) * 0.5
    f_aspect = (1.0 + cos(rg.aspect)) * 0.5
    return f_ata * f_aspect * rg.range_factor


def build_observation(me: AState, bandit: AState, ground_alt: float = 0.0) -> np.ndarray:
    """The egocentric, normalized observation vector (design doc §8.2).

    Layout (31 floats): 14 own frame-invariant + 14 relative-geometry + 3 WEZ.
    Absolute heading ψ is deliberately dropped.
    """
    rg = relative(me, bandit)
    alt_agl = me.alt - ground_alt
    feats = [
        # --- own state (frame-invariant), 14 ---
        me.airspeed / C.V_REF,
        (alt_agl - C.DECK_AGL) / C.ALT_REF,
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
        me.health / C.HP_MAX,
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
        bandit.airspeed / C.V_REF,
        bandit.health / C.HP_MAX,
        1.0 if bandit.alive else 0.0,
        # --- WEZ / gun-solution aids, 3 ---
        1.0 if rg.in_range else 0.0,
        rg.gun_quality,
        cos(rg.lead_error),
    ]
    assert len(feats) == C.OBS_DIM, f"obs dim {len(feats)} != {C.OBS_DIM}"
    return np.asarray(feats, dtype=np.float32)


def _wrap_pi(a: float) -> float:
    from math import pi

    return (a + pi) % (2.0 * pi) - pi
