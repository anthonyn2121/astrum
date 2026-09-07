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
from math import asin, atan, atan2, cos, exp, hypot, sin

import numpy as np

from . import constants as C


def _finite(x, default: float = 0.0):
    """Map any non-finite (NaN/±inf) entry to `default`; finite values pass through.

    Hardens the sim→Python boundary: at a high-g merge JSBSim can occasionally emit a
    non-finite raw field (α/nz/velocity spike at closest approach). Left unsanitized it
    flows through `_unit` (which guards exact-zero, not NaN) into the range/closure geometry
    and the closing-PBRS reward, producing a NaN reward/obs — a rare batch the learner's
    NaN-guard then has to drop. Converting the rare garbage to benign-finite 0 at the source
    keeps that step finite (garbage-but-harmless) instead of poisoning the transition."""
    return np.nan_to_num(x, nan=default, posinf=default, neginf=default)


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
            pos=_finite(np.array([o.pos_e, o.pos_n, o.pos_u], dtype=np.float64)),
            vel=_finite(np.array([o.vel_e, o.vel_n, -o.vel_d], dtype=np.float64)),
            phi=float(_finite(o.phi)),
            theta=float(_finite(o.theta)),
            psi=float(_finite(o.psi)),
            p=float(_finite(o.p)),
            q=float(_finite(o.q)),
            r=float(_finite(o.r)),
            alpha=float(_finite(o.alpha_rad)),
            beta=float(_finite(o.beta_rad)),
            nz=float(_finite(o.nz_g)),
            airspeed=float(_finite(o.airspeed_mps)),
            alt=float(_finite(o.alt_m)),
            health=float(_finite(o.health)),
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
    # `np.isfinite(n)` guards a non-finite input (defense in depth behind `_finite` at the
    # AState boundary): a NaN/inf vector has a non-finite norm, and `NaN > 1e-9` is False, so
    # the un-guarded form would return the garbage vector unchanged. Zero is the safe unit-less.
    return v / n if np.isfinite(n) and n > 1e-9 else np.zeros_like(v)


def sigma_track(rng: float) -> float:
    """Range-dependent HONEST gun-laying lead tolerance (rad) — G1 (curriculum-78).

    The fixed `LEAD_SIGMA`=3° tracking Gaussian is ~3-4× LOOSER than the true 20mm ballistic
    hit threshold at range: the 8 m hit sphere (+ the 8 mrad dispersion cone) subtends only
    `arctan((HIT_RADIUS_M + DISP_MARGIN·DISP_SIGMA·rng)/rng)` — ~1.39° @494 m, ~0.65° @1200 m —
    so a 3° "near solution" at the merge crossing scores gun_quality≈0.36 yet lands 0 hits
    (494·sin3° ≈ 26 m ≫ 8 m). This tolerance TIGHTENS with range but is CLAMPED at
    `LEAD_SIGMA_CAP` (= the old 3°), so at the close chase-finish (≤~180 m, where the 8 m sphere
    already subtends ≥3°) it is IDENTICAL to today — the finish is preserved; only un-hittable
    long-range shots lose their false credit. Drives the reward math (`gun_quality_honest`) AND,
    from curriculum-80 (from-scratch), the OBSERVED gun-solution feature (obs idx 29) — the warm-net
    byte-identity constraint that formerly kept the obs on the fixed `LEAD_SIGMA` no longer applies.
    """
    tol = atan((C.HIT_RADIUS_M + C.DISP_MARGIN * C.DISP_SIGMA * rng) / max(rng, 1.0))
    return min(max(tol, C.LEAD_SIGMA_FLOOR), C.LEAD_SIGMA_CAP)


def p_hit_range(rng: float) -> float:
    """MEASURED per-round hit probability at `rng` — B1's honest range weighting.

    `sigma_track` made the ANGULAR tolerance honest; this makes the RANGE weighting honest. The old
    `range_factor = (R_EFF − rng)/R_EFF` paid 0.42 at 700 m and 0.25 at 900 m for shots the sim's
    ballistics cannot land (measured 0.060 and 0.002), which is why the policy learned to fire at a
    median 1519 m. Linear interpolation of `C.PHIT_PER_ROUND` over `C.PHIT_RANGE_M`, clamped to 1.0
    inside the first knot and 0.0 beyond the last — see the constants block for the measurement
    conditions (5 g bandit, computable lay) and for why the sub-200 m flat spot is deliberate.
    """
    return float(np.interp(rng, C.PHIT_RANGE_M, C.PHIT_PER_ROUND, left=1.0, right=0.0))


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
    lead_error: float  # nose vs the CORRECTED required boresight (rad); LOW = solution
    in_range: bool
    gun_quality: float  # ∈[0,1]: in-range × tracking Gaussian (fixed LEAD_SIGMA; legacy/telemetry only)
    range_factor: float  # ∈[0,1]: B1 — the MEASURED p_hit/round at this range (`p_hit_range`), NOT a linear R_eff ramp
    gun_quality_honest: float = 0.0  # ∈[0,1]: G1 — in-range × range-dependent sigma_track(rng) Gaussian (drives the OBS feature + outcome-mode reward)
    naive_lead_error: float = 0.0  # rad; the PRE-curriculum-86 lead_error (nose→aim-point, no velocity inheritance). Kept so the SCRIPTED bots' fire gates stay byte-identical — the T3 benchmark must not move under a learner-side fix.
    lay_tof: float = 0.0  # s; exact time of flight of the corrected solution (rng/1030 over-states it by 14-20%)
    lay_miss_m: float = 0.0  # m; predicted miss distance at the target = 1030·sin(lead_error)·lay_tof. The physically right "will this round hit?" scalar, directly comparable to the sim's 8 m hit sphere.
    lay_feasible: bool = True  # False ⇒ no forward-time gun solution (never observed: |v_rel| ≪ 1030)
    # --- curriculum-87 (2026-09-04): the arrival-geometry group. All exact closed forms over the
    #     two current states — nothing here needs a previous frame, so `relative` stays a pure
    #     function and no history is plumbed through the env.
    los_rate_az: float = 0.0  # rad/s; LOS rotation in body azimuth (+ = bandit drifting right)
    los_rate_el: float = 0.0  # rad/s; LOS rotation in body elevation (+ = drifting up)
    los_rate: float = 0.0  # rad/s; the TOTAL LOS rotation rate, |v_perp|/rng — the physical quantity
    #     that sets the turn rate a tracking solution demands. Provided BECAUSE the obvious
    #     reconstruction `hypot(los_rate_az, los_rate_el)` is WRONG and four diagnostics used it: those
    #     two are projections of `v_perp/rng` onto body starboard and body up, which span the plane
    #     perpendicular to the NOSE — not to the LOS — so whenever the bandit is off the nose they drop
    #     `v_perp`'s nose-axis component. The hypot therefore UNDER-states by roughly cos(ata), i.e.
    #     worst exactly at the high-ata merge geometries this program studies (measured 0.70× the mean
    #     and 0.32× the p10 of the true value). It is not recoverable from the two stored components,
    #     which is why it is stored rather than left to callers.
    lead_az: float = 0.0  # rad; SIGNED azimuth of the required boresight off the nose (+ = right)
    lead_el: float = 0.0  # rad; SIGNED elevation of the required boresight off the nose (+ = up)
    alpha_req: float = 0.0  # rad; angle between the required boresight and my VELOCITY vector — the
    #     off-velocity cone the wing has to supply. The usable ceiling is lift-limited to ~10.7-15°,
    #     so this is the achievability scalar: a head-on merge needs ~94° and is un-shootable at any
    #     aim quality, while a chase solution sits inside the cone ([[gun-lay-velocity-inheritance]]).


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

    # Gun solution. curriculum-86 (2026-09-04): this is now the VELOCITY-INHERITANCE-CORRECTED
    # lay, i.e. the direction the nose must point so that `me.vel + 1030·nose` (what
    # `fire_guns` actually launches) intercepts the bandit — not the direction to a predicted
    # aim point. See [[gun-lay-velocity-inheritance]]: measured tick-exactly on c84's own 1369
    # rounds vs the current T3, the nose sat 2.16° from the correct lay while the OLD
    # lead_error read 1.53° (gun_quality 0.88), and correcting the lay alone is worth +938.8
    # hits / Ekills 5.93 → 24.71 out of the SAME trigger pulls. The old value was not a
    # loose approximation of this one; at any loaded pull it points somewhere else entirely.
    #
    # `gunlay.intercept` is closed-form and always well-posed here (one positive root whenever
    # the RELATIVE closing speed is under 1030 m/s), and it returns the exact time of flight,
    # which also removes the second error in the old form: `rng/1030` over-states tof by
    # 14-20% because the inherited velocity makes the round's groundspeed ~1230 m/s.
    from .gunlay import intercept  # local: gunlay imports constants, not geometry

    naive_tof = rng / C.MUZZLE_MPS if rng > 1e-6 else 0.0
    naive_dir = _unit(bandit.pos + bandit.vel * naive_tof - me.pos)
    naive_lead_error = atan2(
        float(np.linalg.norm(np.cross(naive_dir, nose))),
        float(np.clip(np.dot(naive_dir, nose), -1.0, 1.0)),
    )
    sol = intercept(me.pos, me.vel, bandit.pos, bandit.vel)
    aim_dir = sol.boresight if sol.feasible else naive_dir
    lead_cos = float(np.clip(np.dot(aim_dir, nose), -1.0, 1.0))
    lead_error = atan2(float(np.linalg.norm(np.cross(aim_dir, nose))), lead_cos)
    tof = sol.tof_s if sol.feasible else naive_tof
    # Being δ off the required boresight puts the round `1030·sin δ·tof` off the target: the
    # muzzle term is the only part of `me.vel + 1030·nose` that rotates with the nose.
    lay_miss_m = C.MUZZLE_MPS * sin(lead_error) * tof
    # `in_range` stays on R_EFF (1200 m) ON PURPOSE — it is what the scripted T3's trigger fires on,
    # so it is benchmark surface, not learner surface (see the R_EFF warning in constants.py).
    # `range_factor` is learner-only, and B1 replaces its linear fiction with the measured p_hit.
    in_range = rng <= C.R_EFF
    range_factor = p_hit_range(rng)
    track = float(np.exp(-0.5 * (lead_error / C.LEAD_SIGMA) ** 2))
    gun_quality = (range_factor * track) if in_range else 0.0
    # G1 (curriculum-78): honest range-dependent quality — same shape, but sigma_track(rng)
    # tightens the tracking Gaussian at range (clamped at LEAD_SIGMA close-in ⇒ identical to
    # `gun_quality` for ≤~180 m). Reward math (K_SOLUTION/K_AIM under outcome_mode) reads this;
    # the observation keeps `gun_quality` on the fixed LEAD_SIGMA.
    # ⚠ "byte-identical obs" USED to be true of that σ choice and is NO LONGER true of the obs as a
    # whole: curriculum-86 made `lead_error` the velocity-inheritance-corrected lay, and
    # `build_observation` feeds `rg.gun_quality` (idx 29) and `cos(rg.lead_error)` (idx 30). So c86 is
    # a reward change AND an observation change, and any checkpoint trained before it sees those two
    # columns OUT OF DISTRIBUTION. Evaluating a pre-c86 policy in this env is therefore NOT a fair
    # baseline for it (measured: c84 scored 2 kills here vs its recorded 10). See
    # [[gun-lay-velocity-inheritance]], [[curriculum-86-corrected-aim]].
    track_honest = float(np.exp(-0.5 * (lead_error / sigma_track(rng)) ** 2))
    gun_quality_honest = (range_factor * track_honest) if in_range else 0.0

    # curriculum-87 arrival geometry. LOS rate is the exact kinematic form: only the component of
    # the relative velocity PERPENDICULAR to the LOS rotates the line, and it does so at |v_perp|/r.
    # Deriving it analytically (rather than finite-differencing a stored previous LOS) keeps this
    # function pure and makes the feature exact at the merge, where a one-frame difference is
    # noisiest and matters most.
    v_perp = v_rel - float(np.dot(v_rel, los_hat)) * los_hat
    inv_r = 1.0 / rng if rng > 1e-6 else 0.0
    los_rate_az = float(np.dot(v_perp, starboard)) * inv_r
    los_rate_el = float(np.dot(v_perp, -down)) * inv_r
    los_rate = float(np.linalg.norm(v_perp)) * inv_r  # the TOTAL rate — see the field comment

    # Signed body decomposition of the SAME corrected boresight `lead_error` measures. `lead_error`
    # is the unsigned total, so on its own it cannot say which way to correct; these two can.
    aim_x = float(np.dot(aim_dir, nose))
    aim_y = float(np.dot(aim_dir, starboard))
    aim_z = float(np.dot(aim_dir, down))
    lead_az = atan2(aim_y, aim_x)
    lead_el = atan2(-aim_z, hypot(aim_x, aim_y))

    # Achievability: how far off my own velocity vector the required boresight sits. The airframe
    # can only hold the nose off the flight path by its trimmed α, so this — not the aim error —
    # is what bounds whether a shot exists at all.
    v_hat = _unit(me.vel)
    alpha_req = atan2(
        float(np.linalg.norm(np.cross(aim_dir, v_hat))),
        float(np.clip(np.dot(aim_dir, v_hat), -1.0, 1.0)),
    )

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
        gun_quality_honest=gun_quality_honest,
        naive_lead_error=naive_lead_error,
        lay_tof=tof,
        lay_miss_m=lay_miss_m,
        lay_feasible=sol.feasible,
        los_rate_az=los_rate_az,
        los_rate_el=los_rate_el,
        los_rate=los_rate,
        lead_az=lead_az,
        lead_el=lead_el,
        alpha_req=alpha_req,
    )


def specific_energy(s: AState) -> float:
    """Specific energy `Es = alt + V²/2g` (m) — the altitude the jet could trade its speed for.

    The scalar the energy fight is actually fought over: a turn that holds Es is sustainable, one
    that bleeds it buys angles on credit. Only the DIFFERENCE between the two jets is observed
    (obs tail), since absolute Es is already recoverable from the airspeed and altitude features.
    """
    return float(s.alt + s.airspeed * s.airspeed / (2.0 * C.G_MPS2))


def flight_path_angle(s: AState) -> float:
    """Flight-path angle γ (rad, + = climbing) — where the jet is GOING, not where its nose points.

    Distinct from pitch attitude θ by the angle of attack (θ = γ + α only wings-level), and γ is
    the one the vertical fight is fought in. F4 (2026-09-07) is the cautionary case: the T3's break
    holds a FIXED 85° of bank, whose level-flight load 1/cos85 = 11.5 g is past the airframe's
    limiter, and a bank/load identity therefore predicts it must descend — yet it CLIMBS 460-980 m,
    because it enters nose-high and its pursuit law has no γ term to take that entry away. θ and
    the bank angle did not predict the flight path; γ is the measurement.
    """
    v = float(np.linalg.norm(s.vel))
    if v <= 1e-6:
        return 0.0
    return asin(max(-1.0, min(1.0, float(s.vel[2]) / v)))


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

    Layout (`C.OBS_DIM` floats): 14 own frame-invariant + 14 relative-geometry + 3 WEZ + 10
    appended by curriculum-87 (arrival geometry / signed aim / achievability / bandit turn state /
    relative energy) + 4 appended by curriculum-97 (the signed vertical axis, gated on
    `C.C97_VERTICAL_OBS`; 45 with it, 41 without). Absolute heading ψ is deliberately dropped.
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
        # curriculum-80 P2 (2026-09-02): REVERTED to the fixed-LEAD_SIGMA `gun_quality` (loose 3°).
        # The P1 attempt fed the HONEST range-dependent `gun_quality_honest` here (and into the reward)
        # to match ballistics — but that was an OWN-GOAL for a COLD learner: the honest ~1.4° σ Gaussian
        # is ≈0 with NO SLOPE for any aim wider than ~2°, so the K_SOLUTION/K_AIM aim-shaping gradient
        # went flat at chase range and the cold net never learned to point at the lead solution (0 kills
        # every aspect, sprayed the mag for ~0 HP). The loose σ's "false credit" ([[curriculum-76-77]])
        # is ALSO the essential aim-shaping gradient; kill-honesty is enforced by the SIM's real 20mm
        # ballistics (50 hits/kill), not by tightening this feature. So obs↔reward are a coherent LOOSE
        # pair again (reward.py q_sol/aim_r also reverted to loose), and the finish is re-earned cold vs
        # the current aggressive T3 with a learnable gradient.
        rg.gun_quality,
        cos(rg.lead_error),
        # --- curriculum-87 append: arrival geometry, turn state, achievability, energy (10) ---
        # Ordered AFTER the original 31 so `gun_quality` keeps absolute frame index 29 and
        # `GUN_QUALITY_OBS_INDEX - OBS_DIM` (= the stack-safe negative index the fire-override
        # reads) keeps resolving to it. Every new column is zero-init'd in the encoder on a warm
        # start, so the resumed policy begins byte-identical and has to EARN any use of them.
        #
        # (a) Arrival geometry — the measured wall. c86 kills 8/8 arriving at ~18°/s LOS rate and
        #     scores 0/8 arriving at ~59°/s; it could not tell those two apart. Signed, so "he is
        #     sliding right/up across my nose" is distinguishable from the mirror case, which is
        #     what picks the direction of a lag/lead correction and of an out-of-plane roll.
        rg.los_rate_az / C.LOS_RATE_REF,
        rg.los_rate_el / C.LOS_RATE_REF,
        # (b) SIGNED aim error. `cos(lead_error)` above is unsigned AND saturates (cos is flat to
        #     second order near 0), so near a solution it carries almost no gradient and never
        #     says which way to pull. `sin` of the signed body components is bounded, monotonic
        #     through zero, and steepest exactly where the shot is decided.
        sin(rg.lead_az),
        sin(rg.lead_el),
        # (c) Achievability. Normalized by the airframe's own AoA limiter, so ~1 means "this shot
        #     needs all the α the wing has" and >1 means no shot exists at any aim quality — the
        #     distinction that makes a head-on merge un-shootable regardless of tracking. CLIPPED
        #     at 2 (=60°) like the miss below: `alpha_req` ranges to 180° (a shot behind me), and
        #     an unclipped 6.0 would dominate the encoder's input scale to say only "impossible",
        #     which 2.0 already says. The resolution that matters is inside the usable cone.
        min(rg.alpha_req / C.ALPHA_REF, 2.0),
        # (d) Predicted miss against the sim's real 8 m hit sphere, clipped so the hopeless tail
        #     (which is most rounds at ≥55° entry) cannot dominate the scale.
        min(rg.lay_miss_m / C.MISS_REF, 2.0),
        # (e) The bandit's turn PLANE and turn RATE — what "out-of-plane" and "turn circle" are
        #     defined against, and previously absent entirely ([[obs-blind-to-opponent-turn]]).
        #     Bank gives the plane he is pulling in; the pitch/yaw rate magnitude gives how hard.
        sin(bandit.phi),
        cos(bandit.phi),
        hypot(bandit.q, bandit.r) / C.RATE_REF,
        # (f) Relative specific energy — the currency of "gain enough energy to reach a killing
        #     position". c86's frontier cells sit 473-528 m down at closest approach vs 294 m in
        #     the cell it wins, and it had no way to see that it was buying angles on credit.
        (specific_energy(me) - specific_energy(bandit)) / C.ES_REF,
    ]
    # --- curriculum-97 append: the VERTICAL axis, signed and rate-bearing (4) ---
    # F5 (2026-09-07, `runs/c97/c97_vertical.txt`) measured 24 whole 180 s episodes on the perched
    # merge: c96 zooms to γ p95 **+58..+68° on every single episode**, then spends 43.7% of its
    # steps below `V_BAND_CTR` and 21.7% below `V_CORNER_FLOOR` (V_min median 114 m/s) and nets
    # −1395 m, climbing in 1/8. That is `reward.py:322-327`'s own named failure verbatim — the
    # merge zoom-to-stall. The reward is NOT the explanation: all six vertical-coupled terms are
    # negative (anti-dive) and total only −25..−40 against returns of −364..−535, and `band_reward`
    # telescopes to −0.00 on all 24 episodes ⇒ this is an OBSERVABILITY box, not a reward box.
    #
    # What the policy cannot see: the checkpoint runs `frame_stack = 1`, so NO rate is recoverable
    # by differencing frames — every rate must be handed over explicitly. The HORIZONTAL axis is:
    # position `rng` (idx 14) AND rate `closure` (idx 23). The VERTICAL axis is position ONLY —
    # `rel_alt` (idx 24), whose first-layer column norm ranks 3rd of 41 in c96/c95/c87 alike, so
    # the policy plainly USES height — with no vertical rate anywhere in the vector, and the
    # bandit's climb rate in no form at all (he is given as scalar airspeed/health/alive plus an
    # UNSIGNED `hypot(q, r)`). Own γ needs *products* of the sin/cos θ, α and sin φ channels that
    # obs 0-13 hold individually, and the first `NormedLinear` — Linear → LayerNorm → Mish —
    # cannot form a product of its inputs. Same class of gap as the curriculum-87 arrival append:
    # information present in principle, not linearly available, therefore not conditionable on.
    #
    # ⚠ Gated so the paired 41-obs NULL CONTROL runs from this same tree — see `C.OBS_DIM`.
    if C.C97_VERTICAL_OBS:
        feats += [
            (bandit.vel[2] - me.vel[2]) / C.V_REF,
            sin(flight_path_angle(me)),
            sin(flight_path_angle(bandit)),
            # SIGNED pitch rate. Feature (e) above ships `hypot(bandit.q, bandit.r)`, and its own
            # sibling (b) already made the argument against that shape: an unsigned magnitude
            # "never says which way to pull". Measured consequence — a bandit pulling UP at
            # 0.3 rad/s and one pushing DOWN at 0.3 rad/s produce a BIT-IDENTICAL observation
            # today. The `hypot` slot is kept untouched (it is what the champion trained on;
            # mutating it would make this two experiments) and the sign is restored alongside it.
            bandit.q / C.RATE_REF,
        ]
    assert len(feats) == C.OBS_DIM, f"obs dim {len(feats)} != {C.OBS_DIM}"
    return np.asarray(feats, dtype=np.float32)


# Absolute index of the gun-solution quality ∈ [0,1] within a single observation frame — the last
# of the three WEZ features, which curriculum-87's 10-feature append deliberately sits BEHIND so
# this stays 29. Consumers reading it off a *frame-stacked* vector should index the current frame
# with `obs[GUN_QUALITY_OBS_INDEX - C.OBS_DIM]` (now obs[-12], was obs[-2]); that form is both
# stack-depth-independent (FrameStack concatenates whole frames, newest last) and append-proof,
# which is why the fire-override in `train_parallel.py` needed no change.
GUN_QUALITY_OBS_INDEX = 29


def _wrap_pi(a: float) -> float:
    from math import pi

    return (a + pi) % (2.0 * pi) - pi
