"""BFM event detection — the measurement layer this program never had.

Every prior BFM claim in this project was made without a detector, and that cost us: two
independent analyses of the SAME flight data reached OPPOSITE conclusions about whether the policy
commits to rolls, purely from statistic choice (one used a one-sided `ail > 0.5` and reported a max
run of 18 steps; the other used a sign-agnostic `abs(ail) > 0.5` with no sign-constancy test and
reported 3.54 s). Both are wrong for the question — and, it turned out, both were wrong about the
QUANTITY too: `a[0]` is a roll-RATE command on this FLCS, so neither statistic could have answered
"did it commit to a roll" at any threshold. See `commitment` for the correction. Until a maneuver
can be *detected* — on the right quantity — a run that
SCORES better is indistinguishable from a run that LEARNED BFM — so score deltas cannot be
attributed, and a reward that pays for the wrong proxy cannot be caught.

This module is a **pure function of a per-tick trace**. It adds no sim state, no observation
features and no GPU work: everything it needs is already on `geometry.AState` plus the flown action.
That matters because it has to be runnable over *historical* checkpoints (c87 and c93 as well as
c95) as a negative control — an "improvement" that also shows up in c87 is a detector artifact, not
a learning result.

WHAT IT MEASURES

The whole family reduces to one scalar. `chi` is the roll of the aircraft's LIFT VECTOR off the
plane containing its velocity vector and the line of sight:

    chi ~ 0        the lift vector is ON the target: an IN-PLANE pull (classic pure pursuit)
    abs(chi) large the lift vector is displaced off the LOS plane: OUT-OF-PLANE maneuvering

That single number is what "out-of-plane", "displacement roll" and "turning room" are all defined
against, and it was previously unmeasured. Everything below is a predicate over `chi`, the
flight-path angle, load factor, and closure.

THE TWO HALVES OF THE QUESTION

The user's framing is that a real pilot has to know both HOW to fly a maneuver and WHEN. These map
onto different statistics and must not be conflated:

  * HOW  — D1/D2/D3 count events whose SHAPE is right (the roll goes out and comes back, the pull is
           loaded, the geometry improves).
  * WHEN — D2's entry gate (`closure > CLOSURE_OVERSHOOT`) and, decisively, **D5**: the ratio of
           lethal-shot probability after an event to the base rate. A policy that learned the shape
           but not the timing raises D1/D2 counts at D5 ratio ~1.0. Only a policy that learned the
           timing moves the ratio. Counts alone are gameable by a policy that rolls at random.

NEVER report as evidence of BFM: episode length (measured not to discriminate), a training log's
`kill%`/`ret` (that is exploration, not the deployed policy), reward-term sums (gamma=1-differenced
and one-sided-gated, hence path-dependent), or `hypot(los_rate_az, los_rate_el)` (which is 0.70x the
mean and 0.32x the p10 of the physical `|v_perp|/rng` -- read `RelGeom.los_rate`, which is that
quantity exactly, and is not reconstructible from the two body components).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .geometry import AState, RelGeom, body_axes_enu, relative, specific_energy

# ======================================================================================
# Thresholds. Module-level so the negative controls (D6) can probe sensitivity without
# editing detector bodies — a threshold that only works at one setting is not a detector.
# ======================================================================================

CHI_INPLANE_DEG = 20.0  # below this, the lift vector is effectively ON the LOS plane
CHI_OUT_DEG = 60.0  # above this, genuinely out of plane (D1)
CHI_ROOM_DEG = 45.0  # the looser out-of-plane bar for a turning-room build (D3)
CHI_RETURN_DEG = 30.0  # "rolled back in" — the closing half of a displacement roll
CHI_MIN_HOLD_S = 0.5  # an out-of-plane INSTANT is noise; a HOLD is a maneuver
CHI_RETURN_S = 4.0  # roll back in within this or it was a departure, not a displacement
CHI_LOOKBACK_S = 4.0  # must have been in-plane this recently (else it was never displaced)

NZ_LOADED = 3.0  # the pull that separates a maneuver from a wings-level drift
LOADED_FRAC = 0.5  # fraction of a yo-yo's ticks that must be loaded (anti zoom-coast)

CLOSURE_OVERSHOOT = 120.0  # m/s; == constants.CLOSURE_HI — the genuine overshoot / yo-yo trigger
YOYO_GAMMA_SWING_DEG = 30.0  # flight-path swing that constitutes a yo-yo
YOYO_GAMMA_EXCURSION_DEG = 15.0  # ...and it must actually leave the horizon by this much
YOYO_RETURN_S = 6.0  # ...and come back within this
YOYO_GAMMA_RETURN_DEG = 10.0  # "came back" tolerance about the entry flight path
YOYO_EXIT_RNG_M = 1500.0  # a yo-yo that ends outside this range solved nothing
YOYO_ASPECT_SLIP_DEG = 10.0  # aspect may not degrade more than this  <-- the c70 zoom-away detector

ASPECT_GAIN_DEG = 15.0  # a displacement roll must actually BUY angles (D1 clause f)

# ======================================================================================
# D1 REPAIR, 2026-09-07 — the four steps of "the minimum instrument repair" that F2's
# positive control made mandatory (`docs/bfm-program-handoff.md`, "the minimum instrument
# repair (not yet made)").
#
# F2 measured D1 = 8/8 on FOUR merge arms that each deal exactly 0.0 HP — one of which fires
# 0 rounds and dives 2.4 km — and 0/8 on the only arm that kills. Not too strict: NOT
# SPECIFIC. Two clauses were being supplied for free by the geometry of a head-on pass:
#   * (d) "rolled back in" had no requirement that the bandit still be in FRONT (the passing
#     arms exit at ata 139-171 deg, i.e. a small |chi| measured against a LOS pointing over
#     the shoulder), and it accepted the `chi_valid` SENTINEL — `out_of_plane_angle` returns
#     `chi = 0.0` when the LOS is within `CHI_VALID_ATA_DEG` of the velocity vector, and 0.0
#     trivially satisfies `|chi| < CHI_RETURN_DEG`. A target dead ahead scored as "rolled
#     back in".
#   * (f) "aspect improved by >= 15 deg" is automatic on ANY head-on pass, because aspect is
#     measured off the TARGET'S TAIL: fly past a jet you approached head-on and aspect
#     collapses whether or not you out-turned him.
#
# Each step is a separate module-level switch so the audit can price it MARGINALLY rather
# than shipping four coupled edits and one number — the same reason the thresholds above are
# module-level. `_diag_f7_detector_repair.py` reports D1 under every configuration on both
# cells; read that before changing a default here.
# ======================================================================================
D1_REQUIRE_FORWARD_EXIT = True  # repair 1: (d) needs `chi_valid` AND the bandit still ahead
D1_EXIT_ATA_MAX_DEG = 120.0  # ...the forward-hemisphere bound. The LOOSE end of the 90-120
#                              range the repair specifies, chosen so the step cannot be
#                              accused of deleting real events to make the merge count fall.
D1_REQUIRE_NONOPENING = True  # repair 2: range non-increasing at the (d) exit. A pass always
#                               exits on the OPENING side of its own minimum (the four F2
#                               arms bottom out at 97-407 m and are already opening when they
#                               roll level), so this is the one clause a pass cannot fake.
D1_ANGLE_CARRIER = "ata"  # repair 3: which angle clause (f) is carried by.
#   "aspect" = pre-repair (contaminated: a pass supplies it free)
#   "ata"    = the angle off MY NOSE. A pass DEGRADES it (+64 to +150 deg on the F2 arms),
#              so it is the quantity a pass cannot supply. This is the shipped default.
#   "both"   = require aspect AND ata. Retained for the audit only: it preserves the
#              historical chase rejection, which is exactly why it must not be the default —
#              picking a carrier because it keeps a previous answer is tuning, not repair.
#   ⚠ KNOWN BIAS of the ata carrier: an event entered from ata < ATA_GAIN_DEG cannot satisfy
#     it (there is not 15 deg of ata left to win). That is a real blind spot, accepted because
#     a bandit already inside 15 deg of the nose does not need a displacement roll.
ATA_GAIN_DEG = 15.0  # ...its magnitude. Separate from ASPECT_GAIN_DEG so a sensitivity probe
#                      can move one without the other; they are different angles.
D1_CLOSURE_OVER_HOLD = True  # repair 4: (c) becomes the MEAN closure over the hold against a
#                              `-CLOSURE_TOL_MS` floor, instead of the single entry tick against
#                              zero. A 1.333 s / 144.6 deg / 5.25 g displacement was being
#                              discarded by a knife-edge -7 m/s reading on a noisy quantity.
#                              ⚠ NOT a pure loosening, though the -10 m/s floor reads like one:
#                              it swaps the STATISTIC, so a hold whose entry tick is closing but
#                              whose mean over the hold is not now REJECTS. F7's smoke measured
#                              exactly that -- `merge:lead_turn`'s pass enters at +253 m/s and
#                              exits at -348, and repair 4 alone kills it. Both directions must
#                              therefore be priced, which is why it gets its own audit column.
CLOSURE_TOL_MS = 10.0  # the tolerance repair 4 and repair 2's sign test share. Repair 2 uses it
#                        unconditionally (its clause is new, so there is no pre-repair form to
#                        preserve); repair 4's clause falls back to a hard 0.0 when its switch is
#                        off, which is exactly the pre-repair behaviour.

AIL_COMMIT = 0.5  # |ail| above this = a real ROLL-RATE command, not a trim correction
COMMIT_WINDOW_S = 1.5  # window for the signed mean-aileron statistic (D4)
BANK_COMMIT_DEG = 45.0  # |phi| above this = genuinely banked up, not a wings-level drift

CHI_VALID_ATA_DEG = 5.0  # below this the LOS is along the velocity vector and "the LOS plane"
#                          does not exist — chi is undefined, NOT zero. See `out_of_plane_angle`.

LETHAL_RANGE_M = C.LETHAL_RANGE_M  # the MEASURED gun envelope (p_hit/round 0.202 @600 m, 0.000
#                         @1200 m) -- deliberately NOT constants.R_EFF (1200 m), which was the
#                         reward's fiction. B1 made the reward honest too, so this is now RE-EXPORTED
#                         from constants rather than duplicated: `in_solution` (reward.py) and the D5
#                         `lethal` tick below read the SAME number, and the instrument therefore
#                         cannot silently disagree with the reward about what a killing shot is.
D5_WINDOW_S = 8.0  # credit window after an event entry
D5_BASELINE_STRIDE_S = 1.0  # stride for sampling non-event baseline windows


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if np.isfinite(n) and n > 1e-9 else np.zeros(3)


# ======================================================================================
# The scalar
# ======================================================================================
def lift_vector_enu(phi: float, theta: float, psi: float) -> np.ndarray:
    """The aircraft's lift direction in ENU: body "up", i.e. `-down` from the body DCM.

    Body-constant, so this is a pure function of attitude — no aerodynamic state needed. (Lift acts
    along the negative body-z axis in the symmetry plane; the small tilt from angle of attack is
    deliberately ignored, since we care about which way the jet is POINTING its lift, not the exact
    force line.)
    """
    _, _, down = body_axes_enu(phi, theta, psi)
    return -down


def out_of_plane_angle(me: AState, bandit: AState) -> tuple[float, bool]:
    """`(chi, valid)` — the roll of the lift vector off the LOS plane, in radians.

    Both the lift vector and the line of sight are projected into the plane PERPENDICULAR TO THE
    VELOCITY VECTOR and compared there, using the local horizon as the zero-roll reference:

        u_ref = "up"    component of world-up perpendicular to v_hat
        r_ref = "right" = v_hat x u_ref
        chi   = atan2(L.r_ref, L.u_ref) - atan2(LOS.r_ref, LOS.u_ref)

    so `chi = 0` means the lift vector points at the bandit (an in-plane pull will drive the nose
    straight to him) and `abs(chi) = 90 deg` means the lift is pulling square across the LOS.

    DEVIATION FROM THE ORIGINAL SPEC, deliberate: the spec paired this with the body-frame
    `phi_pp = atan2(sin ata_az, tan ata_el)` from `bfm._lift_on_target_bank_err`. That expression is
    correct but is measured about the NOSE, while the lift roll is measured about the VELOCITY
    vector, so differencing them mixes two frames and injects an error of order alpha (5-17 deg here,
    and largest exactly at the loaded merge states we care most about). Measuring BOTH about
    `v_hat` removes that bias, and `atan2` of the projected LOS has no `tan(ata_el)` pole to clamp.

    `valid` is False when the LOS lies within `CHI_VALID_ATA_DEG` of the velocity vector: there the
    "LOS plane" is degenerate and `chi` is genuinely undefined rather than zero. Callers get
    `chi = 0.0`, which is the safe default (it cannot satisfy any out-of-plane predicate), but a
    target dead ahead must never be reported as an in-plane PULL — it is a target dead ahead.
    """
    v_hat = _unit(np.asarray(me.vel, float))
    los = _unit(np.asarray(bandit.pos, float) - np.asarray(me.pos, float))
    if not v_hat.any() or not los.any():
        return 0.0, False

    # Local-horizon roll reference perpendicular to the flight path. Degenerate only in a pure
    # vertical climb/dive, where "up" relative to the velocity vector is undefined; fall back to
    # north, which keeps the frame continuous and well-conditioned through the vertical.
    up = np.array([0.0, 0.0, 1.0])
    u_ref = _unit(up - float(np.dot(up, v_hat)) * v_hat)
    if not u_ref.any():
        north = np.array([0.0, 1.0, 0.0])
        u_ref = _unit(north - float(np.dot(north, v_hat)) * v_hat)
    r_ref = np.cross(v_hat, u_ref)

    los_perp = los - float(np.dot(los, v_hat)) * v_hat
    if float(np.linalg.norm(los_perp)) < math.sin(math.radians(CHI_VALID_ATA_DEG)):
        return 0.0, False

    lift = lift_vector_enu(me.phi, me.theta, me.psi)
    lift_perp = lift - float(np.dot(lift, v_hat)) * v_hat
    if float(np.linalg.norm(lift_perp)) < 1e-6:
        return 0.0, False

    phi_lift = math.atan2(float(np.dot(lift_perp, r_ref)), float(np.dot(lift_perp, u_ref)))
    phi_los = math.atan2(float(np.dot(los_perp, r_ref)), float(np.dot(los_perp, u_ref)))
    return _wrap_pi(phi_lift - phi_los), True


def cpa_miss_m(pos_me, vel_me, pos_op, vel_op) -> float:
    """Miss distance at the predicted closest point of approach of two constant-velocity aircraft.

    The TURNING-ROOM quantity: neither the slant range nor the LOS bearing is. `t < 0` (already past
    the CPA) collapses to the present separation, so a blown pass reports the opening range rather
    than a stale prediction behind the jet.

    Duplicated from `train_parallel._cpa_miss_m` rather than imported: that module lives in
    `controllers/rl`, and the env package must not depend on the trainer. Keep the two in step.
    """
    r = np.asarray(pos_op, float) - np.asarray(pos_me, float)
    v = np.asarray(vel_op, float) - np.asarray(vel_me, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-9 else max(0.0, -float(r @ v) / vv)
    return float(np.linalg.norm(r + v * t))


# ======================================================================================
# Trace
# ======================================================================================
@dataclass
class Sample:
    """One tick of a fight, reduced to exactly the scalars the detectors read."""

    t: float
    chi: float  # out-of-plane angle (rad); 0 when `chi_valid` is False
    chi_valid: bool
    gamma: float  # FLIGHT-PATH angle (rad) -- never the pitch attitude
    nz: float
    phi: float  # BANK angle (rad). The STATE the roll-rate command integrates to -- see `commitment`
    alt: float  # altitude (m). Read ONLY to price a hold -- see `commitment`'s dalt statistics
    ail: float  # flown a[0], [-1, 1] (+ = roll right). A ROLL-RATE command, NOT a deflection
    rng: float
    ata: float
    ata_az: float
    aspect: float
    closure: float
    cpa_miss: float
    es_rel: float  # Es(me) - Es(bandit), metres
    lay_miss_m: float
    lethal: bool  # a shot taken NOW would be inside both the lay and the real gun envelope


def make_sample(t: float, me: AState, bandit: AState, action, rel: RelGeom | None = None) -> Sample:
    """Build one `Sample`. `action` is the flown 4- or 5-vector; only aileron [0] is read.

    `lethal` is the RANGE-GATED lay test (`lay_miss_m < HIT_RADIUS_M` AND `rng <= LETHAL_RANGE_M`),
    i.e. `tLeth`, not `tHit`. The un-gated version inflates: the policy's median firing range is
    ~1519 m, where p_hit per round is 0.000, so a "good lay" there is worth nothing and counting it
    would let a policy score D5 lift for shots that cannot connect.
    """
    rel = rel if rel is not None else relative(me, bandit)
    chi, valid = out_of_plane_angle(me, bandit)
    speed = float(np.linalg.norm(np.asarray(me.vel, float)))
    gamma = math.asin(float(np.clip(me.vel[2] / speed, -1.0, 1.0))) if speed > 1e-6 else 0.0
    return Sample(
        t=float(t),
        chi=chi,
        chi_valid=valid,
        gamma=gamma,
        nz=float(me.nz),
        phi=float(me.phi),
        alt=float(me.pos[2]),
        ail=float(np.asarray(action, float).reshape(-1)[0]),
        rng=float(rel.rng),
        ata=float(rel.ata),
        ata_az=float(rel.ata_az),
        aspect=float(rel.aspect),
        closure=float(rel.closure),
        cpa_miss=cpa_miss_m(me.pos, me.vel, bandit.pos, bandit.vel),
        es_rel=specific_energy(me) - specific_energy(bandit),
        lay_miss_m=float(rel.lay_miss_m),
        lethal=bool(rel.lay_miss_m < C.HIT_RADIUS_M and rel.rng <= LETHAL_RANGE_M),
    )


@dataclass
class Event:
    """A detected maneuver. `i0/i1` index the trace; the deltas are entry->exit."""

    kind: str  # "displacement_roll" | "high_yoyo" | "low_yoyo" | "turning_room"
    i0: int
    i1: int
    t0: float
    t1: float
    d_aspect_deg: float
    d_rng_m: float
    d_es_m: float
    detail: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0


# ======================================================================================
# Internals
# ======================================================================================
def _runs(mask) -> list[tuple[int, int]]:
    """All maximal inclusive index ranges where `mask` is True."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _deg(x: float) -> float:
    return math.degrees(x)


def _event_from(tr: list[Sample], kind: str, i0: int, i1: int, **detail) -> Event:
    return Event(
        kind=kind,
        i0=i0,
        i1=i1,
        t0=tr[i0].t,
        t1=tr[i1].t,
        d_aspect_deg=_deg(tr[i1].aspect - tr[i0].aspect),
        d_rng_m=tr[i1].rng - tr[i0].rng,
        d_es_m=tr[i1].es_rel - tr[i0].es_rel,
        detail=detail,
    )


# ======================================================================================
# D1 -- displacement roll
# ======================================================================================
def detect_displacement_rolls(tr: list[Sample]) -> list[Event]:
    """Rolled the lift vector AWAY from the LOS under load, then back, and gained angles.

    All clauses are load-bearing:
      (a) `abs(chi)` was in-plane recently, then exceeds `CHI_OUT_DEG` and HOLDS for
          `CHI_MIN_HOLD_S` -- an instant is noise, and without the in-plane lookback a jet that is
          simply never pointed at the bandit would score continuously.
      (b) `nz > NZ_LOADED` throughout the hold. This is the clause that makes the detector immune to
          the passivity failure mode this program keeps hitting: a wings-level drift with the lift
          vector nowhere near the target is not a maneuver.
      (c) closing across the hold -- a displacement roll is an attacker's tool. **Repair 4**: the
          MEAN over the hold against a `-CLOSURE_TOL_MS` floor, not the single entry tick against
          zero; a real 1.333 s / 144.6 deg / 5.25 g displacement was being discarded by a -7 m/s
          reading on a quantity that noisy.
      (d) `abs(chi)` returns under `CHI_RETURN_DEG` within `CHI_RETURN_S`. Rolling out and NOT
          coming back is a departure/disengage, which this program has mistaken for a maneuver
          before. **Repair 1**: the returning tick must have `chi_valid` (else the `chi = 0.0`
          sentinel for "bandit dead ahead" satisfies it) and `ata <= D1_EXIT_ATA_MAX_DEG` (else a
          small |chi| measured against a LOS pointing over the shoulder satisfies it). Search
          continues past a tick that fails these, so this narrows WHICH tick counts as the return
          rather than aborting on the first near-miss.
      (e) `sign(ata_az)` CONSTANT across the event. A flip means the bandit crossed the nose, i.e.
          an overshoot or a reversal -- a different maneuver with a different meaning.
      (f) the angle carrier improves by its bar. **Repair 3**: carried by **ata** (the angle off my
          own nose) rather than aspect (the angle off the target's tail), because a head-on pass
          supplies aspect for free and DEGRADES ata. `D1_ANGLE_CARRIER` selects; both deltas are
          always recorded in `detail` so a rejection can be attributed either way.
      (g) **Repair 2**, new: range non-increasing at the (d) exit. A pass always exits on the
          opening side of its own minimum. Implemented on the exit tick's closure rather than on
          `d_rng_m` over the whole event, because `d_rng_m` spans the hold AND the return and so
          mixes a legitimate mid-manoeuvre range excursion with the pass signature; `d_rng_m` is on
          the `Event` regardless, so the alternative reading stays auditable.

    ⚠ The clause that REJECTS is recorded on every candidate, in `detail["reject"]` of nothing —
    rather, rejections are not returned at all (unlike D2's `failed_*` tags). That asymmetry is
    deliberate and is why `_diag_f7_detector_repair.py` re-scores the same traces under every
    switch configuration instead of reading a single count.
    """
    if len(tr) < 3:
        return []
    hold = [(abs(s.chi) > math.radians(CHI_OUT_DEG) and s.nz > NZ_LOADED) for s in tr]
    events: list[Event] = []
    for i, j in _runs(hold):
        if tr[j].t - tr[i].t < CHI_MIN_HOLD_S:
            continue
        # (a) in-plane within the lookback, i.e. it genuinely DISPLACED from a pull
        if not any(
            abs(tr[k].chi) < math.radians(CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= CHI_LOOKBACK_S
        ):
            continue
        # (c) closing, as the mean over the hold with a tolerance (repair 4)
        clo_hold = float(np.mean([tr[k].closure for k in range(i, j + 1)]))
        clo_test, clo_floor = (
            (clo_hold, -CLOSURE_TOL_MS) if D1_CLOSURE_OVER_HOLD else (tr[i].closure, 0.0)
        )
        if clo_test <= clo_floor:
            continue
        # (d) rolled back in -- on a VALID chi, with the bandit still in front (repair 1)
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > CHI_RETURN_S:
                break
            if abs(tr[k].chi) >= math.radians(CHI_RETURN_DEG):
                continue
            if D1_REQUIRE_FORWARD_EXIT and not (
                tr[k].chi_valid and tr[k].ata <= math.radians(D1_EXIT_ATA_MAX_DEG)
            ):
                continue
            exit_idx = k
            break
        if exit_idx is None:
            continue
        # (g) the range must not be opening at the exit (repair 2)
        if D1_REQUIRE_NONOPENING and tr[exit_idx].closure <= -CLOSURE_TOL_MS:
            continue
        # (e) same side throughout
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
        if len(sides) != 1:
            continue
        d_ata_deg = _deg(tr[exit_idx].ata - tr[i].ata)
        ev = _event_from(
            tr,
            "displacement_roll",
            i,
            exit_idx,
            chi_peak_deg=max(abs(_deg(tr[k].chi)) for k in range(i, j + 1)),
            nz_peak=max(tr[k].nz for k in range(i, j + 1)),
            hold_s=tr[j].t - tr[i].t,
            # Everything the four repair steps test, recorded so a count is attributable.
            d_ata_deg=d_ata_deg,
            ata_entry_deg=_deg(tr[i].ata),
            ata_exit_deg=_deg(tr[exit_idx].ata),
            # False here means the (d) exit was the `chi = 0.0` SENTINEL, i.e. "bandit dead ahead"
            # scored as "rolled back in" — the exact route the D1-exists fly-by took.
            exit_chi_valid=bool(tr[exit_idx].chi_valid),
            closure_hold=clo_hold,
            closure_entry=tr[i].closure,
            closure_exit=tr[exit_idx].closure,
            rng_min_m=min(tr[k].rng for k in range(i, exit_idx + 1)),
        )
        # (f) it must have bought angles, in whichever currency the carrier names
        gained_aspect = ev.d_aspect_deg <= -ASPECT_GAIN_DEG
        gained_ata = d_ata_deg <= -ATA_GAIN_DEG
        if D1_ANGLE_CARRIER == "aspect":
            gained = gained_aspect
        elif D1_ANGLE_CARRIER == "ata":
            gained = gained_ata
        elif D1_ANGLE_CARRIER == "both":
            gained = gained_aspect and gained_ata
        else:
            raise ValueError(f"D1_ANGLE_CARRIER must be aspect|ata|both, got {D1_ANGLE_CARRIER!r}")
        if not gained:
            continue
        events.append(ev)
    return events


# ======================================================================================
# D2 -- yo-yos, and the WHEN detector
# ======================================================================================
def detect_yoyos(tr: list[Sample]) -> list[Event]:
    """Traded a genuine overshoot for turning room out of plane, then came back on the LOS.

    The entry gate IS the timing question: a yo-yo is only correct when closure is actually
    excessive (`closure > CLOSURE_OVERSHOOT`), so an event count already encodes "did it fire at the
    right moment" in a way a pure shape test would not.

    Clause (e) -- `aspect_exit <= aspect_entry + YOYO_ASPECT_SLIP_DEG` -- is the **c70 failure
    detector**, and it is the reason this detector exists. The recorded c70 pathology (mean ata
    86 deg -> 158 deg, damage dealt 27 -> 0) is exactly "the UP branch fired and (e) failed": the
    policy learned to zoom AWAY and call it a yo-yo. Any future run that raises D2 counts while (e)
    fails has learned to leave the fight. Failures are returned too, tagged
    `detail["failed_exit"]`, precisely so that regression is visible rather than silently filtered.

    Classified `high_yoyo` when the flight-path excursion is UP (trade closure for altitude and
    turning room) and `low_yoyo` when it is DOWN (trade altitude for the corner).
    """
    events: list[Event] = []
    n = len(tr)
    if n < 3:
        return []
    swing = math.radians(YOYO_GAMMA_SWING_DEG)
    excur = math.radians(YOYO_GAMMA_EXCURSION_DEG)
    ret = math.radians(YOYO_GAMMA_RETURN_DEG)

    i = 0
    while i < n:
        # entry: closure crosses INTO the overshoot band (rising edge, so a long hot pass yields
        # one event, not one per tick)
        if not (tr[i].closure > CLOSURE_OVERSHOOT and (i == 0 or tr[i - 1].closure <= CLOSURE_OVERSHOOT)):
            i += 1
            continue
        g0 = tr[i].gamma
        window = [k for k in range(i, n) if tr[k].t - tr[i].t <= YOYO_RETURN_S]
        if len(window) < 3:
            break
        peak = max(window, key=lambda k: abs(tr[k].gamma - g0))
        if abs(tr[peak].gamma - g0) < swing or abs(tr[peak].gamma) < excur:
            i += 1
            continue
        loaded = [tr[k].nz > NZ_LOADED for k in range(i, peak + 1)]
        if not loaded or sum(loaded) / len(loaded) < LOADED_FRAC:
            i += 1  # a zoom-coast, not a pull
            continue
        exit_idx = None
        for k in range(peak + 1, n):
            if tr[k].t - tr[i].t > YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < ret:
                exit_idx = k
                break
        if exit_idx is None:
            i = peak + 1
            continue
        kind = "high_yoyo" if tr[peak].gamma > g0 else "low_yoyo"
        bled = tr[exit_idx].closure < CLOSURE_OVERSHOOT and tr[exit_idx].rng < YOYO_EXIT_RNG_M
        kept = _deg(tr[exit_idx].aspect - tr[i].aspect) <= YOYO_ASPECT_SLIP_DEG
        events.append(
            _event_from(
                tr,
                kind,
                i,
                exit_idx,
                gamma_peak_deg=_deg(tr[peak].gamma),
                gamma_swing_deg=_deg(abs(tr[peak].gamma - g0)),
                closure_entry=tr[i].closure,
                closure_exit=tr[exit_idx].closure,
                loaded_frac=sum(loaded) / len(loaded),
                failed_exit=not bled,  # (d) did not bleed the closure / ended out of range
                failed_aspect=not kept,  # (e) the c70 zoom-away signature
                clean=bool(bled and kept),
            )
        )
        i = exit_idx + 1
    return events


# ======================================================================================
# D3 -- turning-room build
# ======================================================================================
def detect_turning_room(tr: list[Sample]) -> list[Event]:
    """Held the lift vector out of plane on one side while OPENING the predicted CPA, then closed it.

    This is the signature of the only arm in this program's history that ever converted a neutral
    merge (it reached aspect 14 deg with a 1 m predicted lay), so it is the reference SHAPE for the
    merge problem: deliberately give up instantaneous closure to buy the room to turn in, then cash
    it. `cpa_miss` rising during the hold and falling after is what distinguishes it from simply
    flying away -- flying away raises the CPA and never brings it back.
    """
    if len(tr) < 3:
        return []
    hold = [(abs(s.chi) > math.radians(CHI_ROOM_DEG) and s.nz > NZ_LOADED) for s in tr]
    events: list[Event] = []
    for i, j in _runs(hold):
        if tr[j].t - tr[i].t < CHI_MIN_HOLD_S:
            continue
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        if len(sides) != 1:
            continue
        if tr[j].cpa_miss <= tr[i].cpa_miss:  # room must have been BUILT
            continue
        after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= CHI_RETURN_S]
        if not after or min(tr[k].cpa_miss for k in after) >= tr[j].cpa_miss:
            continue  # ...and then CASHED
        exit_idx = min(after, key=lambda k: tr[k].cpa_miss)
        events.append(
            _event_from(
                tr,
                "turning_room",
                i,
                exit_idx,
                cpa_entry_m=tr[i].cpa_miss,
                cpa_peak_m=tr[j].cpa_miss,
                cpa_exit_m=tr[exit_idx].cpa_miss,
                hold_s=tr[j].t - tr[i].t,
            )
        )
    return events


# ======================================================================================
# D4 -- commitment
# ======================================================================================
def commitment(tr: list[Sample]) -> dict:
    """The roll-commitment statistics — measured on the HELD STATE, not on the rate command.

    ⚠ CORRECTED. The original spec's D4 was "longest run with `sign(ail)` constant AND
    `abs(ail) > AIL_COMMIT`", on the assumption that `a[0]` is an aileron DEFLECTION. It is not.
    `Control::Raw` writes `a[0]` to `fcs/aileron-cmd-norm`, and with `fcs/fbw-override` at its
    initial value of 0 — nothing in this repo ever sets it — `f16.xml:364-396` feeds that straight
    into a roll-rate PID (`roll-trim-error = aileron-cmd-norm - roll-rate-norm`). So `a[0]` is a
    normalized **roll-RATE command**: `[[f16-raw-is-flcs-nrt]]`, and `_diag_turnperf.py` steers it as
    `a = 2.0*(bank - phi)` for exactly that reason.

    On a rate interface the command shape of a displacement roll is a PULSE to set the bank followed
    by `a[0] ~ 0` to HOLD it. Holding `abs(a[0]) > 0.5` for 1.5 s does not commit to a roll — it
    commands a sustained high roll rate, i.e. it keeps rolling, which is an aileron roll (a spin
    about the velocity vector), not a displacement. The old statistic was therefore not merely
    uninformative but ANTI-informative: maximizing it means departing the manoeuvre. Measured on
    c95 it read 0.17 s, which would have been reported as "the policy cannot commit to a roll" when
    it is approximately what correct rate-command flying looks like.

    So D4 now reports the state:

      * `longest_bank_hold_s`  -- longest run with `sign(phi)` constant and `abs(phi) >
        BANK_COMMIT_DEG`. The pilot-legible commitment measure, and the same quantity the leadturn
        diagnostic arm used when it recorded "committed correctly -- roll to 84 deg, 6.9 g".
      * `longest_chi_hold_s`   -- longest run with `sign(chi)` constant and `abs(chi) >
        CHI_ROOM_DEG`, over VALID ticks only. The BFM-relevant one: it is `chi`, the program's
        central scalar, held displaced to ONE side, which is what D1/D3 are predicates over. A jet
        can be steeply banked and still in-plane (a flat pull in a bank), so bank alone is not it.

    Both holds additionally require `nz > NZ_LOADED`, the same load clause D1/D2/D3 carry, and for
    the same reason: without it a hold cannot tell a committed turn from an unloaded wings-over
    descent. Shipping them ungated read c87 at a 44.6 s MEAN bank hold and 13.0 s MEAN `chi` hold --
    higher than c95 on every out-of-plane statistic -- while c87 decked 5/5 with zero lethal ticks.
    That is a D6 artifact, not commitment.

    And because a loaded hold can still be a high-g spiral into the ground, each hold reports the
    altitude it SPENT (`bank_hold_dalt_m`, `chi_hold_dalt_m`). Duration alone prices a manoeuvre at
    zero; on this map altitude is the currency BFM is bought with.

    The rate-command statistics are kept, renamed to say what they are (`longest_rate_cmd_run_s`,
    `max_abs_window_mean_ail`), because they still diagnose stick DITHER — but on this interface a
    LOW value is correct, and neither is evidence about commitment. Both halves of the old
    sign-and-magnitude test are retained in them: a one-sided `ail > 0.5` cannot see left rolls at
    all, while `abs(ail) > 0.5` without sign-constancy counts a jet sawing the stick as sustained.
    That original critique was right about the statistic and wrong about the quantity.

    Also returns the `chi` histogram — the cheapest single picture of whether a policy manoeuvres.
    """
    if not tr:
        return dict(
            longest_bank_hold_s=0.0,
            longest_chi_hold_s=0.0,
            bank_hold_dalt_m=0.0,
            chi_hold_dalt_m=0.0,
            bank_max_deg=0.0,
            longest_rate_cmd_run_s=0.0,
            roll_runs=0,
            max_abs_window_mean_ail=0.0,
            chi_hist={},
            chi_abs_median_deg=0.0,
            valid_frac=0.0,
        )
    best, runs = 0.0, 0
    i, n = 0, len(tr)
    while i < n:
        if abs(tr[i].ail) > AIL_COMMIT:
            s = 1 if tr[i].ail > 0 else -1
            j = i
            while j + 1 < n and abs(tr[j + 1].ail) > AIL_COMMIT and (1 if tr[j + 1].ail > 0 else -1) == s:
                j += 1
            best = max(best, tr[j].t - tr[i].t)
            runs += 1
            i = j + 1
        else:
            i += 1

    # Only FULL-length windows count. A truncated tail window is the difference between "held one-way
    # aileron for 1.5 s" and "the last 4 ticks of the episode happened to agree", and the truncated
    # form scores the latter a perfect 0.9 — it would report a sawing stick as fully committed.
    max_wm = 0.0
    span = tr[-1].t - tr[0].t
    starts = range(n) if span >= COMMIT_WINDOW_S else [0]
    for a in starts:
        if span >= COMMIT_WINDOW_S and tr[-1].t - tr[a].t < COMMIT_WINDOW_S:
            break
        w = [tr[k].ail for k in range(a, n) if tr[k].t - tr[a].t <= COMMIT_WINDOW_S]
        if len(w) > 1:
            max_wm = max(max_wm, abs(float(np.mean(w))))

    # The two STATE holds — the corrected D4. Same sign-constancy-plus-magnitude structure as above,
    # applied to the quantity the rate command integrates to. `chi` additionally requires
    # `chi_valid`: an invalid tick means the LOS is along the velocity vector and `chi` was defaulted
    # to 0.0, so treating it as "in plane" would silently break a hold that merely passed through a
    # target-dead-ahead moment, and treating it as continuous would extend a hold across a gap where
    # out-of-plane-ness is undefined. Breaking the run is the conservative reading.
    def _ok(s: Sample, value, bar: float, need_valid: bool) -> bool:
        if need_valid and not s.chi_valid:
            return False
        return abs(value(s)) > bar and s.nz > NZ_LOADED

    def _longest_hold(value, bar: float, need_valid: bool) -> tuple[float, int, int]:
        held, b0, b1 = 0.0, -1, -1
        i = 0
        while i < n:
            if not _ok(tr[i], value, bar, need_valid):
                i += 1
                continue
            s = 1 if value(tr[i]) > 0 else -1
            j = i
            while j + 1 < n:
                nxt = tr[j + 1]
                if not _ok(nxt, value, bar, need_valid):
                    break
                if (1 if value(nxt) > 0 else -1) != s:
                    break
                j += 1
            if tr[j].t - tr[i].t > held:
                held, b0, b1 = tr[j].t - tr[i].t, i, j
            i = j + 1
        return held, b0, b1

    bank_hold, bi, bj = _longest_hold(lambda s: s.phi, math.radians(BANK_COMMIT_DEG), False)
    chi_hold, ci, cj = _longest_hold(lambda s: s.chi, math.radians(CHI_ROOM_DEG), True)
    # What the hold COST. A committed BFM turn trades a few hundred metres for angles; a descending
    # spiral trades everything and arrives at the ground. Without this, the two are the same number.
    bank_dalt = (tr[bj].alt - tr[bi].alt) if bi >= 0 else 0.0
    chi_dalt = (tr[cj].alt - tr[ci].alt) if ci >= 0 else 0.0

    valid = [abs(_deg(s.chi)) for s in tr if s.chi_valid]
    edges = [0, 20, 45, 60, 90, 135, 180]
    hist = {
        f"{lo}-{hi}": sum(1 for v in valid if lo <= v < hi) for lo, hi in zip(edges[:-1], edges[1:])
    }
    return dict(
        longest_bank_hold_s=bank_hold,
        longest_chi_hold_s=chi_hold,
        bank_hold_dalt_m=bank_dalt,
        chi_hold_dalt_m=chi_dalt,
        bank_max_deg=max(abs(_deg(s.phi)) for s in tr),
        longest_rate_cmd_run_s=best,
        roll_runs=runs,
        max_abs_window_mean_ail=max_wm,
        chi_hist=hist,
        chi_abs_median_deg=float(np.median(valid)) if valid else 0.0,
        valid_frac=len(valid) / n,
    )


# ======================================================================================
# D5 -- the number that separates BFM from luck
# ======================================================================================
def lethality_lift(tr: list[Sample], events: list[Event]) -> dict:
    """`P(lethal shot within D5_WINDOW_S of an event) / P(lethal shot | no event)`.

    THE headline statistic, and the only one that speaks to WHEN. Event counts measure shape and are
    gameable -- a policy that rolls out of plane at random raises D1/D2 counts with a ratio of 1.0.
    Only a policy that manoeuvres when it PAYS moves this number above 1.

    Both base rates are returned rather than just the ratio, because the ratio alone is
    uninterpretable: 3/3 over 0/40 and 30/30 over 1/40 are very different evidence, and a ratio
    computed from two lethal ticks total should not be reported as a result. `n_event_windows` and
    `n_base_windows` are returned for the same reason.

    A baseline window must be FULLY event-free and FULL LENGTH — not merely start on an uncovered
    tick. Allowing a baseline window to run into an event's window would let a manoeuvre's own lethal
    ticks count as background and silently pull the ratio toward 1; allowing truncation would let a
    short window's lower hit chance pull the base rate down and inflate it. Event windows ARE
    truncated at the end of the episode, deliberately and asymmetrically: an event that ends the
    fight is exactly the event we want to credit, and the bias from truncating them runs the
    conservative way (a short window is less likely to contain a lethal tick, so `p_event` is if
    anything understated).
    """
    n = len(tr)
    out = dict(
        p_event=float("nan"),
        p_base=float("nan"),
        ratio=float("nan"),
        n_event_windows=0,
        n_base_windows=0,
        event_hits=0,
        base_hits=0,
    )
    if n == 0:
        return out

    def _hit(start: int) -> bool:
        return any(tr[k].lethal for k in range(start, n) if tr[k].t - tr[start].t <= D5_WINDOW_S)

    covered = set()
    for ev in events:
        for k in range(ev.i0, n):
            if tr[k].t - tr[ev.i0].t > D5_WINDOW_S:
                break
            covered.add(k)

    ev_hits = sum(1 for ev in events if _hit(ev.i0))

    def _clean_window(start: int) -> bool:
        if tr[-1].t - tr[start].t < D5_WINDOW_S:
            return False  # full length only
        return not any(
            k in covered for k in range(start, n) if tr[k].t - tr[start].t <= D5_WINDOW_S
        )

    base_starts, last_t = [], None
    for k in range(n):
        if not _clean_window(k):
            continue
        if last_t is None or tr[k].t - last_t >= D5_BASELINE_STRIDE_S:
            base_starts.append(k)
            last_t = tr[k].t
    base_hits = sum(1 for k in base_starts if _hit(k))

    out.update(
        n_event_windows=len(events),
        n_base_windows=len(base_starts),
        event_hits=ev_hits,
        base_hits=base_hits,
    )
    if events:
        out["p_event"] = ev_hits / len(events)
    if base_starts:
        out["p_base"] = base_hits / len(base_starts)
    if events and base_starts and base_hits > 0:
        out["ratio"] = out["p_event"] / out["p_base"]
    elif events and base_starts and base_hits == 0 and ev_hits > 0:
        out["ratio"] = float("inf")  # honest: undefined-but-favourable, not a large finite number
    return out


# ======================================================================================
# Aggregate
# ======================================================================================
def detect_all(tr: list[Sample]) -> dict:
    """Every detector over one episode, plus D4/D5. `events` is D1+D2+D3 in trace order."""
    d1 = detect_displacement_rolls(tr)
    d2 = detect_yoyos(tr)
    d3 = detect_turning_room(tr)
    events = sorted(d1 + d2 + d3, key=lambda e: e.i0)
    return dict(
        ticks=len(tr),
        duration_s=(tr[-1].t - tr[0].t) if tr else 0.0,
        displacement_rolls=d1,
        yoyos=d2,
        yoyos_clean=[e for e in d2 if e.detail.get("clean")],
        turning_room=d3,
        events=events,
        commitment=commitment(tr),
        lethality=lethality_lift(tr, events),
        lethal_ticks=sum(1 for s in tr if s.lethal),
    )


def summarize(episodes: list[list[Sample]]) -> dict:
    """Pool `detect_all` across episodes. Rates are per episode; D5 is pooled over all windows.

    D5 is recomputed from POOLED window counts rather than averaged over per-episode ratios: most
    episodes contain zero lethal ticks, so a mean of per-episode ratios is dominated by
    undefined/degenerate cells.
    """
    per = [detect_all(tr) for tr in episodes]
    n = max(1, len(per))
    ev_w = sum(r["lethality"]["n_event_windows"] for r in per)
    bs_w = sum(r["lethality"]["n_base_windows"] for r in per)
    ev_h = sum(r["lethality"]["event_hits"] for r in per)
    bs_h = sum(r["lethality"]["base_hits"] for r in per)
    p_ev = (ev_h / ev_w) if ev_w else float("nan")
    p_bs = (bs_h / bs_w) if bs_w else float("nan")
    if ev_w and bs_w and bs_h > 0:
        ratio = p_ev / p_bs
    elif ev_w and bs_w and bs_h == 0 and ev_h > 0:
        ratio = float("inf")
    else:
        ratio = float("nan")
    return dict(
        episodes=len(per),
        d1_per_ep=sum(len(r["displacement_rolls"]) for r in per) / n,
        d2_per_ep=sum(len(r["yoyos"]) for r in per) / n,
        d2_clean_per_ep=sum(len(r["yoyos_clean"]) for r in per) / n,
        d2_failed_aspect=sum(
            1 for r in per for e in r["yoyos"] if e.detail.get("failed_aspect")
        ),
        d3_per_ep=sum(len(r["turning_room"]) for r in per) / n,
        longest_bank_hold_s=max((r["commitment"]["longest_bank_hold_s"] for r in per), default=0.0),
        mean_bank_hold_s=sum(r["commitment"]["longest_bank_hold_s"] for r in per) / n,
        longest_chi_hold_s=max((r["commitment"]["longest_chi_hold_s"] for r in per), default=0.0),
        mean_chi_hold_s=sum(r["commitment"]["longest_chi_hold_s"] for r in per) / n,
        bank_max_deg=max((r["commitment"]["bank_max_deg"] for r in per), default=0.0),
        # Signed and MEAN, not max: a mean over episodes is what says "every hold is paid for in
        # altitude". A max would be one bad episode and a min would hide the spiral entirely.
        mean_bank_hold_dalt_m=sum(r["commitment"]["bank_hold_dalt_m"] for r in per) / n,
        mean_chi_hold_dalt_m=sum(r["commitment"]["chi_hold_dalt_m"] for r in per) / n,
        # Kept and reported so a dithering stick is still visible, but this is the RATE command:
        # low is correct here, and it is not a commitment measure. See `commitment`.
        longest_rate_cmd_run_s=max(
            (r["commitment"]["longest_rate_cmd_run_s"] for r in per), default=0.0
        ),
        chi_abs_median_deg=float(
            np.median([r["commitment"]["chi_abs_median_deg"] for r in per])
        )
        if per
        else 0.0,
        lethal_ticks=sum(r["lethal_ticks"] for r in per),
        p_event=p_ev,
        p_base=p_bs,
        ratio=ratio,
        n_event_windows=ev_w,
        n_base_windows=bs_w,
        per_episode=per,
    )


def format_summary(s: dict) -> str:
    """One compact block, with the D5 base rates always shown next to the ratio."""
    return "\n".join(
        [
            f"episodes {s['episodes']}  lethalTicks {s['lethal_ticks']}",
            f"D1 displacement_roll/ep {s['d1_per_ep']:.2f}",
            f"D2 yoyo/ep {s['d2_per_ep']:.2f}  clean/ep {s['d2_clean_per_ep']:.2f}"
            f"  failed_aspect(c70 zoom-away) {s['d2_failed_aspect']}",
            f"D3 turning_room/ep {s['d3_per_ep']:.2f}",
            f"D4 bank hold >{BANK_COMMIT_DEG:.0f}deg nz>{NZ_LOADED:.0f}"
            f" {s['longest_bank_hold_s']:.2f} s (max)  {s['mean_bank_hold_s']:.2f} s (mean)"
            f"  dalt {s['mean_bank_hold_dalt_m']:+.0f} m  peak bank {s['bank_max_deg']:.0f} deg",
            f"D4 chi  hold >{CHI_ROOM_DEG:.0f}deg nz>{NZ_LOADED:.0f}"
            f" {s['longest_chi_hold_s']:.2f} s (max)  {s['mean_chi_hold_s']:.2f} s (mean)"
            f"  dalt {s['mean_chi_hold_dalt_m']:+.0f} m  |chi| median {s['chi_abs_median_deg']:.1f} deg"
            f"  [rate-cmd run {s['longest_rate_cmd_run_s']:.2f} s = DITHER, low is correct]",
            f"D5 P(lethal|event) {s['p_event']:.3f} [{s['n_event_windows']} w]"
            f"  P(lethal|base) {s['p_base']:.3f} [{s['n_base_windows']} w]"
            f"  ratio {s['ratio']:.2f}",
        ]
    )
