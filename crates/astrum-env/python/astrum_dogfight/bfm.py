"""BFM (Basic Fighter Maneuver) primitive library — Phase 1 of the head-on-merge program.

The pure-pursuit dense-reward approach (c33–c61) topped out: from a HEAD-ON merge the learner
either decks (over-pulls, nose slices low into a descending spiral) or goes passive. The user's
thesis is that a real fighter WINS the merge with BFM — vertical reversals, yo-yos, rolls — to
convert a nose-to-nose pass into a rear-quarter kill, and the agent never learned it because
training never taught or tested it. This module builds each maneuver as a CLOSED-LOOP stick
controller (a small attitude/geometry state machine, NOT a timed open-loop sequence) so they can
be (1) flight-validated in the sim, (2) assembled into a doctrine COACH, and later (3) cloned into
the policy + RL-refined on an OUTCOME reward.

ACTION CONVENTION (F-16 raw FLCS interface; matches `train_parallel._vertical_egg_stick`):
each maneuver's `.step()` returns a len-4 `np.float32` array ``[ail, elev, rud, thr]`` — the fire
trigger [4] is NOT included; the `BFMCoach` adds it.
  * ail  : aileron / roll-rate command, +1 = roll RIGHT (right wing down).
  * elev : elevator, NEGATIVE = pull / nose-UP / +Nz  (inverted-sign FLCS). elev = -1 is max pull.
  * rud  : rudder, 0 (the FLCS coordinates turns itself).
  * thr  : throttle, +1 = full.
All four are clamped to [-1, 1].

STATE (`geometry.AState` `me`, `geometry.RelGeom` `rel = relative(me, bandit)`):
  me.phi   bank (rad, + = right wing down)      rel.rng      range (m)
  me.theta pitch (rad, + = nose up)             rel.ata      nose-to-LOS (rad, 0 = nose-on)
  me.psi   true heading (rad)                   rel.ata_az   LOS bearing off nose (rad, right +)
  me.vel   ENU velocity (m/s)                   rel.ata_el   LOS elevation off nose (rad, up +)
  me.airspeed (m/s)                             rel.aspect   angle at bandit tail→LOS (0 = his six)
  me.alt   MSL altitude (m)                     rel.closure  +CLOSING / -OPENING (m/s)  ← verified
                                                rel.gun_quality  ∈[0,1]
Each primitive keys its phase transitions off ATTITUDE / GEOMETRY (heading reversal via the
horizontal-velocity flip, bank angle, pitch, closure sign) so it self-terminates when the maneuver
is actually complete, not after a fixed step count.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from .geometry import AState, RelGeom  # noqa: F401  (public re-export for callers/tests)

_R = math.radians

# P-controller gains, matching _vertical_egg_stick (KROLL 2.0 → full aileron past ~0.5 rad of bank
# error; KPITCH 3.0 → saturates elevator to max pull from any level/nose-low state).
KROLL = 2.0
KPITCH = 3.0

# Merge/deck tuning (coach doctrine knobs; kept here so the reward's CONVERT_* constants stay separate).
MERGE_TRIGGER_M = 900.0  # start the merge reversal once the pass is within this range (else drive in)
COACH_DECK_GUARD_ALT = 1500.0  # below this MSL, never command a nose-low pull; arrest toward a climb


def _clip1(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _roll_to(phi: float, phi_tgt: float, k: float = KROLL) -> float:
    """Aileron to drive bank `phi` → `phi_tgt` the SHORT way around the ±π wrap (+ = roll right)."""
    return _clip1(k * _wrap_pi(phi_tgt - phi))


def _pitch_to(theta: float, theta_tgt: float, k: float = KPITCH) -> float:
    """Elevator to drive pitch `theta` → `theta_tgt`. elev = -k·(tgt-theta): tgt above current ⇒
    negative ⇒ pull/nose-up (from level toward a steep target this saturates to -1 = max pull)."""
    return _clip1(-k * (theta_tgt - theta))


def _horiz_unit(vel: np.ndarray) -> np.ndarray:
    h = np.array([float(vel[0]), float(vel[1])], dtype=float)
    n = math.hypot(h[0], h[1])
    return h / n if n > 1e-6 else np.array([0.0, 0.0])


def _lift_on_target_bank_err(rel: RelGeom) -> float:
    """Roll error (rad) that would put the lift vector (canopy-up) ON the bandit — i.e. bring the
    LOS into the vertical plane above the nose so a pull drives the nose straight to him. Derived
    purely from the body-frame LOS bearing/elevation: atan2(right, up) = atan2(sin ata_az, tan
    ata_el). Target dead ahead → 0 (wings level, pure pull); target abeam-right → +90° (roll right,
    then pull); target below → toward ±180° (roll inverted, then pull). ata_el clamped off ±90°."""
    ael = max(-_R(80.0), min(_R(80.0), rel.ata_el))
    return math.atan2(math.sin(rel.ata_az), math.tan(ael))


def _point_at(rel: RelGeom, pull_gain: float = C.K_PURSUE_PULL, lag_rad: float = 0.0):
    """(ail, elev) that rolls the lift vector onto the bandit and pulls the nose to him with an
    optional `lag_rad` trail (pursue slightly behind for a stable tracking solution)."""
    ail = _clip1(C.K_PURSUE_ROLL * _lift_on_target_bank_err(rel))
    if rel.ata < C.PURSUE_ROLL_DEADBAND:
        ail = 0.0  # on boresight — stop chasing bank, just hold
    pull = max(0.0, rel.ata - lag_rad)
    elev = _clip1(-pull_gain * pull)
    return ail, elev


def _deck_guard(me: AState, ail: float, elev: float) -> tuple[float, float]:
    """Near the deck, refuse the descending-spiral death that kills the pure-pursuit chaser: if we're
    low and not already climbing, force a pull to a climb (elev < 0 = nose-up), and roll upright if
    inverted (a pull while inverted digs toward the ground)."""
    if me.alt < COACH_DECK_GUARD_ALT:
        if me.theta < _R(10.0):
            elev = min(elev, -0.55)  # ensure a climb regardless of the tracking pull
        if abs(_wrap_pi(me.phi)) > _R(90.0):
            ail = _roll_to(me.phi, 0.0)  # get upright before pulling
    return ail, elev


# ======================================================================================
# Base class
# ======================================================================================
class Maneuver:
    """Closed-loop stick primitive. Subclasses implement `step`; `reset` captures the entry frame
    (heading, altitude) used by the geometric completion tests."""

    name = "maneuver"
    max_steps = 480  # ~20 s at the 24 Hz agent rate (action_repeat 5 @ 120 Hz) — a hard safety done

    def reset(self, me: AState, rel: RelGeom) -> None:
        self._h0 = _horiz_unit(me.vel)  # entry horizontal-flight direction (for reversal detection)
        self._alt0 = float(me.alt)
        self._psi0 = float(me.psi)
        self._steps = 0

    def step(self, me: AState, rel: RelGeom):  # -> (np.ndarray[4], bool)
        raise NotImplementedError

    # -- shared completion signals --------------------------------------------------------
    def _reversed(self, me: AState, thresh_deg: float) -> bool:
        """True once the horizontal flight direction has swung `thresh_deg`+ away from entry
        (robust across the vertical: uses the velocity vector, not the Euler ψ which is ill-defined
        near vertical). Guards against the near-vertical transient where horizontal speed is tiny."""
        h = _horiz_unit(me.vel)
        if float(np.hypot(h[0], h[1])) < 0.5:  # essentially vertical — direction undefined
            return False
        align = float(np.dot(h, self._h0))
        return align < math.cos(_R(thresh_deg))

    def _capped(self) -> bool:
        return self._steps >= self.max_steps


def _sign_from(rel: RelGeom) -> float:
    """Turn direction toward the bandit: +1 roll right (he's off to the right), -1 left."""
    return 1.0 if rel.ata_az >= 0.0 else -1.0


# ======================================================================================
# 1-5: reversals (self-terminating; end wings-level on the reciprocal heading)
# ======================================================================================
class Immelmann(Maneuver):
    """Half-loop UP then roll upright: wings-level max-g pull over the top, then roll to upright on
    the reciprocal heading. Net: ~180° reversal + ALTITUDE GAIN. Trades speed for height + a nose
    onto a bandit that merged high/co-alt."""

    name = "immelmann"
    _PULL, _ROLL = 0, 1

    def reset(self, me, rel):
        super().reset(me, rel)
        self._phase = self._PULL

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._PULL:
            # Keep wings level only while still climbing toward vertical; above ~60° pitch stop
            # fighting the natural roll-to-inverted at the apex (a true loop holds zero aileron).
            ail = _roll_to(me.phi, 0.0) if abs(me.theta) < _R(60.0) else 0.0
            elev = -1.0  # max pull up and over
            if self._steps > 8 and self._reversed(me, 120.0):
                self._phase = self._ROLL
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        # ROLL: level the wings (from inverted) and hold the nose near the horizon.
        ail = _roll_to(me.phi, 0.0)
        elev = _pitch_to(me.theta, _R(0.0))
        done = abs(_wrap_pi(me.phi)) < _R(20.0)
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class SplitS(Maneuver):
    """Roll inverted then half-loop DOWN: roll to inverted, then max-g pull the nose down and
    around to upright on the reciprocal heading. Net: ~180° reversal + ALTITUDE LOSS. The fast/high
    reversal — trade height for turn + speed."""

    name = "splits"
    _INVERT, _PULL, _RECOVER = 0, 1, 2

    def reset(self, me, rel):
        super().reset(me, rel)
        self._phase = self._INVERT

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._INVERT:
            ail = _roll_to(me.phi, math.pi)  # roll to inverted (short way)
            elev = _pitch_to(me.theta, _R(0.0))  # hold pitch while rolling
            if abs(_wrap_pi(me.phi)) > _R(160.0):
                self._phase = self._PULL
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        if self._phase == self._PULL:
            ail = 0.0  # hold inverted, pure pull
            elev = -1.0  # max pull — nose slices down and around
            if self._steps > 8 and self._reversed(me, 150.0):
                self._phase = self._RECOVER
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        # RECOVER: level the wings, arrest the nose at the horizon.
        ail = _roll_to(me.phi, 0.0)
        elev = _pitch_to(me.theta, _R(0.0))
        done = abs(_wrap_pi(me.phi)) < _R(20.0)
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class VerticalEgg(Maneuver):
    """The champion-congruent nose-HIGH reversal (the c58 vertical egg): a modest ~40° bank toward
    the bandit + MAX-g pull through a canted (egg-shaped) loop — zoom up and over, come down
    reversed and pointed at him — then roll wings-level. The lift vector stays mostly vertical so it
    climbs (never decks). `sign` = roll direction."""

    name = "egg"
    _LOOP, _LEVEL = 0, 1

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._phase = self._LOOP

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._LOOP:
            # Cant the loop plane at the bandit (~40° bank) while still climbing; above ~60° pitch
            # stop fighting the natural roll-to-inverted at the apex (else the aileron destabilises
            # the loop and it never rolls out — the c58-egg's original failure).
            ail = _roll_to(me.phi, self._sign * _R(40.0)) if abs(me.theta) < _R(60.0) else 0.0
            elev = -1.0  # MAX pull through the canted loop (the egg) — not a fixed-pitch hold
            if self._steps > 8 and self._reversed(me, 120.0):
                self._phase = self._LEVEL
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        ail = _roll_to(me.phi, 0.0)
        elev = _pitch_to(me.theta, _R(0.0))
        done = abs(_wrap_pi(me.phi)) < _R(20.0)
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class Pitchback(Maneuver):
    """Nose-HIGH slice reversal, tighter than the egg: raise the nose into the vertical, then roll
    to a high bank (~110°) and MAX-g pull the nose around and back down on the reciprocal heading,
    then roll wings-level. Near-neutral energy — the tight in-plane reversal. `sign` = turn dir."""

    name = "pitchback"
    _CLIMB, _SLICE, _LEVEL = 0, 1, 2

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._phase = self._CLIMB

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._CLIMB:
            ail = _roll_to(me.phi, self._sign * _R(30.0))
            elev = -1.0  # pull the nose up into the vertical first (nose-high entry)
            if me.theta > _R(35.0):
                self._phase = self._SLICE
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        if self._phase == self._SLICE:
            ail = _roll_to(me.phi, self._sign * _R(110.0))
            elev = -1.0  # roll to high bank + max pull → slice around and over the top
            if self._steps > 6 and self._reversed(me, 150.0):
                self._phase = self._LEVEL
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        ail = _roll_to(me.phi, 0.0)
        elev = _pitch_to(me.theta, _R(0.0))
        done = abs(_wrap_pi(me.phi)) < _R(20.0)
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class Wingover(Maneuver):
    """Low-g reversal for a SLOW jet: climb (low g) to bleed airspeed, then roll ~100° over the top
    and let the nose slice down and around under a moderate pull, recover wings-level on the
    reciprocal heading. Minimal energy loss, no deck. `sign` = roll direction."""

    name = "wingover"
    _CLIMB, _OVER, _RECOVER = 0, 1, 2

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._phase = self._CLIMB

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._CLIMB:
            ail = _roll_to(me.phi, self._sign * _R(25.0))
            elev = _pitch_to(me.theta, _R(45.0))  # low-g climb to bleed speed
            if me.airspeed < 160.0 or (me.theta > _R(40.0) and self._steps > 8):
                self._phase = self._OVER
            return np.array([ail, elev, 0.0, 0.9], np.float32), self._capped()
        if self._phase == self._OVER:
            ail = _roll_to(me.phi, self._sign * _R(100.0))  # roll over the top
            elev = -0.5  # moderate pull, let the nose slice down and around
            if self._reversed(me, 130.0):
                self._phase = self._RECOVER
            return np.array([ail, elev, 0.0, 0.9], np.float32), self._capped()
        ail = _roll_to(me.phi, 0.0)
        elev = _pitch_to(me.theta, _R(0.0))
        done = abs(_wrap_pi(me.phi)) < _R(20.0) and me.theta < _R(15.0)
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


# ======================================================================================
# 6-9: pursuit modifiers (reference a LIVE bandit; end when the geometry is resolved)
# ======================================================================================
class HighYoYo(Maneuver):
    """Kill excess closure / prevent an overshoot from the rear quarter: pull UP out of plane
    (bank into the turn + nose above his plane) to trade airspeed for turning room, then come back
    DOWN onto the LOS once the closure is bled. `sign` = turn direction."""

    name = "high_yoyo"
    _UP, _DOWN = 0, 1

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._phase = self._UP

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._UP:
            ail = _roll_to(me.phi, self._sign * _R(70.0))
            elev = _pitch_to(me.theta, _R(25.0))  # nose above his plane — go high, bleed closure
            # Closure bled (no longer overshooting) OR range reopened ⇒ pull back down to reacquire.
            if rel.closure < C.CLOSURE_LO or rel.rng > C.YOYO_RANGE_EXIT:
                self._phase = self._DOWN
            return np.array([ail, elev, 0.0, 0.75], np.float32), self._capped()
        ail, elev = _point_at(rel)  # roll down onto the LOS and pull to reacquire
        done = rel.ata < _R(20.0)  # nose back on him
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class LowYoYo(Maneuver):
    """Close a gap / cut the corner when lagging: drop the nose below the LOS (hard bank + slight
    nose-low + full throttle) to accelerate and cut across the circle, then pull back up to his
    plane once closed. `sign` = turn direction."""

    name = "low_yoyo"
    _DOWN, _UP = 0, 1

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._phase = self._DOWN

    def step(self, me, rel):
        self._steps += 1
        if self._phase == self._DOWN:
            ail = _roll_to(me.phi, self._sign * _R(80.0))
            elev = _pitch_to(me.theta, _R(-10.0))  # slight nose-low — accelerate, cut the corner
            if rel.rng < C.YOYO_RANGE or rel.closure > C.CLOSURE_HI:
                self._phase = self._UP
            return np.array([ail, elev, 0.0, 1.0], np.float32), self._capped()
        ail, elev = _point_at(rel)  # pull up onto his level / the LOS
        done = rel.ata < _R(20.0) and abs(rel.rel_alt) < 150.0
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class LagDisplacementRoll(Maneuver):
    """Prevent an overshoot by momentarily rolling the lift vector to LAG (bank past pure pursuit,
    into the turn) — adds turning room without an out-of-plane zoom. A brief transient: complete
    once the lag bank is established (a bounded attitude change, per doctrine). `sign` = turn dir."""

    name = "lag_roll"
    _TGT = _R(100.0)  # roll past the pure-pursuit bank to displace behind his flight path

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._dwell = 0

    def step(self, me, rel):
        self._steps += 1
        tgt = self._sign * self._TGT
        ail = _roll_to(me.phi, tgt)
        elev = -0.5  # moderate pull while displaced
        if abs(_wrap_pi(me.phi - tgt)) < _R(15.0):
            self._dwell += 1
        done = self._dwell >= 12  # ~0.5 s holding the lag attitude → re-establish pursuit
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class BarrelRollAttack(Maneuver):
    """Bleed excess closure with a full 360° roll AROUND the flight path (constant aileron +
    moderate pull) — stay behind without an overshoot when closing too fast from the rear quarter.
    Completes after one full revolution. `sign` = roll direction."""

    name = "barrel_roll"

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)
        self._roll_acc = 0.0
        self._prev_phi = float(me.phi)

    def step(self, me, rel):
        self._steps += 1
        d = abs(_wrap_pi(float(me.phi) - self._prev_phi))
        self._roll_acc += d
        self._prev_phi = float(me.phi)
        ail = self._sign * 1.0  # full constant roll
        elev = -0.5  # moderate pull to arc around the flight path
        done = self._roll_acc > _R(350.0)
        return np.array([ail, elev, 0.0, 0.85], np.float32), done or self._capped()


# ======================================================================================
# 10: terminal tracker (never "done" — the coach fires when the solution is good)
# ======================================================================================
class LagPursuitTracker(Maneuver):
    """Terminal gun-tracking mode: roll the lift vector onto the bandit, pull the nose to him with
    a slight lag (a stable trailing solution, not a lead overshoot), and modulate throttle to hold
    closure near zero. Never self-terminates — the coach latches this once behind him and fires
    when gun_quality is high. Deck-guarded so a low tail-chase doesn't spiral into the ground."""

    name = "lag_track"

    def step(self, me, rel):
        self._steps += 1
        ail, elev = _point_at(rel, pull_gain=C.K_PURSUE_PULL, lag_rad=_R(3.0))
        ail, elev = _deck_guard(me, ail, elev)
        # Throttle to bleed excess closure (don't overrun the trail) but never coast to a stall.
        thr = max(0.25, _clip1(0.85 - max(0.0, rel.closure) / 300.0))
        return np.array([ail, elev, 0.0, thr], np.float32), False


class MergeApproach(Maneuver):
    """Drive to the merge pointed at the bandit (slight lead), mil power, and hold fire. Completes
    the instant the pass is imminent (range inside MERGE_TRIGGER_M) OR the bandit has begun to pass
    (closure gone negative) — the coach then commits to a REVERSAL to convert. This is the doctrine
    fix for the head-on trap: a straight merge is always a death, so we do NOT try to gun the face
    (which mag-dumps + flat-turn-chases to a timeout/deck); we set up the reversal instead."""

    name = "merge_approach"

    def step(self, me, rel):
        self._steps += 1
        ail, elev = _point_at(rel, lag_rad=_R(2.0))  # nose toward him, ready to lead-turn at the pass
        done = rel.rng < MERGE_TRIGGER_M or rel.closure < 0.0
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


class LeadTurn(Maneuver):
    """Sustained hard LEAD turn to gain angles on a bandit who reversed WITH us (a fixed 180° reversal
    just re-creates the head-on — the neutral rolling-fight trap). Roll to a hard ~75° bank toward him
    (a vertical component preserved so it's a CLIMBING turn, not an energy-dumping flat groove) and
    pull max-g, continuously driving his aspect down. Completes when we convert to his rear (→ tracker)
    or bleed to corner (→ caller picks a low-speed reversal). Deck-guarded. `sign` = turn direction."""

    name = "lead_turn"
    _BANK = _R(75.0)  # hard but keeps ~sin25° of lift vertical → sustains energy in the turn

    def __init__(self, sign: float | None = None):
        self._sign = sign

    def reset(self, me, rel):
        super().reset(me, rel)
        if self._sign is None:
            self._sign = _sign_from(rel)

    def step(self, me, rel):
        self._steps += 1
        ail = _roll_to(me.phi, self._sign * self._BANK)
        elev = -1.0  # max-g pull through the turn
        ail, elev = _deck_guard(me, ail, elev)
        done = rel.aspect < COACH_REAR_ASPECT or me.airspeed < COACH_V_SLOW
        return np.array([ail, elev, 0.0, 1.0], np.float32), done or self._capped()


# ======================================================================================
# Doctrine coach — first draft (thresholds to be tuned against the sim + a live T3)
# ======================================================================================
# Selection thresholds (named for tuning). Ordered doctrine tree; the head-on merge is discriminated
# by ASPECT (he's ahead / pointing at us), NOT by ata — at a head-on pass our nose is ON him (ata≈0),
# so the v1 tree's `ata>80` merge test never fired and everything fell through to pure pursuit.
COACH_FIRE_Q = 0.30  # gun_quality above which the trigger is squeezed (AND must be in his rear quarter)
COACH_FIRE_ASPECT = _R(60.0)  # only fire from his rear quarter — never spray the head-on face (mag-dump)
COACH_MERGE_ASPECT = _R(100.0)  # bandit aspect above this = he's ahead pointing at us (merge / head-on)
COACH_REAR_ASPECT = _R(60.0)  # aspect below this = we're in his rear quarter (a chase, not a merge)
COACH_TRACK_ATA = _R(35.0)  # AND nose within this of him ⇒ latch the terminal tracker
COACH_V_FAST = 235.0  # m/s; enough energy for a nose-high reversal
COACH_V_SLOW = 150.0  # m/s; too slow for a hard reversal ⇒ wingover
COACH_ALT_HIGH = 3500.0  # m MSL; room below to trade for a Split-S


class BFMCoach:
    """Holds the active primitive and re-selects when it completes. `.act(me, rel)` returns the
    full len-5 action (with the fire trigger). The doctrine tree (documented per branch) is a
    scaffold for Phase 2 — it will be tuned head-on vs T3, then cloned + RL-refined in Phase 3."""

    def __init__(self):
        self._active: Maneuver | None = None

    def _select(self, me: AState, rel: RelGeom) -> Maneuver:
        sign = _sign_from(rel)
        # (a) In his REAR QUARTER with the nose roughly on him → terminal tracker (the coach fires).
        #     Gated on aspect (behind him), NOT gun_quality alone — a head-on face can score high
        #     gun_quality yet is a merge, not a kill setup.
        if rel.aspect < COACH_REAR_ASPECT and rel.ata < COACH_TRACK_ATA:
            return LagPursuitTracker()
        # (b) Rear quarter, closing hot & close → overshoot risk → high yo-yo (or barrel if very hot).
        if rel.aspect < _R(90.0) and rel.rng < C.YOYO_RANGE and rel.closure > C.CLOSURE_HI:
            return BarrelRollAttack(sign) if rel.closure > C.CLOSURE_HI + 60.0 else HighYoYo(sign)
        # (c) Behind him but lagging / too far & low closure → low yo-yo to cut the corner and close.
        if rel.aspect < COACH_REAR_ASPECT and rel.rng > C.YOYO_RANGE_EXIT and rel.closure < C.CLOSURE_LO:
            return LowYoYo(sign)
        # (d) HIGH-ASPECT MERGE — he's ahead pointing at us (the head-on task). Do NOT try to gun the
        #     pass ("a straight merge is always a death"). Far & closing → DRIVE to the merge (no fire);
        #     at/after the pass → a vertical reversal to convert, chosen by energy. NOTE: vs an equal
        #     T3 this fight is symmetric — the coach SURVIVES + does BFM but cannot hand-code a
        #     sustained conversion (twice-confirmed: egg and lead-turn both stay neutral). The nose-high
        #     VerticalEgg is the best SURVIVOR (holds energy, never decks) and the intended RL seed;
        #     LeadTurn is retained as a primitive but bled energy without converting, so it is not the
        #     default. fast+high → Split-S (out-of-plane); slow → Wingover (low-g); else → egg.
        if rel.aspect > COACH_MERGE_ASPECT:
            if rel.rng > MERGE_TRIGGER_M and rel.closure > 0.0:
                return MergeApproach()
            if me.airspeed > COACH_V_FAST and me.alt > COACH_ALT_HIGH:
                return SplitS()
            if me.airspeed < COACH_V_SLOW:
                return Wingover(sign)
            return VerticalEgg(sign)
        # (e) Slow anywhere else → wingover to reverse without decking.
        if me.airspeed < COACH_V_SLOW:
            return Wingover(sign)
        # (f) Default (mid-aspect, not yet behind): drive toward him rather than latch a flat pursuit.
        return MergeApproach()

    def act(self, me: AState, rel: RelGeom) -> np.ndarray:
        if self._active is None:
            self._active = self._select(me, rel)
            self._active.reset(me, rel)
        a4, done = self._active.step(me, rel)
        if done:
            self._active = self._select(me, rel)
            self._active.reset(me, rel)
            a4, _ = self._active.step(me, rel)
        # Fire ONLY from his rear quarter with a real solution in range — never spray the head-on
        # face (the v1 coach dumped all 511 rounds at low-quality merge snapshots and killed nothing).
        #
        # Hold-fire is -1.0, NOT 0.0, because this action is in the AGENT [-1,1] space (see the module
        # docstring) and every consumer steps a `make_tdmpc_env`, whose `_LinearActionWrapper` affine-maps
        # the trigger to the env's native [0,1] before the Rust side gates it at `>= 0.5`. A hold-fire of
        # 0.0 therefore mapped to native 0.5 — exactly ON the gate — so the coach fired on EVERY step of
        # every episode, including the whole head-on merge it is explicitly written not to spray. That
        # inverted the one discipline the v2 coach was built to add, and it is a plausible contributor to
        # the c62/c62b decks (the buffer saw mag-dumping merge trajectories under a waste penalty).
        fire = 1.0 if (
            rel.aspect < COACH_FIRE_ASPECT and rel.gun_quality >= COACH_FIRE_Q and rel.in_range
        ) else -1.0
        return np.array([a4[0], a4[1], a4[2], a4[3], fire], np.float32)

    @property
    def active_name(self) -> str:
        return self._active.name if self._active is not None else "none"
