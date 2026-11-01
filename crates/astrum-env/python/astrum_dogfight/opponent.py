"""Scripted opponents, driven through the raw action interface (design doc §8.5).

The scripted bot is just another row of the `astrum_env.Env.step` action
matrix; its tactics live here in Python. Straight-and-level holds use the sim's
own fixed-wing autopilot law (`astrum_env.FixedWingAutopilot("fighter", dt)`,
one instance per aircraft, reset per episode) and emit its raw
`[ail, elev, rud, thr]`, so the bot can also fire and (at Tier 3) fly vertical
BFM the autopilot can't. Gun decisions use the sim's fire control
(`GunSolution`). Every timer is in seconds, advanced by the env's `dt`.
"""

from __future__ import annotations

from math import cos

import astrum_env

from . import constants as C
from .geometry import AState, GunSolution, aim_direction, body_bearing, relative


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _corner_limited_pull(elev_cmd: float, s: AState) -> float:
    """Ease off a pull command (negative = aft stick) to hold energy near corner
    speed and stay below the AoA soft cap (§8.5). The FLCS already prevents
    departure; this keeps the bot from bleeding to a mush by over-pulling."""
    mag = -elev_cmd
    if mag <= 0.0:
        return elev_cmd
    if s.tas < C.V_CORNER_MPS:
        mag *= _clamp(s.tas / C.V_CORNER_MPS, 0.3, 1.0)
    if s.alpha > C.ALPHA_SOFT_RAD:
        mag *= _clamp(C.ALPHA_SOFT_RAD / max(s.alpha, 1e-3), 0.3, 1.0)
    return -_clamp(mag, 0.0, 1.0)


class _Hold:
    """Heading / HAE altitude / TAS hold through the sim's fighter autopilot."""

    def __init__(self) -> None:
        self.ap: astrum_env.FixedWingAutopilot | None = None
        self.dt = 0.0
        self.target = (0.0, 0.0, 0.0)

    def reset(self, s: AState, dt: float) -> None:
        if self.ap is None or dt != self.dt:
            self.ap = astrum_env.FixedWingAutopilot("fighter", dt)
            self.dt = dt
        else:
            self.ap.reset()
        self.retrim(s)

    def retrim(self, s: AState) -> None:
        """Hold wherever `s` is now (local-level course, HAE altitude, TAS)."""
        self.target = (s.course_local(), s.alt_hae_m, s.tas)

    def act(self, s: AState) -> list:
        return list(self.ap.step(s.row, *self.target))


class Tier1Opponent:
    """Tier-1 target dummy: straight & level at its reset condition. Non-shooting."""

    def __init__(self) -> None:
        self._hold = _Hold()

    def reset(self, s: AState, dt: float) -> None:
        self._hold.reset(s, dt)

    def act(self, s: AState, target: AState, gun: GunSolution | None) -> list:
        """`[aileron, elevator, rudder, throttle, trigger]`; `target`/`gun` unused."""
        return self._hold.act(s) + [0.0]


# ---------------------------------------------------------------------------
# Tier-2: predictable defensive maneuvers (design doc §8.5)
# ---------------------------------------------------------------------------
#
# Each maneuver is a small open-loop phase machine over raw `[ail,elev,rud,thr]`,
# driven by the bot's own attitude/course with a safety cap in seconds. Once
# triggered a maneuver runs to completion — the "predictable" property of Tier 2.


class _Maneuver:
    """Base defensive maneuver: `reset(s, threat, dt)` then `step(s) -> (act4, done)`."""

    name = "maneuver"

    def reset(self, s: AState, threat: AState, dt: float) -> None:
        self.phase = "roll"
        self.course0 = s.course_local()
        self.dt = dt
        self.elapsed_s = 0.0
        self.recover_s = 0.0

    def _swept(self, s: AState) -> float:
        return abs(astrum_env.wrap_pi(s.course_local() - self.course0))

    def _recover(self, s: AState, throttle: float):
        """Shared rollout: roll wings-level, unload."""
        self.recover_s += self.dt
        return [_clamp(-2.0 * s.phi, -1.0, 1.0), -0.05, 0.0, throttle], self.recover_s > C.RECOVER_S


class SplitS(_Maneuver):
    """Roll inverted, pull through the vertical: reverses course and trades
    altitude. A disengagement / hard defensive reversal."""

    name = "split_s"

    def step(self, s: AState):
        self.elapsed_s += self.dt
        if self.elapsed_s > C.MANEUVER_MAX_S:
            return self._recover(s, 0.6)
        if self.phase == "roll":
            if cos(s.phi) < -0.9:  # within ~26° of inverted
                self.phase = "pull"
            return [1.0, -0.15, 0.0, 0.7], False
        if self.phase == "pull":
            if self._swept(s) > C.SPLITS_SWEEP_RAD:
                self.phase = "recover"
            return [0.0, -1.0, 0.0, 0.2], False
        return self._recover(s, 0.6)


class BreakTurn(_Maneuver):
    """Hard break into the attacker: high bank, corner-limited high-g turn."""

    name = "break"

    def reset(self, s: AState, threat: AState, dt: float) -> None:
        super().reset(s, threat, dt)
        self.sign = 1.0 if relative(s, threat, None).ata_az >= 0.0 else -1.0

    def step(self, s: AState):
        self.elapsed_s += self.dt
        if self.elapsed_s > C.MANEUVER_MAX_S:
            return self._recover(s, 0.9)
        if self.phase == "roll":
            if abs(s.phi) >= C.BREAK_BANK_RAD * 0.9:
                self.phase = "pull"
            return [self.sign, _corner_limited_pull(-0.25, s), 0.0, 1.0], False
        if self.phase == "pull":
            ail = _clamp(1.5 * (self.sign * C.BREAK_BANK_RAD - s.phi), -1.0, 1.0)
            if self._swept(s) > C.BREAK_SWEEP_RAD:
                self.phase = "recover"
            return [ail, _corner_limited_pull(-0.9, s), 0.0, 1.0], False
        return self._recover(s, 0.9)


_MANEUVERS = {"break": BreakTurn, "split_s": SplitS}


class Tier2Opponent:
    """Tier-2 predictable defender (§8.5): holds straight & level, then commits
    to a fixed high-g maneuver on detecting an attacker in its rear hemisphere
    (within `R_DETECT`, aft of the 3-9 line). Non-shooting.

    `maneuver` ∈ {"break", "split_s", "alternate"} ("alternate" cycles the two
    across triggers). A post-maneuver cooldown keeps detection from chattering.
    """

    def __init__(self, *, maneuver: str = "break"):
        if maneuver not in ("break", "split_s", "alternate"):
            raise ValueError(f"unknown maneuver {maneuver!r}")
        self.maneuver_kind = maneuver
        self._hold = _Hold()
        self._active: _Maneuver | None = None
        self._cooldown_s = 0.0
        self._alt_idx = 0
        self._dt = 0.0

    def reset(self, s: AState, dt: float) -> None:
        self._hold.reset(s, dt)
        self._dt = dt
        self._active = None
        self._cooldown_s = 0.0
        self._alt_idx = 0

    @property
    def maneuver(self) -> str | None:
        """The maneuver in progress (`"break"` / `"split_s"`), or None while holding."""
        return self._active.name if self._active is not None else None

    def _spawn_maneuver(self) -> _Maneuver:
        kind = self.maneuver_kind
        if kind == "alternate":
            kind = "split_s" if (self._alt_idx % 2) else "break"
            self._alt_idx += 1
        return _MANEUVERS[kind]()

    def act(self, s: AState, target: AState, gun: GunSolution | None) -> list:
        """`target` is the learner (the attacker we defend against)."""
        if self._active is not None:
            act4, done = self._active.step(s)
            if done:
                self._active = None
                self._cooldown_s = C.MANEUVER_COOLDOWN_S
                self._hold.retrim(s)
            return act4 + [0.0]

        if self._cooldown_s > 0.0:
            self._cooldown_s -= self._dt
        else:
            rg = relative(s, target, None)
            if rg.rng < C.R_DETECT and rg.ata > C.REAR_HEMI_RAD:
                self._active = self._spawn_maneuver()
                self._active.reset(s, target, self._dt)
                return self._active.step(s)[0] + [0.0]

        return self._hold.act(s) + [0.0]


# ---------------------------------------------------------------------------
# Tier-3: rule-based FSM tactician + WEZ gun trigger (design doc §8.5)
# ---------------------------------------------------------------------------
#
# A 4-state BFM FSM (+ an energy sub-state) in "godmode" (reads both aircraft's
# full state). Transitions are dead-banded and dwell-gated in seconds; a genuine
# threat preempts into DEFENSIVE at once. Guns fire whenever the sim's fire
# control says the target is in the WEZ, independent of state.

_NEUTRAL = "neutral"
_OFFENSIVE = "offensive"
_DEFENSIVE = "defensive"
_YOYO = "high_yoyo"
_ENERGY = "energy_recovery"


def wez_fire(me: AState, target: AState, gun: GunSolution | None) -> bool:
    """WEZ gate: fire iff armed with ammo, both alive, and the sim's solution
    `gun` (me→target) is in range with the boresight within `FIRE_LEAD_RAD` of
    the required aim (or a round fired now would hit outright)."""
    if gun is None or me.ammo <= 0 or not me.alive or not target.alive:
        return False
    return gun.hit or (gun.in_range and gun.aim_error_rad < C.FIRE_LEAD_RAD)


class Tier3Opponent:
    """Rule-based BFM tactician with a WEZ-gated gun (§8.5).

    States: NEUTRAL (intercept / lead), OFFENSIVE (lead pursuit + guns), HIGH_YOYO
    (bleed closure to avoid an overshoot), DEFENSIVE (hard break into the
    attacker), ENERGY_RECOVERY (unload + accelerate when a break bleeds the jet
    slow). Each control law returns raw `[ail, elev, rud, thr]`.
    """

    def __init__(self) -> None:
        self._state = _NEUTRAL
        self._dwell_s = 0.0
        self._dt = 0.0

    def reset(self, s: AState, dt: float) -> None:
        self._state = _NEUTRAL
        self._dwell_s = 0.0
        self._dt = dt

    @property
    def state(self) -> str:
        return self._state

    def _to(self, new: str) -> None:
        if new != self._state:
            self._state = new
            self._dwell_s = 0.0

    def _dwell_ok(self) -> bool:
        return self._dwell_s >= C.FSM_MIN_DWELL_S

    def _transition(self, s: AState, rg, rg_t) -> None:
        """`rg` = target seen from me; `rg_t` = me seen from the target (threat)."""
        threatened = rg_t.ata < C.THREAT_CONE_RAD and rg_t.rng < C.THREAT_RANGE
        threat_clear = rg_t.ata > C.THREAT_CONE_EXIT_RAD or rg_t.rng > C.THREAT_RANGE_EXIT
        st = self._state

        if st == _DEFENSIVE:
            if s.tas < C.V_ENERGY_LOW:
                self._to(_ENERGY)
            elif threat_clear and self._dwell_ok():
                self._to(_NEUTRAL)
            return
        if st == _ENERGY:
            if s.tas > C.V_ENERGY_OK and self._dwell_ok():
                self._to(_NEUTRAL if threat_clear else _DEFENSIVE)
            return
        if threatened:
            self._to(_DEFENSIVE)
            return

        if st == _YOYO:
            if (rg.closure < C.CLOSURE_LO or rg.rng > C.YOYO_RANGE_EXIT) and self._dwell_ok():
                self._to(_OFFENSIVE)
            return
        if st == _OFFENSIVE:
            if rg.closure > C.CLOSURE_HI and rg.rng < C.YOYO_RANGE and self._dwell_ok():
                self._to(_YOYO)
            elif (rg.ata > C.OFF_ATA_EXIT_RAD or rg.rng > C.OFF_RANGE_EXIT) and self._dwell_ok():
                self._to(_NEUTRAL)
            return
        if rg.ata < C.OFF_ATA_ENTER_RAD and rg.rng < C.OFF_RANGE and self._dwell_ok():
            self._to(_OFFENSIVE)

    def _roll_onto(self, s: AState, direction):
        off, bank_err, _ = body_bearing(s, direction)
        gate = min(off / C.PURSUE_ROLL_DEADBAND, 1.0)  # don't roll once on the nose
        ail = _clamp(C.K_PURSUE_ROLL * bank_err * gate - C.K_ROLL_DAMP * s.p, -1.0, 1.0)
        return ail, off, bank_err

    def _pursue(self, s: AState, target: AState, gun: GunSolution | None):
        """Lead pursuit on the sim's aim direction: lift vector on it, pull onto
        it, hold pursuit speed."""
        ail, off, bank_err = self._roll_onto(s, aim_direction(s, target, gun))
        pull = _clamp(-C.K_PURSUE_PULL * off * max(cos(bank_err), 0.0), -1.0, 0.0)
        thr = _clamp(C.PURSUIT_THR_TRIM + C.K_PURSUIT_THR * (C.V_PURSUIT - s.tas), 0.0, 1.0)
        return [ail, _corner_limited_pull(pull, s), 0.0, thr]

    def _break(self, s: AState, target: AState):
        """Hard defensive break into the attacker (continuous)."""
        sign = 1.0 if relative(s, target, None).ata_az >= 0.0 else -1.0
        bank_err = sign * C.BREAK_BANK_RAD - s.phi
        ail = _clamp(C.K_BREAK_ROLL * bank_err - C.K_ROLL_DAMP * s.p, -1.0, 1.0)
        return [ail, _corner_limited_pull(-0.95, s), 0.0, 1.0]

    def _high_yoyo(self, s: AState, target: AState, gun: GunSolution | None):
        """Pitch up out of plane at low power to dump closure and avoid overshoot."""
        ail, _, _ = self._roll_onto(s, aim_direction(s, target, gun))
        return [ail, _corner_limited_pull(-0.5, s), 0.0, 0.2]

    def _energy_recover(self, s: AState):
        """Wings level, slight unload, full throttle."""
        return [_clamp(-2.0 * s.phi - C.K_ROLL_DAMP * s.p, -1.0, 1.0), 0.12, 0.0, 1.0]

    def act(self, s: AState, target: AState, gun: GunSolution | None) -> list:
        """`target` is the learner; `gun` the sim's solution me→target."""
        self._dwell_s += self._dt
        self._transition(s, relative(s, target, gun), relative(target, s, None))

        st = self._state
        if st == _DEFENSIVE:
            act4 = self._break(s, target)
        elif st == _ENERGY:
            act4 = self._energy_recover(s)
        elif st == _YOYO:
            act4 = self._high_yoyo(s, target, gun)
        else:  # NEUTRAL and OFFENSIVE both fly lead pursuit
            act4 = self._pursue(s, target, gun)
        return act4 + [1.0 if wez_fire(s, target, gun) else 0.0]
