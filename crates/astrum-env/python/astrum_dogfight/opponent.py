"""Scripted opponents, driven through the raw action interface (design doc §8.5).

The scripted bot is just another agent id fed to `DogfightEnv.step`; all of its
logic lives here in Python. `InnerLoop` is a faithful port of the sim's own
fixed-wing autopilot law + f16 gains (`crates/astrum-sim/src/net.rs`
`fixed_wing_autopilot` / `FixedWingGains::f16`), so a Tier-1/2 "hold" behaves
exactly like the in-sim autopilot — but expressed as raw `[ail, elev, rud, thr]`
so the bot can also fire and (at Tier 3) fly vertical BFM the autopilot can't.

We drive the opponent through raw actions rather than exposing the autopilot
because that autopilot has no trigger channel and actively fights inverted /
vertical states (Split-S, yo-yos).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, radians, sin

from . import constants as C
from .geometry import AState, body_bearing, lead_point, relative


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap_pi(a: float) -> float:
    return (a + pi) % (2.0 * pi) - pi


def _course(s: AState) -> float:
    """Ground course (rad, 0=N, +E) from horizontal velocity — gimbal-free through
    the vertical, unlike Euler ψ (a Split-S flips ψ). Falls back to ψ when slow."""
    if hypot(s.vel[0], s.vel[1]) > 1.0:
        return atan2(s.vel[0], s.vel[1])  # atan2(E, N) == ψ convention
    return s.psi


def _corner_limited_pull(elev_cmd: float, s: AState, *, ease_speed: bool = True) -> float:
    """Ease off a pull command (negative = aft stick) to hold energy near corner
    speed and stay below the AoA soft cap — the §8.5 "corner speed ← Nz" note.

    The FLCS already prevents *departure* (~30° AoA / 9 g hard limits); this keeps
    the *scripted* opponent from bleeding to a mush by over-pulling when slow.

    `ease_speed=False` keeps ONLY the AoA soft-cap (departure protection) and drops
    the airspeed easing, so a *max-performance* break rides the alpha limit and bleeds
    down toward corner speed for the tightest turn — trading energy for turn rate the
    way a defender fighting for its life does, instead of preserving energy in a wide
    turn. The alpha cap still prevents a mush/departure.
    """
    mag = -elev_cmd  # pull magnitude in (0, 1]
    if mag <= 0.0:
        return elev_cmd  # not a pull — leave push/neutral untouched
    if ease_speed and s.airspeed < C.V_CORNER_MPS:
        mag *= _clamp(s.airspeed / C.V_CORNER_MPS, 0.3, 1.0)
    if s.alpha > C.ALPHA_SOFT_RAD:
        mag *= _clamp(C.ALPHA_SOFT_RAD / max(s.alpha, 1e-3), 0.3, 1.0)
    return -_clamp(mag, 0.0, 1.0)


@dataclass
class F16Gains:
    """Fighter tuning, ported verbatim from `FixedWingGains::f16()` (net.rs)."""

    max_roll_rad: float = radians(60.0)
    max_pitch_rad: float = radians(24.0)
    k_hdg_to_roll: float = 2.5
    k_roll_to_ail: float = 2.5
    k_roll_rate: float = 0.4
    k_alt_to_pitch: float = 0.02
    k_turn_comp: float = 0.14
    k_pitch_to_elev: float = 2.0
    k_pitch_rate: float = 0.5
    throttle_trim: float = 0.6
    k_spd_to_thr: float = 0.02


class InnerLoop:
    """Heading + altitude + airspeed hold → `[aileron, elevator, rudder, throttle]`.

    Sign conventions match the sim: `aileron+` rolls right (φ+), `elevator+`
    pitches down (θ−). Rudder is left at 0 (the fixed-wing law is coordinated by
    the FLCS). This is `fixed_wing_autopilot` in Python.
    """

    def __init__(self, gains: F16Gains | None = None):
        self.g = gains or F16Gains()

    def compute(self, s: AState, tgt_hdg: float, tgt_alt: float, tgt_spd: float):
        g = self.g
        # Heading → bank → aileron.
        hdg_err = _wrap_pi(tgt_hdg - s.psi)
        roll_target = _clamp(g.k_hdg_to_roll * hdg_err, -g.max_roll_rad, g.max_roll_rad)
        aileron = _clamp(g.k_roll_to_ail * (roll_target - s.phi) - g.k_roll_rate * s.p, -1.0, 1.0)

        # Altitude → pitch → elevator, with load-factor feedforward for the bank.
        load_excess = _clamp(1.0 / max(abs(cos(s.phi)), 1e-3) - 1.0, 0.0, 2.0)
        pitch_target = _clamp(
            g.k_alt_to_pitch * (tgt_alt - s.alt) + g.k_turn_comp * load_excess,
            -g.max_pitch_rad,
            g.max_pitch_rad,
        )
        elevator = _clamp(
            -g.k_pitch_to_elev * (pitch_target - s.theta) + g.k_pitch_rate * s.q, -1.0, 1.0
        )

        # Airspeed → throttle.
        throttle = _clamp(g.throttle_trim + g.k_spd_to_thr * (tgt_spd - s.airspeed), 0.0, 1.0)
        return [aileron, elevator, 0.0, throttle]


class Tier1Opponent:
    """Tier-1 target dummy: straight & level at its trimmed reset condition (§8.5).

    Captures heading / altitude / airspeed at `reset` and holds them. Non-shooting
    (the learner practices pure offense first).
    """

    def __init__(self, inner: InnerLoop | None = None):
        self.inner = inner or InnerLoop()
        self._tgt_hdg = 0.0
        self._tgt_alt = 0.0
        self._tgt_spd = 0.0

    def reset(self, s: AState) -> None:
        self._tgt_hdg = s.psi
        self._tgt_alt = s.alt
        self._tgt_spd = s.airspeed

    def act(self, s: AState, opponent: AState) -> list:
        """Return `[aileron, elevator, rudder, throttle, trigger]`. `opponent` is
        the learner's state (unused at Tier 1; higher tiers react to it)."""
        return self.inner.compute(s, self._tgt_hdg, self._tgt_alt, self._tgt_spd) + [0.0]


# ---------------------------------------------------------------------------
# Tier-2: predictable defensive maneuvers (design doc §8.5)
# ---------------------------------------------------------------------------
#
# Each maneuver is a small **open-loop phase machine** over raw `[ail,elev,rud,thr]`,
# driven by the opponent's own attitude/course (robust to dt) with a wall-clock
# safety cap. Command values for the Split-S are the ones validated end-to-end in
# `crates/astrum-sim/tests/split_s.rs` (roll inverted at max rate, pull through at
# the g-limit). Once triggered a maneuver runs to completion — no mid-maneuver
# re-evaluation — which is the "predictable / fixed" property Tier 2 is meant to be.


class _Maneuver:
    """Base defensive maneuver: `reset(s, threat, dt)` then `step(s) -> (act4, done)`."""

    name = "maneuver"

    def reset(self, s: AState, threat: AState, dt: float) -> None:
        self.phase = "roll"
        self.course0 = _course(s)
        self.ticks = 0
        self.recover_ticks = 0
        self._max_ticks = int(C.MANEUVER_MAX_S / dt)
        self._recover_target = int(C.RECOVER_S / dt)

    def _recover(self, s: AState, throttle: float):
        """Shared rollout: roll wings-level (rate command opposing bank), unload."""
        ail = _clamp(-2.0 * s.phi, -1.0, 1.0)
        self.recover_ticks += 1
        done = self.recover_ticks > self._recover_target
        return [ail, -0.05, 0.0, throttle], done


class SplitS(_Maneuver):
    """Roll inverted, pull through the vertical: reverses heading and dumps ~2 km
    of altitude (`split_s.rs`). A disengagement / hard defensive reversal."""

    name = "split_s"

    def step(self, s: AState):
        self.ticks += 1
        if self.ticks > self._max_ticks:
            return self._recover(s, 0.6)
        if self.phase == "roll":
            # Max roll-rate command to inverted; slight aft stick holds the nose.
            if cos(s.phi) < -0.9:  # within ~26° of inverted
                self.phase = "pull"
            return [1.0, -0.15, 0.0, 0.7], False
        if self.phase == "pull":
            # Full aft stick (FLCS caps the g); idle-ish throttle so the dive is bounded.
            swept = abs(_wrap_pi(_course(s) - self.course0))
            if swept > C.SPLITS_SWEEP_RAD:
                self.phase = "recover"
            return [0.0, -1.0, 0.0, 0.2], False
        return self._recover(s, 0.6)


class BreakTurn(_Maneuver):
    """Hard, level-ish break into the attacker: roll to a high bank and pull a
    corner-limited high-g turn to force an overshoot. Turns toward the threat."""

    name = "break"

    def reset(self, s: AState, threat: AState, dt: float) -> None:
        super().reset(s, threat, dt)
        # Break toward the attacker's bearing (turn into him): +1 = roll right.
        self.sign = 1.0 if relative(s, threat).ata_az >= 0.0 else -1.0

    def step(self, s: AState):
        self.ticks += 1
        if self.ticks > self._max_ticks:
            return self._recover(s, 0.9)
        if self.phase == "roll":
            # Roll to the break bank while beginning to load; full mil power.
            if abs(s.phi) >= C.BREAK_BANK_RAD * 0.9:
                self.phase = "pull"
            return [self.sign, _corner_limited_pull(-0.25, s), 0.0, 1.0], False
        if self.phase == "pull":
            # Hold the bank, pull a corner-speed-limited hard turn until swept round.
            ail = _clamp(1.5 * (self.sign * C.BREAK_BANK_RAD - s.phi), -1.0, 1.0)
            swept = abs(_wrap_pi(_course(s) - self.course0))
            if swept > C.BREAK_SWEEP_RAD:
                self.phase = "recover"
            return [ail, _corner_limited_pull(-0.9, s), 0.0, 1.0], False
        return self._recover(s, 0.9)


_MANEUVERS = {"break": BreakTurn, "split_s": SplitS}


class Tier2Opponent:
    """Tier-2 predictable defender (§8.5): holds straight & level like Tier 1, then
    commits to a fixed high-g maneuver on detecting an attacker in its rear
    hemisphere (within `R_DETECT`, aft of the 3-9 line). Non-shooting.

    `maneuver` ∈ {"break", "split_s", "alternate"}. "alternate" cycles the two
    across successive triggers. A post-maneuver cooldown gives the dwell that keeps
    detection from chattering at the threshold.
    """

    def __init__(
        self,
        inner: InnerLoop | None = None,
        *,
        maneuver: str = "break",
        dt: float = 1.0 / 120.0,
    ):
        if maneuver not in ("break", "split_s", "alternate"):
            raise ValueError(f"unknown maneuver {maneuver!r}")
        self.inner = inner or InnerLoop()
        self.maneuver_kind = maneuver
        self.dt = dt
        self._active: _Maneuver | None = None
        self._cooldown_ticks = 0
        self._alt_idx = 0
        self._tgt_hdg = 0.0
        self._tgt_alt = 0.0
        self._tgt_spd = 0.0

    def reset(self, s: AState) -> None:
        self._tgt_hdg = s.psi
        self._tgt_alt = s.alt
        self._tgt_spd = s.airspeed
        self._active = None
        self._cooldown_ticks = 0
        self._alt_idx = 0

    def _detect(self, s: AState, learner: AState) -> bool:
        rg = relative(s, learner)  # attacker seen from me
        return rg.rng < C.R_DETECT and rg.ata > C.REAR_HEMI_RAD

    def _spawn_maneuver(self) -> _Maneuver:
        kind = self.maneuver_kind
        if kind == "alternate":
            kind = "split_s" if (self._alt_idx % 2) else "break"
            self._alt_idx += 1
        return _MANEUVERS[kind]()

    def act(self, s: AState, opponent: AState) -> list:
        # opponent == the learner's state (the attacker we defend against).
        if self._active is not None:
            act4, done = self._active.step(s)
            if done:
                self._active = None
                self._cooldown_ticks = int(C.MANEUVER_COOLDOWN_S / self.dt)
                # Re-trim the level-hold to wherever the maneuver left us.
                self._tgt_hdg, self._tgt_alt, self._tgt_spd = _course(s), s.alt, s.airspeed
            return act4 + [0.0]

        if self._cooldown_ticks > 0:
            self._cooldown_ticks -= 1
        elif self._detect(s, opponent):
            self._active = self._spawn_maneuver()
            self._active.reset(s, opponent, self.dt)
            act4, _ = self._active.step(s)
            return act4 + [0.0]

        return self.inner.compute(s, self._tgt_hdg, self._tgt_alt, self._tgt_spd) + [0.0]


# ---------------------------------------------------------------------------
# Tier-3: rule-based FSM tactician + WEZ gun trigger (design doc §8.5)
# ---------------------------------------------------------------------------
#
# The first *shooting* opponent. A 4-state BFM FSM (+ two energy sub-states) on
# standard rules-of-thumb, all in "godmode" (reads both aircraft's full state).
# Transitions are dead-banded (separate enter/exit thresholds) and dwell-gated so
# they don't chatter at a boundary; a genuine threat preempts into DEFENSIVE at
# once. Guns fire whenever the target sits in the WEZ, independent of state.

_NEUTRAL = "neutral"
_OFFENSIVE = "offensive"
_DEFENSIVE = "defensive"
_YOYO = "high_yoyo"
_ENERGY = "energy_recovery"


def wez_fire(me: AState, target: AState) -> bool:
    """WEZ gate: fire iff we have ammo and a live target sits inside the effective
    gun range with a tight enough lead solution (§8.5). State-independent — this is
    the "fire when the attacker is in my weapons-engagement zone" rule."""
    if me.ammo <= 0 or not me.alive or not target.alive:
        return False
    rg = relative(me, target)
    # `naive_lead_error`, NOT the corrected `lead_error`: curriculum-86 made `lead_error` the
    # velocity-inheritance-corrected lay ([[gun-lay-velocity-inheritance]]), which would have
    # silently changed this bot's trigger. The scripted T3 IS the benchmark the whole
    # curriculum-01→85 history is measured against, so a learner-side aim fix must leave it
    # byte-identical. (`lead_pursuit`'s steering already uses the naive `lead_point`.)
    return rg.in_range and rg.naive_lead_error < C.FIRE_LEAD_RAD


def lead_pursuit(s: AState, target: AState, g: F16Gains, *, tgt_spd: float = C.V_PURSUIT) -> list:
    """Lead-pursuit control law → raw `[ail, elev, rud, thr]`: roll the lift vector onto
    the gun-lead point, pull onto it (corner-limited), and hold `tgt_spd` (§8.5).

    Extracted from `Tier3Opponent._pursue` so the exploration-only `PursuitExpert` can
    reuse the exact, sim-sign-correct law without duplicating it. `tgt_spd` defaults to
    `V_PURSUIT` (Tier-3's tuned hold, unchanged); the expert overrides it upward so it
    outruns — and thus closes on — a fleeing target instead of decelerating below it."""
    off, bank_err, _ = body_bearing(s, lead_point(s, target))
    roll_gate = min(off / C.PURSUE_ROLL_DEADBAND, 1.0)  # don't roll once on the nose
    ail = _clamp(C.K_PURSUE_ROLL * bank_err * roll_gate - g.k_roll_rate * s.p, -1.0, 1.0)
    aligned = max(cos(bank_err), 0.0)  # pull hardest once rolled onto the target
    elev = _corner_limited_pull(_clamp(-C.K_PURSUE_PULL * off * aligned, -1.0, 0.0), s)
    thr = _clamp(g.throttle_trim + g.k_spd_to_thr * (tgt_spd - s.airspeed), 0.0, 1.0)
    return [ail, elev, 0.0, thr]


class Tier3Opponent:
    """Rule-based BFM tactician with a WEZ-gated gun (§8.5).

    States: NEUTRAL (intercept / lead), OFFENSIVE (lead pursuit + guns), HIGH_YOYO
    (bleed closure to avoid an overshoot), DEFENSIVE (hard break into the attacker),
    ENERGY_RECOVERY (unload + accelerate when a break bleeds the jet slow). Each
    control law returns raw `[ail, elev, rud, thr]`; `act` appends the fire bit.
    """

    def __init__(self, gains: F16Gains | None = None, *, dt: float = 1.0 / 120.0):
        self.g = gains or F16Gains()
        self.dt = dt
        self._min_dwell = int(C.FSM_MIN_DWELL_S / dt)
        self._state = _NEUTRAL
        self._dwell = 0
        self._break_sign = 0.0  # 0 = unlatched; `_break` picks a direction on first use

    def reset(self, s: AState) -> None:
        self._state = _NEUTRAL
        self._dwell = 0
        self._break_sign = 0.0

    @property
    def state(self) -> str:
        return self._state

    # -- state machine -----------------------------------------------------
    def _to(self, new: str) -> None:
        if new != self._state:
            self._state = new
            self._dwell = 0

    def _dwell_ok(self) -> bool:
        return self._dwell >= self._min_dwell

    def _transition(self, s: AState, rg, rg_t) -> None:
        """`rg` = target seen from me; `rg_t` = me seen from the target (threat)."""
        threatened = rg_t.ata < C.THREAT_CONE_RAD and rg_t.rng < C.THREAT_RANGE
        threat_clear = rg_t.ata > C.THREAT_CONE_EXIT_RAD or rg_t.rng > C.THREAT_RANGE_EXIT
        st = self._state

        # Survival chain first (highest priority).
        if st == _DEFENSIVE:
            if s.airspeed < C.V_ENERGY_LOW:
                self._to(_ENERGY)
            elif threat_clear and self._dwell_ok():
                self._to(_NEUTRAL)
            return
        if st == _ENERGY:
            if s.airspeed > C.V_ENERGY_OK and self._dwell_ok():
                self._to(_DEFENSIVE if not threat_clear else _NEUTRAL)
            return
        # A real threat preempts any offensive/neutral state immediately (no dwell).
        if threatened:
            self._to(_DEFENSIVE)
            return

        # Offensive chain.
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
        # NEUTRAL.
        if rg.ata < C.OFF_ATA_ENTER_RAD and rg.rng < C.OFF_RANGE and self._dwell_ok():
            self._to(_OFFENSIVE)

    # -- control laws (each returns [ail, elev, rud, thr]) -----------------
    def _pursue(self, s: AState, target: AState):
        """Lead pursuit: put the lift vector on the lead point, pull onto it, hold
        pursuit speed."""
        return lead_pursuit(s, target, self.g)

    def _break(self, s: AState, target: AState):
        """Hard defensive break into the attacker (continuous, unlike Tier-2's timed
        maneuver): roll the lift vector nearly onto him (85° bank) and pull to the AoA
        limit — a max-performance turn that bleeds energy for turn rate, NOT the old
        energy-preserving corner-hold. `ease_speed=False` keeps only the departure cap.

        The pull is ROLL-GATED by `cos(bank_err)`, the same way `lead_pursuit` gates its own
        ("pull hardest once rolled onto the target") and Tier-2's break gates its 0.9 pull behind
        `abs(phi) >= 0.9 * BREAK_BANK_RAD`. Without the gate this law held max aft stick while
        rolling, and since it also recomputed its roll direction from `ata_az` every tick with no
        hysteresis, every reversal rolled 85° -> 0 -> -85° at max g: near wings-level the whole
        max-g demand became PITCH, so a "level break" turned into a zoom climb (measured at spawn
        aspect 55: theta +7 -> +42° and gamma +3 -> +28° in 4 s, then a locked gamma +38° climbing
        spiral to 8261 m at the 16° alpha ceiling, bleeding 218 -> 138 m/s). Both jets ended
        alpha-saturated at ~150 m/s where NEITHER can point a gun, which is the regime the chase
        curriculum spent ~20 ceilings failing to solve. Gated, the break holds 84° of bank, keeps
        its speed (V 158 -> 203 m/s) and turns in-plane as intended.

        NOTE: unlike `wez_fire` — deliberately pinned to the naive lead angle so a learner-side aim
        fix could not move the benchmark — this DOES move the benchmark, knowingly: scores here are
        not comparable to the curriculum-01→93 chain, which was measured against the ungated break.
        """
        # Latched roll direction with a deadband, so the break commits to a turn instead of
        # chattering when the attacker sits near the nose (`ata_az` ~ 0). Reversals are legitimate
        # BFM, so they are allowed — the threat just has to cross meaningfully to the other side.
        az = relative(s, target).ata_az
        if self._break_sign == 0.0:
            self._break_sign = 1.0 if az >= 0.0 else -1.0
        elif az * self._break_sign < -C.BREAK_REVERSE_DEADBAND_RAD:
            self._break_sign = -self._break_sign

        bank_err = self._break_sign * C.BREAK_BANK_RAD - s.phi
        ail = _clamp(self.g.k_roll_to_ail * bank_err - self.g.k_roll_rate * s.p, -1.0, 1.0)
        # max(..., 0) keeps the scale non-negative when the commanded bank is more than 90° away
        # (mid-reversal) — precisely the window that used to convert max g into pitch.
        aligned = max(cos(bank_err), 0.0)
        return [ail, _corner_limited_pull(-aligned, s, ease_speed=False), 0.0, 1.0]

    def _high_yoyo(self, s: AState, target: AState):
        """Pitch up out of plane at idle to dump closure/energy and avoid overshoot."""
        off, bank_err, _ = body_bearing(s, lead_point(s, target))
        roll_gate = min(off / C.PURSUE_ROLL_DEADBAND, 1.0)
        ail = _clamp(C.K_PURSUE_ROLL * bank_err * roll_gate - self.g.k_roll_rate * s.p, -1.0, 1.0)
        return [ail, _corner_limited_pull(-0.5, s), 0.0, 0.2]

    def _energy_recover(self, s: AState):
        """Roll wings-level, unload (slight nose-down), full throttle to regain speed."""
        ail = _clamp(-2.0 * s.phi - self.g.k_roll_rate * s.p, -1.0, 1.0)
        return [ail, 0.12, 0.0, 1.0]

    def act(self, s: AState, opponent: AState) -> list:
        """`opponent` is the target (the learner). Returns
        `[aileron, elevator, rudder, throttle, trigger]`."""
        self._dwell += 1
        rg = relative(s, opponent)  # target seen from me
        rg_t = relative(opponent, s)  # me seen from the target (threat picture)
        self._transition(s, rg, rg_t)

        st = self._state
        if st == _DEFENSIVE:
            act4 = self._break(s, opponent)
        elif st == _ENERGY:
            act4 = self._energy_recover(s)
        elif st == _YOYO:
            act4 = self._high_yoyo(s, opponent)
        else:  # NEUTRAL and OFFENSIVE both fly lead pursuit
            act4 = self._pursue(s, opponent)

        return act4 + [1.0 if wez_fire(s, opponent) else 0.0]


# ---------------------------------------------------------------------------
# Exploration guide (NOT an opponent): a pure-offensive lead-pursuit gun expert.
# ---------------------------------------------------------------------------


class PursuitExpert:
    """Relentless lead-pursuit gun expert — an EXPLORATION guide for the RL learner, not
    an opponent it fights.

    Always flies Tier-3's proven OFFENSIVE law (`lead_pursuit`) and fires whenever the
    target sits in the WEZ (`wez_fire`), with none of Tier-3's defensive / energy / yoyo
    states: it presses to guns and shoots even when the learner's own prior stalls at
    standoff. The RL collector (`train_parallel`) runs whole episodes under this expert
    with an annealing probability early in a stage — a DAgger-style seed that fills the
    replay buffer with complete close→kill trajectories the stalled prior cannot generate
    on the long-range rungs (where the fire-override is impotent: `gun_quality ≡ 0` beyond
    the gun's effective range, so it can't trigger until the jet has already closed).

    Reads full state ("godmode"), like every scripted law here. `act` returns
    `[aileron, elevator, rudder, throttle, trigger]` in the RAW env action space (the
    `_LinearActionWrapper` inverts it back to the agent's `[-1,1]` space for storage)."""

    def __init__(self, gains: F16Gains | None = None):
        self.g = gains or F16Gains()

    def reset(self, s: AState) -> None:  # stateless — defined for interface symmetry
        pass

    def act(self, me: AState, target: AState) -> list:
        # Proportional range-keeper (see constants): chase at the target's speed + a commanded
        # closure that is max far out and tapers to zero as we reach a stable guns-trail. This
        # both closes from standoff (the bare V_PURSUIT hold is slower than a cruising bandit, so
        # it would open range forever) AND avoids the overshoot a fixed high chase speed causes
        # (blaze past → can't re-solve).
        rg = relative(me, target)
        closure = _clamp(C.K_CHASE_RANGE * (rg.rng - C.R_CHASE_TRAIL), 0.0, C.V_CHASE_MARGIN)
        act4 = lead_pursuit(me, target, self.g, tgt_spd=target.airspeed + closure)
        return act4 + [1.0 if self._fire(me, target, rg) else 0.0]

    @staticmethod
    def _fire(me: AState, target: AState, rg) -> bool:
        """Point-blank fire discipline: shoot only when the aim point falls inside the target's
        hit sphere (predicted lateral miss `rng·sin(naive_lead_error)` ≤ `GUN_FIRE_MISS_M`), so the
        finite magazine is spent on rounds that connect — not hosed at long range where the
        dispersion footprint and any lead error dwarf the 8 m target (see constants).

        Uses `naive_lead_error` (the pre-curriculum-86 nose-to-aim-point angle) on purpose: this
        expert is a fixed REFERENCE bot whose numbers appear all over the run history, so it must
        not shift under the learner-side aim fix ([[gun-lay-velocity-inheritance]]). Its steering
        (`lead_pursuit` → `lead_point`) is naive too, so gate and stick stay consistent."""
        if me.ammo <= 0 or not me.alive or not target.alive or not rg.in_range:
            return False
        return rg.rng * sin(rg.naive_lead_error) <= C.GUN_FIRE_MISS_M
