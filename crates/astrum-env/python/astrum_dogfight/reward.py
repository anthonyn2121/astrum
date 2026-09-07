"""Reward, penalties and wrapper-enforced episode boundaries (design doc §8.3–§8.4).

Operates purely on `AState` snapshots (fabricable), so the reward table and the
boundary predicates are unit-testable with no live sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot

from . import constants as C
from .geometry import AState, relative


@dataclass
class RewardOut:
    reward: float
    terminated: bool
    truncated: bool
    info: dict


def below_hard_deck(me: AState, ground_alt: float = 0.0) -> bool:
    return (me.alt - ground_alt) < C.DECK_AGL


def outside_bubble(me: AState, ground_alt: float = 0.0, center=(0.0, 0.0)) -> bool:
    horiz = hypot(me.pos[0] - center[0], me.pos[1] - center[1])
    return horiz > C.R_BUBBLE or (me.alt - ground_alt) > C.CEILING_AGL


def compute(
    prev_me: AState,
    prev_bandit: AState,
    me: AState,
    bandit: AState,
    ground_alt: float = 0.0,
    gamma: float = C.GAMMA,
    waste_streak: float = 0.0,
    convert_bonus: float = 0.0,
    convert_dwell_used: float = 0.0,
    outcome_mode: bool = False,
) -> RewardOut:
    """One step's reward + episode flags, from the (prev → new) state transition.

    reward-v2 (§8.3): dense terms are damage, a NON-telescoping tracking incentive
    (nose-on × range-closeness, plus in-WEZ gun-solution quality), fire discipline
    (round-by-round trigger cost, cheap in solution / penalised out of it), and the
    time penalty. Terminal precedence: hard deck (worst) → trade → death → kill →
    bubble exit. `gamma` is accepted for signature stability but no longer used (the
    old γΦ′−Φ shaping is gone — see constants.py's reward-v2 note).

    reward-v4: the per-tick tracking sub-total is surfaced in `info["track_reward"]`
    so the env can clamp its per-EPISODE cumulative at `C.TRACK_BUDGET`. Tracking is
    the only unbounded dense term (≈`(K_TRACK+K_SOLUTION)·7200 ≈ 2·R_KILL` if left to
    run a whole episode), which reincarnated the passive-draw trap at a higher reward
    level; capping it keeps a kill strictly dominant. This function stays stateless —
    the accumulation/clamp lives in `DogfightGymEnv.step` (it owns the episode state).

    reward-v5: `TRACK_BUDGET` cut to ≈R_KILL/10 (the +50 cap was still competitive with
    a kill once commit-risk was priced in). reward-v6: adds a potential-based closing-range
    term `K_CLOSE·(Φ(rng′)−Φ(rng))` — because capping tracking removed the gradient that
    pulls the policy INTO gun range, so the v5 policy parked at standoff and coasted to a
    draw. That term uses a discount of 1.0 (see constants.py), so the passed `gamma` remains
    unused here; it is kept only for signature stability.

    reward-v7: a clock-TIMEOUT penalty (an unfinished engagement is a failure) is NOT applied
    here — a timeout is the sim running out of ticks, which this stateless per-transition
    function cannot see. It is added in `DogfightGymEnv.step` from the sim's truncation flag,
    per-stage (`C.R_TIMEOUT` on kill rungs, 0 on survive rungs). See constants.py's reward-v7 note.

    reward-v8: adds a second potential-based term `K_AIM·(q′ − q)` on the gun-solution quality
    (γ = 1, telescoping/non-farmable), because greedy eval of the v7 policy flew one firing pass
    then coasted — the capped tracking term left NO gradient to re-acquire the pipper after the
    first pass. See constants.py's reward-v8 note.

    reward-v10: the fire-discipline waste cost RAMPS with the length of the current out-of-solution
    burst (`waste_streak`), because the c29 champion (which is reward-v9) still empties its whole
    magazine in the first pass and then flies disarmed. To keep `compute` a pure function, the
    caller passes the running `waste_streak` in and reads the updated value back from
    `info["waste_streak"]`; `DogfightGymEnv.step` owns that per-episode state (like `_track_used`).
    At `waste_streak == 0` the cost is exactly `R_FIRE_WASTE` (v9-identical), so this is backward-
    compatible for single-call unit tests and short exploratory bursts. See constants.py's v10 note.

    reward-v11: the two immediate-pursuit dense terms — the non-telescoping nose-on tracking term and
    the closing-TOWARD part of the closing PBRS — are scaled by a pursuit-appropriateness gate
    `f_pursuit(rg.aspect)` (1 when the learner is in the bandit's rear quarter → `C.PURSUIT_FLOOR` at
    the beam/head-on merge). This removes the "rush the bandit nose-on the instant the fight starts"
    bribe at a high-aspect merge (which overshoots), so the policy can learn the lead turn; the gun-
    solution/aim/damage terms are ungated (a real shot pays at any aspect). Strict no-op at low aspect
    (f_pursuit=1 → identical to v10); the closing gate is one-sided (opening keeps full honest cost),
    so the telescoping term stays non-farmable. See constants.py's reward-v11 note.

    reward-v15 RETIRED (2026-08-29): the wounded-bandit re-attack amplifier (scaling the close/aim/stern
    PBRS by a wound factor) destabilized the policy in c42 and is reverted. Grounding the magnitudes
    (DMG_PER_HP=0.5) showed the reward ALREADY favours the kill by ~+260 vs ~−48 for a first-pass-coast, so
    the ≥55° wall is an EXECUTION/exploration gap (the overshoot-recovery re-attack is never learned), not a
    reward-shaping gap — amplifying shaping could not help. Reward math is back to reward-v14b.

    reward-v16 (2026-08-30): a dense ANTI-SPIRAL attitude penalty (K_SPIRAL·nose_low·bank_frac, gated off in
    a gun solution) taxes the nose-low DESCENDING-SPIRAL attitude that decked every c49-c57 merge attempt, so
    the policy reverses in the VERTICAL (a nose-high "egg") instead of corkscrewing at the deck. Added AFTER
    the deck cushion; see the block below and constants.py's reward-v16 note. All other terms are v14b.

    reward-v19 (2026-08-31, the BFM pivot, SUPERSEDED same day): dropped the five dense pure-pursuit terms
    entirely under `outcome_mode`. c62 (pure-outcome) CATASTROPHICALLY COLLAPSED the champion's fine-control
    finish — 18/24 deck at aspect 45/55° by only 1.15M steps (champion baseline under the identical verdict:
    0 losses, 9 kills) — because the dense tracking term IS what maintains the ≤55° kill; removing it
    everywhere (incl. the rehearsal anchor) left nothing to preserve fine control and the warm policy decked.

    reward-v19b (2026-08-31, the HYBRID fix, CURRENT): `outcome_mode` now KEEPS full dense pursuit
    (K_TRACK/K_SOLUTION/K_CLOSE/K_AIM/K_ASPECT — the finish stays intact, and reward-v11's f_pursuit gate
    already damps the "point at him / close now" bribe to PURSUIT_FLOOR at the high-aspect head-on merge, so
    it does NOT force the fatal straight merge) and merely ADDS the outcome extras: a capped SUSTAINED
    rear-quarter conversion DWELL (K_CONVERT_DWELL, the reward-v18 CONVERT_* box, non-terminal) and a
    specific-energy PBRS (K_ENERGY, telescoping/policy-invariant → "keep energy, use the vertical" — also an
    anti-deck signal). The genuinely-new lever is the BFM-coach exploration override (whole-episode doctrine
    BFM seeding the deck-free merge→convert trajectory distribution); the reward's job is only to preserve
    the finish (dense) and reward the conversion+energy (extras). `convert_dwell_used` carries the per-episode
    dwell accumulator (like waste_streak); the updated value is read back from info["convert_dwell_used"].
    See constants.py's v19 note. A strict no-op for every legacy run (outcome_mode=False → v14b/v16b terms).

    curriculum-78 (2026-09-01, "safe-core" — the ONLY survivors of the graph-orchestrate BFM-reward
    ideation that passed adversarial review): two HONESTY fixes, both outcome_mode-gated (strict legacy
    no-op) and near-no-op at the close finish. G1 — the gun-solution REWARD terms (K_SOLUTION, K_AIM)
    read the HONEST range-dependent `gun_quality_honest` (geometry.sigma_track) instead of the fixed
    LEAD_SIGMA=3° quality, fixing the documented ~3-4× fire/reward-gate calibration mismatch (rewarding
    un-hittable 3°@494m shots); the OBSERVED gun_quality is untouched (byte-identical obs for the warm
    net). E1 — a ONE-SIDED corner-speed-BAND airspeed PBRS (K_BAND) pays to regain corner speed and
    charges bleeding below it but is FLAT above (no overspeed tax — the reverse of the disproven
    reward-v26). The anti-spiral gate below is deliberately LEFT on the lenient `gun_quality` (never
    newly taxes a committed finishing pass). Judge ONLY by greedy jitter-0.5; stop on any finish erosion.
    """
    info: dict = {}
    reward = C.TIME_PENALTY

    # Dense: damage dealt / taken (zero-sum in self-play).
    dmg_dealt = max(0.0, prev_bandit.health - bandit.health)
    dmg_taken = max(0.0, prev_me.health - me.health)
    reward += C.DMG_PER_HP * dmg_dealt - C.DMG_PER_HP * dmg_taken
    if dmg_dealt:
        info["damage_dealt"] = dmg_dealt
    if dmg_taken:
        info["damage_taken"] = dmg_taken

    # Dense: non-telescoping tracking reward (reward-v2). Unlike the old PBRS term
    # this does NOT cancel over an episode, so holding the nose on the bandit and
    # closing to a gun solution earns a standing, cumulative payoff — the incentive
    # to *fly offensively* rather than survive to a draw. Only while both are alive.
    rg = relative(me, bandit)
    if me.alive and bandit.alive:
        prev_rg = relative(prev_me, prev_bandit)  # for the telescoping PBRS terms below
        # reward-v15 RETIRED (2026-08-29): the wounded-bandit re-attack amplifier destabilized the policy
        # (c42 regressed the proven 45° kill 12/12 → 16 HP dealt; monotone-declining returns to −99 over
        # 1.1M steps → the passive basin). Grounding the reward magnitudes (DMG_PER_HP=0.5) showed a
        # first-pass-then-coast already nets ≈−48 (the K_CLOSE/K_AIM PBRS go NEGATIVE on the overshoot +
        # R_TIMEOUT −50) vs ≈+260 for a kill — so the reward ALREADY overwhelmingly favours the kill; the
        # ≥55° wall is an EXECUTION/exploration gap (the policy never learned the overshoot-recovery
        # re-attack), NOT a reward-shaping gap. Amplifying position shaping could not help and only broke
        # the value scale. Reverted to reward-v14b (proven best: the c38 24/36-kill peak). WOUND_* constants
        # are left in constants.py, marked retired.
        # reward-v11: pursuit-appropriateness gate from the BANDIT aspect (rg.aspect: low = I'm in
        # his rear quarter, high = abeam/ahead at the merge). 1 when clearly behind → PURSUIT_FLOOR
        # at the beam/head-on merge, so the immediate-pursuit dense terms below no longer bribe a
        # nose-on rush at a high-aspect merge — the policy earns full pursuit reward only once it has
        # flown the lead turn into his rear (aspect drops). See constants.py's reward-v11 note.
        if rg.aspect <= C.PURSUIT_GATE_LO:
            f_pursuit = 1.0
        elif rg.aspect >= C.PURSUIT_GATE_HI:
            f_pursuit = C.PURSUIT_FLOOR
        else:
            frac = (rg.aspect - C.PURSUIT_GATE_LO) / (C.PURSUIT_GATE_HI - C.PURSUIT_GATE_LO)
            f_pursuit = 1.0 - (1.0 - C.PURSUIT_FLOOR) * frac
        info["f_pursuit"] = f_pursuit
        f_nose = (1.0 + cos(rg.ata)) * 0.5  # 1 nose-on → 0 nose-away
        f_range = max(0.0, (C.TRACK_RANGE - rg.rng) / C.TRACK_RANGE)  # 1 point-blank → 0 at TRACK_RANGE
        # reward-v11: gate ONLY the nose-on tracking component (K_TRACK·f_nose·f_range); the WEZ
        # gun-solution term (K_SOLUTION·gun_quality) is NOT gated — a real firing solution pays at
        # any aspect, so the kill incentive is intact and only the "point at him now" bribe is damped.
        # reward-v19b: the HYBRID KEEPS this dense pure-pursuit tracking term (and the three PBRS terms
        # below) in outcome_mode — dropping it (v19) decked the finish. f_pursuit already damps it at the
        # head-on merge, so it preserves the ≤55° kill without bribing the straight merge. The OUTCOME
        # extras (conversion-dwell + energy) are ADDED at the end of this alive/alive block.
        # G1 (curriculum-78): under outcome_mode the gun-solution REWARD formerly read the HONEST
        # range-dependent quality (`gun_quality_honest`). curriculum-80 P2 (2026-09-02) REVERTED that:
        # the tight honest σ flattened this K_SOLUTION term to ≈0 at chase range (no slope), and cold
        # training then never learned to aim (0 kills, sprayed for ~0 HP — see geometry.py obs note).
        # q_sol is LOOSE (fixed LEAD_SIGMA) again for ALL modes so the cold learner has a real aim
        # gradient; the SIM's 20mm ballistics enforce kill-honesty. `gun_quality_honest`/sigma_track
        # stay computed as telemetry (and for a future annealed-σ schedule).
        q_sol = rg.gun_quality
        track_r = C.K_TRACK * f_nose * f_range * f_pursuit + C.K_SOLUTION * q_sol
        reward += track_r
        # reward-v4: surface the tracking sub-total so the env can cap its per-episode
        # cumulative (it is the only unbounded dense term — see TRACK_BUDGET). Damage
        # and the fire bonus self-cap (100 HP = a kill; ammo is finite), so they aren't
        # capped here; the reward itself is left uncapped (the env trims the excess).
        info["track_reward"] = track_r
        info["gun_quality"] = rg.gun_quality
        if outcome_mode:
            info["gun_quality_honest"] = rg.gun_quality_honest  # G1 telemetry (honest signal actually rewarded)

        # Dense: potential-based CLOSING-RANGE shaping (reward-v6). The capped tracking term
        # above is fully earnable at STANDOFF (nose-on at 700 m banks the whole TRACK_BUDGET
        # without closing), so eval of the v5 policy parked it nose-on at range, coasting to a
        # draw (0/20 kills at 700 m). This F = Φ(rng′) − Φ(rng) pays to REDUCE range and costs
        # to open it, restoring the "get to gun range" gradient the cap removed. It is
        # telescoping (sum over an episode = Φ(end) − Φ(start)) so it cannot be farmed and does
        # not shift the optimal policy. The discount is 1.0 by design — a γ<1 PBRS on a negative
        # potential would pay to merely hold range far (Φ(γ−1) > 0); γ=1 makes holding = 0.
        def _phi(r: float) -> float:
            return -min(max(r, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE

        # reward-v11: damp ONLY the closing-toward (positive) part at high aspect; the opening
        # (negative) part keeps its full honest cost, so no close-low/open-high cycle nets positive
        # (every closed range cycle still sums ≤ 0 → non-farmable). At low aspect f_pursuit=1 → the
        # original telescoping term is recovered exactly.
        # reward-v23 (2026-09-01): in the control-position zone, GATE the positive closing bribe on
        # LOW CROSSING. Diagnosis (c73 greedy @1.3M): warm-consolidated on the hot pass the policy
        # CONVERTS to the six (aspect 10°/ata 1°) but the in-zone lead GREW 20°→29° and it stopped
        # firing — because K_CLOSE (120) bribes the OVERTAKE, and overtaking is exactly what sustains
        # the crossing velocity that forces the large lead. In-zone, close_r (pay to close) and the
        # reward-v22 lead_r (pay to match velocity = STOP closing) are directly opposed; the bigger
        # gain (K_CLOSE) wins → the crossing is never nulled. Fix: once in the rear-quarter gun zone
        # (aspect ≤ CONVERT_ASPECT_RAD ∧ rng ≤ CONVERT_RANGE_MAX), the POSITIVE close bribe is paid
        # ONLY at LOW closure (|closure| ≤ CONVERT_CLOSURE_MAX, the reward-v21 matched-velocity gate) —
        # so EASING into range still pays, but the high-overtake crossing no longer does, letting
        # reward-v22's lead-null gradient settle the jet onto a matched-velocity tracking shot. The
        # OPENING (negative) part always keeps its full cost (non-farmable: a fast-close/open cycle
        # nets < 0). Outside the zone (the long-range approach) the bribe is unchanged — closing fast
        # to the merge is still rewarded. Gated behind outcome_mode (a strict no-op for legacy runs).
        raw_close = C.K_CLOSE * (_phi(rg.rng) - _phi(prev_rg.rng))
        if raw_close > 0.0:
            in_gun_zone = rg.aspect <= C.CONVERT_ASPECT_RAD and rg.rng <= C.CONVERT_RANGE_MAX
            if outcome_mode and in_gun_zone and abs(rg.closure) > C.CONVERT_CLOSURE_MAX:
                close_r = 0.0  # reward-v23: no overtake bribe in-zone → let reward-v22 null the crossing
            else:
                close_r = raw_close * f_pursuit
        else:
            close_r = raw_close  # opening always keeps full cost (reward-v11/v19b)
        reward += close_r
        info["close_reward"] = close_r

        # Dense: potential-based AIM shaping (reward-v8). F = K_AIM·(q′ − q) on the gun-solution
        # quality q — PAYS to sharpen the firing solution and COSTS to lose it, supplying the
        # "re-acquire the pipper after the first pass" gradient that the CAPPED tracking term stops
        # providing. Telescoping (γ = 1) → sum over an episode is K_AIM·(q_end − q_start), so it is
        # non-farmable (holding a solution nets 0) and policy-invariant. See constants.py's v8 note.
        # reward-v19b: kept in outcome_mode. G1 (curriculum-78) telescoped the HONEST quality here;
        # curriculum-80 P2 (2026-09-02) REVERTED to the fixed-LEAD_SIGMA quality for ALL modes — the
        # honest σ made this re-acquire PBRS flat-zero at chase range (Δq≈0 for any realistic aim), so
        # cold training got no "sharpen the pipper" gradient (see the q_sol / geometry.py notes).
        # Telescoping is preserved (both endpoints use the same loose q family) → non-farmable and
        # policy-invariant.
        aim_r = C.K_AIM * (rg.gun_quality - prev_rg.gun_quality)
        reward += aim_r
        info["aim_reward"] = aim_r

        # Dense: potential-based STERN-CONVERSION shaping (reward-v13). F = K_ASPECT·(Ψ(A′) − Ψ(A)) on
        # the BANDIT aspect A (rg.aspect, radians; low = in his rear quarter). Ψ(A) = −clip(A, 0,
        # PURSUIT_GATE_HI)/PURSUIT_GATE_HI ∈ [−1, 0] rises toward 0 as A → 0, so flying to his six PAYS
        # and falling off it COSTS — the missing gradient that turns a high-aspect merge into a stern
        # chase (c35 closed range nose-off and sprayed empty air at aspect ≥70°; K_CLOSE rewarded range
        # but nothing rewarded the rear-quarter geometry in between). Telescoping (γ = 1, exactly like
        # the closing/aim PBRS): the episode sum is K_ASPECT·(Ψ_end − Ψ_start), so an aspect wobble nets
        # 0 (non-farmable) and it is policy-invariant. See constants.py's reward-v13 note.
        def _psi(a: float) -> float:
            return -min(max(a, 0.0), C.PURSUIT_GATE_HI) / C.PURSUIT_GATE_HI

        aspect_r = C.K_ASPECT * (_psi(rg.aspect) - _psi(prev_rg.aspect))  # reward-v19b: kept in outcome_mode
        reward += aspect_r
        info["aspect_reward"] = aspect_r

        # reward-v19b OUTCOME extras (only when outcome_mode; a strict no-op otherwise). Added ON TOP OF the
        # (now-preserved) dense pursuit terms, still inside the both-alive guard. See constants.py's v19 note.
        if outcome_mode:
            # (1) SUSTAINED rear-quarter conversion DWELL — the "sustained-tail-conversion" outcome.
            # Pays only inside the reward-v18 SETTLED box: his six (aspect ≤ CONVERT_ASPECT), nose
            # roughly ON him (ata ≤ CONVERT_ATA), wings settled (|phi| ≤ CONVERT_BANK), in the gun band
            # (reward-v20: DWELL_RANGE_MIN ≤ rng ≤ CONVERT_RANGE_MAX — floor decoupled to 40 m so the
            # close-in merge finish, 66-134 m, earns settle-reward instead of swinging through it).
            # Capped per-episode at CONVERT_DWELL_BUDGET (< R_KILL) via the caller-carried accumulator,
            # so a kill (which ends the dwell) dominates a park-on-his-six farm. NOT terminal
            # (convert_bonus=0 here → the finish is the goal).
            if (
                rg.aspect <= C.CONVERT_ASPECT_RAD
                and rg.ata <= C.CONVERT_ATA_RAD
                and abs(me.phi) <= C.CONVERT_BANK_RAD
                and C.DWELL_RANGE_MIN <= rg.rng <= C.CONVERT_RANGE_MAX
            ):
                granted = min(C.K_CONVERT_DWELL, max(0.0, C.CONVERT_DWELL_BUDGET - convert_dwell_used))
                reward += granted
                convert_dwell_used += granted
                info["convert_dwell"] = granted
            info["convert_dwell_used"] = convert_dwell_used

            # (2) SPECIFIC-ENERGY PBRS (telescoping γ=1 → policy-invariant): "keep energy, use the
            # vertical." Es = AGL + V²/2g; Φ_E = clip(Es, 0, 1.5·ENERGY_REF)/ENERGY_REF. Pays to gain
            # energy and costs to bleed it (into a flat energy-dumping turn / the deck), densifying the
            # terminal deck signal WITHOUT an energy-hoarding basin (a standing alt reward would).
            def _es_phi(a: AState) -> float:
                es = max(0.0, a.alt - ground_alt) + a.airspeed * a.airspeed / (2.0 * 9.80665)
                return min(es, C.ENERGY_REF * 1.5) / C.ENERGY_REF

            # reward-v25 (2026-09-01, curriculum-76): ENGAGEMENT-GATE the energy-GAIN side. c75 (reward-v24,
            # rel-Es=50) learned to DISENGAGE and hoard — greedy left the bubble climbing away with
            # endRelEs 4000+ (the flee banks +Es shaping that offsets the −50 bubble/timeout). Both Es
            # PBRS terms pay their GAIN side while the jet climbs OUT of the fight, so fleeing pays. Fix
            # (mirrors the one-sided K_LEAD/K_CLOSE gating): credit energy GAINS only while ENGAGED
            # (rng ≤ TRACK_RANGE=3000, the "in the fight" range) and ALWAYS charge energy LOSSES. Now the
            # only +EV energy play is out-flying him WHILE engaged (press/bait the break), not extending
            # to a high perch. Non-farmable (one-sided). Anti-deck role intact (the loss side is untouched).
            engaged = rg.rng <= C.TRACK_RANGE  # reward-v25: "in the fight" gate on the energy-GAIN side
            raw_energy = C.K_ENERGY * (_es_phi(me) - _es_phi(prev_me))
            energy_r = raw_energy if (raw_energy <= 0.0 or engaged) else 0.0
            reward += energy_r
            info["energy_reward"] = energy_r

            # reward-v21 (a) CORNER-SPEED FLOOR (anti-zoom-stall). The absolute-Es PBRS above is
            # NEUTRAL on a speed↔alt trade, so the merge zoom-to-stall (airspeed 222→80 m/s, trading
            # all speed for altitude) costs it nothing — the exact hole the T3 exploits (it holds
            # ≥128 m/s in the vertical scissors while we stall). This per-tick penalty bites ONLY
            # below V_CORNER_FLOOR, ramping 0→K_CORNER as airspeed falls to 0, so it taxes bleeding
            # below sustainable maneuvering speed directly WITHOUT rewarding raw speed (no run-away
            # bribe) and WITHOUT a standing basin (0 at/above the floor).
            if me.airspeed < C.V_CORNER_FLOOR:
                corner_pen = C.K_CORNER * (C.V_CORNER_FLOOR - me.airspeed) / C.V_CORNER_FLOOR
                reward -= corner_pen
                info["corner_penalty"] = -corner_pen

            # E1 (curriculum-78, safe-core): CORNER-SPEED-BAND airspeed PBRS. A ONE-SIDED telescoping
            # (γ=1 → policy-invariant, non-farmable) potential that pays REGAINING corner speed and
            # charges bleeding BELOW it, but is FLAT at/above corner so carrying or zooming for energy
            # is NEVER taxed. The flat top is the deliberate fix for the reward-v26 overspeed-tax
            # failure (taxing speed ABOVE a ceiling made the learner timid/disengage) and it removes
            # any "coast to exactly corner speed" terminal bias (ending fast gives the same Φ=1). It
            # does NOT oppose K_ENERGY above corner (zero fast-side gradient); below corner both agree
            # (reward speeding back up toward corner). Distinct from the reward-v21 CORNER-SPEED FLOOR
            # one-tick penalty above — this is a smooth potential that also REWARDS the recovery.
            def _band_phi(v: float) -> float:
                d = min(0.0, v - C.V_BAND_CTR) / C.V_BAND_SIGMA
                return exp(-0.5 * d * d)

            band_r = C.K_BAND * (_band_phi(me.airspeed) - _band_phi(prev_me.airspeed))
            reward += band_r
            info["band_reward"] = band_r

            # reward-v26 (curriculum-77) CONVERSION OVERSPEED TAX — REVERTED / DISPROVEN. Taxing in-zone
            # airspeed above a ceiling to force "bleed to corner" was COUNTERPRODUCTIVE: it made the
            # learner timid in the tracking band (won't drive in hard), so it never pressured the T3 into
            # bleeding (bandMinV stayed ~171 vs c76's 92) and it disengaged (left_bubble) instead of
            # pressing. c77@5.38M was strictly worse than c76's reward-v25 peak. See constants.py.

            # reward-v21 (b) RELATIVE-Es PBRS ("out-fly him on energy"). The absolute-Es PBRS is blind
            # to the OPPONENT — a symmetric zoom-stall nets ~0 there. The merge is an ENERGY WAR we
            # currently LOSE; this telescoping term (γ=1 → policy-invariant, non-farmable; episode sum
            # = K_REL_ENERGY·(Φ_rel_end − Φ_rel_start)) pays to END with MORE specific energy than the
            # bandit and costs to fall behind, shaping RL to out-manage the T3's energy rather than
            # merely hoard its own.
            def _es(a: AState) -> float:
                return max(0.0, a.alt - ground_alt) + a.airspeed * a.airspeed / (2.0 * 9.80665)

            def _rel_phi(m: AState, b: AState) -> float:
                return max(-1.0, min(1.0, (_es(m) - _es(b)) / C.ENERGY_REF))

            # reward-v25: same engagement gate as the absolute-Es term above — credit a rel-Es GAIN only
            # while in the fight (rng ≤ TRACK_RANGE), always charge falling behind. Out-energizing the T3
            # only pays when you STAY and press him, never by running away to a high, safe perch.
            raw_rel = C.K_REL_ENERGY * (_rel_phi(me, bandit) - _rel_phi(prev_me, prev_bandit))
            rel_energy_r = raw_rel if (raw_rel <= 0.0 or engaged) else 0.0
            reward += rel_energy_r
            info["rel_energy_reward"] = rel_energy_r

            # reward-v22 (2026-09-01) LEAD-ERROR PBRS — NULL THE CROSSING into a control-position tracking
            # shot. F = K_LEAD·(Φ_L(lead′) − Φ_L(lead)), Φ_L(θ) = (1+cos θ)/2. The gun_quality Gaussian
            # (LEAD_SIGMA=3°) is FLAT for lead_error beyond ~10°, so K_AIM gives zero pull out of a 30-57°
            # co-speed crossing — the exact reason the policy AND the perfect BFM coach reach aspect 4° yet
            # dump the mag for 0 damage. Φ_L has a smooth non-vanishing gradient across the whole range, so
            # it pulls from the crossing down to a solution; following it requires matching his velocity
            # (that is HOW lead drops) = the settle/tracking skill. GATED to the rear-quarter gun zone
            # (aspect ≤ CONVERT_ASPECT_RAD ∧ rng ≤ CONVERT_RANGE_MAX) so it shapes the CONTROL POSITION, not
            # the unavoidable approach/merge crossing. ONE-SIDED like K_CLOSE (pay improvement only in-zone;
            # always charge degradation) → non-farmable. See constants.py's reward-v22 note.
            def _phi_lead(g) -> float:
                return (1.0 + cos(g.lead_error)) * 0.5

            in_zone = rg.aspect <= C.CONVERT_ASPECT_RAD and rg.rng <= C.CONVERT_RANGE_MAX
            raw_lead = C.K_LEAD * (_phi_lead(rg) - _phi_lead(prev_rg))
            lead_r = raw_lead if (raw_lead <= 0.0 or in_zone) else 0.0
            reward += lead_r
            info["lead_reward"] = lead_r

    # Dense: fire discipline (reward-v2 → v9 → v10). Charge per round actually fired (ammo
    # decrement) — no action plumbing needed. reward-v9 keys the bonus/waste split on the
    # PursuitExpert's HIT gate (predicted lateral miss rng·sin(lead_error) inside WEZ_HIT_M — reward-v14
    # decoupled this reward gate from the expert's own GUN_FIRE_MISS_M trigger and tightened it to 6 m,
    # ≈ the target's hit sphere): a round that actually connects pays R_FIRE_WEZ; a long-range
    # spray costs R_FIRE_WASTE. reward-v10 makes the WASTE cost RAMP with the length of the current
    # out-of-solution burst so a sustained mag-dump is clearly −EV, while an in-solution round
    # resets the burst (disciplined) and a trigger release / reposition decays it — see the v10 note.
    rounds = max(0, int(prev_me.ammo) - int(me.ammo))
    if rounds:
        # curriculum-86: `rg.lay_miss_m` replaces `rng·sin(lead_error)`. Same quantity — the predicted
        # miss distance at the target — but exact: the round is displaced `1030·sin(δ)·tof`, and with
        # velocity inheritance its groundspeed is ~1230 m/s, so `tof < rng/1030` and the old form
        # over-stated the miss by 14-20%. Now that `lead_error` is measured to the CORRECTED boresight
        # ([[gun-lay-velocity-inheritance]]) the gate is directly comparable to the sim's own 8 m hit
        # sphere, so `WEZ_HIT_M` finally means what it says: this round connects.
        # B1: `in_range` (1200 m) is NOT the lethal envelope — it is the scripted T3's trigger range,
        # left at 1200 m so the benchmark cannot move. Without the `LETHAL_RANGE_M` term an 1100 m
        # round with a sub-8 m COMPUTABLE lay collected the full "this round connects" bonus at a
        # measured p_hit of 0.000 (+0.51 for a 10-round burst that lands nothing); it now costs
        # −0.19 and feeds the waste ramp. `<=` not `<` so 600 m exactly is still a solution.
        in_solution = (
            rg.in_range
            and rg.lay_feasible
            and rg.lay_miss_m < C.WEZ_HIT_M
            and rg.rng <= C.LETHAL_RANGE_M
        )
        # reward-v17: the merge-CONVERTER (M, convert_bonus>0) is NEVER paid the WEZ fire bonus — it
        # must save the shared jet's ammo for the frozen finisher F, and its win is the CONVERSION
        # terminal (below), not a shot. Firing then only ever COSTS (the waste ramp), so M learns to
        # hold fire and position. A strict no-op for every legacy run (convert_bonus=0 → bonus paid).
        if in_solution and convert_bonus <= 0.0:
            reward += rounds * C.R_FIRE_WEZ
            waste_streak = 0.0  # a hit-gate round is disciplined → forgive the burst
        else:
            # Ramp on the burst length AT THE START of this tick (rounds/tick ≤ 1 at the gun's
            # rate, so this is exact in the live sim and leaves a single-call test at streak 0
            # identical to v9); then grow the streak by the wasted rounds just fired.
            ramp = min(1.0 + waste_streak / C.WASTE_RAMP_TAU, C.WASTE_RAMP_MAX)
            reward += rounds * C.R_FIRE_WASTE * ramp
            waste_streak += rounds
        info["rounds_fired"] = rounds
    else:
        waste_streak = max(0.0, waste_streak - C.WASTE_STREAK_DECAY)  # a pause forgives the burst
    info["waste_streak"] = waste_streak

    # Dense: hard-deck CUSHION penalty (reward-v9). The deck bust below is TERMINAL — a cliff
    # with no gradient. This ramps a per-tick cost 0 → C.K_DECK_FLOOR as AGL falls from
    # (DECK_AGL + DECK_MARGIN) down to DECK_AGL, so a nose-low chase feels the deck approaching
    # and pulls up before busting. Only bites in the bottom cushion (start alt ≫ this); kept
    # milder than the tracking gradient so it shades — never forbids — a committed low gun pass.
    agl = me.alt - ground_alt
    if agl < C.DECK_AGL + C.DECK_MARGIN:
        deck_frac = min(1.0, max(0.0, (C.DECK_AGL + C.DECK_MARGIN - agl) / C.DECK_MARGIN))
        deck_pen = C.K_DECK_FLOOR * deck_frac
        reward += deck_pen
        info["deck_penalty"] = deck_pen

    # Dense: ANTI-SPIRAL attitude penalty (reward-v16 / v16b). Taxes the nose-low DESCENDING-SPIRAL
    # attitude (nose-low × banked) — the c49-c57 decking signature — so the policy reverses in the VERTICAL
    # (a nose-high "egg") instead of corkscrewing at the deck. GATED OFF in a real gun solution (a committed
    # diving guns pass is legitimate); a PRODUCT of nose-low × bank, so a wings-level dive OR a nose-HIGH
    # banked turn (the egg) pays nothing; NOT a climb bonus (no energy-hoarding bribe).
    # reward-v16b ALTITUDE GATE (the alt_frac factor): a death spiral is a LOW-altitude event. The v16
    # penalty fired at every altitude and taxed the legitimate nose-low BANKED converting pursuit turn just
    # as hard, so c58 went globally PASSIVE (55° finish → 0 damage, merge stalled at net 0). alt_frac ramps
    # the penalty 0 → 1 as AGL falls from SPIRAL_ALT_BAND_AGL down to the deck cushion (DECK_AGL+DECK_MARGIN),
    # where the deck-floor term takes over. Above the band the corkscrew pays NOTHING → aggressive turns at
    # fighting altitude are freed (cures the passivity); down low it still bites (anti-deck safety net).
    if me.alive and rg.gun_quality < C.SPIRAL_SOLUTION_Q:
        nose_low = min(1.0, max(0.0, -me.theta / C.SPIRAL_PITCH_RAD))
        bank_frac = min(
            1.0,
            max(0.0, (abs(me.phi) - C.SPIRAL_BANK_MIN_RAD) / (C.SPIRAL_BANK_MAX_RAD - C.SPIRAL_BANK_MIN_RAD)),
        )
        alt_floor = C.DECK_AGL + C.DECK_MARGIN
        alt_frac = min(1.0, max(0.0, (C.SPIRAL_ALT_BAND_AGL - agl) / (C.SPIRAL_ALT_BAND_AGL - alt_floor)))
        spiral = nose_low * bank_frac * alt_frac
        if spiral > 0.0:
            spiral_pen = C.K_SPIRAL * spiral
            reward += spiral_pen
            info["spiral_penalty"] = spiral_pen

    terminated = False
    truncated = False

    # Terminal: hard-deck bust (checked first — the single worst outcome).
    if below_hard_deck(me, ground_alt):
        reward += C.R_DECK
        terminated = True
        info["event"] = "hard_deck"
        return RewardOut(reward, terminated, truncated, info)

    # Terminal: trade / death / kill (mutually exclusive, in that preference order).
    just_killed = prev_bandit.alive and not bandit.alive
    just_died = prev_me.alive and not me.alive
    if just_died and just_killed:  # mutual kill same tick — a trade, its own (soft) penalty
        reward += C.R_TRADE
        terminated = True
        info["event"] = "trade"
        return RewardOut(reward, terminated, truncated, info)
    if just_died:
        reward += C.R_DEATH
        terminated = True
        info["event"] = "killed"
        return RewardOut(reward, terminated, truncated, info)
    if just_killed:
        reward += C.R_KILL
        terminated = True
        info["event"] = "kill"
        return RewardOut(reward, terminated, truncated, info)

    # Terminal: SURVIVE-CONVERT (reward-v17). Only for the merge-converter policy M (convert_bonus>0);
    # a strict no-op for every legacy ladder. M "wins" the instant it reaches a genuine rear-quarter
    # firing position behind a LIVE bandit — in his rear quarter (aspect low), NOSE ON him (ata low, so
    # a head-on fly-through where I'm swept behind but flying AWAY does NOT count), and inside F's
    # prosecution band — at which point the geometric selector (eval) hands the jet to the frozen
    # finisher F. Placed AFTER kill/death/trade/deck so those keep precedence, BEFORE the bubble
    # truncation so a conversion at the bubble edge still counts. Must be `terminated` (not truncated)
    # so TD-MPC2's value bootstrap zeroes past it (the finish is F's problem, not part of M's return).
    # reward-v18: the box also requires the jet be SETTLED — nose genuinely on him (tighter ata) and
    # roughly wings-level (|phi| ≤ CONVERT_BANK_RAD). The c60 verdict showed M "converting" 100% at the
    # 135° merge by clipping the box's far CORNER at ~100° of bank DURING the merge pass (a transient,
    # not an arrival); F, handed that knife-edge, spiralled into the deck (7/12, 0 kills). The bank gate
    # rejects the knife-edge slice so the hand-off only fires from a flyable state inside F's envelope,
    # and it forces M to actually ROLL OUT behind the bandit (the real conversion) to earn the bonus.
    # reward-v21: the box ALSO requires LOW |closure| (matched-velocity settled tail). Without it the
    # box could fire on a high-closure MERGE-PASS transient (nose swept on his six for ~3 s as the jets
    # cross at ~2× speed) — exactly the c60/c61 seam failure: F, handed a fly-through it never trained
    # for, spiralled in. The crossing-velocity finding showed even a wings-level nose-on state at the
    # merge has no gun solution (crossing forces a 30-57° lead → bullets miss by 100-200 m); F can only
    # finish a genuine matched-velocity tail. This conjunct forces M to null the crossing (roll out
    # BEHIND at his speed) before the hand-off — the real conversion — and MUST stay in lockstep with
    # the eval selector (_verdict2p_tmp.py).
    if (
        convert_bonus > 0.0
        and me.alive
        and bandit.alive
        and rg.aspect <= C.CONVERT_ASPECT_RAD
        and rg.ata <= C.CONVERT_ATA_RAD
        and abs(me.phi) <= C.CONVERT_BANK_RAD
        and abs(rg.closure) <= C.CONVERT_CLOSURE_MAX
        and C.CONVERT_RANGE_MIN <= rg.rng <= C.CONVERT_RANGE_MAX
    ):
        reward += convert_bonus
        terminated = True
        info["event"] = "conversion"
        return RewardOut(reward, terminated, truncated, info)

    # Truncate: left the engagement bubble.
    if outside_bubble(me, ground_alt):
        reward += C.R_BUBBLE_EXIT
        truncated = True
        info["event"] = "left_bubble"

    return RewardOut(reward, terminated, truncated, info)
