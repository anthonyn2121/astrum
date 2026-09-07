"""Async multi-process TD-MPC2 trainer — the design-doc §8 throughput scale-out.

    cd controllers/rl
    uv run python -m astrum_rl.tdmpc2.train_parallel --workers 6 \
        --steps 6_000_000 --save runs/par/run.pt

Why this exists — the single-env loop (`train.py`) is **not** simulator-bound: one
JSBSim env alone collects ~1000 control-steps/s, but the loop drops to ~50/s because it
*serializes* MPPI planning-to-act and one gradient update on the same GPU, every step.
This trainer breaks that coupling:

* **K collector processes**, one JSBSim 1v1 env each (JSBSim is process-global + not
  thread-safe, so it MUST be separate processes — see envs.py), acting on **CPU** with
  the policy prior (or optional CPU-MPPI). They stream whole episodes into a queue.
* **One central learner** (this process) owns the GPU: it drains episodes into the
  shared replay buffer and runs gradient updates **back-to-back**, no longer time-shared
  with an actor's per-step planning. It periodically broadcasts fresh weights to the
  collectors.

Net effect: the learner does GPU updates at the device's update rate (not ~50/s), while
K collectors keep the 1M buffer fed and add initial-condition/seed diversity. It is also
the natural substrate for self-play (each collector = one match vs a pooled opponent).
Bounded by GPU update latency, so speedup is sub-linear in K — this trades some sample
efficiency (UTD < 1) for wall-clock, exactly the "more efficient learning at the risk of
over-tuning" tradeoff.

Correctness carried over from `train.py` (design doc §6): episodes are transported and
stored **whole**, so each begins with its `first=True` obs and `terminated` (kill/death/
deck) stays distinct from `truncated` (bubble/time). The `--steps` budget counts
**collected env (control) steps** across all workers — same semantics as the single-env run.

The curriculum is a ladder of (opponent tier, start separation, gate metric) **stages**
(`_CHASE_THEN_DEFEND` / `build_stages`), not a bare tier schedule — because separation is
what actually gates the skill (a 400 m spawn hands the agent a point-blank kill; a real
chase must be *learned* farther out). Advance is **performance-gated on sustained mastery**:
a stage is cleared only once its success-rate — a *kill* on the offensive/converting rungs,
mere *survival* on the point-blank defensive rungs — holds ≥ its bar over a large recent
window (defaults: 100 episodes ≥ 70%/85%, min 120 played), so a lucky streak can't promote.
A per-stage step **backstop** forces progress if the gate is slow, so no stage stalls the
run forever. `--resume` continues a saved run at its recorded stage (backstops rebase to the
resume stage). See docs/dogfight-tdmpc2-design.md §8.7.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import os
import queue
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from .. import _paths
from .agent import TDMPC2, resolve_device
from .buffer import ReplayBuffer
from .config import TDMPC2Config
from .train import build_config


# ======================================================================
# Weight transport: serialize the model to bytes (robust across `spawn`;
# sidesteps shared-memory FD lifetime issues that plague tensor-over-Queue).
# ======================================================================
def _serialize_model(model) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getvalue()


def _load_latest_weights(agent: TDMPC2, weight_q) -> bool:
    """Drain the weight queue keeping only the newest blob; load it if any. Returns
    True iff weights were loaded (a collector uses this to leave the random-seed phase)."""
    blob = None
    while True:
        try:
            blob = weight_q.get_nowait()
        except queue.Empty:
            break
    if blob is None:
        return False
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    agent.model.load_state_dict(sd)
    return True


# ======================================================================
# Self-play (curriculum-21): the collector's scripted opponent is replaced by a frozen copy of
# the learner's own policy, sampled per-episode from a growing pool the learner periodically adds
# to. These two helpers are the wiring the selfplay.py scaffold left to the trainer.
# ======================================================================
def _snapshot_prior(agent, cfg, pool_dir: str, idx: int) -> str:
    """Export the learner's CURRENT policy prior into the self-play pool (design doc P5).

    Traces the deployable `obs→action∈[-1,1]` prior (encoder + π mean) from a *copy* of the live
    world-model state to a CPU TorchScript module — the exact artifact `ScriptedPolicyOpponent`
    loads. Written to `snap_<idx>.pt.tmp` then `os.replace`-d to the final name so a collector
    globbing the pool never sees a half-written file (os.replace is atomic on POSIX). The copy is
    on CPU and detached from the training graph, so this neither perturbs the GPU model nor its
    optimizer. No `.json` sidecar is needed — the opponent maps actions with the fixed
    `export.ACTION_LOW/HIGH`, not the metadata."""
    from .export import PolicyPrior

    prior = PolicyPrior.from_world_model(agent.model, cfg).to(torch.device("cpu")).eval()
    example = torch.zeros(1, cfg.obs_dim)
    with torch.no_grad():
        scripted = torch.jit.trace(prior, example)
    final = os.path.join(pool_dir, f"snap_{idx:06d}.pt")
    tmp = final + ".tmp"
    scripted.save(tmp)
    os.replace(tmp, final)
    return final


def _prune_pool(pool_dir: str, max_pool: int) -> None:
    """Keep only the `max_pool` most-recent snapshots (by name, which is snapshot-ordered), so the
    pool dir + per-episode glob stay bounded on a long run. A collector mid-load of a pruned file
    is covered by `_sample_opponent`'s try/except; the seed/oldest are the least-sampled anyway."""
    if max_pool <= 0:
        return
    snaps = sorted(Path(pool_dir).glob("snap_*.pt"))
    for stale in snaps[:-max_pool]:
        try:
            stale.unlink()
        except OSError:
            pass


def _sample_opponent(pool_dir: str, frame_stack: int, recency_bias: float, rng):
    """Sample one `ScriptedPolicyOpponent` from the current pool dir, or None if it is empty / a
    load raced a write. Re-globs every call so freshly-snapshotted priors enter the pool the
    moment the learner writes them (a stronger opponent as the learner improves)."""
    from .selfplay import OpponentPool

    pool = OpponentPool(frame_stack=frame_stack, device="cpu", recency_bias=recency_bias)
    pool.add_dir(pool_dir, pattern="snap_*.pt")
    if len(pool) == 0:
        return None
    try:
        return pool.sample(rng)
    except Exception:
        return None  # a torch.jit.load racing an os.replace (rare) → keep the prior opponent


# curriculum-56/57/58 (imitation-seed the merge circle): a SCRIPTED reversal in AGENT action space,
# spliced into rollouts by the merge-turn override (below). The champion FREEZES at the merge — it fires
# `spent 0`, holds attitude, and drifts to a deck or timeout — because its weights encode a 5 s
# co-heading kill and NO turning circle, so RL never sees the circle it must learn from (5th
# confirmation c49/c50/c51/c53/c55). The PursuitExpert can't seed it either: it DECKS on the
# offensive-posture merge. So we hand-script the motor prior.
#
# curriculum-58 REWRITE — a VERTICAL EGG, not a horizontal break (the user's directive after the 7th
# collapse: "don't teach it to spiral DOWN — a real pilot does a vertical egg maneuver"). c56/c57 both
# scripted a mostly-HORIZONTAL turn (roll to ~45°+ bank, pitch to only a shallow +8° climb); at that
# bank the pull's lift vector is mostly sideways, so under a hard pull the jet DESCENDS through the turn
# — the nose-low corkscrew that decked the greedy policy 11-12/12 (c57 verdict #3) and that the policy
# never internalised because it FOUGHT the champion's nose-HIGH energy-fighter identity (maximal
# destructive interference). The egg is CONGRUENT with that identity: roll only a MODEST bank toward the
# bandit (angle the loop plane at him) and PULL MAX-g toward a STEEP nose-HIGH pitch target — trade
# airspeed for altitude, pull up and over toward the bandit, come down pointed at him. The lift vector
# stays mostly VERTICAL, so the maneuver climbs (never decks) and it reinforces "point the nose UP, fight
# in the vertical" — the same instinct the champion already has — instead of the descending spiral. This
# is the load-bearing pairing with the reward-v16 anti-spiral penalty (nose-low×banked is now taxed), so
# both the SEED and the GRADIENT push the same way. Sim-validated to climb + reverse from the merge spawn
# before use (_c58_eggcheck.py). Pure scalar function → trivially unit-testable. AGENT action convention:
# [0] aileron (+ = roll RIGHT), [1] elevator (NEGATIVE = pull / nose-UP / +Nz — the inverted-sign FLCS
# interface), [2] rudder (0, FLCS coordinates), [3] throttle (+1 = full). The TRIGGER [4] is left to the
# policy + fire-override — the seed teaches the CIRCLE, not the shot. `sign` = which way to roll (+1
# toward a bandit on the right, ata_az ≥ 0; −1 to the left).
_VE_BANK_RAD = float(np.radians(30.0))  # MODEST bank toward the bandit — keep the pull's lift mostly VERTICAL (egg, not a level/descending turn)
_VE_KROLL = 2.0  # P-gain: bank-angle error (rad) → aileron; full deflection past ~0.5 rad of error
_VE_PITCH_TGT_RAD = float(np.radians(50.0))  # STEEP nose-HIGH pitch target — pull UP into the vertical (the "egg"); commands MAX +Nz from a level/nose-low state
_VE_KPITCH = 3.0  # P-gain: pitch error (rad) → elevator (negative = pull toward the steep climb)


def _vertical_egg_stick(phi_rad: float, theta_rad: float, sign: float) -> np.ndarray:
    """Scripted VERTICAL-EGG reversal, returned as an AGENT-space [ail, elev, rud, thr] (len-4, trigger
    excluded). `phi_rad` = bank (+ = right wing down); `theta_rad` = pitch (< 0 = nose-low); `sign` =
    roll direction toward the bandit (+1 right, −1 left).

    Roll proportionally toward a signed MODEST ~30° bank (angle the loop plane at the bandit while
    keeping the lift vector mostly vertical) AND pull MAX-g toward a STEEP +50° nose-HIGH pitch target:
    elevator = −K·(theta_tgt − theta) → from any level/nose-low state (theta ≪ +50°) this saturates to
    −1.0 (max pull / nose-UP), so the jet zooms up and over — the vertical egg — instead of corkscrewing
    down (the c56/c57 descending-turn signature that decked the greedy policy). It eases the pull as the
    nose approaches +50° and never commands a further dive when already nose-high. Full throttle, rudder 0
    (FLCS coordinates). CONGRUENT with the champion's nose-high identity + paired with the reward-v16
    anti-spiral penalty. Pure & side-effect free → unit-tested in tests/test_curriculum_gate.py."""
    phi_tgt = sign * _VE_BANK_RAD
    ail = max(-1.0, min(1.0, _VE_KROLL * (phi_tgt - phi_rad)))
    elev = max(-1.0, min(1.0, -_VE_KPITCH * (_VE_PITCH_TGT_RAD - theta_rad)))
    return np.asarray([ail, elev, 0.0, 1.0], np.float32)


# --- curriculum-66 LOW-YO-YO energy-conversion stick (the anti-zoom-stall demonstration). ---
_LY_BANK_RAD = float(np.radians(72.0))    # HARD bank toward the bandit — a tight but flyable converting turn
_LY_KROLL = 2.0                           # P-gain: bank error (rad) → aileron
_LY_KPITCH = 2.5                          # P-gain: pitch error (rad) → elevator
_LY_PITCH_LAG_RAD = float(np.radians(-6.0))    # LAGGING/far: nose slightly LOW → hold/gain speed (LowYoYo DOWN; anti-zoom-stall)
_LY_PITCH_CONV_RAD = float(np.radians(12.0))   # CLOSING/near: a GENTLE pull onto him (LowYoYo UP) — NEVER the +50° zoom
_LY_RANGE_KNEE_M = 700.0                  # rng above ⇒ DOWN (accelerate / cut the corner); at/below ⇒ UP (nose onto him)
# --- ROLL-OUT / SETTLE terminal phase (curriculum-66b): the missing HAND-OFF demonstration. The 72°
# corner bank NEVER satisfies the |bank|≤60° convert box, so the pure-corner c66 prior parked M in a
# knife-edge slice it never rolled out of (greedy M over-banked it to ~100° and departed the bubble,
# 0/12 convert). Once M is genuinely in the bandit's rear quarter (aspect small) AND inside the
# finishable range band, roll the wings TOWARD level (well under the 60° gate), ease the throttle to
# bleed the overtake under the 100 m/s low-crossing gate, and hold a gentle nose-on lead — demonstrating
# the SETTLED matched-velocity tail the frozen finisher F needs (the posture RL was never shown before).
_LY_SETTLE_ASPECT_RAD = float(np.radians(55.0))  # inside his rear quarter (aspect ≤ this) ⇒ begin the roll-out
_LY_SETTLE_RANGE_M = 1000.0                       # AND within the finishable band ⇒ settle (else keep cornering)
_LY_SETTLE_BANK_RAD = float(np.radians(35.0))     # ROLL-OUT target: wings toward level, under the 60° gate with margin
_LY_SETTLE_THR = 0.55                             # ease throttle to bleed the overtake under CONVERT_CLOSURE_MAX


def _lowyoyo_stick(phi_rad: float, theta_rad: float, rng_m: float, aspect_rad: float, sign: float) -> np.ndarray:
    """Scripted LOW-YO-YO energy-conversion stick — the OPPOSITE of `_vertical_egg_stick`. Returned as
    AGENT-space [ail, elev, rud, thr] (len-4, trigger excluded). The T3 opponent trace showed the merge
    is a vertical rolling scissors our jet LOSES on energy: it zooms to a nose-HIGH near-stall (222→80
    m/s) while the T3 holds ≥128, so it never nulls the crossing and the merge stays a draw. This stick
    demonstrates KEEPING ENERGY through the converting turn, THEN rolling out to a finishable tail — two
    phases:

      CORNER (default): HARD bank toward the bandit + FULL throttle, nose held NEAR THE HORIZON — slightly
        NOSE-LOW when lagging (rng > knee: accelerate & cut the corner, the LowYoYo DOWN phase) easing to
        a GENTLE pull as the range closes (the LowYoYo UP phase) — never the +50° zoom.
      ROLL-OUT / SETTLE (aspect ≤ _LY_SETTLE_ASPECT_RAD AND rng ≤ _LY_SETTLE_RANGE_M — genuinely behind
        him and in the finishable band): roll the wings TOWARD level (under the |bank|≤60° hand-off gate),
        ease the throttle to bleed the overtake under the low-crossing gate, hold a gentle nose-on lead.
        This is the SETTLED matched-velocity tail the frozen finisher F needs — the demonstration the
        pure-corner prior lacked, so M's exploration is now biased toward the whole convert, not just the
        knife-edge slice it used to depart from.

    `sign` = roll direction toward the bandit (+1 right, −1 left). Pure & side-effect free."""
    if aspect_rad <= _LY_SETTLE_ASPECT_RAD and rng_m <= _LY_SETTLE_RANGE_M:
        phi_tgt = sign * _LY_SETTLE_BANK_RAD
        pitch_tgt = _LY_PITCH_CONV_RAD
        thr = _LY_SETTLE_THR
    else:
        phi_tgt = sign * _LY_BANK_RAD
        pitch_tgt = _LY_PITCH_LAG_RAD if rng_m > _LY_RANGE_KNEE_M else _LY_PITCH_CONV_RAD
        thr = 1.0
    ail = max(-1.0, min(1.0, _LY_KROLL * (phi_tgt - phi_rad)))
    elev = max(-1.0, min(1.0, -_LY_KPITCH * (pitch_tgt - theta_rad)))
    return np.asarray([ail, elev, 0.0, thr], np.float32)


# --- curriculum-70 HIGH-YO-YO closure-management stick (the anti-overshoot demonstration). ---
# User directive (2026-09-01): "Dude you gotta stop the advantageous learning. It must learn to high
# yoyo." The HIGH YO-YO is the rear-quarter BFM that KILLS EXCESS CLOSURE so a converting attacker does
# NOT overshoot (fly out in front of) a hard-turning defender: pull UP out of plane (bank hard INTO the
# turn + nose ABOVE his plane) to trade the surplus airspeed/closure for turning room + potential
# energy, then — once the closure is bled — roll back DOWN onto the LOS and reacquire. Against the
# aggressive T3 (which BREAKS hard and decelerates, [[aggressive-t3-break]]) a co-speed pursuer's
# closure SPIKES at the conversion → overshoot → this is the tool. Fired ONLY inside the hot-closure
# overshoot transient (the doctrine trigger rng<YOYO_RANGE ∧ closure>CLOSURE_HI, same as bfm.HighYoYo /
# Tier3 opponent), so it seeds the missing "pull up + bleed" transitions WITHOUT flooding the buffer
# (the c62 whole-episode-override destabiliser lesson — windowing matters, not the maneuver).
_HY_BANK_UP_RAD = float(np.radians(68.0))     # HARD bank INTO the turn during the UP (bleed) phase
_HY_PITCH_UP_RAD = float(np.radians(24.0))    # nose ABOVE his plane — go high, trade closure for turning room + PE
_HY_THR_UP = 0.70                             # ease throttle to help bleed the surplus closure
_HY_BANK_DOWN_RAD = float(np.radians(45.0))   # roll toward a moderate bank to pull DOWN onto the LOS
_HY_PITCH_DOWN_RAD = float(np.radians(-4.0))  # nose back down onto his plane to reacquire
_HY_CLOSURE_LO = 40.0                         # = constants.CLOSURE_LO (retained for the tests' bled reference)
_HY_CLOSURE_UP = 120.0                        # = constants.CLOSURE_HI; the OVERSHOOT threshold — above ⇒ UP/bleed, below ⇒ DOWN/reacquire
_HY_KROLL = 2.0                               # P-gain: bank error (rad) → aileron
_HY_KPITCH = 2.5                              # P-gain: pitch error (rad) → elevator


def _highyoyo_stick(phi_rad: float, theta_rad: float, closure: float, sign: float) -> np.ndarray:
    """Scripted HIGH-YO-YO closure-management stick, returned as AGENT-space [ail, elev, rud, thr]
    (len-4, trigger excluded). `phi_rad` = bank (+ = right wing down); `theta_rad` = pitch (< 0 =
    nose-low); `closure` = range-rate (m/s, + = closing); `sign` = roll direction toward the bandit
    (+1 right, −1 left). Two phases keyed on closure vs _HY_CLOSURE_UP (= CLOSURE_HI, the doctrine
    OVERSHOOT threshold):

      UP / BLEED (closure > _HY_CLOSURE_UP — genuinely HOT, overshoot imminent): bank HARD into the
        turn (sign·68°) + pull the nose ABOVE his plane (theta_tgt +24°) and EASE the throttle — trade
        the surplus airspeed/closure for turning room and altitude so the jet does NOT fly out in front.
      DOWN / REACQUIRE (closure ≤ _HY_CLOSURE_UP — the normal case): roll toward a moderate bank
        (sign·45°) + bring the nose back DOWN onto the LOS (theta_tgt −4°) at full throttle — come back
        down onto the line of sight to convert.

    c71 FIX (was c70): the splice now fires on the BROAD rng∧ata gate (like the egg/low-yoyo seeds), so
    BOTH phases get demonstrated — UP only on the rare genuine overshoot (closure > CLOSURE_HI), DOWN /
    REACQUIRE the rest of the time (nose back ONTO the LOS). c70 gated the splice on closure > CLOSURE_HI
    yet split the phases at CLOSURE_LO, so ONLY the UP branch ever ran: the policy learned to zoom up and
    away (⟨ata⟩ 86°→158°, dealt 27→0) and never reacquired. Keying the split at CLOSURE_HI makes the UP
    bleed the exception and the nose-on reacquire the demonstrated norm. Pure & side-effect free →
    unit-tested in tests/test_curriculum_gate.py."""
    if closure > _HY_CLOSURE_UP:
        phi_tgt = sign * _HY_BANK_UP_RAD
        pitch_tgt = _HY_PITCH_UP_RAD
        thr = _HY_THR_UP
    else:
        phi_tgt = sign * _HY_BANK_DOWN_RAD
        pitch_tgt = _HY_PITCH_DOWN_RAD
        thr = 1.0
    ail = max(-1.0, min(1.0, _HY_KROLL * (phi_tgt - phi_rad)))
    elev = max(-1.0, min(1.0, -_HY_KPITCH * (pitch_tgt - theta_rad)))
    return np.asarray([ail, elev, 0.0, thr], np.float32)


# --- curriculum-95 LEAD-TURN merge-conversion stick (the first MEASURED merge conversion). ---
# The only scripted maneuver in this program measured to CONVERT a neutral head-on merge into a real gun
# solution ([[merge-leadturn-conversion]]; `_diag_merge_convert_tmp.py` arm `T1000+`, vs the FIXED T3
# with 1500 m of lateral turning room, 8 eps): it reaches the bandit's SIX at aspect 14° / ata 8° with
# `alpha_req` 9° — INSIDE the ~15° usable α cone, where c90/c91 measured 140-146° at the same merge —
# `gun_quality_honest` 0.49 and a predicted lay miss of 1 m against an 8 m hit sphere. Every other arm
# ever measured there, including the greedy c87 champion and the whole `bfm.BFMCoach` doctrine tree,
# reads 0.00 gun quality. THREE phases (`_leadturn_phase`), all three measured to be necessary:
#
#   APPROACH: fly the PREDICTED CLOSEST POINT OF APPROACH to _LT_TURNROOM_M and hold corner speed. The
#     CPA MISS DISTANCE, not the range or the bearing, is the turning-room quantity — it is what decides
#     whether a turning fight exists at all, and a merge flown to a collision course (c90/c91's ±135 m
#     spawn) has none.
#   TURN (latched once inside _LT_COMMIT_RNG_M): a max-rate ONE-CIRCLE lead turn TOWARD his side — hard
#     bank, max Nz gated on roll completion, full throttle.
#   TRACK (once behind him with the nose close): hand the last ~30° to LAG PURSUIT — `bfm._point_at` on a
#     3° trail plus the closure throttle law, i.e. `bfm.LagPursuitTracker.step` verbatim. Measured
#     necessary: without it the blind corner stalls at ata 29° / aspect 47° and climbs out of the arena
#     (`left_bubble` 5/8, dAGL +1782) where the reference arm reaches ata 8° / gq_honest 0.49. The last
#     30° is a TRACKING problem, not a turning one. Deck-guarded in all three phases, as the arm was.
#
# THE PHASE MUST BE LATCHED, and that is measured, not stylistic. The first port kept the stick pure by
# re-deriving the phase from geometry every tick, which is the property that makes the egg/yo-yo seeds
# unit-testable — but BOTH stateless proxies were measured to destroy the conversion, in opposite ways
# (`_diag_merge_convert_tmp.py` arm PROD, 8 eps, same seeds as the latched `T1000+` arm):
#
#   * RANGE ONLY (`rng > _LT_COMMIT_RNG_M`) committed correctly — roll to 84°, 6.9 g, right direction —
#     then REVERTED to APPROACH two seconds later when the pass opened the range back past 1000 m
#     (933 → 1140 m), dropping the pull to nz +1.5 exactly when the turn had to keep coming. ata ran
#     121° → 139° → 155° → 171°: `gq_honest 0.00`, a 169 m lay, `aspMn 85°` vs the latched arm's 14°,
#     i.e. indistinguishable from the do-nothing BASE control.
#   * RANGE **AND** CLOSURE (`rng > _LT_COMMIT_RNG_M and closure > 0`), which looked physically motivated
#     — past the CPA the predicted miss no longer describes a future pass, so keep turning — was WORSE:
#     `timeout 8/8, passes 1.0, tInR 4.9 s, rngMn 875 m, gq_honest 0.00`. Cause: the APPROACH phase
#     deliberately banks AWAY to build turning room, which drops closure through zero transiently, so the
#     test tripped COMMIT at ~2500 m and the jet hard-turned away from a merge it had never entered.
#     Range-only reverts mid-turn; closure-only commits before the merge. There is no stateless split.
#
# So the latch is explicit and lives at the CALL SITE (`_leadturn_committed`, one boolean per episode),
# which keeps the stick itself pure and unit-testable while reproducing the reference arm's behaviour.
# Note the consequence: the latch is never cleared, so a SECOND merge in the same episode is flown
# straight into the corner with no approach. That matches the reference arm and is sane BFM — by then the
# jets are slow and close — but it is a real behavioural difference from the first pass.
#
# The stick takes the CPA for its MAGNITUDE only and its roll direction from the splice's ata_az `_sgn`,
# which is deliberate: the CPA *side* is not usable as a per-tick turn direction, because it passes
# through zero the moment the nose leads his future position and then FLIPS, reversing the turn out of
# the very conversion it was committing to. The diag arm could steer on it only because it LATCHED the
# side once at commit — and its own docstring flags re-deriving a turn sign every tick as the defect
# that made the scripted T3 zoom-climb. `_sgn` is the stateless equivalent: it cannot flip while the
# splice is active, since ata_az can only change sign by passing through the gate's own `ata >
# merge_override_ata_rad` dead zone, which shuts the stick off first. At the commit the two agree
# anyway — the bearing side IS the pass side whenever the miss is large next to the range.
#
# THREE measured facts this encodes, each established one-variable:
#   1. Commit range is decisive and the window is only ~200 m wide. 1500/1200 m → aspect ~40° and no
#      solution; 1000 m → aspect 14° and a 1 m lay; 800 m → blows through (`left_bubble` 8/8 in 58 s).
#      A commit that narrow and that precisely timed inside a 180 s episode is plausibly WHY RL has not
#      found it unaided (c94 held kill 0% for 3M steps warm from c87) — which is what a prior is for.
#   2. ONE-circle beats two-circle. Turning TOWARD his pass side converts; turning AWAY survives
#      beautifully (0 decks, +1042 m, 7 passes) and stays neutral at aspect 54° forever. Hence `sign`
#      always rolls INTO the bandit — there is no two-circle branch to select at runtime.
#   3. Guard the FLIGHT PATH, not the pitch attitude. The first version of this arm commanded 80° bank
#      + max Nz and ZOOM-CLIMBED +4.4 km while bleeding 252→152 m/s: once γ goes positive at 80° of bank
#      the pull is nearly horizontal, so nothing restores the flight path — and an attitude test like
#      `bfm._deck_guard`'s `theta < 10° ⇒ pull` cannot arrest a flight-path excursion at high bank
#      either. Feeding γ to the BANK instead (`bank_tgt = 80° + 2γ`: γ>0 ⇒ roll PAST vertical so the
#      pull drags the nose back down to the horizon; γ<0 ⇒ shallow out) pinned θ at +5-6°, held V at
#      224 m/s on nz 5-6, and produced the conversion. Same class of bug as the T3's `_break` defect.
#
# HAND-OFF is free via that same gate: the conversion ends at ata ≈ 8°, so the gate closes and the POLICY
# flies the finish. That is deliberate — no scripted arm can FINISH (12 rounds / 10-11 HP of the ~50 hits
# a kill needs; the diag's damage came from a `bfm.LagPursuitTracker` hand-off this stick does NOT
# reproduce), and whole-episode overrides are the known destabiliser ([[curriculum-62-bfm-coach-outcome]]).
# So what this seeds is the CONVERSION, not the kill. Pair it with a WIDE window: the measured geometry
# puts ata ≈ 30° at the 3000 m spawn rising to ≈ 53° at the 1000 m commit (asin(room/rng)), and the
# post-pass turn needs ~12 s at 14°/s to bring the nose around while the bandit opens to 2-3 km, so
# --merge-override-range-m 3000 --merge-override-ata-deg 25 is the sane pairing. A 2000 m window was
# measured to hand control back MID-CONVERSION, and c87 then flew away (throttle to idle, 1200 m of
# descent, range ballooning to 3.6 km) — its known head-on mis-specialization. Do NOT narrow the ata
# gate to the diag's own 50° TRACK hand-off: that gates the APPROACH out entirely and leaves nothing to
# build the turning room with. Over-rotating to 25° is fine and probably better, since it hands the
# policy a rear-quarter bandit at ata ≈ 25° and c87's chase is 20/20 from there.
_LT_COMMIT_RNG_M = 1000.0  # THE measured commit range — 1200 m leaves aspect 40°, 800 m blows through
_LT_TURNROOM_M = 800.0  # predicted CPA the approach manages toward (the measured turning room)
_LT_ROOM_K = 2.0  # P-gain: fractional CPA error → fraction of _LT_ROOM_BANK_RAD
_LT_ROOM_BANK_RAD = float(np.radians(45.0))  # approach bank authority (gentle — this is setup, not the corner)
_LT_APPROACH_PITCH_RAD = float(np.radians(-4.0))  # nose slightly LOW while below corner speed — accelerate into the commit
_LT_V_CORNER_TGT = 225.0  # corner-speed target: turn radius scales with V², so arriving at 331 m/s costs 1241 m vs 574 m
_LT_V_BAND_MPS = 15.0  # deadband under _LT_V_CORNER_TGT before the nose drops to accelerate
_LT_THR_K_MPS = 40.0  # airspeed error (m/s) → throttle about 0.5
_LT_TURN_BANK_RAD = float(np.radians(80.0))  # COMMIT: max-rate corner bank into the one-circle turn
_LT_GAMMA_K = 2.0  # flight-path guard: γ error (rad) → bank offset (see fact 3 — NOT fed to the pull)
_LT_BANK_MIN_RAD = float(np.radians(40.0))  # γ<0 (descending) ⇒ shallow out, but never below a real turn
_LT_BANK_MAX_RAD = float(np.radians(115.0))  # γ>0 (climbing) ⇒ roll PAST vertical so the pull comes back down
_LT_V_SUSTAIN_MPS = 190.0  # below this the pull is eased toward the sustained-rate side (don't bleed to a stall)
_LT_V_EASE_FLOOR = 0.45  # never ease the pull below this fraction — a slow jet must still turn
_LT_KROLL = 2.0  # P-gain: bank error (rad) → aileron
_LT_KPITCH = 3.0  # P-gain: pitch error (rad) → elevator (approach phase only)
_LT_TRACK_ASPECT_RAD = float(np.radians(70.0))  # TRACK entry: aspect below this = we are BEHIND him...
_LT_TRACK_ATA_RAD = float(np.radians(50.0))  # ...and the nose is close enough that TRACKING converts
_LT_TRACK_EXIT_ASPECT_RAD = float(np.radians(110.0))  # he got out front pointing back ⇒ a new merge
_LT_TRACK_LAG_RAD = float(np.radians(3.0))  # trail slightly behind him — a stable solution, not a lead overshoot


def _leadturn_phase(phase: str, rng_m: float, ata_rad: float, aspect_rad: float) -> str:
    """APPROACH → TURN → TRACK state machine for the lead turn, ported from the measured arm
    (`_diag_merge_convert_tmp.py` `LeadTurnArm.act`) rather than re-derived. `phase` is the caller's
    previous phase — the ONLY state in any of these exploration seeds, and it is load-bearing:

      APPROACH → TURN  LATCHES the first time the bandit is inside _LT_COMMIT_RNG_M, and never unlatches
        on range. Both stateless phase tests were MEASURED to destroy the conversion (block comment
        above): range-only reverted mid-turn when the pass opened the range back past 1000 m (gq_honest
        0.49 → 0.00, aspMn 14° → 85°), and range∧closure tripped COMMIT at ~2500 m because the approach
        banks away to build turning room, so the jet hard-turned away from a merge it never entered
        (timeout 8/8, rngMn 875 m).
      TURN → TRACK  once we are behind him (aspect < _LT_TRACK_ASPECT_RAD) with the nose close
        (ata < _LT_TRACK_ATA_RAD). Also measured: WITHOUT this phase the blind max-rate corner stalls
        at ata 29° / aspect 47° and climbs out of the arena (`left_bubble` 5/8, dAGL +1782), while the
        reference arm's tracker takes the same geometry to ata 8° / aspect 14° / gq_honest 0.49. The
        last 30° of the conversion is a TRACKING problem, not a turning one.
      TRACK → APPROACH  if he gets back out in front pointing at us (aspect > _LT_TRACK_EXIT_ASPECT_RAD)
        — that is another merge, not a chase, so re-enter the state machine at the top (which also
        clears the TURN latch, exactly as the reference arm does).

    Pure & side-effect free → unit-tested in tests/test_curriculum_gate.py."""
    if phase == "TRACK":
        return "TRACK" if aspect_rad <= _LT_TRACK_EXIT_ASPECT_RAD else "APPROACH"
    if phase == "APPROACH" and rng_m <= _LT_COMMIT_RNG_M:
        phase = "TURN"
    if phase == "TURN" and aspect_rad < _LT_TRACK_ASPECT_RAD and ata_rad < _LT_TRACK_ATA_RAD:
        phase = "TRACK"
    return phase


def _cpa_miss_m(pos_me, vel_me, pos_op, vel_op) -> float:
    """Miss distance at the predicted CLOSEST POINT OF APPROACH of two constant-velocity aircraft, in
    metres. `t < 0` — already past the CPA — collapses to the present separation, so a blown pass
    reports the opening range rather than a stale prediction behind the jet. This is the turning-room
    quantity a lead turn must manage (see `_leadturn_stick`); neither the slant range nor the LOS
    bearing is. Pure & side-effect free → unit-tested in tests/test_curriculum_gate.py."""
    r = np.asarray(pos_op, float) - np.asarray(pos_me, float)
    v = np.asarray(vel_op, float) - np.asarray(vel_me, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-9 else max(0.0, -float(r @ v) / vv)
    return float(np.linalg.norm(r + v * t))


def _leadturn_stick(
    phi_rad: float,
    theta_rad: float,
    gamma_rad: float,
    airspeed: float,
    committed: bool,
    miss_m: float,
    sign: float,
) -> np.ndarray:
    """Scripted ONE-CIRCLE LEAD-TURN merge conversion, returned as AGENT-space [ail, elev, rud, thr]
    (len-4, trigger excluded). `phi_rad` = bank (+ = right wing down); `theta_rad` = pitch ATTITUDE;
    `gamma_rad` = FLIGHT-PATH angle (+ = climbing — the load-bearing input, see the block comment);
    `committed` = the LATCHED phase flag from `_leadturn_committed` (the caller's one bit of per-episode
    state); `miss_m` = predicted CPA miss from `_cpa_miss_m`; `sign` = roll direction toward the bandit
    (+1 right, −1 left), same convention as the egg/yo-yo sticks. Two phases:

      APPROACH (not committed): bank proportionally to the FRACTIONAL CPA error — TOWARD him when the
        miss is too big (cut the corner into the merge), AWAY when it is too small (check away to open
        turning room; a collision-course pass has none at all) — with the nose slightly low while below
        corner speed and throttle closing on _LT_V_CORNER_TGT. Arriving at corner speed is not a nicety:
        turn radius scales with V², so 331 m/s costs 1241 m of radius against 574 at 225.
      COMMIT (latched): roll to a γ-CORRECTED hard bank toward him and pull max Nz — but the pull is
        GATED ON ROLL COMPLETION (`cos(bank_err)`, zero while still rolling) so the g goes into the turn
        rather than into a zoom, and eased below _LT_V_SUSTAIN_MPS so the corner is sustained instead of
        bled to a stall. Full throttle, and it NEVER reverts — an opening range means the pass is behind
        you and the nose still has to come around.

    Pure & side-effect free → unit-tested in tests/test_curriculum_gate.py."""
    if not committed:
        room_err = _LT_ROOM_K * (miss_m - _LT_TURNROOM_M) / _LT_TURNROOM_M
        phi_tgt = sign * max(-1.0, min(1.0, room_err)) * _LT_ROOM_BANK_RAD
        pitch_tgt = (
            _LT_APPROACH_PITCH_RAD if airspeed < _LT_V_CORNER_TGT - _LT_V_BAND_MPS else 0.0
        )
        ail = max(-1.0, min(1.0, _LT_KROLL * (phi_tgt - phi_rad)))
        elev = max(-1.0, min(1.0, -_LT_KPITCH * (pitch_tgt - theta_rad)))
        thr = max(0.0, min(1.0, 0.5 + (_LT_V_CORNER_TGT - airspeed) / _LT_THR_K_MPS))
        return np.asarray([ail, elev, 0.0, thr], np.float32)
    bank_tgt = max(
        _LT_BANK_MIN_RAD,
        min(_LT_BANK_MAX_RAD, _LT_TURN_BANK_RAD + _LT_GAMMA_K * gamma_rad),
    )
    bank_err = sign * bank_tgt - phi_rad
    ail = max(-1.0, min(1.0, _LT_KROLL * bank_err))
    pull = max(float(np.cos(bank_err)), 0.0)  # no g until the roll is in — g while rolling is a zoom
    if airspeed < _LT_V_SUSTAIN_MPS:
        pull *= max(_LT_V_EASE_FLOOR, min(1.0, airspeed / _LT_V_SUSTAIN_MPS))
    return np.asarray([ail, -min(1.0, pull), 0.0, 1.0], np.float32)


# ======================================================================
# Collector process: one env, CPU acting, whole-episode transport.
# ======================================================================
def _collector_loop(
    worker_id: int,
    cfg: TDMPC2Config,
    env_kwargs: dict,
    stages,
    stage_value,
    episode_q,
    weight_q,
    stop_event,
    *,
    seed: int,
    weight_check_every: int,
    fire_override_p0: float = 0.0,
    fire_override_anneal: float = 0.0,
    fire_override_quality: float = 0.6,
    reattack_override_p0: float = 0.0,
    reattack_override_anneal: float = 0.0,
    reattack_override_ata_rad: float = 0.0,
    merge_override_p0: float = 0.0,
    merge_override_anneal: float = 0.0,
    merge_override_range_m: float = 1500.0,
    merge_override_ata_rad: float = 0.0,
    merge_override_mode: str = "egg",
    bfm_override_p0: float = 0.0,
    bfm_override_anneal: float = 0.0,
    expert_p0: float = 0.0,
    expert_anneal: float = 0.0,
    probe_every: int = 0,
    selfplay: bool = False,
    pool_dir: str = "",
    selfplay_recency_bias: float = 0.0,
    selfplay_scripted_prob: float = 0.0,
    rehearse_stage=None,
):
    """Collect episodes against the current curriculum stage and push them to the learner.

    Acts with random actions until the first weight broadcast arrives (the seed phase),
    then with the policy prior (or CPU-MPPI if `cfg.mpc`), refreshing weights between and
    during episodes. Rebuilds its env when the shared stage Value changes (curriculum) —
    each stage carries both an opponent `tier` and a start `separation_m`.

    Guided-fire override (idea 2): early in *each* stage, whenever the agent is looking at a
    good gun solution (`obs[gun_quality] ≥ fire_override_quality`), the trigger is forced on
    with probability `p = fire_override_p0·(1 − steps_in_stage/fire_override_anneal)`, annealing
    to 0 over the stage's opening `fire_override_anneal` control-steps. This seeds the replay
    buffer with *fire-in-WEZ* transitions so the learner can discover that shooting on a
    solution pays — reward-v4 alone left the policy parked at the capped track-draw because
    the safe standing reward beat the EV of an imperfect, self-initiated shot. Off-policy
    native: the forced action is what gets stored (line below), so TD-MPC2 learns from it
    cleanly with no core-loop change. Disabled when `fire_override_p0`/`_anneal` ≤ 0.

    Expert-rollout override (long-range fix): early in *each* stage, whole episodes are flown
    by the scripted `PursuitExpert` (surfaced by the env as `info["expert_action"]`) with
    probability `p = expert_p0·(1 − steps_in_stage/expert_anneal)`. Unlike the per-step
    fire-override — impotent past the gun's effective range, where `gun_quality ≡ 0` so the
    trigger can never fire until the jet has already closed — the expert PRESSES to guns and
    shoots, seeding the buffer with complete close→kill trajectories the stalled prior can't
    generate on its own. These episodes are marked `override_fired` (excluded from the honest
    gate) but train the buffer. SCOPED TO TIERS 1-2 — the target-ahead geometry the pursuit law
    can actually fly; on the reversed tier-3 rungs it hard-decks and is disabled (see the
    use_expert guard below). Disabled entirely when `expert_p0`/`_anneal` ≤ 0.

    BFM-coach exploration override (curriculum-62, the BFM pivot): early in *each* stage, whole
    episodes on the HEAD-ON offensive-posture tier-3 rungs are flown by the doctrine `BFMCoach`
    (bfm.py — MergeApproach→VerticalEgg/reversal→rear-quarter track, fire from the rear quarter only)
    with probability `p = bfm_override_p0·(1 − steps_in_stage/bfm_override_anneal)`. Unlike the
    PursuitExpert — which DECKS at the head-on merge (it hauls its nose flat at an incoming bandit,
    bleeds energy, spirals in) — the coach SURVIVES the merge, keeps its energy, and does textbook
    BFM (validated: 0 deck / 0 loss over 24 head-on eps, but 0 kills — a hand-coded tree can't convert
    a symmetric merge; see bfm-library-coach memory). So it is used ONLY as an exploration/imitation
    PRIOR: it seeds the buffer with SAFE, BFM-literate merge→convert trajectories the deck-prone
    pure-pursuit policy never generates, from which RL (under the reward-v19 OUTCOME reward that DROPS
    the dense pure-pursuit terms — see env `outcome_mode`) discovers the conversion→kill. Supplies the
    FULL len-5 action incl. its own rear-quarter fire, so the fire/reattack/merge per-step overrides are
    suppressed while the coach flies (`and not use_coach`). A fresh `BFMCoach()` per episode (it is
    stateful — holds the active maneuver and re-selects on done). Marked `override_fired` (excluded from
    the honest gate); never on probes. Off when `bfm_override_p0`/`_anneal` ≤ 0.

    Deterministic probe episodes (curriculum-14 honest gate): every `probe_every`-th episode
    (once weights exist) is flown by the DEPLOYABLE policy — `eval_mode=True` (deterministic,
    no exploration noise), no expert, no fire-override — and tagged `probe=True`. These are the
    ONLY episodes the learner's stage gate counts, so advancement measures the greedy policy we
    would actually deploy, over the jittered start distribution, rather than the stochastic
    exploration prior (which the curriculum-13 ground-truth eval proved advanced stages the
    deployable policy scored 0 kills on). Probes still train the buffer. `probe_every`≤0 keeps
    the legacy behaviour (gate on any non-override episode).

    Self-play (curriculum-21): when `selfplay` is set, the env's scripted opponent is REPLACED
    before every reset by a `ScriptedPolicyOpponent` sampled from the pool dir (`_sample_opponent`,
    re-globbed each episode so fresh learner snapshots enter immediately) — i.e. the learner fights
    frozen copies of itself (fictitious self-play). The env is still built by `_make_env` for its
    tier-3 GEOMETRY (a neutral merge); only `env.unwrapped.opponent` is swapped. Expert-rollout
    seeding is inapplicable here (no scripted teacher for the merge — the whole reason self-play
    exists) and does not fire at this geometry.

    ANCHORING (curriculum-24): pure fictitious self-play (c21-c23) drifted OFF the real target — the
    pool converged to a passive mutual-draw equilibrium and the policy, never made to fight the
    scripted bot, REGRESSED (c23 run.pt scored 0 kills and died half the time vs the T3 that the c19
    warm-start killed 4/10, 91 HP / 0 taken). So `selfplay_scripted_prob` reserves a fraction of
    episodes — AND every deterministic probe — for the SCRIPTED tier-3 bot (the geometry's native
    opponent) instead of a pooled self-copy. Anchored probes make the honest gate measure damage vs
    the ACTUAL target, and the mixed-in scripted episodes hold the offense in place while self-copies
    still supply merge/reversal ACM variety. `selfplay_scripted_prob=0` is the old pure-self-play."""
    import traceback

    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight.geometry import GUN_QUALITY_OBS_INDEX, relative

    from ..envs import make_tdmpc_env

    # gun_quality lives at index 29 of a single 31-float frame; index from the END so the
    # override reads the *current* frame's value regardless of --frame-stack (FrameStack
    # concatenates whole frames newest-last, so obs[-2] is always the latest gun_quality).
    gun_q_idx = GUN_QUALITY_OBS_INDEX - C.OBS_DIM
    fire_override_on = fire_override_p0 > 0.0 and fire_override_anneal > 0.0
    reattack_override_on = reattack_override_p0 > 0.0 and reattack_override_anneal > 0.0
    merge_override_on = merge_override_p0 > 0.0 and merge_override_anneal > 0.0
    bfm_override_on = bfm_override_p0 > 0.0 and bfm_override_anneal > 0.0
    expert_on = expert_p0 > 0.0 and expert_anneal > 0.0
    # curriculum-62: lazily import the doctrine coach only when the BFM override is enabled (keeps
    # legacy runs free of the dependency). Fresh instance per episode — see the per-episode block.
    BFMCoach = None
    if bfm_override_on:
        from astrum_dogfight.bfm import BFMCoach

    # curriculum-95: the lead turn's TRACK phase IS `bfm.LagPursuitTracker.step`, so borrow that class's
    # two helpers rather than re-deriving pursuit math here (the reference arm hands off to the class
    # itself; it is effectively stateless — `step` never self-terminates and reads no entry frame).
    _lt_point_at = _lt_deck_guard = None
    if merge_override_on and merge_override_mode == "leadturn":
        from astrum_dogfight.bfm import _deck_guard as _lt_deck_guard
        from astrum_dogfight.bfm import _point_at as _lt_point_at

    # curriculum-57 dedicated REHEARSAL worker: when `rehearse_stage` is set this collector IGNORES the
    # shared stage ladder and flies a FIXED co-heading finish env forever (the champion's ≤55° tracking-
    # kill home turf), feeding those transitions into the shared replay buffer so the merge workers'
    # gradient can't FORGET the deliverable (the c56 collapse: training only on the merge rungs
    # destroyed the finish — 45° 12/12 → 0/12). It NEVER probes (its episodes must not vote the honest
    # head-on gate) and runs NO merge override (co-heading — there is no merge to turn; the fire-
    # override stays on so the finish keeps firing). JSBSim is process-global (one env per process), so
    # this is a SEPARATE collector process, not a 2nd env inside a merge worker. Off when rehearse_stage
    # is None (every legacy run unchanged).
    is_rehearse = rehearse_stage is not None
    if is_rehearse:
        merge_override_on = False
        bfm_override_on = False  # the rehearsal worker flies the co-heading finish (no merge to turn)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)  # each worker is one env; don't let BLAS oversubscribe cores

    # Self-play: the opponent is a pooled copy of the learner, re-sampled per episode. The
    # opponent's own frame-stack must match the env's (its prior was trained at this frame_stack).
    sp_frame_stack = int(env_kwargs.get("frame_stack", 1))
    opp_rng = np.random.default_rng(seed * 7919 + 17)

    ccfg = dataclasses.replace(cfg, device="cpu")
    agent = TDMPC2(ccfg, torch.device("cpu"))
    agent.model.eval()

    # Distinct jitter seed per (worker, stage) so collectors fly independent jittered starts
    # and a stage revisit doesn't replay the same geometry stream (only used when jitter>0).
    def _make_env(stage, stage_idx):
        # The learner's perch (--me-dalt-ft, curriculum-79), the block altitude (--merge-alt-ft,
        # curriculum-91) and the merge lateral separation (--merge-lateral-m, curriculum-92) are
        # RUN-level knobs that only `tier3_scenario` accepts, so they ride in as
        # sentinels in env_kwargs and must be filtered out of the generic splat below — the tier-1/2
        # factories reject unknown kwargs. `stage_env_extra` decides where they actually apply.
        me_dalt_ft = env_kwargs.get("me_dalt_ft", 0.0)
        merge_alt_ft = env_kwargs.get("merge_alt_ft", 0.0)
        merge_alt_ft_max = env_kwargs.get("merge_alt_ft_max", 0.0)
        merge_lateral_m = env_kwargs.get("merge_lateral_m", 0.0)
        passthru = {
            k: v
            for k, v in env_kwargs.items()
            if k
            not in ("me_dalt_ft", "merge_alt_ft", "merge_alt_ft_max", "merge_lateral_m")
        }
        # A stage may pin its OWN episode clock (Stage.max_time_s) instead of inheriting the
        # run-level --max-time-s. The rehearsal anchor uses this to stay at the 60 s episodes
        # the chase skill was learned at while the merge workers run long.
        if stage.max_time_s > 0.0:
            passthru["max_time_s"] = stage.max_time_s
        # Which per-stage scenario kwargs apply (and which run-level knobs the anchor is shielded
        # from) is a pure decision — see `stage_env_extra`, where it is unit-tested.
        extra = stage_env_extra(
            stage,
            is_rehearse=is_rehearse,
            me_dalt_ft=me_dalt_ft,
            merge_alt_ft=merge_alt_ft,
            merge_alt_ft_max=merge_alt_ft_max,
            merge_lateral_m=merge_lateral_m,
        )
        return make_tdmpc_env(
            stage.tier,
            separation_m=stage.separation_m,
            timeout_penalty=stage.timeout_penalty,
            jitter_seed=(seed * 100003 + stage_idx),
            **extra,
            **passthru,
        )

    env = None
    try:
        cur_stage = int(stage_value.value)
        # A rehearsal worker pins its own fixed stage (co-heading finish) and never follows the ladder.
        st = rehearse_stage if is_rehearse else stages[cur_stage]
        env = _make_env(st, cur_stage)
        # Self-play anchoring: the opponent the geometry natively builds IS the scripted tier-3 bot
        # (opponent_tier 0). Hold a reference so a fraction of episodes (and every probe) can be
        # routed to the REAL target instead of a pooled self-copy (see the per-episode swap below).
        scripted_opp = getattr(env.unwrapped, "opponent", None)
        # Seed only the first reset of each env, then let it advance its own RNG — the
        # single-env trainer does the same (train.py), and the env treats reset(seed=)
        # as a no-op anyway, so re-seeding every episode is neither needed nor free.
        fresh_env = True
        have_weights = False
        ep_seed = seed
        steps_in_stage = 0  # control-steps at the current stage; drives the fire-override anneal
        ep_idx = 0  # episodes this worker has started; drives the deterministic-probe cadence

        while not stop_event.is_set():
            want_stage = int(stage_value.value)
            if not is_rehearse and want_stage != cur_stage:
                cur_stage = want_stage
                st = stages[cur_stage]
                env.close()
                env = _make_env(st, cur_stage)
                scripted_opp = getattr(
                    env.unwrapped, "opponent", None
                )  # re-capture for the new geometry
                fresh_env = True
                steps_in_stage = 0  # a new scenario → refresh the guided-fire exploration burst
            have_weights = _load_latest_weights(agent, weight_q) or have_weights

            # Probe cadence — computed BEFORE the opponent swap so an anchored probe can be routed to
            # the scripted target. Every probe_every-th episode (once weights exist) is flown greedy
            # and is the ONLY episode the gate counts (ep_idx is advanced below, after reset).
            is_probe = (
                not is_rehearse
                and probe_every > 0
                and have_weights
                and (ep_idx % probe_every == 0)
            )

            # Self-play: choose this episode's opponent and swap it in BEFORE reset (reset() calls
            # opponent.reset(state)). ANCHORING (curriculum-24): with prob selfplay_scripted_prob —
            # and on EVERY probe — the episode faces the SCRIPTED tier-3 bot (the geometry's native
            # opponent, captured above) instead of a pooled self-copy, so the gate measures damage vs
            # the REAL target and the offense stays anchored (pure self-play regressed off it; see the
            # docstring). Otherwise sample a frozen copy of the learner from the pool. A None sample
            # (empty pool at startup, or a load racing a snapshot write) keeps the previous opponent.
            if selfplay:
                face_scripted = is_probe or (opp_rng.random() < selfplay_scripted_prob)
                if face_scripted and scripted_opp is not None:
                    try:
                        env.unwrapped.opponent = scripted_opp
                    except Exception:
                        pass
                else:
                    new_opp = _sample_opponent(
                        pool_dir, sp_frame_stack, selfplay_recency_bias, opp_rng
                    )
                    if new_opp is not None:
                        try:
                            env.unwrapped.opponent = new_opp
                        except Exception:
                            pass

            if fresh_env:
                obs, info = env.reset(seed=ep_seed)
                ep_seed += 1
                fresh_env = False
            else:
                obs, info = env.reset()
            obs_hist = [np.asarray(obs, np.float32)]
            acts, rews, terms, truncs = [], [], [], []
            ep_return = 0.0
            t0 = True
            interrupted = False
            last_event = None  # terminal tag (kill/killed/trade/deck/bubble) for the win-rate gate
            # Baseline opponent HP for the DAMAGE gate metric: read once post-reset, re-read at
            # episode end → HP dealt. `unwrapped` reaches the DogfightGymEnv through the
            # action-repeat / frame-stack wrappers (the same access eval_policy uses); the env's
            # AState is refreshed each step, so we fetch by id at both ends rather than hold a ref.
            # curriculum-27: also baseline the LEARNER's HP so the episode can report HP TAKEN,
            # the term the pure-damage gate was blind to (it advanced net-losing policies).
            _base, _opp_id, _lnr_id = None, -1, -1
            opp_hp0, lnr_hp0 = float("nan"), float("nan")
            probe_aspect = None  # this probe's START aspect (deg) for the c29 honest frontier gate
            try:
                _base = env.unwrapped
                _opp_id = _base.opponent_id
                _lnr_id = _base.learner_id
                opp_hp0 = float(_base._states[_opp_id].health)
                lnr_hp0 = float(_base._states[_lnr_id].health)
                if is_probe:
                    # curriculum-29 honest FRONTIER gate: record the bandit's start bearing off
                    # our nose (≈ the scenario aspect_deg after jitter, since both jets spawn
                    # heading north). The learner votes ONLY probes whose start aspect falls in
                    # the band's hard outer slice, so a base-dominated band average can no longer
                    # mask an unmastered frontier (the c28 lie) — see _in_frontier / ingest.
                    _rg0 = relative(_base._states[_lnr_id], _base._states[_opp_id])
                    probe_aspect = float(np.degrees(_rg0.ata))
            except Exception:
                _base = None
            override_fired = False  # did an exploration override touch this episode's actions?

            # Expert-rollout override: decide ONCE per episode whether the scripted PursuitExpert
            # flies the whole thing (annealing p over the stage's opening `expert_anneal` steps).
            # SCOPED TO TIERS 1-2 (target-ahead geometry): the PursuitExpert is a pure lead-pursuit
            # offense closer, validated to close→kill the T1 dummy (≤1600 m) and to close-and-fire on
            # the T2 breaker. On the tier-3 rungs the geometry is REVERSED (offensive posture spawns
            # the bot on the learner's SIX — the eval/acceptance geometry), a guns-defense-into-
            # reversal problem the pursuit law cannot fly: it just hauls its nose at a bandit behind
            # it, bleeds energy, and flies into the deck (diagnosed: 0 HP dealt, hard_deck at 1100/
            # 1600 m). Seeding that would only poison the T3 buffer, so the expert is disabled there;
            # the learner converts those rungs from its T1/T2 offense + reward shaping + backstop.
            # Deterministic probe: every probe_every-th episode (once weights exist) is flown by
            # the greedy deployable policy — no expert, no fire-override — and is the ONLY episode
            # the honest gate counts (see docstring). Its outcome is what we would actually deploy.
            # `is_probe` was decided above (before the opponent swap, so anchored probes hit the
            # scripted target); here we only advance the per-worker episode cadence counter.
            ep_idx += 1

            expert_act = info.get("expert_action")  # env's expert action for the current obs
            use_expert = False
            # The PursuitExpert is a lead-pursuit tracker: valid wherever the target starts AHEAD —
            # tiers 1/2, AND Tier-3 rungs flown in the "defensive" posture (bot ahead, learner on its
            # tail = an OFFENSIVE chase). It is NOT flyable on "offensive"-posture T3 rungs (bot on the
            # SIX): there it just hauls its nose at a bandit behind it and decks (probed: 0 HP, hard_deck).
            # NB its own trigger stays shut on the jinker (4° cone) — the fire-override below supplies the
            # spray, exactly the expert-stick + liberal-fire combo that killed the T3 in _diag_t3_spray.py.
            # curriculum-20: ALSO flyable on a tier-3-geometry merge rung whose OPPONENT is a straight
            # Tier1 dummy in the forward hemisphere (aspect ≤ 90) — a beam merge vs a NON-evading target
            # IS convertible by lead pursuit and does not deck (a probe showed offensive-posture never
            # decks at aspect ≤ 90). It stays OFF vs the T2/T3 evader off-axis (probed: lead pursuit lands
            # ~0 HP there and cannot reverse), so those rungs learn from warm-start + shaping + fire-override.
            expert_flyable = (
                st.tier in (1, 2)
                or (st.tier == 3 and st.posture == "defensive")
                or (
                    st.tier == 3
                    and st.posture == "offensive"
                    and st.opponent_tier == 1
                    and st.aspect_deg <= 90.0
                )
            )
            if expert_on and not is_probe and expert_flyable and steps_in_stage < expert_anneal:
                p = expert_p0 * (1.0 - steps_in_stage / expert_anneal)
                use_expert = np.random.rand() < p
            if use_expert:
                override_fired = True  # expert-assisted → excluded from the honest gate

            # BFM-coach exploration override (curriculum-62): decide ONCE per episode whether the
            # doctrine coach flies the whole thing (annealing p). SCOPED to the offensive-posture
            # tier-3 HEAD-ON rungs — the merge geometry the coach was built/validated for and where the
            # PursuitExpert decks. A fresh BFMCoach() per episode (it is stateful). It supplies the full
            # len-5 action incl. its own rear-quarter fire, so it takes precedence over the pursuit
            # expert AND suppresses the per-step fire/reattack/merge splices while it flies.
            use_coach = False
            coach = None
            coach_flyable = st.tier == 3 and st.posture == "offensive"
            if (
                bfm_override_on
                and not is_probe
                and coach_flyable
                and steps_in_stage < bfm_override_anneal
            ):
                p = bfm_override_p0 * (1.0 - steps_in_stage / bfm_override_anneal)
                use_coach = np.random.rand() < p
            if use_coach:
                coach = BFMCoach()
                use_expert = False  # coach takes precedence over the (deck-prone) pursuit expert here
                override_fired = True  # BFM-coach demo → excluded from the honest gate

            # curriculum-95: the lead turn's one piece of per-EPISODE state. The sticks are pure, but the
            # PHASE cannot be re-derived from geometry — both stateless tests were measured to break the
            # conversion (see the block comment). Reset here, advanced in the splice below.
            lt_phase = "APPROACH"
            # ...and its coin, drawn ONCE per episode like the coach's. egg/lowyoyo/highyoyo are
            # memoryless single-instant sticks, so their per-STEP draw degrades gracefully: a skipped
            # tick just means the policy flies that instant of the same maneuver. The lead turn is the
            # opposite — a ~12 s SUSTAINED corner whose latched phase machine only advances on drawn
            # ticks, so at p<1 a per-step draw interleaves the policy's own (fly-away) stick mid-corner,
            # stalls the latch, and the conversion never completes. Drawing per episode keeps every
            # demonstration coherent at ANY p. This is still a SPLICE, not c62's whole-episode flood:
            # the rng ∧ ata WINDOW below is unchanged, so the policy flies everything outside the merge.
            lt_episode_on = True
            if (
                merge_override_on
                and merge_override_mode == "leadturn"
                and not is_probe
                and steps_in_stage < merge_override_anneal
            ):
                lt_episode_on = np.random.rand() < merge_override_p0 * (
                    1.0 - steps_in_stage / merge_override_anneal
                )

            while True:
                if use_coach and coach is not None and _base is not None:
                    # curriculum-62: the doctrine coach flies from the LIVE env states (refreshed each
                    # step). Returns the full len-5 [ail,elev,rud,thr,fire] incl. its own rear-quarter
                    # fire, so the per-step overrides below are suppressed (`and not use_coach`).
                    _me_c = _base._states[_lnr_id]
                    _bd_c = _base._states[_opp_id]
                    action = np.asarray(coach.act(_me_c, relative(_me_c, _bd_c)), np.float32)
                elif use_expert and expert_act is not None:
                    # .copy() so the fire-override below can mutate the trigger without touching
                    # the env's shared expert_action buffer.
                    action = np.array(expert_act, np.float32)  # scripted expert flies (DAgger seed)
                elif have_weights:
                    action = agent.act(obs, t0=t0, eval_mode=is_probe).numpy()
                else:
                    action = env.action_space.sample()
                t0 = False
                # Guided-fire override: force the trigger on a good solution, annealing p→0 over the
                # stage's opening `fire_override_anneal` control-steps (see docstring). Applies to BOTH
                # policy AND expert rollouts (curriculum-17): the PursuitExpert's own 4° fire cone never
                # opens on the jinking T3, so on the T3-offense rungs the expert's decent tracking STICK
                # only produces kill/damage completions when this override supplies the spray trigger
                # (the exact combo _diag_t3_spray.py used to kill the T3). Still skipped on probes, which
                # must fly the unassisted greedy policy so the honest gate measures what we deploy.
                if (
                    not is_probe
                    and not use_coach
                    and fire_override_on
                    and steps_in_stage < fire_override_anneal
                ):
                    p = fire_override_p0 * (1.0 - steps_in_stage / fire_override_anneal)
                    if obs[gun_q_idx] >= fire_override_quality and np.random.rand() < p:
                        action[4] = 1.0  # max agent-trigger → env-trigger 1.0 ≥ 0.5 → fire
                        override_fired = True  # this episode's outcome is override-ASSISTED
                # Re-attack override (curriculum-43): the ≥55° wall is an UNDER-EXPLORED overshoot-
                # recovery skill, not a reward/model gap (grounded reward favours the kill ≈+260 vs
                # ≈−48 for a coast; MPC over the learned world model is WORSE — decks + mag-dumps; 4
                # reward iterations failed). Finish-diag on the c41 killer: it lands a crushing first
                # pass (bandit → ~18 HP), OVERSHOOTS, gun_quality collapses, then COASTS to timeout
                # firing 0 rounds — it never turns back. This splices the env's PursuitExpert STICK (a
                # pure lead-pursuit reversal, already in AGENT space in info["expert_action"]) whenever
                # the agent has overshot a LIVE bandit it can still shoot — range OPENING (closure<0) ∧
                # bandit well off the nose (ata > threshold) ∧ alive ∧ ammo left — with an annealing p,
                # seeding the buffer with the missing "chase it down after losing it" transitions so the
                # value fn learns their worth (exactly how the fire-override seeded fire-in-WEZ demos).
                # STICK-ONLY [0:4]: the trigger stays with the policy + fire-override, so learned fire
                # discipline is preserved and the reverse itself carries no shot (ata high → gun_q≈0).
                # Excluded from the honest gate (override_fired); never on probes. Off at p0/anneal ≤ 0.
                if (
                    not is_probe
                    and not use_coach
                    and reattack_override_on
                    and expert_act is not None
                    and _base is not None
                    and steps_in_stage < reattack_override_anneal
                ):
                    p = reattack_override_p0 * (1.0 - steps_in_stage / reattack_override_anneal)
                    if np.random.rand() < p:
                        try:
                            _me_s = _base._states[_lnr_id]
                            _bd_s = _base._states[_opp_id]
                        except Exception:
                            _me_s = _bd_s = None
                        if (
                            _me_s is not None
                            and _bd_s is not None
                            and _bd_s.alive
                            and _me_s.ammo > 0
                        ):
                            _rgo = relative(_me_s, _bd_s)
                            if _rgo.closure < 0.0 and _rgo.ata > reattack_override_ata_rad:
                                action[:4] = np.asarray(expert_act, np.float32)[:4]
                                override_fired = True  # re-attack demo → excluded from the honest gate
                # Merge-turn override (curriculum-56): imitation-seed the MERGE CIRCLE. The champion
                # FREEZES at the merge (fires spent 0, holds attitude, drifts to a deck/timeout) — it
                # carries a 5 s co-heading kill but NO turning circle, so RL never generates the merge
                # experience it must learn from (5th warm-start collapse c49/c50/c51/c53/c55). This
                # SPLICES a SCRIPTED VERTICAL EGG (_vertical_egg_stick: roll a MODEST ~30° bank toward
                # the bandit + pull MAX-g to a steep +50° nose-HIGH pitch — c58 replaced c56/c57's
                # descending horizontal break, the nose-low corkscrew that decked the greedy policy and
                # fought the champion's nose-high identity) whenever a LIVE bandit sits well off the nose
                # within merge range (rng < merge_override_range_m ∧ ata > merge_override_ata_rad), with
                # an annealing p — building the "pull UP into the vertical and reverse onto the target"
                # transitions the frozen prior cannot produce on its own. Unlike the re-attack/expert
                # overrides it does NOT use info["expert_action"] (the PursuitExpert DECKS at the
                # merge), so it works on the offensive-posture HEAD-ON rungs where the expert is
                # disabled. STICK-ONLY [0:4] — the trigger stays with the policy + fire-override (the
                # seed teaches the CIRCLE, not the shot; ata is high there so gun_q≈0 anyway). Excluded
                # from the honest gate (override_fired); never on probes. Off at p0/anneal ≤ 0. The
                # egg needs no altitude input, so only phi/theta/sign feed the stick.
                if (
                    not is_probe
                    and not use_coach
                    and merge_override_on
                    and _base is not None
                    and steps_in_stage < merge_override_anneal
                ):
                    if merge_override_mode == "leadturn":
                        _draw = lt_episode_on  # per-EPISODE coin (see above) — coherence, not chance
                    else:
                        _draw = np.random.rand() < merge_override_p0 * (
                            1.0 - steps_in_stage / merge_override_anneal
                        )
                    if _draw:
                        try:
                            _me_s = _base._states[_lnr_id]
                            _bd_s = _base._states[_opp_id]
                        except Exception:
                            _me_s = _bd_s = None
                        if _me_s is not None and _bd_s is not None and _bd_s.alive:
                            _rgm = relative(_me_s, _bd_s)
                            if (
                                _rgm.rng < merge_override_range_m
                                and _rgm.ata > merge_override_ata_rad
                            ):
                                _sgn = 1.0 if _rgm.ata_az >= 0.0 else -1.0
                                if merge_override_mode == "highyoyo":
                                    # c71 (fixes c70): HIGH-YO-YO closure-management seed on the SAME
                                    # broad rng∧ata gate as the egg/low-yoyo seeds — the stick's own
                                    # closure logic picks the phase: UP/bleed (bank up + nose-high, ease
                                    # throttle) ONLY on a genuine overshoot (closure > CLOSURE_HI), else
                                    # DOWN/REACQUIRE (roll onto the LOS, full throttle). c70 gated the
                                    # splice itself on closure > CLOSURE_HI, so ONLY the UP branch ever
                                    # ran → the policy zoomed up and away (⟨ata⟩ 86°→158°, dealt 27→0)
                                    # and never reacquired. Demonstrating BOTH phases (UP the exception,
                                    # nose-on reacquire the norm) teaches the COMPLETE maneuver. Still
                                    # windowed (probabilistic p0 + anneal + rng∧ata gate), NOT the c62
                                    # whole-episode flood; the DOWN stick is a benign reacquiring turn.
                                    action[:4] = _highyoyo_stick(
                                        float(_me_s.phi),
                                        float(_me_s.theta),
                                        float(_rgm.closure),
                                        _sgn,
                                    )
                                    override_fired = True  # high-yoyo demo → excluded from the honest gate
                                elif merge_override_mode == "leadturn":
                                    # c95: the MEASURED one-circle lead turn — the only scripted arm
                                    # that converts a neutral head-on merge to a real gun solution
                                    # (aspect 14°, alpha_req 9°, gq_honest 0.49, a 1 m predicted lay,
                                    # against 0.00 for every other arm ever measured there, including
                                    # greedy c87 and the whole BFM coach doctrine tree). It needs two
                                    # inputs the other sticks don't: the predicted CPA MISS (the
                                    # turning-room magnitude the approach flies to) and the FLIGHT-PATH
                                    # angle γ, which is fed to the BANK — the correction that stopped
                                    # this arm zoom-climbing +4.4 km. The roll direction is the ordinary
                                    # ata_az `_sgn`; see the block comment for why the CPA side is NOT
                                    # usable per-tick. It also needs the LATCHED PHASE — the one piece of
                                    # state in any of these seeds, and measured twice over to be
                                    # indispensable. Hand-off is the gate's job: the conversion ends at
                                    # ata ≈ 8°, so the `ata > merge_override_ata_rad` test closes and the
                                    # policy flies the FINISH, which no scripted arm can do anyway.
                                    _vm = np.asarray(_me_s.vel, float)
                                    _sp = float(np.linalg.norm(_vm))
                                    _gam = (
                                        float(np.arcsin(np.clip(_vm[2] / _sp, -1.0, 1.0)))
                                        if _sp > 1e-6
                                        else 0.0
                                    )
                                    lt_phase = _leadturn_phase(
                                        lt_phase,
                                        float(_rgm.rng),
                                        float(_rgm.ata),
                                        float(_rgm.aspect),
                                    )
                                    if lt_phase == "TRACK":
                                        # `bfm.LagPursuitTracker.step` verbatim: lag pursuit on a 3°
                                        # trail, throttle bleeding excess closure without coasting to a
                                        # stall. The class itself is stateful only in a step counter it
                                        # never acts on, so calling its two helpers is the faithful port.
                                        _ail, _elev = _lt_point_at(
                                            _rgm,
                                            pull_gain=C.K_PURSUE_PULL,
                                            lag_rad=_LT_TRACK_LAG_RAD,
                                        )
                                        _thr = max(
                                            0.25,
                                            min(1.0, 0.85 - max(0.0, float(_rgm.closure)) / 300.0),
                                        )
                                    else:
                                        _ail, _elev, _, _thr = _leadturn_stick(
                                            float(_me_s.phi),
                                            float(_me_s.theta),
                                            _gam,
                                            float(_me_s.airspeed),
                                            lt_phase == "TURN",
                                            _cpa_miss_m(
                                                _me_s.pos, _me_s.vel, _bd_s.pos, _bd_s.vel
                                            ),
                                            _sgn,
                                        )
                                    # deck-guarded in ALL phases, as the measured arm was — the
                                    # conversion spends the whole altitude budget (dAGL −3308).
                                    _ail, _elev = _lt_deck_guard(_me_s, _ail, _elev)
                                    action[:4] = np.asarray(
                                        [_ail, _elev, 0.0, _thr], np.float32
                                    )
                                    override_fired = True  # lead-turn demo → excluded from the honest gate
                                elif merge_override_mode == "lowyoyo":
                                    # c66: energy-keeping LOW-YO-YO conversion (anti-zoom-stall) —
                                    # the opposite of the nose-high egg, for a warm-from-BASELINE M
                                    # that already flies the scissors but bleeds to a stall.
                                    action[:4] = _lowyoyo_stick(
                                        float(_me_s.phi),
                                        float(_me_s.theta),
                                        float(_rgm.rng),
                                        float(_rgm.aspect),
                                        _sgn,
                                    )
                                    override_fired = True  # merge-turn demo → excluded from the honest gate
                                else:
                                    action[:4] = _vertical_egg_stick(
                                        float(_me_s.phi),
                                        float(_me_s.theta),
                                        _sgn,
                                    )
                                    override_fired = True  # merge-turn demo → excluded from the honest gate
                obs, reward, terminated, truncated, info = env.step(action)
                expert_act = info.get("expert_action")  # expert action for the NEW obs
                steps_in_stage += 1
                if info.get("event"):
                    last_event = info["event"]  # last recognized tag wins (as eval_policy does)

                obs_hist.append(np.asarray(obs, np.float32))
                acts.append(np.asarray(action, np.float32))
                rews.append(float(reward))
                terms.append(float(terminated))
                truncs.append(float(truncated))
                ep_return += float(reward)

                if len(acts) % weight_check_every == 0:
                    have_weights = _load_latest_weights(agent, weight_q) or have_weights
                if terminated or truncated:
                    break
                if stop_event.is_set():
                    interrupted = True
                    break

            if interrupted or not acts:
                continue  # drop partial episodes on shutdown — never send a non-terminal tail

            try:
                dmg_dealt = (
                    opp_hp0 - float(_base._states[_opp_id].health)
                    if _base is not None
                    else float("nan")
                )
                dmg_taken = (
                    lnr_hp0 - float(_base._states[_lnr_id].health)
                    if _base is not None
                    else float("nan")
                )
            except Exception:
                dmg_dealt = dmg_taken = float("nan")

            episode = {
                "obs": np.stack(obs_hist),  # (T+1, obs_dim)
                "action": np.stack(acts),  # (T, act_dim)
                "reward": np.asarray(rews, np.float32),
                "terminated": np.asarray(terms, np.float32),
                "truncated": np.asarray(truncs, np.float32),
                "ret": ep_return,
                "tier": st.tier,
                "stage": cur_stage,
                "worker": worker_id,
                "event": last_event,  # for the learner's performance gate
                "dmg_dealt": dmg_dealt,  # opponent HP removed this episode → the DAMAGE gate metric
                "dmg_taken": dmg_taken,  # learner HP lost this episode → the NET gate metric (c27)
                "override_fired": override_fired,  # exclude assisted episodes from the gate
                "probe": is_probe,  # deterministic deployable-policy episode → the honest gate
                "probe_aspect": probe_aspect,  # c29: start aspect (deg) → frontier-slice gate vote
            }
            while not stop_event.is_set():
                try:
                    episode_q.put(episode, timeout=0.5)
                    break
                except queue.Full:
                    continue  # backpressure: learner is busy (e.g. pretraining burst)
    except Exception:
        # A silent collector death would leave the learner spinning with no data; make
        # it loud (to stdout, which survives stderr filtering) so the cause is visible.
        print(f"[collector {worker_id}] CRASHED:\n{traceback.format_exc()}", flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        # Don't let unflushed queue writes keep this process alive at exit (a classic
        # mp.Queue feeder-thread hang); the learner tolerates a dropped trailing episode.
        episode_q.cancel_join_thread()
        weight_q.cancel_join_thread()


# ======================================================================
# Learner (this process): ingest episodes, update, broadcast, checkpoint.
# ======================================================================
def _drain(episode_q, max_items: int) -> list:
    out = []
    for _ in range(max_items):
        try:
            out.append(episode_q.get_nowait())
        except queue.Empty:
            break
    return out


def _milestone_path(save: str, stage_idx: int, stage) -> str:
    """Per-stage milestone checkpoint path, e.g. run_stage03_t1_1600m.pt."""
    p = Path(save)
    tag = f"stage{stage_idx:02d}_t{stage.tier}_{int(stage.separation_m)}m"
    return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))


# ======================================================================
# Curriculum as a ladder of (tier, separation, gate-metric) STAGES.
# ----------------------------------------------------------------------
# A rung is not just an opponent tier but a *scenario distance*, because separation is
# what actually gates the skill: for the offensive tiers (1/2) it is how far the learner
# must CLOSE (small = a handed point-blank kill, large = a real chase), while for the
# defensive tier (3) it is how close the bot opens on the six (small = a lethal ambush,
# large = room to convert). So the ladder climbs close→far on offense (learn to run the
# bandit down) then far→near on defense (convert with room, then survive the point-blank
# break). See docs/dogfight-tdmpc2-design.md §8.7.
# ======================================================================
LOSS_EVENTS = frozenset({"death", "killed", "trade", "hard_deck", "left_bubble"})


def _is_win(event, metric: str) -> bool:
    """Did this episode clear an event-based stage bar? `kill` counts only a kill; `survive`
    counts any non-losing terminal (a timeout draw or a kill — anything that is not death/
    trade/deck/bubble-exit), because at point-blank defensive stages defeating the shot is the
    milestone and a kill may not be physically reachable. `convert` (reward-v17, the two-policy
    merge-converter M) counts ONLY the rear-quarter-conversion terminal — a KILL is F's job, not
    M's, so it does NOT clear M's bar; only reaching the hand-off geometry does. (The `damage`
    metric is numeric and is evaluated by `_stage_win`, which reads the episode's HP-dealt.)
    Pure → unit-tested."""
    if metric == "survive":
        return event not in LOSS_EVENTS
    if metric == "convert":
        return event == "conversion"
    return event == "kill"


def _stage_win(ep, stage) -> bool:
    """Did this episode clear `stage`'s success bar? Dispatches on the stage metric. The
    `damage` metric (feasibility-driven, curriculum-14) counts an episode a win iff it removed
    at least `stage.damage_threshold` opponent HP — an aggression-forcing bar for the maneuvering
    Tier-2 opponent, whose evasion + the scripted expert's tight fire cone make a clean KILL
    un-teachable under start jitter (probed: expert 0/3 kills even at 400 m) yet leave 40-50 HP
    of chip damage readily gettable and a kill physically feasible; a bare `survive` gate there
    would instead reward passive drift. `kill`/`survive` defer to `_is_win` on the event tag.

    The `net` metric (curriculum-27) is the honest engagement outcome: a win iff the episode
    KILLED the bot, OR ended with NO losing terminal (no death/trade/deck/bubble-exit) AND net
    HP ahead by ≥ `stage.net_threshold` (HP dealt − HP taken). Unlike `damage`, it cannot be
    passed by chipping while being shredded — the exact hole that let the c26 damage-gate certify
    a net-losing policy. Requires the collector to record `dmg_taken` alongside `dmg_dealt`."""
    if stage.metric == "damage":
        dmg = ep.get("dmg_dealt")
        return dmg is not None and float(dmg) >= stage.damage_threshold
    if stage.metric == "net":
        event = ep.get("event")
        if event == "kill":
            return True
        if event in LOSS_EVENTS:
            return False
        dealt, taken = ep.get("dmg_dealt"), ep.get("dmg_taken")
        if dealt is None or taken is None:
            return False
        return float(dealt) - float(taken) >= stage.net_threshold
    return _is_win(ep.get("event"), stage.metric)


@dataclasses.dataclass(frozen=True)
class Stage:
    tier: int
    separation_m: float
    metric: str  # "kill" | "survive" | "damage" | "net"
    winrate: float  # gate threshold for THIS stage (0 disables the gate here)
    backstop: int  # collected-step floor that FORCES entry to this stage (gate-slow safety net)
    timeout_penalty: float = (
        0.0  # reward-v7: clock-timeout penalty for THIS stage (0 on survive/damage rungs)
    )
    damage_threshold: float = 0.0  # HP-dealt bar for the "damage" metric (ignored otherwise)
    # curriculum-17: starting GEOMETRY for a Tier-3 rung (ignored for tiers 1/2). "offensive" =
    # the bot opens on the learner's SIX (tier3_scenario, the defensive/acceptance fight, gated
    # on survive); "defensive" = the bot starts AHEAD with the learner on its tail (tier1_scenario,
    # an OFFENSIVE chase, gated on damage/kill). The spray-ceiling probe (_diag_t3_spray.py) proved
    # the jinking T3 IS gunnable from the offensive start (kill + 90-100 HP passes) — c16 only ever
    # trained the "offensive"-posture SURVIVE fight, which is why its policy never learned to attack.
    posture: str = "offensive"
    # curriculum-18 (reversal ladder): starting BEARING of the bot around the learner for a Tier-3
    # "offensive"-posture rung (ignored for tiers 1/2 and for "defensive" posture). 0°=dead ahead
    # (learner on the bot's six = the c17-mastered OFFENSE), 90°=beam (neutral merge), 180°=dead six
    # (the acceptance DEFENSIVE fight). Sweeping 0→180 grows the reversal difficulty gradually so the
    # warm-started offense champion learns anti-deck + defensive BFM before facing the point-blank six.
    aspect_deg: float = 180.0
    # curriculum-20 (opponent-difficulty ladder): which OPPONENT AI flies the tier-3 start
    # GEOMETRY, decoupled from `tier`. 0 (default) = the geometry's native opponent (tier-3 →
    # Tier3Opponent); 1 = a straight Tier1 dummy; 2 = a Tier2 breaker. Lets a neutral/off-axis
    # merge (aspect_deg) be flown against a weaker AI so the beam→six turn-to-guns conversion is
    # learnable before the same merge is faced against the full T3 evader (which no scripted
    # expert can demonstrate off-axis, and which a geometry-only ladder stalled passive on).
    opponent_tier: int = 0
    # curriculum-21 / self-play: the bot's ABSOLUTE start heading for a Tier-3 offensive-posture
    # rung, overriding the co-heading (north) default. None = co-heading (jittered). Set to 180
    # with aspect_deg=0 for a HEAD-ON merge (pair closes at ~2× speed, no overtake needed); at
    # aspect_deg=90 it leaves the neutral line-abreast merge co-heading. Threaded to
    # tier3_scenario's `opp_heading_deg`. Only meaningful on offensive-posture Tier-3 rungs.
    opp_heading_deg: float | None = None
    # curriculum-27: HP-margin bar for the "net" metric — an episode wins iff it KILLs, or (no
    # death/deck) ends net HP ahead by ≥ net_threshold. Closes the c26 blind spot where the pure
    # "damage" gate counted only OUTGOING chip and certified a policy that chipped while being
    # shredded (final c26 policy: net −27 at its own gated aspect). Ignored for other metrics.
    net_threshold: float = 0.0
    # curriculum-27: when ≥0 on an offensive-posture Tier-3 rung, the bot's start bearing is drawn
    # per-episode from the RANDOM band [aspect_deg, aspect_deg_max] instead of the fixed aspect_deg.
    # Anchoring the band at 0 (e.g. [0, 60]) makes every stage rehearse the whole forward hemisphere,
    # so ramping the ceiling can't catastrophically forget the tail-chase (the c26 failure mode).
    # −1 (default) disables banding → the fixed aspect_deg is used (legacy ladders unchanged).
    aspect_deg_max: float = -1.0
    # curriculum-49 (re-attack ladder): the OPPONENT's spawn HP for an offensive-posture Tier-3 rung.
    # None (default) = full-HP (100), every legacy ladder unchanged. A wounded value (e.g. 30) spawns
    # the bandit already damaged, so every episode starts in the post-first-pass STATE (wounded bandit
    # off the nose, full ammo) and — kill-gated — the policy must reverse and FINISH rather than coast
    # (the ≥55° terminal-finish wall = a missing reverse→re-acquire→finish skill; c42 finish-diag).
    # Threaded to tier3_scenario's `opp_health`; only meaningful on offensive-posture Tier-3 rungs.
    opp_health: float | None = None
    # Per-stage EPISODE LENGTH override in seconds. 0 (default) = use the run-level
    # `--max-time-s`, i.e. every legacy ladder unchanged.
    #
    # Why this exists: `_make_env` splats the run-level `env_kwargs` — including `max_time_s` —
    # into EVERY collector, so `--max-time-s 180` (needed by the multi-turn merge program)
    # also stretched the curriculum-57 chase-REHEARSAL anchor from 60 s to 180 s. That anchor's
    # whole job is to hold the ≤55° finish skill by keeping its episodes in the buffer at the
    # distribution the skill was learned at; tripling their length changed that distribution,
    # so the anti-forgetting mechanism was itself confounded on exactly the runs it was added
    # to protect (c85's chase retention dropped 10 kills → 1). Pin the anchor's own clock here.
    max_time_s: float = 0.0


# The chase-then-defend ladder (curriculum-14). Each rung's GATE METRIC is chosen from what the
# feasibility probe (crates/astrum-env/python/_diag_feasible.py) proved is both achievable AND
# skill-forcing UNDER the per-episode start jitter the policy trains on — the fix for prior
# curricula that either backstop-marched through un-winnable kill rungs or passed passivity-
# rewarding survive rungs:
#   • T1 (non-maneuvering dummy) — the scripted PursuitExpert KILLS it under jitter (probed 4/4
#     @400 m, 3/4 @1400 m), so a KILL gate is honest and expert-teachable. Widen close→far to
#     grow the closing skill; the learner is the attacker here.
#   • T2 (maneuvering) — the expert CANNOT kill it under jitter (0/3 even @400 m: its 4° fire
#     cone won't open on a jinking target), yet a kill is physically feasible (≤511 rounds) and
#     40-50 HP of chip damage is readily gettable. A KILL gate would just backstop-march; a
#     SURVIVE gate would reward passive drift. So gate on DAMAGE DEALT — aggression-forcing
#     (must close AND employ guns) and achievable — with the HP bar easing as entry range widens.
#   • T3 (defender: the bot opens on the learner's SIX — the eval/acceptance geometry) — run
#     wide→near, gated on SURVIVAL (far = room to defend/reverse, near = a lethal point-blank
#     ambush). Kills stay rewarded so the policy reverses opportunistically, but survival is the
#     physically-assured, acceptance-aligned bar.
# Win bars are HONEST deterministic-probe rates (see the probe gate): a deployable greedy
# success-RATE over the jittered envelope, not the old stochastic-prior gate's lucky-noise 0.70.
# `damage` rows carry a 5th field = the HP-dealt threshold.
_CHASE_THEN_DEFEND = (
    (1, 400, "kill", 0.60),
    (1, 800, "kill", 0.55),
    (1, 1400, "kill", 0.45),
    (2, 400, "damage", 0.50, 60.0),
    # curriculum-16 recalibration (TWICE — grounded in a per-episode greedy feasibility eval, not a
    # rubber-stamp). A 20-episode greedy probe of the run.pt policy at T2@700m (jitter 0.5) showed
    # the damage distribution is heavily RIGHT-SKEWED: dmg_taken = 0.0 on ALL 20 (perfectly safe),
    # but dmg_dealt mean 15.4 / MEDIAN ~13 — ~30% of starts deal 0 HP, only 25% clear 30 HP, while
    # ≥15 HP clears 50%. The jinker's evasion + the prior-vs-plan gap cap the FAST greedy prior far
    # below the earlier lucky "mean 32". So ≥30 HP @ 40% sat above the feasible greedy ceiling and
    # the (backstop-free) run stalled at 15-20% for >150k updates. The rung's real OBJECTIVE — press
    # a damaging 700m attack WITHOUT ever dying — is already met (0 taken, mean 15 landed = closing &
    # firing). Drop to ≥15 HP (≈7-8 hits): ~50% feasible → real headroom over the 40% RATE bar (the
    # consistency requirement stays), certifies a competent 700m gun pass, and gates promptly so we
    # reach the T3 SURVIVE ladder (the acceptance geometry + real unknown). Deployment uses MPC,
    # which converts far better than this greedy gate. See _diag_feasible.py + dist_s4_greedy.txt.
    (2, 700, "damage", 0.40, 15.0),
    # curriculum-17: OFFENSE vs the Tier-3 AI, flown from the "defensive" posture (bot AHEAD, learner
    # on its tail — an offensive chase; the trailing "defensive" string sets Stage.posture). c16
    # mastered only the "offensive"-posture SURVIVE fight (bot on our six) and produced a pure passive
    # evader that never fires; two feasibility probes then showed why offense looked impossible and
    # why it actually isn't: (a) _diag_t3_chase.py — the PursuitExpert's cone-gated trigger fires 0
    # rounds on the continuous jinker even sitting 380-470 m behind it (so expert *seeding* alone can't
    # teach it, but note the expert does NOT deck from this posture — the deck was a bot-on-six artifact);
    # (b) _diag_t3_spray.py — the SAME expert stick with a LIBERAL trigger (fire when gun_quality≥0.05)
    # KILLED the T3 and landed 90-100 HP passes @400m (33% of jittered starts ≥30 HP) and 36-42 HP
    # @700m (33% ≥15 HP). The env hit model is physics-based (8 m round radius, not a cone), so spraying
    # while tracking works — the RL trigger is a learned action free to do exactly that, and the
    # fire-override now also drives the expert-rollout trigger so seeds complete kills. Gate on DAMAGE
    # at the probed ceiling: a greedy-probe RATE just above the so-so expert stick's 33% pass-rate
    # certifies the policy tracks/sprays the jinker BETTER than lead-pursuit. Deployment (MPC) converts
    # harder still. Only after offense is mastered do we return to the bot-on-six SURVIVE ladder — now
    # the policy carries a T3 gun skill to convert opportunistically into the reversal-kill c16 lacked.
    (3, 400, "damage", 0.35, 30.0, "defensive"),
    (3, 700, "damage", 0.35, 15.0, "defensive"),
    (3, 2200, "survive", 0.75),
    (3, 1400, "survive", 0.70),
    (3, 900, "survive", 0.65),
    (3, 600, "survive", 0.60),
    (3, 400, "survive", 0.55),
)


# curriculum-18: the REVERSAL ladder. c17 proved (and an independent greedy eval confirmed) that
# the policy guns the T3 from an OFFENSIVE start (aspect 0: 6/10 kills @400 m, 81 HP mean, 0 taken;
# chip @700 m). But an aspect-sweep feasibility probe of that champion (run_stage06 flown greedy vs
# the bot-on-six at aspects 0→180) showed the offense skill does NOT transfer off-axis: aspect 0 →
# 32 HP/0 taken, but aspect 45/135 → the policy DECKS (5-6/8 fly into the ground), aspect 90 → passive
# standoff (0 HP), aspect 180 (dead six) → dominated 8/8 (7 deck + 1 death, 30 HP taken). The dead-six
# reversal is a genuinely harder skill (defensive BFM + energy/altitude management the fast offense
# fight never exercised — it killed in ~5 s, long before altitude bled off). It IS learnable: the
# env already imposes a large hard-deck TERMINAL penalty (R_DECK, the worst outcome), so training on
# these geometries gives strong gradient to stop decking, and warm-starting from the offense champion
# carries the gun skill in. So slide the bot's start BEARING 0→180 in fine steps (aspect_deg, the 6th
# row field), each rung DAMAGE-gated so the policy must reverse and shoot — never merely evade (the
# c16 passivity trap the SURVIVE ladder re-taught). HP bars ease as aspect grows (a dead-six reversal
# lands far fewer rounds than a point-blank chase) but stay strictly > 0, forcing aggression at every
# rung. posture stays "offensive" (bot-on-six family) so NO expert-rollout scaffold fires (pursuit
# cannot reverse — c17 edit scopes the expert to tiers 1/2 + defensive posture); the fire-override
# still supplies the trigger. Run with --ladder reversal --resume <c17 stage06> --start-stage 0.
_REVERSAL = (
    # (tier, sep, metric, winrate, HP-bar, aspect_deg)  — posture defaults "offensive" (bot-on-six family)
    (
        3,
        700,
        "damage",
        0.35,
        15.0,
        0.0,
    ),  # re-consolidate the offense (bot dead ahead) + learn anti-deck under jitter
    (3, 700, "damage", 0.30, 12.0, 30.0),  # bot ahead-off-axis: the first real turn to the guns
    (3, 700, "damage", 0.30, 10.0, 60.0),
    (3, 700, "damage", 0.28, 8.0, 90.0),  # beam / neutral merge — win the turn to gain the six
    (3, 800, "damage", 0.25, 6.0, 115.0),  # bot slightly behind the beam — begin the reversal
    (3, 800, "damage", 0.25, 6.0, 140.0),
    (3, 800, "damage", 0.22, 6.0, 160.0),
    (
        3,
        800,
        "damage",
        0.20,
        6.0,
        180.0,
    ),  # DEAD SIX — the acceptance: reverse the bot on your six and land ≥3 hits
)


# curriculum-20: the OPPONENT-DIFFICULTY ladder — the fix for the wall c18/c19 hit. Those swept the
# START GEOMETRY (aspect 0→180) against the FULL T3 evader at every rung; aspect 0 (a clean tail-chase)
# was masterable (converts to kills under the -40 damage-timeout penalty) but EVERY off-axis rung
# stalled PASSIVE — the policy ate the timeout penalty rather than attempt a turn-to-guns it never
# discovered, and two feasibility probes proved neither the fire-override (supplies the trigger, not
# the MANEUVER) nor a lead-pursuit expert SEED could teach it: at offensive-posture aspect 30/60/90 the
# expert stick + liberal spray deals ~0 HP vs the jinking T3 (it can't convert a deflection merge into a
# tail-chase gun solution). The missing skill is the beam→six CONVERSION, and there is no scripted
# teacher for it vs T3. So instead of curriculuming geometry against a fixed hard opponent, curriculum
# the OPPONENT against a fixed neutral/forward merge: learn the conversion against a STRAIGHT Tier1 dummy
# (learnable — the aim/close PBRS gradient points the right way, and the lead-pursuit expert CAN seed it
# there), harden to a Tier2 breaker, then face the full Tier3 evader at the same merge. The bet: compose
# "convert a merge to the six" (learned vs T1/T2 here) with "track & gun a jinking T3" (already solved
# from the tail-chase, carried in via warm-start) → win the neutral merge vs T3 = beat it head-to-head.
# opponent_tier (dict key) is the difficulty knob, decoupled from the tier-3 START GEOMETRY (aspect_deg).
# T1/T2 rungs are KILL/DAMAGE-gated (a dummy is killable); T3 rungs are DAMAGE-gated (kill is hard,
# damage forces aggression) with the -40 timeout penalty so a no-damage draw stays a loss.
# Warm-start from the c19 aspect-0 T3-killer: --ladder merge --resume <c19 stage00> --start-stage 0.
_MERGE = (
    # forward-off-axis vs a straight DUMMY: turn 45° onto its six and gun (expert-seeded, easy warm-up)
    dict(
        tier=3,
        opponent_tier=1,
        posture="offensive",
        aspect_deg=45.0,
        separation_m=900.0,
        metric="kill",
        winrate=0.55,
    ),
    # BEAM merge vs the dummy: the core turn-to-six conversion skill, still expert-seedable
    dict(
        tier=3,
        opponent_tier=1,
        posture="offensive",
        aspect_deg=90.0,
        separation_m=900.0,
        metric="kill",
        winrate=0.50,
    ),
    # beam merge vs a Tier2 BREAKER: convert against a mild evader (no expert seed; RL + warm-start)
    dict(
        tier=3,
        opponent_tier=2,
        posture="offensive",
        aspect_deg=90.0,
        separation_m=900.0,
        metric="damage",
        winrate=0.40,
        damage_threshold=30.0,
    ),
    # forward-off-axis vs the FULL T3 evader: the bridge from the solved tail-chase into off-axis vs T3
    dict(
        tier=3,
        opponent_tier=3,
        posture="offensive",
        aspect_deg=45.0,
        separation_m=900.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=20.0,
    ),
    # BEAM merge vs the full T3 evader — THE ACCEPTANCE: win the neutral merge = beat T3 head-to-head
    dict(
        tier=3,
        opponent_tier=3,
        posture="offensive",
        aspect_deg=90.0,
        separation_m=900.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
)


# curriculum-21: SELF-PLAY (design doc P5) — the escape from the merge-conversion wall. c18/c19/c20
# proved the beam→six / neutral-merge CONVERSION has NO scripted teacher: 4 lead-pursuit teachers ×
# 5 geometries all deal ~0 reliable damage vs the T3 off-axis, and RL-alone stalls passive there. The
# missing skill (turn a neutral merge into a gun solution, then reverse a bandit that reverses back)
# only emerges from an opponent that itself maneuvers and improves — i.e. copies of the learner. So
# the collector's scripted opponent is replaced by a `ScriptedPolicyOpponent` sampled per-episode from
# an `OpponentPool` of the learner's own frozen priors (fictitious self-play); the learner periodically
# exports a fresh prior into the pool (--selfplay-snapshot-every), so the opponent tracks the learner's
# skill and the arms race grows merge/neutral/reversal ACM the scripted ladder could not. The GEOMETRY
# is a single NEUTRAL line-abreast merge (aspect 90, co-heading, jittered): symmetric (neither jet has
# the opening advantage), and both policies naturally turn in — forcing exactly the merge conversion.
# DAMAGE-gated with the -40 timeout penalty (the validated c19/c20 anti-passivity setup) so a fly-past
# draw is a loss and gradient must convert; the single top stage means the gate is a no-op (nowhere to
# advance) — the pool grows on the snapshot clock, and the logged damage-rate + out-of-band eval vs the
# scripted T3 are the progress signals. Warm-start the learner (and seed the pool's first opponent)
# from a champion: --ladder selfplay --resume <c17/c19 champion> --save runs/sp/run.pt.
_SELFPLAY = (
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=90.0,
        opp_heading_deg=None,
        separation_m=900.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
)


# curriculum-51 (Phase 2, the user's plan): NOSE-TO-NOSE self-play — the pivot after 8 curricula
# (c43–c50) confirmed the ≥60° terminal FINISH is a method ceiling for single-agent-vs-SCRIPTED-T3
# (the wounded-bandit family collapsed into decking; every ≥60° push degraded the fragile champion).
# The diagnosis: the scripted T3 never densely presents the high-aspect reversal-and-finish situation the
# finish skill lives in, and hand-crafted scenarios that force it destroy the ≤55° kill. Self-play
# attacks both — a POOLED, ever-improving copy of the learner is a roughly-EQUAL adversary, so genuine
# neutral/high-aspect merges and reversals arise every episode, and warm-starting from the champion makes
# its 45° kill the STARTING point (nothing hand-crafted can shred it). GEOMETRY = head-on merge
# (aspect 0 + opp_heading 180): both jets close at ~2× speed, pass, and it becomes the canonical neutral
# turning fight — "kill-first vs survive-first". KILL-gated (honest, not the climbable damage gate) with
# ALL probes routed to the SCRIPTED T3 via --selfplay-scripted-prob, so the gate measures a REAL kill-rate
# vs the true target and can't be fooled by weak self-copies (the c22–c24 self-play gate lie). Single
# stage: self-play difficulty auto-ramps as the pool tracks the learner — no geometry ladder needed; the
# real progress signal is the OUT-OF-BAND greedy verdict vs the scripted T3 at 45/55/60/65°. Auto-enables
# --selfplay; warm-start + seed the pool with --resume <champion> (curriculum-46/_snap_2244k).
_SELFPLAY_NOSE = (
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=180.0,
        separation_m=1200.0,
        metric="kill",
        winrate=0.20,
    ),
)


# curriculum-22: SELF-PLAY ANGLE LADDER — the fix for the c21 stall. c21 launched self-play directly at
# the neutral BEAM merge (aspect 90) and drew 0% damage for 1.48M steps / 1000+ episodes: a symmetric
# merge between two NON-converters has no conversion gradient (the c19 warm-start is a tail-chase killer,
# out-of-distribution at the beam, and so is its frozen pooled copy — neither lands a hit, so there is no
# signal either way and the guided-fire override never triggers because gun_quality never clears 0.05).
# The escape is the same distance-curriculum insight that cracked offense, applied to ANGLE: START where
# the warm-start ALREADY converts vs a copy of itself — aspect 0, co-heading (learner opens on the pooled
# bandit's SIX, the tail-chase c19 masters) — so damage flows immediately, the pool seeds with converting
# behaviour, and the arms race ignites. Then HARDEN the merge angle outward in FINE 15° steps, advancing
# only on sustained damage MASTERY (gate), so the beam-merge conversion is built up from a geometry the
# policy can already win rather than demanded cold. Every rung is DAMAGE-gated with the -40 timeout penalty
# (a fly-past draw is a loss); the pool is shared across rungs (opponents keep getting stronger as the
# angle hardens). Warm-start + seed from the c19 tail-chase T3 killer:
#   --ladder selfplay-ladder --resume <c19 stage00> --start-stage 0 --stage-backstop-steps 3000000
#
# c22 (first cut) used 30° steps and STALLED at aspect 30: damage cratered 50%→0% for ~1.3M steps then only
# oscillated 0-8% (never near the 35% bar) — the jump from "shoot the six I'm handed" (aspect 0) to "turn
# 30° to convert" (aspect 30) is the whole BFM-conversion skill, too big to cross in one rung. c23 halves
# the step to 15°: at aspect 15 the bandit is nearly dead-ahead so a small turn/lead converts it (close to
# the dead-six skill the warm-start has), and each rung is a small increment on the last — so the angles
# fight grows incrementally instead of being demanded whole.
#
# curriculum-24: the c23 15° ladder gated cleanly through aspect 75 — but a GPU eval vs the SCRIPTED T3
# exposed the trap: the gated policy scored 0 KILLS and DIED half the time at the wider aspects, having
# REGRESSED off the c19 warm-start that killed the T3 4/10 (91 HP / 0 taken) at the tail-chase. The `dama`
# gate had measured damage vs SELF-COPIES, which converged to a passive mutual-draw equilibrium — a false
# signal, and the policy, never made to fight the real bot, drifted off it entirely. The fix is ANCHORING
# (--selfplay-scripted-prob): a fraction of episodes AND every deterministic probe face the SCRIPTED T3,
# so the gate measures the REAL target and the offense stays pinned to the c19 killer while self-copies
# still supply merge/reversal variety. Relaunch warm-started from the c19 tail-chase killer:
#   --ladder selfplay-ladder --resume <c19 stage00> --start-stage 0 --stage-backstop-steps 3000000 \
#   --selfplay-scripted-prob 0.5
_SELFPLAY_LADDER = (
    # rung 0 — tail-chase (bot dead-ahead, co-heading = learner on its six): the c19 warm-start converts
    # this cold, so damage flows from step 0. Higher bar (0.45) — prove ROBUST mastery before hardening,
    # so the skill transferred to the first offset rung is sturdy (c22 advanced at a shaky 40%).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.45,
        damage_threshold=12.0,
    ),
    # rungs 1-5 — harden the merge angle in 15° steps; each removes a little more of the opening advantage
    # so the learner must actively convert a slightly wider offset than the rung it just mastered.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=15.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.40,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=30.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=45.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
    # rung 6 — the neutral BEAM merge (aspect 90): THE acceptance geometry. Converting it vs a maneuvering
    # copy of the learner even 30% of the time is beating a peer head-to-head from neutral.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
)


# curriculum-25: the BEAM-MERGE ladder with EXPERT-BFM seeding — the fix for the c24 wall. Anchored self-play
# (c24) worked: it gated HONESTLY (probes vs the SCRIPTED T3) through aspect 0→15→30→45, each on merit well
# inside its backstop — the neutral-merge conversion that stalled every prior curriculum. But it then hit a
# hard wall at aspect 60: `dama` cratered to 0% and stayed there for ~8.3M steps (1707/1707 stage-4 probe
# lines at 0%, max 33% only on the boundary) while the policy ENTROPY collapsed (pi +0.14 → ~0.01) — it
# converged to a passive, non-converting equilibrium and could not climb out. Root cause is UPSTREAM of the
# trigger (the fire-override re-arms each stage, so it was live and still impotent): at a 60°+ beam merge the
# problem is winning the turning-circle geometry to get the nose on the bandit and CREATE a gun solution at
# all — qualitatively new merge BFM (lead/lag pursuit) the aspect-0-warm-started prior never discovered, and
# the fire-override can't force a shot that geometry never sets up. Two changes:
#   1. EXPERT-BFM seeding (--expert-p0): early in each stage, fly whole episodes with the scripted PursuitExpert
#      (annealing p→0, EXCLUDED from the honest gate) so the buffer gets complete close→convert demos the
#      passive prior can't produce — the model learns the merge maneuver by imitation, then the gate measures
#      whether the GREEDY policy reproduces it. This is the long-range lever the geometry-only ladder lacked.
#   2. FINER rungs concentrated on the wall: re-anchor at the mastered aspect 45, then harden in ~7-8° steps
#      through the exact 45→60 transition that broke (vs the 15° step that stalled), out to the beam merge.
# Warm-start from the c24 aspect-45 converter (its best HONEST milestone, NOT the passive stage-4 policy):
#   --ladder selfplay-beam --resume runs/curriculum-24/run_stage03_t3_700m.pt --start-stage 0 \
#   --selfplay-scripted-prob 0.5 --expert-p0 0.5 --expert-steps 2000000 --stage-backstop-steps 3000000
_SELFPLAY_BEAM = (
    # rung 0 — re-anchor at aspect 45, which the warm-start already converts (gate clears fast); this re-seeds
    # the pool with CONVERTING behaviour at the offset before the fine hardening begins.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=45.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=12.0,
    ),
    # rungs 1-5 — cross the 45→60 wall in ~7-8° steps (half the 15° step that stalled c24), each a small
    # increment so the merge BFM grows continuously with the expert demos rather than being demanded whole.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=52.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.32,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=68.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.28,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=76.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.25,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=83.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.25,
        damage_threshold=12.0,
    ),
    # rung 6 — the neutral BEAM merge (aspect 90): THE acceptance geometry, eased to a 0.25 bar (converting a
    # true perpendicular merge vs a maneuvering peer even a quarter of the time is a strong neutral-merge win).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.25,
        damage_threshold=12.0,
    ),
)


# curriculum-25 (HONEST, non-self-play): the fix after the c24 gate was proven to LIE. The _SELFPLAY_BEAM
# comment above claims "anchored self-play gated honestly (probes vs the scripted T3)" — that is FALSE. A
# ground-truth audit of c24's own saved stage-3 milestone (run_stage03, the aspect-45 gate-passer) showed:
# the gate reported a smoothly-climbing damage-rate to 35% vs the "scripted T3", yet an exact replication of
# that probe — same env recipe, same greedy prior, a FRESHLY-BUILT Tier3Opponent — deals 0% ≥12HP / 0.7 HP at
# aspect 45 AND aspect 30 (all timeouts). The anchoring opponent-swap does NOT deliver the T3 to the gated
# probes, so every self-play probe scored damage against a WEAK POOLED SELF-COPY, not the bot. c22→c24's
# entire aspect 0→45 "progress" was illusory, and the detour REGRESSED the real skill: c24's policy only chips
# 40%/0-kills at aspect 0, where the honest c19 anchor KILLS (25% kills, 69 HP mean, 100% ≥12HP).
#
# Non-self-play makes the gate honest BY CONSTRUCTION: with selfplay=False the per-episode opponent-swap block
# is never entered, so the env keeps the native Tier3Opponent it builds — every collection episode AND every
# deterministic probe faces the REAL bot, and `dama` is damage vs the target with no swap to fail silently.
# Ground-truth of the c19 anchor maps the TRUE frontier: aspect 0 = 100% ≥12HP / 25% kills, aspect 15 = 20%,
# aspect 30 = 0% (and the nose-low conversion DECKS — 5/20 hard_deck). So the wall is aspect ~15-30. This
# ladder re-consolidates the aspect-0 kill, then hardens in FINE ~6° steps THROUGH that wall (vs the 15° that
# only ever "passed" dishonestly), widening the steps past 30° where the geometry is already lost. Strict gate
# (--gate-window 100 --gate-min-episodes 120) + a huge backstop make advancement gate-DRIVEN: a rung the policy
# cannot honestly master parks with dama~0 and shows the true frontier instead of skipping on a lucky streak.
# Warm-start from the GENUINE aspect-0 killer, gate-only advancement, anti-passivity timeout penalty:
#   --ladder beam-t3 --resume runs/curriculum-19/run_stage00_t3_700m.pt --start-stage 0 \
#   --gate-window 100 --gate-min-episodes 120 --stage-backstop-steps 30000000 --damage-timeout-penalty -40
_BEAM_T3 = (
    # rung 0 — re-consolidate the tail chase (bot dead-ahead, co-heading = learner on its six): the c19 warm-
    # start KILLS this cold (25% kills / 69 HP), so a robust 0.50 bar proves the skill is sturdy before hardening.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.50,
        damage_threshold=12.0,
    ),
    # rungs 1-5 — FINE ~6° steps straight through the honest wall (aspect 0→30, where c19 fell 100%→20%→0% and
    # began decking): each removes a sliver more opening advantage so the merge BFM grows continuously.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=6.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.40,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=12.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=18.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.30,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=24.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.28,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=30.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.25,
        damage_threshold=12.0,
    ),
    # rungs 6-9 — widen the steps past the wall (the offset is already large; the marginal difficulty per degree
    # falls) out to the neutral beam merge, easing the bar as the geometry hardens toward a true perpendicular.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=40.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.22,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=55.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.20,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=70.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.18,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.15,
        damage_threshold=12.0,
    ),
)


# curriculum-26 (HONEST, non-self-play): the fix after c25 (--ladder beam-t3) demolished the 0→40° wall
# honestly (aspect 0→40 all mastered vs the NATIVE T3: gate advances 92/80/35/69/43/63/33%, and an
# independent eval of the aspect-40 milestone confirms 87% ≥12HP at 40°) but then WALLED HARD at rung 7
# (aspect 55): ~1.6M steps parked at dama 0-3% (gate 20%). A ground-truth probe of the c25 aspect-40
# milestone (run_stage06) pinpoints a SHARP skill cliff at ~45-48°, NOT a range problem:
#   aspect 40 → 87% ≥12HP   aspect 45 → 7% (marginal, max 34 HP)   aspect 48/50/55 → 0% / 0 HP (all timeouts)
# and reducing separation makes it WORSE, not better — 55°@450m = 0% and the learner got KILLED (close range
# at a beam merge favors the more-agile T3; the "kill nearly free at 400m" only holds for the LOW-aspect
# tail-chase). So the c25 tail (40→55→70→90, up to 15° per rung) threw the policy straight past its ~45°
# skill boundary into a 0%-signal void with nothing to bootstrap from. The 7% at 45° is the tell: a nonzero
# seed the guided-fire override can grow — but only if the policy actually TRAINS at 45° before facing 50°.
# Fix = a FINE 5° aspect staircase 40→45→…→90 at the SAME 700 m, warm-started from the c25 aspect-40 chipper,
# so every rung starts only 5° from a trained checkpoint (transfer stays nonzero) and climbs to gate before
# the next. Strict gate (100/120) + huge backstop keep advancement gate-DRIVEN, so if a rung still walls it
# marks the true skill ceiling precisely (next lever there = longer episode clock or a working BFM expert):
#   --ladder beam-t3-fine --resume runs/curriculum-25/run_stage06_t3_700m.pt --start-stage 0 \
#   --gate-window 100 --gate-min-episodes 120 --stage-backstop-steps 30000000 --damage-timeout-penalty -40
_BEAM_T3_FINE = (
    # rung 0 — re-consolidate the aspect-40 chip the c25 warm-start already owns (87% ≥12HP), so the base
    # BFM is proven sturdy before the staircase climbs into the cliff. Easy 0.35 bar over the 40° milestone.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=40.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.35,
        damage_threshold=12.0,
    ),
    # rungs 1-10 — FINE 5° steps through the ~45-48° cliff that stopped c25, out to the neutral beam merge.
    # Each rung is only 5° off a trained checkpoint (45° transfers at ~7% from the 40° ckpt — a live seed the
    # guided-fire override grows), so the merge lead-turn deepens continuously instead of vanishing at a jump.
    # The bar eases as the offset hardens toward a true perpendicular where a nose-on gun solution is scarcest.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=45.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.28,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=50.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.25,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=55.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.22,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.20,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=65.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.18,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=70.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.17,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.16,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=80.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.15,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=85.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.14,
        damage_threshold=12.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="damage",
        winrate=0.13,
        damage_threshold=12.0,
    ),
)


# curriculum-27: the c25/c26 "beam mastery" was HOLLOW. The pure `damage` gate counts only OUTGOING
# chip (HP dealt) — it never saw HP TAKEN or deaths/decks. An honest verdict eval of the c26 final
# policy vs the NATIVE T3 (greedy prior) exposed the lie: at its OWN gated aspects it is net-NEGATIVE
# everywhere but 0 (aspect 60 → 0 kills / 3 deaths / dealt 10.8 taken 41.2 NET −30.4; aspect 80 → NET
# −27.6; aspect 90 → NET −55.7) AND it catastrophically FORGOT the low aspects (0 HP dealt at 0–40).
# The gate had certified a jet that chips while being shredded. Two root causes, two fixes here:
#   (1) GATE BLIND SPOT → the `net` metric: a probe wins iff it KILLs, or ends with NO loss terminal
#       (death/trade/deck/bubble) AND net HP ahead by ≥ net_threshold. Getting gunned or decking is
#       now a loss no matter how much chip was dealt. (The reward was ALREADY net-aware — dense term
#       is +0.5·dealt −0.5·taken plus a death penalty — so the honest gate finally AGREES with the
#       gradient; the gate, not the reward, was the broken link.)
#   (2) CATASTROPHIC FORGETTING → aspect BANDING: every rung samples the bearing per-episode from
#       [0, ceiling] (aspect_deg=0, aspect_deg_max=ceiling), so the learner rehearses the WHOLE
#       forward hemisphere at every stage. Ramping the ceiling can no longer forget the tail-chase.
# The gate winrate is now the fraction of band-random probes that genuinely net-WIN; low aspects are
# near-free net-wins (c25: +32 HP / 0 taken at 40°) so the floor is high, letting a firm bar force the
# competence FRONTIER up the band: at bar θ over ceiling C the policy must net-win out to ≈θ·C. If a
# rung walls, that frontier IS the honest beam ceiling (next lever = a working BFM expert or a longer
# clock — NOT a lower bar). Warm-start from the genuine c25 aspect-0→40 net-winner (run_stage06):
#   --ladder beam-t3-net --resume runs/curriculum-25/run_stage06_t3_700m.pt --start-stage 0 \
#   --gate-window 100 --gate-min-episodes 120 --stage-backstop-steps 40000000 --damage-timeout-penalty -40
_BEAM_T3_NET = (
    # rung 0 — CONSOLIDATE the forward hemisphere [0,40] as ONE policy (c25's milestones were
    # per-aspect; prove a single net-winner holds the whole band). c25 nets ~+32/0-taken here, so
    # the 0.72 bar is an easy confirmation that banding re-fused the skill (and anti-forgets it).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=40.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.72,
        net_threshold=8.0,
    ),
    # rungs 1-3 — widen the ceiling through the ~45-48° cliff in fine steps; the band keeps 0-40 in
    # rotation so the frontier extends without shedding the base. Bar eases slightly as the hard
    # high-aspect slice of the band grows; net bar eases as clean margins shrink toward the beam.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=50.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.70,
        net_threshold=7.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=55.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.68,
        net_threshold=6.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.66,
        net_threshold=5.0,
    ),
    # rungs 4-6 — out to the neutral beam merge. If the policy cannot net-win these bands, it STALLS
    # here and that is the honest T3 beam ceiling (the c26 "pass" here was the damage-gate lie).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=70.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.64,
        net_threshold=4.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=80.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.62,
        net_threshold=4.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.60,
        net_threshold=3.0,
    ),
)


# curriculum-28: c27 honestly GATED the offensive hemisphere and an independent verdict eval of its
# run_stage05 milestone ([0,80] master) vs the native T3 confirmed the net gate is trustworthy —
# DOMINANT with kills at aspect 0-45 (30°→7 kills/+91 NET/100% win; 45°→3 kills/+82/100%, past the
# c25 45° cliff) — but exposed the true ceiling: aspect 60→50% (chip only), 70-80→0% (pure passive
# timeout, 0 HP dealt), 90→decks 4/10 / −18 NET. ROOT CAUSE the wide [0,ceiling] band let 0-45
# dominance MASK the frontier: band-avg over [0,80] ≈ (100·4 + 50 + 0 + 0)/7 ≈ 60% cleared the 0.62
# bar almost entirely on low-aspect wins while 70-90 stayed at ZERO. A firm bar on a wide band does
# NOT force the frontier once the base is maxed. c28 fixes this with NARROW frontier-weighted bands:
# a 25°-wide window centered on the unmastered slice so band-avg ≈ the frontier's own net-win rate —
# a modest bar there forces REAL advance (at bar θ over band [lo,hi] the policy must net-win out to
# ≈ lo + θ·(hi−lo)). Frontier-push rungs (high floor) interleave with [0,hi] CONSOLIDATION rungs that
# re-fuse the whole hemisphere so the narrow bands can't shed the c27 low-aspect kills. Bars are
# per-band (NOT monotonic across the ladder): a consolidation rung's 0.68 over [0,65] is EASIER than
# a frontier rung's 0.48 over [50,75]. If a beam rung walls, THAT is the honest ceiling (next lever =
# self-play or a BFM expert, NOT a lower bar). Warm-start = the c27 [0,80] master; fewer collectors
# (workers 3) lift utd ~0.02→~0.05 so the rare beam net-wins get reused (the c15/c16 lesson):
#   --ladder beam-t3-push --resume runs/curriculum-27/run_stage05_t3_700m.pt --start-stage 0 \
#   --workers 3 --gate-window 100 --gate-min-episodes 120 --stage-backstop-steps 40000000 \
#   --damage-timeout-penalty -40
_BEAM_T3_NET_PUSH = (
    # rung 0 — FRONTIER [40,65]: probes concentrate on the 55-65 slice the c27 master half-holds
    # (60°→50% net-win). 0.55 bar forces mastery out to ≈40+0.55·25≈54° with the hard 60-65 slice
    # weighted enough to matter — no longer drowned by the free 0-40 tail-chase.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=40.0,
        aspect_deg_max=65.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.55,
        net_threshold=5.0,
    ),
    # rung 1 — CONSOLIDATE [0,65]: re-fuse the whole forward hemisphere as ONE net-winner so the
    # narrow push didn't shed the 0-45 kills. Bar high (includes the big-margin low aspects) but the
    # 55-65 slice must now hold too — this is the anti-forgetting stitch.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=65.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.68,
        net_threshold=6.0,
    ),
    # rung 2 — FRONTIER [50,75]: push into the passive-timeout band (c27 master: 0 HP dealt at 70°).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=50.0,
        aspect_deg_max=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.48,
        net_threshold=4.0,
    ),
    # rung 3 — CONSOLIDATE [0,75].
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.60,
        net_threshold=4.0,
    ),
    # rung 4 — FRONTIER [60,85]: the near-beam. Bar eases as clean net margins vanish toward 90°.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=60.0,
        aspect_deg_max=85.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.42,
        net_threshold=3.0,
    ),
    # rung 5 — FRONTIER [65,90]: the pure neutral beam merge (c27 master decked here). If this walls,
    # the honest T3 offensive-hemisphere ceiling is ≈65-70° and self-play is the next lever.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=65.0,
        aspect_deg_max=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.38,
        net_threshold=3.0,
    ),
    # rung 6 — FINAL CONSOLIDATE [0,90]: prove the ENTIRE forward hemisphere holds as one net-winner
    # (this is exactly the [0,90] rung c27 walled on at 0.60 — 0.55 here because the frontier pushes
    # should have lifted the high slice; still walls → milestones capture the true best).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.55,
        net_threshold=3.0,
    ),
)


# curriculum-29: the SYNTHESIS of c27's and c28's opposite failures. c27 = always-[0,hi] bands (no
# forgetting) + BAND-AVERAGE gate → the base masked the frontier ([0,80] read 60% on free 0-45 kills
# while 70-90 sat at 0%): it ADVANCED on a lie. c28 = narrow FRONTIER-only bands (no masking) but
# EXCLUDED the base from training → SGD catastrophically forgot the 0-45 kills the c27 warm-start
# had (an independent eval found the c28 "champion" inert at easy aspects). c29 keeps BOTH virtues:
#   * TRAIN the full [0, hi] band every rung  → the base stays in rotation, so no forgetting.
#   * GATE on the FRONTIER SLICE only ([hi−FRONTIER_SLICE_DEG, hi], via _in_frontier / probe_aspect)
#     → advance measures the hard outer wedge's OWN net-win rate, so the base can't mask it.
# The winrate bar is therefore the FRONTIER's required net-win rate (not a band average), eased
# toward the beam where clean net margins vanish. Two reward-v9 fixes ride along to convert the
# frontier the c27 champion lost on: (a) a DENSE deck-cushion penalty (its 85-90° losses were
# self-inflicted decks, not gun deaths) and (b) the EXPERT's hit-gated fire bonus (it dumped the
# whole magazine on the first long pass). Warm-start = the honest c27 [0,80] master (run_stage05):
#   --ladder beam-t3-frontier --resume runs/curriculum-27/run_stage05_t3_700m.pt --start-stage 0 \
#   --workers 3 --gate-window 60 --gate-min-episodes 80 --stage-backstop-steps 60000000 \
#   --damage-timeout-penalty -40 --probe-every 6
# (gate window/min-eps are LOWER than c27/c28's 100/120 because only frontier-slice probes vote —
# ~FRONTIER_SLICE_DEG/hi of them — so votes accrue slower; the bar still demands sustained mastery.)
_BEAM_T3_FRONTIER = (
    # rung 0 — [0,60], frontier [35,60]. c27's master half-holds 60° (50% net-win); with the deck +
    # fire fixes the outer slice should firm up. Re-fuses the whole hemisphere as ONE net-winner.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.55,
        net_threshold=5.0,
    ),
    # rungs 1-3 — widen the ceiling through the near-beam. Frontier slice slides [40,65]→[45,70]→
    # [50,75]; the bar eases as the clean net margin at the merge shrinks. The full [0,hi] band keeps
    # the 0-45 kills in rotation the whole way (anti-forgetting), but ONLY the outer wedge gates.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=65.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.52,
        net_threshold=5.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=70.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.48,
        net_threshold=4.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.45,
        net_threshold=4.0,
    ),
    # rungs 4-6 — the near-beam / pure neutral merge (c27's 70-90 wall: 0 HP dealt or decked). If a
    # rung walls HERE with the frontier gate honest, THAT is the true offensive-hemisphere ceiling
    # and self-play (task #8) is the next lever — NOT a lower bar. Milestones capture the best policy.
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=80.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.42,
        net_threshold=3.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=85.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.38,
        net_threshold=3.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.35,
        net_threshold=3.0,
    ),
)

# curriculum-34: the c33 post-mortem. c33 (reward-v10, 25M, warm c29 champion) climbed all 7
# _BEAM_T3_FRONTIER rungs to the beam, but an HONEST out-of-band verdict of its 13.25M policy vs a
# fresh native T3 came back 0 KILLS at EVERY aspect (0/45/90) — 0 deaths, full 511-round mag dumped
# each engagement, ~68 HP chipped at 45° then flown disarmed. It had REGRESSED from the ~10.9M
# checkpoint's clean 45° kills. Root cause: EVERY frontier rung gates on metric="net" (win = kill OR
# survived-net-HP-ahead ≥ threshold, thresholds as low as 3 HP), so the ladder is climbable — and was
# climbed (stage 5→6 gated at 50% net-rate) — by CHIP-AND-SURVIVE without ever killing; 7M steps parked
# on the un-killable 90° rung under that gate then eroded the finishing behaviour even at 45°.
# THE KILL-GATE FIX: the low/mid rungs where a clean kill is demonstrably reachable (ceiling ≤75°,
# stages 0-3) gate on metric="kill" — advancement now REQUIRES genuine kills, so the curriculum
# certifies a real finisher before widening toward the beam and the chip-survive optimum is closed off.
# The near-beam rungs (80/85/90, stages 4-6) stay metric="net": a clean kill at a pure neutral merge
# may be physically unreachable (see the merge-kill-not-scripted finding — no scripted expert can
# demonstrate it), so net-HP-ahead-survive is the honest bar at the true ceiling. Kill rungs auto-carry
# reward-v7's R_TIMEOUT (a no-kill clock draw is a loss → gradient must convert, not park nose-on).
# Rides with reward-v11 (aspect-gated pursuit → learn the lead turn at the merge instead of the
# overshoot-then-spray rush). The full [0,hi] band still trains every rung (anti-forgetting) and only
# the frontier slice gates. Kill-rung winrates ease 0.45→0.30 (a kill is strictly harder than net-ahead;
# the c29 champion already kills ≤45°, so 45% at the [35,60] frontier is a real but reachable bar).
# Launch: --ladder beam-t3-killgate --resume runs/curriculum-29/run_stage05_t3_700m.pt --start-stage 0
#   --workers 3 --gate-window 60 --gate-min-episodes 80 --stage-backstop-steps 12000000
#   --damage-timeout-penalty -40 --probe-every 6 --steps 25000000
_BEAM_T3_KILLGATE = (
    # rungs 0-3 — [0,60]→[0,75], frontier slice [hi−25,hi]. metric="kill": ADVANCE only on genuine
    # kills at the outer wedge, so the finisher can't be faked by chip-and-survive (the c33 lie).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=60.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=0.45,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=65.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=0.40,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=70.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=0.35,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=75.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=0.30,
    ),
    # rungs 4-6 — the near-beam / pure neutral merge (80/85/90). metric="net": a clean kill here may be
    # physically unreachable, so net-HP-ahead-survive is the honest ceiling bar (unchanged from frontier).
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=80.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.42,
        net_threshold=3.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=85.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.38,
        net_threshold=3.0,
    ),
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=90.0,
        opp_heading_deg=None,
        separation_m=700.0,
        metric="net",
        winrate=0.35,
        net_threshold=3.0,
    ),
)

# curriculum-35: the c34 post-mortem + the user's reframe. c34 (kill-gate 0-3 / net-gate 4-6, reward-v11)
# proved the KILL-GATE fixes the chip-survive lie (stages 0-3 advanced on genuine 30-52% kills, no c33
# regression) BUT its 9.7M-step grind on the net-gated 90° rung drove 75-90° into a NEVER-FIRE TRAP: the
# 19M greedy verdict killed ≤60° yet at 75-90° spent 0 rounds, dealt 0 HP, and timed out — it learned to
# SURVIVE the beam merge (0 decks, down from 40% at entry) by never engaging at all. Two causes: (1) the
# net gate on 4-6 rewards chip-AND-survive, so passivity gates; (2) reward-v11 damped the re-attack turn
# and a passive draw was cheap. The user's acceptance is now explicit: "I still want to KILL the bandit —
# that's the direct product of chasing it down and spending its remaining ammo." So the KILL is the goal
# at EVERY aspect (the beam kill arrives AFTER the merge via the chase, not at the merge), and the chase
# is the means. THE FIX: (a) every rung gates on metric="kill" — the net-gate (chip-survive) is retired
# entirely, so the ladder can be climbed ONLY by finishing; (b) reward-v12 supplies the chase — v11's
# aspect damping is neutralised (PURSUIT_FLOOR→1) so the re-attack turn pays at any aspect, and a PASSIVE
# clock draw (dealt < 25 HP — never ran him down) is penalised 2× harder than a hard-fought no-finish
# draw, so orbiting to the clock is clearly −EV. Warm-start = the c34 19M policy (runs/curriculum-34/
# run.pt): it already merges + kills at 45-60° with ZERO deaths/decks, so the survive-and-kill skill is
# real and generalizable — it simply declined to apply it at 75-90° where the old reward paid passivity;
# reward-v12 changes that EV. Kill winrates ease 0.45→0.15 up the band (a beam kill is strictly hardest;
# a walled rung IS the honest ceiling, and the out-of-band verdict then shows the CHASE — spent-in-solution
# + dealt>0 + 0 decks — which is the acceptance behaviour even short of a beam kill).
# Launch: --ladder chase-t3-killgate --resume runs/curriculum-34/run.pt --start-stage 0 --workers 3
#   --gate-window 60 --gate-min-episodes 80 --stage-backstop-steps 12000000 --damage-timeout-penalty -40
#   --probe-every 6 --steps 25000000 --device auto --save runs/curriculum-35/run.pt --save-every 50000
_CHASE_T3_KILLGATE = tuple(
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=float(hi),
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=float(wr),
    )
    # ceilings 60→90 (frontier slice [hi−25, hi] gates); kill bar eases as the frontier nears the beam,
    # where a clean gun kill off a chase is strictly harder (never below 0.15 — still a real finisher bar).
    for hi, wr in ((60, 0.45), (65, 0.40), (70, 0.35), (75, 0.30), (80, 0.25), (85, 0.20), (90, 0.15))
)


# curriculum-70 (the NO-GIFT HOT-PASS + HIGH-YO-YO ladder). User directive (2026-09-01): "Dude you
# gotta stop the advantageous learning. It must learn to high yoyo." The chase-t3-killgate program
# bands aspect [0, hi] — every episode SAMPLES the aspect-0 rear-quarter GIFT (the learner handed a
# stable tracking solution on the six), so a flat pure-pursuit point-and-shoot works and the agent
# never has to fly BFM. c67/c68/c69 all confirmed the same non-holdable-peak ceiling from that band.
# *** DESIGN PIVOT grounded in _hy_trigger_probe_tmp.py (2026-09-01): the HIGH YO-YO is intrinsically a
# REAR-QUARTER OVERTAKE maneuver — it manages EXCESS CLOSURE. At co-speed the triggering closure
# (>CLOSURE_HI within YOYO_RANGE) is only reachable in the ~45-60° band, where the aggressive T3's hard
# break decelerates it in front of the co-speed pursuer (probe: aspect55 cloP90=128, fires 12%/all eps;
# by aspect75 cloP90=18 fires 2%; at the beam90+ cloMax≤123 usually ~76, fires ~0 — the beam's overshoot
# is LATERAL, a low-yoyo problem c66 already targets, NOT a closure problem). *** So this ladder does NOT
# move to the beam (where the high yoyo can't fire); it REMOVES THE GIFT differently: band anchored at
# 45° (EXCLUDES the 0-45 on-the-six tracking gift) through the beam-approach, vs the aggressive T3 —
# a HOT DYNAMIC CONVERTING PASS whose break forces an overshoot the agent must HIGH-YO-YO through, not a
# handed gun solution (c67-69 proved this band is an unsolved hard BFM problem, 0-4 kills, non-holdable).
# Paired with:
#   • the WINDOWED high-yoyo seed (--merge-override-mode highyoyo --merge-override-range-m 900
#     --merge-override-ata-deg 0): splices the scripted high-yoyo ONLY in the hot-closure overshoot
#     transient (closure > CLOSURE_HI within 900 m — the doctrine trigger, probe-confirmed to fire in
#     this band's lower half), seeding the "pull UP out of plane + bleed the surplus closure" transitions
#     the flat-pursuit prior can't produce — windowed (probabilistic + annealing + transient-gated), NOT
#     the c62 whole-episode override that decked;
#   • the ENERGY reward (--outcome-reward): K_ENERGY specific-energy PBRS + the anti-zoom-stall corner
#     floor + rel-Es (out-energy the bandit) + WEZ conversion-dwell — so the jet keeps the energy to
#     reach AND hold the WEZ instead of zoom-stalling out of it ("gain enough energy to put us into
#     WEZ"). Co-ALTITUDE / co-SPEED spawn (the user's explicit constraint — no spawn handicap; the
#     closure/energy is EARNED off the T3's break, not gifted). Co-heading vs the native aggressive T3,
#     NOT the confirmed-dead head-on merge (opp_heading None, not 180) → stays out of the symmetric-
#     crossing geometric-draw ceiling ([[merge-crossing-velocity-ceiling]]).
# Anchored at 45° (top of the mastered chase band → a thin anti-catastrophic-forgetting tail + warm-start
# overlap, the c26/c27 banding lesson) so a warm-from-c69 converter isn't thrown stone-cold (the c51
# cold-collapse failure). KILL-gated (no cowardice survive-stage); frontier slice (25°) votes on the
# hardest end. Warm from runs/curriculum-69/_auto_002.pt (the 3M energy-managed converter peak). Judge
# ONLY the out-of-band greedy jitter-0.5 verdict/WEZ probe ([[display-killpct-is-exploration]]); a kill
# recovery is the win, else a rising wez%/relEs + high-yoyo count (via _yoyo_probe_tmp.py) shows the BFM
# skill forming. If the high yoyo never fires enough to teach (probe the run), lower the highyoyo trigger
# (_HY splice gate) below CLOSURE_HI — edit picked up at process restart.
# rows: (aspect_deg, aspect_deg_max, winrate)  native aggressive T3, co-heading, offensive, sep 700.
# Backstops are set at launch via --stage-backstop-steps (cumulative per index in build_stages): a big
# value keeps advancement GATE-driven so a walled rung doesn't backstop-decay onto a harder band.
_CHASE_T3_NEUTRAL = tuple(
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=float(lo),
        aspect_deg_max=float(hi),
        opp_heading_deg=None,
        separation_m=700.0,
        metric="kill",
        winrate=float(wr),
    )
    for lo, hi, wr in (
        (45.0, 80.0, 0.30),  # 0: hot converting pass — excludes the 0-45 gift; the 45-60 zone fires the high-yoyo seed
        (55.0, 90.0, 0.22),  # 1: pushes toward the beam (less seed, reward-driven conversion) — the harder end
    )
)


# curriculum-53 (the HEAD-ON MERGE program): kill a peer at the neutral head-on merge — the skill
# NOTHING we've trained can do (fresh head-on verdict: champion 0 kills/decks 1/3; c51 pure nose-to-nose
# self-play 100% deck; c52 rehearsal self-play merge-STABLE but passive/0-dmg). The reframe: hold the
# head-on geometry FIXED (aspect 0 + opp_heading 180) and make the OPPONENT the difficulty ladder — the
# inverse of chase-t3-killgate (which held the scripted T3 and scaled OUR positional disadvantage). The
# key lever the infra always had but we never used at the merge: `opponent_tier` decouples the AI from the
# geometry (env.py make_tier3_env) — 1 = straight NON-shooter, 2 = breaker (evades, NON-shooting), 3 = the
# full BFM shooter. c51 collapsed because it threw the agent COLD into a full-T3 head-on (hardest geometry
# + only shooter, no ramp). Here the agent first learns the merge PASS + convert-to-six against a target
# that can't shoot back (T1), then follows an evading turn (T2), THEN faces return fire (T3). DESIRED
# TACTIC (not head-on face-shot spray — a low-% mutual trade): lead-turn off the merge to gain angles,
# pursue to the rear quarter, kill with a TRACKING shot from behind (low LOS-rate → the 4° WEZ cone tracks).
# reward-v14b already biases this (K_ASPECT stern-conversion, K_AIM/K_CLOSE, WEZ/waste discipline); the
# ramp + self-play (Phase B) must teach only the MERGE-TURN decision (one/two-circle, lead-turn, energy).
# Anti-deck: spawn alt 15000 ft (~14k ft AGL of room) + the existing deck cushion; the T1-first ramp is
# itself the primary anti-deck mechanism (no incoming fire while learning the turn). Reserve lever if a
# rung decks >20%: raise spawn altitude / strengthen K_DECK_FLOOR. KILL-gated throughout (no cowardice
# survive-stage). Judge ONLY by the out-of-band jittered greedy head-on verdict (_verdict_tmp.py
# --opp-heading 180). NOT self-play (Phase A); Phase B (selfplay at the merge) warm-starts from the
# Phase-A T3-head-on killer. Launch TWO in parallel: c53 warm from champion (curriculum-46/_snap_2244k,
# --lr-scale 0.5, reuses the turning-fight finish — the post-merge fight IS a chase, so it won't forget),
# c53b from scratch (--resume "", --lr-scale 1.0, the clean test unconfounded by chase overfit).
_MERGE_T123 = tuple(
    dict(
        tier=3,
        opponent_tier=ot,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=180.0,
        separation_m=float(sep),
        metric="kill",
        winrate=float(wr),
    )
    # opponent ramp T1(no-fire)→T2(evade,no-fire)→T3(shooter); separation tightens each pair. Rung 4
    # (first full-T3 head-on kill) is the milestone "beat T3 head-on"; winrate eases 0.50→0.25 up the ramp.
    for ot, sep, wr in (
        (1, 1500.0, 0.50),
        (1, 1000.0, 0.50),
        (2, 1200.0, 0.40),
        (2, 900.0, 0.35),
        (3, 1200.0, 0.30),
        (3, 900.0, 0.25),
    )
)


# curriculum-75 (the LONG-RANGE NEUTRAL-MERGE ENERGY FIGHT). User authorized a PROBLEM REFRAME (option B)
# out of the confirmed one-pass merge-kill ceiling: 6 reward families × ~16 runs × 5 architectures all
# deck-or-draw the SYMMETRIC co-speed merge KILL because it is a geometric no-gun-solution IN ONE PASS
# ([[merge-crossing-velocity-ceiling]], [[curriculum-74-lead-null-disproof]]). Real pilots don't win the
# neutral merge on the first pass ("a straight merge is always a death") — they SURVIVE it, OUT-ENERGY the
# adversary over MULTIPLE turns, and finish from the rear quarter once he has gone slow. So this ladder
# stops judging the first merge and instead gives the fight ROOM to develop:
#   • FULLY NEUTRAL LONG-RANGE start — head-on (aspect_deg=0 + opp_heading_deg=180) at 2.2-3.0 km,
#     co-ALTITUDE / co-SPEED (NO handicap — the user's hard constraint: energy is EARNED, not gifted).
#     The long approach gives room to enter the merge with a self-built energy edge (climb/dive on the
#     run-in); the neutral head-on is the hardest fair geometry and the actual goal ("kill a peer coming
#     AT it"). vs the NATIVE aggressive T3 (opponent_tier=0 = the full Tier3Opponent shooter, the goal).
#   • LONG 180 s episode (--max-time-s 180, 1.5× the c55 cap) so the post-merge TURNING FIGHT has room
#     for 2-4 reversals — the win comes on a LATER pass, not the (drawn) first merge (c55/merge-t123 all
#     truncated/gated the first pass, so the multi-turn kill signal never arrived).
#   • WARM from runs/BASELINE_merge_survivor_2480k.pt — it already SURVIVES the head-on merge (0-deck /
#     0-loss) AND carries the ≤55° rear-quarter TRACKING-KILL finish (the hard skill). The post-merge fight
#     IS a chase, so warm-starting won't forget the finish; --lr-scale 0.5 protects the fragile peak.
#   • reward-v24 (K_REL_ENERGY 20→50, constants.py): make OUT-ENERGIZING the T3 a first-class objective so
#     RL takes the lag / keeps the vertical / waits for the T3's energy-bleeding break to go slow, THEN
#     converts — instead of matching his break into a symmetric-bleed draw. The energy advantage is what
#     DISSOLVES the crossing (a slow T3 is velocity-matchable → lead nulls → tracking kill). Paired with
#     the existing K_ENERGY specific-Es PBRS + corner-speed anti-stall floor + WEZ convert-dwell.
#   • --rehearse-workers 2 keeps a co-heading FINISH anchor rolling (anti-catastrophic-forgetting; the
#     c26/c51/c56 lesson that cold single-geometry training collapses the deliverable).
# KILL-gated throughout (no cowardice survive-stage); backstop huge (--stage-backstop-steps) so
# advancement is GATE-driven. Judge ONLY the out-of-band greedy jitter-0.5 head-on verdict via
# _energy_merge_probe_tmp.py — WIN SIGNAL: survives (low deck), ends rel-Es > 0 (out-energized the T3),
# and KILLS from a LOW-closure rear-quarter tracking shot (not a high-speed crossing snapshot).
# rows: (separation_m, winrate)  head-on neutral, native aggressive T3, offensive, co-alt/co-speed.
_ENERGY_MERGE = tuple(
    dict(
        tier=3,
        opponent_tier=0,          # native aggressive T3 (the real adversary / the goal)
        posture="offensive",
        aspect_deg=0.0,           # bot dead ahead of the learner
        opp_heading_deg=180.0,    # bot flying AT the learner ⇒ neutral head-on merge (no overtake gift)
        separation_m=float(sep),
        metric="kill",
        winrate=float(wr),
    )
    for sep, wr in (
        (3000.0, 0.25),   # 0: long-range neutral merge — room to build an energy edge on the run-in + multi-turn after
        (2200.0, 0.22),   # 1: tighter merge, less approach room — must carry the energy skill in
    )
)


# curriculum-55: the MERGE→CHASE FUSION ladder. Diagnosis after c53b/c54 (merge-t123): the NEUTRAL
# HEAD-ON merge-KILL vs the scripted T3 inside 60 s is a GEOMETRIC DRAW — both merge-t123 runs
# passive-stalled at the T3 head-on rung (dealt ~0, fired ~1 round), and even scripted T3-vs-T3
# can't win the neutral merge (both deck/disengage) [[merge-kill-not-scripted]]. The agent's
# passive-draw there is the game-theoretically CORRECT refusal of a losing mutual gunfight, NOT
# cowardice. But the user's REAL goal ("kill an equally intelligent opponent coming AT it") does
# NOT require winning the merge FACE-SHOT — it requires SURVIVING the head-on pass, CONVERTING to
# a turning fight, and finishing from the REAR QUARTER: exactly the champion's proven ≤55° chase
# kill. So fuse the two halves the split runs each had (c54 had merge-fly-but-no-finish; the
# champion has finish-but-no-merge):
#   • WARM from the champion (curriculum-46/_snap_2244k) — it carries the HARD skill (the terminal
#     tracking kill from the six that no from-scratch run could discover); merge-survival is the
#     EASY skill (c54 learned it in ~1M). --lr-scale 0.5 keeps the fragile peak from degrading.
#   • LONG 120 s episodes (--max-time-s 120, ~2× the 60 s cap) so the post-merge turning fight has
#     ROOM to develop into a finishable rear-quarter chase — c51/c53's 60 s cap truncated the fight
#     before the conversion could complete, so the kill signal never arrived.
#   • EASE the champion into the head-on PASS via the T2 non-shooting EVADER + a SEPARATION ramp
#     (rung 0 co-heading = home turf, anti-forgetting anchor; then a quartering-then-full head-on
#     merge, rather than the COLD 180 start that decked c53's champion). *** c55-v1 REVISION ***:
#     the original design ramped the bot's START HEADING co→90→135→180 vs a straight-flying T1
#     DUMMY — but a stage-1 diagnostic (opp_heading 90 vs T1, greedy) returned spent=0 / dealt=0 /
#     0 kills-losses-timeouts, all 8 eps ending as left_bubble: a straight-flyer on ANY off-head
#     heading flies PERPENDICULAR STRAIGHT OUT of the engagement bubble before the learner can
#     engage (only 0° chase and 180° head-on are non-degenerate for a straight target — everything
#     between escapes). So the heading ramp is fundamentally incompatible with a straight dummy.
#     FIX: drop the T1 dummy entirely (it escapes + only teaches a merge FACE-SHOT, not conversion)
#     and use the T2 EVADER as the merge/conversion teacher — it MANEUVERS (break-turns), so it
#     STAYS in the bubble at any heading AND gives a real turning fight to convert against (the
#     whole point of the fusion), while still NOT shooting (anti-deck, no fire-pressure). c54
#     proved head-on T1/T2 are killable; the T3 SHOOTER head-on was the passive-stall wall.
#   • RAMP the opponent T2(evader)→T3(shooter): rungs 1-3 are the head-on merge + conversion vs the
#     non-shooting T2 evader (learn the pass + rear-quarter finish with NO return fire), eased by
#     separation 1500→1000; then rungs 4-6 introduce the T3 SHOOTER head-on, eased by separation
#     1800→1200→900 (rung 5 sep1200 = THE MILESTONE: survive the pass → convert → rear-quarter
#     finish vs return fire).
# KILL-gated throughout (metric="kill" → kill_timeout_penalty −50 on a passive draw, no cowardice
# survive-stage; the long episode makes passivity eat the full −50). rung 0 is BANDED [0,45] so
# the honest frontier slice votes on the champion's 45° strength (quick re-anchor); rungs 1-6 are
# fixed-aspect-0 merges (no band → the whole rung votes, frontier-slice a no-op). Judge ONLY by the
# out-of-band jittered-0.5 greedy head-on verdict (_verdict_tmp.py --opp-heading 180). Run:
# --ladder merge-chase --resume runs/curriculum-46/_snap_2244k.pt --start-stage 0 --max-time-s 120
# --lr-scale 0.5. If the champion still decks the merge despite the ease-in → raise spawn alt / fall
# back to a c54 merge-flier warm-start; if it holds the pass but stalls passive at the T3 shooter
# (rung 5) → that IS the confirmed geometric ceiling → escalate with the evidence.
# rows: (opponent_tier, opp_heading_deg, separation_m, winrate, aspect_deg_max)  aspect_deg=0 throughout
_MERGE_CHASE = tuple(
    dict(
        tier=3,
        opponent_tier=ot,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=hdg,
        separation_m=float(sep),
        aspect_deg_max=float(amax),
        metric="kill",
        winrate=float(wr),
    )
    for ot, hdg, sep, wr, amax in (
        (0, None, 700.0, 0.45, 45.0),  # 0: REHEARSE the champion's finish (co-heading chase, band [0,45])
        (2, 135.0, 1500.0, 0.40, -1.0),  # 1: quartering merge vs the NON-shooting T2 EVADER — ease into the pass
        (2, 180.0, 1500.0, 0.35, -1.0),  # 2: FULL head-on merge vs the T2 evader — survive pass → follow break → convert
        (2, 180.0, 1000.0, 0.30, -1.0),  # 3: tighter head-on vs the T2 evader — higher closure, master the conversion
        (3, 180.0, 1800.0, 0.25, -1.0),  # 4: head-on vs the T3 SHOOTER at long sep — introduce return fire gently
        (3, 180.0, 1200.0, 0.22, -1.0),  # 5: FULL head-on merge vs the T3 SHOOTER = THE MILESTONE
        (3, 180.0, 900.0, 0.20, -1.0),  # 6: full head-on T3 at closer range (higher closure)
    )
)


# curriculum-56: the SURVIVE-CONVERT ladder — the honest reframe of the head-on merge program after
# 5 warm-start collapses (c49/c50/c51/c53/c55) confirmed that a reliable neutral-merge FACE-SHOT kill
# is a geometric draw even for a perfect pilot [[merge-kill-not-scripted]], and that the champion
# FREEZES at every merge geometry (no turning circle in its weights → RL can't learn from experience
# it never generates). Two changes from _MERGE_CHASE (the c55 kill-gated version that PASSIVE-STALLED):
#   (1) the win bar is the user-chosen SURVIVE + CONVERT — a `net`/`survive` gate, NOT a kill gate —
#       so a clean-survive-then-convert advances even when the merge face-shot is a draw. `net` = kill
#       OR (no losing terminal: no deck/death/trade/bubble AND net HP dealt−taken ≥ threshold); a pure
#       passive clock-draw deals 0 → net 0 → FAILS (+ the `net` timeout penalty), so it still forbids
#       cowardice. The top rung is `survive` (just DON'T LOSE the pass) where positive net at a close
#       head-on may be a true geometric draw.
#   (2) it is paired with the IMITATION-SEED merge-turn override (--merge-override-p0): a scripted
#       non-decking VERTICAL EGG (_vertical_egg_stick, c58) spliced into rollouts whenever a live bandit
#       is off the nose within merge range, giving the frozen champion the merge-circle MOTOR PRIOR it
#       lacks (the c55 wall was that NO override taught the turn). RL then adds the finish; the champion
#       supplies the ≤55° rear-quarter tracking kill for the "finish when geometry allows" part.
# Structure mirrors _MERGE_CHASE's opponent+separation ramp (so the geometry difficulty is identical,
# only the gate is honest): rung 0 co-heading REHEARSE (anti-forgetting anchor on the champion's 45°
# strength, band [0,45] so frontier-slice-8 votes there); rungs 1-3 EASE into the pass vs the
# NON-shooting T2 EVADER (it maneuvers → stays in the bubble at any heading AND gives a real turning
# fight to convert against, unlike a straight T1 dummy which escapes off-head — the c55 revision
# lesson), quartering 135°→full head-on 180°, separation 1500→1000; rungs 4-5 introduce the T3 SHOOTER
# head-on (return fire), separation 1800→1200 (rung 5 = THE MILESTONE: survive the pass → convert →
# come out net-ahead vs a peer coming AT it); rung 6 = the closest T3 head-on, `survive`-gated (don't
# lose the pass). 60 s episodes (NOT 120 s — c55 proved 120 s is a deck-inducer for a passive policy).
# Warm from the champion (curriculum-46/_snap_2244k) + --lr-scale 0.5. Judge ONLY by the out-of-band
# jittered-0.5 greedy head-on verdict (_verdict_tmp.py --opp-heading 180). NOT self-play (Phase A);
# Phase B (self-play at the merge) would warm from a c56 survive-convert flier.
# rows: (opponent_tier, opp_heading_deg, separation_m, winrate, net_threshold, metric, aspect_deg_max)
_MERGE_SURVIVE = tuple(
    dict(
        tier=3,
        opponent_tier=ot,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=hdg,
        separation_m=float(sep),
        aspect_deg_max=float(amax),
        metric=metric,
        winrate=float(wr),
        net_threshold=float(nthr),
    )
    for ot, hdg, sep, wr, nthr, metric, amax in (
        (0, None, 700.0, 0.50, 5.0, "net", 45.0),  # 0: REHEARSE co-heading finish (band [0,45], anchor)
        (2, 135.0, 1500.0, 0.45, 5.0, "net", -1.0),  # 1: quartering merge vs the NON-shooting T2 evader — ease in
        (2, 180.0, 1500.0, 0.40, 4.0, "net", -1.0),  # 2: FULL head-on vs T2 — survive pass → follow break → convert
        (2, 180.0, 1000.0, 0.35, 4.0, "net", -1.0),  # 3: tighter head-on vs T2 — higher closure, master conversion
        (0, 180.0, 1800.0, 0.30, 3.0, "net", -1.0),  # 4: head-on vs the T3 SHOOTER, long sep — gentle return fire
        (0, 180.0, 1200.0, 0.25, 3.0, "net", -1.0),  # 5: FULL head-on vs T3 = THE MILESTONE (survive → convert → net+)
        (0, 180.0, 900.0, 0.45, 0.0, "survive", -1.0),  # 6: closest T3 head-on — just DON'T LOSE the pass
    )
)


# curriculum-60 (reward-v17): the MERGE-CONVERT ladder for the two-policy converter M (see the
# CONVERT_* / reward-v17 note in constants.py, and the curriculum-59 memory for why the shared net
# hit a ceiling). M is FROZEN-finisher-free: its ONLY job is to survive the pass and reach a
# rear-quarter firing position behind a LIVE bandit (the "conversion" terminal), where a geometric
# selector hands the jet to the frozen champion F. So EVERY rung is `convert`-gated (a KILL does not
# clear M's bar — that is F's job) with convert_bonus wired run-level via --convert-bonus → env_kwargs.
#
# Geometry note: unlike _MERGE_SURVIVE's rung 0 (co-heading, which spawns M ALREADY behind → a
# degenerate instant conversion), EVERY rung here starts at a real MERGE (opp_heading 120→180 →
# high spawn aspect), so conversion always requires the reversal skill we are training. Difficulty
# ramps on three axes at once: heading 120°(shallow quartering)→180°(full head-on), opponent
# T2-breaker→T3-shooter (the _MERGE_SURVIVE lesson: a straight T1 dummy escapes off-head, so use the
# maneuvering-but-NON-shooting T2 as the stay-in-the-bubble conversion target before the T3 shooter),
# and separation wide→close (wide = more room/time to lead-turn behind = easier). The convert winrate
# eases 0.50→0.30 toward the milestone. Rungs 5-6 = THE DELIVERABLE: convert the head-on pass vs the
# full scripted T3 shooter, then hand off to F to finish (measured by the two-policy hand-off eval).
# rows: (opponent_tier, opp_heading_deg, separation_m, winrate)   [metric always "convert"]
_MERGE_CONVERT = tuple(
    dict(
        tier=3,
        opponent_tier=ot,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=hdg,
        separation_m=float(sep),
        aspect_deg_max=-1.0,
        metric="convert",
        winrate=float(wr),
    )
    for ot, hdg, sep, wr in (
        (2, 120.0, 1800.0, 0.50),  # 0: shallow quartering merge vs the NON-shooting T2 evader, wide — easiest reversal
        (2, 135.0, 1600.0, 0.45),  # 1: steeper quartering vs T2
        (2, 150.0, 1500.0, 0.45),  # 2: near-head-on vs T2
        (2, 165.0, 1500.0, 0.40),  # 3: almost head-on vs T2
        (2, 180.0, 1500.0, 0.40),  # 4: FULL head-on vs the NON-shooting T2 — master the pure conversion (no return fire)
        (0, 180.0, 1800.0, 0.35),  # 5: head-on vs the T3 SHOOTER, wide — gentle return fire (MILESTONE begins)
        (0, 180.0, 1200.0, 0.30),  # 6: closest head-on vs T3 = THE DELIVERABLE (convert → hand off to F to finish)
    )
)


# curriculum-62 (reward-v19, the BFM pivot): the HEAD-ON ladder for the single-net BFM program. Every
# rung is a FULL head-on merge (opp_heading 180°, aspect 0) — the geometry the user demanded we train
# ("fly BFM to reach the tail from a head-on IC", "a straight merge is always a death"). Difficulty
# ramps on two axes: opponent T2-breaker (non-shooting evader — learn the convert→kill with NO return
# fire) → T3-shooter (the deliverable), and separation wide→close (wide = more room/time to BFM behind).
# UNLIKE the c60 two-policy _MERGE_CONVERT ladder (whose rungs were `convert`-gated — reach the rear
# quarter, then hand off to a frozen finisher), EVERY rung here is `kill`-gated: this is the ONE net,
# the coach seeds the merge→convert→FINISH trajectory, and reward-v19's outcome reward (dropped dense
# pursuit; kept sparse damage + conversion-dwell + energy) rewards the whole thing. The kill winrates
# are modest — a head-on kill vs T3 is the hard unsolved problem — and the honest greedy verdict
# (--jitter 0.5 --opp-heading 180) is the real judge, not the train-log kill%.
# rows: (opponent_tier, separation_m, kill-winrate)   [tier 3, posture offensive, aspect 0, hdg 180]
_BFM_HEADON = tuple(
    dict(
        tier=3,
        opponent_tier=ot,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=180.0,
        separation_m=float(sep),
        aspect_deg_max=-1.0,
        metric="kill",
        winrate=float(wr),
    )
    for ot, sep, wr in (
        (2, 1500.0, 0.30),  # 0: head-on vs the NON-shooting T2 evader, wide — learn merge→BFM→kill, no return fire
        (2, 1100.0, 0.25),  # 1: tighter head-on vs T2 — higher closure, master the conversion→finish
        (0, 1800.0, 0.20),  # 2: head-on vs the T3 SHOOTER, wide — gentle return fire (MILESTONE begins)
        (0, 1200.0, 0.18),  # 3: FULL head-on vs T3 — the real fight
        (0, 900.0, 0.15),  # 4: closest head-on vs T3 = THE DELIVERABLE (survive → BFM behind → gun the kill)
    )
)


# curriculum-49: the RE-ATTACK ladder — a SCENARIO attack on the ≥55° terminal-FINISH wall.
# The wall is a MISSING SKILL, not a reward/planning defect: after the first pass crushes the bandit
# to ~18 HP the greedy policy OVERSHOOTS to ~1.25 km, gun_quality collapses to ~0, and it COASTS to
# the clock firing 0 of its ~400 remaining rounds — it never turns back (c42 finish-diag). Reward
# v12–v15, MPC planning, and the scripted re-attack override (splices a PursuitExpert reversal) ALL
# failed to instil the reverse→re-acquire→FINISH: the stick-only override taught a reverse-COAST (the
# very cowardice we must kill), and every reward/exploration/stability lever hit a method ceiling
# (c43–c48). The untried KIND of lever = the SCENARIO: spawn the bandit already WOUNDED (opp_health
# ~30 HP) so EVERY episode starts in the exact post-first-pass STATE — wounded bandit off the nose,
# learner high-energy with full ammo — and, KILL-gated, the policy must reverse and finish or fail.
# Because TD-MPC2 conditions on the OBSERVATION (bandit HP / bearing / range / closure), not on how
# that state was reached, the "finish a wounded bandit off the nose" skill learned here is the SAME
# state the policy meets after wounding a full-HP bandit itself — so it transfers to the real fight
# (measured OUT-OF-BAND vs the FULL-HP T3 at 45/55/60/65°). Band [0, hi] keeps the whole forward
# hemisphere in rotation (no forgetting the mastered low-aspect kill) while the CEILING is pushed PAST
# the beam into the reversal regime (90→180); the honest frontier-slice gate (--frontier-slice-deg)
# requires the outer reversal wedge itself to net-KILL before the band widens. Warm-start from the
# best 45° killer (curriculum-46/_snap_2244k) so the first-pass gun skill is carried in.
#
# curriculum-49 → curriculum-50 CORRECTION: 30 HP DESTROYED the champion (@2.35M greedy vs full-HP T3:
# 0 kills, 26 deaths, 0 damage dealt at every aspect; 11/12 decked at 60/65°). Root cause = FIRE
# STARVATION: a 30-HP bandit dies to a ~2-round override burst, so the buffer accrues almost no
# "track-then-fire-in-WEZ" transitions for the pi head to imitate → greedy firing decays to zero → the
# whole gun-BFM strategy destabilises and the policy flies nose-low into the ground. (c38/c46 worked
# because a FULL-HP bandit forces the override to fire a long sustained burst = plenty of firing
# experience.) A wounded bandit + fire-override are self-defeating together at 30 HP. FIX: 65 HP
# (≈72 rounds to kill at 45° → a sustained finishing burst, firing experience preserved, distribution
# shift from 100 HP kept small) + a GENTLE ceiling ladder that starts just above the mastered 45° kill.
_REATTACK_WOUNDED = tuple(
    dict(
        tier=3,
        opponent_tier=0,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=float(hi),
        opp_heading_deg=None,
        separation_m=float(sep),
        opp_health=65.0,  # curriculum-50: MILDLY wounded (see the c49→c50 note above) — a sustained-burst finish
        metric="kill",
        winrate=float(wr),
    )
    # curriculum-50 GENTLE ladder: ceilings 50→90 climbing from JUST ABOVE the champion's competence
    # (45°=8/12) so stage 0 is masterable and the aspect increases only after a kill-gate pass — no
    # cold jump to the beam. c49's [0,90]→[0,180] ladder decked the policy at ≥60° (11/12 crashes):
    # a warm champion thrown straight at beam-and-reversal geometry over-committed nose-low and hit the
    # ground. 50 = barely above the mastered 45° kill; 65 = the ≥60° TARGET; 80/90 = beam. Kill bar
    # eases 0.45→0.30 as the frontier nears the beam; separation held at 900 m.
    for hi, sep, wr in (
        (50, 900.0, 0.45),
        (65, 900.0, 0.40),
        (80, 900.0, 0.35),
        (90, 900.0, 0.30),
    )
)


def build_stages(
    rows=_CHASE_THEN_DEFEND,
    *,
    backstop_steps: int,
    start_stage: int = 0,
    sep_override: float = 0.0,
    winrate_override: float = 0.0,
    kill_timeout_penalty: float = 0.0,
    damage_timeout_penalty: float = 0.0,
) -> list:
    """Turn (tier, sep, metric, winrate) rows into `Stage`s with a cumulative step-backstop.

    Backstops are RELATIVE to `start_stage` (stage i fires at max(0, i−start_stage)·
    backstop_steps), so a `--resume` run budgets only the remaining stages from collected=0.
    `sep_override`>0 forces every stage's separation and `winrate_override`>0 forces every
    stage's gate bar (both for debugging / the CPU smoke); 0 keeps the per-stage values.
    `kill_timeout_penalty` (reward-v7, ≤0) is applied on a clock timeout at every KILL-metric
    rung and suppressed (0) at SURVIVE rungs, where surviving to the clock is the win.
    `damage_timeout_penalty` (≤0) does the same for DAMAGE rungs — added for the curriculum-18
    reversal ladder: a greedy eval of the c18 stage-1 (aspect-30) policy came back 12/12
    timeouts / 0 HP dealt / +12 return, i.e. it reverted to the c16 passive-draw optimum because
    with a 0 timeout penalty a nose-on track-draw banks the +15 TRACK_BUDGET for a positive
    return and off-axis conversion (which risks a deck) is EV-negative. A moderate penalty makes
    a no-damage draw a loss so gradient descent must convert; keep it milder than |R_DEATH| so a
    draw still beats a death (no suicide incentive) and far above R_DECK (deck stays worst)."""

    def _timeout_penalty(metric: str) -> float:
        # SURVIVE: 0 (clock = win). DAMAGE: damage_timeout_penalty (c18 reversal / c20 merge —
        # a no-damage draw must be a loss). KILL: kill_timeout_penalty.
        if str(metric) == "survive":
            return 0.0
        if str(metric) in ("damage", "net"):
            # net (c27) shares the damage-rung rationale: a no-outcome clock draw must be a loss
            # so gradient descent converts rather than parking nose-on to bank TRACK_BUDGET.
            return float(damage_timeout_penalty)
        return float(kill_timeout_penalty)

    stages = []
    for i, row in enumerate(rows):
        # curriculum-20 rows are dicts (explicit keys) so a merge rung can carry an
        # `opponent_tier` without overloading the legacy positional (type-sniffed) tuple parser.
        if isinstance(row, dict):
            metric = str(row["metric"])
            stages.append(
                Stage(
                    tier=int(row.get("tier", 3)),
                    separation_m=(
                        float(sep_override) if sep_override > 0 else float(row["separation_m"])
                    ),
                    metric=metric,
                    winrate=(
                        float(winrate_override) if winrate_override > 0 else float(row["winrate"])
                    ),
                    backstop=max(0, i - start_stage) * int(backstop_steps),
                    timeout_penalty=_timeout_penalty(metric),
                    damage_threshold=float(row.get("damage_threshold", 0.0)),
                    net_threshold=float(row.get("net_threshold", 0.0)),
                    posture=str(row.get("posture", "offensive")),
                    aspect_deg=float(row.get("aspect_deg", 180.0)),
                    aspect_deg_max=float(row.get("aspect_deg_max", -1.0)),
                    opponent_tier=int(row.get("opponent_tier", 0)),
                    opp_heading_deg=(
                        None
                        if row.get("opp_heading_deg") is None
                        else float(row["opp_heading_deg"])
                    ),
                    opp_health=(
                        None
                        if row.get("opp_health") is None
                        else float(row["opp_health"])
                    ),
                )
            )
            continue
        tier, sep, metric, wr = row[0], row[1], row[2], row[3]
        # Trailing optional fields (curriculum-17): a numeric = the HP-dealt bar (damage rungs);
        # a string = the Tier-3 starting posture. Parse by type so a row may carry either, both
        # (numeric then string), or neither — keeping the legacy 4/5-tuple rows unchanged.
        extras = row[4:]
        nums = [float(x) for x in extras if isinstance(x, (int, float))]
        dmg_thr = nums[0] if nums else 0.0
        # curriculum-18: a 2nd numeric extra is the Tier-3 offensive-posture start BEARING
        # (aspect_deg); legacy rows carry ≤1 numeric extra, so aspect defaults to 180° (dead six).
        aspect_deg = nums[1] if len(nums) > 1 else 180.0
        posture = next((str(x) for x in extras if isinstance(x, str)), "offensive")
        stages.append(
            Stage(
                tier=int(tier),
                separation_m=(float(sep_override) if sep_override > 0 else float(sep)),
                metric=str(metric),
                winrate=(float(winrate_override) if winrate_override > 0 else float(wr)),
                backstop=max(0, i - start_stage) * int(backstop_steps),
                # Clock-timeout penalty (reward-v7). SURVIVE rungs: 0 (clock = win). KILL rungs:
                # kill_timeout_penalty. DAMAGE rungs: damage_timeout_penalty — 0 by default, but
                # the c18 reversal ladder sets it <0 so a no-damage draw is a loss (a 0 penalty
                # let the greedy policy park nose-on and bank TRACK_BUDGET for a passive +draw).
                timeout_penalty=_timeout_penalty(metric),
                damage_threshold=dmg_thr,
                posture=posture,
                aspect_deg=aspect_deg,
            )
        )
    return stages


# curriculum-29 honest FRONTIER gate: width (deg) of the hard OUTER slice of a training band that
# alone votes on stage mastery. Every c29 rung trains the FULL [0, hi] band (base kept in rotation →
# no catastrophic forgetting, the c28 bug), but ADVANCE requires the outer wedge [hi−slice, hi] to
# genuinely net-win — so a base-dominated band AVERAGE (the c27/c28 masking lie: [0,80] read 60% on
# free 0-45 kills while 70-90 sat at 0%) can no longer promote a rung whose frontier is unmastered.
FRONTIER_SLICE_DEG = 25.0


def _in_frontier(aspect, stage, slice_deg: float = FRONTIER_SLICE_DEG) -> bool:
    """Does this probe's START aspect sit in the current band's FRONTIER slice — the hard outer
    wedge [aspect_deg_max − slice_deg, aspect_deg_max] the c29 honest gate requires the
    policy to master before the band widens? A rung with no band ceiling (aspect_deg_max < 0, i.e.
    a fixed single-aspect or non-banded rung) has no frontier → the whole rung counts, preserving
    the legacy band-average gate. A missing aspect (non-tier3 / pre-c29 collector) also counts, so
    an older collector or a non-banded ladder never silently drops every vote.

    curriculum-36: `slice_deg` is configurable (--frontier-slice-deg, default FRONTIER_SLICE_DEG=25).
    c35 exposed the 25° slice as still too wide on the [0, hi] chase bands — a stage that "mastered
    75°" killed 0/40 at 75° because 50–60° kills (well inside the [50,75] slice) carried the gate
    vote. A narrow slice (c36 uses 10°) forces the outer wedge itself to net-win before advancing."""
    hi = stage.aspect_deg_max
    if hi < 0.0:
        return True  # no band ceiling → not a frontier ladder; count the whole rung
    if aspect is None:
        return True  # aspect unknown → don't discard the vote
    lo = max(float(stage.aspect_deg), hi - slice_deg)
    return float(aspect) >= lo


def _gate_should_advance(win_flags, window: int, min_episodes: int, winrate: float) -> bool:
    """Performance gate: has the learner *mastered* the current stage? True iff at least
    `min_episodes` episodes have been played at this stage AND the success-rate over the
    most recent `window` of them is ≥ `winrate`. `win_flags` is the per-episode 1/0 success
    log for the CURRENT stage (the caller clears it on every advance and applies the stage's
    metric). Disabled — always False — when `winrate <= 0`. A large window at a high bar is
    what makes advancement demand *sustained* mastery, not a lucky streak. Pure function →
    unit-tested in tests/test_curriculum_gate.py."""
    if winrate <= 0.0:
        return False
    n = len(win_flags)
    if n < max(1, min_episodes):
        return False
    recent = win_flags[-window:] if window > 0 else win_flags
    return (sum(recent) / len(recent)) >= winrate


def stage_at(stages, collected: int, current: int) -> int:
    """Highest stage index whose step-backstop has elapsed, never below `current`
    (monotonic — the gate may already have advanced past the backstop)."""
    idx = current
    for i in range(current + 1, len(stages)):
        if collected >= stages[i].backstop:
            idx = i
    return idx


def _next_stage(stages, current: int):
    """The next rung up, or None if `current` is already the top of the ladder."""
    return current + 1 if current + 1 < len(stages) else None


def _advance_stage(stages, current, collected, wins, *, gate_window, gate_min_episodes):
    """Which stage to be at now — one pure, unit-testable decision. Monotonic (never steps
    back). Backstop first (a slow gate cannot stall the run forever), then the performance
    gate at THIS stage's metric + bar. Returns `(target_idx, reason)`; target==current means
    hold. `wins` is the per-episode 1/0 success log at the current stage (its metric already
    applied by the caller), cleared on every advance."""
    bs = stage_at(stages, collected, current)
    if bs > current:
        return bs, f"step {collected:,} backstop"
    st = stages[current]
    if _gate_should_advance(wins, gate_window, gate_min_episodes, st.winrate):
        nxt = _next_stage(stages, current)
        if nxt is not None:
            recent = wins[-gate_window:] if gate_window > 0 else wins
            return (
                nxt,
                f"gate: {st.metric}-rate {sum(recent) / len(recent):.0%} over last {len(recent)}",
            )
    return current, ""


def _save_ckpt(
    agent, path: str, stage_idx: int, collected: int, provenance: dict | None = None
) -> None:
    """Persist a resume-able checkpoint: the agent's full state (model + both optimizers +
    cfg, as `eval_policy` expects) plus the curriculum position. `load_state_dict` and the
    eval harness both read only their own keys, so the extra `stage_idx`/`collected` ride
    along harmlessly and let `--resume` continue exactly where the run left off.

    `provenance` carries the run settings that are NOT part of cfg and therefore not
    restored by `--resume` — above all `outcome_mode`, which selects the reward SHAPE
    (convert-dwell, K_ENERGY, rel-Es PBRS, corner floor, the in-zone closure gate). A
    checkpoint that does not say which reward trained it cannot be warm-started safely,
    and silently resuming it under the other reward reads as a training collapse."""
    sd = agent.state_dict()
    sd["stage_idx"] = int(stage_idx)
    sd["collected"] = int(collected)
    if provenance:
        sd["provenance"] = dict(provenance)
    torch.save(sd, path)


def ramped_discount(done: int, span: float, start: float, target: float) -> float:
    """γ at `done` collected steps on a linear walk from `start` to `target` over `span` steps.

    Quantised into 1000 discrete increments: a `--compile`d `agent.update` retraces when a
    Python float it closes over changes, so a fresh γ every step would recompile forever.
    Clamped at both ends, so `done > span` holds the target and a `span <= 0` caller is the
    no-ramp case (handled before this is ever called)."""
    frac = min(1.0, max(0.0, done / span))
    frac = round(frac * 1000.0) / 1000.0
    return start + (target - start) * frac


def stage_env_extra(
    stage,
    *,
    is_rehearse: bool,
    me_dalt_ft: float = 0.0,
    merge_alt_ft: float = 0.0,
    merge_alt_ft_max: float = 0.0,
    merge_lateral_m: float = 0.0,
) -> dict:
    """The scenario kwargs a stage adds on top of the run-level `env_kwargs` splat.

    Lifted out of `_make_env` to be testable. It was a closure, and that is exactly how the
    `--me-dalt-ft` anchor leak survived a whole run unnoticed: the perch is a RUN-level knob and
    every run-level knob reaches the rehearsal anchor unless something explicitly stops it, since
    the anchor is itself a tier-3/offensive stage and matches the same guard. Only `Stage` fields
    distinguish the anchor, so `is_rehearse` has to be passed in and checked here.

    Three knobs must never touch the anchor. The anchor's entire job is to keep replaying the exact
    regime the champion earned — co-altitude, ≤55°, 700 m, 60 s — so that a long merge run cannot
    forget the finish. Shifting its start distribution destroys the one guarantee it provides:
      * `me_dalt_ft` (curriculum-79 perch) — the learner's altitude advantage;
      * `merge_alt_ft` (curriculum-91 block altitude) — the fight's altitude BUDGET;
      * `merge_lateral_m` (curriculum-92 merge lateral separation) — cross-range turning room.
    All are silently dropped on non-tier-3 and defensive-posture rungs too, because only
    `tier3_scenario` accepts them and the tier-1/2 factories reject unknown kwargs."""
    extra: dict = {}
    # `posture` is Tier-3-only (which start geometry vs the Tier3Opponent); the tier-1/2 factories
    # don't accept it. `aspect_deg` (the c18 reversal ladder) is only meaningful for the offensive
    # builder (tier3_scenario) — the defensive one (tier1_scenario) has no such knob.
    if stage.tier != 3:
        return extra
    extra["posture"] = stage.posture
    if stage.posture != "offensive":
        # curriculum-20's weaker-AI swap is posture-independent, so it applies on both branches.
        if stage.opponent_tier > 0:
            extra["opponent_tier"] = stage.opponent_tier
        return extra
    extra["aspect_deg"] = stage.aspect_deg
    if me_dalt_ft and not is_rehearse:
        extra["me_dalt_ft"] = me_dalt_ft
    if merge_alt_ft and not is_rehearse:
        extra["alt_ft"] = merge_alt_ft
        # The band ceiling only means anything alongside a floor, so it rides inside this guard —
        # a ceiling with no floor would otherwise silently do nothing.
        if merge_alt_ft_max:
            extra["alt_ft_max"] = merge_alt_ft_max
    # curriculum-92: the merge's LATERAL SEPARATION — a signed ±merge_lateral_m cross-range band on
    # top of the existing jitter wobble. Shielded from the anchor for the same reason as the two
    # altitude knobs: the anchor is a 700 m tail-chase drill, and ±1200 m of cross-range at 700 m
    # separation is not "the chase with an offset", it is a different fight entirely.
    if merge_lateral_m and not is_rehearse:
        extra["lateral_m_max"] = merge_lateral_m
    # curriculum-27: a per-episode RANDOM bearing band [aspect_deg, aspect_deg_max] instead of a
    # fixed bearing, so widening the ceiling can't catastrophically forget the easy tail-chase
    # (the c26 failure). Only meaningful under jitter (it needs the reset rng); disabled at −1.
    if stage.aspect_deg_max >= 0.0:
        extra["aspect_deg_max"] = stage.aspect_deg_max
    # curriculum-21: the bot's ABSOLUTE start heading (None = co-heading north). Head-on merge =
    # aspect 0 + heading 180.
    if stage.opp_heading_deg is not None:
        extra["opp_heading_deg"] = stage.opp_heading_deg
    # curriculum-49: spawn the bandit "wounded" so every episode is a re-attack finish drill (the
    # learner starts in the post-first-pass state). None = full-HP (100), the legacy behaviour.
    if stage.opp_health is not None:
        extra["opp_health"] = stage.opp_health
    # curriculum-20: swap in a weaker opponent AI at the tier-3 geometry (0 = native T3). Under
    # self-play the env opponent is overwritten per episode, so this is only a placeholder there.
    if stage.opponent_tier > 0:
        extra["opponent_tier"] = stage.opponent_tier
    return extra


def run_parallel(
    cfg: TDMPC2Config,
    *,
    workers: int,
    steps: int,
    seed_steps: int | None,
    stages,
    env_kwargs: dict,
    weight_sync_every: int,
    weight_check_every: int,
    utd: float,
    queue_max: int,
    save: str,
    save_every: int,
    log_every: int,
    device: torch.device,
    start_stage: int = 0,
    resume_sd: dict | None = None,
    lr_scale: float = 1.0,
    gate_window: int = 100,
    gate_min_episodes: int = 120,
    frontier_slice_deg: float = FRONTIER_SLICE_DEG,
    fire_override_p0: float = 0.0,
    fire_override_steps: int = 0,
    fire_override_quality: float = 0.6,
    reattack_override_p0: float = 0.0,
    reattack_override_steps: int = 0,
    reattack_override_ata_deg: float = 50.0,
    merge_override_p0: float = 0.0,
    merge_override_steps: int = 0,
    merge_override_range_m: float = 1500.0,
    merge_override_ata_deg: float = 30.0,
    merge_override_mode: str = "egg",
    bfm_override_p0: float = 0.0,
    bfm_override_steps: int = 0,
    expert_p0: float = 0.0,
    expert_steps: int = 0,
    probe_every: int = 0,
    selfplay: bool = False,
    pool_dir: str = "",
    selfplay_snapshot_every: int = 0,
    selfplay_recency_bias: float = 0.0,
    selfplay_max_pool: int = 50,
    selfplay_scripted_prob: float = 0.0,
    rehearse_workers: int = 0,
    rehearse_stage=None,
    discount_ramp_steps: int = 0,
    discount_start: float = 0.0,
) -> dict:
    """The central learner loop. Spawns `workers` collectors, then ingests + updates.

    Returns a summary `{collected, updates, episodes, ret_last, tier}` for callers/tests."""
    probe_gate = probe_every > 0  # gate on deterministic probes (honest) vs any unassisted ep
    agent = TDMPC2(cfg, device)
    if cfg.compile:
        agent.update = torch.compile(agent.update)  # type: ignore[method-assign]
    if resume_sd is not None:
        agent.load_state_dict(resume_sd)  # model + both optimizers, restored from --resume
    # Gentle-consolidation lever: scale ALL optimizer LRs (both world-model groups incl. the
    # encoder-at-enc_lr_scale group, and pi_optim) by lr_scale AFTER the load, so it overrides
    # the checkpoint's restored param-group LR. Preserves the enc-vs-rest ratio. lr_scale<1 makes
    # a warm-started converged policy consolidate instead of oscillate off its peak.
    if lr_scale != 1.0:
        for _og in (agent.optim, agent.pi_optim):
            for _pg in _og.param_groups:
                _pg["lr"] *= lr_scale
        _lrs = [f"{_pg['lr']:.2e}" for _og in (agent.optim, agent.pi_optim) for _pg in _og.param_groups]
        print(f"[lr-scale] x{lr_scale} -> optimizer LRs: {_lrs}", flush=True)
    # curriculum-89 DISCOUNT RAMP. Raising γ on a warm champion as a step change collapses it:
    # c88 (0.995→0.9985 instantly) fell monotonically (ret +59/+58/−42/−91/−287/−463 per 100k,
    # kill% 13→6) and c88b (→0.998) traced the same curve ~40% slower, so the collapse is
    # magnitude-dependent, not a threshold. Cause: 1/(1−γ) rescales every bootstrapped value
    # target (200→667 at 0.9985), so the restored critic is badly miscalibrated at the new γ and
    # the policy — which just maximises Q — follows it down. (Not head saturation: the symlog
    # two-hot head spans ±10 in SYMLOG space ≈ ±22,000 real.) Fix: walk γ from the CHECKPOINT's
    # value to the target over `discount_ramp_steps` collected steps so the critic re-fits as it
    # moves. Quantised into 1000 discrete increments so a `--compile`d `agent.update` can't
    # recompile per step. Collectors are unaffected (prior-only collection never reads discount;
    # under --mpc they re-read it on the next broadcast). ⚠ a mid-ramp --resume RESTARTS the ramp
    # from the checkpoint's cfg discount, which is the target — i.e. resuming mid-ramp lands you
    # at the target immediately; finish a ramp inside one run.
    discount_target = float(agent.discount)
    ramp_span = float(discount_ramp_steps) if discount_ramp_steps > 0 else 0.0
    ramp_from = float(discount_start) if discount_start > 0.0 else discount_target
    ramping = ramp_span > 0.0 and abs(discount_target - ramp_from) > 1e-9
    if ramping:
        agent.discount = ramp_from
        print(
            f"[discount-ramp] γ {ramp_from:.4f} -> {discount_target:.4f} over "
            f"{int(ramp_span):,} collected steps "
            f"(credit horizon {1.0 / (1.0 - ramp_from):.0f} -> "
            f"{1.0 / (1.0 - discount_target):.0f} steps)",
            flush=True,
        )
    elif ramp_span > 0.0:
        print(
            f"[discount-ramp] no-op: γ already {discount_target:.4f} "
            "(pass --discount to set a different target)",
            flush=True,
        )

    buffer = ReplayBuffer(cfg, device)
    seed_steps = seed_steps or cfg.resolved_seed_steps()
    stage_idx = int(start_stage)

    ctx = mp.get_context("spawn")
    episode_q = ctx.Queue(maxsize=queue_max)
    weight_qs = [ctx.Queue(maxsize=2) for _ in range(workers)]
    stage_value = ctx.Value("i", stage_idx)
    stop_event = ctx.Event()

    # The fire-override step budget is GLOBAL (same unit as --steps / --curriculum), so
    # split it across the K collectors: each sees ~1/workers of the aggregate control-steps,
    # and the aggregate hits `fire_override_steps` exactly when every worker has run this far
    # into the tier. p0 ≤ 0 or steps ≤ 0 disables the override.
    fire_override_anneal = (fire_override_steps / workers) if fire_override_steps > 0 else 0.0
    # Same GLOBAL→per-worker split for the re-attack override (curriculum-43); the ata overshoot gate
    # is converted deg→rad once here (bandit this far off the nose, with range opening, = overshot).
    reattack_override_anneal = (
        (reattack_override_steps / workers) if reattack_override_steps > 0 else 0.0
    )
    reattack_override_ata_rad = float(np.radians(reattack_override_ata_deg))
    # Same GLOBAL→per-worker split for the merge-turn override (curriculum-56); the ata trigger gate
    # is converted deg→rad once here (bandit this far off the nose, within range, = a merge to turn on).
    merge_override_anneal = (
        (merge_override_steps / workers) if merge_override_steps > 0 else 0.0
    )
    merge_override_ata_rad = float(np.radians(merge_override_ata_deg))
    # Same GLOBAL→per-worker split for the BFM-coach exploration budget (curriculum-62).
    bfm_override_anneal = (bfm_override_steps / workers) if bfm_override_steps > 0 else 0.0
    # Same GLOBAL→per-worker split for the expert-rollout budget (see fire_override_anneal).
    expert_anneal = (expert_steps / workers) if expert_steps > 0 else 0.0

    # Self-play: SEED the opponent pool before any collector starts, so episode 1 already has an
    # opponent to fight (the warm-started champion if --resume, else the fresh policy). The learner
    # then snapshots itself into the pool every `selfplay_snapshot_every` steps (an ever-stronger
    # opponent as it improves). A resumed self-play run whose pool dir already holds snapshots keeps
    # them (the ladder continues) and just appends past the highest existing index.
    snapshot_idx = 0
    if selfplay:
        if not pool_dir:
            raise ValueError("self-play requires a pool_dir (set --pool-dir or --save)")
        os.makedirs(pool_dir, exist_ok=True)
        existing = sorted(Path(pool_dir).glob("snap_*.pt"))
        if existing:
            snapshot_idx = 1 + max(int(p.stem.split("_")[-1]) for p in existing)
            print(f"self-play: pool at {pool_dir} has {len(existing)} snapshot(s); appending")
        else:
            path = _snapshot_prior(agent, cfg, pool_dir, snapshot_idx)
            snapshot_idx += 1
            print(f"self-play: seeded pool → {path}")

    # curriculum-57: the LAST `rehearse_workers` collectors are dedicated co-heading-finish rehearsers
    # (each pins `rehearse_stage`, never follows the ladder, never probes) so the shared buffer keeps a
    # steady stream of the champion's ≤55° tracking-kill transitions — the anti-forgetting anchor the
    # c56 merge run lacked (it forgot the finish: 45° 12/12 → 0/12). The first workers stay merge
    # collectors that drive the honest gate. rehearse_workers=0 → every legacy run unchanged.
    n_rehearse = int(rehearse_workers) if rehearse_stage is not None else 0
    n_rehearse = max(0, min(n_rehearse, workers - 1))  # always keep ≥1 merge worker to drive the gate
    procs = []
    for w in range(workers):
        w_rehearse_stage = rehearse_stage if w >= (workers - n_rehearse) else None
        p = ctx.Process(
            target=_collector_loop,
            args=(w, cfg, env_kwargs, stages, stage_value, episode_q, weight_qs[w], stop_event),
            kwargs={
                "seed": cfg.seed + 1 + w * 100003,
                "weight_check_every": weight_check_every,
                "rehearse_stage": w_rehearse_stage,
                "fire_override_p0": fire_override_p0,
                "fire_override_anneal": fire_override_anneal,
                "fire_override_quality": fire_override_quality,
                "reattack_override_p0": reattack_override_p0,
                "reattack_override_anneal": reattack_override_anneal,
                "reattack_override_ata_rad": reattack_override_ata_rad,
                "merge_override_p0": merge_override_p0,
                "merge_override_anneal": merge_override_anneal,
                "merge_override_range_m": merge_override_range_m,
                "merge_override_ata_rad": merge_override_ata_rad,
                "merge_override_mode": merge_override_mode,
                "bfm_override_p0": bfm_override_p0,
                "bfm_override_anneal": bfm_override_anneal,
                "expert_p0": expert_p0,
                "expert_anneal": expert_anneal,
                "probe_every": probe_every,
                "selfplay": selfplay,
                "pool_dir": pool_dir,
                "selfplay_recency_bias": selfplay_recency_bias,
                "selfplay_scripted_prob": selfplay_scripted_prob,
            },
            daemon=True,
        )
        p.start()
        procs.append(p)

    # `outcome_mode` is a REWARD SHAPE switch that lives only in env_kwargs — it is not in
    # cfg, so `--resume` does not restore it. It gates the convert-dwell term, K_ENERGY, the
    # rel-Es PBRS, the corner-speed floor and the in-zone CONVERT_CLOSURE_MAX gate, i.e. the
    # constants curriculum-83/84 earned. Omitting --outcome-reward silently deletes them and
    # the resulting flat run reads as yet another non-holdability collapse. Echo it here so
    # the log says which reward actually ran.
    print(
        f"device {device} | workers {workers} | start_stage {stage_idx}"
        f"{' (RESUMED)' if resume_sd is not None else ''} | "
        f"obs {cfg.obs_dim} act {cfg.action_dim} | "
        f"collect {'CPU-MPPI' if cfg.mpc else 'prior-only'} | discount {agent.discount:.4f} | "
        f"horizon {cfg.horizon} | reward {'OUTCOME' if env_kwargs.get('outcome_mode') else 'dense-only'} | "
        f"seed_steps {seed_steps} | episode_length {cfg.episode_length} | steps {steps:,}"
    )
    # The exact launch line. No run before curriculum-86 recorded one, so reconstructing which
    # flags a checkpoint was actually trained with meant guessing from the log's derived fields —
    # and the flags that matter most (--outcome-reward, --rehearse-workers, --me-dalt-ft) leave no
    # other trace. One line here makes every run self-describing.
    print(f"argv: {' '.join(sys.argv[1:])}")
    # ...and the env-var switches, which argv cannot show. `ASTRUM_C97_VERTICAL` selects the obs
    # width (45 = the c97 vertical treatment, 41 = its null control) and is exactly the
    # [[outcome-reward-invisible-switch]] shape: it changes what the policy can SEE and leaves no
    # trace in the launch line, so it is echoed here alongside the width it produced.
    from astrum_dogfight import constants as _C

    print(
        f"env: ASTRUM_C97_VERTICAL={os.environ.get('ASTRUM_C97_VERTICAL', '<unset>')} "
        f"=> OBS_DIM {_C.OBS_DIM} (c97 vertical obs {'ON' if _C.C97_VERTICAL_OBS else 'OFF'})"
    )
    # Non-cfg run settings a checkpoint must carry so a later --resume can tell whether the
    # reward it was trained under matches the reward it is being resumed into.
    provenance = {
        "outcome_mode": bool(env_kwargs.get("outcome_mode", False)),
        "ladder_len": len(stages),
        "rehearse_workers": int(n_rehearse),
        "me_dalt_ft": float(env_kwargs.get("me_dalt_ft", 0.0)),
        "merge_alt_ft": float(env_kwargs.get("merge_alt_ft", 0.0)),
        "merge_alt_ft_max": float(env_kwargs.get("merge_alt_ft_max", 0.0)),
        "merge_lateral_m": float(env_kwargs.get("merge_lateral_m", 0.0)),
        "max_time_s": float(env_kwargs.get("max_time_s", 0.0)),
        "argv": list(sys.argv[1:]),
    }
    print(
        f"curriculum: {len(stages)} stages | gate window {gate_window} min_eps {gate_min_episodes}"
        f" | frontier slice {frontier_slice_deg:.0f}°"
    )
    # Echo the altitude setup on its own line: the block altitude decides how many seconds of
    # fight the floor allows, so a run's altitude budget should never have to be inferred.
    if env_kwargs.get("merge_alt_ft") or env_kwargs.get("me_dalt_ft"):
        block_ft = float(env_kwargs.get("merge_alt_ft") or 15000.0)
        block_hi_ft = float(env_kwargs.get("merge_alt_ft_max") or block_ft)
        perch_ft = float(env_kwargs.get("me_dalt_ft", 0.0))
        band = f"{block_ft:,.0f}" if block_hi_ft == block_ft else f"{block_ft:,.0f}-{block_hi_ft:,.0f}"

        def _budget_s(ft):  # 68.3 m/s = the curriculum-90-measured head-on bleed rate
            return (ft + perch_ft) * 0.3048 / 68.3

        budget = (
            f"~{_budget_s(block_ft):.0f} s"
            if block_hi_ft == block_ft
            else f"~{_budget_s(block_ft):.0f}-{_budget_s(block_hi_ft):.0f} s"
        )
        print(
            f"altitude: block {band} ft + perch {perch_ft:,.0f} ft = "
            f"{(block_ft + perch_ft) * 0.3048:,.0f}-{(block_hi_ft + perch_ft) * 0.3048:,.0f} m AGL "
            f"⇒ {budget} of budget at the curriculum-90 bleed rate "
            f"(clock {env_kwargs.get('max_time_s', 60.0):.0f} s)"
        )
    # curriculum-92: echo the merge's LATERAL SEPARATION, and check it against the OPPONENT's threat
    # cone in the same breath. Turning room is only meaningful relative to a turn radius, and the
    # band is only valid while the T3 still breaks — past the cone the header would still say "T3"
    # while the benchmark had quietly become an easier opponent, which is unfalsifiable-result
    # territory. Both numbers are derived here so neither has to be recomputed by hand later.
    if env_kwargs.get("merge_lateral_m"):
        from math import sin, sqrt

        from astrum_dogfight import constants as C

        lat_m = float(env_kwargs["merge_lateral_m"])
        jit = float(env_kwargs.get("jitter", 0.0))
        # Worst case over the offensive tier-3 rungs: the wobble scales with each rung's separation.
        seps = [s.separation_m for s in stages if s.tier == 3 and s.posture == "offensive"]
        wobble_m = jit * 0.09 * max(seps or [0.0])
        total_m = lat_m + wobble_m
        # 634 m = the F-16 turn radius at the curriculum-90-measured 225 m/s merge speed and the
        # 8.2 g the T3 bot actually pulls (V²/(g·√(n²−1))).
        radius_m = 634.0
        dz_m = float(env_kwargs.get("me_dalt_ft", 0.0)) * 0.3048
        # The T3 goes defensive only inside THREAT_CONE_RAD at THREAT_RANGE, and `ata` is the TOTAL
        # off-nose angle (elevation counts), so the perch eats into the horizontal allowance.
        cone_m = C.THREAT_RANGE * sin(C.THREAT_CONE_RAD)
        cliff_m = sqrt(cone_m**2 - dz_m**2) if cone_m > dz_m else 0.0
        print(
            f"merge lateral: ±{lat_m:,.0f} m cross-range band (±{wobble_m:,.0f} m jitter wobble on "
            f"top ⇒ |L| ≤ {total_m:,.0f} m = {total_m / radius_m:.2f} turn radii @{radius_m:.0f} m) "
            f"| T3 break holds to {cliff_m:,.0f} m "
            f"({'margin ' + format(cliff_m - total_m, ',.0f') + ' m' if total_m < cliff_m else 'EXCEEDED — benchmark moved, result uninterpretable'})"
            f" | anchor EXEMPT"
        )
    if n_rehearse:
        print(
            f"rehearse: {n_rehearse}/{workers} workers pinned to the chase anchor "
            f"(t3 aspect [{rehearse_stage.aspect_deg:.0f},{rehearse_stage.aspect_deg_max:.0f}]° "
            f"@{rehearse_stage.separation_m:.0f} m, {rehearse_stage.max_time_s or env_kwargs.get('max_time_s', 60.0):.0f} s episodes)"
        )
    for i, s in enumerate(stages):
        mark = "→" if i == stage_idx else ("·" if i > stage_idx else "✓")
        tp = f" | timeout {s.timeout_penalty:+.0f}" if s.timeout_penalty else ""
        if s.metric == "damage":
            goal = f"{s.metric}(≥{s.damage_threshold:.0f}HP)"
        elif s.metric == "net":
            goal = f"{s.metric}(≥{s.net_threshold:.0f}HP, no-death)"
        else:
            goal = s.metric
        # Show the start geometry for Tier-3 rungs: posture + (offensive-posture) bot bearing.
        geo = ""
        if s.tier == 3:
            geo = f" | {s.posture}"
            if s.posture == "offensive":
                # c27: a random bearing BAND [aspect, aspect_max] prints as e.g. aspect0-60.
                if s.aspect_deg_max >= 0.0:
                    geo += f"/aspect{s.aspect_deg:.0f}-{s.aspect_deg_max:.0f}"
                else:
                    geo += f"/aspect{s.aspect_deg:.0f}"
            if s.opponent_tier > 0:  # curriculum-20: opponent AI decoupled from geometry
                geo += f"/opp-T{s.opponent_tier}"
        print(
            f"  {mark} stage {i:2d}: tier {s.tier} @ {int(s.separation_m):>4d} m | "
            f"gate {goal}-rate ≥ {s.winrate:.0%} | backstop {s.backstop:,} steps{tp}{geo}"
        )
    if fire_override_anneal > 0.0:
        print(
            f"guided-fire override | p0 {fire_override_p0:.2f} → 0 over {fire_override_steps:,} "
            f"global steps ({fire_override_anneal:,.0f}/worker) per stage | "
            f"gun_quality ≥ {fire_override_quality:.2f}"
        )
    if reattack_override_anneal > 0.0:
        print(
            f"re-attack override | p0 {reattack_override_p0:.2f} → 0 over {reattack_override_steps:,} "
            f"global steps ({reattack_override_anneal:,.0f}/worker) per stage | overshoot = opening ∧ "
            f"ata > {reattack_override_ata_deg:.0f}° ∧ bandit-alive ∧ ammo | PursuitExpert stick (gate-excluded)"
        )
    if merge_override_anneal > 0.0:
        print(
            f"merge-turn override | p0 {merge_override_p0:.2f} → 0 over {merge_override_steps:,} "
            f"global steps ({merge_override_anneal:,.0f}/worker) per stage | merge = bandit-alive ∧ "
            f"rng < {merge_override_range_m:.0f} m ∧ ata > {merge_override_ata_deg:.0f}° | scripted "
            + (
                "HIGH-YO-YO stick (UP/bleed = bank up + nose-high only on a genuine overshoot closure > "
                "CLOSURE_HI, else DOWN/reacquire = roll onto the LOS at full throttle; anti-overshoot; gate-excluded)"
                if merge_override_mode == "highyoyo"
                else "LOW-YO-YO stick (hard bank + nose-near-horizon, hold energy; anti-zoom-stall; gate-excluded)"
                if merge_override_mode == "lowyoyo"
                # curriculum-95: a banner that prints "VERTICAL-EGG" for every mode is the same
                # unauditable-header defect as [[--outcome-reward invisible switch]] — the lever is in
                # argv but not in the human-readable line anyone reads to check WHAT ran.
                else "one-circle LEAD-TURN stick, per-EPISODE coin (APPROACH: hold the predicted CPA "
                "at ~800 m of turning room at corner speed → TURN: γ-corrected max-rate turn TOWARD "
                "his pass side, committed at 1,000 m → TRACK: lag-pursuit to the shot; latched "
                "3-phase, deck-guarded in all phases; gate-excluded)"
                if merge_override_mode == "leadturn"
                else "VERTICAL-EGG stick (modest bank + max-g nose-high pull; gate-excluded)"
            )
        )
    if bfm_override_anneal > 0.0:
        print(
            f"BFM-coach override | p0 {bfm_override_p0:.2f} → 0 over {bfm_override_steps:,} "
            f"global steps ({bfm_override_anneal:,.0f}/worker) per stage | doctrine BFMCoach flies "
            f"whole HEAD-ON offensive-t3 episodes (merge→egg/reversal→rear-quarter track+fire; "
            f"gate-excluded) — the reward-v19 outcome-reward exploration prior"
        )
    if expert_anneal > 0.0:
        print(
            f"expert-rollout override | p0 {expert_p0:.2f} → 0 over {expert_steps:,} "
            f"global steps ({expert_anneal:,.0f}/worker) per stage | scripted PursuitExpert "
            f"flies whole episodes (excluded from the honest gate)"
        )
    if n_rehearse > 0:
        rs = rehearse_stage
        print(
            f"rehearsal workers | {n_rehearse}/{workers} collectors PINNED to a co-heading FINISH env "
            f"(t{rs.tier}@{int(rs.separation_m)}m, aspect [{rs.aspect_deg:.0f},{rs.aspect_deg_max:.0f}]°, "
            f"opp_tier {rs.opponent_tier}, {rs.metric}) | anti-forgetting anchor | never probes (gate stays "
            f"the {workers - n_rehearse} merge worker(s))"
        )
    jit = float(env_kwargs.get("jitter", 0.0))
    if probe_gate:
        print(
            f"honest gate: every {probe_every}th episode is a DETERMINISTIC probe (greedy, no "
            f"expert/fire-override) | ONLY probes vote on stage mastery"
        )
    if jit > 0.0:
        print(f"start jitter: strength {jit:.2f} (per-episode range/bearing/heading/alt/speed)")
    if selfplay:
        rb = f"recency-bias {selfplay_recency_bias:.2f}" if selfplay_recency_bias > 0 else "uniform"
        snap = (
            f"snapshot every {selfplay_snapshot_every:,} steps (max pool {selfplay_max_pool})"
            if selfplay_snapshot_every > 0
            else "no periodic snapshot (seed-only pool)"
        )
        anchor = (
            f"ANCHORED: {selfplay_scripted_prob:.0%} of exploration eps + ALL probes vs the SCRIPTED "
            f"T3 (gate measures the real target)"
            if selfplay_scripted_prob > 0
            else "pure fictitious self-play (no scripted anchor)"
        )
        print(
            f"SELF-PLAY: opponent = pooled learner priors at {pool_dir} | {rb} | {snap} | "
            f"the scripted opponent is replaced per-episode by a frozen copy of the learner\n"
            f"           {anchor}"
        )

    def broadcast():
        blob = _serialize_model(agent.model)
        for q in weight_qs:
            try:
                q.put_nowait(blob)
            except queue.Full:
                pass  # collector hasn't consumed the last one; it'll get the next

    collected = 0
    updates = 0
    ep_count = 0
    recent_returns: list[float] = []
    stage_wins: list[
        float
    ] = []  # per-episode 1/0 success log at the CURRENT stage (cleared on advance)
    last_stats: dict = {}
    seeded = (
        resume_sd is not None
    )  # a resumed agent is already trained → skip the cold pretrain burst
    next_log = log_every
    next_save = save_every
    # Self-play: first prior snapshot fires `selfplay_snapshot_every` collected steps in (step 0
    # was seeded before the collectors spawned); 0 disables the cadence (pool stays at the seed).
    next_snapshot = selfplay_snapshot_every if (selfplay and selfplay_snapshot_every > 0) else 0
    t_start = time.monotonic()

    def ingest(episodes) -> None:
        nonlocal collected, ep_count, stage_idx
        for ep in episodes:
            buffer.add_episode(
                ep["obs"], ep["action"], ep["reward"], ep["terminated"], ep["truncated"]
            )
            collected += int(ep["action"].shape[0])
            ep_count += 1
            recent_returns.append(float(ep["ret"]))
            # Honest gate (curriculum-14): only DETERMINISTIC PROBE episodes vote on mastery —
            # the greedy, deployable policy flown with no exploration noise, no expert and no
            # fire-override, over the jittered start distribution. The curriculum-13 ground-truth
            # eval proved the earlier "unassisted stochastic-prior" gate lied: it read 70% while
            # the deployable policy killed 0/20 greedy (lucky-noise kills inflated the rate). The
            # probe outcome is exactly what an acceptance eval would measure, so gating on it means
            # a stage clears only once the policy we ACTUALLY deploy has mastered it. Probes still
            # train the buffer (added above); the step-backstop remains the slow-gate safety net.
            # (probe_gate off / legacy: fall back to counting any non-override episode.)
            counts_for_gate = ep.get("probe") if probe_gate else not ep.get("override_fired")
            if counts_for_gate:
                # curriculum-29 honest FRONTIER gate: on a banded rung, only probes whose START
                # aspect lands in the band's hard outer slice [hi−FRONTIER_SLICE_DEG, hi] vote on
                # mastery. The full [0, hi] band is still FLOWN and TRAINED (base replay → no
                # forgetting), but ADVANCE now measures the frontier's own net-win rate, so a
                # base-dominated band average can't promote an unmastered frontier (the c28 lie).
                # Non-banded rungs / missing aspect fall through to the legacy whole-rung vote.
                if _in_frontier(ep.get("probe_aspect"), stages[stage_idx], frontier_slice_deg):
                    stage_wins.append(1.0 if _stage_win(ep, stages[stage_idx]) else 0.0)

            # Curriculum advance (buffer + model persist across it): the performance gate
            # (only clear a stage once the learner reliably meets its metric — kill or
            # survive) or the step backstop, whichever fires first, monotonic. See
            # _advance_stage.
            target, reason = _advance_stage(
                stages,
                stage_idx,
                collected,
                stage_wins,
                gate_window=gate_window,
                gate_min_episodes=gate_min_episodes,
            )
            if target != stage_idx:
                if save:
                    mp_path = _milestone_path(save, stage_idx, stages[stage_idx])
                    # Store the position to RESUME at (the next stage), so --resume from a
                    # milestone continues forward rather than repeating the mastered stage.
                    _save_ckpt(agent, mp_path, target, collected, provenance)
                    print(f"  ↳ saved stage-{stage_idx} milestone → {mp_path}")
                print(f"  ↳ step {collected:,}: curriculum stage {stage_idx} → {target} ({reason})")
                stage_idx = target
                stage_value.value = stage_idx
                stage_wins.clear()  # restart the mastery measurement for the new stage

    def collectors_alive() -> bool:
        return any(p.is_alive() for p in procs)

    def abort_if_collectors_dead() -> None:
        """Collectors never exit on their own (they loop until `stop_event`), so all of
        them being gone means they crashed — and we can never reach the budget. Sweep up
        any episodes still in flight (`mp.Queue.empty()` is unreliable across the feeder
        thread, so drain with a short block instead), then fail loudly. Runs on every
        idle iteration so a crash aborts promptly regardless of the UTD state — not only
        in the capped branch, which a busy learner may never reach."""
        if collectors_alive():
            return
        stragglers = []
        for _ in range(20):
            try:
                stragglers.append(episode_q.get(timeout=0.05))
            except queue.Empty:
                break
        if stragglers:
            ingest(stragglers)
            return
        raise RuntimeError(
            f"all {workers} collectors exited before the {steps:,}-step budget "
            f"(collected {collected:,}, {updates:,} updates) — see the collector "
            f"traceback above for the cause"
        )

    if resume_sd is not None:
        broadcast()  # push the trained policy to collectors immediately (no cold seed phase)

    try:
        while collected < steps:
            episodes = _drain(episode_q, max_items=queue_max)
            if episodes:
                ingest(episodes)
            else:
                abort_if_collectors_dead()

            if not seeded:
                if len(buffer) >= seed_steps:
                    print(f"  pretraining on {len(buffer):,} seed transitions...")
                    for i in range(seed_steps):
                        last_stats = agent.update(buffer)
                        updates += 1
                        if i % 64 == 0:  # keep draining so collectors don't stall on a full queue
                            ingest(_drain(episode_q, max_items=queue_max))
                    seeded = True
                    broadcast()
                elif not episodes:
                    # Buffer still warming up; block briefly for the first episodes.
                    # (A dead-collector stall is caught by abort_if_collectors_dead above.)
                    try:
                        ingest([episode_q.get(timeout=1.0)])
                    except queue.Empty:
                        pass
                continue

            # Steady state: update back-to-back, capped at UTD to protect sample quality
            # if collectors ever fall behind the learner (utd<=0 disables the cap).
            can_update = len(buffer) >= buffer.slice_len
            under_cap = utd <= 0 or updates < utd * max(collected, 1)
            if can_update and under_cap:
                if ramping:
                    agent.discount = ramped_discount(
                        collected, ramp_span, ramp_from, discount_target
                    )
                last_stats = agent.update(buffer)
                updates += 1
                if updates % weight_sync_every == 0:
                    broadcast()
            else:
                time.sleep(0.001)  # UTD-capped: yield until fresh data arrives

            if collected >= next_log:
                next_log += log_every
                elapsed = time.monotonic() - t_start
                sps = collected / elapsed
                ups = updates / elapsed
                ret = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
                wr = float(np.mean(stage_wins[-gate_window:])) if stage_wins else float("nan")
                st = stages[stage_idx]
                utd_now = updates / max(collected, 1)
                msg = (
                    f"step {collected:>9,d} | upd {updates:>8,d} | utd {utd_now:4.2f} | "
                    f"eps {ep_count:4d} | ret(20) {ret:8.2f} | {st.metric[:4]} {wr:4.0%} | "
                    f"buf {len(buffer):>8,d} | s{stage_idx:02d} t{st.tier}@{int(st.separation_m)}m | "
                    f"{sps:6.0f} sps | {ups:5.0f} ups"
                    # Show γ only while it is MOVING — a ramped run's collapse (or survival) has to
                    # be readable against where γ actually was, not where it ended up (curriculum-88).
                    + (f" | γ {agent.discount:.4f}" if ramping else "")
                )
                if last_stats:
                    msg += (
                        f" | cons {last_stats['consistency_loss']:.3f} "
                        f"rew {last_stats['reward_loss']:.3f} val {last_stats['value_loss']:.3f} "
                        f"pi {last_stats['pi_loss']:+.3f}"
                    )
                print(msg, flush=True)

            if save and save_every and collected >= next_save:
                next_save += save_every
                _save_ckpt(agent, save, stage_idx, collected, provenance)

            # Self-play: drop a fresh frozen prior into the pool so collectors begin facing the
            # improved policy (fictitious self-play — the opponent tracks the learner's skill).
            if next_snapshot and collected >= next_snapshot:
                next_snapshot += selfplay_snapshot_every
                path = _snapshot_prior(agent, cfg, pool_dir, snapshot_idx)
                snapshot_idx += 1
                _prune_pool(pool_dir, selfplay_max_pool)
                live = len(list(Path(pool_dir).glob("snap_*.pt")))
                print(f"  ↳ step {collected:,}: self-play snapshot → {path} (pool {live})")
    except KeyboardInterrupt:
        print("\ninterrupted — shutting down collectors and saving...")
    finally:
        stop_event.set()
        # Don't let this process block at interpreter exit flushing queue feeder threads:
        # the learner puts weight blobs, and if a collector already died without draining
        # them the pipe can be full and the feeder-join would hang (we saw a shutdown
        # stall exactly here). The collectors cancel their own ends symmetrically.
        for q in weight_qs:
            q.cancel_join_thread()
        episode_q.cancel_join_thread()
        # Unblock any collector parked on a full queue, then let them exit.
        for _ in range(workers * 2):
            try:
                episode_q.get_nowait()
            except queue.Empty:
                break
        for p in procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
        if save:
            _save_ckpt(agent, save, stage_idx, collected, provenance)
            print(f"saved checkpoint → {save}  ({collected:,} steps, {updates:,} updates)")

    return {
        "collected": collected,
        "updates": updates,
        "episodes": ep_count,
        "ret_last": recent_returns[-1] if recent_returns else float("nan"),
        "stage": stage_idx,
        "tier": stages[stage_idx].tier,
    }


# Config fields a `--resume` is allowed to change, mapped to the CLI flags that decide
# them. Shape-BEARING fields (obs_dim, action_dim, latent_dim, num_q) are deliberately
# absent: they must match the checkpoint's tensors. `action_repeat` is absent too —
# `ActionRepeat` SUMS reward over its repeats, so changing it mid-run rescales the whole
# dense signal under a warm head. `episode_length` is derived from --max-time-s, hence the
# many-to-one mapping.
_RESUME_OVERRIDABLE: dict[str, frozenset[str]] = {
    "discount": frozenset({"--discount"}),
    "horizon": frozenset({"--horizon"}),
    "batch_size": frozenset({"--batch-size"}),
    "buffer_size": frozenset({"--buffer-size"}),
    "episode_length": frozenset({"--max-time-s", "--rate-hz"}),
    "seed_steps": frozenset({"--seed-steps"}),
    # Collection-only exploration spectrum (PROBE-2). MUST be listed here or a warm
    # `--resume` silently keeps the checkpoint's "white" and the flag becomes another
    # invisible switch (the [[outcome-reward-invisible-switch]] failure mode).
    "explore_noise": frozenset({"--explore-noise"}),
    "explore_beta": frozenset({"--explore-beta"}),
    "explore_tau_steps": frozenset({"--explore-tau-steps"}),
    # NB `steps` is deliberately absent: in THIS trainer the collection budget is `args.steps`
    # passed straight into `run_parallel`, and `cfg.steps` is unread (only the single-env
    # `train.py` loops on it). It is force-set from the CLI below so the saved cfg records the
    # budget the run actually had instead of the origin run's.
}


def _widen_resume_obs(resume_sd: dict, old_obs_dim: int, new_obs_dim: int, frame_stack: int) -> int:
    """Zero-init encoder-column surgery: make a checkpoint trained on a NARROWER obs loadable
    after `astrum_dogfight.constants.OBS_DIM` grew. Returns the new stacked obs dim.

    curriculum-87 appended 10 features to the observation ([[obs-blind-to-opponent-turn]]). Only
    ONE tensor in the whole model depends on the obs width — the encoder's first `NormedLinear`
    weight, `(enc_dim, obs_dim·frame_stack)` — because everything after it sees a fixed-width
    latent. Widening it with ZERO columns makes the resumed policy's output **bit-identical** on
    the old features while leaving the new ones connected but silent, so the new signal has to be
    EARNED by gradient descent rather than arriving as a step change in behaviour. That property
    is what makes an obs append a legitimate single-variable experiment on a warm champion, which
    matters here: every collapse in [[curriculum-65-settle-and-shoot]] began with a warm start
    that perturbed the policy on contact.

    Frame stacking is handled per-frame, not by appending at the end of the whole vector:
    `FrameStack` concatenates whole frames, so the live obs is `[frame_0(new) … frame_{k-1}(new)]`
    and the trained columns for frame f sit at `[f·new, f·new + old)`. Appending the pad at the
    very end instead would silently re-map every frame but the first onto the wrong weights.

    Adam's moments for the widened tensor are padded to match, preserving momentum for the trained
    columns; the new columns start at zero moment, which is exactly a fresh param's state.
    """
    key = "_encoder.0.weight"
    model = resume_sd["model"]
    w = model[key]
    old_stacked, new_stacked = old_obs_dim * frame_stack, new_obs_dim * frame_stack
    if w.shape[1] != old_stacked:
        raise SystemExit(
            f"[resume] cannot widen obs: {key} is {tuple(w.shape)} but the checkpoint's cfg implies "
            f"obs_dim·frame_stack = {old_stacked}. Refusing to guess the layout."
        )

    def _pad(t):
        out = torch.zeros(t.shape[0], new_stacked, dtype=t.dtype, device=t.device)
        for f in range(frame_stack):
            out[:, f * new_obs_dim : f * new_obs_dim + old_obs_dim] = t[
                :, f * old_obs_dim : (f + 1) * old_obs_dim
            ]
        return out

    model[key] = _pad(w)

    # The optimizer keys state by POSITIONAL param index, so locate the entry by matching the old
    # tensor's exact shape within the encoder param group (group 0 is `self._encoder.parameters()`,
    # see world_model._optim_groups) rather than hard-coding index 0. If that is ambiguous, drop
    # the state instead of corrupting it — Adam lazily re-creates zeroed moments.
    enc_group = resume_sd["optim"]["param_groups"][0]["params"]
    state = resume_sd["optim"]["state"]
    hits = [
        i
        for i in enc_group
        if i in state
        and getattr(state[i].get("exp_avg"), "shape", None) == torch.Size([w.shape[0], old_stacked])
    ]
    if len(hits) == 1:
        for m in ("exp_avg", "exp_avg_sq"):
            if m in state[hits[0]]:
                state[hits[0]][m] = _pad(state[hits[0]][m])
        moments = f"Adam moments padded (param {hits[0]})"
    else:
        for i in hits:
            state.pop(i, None)
        moments = f"Adam moments RESET ({len(hits)} ambiguous matches)"

    resume_sd["cfg"] = {**resume_sd["cfg"], "obs_dim": new_stacked}
    print(
        f"[resume] OBS WIDENED {old_obs_dim} -> {new_obs_dim} per frame (frame_stack {frame_stack}: "
        f"{old_stacked} -> {new_stacked} encoder inputs). {key} zero-padded per frame, {moments}. "
        f"The resumed policy is bit-identical on the old features; the {new_obs_dim - old_obs_dim} "
        f"new columns start silent.",
        flush=True,
    )
    return new_stacked


def _flags_passed(argv) -> frozenset[str]:
    """The long flags EXPLICITLY present on the command line.

    A resume override must fire only on an explicit flag, never on a parser default that
    happens to differ from the checkpoint — otherwise `--resume` of a 180 s run would
    silently snap `episode_length` back to the 60 s default. Handles both `--flag value`
    and `--flag=value`."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    return frozenset(t.split("=", 1)[0] for t in tokens if t.startswith("--"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Async multi-collector TD-MPC2 for the ASTRUM dogfight."
    )
    # -- shared with the single-env trainer (the opponent tier & schedule are NOT here — the
    #    curriculum is the per-stage (tier, separation, gate) ladder below, not a bare tier). --
    p.add_argument("--steps", type=int, default=6_000_000, help="total *collected* env steps")
    p.add_argument("--seed-steps", type=int, default=0, help="random warmup; 0 = derive from cfg")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    # -- start-geometry override. Separation is normally PER-STAGE (the distance curriculum;
    #    see _CHASE_THEN_DEFEND), where it widens close→far on offense and narrows far→near on
    #    defense. This flag, when >0, forces the SAME separation on every stage (debugging / the
    #    CPU smoke); 0 = use each stage's per-stage separation.
    p.add_argument(
        "--separation-m",
        type=float,
        default=0.0,
        help="force one separation on all stages (0 = per-stage)",
    )
    p.add_argument(
        "--airspeed-kts",
        type=float,
        default=0.0,
        help="both aircraft start airspeed (0 = tier default)",
    )
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--horizon", type=int, default=3)
    # -- credit horizon. The discount is applied PER CONTROL STEP (rate_hz/action_repeat =
    #    24 Hz), so the default resolved 0.995 is a 1/(1-γ) = 200-step = 8.33 s horizon —
    #    shorter than a single 180° reversal. A kill 30 s out is worth 4.06 of its 150,
    #    against a convert-dwell farm paying ~12/s, so multi-turn BFM is an investment that
    #    cannot pay. `TDMPC2Config.discount` already short-circuits `resolved_discount()`;
    #    this flag is the missing CLI surface. 0 = derive from episode_length as before.
    p.add_argument(
        "--discount",
        type=float,
        default=0.0,
        help="fixed per-control-step discount (0 = derive from episode_length; "
        "0.9985 ≈ 28 s credit horizon)",
    )
    # PROBE-2 (discoverability). ⚠ This block's ORIGINAL arithmetic was measured on the WRONG code
    # path and is superseded — kept here only because the number it quoted (σ=0.21, P≈1e-50) is cited
    # elsewhere in the record. It read the MPPI plan's first-action std via `agent.plan`, but
    # `cfg.mpc` is False in every shipped checkpoint, so `plan()` never runs during collection
    # (`agent.py:203`) and neither did the noise objects: they were only ever sampled at
    # `agent.py:272`, INSIDE `plan`. The live path is the actor's own Gaussian at `world_model.pi`.
    #
    # D2a re-measured it there (`_diag_actor_std.py`, handoff §9): σ_ail is **0.67-0.90**, i.e. 4× the
    # 0.21 quoted above and 13-18× `min_std` ⇒ **amplitude was never the limiter and an actor-side std
    # floor could not bind.** What IS the limiter is temporal correlation, and the honest statistic is
    # N_eff = var(x)/var(36-step window mean), which is exactly 36 for white noise: the actor measured
    # **34-41**, i.e. indistinguishable from independent, and P(|1.5 s window mean| > 0.5) was 0.000 in
    # all 40 channel-views. That is a stronger claim than the old one and does not depend on σ.
    #
    # D2b then wired these flags into the LIVE path (`agent._explore_eps` → `pi(z, eps=...)`,
    # pre-squash, on channels {0,1} = roll and g only) and D2c measured it closed-loop: the same
    # statistic goes **0.000 → 0.256** at `ou --explore-tau-steps 36`. Use `ou`, NOT `pink --beta 1`
    # (β=1 is the weakest coloured option at N_eff 2.20; see `config.py`'s measured table).
    # Collection-side only — no network, loss or buffer semantics change — so it remains a legitimate
    # single-variable warm-start experiment. Whether it CONVERTS to BFM or kills is still unrun.
    p.add_argument(
        "--explore-noise",
        default="white",
        choices=("white", "pink", "ou"),
        help="temporal spectrum of the per-control-step exploration noise: 'white' (default) = "
        "the TD-MPC2 reference i.i.d. draw; 'pink' = 1/f (Eberhard et al. ICLR 2023); "
        "'ou' = AR(1) at --explore-tau-steps. Same marginal variance in every mode.",
    )
    p.add_argument(
        "--explore-beta", type=float, default=1.0, help="pink spectral exponent (1=pink, 2=brown)"
    )
    p.add_argument(
        "--explore-tau-steps",
        type=float,
        default=36.0,
        help="OU correlation time in CONTROL steps (36 ≈ 1.5 s at 24 Hz)",
    )
    # curriculum-89: raising --discount on a warm champion as a STEP collapses it (c88/c88b),
    # because 1/(1-γ) rescales every value target and the restored critic is miscalibrated at the
    # new γ. Walk γ there instead, so the critic re-fits as it moves. Only meaningful with
    # --discount AND --resume; 0 = old behaviour (instant).
    p.add_argument(
        "--discount-ramp-steps",
        type=int,
        default=0,
        help="ramp γ from the checkpoint's value to --discount over this many collected steps "
        "(0 = apply instantly, which collapsed a warm champion in curriculum-88)",
    )
    p.add_argument("--compile", action="store_true", help="torch.compile the learner update")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="'auto' | 'cuda' | 'cpu' (learner device)")
    p.add_argument("--log-every", type=int, default=2000, help="collected steps between summaries")
    p.add_argument("--save", default="", help="checkpoint path (overwritten every --save-every)")
    p.add_argument("--save-every", type=int, default=0, help="checkpoint every N collected steps")
    # -- parallel-specific -------------------------------------------------
    p.add_argument("--workers", type=int, default=0, help="collector processes; 0 = auto")
    p.add_argument(
        "--rehearse-workers",
        type=int,
        default=0,
        help="curriculum-57: dedicate the LAST N collectors to a FIXED co-heading finish env "
        "(aspect [0,55]° vs native T3, sep 700 m, kill-gated) — an anti-forgetting anchor whose "
        "transitions feed the shared buffer so a merge ladder can't forget the ≤55° tracking kill. "
        "These workers never probe (the honest gate stays on the merge workers). 0 = disabled.",
    )
    p.add_argument(
        "--collector-mpc",
        action="store_true",
        help="collectors plan with CPU-MPPI (higher-quality data, slower) instead of prior-only",
    )
    p.add_argument(
        "--weight-sync-every", type=int, default=200, help="learner updates between broadcasts"
    )
    p.add_argument(
        "--weight-check-every", type=int, default=100, help="collector steps between weight polls"
    )
    p.add_argument(
        "--utd", type=float, default=1.0, help="max updates-per-collected-step (0 = uncapped)"
    )
    p.add_argument("--queue-max", type=int, default=0, help="episode queue depth; 0 = 4·workers")
    # -- distance-curriculum stages: the ladder is (tier, separation, gate-metric) rungs
    #    (see _CHASE_THEN_DEFEND / build_stages). Advance a stage only on SUSTAINED mastery
    #    of its metric (kill or survive); a per-stage step-backstop forces progress if the
    #    gate is slow so a stuck stage can't stall the run forever.
    p.add_argument(
        "--ladder",
        default="chase_then_defend",
        choices=(
            "chase_then_defend",
            "reversal",
            "merge",
            "beam-t3",
            "beam-t3-fine",
            "beam-t3-net",
            "beam-t3-push",
            "beam-t3-frontier",
            "beam-t3-killgate",
            "chase-t3-killgate",
            "chase-t3-neutral",
            "reattack-wounded",
            "selfplay",
            "selfplay-ladder",
            "selfplay-beam",
            "selfplay-nose",
            "merge-t123",
            "merge-chase",
            "merge-survive",
            "merge-convert",
            "bfm-headon",
            "energy-merge",
        ),
        help="which stage ladder: 'chase_then_defend' (T1→T2→T3 offense→survive, the default); "
        "'reversal' (c18: sweep the Tier-3 bot's start bearing 0°→180° / dead-six, damage-gated); "
        "'merge' (c20: OPPONENT-difficulty ladder at a fixed neutral/forward merge — learn the beam→six "
        "conversion vs a T1 dummy, harden T1→T2→T3; warm-start from a c19 aspect-0 killer with "
        "--resume + --start-stage 0); or 'beam-t3' (c25 HONEST: NON-self-play FINE aspect ladder 0→6→12→…→90 "
        "vs the NATIVE Tier-3 bot every episode — gate honest by construction (no self-play swap to fail "
        "silently, unlike the c24 anchoring that scored probes vs weak pooled copies); warm-start from the "
        "GENUINE c19 aspect-0 killer, pair with --gate-window 100 --gate-min-episodes 120 for a strict gate); "
        "or 'beam-t3-fine' (c26: the beam-t3 continuation after it demolished 0→40° honestly but WALLED at a "
        "sharp ~45-48° skill cliff — a FINE 5° aspect staircase 40→45→…→90 at the SAME 700 m so every rung is "
        "only 5° off a trained checkpoint (transfer stays nonzero); warm-start from the c25 aspect-40 milestone "
        "runs/curriculum-25/run_stage06_t3_700m.pt with --start-stage 0); "
        "or 'beam-t3-net' (c27: the HONEST rebuild after the c26 verdict exposed the damage-gate as blind to "
        "HP TAKEN — it certified a net-LOSING policy (net −27 at its own gated aspect) that also FORGOT low "
        "aspects. Fixes: a NET-outcome gate (win = kill OR net-HP-positive survival, no death/deck) + per-episode "
        "aspect BANDING [0,ceiling] so every stage rehearses the whole forward hemisphere (anti-forgetting). "
        "Ceiling ramps 40→50→55→60→70→80→90; warm-start the c25 aspect-0→40 net-winner run_stage06 with "
        "--start-stage 0, --damage-timeout-penalty -40, strict --gate-window 100 --gate-min-episodes 120); "
        "or 'beam-t3-frontier' (c29: the SYNTHESIS after c27's band-average gate MASKED the frontier (advanced "
        "on free 0-45 kills while 70-90 sat at 0%%) and c28's narrow frontier-only bands catastrophically FORGOT "
        "the base. Keeps the full [0,hi] band in TRAINING (base replay → no forgetting) but GATES on the outer "
        "slice [hi-25°,hi] ONLY (via per-probe start-aspect → the frontier's own net-win rate, not a band avg). "
        "Ceiling ramps 60→65→…→90, bar eases toward the beam; rides reward-v9's dense deck-cushion + expert "
        "hit-gated fire fixes. Warm-start the honest c27 [0,80] master run_stage05, --workers 3, lower "
        "--gate-window 60 --gate-min-episodes 80 (frontier probes are scarcer), --damage-timeout-penalty -40); "
        "or 'selfplay' (c21: single neutral-merge stage vs POOLED copies "
        "of the learner — auto-enables --selfplay; warm-start + seed the pool with --resume <champion>); "
        "or 'selfplay-ladder' (c22: self-play ANGLE ladder aspect 0→30→45→60→90 — bootstrap from a geometry "
        "the warm-start already converts vs its pooled self, harden to the beam merge; auto-enables --selfplay); "
        "or 'selfplay-beam' (c25: FINE beam-merge ladder aspect 45→52→60→…→90 in ~7-8° steps with EXPERT-BFM "
        "seeding (--expert-p0) — the fix for the c24 aspect-60 wall; warm-start from the c24 aspect-45 milestone); "
        "or 'merge-t123' (c53: head-on merge program — fix aspect0+opp_hdg180, ramp the OPPONENT T1→T2→T3); "
        "or 'merge-chase' (c55 FUSION: rung0 co-heading chase REHEARSAL, then ramp opp_heading co-head→90→135→180 "
        "vs a T1 dummy so the fragile champion eases into the head-on PASS + rear-quarter conversion with no "
        "return fire, then T2 evader, then the T3 shooter head-on = milestone; run with --max-time-s 120 so the "
        "post-merge turning fight develops into a finishable chase, warm champion + --lr-scale 0.5); "
        "or 'merge-survive' (c56 HONEST reframe: _MERGE_CHASE's opp+sep ramp but SURVIVE-CONVERT gated "
        "(net/survive, not kill — a merge face-shot is a geometric draw) + paired with the imitation-seed "
        "merge-turn override --merge-override-p0 that splices a scripted non-decking break/lead-turn to "
        "give the frozen champion the merge-circle motor prior; 60 s episodes, warm champion + --lr-scale 0.5); "
        "or 'merge-convert' (c60 TWO-POLICY: train ONLY the merge-converter M — every rung `convert`-gated "
        "(reach the bandit's rear quarter → episode ends, hand off to the FROZEN champion F to finish), "
        "opp_heading 120°→180° / opponent T2→T3 / sep wide→close; pair with --convert-bonus 150 and the "
        "merge-override, NO fire-override/rehearse-workers; warm champion for flight competence); "
        "or 'bfm-headon' (c62 the BFM PIVOT: SINGLE-net head-on ladder — every rung a full head-on merge "
        "(aspect0+opp_hdg180), opponent T2→T3 / sep wide→close, ALL kill-gated; pair with --outcome-reward "
        "(reward-v19 drops the dense pure-pursuit terms → sparse damage + conversion-dwell + energy) and the "
        "--bfm-override that flies whole episodes with the doctrine BFMCoach as the exploration prior; warm "
        "champion + --rehearse-workers 1 anti-forgetting anchor)",
    )
    p.add_argument(
        "--stage-backstop-steps",
        type=int,
        default=600_000,
        help="per-stage collected-step backstop that forces advancement if the gate is slow",
    )
    p.add_argument(
        "--damage-timeout-penalty",
        type=float,
        default=0.0,
        help="clock-timeout terminal penalty on DAMAGE-metric rungs (≤0; 0 = none). The c18 "
        "reversal ladder sets it <0 (e.g. -40) so a no-damage draw is a loss and the policy "
        "cannot park nose-on banking TRACK_BUDGET; keep it milder than |R_DEATH| (60) so a draw "
        "still beats a death (no suicide incentive). No effect on kill/survive rungs.",
    )
    p.add_argument(
        "--convert-bonus",
        type=float,
        default=0.0,
        help="reward-v17 (c60 two-policy): terminal bonus (>0) paid to the merge-converter M the "
        "instant it reaches a genuine rear-quarter firing position behind a LIVE bandit (aspect ≤ "
        "CONVERT_ASPECT, nose-on ata ≤ CONVERT_ATA, range ∈ [CONVERT_RANGE_MIN, MAX]) — at which "
        "point the geometric selector hands the jet to the frozen finisher F. Also SUPPRESSES M's "
        "WEZ fire bonus (M saves ammo for F). Run-level (all merge-convert rungs are convert-metric); "
        "0 (default) = a legacy finisher/chase run. Pair with --ladder merge-convert (e.g. 150).",
    )
    p.add_argument(
        "--me-dalt-ft",
        type=float,
        default=0.0,
        help="curriculum-79 (altitude-advantage 'perch'): spawn the LEARNER this many feet ABOVE the "
        "co-altitude T3 — a slight ENTRY ENERGY edge (potential energy → sustained-pull / dive-for-"
        "closure) so the policy can press into the WEZ against the aggressive corner-holding T3 whose "
        "8.2 g break defeats a co-altitude tracking solution. Run-level but applied ONLY on offensive "
        "tier-3 rungs (tier1/2 scenario builders reject it); the co-heading rehearse anchor stays co-alt "
        "so the foundational finish skill is preserved. 0 (default) = legacy co-altitude start.",
    )
    p.add_argument(
        "--merge-alt-ft",
        type=float,
        default=0.0,
        help="curriculum-91 (altitude BUDGET): the BLOCK altitude both jets spawn at, overriding the "
        "scenario default of 15,000 ft. Measured on curriculum-90: a head-on energy fight bleeds ~68 m/s "
        "of altitude, so 15,000 ft (4,572 m AGL) is only ~63 s of altitude against a 180 s clock — 11/12 "
        "greedy episodes ended hard_deck at a mean 86 s, i.e. the fight is truncated by the FLOOR at less "
        "than half its clock and a multi-pass conversion is unobservable. 30,000 ft = ~129 s of budget, so "
        "the clock and the altitude become commensurate and the vertical BFM the program is trying to teach "
        "(egg turns, yo-yos, displacement rolls) finally has room. Run-level but applied ONLY on offensive "
        "tier-3 rungs, and NOT on the rehearse anchor (same reasoning as --me-dalt-ft). 0 = scenario default. "
        "Prefer setting this as the FLOOR of a band with --merge-alt-ft-max rather than as a fixed jump.",
    )
    p.add_argument(
        "--merge-alt-ft-max",
        type=float,
        default=0.0,
        help="curriculum-91: make the block altitude a per-episode RANDOM band [--merge-alt-ft, this], the "
        "same mechanism as the curriculum-27 aspect band. Use this rather than a fixed --merge-alt-ft jump: "
        "obs feature 2 is (agl − DECK_AGL)/ALT_REF, so 15,000 ft reads 0.81 and 30,000 ft reads 1.58 — past "
        "1.0 into a range no checkpoint has seen, which is exactly how the curriculum-79 perch went OOD and "
        "backfired. A band keeps the familiar altitude in-distribution while extending coverage, and forces "
        "the policy to READ its remaining altitude budget instead of memorising one block. Needs --jitter>0 "
        "(the band is sampled from the reset rng). 0 (default) = no band, use the fixed altitude.",
    )
    p.add_argument(
        "--merge-lateral-m",
        type=float,
        default=0.0,
        help="curriculum-92 (merge LATERAL SEPARATION): half-width, in metres, of a SIGNED per-episode "
        "cross-range band added to the merge spawn — the turning room the head-on merge never had. The only "
        "cross-range term until now was the jitter wobble, ±0.09·jitter·separation = ±135 m at jitter 0.5 / "
        "3000 m = 2.6° off the nose; against a ~634 m F-16 turn radius (225 m/s, the 8.2 g the T3 pulls) that "
        "is 0.21 radii, i.e. a COLLISION COURSE with no one-circle/two-circle decision and no lead turn in it. "
        "That matches the measured failure exactly: curriculum-90 spends 14.2 s/episode inside 500 m at ata "
        "128°, aspect 147°, closure ≈0, with 0.0%% of 4100 ticks hittable OR feasible — while the same "
        "checkpoint kills 11/20 at chase aspect 30, so the skill exists and the geometry never invokes it. "
        "In-sim the T3 changes regime at ~800-1200 m: at ≤500 m it flies a nose-to-nose 7-8.3 g break (HCA "
        "152-165°, a face pass), at ≥800 m it COMMITS A LEAD TURN and HCA collapses to 124/106/96°. Signed and "
        "ADDED to the wobble so the band spans today's familiar ±135 m (the band-not-step pattern that worked "
        "for the aspect band; the fixed --me-dalt-ft/--merge-alt-ft jumps skipped it and both backfired), and "
        "so the policy cannot hard-code 'break right' — every rung in c01→c91 spawned the bandit to the RIGHT. "
        "CEILING: keep |L| under √((THREAT_RANGE·sin THREAT_CONE_RAD)² − perch²) = 1,659 m at a 2,000 ft perch, "
        "or the T3 stops going defensive and the benchmark has silently moved (the header checks this). "
        "Run-level, applied ONLY on offensive tier-3 rungs and NOT on the rehearse anchor (same reasoning as "
        "--me-dalt-ft). Needs --jitter>0 (sampled from the reset rng). 0 (default) = legacy wobble only.",
    )
    p.add_argument(
        "--outcome-reward",
        action="store_true",
        help="reward-v19b (the CURRENT reward every curriculum-82/83/84 result was earned with). KEEPS all "
        "five dense pure-pursuit terms (track/solution/close/aim/aspect) and ADDS, on top of them: the capped "
        "rear-quarter conversion-DWELL, the K_ENERGY specific-energy PBRS + rel-Es term, the anti-zoom-stall "
        "corner-speed floor, and the in-zone CONVERT_CLOSURE_MAX gate. (v19's original pure-outcome form DID "
        "drop the dense terms and decked the champion at ~1.13M; v19b is the hybrid that replaced it — this "
        "flag's old help text described the reverted version.) Effectively MANDATORY for any run warm-started "
        "from c82 or later: omitting it silently deletes the constants those runs earned and the flat result "
        "reads as another non-holdability collapse. Off (default) = pre-c82 dense-only.",
    )
    p.add_argument(
        "--allow-reward-switch",
        action="store_true",
        help="permit a --resume whose checkpoint was trained under the OTHER reward shape (see the "
        "outcome_mode guard). Changing the objective under a converged policy is a real experiment, but it "
        "is indistinguishable from a non-holdability collapse after the fact, so it must be deliberate.",
    )
    p.add_argument(
        "--gate-winrate",
        type=float,
        default=0.0,
        help="override EVERY stage's gate bar (0 = use the per-stage preset thresholds)",
    )
    p.add_argument(
        "--gate-window",
        type=int,
        default=40,
        help="PROBE episodes over which the gate success-rate is measured (large ⇒ demands consistency)",
    )
    p.add_argument(
        "--gate-min-episodes",
        type=int,
        default=60,
        help="min PROBE episodes at a stage before the gate can fire (settle-in floor; ≥ window)",
    )
    p.add_argument(
        "--frontier-slice-deg",
        type=float,
        default=FRONTIER_SLICE_DEG,
        help=(
            "width (deg) of the band's OUTER frontier slice [hi−slice, hi] whose probes alone vote on "
            "mastery (curriculum-36: 10 forces the ceiling itself to net-win; 25=legacy c29 default — "
            "wider let low-aspect kills pad the gate, promoting an unmastered frontier)"
        ),
    )
    # -- resume / warm-start: continue a run from a saved checkpoint (model + both optimizers),
    #    picking up at its recorded stage (or --start-stage). Lets us iterate on the hard T3
    #    stages without re-running the T1/T2 offense curriculum each time.
    p.add_argument(
        "--resume", default="", help="checkpoint path to resume from (model + optimizer state)"
    )
    p.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help=(
            "multiply ALL optimizer learning rates by this factor AFTER a --resume load "
            "(overrides the checkpoint's restored LR; preserves enc-vs-rest ratio). <1 = gentle "
            "consolidation of a warm-started converged policy (damps oscillation off its peak)"
        ),
    )
    p.add_argument(
        "--start-stage",
        type=int,
        default=-1,
        help="stage index to start at (−1 = the checkpoint's recorded stage, or 0 for a fresh run)",
    )
    # -- guided-fire exploration override (idea 2): early in each tier, force the trigger on a
    #    good gun solution with an annealing probability, so the buffer contains fire-in-WEZ
    #    transitions and the learner discovers that shooting on a solution pays.
    p.add_argument(
        "--fire-override-p0",
        type=float,
        default=0.0,
        help="initial prob. of forcing the trigger when in a gun solution (0 = disabled)",
    )
    p.add_argument(
        "--fire-override-steps",
        type=int,
        default=1_000_000,
        help="global control-steps over which the fire-override prob. anneals p0→0, restarted each tier",
    )
    p.add_argument(
        "--fire-override-quality",
        type=float,
        default=0.6,
        help="gun_quality threshold (obs∈[0,1]) above which the override may fire",
    )
    # -- re-attack exploration override (curriculum-43): the ≥55° wall is an under-explored overshoot-
    #    recovery skill (finish-diag on the c41 killer: crushing first pass → overshoot → coast, 0 rounds
    #    fired; reward AND MPC/world-model ruled out as levers). When the agent has overshot a LIVE bandit
    #    it can still shoot (range opening ∧ bandit off the nose ∧ ammo left), splice the PursuitExpert
    #    STICK (a lead-pursuit reversal, already agent-space in info["expert_action"]) with an annealing
    #    prob., seeding the buffer with "chase it down after losing it" demos. Gate-excluded; trigger left
    #    to the policy + fire-override so learned fire discipline is preserved. Disabled at p0/steps ≤ 0.
    p.add_argument(
        "--reattack-override-p0",
        type=float,
        default=0.0,
        help="initial prob. of splicing a PursuitExpert reversal when overshooting a live bandit (0 = disabled)",
    )
    p.add_argument(
        "--reattack-override-steps",
        type=int,
        default=15_000_000,
        help="global control-steps over which the re-attack-override prob. anneals p0→0, restarted each stage",
    )
    p.add_argument(
        "--reattack-override-ata-deg",
        type=float,
        default=50.0,
        help="bandit must be this many degrees off the nose (ata), with range opening, to count as overshot",
    )
    # -- merge-turn exploration override (curriculum-56): imitation-seed the MERGE CIRCLE the champion
    #    lacks. When a LIVE bandit sits off the nose within merge range (ata > threshold ∧ rng < range),
    #    splice a SCRIPTED non-decking break/lead-turn (roll toward the bandit + hold nose-up + anti-deck
    #    AGL floor) with an annealing prob., building the merge-turn motor prior + rear-quarter conversion
    #    transitions the frozen prior can't produce. STICK-ONLY [0:4] (trigger stays with the policy).
    #    Unlike the re-attack/expert overrides it does NOT use the PursuitExpert (which decks at the merge),
    #    so it works on the offensive-posture HEAD-ON rungs. Gate-excluded; off at p0/steps ≤ 0.
    p.add_argument(
        "--merge-override-p0",
        type=float,
        default=0.0,
        help="initial prob. of splicing a scripted non-decking break/lead-turn at the merge (0 = disabled)",
    )
    p.add_argument(
        "--merge-override-steps",
        type=int,
        default=15_000_000,
        help="global control-steps over which the merge-override prob. anneals p0→0, restarted each stage",
    )
    p.add_argument(
        "--merge-override-range-m",
        type=float,
        default=1500.0,
        help="bandit must be within this range (m) for the merge-turn override to fire",
    )
    p.add_argument(
        "--merge-override-ata-deg",
        type=float,
        default=30.0,
        help="bandit must be this many degrees off the nose (ata) to trigger the merge turn "
        "(below this the bandit is near the nose → hand off to the policy for the tracking shot)",
    )
    p.add_argument(
        "--merge-override-mode",
        type=str,
        default="egg",
        choices=("egg", "lowyoyo", "highyoyo", "leadturn"),
        help="which scripted stick the merge-override splices: 'egg' = c58 nose-HIGH vertical egg "
        "(entry reversal for a champion that FREEZES at the merge); 'lowyoyo' = c66 energy-keeping "
        "LOW-YO-YO (hard bank + nose near the horizon + full throttle → hold corner speed, the "
        "anti-zoom-stall conversion for a warm-from-BASELINE M that already flies but bleeds to a stall); "
        "'highyoyo' = c70 anti-overshoot HIGH-YO-YO (bank hard into the turn + nose ABOVE his plane + "
        "ease throttle → bleed excess closure, then roll down onto the LOS) — fired ONLY in the "
        "hot-closure transient (closure > CLOSURE_HI within --merge-override-range-m) to teach the "
        "converting attacker not to fly out in front of the hard-breaking T3; 'leadturn' = c95 "
        "one-circle LEAD TURN (manage the predicted CPA to ~800 m of turning room at corner speed, "
        "then commit a gamma-corrected max-rate turn TOWARD his pass side at 1000 m, then TRACK) — the "
        "only scripted arm MEASURED to convert a neutral head-on merge to a real gun solution, so pair "
        "it with a wide window (--merge-override-range-m 3000 --merge-override-ata-deg 25: 2000 m was "
        "measured to hand back MID-conversion, and a deeper 8 deg ata gate measured WORSE than 25 - "
        "gq_honest 0.58 -> 0.00 - because the policy's own terminal beats the scripted tracker's) and "
        "let the ata gate hand the finish back to the policy at the conversion. Its coin is drawn "
        "ONCE PER EPISODE (the others draw per step) because it is a sustained latched maneuver",
    )
    # -- expert-rollout exploration override (long-range fix): early in each stage, fly whole
    #    episodes with the scripted PursuitExpert at an annealing probability, so the buffer
    #    gets complete close→kill demos the stalled prior can't produce past gun range (where
    #    the fire-override is impotent). Expert episodes are excluded from the honest gate.
    p.add_argument(
        "--expert-p0",
        type=float,
        default=0.0,
        help="initial prob. of flying a whole episode with the scripted PursuitExpert (0 = disabled)",
    )
    p.add_argument(
        "--expert-steps",
        type=int,
        default=6_000_000,
        help="global control-steps over which the expert-rollout prob. anneals p0→0, restarted each stage",
    )
    # -- BFM-coach exploration override (curriculum-62, the BFM pivot): early in each stage, fly whole
    #    HEAD-ON offensive-t3 episodes with the doctrine BFMCoach (bfm.py) at an annealing probability. The
    #    coach SURVIVES the merge and does textbook BFM (merge→egg/reversal→rear-quarter track+fire) where
    #    the PursuitExpert decks, seeding the buffer with SAFE, BFM-literate merge→convert→finish demos the
    #    deck-prone pure-pursuit policy can't generate — the exploration prior for the reward-v19 outcome
    #    reward (--outcome-reward). Full len-5 action incl. its own fire; gate-excluded; off at p0/steps ≤ 0.
    p.add_argument(
        "--bfm-override-p0",
        type=float,
        default=0.0,
        help="initial prob. of flying a whole head-on episode with the doctrine BFMCoach (0 = disabled)",
    )
    p.add_argument(
        "--bfm-override-steps",
        type=int,
        default=15_000_000,
        help="global control-steps over which the BFM-coach override prob. anneals p0→0, restarted each stage",
    )
    # -- honest deterministic-probe gate (curriculum-14): every Nth episode is flown by the
    #    greedy deployable policy (no expert/fire-override) and is the ONLY episode the stage
    #    gate counts, so advancement tracks the policy we actually deploy — not the stochastic
    #    exploration prior, which the curriculum-13 eval proved advanced stages on lucky-noise
    #    kills the greedy policy could not reproduce. 0 = legacy (gate on any non-override ep).
    p.add_argument(
        "--probe-every",
        type=int,
        default=3,
        help="every Nth episode is a deterministic gate probe (0 = legacy non-override gate)",
    )
    # -- self-play (curriculum-21): replace the collector's scripted opponent with a frozen copy of
    #    the learner, sampled per-episode from a growing pool the learner snapshots into. The escape
    #    from the merge-conversion wall — the neutral-merge skill has no scripted teacher, so the
    #    opponent must be a maneuvering, improving copy of the learner itself. --ladder selfplay
    #    auto-enables --selfplay; warm-start + seed the pool with --resume <champion>.
    p.add_argument(
        "--selfplay",
        action="store_true",
        help="fight POOLED copies of the learner instead of the scripted opponent (implied by "
        "--ladder selfplay)",
    )
    p.add_argument(
        "--pool-dir",
        default="",
        help="dir of frozen learner priors to sample self-play opponents from "
        "(default: <save-dir>/pool)",
    )
    p.add_argument(
        "--selfplay-snapshot-every",
        type=int,
        default=25_000,
        help="collected steps between the learner exporting a fresh prior into the pool (0 = "
        "seed-only pool, never grows)",
    )
    p.add_argument(
        "--selfplay-recency-bias",
        type=float,
        default=0.0,
        help="opponent-pool sampling bias toward recent (stronger) snapshots; 0 = uniform "
        "(fictitious self-play — the stable default)",
    )
    p.add_argument(
        "--selfplay-max-pool",
        type=int,
        default=50,
        help="cap on pooled snapshots (oldest pruned beyond this); keeps the per-episode glob bounded",
    )
    p.add_argument(
        "--selfplay-scripted-prob",
        type=float,
        default=0.0,
        help="curriculum-24 ANCHORING: fraction of self-play episodes (and EVERY deterministic probe) "
        "flown vs the SCRIPTED tier-3 bot instead of a pooled self-copy, so the gate measures the "
        "real target and the offense stays anchored (pure self-play regressed off it). 0 = pure "
        "fictitious self-play (the old c21-c23 behaviour); 0.5 = half scripted / half self-copies",
    )
    # -- per-episode start-geometry jitter (curriculum-14): perturbs the opening range, LOS
    #    bearing, opponent heading, altitude and airspeeds every episode so a deterministic
    #    policy must generalise across the envelope instead of memorising one fixed start
    #    (the curriculum-13 overfit: killed from its training separation, chip-drew when closer).
    p.add_argument(
        "--jitter",
        type=float,
        default=0.5,
        help="start-geometry jitter strength ∈ [0,1] (0 = fixed legacy geometry)",
    )
    return p.parse_args(argv)


def _auto_workers() -> int:
    import os

    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu - 2))  # leave headroom for the learner + GPU host thread


def main(argv=None) -> None:
    args = parse_args(argv)
    _paths.require_astrum_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    workers = args.workers or _auto_workers()
    queue_max = args.queue_max or max(8, 4 * workers)
    if device.type == "cpu":
        # A CPU learner shares cores with the K collectors; leave them room and stop torch
        # from oversubscribing (which is dispatch-bound thrash on small ops). No effect on
        # GPU runs, where torch's CPU threads only handle light host-side work.
        import os

        torch.set_num_threads(max(1, (os.cpu_count() or 4) - workers - 1))
    from astrum_dogfight import constants as C

    # build_config is shared with the single-env trainer, which derives cfg.mpc from a
    # `--no-mpc` flag; the parallel trainer exposes the same knob as --collector-mpc
    # (prior-only collection is the default — it keeps the GPU free for the learner; see
    # the module docstring). Bridge the two names so build_config can read `no_mpc`.
    args.no_mpc = not args.collector_mpc

    # --resume: rebuild cfg from the checkpoint so the ARCHITECTURE matches exactly, then
    # re-apply the CLI knobs that change no tensor shape.
    #
    # Until 2026-09-04 only device/compile/mpc were re-taken, so on every warm start
    # `--horizon`, `--batch-size`, `--discount` and the `--max-time-s`-derived
    # `episode_length` were silently DISCARDED — c85 ran 180 s episodes with cfg.episode_length
    # still logged and used as 1440 (60 s), and there was no `--discount` flag at all. Two of
    # the highest-value experiments (a longer credit horizon, a longer planning horizon) were
    # therefore untestable on any warm run, which is every run past curriculum-01.
    #
    # `_RESUME_OVERRIDABLE` is an explicit allow-list, not a blanket re-derive: `latent_dim`,
    # `num_q`, `obs_dim` and `action_dim` are shape-BEARING and must keep the checkpoint's
    # values (widening obs_dim needs the zero-init encoder-column surgery, not a flag), and
    # `action_repeat` is excluded because `ActionRepeat` SUMS reward — changing it on a warm
    # head doubles the dense reward scale mid-run.
    resume_sd = None
    if args.resume:
        resume_sd = torch.load(args.resume, map_location="cpu", weights_only=False)
        # An obs APPEND is the one shape change a warm start can absorb losslessly, so it is
        # handled here rather than being rejected as a shape mismatch. Automatic and loud, not
        # flag-gated: `constants.OBS_DIM` growing is a fact about the env the run is about to
        # collect in, and there is no correct way to resume into it WITHOUT widening — a flag
        # would only add a way to forget. Shrinking is still refused: dropping trained columns is
        # a real capability loss and needs a deliberate decision, not an inferred one.
        want_obs = C.OBS_DIM * args.frame_stack
        ckpt_obs = int(resume_sd["cfg"]["obs_dim"])
        if ckpt_obs != want_obs:
            per_frame = ckpt_obs // args.frame_stack
            if ckpt_obs > want_obs or ckpt_obs % args.frame_stack or per_frame > C.OBS_DIM:
                raise SystemExit(
                    f"[resume] obs mismatch: checkpoint {ckpt_obs} vs env {want_obs} "
                    f"(OBS_DIM {C.OBS_DIM} × frame_stack {args.frame_stack}). Only an APPEND "
                    f"(checkpoint narrower, same frame_stack) can be resumed."
                )
            _widen_resume_obs(resume_sd, per_frame, C.OBS_DIM, args.frame_stack)
        ckpt_cfg = TDMPC2Config(**resume_sd["cfg"])
        cli_cfg = build_config(args, ckpt_cfg.obs_dim, ckpt_cfg.action_dim)
        passed = _flags_passed(argv)
        overrides = {
            field: getattr(cli_cfg, field)
            for field, flags in _RESUME_OVERRIDABLE.items()
            if passed & flags and getattr(cli_cfg, field) != getattr(ckpt_cfg, field)
        }
        cfg = dataclasses.replace(
            ckpt_cfg,
            device=args.device,
            compile=args.compile,
            mpc=not args.no_mpc,
            steps=args.steps,  # bookkeeping only; the real budget is run_parallel(steps=…)
            **overrides,
        )
        default_stage = int(resume_sd.get("stage_idx", 0))
        if overrides:
            applied = ", ".join(
                f"{k}: {getattr(ckpt_cfg, k)} -> {v}" for k, v in sorted(overrides.items())
            )
            print(f"[resume] cfg overridden from CLI — {applied}", flush=True)
        ignored = sorted(
            flag
            for field, flags in _RESUME_OVERRIDABLE.items()
            for flag in flags
            if flag in passed and field not in overrides
        )
        if ignored:
            print(f"[resume] no-op (already matches the checkpoint): {', '.join(ignored)}", flush=True)
    else:
        cfg = build_config(args, C.OBS_DIM * args.frame_stack, C.ACTION_DIM)
        overrides = {}
        default_stage = 0
    start_stage = args.start_stage if args.start_stage >= 0 else default_stage

    ladder_rows = {
        "reversal": _REVERSAL,
        "merge": _MERGE,
        "beam-t3": _BEAM_T3,
        "beam-t3-fine": _BEAM_T3_FINE,
        "beam-t3-net": _BEAM_T3_NET,
        "beam-t3-push": _BEAM_T3_NET_PUSH,
        "beam-t3-frontier": _BEAM_T3_FRONTIER,
        "beam-t3-killgate": _BEAM_T3_KILLGATE,
        "chase-t3-killgate": _CHASE_T3_KILLGATE,
        "chase-t3-neutral": _CHASE_T3_NEUTRAL,
        "reattack-wounded": _REATTACK_WOUNDED,
        "selfplay": _SELFPLAY,
        "selfplay-ladder": _SELFPLAY_LADDER,
        "selfplay-beam": _SELFPLAY_BEAM,
        "selfplay-nose": _SELFPLAY_NOSE,
        "merge-t123": _MERGE_T123,
        "merge-chase": _MERGE_CHASE,
        "merge-survive": _MERGE_SURVIVE,
        "merge-convert": _MERGE_CONVERT,
        "bfm-headon": _BFM_HEADON,
        "energy-merge": _ENERGY_MERGE,
    }.get(args.ladder, _CHASE_THEN_DEFEND)
    # --ladder selfplay[-ladder|-beam] implies self-play (fight pooled copies of the learner, not the scripted bot).
    selfplay = args.selfplay or args.ladder in (
        "selfplay",
        "selfplay-ladder",
        "selfplay-beam",
        "selfplay-nose",
    )
    pool_dir = args.pool_dir
    if selfplay and not pool_dir:
        if not args.save:
            raise SystemExit("self-play needs a pool location: pass --pool-dir or --save")
        pool_dir = str(Path(args.save).parent / "pool")
    stages = build_stages(
        rows=ladder_rows,
        backstop_steps=args.stage_backstop_steps,
        start_stage=start_stage,
        sep_override=args.separation_m,
        winrate_override=args.gate_winrate,
        kill_timeout_penalty=C.R_TIMEOUT,  # reward-v7: penalize a no-kill draw on kill rungs
        damage_timeout_penalty=args.damage_timeout_penalty,  # c18: penalize a no-damage draw
    )
    if not (0 <= start_stage < len(stages)):
        raise SystemExit(f"--start-stage {start_stage} out of range 0..{len(stages) - 1}")

    # curriculum-57: the fixed co-heading FINISH stage flown by the dedicated rehearsal workers
    # (--rehearse-workers). It reproduces the champion's proven chase-t3-killgate rung-0 home turf —
    # native T3 shooter, aspect band [0,55]° (the mastered ≤55° tracking kill), separation 700 m,
    # KILL-gated so the timeout penalty keeps it aggressive — so the buffer never loses the finish
    # while the merge workers push the head-on program. Built here (C.R_TIMEOUT is the kill-rung
    # timeout penalty, matching build_stages). None when no rehearsal workers are requested.
    rehearse_stage = None
    if args.rehearse_workers > 0:
        rehearse_stage = Stage(
            tier=3,
            separation_m=700.0,
            metric="kill",
            winrate=0.0,  # never gates (rehearsal workers don't probe/advance)
            backstop=0,
            timeout_penalty=C.R_TIMEOUT,
            posture="offensive",
            aspect_deg=0.0,
            aspect_deg_max=55.0,
            opponent_tier=0,
            opp_heading_deg=None,
            # Pin the anchor's own 60 s clock: the ≤55° finish it exists to preserve was learned
            # at 60 s episodes (curriculum-82/83/84), and inheriting a long --max-time-s from the
            # merge program silently changed the very distribution the anchor is meant to hold.
            max_time_s=60.0,
        )

    env_kwargs = dict(
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        max_time_s=args.max_time_s,
        rate_hz=args.rate_hz,
    )
    # Separation is per-stage (folded into `stages`); only airspeed is a global override here.
    if args.airspeed_kts > 0.0:
        env_kwargs["airspeed_kts"] = args.airspeed_kts
    # Per-episode start-geometry jitter (the collector supplies a distinct jitter_seed per worker).
    if args.jitter > 0.0:
        env_kwargs["jitter"] = args.jitter
    # reward-v17 (c60 two-policy): the merge-converter M's terminal convert bonus. Run-level (every
    # merge-convert rung is convert-metric), so it rides env_kwargs into every worker's make_tdmpc_env
    # → the factory's convert_bonus param → reward.compute. Only set when >0 so a legacy run is untouched.
    if args.convert_bonus > 0.0:
        env_kwargs["convert_bonus"] = args.convert_bonus
    # reward-v19 (c62 the BFM pivot): the OUTCOME reward. Run-level like convert_bonus — rides
    # env_kwargs into every worker's make_tdmpc_env → the factory's outcome_mode param →
    # reward.compute. It does NOT drop the dense pursuit terms (an older comment and the flag's
    # help text both claimed it did; `reward.py` keeps full dense pursuit either way). What it
    # ADDS is the convert-dwell term, K_ENERGY, the rel-Es PBRS, the corner-speed floor, and the
    # in-zone `CONVERT_CLOSURE_MAX` gate — i.e. the constants curriculum-83 and curriculum-84
    # earned. Only set when enabled so legacy runs are byte-for-byte untouched.
    if args.outcome_reward:
        env_kwargs["outcome_mode"] = True
    # …and because it is NOT in cfg, `--resume` cannot restore it. Warm-starting a checkpoint
    # trained with the outcome reward into a dense-only run (or vice versa) changes the objective
    # under a converged policy, which presents exactly like the non-holdability collapses this
    # program has spent ~17 runs on. Refuse to do it silently.
    if resume_sd is not None:
        was = resume_sd.get("provenance", {}).get("outcome_mode")
        now = bool(env_kwargs.get("outcome_mode", False))
        if was is None:
            print(
                f"[resume] WARNING: {args.resume} predates provenance recording — it does not say "
                f"which reward trained it. This run uses {'OUTCOME' if now else 'dense-only'}; "
                f"check the origin run's launch line before trusting the comparison.",
                flush=True,
            )
        elif bool(was) != now:
            msg = (
                f"reward-shape mismatch: {args.resume} was trained with "
                f"outcome_mode={bool(was)} but this run would use outcome_mode={now}. "
                f"{'Add' if was else 'Drop'} --outcome-reward, or pass --allow-reward-switch "
                f"if changing the objective is the experiment."
            )
            if not args.allow_reward_switch:
                raise SystemExit(msg)
            print(f"[resume] --allow-reward-switch: {msg}", flush=True)
    # curriculum-79 (altitude-advantage perch): carried as a sentinel in env_kwargs; _make_env reads
    # it and injects into the offensive tier-3 `extra` only, filtering it out of the generic splat so
    # tier1/2 factories never see the unknown kwarg. 0 (default) = legacy co-altitude, untouched.
    if args.me_dalt_ft != 0.0:
        env_kwargs["me_dalt_ft"] = args.me_dalt_ft
    # curriculum-91 (block altitude = the fight's altitude BUDGET): same sentinel mechanism.
    if args.merge_alt_ft != 0.0:
        env_kwargs["merge_alt_ft"] = args.merge_alt_ft
    if args.merge_alt_ft_max != 0.0:
        if args.merge_alt_ft == 0.0:
            raise SystemExit(
                "--merge-alt-ft-max is the CEILING of the block-altitude band and needs a floor: "
                "pass --merge-alt-ft too (e.g. --merge-alt-ft 15000 --merge-alt-ft-max 30000)."
            )
        if args.jitter <= 0.0:
            raise SystemExit(
                "--merge-alt-ft-max needs --jitter > 0: the band is sampled from the per-episode "
                "reset rng, which only exists under jitter, so without it the ceiling is silently ignored."
            )
        env_kwargs["merge_alt_ft_max"] = args.merge_alt_ft_max
    # curriculum-92 (merge lateral separation = the merge's turning room): same sentinel mechanism.
    if args.merge_lateral_m != 0.0:
        if args.jitter <= 0.0:
            raise SystemExit(
                "--merge-lateral-m needs --jitter > 0: the signed band is sampled from the "
                "per-episode reset rng, which only exists under jitter, so without it the offset "
                "is silently ignored (and `lateral_m` is forced to 0.0)."
            )
        # Refuse to spawn past the opponent's threat cone. Beyond it the T3 never goes defensive and
        # its min range grows past gun range: a materially EASIER opponent, still labelled T3, so the
        # run would be uninterpretable rather than merely negative. Derived, not hardcoded, so that
        # changing the perch or either threat constant re-derives the bound instead of silently
        # invalidating it.
        from math import sin, sqrt

        from astrum_dogfight import constants as C

        _cone = C.THREAT_RANGE * sin(C.THREAT_CONE_RAD)
        _dz = args.me_dalt_ft * 0.3048
        _cliff = sqrt(_cone**2 - _dz**2) if _cone > _dz else 0.0
        _total = abs(args.merge_lateral_m) + args.jitter * 0.09 * 3000.0
        if _total >= _cliff:
            raise SystemExit(
                f"--merge-lateral-m {args.merge_lateral_m:,.0f} (+{args.jitter * 0.09 * 3000.0:,.0f} m "
                f"jitter wobble = {_total:,.0f} m) reaches the T3's threat cone at {_cliff:,.0f} m "
                f"(THREAT_RANGE {C.THREAT_RANGE:,.0f} m × sin {C.THREAT_CONE_RAD * 57.2958:.0f}°, minus "
                f"the {_dz:,.0f} m perch). Past it the bot never breaks and the benchmark has moved — "
                f"the result would be uninterpretable, not just negative. Lower the offset or the perch."
            )
        env_kwargs["merge_lateral_m"] = args.merge_lateral_m

    run_parallel(
        cfg,
        workers=workers,
        steps=args.steps,
        seed_steps=args.seed_steps or None,
        stages=stages,
        env_kwargs=env_kwargs,
        weight_sync_every=args.weight_sync_every,
        weight_check_every=args.weight_check_every,
        utd=args.utd,
        queue_max=queue_max,
        save=args.save,
        save_every=args.save_every,
        log_every=args.log_every,
        device=device,
        start_stage=start_stage,
        resume_sd=resume_sd,
        lr_scale=args.lr_scale,
        gate_window=args.gate_window,
        gate_min_episodes=args.gate_min_episodes,
        frontier_slice_deg=args.frontier_slice_deg,
        fire_override_p0=args.fire_override_p0,
        fire_override_steps=args.fire_override_steps,
        fire_override_quality=args.fire_override_quality,
        reattack_override_p0=args.reattack_override_p0,
        reattack_override_steps=args.reattack_override_steps,
        reattack_override_ata_deg=args.reattack_override_ata_deg,
        merge_override_p0=args.merge_override_p0,
        merge_override_steps=args.merge_override_steps,
        merge_override_range_m=args.merge_override_range_m,
        merge_override_ata_deg=args.merge_override_ata_deg,
        merge_override_mode=args.merge_override_mode,
        bfm_override_p0=args.bfm_override_p0,
        bfm_override_steps=args.bfm_override_steps,
        expert_p0=args.expert_p0,
        expert_steps=args.expert_steps,
        probe_every=args.probe_every,
        selfplay=selfplay,
        pool_dir=pool_dir,
        selfplay_snapshot_every=args.selfplay_snapshot_every,
        selfplay_recency_bias=args.selfplay_recency_bias,
        selfplay_max_pool=args.selfplay_max_pool,
        selfplay_scripted_prob=args.selfplay_scripted_prob,
        rehearse_workers=args.rehearse_workers,
        rehearse_stage=rehearse_stage,
        discount_ramp_steps=args.discount_ramp_steps,
        # Ramp FROM whatever the checkpoint was actually trained at (its own resolved γ), not from
        # the CLI target — that is the value the restored critic is calibrated for. Cold runs have
        # nothing to ramp from, so they leave it 0 (= no ramp).
        discount_start=(ckpt_cfg.resolved_discount() if resume_sd is not None else 0.0),
    )


if __name__ == "__main__":
    main()
