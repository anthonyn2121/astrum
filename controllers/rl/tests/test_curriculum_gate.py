"""Distance-curriculum stage ladder + performance gate (train_parallel).

The curriculum is a ladder of (tier, separation, gate-metric) *stages*, advanced only on
SUSTAINED mastery of each stage's metric — a *kill* on the offensive/converting rungs, mere
*survival* on the point-blank defensive rungs — with a per-stage step backstop as the safety
net for a slow gate. These exercise the pure decision helpers with no learner/GPU/env:

* the ladder builder (cumulative, resume-rebased backstops; per-stage vs override thresholds),
* the metric predicate (`kill` vs `survive`),
* the mastery gate itself, including that a *lucky trailing streak is not enough* — a large
  window at a high bar is what makes advancement demand consistency (the user's requirement),
* and that advance is backstop-first, per-stage-metric-gated, monotonic, and one rung at a time.
"""

from __future__ import annotations

import numpy as np
import pytest
from astrum_rl.tdmpc2.train_parallel import (
    _BEAM_T3,
    _BEAM_T3_FINE,
    _BEAM_T3_FRONTIER,
    _BEAM_T3_NET,
    _BEAM_T3_NET_PUSH,
    _CHASE_T3_NEUTRAL,
    _HY_BANK_UP_RAD,
    _HY_CLOSURE_LO,
    _HY_CLOSURE_UP,
    _HY_PITCH_UP_RAD,
    _LT_BANK_MAX_RAD,
    _LT_BANK_MIN_RAD,
    _LT_COMMIT_RNG_M,
    _LT_TRACK_ASPECT_RAD,
    _LT_TRACK_ATA_RAD,
    _LT_TRACK_EXIT_ASPECT_RAD,
    _LT_TURN_BANK_RAD,
    _LT_TURNROOM_M,
    _LT_V_CORNER_TGT,
    _LT_V_EASE_FLOOR,
    _LT_V_SUSTAIN_MPS,
    _MERGE,
    _MERGE_SURVIVE,
    _SELFPLAY,
    _SELFPLAY_BEAM,
    _SELFPLAY_LADDER,
    _VE_BANK_RAD,
    _VE_PITCH_TGT_RAD,
    FRONTIER_SLICE_DEG,
    Stage,
    _advance_stage,
    _cpa_miss_m,
    _gate_should_advance,
    _highyoyo_stick,
    _in_frontier,
    _is_win,
    _leadturn_phase,
    _leadturn_stick,
    _next_stage,
    _stage_win,
    _vertical_egg_stick,
    build_stages,
    stage_at,
)

# A compact 3-rung ladder: an offensive kill rung, then two defensive *survive* rungs (so the
# survive gate has somewhere to advance TO, unlike the real ladder's top rung). backstop 300k.
ROWS = (
    (1, 400, "kill", 0.70),
    (3, 700, "survive", 0.85),
    (3, 450, "survive", 0.85),
)
BACKSTOP = 300_000
STAGES = build_stages(ROWS, backstop_steps=BACKSTOP)
GATE = dict(gate_window=40, gate_min_episodes=60)


# -- build_stages ----------------------------------------------------------
def test_build_stages_maps_rows_and_cumulative_backstops():
    assert [s.tier for s in STAGES] == [1, 3, 3]
    assert [s.separation_m for s in STAGES] == [400.0, 700.0, 450.0]
    assert [s.metric for s in STAGES] == ["kill", "survive", "survive"]
    assert [s.winrate for s in STAGES] == [0.70, 0.85, 0.85]
    # Backstop i = i · backstop_steps (relative to start_stage 0).
    assert [s.backstop for s in STAGES] == [0, BACKSTOP, 2 * BACKSTOP]
    assert all(isinstance(s, Stage) for s in STAGES)


def test_build_stages_backstops_rebase_to_start_stage():
    # Resuming at stage 1 must budget only the REMAINING rungs from collected=0: stage 1
    # becomes the new zero-floor, stage 2 fires one backstop later. Prior rungs pin to 0.
    s = build_stages(ROWS, backstop_steps=BACKSTOP, start_stage=1)
    assert [x.backstop for x in s] == [0, 0, BACKSTOP]


def test_build_stages_overrides():
    s = build_stages(ROWS, backstop_steps=BACKSTOP, sep_override=1234.0, winrate_override=0.5)
    assert all(x.separation_m == 1234.0 for x in s)
    assert all(x.winrate == 0.5 for x in s)
    # 0 overrides are inert — the per-stage presets stand.
    s0 = build_stages(ROWS, backstop_steps=BACKSTOP, sep_override=0.0, winrate_override=0.0)
    assert [x.separation_m for x in s0] == [400.0, 700.0, 450.0]
    assert [x.winrate for x in s0] == [0.70, 0.85, 0.85]


# -- _is_win (metric predicate) --------------------------------------------
def test_is_win_kill_metric_counts_only_kills():
    assert _is_win("kill", "kill")
    for loss in ("killed", "death", "trade", "hard_deck", "left_bubble"):
        assert not _is_win(loss, "kill")
    # A timeout draw or a missing tag is NOT a kill.
    assert not _is_win("timeout", "kill")
    assert not _is_win(None, "kill")


def test_is_win_survive_metric_counts_any_non_loss():
    # Survival = defeating the shot: anything that is not a recognized loss counts, including
    # a timeout draw, an unrecognized/absent tag, AND an outright kill.
    assert _is_win("kill", "survive")
    assert _is_win("timeout", "survive")
    assert _is_win(None, "survive")
    for loss in ("killed", "death", "trade", "hard_deck", "left_bubble"):
        assert not _is_win(loss, "survive")


# -- _stage_win (metric dispatch, incl. the numeric DAMAGE gate) -----------
def test_stage_win_defers_to_is_win_for_event_metrics():
    # kill/survive rungs ignore dmg_dealt and grade on the event tag (via _is_win).
    kill_stage = Stage(tier=2, separation_m=400.0, metric="kill", winrate=0.5, backstop=0)
    surv_stage = Stage(tier=3, separation_m=400.0, metric="survive", winrate=0.5, backstop=0)
    assert _stage_win({"event": "kill", "dmg_dealt": 0.0}, kill_stage)
    assert not _stage_win({"event": "timeout", "dmg_dealt": 99.0}, kill_stage)  # HP irrelevant
    assert _stage_win({"event": "timeout", "dmg_dealt": 0.0}, surv_stage)
    assert not _stage_win({"event": "death", "dmg_dealt": 99.0}, surv_stage)


def test_stage_win_damage_metric_thresholds_on_hp_dealt():
    dmg_stage = Stage(
        tier=2, separation_m=400.0, metric="damage", winrate=0.5, backstop=0, damage_threshold=60.0
    )
    assert _stage_win({"dmg_dealt": 60.0}, dmg_stage)  # exactly the bar counts
    assert _stage_win({"dmg_dealt": 100.0}, dmg_stage)  # a kill (full HP) trivially clears it
    assert not _stage_win({"dmg_dealt": 59.9}, dmg_stage)
    # A missing / unreadable HP reading (None or NaN) must NOT be credited as a win.
    assert not _stage_win({"dmg_dealt": None}, dmg_stage)
    assert not _stage_win({"dmg_dealt": float("nan")}, dmg_stage)
    assert not _stage_win({}, dmg_stage)


def test_stage_win_net_metric_grades_on_outcome_not_chip():
    # curriculum-27: the honest gate. A kill always wins; any losing terminal (death/deck/trade/
    # bubble) is a loss NO MATTER how much chip was dealt (the c26 blind spot); a clock draw wins
    # iff net HP (dealt − taken) clears the bar. This is exactly what the damage gate could not see.
    net_stage = Stage(
        tier=3, separation_m=700.0, metric="net", winrate=0.6, backstop=0, net_threshold=5.0
    )
    # a kill wins outright regardless of HP bookkeeping.
    assert _stage_win({"event": "kill", "dmg_dealt": 12.0, "dmg_taken": 40.0}, net_stage)
    # chipping-while-shredded (the c26 policy): dealt 30 but taken 45 and gunned → a LOSS.
    assert not _stage_win({"event": "killed", "dmg_dealt": 30.0, "dmg_taken": 45.0}, net_stage)
    assert not _stage_win({"event": "hard_deck", "dmg_dealt": 30.0, "dmg_taken": 0.0}, net_stage)
    # clock draws grade on net HP against the threshold.
    assert _stage_win({"event": "timeout", "dmg_dealt": 20.0, "dmg_taken": 12.0}, net_stage)  # +8
    assert _stage_win({"event": None, "dmg_dealt": 5.0, "dmg_taken": 0.0}, net_stage)  # exactly +5
    assert not _stage_win(
        {"event": "timeout", "dmg_dealt": 12.0, "dmg_taken": 10.0}, net_stage
    )  # +2
    assert not _stage_win(
        {"event": "timeout", "dmg_dealt": 40.0, "dmg_taken": 40.0}, net_stage
    )  # 0
    # missing bookkeeping is not a win.
    assert not _stage_win({"event": "timeout", "dmg_dealt": None, "dmg_taken": 0.0}, net_stage)
    assert not _stage_win({"event": "timeout"}, net_stage)


def test_build_stages_parses_damage_threshold_and_suppresses_timeout_penalty():
    rows = (
        (1, 400, "kill", 0.60),
        (2, 400, "damage", 0.50, 60.0),
        (3, 400, "survive", 0.55),
    )
    s = build_stages(rows, backstop_steps=BACKSTOP, kill_timeout_penalty=-5.0)
    assert [x.metric for x in s] == ["kill", "damage", "survive"]
    assert [x.damage_threshold for x in s] == [0.0, 60.0, 0.0]  # only the 5-field row carries a bar
    # reward-v7 timeout penalty applies to the KILL rung only; damage & survive suppress it.
    assert s[0].timeout_penalty == -5.0
    assert s[1].timeout_penalty == 0.0
    assert s[2].timeout_penalty == 0.0


def test_build_stages_parses_dict_rows_with_opponent_tier():
    # curriculum-20: dict rows carry an explicit opponent_tier (decoupled from `tier`) plus the
    # damage-timeout penalty on damage rungs. Legacy positional rows default opponent_tier to 0.
    rows = (
        dict(
            tier=3,
            opponent_tier=1,
            posture="offensive",
            aspect_deg=90.0,
            separation_m=900.0,
            metric="kill",
            winrate=0.5,
        ),
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
    )
    s = build_stages(
        rows, backstop_steps=BACKSTOP, kill_timeout_penalty=-80.0, damage_timeout_penalty=-40.0
    )
    assert [x.tier for x in s] == [3, 3]
    assert [x.opponent_tier for x in s] == [1, 3]
    assert [x.aspect_deg for x in s] == [90.0, 45.0]
    assert [x.separation_m for x in s] == [900.0, 900.0]
    assert [x.metric for x in s] == ["kill", "damage"]
    assert [x.damage_threshold for x in s] == [0.0, 20.0]
    # kill rung → kill_timeout_penalty; damage rung → damage_timeout_penalty.
    assert s[0].timeout_penalty == -80.0
    assert s[1].timeout_penalty == -40.0
    # cumulative backstops apply to dict rows just like tuple rows.
    assert [x.backstop for x in s] == [0, BACKSTOP]
    # legacy positional rows leave opponent_tier at its 0 default.
    assert build_stages(ROWS, backstop_steps=BACKSTOP)[0].opponent_tier == 0


def test_build_stages_parses_opp_heading_deg():
    # curriculum-21: a dict row may pin the bot's ABSOLUTE start heading (head-on merge = aspect 0
    # + hdg 180). Absent / None → co-heading default; a value round-trips as a float.
    rows = (
        dict(
            tier=3,
            posture="offensive",
            aspect_deg=0.0,
            opp_heading_deg=180.0,
            separation_m=900.0,
            metric="damage",
            winrate=0.3,
            damage_threshold=12.0,
        ),
        dict(
            tier=3,
            posture="offensive",
            aspect_deg=90.0,
            separation_m=900.0,
            metric="damage",
            winrate=0.3,
            damage_threshold=12.0,
        ),  # no opp_heading_deg key
    )
    s = build_stages(rows, backstop_steps=BACKSTOP)
    assert s[0].opp_heading_deg == 180.0
    assert s[1].opp_heading_deg is None  # absent key → co-heading
    # explicit None also stays None (the _SELFPLAY neutral-merge form).
    none_row = (
        dict(
            tier=3,
            posture="offensive",
            aspect_deg=90.0,
            opp_heading_deg=None,
            separation_m=900.0,
            metric="damage",
            winrate=0.3,
            damage_threshold=12.0,
        ),
    )
    assert build_stages(none_row, backstop_steps=BACKSTOP)[0].opp_heading_deg is None
    # tuple rows never carry it → None.
    assert build_stages(ROWS, backstop_steps=BACKSTOP)[0].opp_heading_deg is None


def test_selfplay_ladder_is_a_single_neutral_merge_stage():
    # curriculum-21: the self-play ladder is ONE neutral (beam, co-heading) tier-3 rung, damage-
    # gated with the -40 anti-passivity timeout penalty. Single stage ⇒ the gate is a no-op (the
    # pool grows on the snapshot clock, not the gate) — assert there is exactly one and nowhere to
    # advance to, and that the timeout penalty threads through on the damage metric.
    s = build_stages(_SELFPLAY, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 1
    st = s[0]
    assert st.tier == 3 and st.posture == "offensive" and st.aspect_deg == 90.0
    assert st.opp_heading_deg is None  # co-heading line-abreast merge
    assert st.metric == "damage" and st.timeout_penalty == -40.0
    assert _next_stage(s, 0) is None  # top rung — gate has nowhere to promote to
    # the top-stage gate is inert even on a crushing win log (no regression, no advance).
    assert _advance_stage(s, 0, 10 * BACKSTOP, [1.0] * 200, **GATE) == (0, "")


def test_merge_ladder_rows_leave_opp_heading_deg_none():
    # The c20 merge ladder predates the heading knob; every rung stays co-heading.
    s = build_stages(_MERGE, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert all(x.opp_heading_deg is None for x in s)


def test_selfplay_angle_ladder_hardens_aspect_and_is_damage_gated():
    # curriculum-23: the self-play angle ladder is a 7-rung sweep that HARDENS the merge angle in fine
    # 15° steps, from a tail-chase the warm-start already converts (aspect 0) out to the neutral beam
    # merge (aspect 90). c22 used 30° steps and stalled at aspect 30 (the jump from "shoot the six" to
    # "convert a 30° offset" was the whole BFM skill in one rung); 15° lets it grow incrementally. Every
    # rung is a tier-3 damage-gated self-play rung (co-heading) carrying the -40 anti-passivity timeout
    # penalty; the aspect is monotonically non-decreasing and ends at the beam-merge acceptance.
    s = build_stages(_SELFPLAY_LADDER, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 7
    assert [x.aspect_deg for x in s] == [
        0.0,
        15.0,
        30.0,
        45.0,
        60.0,
        75.0,
        90.0,
    ]  # 15° hardening ladder
    assert all(x.tier == 3 and x.posture == "offensive" for x in s)
    assert all(
        x.opp_heading_deg is None for x in s
    )  # co-heading (learner opens on the bandit's six)
    assert all(x.metric == "damage" and x.timeout_penalty == -40.0 for x in s)
    assert all(x.damage_threshold == 12.0 for x in s)
    # non-decreasing aspect (a monotone ladder — never eases the geometry mid-run).
    assert all(b.aspect_deg >= a.aspect_deg for a, b in zip(s, s[1:]))
    assert s[-1].aspect_deg == 90.0  # tops out at the beam-merge acceptance
    # cumulative backstops + a real top rung to promote through (unlike the single-stage _SELFPLAY).
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None


def test_selfplay_beam_ladder_is_fine_grained_and_reanchors_at_45():
    # curriculum-25: the beam-merge ladder is the fix for the c24 aspect-60 wall. It RE-ANCHORS at the
    # mastered aspect 45, then hardens the merge in FINE ~7-8° steps (half the 15° step that stalled c24)
    # through the exact 45→60 transition that broke, out to the beam merge (90). Every rung is a tier-3
    # damage-gated self-play rung (co-heading) carrying the -40 anti-passivity timeout penalty; expert-BFM
    # seeding is a launch-time knob (--expert-p0), not a ladder property, so it isn't asserted here.
    s = build_stages(_SELFPLAY_BEAM, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 7
    assert [x.aspect_deg for x in s] == [45.0, 52.0, 60.0, 68.0, 76.0, 83.0, 90.0]
    assert s[0].aspect_deg == 45.0  # re-anchor at the mastered offset
    assert s[-1].aspect_deg == 90.0  # tops out at the beam merge
    assert all(x.tier == 3 and x.posture == "offensive" for x in s)
    assert all(x.opp_heading_deg is None for x in s)  # co-heading
    assert all(x.metric == "damage" and x.timeout_penalty == -40.0 for x in s)
    assert all(x.damage_threshold == 12.0 for x in s)
    # monotone non-decreasing aspect, and every step is FINER than the 15° that stalled c24.
    steps = [b.aspect_deg - a.aspect_deg for a, b in zip(s, s[1:])]
    assert all(d > 0 for d in steps) and max(steps) < 15.0
    # bars ease as the merge widens (the beam is the hardest) but never below the 0.25 acceptance floor.
    assert [x.winrate for x in s] == [0.35, 0.32, 0.30, 0.28, 0.25, 0.25, 0.25]
    # cumulative backstops + a real top rung to promote through.
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None


def test_beam_t3_ladder_is_honest_fine_and_reconsolidates_at_zero():
    # curriculum-25 (HONEST): the non-self-play aspect ladder is the fix after the c24 self-play gate was
    # proven to score probes vs weak pooled copies (0% ≥12HP vs a real T3 at the "mastered" aspect 45). It
    # re-consolidates the aspect-0 tail chase the c19 anchor genuinely KILLS, then hardens in FINE ~6° steps
    # THROUGH the honest wall (aspect 0→30, where the anchor fell 100%→20%→0% and began decking), widening
    # past 30°. Every rung is a tier-3 damage-gated OFFENSIVE rung vs the NATIVE bot (opponent_tier 0), co-
    # heading, carrying the -40 anti-passivity timeout penalty. Self-play is a launch flag (NOT set for this
    # ladder), so it isn't a ladder property and isn't asserted here — the geometry/gate shape is.
    s = build_stages(_BEAM_T3, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 10
    assert [x.aspect_deg for x in s] == [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 40.0, 55.0, 70.0, 90.0]
    assert s[0].aspect_deg == 0.0  # re-consolidate the genuinely-mastered tail chase
    assert s[-1].aspect_deg == 90.0  # tops out at the neutral beam merge
    assert all(x.tier == 3 and x.posture == "offensive" for x in s)
    assert all(x.opponent_tier == 0 for x in s)  # NATIVE Tier-3 bot every episode (honest gate)
    assert all(
        x.opp_heading_deg is None for x in s
    )  # co-heading (learner opens on the bandit's six)
    assert all(x.metric == "damage" and x.timeout_penalty == -40.0 for x in s)
    assert all(x.damage_threshold == 12.0 for x in s)
    # monotone non-decreasing aspect; steps THROUGH the wall (rungs 0→5, aspect ≤30) are FINE (≤6°).
    steps = [b.aspect_deg - a.aspect_deg for a, b in zip(s, s[1:])]
    assert all(d > 0 for d in steps)
    assert max(steps[:5]) <= 6.0  # the 0→30 wall zone is crossed in ~6° increments
    # bars ease monotonically as the merge widens and stay meaningful (the beam is hardest).
    assert [x.winrate for x in s] == [0.50, 0.40, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15]
    assert all(b.winrate <= a.winrate for a, b in zip(s, s[1:]))
    # cumulative backstops + a real top rung to promote through.
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(10)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 9) is None


def test_beam_t3_fine_ladder_is_a_5deg_staircase_through_the_45_cliff():
    # curriculum-26 (HONEST): the continuation after c25's beam-t3 demolished 0→40° vs the NATIVE bot but
    # WALLED at a sharp ~45-48° skill cliff (aspect-40 milestone: 87% ≥12HP at 40°, 7% at 45°, 0% at ≥48°;
    # reducing separation made it WORSE — got killed at 55°@450m). The fix is a FINE 5° aspect staircase
    # 40→45→…→90 at the SAME 700 m so every rung is only 5° off a trained checkpoint (transfer stays nonzero),
    # warm-started from the c25 aspect-40 chipper. Still tier-3 damage-gated OFFENSIVE vs the native bot, co-
    # heading, -40 anti-passivity timeout penalty; non-self-play (a launch flag, not a ladder property).
    s = build_stages(_BEAM_T3_FINE, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 11
    assert [x.aspect_deg for x in s] == [
        40.0,
        45.0,
        50.0,
        55.0,
        60.0,
        65.0,
        70.0,
        75.0,
        80.0,
        85.0,
        90.0,
    ]
    assert (
        s[0].aspect_deg == 40.0
    )  # re-consolidate the aspect-40 chip the c25 warm-start already owns
    assert s[-1].aspect_deg == 90.0  # tops out at the neutral beam merge
    assert all(x.tier == 3 and x.posture == "offensive" for x in s)
    assert all(x.opponent_tier == 0 for x in s)  # NATIVE Tier-3 bot every episode (honest gate)
    assert all(x.opp_heading_deg is None for x in s)  # co-heading
    assert all(x.metric == "damage" and x.timeout_penalty == -40.0 for x in s)
    assert all(x.damage_threshold == 12.0 for x in s)
    assert all(
        x.separation_m == 700.0 for x in s
    )  # SAME range — closer is worse at a beam merge (gets killed)
    # a uniform FINE 5° staircase — the whole point (c25 walled on its 40→55 jump of 15°).
    steps = [round(b.aspect_deg - a.aspect_deg, 6) for a, b in zip(s, s[1:])]
    assert steps == [5.0] * 10
    # bars ease monotonically as the merge widens toward the (hardest) perpendicular beam.
    assert [x.winrate for x in s] == [
        0.35,
        0.28,
        0.25,
        0.22,
        0.20,
        0.18,
        0.17,
        0.16,
        0.15,
        0.14,
        0.13,
    ]
    assert all(b.winrate <= a.winrate for a, b in zip(s, s[1:]))
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(11)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 10) is None


def test_beam_t3_net_ladder_is_net_gated_with_a_widening_aspect_band():
    # curriculum-27 (HONEST rebuild): after the c26 verdict exposed the damage gate as blind to HP
    # TAKEN (it certified a net-losing, low-aspect-forgetting policy), this ladder swaps to a NET
    # outcome gate and a per-episode aspect BAND [0, ceiling] so every stage rehearses the whole
    # forward hemisphere (anti-forgetting). Ceiling ramps 40→90; warm-start c25's run_stage06.
    s = build_stages(_BEAM_T3_NET, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 7
    # every rung is a BAND anchored at 0 whose ceiling (aspect_deg_max) ramps up.
    assert all(
        x.aspect_deg == 0.0 for x in s
    )  # band floor is the free tail-chase, always in rotation
    assert [x.aspect_deg_max for x in s] == [40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0]
    assert (
        s[0].aspect_deg_max == 40.0
    )  # consolidate the c25-mastered forward hemisphere as ONE policy
    assert s[-1].aspect_deg_max == 90.0  # tops out at the neutral beam merge
    # the NET metric — the whole point (damage gate was blind to deaths/decks/taken HP).
    assert all(x.metric == "net" for x in s)
    assert all(x.tier == 3 and x.posture == "offensive" for x in s)
    assert all(
        x.opponent_tier == 0 for x in s
    )  # NATIVE Tier-3 bot every episode (honest by construction)
    assert all(x.opp_heading_deg is None for x in s)  # co-heading
    assert all(
        x.separation_m == 700.0 for x in s
    )  # closer is worse at a beam merge (c25/c26 finding)
    # a no-outcome clock draw must be a loss so the policy converts (shared with damage rungs).
    assert all(x.timeout_penalty == -40.0 for x in s)
    # ceilings widen monotonically; net-HP bar eases as clean margins shrink toward the beam.
    assert all(b.aspect_deg_max > a.aspect_deg_max for a, b in zip(s, s[1:]))
    assert [x.net_threshold for x in s] == [8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 3.0]
    assert all(b.net_threshold <= a.net_threshold for a, b in zip(s, s[1:]))
    # winrate bar stays firm (0.6+) so passing forces the competence frontier UP the band, not just
    # farming the free low-aspect wins; eases only gently as the hard high-aspect slice grows.
    assert [x.winrate for x in s] == [0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.60]
    assert all(b.winrate <= a.winrate for a, b in zip(s, s[1:]))
    assert all(x.winrate >= 0.60 for x in s)
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None


def test_beam_t3_net_ladder_disables_banding_only_via_negative_sentinel():
    # aspect_deg_max ≥ 0 means "band active"; the legacy fixed-aspect ladders must stay at −1 so
    # _make_env doesn't accidentally randomize their bearing.
    net = build_stages(_BEAM_T3_NET, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert all(x.aspect_deg_max >= 0.0 for x in net)  # c27 rungs are banded
    fine = build_stages(_BEAM_T3_FINE, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert all(x.aspect_deg_max == -1.0 for x in fine)  # fixed-aspect ladder: banding OFF
    assert all(x.net_threshold == 0.0 for x in fine)  # and net_threshold is inert there


def test_beam_t3_push_ladder_interleaves_narrow_frontier_and_consolidation_bands():
    # curriculum-28: the c27 verdict showed a wide [0,ceiling] band lets 0-45 dominance MASK the
    # frontier (band-avg [0,80]≈60% cleared the bar while 70-90 was 0% net-win). c28 uses NARROW
    # 25°-wide frontier-weighted bands (high floor) so the gate measures the hard slice directly,
    # interleaved with [0,hi] CONSOLIDATION rungs (floor 0) that re-fuse the base against forgetting.
    s = build_stages(_BEAM_T3_NET_PUSH, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 7
    # frontier rungs (0,2,4,5) have a NON-ZERO floor; consolidation rungs (1,3,6) anchor at 0.
    assert [x.aspect_deg for x in s] == [40.0, 0.0, 50.0, 0.0, 60.0, 65.0, 0.0]
    assert [x.aspect_deg_max for x in s] == [65.0, 65.0, 75.0, 75.0, 85.0, 90.0, 90.0]
    # every frontier band is exactly 25° wide (the frontier-weighting invariant); the gate-avg then
    # tracks the frontier's own net-win rate, so a modest bar forces real advance out to lo+θ·width.
    frontier = [x for x in s if x.aspect_deg > 0.0]
    assert all(x.aspect_deg_max - x.aspect_deg == 25.0 for x in frontier)
    # ceilings never regress; the ladder tops out at the pure beam merge and ends consolidated at 0.
    assert all(b.aspect_deg_max >= a.aspect_deg_max for a, b in zip(s, s[1:]))
    assert s[-1].aspect_deg == 0.0 and s[-1].aspect_deg_max == 90.0  # final full-hemisphere stitch
    # bars are PER-BAND, deliberately NON-monotonic across the ladder: a wide consolidation rung's
    # bar is higher than the adjacent narrow frontier rung's (0.68 over [0,65] is easier than 0.48
    # over [50,75]). Every frontier bar is lower than both its neighboring consolidation bars.
    cons = {1, 3, 6}
    # each consolidation rung RAISES the bar over its immediately-preceding frontier push (the wide
    # band re-admits the big-margin low aspects, so a higher bar is both fair and forcing there).
    assert all(s[i].winrate > s[i - 1].winrate for i in sorted(cons))
    assert [x.winrate for x in s] == [0.55, 0.68, 0.48, 0.60, 0.42, 0.38, 0.55]
    # same honest net gate + native T3 + closer separation + co-heading + convert-or-lose clock as c27.
    assert all(x.metric == "net" for x in s)
    assert all(x.tier == 3 and x.posture == "offensive" and x.opponent_tier == 0 for x in s)
    assert all(x.opp_heading_deg is None and x.separation_m == 700.0 for x in s)
    assert all(x.timeout_penalty == -40.0 for x in s)
    assert all(x.aspect_deg_max >= 0.0 for x in s)  # banding active on every rung
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    # sanity: the interleave alternates floor∈{narrow,0}; consolidation indices anchor at 0.
    assert all(s[i].aspect_deg == 0.0 for i in cons)


def test_beam_t3_frontier_ladder_trains_full_band_but_gates_the_frontier_slice():
    # curriculum-29: the synthesis of c27 (always-[0,hi] → no forgetting, but a band-AVERAGE gate
    # the base masked) and c28 (narrow frontier → no masking, but excluded-base forgetting). Every
    # rung is a FULL [0,hi] band (floor 0 → base always trained, anti-forgetting) but the learner
    # gates on the outer slice only (via _in_frontier). Ceiling ramps 60→90; bar eases toward beam.
    s = build_stages(_BEAM_T3_FRONTIER, backstop_steps=BACKSTOP, damage_timeout_penalty=-40.0)
    assert len(s) == 7
    # ALWAYS-[0,hi]: every rung's band FLOOR is 0 (unlike c28's narrow frontier bands) — this is the
    # anti-forgetting property, since the base tail-chase is flown and trained on every rung.
    assert all(x.aspect_deg == 0.0 for x in s)
    assert [x.aspect_deg_max for x in s] == [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
    assert (
        s[0].aspect_deg_max == 60.0
    )  # picks up where the honest c27 [0,80] master half-held (60°)
    assert s[-1].aspect_deg_max == 90.0  # tops out at the pure neutral beam merge
    # same honest net gate + native T3 + closer separation + co-heading + convert-or-lose clock.
    assert all(x.metric == "net" for x in s)
    assert all(x.tier == 3 and x.posture == "offensive" and x.opponent_tier == 0 for x in s)
    assert all(x.opp_heading_deg is None and x.separation_m == 700.0 for x in s)
    assert all(x.timeout_penalty == -40.0 for x in s)
    assert all(x.aspect_deg_max >= 0.0 for x in s)  # banding active on every rung
    # ceilings widen monotonically; bar + net-HP threshold ease as clean margins vanish at the beam.
    assert all(b.aspect_deg_max > a.aspect_deg_max for a, b in zip(s, s[1:]))
    assert [x.winrate for x in s] == [0.55, 0.52, 0.48, 0.45, 0.42, 0.38, 0.35]
    assert all(b.winrate <= a.winrate for a, b in zip(s, s[1:]))
    assert [x.net_threshold for x in s] == [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 3.0]
    assert all(b.net_threshold <= a.net_threshold for a, b in zip(s, s[1:]))
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None


# -- _in_frontier (curriculum-29 honest frontier-slice gate vote filter) ----
def test_in_frontier_counts_only_the_outer_slice_of_a_band():
    # A [0,80] band: only probes whose START aspect is in [80−FRONTIER_SLICE_DEG, 80] vote. A
    # base/mid-band probe (0-45, the c27-free kills) is FLOWN + TRAINED but does NOT count toward
    # advancement — so a base-dominated band average can no longer mask an unmastered frontier.
    band = Stage(
        tier=3,
        separation_m=700.0,
        metric="net",
        winrate=0.5,
        backstop=0,
        aspect_deg=0.0,
        aspect_deg_max=80.0,
    )
    lo = 80.0 - FRONTIER_SLICE_DEG  # = 55.0
    assert _in_frontier(80.0, band)  # the ceiling itself
    assert _in_frontier(lo, band)  # exactly on the slice edge counts
    assert _in_frontier(70.0, band)  # inside the slice
    assert not _in_frontier(lo - 0.1, band)  # just below the slice → base/mid, no vote
    assert not _in_frontier(30.0, band)  # a free tail-chase kill: trained, but not a gate vote
    assert not _in_frontier(0.0, band)


def test_in_frontier_configurable_slice_narrows_the_voting_wedge():
    # curriculum-36: a caller-supplied slice_deg narrows the outer voting wedge so low-aspect kills
    # can no longer pad the gate (c35 "mastered 75°" while killing 0/40 at 75° — 50-60° kills inside
    # the default [50,75] slice carried the vote). With a 10° slice on a [0,75] band the wedge is
    # [65,75]: a 60° probe (which the 25° default would count) no longer votes.
    band = Stage(
        tier=3,
        separation_m=700.0,
        metric="kill",
        winrate=0.3,
        backstop=0,
        aspect_deg=0.0,
        aspect_deg_max=75.0,
    )
    assert _in_frontier(60.0, band)  # default 25° slice [50,75] counts a 60° probe
    assert not _in_frontier(60.0, band, 10.0)  # a 10° slice [65,75] does NOT — the honest fix
    assert _in_frontier(70.0, band, 10.0) and _in_frontier(75.0, band, 10.0)  # the outer wedge still votes
    assert not _in_frontier(64.9, band, 10.0)  # just below the narrowed wedge


def test_in_frontier_slice_floor_never_dips_below_the_band_floor():
    # A band NARROWER than FRONTIER_SLICE_DEG (e.g. [0,20] with a 25° slice) clamps the slice floor
    # to the band floor, so the whole (tiny) band is its own frontier — never a negative aspect.
    narrow = Stage(
        tier=3,
        separation_m=700.0,
        metric="net",
        winrate=0.5,
        backstop=0,
        aspect_deg=0.0,
        aspect_deg_max=20.0,
    )
    assert _in_frontier(0.0, narrow) and _in_frontier(20.0, narrow)


def test_in_frontier_non_banded_or_missing_aspect_counts_the_whole_rung():
    # A fixed-aspect / non-banded rung (aspect_deg_max < 0) has no frontier → legacy whole-rung vote.
    fixed = Stage(
        tier=3,
        separation_m=700.0,
        metric="net",
        winrate=0.5,
        backstop=0,
        aspect_deg=45.0,
        aspect_deg_max=-1.0,
    )
    assert _in_frontier(45.0, fixed) and _in_frontier(None, fixed)
    # A banded rung with an UNKNOWN aspect (older collector / non-tier3) also counts — never
    # silently drop every vote and stall the gate forever.
    band = Stage(
        tier=3,
        separation_m=700.0,
        metric="net",
        winrate=0.5,
        backstop=0,
        aspect_deg=0.0,
        aspect_deg_max=80.0,
    )
    assert _in_frontier(None, band)


# -- _gate_should_advance (mastery gate) -----------------------------------
def test_gate_disabled_when_winrate_nonpositive():
    wins = [1.0] * 200
    assert not _gate_should_advance(wins, window=40, min_episodes=60, winrate=0.0)
    assert not _gate_should_advance(wins, window=40, min_episodes=60, winrate=-1.0)


def test_gate_needs_min_episodes():
    assert not _gate_should_advance([1.0] * 59, window=40, min_episodes=60, winrate=0.6)
    assert _gate_should_advance([1.0] * 60, window=40, min_episodes=60, winrate=0.6)


def test_gate_measures_success_rate_over_window():
    below = [1.0] * 20 + [0.0] * 40  # last-40 = 0%
    assert not _gate_should_advance(below, window=40, min_episodes=60, winrate=0.6)
    at = [0.0] * 20 + ([1.0] * 24 + [0.0] * 16)  # last-40 = 24/40 = 60%
    assert _gate_should_advance(at, window=40, min_episodes=60, winrate=0.6)
    recovered = [0.0] * 100 + [1.0] * 40  # last-40 = 100%
    assert _gate_should_advance(recovered, window=40, min_episodes=60, winrate=0.6)


def test_strict_gate_rejects_a_lucky_trailing_streak():
    # The consistency requirement: a hot tail cannot promote a mostly-losing learner. A
    # 40-win streak riding on 80 prior losses is only 40% over the strict last-100 window.
    streak = [0.0] * 80 + [1.0] * 40
    assert not _gate_should_advance(streak, window=100, min_episodes=120, winrate=0.70)
    # 120 episodes played, but the deciding last-100 window is 69% — just under the bar ⇒ hold.
    just_under = [1.0] * 20 + [0.0] * 31 + [1.0] * 69  # len 120, last-100 = 69/100
    assert not _gate_should_advance(just_under, window=100, min_episodes=120, winrate=0.70)
    # A genuinely sustained ≥70% over the full 100-window clears it.
    sustained = [1.0] * 20 + [0.0] * 30 + [1.0] * 70  # len 120, last-100 = 70/100 = 70%
    assert _gate_should_advance(sustained, window=100, min_episodes=120, winrate=0.70)


def test_strict_gate_min_episodes_blocks_a_short_perfect_window():
    # Even a perfect last-100 is held until min_episodes have actually been played — a high
    # window bar and a high episode floor together are what forbid a lucky-streak promotion.
    assert not _gate_should_advance([1.0] * 100, window=100, min_episodes=120, winrate=0.70)
    assert _gate_should_advance([1.0] * 120, window=100, min_episodes=120, winrate=0.70)


# -- ladder navigation -----------------------------------------------------
def test_next_stage():
    assert _next_stage(STAGES, 0) == 1
    assert _next_stage(STAGES, 1) == 2
    assert _next_stage(STAGES, 2) is None  # top of the ladder


def test_stage_at_backstop_elapsed():
    assert stage_at(STAGES, 0, 0) == 0
    assert stage_at(STAGES, BACKSTOP - 1, 0) == 0
    assert stage_at(STAGES, BACKSTOP, 0) == 1
    assert stage_at(STAGES, 2 * BACKSTOP, 0) == 2


def test_stage_at_is_monotonic():
    # Already ahead of where the backstop alone would put us (the gate advanced us early):
    # never step back down.
    assert stage_at(STAGES, 0, 2) == 2
    assert stage_at(STAGES, BACKSTOP, 2) == 2


# -- _advance_stage (the composed decision) --------------------------------
def test_advance_holds_until_mastery():
    weak = [0.0] * 60
    target, reason = _advance_stage(STAGES, 0, 50_000, weak, **GATE)
    assert target == 0 and reason == ""


def test_advance_on_kill_gate_ahead_of_schedule():
    # Stage 0 is kill-gated at 0.70; a strong kill log promotes well before the backstop.
    strong = [1.0] * 60
    target, reason = _advance_stage(STAGES, 0, 50_000, strong, **GATE)
    assert target == 1 and "gate" in reason and "kill" in reason


def test_advance_uses_the_stages_own_metric_bar():
    # 80% success (32/40) clears the kill rung (bar 0.70) but NOT a survive rung (bar 0.85):
    # the gate reads each stage's own threshold, not a global one.
    log = [1.0] * 32 + [0.0] * 8 + [1.0] * 20  # last-40 = 32/40 = 80%
    assert _advance_stage(STAGES, 0, 50_000, log, **GATE)[0] == 1  # kill @0.70 → advance
    assert _advance_stage(STAGES, 1, 50_000, log, **GATE)[0] == 1  # survive @0.85 → hold


def test_advance_on_survive_gate():
    # A sustained ≥85% survival on the defensive rung clears it (the point-blank milestone).
    survived = [1.0] * 60
    target, reason = _advance_stage(STAGES, 1, 50_000, survived, **GATE)
    assert target == 2 and "gate" in reason and "survive" in reason


def test_step_backstop_forces_slow_gate():
    losing = [0.0] * 200
    target, reason = _advance_stage(STAGES, 0, BACKSTOP, losing, **GATE)
    assert target == 1 and "backstop" in reason


def test_advance_is_monotonic_never_regresses():
    # Gate put us on stage 2 early; the backstop at this step still nominally wants a lower
    # rung — never pull back down.
    target, reason = _advance_stage(STAGES, 2, BACKSTOP, [0.0] * 5, **GATE)
    assert target == 2 and reason == ""


def test_gate_steps_one_stage_at_a_time():
    # Crushing stage 0 advances to 1, never skipping straight to 2.
    target, _ = _advance_stage(STAGES, 0, 50_000, [1.0] * 60, **GATE)
    assert target == 1


def test_top_stage_gate_is_a_noop():
    # Mastering the final rung has nowhere to go — hold (self-play is a separate phase).
    target, reason = _advance_stage(STAGES, 2, 10 * BACKSTOP, [1.0] * 60, **GATE)
    assert target == 2 and reason == ""


# -- curriculum-56: merge-survive ladder + the imitation-seed break-turn stick ------------
def test_merge_survive_ladder_is_the_survive_convert_reframe():
    # c56: the honest reframe of the head-on merge program — _MERGE_CHASE's opponent+separation ramp
    # but SURVIVE-CONVERT gated (net/survive, not kill) so a clean-survive-then-convert advances even
    # when the neutral-merge FACE-SHOT is a geometric draw. `net` rungs carry the anti-passivity
    # timeout penalty (a no-outcome clock draw must be a loss), the top `survive` rung suppresses it.
    s = build_stages(_MERGE_SURVIVE, backstop_steps=BACKSTOP, damage_timeout_penalty=-25.0)
    assert len(s) == 7
    # Head-on geometry throughout the merge rungs: aspect_deg 0 on every rung; opp_heading 180 on the
    # full head-on rungs (rung 0 is the co-heading REHEARSAL anchor; rung 1 is a 135° quartering ease-in).
    assert all(x.tier == 3 and x.posture == "offensive" and x.aspect_deg == 0.0 for x in s)
    assert s[0].opp_heading_deg is None  # rung 0: co-heading rehearsal anchor
    assert s[1].opp_heading_deg == 135.0  # rung 1: quartering ease-in
    assert all(x.opp_heading_deg == 180.0 for x in s[2:])  # rungs 2-6: full head-on merge
    # Opponent ramp: native T3 rehearsal (opp 0), then the NON-shooting T2 evader teaches the pass +
    # conversion (rungs 1-3), then the T3 SHOOTER for the milestone + survive rungs (rungs 4-6).
    assert [x.opponent_tier for x in s] == [0, 2, 2, 2, 0, 0, 0]
    # SURVIVE-CONVERT gate: net on the rehearsal + merge rungs, `survive` (just don't lose) at the top.
    assert [x.metric for x in s] == ["net", "net", "net", "net", "net", "net", "survive"]
    # net rungs get the anti-passivity timeout penalty; the survive rung suppresses it (clock = win).
    assert all(x.timeout_penalty == -25.0 for x in s[:6])
    assert s[6].timeout_penalty == 0.0
    # separation eases the pass (long → tight) within each opponent block; net threshold + bar ease up.
    assert [x.separation_m for x in s] == [700.0, 1500.0, 1500.0, 1000.0, 1800.0, 1200.0, 900.0]
    assert [x.net_threshold for x in s[:6]] == [5.0, 5.0, 4.0, 4.0, 3.0, 3.0]
    assert all(b.net_threshold <= a.net_threshold for a, b in zip(s[:6], s[1:6]))
    # rung 0 is BANDED [0,45] (frontier slice votes on the champion's 45° strength); merge rungs fixed.
    assert s[0].aspect_deg_max == 45.0
    assert all(x.aspect_deg_max < 0.0 for x in s[1:])
    assert [x.backstop for x in s] == [i * BACKSTOP for i in range(7)]
    assert _next_stage(s, 0) == 1 and _next_stage(s, 6) is None
    # a clean survive-convert (positive net, no losing terminal) clears rung 5 (the milestone) — the
    # exact outcome a kill gate would reject as a non-kill; a passive 0-net clock draw does NOT.
    milestone = s[5]
    assert _stage_win({"event": "timeout", "dmg_dealt": 40.0, "dmg_taken": 10.0}, milestone)  # +30
    assert not _stage_win({"event": "timeout", "dmg_dealt": 0.0, "dmg_taken": 0.0}, milestone)
    assert not _stage_win({"event": "hard_deck", "dmg_dealt": 80.0, "dmg_taken": 0.0}, milestone)


def test_vertical_egg_stick_turns_toward_the_bandit_and_pulls_up_hard():
    # c58 VERTICAL EGG: rolls a MODEST ~30° bank toward the bandit (sign) and PULLS MAX-g toward a steep
    # +50° nose-HIGH pitch (elevator NEGATIVE = pull, the inverted FLCS sign), rudder 0, full throttle.
    # The TRIGGER is not part of the [:4] stick. From a level state the pull must SATURATE (−1.0) — this
    # is what makes it a vertical zoom, not the old shallow descending break.
    right = _vertical_egg_stick(phi_rad=0.0, theta_rad=0.0, sign=+1.0)  # bandit on the right
    left = _vertical_egg_stick(phi_rad=0.0, theta_rad=0.0, sign=-1.0)  # bandit on the left
    assert right.shape == (4,)
    assert right[0] > 0.0  # +sign → roll RIGHT (aileron +)
    assert left[0] < 0.0  # -sign → roll LEFT
    assert right[1] == -1.0  # from level → MAX PULL (nose-up) toward the steep +50° egg target
    assert right[2] == 0.0  # rudder stays 0 (FLCS coordinates)
    assert right[3] == 1.0  # full throttle to trade speed for altitude over the top
    # the bank target is MODEST (keep the pull mostly vertical) — well under a 45° horizontal break.
    assert _VE_BANK_RAD < np.radians(45.0)
    # rolls toward, then HOLDS, the bank: at the ~30° target bank the aileron command drops to ~0.
    at_bank = _vertical_egg_stick(phi_rad=_VE_BANK_RAD, theta_rad=0.0, sign=+1.0)
    assert abs(at_bank[0]) < abs(right[0])  # near the target bank → less roll command
    for a in (right, left, at_bank):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_vertical_egg_stick_always_pulls_up_out_of_a_nose_low_state():
    # c58: the SAME unconditional stick always pulls a nose-low attitude UP into the vertical (no AGL
    # branch — the c57 fix for the c56 non-imitable floor crutch stands). A nose-low state commands a
    # HARD pull; the pull only eases once the nose is near/above the steep +50° egg target — it never
    # commands a further dive.
    deep = _vertical_egg_stick(phi_rad=+_VE_BANK_RAD, theta_rad=-0.5, sign=+1.0)  # steep nose-low
    shallow = _vertical_egg_stick(phi_rad=+_VE_BANK_RAD, theta_rad=-0.1, sign=+1.0)  # mild nose-low
    assert deep[1] == -1.0 and shallow[1] == -1.0  # both SATURATE the pull — always raise the nose hard
    # near the +50° egg target the pull relaxes (no runaway pull), and above it eases (never a dive cmd).
    near = _vertical_egg_stick(phi_rad=0.0, theta_rad=_VE_PITCH_TGT_RAD - np.radians(2.0), sign=+1.0)
    assert -1.0 < near[1] < 0.0  # just below the target → gentle pull, no longer saturated
    at_tgt = _vertical_egg_stick(phi_rad=0.0, theta_rad=_VE_PITCH_TGT_RAD, sign=+1.0)
    assert at_tgt[1] == 0.0  # on the +50° egg target → no elevator command
    above = _vertical_egg_stick(phi_rad=0.0, theta_rad=_VE_PITCH_TGT_RAD + 0.2, sign=+1.0)
    assert above[1] >= 0.0  # above the target → ease (push), never a further pull
    for a in (deep, shallow, near, at_tgt, above):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_highyoyo_stick_pulls_up_and_banks_hard_while_closure_is_hot():
    # HIGH YO-YO — UP/BLEED phase (closure > CLOSURE_HI = a GENUINE overshoot): bank HARD toward the
    # bandit (sign) + pull the nose UP above his plane (elevator NEGATIVE = pull) and EASE the throttle
    # (< full) to trade the surplus closure for turning room. The TRIGGER is not part of the [:4] stick.
    hot = 150.0  # well above CLOSURE_HI (120) — a hot, overshooting closure
    right = _highyoyo_stick(phi_rad=0.0, theta_rad=0.0, closure=hot, sign=+1.0)  # bandit on the right
    left = _highyoyo_stick(phi_rad=0.0, theta_rad=0.0, closure=hot, sign=-1.0)   # bandit on the left
    assert right.shape == (4,)
    assert right[0] > 0.0   # +sign → roll RIGHT (aileron +)
    assert left[0] < 0.0    # -sign → roll LEFT
    assert right[1] < 0.0   # from level → PULL (nose-up) toward the +24° above-plane target
    assert right[2] == 0.0  # rudder stays 0 (FLCS coordinates)
    assert 0.0 < right[3] < 1.0  # EASE the throttle to bleed the surplus closure (not full)
    # the bank target is HARD (a tight bleeding turn), well past a 45° break.
    assert _HY_BANK_UP_RAD > np.radians(45.0)
    # rolls toward, then HOLDS, the bank: at the hard target bank the aileron command drops.
    at_bank = _highyoyo_stick(phi_rad=_HY_BANK_UP_RAD, theta_rad=0.0, closure=hot, sign=+1.0)
    assert abs(at_bank[0]) < abs(right[0])
    # once the nose is above the +24° up-target, the pull relaxes (no runaway).
    above = _highyoyo_stick(phi_rad=0.0, theta_rad=_HY_PITCH_UP_RAD + 0.2, closure=hot, sign=+1.0)
    assert above[1] >= 0.0
    for a in (right, left, at_bank, above):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_highyoyo_stick_reacquires_down_unless_closure_is_a_genuine_overshoot():
    # HIGH YO-YO — DOWN/REACQUIRE phase (closure <= CLOSURE_HI): roll toward a MODERATE bank + bring the
    # nose back DOWN onto the LOS (theta_tgt < 0) at FULL throttle to come down and convert. c71 FIX: the
    # UP/DOWN split is at CLOSURE_HI (120), NOT CLOSURE_LO (40), so a MODERATE closure (40 < c < 120) now
    # REACQUIRES (down) instead of zooming up — the c70 bug was UP-only, which flew the nose 158° away.
    bled = _HY_CLOSURE_LO - 5.0        # 35 — closure fully bled off
    moderate = 0.5 * (_HY_CLOSURE_LO + _HY_CLOSURE_UP)  # 80 — between the two thresholds (the c70-bug regime)
    overshoot = _HY_CLOSURE_UP + 30.0  # 150 — a genuine overshoot
    down = _highyoyo_stick(phi_rad=0.0, theta_rad=0.0, closure=bled, sign=+1.0)
    down_mod = _highyoyo_stick(phi_rad=0.0, theta_rad=0.0, closure=moderate, sign=+1.0)
    up = _highyoyo_stick(phi_rad=0.0, theta_rad=0.0, closure=overshoot, sign=+1.0)
    # moderate closure must behave like the bled DOWN phase (the fix), NOT the UP phase.
    assert np.allclose(down_mod, down)
    assert down[3] == 1.0            # DOWN phase → FULL throttle (rebuild energy for the reacquire)
    assert down[3] > up[3]           # the overshoot UP phase eased the throttle; below it goes full
    assert down[1] > 0.0             # from level toward a NEGATIVE (nose-down) target → push/ease down
    assert up[1] < down[1]           # the UP phase pulls harder (more nose-up) than the DOWN phase
    for a in (down, down_mod, up):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_cpa_miss_reads_the_pass_geometry_not_the_present_range():
    # The lead turn manages the PREDICTED CLOSEST POINT OF APPROACH, because that — not the slant range
    # — is the turning-room quantity at a head-on merge. Me at the origin tracking due NORTH; a bandit
    # 1400 m EAST and 2653 m north, closing head-on (reciprocal heading). He is 3000 m away right now,
    # yet the PASS will carry the full 1400 m of lateral offset: that difference is the whole point.
    vel_me = np.asarray([0.0, 200.0, 0.0])   # ENU: due north
    vel_op = np.asarray([0.0, -200.0, 0.0])  # reciprocal
    miss = _cpa_miss_m([0.0, 0.0, 0.0], vel_me, [1400.0, 2653.0, 0.0], vel_op)
    assert miss == pytest.approx(1400.0, abs=1.0)
    # mirrored offset ⇒ same turning room (it is a magnitude; the turn DIRECTION comes from ata_az).
    assert _cpa_miss_m([0.0, 0.0, 0.0], vel_me, [-1400.0, 2653.0, 0.0], vel_op) == pytest.approx(
        1400.0, abs=1.0
    )
    # a collision-course spawn has NO turning room at all — the c90/c91 ±135 m merge, measured as ~0.
    assert _cpa_miss_m([0.0, 0.0, 0.0], vel_me, [0.0, 3000.0, 0.0], vel_op) == pytest.approx(
        0.0, abs=1.0
    )
    # already PAST the CPA (he is behind me and opening): t clamps to 0 ⇒ report the present separation,
    # never a stale prediction behind the jet.
    past = _cpa_miss_m([0.0, 0.0, 0.0], vel_me, [100.0, -2000.0, 0.0], vel_op)
    assert past == pytest.approx(np.hypot(100.0, 2000.0), abs=1.0)
    # co-heading, matched speed: no closure ⇒ the CPA is now, and it must not divide by zero.
    assert _cpa_miss_m([0.0, 0.0, 0.0], vel_me, [500.0, 800.0, 0.0], vel_me) == pytest.approx(
        np.hypot(500.0, 800.0), abs=1.0
    )


def test_leadturn_approach_manages_the_cpa_toward_turning_room_at_corner_speed():
    # c95 APPROACH phase (rng > commit): bank proportionally to the FRACTIONAL CPA error — toward his
    # pass side when the miss is too big, AWAY when it is too small (a collision course has no turning
    # room) — nose slightly low while slow, throttle closing on corner speed.
    wide = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, 3.0 * _LT_TURNROOM_M, +1.0)
    slight = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, _LT_TURNROOM_M + 100.0, +1.0)
    tight = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, 0.1 * _LT_TURNROOM_M, +1.0)
    onroom = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, _LT_TURNROOM_M, +1.0)
    assert wide.shape == (4,)
    assert wide[0] > 0.0                     # miss too BIG → bank TOWARD him (sign +1) to cut the corner
    assert tight[0] < 0.0                    # miss too SMALL → bank AWAY to open turning room
    assert onroom[0] == 0.0                  # exactly on the measured turning room → hold wings level
    assert wide[0] > slight[0] > 0.0         # proportional, not bang-bang
    assert wide[2] == 0.0                    # rudder stays 0 (FLCS coordinates)
    # mirrored: a bandit off to the LEFT inverts every bank command.
    mirror = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, 3.0 * _LT_TURNROOM_M, -1.0)
    assert mirror[0] < 0.0
    # SPEED law: turn radius scales with V², so the approach flies TO corner speed — full throttle and
    # nose slightly low while slow, throttle back once there (the diag arm arrived at 331 m/s and its
    # radius blew out to 1241 m against 574 m at 225).
    slow = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT - 60.0, False, _LT_TURNROOM_M, +1.0)
    fast = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT + 60.0, False, _LT_TURNROOM_M, +1.0)
    assert slow[3] > onroom[3] > fast[3]
    assert slow[1] > 0.0                     # slow → nose LOW (elevator positive = push) to accelerate
    assert onroom[1] == 0.0                  # at corner speed → level target, no pitch command
    for a in (wide, slight, tight, onroom, mirror, slow, fast):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_leadturn_phase_latches_the_turn_and_hands_the_last_30deg_to_the_tracker():
    HEADON, BEHIND = np.radians(170.0), np.radians(30.0)   # aspect at the merge entry / after converting
    far, near = _LT_COMMIT_RNG_M + 1.0, _LT_COMMIT_RNG_M
    # THE PHASE MUST BE LATCHED, and that is measured: BOTH stateless phase tests were tried on the real
    # merge and BOTH destroyed the conversion, in opposite directions.
    #   * range only (commit ⟺ rng <= _LT_COMMIT_RNG_M every tick): committed correctly, then REVERTED to
    #     APPROACH ~2 s later when the pass opened the range back past 1000 m (933 → 1140), dropping the
    #     pull from 6.9 g to nz +1.5 exactly when the turn had to keep coming. ata ran 121° → 139° → 155°
    #     → 171°: gq_honest 0.00 / a 169 m lay / aspMn 85° — the do-nothing control's numbers.
    #   * range AND closure: WORSE — the APPROACH banks AWAY to build turning room, which drops closure
    #     through zero transiently, so COMMIT tripped at ~2500 m and the jet hard-turned away from a merge
    #     it had never entered (timeout 8/8, passes 1.0, tInR 4.9 s, rngMn 875 m).
    assert _leadturn_phase("APPROACH", far, np.radians(30.0), HEADON) == "APPROACH"
    assert _leadturn_phase("APPROACH", near, np.radians(30.0), HEADON) == "TURN"
    # ...and the TURN is STICKY at any range, however far the pass opens it back up. This one assertion
    # is the range-only disproof above.
    assert _leadturn_phase("TURN", 10.0 * _LT_COMMIT_RNG_M, np.radians(120.0), HEADON) == "TURN"
    # TRACK takes over for the last ~30°, and that is measured too: WITHOUT it the blind max-rate corner
    # stalls at ata 29° / aspect 47° and climbs out of the arena (left_bubble 5/8, dAGL +1782), where the
    # reference arm's tracker took the same geometry to ata 8° / aspect 14° / gq_honest 0.49.
    assert _leadturn_phase("TURN", 600.0, np.radians(29.0), np.radians(47.0)) == "TRACK"
    # both conditions are required: behind him but nose still wide, or nose-on but he is still head-on
    # (i.e. we are the one being converted), are BOTH still a turning problem.
    assert _leadturn_phase("TURN", 600.0, np.radians(80.0), BEHIND) == "TURN"
    assert _leadturn_phase("TURN", 600.0, np.radians(10.0), HEADON) == "TURN"
    # the entry thresholds are the boundaries themselves (strict), so the phase can't chatter on equality
    assert _leadturn_phase("TURN", 600.0, _LT_TRACK_ATA_RAD, BEHIND) == "TURN"
    assert _leadturn_phase("TURN", 600.0, np.radians(10.0), _LT_TRACK_ASPECT_RAD) == "TURN"
    # TRACK holds while he stays in front of the wing line, and never falls back to a blind TURN...
    assert _leadturn_phase("TRACK", 4000.0, np.radians(85.0), _LT_TRACK_EXIT_ASPECT_RAD) == "TRACK"
    # ...but if he gets out in FRONT pointing back at us that is another merge, not a chase: restart the
    # machine at the top, which also clears the TURN latch so the next pass gets its turning room built.
    assert _leadturn_phase("TRACK", 4000.0, np.radians(5.0), np.radians(150.0)) == "APPROACH"
    assert _LT_TRACK_ASPECT_RAD < _LT_TRACK_EXIT_ASPECT_RAD


def test_leadturn_commit_is_a_hard_one_circle_turn_with_the_pull_gated_on_roll():
    # c95 COMMIT phase (latched): roll hard TOWARD his pass side at full throttle and pull max Nz, but
    # only ONCE THE ROLL IS IN. g spent while still rolling is a zoom, not a turn: that is exactly how
    # the first version of this arm climbed +4.4 km.
    rolling = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, True, _LT_TURNROOM_M, +1.0)
    rolled = _leadturn_stick(
        _LT_TURN_BANK_RAD, 0.0, 0.0, _LT_V_CORNER_TGT, True, _LT_TURNROOM_M, +1.0
    )
    assert rolling[0] > 0.0                  # wings level → roll RIGHT toward his pass side
    assert rolled[0] == pytest.approx(0.0)   # at the corner bank → hold it
    assert rolled[1] == pytest.approx(-1.0)  # roll complete → MAX pull (negative = nose-up/+Nz)
    assert rolled[1] < rolling[1] < 0.0      # still rolling → the pull is held back
    assert rolling[3] == 1.0 and rolled[3] == 1.0  # full throttle through the corner
    assert rolling[2] == 0.0
    # a real max-rate corner, not a 45° break — and the turning room is inside the commit range.
    assert _LT_TURN_BANK_RAD > np.radians(60.0)
    assert _LT_TURNROOM_M < _LT_COMMIT_RNG_M
    # mirrored pass side ⇒ mirrored roll, same pull once established (one-circle either way).
    left = _leadturn_stick(
        -_LT_TURN_BANK_RAD, 0.0, 0.0, _LT_V_CORNER_TGT, True, _LT_TURNROOM_M, -1.0
    )
    assert left[1] == pytest.approx(-1.0)
    assert _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, True, _LT_TURNROOM_M, -1.0)[0] < 0.0
    # rolled well PAST the target (bank error > 90°) must never command a PUSH — the pull just stops.
    overbanked = _leadturn_stick(
        -np.radians(40.0), 0.0, 0.0, _LT_V_CORNER_TGT, True, _LT_TURNROOM_M, +1.0
    )
    assert overbanked[1] == 0.0
    # the two phases are genuinely different sticks: committed = full throttle + max pull; not committed
    # (identical geometry) = a gentle setup bank at part throttle. Nothing but the latch distinguishes
    # them, so the CPA and the range can no longer silently flip the jet out of its own turn.
    approach = _leadturn_stick(0.0, 0.0, 0.0, _LT_V_CORNER_TGT, False, _LT_TURNROOM_M, +1.0)
    assert rolling[3] == 1.0 and approach[3] < 1.0
    assert rolling[1] < 0.0 and approach[1] == 0.0
    for a in (rolling, rolled, left, overbanked, approach):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_leadturn_guards_the_flight_path_with_the_bank_not_the_pull():
    # THE load-bearing fix ([[merge-leadturn-conversion]]): 80° bank + max Nz zoom-climbed +4.4 km and
    # bled 252→152 m/s, because once γ goes positive at high bank the pull is nearly HORIZONTAL — so
    # nothing restores the flight path, and an attitude test (bfm._deck_guard's `theta < 10° ⇒ pull`)
    # cannot arrest it either. The correction goes to the BANK: γ>0 ⇒ roll PAST vertical so the pull
    # drags the nose back down to the horizon; γ<0 ⇒ shallow out.
    args = (_LT_V_CORNER_TGT, True, _LT_TURNROOM_M, +1.0)
    # held AT the nominal corner bank: level flight path ⇒ hold; climbing ⇒ roll FURTHER over;
    # descending ⇒ roll BACK toward level. Note the input is γ (flight path), not θ (attitude).
    level = _leadturn_stick(_LT_TURN_BANK_RAD, 0.0, 0.0, *args)
    climbing = _leadturn_stick(_LT_TURN_BANK_RAD, 0.0, np.radians(15.0), *args)
    diving = _leadturn_stick(_LT_TURN_BANK_RAD, 0.0, np.radians(-15.0), *args)
    assert level[0] == pytest.approx(0.0)
    assert climbing[0] > 0.0   # zooming → steepen the bank so the g pulls the nose DOWN to the horizon
    assert diving[0] < 0.0     # descending → shallow the bank so the g pulls the nose UP
    # the correction is CLAMPED to a flyable band: never past _LT_BANK_MAX_RAD, never below a real turn.
    assert _leadturn_stick(_LT_BANK_MAX_RAD, 0.0, np.radians(60.0), *args)[0] == pytest.approx(0.0)
    assert _leadturn_stick(_LT_BANK_MIN_RAD, 0.0, np.radians(-60.0), *args)[0] == pytest.approx(0.0)
    assert _LT_BANK_MIN_RAD < _LT_TURN_BANK_RAD < _LT_BANK_MAX_RAD
    # and γ NEVER weakens the pull: at each γ's own settled bank the jet still pulls max g. This is what
    # distinguishes the fix from "ease off when climbing" — the corner is never traded away.
    for gamma in (np.radians(-15.0), 0.0, np.radians(15.0)):
        settled = float(np.clip(_LT_TURN_BANK_RAD + 2.0 * gamma, _LT_BANK_MIN_RAD, _LT_BANK_MAX_RAD))
        assert _leadturn_stick(settled, 0.0, gamma, *args)[1] == pytest.approx(-1.0)


def test_leadturn_eases_the_pull_below_sustain_speed_but_never_stops_turning():
    # A max-rate corner held past the sustained-rate speed bleeds the jet to a stall (the c66 zoom-stall
    # signature: 222→80 m/s while the T3 holds ≥128). So below _LT_V_SUSTAIN_MPS the pull is scaled back
    # proportionally — but never below _LT_V_EASE_FLOOR, because a slow jet must still turn.
    at_bank = (_LT_TURN_BANK_RAD, 0.0, 0.0)
    fast = _leadturn_stick(*at_bank, _LT_V_SUSTAIN_MPS + 40.0, True, _LT_TURNROOM_M, +1.0)
    onsustain = _leadturn_stick(*at_bank, _LT_V_SUSTAIN_MPS, True, _LT_TURNROOM_M, +1.0)
    slow = _leadturn_stick(*at_bank, 0.6 * _LT_V_SUSTAIN_MPS, True, _LT_TURNROOM_M, +1.0)
    crawling = _leadturn_stick(*at_bank, 5.0, True, _LT_TURNROOM_M, +1.0)
    assert fast[1] == pytest.approx(-1.0)        # above sustain → full corner
    assert onsustain[1] == pytest.approx(-1.0)   # AT sustain → still full (the ease starts below it)
    assert -1.0 < slow[1] < 0.0                  # below sustain → eased, still pulling
    assert crawling[1] == pytest.approx(-_LT_V_EASE_FLOOR)  # floored, never zero
    assert slow[1] < crawling[1]                 # monotone in airspeed
    assert 0.0 < _LT_V_EASE_FLOOR < 1.0
    for a in (fast, onsustain, slow, crawling):
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_chase_t3_neutral_ladder_is_a_hot_pass_that_excludes_the_on_six_gift():
    # c70: the NO-GIFT HOT-PASS ladder — a co-heading converting pass vs the native aggressive T3
    # (sep 700), kill-gated. Physics (probe): the HIGH YOYO manages EXCESS CLOSURE, which at co-speed
    # is only reachable in the ~45-60° band where the T3's hard break decelerates it in front of the
    # co-speed pursuer (the beam has no overtake → no closure → the high yoyo can't fire there). So the
    # band anchors at 45° (EXCLUDES the 0-45 on-the-six tracking gift) through the beam-approach — the
    # overshoot regime the high yoyo solves — NOT the beam and NOT the head-on merge (opp_heading None).
    stages = build_stages(_CHASE_T3_NEUTRAL, backstop_steps=25_000_000, start_stage=0)
    assert len(stages) == 2
    for s in stages:
        assert s.tier == 3 and s.opponent_tier == 0        # native aggressive T3
        assert s.posture == "offensive"
        assert s.metric == "kill"                          # kill-gated (no cowardice survive-stage)
        assert s.opp_heading_deg is None                   # co-heading converting pass, NOT the head-on merge
        assert s.separation_m == 700.0
        assert s.aspect_deg_max > s.aspect_deg             # a real per-episode band
        assert s.aspect_deg >= 45.0                        # anchored above the on-the-six gift (0–45 excluded)
    # rung 0 is the hot converting pass (its 45-60 zone fires the high-yoyo seed); rung 1 pushes toward
    # the beam (harder, reward-driven — less seed).
    assert stages[0].aspect_deg == 45.0 and stages[0].aspect_deg_max == 80.0
    assert stages[1].aspect_deg == 55.0 and stages[1].aspect_deg_max == 90.0
    # cumulative backstops from the CLI: rung 0 starts immediately, rung 1 fires at one backstop step.
    assert stages[0].backstop == 0 and stages[1].backstop == 25_000_000


def test_rehearse_stage_is_the_coheading_finish_anchor():
    # c57: the dedicated rehearsal workers fly a FIXED co-heading finish stage — the champion's proven
    # chase-t3-killgate rung-0 home turf (native T3, aspect band [0,55]°, sep 700 m, KILL-gated) — so
    # the shared buffer never forgets the ≤55° tracking kill while the merge workers push the head-on
    # program. Build the same Stage main() constructs and assert its shape.
    from astrum_dogfight import constants as C  # reward constants (R_TIMEOUT = kill-rung penalty)

    rs = Stage(
        tier=3,
        separation_m=700.0,
        metric="kill",
        winrate=0.0,
        backstop=0,
        timeout_penalty=C.R_TIMEOUT,
        posture="offensive",
        aspect_deg=0.0,
        aspect_deg_max=55.0,
        opponent_tier=0,
        opp_heading_deg=None,
    )
    assert rs.tier == 3 and rs.posture == "offensive"
    assert rs.opp_heading_deg is None  # CO-HEADING (the finish), not a head-on merge
    assert rs.aspect_deg == 0.0 and rs.aspect_deg_max == 55.0  # band [0,55]° = the mastered finish
    assert rs.opponent_tier == 0  # the native T3 (the real target), not a weak dummy
    assert rs.metric == "kill" and rs.timeout_penalty == C.R_TIMEOUT  # kill-gated, penalize a draw
    # a clean ≤55° kill clears it; a passive draw is a loss (keeps rehearsal AGGRESSIVE, never coasting).
    assert _stage_win({"event": "kill", "dmg_dealt": 100.0, "dmg_taken": 0.0}, rs)
    assert not _stage_win({"event": "timeout", "dmg_dealt": 0.0, "dmg_taken": 0.0}, rs)
