"""`stage_env_extra` — which scenario kwargs a curriculum stage adds, and what the anchor is shielded from.

This exists because of a real defect. `--me-dalt-ft` (the curriculum-79 altitude perch) is a
RUN-level knob carried in `env_kwargs`, and `_make_env` applied it to *any* tier-3/offensive stage.
The rehearsal anchor is itself tier-3/offensive, so the perch silently shifted the anchor's start
distribution — destroying the exact anti-forgetting guarantee the anchor exists to provide, and
contradicting `--me-dalt-ft`'s own help text ("the co-heading rehearse anchor stays co-alt"). It
went unnoticed for a whole run because the logic lived in a closure inside `run_parallel` and was
therefore untestable. The lesson generalises: **run-level `env_kwargs` reach the anchor; only
`Stage` fields distinguish it.** Every new run-level env knob needs a case here.
"""

import dataclasses

import pytest

from astrum_rl.tdmpc2.train_parallel import Stage, stage_env_extra


def t3(**kw):
    """An offensive tier-3 stage (the shape both the merge rungs and the anchor take)."""
    base = Stage(
        tier=3, separation_m=3000.0, metric="kill", winrate=0.0, backstop=0, posture="offensive"
    )
    return dataclasses.replace(base, **kw) if kw else base


# --- the anchor shield: the whole reason this module exists -------------------------------------


@pytest.mark.parametrize(
    "knob,scenario_key",
    [("me_dalt_ft", "me_dalt_ft"), ("merge_alt_ft", "alt_ft"), ("merge_lateral_m", "lateral_m_max")],
)
def test_run_level_altitude_knobs_reach_a_merge_rung(knob, scenario_key):
    extra = stage_env_extra(t3(), is_rehearse=False, **{knob: 2000.0})
    assert extra[scenario_key] == 2000.0


@pytest.mark.parametrize(
    "knob,scenario_key",
    [("me_dalt_ft", "me_dalt_ft"), ("merge_alt_ft", "alt_ft"), ("merge_lateral_m", "lateral_m_max")],
)
def test_run_level_altitude_knobs_never_reach_the_rehearsal_anchor(knob, scenario_key):
    """The regression. The anchor is tier-3/offensive too, so only `is_rehearse` can stop these."""
    extra = stage_env_extra(t3(), is_rehearse=True, **{knob: 2000.0})
    assert scenario_key not in extra


def test_anchor_still_gets_its_own_stage_fields():
    """Shielding the anchor from run-level knobs must not strip the fields that DEFINE it."""
    anchor = t3(aspect_deg=0.0, aspect_deg_max=55.0, separation_m=700.0, max_time_s=60.0)
    extra = stage_env_extra(anchor, is_rehearse=True, me_dalt_ft=2000.0, merge_alt_ft=30000.0)
    assert extra["posture"] == "offensive"
    assert extra["aspect_deg"] == 0.0
    assert extra["aspect_deg_max"] == 55.0
    assert "me_dalt_ft" not in extra and "alt_ft" not in extra


def test_both_altitude_knobs_are_independent():
    """The perch is an EDGE over the bandit; the block altitude is the shared budget. Not the same."""
    extra = stage_env_extra(t3(), is_rehearse=False, me_dalt_ft=2000.0, merge_alt_ft=30000.0)
    assert extra["me_dalt_ft"] == 2000.0
    assert extra["alt_ft"] == 30000.0


@pytest.mark.parametrize("knob", ["me_dalt_ft", "merge_alt_ft", "merge_lateral_m"])
def test_zero_means_absent_not_zero(knob):
    """0 must mean "use the scenario default", never "spawn at 0 ft" — these are sentinels."""
    assert stage_env_extra(t3(), is_rehearse=False, **{knob: 0.0}) == {
        "posture": "offensive",
        "aspect_deg": 180.0,
    }


# --- the block-altitude BAND (curriculum-91) ----------------------------------------------------


def test_block_altitude_band_is_forwarded_as_a_pair():
    extra = stage_env_extra(t3(), is_rehearse=False, merge_alt_ft=15000.0, merge_alt_ft_max=30000.0)
    assert extra["alt_ft"] == 15000.0
    assert extra["alt_ft_max"] == 30000.0


def test_band_ceiling_alone_does_nothing():
    """A ceiling with no floor must not leak a lone `alt_ft_max` — the scenario samples [lo, hi]."""
    extra = stage_env_extra(t3(), is_rehearse=False, merge_alt_ft=0.0, merge_alt_ft_max=30000.0)
    assert "alt_ft_max" not in extra and "alt_ft" not in extra


def test_the_band_is_shielded_from_the_anchor_too():
    """Whole point of the band is a MERGE budget; the anchor keeps the 15,000 ft it was learned at."""
    extra = stage_env_extra(t3(), is_rehearse=True, merge_alt_ft=15000.0, merge_alt_ft_max=30000.0)
    assert "alt_ft" not in extra and "alt_ft_max" not in extra


def test_a_fixed_block_altitude_sends_no_band():
    extra = stage_env_extra(t3(), is_rehearse=False, merge_alt_ft=22000.0)
    assert extra["alt_ft"] == 22000.0
    assert "alt_ft_max" not in extra


# --- the merge LATERAL SEPARATION (curriculum-92) ------------------------------------------------


def test_lateral_separation_is_shielded_from_the_anchor():
    """The anchor is a 700 m tail-chase drill; ±1200 m of cross-range there is a different fight,
    not "the chase with an offset". Same defect shape as the --me-dalt-ft leak."""
    extra = stage_env_extra(t3(separation_m=700.0), is_rehearse=True, merge_lateral_m=1200.0)
    assert "lateral_m_max" not in extra


def test_lateral_separation_is_independent_of_the_altitude_knobs():
    """Cross-range room, the shared block altitude and the learner's perch are three separate
    quantities; c92 changes exactly one of them off c90's argv."""
    extra = stage_env_extra(
        t3(), is_rehearse=False, me_dalt_ft=2000.0, merge_alt_ft=15000.0, merge_lateral_m=1200.0
    )
    assert extra["me_dalt_ft"] == 2000.0
    assert extra["alt_ft"] == 15000.0
    assert extra["lateral_m_max"] == 1200.0


def test_lateral_separation_is_forwarded_as_a_bare_half_width():
    """It is a HALF-WIDTH, not a floor/ceiling pair: the scenario draws U(−L, +L) itself, so there
    is deliberately only one flag and the band's centre is the legacy 0."""
    extra = stage_env_extra(t3(), is_rehearse=False, merge_lateral_m=1200.0)
    assert extra["lateral_m_max"] == 1200.0
    assert "lateral_m" not in extra


# --- the tier/posture guards: tier-1/2 factories reject unknown kwargs --------------------------


@pytest.mark.parametrize("tier", [1, 2])
def test_non_tier3_stages_get_nothing(tier):
    """`posture` itself is a tier-3-only kwarg, so a tier-1/2 rung must come back empty."""
    stage = t3(tier=tier)
    assert stage_env_extra(stage, is_rehearse=False, me_dalt_ft=2000.0, merge_alt_ft=30000.0) == {}


def test_defensive_posture_drops_the_offensive_only_knobs():
    """tier1_scenario (the defensive builder) accepts neither the bearing nor either altitude knob."""
    stage = t3(posture="defensive")
    extra = stage_env_extra(stage, is_rehearse=False, me_dalt_ft=2000.0, merge_alt_ft=30000.0)
    assert extra == {"posture": "defensive"}


def test_defensive_posture_still_forwards_the_opponent_tier_swap():
    """curriculum-20's weaker-AI swap is posture-independent — it must survive the early return."""
    stage = t3(posture="defensive", opponent_tier=2)
    assert stage_env_extra(stage, is_rehearse=False) == {"posture": "defensive", "opponent_tier": 2}


# --- the optional per-stage knobs ---------------------------------------------------------------


def test_optional_stage_knobs_are_omitted_at_their_sentinels():
    """aspect_deg_max −1, opp_heading_deg None and opp_health None all mean "don't pass it"."""
    extra = stage_env_extra(
        t3(aspect_deg_max=-1.0, opp_heading_deg=None, opp_health=None), is_rehearse=False
    )
    assert "aspect_deg_max" not in extra
    assert "opp_heading_deg" not in extra
    assert "opp_health" not in extra


def test_head_on_merge_geometry_is_forwarded():
    """The head-on merge IS aspect 0 + opp_heading 180 — the geometry the whole program turns on."""
    extra = stage_env_extra(
        t3(aspect_deg=0.0, opp_heading_deg=180.0), is_rehearse=False, merge_alt_ft=30000.0
    )
    assert extra["aspect_deg"] == 0.0
    assert extra["opp_heading_deg"] == 180.0
    assert extra["alt_ft"] == 30000.0


def test_aspect_deg_max_zero_is_forwarded_not_treated_as_a_sentinel():
    """The sentinel is negative, so a band pinned at [0, 0] must still be passed through."""
    assert stage_env_extra(t3(aspect_deg_max=0.0), is_rehearse=False)["aspect_deg_max"] == 0.0


def test_opp_health_zero_is_forwarded():
    """None is the sentinel, so an explicit 0 HP must not be silently dropped."""
    assert stage_env_extra(t3(opp_health=0.0), is_rehearse=False)["opp_health"] == 0.0
