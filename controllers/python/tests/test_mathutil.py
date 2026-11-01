import math

import pytest
from astrum_client.mathutil import (
    MAX_INTEGRATION_STEP_S,
    clamp,
    integrate_conditionally,
    integration_step,
    wrap_pi,
)


@pytest.mark.parametrize(
    "a, expected",
    [
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (math.pi / 2, math.pi / 2),
        (-math.pi / 2, -math.pi / 2),
        (2.0 * math.pi, 0.0),
        (7.0 * math.pi / 4, -math.pi / 4),
        (-7.0 * math.pi / 4, math.pi / 4),
    ],
)
def test_wrap_pi_values(a, expected):
    assert wrap_pi(a) == pytest.approx(expected, abs=1e-12)


def test_wrap_pi_range_is_half_open():
    tiny = math.ulp(math.pi)
    samples = [k * 0.37 for k in range(-2000, 2000)]
    samples += [1e6, -1e6, 1e15, -1e15, math.pi - tiny, -math.pi - tiny, -1e-300, 1e-300]
    for a in samples:
        w = wrap_pi(a)
        assert -math.pi <= w < math.pi, a
        # Same angle: differs from the input by a whole number of turns.
        turns = (a - w) / (2.0 * math.pi)
        assert turns == pytest.approx(round(turns), abs=1e-6 * max(1.0, abs(a))), a


@pytest.mark.parametrize("a", [math.nan, math.inf, -math.inf])
def test_wrap_pi_non_finite_is_nan(a):
    assert math.isnan(wrap_pi(a))


def test_clamp():
    assert clamp(2.0, -1.0, 1.0) == 1.0
    assert clamp(-2.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5
    assert math.isnan(clamp(math.nan, -1.0, 1.0))


def test_integration_step_skips_first_repeat_and_gaps():
    assert integration_step(None, 1.0) == 0.0
    assert integration_step(1.0, 1.0) == 0.0
    assert integration_step(1.0, 0.5) == 0.0
    assert integration_step(1.0, 1.0 + MAX_INTEGRATION_STEP_S + 0.01) == 0.0
    assert integration_step(1.0, 1.25) == pytest.approx(0.25)


def test_conditional_integration_freezes_into_saturation_only():
    # Output saturated high: a positive step is frozen, a negative one integrates.
    assert integrate_conditionally(0.1, 1.0, 0.1, 2.0, 0.0, 1.0, 5.0) == 0.1
    assert integrate_conditionally(0.1, -1.0, 0.1, 2.0, 0.0, 1.0, 5.0) == pytest.approx(0.0)
    # Saturated low: mirror image.
    assert integrate_conditionally(0.1, -1.0, 0.1, -2.0, 0.0, 1.0, 5.0) == 0.1
    # Unsaturated: integrates but never beyond the bound.
    assert integrate_conditionally(0.95, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0) == 1.0
