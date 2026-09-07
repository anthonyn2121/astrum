"""γ (discount) ramp for warm-started runs (train_parallel).

Raising `--discount` on a warm champion as a STEP change collapses it: curriculum-88 went
0.995 → 0.9985 in one jump and fell monotonically (mean undiscounted return +59/+58/−42/−91/
−287/−463 per 100k steps), and c88b's gentler 0.998 traced the same curve ~40% slower. The
cause is arithmetic, not exploration: 1/(1−γ) rescales every bootstrapped value target
(200 → 667), so a critic restored at the old γ is badly miscalibrated at the new one and the
policy, which only maximises Q, follows it down.

`ramped_discount` is the fix — walk γ from the checkpoint's value to the target across the
run so the critic re-fits as it moves. These pin the properties the learner loop relies on:
endpoint exactness, monotonicity, clamping past the span, and the 1000-step quantisation that
keeps a `--compile`d update from retracing on every step.
"""

from __future__ import annotations

import numpy as np

from astrum_rl.tdmpc2.train_parallel import ramped_discount

SPAN = 1_000_000.0
START = 0.995
TARGET = 0.998


def test_endpoints_are_exact():
    # A warm run must begin at EXACTLY the checkpoint's γ — starting even slightly above it is
    # the miscalibration c88 died of — and must actually reach the target, not asymptote near it.
    assert ramped_discount(0, SPAN, START, TARGET) == START
    assert ramped_discount(int(SPAN), SPAN, START, TARGET) == TARGET


def test_clamped_past_the_span():
    # The ramp ends but the run does not: every step after `span` holds the target.
    for done in (int(SPAN) + 1, int(SPAN) * 2, 10**9):
        assert ramped_discount(done, SPAN, START, TARGET) == TARGET
    # Negative/zero progress cannot undershoot the checkpoint's γ either.
    assert ramped_discount(-5, SPAN, START, TARGET) == START


def test_monotone_and_bounded():
    prev = -1.0
    for done in range(0, int(SPAN) + 1, 1_000):
        g = ramped_discount(done, SPAN, START, TARGET)
        assert g >= prev, "γ must never step backwards; the critic tracks it in one direction"
        assert START <= g <= TARGET
        prev = g


def test_midpoint_is_linear_in_credit_horizon_terms():
    mid = ramped_discount(int(SPAN // 2), SPAN, START, TARGET)
    assert np.isclose(mid, 0.5 * (START + TARGET))
    # The point of the ramp is that the horizon it implies also moves gradually: 200 → 400 → 500
    # steps, not 200 → 500 in one update.
    assert 1.0 / (1.0 - START) < 1.0 / (1.0 - mid) < 1.0 / (1.0 - TARGET)


def test_quantised_to_1000_increments():
    # `agent.update` may be torch.compile'd, and it closes over `discount` as a Python float, so a
    # brand-new value every step would retrace forever. Steps within the same 1/1000 bucket of the
    # span must return the BIT-IDENTICAL float, and the whole ramp must span few distinct values.
    bucket = SPAN / 1000.0
    base = ramped_discount(0, SPAN, START, TARGET)
    for done in range(0, int(bucket // 2), 37):
        assert ramped_discount(done, SPAN, START, TARGET) == base

    seen = {ramped_discount(d, SPAN, START, TARGET) for d in range(0, int(SPAN) + 1, 97)}
    assert len(seen) <= 1001


def test_downward_ramp_is_supported():
    # Nothing in the helper assumes target > start; lowering γ on a warm checkpoint rescales the
    # value targets just as violently in the other direction and deserves the same treatment.
    assert ramped_discount(0, SPAN, 0.998, 0.995) == 0.998
    assert ramped_discount(int(SPAN), SPAN, 0.998, 0.995) == 0.995
    assert 0.995 < ramped_discount(int(SPAN // 2), SPAN, 0.998, 0.995) < 0.998


def test_equal_endpoints_are_flat():
    # `--discount-ramp-steps` with a target equal to the checkpoint's γ is a no-op, not a divide
    # or a drift (run_parallel also detects this and prints a no-op notice instead of ramping).
    for done in (0, 1, int(SPAN // 3), int(SPAN), int(SPAN) * 3):
        assert ramped_discount(done, SPAN, START, START) == START
