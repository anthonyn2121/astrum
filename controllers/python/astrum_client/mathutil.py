"""Angle, saturation and integrator helpers shared by every ASTRUM Python controller."""

import math

TWO_PI = 2.0 * math.pi

# Integrate over at most this long a gap between samples; a longer gap (dropped
# telemetry, a paused sim) is not evidence of a persistent error.
MAX_INTEGRATION_STEP_S = 0.5


def wrap_pi(a: float) -> float:
    """Wrap an angle (rad) into ``[-pi, pi)``.

    ``pi`` maps to ``-pi``. A non-finite input (NaN, +/-inf) has no meaningful
    wrapped value and returns NaN, so a corrupt angle propagates visibly instead
    of turning into a plausible-looking number.
    """
    if not math.isfinite(a):
        return math.nan
    w = math.fmod(a + math.pi, TWO_PI)
    if w < 0.0:
        w += TWO_PI
    if w >= TWO_PI:  # a tiny negative fmod result rounds up to exactly 2*pi
        w = 0.0
    return w - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    """Saturate ``x`` to ``[lo, hi]``; NaN passes through unchanged."""
    if x != x:
        return x
    return lo if x < lo else hi if x > hi else x


def integration_step(prev_t: float | None, t: float) -> float:
    """Sim-time step to integrate over between two samples: 0 on the first
    sample, a repeated/backwards time, or a gap longer than
    :data:`MAX_INTEGRATION_STEP_S`."""
    if prev_t is None:
        return 0.0
    dt = t - prev_t
    return dt if 0.0 < dt <= MAX_INTEGRATION_STEP_S else 0.0


def integrate_conditionally(
    integ: float, rate: float, dt: float, unsaturated: float, lo: float, hi: float, bound: float
) -> float:
    """One anti-windup integrator step: ``integ + rate*dt`` unless the output the
    integrator feeds (``unsaturated``, before clamping to ``[lo, hi]``) is already
    saturated and the step would push it further; the result stays in ``±bound``.
    ``integ`` and ``rate`` are in the units of that output."""
    step = rate * dt
    if (unsaturated >= hi and step > 0.0) or (unsaturated <= lo and step < 0.0):
        return integ
    return clamp(integ + step, -bound, bound)
