"""Guidance helpers for the autopilot laws (``figure8``, ``formation``).

Both laws plan in the scenario's ENU *grid* (positions ``e/n``, velocities
``vel_enu_*``) but the sim's autopilot takes a *true* heading. The two differ
by the grid convergence at the aircraft, which each telemetry sample carries
implicitly: the same velocity is given in the grid (``velEnu``) and in the
local level (``velNed``).
"""

from __future__ import annotations

import math

from astrum_client.mathutil import wrap_pi
from astrum_client.wire import VehicleState

# Below this horizontal speed the velocity direction (and so the convergence
# estimate) is noise.
_MIN_TRACK_SPEED_MPS = 1.0


def grid_convergence(s: VehicleState) -> float:
    """True-north minus grid-north direction at the aircraft (rad)."""
    if math.hypot(s.vel_enu_e, s.vel_enu_n) < _MIN_TRACK_SPEED_MPS:
        return 0.0
    true_track = math.atan2(s.vel_ned_e, s.vel_ned_n)
    grid_track = math.atan2(s.vel_enu_e, s.vel_enu_n)
    return wrap_pi(true_track - grid_track)


def grid_track(s: VehicleState) -> float:
    """Direction of the ground velocity in the grid (rad, 0 = grid north, +east)."""
    return math.atan2(s.vel_enu_e, s.vel_enu_n)


def true_heading_for_grid_track(s: VehicleState, track_grid: float) -> float:
    """Autopilot heading target that flies the grid-frame ground track
    ``track_grid``. Assumes NO WIND and coordinated flight (no sideslip), where
    heading equals track; with wind the aircraft would crab and drift."""
    return wrap_pi(track_grid + grid_convergence(s))
