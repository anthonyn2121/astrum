"""Scenario TOML (v2) builders for the dogfight adapter (design doc §6, §8.5).

The env takes a scenario as a TOML string, so training code can synthesize starts
without touching `config/`. Units follow scenario v2: heights above the WGS-84
ellipsoid (m) and TRUE airspeed (m/s).

The default start (4572 m HAE, 222 m/s TAS) is the former 15 000 ft / 350 KCAS
start converted intentionally: 350 KCAS at 15 000 ft ISA is Mach 0.688 ≈ 222 m/s
TAS (431 kt).
"""

from __future__ import annotations

# SF-Bay datum (matches the other scenarios / demos).
_ORIGIN_LAT = 37.6188
_ORIGIN_LON = -122.3756
_M_PER_DEG_LAT = 111_320.0

DEFAULT_ALT_HAE_M = 4572.0
DEFAULT_TAS_MPS = 222.0


def _header(seed: int, rate_hz: float, max_time_s: float) -> str:
    return f"""
[world]
origin_lat_deg = {_ORIGIN_LAT}
origin_lon_deg = {_ORIGIN_LON}
origin_alt_hae_m = 0.0

[sim]
rate_hz = {rate_hz}
max_time_s = {max_time_s}
realtime = false
seed = {seed}
"""


def _aircraft(id: int, team: int, lat: float, alt_hae_m: float, tas_mps: float) -> str:
    return f"""
[[aircraft]]
id = {id}
model = "f16"
lat_deg = {lat:.7f}
lon_deg = {_ORIGIN_LON}
alt_hae_m = {alt_hae_m}
heading_deg = 0.0
tas_mps = {tas_mps}
team = {team}
"""


def _pair(
    offset_m: float,
    *,
    seed: int,
    rate_hz: float,
    max_time_s: float,
    alt_hae_m: float,
    tas_mps: float,
    learner_id: int,
    opponent_id: int,
) -> str:
    """Learner (team 0) at the origin, opponent (team 1) `offset_m` north of it;
    both heading north at the same height and speed."""
    opp_lat = _ORIGIN_LAT + offset_m / _M_PER_DEG_LAT
    return (
        _header(seed, rate_hz, max_time_s)
        + _aircraft(learner_id, 0, _ORIGIN_LAT, alt_hae_m, tas_mps)
        + _aircraft(opponent_id, 1, opp_lat, alt_hae_m, tas_mps)
    )


def tier1_scenario(
    *,
    seed: int = 7,
    rate_hz: float = 120.0,
    max_time_s: float = 60.0,
    alt_hae_m: float = DEFAULT_ALT_HAE_M,
    tas_mps: float = DEFAULT_TAS_MPS,
    separation_m: float = 2000.0,
    learner_id: int = 1,
    opponent_id: int = 2,
) -> str:
    """The learner `separation_m` behind (south of) the opponent — a clean
    offensive-intercept setup on the opponent's tail."""
    return _pair(
        separation_m,
        seed=seed,
        rate_hz=rate_hz,
        max_time_s=max_time_s,
        alt_hae_m=alt_hae_m,
        tas_mps=tas_mps,
        learner_id=learner_id,
        opponent_id=opponent_id,
    )


def tier3_scenario(
    *,
    seed: int = 11,
    rate_hz: float = 120.0,
    max_time_s: float = 60.0,
    alt_hae_m: float = DEFAULT_ALT_HAE_M,
    tas_mps: float = DEFAULT_TAS_MPS,
    separation_m: float = 700.0,
    learner_id: int = 1,
    opponent_id: int = 2,
) -> str:
    """The opponent `separation_m` behind (south of) the learner — the bot opens
    on the learner's six in lead pursuit (mirror of the Tier-1/2 setup)."""
    return _pair(
        -separation_m,
        seed=seed,
        rate_hz=rate_hz,
        max_time_s=max_time_s,
        alt_hae_m=alt_hae_m,
        tas_mps=tas_mps,
        learner_id=learner_id,
        opponent_id=opponent_id,
    )
