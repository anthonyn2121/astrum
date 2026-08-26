"""Scenario TOML builders for the dogfight adapter (design doc §6, §8.5).

The env takes a scenario as a TOML string, so training code can synthesize starts
without touching `config/`. Tier-1 = one learner behind a straight-and-level f16
dummy over SF Bay.
"""

from __future__ import annotations

# SF-Bay datum (matches the other scenarios / demos).
_ORIGIN_LAT = 37.6188
_ORIGIN_LON = -122.3756
_M_PER_DEG_LAT = 111_320.0


def _aircraft(
    id: int, team: int, lat: float, lon: float, alt_ft: float, hdg: float, kts: float
) -> str:
    return f"""
[[aircraft]]
id = {id}
model = "f16"
lat_deg = {lat:.6f}
lon_deg = {lon:.6f}
alt_ft = {alt_ft}
heading_deg = {hdg}
airspeed_kts = {kts}
team = {team}
"""


def tier1_scenario(
    *,
    seed: int = 7,
    rate_hz: float = 120.0,
    max_time_s: float = 60.0,
    alt_ft: float = 15000.0,
    airspeed_kts: float = 350.0,
    separation_m: float = 2000.0,
    learner_id: int = 1,
    opponent_id: int = 2,
) -> str:
    """Two f16: the learner (team 0) behind a Tier-1 dummy (team 1) flying north.

    The dummy starts `separation_m` north of the learner on the same heading, so
    the learner begins roughly on its tail — a clean offensive-intercept setup.
    """
    opp_lat = _ORIGIN_LAT + separation_m / _M_PER_DEG_LAT
    return (
        f"""
[world]
origin_lat_deg = {_ORIGIN_LAT}
origin_lon_deg = {_ORIGIN_LON}
origin_alt_m = 0.0

[sim]
rate_hz = {rate_hz}
max_time_s = {max_time_s}
realtime = false
seed = {seed}
"""
        + _aircraft(learner_id, 0, _ORIGIN_LAT, _ORIGIN_LON, alt_ft, 0.0, airspeed_kts)
        + _aircraft(opponent_id, 1, opp_lat, _ORIGIN_LON, alt_ft, 0.0, airspeed_kts)
    )


def tier3_scenario(
    *,
    seed: int = 11,
    rate_hz: float = 120.0,
    max_time_s: float = 60.0,
    alt_ft: float = 15000.0,
    airspeed_kts: float = 350.0,
    separation_m: float = 700.0,
    learner_id: int = 1,
    opponent_id: int = 2,
) -> str:
    """Two f16 with the *opponent* on the learner's six (offensive setup).

    The Tier-3 bot (team 1) starts `separation_m` behind (south of) the learner
    (team 0), both heading north, so its FSM opens in lead pursuit and can take gun
    shots — the mirror of the Tier-1/2 setup, which puts the learner on the bot's tail.
    """
    opp_lat = _ORIGIN_LAT - separation_m / _M_PER_DEG_LAT
    return (
        f"""
[world]
origin_lat_deg = {_ORIGIN_LAT}
origin_lon_deg = {_ORIGIN_LON}
origin_alt_m = 0.0

[sim]
rate_hz = {rate_hz}
max_time_s = {max_time_s}
realtime = false
seed = {seed}
"""
        + _aircraft(learner_id, 0, _ORIGIN_LAT, _ORIGIN_LON, alt_ft, 0.0, airspeed_kts)
        + _aircraft(opponent_id, 1, opp_lat, _ORIGIN_LON, alt_ft, 0.0, airspeed_kts)
    )
