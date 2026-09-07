"""Scenario TOML builders for the dogfight adapter (design doc §6, §8.5).

The env takes a scenario as a TOML string, so training code can synthesize starts
without touching `config/`. Tier-1 = one learner behind a straight-and-level f16
dummy over SF Bay.
"""

from __future__ import annotations

import math

# SF-Bay datum (matches the other scenarios / demos).
_ORIGIN_LAT = 37.6188
_ORIGIN_LON = -122.3756
_M_PER_DEG_LAT = 111_320.0
# metres per degree of longitude at the datum latitude (for lateral offset jitter).
_M_PER_DEG_LON = _M_PER_DEG_LAT * math.cos(math.radians(_ORIGIN_LAT))


def _jitter_geometry(rng, jitter: float, separation_m: float, airspeed_kts: float) -> dict:
    """Deterministic per-episode start-geometry perturbation (jitter ∈ [0,1] = strength).

    The scripted scenarios are otherwise fixed, so a deterministic policy sees ONE exact
    geometry forever and memorises it (ground-truth eval: the curriculum-13 policy killed
    from its training separation but chip-drew — 28 HP — at a closer range against a sitting
    dummy, and the stochastic-prior gate mistook lucky-noise kills for mastery). Jittering the
    opening range, line-of-sight bearing, opponent heading, altitude and both airspeeds every
    episode forces a policy that generalises across the engagement envelope and makes the
    deterministic-probe gate a real kill-RATE over a distribution rather than one bit. `rng`
    is a numpy Generator (the env advances it per reset, seeding once per worker), so a run is
    reproducible. jitter≤0 (or no rng) → the legacy fixed geometry (existing tests unchanged)."""
    if jitter <= 0.0 or rng is None:
        return {
            "separation_m": separation_m,
            "lateral_m": 0.0,
            "opp_hdg_deg": 0.0,
            "opp_dalt_ft": 0.0,
            "me_kts": airspeed_kts,
            "opp_kts": airspeed_kts,
        }
    j = float(jitter)
    return {
        "separation_m": separation_m * (1.0 + j * 0.18 * rng.uniform(-1.0, 1.0)),
        "lateral_m": j * 0.09 * separation_m * rng.uniform(-1.0, 1.0),  # ±~5° LOS off the nose
        "opp_hdg_deg": j * 15.0 * rng.uniform(-1.0, 1.0),
        "opp_dalt_ft": j * 240.0 * rng.uniform(-1.0, 1.0),
        "me_kts": airspeed_kts + j * 15.0 * rng.uniform(-1.0, 1.0),
        "opp_kts": airspeed_kts + j * 15.0 * rng.uniform(-1.0, 1.0),
    }


def _aircraft(
    id: int,
    team: int,
    lat: float,
    lon: float,
    alt_ft: float,
    hdg: float,
    kts: float,
    health: float | None = None,
) -> str:
    # `health` (curriculum-49): spawn HP for this aircraft. None → the Rust default (100, via
    # AircraftSpec::default_health in astrum-sim/src/scenario.rs). Set on the OPPONENT to drill the
    # re-attack finish: a bandit that spawns "wounded" (e.g. 30 HP) puts the learner in the exact
    # post-first-pass STATE (wounded bandit off the nose, full ammo) every episode, so — kill-gated —
    # it must reverse and finish rather than coast (the ≥55° wall). The sim sets hp_max to the same
    # value (Health::full), and the learner's obs normalises by the constant C.HP_MAX=100, so the
    # bandit reads correctly as wounded (30/100 = 0.3).
    health_line = "" if health is None else f"health = {float(health)}\n"
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
{health_line}"""


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
    jitter: float = 0.0,
    rng=None,
) -> str:
    """Two f16: the learner (team 0) behind a Tier-1 dummy (team 1) flying north.

    The dummy starts `separation_m` north of the learner on the same heading, so
    the learner begins roughly on its tail — a clean offensive-intercept setup.
    `jitter`>0 perturbs the start every episode (see `_jitter_geometry`).
    """
    g = _jitter_geometry(rng, jitter, separation_m, airspeed_kts)
    opp_lat = _ORIGIN_LAT + g["separation_m"] / _M_PER_DEG_LAT
    opp_lon = _ORIGIN_LON + g["lateral_m"] / _M_PER_DEG_LON
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
        + _aircraft(learner_id, 0, _ORIGIN_LAT, _ORIGIN_LON, alt_ft, 0.0, g["me_kts"])
        + _aircraft(
            opponent_id,
            1,
            opp_lat,
            opp_lon,
            alt_ft + g["opp_dalt_ft"],
            g["opp_hdg_deg"],
            g["opp_kts"],
        )
    )


def tier3_scenario(
    *,
    seed: int = 11,
    rate_hz: float = 120.0,
    max_time_s: float = 60.0,
    alt_ft: float = 15000.0,
    alt_ft_max: float | None = None,
    me_dalt_ft: float = 0.0,
    lateral_m_max: float = 0.0,
    airspeed_kts: float = 350.0,
    separation_m: float = 700.0,
    aspect_deg: float = 180.0,
    aspect_deg_max: float | None = None,
    opp_heading_deg: float | None = None,
    opp_health: float | None = None,
    learner_id: int = 1,
    opponent_id: int = 2,
    jitter: float = 0.0,
    rng=None,
) -> str:
    """Two f16 with the *opponent* at bearing `aspect_deg` around the learner.

    Both jets start heading north; the Tier-3 bot (team 1) is placed `separation_m`
    away from the learner (team 0) at compass bearing `aspect_deg` (measured from the
    learner's nose, clockwise). `aspect_deg` is the reversal-difficulty knob for the
    c18 curriculum:
      * 180° (default) → bot due SOUTH = dead six, co-heading → the hard DEFENSIVE
        fight (bot opens in lead pursuit with a gun solution; the learner must reverse
        to score). This exactly reproduces the legacy dead-six `tier3_scenario`.
      * 90° → bot on the BEAM = a near-neutral line-abreast merge.
      * 0° → bot due NORTH = dead ahead → the learner is on its six (OFFENSIVE, the
        c17-mastered geometry). Sweeping 0°→180° slides offensive→neutral→defensive.
    `jitter`>0 perturbs the start every episode (see `_jitter_geometry`); its lateral
    term rides on the cross-range (east) axis as a small line-of-sight wobble.

    `opp_heading_deg` (curriculum-21) sets the bot's ABSOLUTE start heading, overriding the
    co-heading (north) default. Placing the bot dead-ahead (`aspect_deg=0`) and pointing it
    back at the learner (`opp_heading_deg=180`) makes a HEAD-ON MERGE: the pair close at ~2×
    speed and pass, so no overtake is needed to reach guns range (the wall that stalls the
    co-heading offset merge) — vs a Tier-3 that turns to fight, it becomes the canonical
    neutral turning fight where the tail-chase gun skill (carried in by warm-start) applies.
    None (default) keeps the jittered co-heading start. When set, the jitter heading term is
    replaced (range/lateral/alt/speed jitter still apply).

    `aspect_deg_max` (curriculum-27) turns the fixed bearing into a per-episode RANDOM band:
    when set (and an `rng` is supplied — i.e. under jitter), the bearing is sampled uniformly
    from [aspect_deg, aspect_deg_max] every reset. A band anchored at 0 (e.g. [0, 90]) makes the
    learner practise the WHOLE forward hemisphere every stage, so widening the ceiling can no
    longer catastrophically forget the easy tail-chase (the c26 failure: a fixed-aspect ladder
    marched the ceiling up and the final policy had forgotten aspects 0–40). The deterministic
    eval/probe path leaves `aspect_deg_max=None` for a fixed, repeatable per-aspect readout.

    `me_dalt_ft` (altitude-advantage, "perch") raises the LEARNER's spawn altitude by this many feet
    above the opponent's `alt_ft` — a slight ENTRY ENERGY advantage (potential energy convertible to
    sustained-pull / dive-for-closure) so the learner can press into the WEZ against the aggressive
    corner-holding T3 whose 8.2 g break otherwise defeats a co-altitude tracking solution. 0 (default) =
    legacy co-altitude start. Opponent altitude jitter (`opp_dalt_ft`, ±240·jitter ft) still applies on
    top, so the net edge is `me_dalt_ft − opp_dalt_ft`.

    `alt_ft_max` (curriculum-91) turns the fixed BLOCK altitude into a per-episode RANDOM band
    [alt_ft, alt_ft_max], exactly like `aspect_deg_max` does for the bearing — and for the same
    reason. The block altitude is really the fight's altitude BUDGET: a head-on energy fight was
    measured bleeding ~68 m/s, so the 15,000 ft default is only ~63 s of altitude against the 180 s
    merge clock, and 11/12 greedy episodes ended `hard_deck` at a mean 86 s — the FLOOR truncates
    the fight long before a multi-pass conversion can happen. Raising it outright is not safe,
    though: obs feature 2 is `(agl − DECK_AGL)/ALT_REF`, so 15,000 ft reads 0.81 while 30,000 ft
    reads 1.58 — nearly double, and past 1.0 into a range no checkpoint has ever seen (the
    curriculum-79 lesson: an untrained altitude change lands out-of-distribution and backfires).
    A band keeps the familiar altitude in the training distribution while extending coverage
    smoothly, and it makes the policy READ its remaining budget rather than memorise one altitude.
    None (default) = the fixed `alt_ft`, every legacy scenario unchanged. Needs an `rng`
    (i.e. jitter > 0); the deterministic eval/probe path leaves it None for a repeatable readout.

    `lateral_m_max` (curriculum-92) is the LATERAL SEPARATION of the merge — the half-width, in
    metres, of a signed per-episode band added to the cross-range (east) offset. It is the one
    scenario quantity the head-on merge had measurably WRONG BY OMISSION: the only cross-range term
    was `_jitter_geometry`'s ±0.09·jitter·separation wobble = ±135 m at jitter 0.5 / 3000 m, i.e.
    2.6° off the nose. At the measured 225 m/s merge speed an F-16 turn radius is ~634 m at the
    8.2 g the T3 actually pulls, so ±135 m is 0.21 turn radii — not "a small offset" but a
    COLLISION COURSE, in which no one-circle/two-circle decision and no lead turn exist at all. It
    is why c90 spends 14.2 s/episode inside 500 m at ata 128°, aspect 147°, closure ≈ 0 with 0.0%
    of 4100 ticks hittable *or* feasible: a fight with no turning room in it. Measured in-sim, the
    T3 changes regime at ~800–1200 m of offset — at ≤500 m it flies a nose-to-nose 7–8.3 g break
    and passes at HCA 152–165° (a face pass), while at ≥800 m it COMMITS A LEAD TURN (heading
    180°→247/264/277° at 1200/1500/2000 m) and HCA at closest approach collapses to 124/106/96°.
    It erases 41–58% of the offset by then, so budget the room actually delivered at the pass as
    ~0.5·|L|.

    Bounded above by the OPPONENT's own threat gate, which must not move or the benchmark is not
    the benchmark: the T3 goes defensive only when the learner's total off-nose angle is inside
    `THREAT_CONE_RAD` (45°) at `THREAT_RANGE` (2500 m), so the offset must stay under
    √((2500·sin45°)² − dz²) = 1659 m at the +2000 ft perch (dz = 610 m). Past that the T3 never
    breaks and its min range grows past gun range — a materially easier opponent still labelled T3.

    SIGNED, and composed on top of the existing wobble rather than replacing it, so the band spans
    today's familiar ±135 m (the band-not-step pattern that worked for `aspect_deg_max`, and that
    the fixed `me_dalt_ft`/`alt_ft` jumps skipped — both backfired). Signed also because every rung
    in c01→c91 spawned the bandit to the RIGHT, so a one-sided band would let the policy hard-code
    "break right" and score a fake win. Drawn LAST, after every other band, so that with the knob
    off the rng stream is untouched and a c92 episode differs from the same-seed c90 episode ONLY
    by this offset. 0 (default) = legacy. Needs an `rng` (jitter > 0); the deterministic eval/probe
    path leaves it 0 and instead probes a FIXED offset via `aspect_deg` + `separation_m`
    (lateral = separation·sin(aspect)), which is repeatable.
    """
    g = _jitter_geometry(rng, jitter, separation_m, airspeed_kts)
    if opp_heading_deg is not None:
        g["opp_hdg_deg"] = float(opp_heading_deg)
    if aspect_deg_max is not None and rng is not None:
        lo, hi = sorted((float(aspect_deg), float(aspect_deg_max)))
        aspect_deg = float(rng.uniform(lo, hi))
    if alt_ft_max is not None and rng is not None:
        lo, hi = sorted((float(alt_ft), float(alt_ft_max)))
        alt_ft = float(rng.uniform(lo, hi))
    if lateral_m_max and rng is not None:
        g["lateral_m"] += float(lateral_m_max) * rng.uniform(-1.0, 1.0)
    a = math.radians(aspect_deg)
    north_m = g["separation_m"] * math.cos(a)
    east_m = g["separation_m"] * math.sin(a) + g["lateral_m"]
    opp_lat = _ORIGIN_LAT + north_m / _M_PER_DEG_LAT
    opp_lon = _ORIGIN_LON + east_m / _M_PER_DEG_LON
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
        + _aircraft(learner_id, 0, _ORIGIN_LAT, _ORIGIN_LON, alt_ft + me_dalt_ft, 0.0, g["me_kts"])
        + _aircraft(
            opponent_id,
            1,
            opp_lat,
            opp_lon,
            alt_ft + g["opp_dalt_ft"],
            g["opp_hdg_deg"],
            g["opp_kts"],
            opp_health,  # curriculum-49: None = full-HP (legacy); >0 = wounded re-attack drill
        )
    )
