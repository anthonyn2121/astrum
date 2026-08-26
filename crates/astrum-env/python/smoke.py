#!/usr/bin/env python3
"""Smoke test for the `astrum_env` embedded RL environment.

Proves the pyo3 seam end-to-end from Python: build → reset → observe → step
(continuous controls + the learned trigger) → episode flags. It asserts the same
invariants the Rust integration test (`astrum-sim/tests/env.rs`) does, so a green
run here means the FFI marshalling is faithful.

Run it via the crate's build helper (which compiles the .so and puts it on the
path):  bash crates/astrum-env/python/run_smoke.sh
"""

import astrum_env

# A minimal two-team engagement (two C172s on opposing teams over SF Bay).
SCENARIO = """
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_m = 0.0

[sim]
rate_hz = 120.0
max_time_s = 0.0
realtime = false
seed = 7

[[aircraft]]
id = 1
model = "c172x"
lat_deg = 37.6188
lon_deg = -122.3756
alt_ft = 5000.0
heading_deg = 90.0
airspeed_kts = 100.0
team = 0

[[aircraft]]
id = 2
model = "c172x"
lat_deg = 37.6190
lon_deg = -122.3756
alt_ft = 5000.0
heading_deg = 270.0
airspeed_kts = 100.0
team = 1
"""


def main() -> None:
    env = astrum_env.DogfightEnv(SCENARIO)

    ids = env.agent_ids()
    assert ids == [1, 2], f"expected agents [1, 2], got {ids}"

    obs = env.reset()
    assert len(obs) == 2, f"expected 2 observations, got {len(obs)}"
    for o in obs:
        assert o.alive, f"aircraft {o.id} should start alive"
        assert abs(o.health - 100.0) < 1e-3, f"aircraft {o.id} should start at full HP"
        assert o.ammo == 511, f"aircraft {o.id} should start armed, got {o.ammo}"
        assert o.airspeed_mps > 0.0, f"aircraft {o.id} should be trimmed and moving"
        # α/β/Nz (design doc §8 obs): finite, and trimmed flight sits near 1 g.
        assert abs(o.nz_g) < 10.0, f"aircraft {o.id} Nz should be ~1 g, got {o.nz_g}"
        assert o.alpha_rad == o.alpha_rad, f"aircraft {o.id} alpha must be finite"
    print("reset:", *(repr(o) for o in obs), sep="\n  ")

    # Agent 1 rolls hard and holds the trigger; agent 2 flies neutral, no trigger.
    roll_and_fire = [0.2, -0.05, 0.0, 0.7, 1.0]
    neutral = [0.0, 0.0, 0.0, 0.7, 0.0]
    info = None
    for _ in range(40):
        obs, info = env.step([(1, roll_and_fire), (2, neutral)])

    by_id = {o.id: o for o in obs}
    assert by_id[1].ammo < 511, "a held trigger must fire rounds"
    assert by_id[2].ammo == 511, "a released trigger fires nothing"
    assert abs(by_id[1].p) > 0.3, f"full aileron should roll, got p={by_id[1].p:.3f}"
    assert not info.terminated and not info.truncated, "40 ticks shouldn't end this episode"
    print("after 40 ticks:", *(repr(o) for o in obs), sep="\n  ")
    print("info:", repr(info))

    # reset() clears state for a fresh episode.
    obs2 = env.reset()
    assert all(o.alive for o in obs2) and all(o.ammo == 511 for o in obs2)
    print(f"\nOK — pyo3 env seam works: {len(ids)} agents, reset/step/observe faithful.")


if __name__ == "__main__":
    main()
