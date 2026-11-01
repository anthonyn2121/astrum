#!/usr/bin/env python3
"""Smoke test for the `astrum_env` binding (run by `make test-python`).

Exercises the PyO3 seam end-to-end: construct → reset(seed) → step (raw
controls + trigger, autopilot targets) → gun_solution → body/grid rotations →
BatchEnv, asserting the invariants a faithful marshalling layer must keep.

    PYTHONPATH="$(bash scripts/build-astrum-env.sh):crates/astrum-env/python" \
        python crates/astrum-env/python/smoke.py
"""

import astrum_env
import numpy as np

# Two C172s on opposing teams over SF Bay, ~22 m apart north/south, both heading
# east, so neither is in the other's gun line.
SCENARIO = """
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_hae_m = 0.0

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
alt_hae_m = 1524.0
heading_deg = 90.0
tas_mps = 55.0
team = 0

[[aircraft]]
id = 2
model = "c172x"
lat_deg = 37.6190
lon_deg = -122.3756
alt_hae_m = 1524.0
heading_deg = 90.0
tas_mps = 55.0
team = 1
"""

COL = {n: i for i, n in enumerate(astrum_env.OBS_FIELDS)}


def main() -> None:
    env = astrum_env.Env(SCENARIO)
    assert env.agent_ids == [1, 2], env.agent_ids
    assert abs(env.dt - 1.0 / 120.0) < 1e-12, env.dt

    obs = env.reset(seed=3)
    assert obs.shape == (2, len(astrum_env.OBS_FIELDS)) and obs.dtype == np.float64
    assert np.all(np.isfinite(obs))
    assert np.all(obs[:, COL["tick"]] == 1), "first obs after reset is tick 1"
    assert np.all(obs[:, COL["alive"]] == 1)
    assert np.all(obs[:, COL["ammo"]] == astrum_env.GUN_AMMO)
    assert np.all(np.abs(obs[:, COL["tas"]] - 55.0) < 2.0), obs[:, COL["tas"]]
    assert np.all(np.abs(obs[:, COL["alt_hae_m"]] - 1524.0) < 5.0)

    # Agent 1 holds the trigger with a right roll; agent 2 flies the sim autopilot.
    act = np.array([[0.2, -0.05, 0.0, 0.7, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    hold = obs[1, COL["psi"]], obs[1, COL["alt_hae_m"]], 55.0
    ap = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, *hold]])
    for _ in range(40):
        obs, info = env.step(act, autopilot=ap)
    assert obs[0, COL["ammo"]] < astrum_env.GUN_AMMO, "a held trigger fires rounds"
    assert obs[1, COL["ammo"]] == astrum_env.GUN_AMMO, "a released trigger fires nothing"
    assert obs[0, COL["p"]] > 0.1, f"right aileron rolls right, p={obs[0, COL['p']]:.3f}"
    modes = [astrum_env.CONTROL_MODES[int(m)] for m in obs[:, COL["control_mode"]]]
    assert modes == ["raw", "autopilot"], modes
    assert not info.done() and info.tick == 41, info
    assert np.array_equal(env.observe(), obs)

    sol = env.gun_solution(1, 2)
    assert set(sol) == {"miss_m", "tof_s", "hit", "aim_dir", "aim_error_rad", "in_range"}
    assert abs(np.linalg.norm(sol["aim_dir"]) - 1.0) < 1e-9

    # Rotations are inverses and preserve length; nose · velocity ≈ cos(α)cos(β) · |v|.
    v = [3.0, -1.0, 2.0]
    assert np.allclose(env.grid_to_body(1, env.body_to_grid(1, v)), v, atol=1e-9)
    nose = np.array(env.body_to_grid(1, [1.0, 0.0, 0.0]))
    vel = obs[0, COL["vel_e"] : COL["vel_u"] + 1]
    assert np.dot(nose, vel) / np.linalg.norm(vel) > 0.9

    # Same seed ⇒ identical episode.
    first = env.reset(seed=3)
    for _ in range(40):
        a, _ = env.step(act, autopilot=ap)
    again = env.reset(seed=3)
    for _ in range(40):
        b, _ = env.step(act, autopilot=ap)
    assert np.array_equal(first, again) and np.array_equal(a, b)

    # BatchEnv: K envs stepped together match the single env bit-for-bit.
    batch = astrum_env.BatchEnv([SCENARIO] * 3)
    assert len(batch) == 3 and batch.agent_ids == [[1, 2]] * 3
    bobs = batch.reset(seeds=[3, 3, 3])
    assert bobs.shape == (3, 2, len(astrum_env.OBS_FIELDS))
    acts = np.broadcast_to(act, (3, 2, 5)).copy()
    aps = np.broadcast_to(ap, (3, 2, 4)).copy()
    for _ in range(40):
        bobs, term, trunc, tick = batch.step(acts, autopilot=aps)
    assert not term.any() and not trunc.any() and np.all(tick == 41)
    for k in range(3):
        assert np.array_equal(bobs[k], b), f"batch env {k} diverged from Env"

    print(f"OK — astrum_env binding: {len(env.agent_ids)} agents, Env + BatchEnv faithful.")


if __name__ == "__main__":
    main()
