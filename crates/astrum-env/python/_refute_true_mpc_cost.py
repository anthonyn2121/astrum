"""Cost + determinism precheck for the proposed true-sim CEM harness (_diag_true_mpc.py).

Measures (a) raw sim throughput in sim-seconds per wall-second for a single process,
(b) env rebuild cost, (c) whether an action PREFIX replay reproduces state bit-exactly
on the tier-3 merge scenario (the property the whole harness rests on).
"""
import sys
import time

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401

import numpy as np

import astrum_env
from astrum_dogfight.scenario import tier3_scenario

RATE = 120.0  # sim ticks per second (action_repeat 5 @ 24 Hz control)


def build():
    return astrum_env.DogfightEnv(tier3_scenario())


t0 = time.perf_counter()
e = build()
e.reset()
t1 = time.perf_counter()
print(f"COLD build+reset: {(t1 - t0) * 1e3:.1f} ms")

t0 = time.perf_counter()
for _ in range(5):
    ee = build()
    ee.reset()
t1 = time.perf_counter()
build_ms = (t1 - t0) / 5 * 1e3
print(f"WARM build+reset avg: {build_ms:.1f} ms")

ids = e.agent_ids()
me, opp = ids[0], ids[1]
act = [(me, [0.2, -0.4, 0.0, 1.0, 0.0]), (opp, [0.0, -0.1, 0.0, 0.8, 0.0])]

N = 2400  # 20 s of sim at 120 Hz
e.reset()
t0 = time.perf_counter()
for _ in range(N):
    e.step(act)
t1 = time.perf_counter()
wall = t1 - t0
simsec = N / RATE
print(f"STEP: {N} ticks = {simsec:.1f} sim-s in {wall:.2f} wall-s "
      f"=> {N / wall:.0f} ticks/s = {simsec / wall:.2f}x realtime, "
      f"{wall / simsec * 1e3:.2f} ms wall per sim-second")

# --- determinism / prefix replay -------------------------------------------------
def run_prefix(n, seedmix=0):
    env = build()
    obs = env.reset()
    last = None
    for i in range(n):
        a = [(me, [np.sin(i * 0.01), -0.5 + 0.3 * np.cos(i * 0.013), 0.0, 1.0, 0.0]),
             (opp, [0.0, 0.0, 0.0, 0.9, 0.0])]
        last, _info = env.step(a)
    return last


a1 = run_prefix(600)
a2 = run_prefix(600)


def vec(obs_list):
    out = []
    for o in obs_list:
        for f in ("pos", "vel"):
            v = getattr(o, f, None)
            if v is not None:
                out.extend(list(np.asarray(v, dtype=np.float64).ravel()))
        for f in ("airspeed", "health", "ammo", "phi", "theta", "psi", "alpha"):
            v = getattr(o, f, None)
            if v is not None:
                out.append(float(v))
    return np.asarray(out)


v1, v2 = vec(a1), vec(a2)
print(f"determinism: {len(v1)} floats, max |div| = {np.max(np.abs(v1 - v2)):.3e}")

# --- what the proposal's own recipe costs ---------------------------------------
# 3 s lookahead, 0.5 s commit, out to 30 s => 60 replans.
# CEM 4 iters x 32 samples = 128 rollouts per replan.
# NO snapshot API => each rollout must replay the committed prefix from t=0.
ms_per_simsec = wall / simsec * 1e3
tot = 0.0
for k in range(60):
    prefix_s = 0.5 * k
    per_rollout_s = prefix_s + 3.0
    tot += 128 * (per_rollout_s * ms_per_simsec + build_ms)
print(f"\nONE 30 s engagement, one seed, SINGLE process: {tot / 1000:.0f} s "
      f"= {tot / 3.6e6:.2f} h   (sharded over 6: {tot / 6 / 3600:.2f} h)")
print(f"  total simulated flight: "
      f"{sum(128 * (0.5 * k + 3.0) for k in range(60)) / 3600:.1f} sim-hours")
