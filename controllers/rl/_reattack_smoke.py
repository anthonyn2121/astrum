"""curriculum-49 smoke: confirm opp_health plumbing spawns a wounded bandit + env steps.

Run inside the dogfight-train container:
  cd /workspace/controllers/rl && \
  PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
  uv run python _reattack_smoke.py
"""
import numpy as np

from astrum_dogfight.scenario import tier3_scenario

# (1) TOML emits exactly one health line, on the opponent only.
toml = tier3_scenario(opp_health=30.0, aspect_deg=0.0)
assert "health = 30.0" in toml, "opponent health line missing from TOML"
assert toml.count("health = ") == 1, f"expected 1 health line, got {toml.count('health = ')}"
print("TOML OK: single opponent health line (learner left at default)")

# legacy default (no opp_health) must emit NO health line -> full-HP, unchanged behaviour.
toml_legacy = tier3_scenario(aspect_deg=0.0)
assert "health = " not in toml_legacy, "legacy scenario unexpectedly emits a health line"
print("TOML OK: legacy (opp_health=None) emits no health line")

# (2) Build the actual training env for reattack-wounded stage 0 and check the spawned HP.
from astrum_rl.envs import make_tdmpc_env

env = make_tdmpc_env(
    3,
    posture="offensive",
    aspect_deg=0.0,
    aspect_deg_max=90.0,
    opp_health=30.0,
    separation_m=900.0,
    jitter=0.5,
    jitter_seed=1,
)
obs, info = env.reset()
base = env.unwrapped
opp = base._states[base.opponent_id]
lnr = base._states[base.learner_id]
print(f"SPAWN: opp.health={opp.health:.1f}  lnr.health={lnr.health:.1f}")
assert abs(opp.health - 30.0) < 1e-3, f"bandit not wounded: {opp.health}"
assert abs(lnr.health - 100.0) < 1e-3, f"learner not full HP: {lnr.health}"
print("ENV OK: bandit spawns wounded 30, learner full 100")

# obs must normalise the bandit HP by C.HP_MAX=100 -> ~0.3 somewhere in the vector.
import astrum_dogfight.constants as C
assert any(abs(float(v) - 30.0 / C.HP_MAX) < 1e-2 for v in np.asarray(obs).ravel()), \
    "no obs feature ~0.30 (wounded-bandit HP) found"
print(f"OBS OK: a feature reads ~{30.0 / C.HP_MAX:.2f} (bandit HP normalised by HP_MAX={C.HP_MAX})")

# (3) a handful of steps run without exception.
for _ in range(20):
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
    if term or trunc:
        break
print("STEP OK: rollout ran, no exception")
print("REATTACK_SMOKE_OK")
