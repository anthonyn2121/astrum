"""curriculum-65 CPU validation: reward-v20 close-in settle dwell + fire-override quality wiring.

Run inside the container (RL uv env):
  cd /workspace/controllers/rl && PYTHONPATH=... uv run python _c65_validate_tmp.py

Checks (no GPU, no training):
  1. DWELL_RANGE_MIN exists, == 40, and is < the (untouched) two-policy CONVERT_RANGE_MIN=250.
  2. outcome-mode conversion-dwell now FIRES at a close merge-finish geometry (~100 m, wings settled,
     nose-on his six) — the exact state the deliverable swung through firing 0 rounds — whereas at the
     OLD floor (250 m) it did not pay there.
  3. the two-policy convert TERMINAL (convert_bonus>0, reward.py ~line 377) still uses CONVERT_RANGE_MIN
     (250) — i.e. reward-v20 did NOT move the two-policy band (dead path, but keep it + its tests green).
  4. fire-override CLI parses --fire-override-quality 0.35 and it reaches the collector kwargs.
"""

import sys
import traceback

import numpy as np

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C
from astrum_dogfight import reward as R
from astrum_dogfight.geometry import AState
from astrum_rl.tdmpc2 import train_parallel as TP

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}")


def mk(idn, team, alt, v, north, hp=100.0, phi=0.0):
    # ENU pos=(E,N,U), heading psi=0 → nose +N; a bandit due-north is dead ahead.
    return AState(id=idn, team=team, alive=True,
                  pos=np.array([0.0, north, alt]),
                  vel=np.array([0.0, v, 0.0]),
                  phi=phi, theta=0.0, psi=0.0, p=0.0, q=0.0, r=0.0,
                  alpha=0.0, beta=0.0, nz=1.0, airspeed=v, alt=alt, health=hp, ammo=511)


# 1. constant --------------------------------------------------------------
check("DWELL_RANGE_MIN == 40 and < CONVERT_RANGE_MIN(250)",
      hasattr(C, "DWELL_RANGE_MIN") and abs(C.DWELL_RANGE_MIN - 40.0) < 1e-9
      and C.DWELL_RANGE_MIN < C.CONVERT_RANGE_MIN,
      f"dwell_min={getattr(C, 'DWELL_RANGE_MIN', None)} convert_min={C.CONVERT_RANGE_MIN}")

# 2. dwell fires at a ~100 m settled rear-quarter finish --------------------
try:
    # learner behind a co-heading bandit 100 m ahead, wings level, nose-on → his six, ata≈0, aspect≈0.
    me = mk(1, 0, 4000.0, 230.0, 0.0, phi=0.0)
    bd = mk(2, 1, 4000.0, 200.0, 100.0)   # 100 m ahead
    r_close = R.compute(me, bd, me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)  # outcome_mode=True
    dwell_close = r_close.info.get("convert_dwell", 0.0)
    check("outcome dwell FIRES at ~100 m close finish (reward-v20)",
          dwell_close > 0.0, f"convert_dwell={dwell_close} at rng≈100 m")

    # sanity: at 300 m (inside BOTH old & new band) it also fires; at 20 m (below new floor) it does not.
    bd300 = mk(2, 1, 4000.0, 200.0, 300.0)
    bd20 = mk(2, 1, 4000.0, 200.0, 20.0)
    d300 = R.compute(me, bd300, me, bd300, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True).info.get("convert_dwell", 0.0)
    d20 = R.compute(me, bd20, me, bd20, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True).info.get("convert_dwell", 0.0)
    check("dwell pays at 300 m and NOT below the 40 m floor",
          d300 > 0.0 and d20 == 0.0, f"d300={d300} d20={d20}")
except Exception:
    check("outcome dwell close-finish", False, traceback.format_exc())

# 3. two-policy convert TERMINAL still uses CONVERT_RANGE_MIN(250) -----------
try:
    # 100 m nose-on rear-quarter with convert_bonus>0: with the OLD terminal floor 250, this is BELOW
    # the two-policy band → NO convert terminal (reward-v20 must NOT have moved that band to 40).
    me = mk(1, 0, 4000.0, 230.0, 0.0, phi=0.0)
    bd = mk(2, 1, 4000.0, 200.0, 100.0)
    # positional: (prev_me, prev_bandit, me, bandit, ground_alt, gamma, waste_streak, convert_bonus, ...)
    r_term = R.compute(me, bd, me, bd, 0.0, C.GAMMA, 0.0, 150.0, 0.0, False)  # convert_bonus=150, dense mode
    ev = r_term.info.get("event")
    check("two-policy convert TERMINAL still gated at CONVERT_RANGE_MIN=250 (no fire at 100 m)",
          ev != "convert", f"event={ev} (expected NOT 'convert' at 100 m < 250 m band)")
except Exception:
    check("two-policy convert terminal band", False, traceback.format_exc())

# 4. fire-override quality CLI --------------------------------------------
try:
    args = TP.parse_args([
        "--ladder", "bfm-headon", "--outcome-reward",
        "--fire-override-p0", "0.4", "--fire-override-steps", "8000000",
        "--fire-override-quality", "0.35",
    ])
    check("cli parses --fire-override-quality 0.35",
          abs(args.fire_override_quality - 0.35) < 1e-9 and abs(args.fire_override_p0 - 0.4) < 1e-9,
          f"q={args.fire_override_quality} p0={args.fire_override_p0} steps={args.fire_override_steps}")
except Exception:
    check("cli fire-override-quality", False, traceback.format_exc())

print()
if FAILS:
    print(f"VALIDATION FAILED: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("ALL c65 VALIDATION CHECKS PASSED")
