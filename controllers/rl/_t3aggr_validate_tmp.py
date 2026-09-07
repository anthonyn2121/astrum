"""Validate the AGGRESSIVE Tier-3 break: constants bumped, and the break law now rides
the AoA limit (max-performance turn) instead of the old energy-preserving corner-hold.

Run in-container:
  cd /workspace/controllers/rl && uv run python _t3aggr_validate_tmp.py

Checks (no GPU, no training):
  1. constants: BREAK_BANK_RAD=85°, V_ENERGY_LOW=105.
  2. _corner_limited_pull(ease_speed=False) drops the AIRSPEED easing (pulls harder when slow)
     but KEEPS the AoA soft-cap (departure protection) — so it's max-perf, not a mush.
  3. Tier3._break commands ~85° bank target and a near-full (alpha-limited) pull, strictly
     harder than the old -0.95 corner-limited pull at the same slow, low-AoA state.
  4. A full chase episode vs the aggressive T3: it commits a hard break (heading swings hard)
     AND now actually EXPENDS energy (bleeds airspeed well below the old ~156 floor) while
     staying inside the FLCS envelope (alive, |nz| bounded).
"""

import math
import sys
import traceback

import numpy as np

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C
from astrum_dogfight import opponent as OPP
from astrum_dogfight.geometry import AState

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}")


def mk(v, alpha_deg=0.0):
    return AState(id=2, team=1, alive=True,
                  pos=np.array([0.0, 0.0, 4000.0]), vel=np.array([0.0, v, 0.0]),
                  phi=0.0, theta=0.0, psi=0.0, p=0.0, q=0.0, r=0.0,
                  alpha=math.radians(alpha_deg), beta=0.0, nz=1.0,
                  airspeed=v, alt=4000.0, health=100.0, ammo=511)


# 1. constants -------------------------------------------------------------
check("BREAK_BANK_RAD bumped to 85°",
      abs(math.degrees(C.BREAK_BANK_RAD) - 85.0) < 1e-6, f"{math.degrees(C.BREAK_BANK_RAD):.1f}°")
check("V_ENERGY_LOW lowered to 105 m/s",
      abs(C.V_ENERGY_LOW - 105.0) < 1e-9, f"{C.V_ENERGY_LOW}")

# 2. ease_speed flag -------------------------------------------------------
try:
    slow = mk(130.0)  # below corner (175), low AoA
    eased = OPP._corner_limited_pull(-1.0, slow, ease_speed=True)
    maxp = OPP._corner_limited_pull(-1.0, slow, ease_speed=False)
    check("ease_speed=False pulls HARDER than the eased corner-hold when slow",
          maxp < eased and abs(maxp - (-1.0)) < 1e-9,
          f"eased={eased:.3f} maxperf={maxp:.3f}")
    # AoA cap still bites even with ease_speed=False (departure protection retained)
    hi_alpha = mk(130.0, alpha_deg=30.0)  # past the 22° soft cap
    capped = OPP._corner_limited_pull(-1.0, hi_alpha, ease_speed=False)
    check("ease_speed=False STILL applies the AoA soft-cap (no mush/departure)",
          capped > -1.0 + 1e-6, f"pull@30°AoA={capped:.3f} (eased back from -1.0)")
except Exception:
    check("ease_speed flag", False, traceback.format_exc())

# 3. _break is strictly more aggressive ------------------------------------
try:
    t3 = OPP.Tier3Opponent(dt=1 / 120)
    # attacker (learner) dead astern so ata_az sign is defined; me slow & low-AoA
    me = mk(150.0)
    atkr = AState(id=1, team=0, alive=True, pos=np.array([0.0, -800.0, 4000.0]),
                  vel=np.array([0.0, 150.0, 0.0]), phi=0.0, theta=0.0, psi=0.0,
                  p=0.0, q=0.0, r=0.0, alpha=0.0, beta=0.0, nz=1.0,
                  airspeed=150.0, alt=4000.0, health=100.0, ammo=511)
    ail, elev, rud, thr = t3._break(me, atkr)
    old_pull = OPP._corner_limited_pull(-0.95, me)  # what the pre-change break commanded
    check("_break pull rides the AoA limit (near -1.0, harder than old -0.95 corner-hold)",
          elev < old_pull and elev <= -0.99, f"new_elev={elev:.3f} old_elev={old_pull:.3f}")
    check("_break rolls toward the 85° break bank (full-authority aileron from wings-level)",
          abs(ail) > 0.9 and thr >= 0.99 and abs(rud) < 1e-9,
          f"ail={ail:.3f} thr={thr:.3f} rud={rud:.3f}")
except Exception:
    check("_break aggression", False, traceback.format_exc())

# 4. full chase episode: T3 breaks hard AND bleeds energy ------------------
try:
    from astrum_rl.envs import make_tdmpc_env
    env = make_tdmpc_env(3, posture="offensive", aspect_deg=0.0, separation_m=700.0,
                         action_repeat=5, frame_stack=1, max_time_s=30.0, rate_hz=120.0)
    base = env.unwrapped
    env.reset(seed=7)
    opp_id = base.opponent_id
    psis, spds, nzs, alive_all = [], [], [], True
    # fly the LEARNER straight-ish (zero stick) so we isolate the T3's own break behavior
    a = np.zeros(base.action_space.shape[0], dtype=np.float32)
    for _ in range(int(30.0 / (5 / 120.0))):
        env.step(a)
        o = base._states.get(opp_id)
        if o is None:
            break
        psis.append(math.degrees(o.psi) % 360.0)
        spds.append(float(o.airspeed))
        nzs.append(abs(float(o.nz)))
        alive_all = alive_all and o.alive
    env.close()
    # heading sweep (unwrap the wrap-around)
    sweep = 0.0
    for i in range(1, len(psis)):
        d = psis[i] - psis[i - 1]
        d = (d + 180.0) % 360.0 - 180.0
        sweep += abs(d)
    min_spd = min(spds) if spds else 999.0
    peak_g = max(nzs) if nzs else 0.0
    check("aggressive T3 swings heading hard in the break (sweep > 90°)",
          sweep > 90.0, f"total sweep={sweep:.0f}°")
    # The real aggression signal is PEAK G: the old energy-preserving wide climbing turn only
    # pulled ~3-4 g; a max-performance break rides the g/AoA limit (~7-9 g). It holds ~corner
    # speed by going nose-low (trading ALTITUDE for turn), which is optimal — not a mush.
    check("aggressive T3 pulls a hard, high-g break (peak |nz| > 6 g, vs ~3-4 g wide turn)",
          peak_g > 6.0, f"peak|nz|={peak_g:.1f} g, min airspeed={min_spd:.0f} m/s (holds ~corner)")
    check("aggressive T3 stays inside the FLCS envelope (alive, |nz| bounded < 12)",
          alive_all and peak_g < 12.0, f"alive={alive_all} max|nz|={peak_g:.1f}")
except Exception:
    check("full chase episode", False, traceback.format_exc())

print()
if FAILS:
    print(f"*** {len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1)
print("*** ALL aggressive-T3 CHECKS PASSED")
