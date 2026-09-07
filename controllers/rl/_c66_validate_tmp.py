"""curriculum-66 CPU validation: reward-v21 (corner-speed floor + relative-Es PBRS + low-crossing
hand-off gate) and the LOW-YO-YO merge-override stick / --merge-override-mode wiring.

Run inside the container (RL uv env):
  cd /workspace/controllers/rl && PYTHONPATH=... uv run python _c66_validate_tmp.py

Checks (no GPU, no training):
  1. reward-v21 constants exist with the intended values.
  2. CORNER-SPEED FLOOR (a): penalty fires below V_CORNER_FLOOR in outcome_mode, is 0 at/above it,
     and is a strict NO-OP when outcome_mode=False (legacy runs untouched).
  3. RELATIVE-Es PBRS (b): pays POSITIVE when we gain specific energy relative to the bandit,
     NEGATIVE when we fall behind, ~0 at steady state; strict no-op when outcome_mode=False.
  4. LOW-CROSSING hand-off gate (c): the two-policy convert TERMINAL fires from a settled
     matched-velocity tail (low |closure|) but is REJECTED at the same geometry with high |closure|
     (the merge-pass fly-through the c60/c61 seam handed F).
  5. _lowyoyo_stick: len-4, full throttle, rudder 0, banks TOWARD the sign, and is the ANTI-zoom of
     the egg — a gentle pull near the horizon (not the egg's saturated +50° nose-high max-g pull).
  6. --merge-override-mode parses {egg,lowyoyo} and reaches the run() plumbing; bad choice rejected.
"""

import argparse
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


def mk(idn, team, alt, v, north, hp=100.0, phi=0.0, theta=0.0):
    # ENU pos=(E,N,U), heading psi=0 → nose +N; a bandit due-north is dead ahead.
    return AState(id=idn, team=team, alive=True,
                  pos=np.array([0.0, north, alt]),
                  vel=np.array([0.0, v, 0.0]),
                  phi=phi, theta=theta, psi=0.0, p=0.0, q=0.0, r=0.0,
                  alpha=0.0, beta=0.0, nz=1.0, airspeed=v, alt=alt, health=hp, ammo=511)


# 1. constants -------------------------------------------------------------
check("reward-v21 constants exist w/ intended values",
      abs(C.V_CORNER_FLOOR - 150.0) < 1e-9 and abs(C.K_CORNER - 0.03) < 1e-9
      and abs(C.K_REL_ENERGY - 20.0) < 1e-9 and abs(C.CONVERT_CLOSURE_MAX - 100.0) < 1e-9,
      f"V_CORNER_FLOOR={C.V_CORNER_FLOOR} K_CORNER={C.K_CORNER} "
      f"K_REL_ENERGY={C.K_REL_ENERGY} CONVERT_CLOSURE_MAX={C.CONVERT_CLOSURE_MAX}")

# 2. corner-speed floor ----------------------------------------------------
try:
    # both co-heading (steady, prev==cur) so the telescoping terms are 0 and only the corner term
    # (and the always-on energy PBRS, also 0 at steady state) can move; bandit far off to isolate.
    bd = mk(2, 1, 4000.0, 200.0, 2000.0)
    slow = mk(1, 0, 4000.0, 80.0, 0.0)     # airspeed 80 < 150 floor
    fast = mk(1, 0, 4000.0, 200.0, 0.0)    # airspeed 200 ≥ floor
    slow_om = R.compute(slow, bd, slow, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
    fast_om = R.compute(fast, bd, fast, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
    slow_legacy = R.compute(slow, bd, slow, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, False)
    cp_slow = slow_om.info.get("corner_penalty")
    cp_fast = fast_om.info.get("corner_penalty")
    cp_legacy = slow_legacy.info.get("corner_penalty")
    check("corner penalty FIRES (<0) below the floor in outcome_mode",
          cp_slow is not None and cp_slow < 0.0, f"corner_penalty@80={cp_slow}")
    check("corner penalty ABSENT at/above the floor",
          cp_fast is None, f"corner_penalty@200={cp_fast}")
    check("corner penalty is a NO-OP when outcome_mode=False (legacy)",
          cp_legacy is None, f"corner_penalty(legacy)={cp_legacy}")
    # ramp shape: ~K_CORNER * (150-80)/150
    exp = C.K_CORNER * (C.V_CORNER_FLOOR - 80.0) / C.V_CORNER_FLOOR
    check("corner penalty magnitude matches the ramp",
          cp_slow is not None and abs(-cp_slow - exp) < 1e-6, f"got {cp_slow}, expected {-exp:.5f}")
except Exception:
    check("corner-speed floor", False, traceback.format_exc())

# 3. relative-Es PBRS ------------------------------------------------------
try:
    bd = mk(2, 1, 4000.0, 200.0, 2000.0)
    prev_me = mk(1, 0, 4000.0, 150.0, 0.0)   # lower energy than "now"
    now_me = mk(1, 0, 4000.0, 250.0, 0.0)    # gained speed → gained energy vs the (steady) bandit
    gained = R.compute(prev_me, bd, now_me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
    lost = R.compute(now_me, bd, prev_me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)  # reverse → fell behind
    steady = R.compute(now_me, bd, now_me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
    legacy = R.compute(prev_me, bd, now_me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, False)
    rg = gained.info.get("rel_energy_reward")
    rl = lost.info.get("rel_energy_reward")
    rs = steady.info.get("rel_energy_reward")
    check("rel-Es PBRS POSITIVE when gaining energy vs bandit",
          rg is not None and rg > 0.0, f"rel_energy@gain={rg}")
    check("rel-Es PBRS NEGATIVE when falling behind",
          rl is not None and rl < 0.0, f"rel_energy@lose={rl}")
    check("rel-Es PBRS ~0 at steady state (telescoping)",
          rs is not None and abs(rs) < 1e-9, f"rel_energy@steady={rs}")
    check("rel-Es PBRS is a NO-OP when outcome_mode=False (legacy)",
          legacy.info.get("rel_energy_reward") is None, f"rel_energy(legacy)={legacy.info.get('rel_energy_reward')}")
except Exception:
    check("relative-Es PBRS", False, traceback.format_exc())

# 4. low-crossing hand-off gate on the convert TERMINAL --------------------
try:
    # Settled rear-quarter tail 400 m ahead (in [250,1000]), wings level, nose-on his six.
    # LOW closure: me 210 vs bandit 200 → |closure|≈10 ≤ 100 → CONVERT.
    # HIGH closure: me 450 vs bandit 100 → |closure|≈350 > 100 → REJECT (a merge-pass fly-through).
    bd = mk(2, 1, 4000.0, 200.0, 400.0)
    me_lo = mk(1, 0, 4000.0, 210.0, 0.0)
    bd_hi = mk(2, 1, 4000.0, 100.0, 400.0)
    me_hi = mk(1, 0, 4000.0, 450.0, 0.0)
    from astrum_dogfight.geometry import relative
    clos_lo = relative(me_lo, bd).closure
    clos_hi = relative(me_hi, bd_hi).closure
    r_lo = R.compute(me_lo, bd, me_lo, bd, 0.0, C.GAMMA, 0.0, 150.0, 0.0, False)   # convert_bonus=150
    r_hi = R.compute(me_hi, bd_hi, me_hi, bd_hi, 0.0, C.GAMMA, 0.0, 150.0, 0.0, False)
    check("convert TERMINAL fires from a LOW-crossing settled tail",
          r_lo.info.get("event") == "conversion" and r_lo.terminated,
          f"|clos|={abs(clos_lo):.0f} event={r_lo.info.get('event')}")
    check("convert TERMINAL REJECTS the same geometry at HIGH crossing (merge-pass)",
          r_hi.info.get("event") != "conversion",
          f"|clos|={abs(clos_hi):.0f} event={r_hi.info.get('event')}")
except Exception:
    check("low-crossing hand-off gate", False, traceback.format_exc())

# 5. _lowyoyo_stick: CORNER (keep energy) vs ROLL-OUT/SETTLE (finishable hand-off) ---------
try:
    import math as _m5
    check("roll-out/settle constants exist w/ intended values",
          abs(_m5.degrees(TP._LY_SETTLE_ASPECT_RAD) - 55.0) < 1e-6
          and abs(TP._LY_SETTLE_RANGE_M - 1000.0) < 1e-9
          and abs(_m5.degrees(TP._LY_SETTLE_BANK_RAD) - 35.0) < 1e-6
          and abs(TP._LY_SETTLE_THR - 0.55) < 1e-9,
          f"aspect={_m5.degrees(TP._LY_SETTLE_ASPECT_RAD)} rng={TP._LY_SETTLE_RANGE_M} "
          f"bank={_m5.degrees(TP._LY_SETTLE_BANK_RAD)} thr={TP._LY_SETTLE_THR}")
    # CORNER: not yet behind (aspect big) and/or outside the band → hard bank, full throttle, nose near horizon
    cornr = TP._lowyoyo_stick(0.0, 0.0, 1200.0, _m5.radians(120.0), +1.0)      # lagging, wide aspect
    cornr_near = TP._lowyoyo_stick(0.0, 0.0, 800.0, _m5.radians(120.0), +1.0)  # close but NOT behind → still corner
    # SETTLE: genuinely behind (aspect ≤ 55°) AND inside the band (rng ≤ 1000) → roll out, ease throttle, gentle pull
    setl = TP._lowyoyo_stick(0.0, 0.0, 300.0, _m5.radians(30.0), +1.0)
    egg = TP._vertical_egg_stick(0.0, 0.0, +1.0)
    check("_lowyoyo_stick len-4, rudder 0 (both phases)",
          cornr.shape == (4,) and abs(cornr[2]) < 1e-9 and abs(setl[2]) < 1e-9,
          f"corner={cornr} settle={setl}")
    check("_lowyoyo CORNER: FULL throttle (keep energy)",
          abs(cornr[3] - 1.0) < 1e-6, f"corner_thr={cornr[3]}")
    check("_lowyoyo SETTLE: throttle EASED to bleed overtake (< corner)",
          setl[3] < cornr[3] and abs(setl[3] - TP._LY_SETTLE_THR) < 1e-6, f"settle_thr={setl[3]}")
    check("_lowyoyo banks TOWARD +sign both phases (ail>0)",
          cornr[0] > 0.0 and setl[0] > 0.0, f"corner_ail={cornr[0]} settle_ail={setl[0]}")
    check("_lowyoyo banks TOWARD -sign (ail<0)",
          TP._lowyoyo_stick(0.0, 0.0, 300.0, _m5.radians(30.0), -1.0)[0] < 0.0)
    # THE FIX: from an already-hard +72° bank, CORNER HOLDS it (ail≈0) but SETTLE rolls OUT toward level (ail<0)
    corner_hold = TP._lowyoyo_stick(TP._LY_BANK_RAD, 0.0, 1200.0, _m5.radians(120.0), +1.0)
    settle_rollout = TP._lowyoyo_stick(TP._LY_BANK_RAD, 0.0, 300.0, _m5.radians(30.0), +1.0)
    check("_lowyoyo SETTLE rolls OUT toward level from a hard bank (ail<0); CORNER holds it (ail~0)",
          settle_rollout[0] < -0.01 and abs(corner_hold[0]) < 0.05,
          f"settle_rollout_ail={settle_rollout[0]:.3f} corner_hold_ail={corner_hold[0]:.3f}")
    # egg saturates elevator to full nose-up pull (-1.0); lowyoyo CORNER does NOT (energy-keeping)
    check("_lowyoyo CORNER is the ANTI-zoom of the egg (elev NOT a saturated nose-up pull)",
          egg[1] <= -0.999 and cornr[1] > -0.9,
          f"egg_elev={egg[1]:.3f} corner_elev={cornr[1]:.3f}")
    # CORNER far/lagging → nose-LOW to hold speed (elev > 0 in this NEG=pull convention)
    check("_lowyoyo CORNER noses LOW when lagging (rng>knee) to hold speed",
          cornr[1] > 0.0, f"corner_elev(far)={cornr[1]:.3f}")
    # phase GATE: close but NOT behind (aspect large) stays in CORNER (full throttle) — needs BOTH aspect & range
    check("_lowyoyo stays in CORNER when close but NOT behind (aspect large)",
          abs(cornr_near[3] - 1.0) < 1e-6, f"thr={cornr_near[3]}")
except Exception:
    check("_lowyoyo_stick", False, traceback.format_exc())

# 6. --merge-override-mode CLI wiring --------------------------------------
try:
    a_egg = TP.parse_args(["--merge-override-mode", "egg"])
    a_ly = TP.parse_args(["--merge-override-mode", "lowyoyo"])
    a_def = TP.parse_args([])
    check("--merge-override-mode parses egg/lowyoyo + defaults to egg",
          a_egg.merge_override_mode == "egg" and a_ly.merge_override_mode == "lowyoyo"
          and a_def.merge_override_mode == "egg",
          f"egg={a_egg.merge_override_mode} lowyoyo={a_ly.merge_override_mode} default={a_def.merge_override_mode}")
    bad = False
    try:
        TP.parse_args(["--merge-override-mode", "spiral"])
    except SystemExit:
        bad = True
    check("--merge-override-mode rejects an invalid choice", bad)
except Exception:
    check("--merge-override-mode CLI", False, traceback.format_exc())

print()
if FAILS:
    print(f"*** {len(FAILS)} CHECK(S) FAILED: {FAILS}")
    sys.exit(1)
print("*** ALL c66 CHECKS PASSED")
