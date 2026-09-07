"""curriculum-62 CPU validation: reward-v19 outcome reward + BFM-coach override wiring.

Run inside the container:
  cd controllers/rl && uv run python _c62_validate_tmp.py
Checks (no GPU, no training):
  1. all edited modules import + the CLI parses --ladder bfm-headon --outcome-reward --bfm-override-*
  2. build_stages(bfm-headon) yields the 5 head-on kill-gated rungs
  3. reward.compute (v19b HYBRID): outcome_mode keeps dense terms AND adds energy/dwell on top
  4. a live outcome_mode T3 env flown by the BFMCoach runs an episode with no crash and surfaces
     the reward-v19 info keys (convert_dwell_used, energy_reward)
"""

import sys
import traceback

import numpy as np

from astrum_rl import _paths

_paths.require_astrum_env()  # put the vendored astrum_env + adapter dir on sys.path
from astrum_dogfight import constants as C
from astrum_dogfight import reward as R
from astrum_dogfight.bfm import BFMCoach
from astrum_dogfight.env import make_tier3_env
from astrum_dogfight.geometry import AState, relative
from astrum_rl.tdmpc2 import train_parallel as TP

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(name)
    print(f"[{tag}] {name}{(' — ' + detail) if detail else ''}")


# 1. CLI parse ---------------------------------------------------------------
try:
    args = TP.parse_args(
        [
            "--ladder", "bfm-headon",
            "--outcome-reward",
            "--bfm-override-p0", "0.5",
            "--bfm-override-steps", "15000000",
            "--fire-override-p0", "0.2",
            "--fire-override-steps", "8000000",
        ]
    )
    check("cli parses bfm-headon/outcome/bfm-override",
          args.ladder == "bfm-headon" and args.outcome_reward is True
          and abs(args.bfm_override_p0 - 0.5) < 1e-9 and args.bfm_override_steps == 15_000_000,
          f"ladder={args.ladder} outcome={args.outcome_reward} p0={args.bfm_override_p0}")
except Exception:
    check("cli parses bfm-headon/outcome/bfm-override", False, traceback.format_exc())

# 2. ladder / stages ---------------------------------------------------------
try:
    rows = TP._BFM_HEADON
    stages = TP.build_stages(rows=rows, backstop_steps=40_000_000, start_stage=0,
                             kill_timeout_penalty=C.R_TIMEOUT)
    ok = (
        len(stages) == 5
        and all(s.tier == 3 and s.posture == "offensive" and s.metric == "kill" for s in stages)
        and all(abs(s.aspect_deg - 0.0) < 1e-9 and s.opp_heading_deg == 180.0 for s in stages)
        and [s.opponent_tier for s in stages] == [2, 2, 0, 0, 0]
        and all(s.timeout_penalty == C.R_TIMEOUT for s in stages)  # kill rungs → timeout penalty
    )
    check("bfm-headon builds 5 head-on kill-gated rungs", ok,
          f"n={len(stages)} ot={[s.opponent_tier for s in stages]} "
          f"sep={[int(s.separation_m) for s in stages]} wr={[s.winrate for s in stages]}")
except Exception:
    check("bfm-headon builds 5 head-on kill-gated rungs", False, traceback.format_exc())

# 3. reward.compute dense-vs-outcome ----------------------------------------
# Build two states with the learner nose-on a bandit ahead (a would-be tracking-reward geometry).
# ENU: pos=(E,N,U); heading psi=0 → nose points +N, so a bandit due-north is straight ahead.
def mk(idn, team, alt, v, north, hp=100.0, phi=0.0):
    return AState(id=idn, team=team, alive=True,
                  pos=np.array([0.0, north, alt]),
                  vel=np.array([v * np.sin(0.0), v * np.cos(0.0), 0.0]),
                  phi=phi, theta=0.0, psi=0.0, p=0.0, q=0.0, r=0.0,
                  alpha=0.0, beta=0.0, nz=1.0, airspeed=v, alt=alt, health=hp, ammo=511)


try:
    # learner at N=0 heading north; bandit 400 m ahead co-heading (I'm on his six: low aspect, nose-on
    # → dense track pays). me1 has gained energy (alt + airspeed up) for the energy-PBRS check.
    me0 = mk(1, 0, 4000.0, 250.0, 0.0)
    bd0 = mk(2, 1, 4000.0, 200.0, 400.0)
    me1 = mk(1, 0, 4050.0, 255.0, 10.0)  # gained energy (alt+speed up)
    bd1 = mk(2, 1, 4000.0, 200.0, 405.0)
    r_dense = R.compute(me0, bd0, me1, bd1, 0.0, C.GAMMA, 0.0, 0.0, 0.0, False)
    r_out = R.compute(me0, bd0, me1, bd1, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
    di, oi = r_dense.info, r_out.info
    check("dense mode keeps track_reward",
          di.get("track_reward", 0.0) != 0.0 or di.get("close_reward", 0.0) != 0.0,
          f"track={di.get('track_reward')} close={di.get('close_reward')}")
    check("v19b outcome mode KEEPS dense pursuit (track/close match dense mode)",
          oi.get("track_reward", 0.0) == di.get("track_reward", 0.0)
          and oi.get("close_reward", 0.0) == di.get("close_reward", 0.0)
          and oi.get("aim_reward", 0.0) == di.get("aim_reward", 0.0)
          and oi.get("aspect_reward", 0.0) == di.get("aspect_reward", 0.0),
          f"out track={oi.get('track_reward')} close={oi.get('close_reward')} | "
          f"dense track={di.get('track_reward')} close={di.get('close_reward')}")
    check("outcome mode adds energy PBRS (>0 on an energy gain)",
          "energy_reward" in oi and oi["energy_reward"] > 0.0, f"energy={oi.get('energy_reward')}")
    check("outcome mode surfaces convert_dwell_used", "convert_dwell_used" in oi,
          f"keys={sorted(k for k in oi if 'convert' in k or 'energy' in k)}")

    # dwell cap: park in the settled box, accumulate past the budget → capped
    me_box = mk(1, 0, 4000.0, 250.0, 0.0)
    bd_box = mk(2, 1, 4000.0, 200.0, 500.0)  # 500 m ahead, aspect≈0 ata≈0, |phi|=0 → in the settled box
    used = 0.0
    granted_any = False
    for _ in range(2000):
        rr = R.compute(me_box, bd_box, me_box, bd_box, 0.0, C.GAMMA, 0.0, 0.0, used, True)
        used = rr.info.get("convert_dwell_used", used)
        if rr.info.get("convert_dwell", 0.0) > 0.0:
            granted_any = True
    check("conversion-dwell fires in the settled box and caps at budget",
          granted_any and used <= C.CONVERT_DWELL_BUDGET + 1e-6,
          f"used={used:.3f} budget={C.CONVERT_DWELL_BUDGET}")
except Exception:
    check("reward.compute dense-vs-outcome", False, traceback.format_exc())

# 4. live outcome_mode env flown by the BFMCoach -----------------------------
try:
    env = make_tier3_env(posture="offensive", opponent_tier=2, aspect_deg=0.0,
                         opp_heading_deg=180.0, separation_m=1500.0, outcome_mode=True,
                         max_time_s=8.0, rate_hz=120.0)
    obs, info = env.reset()
    base = env  # DogfightGymEnv (make_tier3_env returns the raw env, no wrappers here)
    coach = BFMCoach()
    steps = 0
    saw_energy = False
    last_info = {}
    while steps < 1500:
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        act = coach.act(me, relative(me, bd))
        obs, rew, term, trunc, last_info = env.step(act)
        saw_energy = saw_energy or ("energy_reward" in last_info)
        steps += 1
        if term or trunc:
            break
    check("live BFMCoach + outcome_mode env runs an episode with no crash", steps > 0,
          f"steps={steps} term/trunc after")
    check("live episode surfaces reward-v19 info (energy_reward)", saw_energy,
          f"convert_dwell_used={last_info.get('convert_dwell_used')} "
          f"energy_reward={last_info.get('energy_reward')} event={last_info.get('event')}")
    env.close()
except Exception:
    check("live BFMCoach + outcome_mode env", False, traceback.format_exc())

print()
if FAILS:
    print(f"VALIDATION FAILED: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("ALL c62 VALIDATION CHECKS PASSED")
