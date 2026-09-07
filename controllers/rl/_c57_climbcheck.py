"""curriculum-57 sim-validation: fly the CLIMBING _merge_break_stick continuously from the head-on
merge spawn and confirm it HOLDS/GAINS altitude (never decks). The c56 collapse root cause #2 was an
un-imitable AGL-floor branch in the break-turn: it masked training decks yet flew the greedy policy
into the ground. c57 replaces it with a single unconditional climbing turn (roll to 45° bank + pitch
to a +8° climb). This script proves that stick actually climbs from the real merge geometry before we
commit a GPU run to it. NOT a committed test — a throwaway validation harness (controllers/rl is
untracked). Run inside the container:
  PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
    uv run python _c57_climbcheck.py
"""

import numpy as np

from astrum_dogfight.geometry import relative
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2.train_parallel import _merge_break_stick


def fly(seed: int, sep: float, max_time_s: float = 60.0) -> dict:
    env = make_tdmpc_env(
        3,
        separation_m=sep,
        posture="offensive",
        aspect_deg=0.0,
        opp_heading_deg=180.0,  # head-on merge
        opponent_tier=0,  # native T3
        jitter=0.5,
        jitter_seed=seed,
        max_time_s=max_time_s,
        action_repeat=5,
        frame_stack=1,
    )
    base = env.unwrapped
    lnr, opp = base.learner_id, base.opponent_id
    env.reset(seed=seed)
    alt0 = float(base._states[lnr].alt)
    min_alt = alt0
    event = None
    steps = 0
    while True:
        me = base._states[lnr]
        bd = base._states[opp]
        rg = relative(me, bd)
        sign = 1.0 if rg.ata_az >= 0.0 else -1.0
        stick = _merge_break_stick(float(me.phi), float(me.theta), sign)
        action = np.array([stick[0], stick[1], stick[2], stick[3], -1.0], np.float32)  # no fire
        _obs, _r, term, trunc, info = env.step(action)
        steps += 1
        min_alt = min(min_alt, float(base._states[lnr].alt))
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    final_alt = float(base._states[lnr].alt)
    env.close()
    return {
        "seed": seed,
        "sep": sep,
        "alt0": alt0,
        "min_alt": min_alt,
        "final_alt": final_alt,
        "event": event,
        "steps": steps,
        "decked": event == "hard_deck",
    }


if __name__ == "__main__":
    print("=== c57 climbing-turn altitude validation (head-on merge, opp_hdg 180, native T3) ===")
    any_deck = False
    for sep in (1200.0, 1800.0, 900.0):
        for seed in range(1, 6):
            r = fly(seed, sep)
            any_deck = any_deck or r["decked"]
            print(
                f"sep {r['sep']:6.0f} seed {r['seed']} | alt0 {r['alt0']:6.0f} "
                f"min {r['min_alt']:6.0f} final {r['final_alt']:6.0f} | "
                f"steps {r['steps']:4d} | event {str(r['event']):>12} | "
                f"{'DECKED' if r['decked'] else 'ok'}"
            )
    print("=== VERDICT:", "FAIL (a run decked)" if any_deck else "PASS (no decks — the climb holds)")
