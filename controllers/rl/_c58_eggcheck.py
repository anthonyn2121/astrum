"""curriculum-58 sim-validation: fly the VERTICAL-EGG _vertical_egg_stick continuously from the head-on
merge spawn and confirm it (a) PULLS UP INTO THE VERTICAL (pitch goes strongly nose-HIGH, altitude
GAINED) and (b) NEVER decks — i.e. it is the vertical egg the user asked for, not the c56/c57 descending
corkscrew. The c57 climbing-break stick (roll ~45° bank, pitch to only +8°) still descended enough under
a hard pull that the greedy policy decked; c58 rolls only a MODEST ~30° bank and pulls MAX-g to a steep
+50° nose-high target, so the lift vector stays mostly vertical and the jet zooms up-and-over. This proves
the stick actually climbs (and how nose-high it gets) from the real merge geometry before we commit a GPU
run to it. NOT a committed test — a throwaway validation harness (controllers/rl is untracked). Run inside
the container:
  PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
    uv run python _c58_eggcheck.py
"""

import math

import numpy as np

from astrum_dogfight.geometry import relative
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2.train_parallel import _vertical_egg_stick


def fly(seed: int, sep: float, max_time_s: float = 30.0) -> dict:
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
    max_alt = alt0
    max_pitch = math.degrees(float(base._states[lnr].theta))
    min_pitch = max_pitch
    event = None
    steps = 0
    while True:
        me = base._states[lnr]
        bd = base._states[opp]
        rg = relative(me, bd)
        sign = 1.0 if rg.ata_az >= 0.0 else -1.0
        stick = _vertical_egg_stick(float(me.phi), float(me.theta), sign)
        action = np.array([stick[0], stick[1], stick[2], stick[3], -1.0], np.float32)  # no fire
        _obs, _r, term, trunc, info = env.step(action)
        steps += 1
        me = base._states[lnr]
        min_alt = min(min_alt, float(me.alt))
        max_alt = max(max_alt, float(me.alt))
        pit = math.degrees(float(me.theta))
        max_pitch = max(max_pitch, pit)
        min_pitch = min(min_pitch, pit)
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
        "max_alt": max_alt,
        "final_alt": final_alt,
        "max_pitch": max_pitch,
        "min_pitch": min_pitch,
        "event": event,
        "steps": steps,
        "decked": event == "hard_deck",
    }


if __name__ == "__main__":
    print("=== c58 VERTICAL-EGG altitude+pitch validation (head-on merge, opp_hdg 180, native T3) ===")
    any_deck = False
    never_nose_high = False
    for sep in (1200.0, 1800.0, 900.0):
        for seed in range(1, 6):
            r = fly(seed, sep)
            any_deck = any_deck or r["decked"]
            # the egg must actually pull the nose well UP at some point (vertical, not a level break)
            if r["max_pitch"] < 30.0:
                never_nose_high = True
            print(
                f"sep {r['sep']:6.0f} seed {r['seed']} | alt0 {r['alt0']:6.0f} "
                f"min {r['min_alt']:6.0f} max {r['max_alt']:6.0f} final {r['final_alt']:6.0f} | "
                f"pitch [{r['min_pitch']:+5.0f},{r['max_pitch']:+5.0f}]° | "
                f"steps {r['steps']:4d} | event {str(r['event']):>12} | "
                f"{'DECKED' if r['decked'] else 'ok'}"
            )
    verdict = "PASS (climbs into the vertical, no decks)"
    if any_deck:
        verdict = "FAIL (a run decked)"
    elif never_nose_high:
        verdict = "FAIL (never pulled the nose above +30° — not a vertical egg)"
    print("=== VERDICT:", verdict)
