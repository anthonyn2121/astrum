"""Sim flight-validation for the BFM primitive library (astrum_dogfight.bfm) — Phase 1.

Flies each maneuver as an open-loop-of-the-agent controller from the head-on merge spawn against a
STRAIGHT non-shooting Tier-1 bandit (opponent_tier=1) so the signature is clean (no incoming fire,
no bandit maneuvering). Reports a per-maneuver SIGNATURE and PASS/FAIL vs the expected reversal
kinematics. The yo-yos / lag-roll / barrel / tracker need a live maneuvering bandit to fully
validate (Phase 2); here we only confirm they fly SANE (no deck, no tumble).

    docker compose --profile rl run --rm --no-deps --name bfm-check dogfight-train \
      bash -lc 'cd /workspace/controllers/rl && PYTHONPATH=/workspace/controllers/rl:\
/workspace/controllers/rl/vendor:/workspace/crates/astrum-env/python uv run python _bfm_check_tmp.py'
"""

from __future__ import annotations

import math

import numpy as np

from astrum_rl import _paths

_paths.require_astrum_env()

from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402

MPS_TO_KT = 1.94384
DECK_FLOOR_M = 320.0  # DECK_AGL is 300 m; flag anything that got within ~20 m of busting


def _course(vel) -> float:
    return math.atan2(float(vel[0]), float(vel[1]))  # atan2(E, N)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def fly(factory, seed: int = 0, max_time_s: float = 30.0):
    env = make_tdmpc_env(
        3,
        posture="offensive",
        aspect_deg=0,
        separation_m=4000.0,
        opp_heading_deg=180.0,
        opponent_tier=1,  # straight, NON-shooting bandit → clean signature
        action_repeat=5,
        frame_stack=1,
        max_time_s=max_time_s,
        rate_hz=120.0,
    )
    base = env.unwrapped
    env.reset(seed=seed)
    me0 = base._states[base.learner_id]
    bd0 = base._states[base.opponent_id]
    man = factory()
    man.reset(me0, relative(me0, bd0))

    course0 = _course(me0.vel)
    alt0 = float(me0.alt)
    v_entry = float(me0.airspeed)
    trace = []  # (t, alt, spd, phi, theta, psi, course, rng, closure)
    t = 0.0
    dt = 5.0 / 120.0
    reason = "done"
    while True:
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        a4, done = man.step(me, rel)
        # -1.0, not 0.0, is hold-fire: the trigger is affine-mapped from [-1,1] to the env's native
        # [0,1] and gated at >= 0.5, so 0.0 lands exactly ON the gate and fires every step.
        a = np.array([a4[0], a4[1], a4[2], a4[3], -1.0], np.float32)
        _obs, _r, term, trunc, _info = env.step(a)
        t += dt
        me2 = base._states[base.learner_id]
        rel2 = relative(me2, base._states[base.opponent_id])
        trace.append(
            (t, float(me2.alt), float(me2.airspeed), float(me2.phi), float(me2.theta),
             float(me2.psi), _course(me2.vel), float(rel2.rng), float(rel2.closure))
        )
        if done:
            reason = "maneuver-done"
            break
        if term:
            reason = "terminated"
            break
        if trunc:
            reason = "truncated(clock)"
            break
    me_end = base._states[base.learner_id]
    alive = bool(me_end.alive)
    env.close()

    alts = [r[1] for r in trace]
    spds = [r[2] for r in trace]
    course_end = trace[-1][6]
    d_hdg = abs(_wrap_deg(math.degrees(course_end - course0)))
    d_alt = alts[-1] - alt0
    phi_end = trace[-1][3]
    upright = abs(bfm._wrap_pi(phi_end)) < math.radians(20.0)
    return {
        "d_hdg": d_hdg,
        "d_alt": d_alt,
        "v_entry_kt": v_entry * MPS_TO_KT,
        "v_min_kt": min(spds) * MPS_TO_KT,
        "v_max_kt": max(spds) * MPS_TO_KT,
        "alt_min": min(alts),
        "alt_end": alts[-1],
        "upright": upright,
        "phi_end_deg": math.degrees(phi_end),
        "alive": alive,
        "steps": len(trace),
        "t": t,
        "reason": reason,
        "decked": min(alts) < DECK_FLOOR_M or not alive,
    }


def sig_line(name: str, s: dict, verdict: str) -> str:
    return (
        f"SIGNATURE {name:<14} dHdg {s['d_hdg']:>5.0f}  dAlt {s['d_alt']:>+7.0f}m  "
        f"spd[{s['v_min_kt']:>4.0f}..{s['v_max_kt']:>4.0f}]kt  altmin {s['alt_min']:>6.0f}m  "
        f"altend {s['alt_end']:>6.0f}m  upright {str(s['upright']):>5}  phi_end {s['phi_end_deg']:>+5.0f}  "
        f"t {s['t']:>4.1f}s  {s['reason']:<16} {verdict}"
    )


def reversal_verdict(s: dict, hdg_lo, hdg_hi, dalt_sign) -> str:
    ok_hdg = hdg_lo <= s["d_hdg"] <= hdg_hi
    ok_alt = (s["d_alt"] > 300.0) if dalt_sign > 0 else (s["d_alt"] < -300.0 if dalt_sign < 0 else True)
    ok = ok_hdg and ok_alt and s["upright"] and not s["decked"]
    tags = []
    if not ok_hdg:
        tags.append(f"hdg∉[{hdg_lo},{hdg_hi}]")
    if not ok_alt:
        tags.append("alt")
    if not s["upright"]:
        tags.append("not-upright")
    if s["decked"]:
        tags.append("DECKED")
    return "PASS" if ok else "FAIL(" + ",".join(tags) + ")"


def sane_verdict(s: dict) -> str:
    return "SANE" if (not s["decked"] and s["alive"]) else "UNSANE(" + ("DECKED" if s["decked"] else "dead") + ")"


def main() -> None:
    print("=== BFM PRIMITIVE FLIGHT VALIDATION (head-on spawn, T1 straight bandit) ===")

    # --- reversals: expected signature enforced ---
    reversals = [
        ("Immelmann", lambda: bfm.Immelmann(), (150, 210), +1),
        ("SplitS", lambda: bfm.SplitS(), (150, 210), -1),
        ("VerticalEgg", lambda: bfm.VerticalEgg(1.0), (120, 220), 0),
        ("Pitchback", lambda: bfm.Pitchback(1.0), (120, 220), 0),
        ("Wingover", lambda: bfm.Wingover(1.0), (120, 220), 0),
    ]
    for name, fac, (lo, hi), dalt in reversals:
        s = fly(fac)
        v = reversal_verdict(s, lo, hi, dalt)
        if name == "Wingover":  # extra: must have bled speed below entry
            if s["v_min_kt"] >= s["v_entry_kt"] - 5.0 and v == "PASS":
                v = "FAIL(no-speed-bleed)"
        print(sig_line(name, s, v))

    # --- pursuit modifiers: only "flies sane" (need a live bandit for the real test) ---
    modifiers = [
        ("HighYoYo", lambda: bfm.HighYoYo(1.0)),
        ("LowYoYo", lambda: bfm.LowYoYo(1.0)),
        ("LagDispRoll", lambda: bfm.LagDisplacementRoll(1.0)),
        ("BarrelRoll", lambda: bfm.BarrelRollAttack(1.0)),
        ("LagTracker", lambda: bfm.LagPursuitTracker()),
    ]
    for name, fac in modifiers:
        s = fly(fac)
        print(sig_line(name, s, sane_verdict(s)))

    # --- the coach, end-to-end, sane check ---
    s = fly(lambda: _CoachDriver())
    print(sig_line("BFMCoach", s, sane_verdict(s)))


class _CoachDriver:
    """Adapts BFMCoach (returns len-5) to the Maneuver .reset/.step(→len-4,done) harness API."""

    def __init__(self):
        self._c = bfm.BFMCoach()

    def reset(self, me, rel):
        pass

    def step(self, me, rel):
        a5 = self._c.act(me, rel)
        return np.asarray(a5[:4], np.float32), False


if __name__ == "__main__":
    main()
