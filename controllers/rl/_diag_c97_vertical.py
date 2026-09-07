"""c97 PRE-CHECK — over WHOLE 180 s greedy episodes, which way does the fight actually GO, and what
does the reward pay for going there?

Three pre-registered questions, all on the SAME perched merge the checkpoints were trained and
measured on (sep 3000 m, aspect 30 deg, opp_heading 180, perch +2000 ft, block 15,000 ft, 180 s).

Q1 — THE BOX'S PREMISE.  F4 concluded "the T3 climbs 460-980 m while every trained checkpoint
     DIVES", and I put that in the handoff doc.  The c97 pre-flight then measured c96 flying this
     cell at gamma p50 +26.5 deg -- CLIMBING HARDER THAN THE T3 -- over 24 s.  So the premise is in
     doubt and the number it rested on (c96 "-4,670 m") is a pooled per-EXCURSION sum over gated
     windows, not a net episode descent.  PREDICTION, written before the run: c96's NET altitude
     change over a whole episode is POSITIVE on this cell, i.e. my own earlier framing is wrong and
     "the policies dive" must be withdrawn or re-scoped to c95/c87.

Q2 — THE ZOOM-STALL LOOP.  The pre-flight's gamma p95 was +63.6 deg, a near-vertical zoom, and
     `reward.py`'s own comment at the corner-speed floor names that exact failure: "the merge
     zoom-to-stall (airspeed 222->80 m/s, trading all speed for altitude) ... the exact hole the T3
     exploits (it holds >=128 m/s in the vertical scissors while we stall)".  PREDICTION: c96 spends
     real time below V_BAND_CTR = 175 m/s and is charged for it, so the vertical problem is not a
     SIGN error but an ENERGY-MANAGEMENT one: it goes up, but by trading away the speed it needs.

Q3 — THE REWARD'S NET VERTICAL TILT (unmeasured until now; the prerequisite for any claim that the
     reward does or does not price the vertical).  Sum ALL SIX vertical-coupled terms per episode:
     energy_reward, rel_energy_reward, band_reward, corner_penalty, deck_penalty, spiral_penalty.
     `reward.py` contains no gamma, no vertical-rate and no rel_alt term -- but K_BAND (20.0, flat
     at/above 175 m/s, decaying below) and K_CORNER (0.03/tick) price AIRSPEED one-sidedly, and
     altitude BUYS airspeed, so an exactly Es-neutral trade is NOT reward-neutral.  This measures
     the sign and size of that tilt on the real distribution instead of on a hypothetical.

⚠ A CADENCE DEFECT THIS SCRIPT EXISTS PARTLY TO AVOID.  `envs.ActionRepeat.step` SUMS reward across
its 5 ticks but returns only the LAST tick's `info`.  Reading `info["spiral_penalty"]` once per
AGENT step therefore samples 1 tick in 5 -- a ~5x undercount on every per-tick term and a random
4/5 loss of every PBRS increment.  `_refute_alt_tax_episode.py` accumulates exactly that way, so its
sums are not comparable to a reward total.  Here the env is built with `action_repeat=1` and the
5-tick hold is done in this loop, accumulating `info` on EVERY tick, which reproduces
`ActionRepeat`'s reward exactly while keeping the per-term sums honest.

⚠ The three checkpoints are 41-wide and `OBS_DIM` is now 45, so each is widened at load with the
trainer's own `_widen_resume_obs`.  The c97 pre-flight measured that this is bit-identical
(max |dAction| = 0.000e+00 over 1152 merge steps), so these are the SAME policies, not variants.
"""

from __future__ import annotations

import argparse
import copy
import math
import os

import numpy as np
import torch
from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import flight_path_angle  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402
from astrum_rl.tdmpc2.train_parallel import _widen_resume_obs  # noqa: E402

DEG = 180.0 / math.pi
OUT = "/workspace/controllers/rl/runs/c97"
REPEAT = 5

# Every reward term that is a function of altitude or of the airspeed altitude buys.
# ⚠ All six are stored in `info` ALREADY SIGNED as their contribution to `reward`: the two penalties
# whose constants are negative (K_DECK_FLOOR = K_SPIRAL = -0.02) are `reward += pen`, and the corner
# floor is `reward -= pen` with `info[...] = -pen`.  So the six are directly summable and their total
# is a real slice of the episode return, not a magnitude tally.
VERT_KEYS = (
    "energy_reward",  # K_ENERGY  PBRS, Es = AGL + V^2/2g  (neutral on an alt<->speed trade)
    "rel_energy_reward",  # K_REL_ENERGY PBRS, same but vs the bandit
    "band_reward",  # K_BAND    PBRS, flat >= 175 m/s, decays below  => PRO-DIVE when slow
    "corner_penalty",  # K_CORNER  per-tick, bites below 150 m/s      => PRO-DIVE when slow
    "deck_penalty",  # K_DECK_FLOOR per-tick, bottom 800 m           => ANTI-dive
    "spiral_penalty",  # K_SPIRAL  per-tick, nose-low x banked x low   => ANTI-dive
)

CKPTS = (
    ("c96", "/workspace/controllers/rl/runs/curriculum-96/run.pt"),
    ("c95", "/workspace/controllers/rl/runs/curriculum-95/run.pt"),
    ("c87", "/workspace/controllers/rl/runs/curriculum-87/c87_arrival_obs_5000k.pt"),
)

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def load_agent(ckpt: str, device) -> TDMPC2:
    sd = copy.deepcopy(torch.load(ckpt, map_location=device, weights_only=False))
    fs = int(sd["cfg"].get("frame_stack", 1) or 1)
    per_frame = int(sd["cfg"]["obs_dim"]) // fs
    if per_frame != C.OBS_DIM:
        _widen_resume_obs(sd, per_frame, C.OBS_DIM, fs)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # E3: the planner is death-seeking until the terminal is priced
    agent = TDMPC2(cfg, device=device)
    agent.model.load_state_dict(sd["model"])
    agent.model.eval()
    return agent


def run_episode(env, base, agent, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    t0 = True
    sums = dict.fromkeys(VERT_KEYS, 0.0)
    ret = 0.0
    me_alt, bd_alt, me_gam, bd_gam, me_v = [], [], [], [], []
    event = None
    n_agent = 0
    while True:
        act = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        me_alt.append(float(me.alt))
        bd_alt.append(float(bd.alt))
        me_gam.append(flight_path_angle(me) * DEG)
        bd_gam.append(flight_path_angle(bd) * DEG)
        me_v.append(float(me.airspeed))
        n_agent += 1
        # The 5-tick hold, done here so `info` is read on EVERY tick (see the module docstring).
        term = trunc = False
        for _ in range(REPEAT):
            obs, r, term, trunc, info = env.step(act)
            ret += float(r)
            for k in VERT_KEYS:
                sums[k] += float(info.get(k, 0.0) or 0.0)
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
        if term or trunc:
            break
    me_alt_a, bd_alt_a = np.asarray(me_alt), np.asarray(bd_alt)
    me_v_a = np.asarray(me_v)
    return dict(
        event=event,
        n=n_agent,
        dur=n_agent * REPEAT / 120.0,
        ret=ret,
        sums=sums,
        me_dalt=float(me_alt_a[-1] - me_alt_a[0]),
        bd_dalt=float(bd_alt_a[-1] - bd_alt_a[0]),
        me_alt_min=float(me_alt_a.min()),
        me_gam=float(np.median(me_gam)),
        me_gam_p95=float(np.percentile(me_gam, 95)),
        bd_gam=float(np.median(bd_gam)),
        me_v_p50=float(np.median(me_v_a)),
        me_v_min=float(me_v_a.min()),
        f_slow=float(np.mean(me_v_a < C.V_BAND_CTR)),
        f_stall=float(np.mean(me_v_a < C.V_CORNER_FLOOR)),
        f_lowband=float(np.mean(me_alt_a < C.SPIRAL_ALT_BAND_AGL)),
        hp_dealt=C.HP_MAX - float(base._states[base.opponent_id].health),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("auto")
    log("#" * 118)
    log("#  c97 PRE-CHECK — whole-episode vertical behaviour and the reward's vertical tilt")
    log("#" * 118)
    log("")
    log(
        f"cell: perched merge  sep 3000 m  aspect 30 deg  opp_heading 180  perch +2000 ft  "
        f"block 15,000 ft  {a.max_time_s:.0f} s  jitter {a.jitter}  greedy  {a.episodes} eps"
    )
    log(
        f"device {device}   OBS_DIM {C.OBS_DIM} (checkpoints widened from 41 at load; bit-identical)"
    )
    log(
        f"thresholds: V_BAND_CTR {C.V_BAND_CTR:.0f}  V_CORNER_FLOOR {C.V_CORNER_FLOOR:.0f} m/s   "
        f"SPIRAL band < {C.SPIRAL_ALT_BAND_AGL:.0f} m AGL   deck cushion < "
        f"{C.DECK_AGL + C.DECK_MARGIN:.0f} m"
    )

    kw = dict(
        posture="offensive",
        aspect_deg=30.0,
        separation_m=3000.0,
        action_repeat=1,  # the 5-tick hold is done in run_episode so info is read per TICK
        frame_stack=1,
        max_time_s=a.max_time_s,
        rate_hz=120.0,
        opp_heading_deg=180.0,
        jitter=a.jitter,
        me_dalt_ft=2000.0,
        alt_ft=15000.0,
        outcome_mode=True,
    )

    table: dict[str, list[dict]] = {}
    for name, path in CKPTS:
        if not os.path.exists(path):
            log(f"\n[skip] {name}: {path} not found")
            continue
        agent = load_agent(path, device)
        env = make_tdmpc_env(3, **kw)
        base = env
        while hasattr(base, "env"):
            base = base.env
        eps = [run_episode(env, base, agent, 1000 + i) for i in range(a.episodes)]
        env.close()
        table[name] = eps

        log("")
        log("-" * 118)
        log(f"ARM {name}")
        log("-" * 118)
        log(
            f"  {'ep':>3} {'event':>12} {'T':>6} | {'ME dalt':>9} {'gam50':>7} {'gam95':>7} "
            f"{'minAlt':>8} | {'T3 dalt':>9} {'gam50':>7} | {'V50':>6} {'Vmin':>6} "
            f"{'%<175':>6} {'%<150':>6} {'%<2km':>6} | {'dealt':>6}"
        )
        for i, e in enumerate(eps):
            log(
                f"  {i:>3} {str(e['event']):>12} {e['dur']:6.1f} | {e['me_dalt']:+9.1f} "
                f"{e['me_gam']:+7.1f} {e['me_gam_p95']:+7.1f} {e['me_alt_min']:8.1f} | "
                f"{e['bd_dalt']:+9.1f} {e['bd_gam']:+7.1f} | {e['me_v_p50']:6.1f} "
                f"{e['me_v_min']:6.1f} {e['f_slow']:6.1%} {e['f_stall']:6.1%} "
                f"{e['f_lowband']:6.1%} | {e['hp_dealt']:6.1f}"
            )

        n_up = sum(1 for e in eps if e["me_dalt"] > 0)
        log("")
        log(
            f"  Q1  net ME dalt: median {np.median([e['me_dalt'] for e in eps]):+9.1f} m   "
            f"CLIMBS in {n_up}/{len(eps)} episodes   (T3 median "
            f"{np.median([e['bd_dalt'] for e in eps]):+9.1f} m)"
        )
        log(
            f"  Q2  slow: median {np.median([e['f_slow'] for e in eps]):.1%} of steps < 175 m/s, "
            f"{np.median([e['f_stall'] for e in eps]):.1%} < 150 m/s;  "
            f"Vmin median {np.median([e['me_v_min'] for e in eps]):.1f} m/s"
        )
        log("")
        log(f"  Q3  EPISODE REWARD SUMS (median over {len(eps)} eps)")
        log(f"      {'term':>20} {'median':>10} {'min':>10} {'max':>10}   role")
        roles = {
            "energy_reward": "Es PBRS (alt<->speed NEUTRAL)",
            "rel_energy_reward": "rel-Es PBRS vs bandit",
            "band_reward": "airspeed band  => PRO-DIVE when slow",
            "corner_penalty": "corner floor   => PRO-DIVE when slow",
            "deck_penalty": "deck cushion   => ANTI-dive",
            "spiral_penalty": "anti-spiral    => ANTI-dive",
        }
        for k in VERT_KEYS:
            v = [e["sums"][k] for e in eps]
            log(
                f"      {k:>20} {np.median(v):+10.2f} {np.min(v):+10.2f} {np.max(v):+10.2f}   "
                f"{roles[k]}"
            )
        pro_dive = [e["sums"]["band_reward"] + e["sums"]["corner_penalty"] for e in eps]
        anti_dive = [e["sums"]["deck_penalty"] + e["sums"]["spiral_penalty"] for e in eps]
        net_all = [sum(e["sums"].values()) for e in eps]
        log(
            f"      {'--- pro-dive pair':>20} {np.median(pro_dive):+10.2f} "
            f"{np.min(pro_dive):+10.2f} {np.max(pro_dive):+10.2f}   band + corner"
        )
        log(
            f"      {'--- anti-dive pair':>20} {np.median(anti_dive):+10.2f} "
            f"{np.min(anti_dive):+10.2f} {np.max(anti_dive):+10.2f}   deck + spiral"
        )
        log(
            f"      {'--- ALL SIX':>20} {np.median(net_all):+10.2f} {np.min(net_all):+10.2f} "
            f"{np.max(net_all):+10.2f}   vs episode RET median "
            f"{np.median([e['ret'] for e in eps]):+.1f}"
        )

    log("")
    log("=" * 118)
    log("VERDICT")
    log("=" * 118)
    for name, eps in table.items():
        n_up = sum(1 for e in eps if e["me_dalt"] > 0)
        log(
            f"  {name}: net dalt median {np.median([e['me_dalt'] for e in eps]):+9.1f} m, climbs "
            f"{n_up}/{len(eps)};  T3 {np.median([e['bd_dalt'] for e in eps]):+9.1f} m;  "
            f"slow {np.median([e['f_slow'] for e in eps]):.0%};  "
            f"band+corner {np.median([e['sums']['band_reward'] + e['sums']['corner_penalty'] for e in eps]):+7.2f};  "
            f"deck+spiral {np.median([e['sums']['deck_penalty'] + e['sums']['spiral_penalty'] for e in eps]):+7.2f};  "
            f"dealt {np.median([e['hp_dealt'] for e in eps]):5.1f}"
        )
    log("")
    log("  Read Q1 against the PREDICTION in the docstring: I predicted c96 NET CLIMBS, i.e. that")
    log("  my own F4-derived 'every trained policy dives' is wrong on this cell and must be")
    log("  withdrawn.  If c96 net-descends instead, the original framing survives and this")
    log("  pre-check is the refutation of the pre-flight's 24 s window, not of the doc.")

    p = os.path.join(OUT, "c97_vertical.txt")
    with open(p, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    log(f"\n[written] {p}")


if __name__ == "__main__":
    main()
