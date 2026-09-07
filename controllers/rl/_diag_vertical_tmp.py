"""OUT-OF-PLANE / VERTICAL usage autopsy (read-only forensics; throwaway like _yoyo_probe_tmp.py).

Question: does a checkpoint fight in the VERTICAL at all, or purely in-plane?
`_yoyo_probe_tmp.py` answers that for the CO-HEADING chase only (no --opp-heading), and it
never reports absolute altitude / bubble-exit cause. This probe supports the NEUTRAL HEAD-ON
geometry (aspect 0 + opp_heading 180, sep 3000, 180 s) and reports, per episode:

  * event + WHY the bubble was left (horiz > R_BUBBLE vs alt > CEILING_AGL)
  * absolute altitude envelope: alt0 / maxAlt / minAlt / peak-to-trough amplitude
  * vertical separation vs the bandit: max/min (me.alt - bd.alt)
  * attitude usage while ENGAGED (rng < engage_rng AND |bank| > bank_deg):
      climb% / dive% / flat% (pitch deadband), |bank| p95, vertical-amplitude std
  * out-of-plane events: prominence-gated yoyo swings on (me.alt-bd.alt), hi/lo + convert rate
  * roll reversals (sign flips of bank through >bank_deg) = displacement-roll proxy
  * best gun-quality moment and its range/closure/lead (matches _energy_merge_probe_tmp.py)
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

DEG = 180.0 / math.pi
G = 9.80665


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True, help="comma list PATH[=LABEL]")
    p.add_argument("--aspect", type=float, default=0.0)
    p.add_argument("--opp-heading", type=float, default=None, help="180 = head-on; omit = co-heading chase")
    p.add_argument("--sep", type=float, default=3000.0)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--max-time-s", type=float, default=180.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--me-dalt-ft", type=float, default=0.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--engage-rng", type=float, default=1400.0)
    p.add_argument("--bank-deg", type=float, default=35.0)
    p.add_argument("--pitch-deg", type=float, default=5.0)
    p.add_argument("--prom-m", type=float, default=60.0)
    p.add_argument("--conv-s", type=float, default=3.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=2000)
    return p.parse_args()


def _smooth(y, w=7):
    if len(y) < w or w < 2:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _swings(y, prom):
    n = len(y)
    if n < 3:
        return []
    ext, ext_idx, ext_val, up = [], 0, y[0], None
    for i in range(1, n):
        v = y[i]
        if up is None:
            if v > ext_val + 1e-6:
                up = True
            elif v < ext_val - 1e-6:
                up = False
            ext_val, ext_idx = v, i
            continue
        if up:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif v <= ext_val - prom:
                ext.append((ext_idx, "max"))
                up, ext_val, ext_idx = False, v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif v >= ext_val + prom:
                ext.append((ext_idx, "min"))
                up, ext_val, ext_idx = True, v, i
    return ext


def run_ep(agent, env, base, C, geom, seed, args, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    S = {k: [] for k in ("rng", "asp", "ata", "gq", "clo", "lead", "alt", "balt", "dh",
                         "th", "phi", "spd", "bspd", "horiz")}
    best_gq, bgq = -1.0, (float("nan"),) * 3
    fire_cmds = 0
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        # The agent acts in [-1,1]; `map_tdmpc_action` rescales the trigger to native [0,1],
        # so the gun fires (native >= 0.5) exactly when the AGENT's trigger >= 0 — not 0.5.
        if a[4] >= 0.0:
            fire_cmds += 1
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rg = geom.relative(me, bd)
        S["rng"].append(rg.rng); S["asp"].append(rg.aspect * DEG); S["ata"].append(rg.ata * DEG)
        S["gq"].append(rg.gun_quality); S["clo"].append(rg.closure)
        S["lead"].append(rg.lead_error * DEG)
        S["alt"].append(me.alt); S["balt"].append(bd.alt); S["dh"].append(me.alt - bd.alt)
        S["th"].append(me.theta * DEG); S["phi"].append(me.phi * DEG)
        S["spd"].append(me.airspeed); S["bspd"].append(bd.airspeed)
        S["horiz"].append(float(math.hypot(me.pos[0], me.pos[1])))
        if rg.gun_quality > best_gq:
            best_gq = rg.gun_quality
            bgq = (rg.rng, rg.closure, rg.lead_error * DEG)
        if term or trunc:
            break
    A = {k: np.asarray(v) for k, v in S.items()}
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]

    eng = (A["rng"] < args.engage_rng) & (np.abs(A["phi"]) > args.bank_deg)
    ne = int(eng.sum())
    out = dict(
        event=event, dealt=C.HP_MAX - float(bd.health), taken=C.HP_MAX - float(me.health),
        steps=len(A["rng"]), fire=fire_cmds, best_gq=best_gq,
        gq_rng=bgq[0], gq_clo=bgq[1], gq_lead=bgq[2],
        alt0=float(A["alt"][0]), max_alt=float(A["alt"].max()), min_alt=float(A["alt"].min()),
        alt_amp=float(A["alt"].max() - A["alt"].min()),
        max_dh=float(A["dh"].max()), min_dh=float(A["dh"].min()),
        end_horiz=float(A["horiz"][-1]), end_alt=float(A["alt"][-1]),
        min_rng=float(A["rng"].min()), min_asp=float(A["asp"].min()),
        bmin_v=float(A["bspd"].min()),
        eng=ne, climb=float("nan"), dive=float("nan"), flat=float("nan"),
        vamp=float("nan"), bank95=float(np.percentile(np.abs(A["phi"]), 95)),
        eng_alt_amp=float("nan"), th_max=float(A["th"].max()), th_min=float(A["th"].min()),
        hi=0, lo=0, hi_c=0, lo_c=0, rolls=0,
    )
    if ne >= 3:
        th = A["th"][eng]
        out["climb"] = float(np.mean(th > args.pitch_deg))
        out["dive"] = float(np.mean(th < -args.pitch_deg))
        out["flat"] = float(np.mean(np.abs(th) <= args.pitch_deg))
        out["vamp"] = float(np.std(A["dh"][eng]))
        out["eng_alt_amp"] = float(A["alt"][eng].max() - A["alt"][eng].min())
    # yoyo swings on the vertical separation, credited only while engaged+banked
    cs = max(1, int(args.conv_s / dt))
    for idx, kind in _swings(_smooth(A["dh"]), args.prom_m):
        if idx >= len(A["dh"]) or not eng[idx]:
            continue
        j = min(len(A["dh"]), idx + cs)
        conv = (A["ata"][idx:j].min() <= A["ata"][idx] - 8.0) or (A["gq"][idx:j].max() >= A["gq"][idx] + 0.05)
        if kind == "max":
            out["hi"] += 1; out["hi_c"] += int(conv)
        else:
            out["lo"] += 1; out["lo_c"] += int(conv)
    # roll reversals: sign flips of bank between excursions beyond +-bank_deg
    sign, rolls = 0, 0
    for v in A["phi"]:
        if v > args.bank_deg:
            s = 1
        elif v < -args.bank_deg:
            s = -1
        else:
            continue
        if sign != 0 and s != sign:
            rolls += 1
        sign = s
    out["rolls"] = rolls
    return out


def main():
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight import geometry as geom

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    kw = dict(posture="offensive", aspect_deg=args.aspect, separation_m=args.sep,
              action_repeat=args.action_repeat, frame_stack=1, max_time_s=args.max_time_s,
              rate_hz=args.rate_hz)
    if args.jitter > 0:
        kw["jitter"] = args.jitter
    if args.opp_heading is not None:
        kw["opp_heading_deg"] = args.opp_heading
    if args.me_dalt_ft:
        kw["me_dalt_ft"] = args.me_dalt_ft

    print(f"VERTICAL PROBE | aspect {args.aspect:.0f} opp_hdg {args.opp_heading} sep {args.sep:.0f}m "
          f"{args.max_time_s:.0f}s greedy jitter {args.jitter} | R_BUBBLE={C.R_BUBBLE:.0f}m "
          f"CEILING_AGL={C.CEILING_AGL:.0f}m DECK_AGL={C.DECK_AGL:.0f}m")
    print(f"engaged := rng<{args.engage_rng:.0f}m & |bank|>{args.bank_deg:.0f}deg | "
          f"pitch deadband {args.pitch_deg:.0f}deg | yoyo prominence {args.prom_m:.0f}m")

    for tok in args.checkpoints.split(","):
        path, _, label = tok.strip().partition("=")
        label = label or path.split("/")[-1]
        sd = torch.load(path, map_location=device, weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"]); cfg.mpc = False
        agent = TDMPC2(cfg, device); agent.load_state_dict(sd); agent.model.eval()
        env = make_tdmpc_env(3, **kw)
        base = env.unwrapped
        rows = [run_ep(agent, env, base, C, geom, args.seed + ep, args, dt) for ep in range(args.episodes)]
        env.close()
        print(f"\n### {label} (collected={int(sd.get('collected', -1)):,})")
        print(f"{'ep':>3} {'event':>11} {'dlt':>4} {'alt0':>5} {'maxAlt':>6} {'minAlt':>6} {'altAmp':>6} "
              f"{'maxDH':>6} {'minDH':>6} {'engAmp':>6} {'eng':>4} {'clm%':>4} {'div%':>4} {'flt%':>4} "
              f"{'vamp':>5} {'bnk95':>5} {'th+':>4} {'th-':>4} {'hi':>2} {'lo':>2} {'rol':>3} "
              f"{'endHz':>6} {'bGQ':>4} {'gqR':>5} {'gqC':>4} {'fire':>4}")
        for i, r in enumerate(rows):
            def pc(x):
                return f"{100 * x:4.0f}" if x == x else "  --"
            print(f"{i:>3} {str(r['event']):>11} {r['dealt']:>4.0f} {r['alt0']:>5.0f} {r['max_alt']:>6.0f} "
                  f"{r['min_alt']:>6.0f} {r['alt_amp']:>6.0f} {r['max_dh']:>6.0f} {r['min_dh']:>6.0f} "
                  f"{r['eng_alt_amp'] if r['eng_alt_amp'] == r['eng_alt_amp'] else float('nan'):>6.0f} "
                  f"{r['eng']:>4} {pc(r['climb'])} {pc(r['dive'])} {pc(r['flat'])} "
                  f"{r['vamp']:>5.0f} {r['bank95']:>5.0f} {r['th_max']:>4.0f} {r['th_min']:>4.0f} "
                  f"{r['hi']:>2} {r['lo']:>2} {r['rolls']:>3} {r['end_horiz']:>6.0f} "
                  f"{r['best_gq']:>4.2f} {r['gq_rng']:>5.0f} {r['gq_clo']:>4.0f} {r['fire']:>4}")
        ok = [r for r in rows if r["eng"] >= 3]
        def mn(k, src=None):
            src = src if src is not None else rows
            v = [r[k] for r in src if r[k] == r[k]]
            return float(np.mean(v)) if v else float("nan")
        print(f"  AGG {label}: altAmp {mn('alt_amp'):.0f}m | engaged-altAmp {mn('eng_alt_amp', ok):.0f}m | "
              f"maxDH {mn('max_dh'):.0f}m minDH {mn('min_dh'):.0f}m | "
              f"climb {100*mn('climb', ok):.0f}% dive {100*mn('dive', ok):.0f}% flat {100*mn('flat', ok):.0f}% | "
              f"vamp {mn('vamp', ok):.0f}m bank95 {mn('bank95'):.0f}deg | "
              f"hi/ep {mn('hi'):.2f} lo/ep {mn('lo'):.2f} rolls/ep {mn('rolls'):.1f} | "
              f"bestGQ {mn('best_gq'):.2f} minRng {mn('min_rng'):.0f}m bandMinV {mn('bmin_v'):.0f} "
              f"fire/ep {mn('fire'):.0f} | engSteps {mn('eng'):.0f}/{mn('steps'):.0f}")


if __name__ == "__main__":
    main()
