"""INDEPENDENT refutation probe: measure the ACTUAL per-episode magnitude and the
POLICY-DIFFERENTIAL of the alleged one-sided-gate ratchet, on real sim rollouts of the
current champion on the merge rungs the claim targets.

Records EVERY sim tick's reward decomposition (not the last-of-5 that ActionRepeat leaks),
plus the counterfactual PATCH-B value of lead_r, and the control-step discount index.
"""
import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import env as _denv  # noqa: E402
from astrum_dogfight import reward as _reward  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DEG = 180.0 / math.pi
GAM = 0.995
REPEAT = 5

TERMS = ["close_reward", "aim_reward", "aspect_reward", "energy_reward",
         "rel_energy_reward", "band_reward", "lead_reward", "convert_dwell",
         "track_reward", "corner_penalty"]

REC = []


def _phi_lead(g):
    return (1.0 + math.cos(g.lead_error)) * 0.5


_orig_step = _denv.DogfightGymEnv.step


def _patched_step(self, action):
    me_prev = self._states[self.learner_id]
    bd_prev = self._states[self.opponent_id]
    out = _orig_step(self, action)
    obs, rew, term, trunc, info = out
    me = self._states[self.learner_id]
    bd = self._states[self.opponent_id]
    row = {k: float(info.get(k, 0.0)) for k in TERMS}
    row["reward"] = float(rew)
    row["event"] = info.get("event")
    if me_prev.alive and bd_prev.alive and me.alive and bd.alive:
        rg = relative(me, bd)
        prg = relative(me_prev, bd_prev)
        row["rng"] = rg.rng
        row["aspect"] = rg.aspect
        row["lead"] = rg.lead_error
        row["in_zone"] = bool(rg.aspect <= C.CONVERT_ASPECT_RAD and rg.rng <= C.CONVERT_RANGE_MAX)
        row["raw_lead"] = C.K_LEAD * (_phi_lead(rg) - _phi_lead(prg))
        raw_close = C.K_CLOSE * (
            -min(max(rg.rng, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE
            - -min(max(prg.rng, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE
        )
        row["raw_close"] = raw_close
        row["closure"] = rg.closure
        row["engaged"] = bool(rg.rng <= C.TRACK_RANGE)
    else:
        row["rng"] = float("nan")
        row["in_zone"] = False
        row["raw_lead"] = 0.0
        row["raw_close"] = 0.0
        row["engaged"] = False
        row["aspect"] = float("nan")
        row["lead"] = float("nan")
        row["closure"] = float("nan")
    REC.append(row)
    return out


_denv.DogfightGymEnv.step = _patched_step


def load_agent(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--sep", type=float, default=2200.0)
    ap.add_argument("--lateral", type=float, default=1400.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--aspect", type=float, default=0.0)
    ap.add_argument("--opp-heading", type=float, default=180.0)
    args = ap.parse_args()

    dev = resolve_device("auto")
    agent = load_agent(args.checkpoint, dev)

    kw = dict(posture="offensive", opponent_tier=0, aspect_deg=args.aspect,
              opp_heading_deg=args.opp_heading,
              separation_m=args.sep, action_repeat=REPEAT, frame_stack=1,
              max_time_s=args.max_time_s, rate_hz=120.0, outcome_mode=True,
              jitter=args.jitter)
    if args.lateral:
        kw["lateral_m_max"] = args.lateral
    env = make_tdmpc_env(3, **kw)

    tot = {}
    for ep in range(args.episodes):
        REC.clear()
        obs, _ = env.reset(seed=1000 + ep)
        agent._prev_mean.zero_()
        t0 = True
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows = list(REC)
        n = len(rows)
        # discount index = control step
        disc = np.array([GAM ** (i // REPEAT) for i in range(n)])
        s = {}
        for k in TERMS:
            v = np.array([r[k] for r in rows])
            s[k] = (v.sum(), (v * disc).sum())
        raw_lead = np.array([r["raw_lead"] for r in rows])
        lead = np.array([r["lead_reward"] for r in rows])
        inz = np.array([r["in_zone"] for r in rows])
        # patch B counterfactual: lead_r = raw_lead if in_zone else 0
        patchB_lead = np.where(inz, raw_lead, 0.0)
        d_lead = patchB_lead - lead
        raw_close = np.array([r["raw_close"] for r in rows])
        close = np.array([r["close_reward"] for r in rows])
        d_close = raw_close - close  # patch B item 2 restores raw_close
        rng = np.array([r["rng"] for r in rows])
        totrew = np.array([r["reward"] for r in rows])
        ev = [r["event"] for r in rows if r["event"]]
        print(f"\n=== ep {ep}  n_ticks={n} ({n/120:.1f}s)  event={ev[-1] if ev else None} "
              f"return={totrew.sum():+.2f} disc={(totrew*disc).sum():+.2f}")
        print(f"  minRng {np.nanmin(rng):7.0f}  frac in_zone {inz.mean():.3f}  "
              f"ticks in_zone {int(inz.sum())}")
        for k in TERMS:
            if abs(s[k][0]) > 1e-9:
                print(f"  {k:20s} undisc {s[k][0]:+9.2f}   disc {s[k][1]:+9.2f}")
        print(f"  --- lead ratchet ---")
        print(f"  lead_reward shipped   undisc {lead.sum():+9.2f}  disc {(lead*disc).sum():+9.2f}")
        print(f"  lead_reward PATCH-B   undisc {patchB_lead.sum():+9.2f}  disc {(patchB_lead*disc).sum():+9.2f}")
        print(f"  DELTA (patchB-ship)   undisc {d_lead.sum():+9.2f}  disc {(d_lead*disc).sum():+9.2f}")
        oz = ~inz
        print(f"  out-of-zone neg-lead charges: {int(((raw_lead<0)&oz).sum())} ticks, "
              f"sum {raw_lead[(raw_lead<0)&oz].sum():+.2f}")
        print(f"  out-of-zone pos-lead forgone: {int(((raw_lead>0)&oz).sum())} ticks, "
              f"sum {raw_lead[(raw_lead>0)&oz].sum():+.2f}")
        print(f"  net out-of-zone raw_lead (if two-sided) {raw_lead[oz].sum():+.2f}")
        print(f"  --- close gate ---")
        print(f"  close shipped undisc {close.sum():+9.2f}  raw undisc {raw_close.sum():+9.2f}  "
              f"DELTA {d_close.sum():+9.2f} disc {(d_close*disc).sum():+9.2f}")
        for k, v in [("d_lead_disc", (d_lead*disc).sum()), ("d_close_disc", (d_close*disc).sum()),
                     ("ret_disc", (totrew*disc).sum()), ("ret", totrew.sum()),
                     ("d_lead", d_lead.sum()), ("d_close", d_close.sum())]:
            tot.setdefault(k, []).append(v)

    print("\n==== SUMMARY over episodes ====")
    for k, v in tot.items():
        v = np.array(v)
        print(f"  {k:14s} mean {v.mean():+9.2f}  min {v.min():+9.2f}  max {v.max():+9.2f}")


if __name__ == "__main__":
    main()
