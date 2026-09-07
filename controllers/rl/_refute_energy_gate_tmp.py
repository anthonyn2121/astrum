"""REFUTATION probe for the "fold the engagement gate into the Es potential" candidate.

Measures, per sim tick, on real greedy rollouts:
  (a) the DELETED GAIN the candidate claims to restore:
        sum over ticks of (raw_energy - energy_r) + (raw_rel - rel_energy_r)
      i.e. exactly the positive Es / rel-Es increments that reward.py:316/:371 zero out
      because rng > TRACK_RANGE.
  (b) the candidate's OWN new form's delta-J vs shipped, with w(rng) ramping 1 -> 0
      over TRACK_RANGE -> TRACK_RANGE + RAMP, both undiscounted and discounted at
      gamma=0.995 per CONTROL step.
  (c) the boundary swing the range-coupling introduces (max |new term| per control step)
      and the range profile (frac ticks outside the gate, max range).
"""
import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import env as _denv  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

GAM = 0.995
REPEAT = 5
RAMP = 1500.0
REC = []
_orig_step = _denv.DogfightGymEnv.step


def _es(a, g):
    return max(0.0, a.alt - g) + a.airspeed * a.airspeed / (2.0 * 9.80665)


def _es_phi(a, g):
    return min(_es(a, g), C.ENERGY_REF * 1.5) / C.ENERGY_REF


def _rel_phi(m, b, g):
    return max(-1.0, min(1.0, (_es(m, g) - _es(b, g)) / C.ENERGY_REF))


def _w(rng):
    return max(0.0, min(1.0, (C.TRACK_RANGE + RAMP - rng) / RAMP))


def _patched_step(self, action):
    me_prev = self._states[self.learner_id]
    bd_prev = self._states[self.opponent_id]
    out = _orig_step(self, action)
    obs, rew, term, trunc, info = out
    me = self._states[self.learner_id]
    bd = self._states[self.opponent_id]
    g = self.ground_alt
    row = {
        "reward": float(rew),
        "event": info.get("event"),
        "energy_reward": float(info.get("energy_reward", 0.0)),
        "rel_energy_reward": float(info.get("rel_energy_reward", 0.0)),
    }
    if me_prev.alive and bd_prev.alive and me.alive and bd.alive:
        rg = relative(me, bd)
        prg = relative(me_prev, bd_prev)
        row["rng"] = rg.rng
        row["prng"] = prg.rng
        row["engaged"] = bool(rg.rng <= C.TRACK_RANGE)
        pe, ppe = _es_phi(me, g), _es_phi(me_prev, g)
        pr, ppr = _rel_phi(me, bd, g), _rel_phi(me_prev, bd_prev, g)
        row["raw_energy"] = C.K_ENERGY * (pe - ppe)
        row["raw_rel"] = C.K_REL_ENERGY * (pr - ppr)
        row["new_energy"] = C.K_ENERGY * (_w(rg.rng) * pe - _w(prg.rng) * ppe)
        row["new_rel"] = C.K_REL_ENERGY * (_w(rg.rng) * pr - _w(prg.rng) * ppr)
        row["phiE"] = pe
        row["phiR"] = pr
    else:
        for k in ("rng", "prng", "raw_energy", "raw_rel", "new_energy", "new_rel", "phiE", "phiR"):
            row[k] = float("nan")
        row["engaged"] = False
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
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--lateral", type=float, default=1400.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--aspect", type=float, default=0.0)
    ap.add_argument("--opp-heading", type=float, default=180.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = resolve_device("auto")
    agent = load_agent(args.checkpoint, dev)
    kw = dict(posture="offensive", opponent_tier=0, aspect_deg=args.aspect,
              opp_heading_deg=args.opp_heading, separation_m=args.sep,
              action_repeat=REPEAT, frame_stack=1, max_time_s=args.max_time_s,
              rate_hz=120.0, outcome_mode=True, jitter=args.jitter)
    if args.lateral:
        kw["lateral_m_max"] = args.lateral
    env = make_tdmpc_env(3, **kw)

    agg = {}
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
        disc = np.array([GAM ** (i // REPEAT) for i in range(n)])
        arr = lambda k: np.array([rows[i].get(k, 0.0) for i in range(n)], dtype=float)  # noqa: E731
        rng = arr("rng")
        eng = np.array([bool(rows[i]["engaged"]) for i in range(n)])
        raw_e, raw_r = arr("raw_energy"), arr("raw_rel")
        ship_e, ship_r = arr("energy_reward"), arr("rel_energy_reward")
        new_e, new_r = arr("new_energy"), arr("new_rel")
        raw_e, raw_r, new_e, new_r = (np.nan_to_num(x) for x in (raw_e, raw_r, new_e, new_r))
        # (a) deleted gain: what the increment-level gate zeroes out
        del_e = raw_e - ship_e
        del_r = raw_r - ship_r
        deleted = del_e + del_r
        # (b) the candidate's delta-J
        dJ = (new_e + new_r) - (ship_e + ship_r)
        # (c) boundary swing per control step
        cs = np.add.reduceat(new_e + new_r, np.arange(0, n, REPEAT)) if n else np.array([0.0])
        outside = (~eng) & np.isfinite(rng)
        ev = [r["event"] for r in rows if r["event"]]
        print(f"\n=== ep {ep} n={n} ({n/120:.1f}s) event={ev[-1] if ev else None} "
              f"ret={arr('reward').sum():+.2f}")
        print(f"  rng: max {np.nanmax(rng):7.0f}  frac ticks OUTSIDE gate (>3000m) {outside.mean():.4f} "
              f"({int(outside.sum())} ticks)  maxPhiE {np.nanmax(arr('phiE')):.3f}")
        print(f"  (a) DELETED GAIN  energy {del_e.sum():+8.3f}  rel {del_r.sum():+8.3f}  "
              f"TOTAL {deleted.sum():+8.3f}   (ticks affected {int((deleted != 0).sum())})")
        print(f"      charged-in-full losses at rng>3000: energy {raw_e[outside][raw_e[outside] < 0].sum():+8.3f} "
              f" rel {raw_r[outside][raw_r[outside] < 0].sum():+8.3f}")
        print(f"  (b) candidate delta-J undisc {dJ.sum():+9.3f}   disc {(dJ*disc).sum():+9.3f}")
        print(f"      shipped E+relE undisc {(ship_e+ship_r).sum():+9.3f}   "
              f"new form undisc {(new_e+new_r).sum():+9.3f}")
        print(f"  (c) per-CONTROL-STEP new term: max {cs.max():+8.3f}  min {cs.min():+8.3f}  "
              f"|>2| in {int((np.abs(cs) > 2).sum())} of {len(cs)} steps")
        for k, v in [("deleted", deleted.sum()), ("dJ", dJ.sum()), ("dJ_disc", (dJ*disc).sum()),
                     ("ret", arr("reward").sum()), ("fracOut", outside.mean()),
                     ("maxRng", np.nanmax(rng)), ("csmax", cs.max()), ("csmin", cs.min())]:
            agg.setdefault(k, []).append(float(v))

    print(f"\n==== SUMMARY {args.tag} sep={args.sep} lat={args.lateral} asp={args.aspect} "
          f"opph={args.opp_heading} eps={args.episodes} ====")
    for k, v in agg.items():
        v = np.array(v)
        print(f"  {k:9s} mean {v.mean():+10.3f}  min {v.min():+10.3f}  max {v.max():+10.3f}")


if __name__ == "__main__":
    main()
