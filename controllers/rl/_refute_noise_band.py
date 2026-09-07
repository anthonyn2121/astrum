"""Refutation probe: does colored exploration noise move the MANEUVER-BAND amplitude
of the action the sim ACTUALLY FLIES (and that the buffer stores), or only that of the
injected increment?

Reuses _diag_explore_noise's paired-plan machinery verbatim, but reports the T-window
mean statistics for A_e (collected/executed), A_g (greedy counterfactual) and NZ (noise)
side by side, plus P(|window mean| > D) on the EXECUTED aileron.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import astrum_rl._paths  # noqa: F401
from astrum_dogfight import constants as C  # noqa: F401
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

from _diag_explore_noise import (  # noqa: E402
    OUNoise,
    PinkNoise,
    WhiteNoise,
    acf,
    roll,
    runs_above,
    window_mean_std,
)

CH = ["ail", "elev", "rud", "thr", "trig"]


def wm(x, T):
    x = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[T:] - c[:-T]) / T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--arms", default="white,ou")
    ap.add_argument("--tau-s", type=float, default=1.5)
    ap.add_argument("--separation", type=float, default=3000.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--rate-hz", type=float, default=120.0)
    ap.add_argument("--action-repeat", type=int, default=5)
    ap.add_argument("--me-dalt-ft", type=float, default=2000.0)
    ap.add_argument("--lateral-m", type=float, default=1400.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    print(f"ckpt collected={sd.get('collected', -1):,} explore_noise={getattr(cfg,'explore_noise','?')} "
          f"horizon={cfg.horizon} dt={dt*1000:.0f} ms")
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    env = make_tdmpc_env(
        3, posture="offensive", aspect_deg=0.0, opp_heading_deg=180.0,
        separation_m=args.separation, action_repeat=args.action_repeat, frame_stack=1,
        max_time_s=args.max_time_s, rate_hz=args.rate_hz, outcome_mode=True,
        jitter=args.jitter, me_dalt_ft=args.me_dalt_ft, lateral_m_max=args.lateral_m,
    )
    base = env.unwrapped
    tau_steps = args.tau_s / dt
    gen = torch.Generator()
    gen.manual_seed(args.seed + 4242)

    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        noise = {
            "white": lambda: WhiteNoise(),
            "ou": lambda: OUNoise(tau_steps),
            "pink": lambda: PinkNoise(int(args.max_time_s / dt) + 8, 1.0, seed=args.seed),
        }[name]()
        rows = [roll(agent, env, base, noise, gen, args.seed + 100 * i, dt)
                for i in range(args.episodes)]
        Ae = np.concatenate([r["A_e"] for r in rows])
        Ag = np.concatenate([r["A_g"] for r in rows])
        NZ = Ae - Ag
        print(f"\n===== ARM {name}: steps {len(Ae)} events {[r['event'] for r in rows]} "
              f"dealt {[round(r['dealt'],1) for r in rows]} spent {[int(r['spent']) for r in rows]}")
        for i in (0, 1):
            print(f" {CH[i]}: std  exec {Ae[:,i].std():.3f}  greedy {Ag[:,i].std():.3f}  "
                  f"noise {NZ[:,i].std():.3f}   noise share of var "
                  f"{NZ[:,i].var()/max(1e-9,Ae[:,i].var()):.3f}")
            for T in (12, 36, 72):
                a = window_mean_std(Ae[:, i], T)
                g = window_mean_std(Ag[:, i], T)
                n = window_mean_std(NZ[:, i], T)
                print(f"   T={T:>3}({T*dt:.2f}s) window-mean std: EXEC {a:.3f}  greedy {g:.3f}  noise {n:.3f}")
            # behaviourally relevant: how often does the EXECUTED command hold a large mean?
            for T in (36, 72):
                w = wm(Ae[:, i], T)
                wg = wm(Ag[:, i], T)
                for D in (0.25, 0.5, 0.75):
                    print(f"   P(|exec {T}-step window mean| > {D}) = {np.mean(np.abs(w)>D):.4f}"
                          f"   (greedy {np.mean(np.abs(wg)>D):.4f})")
            L = np.concatenate([runs_above(np.abs(r["A_e"][:, i]), 0.5) for r in rows])
            Lg = np.concatenate([runs_above(np.abs(r["A_g"][:, i]), 0.5) for r in rows])
            print(f"   |a|>0.5 runs (SIGN-AGNOSTIC): exec max {L.max()} ({L.max()*dt:.2f}s) "
                  f"frac>=24 {np.mean(L>=24):.3f} | greedy max {Lg.max()} frac>=24 {np.mean(Lg>=24):.3f}")
            r = acf(Ae[:, i], 120)
            s = 0.0
            for k in range(1, len(r)):
                if r[k] <= 0:
                    break
                s += r[k]
            print(f"   exec tau_int {(1+2*s)*dt:.2f}s  rho12 {r[12]:.3f} rho36 {r[36]:.3f}")
    env.close()


if __name__ == "__main__":
    main()
