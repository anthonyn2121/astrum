"""R5 threshold probe: is `|bank| <= CONVERT_BANK_RAD` the right SETTLED test for the
convert-dwell box, or should it be an LOS-RATE test?

The convert-dwell box (reward.py, outcome_mode only) pays K_CONVERT_DWELL per tick inside

    aspect <= CONVERT_ASPECT_RAD  and  ata <= CONVERT_ATA_RAD
    and |me.phi| <= CONVERT_BANK_RAD          <-- the term under test
    and DWELL_RANGE_MIN <= rng <= CONVERT_RANGE_MAX

The bank term came from reward-v18, where it meant "the frozen finisher F inherits a
FLYABLE state, not a 100-degree knife edge" — a two-policy hand-off that is now dead
(convert_bonus defaults to 0, so the hand-off itself is inert; only this dwell box still
reads CONVERT_BANK_RAD). Its side effect on the surviving single-policy chase is that the
dwell reward is DENIED for exactly the hard-banked turn a high-aspect entry demands — the
native T3's own break rides 85 degrees of bank.

What we actually want "settled" to mean is a STABLE TRACKING SOLUTION, and curriculum-87
measured which scalar carries that: kills arrive at ~18.6 deg/s LOS rate, failures spray
through at ~59 deg/s. LOS rate is also now OBSERVABLE (obs 31/32), so a gate on it is
learnable in a way the old bank gate's proxy never was.

This probe does NOT assume the swap is right. Per tick inside the BANK-FREE box it records
|phi|, the analytic LOS-rate magnitude, lay_miss_m and ballistic hittability, then scores
each candidate gate on the only thing that matters:

    ADMIT%  — how much of the rear-quarter box the gate pays for at all
    HIT%    — of admitted ticks, how many are geometry a round would actually HIT from

A good gate has HIT% well above the ungated baseline (it is selective for shootable
geometry) while admitting enough ticks to be a usable gradient. A gate with high HIT% but
~0 ADMIT% is a reward that never pays; one with baseline HIT% is a gate that discriminates
nothing. Reported per aspect too, because the hypothesis is specifically that the bank gate
starves the 55/60-degree cells.

Run:  cd controllers/rl && uv run python _diag_dwell_gate_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --aspects 0,30,45,55,60,75 --episodes 6 --jitter 0.5
"""

from __future__ import annotations

import argparse
from math import degrees, hypot

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

# Candidate LOS-rate ceilings (deg/s). 18.6 = the measured arrival rate of c86's 8/8 killing
# cell; 58.7 = the measured rate of its 0%-hittable spray band. The bracket spans both.
LOS_GATES_DEG = [15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert-dwell gate probe (bank vs LOS rate).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--aspects", default="0,30,45,55,60,75")
    p.add_argument("--mode", default="greedy", choices=["greedy", "mpc"])
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _roll(agent, env, base, C, geom, sigma_track, seed: int) -> list[dict]:
    """One greedy episode -> the in-box ticks (bank gate NOT applied)."""
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    ticks: list[dict] = []
    t0 = True
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, terminated, truncated, _info = env.step(a)
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rg = geom.relative(me, opp)
        in_box = (
            rg.aspect <= C.CONVERT_ASPECT_RAD
            and rg.ata <= C.CONVERT_ATA_RAD
            and C.DWELL_RANGE_MIN <= rg.rng <= C.CONVERT_RANGE_MAX
        )
        if in_box:
            ticks.append(
                {
                    "phi": abs(degrees(me.phi)),
                    "omega": degrees(rg.los_rate),
                    "miss": rg.lay_miss_m,
                    "rng": rg.rng,
                    "hittable": bool(rg.in_range and rg.lead_error <= sigma_track(rg.rng)),
                }
            )
        if terminated or truncated:
            break
    return ticks


def _score(ticks: list[dict], keep) -> tuple[int, float, float, float]:
    """(admitted, admit%, hit% among admitted, mean miss m among admitted)."""
    adm = [t for t in ticks if keep(t)]
    if not adm:
        return 0, 0.0, float("nan"), float("nan")
    hit = sum(t["hittable"] for t in adm)
    return (
        len(adm),
        100.0 * len(adm) / len(ticks),
        100.0 * hit / len(adm),
        float(np.mean([t["miss"] for t in adm])),
    )


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight import geometry as geom
    from astrum_dogfight.geometry import sigma_track

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = args.mode == "mpc"
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    print(
        f"DWELL-GATE PROBE {args.checkpoint} "
        f"(recorded collected={int(sd.get('collected', -1)):,}) | mode={args.mode} | "
        f"vs native T3 | sep {args.separation:.0f}m jitter{args.jitter:g} | "
        f"{args.episodes} eps/aspect"
    )
    print(
        f"box: aspect<={degrees(C.CONVERT_ASPECT_RAD):.0f} ata<={degrees(C.CONVERT_ATA_RAD):.0f} "
        f"rng {C.DWELL_RANGE_MIN:.0f}-{C.CONVERT_RANGE_MAX:.0f}m | current gate "
        f"|phi|<={degrees(C.CONVERT_BANK_RAD):.0f} | hittable = lead<=sigma_track(rng)"
    )

    per_aspect: dict[float, list[dict]] = {}
    for aspect in aspects:
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=aspect,
            separation_m=args.separation,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
        )
        base = env.unwrapped
        ticks: list[dict] = []
        for ep in range(args.episodes):
            ticks.extend(
                _roll(
                    agent, env, base, C, geom, sigma_track,
                    seed=args.seed + int(aspect) * 1000 + ep,
                )
            )
        env.close()
        per_aspect[aspect] = ticks
        print(f"  aspect {aspect:>3.0f}: {len(ticks):>5} in-box ticks", flush=True)

    pooled = [t for ts in per_aspect.values() for t in ts]
    if not pooled:
        print("\nNO in-box ticks at all — the rear-quarter box is never entered; "
              "the bank gate is not the binding constraint.")
        return

    # ---- (a) how much does the CURRENT bank gate deny, per aspect? ----
    print("\n(a) CURRENT bank gate, per aspect — 'denied' ticks are rear-quarter gun-band")
    print("    geometry the dwell reward refuses to pay for:")
    print(f"{'aspect':>6} {'in-box':>6} {'pass bank':>9} {'admit%':>6} {'hit% all':>8} "
          f"{'hit% pass':>9} {'hit% denied':>11} {'mean|phi|':>9} {'mean w':>7}")
    for aspect in aspects:
        ts = per_aspect[aspect]
        if not ts:
            print(f"{aspect:>6.0f} {0:>6}          —      —        —         —           — "
                  f"        —       —")
            continue
        n, admit, hit_pass, _ = _score(ts, lambda t: t["phi"] <= degrees(C.CONVERT_BANK_RAD))
        denied = [t for t in ts if t["phi"] > degrees(C.CONVERT_BANK_RAD)]
        hit_all = 100.0 * sum(t["hittable"] for t in ts) / len(ts)
        hit_den = 100.0 * sum(t["hittable"] for t in denied) / len(denied) if denied else float("nan")
        print(
            f"{aspect:>6.0f} {len(ts):>6} {n:>9} {admit:>5.0f}% {hit_all:>7.0f}% "
            f"{hit_pass:>8.0f}% {hit_den:>10.0f}% {np.mean([t['phi'] for t in ts]):>9.0f} "
            f"{np.mean([t['omega'] for t in ts]):>7.1f}"
        )

    # ---- (b) candidate gates on the POOLED ticks ----
    base_hit = 100.0 * sum(t["hittable"] for t in pooled) / len(pooled)
    print(f"\n(b) POOLED {len(pooled)} in-box ticks | UNGATED hit% = {base_hit:.0f}% "
          f"(the baseline every gate must beat)")
    print(f"{'gate':>22} {'admitted':>8} {'admit%':>6} {'hit%':>5} {'lift':>6} {'mean miss m':>11}")

    def row(name, keep):
        n, admit, hit, miss = _score(pooled, keep)
        lift = hit - base_hit
        print(f"{name:>22} {n:>8} {admit:>5.0f}% {hit:>4.0f}% {lift:>+5.0f} {miss:>11.1f}")

    row(f"|phi|<={degrees(C.CONVERT_BANK_RAD):.0f} (CURRENT)", lambda t: t["phi"] <= degrees(C.CONVERT_BANK_RAD))
    row("no gate", lambda _t: True)
    for g in LOS_GATES_DEG:
        row(f"omega<={g:.0f} deg/s", lambda t, g=g: t["omega"] <= g)
    # the conjunction, in case bank still adds something on top of a LOS gate
    for g in (20.0, 30.0):
        row(
            f"omega<={g:.0f} AND bank",
            lambda t, g=g: t["omega"] <= g and t["phi"] <= degrees(C.CONVERT_BANK_RAD),
        )

    # ---- (c) is omega monotone in hittability? the mechanism check ----
    print("\n(c) hittability vs LOS rate (pooled in-box ticks) — if the gate idea is sound this")
    print("    must fall monotonically:")
    print(f"{'omega band':>14} {'ticks':>6} {'hit%':>5} {'mean miss m':>11} {'mean rng':>8} {'mean|phi|':>9}")
    edges = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90), (90, 1e9)]
    for lo, hi in edges:
        rec = [t for t in pooled if lo <= t["omega"] < hi]
        if not rec:
            continue
        label = f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f"{lo:.0f}+"
        print(
            f"{label:>14} {len(rec):>6} "
            f"{100.0 * sum(t['hittable'] for t in rec) / len(rec):>4.0f}% "
            f"{np.mean([t['miss'] for t in rec]):>11.1f} "
            f"{np.mean([t['rng'] for t in rec]):>8.0f} {np.mean([t['phi'] for t in rec]):>9.0f}"
        )

    # ---- (d) same for bank, as the control: does bank predict hittability AT ALL? ----
    print("\n(d) CONTROL — hittability vs |bank|. If this is FLAT, the bank gate is selecting")
    print("    on a quantity unrelated to whether the shot hits:")
    print(f"{'|phi| band':>14} {'ticks':>6} {'hit%':>5} {'mean miss m':>11} {'mean w':>7}")
    for lo, hi in [(0, 30), (30, 60), (60, 80), (80, 1e9)]:
        rec = [t for t in pooled if lo <= t["phi"] < hi]
        if not rec:
            continue
        label = f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f"{lo:.0f}+"
        print(
            f"{label:>14} {len(rec):>6} "
            f"{100.0 * sum(t['hittable'] for t in rec) / len(rec):>4.0f}% "
            f"{np.mean([t['miss'] for t in rec]):>11.1f} "
            f"{np.mean([t['omega'] for t in rec]):>7.1f}"
        )


if __name__ == "__main__":
    main()
