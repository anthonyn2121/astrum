"""Phase-2 verdict: does the doctrine BFMCoach convert a HEAD-ON merge into a kill vs a LIVE T3 —
with NO training at all?

    cd controllers/rl && uv run python _bfm_coach_tmp.py --aspects 0 --opp-heading 180 --opp-tier 0

The pure-pursuit policy program (c33-c61) could never win the neutral head-on merge: it demands a
flat turn tighter than the sustainable radius, dumps energy, and decks (or goes passive). The
user's thesis is that a real fighter wins the merge with BFM — vertical reversals + yo-yos + rolls
— to get from a nose-to-nose pass to the bandit's rear quarter, then guns him. `astrum_dogfight.bfm`
implements those maneuvers as closed-loop stick controllers; `BFMCoach` picks among them from the
live geometry/energy. This harness flies the coach (NOT a trained policy) against the SAME native
scripted Tier-3 bot `_verdict_tmp.py` uses, from a head-on IC, and reports whether it converts +
kills — the go/no-go for Phase 3 (clone the coach into the policy + RL-refine on an outcome reward).

The coach acts on the raw AState/RelGeom (read from `env.unwrapped._states`), so this does not load
any checkpoint. Per episode it also traces the ORDERED maneuver sequence the coach flew, the deepest
rear-quarter it reached (min aspect while in a firing band), min altitude (deck check), peak
gun_quality, and rounds fired — so we can see what the doctrine actually does, not just the score.
Runs on CPU by default.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.eval_policy import _classify

_REAR_ASPECT_DEG = 60.0  # aspect below this (in a firing band) = we made it to his rear quarter
_FIRE_BAND_M = 1500.0  # "in a firing band" range ceiling for the convert test


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BFMCoach vs native T3 from a head-on merge (no training).")
    p.add_argument("--aspects", default="0", help="comma list of nominal start aspects (0 = dead-ahead merge)")
    p.add_argument("--episodes", type=int, default=12, help="episodes per aspect")
    p.add_argument("--separation", type=float, default=1200.0)
    p.add_argument(
        "--opp-heading",
        type=lambda s: None if str(s).lower() == "none" else float(s),
        default=180.0,
        help="opponent ABSOLUTE start heading deg; 180 = head-on merge; 'none' = co-heading chase (env default)",
    )
    p.add_argument(
        "--opp-tier",
        type=int,
        default=0,
        help="0/3 = full scripted Tier-3 shooter (north-star); 1 = straight non-shooter; 2 = breaker",
    )
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=120.0, help="BFM needs room to reverse+convert+finish")
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="print per-episode maneuver trace")
    return p.parse_args(argv)


def _roll(env, base, C, episodes: int, seed0: int, verbose: bool) -> list[dict]:
    from astrum_dogfight.bfm import BFMCoach
    from astrum_dogfight.geometry import relative

    rows: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        coach = BFMCoach()
        seq: list[str] = []  # ordered maneuvers flown (dedup consecutive)
        min_aspect = 180.0
        min_aspect_rng = float("nan")
        min_alt = float("inf")
        max_q = 0.0
        firing_pos_steps = 0  # SUSTAINED real gun-track position (behind + nose-on + in band)
        event = None
        terminated = truncated = False
        while True:
            me = base._states[base.learner_id]
            bandit = base._states[base.opponent_id]
            rel = relative(me, bandit)
            a = coach.act(me, rel)
            nm = coach.active_name
            if not seq or seq[-1] != nm:
                seq.append(nm)
            # deepest rear-quarter reached while in a firing band (the conversion signal)
            import math as _m
            asp_deg = _m.degrees(float(rel.aspect))
            ata_deg = _m.degrees(float(rel.ata))
            if bandit.alive and rel.rng <= _FIRE_BAND_M and asp_deg < min_aspect:
                min_aspect = asp_deg
                min_aspect_rng = float(rel.rng)
            # A genuine conversion = SUSTAINED rear-quarter gun-track, not the instantaneous aspect
            # dip during the merge cross-tangle: behind him (aspect<50) + nose on (ata<35) + in band.
            if bandit.alive and asp_deg < 50.0 and ata_deg < 35.0 and 200.0 <= rel.rng <= 1000.0:
                firing_pos_steps += 1
            min_alt = min(min_alt, float(me.alt))
            max_q = max(max_q, float(rel.gun_quality))
            obs, _r, terminated, truncated, info = env.step(a)
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        dealt = C.HP_MAX - float(opp.health)
        taken = C.HP_MAX - float(me.health)
        outcome = _classify(event, terminated, truncated)
        converted = firing_pos_steps >= 12  # ~0.5 s held in a real rear-quarter gun-track position
        # compress the maneuver sequence for display (collapse runs)
        rows.append(
            {
                "outcome": outcome,
                "event": event,
                "dealt": dealt,
                "taken": taken,
                "net": dealt - taken,
                "converted": converted,
                "min_aspect": min_aspect,
                "min_aspect_rng": min_aspect_rng,
                "min_alt": min_alt,
                "max_q": max_q,
                "ammo_spent": float(C.AMMO_MAX) - float(me.ammo),
                "end_alt": float(me.alt),
                "seq": seq,
            }
        )
        if verbose:
            conv = "CONV" if converted else "----"
            print(
                f"§ ep{ep:>2} {outcome:>8} {conv} | minAsp {min_aspect:>3.0f}@{min_aspect_rng:>5.0f}m "
                f"fpos {firing_pos_steps:>4} minAlt {min_alt:>5.0f} peakQ {max_q:>4.2f} "
                f"fired {int(float(C.AMMO_MAX) - float(me.ammo)):>4} net {dealt - taken:>+5.0f} | {' > '.join(seq[:14])}"
            )
    return rows


def main() -> None:
    args = parse_args(argv=None)
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]

    print(
        f"§ BFM-COACH VERDICT (NO TRAINING) vs T{args.opp_tier or 3} | opp_hdg "
        f"{'co-head' if args.opp_heading is None else format(args.opp_heading, '.0f') + '°'} "
        f"sep {args.separation:.0f}m mt{args.max_time_s:.0f}s jitter{args.jitter:g} | {args.episodes} eps/aspect"
    )
    print(
        f"§ {'aspect':>6} {'N':>3} {'kill':>4} {'death':>5} {'deck':>4} {'trade':>5} {'t/out':>5} "
        f"{'conv%':>6} {'minAsp':>7} {'dealt':>6} {'taken':>6} {'NET':>7} {'spent':>6}"
    )
    overall: list[dict] = []
    seq_hist: Counter = Counter()
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
            **({"opp_heading_deg": args.opp_heading} if args.opp_heading is not None else {}),
            **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
        )
        base = env.unwrapped
        rows = _roll(env, base, C, args.episodes, args.seed + int(aspect) * 1000, args.verbose)
        env.close()
        overall.extend(rows)
        for r in rows:
            seq_hist.update(r["seq"])
        n = len(rows)
        kill = sum(r["outcome"] == "kill" for r in rows)
        death = sum(r["outcome"] == "death" for r in rows)
        deck = sum(r["outcome"] == "deck" for r in rows)
        trade = sum(r["event"] == "trade" for r in rows)
        tout = sum(r["outcome"] == "timeout" for r in rows)
        conv = 100.0 * statistics.fmean(r["converted"] for r in rows)
        min_asp = statistics.fmean(r["min_aspect"] for r in rows)
        dealt = statistics.fmean(r["dealt"] for r in rows)
        taken = statistics.fmean(r["taken"] for r in rows)
        net = statistics.fmean(r["net"] for r in rows)
        spent = statistics.fmean(r["ammo_spent"] for r in rows)
        print(
            f"§ {aspect:>6.0f} {n:>3} {kill:>4} {death:>5} {deck:>4} {trade:>5} {tout:>5} "
            f"{conv:>5.0f}% {min_asp:>7.0f} {dealt:>6.1f} {taken:>6.1f} {net:>+7.1f} {spent:>6.0f}"
        )
    n = len(overall)
    kill = sum(r["outcome"] == "kill" for r in overall)
    loss = sum(r["outcome"] in ("death", "deck") for r in overall)
    conv = 100.0 * statistics.fmean(r["converted"] for r in overall)
    net = statistics.fmean(r["net"] for r in overall)
    print(
        f"§ ALL: {n} eps | kills {kill} losses {loss} | converted {conv:.0f}% | mean NET {net:+.1f}"
    )
    print("§ maneuver usage (episode-entries):", dict(seq_hist.most_common()))


if __name__ == "__main__":
    main()
