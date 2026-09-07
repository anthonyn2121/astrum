"""Two-policy (c60) hand-off verdict: merge-converter M → geometric selector → frozen finisher F.

    cd controllers/rl && uv run python _verdict2p_tmp.py \
        --checkpoint-m runs/curriculum-60/run.pt \
        --checkpoint-f runs/curriculum-46/_snap_2244k.pt \
        --opp-heading 180 --opp-tier 0 --aspects 0 --episodes 12 --jitter 0.5

The deliverable metric for the two-policy architecture (see the curriculum-60 memory): from a
HEAD-ON merge vs the scripted Tier-3, can M survive the pass and CONVERT to the bandit's rear
quarter, and does the FROZEN champion F then FINISH the kill? This rolls the exact same native-T3
env as `_verdict_tmp.py`, but flies TWO greedy policies in one episode with a mid-episode hand-off:

  fly M   until the GEOMETRIC SELECTOR fires — a genuine rear-quarter firing position behind a LIVE
          bandit: aspect ≤ CONVERT_ASPECT_RAD (in his rear quarter) AND my nose-to-LOS ata ≤
          CONVERT_ATA_RAD (actually pointing AT him, not a swept-behind head-on fly-through) AND
          range ∈ [CONVERT_RANGE_MIN, CONVERT_RANGE_MAX] (F's prosecution band) — then LATCH to
  fly F   for the finish, for the rest of the SAME episode.

The selector's condition is read straight from `constants.py` so it matches reward-v17's conversion
TERMINAL exactly (the eval env is built with convert_bonus=0 so the episode does NOT auto-terminate
at conversion — we detect it here and switch, then let F run to a real kill/death/deck/timeout).

Greedy only (cfg.mpc=False): the policy-prior mean carries no MPPI planning state, so switching the
acting policy mid-episode is clean. Reports, per aspect: convert% (M delivered a hand-off) and mean
step@convert, the END-TO-END kills/deaths/decks/timeouts (F's job once handed the jet), conv→kill%
(finish quality on the hand-offs M actually delivered), and mean HP dealt/taken/net. Runs on CPU by
default. This is a throwaway diagnostic (untracked), like `_verdict_tmp.py`.
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

_LOSE_EVENTS = {"killed", "death", "trade", "hard_deck", "left_bubble"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-policy M→F hand-off verdict vs native T3.")
    p.add_argument(
        "--checkpoint-m",
        required=True,
        help="the merge-CONVERTER policy M (trained on --ladder merge-convert)",
    )
    p.add_argument(
        "--checkpoint-f",
        default="runs/curriculum-46/_snap_2244k.pt",
        help="the FROZEN finisher F (default = the champion; the ≤55° chase-kill deliverable)",
    )
    p.add_argument("--episodes", type=int, default=12, help="episodes per aspect")
    p.add_argument(
        "--aspects",
        default="0",
        help="comma list of nominal start aspects in deg (0=dead-ahead merge with --opp-heading 180)",
    )
    p.add_argument("--separation", type=float, default=1200.0)
    p.add_argument(
        "--opp-heading",
        type=float,
        default=180.0,
        help="opponent ABSOLUTE start heading deg; 180 = head-on nose-to-nose merge (the deliverable)",
    )
    p.add_argument(
        "--opp-tier",
        type=int,
        default=0,
        help="opponent AI tier (0/3 = full scripted Tier-3 shooter = north-star; 1/2 = weaker)",
    )
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.0)
    # Selector-gate OVERRIDES (deg / m). Default = read from constants.py (matches reward-v18 exactly).
    # Use these to TEST whether F can finish from a looser hand-off (e.g. behind+settled but nose-off)
    # WITHOUT retraining — F is a chase policy and may pull its own nose on. -1 = use the constant.
    p.add_argument("--select-aspect-deg", type=float, default=-1.0)
    p.add_argument("--select-ata-deg", type=float, default=-1.0)
    p.add_argument("--select-bank-deg", type=float, default=-1.0)
    p.add_argument(
        "--select-pitch-deg",
        type=float,
        default=-1.0,
        help="|pitch| ≤ this (deg) to hand off — skips nose-low diving crossings (-1 = no pitch gate)",
    )
    p.add_argument("--select-range-min", type=float, default=-1.0)
    p.add_argument("--select-range-max", type=float, default=-1.0)
    p.add_argument(
        "--select-closure-max",
        type=float,
        default=-1.0,
        help="|closure| ≤ this (m/s) to hand off — reward-v21 LOW-CROSSING gate: F inherits a "
        "matched-velocity settled tail, NOT a high-closure merge-pass fly-through (-1 = use the constant)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load(path: str, device) -> TDMPC2:
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # greedy policy-prior mean → no MPPI state → clean mid-episode hand-off
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent, int(sd.get("stage_idx", -1)), int(sd.get("collected", -1))


def _roll_2p(agent_m, agent_f, env, base, C, episodes: int, seed0: int, gate: dict) -> list[dict]:
    from astrum_dogfight.geometry import relative

    g_aspect = gate["aspect_rad"]
    g_ata = gate["ata_rad"]
    g_bank = gate["bank_rad"]
    g_pitch = gate["pitch_rad"]  # None = no pitch gate
    g_rmin = gate["rng_min"]
    g_rmax = gate["rng_max"]
    g_closure = gate["closure_max"]  # reward-v21 low-crossing gate

    rows: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent_m._prev_mean.zero_()
        agent_f._prev_mean.zero_()
        active = agent_m
        converted = False
        convert_step = None
        handoff = None  # kinematic state F inherits at the conversion instant
        best = None  # deepest rear-quarter M reaches (min aspect while alive+in-band) — settled or slicing?
        t0 = True
        event = None
        terminated = truncated = False
        step_i = 0
        while True:
            a = active.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            obs, _r, terminated, truncated, info = env.step(a)
            step_i += 1
            if info.get("event"):
                event = info["event"]
            # Geometric selector — hand M's jet to the frozen finisher F the instant M reaches a
            # genuine rear-quarter firing position behind a LIVE bandit. Condition mirrors reward-v17's
            # conversion terminal (constants.py) so eval matches training exactly.
            if not converted:
                me = base._states[base.learner_id]
                bandit = base._states[base.opponent_id]
                if me.alive and bandit.alive:
                    rg = relative(me, bandit)
                    # Track the DEEPEST rear-quarter posture M reaches while in the prosecution range
                    # band — was it a settled roll-out (bank small, ata small) that just missed, or a
                    # knife-edge slice the bank gate correctly killed, or did M never get behind at all?
                    if g_rmin <= rg.rng <= g_rmax:
                        import math as _mb
                        if best is None or rg.aspect < best["aspect_rad"]:
                            best = {
                                "aspect_rad": float(rg.aspect),
                                "aspect": _mb.degrees(float(rg.aspect)),
                                "ata": _mb.degrees(float(rg.ata)),
                                "bank": _mb.degrees(float(me.phi)),
                                "pitch": _mb.degrees(float(me.theta)),
                                "alt": float(me.alt),
                                "rng": float(rg.rng),
                                "step": step_i,
                            }
                    if (
                        rg.aspect <= g_aspect
                        and rg.ata <= g_ata
                        and abs(me.phi) <= g_bank  # reward-v18: settled/flyable hand-off
                        and (g_pitch is None or abs(me.theta) <= g_pitch)  # skip nose-low diving crossings
                        and abs(rg.closure) <= g_closure  # reward-v21: matched-velocity tail, not a merge-pass fly-through
                        and g_rmin <= rg.rng <= g_rmax
                    ):
                        converted = True
                        convert_step = step_i
                        import math as _m
                        handoff = {  # the state F is handed — is it finishable?
                            "alt": float(me.alt),
                            "spd_kt": float(me.airspeed) * 1.94384,
                            "theta": _m.degrees(float(me.theta)),
                            "phi": _m.degrees(float(me.phi)),
                            "rng": float(rg.rng),
                            "aspect": _m.degrees(float(rg.aspect)),
                            "ata": _m.degrees(float(rg.ata)),
                            "closure": float(rg.closure),
                        }
                        active = agent_f
                        t0 = True  # fresh (no-op in greedy, but keeps the hand-off explicit)
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        dealt = C.HP_MAX - float(opp.health)
        taken = C.HP_MAX - float(me.health)
        outcome = _classify(event, terminated, truncated)
        rows.append(
            {
                "outcome": outcome,
                "event": event,
                "converted": converted,
                "convert_step": convert_step,
                "handoff": handoff,
                "end_alt": float(me.alt),
                "dealt": dealt,
                "taken": taken,
                "net": dealt - taken,
                "me_alive": bool(me.alive),
            }
        )
        if handoff is not None:
            print(
                f"    ep{ep:>2} HANDOFF@{convert_step:>3}: alt {handoff['alt']:>6.0f}m spd {handoff['spd_kt']:>5.0f}kt "
                f"pitch {handoff['theta']:>+5.0f} bank {handoff['phi']:>+5.0f} rng {handoff['rng']:>5.0f} "
                f"asp {handoff['aspect']:>3.0f} ata {handoff['ata']:>3.0f} clos {handoff['closure']:>+5.0f} "
                f"→ F → {outcome:>8} end_alt {float(me.alt):>6.0f}m"
            )
        else:
            if best is not None:
                print(
                    f"    ep{ep:>2} NO-CONVERT → {outcome:>8} end_alt {float(me.alt):>6.0f}m | "
                    f"deepest@{best['step']:>3}: asp {best['aspect']:>3.0f} ata {best['ata']:>3.0f} "
                    f"bank {best['bank']:>+4.0f} pitch {best['pitch']:>+4.0f} alt {best['alt']:>5.0f} "
                    f"rng {best['rng']:>5.0f}m"
                )
            else:
                print(
                    f"    ep{ep:>2} NO-CONVERT → {outcome:>8} end_alt {float(me.alt):>6.0f}m | "
                    f"never entered the [{C.CONVERT_RANGE_MIN:.0f},{C.CONVERT_RANGE_MAX:.0f}]m band"
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
    device = resolve_device(args.device)

    agent_m, m_stage, m_coll = _load(args.checkpoint_m, device)
    agent_f, f_stage, f_coll = _load(args.checkpoint_f, device)
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]

    from math import radians as _rad
    gate = {
        "aspect_rad": _rad(args.select_aspect_deg) if args.select_aspect_deg >= 0 else C.CONVERT_ASPECT_RAD,
        "ata_rad": _rad(args.select_ata_deg) if args.select_ata_deg >= 0 else C.CONVERT_ATA_RAD,
        "bank_rad": _rad(args.select_bank_deg) if args.select_bank_deg >= 0 else C.CONVERT_BANK_RAD,
        "pitch_rad": _rad(args.select_pitch_deg) if args.select_pitch_deg >= 0 else None,
        "rng_min": args.select_range_min if args.select_range_min >= 0 else C.CONVERT_RANGE_MIN,
        "rng_max": args.select_range_max if args.select_range_max >= 0 else C.CONVERT_RANGE_MAX,
        "closure_max": args.select_closure_max if args.select_closure_max >= 0 else C.CONVERT_CLOSURE_MAX,
    }

    print(
        f"TWO-POLICY VERDICT | M={args.checkpoint_m} (stage {m_stage}, {m_coll:,}) "
        f"→ F={args.checkpoint_f} (stage {f_stage}, {f_coll:,})"
    )
    print(
        f"  hand-off gate: aspect≤{gate['aspect_rad'] * 57.2958:.0f}° ata≤{gate['ata_rad'] * 57.2958:.0f}° "
        f"bank≤{gate['bank_rad'] * 57.2958:.0f}° |clos|≤{gate['closure_max']:.0f}m/s rng∈[{gate['rng_min']:.0f},{gate['rng_max']:.0f}]m | "
        f"vs T{args.opp_tier or 3} opp_hdg {args.opp_heading:.0f}° sep {args.separation:.0f}m "
        f"jitter{args.jitter:g} | {args.episodes} eps/aspect (greedy)"
    )
    print(
        f"{'aspect':>6} {'N':>3} {'conv%':>6} {'conv@':>6} {'kill':>4} {'death':>5} {'deck':>4} "
        f"{'trade':>5} {'t/out':>5} {'conv>kill%':>10} {'dealt':>6} {'taken':>6} {'NET':>7}"
    )
    overall: list[dict] = []
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
        rows = _roll_2p(agent_m, agent_f, env, base, C, args.episodes, args.seed + int(aspect) * 1000, gate)
        env.close()
        overall.extend(rows)
        n = len(rows)
        conv_rows = [r for r in rows if r["converted"]]
        conv = len(conv_rows)
        conv_pct = 100.0 * conv / n if n else 0.0
        conv_at = statistics.fmean(r["convert_step"] for r in conv_rows) if conv_rows else float("nan")
        kill = sum(r["outcome"] == "kill" for r in rows)
        death = sum(r["outcome"] == "death" for r in rows)
        deck = sum(r["outcome"] == "deck" for r in rows)
        trade = sum(r["event"] == "trade" for r in rows)
        tout = sum(r["outcome"] == "timeout" for r in rows)
        # conv→kill%: of the hand-offs M actually delivered, how many did F finish
        conv_kill = 100.0 * statistics.fmean(r["outcome"] == "kill" for r in conv_rows) if conv_rows else float("nan")
        dealt = statistics.fmean(r["dealt"] for r in rows)
        taken = statistics.fmean(r["taken"] for r in rows)
        net = statistics.fmean(r["net"] for r in rows)
        print(
            f"{aspect:>6.0f} {n:>3} {conv_pct:>5.0f}% {conv_at:>6.0f} {kill:>4} {death:>5} {deck:>4} "
            f"{trade:>5} {tout:>5} {conv_kill:>9.0f}% {dealt:>6.1f} {taken:>6.1f} {net:>+7.1f}"
        )
    # rollup: the deliverable end-to-end story
    n = len(overall)
    conv = sum(r["converted"] for r in overall)
    kill = sum(r["outcome"] == "kill" for r in overall)
    loss = sum(r["outcome"] in ("death", "deck") for r in overall)
    net = statistics.fmean(r["net"] for r in overall)
    print(
        f"  ALL: {n} eps | converted {conv}/{n} ({100.0 * conv / n:.0f}%) | "
        f"e2e kills {kill} losses {loss} | mean NET {net:+.1f}"
    )


if __name__ == "__main__":
    main()
