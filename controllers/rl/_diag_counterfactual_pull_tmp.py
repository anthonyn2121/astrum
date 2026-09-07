"""Does the REWARD forbid the hard turn, or does the POLICY just not know to make it?

`_diag_terms_tmp.py` summed every reward term over the ACTUAL greedy rollout and found the policy
already charged -75.7 (close) for letting the range grow at spawn aspect 55/60 — from which I
concluded the gradient points the right way and shaping is exhausted. That conclusion has a hole:
an ON-POLICY term sum prices the trajectory the policy FLEW, never the one it DECLINED. The
maneuver in question is a max-g turn, and a max-g turn BLEEDS, and bleed is taxed twice
(K_ENERGY=25 absolute + K_REL_ENERGY=50 relative, both over ENERGY_REF=10000 m). Hand-priced,
180->130 m/s is dEs ~= 7750 m => -19 - 39 = -58, against K_CLOSE's +15.2 for closing 831->450 m.
If that is real, the reward FORBIDS the pull and no amount of new observation will buy it.

So: measure the counterfactual directly, by splicing a stick override into a deterministic rollout.

  1. Run the greedy policy from a fixed seed and find t* = the first control step inside the
     measured park (rng 600-1000 m, |closure| < 50 m/s) that PERSISTS for HOLD_S seconds — the
     state where the policy is choosing 3 g with 9 g available.
  2. Re-run the SAME seed. Greedy is deterministic (mpc off, scripted opponent, seeded jitter), so
     the prefix replays tick-identically — asserted, not assumed, via `prefix_dev`.
  3. From t* on, fly an override instead, to the end of the episode.

Three arms, chosen so the result discriminates between three different diagnoses:

  BASE  the policy's own choice — the control.
  PURE  lift vector on the bandit, elevator pegged at max pull. The literal "turn harder" the term
        audit says is unpriced. If PURE scores WORSE than BASE, the reward forbids it and the next
        lever is the energy tax, not the observation.
  LAG   the same max pull, aimed LAG_DEG behind him. This is the turn-circle answer (cut across the
        circle rather than chase the tail), and it bleeds nearly as much while pointing less. PURE
        vs LAG separates "hard turning is taxed" from "over-pulling is taxed".

Reported per arm: the env's OWN summed reward (ground truth) over 3/6/10 s and to end; the
per-term attribution from re-running `reward.compute`; and the physics (min range, speed bled,
dEs, nz actually achieved, seconds with an honest 8 m solution, HP dealt). nz matters
independently: if the override cannot get past ~4 g either, the 9 g premise is wrong and the
stalemate is a CAPABILITY wall after all.

    cd controllers/rl && uv run python -u _diag_counterfactual_pull_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt --aspects 55,60 --episodes 4
"""

import argparse
from math import degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import relative, specific_energy  # noqa: E402
from astrum_rl.envs import invert_tdmpc_action, make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

TERMS = ["damage_reward", "track_reward", "close_reward", "aim_reward", "aspect_reward",
         "convert_bonus", "energy_reward", "band_reward", "corner_penalty", "rel_energy_reward",
         "lead_reward", "spiral_penalty", "fire_reward", "time_penalty"]

LAG_DEG = 20.0     # trail behind his tail — the lag-pursuit / cut-the-circle aim point
HOLD_S = 2.0       # the park has to persist this long before t* counts (skip transient passes)
# Round 1 used |closure| < 50 (copied from `_diag_stalemate_tmp.py`'s pooled window) and EVERY t*
# came back at exactly -49/-50: the filter was binding on its own boundary, so the "park" it found
# was the first tick of a separation, not the stalemate. The stalemate's pooled MEDIAN closure was
# -23/-30, so tighten to catch the genuinely stationary state.
PARK_CLO = 25.0
HORIZ_BANK = 85.0  # the max-rate HORIZONTAL turn: lift nearly in-plane, so the pull buys turn, not alt


def _override(me, rel, low, high, lag_rad, horiz=False):
    """Max-performance turn toward the bandit, in WRAPPER space.

    elev = -1 is max pull on the inverted-sign FLCS (bfm.py's convention). `_point_at`'s
    proportional pull is deliberately DISCARDED and replaced with the peg: the whole question is
    what a max-g turn costs, and a P-controller settles well short of max g once ata closes.
    `_deck_guard` stays, because an override that simply flies into the ground would confound the
    energy accounting with a deck termination.

    `horiz` swaps the turn PLANE: instead of rolling the lift vector onto the bandit (which, when he
    is high, points the pull up and converts the g into altitude), bank HORIZ_BANK toward his side
    and peg the pull, giving the classic max-rate level turn. Round 1 showed every arm GAINING
    +2600-3000 m of Es while the range grew to ~2 km, so "the g is going into a zoom-climb instead
    of into turn rate" is the live hypothesis and this arm is its control.
    """
    if horiz:
        sgn = 1.0 if rel.ata_az >= 0.0 else -1.0
        ail = bfm._roll_to(me.phi, sgn * radians(HORIZ_BANK))
        elev = -1.0
    else:
        ail, _ = bfm._point_at(rel, lag_rad=lag_rad)
        pull = max(0.0, rel.ata - lag_rad)
        elev = -1.0 if pull > radians(2.0) else -0.2
    ail, elev = bfm._deck_guard(me, ail, elev)
    # Fire whenever the loose gun gate would — so the arm is not handicapped on trigger volume.
    trig = 1.0 if (rel.in_range and rel.gun_quality > 0.30) else 0.0
    raw = np.array([ail, elev, 0.0, 1.0, trig], dtype=np.float32)
    return invert_tdmpc_action(raw, low, high)


NCOL = 19
ALPHA_SAT = 14.0   # deg — round 2 found α pinned 13.8-16.7 in EVERY arm, so this is "out of lift"


def roll(agent, env, base, seed, ground_alt, dt, switch_at=None, lag_rad=0.0, horiz=False):
    """One episode. `switch_at=None` = pure greedy (and returns the park index). Otherwise greedy
    until that control step, then the override to the end."""
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    low = np.asarray(base.action_space.low, dtype=np.float32)
    high = np.asarray(base.action_space.high, dtype=np.float32)
    t0, event = True, None
    prev_me = prev_op = None
    k = 0
    rows, terms, park = [], [], []
    while True:
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        if switch_at is not None and k >= switch_at:
            a = _override(me, rg, low, high, lag_rad, horiz=horiz)
        else:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        if prev_me is not None:
            out = R.compute(prev_me, prev_op, me, op, ground_alt=ground_alt, outcome_mode=True)
            terms.append([float(out.info.get(t, 0.0)) for t in TERMS])
        else:
            terms.append([0.0] * len(TERMS))
        park.append(600.0 <= rg.rng <= 1000.0 and abs(rg.closure) < PARK_CLO)
        prev_me, prev_op = me, op
        obs, r, term, trunc, info = env.step(a)
        rows.append((rg.rng, rg.closure, me.airspeed, me.nz, specific_energy(me),
                     rg.lay_miss_m, degrees(rg.aspect), float(r),
                     # Added after round 1: without ALTITUDE the +2700 m dEs is uninterpretable, and
                     # without ALPHA there is no way to tell "declined to pull" from "out of lift".
                     me.alt - ground_alt, op.alt - ground_alt, degrees(me.alpha),
                     degrees(rg.ata), float(op.health), degrees(me.theta),
                     # Round 3: the BANDIT's vertical state. Round 2 showed every arm climbing ~3 km
                     # and ending α-saturated at <2 g — but "he dragged me up" and "I spent my energy
                     # on altitude while he kept his lift vector low" predict opposite levers, and
                     # only his numbers separate them.
                     op.airspeed, op.nz, degrees(op.alpha), degrees(op.theta), rg.rel_alt))
        if info.get("event"):
            event = info["event"]
        k += 1
        if term or trunc:
            break
    me = base._states[base.learner_id]
    op = base._states[base.opponent_id]
    M = np.asarray(rows).reshape(-1, NCOL)
    T = np.asarray(terms).reshape(-1, len(TERMS))
    # t* = first index of a park that PERSISTS for HOLD_S (a pass through the band is not the park).
    need = max(1, int(round(HOLD_S / dt)))
    tstar = None
    p = np.asarray(park, dtype=bool)
    for i in range(len(p) - need):
        if p[i:i + need].all():
            tstar = i
            break
    return dict(event=event, M=M, T=T, tstar=tstar,
                dealt=C.HP_MAX - float(op.health), taken=C.HP_MAX - float(me.health),
                spent=int(C.AMMO_MAX - me.ammo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--aspects", default="55,60")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--separation", type=float, default=700.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    dt = 5.0 / 120.0

    print(f"CF {args.checkpoint} (collected={sd.get('collected', 0):,}) | greedy prefix + spliced "
          f"override | chase sep {args.separation:.0f}m {args.max_time_s:.0f}s jitter {args.jitter:g}"
          f" | park = rng 600-1000 ∧ |clo|<50 held {HOLD_S:g}s | lag {LAG_DEG:.0f}°", flush=True)
    print(f"CF| K_CLOSE {C.K_CLOSE:g}/{C.TRACK_RANGE:g}m  K_ENERGY {C.K_ENERGY:g}  "
          f"K_REL_ENERGY {C.K_REL_ENERGY:g}  over ENERGY_REF {C.ENERGY_REF:g}m  "
          f"K_BAND {C.K_BAND:g}@{C.V_BAND_CTR:g}±{C.V_BAND_SIGMA:g}m/s", flush=True)

    for asp in [float(x) for x in args.aspects.split(",") if x.strip()]:
        env = make_tdmpc_env(3, posture="offensive", aspect_deg=asp, separation_m=args.separation,
                             action_repeat=5, frame_stack=1, max_time_s=args.max_time_s,
                             rate_hz=120.0, **({"jitter": args.jitter} if args.jitter > 0 else {}))
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        ground_alt = float(base.ground_alt)
        print(f"CF| ===== spawn aspect {asp:.0f}° =====", flush=True)
        agg = {}
        for i in range(args.episodes):
            seed = 5000 + int(asp) * 100 + i
            a0 = roll(agent, env, base, seed, ground_alt, dt)
            # WHOLE-EPISODE line, printed whether or not a park happened — this is what makes a
            # bucket the policy WINS (spawn 30) usable as the control. The splice can only be taken
            # where the policy actually parks, and where it kills it never does.
            E = a0["M"]
            sat = 100.0 * float((E[:, 10] >= ALPHA_SAT).mean())
            print(f"CF|  ep{i} EPISODE {a0['event']:<9} rngMin {E[:, 0].min():4.0f} "
                  f"| dAGL me {E[-1, 8] - E[0, 8]:+6.0f} op {E[-1, 9] - E[0, 9]:+6.0f} "
                  f"| θmed me {np.median(E[:, 13]):+4.0f} op {np.median(E[:, 17]):+4.0f} "
                  f"| V med me {np.median(E[:, 2]):3.0f} op {np.median(E[:, 14]):3.0f} "
                  f"min me {E[:, 2].min():3.0f} op {E[:, 14].min():3.0f} "
                  f"| relAlt med {np.median(E[:, 18]):+5.0f} "
                  f"| nz med me {np.median(E[:, 3]):3.1f} op {np.median(E[:, 15]):3.1f} "
                  f"| α med me {np.median(E[:, 10]):4.1f} op {np.median(E[:, 16]):4.1f} "
                  f"| α≥{ALPHA_SAT:g}° {sat:3.0f}% of ticks "
                  f"| dealt {a0['dealt']:3.0f}", flush=True)
            if a0["tstar"] is None:
                print(f"CF|      (never parked — no splice)", flush=True)
                continue
            ts = a0["tstar"]
            arms = [("BASE", a0)]
            for name, lag, hz in (("PURE", 0.0, False), ("HORIZ", 0.0, True)):
                arms.append((name, roll(agent, env, base, seed, ground_alt, dt,
                                        switch_at=ts, lag_rad=lag, horiz=hz)))
            # The splice is only meaningful if the prefix really replayed — check it, don't trust it.
            dev = max(float(np.abs(r["M"][:ts, 0] - a0["M"][:ts, 0]).max()) if ts > 0 else 0.0
                      for _, r in arms[1:])
            print(f"CF|  ep{i} seed {seed}  t* {ts * dt:5.1f}s  rng {a0['M'][ts, 0]:4.0f}m  "
                  f"V {a0['M'][ts, 2]:3.0f}m/s  clo {a0['M'][ts, 1]:+4.0f}  "
                  f"AGL me {a0['M'][ts, 8]:5.0f} op {a0['M'][ts, 9]:5.0f}  "
                  f"ata {a0['M'][ts, 11]:3.0f}° asp {a0['M'][ts, 6]:3.0f}°  "
                  f"prefix_dev {dev:.2e} m", flush=True)
            for name, r in arms:
                M, T = r["M"][ts:], r["T"][ts:]
                if not len(M):
                    continue
                es0 = M[0, 4]
                w = {f"{s:g}s": M[:min(len(M), int(round(s / dt))), 7].sum() for s in (3, 6, 10)}
                hit_s = float((M[:, 5] <= C.HIT_RADIUS_M).sum()) * dt
                # Damage AFTER the splice — the episode totals are dominated by the shared prefix
                # (round 1 printed identical `dealt` for all arms and it meant nothing).
                dealt_post = float(M[0, 12] - M[-1, 12])
                parts = [(TERMS[j], float(T[:, j].sum())) for j in range(len(TERMS))]
                parts = sorted([p for p in parts if abs(p[1]) >= 0.5], key=lambda p: -abs(p[1]))
                print(f"CF|    {name:<5} Σr " + " ".join(f"{k} {v:+7.1f}" for k, v in w.items())
                      + f" end {M[:, 7].sum():+8.1f} | rng min {M[:, 0].min():4.0f} "
                      f"end {M[-1, 0]:4.0f} | V min {M[:, 2].min():3.0f} | dAGL "
                      f"{M[-1, 8] - M[0, 8]:+6.0f} dEs {M[-1, 4] - es0:+6.0f} | nz med "
                      f"{np.median(M[:, 3]):3.1f} max {M[:, 3].max():3.1f} | α med "
                      f"{np.median(M[:, 10]):4.1f} max {M[:, 10].max():4.1f} | θ med "
                      f"{np.median(M[:, 13]):+4.0f} | ata med {np.median(M[:, 11]):3.0f} | tHit "
                      f"{hit_s:4.1f}s | dealt+ {dealt_post:3.0f} | {r['event']}", flush=True)
                print(f"CF|          terms " + "  ".join(
                    f"{k.replace('_reward', '').replace('_penalty', '')} {v:+.1f}"
                    for k, v in parts), flush=True)
                d = agg.setdefault(name, {t: 0.0 for t in TERMS})
                d["_n"] = d.get("_n", 0) + 1
                d["_end"] = d.get("_end", 0.0) + float(M[:, 7].sum())
                d["_10s"] = d.get("_10s", 0.0) + w["10s"]
                d["_dealt"] = d.get("_dealt", 0.0) + dealt_post
                d["_dEs"] = d.get("_dEs", 0.0) + float(M[-1, 4] - es0)
                d["_dagl"] = d.get("_dagl", 0.0) + float(M[-1, 8] - M[0, 8])
                d["_rmin"] = d.get("_rmin", 0.0) + float(M[:, 0].min())
                d["_rend"] = d.get("_rend", 0.0) + float(M[-1, 0])
                d["_nz"] = d.get("_nz", 0.0) + float(np.median(M[:, 3]))
                d["_nzmax"] = d.get("_nzmax", 0.0) + float(M[:, 3].max())
                d["_amax"] = d.get("_amax", 0.0) + float(M[:, 10].max())
                for j, t in enumerate(TERMS):
                    d[t] += float(T[:, j].sum())
            print("CF|", flush=True)
        for name in ("BASE", "PURE", "HORIZ"):
            d = agg.get(name)
            if not d:
                continue
            n = max(1, d["_n"])
            parts = sorted([(t, d[t] / n) for t in TERMS if abs(d[t] / n) >= 0.5],
                           key=lambda p: -abs(p[1]))
            print(f"CF| MEAN {name:<5} (n={n}) Σr10s {d['_10s'] / n:+7.1f} Σrend "
                  f"{d['_end'] / n:+8.1f} dealt+ {d['_dealt'] / n:5.1f} rngMin "
                  f"{d['_rmin'] / n:4.0f} rngEnd {d['_rend'] / n:4.0f} dAGL {d['_dagl'] / n:+6.0f} "
                  f"dEs {d['_dEs'] / n:+6.0f} nz {d['_nz'] / n:3.1f}/{d['_nzmax'] / n:3.1f} αmax "
                  f"{d['_amax'] / n:4.1f} | " + "  ".join(
                      f"{k.replace('_reward', '').replace('_penalty', '')} {v:+.1f}"
                      for k, v in parts), flush=True)
        env.close()
        print("CF|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
