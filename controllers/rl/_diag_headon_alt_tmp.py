"""curriculum-90 ALTITUDE-BUDGET diagnostic for the neutral head-on merge.

c90's greedy head-on probe ends `hard_deck` 12/12 while simultaneously WINNING the energy war
(endRelEs +1756 m, T3 bled to 156 m/s, passes 2.4, peakGQ 0.31). It starts at 15,000 ft = 4572 m
AGL with the hard deck at 300 m, so it spends ~4.3 km of altitude inside 180 s and hits the floor.
The reward's anti-descent signals only ramp in BELOW `SPIRAL_ALT_BAND_AGL = 2000 m` (anti-spiral)
and below `DECK_AGL + DECK_MARGIN = 800 m` (cushion), leaving the first ~2.5 km unpriced.

Before touching a constant, this answers WHERE the altitude actually goes:

  • the altitude budget split across the three reward bands (free / anti-spiral / cushion) — how
    much is lost where, and at what descent rate,
  • whether the loss is GRADUAL (a sustained descending fight) or a terminal PLUNGE,
  • whether it ever CLIMBS (`maxClimb`, `nUp`) — i.e. whether it uses the VERTICAL at all, which is
    the out-of-plane/egg behaviour the whole program is trying to teach,
  • whether the deck arrives during a close pass (fighting) or between passes (mismanagement),
  • and the AGL at which its best gun solutions occur.

    cd controllers/rl && uv run python _diag_headon_alt_tmp.py \
        --checkpoint runs/curriculum-90/run.pt --me-dalt-ft 2000 --episodes 12
"""

import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import relative
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2.agent import TDMPC2
from astrum_rl.tdmpc2.config import TDMPC2Config

DEG = 180.0 / math.pi
BAND_HI = C.SPIRAL_ALT_BAND_AGL           # 2000 m; above this NO anti-spiral penalty
BAND_LO = C.DECK_AGL + C.DECK_MARGIN      # 800 m; below this the deck cushion takes over


def make_env(sep, jitter, max_time_s, me_dalt_ft, alt_ft, alt_ft_max, aspect_deg=0.0):
    # LATERAL SEPARATION, free: with `opp_heading_deg=180` the two jets are on RECIPROCAL headings
    # whatever the bearing, so placing the bot at bearing `aspect_deg` and range `separation_m`
    # gives an OFFSET head-on merge — lateral offset = sep·sin(aspect), along-track = sep·cos(aspect),
    # range unchanged. Today's merge is aspect 0 + a ±136 m jitter wobble = 2.6° = 0.19 of a 712 m
    # 9 g turn radius, i.e. a collision-course pass with NO turning room. Sweeping this measures the
    # lateral-separation hypothesis on an EXISTING checkpoint, with no scenario change to trust.
    kw = dict(posture="offensive", aspect_deg=float(aspect_deg), separation_m=float(sep),
              action_repeat=5,
              frame_stack=1, max_time_s=float(max_time_s), rate_hz=120.0, opp_heading_deg=180.0)
    if jitter > 0:
        kw["jitter"] = jitter
    if me_dalt_ft:
        kw["me_dalt_ft"] = float(me_dalt_ft)
    # curriculum-91: the BLOCK altitude is the fight's budget, so it has to be a diagnostic knob.
    # Read a banded policy TWICE: at 15,000 ft (the exact scenario c90 was measured in ⇒ an
    # apples-to-apples "did it learn to manage the budget") and across its own training band
    # (⇒ "does more budget actually buy more passes"). Leaving it hardwired at 15,000 would score
    # a banded policy off-centre from its training distribution and read as a false negative.
    if alt_ft:
        kw["alt_ft"] = float(alt_ft)
        if alt_ft_max:
            kw["alt_ft_max"] = float(alt_ft_max)
    return make_tdmpc_env(3, **kw)


def run_ep(agent, env, base, seed, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    t0, event = True, None
    agl = []          # AGL per control step
    close = []        # was the bandit inside 500 m this step
    rngs = []         # slant range per control step (min range was never reported before c92)
    gq_agl = (-1.0, float("nan"))  # (best gun quality, the AGL it happened at)
    fire_cmds = 0
    # Per-step aim geometry for the ticks that are actually INSIDE guns range. c90@5M reaches
    # <500 m for 14.2 s and fires 0.8 rounds for 0 kills, so the wall is one of two things and
    # these columns separate them: an un-shootable lay the whole time (⇒ the CONVERSION is the
    # wall) vs a usable solution it declines to take (⇒ FIRE DISCIPLINE is the wall).
    #
    # curriculum-92: the window is R_EFF (1200 m), NOT 500 m, and `rng` is a column so it can be
    # sliced back down. A 500 m gate is the wrong instrument for a LATERAL-SEPARATION experiment: the
    # whole point of the offset is to move closest approach out to ~587 m at L=1200, so a merge that
    # genuinely improved into a 600-800 m tracking pass would empty a 500 m window and read as a
    # total null — the measurement would report the opposite of what happened. The gun is nominally
    # effective to R_EFF anyway (range_factor 0.58 at 500 m, 0.42 at 700 m), so 1200 m is the honest
    # window and the <500 m slice below keeps the c90/c91 numbers comparable.
    inr = []  # (rng, lay_miss, alpha_req°, ata°, aspect°, closure, gq_honest, los_rate°/s, fired)
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        h = me.alt - base.ground_alt
        agl.append(h)
        close.append(rel.rng < 500.0)
        if rel.gun_quality > gq_agl[0]:
            gq_agl = (rel.gun_quality, h)
        # a[4] is AGENT space: map_tdmpc_action gives native = 0.5(a+1) and the env fires at
        # native >= 0.5, i.e. agent >= 0. Testing >= 0.5 here undercounts every trigger-down figure.
        fired = a[4] >= 0.0
        if fired:
            fire_cmds += 1
        rngs.append(rel.rng)
        if rel.rng < C.R_EFF:
            # LOS RATE is the mechanism this experiment actually turns: at closest approach
            # ω ≈ V_rel/miss, so a 135 m collision-course pass at ~450 m/s closure spins the line of
            # sight at 191°/s, while 587 m gives 44°/s and 800 m gives 32°/s. Against the strongest
            # empirical finding in the program — c87 kills arrive at 18.6°/s and miss at 59°/s
            # ([[curriculum-87-arrival-obs]]) — the offset is the only knob that moves the pass from
            # ~10× the failing rate toward the working band, and unlike "turning room" it is IN the
            # observation (obs 31/32), so the policy can actually condition on it.
            los = rel.los_rate * DEG
            inr.append((rel.rng, rel.lay_miss_m, rel.alpha_req * DEG, rel.ata * DEG,
                        rel.aspect * DEG, rel.closure, rel.gun_quality_honest, los, float(fired)))
        obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if terminated or truncated:
            break

    agl = np.asarray(agl)
    d = np.diff(agl)                      # per-control-step altitude change
    lost_free = -d[(agl[:-1] >= BAND_HI) & (d < 0)].sum()
    lost_spiral = -d[(agl[:-1] < BAND_HI) & (agl[:-1] >= BAND_LO) & (d < 0)].sum()
    lost_cush = -d[(agl[:-1] < BAND_LO) & (d < 0)].sum()
    t_free = float((agl >= BAND_HI).sum()) * dt
    t_spiral = float(((agl < BAND_HI) & (agl >= BAND_LO)).sum()) * dt
    t_cush = float((agl < BAND_LO).sum()) * dt
    # Terminal plunge vs gradual: how much of the total loss happened in the LAST 15 s.
    n15 = max(1, int(15.0 / dt))
    tail = -d[-n15:][d[-n15:] < 0].sum()
    total_lost = -d[d < 0].sum()
    # PASSES = the quantity the budget hypothesis actually predicts. The merge kill has to come from
    # CONVERTING on a later pass ([[merge-crossing-velocity-ceiling]]: the pass itself is
    # un-shootable), so "budget bought a fight" means MORE rising edges into guns range — not a
    # longer episode. c90 averaged 2.4 passes and died between them, never inside one.
    c = np.asarray(close)
    passes = int(np.count_nonzero(c[1:] & ~c[:-1]) + (1 if c[0] else 0))
    return dict(
        event=event, n=len(agl), t_s=len(agl) * dt,
        agl0=agl[0], aglEnd=agl[-1], aglMin=agl.min(), aglMax=agl.max(),
        lost_free=lost_free, lost_spiral=lost_spiral, lost_cush=lost_cush,
        t_free=t_free, t_spiral=t_spiral, t_cush=t_cush,
        # Does it use the vertical? total metres CLIMBED and the single biggest sustained climb.
        climbed=d[d > 0].sum(), maxClimb=(agl.max() - agl[0]),
        tail15=tail, total_lost=total_lost,
        deck_in_pass=bool(close[-1]), gq=gq_agl[0], gq_agl=gq_agl[1], fire=fire_cmds,
        passes=passes, t_close=float(c.sum()) * dt, inr=np.asarray(inr).reshape(-1, 9),
        # curriculum-92: min RANGE, never reported before. If the offset is being flown AROUND
        # rather than through, this is the number that says so — and it is the one metric a 500 m
        # aim gate silently destroys rather than merely missing.
        min_rng=float(np.min(rngs)),
        t_eff=float(np.count_nonzero(np.asarray(rngs) < C.R_EFF)) * dt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sep", type=float, default=3000.0)
    ap.add_argument("--me-dalt-ft", type=float, default=0.0)
    ap.add_argument("--alt-ft", type=float, default=0.0,
                    help="block altitude; 0 = the scenario default (15,000 ft = what c90 flew)")
    ap.add_argument("--alt-ft-max", type=float, default=0.0,
                    help="band ceiling — sample [--alt-ft, this] per episode (needs --jitter > 0)")
    ap.add_argument("--aspect-deg", type=float, default=0.0,
                    help="bot BEARING off the learner's nose; with opp_heading 180 this IS the "
                         "lateral-separation knob (lateral = sep·sin(aspect)). 0 = today's "
                         "collision-course merge")
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.5)
    args = ap.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, torch.device("cpu"))
    agent.load_state_dict(sd)
    agent.model.eval()
    if args.alt_ft_max and not args.alt_ft:
        raise SystemExit("--alt-ft-max is the band CEILING; pass --alt-ft as the floor too")
    if args.alt_ft_max and args.jitter <= 0:
        raise SystemExit("--alt-ft-max samples per episode via the env rng ⇒ needs --jitter > 0")
    block = "15,000 (default)" if not args.alt_ft else (
        f"{args.alt_ft:,.0f}" + (f"-{args.alt_ft_max:,.0f}" if args.alt_ft_max else ""))
    lat = args.sep * math.sin(math.radians(args.aspect_deg))
    print(f"DIAG {args.checkpoint} (collected={sd.get('collected', 0):,}) | "
          f"head-on sep {args.sep:.0f}m block {block} ft dalt {args.me_dalt_ft:.0f}ft "
          f"{args.max_time_s:.0f}s | aspect {args.aspect_deg:.1f}° ⇒ LATERAL {lat:.0f} m "
          f"({lat / 712.0:.2f} of a 9 g turn radius) | bands: free ≥{BAND_HI:.0f}m, "
          f"spiral {BAND_LO:.0f}-{BAND_HI:.0f}m, cushion <{BAND_LO:.0f}m")

    env = make_env(args.sep, args.jitter, args.max_time_s, args.me_dalt_ft,
                   args.alt_ft, args.alt_ft_max, args.aspect_deg)
    base = env.unwrapped
    while hasattr(base, "env"):
        base = base.env
    dt = 5.0 / 120.0  # action_repeat / rate_hz

    rows = [run_ep(agent, env, base, 1000 + i, dt) for i in range(args.episodes)]
    hdr = (f"{'ep':>3} {'event':>10} {'t_s':>5} {'agl0':>6} {'aglEnd':>7} {'lostFree':>9} "
           f"{'lostSpir':>9} {'lostCush':>9} {'tFree':>6} {'tSpir':>6} {'tCush':>6} "
           f"{'climbed':>8} {'tail15':>7} {'inPass':>7} {'gqAGL':>6} {'fire':>5} "
           f"{'pass':>5} {'tClose':>7} {'minRng':>7} {'gq':>5}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(rows):
        print(f"{i:>3} {str(r['event']):>10} {r['t_s']:>5.0f} {r['agl0']:>6.0f} {r['aglEnd']:>7.0f} "
              f"{r['lost_free']:>9.0f} {r['lost_spiral']:>9.0f} {r['lost_cush']:>9.0f} "
              f"{r['t_free']:>6.0f} {r['t_spiral']:>6.0f} {r['t_cush']:>6.0f} "
              f"{r['climbed']:>8.0f} {r['tail15']:>7.0f} {str(r['deck_in_pass']):>7} "
              f"{r['gq_agl']:>6.0f} {r['fire']:>5} {r['passes']:>5} {r['t_close']:>7.1f} "
              f"{r['min_rng']:>7.0f} {r['gq']:>5.2f}")
    print("-" * len(hdr))

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    decks = sum(r["event"] == "hard_deck" for r in rows)
    print(f"\nAGG {len(rows)} eps | hard_deck {decks} | mean t {m('t_s'):.0f}s "
          f"(cap {args.max_time_s:.0f}s ⇒ {'ENDS EARLY = deck truncates the fight' if m('t_s') < args.max_time_s * 0.9 else 'runs the clock'})")
    print(f"  altitude LOST: free-band {m('lost_free'):.0f} m | spiral-band {m('lost_spiral'):.0f} m | "
          f"cushion {m('lost_cush'):.0f} m  (total {m('total_lost'):.0f} m)")
    print(f"  time IN band:  free {m('t_free'):.0f} s | spiral {m('t_spiral'):.0f} s | cushion {m('t_cush'):.0f} s")
    print(f"  descent RATE:  free {m('lost_free')/max(1e-9, m('t_free')):.1f} m/s | "
          f"spiral {m('lost_spiral')/max(1e-9, m('t_spiral')):.1f} m/s | "
          f"cushion {m('lost_cush')/max(1e-9, m('t_cush')):.1f} m/s")
    print(f"  VERTICAL use:  climbed {m('climbed'):.0f} m total, peak gain over start {m('maxClimb'):.0f} m "
          f"⇒ {'uses the vertical' if m('climbed') > 300 else 'essentially NEVER climbs'}")
    print(f"  last 15 s held {100.0 * m('tail15') / max(1e-9, m('total_lost')):.0f}% of the total loss "
          f"⇒ {'terminal PLUNGE' if m('tail15') > 0.5 * m('total_lost') else 'GRADUAL descending fight'}")
    print(f"  deck arrived inside a <500 m pass in {sum(r['deck_in_pass'] for r in rows)}/{len(rows)} eps | "
          f"mean best-GQ AGL {m('gq_agl'):.0f} m | mean commanded fire {m('fire'):.1f} rounds")
    # The headline number for curriculum-91: budget is only worth buying if it buys PASSES.
    print(f"  PASSES <500 m: {m('passes'):.1f} per ep, {m('t_close'):.1f} s total inside guns range "
          f"⇒ {'multi-pass fight' if m('passes') >= 3.0 else 'ONE-PASS fight (no conversion possible)'}")

    # curriculum-92: min RANGE and peak GUN QUALITY are range-independent, so they are the two
    # metrics that survive an offset moving closest approach outside any fixed window. `gq` was
    # computed and thrown away before now.
    print(f"  CLOSEST APPROACH: min range {m('min_rng'):.0f} m per ep | {m('t_eff'):.1f} s inside "
          f"R_EFF ({C.R_EFF:.0f} m) | peak gun_quality {m('gq'):.2f} "
          f"⇒ {'flown AROUND, not through' if m('min_rng') > 500.0 else 'reaches knife-fight range'}")

    # --- IN-RANGE AIM: is the lay ever usable, and does it shoot when it is? -------------------
    A = np.concatenate([r["inr"] for r in rows], axis=0) if rows else np.zeros((0, 9))
    if not len(A):
        print(f"  IN-RANGE AIM: never inside R_EFF ({C.R_EFF:.0f} m) — nothing to measure\n")
        return
    # HITTABLE = predicted miss inside the sim's 8 m hit sphere (the honest ballistic test, not the
    # 3-4x looser reward/fire gate). FEASIBLE = the required lay is inside the usable ~15 deg alpha
    # cone; outside it the jet physically cannot point there ([[headon-merge-unshootable]]).
    def _block(M, label):
        if not len(M):
            print(f"  IN-RANGE AIM {label}: 0 ticks")
            return
        rg, miss, areq, ata, asp, clo, gqh, los, fired = (M[:, i] for i in range(9))
        hit = miss <= 8.0
        feas = areq <= 15.0
        both = hit & feas
        print(f"  IN-RANGE AIM {label} ({len(M)} ticks = {len(M) * dt:.1f} s):")
        print(f"    lay_miss_m   min {miss.min():.1f} | median {np.median(miss):.0f} | "
              f"HITTABLE (<=8 m) {100.0 * hit.mean():.1f}% of ticks")
        print(f"    alpha_req    min {areq.min():.1f}° | median {np.median(areq):.0f}° | "
              f"FEASIBLE (<=15°) {100.0 * feas.mean():.1f}% of ticks")
        print(f"    ata median {np.median(ata):.0f}° | aspect median {np.median(asp):.0f}° | "
              f"closure median {np.median(clo):+.0f} m/s | gq_honest max {gqh.max():.2f}")
        # c87: kills arrive at 18.6°/s LOS rate, misses at 59°/s. This is the dose-responsive
        # mechanism the lateral offset moves (ω ≈ V_rel/miss), and it is in the obs, so it is the
        # column most likely to show a gradient even if nothing gets shot.
        print(f"    LOS rate     median {np.median(los):.0f}°/s | min {los.min():.0f}°/s | "
              f"in the c87 KILL band (<=25°/s) {100.0 * (los <= 25.0).mean():.0f}% of ticks "
              f"| above the c87 MISS rate (>=59°/s) {100.0 * (los >= 59.0).mean():.0f}%")
        print(f"    SHOOTABLE (hittable AND feasible) {100.0 * both.mean():.1f}% of ticks "
              f"({int(both.sum())} ticks = {both.sum() * dt:.1f} s)")
        if both.sum():
            print(f"    → fired on {100.0 * fired[both].mean():.0f}% of the SHOOTABLE ticks "
                  f"⇒ {'FIRE DISCIPLINE is the wall' if fired[both].mean() < 0.5 else 'it does shoot when it can'}")
        else:
            print("    → 0 shootable ticks ⇒ the CONVERSION is the wall, not fire discipline: "
                  "it reaches guns RANGE but never a usable LAY")

    rng_col = A[:, 0]
    _block(A, f"<{C.R_EFF:.0f} m (R_EFF, the honest gun window)")
    # The <500 m slice is kept ONLY so these numbers stay directly comparable to the c90/c91
    # measurements already on record — it is not the primary read-out any more.
    _block(A[rng_col < 500.0], "<500 m (legacy slice, comparable to c90/c91)")
    _block(A[(rng_col >= 500.0) & (rng_col < 800.0)], "500-800 m (where an offset pass lands)")
    print()


if __name__ == "__main__":
    main()
