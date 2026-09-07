"""curriculum-94: can ANY controller CONVERT the lateral head-on merge vs the FIXED T3?

`[[curriculum-94-headon-lateral]]` measured that lateral separation turns the head-on from a ~191°/s
collision-course crossing into a 10-14°/s co-orbit, 99-100% of ticks inside c87's proven kill band,
closure ≈0, 24-55 s/ep inside R_EFF. So the OPPORTUNITY is now in the data. What is missing is the
CONVERSION: ata median 124-133°, alpha_req median 146°, gq_honest max 0.00, ZERO rounds commanded.

`[[bfm-library-coach]]` says hand-code cannot convert a symmetric peer merge (twice-confirmed, 0
kills) — but that verdict is confounded twice over, exactly like the c90/c91 merge disproof was:

  1. it was measured against the DEFECTIVE zoom-climbing T3 ([[t3-break-ungated-pull]]), and
  2. it was measured at the default ±135 m lateral = a collision-course pass with no turning room,

so "hand-code can't convert" was never actually tested WITH turning room.

(A third, structural reason was HYPOTHESISED and then MEASURED FALSE, recorded here so it is not
re-proposed: `BFMCoach`'s merge branch (d) enters via `MergeApproach`, which flies `_point_at(rel)`,
so it looked like it must collapse the lateral offset back to a collision course before the pass. It
does not — COACH holds `room` 1067-1130 m of predicted CPA all the way in, because pointing at a
bandit who is himself translating does not drive the miss distance to zero. The coach's failure to
convert is NOT an approach-geometry defect.)

This asks the reachability question properly. Arms (all whole-episode, same seeds, same scenario):

  BASE     the c87 champion, greedy. The control — it flies AROUND (min rng 242 → 552 m, passes 0.1).
  COACH    `bfm.BFMCoach` as shipped. The second control, and the measurement of the flaw above.
  T<r>+    LEAD TURN, one-circle (turn TOWARD him at the pass). Three phases:
             APPROACH — MANAGE the merge miss distance instead of pointing: hold him at bearing
                        asin(D/rng) on the side he is already on, i.e. progressively "check away"
                        to BUILD turning room, nose slightly low to sit at corner speed.
             TURN     — at rng < r, roll to a hard bank INTO him and pull max-g for max rate.
             TRACK    — once inside his rear quarter, latch `LagPursuitTracker` and fire.
           `r` is swept because the lead-turn START RANGE is the whole trick: turn too early and he
           just re-points at you, too late and you are a stern conversion at zero closure.
  T<r>-    the same machine turning AWAY at the pass = the TWO-CIRCLE / rate fight. Against a bandit
           whose break is now a competent energy-preserving max-rate turn, the one-circle vs
           two-circle choice is a real fork, so both branches get measured rather than assumed.
  PROD     the SHIPPED c95 splice, exactly as the collector runs it: `_leadturn_stick` + `_cpa_miss_m`
           from train_parallel, driven at p=1.0 inside the production gate (rng < PROD_GATE_RNG_M ∧
           ata > PROD_GATE_ATA), the POLICY flying everything else INCLUDING the trigger. This is the
           regression test for the port, and it EARNED that name twice: the T<r>+ arm LATCHES both its
           turn direction and its phase and hands off to a `LagPursuitTracker`, while the shipped stick
           takes its roll direction from ata_az and has no tracker. Two stateless phase tests were
           measured HERE and both destroyed the conversion (range-only reverted mid-turn → gqh 0.00 /
           aspMn 85°; range∧closure committed before the merge → timeout 8/8, rngMn 875 m), so the
           shipped stick now latches its phase too. Its `dealt` is also the honest answer to "can c87
           finish a merge someone else converted for it?" — which the scripted arms, with their own
           fire gate, cannot answer.

If any arm converts, RL has a proven target and c94's exploration prior has something to imitate —
which would overturn [[merge-kill-not-scripted]] for the first time. If nothing converts even WITH
turning room and a competent bandit, the honest reading is that this merge is a draw by construction
against a peer, and the next lever is asymmetry (energy/perch), not more merge shaping.

    cd controllers/rl && uv run python -u _diag_merge_convert_tmp.py \
        --checkpoint runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --laterals 1500 --me-dalt-ft 2000 --episodes 4
"""

import argparse
from math import asin, degrees, radians

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402
from astrum_rl.tdmpc2.train_parallel import (  # noqa: E402
    _cpa_miss_m,
    _leadturn_phase,
    _leadturn_stick,
)

DEG = 180.0 / np.pi

# --- the PROD arm = the shipped c95 merge-override, gate and all ---------------------------------
PROD_GATE_RNG_M = 3000.0        # --merge-override-range-m: 2000 was MEASURED to hand back mid-turn
                                # (the pass opens the range past it), after which c87 flew away.
PROD_GATE_ATA = radians(25.0)   # --merge-override-ata-deg: also the hand-off — below it the policy flies
# Arm `PRODF` = PROD with ONE variable changed: inside the gate the honest ballistic trigger fires
# instead of the policy's. It exists because PROD measured a real solution it could not cash —
# `layMn 14 m` / `gqhMx 0.25` (both firsts for the shipped stick) yet **0 HP off 395 policy rounds**,
# where the reference `T1000+` deals 10 HP off 12 honest ones. So it separates the two halves of the
# demo: PROD answers "does the shipped stick CONVERT", PRODF answers "is the converted geometry
# SHOOTABLE, i.e. is the residual gap the aim or the trigger". Only PRODF's answer decides whether
# c95 must pair `--fire-override-*` with the prior (the trigger is a run FLAG, not a code change).

# --- lead-turn arm tuning (all measured-from, not guessed) --------------------------------------
TURN_BANK = radians(80.0)     # hard enough for max rate; 10° of lift still vertical so it is not
                              # a pure energy-dumping flat groove (cf. LeadTurn's 75°, which bled out)
D_TURNROOM = 800.0            # m; the merge miss distance the APPROACH phase steers for. Real BFM
                              # turning room is 500-1500 m; 800 sits mid-band and inside the ±1,535 m
                              # band c94 actually trains on.
V_CORNER_TGT = 225.0          # m/s; approach speed target — c87's KILLING buckets sit at 223-225
                              # (`_diag_can_anything_kill_tmp`), its losing ones at 136-155.
V_SUSTAIN_MPS = 190.0         # m/s; ease off max-g below this so the turn becomes SUSTAINED-rate.
                              # Above C.V_CORNER_MPS (175) with margin to spend in the pull.
TRACK_ASPECT = radians(70.0)  # aspect below this = we are behind him ⇒ hand off to the tracker
TRACK_ATA = radians(50.0)     # ...and the nose is close enough that tracking (not turning) converts
FIRE_LAY_MISS_M = 15.0        # m; fire when the CORRECTED lay predicts a miss inside ~2 hit spheres.
                              # An honest ballistic gate, not the reward's 3-4x-loose one
                              # ([[curriculum-76-77-bleed-corner]]) — a reachability test must not
                              # score sprays that cannot hit.
R_LETHAL_M = 600.0            # m; the range beyond which a kill is not buyable with the whole
                              # magazine. Measured, not assumed (`_diag_phit_range.py`): scoring the
                              # COMPUTABLE lay (`gunlay.intercept` — constant-velocity, no drag, no
                              # drop, exactly what `lay_miss_m` scores) against a 5 g breaker with
                              # the repo's own tick-exact propagator gives p_hit 0.98@300 m,
                              # 0.83@395, 0.70@440, 0.48@500, 0.20@600, 0.06@700, 0.002@900,
                              # 0.000@1200 ⇒ rounds needed for the 50 hits of a kill: 51, 60, 71,
                              # 103, 247, 838, 33k, never. AMMO_MAX is 511, so 600 m is the edge.
                              # C.R_EFF (1200 m) — the range the reward pays aim and fire inside —
                              # is 2x that in radius and 4x in area, i.e. most of the advertised
                              # WEZ is a range at which p_hit is literally zero.


def cpa(me, op):
    """Predicted (miss distance, pass side) at the closest point of approach, straight-line.

    This — NOT the LOS bearing — is "how much turning room will the pass give me". The distinction is
    not cosmetic here: `opp_heading_deg=180` sets the bandit's heading reciprocal to MY HEADING, so at
    spawn the nose is on him (`ata_az≈0`) while the geometry already carries the full `sep·sin(bearing)`
    of lateral offset. Steering on `ata_az` therefore reads "no turning room" when there is 1500 m of
    it. Returns `side` = +1 if he will pass down my right, -1 my left.
    """
    r = np.asarray(op.pos, float) - np.asarray(me.pos, float)
    v = np.asarray(op.vel, float) - np.asarray(me.vel, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-9 else max(0.0, -float(r @ v) / vv)   # t<0 ⇒ already past ⇒ use now
    m = r + v * t
    # Body-right in the horizontal plane: velocity (vE, vN) rotated -90° → (vN, -vE).
    vm = np.asarray(me.vel, float)
    right = np.array([vm[1], -vm[0]])
    nr = float(np.hypot(right[0], right[1]))
    s = float(m[0] * right[0] + m[1] * right[1]) / nr if nr > 1e-6 else 0.0
    return float(np.linalg.norm(m)), (1.0 if s >= 0.0 else -1.0)


class LeadTurnArm:
    """APPROACH (build turning room) → TURN (max-rate lead turn) → TRACK (rear-quarter guns).

    `toward=True` turns INTO his pass side at the merge (one-circle / radius fight); False turns AWAY
    (two-circle / rate fight). `turn_at` is the range at which the turn is committed.
    """

    def __init__(self, turn_at: float, toward: bool):
        self.turn_at = float(turn_at)
        self.toward = bool(toward)
        self.phase = "APPROACH"
        self._sign = 0.0
        self._tracker = bfm.LagPursuitTracker()
        self._tracker_live = False

    def act(self, me, op, rel):
        miss, side = cpa(me, op)
        if self.phase == "APPROACH" and rel.rng <= self.turn_at:
            # Latch the turn direction ONCE, off the side he is actually going to pass. Re-deriving a
            # turn sign every tick is the defect that made the scripted T3 zoom
            # ([[t3-break-ungated-pull]]), so the latch is deliberate.
            self._sign = side if self.toward else -side
            self.phase = "TURN"
        if self.phase == "TURN" and rel.aspect < TRACK_ASPECT and rel.ata < TRACK_ATA:
            self.phase = "TRACK"

        if self.phase == "APPROACH":
            # MANAGE the merge miss distance to D_TURNROOM. Too little room ⇒ check AWAY from his pass
            # side; too much ⇒ ease back toward him rather than fly a wide, disengaging arc. This is
            # the opposite of `MergeApproach`'s `_point_at`, which collapses the offset to zero.
            phi_cmd = side * float(np.clip(2.0 * (miss - D_TURNROOM) / D_TURNROOM,
                                           -1.0, 1.0)) * radians(45.0)
            ail = bfm._roll_to(me.phi, phi_cmd)
            # Arrive AT corner, not fast: turn radius goes as V², so the 331 m/s a full-throttle dive
            # reaches would need a 1241 m radius vs 574 m at 225 m/s — un-turnable. Cash the perch
            # into speed only while slow, then hold the target on the throttle.
            th_tgt = radians(-4.0) if me.airspeed < V_CORNER_TGT - 15.0 else 0.0
            elev = bfm._pitch_to(me.theta, th_tgt)
            thr = float(np.clip(0.5 + (V_CORNER_TGT - me.airspeed) / 40.0, 0.0, 1.0))
        elif self.phase == "TURN":
            # Hold the FLIGHT PATH level, not the pitch attitude. Commanding 80° of bank + max Nz and
            # nothing else produced a climbing spiral: γ crept positive, and at 80° of bank the pull is
            # nearly horizontal so nothing pulls the nose back down — measured +3.2 km of climb and a
            # 252→152 m/s bleed. The pilot's answer is to roll PAST vertical and pull the nose down to
            # the horizon, so γ error is fed to the BANK (γ>0 ⇒ steepen past 90°, γ<0 ⇒ shallow out).
            vm = np.asarray(me.vel, float)
            sp = float(np.linalg.norm(vm))
            gamma = float(np.arcsin(np.clip(vm[2] / sp, -1.0, 1.0))) if sp > 1e-6 else 0.0
            bank_tgt = float(np.clip(TURN_BANK + 2.0 * gamma, radians(40.0), radians(115.0)))
            bank_err = self._sign * bank_tgt - me.phi
            ail = float(np.clip(bfm.KROLL * bank_err, -1.0, 1.0))
            # GATE the pull on roll completion. Holding max aft stick while still rolling turns the
            # whole g demand into PITCH — that is precisely the defect that made the scripted T3
            # zoom-climb ([[t3-break-ungated-pull]]), and an ungated version of this arm reproduced it
            # exactly: +4,446 m of climb and a bleed to 166 m/s. Matches the house convention
            # (`lead_pursuit`, the Tier-2/Tier-3 breaks).
            mag = max(np.cos(bank_err), 0.0)
            # Max-RATE while fast, then settle to a SUSTAINED turn instead of bleeding below corner:
            # a max-g pull held to 153 m/s throws away the turn rate it was bought for.
            if me.airspeed < V_SUSTAIN_MPS:
                mag *= float(np.clip(me.airspeed / V_SUSTAIN_MPS, 0.45, 1.0))
            elev = -float(np.clip(mag, 0.0, 1.0))
            thr = 1.0
        else:
            if not self._tracker_live:
                self._tracker.reset(me, rel)   # mandatory: `step` reads the entry frame + _steps
                self._tracker_live = True
            a4, _done = self._tracker.step(me, rel)
            ail, elev, thr = float(a4[0]), float(a4[1]), float(a4[3])
            if rel.aspect > radians(110.0):
                # He got out in front pointing back at us — that is another merge, not a chase.
                # (`LagPursuitTracker` never self-terminates, so this is the only way out.)
                self.phase, self._tracker_live = "APPROACH", False
                self._tracker = bfm.LagPursuitTracker()

        ail, elev = bfm._deck_guard(me, ail, elev)
        # Honest ballistic trigger, uniform across the scripted arms. -1.0 is hold-fire in AGENT
        # space (0.0 maps to native 0.5 = exactly ON the sim's gate ⇒ fires every tick; see bfm.py).
        fire = 1.0 if (rel.in_range and rel.lay_miss_m <= FIRE_LAY_MISS_M) else -1.0
        return np.array([ail, elev, 0.0, thr, fire], dtype=np.float32)


NCOL = 12


def first_pass(rng_col):
    """Index of the first tick inside gun range (len(rng_col) if it never got there)."""
    w = np.flatnonzero(rng_col <= C.R_EFF)
    return int(w[0]) if len(w) else len(rng_col)


def n_passes(rng_col):
    """Count ENTRIES into gun range. [[curriculum-91-altitude-budget]] measured that passes and
    time-in-range discriminate a converting policy while episode length does not."""
    c = rng_col <= C.R_EFF
    if len(c) == 0:
        return 0
    return int(np.count_nonzero(c[1:] & ~c[:-1]) + (1 if c[0] else 0))


def roll(agent, env, base, seed, dt, arm, trace=False):
    obs, info = env.reset(seed=seed)
    agent._prev_mean.zero_()
    ctrl = None
    if arm == "COACH":
        ctrl = bfm.BFMCoach()
    elif arm not in ("BASE", "PROD", "PRODF"):
        ctrl = LeadTurnArm(float(arm[1:-1]), arm[-1] == "+")
    t0, event, rows = True, None, []
    phase = arm
    lt_phase = "APPROACH"  # the PROD splice's per-EPISODE state (see _leadturn_phase)
    while True:
        me = base._states[base.learner_id]
        op = base._states[base.opponent_id]
        rg = relative(me, op)
        if arm in ("PROD", "PRODF"):
            # Reproduce the collector's splice verbatim: the policy acts, then the shipped stick
            # overwrites [0:4] inside the gate. The TRIGGER stays with the policy, as in production.
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            if rg.rng < PROD_GATE_RNG_M and rg.ata > PROD_GATE_ATA:
                vm = np.asarray(me.vel, float)
                sp = float(np.linalg.norm(vm))
                gam = float(np.arcsin(np.clip(vm[2] / sp, -1.0, 1.0))) if sp > 1e-6 else 0.0
                lt_phase = _leadturn_phase(
                    lt_phase, float(rg.rng), float(rg.ata), float(rg.aspect)
                )
                if lt_phase == "TRACK":
                    ail, elev = bfm._point_at(rg, pull_gain=C.K_PURSUE_PULL, lag_rad=radians(3.0))
                    thr = max(0.25, min(1.0, 0.85 - max(0.0, float(rg.closure)) / 300.0))
                else:
                    ail, elev, _, thr = _leadturn_stick(
                        float(me.phi), float(me.theta), gam, float(me.airspeed),
                        lt_phase == "TURN",
                        _cpa_miss_m(me.pos, me.vel, op.pos, op.vel),
                        1.0 if rg.ata_az >= 0.0 else -1.0,
                    )
                ail, elev = bfm._deck_guard(me, ail, elev)
                a[:4] = np.asarray([ail, elev, 0.0, thr], np.float32)
                if arm == "PRODF":
                    # ONE variable vs PROD: the honest ballistic trigger, same gate as every scripted
                    # arm. Outside the window the policy keeps its own trigger in BOTH arms.
                    a[4] = 1.0 if (rg.in_range and rg.lay_miss_m <= FIRE_LAY_MISS_M) else -1.0
                phase = lt_phase
            else:
                phase = "policy"
        elif ctrl is None:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        elif arm == "COACH":
            a = ctrl.act(me, rg)                 # BFMCoach's shipped signature
        else:
            a = ctrl.act(me, op, rg)             # LeadTurnArm needs `op` for the CPA solve
        t0 = False
        if trace and len(rows) % 24 == 0:      # ~1 Hz at the 24 Hz agent rate
            print(f"TR| t{len(rows) * dt:>5.1f} {getattr(ctrl, 'phase', phase):<8} "
                  f"rng{rg.rng:>6.0f} cpa{cpa(me, op)[0]:>6.0f} "
                  f"φ{degrees(me.phi):>+5.0f} θ{degrees(me.theta):>+4.0f} α{degrees(me.alpha):>4.1f} "
                  f"nz{me.nz:>+5.1f} V{me.airspeed:>4.0f} agl{me.alt - base.ground_alt:>6.0f} "
                  f"ata{degrees(rg.ata):>4.0f} asp{degrees(rg.aspect):>4.0f} | "
                  f"ail{a[0]:>+5.2f} elv{a[1]:>+5.2f} thr{a[3]:>4.2f} | "
                  # The two candidate turn directions, side by side. The shipped stick steers on
                  # `sg` (ata_az); the arm that CONVERTED latched `sd` (the CPA pass side) once. If
                  # they disagree, the shipped stick is flying the two-circle fight by accident.
                  f"sd{cpa(me, op)[1]:>+2.0f} sg{1 if rg.ata_az >= 0.0 else -1:>+2d} "
                  f"az{degrees(rg.ata_az):>+4.0f}", flush=True)
        rows.append((rg.rng, degrees(rg.ata), degrees(rg.aspect), degrees(rg.alpha_req),
                     rg.lay_miss_m, rg.gun_quality_honest,
                     DEG * float(rg.los_rate),
                     rg.closure, me.airspeed, me.alt - base.ground_alt, float(a[4] >= 0.0),
                     cpa(me, op)[0]))
        obs, _r, term, trunc, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    op = base._states[base.opponent_id]
    return dict(event=event, M=np.asarray(rows).reshape(-1, NCOL),
                dealt=C.HP_MAX - float(op.health), taken=C.HP_MAX - float(me.health),
                spent=int(C.AMMO_MAX - me.ammo))


def main():
    global PROD_GATE_RNG_M, PROD_GATE_ATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    # Lateral separation, free: with `opp_heading_deg=180` both jets sit on RECIPROCAL headings at any
    # bearing, so bearing IS lateral offset (lateral = sep·sin(aspect)). Given in METRES here.
    ap.add_argument("--laterals", default="1500")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--separation", type=float, default=3000.0)
    ap.add_argument("--max-time-s", type=float, default=180.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--me-dalt-ft", type=float, default=2000.0)
    ap.add_argument("--arms", default="BASE,COACH,T1500+,T1000+,T600+,T1000-")
    # The PROD/PRODF window IS the production hand-off (`--merge-override-range-m` /
    # `--merge-override-ata-deg`), so make it sweepable in an 8-episode diag instead of a 40-min run.
    ap.add_argument("--prod-gate-rng-m", type=float, default=PROD_GATE_RNG_M)
    ap.add_argument("--prod-gate-ata-deg", type=float, default=degrees(PROD_GATE_ATA))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    # Per-tick trace of ONE episode: the arms are hand-flown, so when an arm misbehaves (the first
    # version zoom-climbed +4.4 km) the trace is the only honest way to find out why.
    ap.add_argument("--trace", default="")
    args = ap.parse_args()
    PROD_GATE_RNG_M = float(args.prod_gate_rng_m)
    PROD_GATE_ATA = radians(float(args.prod_gate_ata_deg))

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    dt = 5.0 / 120.0
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"CONVERT {args.checkpoint} (collected={sd.get('collected', 0):,}) | NEUTRAL head-on "
          f"sep {args.separation:.0f}m perch +{args.me_dalt_ft:.0f}ft {args.max_time_s:.0f}s "
          f"jitter {args.jitter:g} | {args.episodes} eps/cell | arms {'/'.join(arms)} | "
          f"turnroom {D_TURNROOM:.0f}m bank {degrees(TURN_BANK):.0f}° fire lay_miss≤"
          f"{FIRE_LAY_MISS_M:.0f}m | PROD gate rng<{PROD_GATE_RNG_M:.0f}m ata>"
          f"{degrees(PROD_GATE_ATA):.0f}°", flush=True)
    hdr = (f"CV| {'lat':>5} {'arm':<7} {'kills':>5} {'events':<30} {'dealt':>5} {'taken':>5} "
           f"{'rds':>4} {'epS':>4} {'pass':>4} {'tInR':>5} {'room':>5} {'rngMn':>5} {'ataMn':>5} "
           f"{'aspMn':>5} {'aReqMn':>6} {'gqhMx':>5} {'layMn':>5} {'ωmed':>5} {'tHit':>5} "
           f"{'tLeth':>5} {'fRng':>5} {'f<Rl':>5} {'tLe6':>5} {'Vmed':>4} {'dAGL':>6}")
    print(hdr, flush=True)
    print("CV| " + "-" * (len(hdr) - 4), flush=True)

    for lat in [float(x) for x in args.laterals.split(",") if x.strip()]:
        # lateral = sep·sin(bearing) ⇒ bearing = asin(lat/sep)
        asp = degrees(asin(min(1.0, lat / args.separation)))
        kw = dict(posture="offensive", aspect_deg=asp, separation_m=args.separation,
                  action_repeat=5, frame_stack=1, max_time_s=args.max_time_s, rate_hz=120.0,
                  opp_heading_deg=180.0)
        if args.jitter > 0:
            kw["jitter"] = args.jitter
        if args.me_dalt_ft:
            kw["me_dalt_ft"] = args.me_dalt_ft
        env = make_tdmpc_env(3, **kw)
        base = env.unwrapped
        while hasattr(base, "env"):
            base = base.env
        for arm in arms:
            rows = [roll(agent, env, base, 7000 + int(lat) + i, dt, arm,
                         trace=(args.trace == arm and i == 0))
                    for i in range(args.episodes)]
            M = np.concatenate([r["M"] for r in rows], axis=0)
            inr = M[M[:, 0] <= C.R_EFF]
            # `room` must be the setup room BEFORE the first pass only. Taking it over every tick
            # outside 1 km folds in the post-merge separation and reported 1,680 m of "turning room"
            # for an approach that never had it.
            appr = np.concatenate([r["M"][:first_pass(r["M"][:, 0])] for r in rows], axis=0)
            ev = {}
            for r in rows:
                ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
            n = len(rows)
            passes = np.mean([n_passes(r["M"][:, 0]) for r in rows])
            print(f"CV| {lat:>5.0f} {arm:<7} {sum(r['event'] == 'kill' for r in rows):>5} "
                  f"{' '.join(f'{k}×{v}' for k, v in sorted(ev.items())):<30} "
                  f"{np.mean([r['dealt'] for r in rows]):>5.0f} "
                  f"{np.mean([r['taken'] for r in rows]):>5.0f} "
                  f"{np.mean([r['spent'] for r in rows]):>4.0f} "
                  f"{np.mean([len(r['M']) for r in rows]) * dt:>4.0f} "
                  f"{passes:>4.1f} {len(inr) * dt / n:>5.1f} "
                  # `room` = the turning room the pass was actually SET UP with (median predicted CPA
                  # while still on the approach, outside 1 km). Distinguishes "built room and failed to
                  # use it" from "collapsed the offset before the merge" — the coach's structural flaw.
                  f"{(np.median(appr[:, 11]) if len(appr) else 0.0):>5.0f} "
                  f"{np.mean([r['M'][:, 0].min() for r in rows]):>5.0f} "
                  f"{np.mean([r['M'][:, 1].min() for r in rows]):>5.0f} "
                  f"{np.mean([r['M'][:, 2].min() for r in rows]):>5.0f} "
                  f"{np.mean([r['M'][:, 3].min() for r in rows]):>6.0f} "
                  f"{M[:, 5].max():>5.2f} "
                  # The direct "how far from a gun solution" scalar, in metres against an 8 m hit
                  # sphere — measured IN RANGE only, since out-of-range lay misses are meaningless.
                  f"{(inr[:, 4].min() if len(inr) else 9999):>5.0f} "
                  f"{(np.median(inr[:, 6]) if len(inr) else 0.0):>5.0f} "
                  # DWELL inside the hit sphere. ⚠ `tHit` is NOT range-gated, and `lay_miss_m`
                  # (geometry.py:227) is drag-free/gravity-free/constant-velocity, so an 8 m lay is
                  # satisfiable at 3 km with ≤3.7 mrad of lead and this column happily counts ticks
                  # where p_hit is ZERO. It is kept only for continuity with the c87→c95 record;
                  # `tLeth` beside it is the honest version (see R_LETHAL_M).
                  f"{(M[:, 4] <= C.HIT_RADIUS_M).sum() * dt / n:>5.2f} "
                  # tLeth = dwell with the lay inside the sphere AND inside the range where a round
                  # can actually connect. This is the number a kill is made of: ~0.5 s of it.
                  f"{((M[:, 4] <= C.HIT_RADIUS_M) & (M[:, 0] <= R_LETHAL_M)).sum() * dt / n:>5.2f} "
                  # WHERE the trigger is actually pulled (col 10 = fire commanded). Against a ~5 g
                  # breaker the computable lay gives p_hit 0.98@300m, 0.70@440, 0.48@500, 0.06@700,
                  # 0.00@1200 (_diag_phit_range.py), i.e. one 511-round magazine buys a kill only
                  # inside ~600 m. `fRng` outside that band means the rounds cannot land whatever
                  # the lay says — which is how 230 rounds at layMn 2 m deal 0 HP.
                  f"{(np.median(M[M[:, 10] > 0, 0]) if (M[:, 10] > 0).any() else 0.0):>5.0f} "
                  f"{(np.mean(M[M[:, 10] > 0, 0] <= R_LETHAL_M) if (M[:, 10] > 0).any() else 0.0):>5.2f} "
                  f"{(M[:, 10] > 0).sum() * dt / n:>5.2f} "
                  f"{np.median(M[:, 8]):>4.0f} "
                  f"{np.mean([r['M'][-1, 9] - r['M'][0, 9] for r in rows]):>+6.0f}", flush=True)
        env.close()
        print("CV|", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
