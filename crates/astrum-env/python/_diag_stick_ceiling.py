#!/usr/bin/env python3
"""A2 — what BFM do the demonstration sticks ACTUALLY fly? (checklist item A2, `bfm-program-handoff`)

The four exploration seeds spliced into training (`_vertical_egg_stick`, `_lowyoyo_stick`,
`_highyoyo_stick`, `_leadturn_stick` in `astrum_rl.tdmpc2.train_parallel`) have never been measured
against a BFM detector — only against outcomes. Three defects were then found in them by reading:

  1. the per-episode coin is `leadturn`-only, so egg/lowyoyo/highyoyo redraw at 24 Hz and the
     "demonstration" is a quarter-second of script alternating with a quarter-second of policy;
  2. rudder is nailed to 0.0 in all four;
  3. the throttle constants are written as POWER FRACTIONS but returned in AGENT space, and
     `map_tdmpc_action` maps agent `a` to native `0.5(a+1)` — so `_HY_THR_UP = 0.70` commands native
     0.85 and `_LY_SETTLE_THR = 0.55` commands 0.775. The bleed half of a yo-yo runs at nearly full
     power, and a high yo-yo that cannot bleed is a zoom.

This script measures the size of each, one variable at a time, BEFORE any GPU is spent — and before
any of them is "fixed", so the fix is chosen by measurement rather than by reading. Nothing here
imports a checkpoint: the sticks are scalar functions and this is their own ceiling.

Two questions, two modes:

  --mode ceiling  the stick flies the WHOLE episode. "Is this maneuver worth copying at all?" —
                  the gate on checklist item C1 (actor-side BC). Coin and splice window are absent
                  by construction, so this isolates the maneuver from the plumbing.
  --mode splice   the stick flies only inside the training window (`rng < --window-m` AND
                  `ata > --window-ata-deg`), with a coin, exactly as `_collector_loop` does. The
                  env's own PursuitExpert flies the rest. That expert is a STAND-IN for the policy,
                  not the policy — it is deterministic and available, which is what makes
                  `--coin perstep` vs `--coin perepisode` a clean one-variable comparison.

Cross-cutting switches:

  --thr asflown|fixed   `fixed` converts the stick's throttle from the power fraction its constants
                        were written as into agent space (`2p - 1`). `asflown` leaves defect 3 in.
  --coin perstep|perepisode  (splice mode only) defect 1.

Rudder (defect 2) needs no switch: it is answered by MEASURING SIDESLIP. The F-16 FLCS in
`f16.xml` has a yaw channel (`fcs/yaw-load-pid`) driven by `accelerations/n-pilot-y-norm` and yaw
rate, i.e. an automatic turn coordinator. If `|beta|` stays small through committed rolls then
rudder 0.0 is CORRECT and defect 2 is not a defect. The `beta` percentiles below are conditioned on
rolling ticks (`|ail| > AIL_COMMIT`), because that is when a missing rudder would show.

Geometry is the benchmark one (`_ENERGY_MERGE` stage 0 + curriculum-94 lateral separation): neutral
head-on, native aggressive T3, 3000 m, +/-1400 m of lateral turning room, 180 s, jitter 0.5.

Run from `controllers/rl`:
    uv run python /workspace/crates/astrum-env/python/_diag_stick_ceiling.py --mode ceiling
    uv run python /workspace/crates/astrum-env/python/_diag_stick_ceiling.py --mode splice
"""

import argparse
import math
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import astrum_rl  # noqa: F401  (path wiring: puts astrum_env.so + astrum_dogfight on sys.path)

import numpy as np
from astrum_rl.envs import make_tdmpc_env
from astrum_rl.tdmpc2 import train_parallel as TP

from astrum_dogfight import bfm as BFM
from astrum_dogfight import bfm_detect as B
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import relative

DEG = 180.0 / math.pi


# ======================================================================================
# Arms. Each returns an AGENT-space [ail, elev, rud, thr]; `st` is per-episode scratch
# (only the lead turn has state, and that state is the whole point of it).
# ======================================================================================
def _egg(me, bd, rel, sgn, st):
    return TP._vertical_egg_stick(float(me.phi), float(me.theta), sgn)


def _lowyoyo(me, bd, rel, sgn, st):
    return TP._lowyoyo_stick(
        float(me.phi), float(me.theta), float(rel.rng), float(rel.aspect), sgn
    )


def _highyoyo(me, bd, rel, sgn, st):
    return TP._highyoyo_stick(float(me.phi), float(me.theta), float(rel.closure), sgn)


def _leadturn(me, bd, rel, sgn, st):
    """The c95 stick, call site and all — ported from `_collector_loop` so the latch behaves."""
    vm = np.asarray(me.vel, float)
    sp = float(np.linalg.norm(vm))
    gam = float(np.arcsin(np.clip(vm[2] / sp, -1.0, 1.0))) if sp > 1e-6 else 0.0
    st["lt_phase"] = TP._leadturn_phase(
        st.get("lt_phase", "APPROACH"), float(rel.rng), float(rel.ata), float(rel.aspect)
    )
    if st["lt_phase"] == "TRACK":
        ail, elev = BFM._point_at(rel, pull_gain=C.K_PURSUE_PULL, lag_rad=TP._LT_TRACK_LAG_RAD)
        thr = max(0.25, min(1.0, 0.85 - max(0.0, float(rel.closure)) / 300.0))
    else:
        ail, elev, _, thr = TP._leadturn_stick(
            float(me.phi),
            float(me.theta),
            gam,
            float(me.airspeed),
            st["lt_phase"] == "TURN",
            TP._cpa_miss_m(me.pos, me.vel, bd.pos, bd.vel),
            sgn,
        )
    ail, elev = BFM._deck_guard(me, ail, elev)
    return np.asarray([ail, elev, 0.0, thr], np.float32)


def _inplane(me, bd, rel, sgn, st):
    """D6 negative control (i): the scripted IN-PLANE tracker must score ~0 D1/D2/D3.

    If it does not, the detector's thresholds are too loose and every count below is worthless.
    """
    trk = st.get("trk")
    if trk is None:
        trk = st["trk"] = BFM.LagPursuitTracker()
        trk.reset(me, rel)  # `Maneuver.step` reads the entry frame `reset` captures
    a, _ = trk.step(me, rel)
    return np.asarray(a, np.float32)


ARMS = {
    "egg": _egg,
    "lowyoyo": _lowyoyo,
    "highyoyo": _highyoyo,
    "leadturn": _leadturn,
    "inplane": _inplane,
}


def _fix_thr(a4):
    """Defect 3: reinterpret the stick's throttle as the POWER FRACTION its constants were written
    as, and convert to agent space. Inverse of `map_tdmpc_action`'s `native = 0.5(a+1)`."""
    a = np.asarray(a4, np.float32).copy()
    a[3] = float(np.clip(2.0 * float(a[3]) - 1.0, -1.0, 1.0))
    return a


# ======================================================================================
# Rollout
# ======================================================================================
def make_env(seed, args):
    return make_tdmpc_env(
        3,
        posture="offensive",
        opponent_tier=0,  # the native aggressive T3 — the benchmark adversary
        aspect_deg=0.0,
        opp_heading_deg=180.0,  # flying AT us ⇒ neutral head-on merge
        separation_m=args.separation_m,
        lateral_m_max=args.lateral_m,  # curriculum-94 turning room (needs jitter > 0)
        max_time_s=args.max_time_s,
        rate_hz=120.0,
        action_repeat=5,  # the trainer's 24 Hz control rate — the sticks are tuned at it
        frame_stack=1,
        jitter=args.jitter,
        jitter_seed=seed,
    )


def fly(arm, seed, args):
    """One episode. Returns (trace, extras, outcome)."""
    env = make_env(seed, args)
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    lnr, opp = base.learner_id, base.opponent_id
    dt = 5 / 120.0
    ata_gate = math.radians(args.window_ata_deg)

    st = {"lt_phase": "APPROACH"}
    # Defect 1: the coin. Per-EPISODE is one draw here; per-STEP redraws every control step, which
    # is what egg/lowyoyo/highyoyo actually get in training.
    rs = np.random.default_rng(seed)
    ep_coin = rs.random() < args.p if args.coin == "perepisode" else True

    tr, extras = [], []
    t = 0.0
    hp0 = float(base._states[opp].health)
    ammo0 = int(base._states[lnr].ammo)
    event = None
    for _ in range(int(args.max_time_s / dt) + 10):
        me, bd = base._states[lnr], base._states[opp]
        rel = relative(me, bd)
        # The env's PursuitExpert stands in for the policy outside the splice window.
        stand_in = info.get("expert_action")
        action = np.zeros(C.ACTION_DIM, np.float32)
        if stand_in is not None:
            action[:] = np.asarray(stand_in, np.float32)

        active = False
        if args.mode == "ceiling":
            active = True
        else:
            draw = ep_coin if args.coin == "perepisode" else (rs.random() < args.p)
            active = bool(
                draw and bd.alive and rel.rng < args.window_m and rel.ata > ata_gate
            )
        if active:
            sgn = 1.0 if rel.ata_az >= 0.0 else -1.0
            a4 = ARMS[arm](me, bd, rel, sgn, st)
            if args.thr == "fixed":
                a4 = _fix_thr(a4)
            action[:4] = np.asarray(a4, np.float32)[:4]

        tr.append(B.make_sample(t, me, bd, action, rel))
        extras.append(
            dict(
                active=active,
                beta_deg=abs(float(me.beta)) * DEG,
                nz=float(me.nz),
                tas=float(me.airspeed),
                thr_native=0.5 * (float(action[3]) + 1.0),
                alt=float(me.alt),
                gq=float(rel.gun_quality_honest),
                lay=float(rel.lay_miss_m),
                phase=st.get("lt_phase", "-"),
            )
        )
        obs, rew, term, trunc, info = env.step(action)
        t += dt
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break

    me, bd = base._states[lnr], base._states[opp]
    out = dict(
        event=event,
        dmg_dealt=hp0 - float(bd.health),
        dmg_taken=C.HP_MAX - float(me.health),
        rounds=ammo0 - int(me.ammo),
        t_end=t,
    )
    env.close()
    return tr, extras, out


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def sweep(eps):
    """Threshold sensitivity — the ONLY thing that separates "the stick is hollow" from "my
    thresholds are too tight", and the reading D6 control (ii) demands when a fixed `_highyoyo_stick`
    scores 0.00 D2/episode.

    Re-runs the three shape detectors on the SAME traces with `NZ_LOADED` relaxed all the way to
    0.5 g (below which "wings-level cruise" would itself count as a maneuver, so there is nothing
    left to relax). If a count stays 0.00 across the whole sweep then the load clause is not what
    rejected the event and the stick does not fly the SHAPE either — no threshold rescues it.
    `CHI_OUT_DEG` gets the same treatment down to 30 deg for D1/D3.

    The detectors read these as module globals at call time, so patching the module is sufficient.
    """
    nz0, chi0, room0 = B.NZ_LOADED, B.CHI_OUT_DEG, B.CHI_ROOM_DEG
    n = max(1, len(eps))
    try:
        # D2 keys on closure and the flight-path swing, so the chi column moves D1/D3 only.
        print("  sensitivity   nz>   chi>   D1/ep  D2/ep  D3/ep")
        for nzt, chit in [
            (3.0, 60.0),  # shipped
            (2.0, 60.0),
            (1.0, 60.0),
            (0.5, 60.0),
            (3.0, 30.0),
            (0.5, 30.0),  # everything relaxed at once
        ]:
            B.NZ_LOADED, B.CHI_OUT_DEG = nzt, chit
            B.CHI_ROOM_DEG = min(room0, chit)
            d1 = sum(len(B.detect_displacement_rolls(tr)) for tr in eps) / n
            d2 = sum(len(B.detect_yoyos(tr)) for tr in eps) / n
            d3 = sum(len(B.detect_turning_room(tr)) for tr in eps) / n
            tag = "  <- shipped" if (nzt, chit) == (nz0, chi0) else ""
            print(f"               {nzt:4.1f}  {chit:5.0f}    {d1:5.2f}  {d2:5.2f}  {d3:5.2f}{tag}")
    finally:
        B.NZ_LOADED, B.CHI_OUT_DEG, B.CHI_ROOM_DEG = nz0, chi0, room0


def run(arm, args):
    hdr = f"{arm:9s} mode={args.mode:7s} thr={args.thr:7s}"
    if args.mode == "splice":
        hdr += f" coin={args.coin:11s} p={args.p:.2f}"
    print(f"\n=== {hdr} ===")
    eps, ex_all, outs = [], [], []
    for s in range(args.episodes):
        tr, ex, out = fly(arm, args.seed0 + s, args)
        eps.append(tr)
        ex_all.append(ex)
        outs.append(out)

    print(B.format_summary(B.summarize(eps)))

    flat = [e for ex in ex_all for e in ex]
    act = [e for e in flat if e["active"]]
    # beta conditioned on ROLLING ticks: a missing rudder shows during roll, not in cruise.
    roll = [
        e
        for ex, tr in zip(ex_all, eps)
        for e, s in zip(ex, tr)
        if abs(s.ail) > B.AIL_COMMIT
    ]
    nz = [e["nz"] for e in act] or [0.0]
    print(
        f"  stick active {100.0 * len(act) / max(1, len(flat)):4.1f}% of ticks"
        f" | nz p50 {_pct(nz, 50):4.2f} p90 {_pct(nz, 90):4.2f} max {max(nz):4.2f}"
        f" | native thr p50 {_pct([e['thr_native'] for e in act] or [0], 50):4.2f}"
        f" | TAS p50 {_pct([e['tas'] for e in act] or [0], 50):5.1f} m/s"
    )
    # Which BRANCH did each stick actually take? Defect 3 only bites on the branches that carry a
    # sub-1.0 throttle constant (the yo-yos' bleed halves), so its reach is bounded by these shares.
    clo = [s.closure for tr, ex in zip(eps, ex_all) for s, e in zip(tr, ex) if e["active"]] or [0.0]
    hot = sum(1 for c in clo if c > B.CLOSURE_OVERSHOOT)
    print(
        f"  active-tick closure p50 {_pct(clo, 50):6.1f} p90 {_pct(clo, 90):6.1f} m/s"
        f" | > CLOSURE_OVERSHOOT on {100.0 * hot / len(clo):4.1f}% (the yo-yo UP/bleed branch)"
        f" | alt p50 {_pct([e['alt'] for e in act] or [0], 50):6.0f} m"
    )
    b = [e["beta_deg"] for e in roll]
    print(
        f"  |beta| on rolling ticks ({len(b)} of {len(flat)}):"
        f" p50 {_pct(b, 50):4.1f} p90 {_pct(b, 90):4.1f} max {max(b) if b else float('nan'):4.1f} deg"
    )
    kills = sum(1 for o in outs if o["event"] == "kill")
    decks = sum(1 for o in outs if o["event"] in ("deck", "crash", "ground"))
    print(
        f"  outcome: kills {kills}/{len(outs)}  decks {decks}  "
        f"dmg dealt mean {np.mean([o['dmg_dealt'] for o in outs]):5.1f} "
        f"taken {np.mean([o['dmg_taken'] for o in outs]):5.1f}  "
        f"rounds {np.mean([o['rounds'] for o in outs]):5.0f}  "
        f"t_end {np.mean([o['t_end'] for o in outs]):5.1f}s  "
        f"events {sorted({str(o['event']) for o in outs})}"
    )
    print(
        f"  best geometry reached: gq_honest max {max(e['gq'] for e in flat):.2f}  "
        f"lay_miss min {min(e['lay'] for e in flat):6.1f} m  "
        f"rng min {min(s.rng for tr in eps for s in tr):6.0f} m  "
        f"aspect min {DEG * min(s.aspect for tr in eps for s in tr):5.1f} deg"
    )
    if args.sweep:
        sweep(eps)


p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["ceiling", "splice"], default="ceiling")
p.add_argument("--thr", choices=["asflown", "fixed"], default="asflown")
p.add_argument("--coin", choices=["perstep", "perepisode"], default="perepisode")
p.add_argument("--arms", default="inplane,egg,lowyoyo,highyoyo,leadturn")
p.add_argument("--episodes", type=int, default=6)
p.add_argument("--seed0", type=int, default=7000)
p.add_argument("--p", type=float, default=1.0, help="splice coin probability (defect 1's knob)")
p.add_argument("--window-m", type=float, default=3000.0, help="--merge-override-range-m")
p.add_argument("--window-ata-deg", type=float, default=25.0, help="--merge-override-ata-deg")
p.add_argument("--separation-m", type=float, default=3000.0)
p.add_argument("--lateral-m", type=float, default=1400.0)
p.add_argument("--max-time-s", type=float, default=180.0)
p.add_argument("--jitter", type=float, default=0.5)
p.add_argument(
    "--sweep",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="threshold sensitivity table (D6 control ii: hollow stick vs too-tight detector)",
)
args = p.parse_args()

print(
    f"geometry: neutral head-on vs native T3 | sep {args.separation_m:.0f} m  "
    f"lateral +/-{args.lateral_m:.0f} m  {args.max_time_s:.0f} s  jitter {args.jitter}  "
    f"{args.episodes} eps from seed {args.seed0}"
)
print(
    f"detector: CHI_OUT {B.CHI_OUT_DEG:.0f} deg  NZ_LOADED {B.NZ_LOADED:.1f}  "
    f"CLOSURE_OVERSHOOT {B.CLOSURE_OVERSHOOT:.0f} m/s  LETHAL_RANGE {B.LETHAL_RANGE_M:.0f} m"
)
for a in args.arms.split(","):
    if a.strip():
        run(a.strip(), args)
