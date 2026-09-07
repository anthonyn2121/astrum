#!/usr/bin/env python3
"""D3a -- THE DWELL BUDGET. How many seconds of sub-600 m / sub-8 m-lay gun dwell does a
kill REQUIRE, and how many can the best available tracker actually HOLD against the fixed
(aggressive) T3 break? (checklist item D3a, `docs/bfm-program-handoff.md`)

Nothing in this program has ever priced the two halves against each other. Every "the wall
is X" claim has been about a mechanism; this box asks whether the arithmetic closes at all,
for zero GPU cost, and it is the one item that can invalidate the framing rather than move a
metric.

================================================================================
WHAT IS MEASURED
================================================================================
R0  Every load-bearing gun/health constant, read out of the Rust source (printed with its
    file), not out of any prose: rate_hz, damage_per_hit, hp_max, hit_radius_m,
    dispersion_rad, muzzle_speed, ammo.
R1  The rate of fire *the sim actually delivers*, measured by holding the trigger in a live
    env and counting the ammo decrement, under three duty cycles. `fire_guns`
    (`combat/systems.rs:57-60`) ZEROES the fractional-round accumulator on a released
    trigger, so a pulsed trigger loses a partial round per pulse and the effective rate is
    below `rate_hz`. The handoff's "~82 rounds/s" is unsourced; this replaces it.
R2  p_hit per round on a (range x systematic lay-miss x bandit g) grid, Monte-Carlo through
    the repo's tick-exact mirror of `combat/ballistics.rs` (`gunlay.p_hit_vec`, pinned to
    Rust by `crates/astrum-sim/tests/ballistics.rs::golden_trajectory_pins_the_python_mirror`),
    including the real 8 mrad SplitMix-equivalent dispersion cone, drag, gravity, Coriolis
    and the target's maneuver during time of flight. The lay-miss axis exists because a
    "lethal tick" is `lay_miss_m < 8 m`, i.e. an ANNULUS, not a point: the required dwell at
    the edge of the detector's own definition is not the required dwell at zero lay error.
R3  The required CUMULATIVE dwell as a distribution, not a point: rounds-to-50-hits is
    NegativeBinomial(50, p) sampled exactly, dwell = rounds / measured rate. Plus the
    magazine feasibility bound (511 rounds => 50 hits needs p >= 0.0978).
A2  The achievable dwell, in the TRUE sim: four scripted trackers x two opponents x four
    cells. Per tick it records `bfm_detect.Sample.lethal` (the program's own tLeth
    predicate), and then -- the point of the section -- the TRUE p_hit of a shot taken at
    that tick, propagated with real drag/gravity/dispersion against the bandit's own
    RECORDED future track. Summing that over the trajectory gives E[hits | trigger held],
    which is the honest "could this trajectory have killed?" number and needs no transfer
    model at all.
    The arms: `expert` = the env's own `PursuitExpert` (with its own fire gate);
    `lagtrack_fixed` = `bfm.LagPursuitTracker` with A2's throttle-units defect corrected;
    `lead_keep` / `lead_max` = written HERE, because every stick in `bfm.py` aims at the
    bandit's POSITION (`bfm._point_at` consumes `rel.ata`) and a gun solution has to aim at
    the LEAD point -- these roll the lift vector onto `gunlay.intercept`'s boresight and pull
    at gain 2.0, `lead_keep` holding closure near zero and `lead_max` at full power. They
    hold the trigger down: the question is geometry, not trigger discipline.
A1  The analytic bound: the co-turning steady state for two identical airframes. Solves
    Delta/2 + asin(2V sin(Delta/2)/1030) = available nose-off-velocity angle for the
    sustainable tracking separation, converts it to a range, and shows the sensitivity to
    altitude through the measured (V, n) pair. Parameters come from A2's measurement of the
    T3's actual break, not from assumption.

D6 CONTROL, IN THE SAME INVOCATION, SAME ENV BUILD, SAME SEEDS, SAME THRESHOLDS. The
control for a dwell bound is a target that does NOT break: every arm is also flown against
`opponent_tier=1` (the straight-and-level Tier-1 dummy). Without it, "the tracker holds
0.2 s" cannot be told apart from "the tracker cannot track anything". The T3-vs-dummy
contrast IS the attribution to the break, and it is the control with power here (a
checkpoint control is not: the question is about the airframe/opponent pair, not about a
policy).

================================================================================
WHAT IS EXPLICITLY *NOT* MEASURED
================================================================================
* Not a search and not an existence proof for a winning trajectory. A2's arms are two fixed
  scripted laws; a low number is a LOWER bound on achievability and NEVER evidence that the
  task is unwinnable. That is D3b's job, and even D3b may only ever say "no solution found".
* Not a claim about any learned policy. No checkpoint is loaded. The recorded c96/c95/c87
  numbers quoted in the verdict are quoted from the record as such, not re-measured here.
* Not a reward/benchmark change. `R_EFF`, the scripted fire gates and tier 3 are untouched;
  this file imports them read-only.
* `lay_miss_m` (and hence `Sample.lethal`, and hence tLeth) is drag- and gravity-free, so
  the dwell numbers are OPTIMISTIC as a proxy for lethality -- a perfect drag-free lay still
  misses by ~15 m at 700 m. That optimism is not corrected in the dwell column; it is
  QUANTIFIED by the true-p_hit column beside it, which uses the real ballistics.
* Episode length, train-log kill%, reward-term sums and the D5 ratio are not read anywhere.

================================================================================
PRE-REGISTERED INTERPRETATION (fixed before the run)
================================================================================
P1  If achievable >> required in some range band => the fight is winnable ON PAPER, the gap
    is skill/discovery, and D3b is worth building. The band must be NAMED.
P2  If achievable << required everywhere, the only admissible sentence is "no dwell budget
    found at ranges R under model M". It bounds achievability from below only and may not
    motivate changing the task.
P3  The TRACKING-vs-GEOMETRY split is pre-registered as a decomposition of the ticks inside
    600 m:
      geometrically shootable = angle(required boresight, my velocity vector) <= the
        measured alpha ceiling (16.7 deg). Below it, no stick input can produce the lay:
        the nose cannot be pointed there. That is a GEOMETRY problem.
      tracking conversion = P(lay_miss < 8 m | geometrically shootable). A high shootable
        fraction with a low conversion is a TRACKING problem.
    If the shootable fraction inside 600 m is small, the lever is BFM/entry geometry; if it
    is large and conversion is small, the lever is the inner loop / aim.
P4  Units are quoted twice everywhere: one control step = action_repeat 5 / 120 Hz =
    41.7 ms, 12 steps = 0.5 s, 36 steps = 1.5 s.

Run (from `controllers/rl`, inside the sim container, NO --user):
    uv run python _diag_dwell_budget.py 2>&1 | tee _dwell_budget_out.txt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
from astrum_rl import _paths  # noqa: E402

_paths.require_astrum_env()

from astrum_dogfight import bfm as BFM  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import gunlay  # noqa: E402
from astrum_dogfight.env import make_tier3_env  # noqa: E402
from astrum_dogfight.geometry import body_axes_enu, relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402

DEG = 180.0 / math.pi
G0 = gunlay.G0

# The measured airframe facts this box is required to USE rather than assume
# (docs/bfm-program-handoff.md, A3 + curriculum-93 autopsy).
ALPHA_CEIL_DEG = 16.7  # measured real alpha ceiling (vs C.ALPHA_REF = 30 deg)
NZ_SUSTAINED_G = 5.8  # measured max sustained pull (9 g was the envelope CORNER)
RADIUS_RECORDED_M = 795.0  # the F-16 turn radius figure carried in the board


# ======================================================================================
# R0 -- constants, with provenance
# ======================================================================================
def sec_r0() -> dict:
    print("=" * 100)
    print("R0  GUN / DAMAGE CONSTANTS, READ OUT OF SOURCE")
    print("=" * 100)
    rows = [
        ("rate_hz (cyclic)", gunlay.GUN_RATE_HZ, "rd/s", "combat/components.rs Gun::m61a1"),
        ("muzzle_speed", gunlay.MUZZLE_SPEED_MPS, "m/s", "combat/components.rs Gun::m61a1"),
        ("dispersion (1sigma)", gunlay.DISPERSION_RAD * 1e3, "mrad", "combat/components.rs"),
        ("ammo (magazine)", 511, "rd", "combat/components.rs Gun::m61a1"),
        ("damage_per_hit", gunlay.DAMAGE_PER_HIT, "HP", "combat/resources.rs CombatParams"),
        ("hit_radius_m", gunlay.HIT_RADIUS_M, "m", "combat/resources.rs CombatParams"),
        ("bullet_ttl_s", gunlay.BULLET_TTL_S, "s", "combat/resources.rs CombatParams"),
        ("hp_max (spawn)", gunlay.HP_FULL, "HP", "scenario.rs default_health / env health=None"),
        ("sim dt", gunlay.SIM_DT * 1e3, "ms", "scenario rate_hz 120"),
    ]
    for name, val, unit, src in rows:
        print(f"  {name:<22} {val:>10.4g} {unit:<6} <- {src}")
    hits = gunlay.HITS_TO_KILL
    print(f"\n  HITS_TO_KILL = ceil(hp_max/damage_per_hit) = {hits}")
    print(
        f"  1 sigma dispersion footprint: {gunlay.DISPERSION_RAD * 300:.2f} m @300 m, "
        f"{gunlay.DISPERSION_RAD * 600:.2f} m @600 m  vs HIT_RADIUS {gunlay.HIT_RADIUS_M:.1f} m"
    )
    print(
        f"  magazine as TRIGGER TIME at the cyclic rate: 511/{gunlay.GUN_RATE_HZ:.0f} = "
        f"{511.0 / gunlay.GUN_RATE_HZ:.2f} s"
    )
    print(f"  p_hit needed for a kill inside ONE magazine: {hits}/511 = {hits / 511.0:.4f}")
    print(
        f"  LETHAL predicate (bfm_detect.Sample.lethal): lay_miss_m < {C.HIT_RADIUS_M:.0f} m "
        f"AND rng <= {B.LETHAL_RANGE_M:.0f} m"
    )
    print("  repo p_hit curve (constants.PHIT_*, 5 g bandit, computable lay, 5000 m):")
    print("    " + "  ".join(f"{r:.0f}m:{p:.3f}" for r, p in zip(C.PHIT_RANGE_M, C.PHIT_PER_ROUND)))
    return dict(hits_to_kill=hits)


# ======================================================================================
# R1 -- the rate of fire the sim actually delivers
# ======================================================================================
def _rate_env(args):
    # Bandit on the BEAM at 700 m so a forward burst cannot reach it (we are counting the
    # gun's own cadence, not scoring hits). Tier-1 dummy, no jitter => deterministic.
    return make_tier3_env(
        posture="offensive",
        opponent_tier=1,
        aspect_deg=90.0,
        separation_m=700.0,
        max_time_s=30.0,
        rate_hz=120.0,
    )


def sec_r1(args) -> dict:
    print()
    print("=" * 100)
    print("R1  RATE OF FIRE, MEASURED IN THE TRUE SIM (ammo decrement, trigger duty cycles)")
    print("=" * 100)
    print("  `fire_guns` (combat/systems.rs:57-60) zeroes `Gun::accum` whenever the trigger is")
    print("  released, so a pulsed trigger discards a partial round per pulse. Patterns are in")
    print("  120 Hz SIM TICKS; one control step at action_repeat 5 = 5 ticks = 41.7 ms.")
    patterns = [
        ("continuous", 1, 0),
        ("5 on / 5 off (alternate control steps)", 5, 5),
        ("5 on / 10 off (1 in 3 control steps)", 5, 10),
        ("1 on / 4 off (per-tick pulse)", 1, 4),
    ]
    out = {}
    for label, on, off in patterns:
        env = _rate_env(args)
        env.reset()
        base = env.unwrapped
        lnr = base.learner_id
        ammo0 = int(base._states[lnr].ammo)
        ticks = int(args.rate_ticks)
        on_ticks = 0
        k = 0
        while k < ticks:
            for _ in range(on):
                if k >= ticks:
                    break
                env.step([0.0, 0.0, 0.0, 0.6, 1.0])
                on_ticks += 1
                k += 1
            for _ in range(off):
                if k >= ticks:
                    break
                env.step([0.0, 0.0, 0.0, 0.6, 0.0])
                k += 1
        spent = ammo0 - int(base._states[lnr].ammo)
        wall_s = ticks * gunlay.SIM_DT
        on_s = on_ticks * gunlay.SIM_DT
        r_wall = spent / wall_s
        r_on = spent / on_s if on_s > 0 else float("nan")
        out[label] = dict(spent=spent, r_wall=r_wall, r_on=r_on, on_s=on_s)
        print(
            f"  {label:<42} {spent:>4d} rd in {wall_s:.2f} s "
            f"({on_s:.2f} s trigger-on) -> {r_wall:6.1f} rd/s wall, {r_on:6.1f} rd/s trigger-on"
        )
        env.close()
    return out


# ======================================================================================
# R2 -- p_hit per round: range x systematic lay-miss x bandit g
# ======================================================================================
def _true_arc(v: float, n: float, horizon_s: float, alt_m: float) -> np.ndarray:
    """(M,3) ENU path of a level n-g turn from the origin heading +E, at `alt_m`."""
    dt = gunlay.SIM_DT
    ticks = np.arange(int(horizon_s / dt) + 3) * dt
    if n <= 1.0:
        return np.stack([v * ticks, np.zeros_like(ticks), np.full_like(ticks, alt_m)], axis=1)
    omega = G0 * math.sqrt(n * n - 1.0) / v
    r = v / omega
    th = omega * ticks
    return np.stack([r * np.sin(th), r * (1.0 - np.cos(th)), np.full_like(th, alt_m)], axis=1)


def _rotate_off(d: np.ndarray, delta: float, az: float) -> np.ndarray:
    """Rotate unit `d` by `delta` rad, in the tangent direction at azimuth `az`."""
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    t = math.cos(az) * e1 + math.sin(az) * e2
    out = math.cos(delta) * d + math.sin(delta) * t
    return out / np.linalg.norm(out)


def sec_r2(args) -> dict:
    print()
    print("=" * 100)
    print("R2  p_hit PER ROUND vs (range, systematic lay-miss, bandit g)")
    print("=" * 100)
    print(
        f"  Model M: tail chase, shooter {args.v_shooter:.0f} m/s co-heading, bandit "
        f"{args.v_target:.0f} m/s in a level n-g turn, alt {args.phit_alt:.0f} m."
    )
    print("  Lay = gunlay.intercept (the COMPUTABLE, velocity-inheritance-corrected, drag-free")
    print("  solution -- exactly the lay `geometry.lay_miss_m` scores), then rotated off by the")
    print("  angle that produces the stated lay_miss, averaged over 4 tangent azimuths.")
    print("  Propagation = the tick-exact ballistics mirror (drag+gravity+Coriolis) with the real")
    print(
        f"  8 mrad dispersion cone, {args.phit_samples} rounds/cell. HIT = closest approach of the"
    )
    print("  swept segment <= 8 m, the same test `resolve_hits` runs.")
    ranges = [float(x) for x in args.phit_ranges.split(",")]
    misses = [float(x) for x in args.phit_misses.split(",")]
    gees = [float(x) for x in args.phit_gees.split(",")]
    grid = {}
    for n in gees:
        omega = 0.0 if n <= 1.0 else math.degrees(G0 * math.sqrt(n * n - 1.0) / args.v_target)
        print(
            f"\n  --- bandit {n:.1f} g ({omega:.1f} deg/s, radius "
            f"{(args.v_target / max(1e-9, G0 * math.sqrt(max(n * n - 1.0, 1e-9)) / args.v_target)):.0f} m) ---"
        )
        hdr = "   rng |" + "".join(f"  lay{m:>4.1f}m" for m in misses) + "  | perfect-lay sys miss"
        print(hdr)
        for R in ranges:
            horizon = 2.5 * R / gunlay.MUZZLE_SPEED_MPS + 0.10
            arc = _true_arc(args.v_target, n, horizon, args.phit_alt)
            tp = arc[0]
            tv = (arc[1] - arc[0]) / gunlay.SIM_DT
            sp = tp - np.array([R, 0.0, 0.0])
            sv = np.array([args.v_shooter, 0.0, 0.0])
            path = np.concatenate([np.arange(len(arc))[:, None] * gunlay.SIM_DT, arc], axis=1)
            sol = gunlay.intercept(sp, sv, tp, tv)
            row = [f"  {R:>4.0f} |"]
            sysmiss = float("nan")
            for m in misses:
                if not sol.feasible:
                    grid[(n, R, m)] = 0.0
                    row.append("   infeas")
                    continue
                delta = math.asin(min(1.0, m / max(1e-9, gunlay.MUZZLE_SPEED_MPS * sol.tof_s)))
                ps = []
                for j in range(4):
                    d = _rotate_off(sol.boresight, delta, j * math.pi / 2.0)
                    p, mm = gunlay.p_hit_vec(
                        d,
                        sp,
                        sv,
                        tp,
                        tv,
                        samples=args.phit_samples,
                        seed=1000 + j,
                        target_path=path,
                        alt_offset=0.0,
                    )
                    ps.append(p)
                    if delta == 0.0:
                        sysmiss = mm
                        break
                p = float(np.mean(ps))
                grid[(n, R, m)] = p
                row.append(f"   {p:6.3f}")
            row.append(f"  | {sysmiss:8.2f} m (mean closest approach, zero-lay-error shot)")
            print(" ".join(row))
    # reproduce the repo's own curve as a pin
    print("\n  PIN vs constants.PHIT_PER_ROUND (5 g, zero lay miss):")
    for r, p_repo in zip(C.PHIT_RANGE_M, C.PHIT_PER_ROUND):
        if (5.0, r, 0.0) in grid:
            print(f"    {r:>5.0f} m  here {grid[(5.0, r, 0.0)]:.3f}  repo {p_repo:.3f}")
    return dict(grid=grid, ranges=ranges, misses=misses, gees=gees)


# ======================================================================================
# R3 -- the required cumulative dwell, as a distribution
# ======================================================================================
def sec_r3(r2: dict, rate: float, args) -> dict:
    print()
    print("=" * 100)
    print(f"R3  REQUIRED CUMULATIVE DWELL for {gunlay.HITS_TO_KILL} hits, at {rate:.1f} rd/s")
    print("=" * 100)
    print("  Probability model: within a burst the SYSTEMATIC miss (lay error + drag/gravity +")
    print("  the bandit's maneuver during time of flight) is ~constant, and the only per-round")
    print("  randomness is the 8 mrad dispersion kick, which `disperse` draws i.i.d. per round")
    print("  from the shared SplitMix64 stream. So hits within a burst are Binomial(p) at the p")
    print("  for that (range, systematic lay-miss) cell, and rounds-to-50-hits is")
    print("  NegativeBinomial(50, p). Dwell = rounds / rate. ACROSS cells p varies, so the table")
    print("  IS the distribution over geometry; it is not folded into one number.")
    print("  Reported: mean = 50/p exactly, plus sampled median and p90 (200k draws).")
    print("  Dwell is CUMULATIVE, not contiguous -- HP does not regenerate. The binding")
    print("  contiguity constraint is instead the magazine: 511 rd = 5.11 s of trigger time.")
    rng = np.random.default_rng(12345)
    grid = r2["grid"]
    out = {}
    for n in r2["gees"]:
        print(
            f"\n  --- bandit {n:.1f} g --- (INFEAS = 50 hits impossible inside one 511-rd magazine)"
        )
        print("   rng | " + " | ".join(f"lay{m:>4.1f}m: p / mean s / p90 s" for m in r2["misses"]))
        for R in r2["ranges"]:
            cells = []
            for m in r2["misses"]:
                p = grid.get((n, R, m), 0.0)
                if p <= 0.0:
                    cells.append(f"{'0.000':>5} /  never  /  never")
                    out[(n, R, m)] = None
                    continue
                mean_rd = gunlay.HITS_TO_KILL / p
                fails = rng.negative_binomial(gunlay.HITS_TO_KILL, p, size=args.nb_samples)
                trials = fails + gunlay.HITS_TO_KILL
                med, p90 = np.percentile(trials, [50, 90])
                tag = "" if mean_rd <= 511 else "!"
                cells.append(f"{p:5.3f} / {mean_rd / rate:6.2f}{tag} / {p90 / rate:6.2f}")
                out[(n, R, m)] = dict(
                    p=p, mean_rd=mean_rd, mean_s=mean_rd / rate, med_s=med / rate, p90_s=p90 / rate
                )
            print(f"  {R:>4.0f} | " + " | ".join(cells))
    print("\n  ! = mean rounds exceeds the 511-round magazine (one belt cannot finish the kill)")
    # the magazine-limited maximum kill range, on the repo curve and on the measured grid
    p_need = gunlay.HITS_TO_KILL / 511.0
    xs = np.linspace(100.0, 1200.0, 4401)
    ps = np.interp(xs, C.PHIT_RANGE_M, C.PHIT_PER_ROUND, left=1.0, right=0.0)
    ok = xs[ps >= p_need]
    r_max = float(ok.max()) if len(ok) else float("nan")
    print(f"\n  MAGAZINE BOUND: 50 hits inside 511 rd needs p >= {p_need:.4f}; on the repo's own")
    print(f"  5 g curve that fails beyond rng ~= {r_max:.0f} m. Independent of any dwell claim,")
    print(
        f"  a one-belt kill is arithmetically impossible past ~{r_max:.0f} m against a 5 g bandit."
    )
    return out


# ======================================================================================
# A2 -- achievable dwell in the true sim
# ======================================================================================
CELLS = {
    # label: (kwargs for make_tdmpc_env, max_time_s)
    "chase-asp0": dict(aspect_deg=0.0, separation_m=700.0, max_time_s=60.0, me_dalt_ft=0.0),
    "chase-asp30": dict(aspect_deg=30.0, separation_m=700.0, max_time_s=60.0, me_dalt_ft=0.0),
    "chase-asp55": dict(aspect_deg=55.0, separation_m=700.0, max_time_s=60.0, me_dalt_ft=0.0),
    "merge-lat1400": dict(
        aspect_deg=27.8,
        separation_m=3000.0,
        max_time_s=180.0,
        me_dalt_ft=2000.0,
        opp_heading_deg=180.0,
    ),
}


def _fix_thr(a4):
    """Reinterpret a stick's throttle as the POWER FRACTION its constants were written as and
    convert to agent space (inverse of `map_tdmpc_action`'s native = 0.5(a+1)). This is A2's
    documented defect 3 in `_diag_stick_ceiling.py`; the tracker arm is run BOTH ways."""
    a = np.asarray(a4, np.float32).copy()
    a[3] = float(np.clip(2.0 * float(a[3]) - 1.0, -1.0, 1.0))
    return a


def _point_at_dir(me, des, pull_gain: float, deadband_rad: float = math.radians(5.0)):
    """(ail, elev) that rolls the lift vector onto an ARBITRARY direction `des` and pulls the nose
    to it. Sign conventions are taken from `bfm._lift_on_target_bank_err` / `bfm._point_at`:
    atan2(starboard component, up component), target abeam-right => +90 deg => roll right; and
    elev < 0 = nose-up pull (`Control::Raw` is an Nz interface with the n_z sign inverted).

    `bfm._point_at` can only aim at the bandit's POSITION (it consumes `rel.ata*`). A gun solution
    has to aim at the LEAD point, which is what this exists for."""
    nose, stbd, down = body_axes_enu(me.phi, me.theta, me.psi)
    d = np.asarray(des, float)
    d = d / max(1e-9, float(np.linalg.norm(d)))
    off = math.acos(float(np.clip(np.dot(d, nose), -1.0, 1.0)))
    bank_err = math.atan2(float(np.dot(d, stbd)), float(-np.dot(d, down)))
    ail = float(np.clip(C.K_PURSUE_ROLL * bank_err, -1.0, 1.0))
    if off < deadband_rad:
        ail = 0.0
    elev = float(np.clip(-pull_gain * off, -1.0, 1.0))
    return ail, elev


def _make_arm(name):
    """Return `act(me, bd, rel, st, info) -> agent-space len-5 action`."""
    if name in ("lead_max", "lead_keep", "pure_max"):

        def act(me, bd, rel, st, info):
            if name == "pure_max":
                des = np.asarray(bd.pos, float) - np.asarray(me.pos, float)
            else:
                sol = gunlay.intercept(me.pos, me.vel, bd.pos, bd.vel)
                des = (
                    sol.boresight
                    if sol.feasible
                    else np.asarray(bd.pos, float) - np.asarray(me.pos, float)
                )
            ail, elev = _point_at_dir(me, des, pull_gain=2.0)
            ail, elev = BFM._deck_guard(me, ail, elev)
            if name == "lead_keep":
                thr_native = max(0.25, min(1.0, 0.85 - max(0.0, float(rel.closure)) / 300.0))
            else:
                thr_native = 1.0
            a = np.zeros(C.ACTION_DIM, np.float32)
            a[0], a[1], a[2] = ail, elev, 0.0
            a[3] = float(np.clip(2.0 * thr_native - 1.0, -1.0, 1.0))
            a[4] = 1.0
            return a

        return act
    if name == "expert":

        def act(me, bd, rel, st, info):
            a = np.zeros(C.ACTION_DIM, np.float32)
            ea = info.get("expert_action")
            if ea is not None:
                a[:] = np.asarray(ea, np.float32)
            return a

        return act
    if name in ("lagtrack_fixed", "lagtrack_asflown"):
        fix = name.endswith("fixed")

        def act(me, bd, rel, st, info):
            trk = st.get("trk")
            if trk is None:
                trk = st["trk"] = BFM.LagPursuitTracker()
                trk.reset(me, rel)
            a4, _ = trk.step(me, rel)
            a4 = _fix_thr(a4) if fix else np.asarray(a4, np.float32)
            a = np.zeros(C.ACTION_DIM, np.float32)
            a[:4] = np.asarray(a4, np.float32)[:4]
            a[4] = 1.0  # hold the trigger: the dwell question is about geometry, not discipline
            return a

        return act
    raise ValueError(name)


def _roll_cell(arm_name, opp_tier, cell, args, seed0):
    """Fly `args.episodes` episodes. Returns a list of per-episode records."""
    kw = dict(CELLS[cell])
    mt = kw.pop("max_time_s")
    dalt = kw.pop("me_dalt_ft", 0.0)
    env = make_tdmpc_env(
        3,
        posture="offensive",
        opponent_tier=opp_tier,
        action_repeat=args.action_repeat,
        frame_stack=1,
        max_time_s=mt,
        rate_hz=args.rate_hz,
        jitter=args.jitter,
        jitter_seed=seed0,
        **({"me_dalt_ft": dalt} if dalt else {}),
        **kw,
    )
    base = env.unwrapped
    lnr, opp = base.learner_id, base.opponent_id
    dt = args.action_repeat / args.rate_hz
    arm = _make_arm(arm_name)
    eps = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=seed0 + ep)
        st = {}
        rec = dict(
            t=[],
            me_pos=[],
            me_vel=[],
            bd_pos=[],
            bd_vel=[],
            nose=[],
            rng=[],
            ata=[],
            aspect=[],
            lay=[],
            lethal=[],
            nz_me=[],
            nz_bd=[],
            alt_me=[],
            alt_bd=[],
            opp_state=[],
            alpha_req=[],
            nose_off_vel=[],
            tas_me=[],
            tas_bd=[],
            opp_turn_rate=[],
            alpha_me=[],
        )
        hp0 = float(base._states[opp].health)
        ammo0 = int(base._states[lnr].ammo)
        t = 0.0
        event = None
        prev_bd_vhat = None
        while True:
            me, bd = base._states[lnr], base._states[opp]
            rel = relative(me, bd)
            a = arm(me, bd, rel, st, info)
            nose, _, _ = body_axes_enu(me.phi, me.theta, me.psi)
            sol = gunlay.intercept(me.pos, me.vel, bd.pos, bd.vel)
            vhat_me = me.vel / max(1e-9, np.linalg.norm(me.vel))
            if sol.feasible:
                areq = math.acos(float(np.clip(np.dot(sol.boresight, vhat_me), -1.0, 1.0)))
            else:
                areq = float("nan")
            vhat_bd = np.asarray(bd.vel, float) / max(1e-9, float(np.linalg.norm(bd.vel)))
            if prev_bd_vhat is not None:
                dot = float(np.clip(np.dot(vhat_bd, prev_bd_vhat), -1.0, 1.0))
                rec["opp_turn_rate"].append(math.acos(dot) / dt)
            else:
                rec["opp_turn_rate"].append(float("nan"))
            prev_bd_vhat = vhat_bd
            rec["t"].append(t)
            rec["me_pos"].append(np.asarray(me.pos, float).copy())
            rec["me_vel"].append(np.asarray(me.vel, float).copy())
            rec["bd_pos"].append(np.asarray(bd.pos, float).copy())
            rec["bd_vel"].append(np.asarray(bd.vel, float).copy())
            rec["nose"].append(nose.copy())
            rec["rng"].append(float(rel.rng))
            rec["ata"].append(float(rel.ata))
            rec["aspect"].append(float(rel.aspect))
            rec["lay"].append(float(rel.lay_miss_m))
            rec["lethal"].append(
                bool(rel.lay_miss_m < C.HIT_RADIUS_M and rel.rng <= B.LETHAL_RANGE_M)
            )
            rec["nz_me"].append(float(me.nz))
            rec["nz_bd"].append(float(bd.nz))
            rec["alt_me"].append(float(me.pos[2]))
            rec["alt_bd"].append(float(bd.pos[2]))
            rec["tas_me"].append(float(me.airspeed))
            rec["tas_bd"].append(float(bd.airspeed))
            rec["alpha_req"].append(areq)
            rec["alpha_me"].append(float(me.alpha))
            rec["nose_off_vel"].append(math.acos(float(np.clip(np.dot(nose, vhat_me), -1.0, 1.0))))
            rec["opp_state"].append(str(getattr(base.opponent, "state", "-")))
            obs, _r, term, trunc, info = env.step(a)
            t += dt
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
        me, bd = base._states[lnr], base._states[opp]
        for k in (
            "t",
            "rng",
            "ata",
            "aspect",
            "lay",
            "nz_me",
            "nz_bd",
            "alt_me",
            "alt_bd",
            "alpha_req",
            "nose_off_vel",
            "tas_me",
            "tas_bd",
            "opp_turn_rate",
            "alpha_me",
        ):
            rec[k] = np.asarray(rec[k], float)
        rec["lethal"] = np.asarray(rec["lethal"], bool)
        for k in ("me_pos", "me_vel", "bd_pos", "bd_vel", "nose"):
            rec[k] = np.asarray(rec[k], float)
        rec["event"] = event
        rec["dealt"] = hp0 - float(bd.health)
        rec["taken"] = C.HP_MAX - float(me.health)
        rec["rounds"] = ammo0 - int(me.ammo)
        rec["t_end"] = t
        eps.append(rec)
    env.close()
    return eps


def _true_phit_integral(rec, args, dt):
    """E[hits | trigger held] over the ticks inside `args.phit_gate_m`, using the TRUE
    ballistics against the bandit's own recorded future track and the shooter's ACTUAL nose.

    Returns (e_hits_all, e_hits_lethal_only, n_eval, n_skipped_tail, p_max)."""
    rounds_per_step = args.rate_used * dt
    n = len(rec["t"])
    tail = int(math.ceil(1.0 / dt))  # the propagator needs ~tof of future track
    idx = [i for i in range(max(0, n - tail)) if rec["rng"][i] <= args.phit_gate_m]
    n_skip = sum(1 for i in range(max(0, n - tail), n) if rec["rng"][i] <= args.phit_gate_m)
    if args.phit_max_ticks and len(idx) > args.phit_max_ticks:
        stride = int(math.ceil(len(idx) / args.phit_max_ticks))
        idx = idx[::stride]
    else:
        stride = 1
    e_all = 0.0
    e_leth = 0.0
    p_max = 0.0
    for i in idx:
        t0 = rec["t"][i]
        path = np.column_stack([rec["t"][i:] - t0, rec["bd_pos"][i:]])
        p, _ = gunlay.p_hit_vec(
            rec["nose"][i],
            rec["me_pos"][i],
            rec["me_vel"][i],
            rec["bd_pos"][i],
            rec["bd_vel"][i],
            samples=args.true_samples,
            seed=7,
            target_path=path,
            alt_offset=0.0,
        )
        e_all += p * rounds_per_step * stride
        if rec["lethal"][i]:
            e_leth += p * rounds_per_step * stride
        p_max = max(p_max, p)
    return e_all, e_leth, len(idx), n_skip, p_max, stride


def _runs_true(mask) -> list:
    out, cur = [], 0
    for v in mask:
        if v:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def sec_a2(args) -> dict:
    print()
    print("=" * 100)
    print("A2  ACHIEVABLE DWELL IN THE TRUE SIM  (D6 control = the non-breaking Tier-1 dummy)")
    print("=" * 100)
    dt = args.action_repeat / args.rate_hz
    print(
        f"  {args.episodes} eps/cell, jitter {args.jitter}, one Sample per control step "
        f"(dt = {dt * 1000:.1f} ms). tLeth = lay_miss < 8 m AND rng <= {B.LETHAL_RANGE_M:.0f} m."
    )
    print(
        f"  E[hits] uses the TRUE ballistics at each tick inside {args.phit_gate_m:.0f} m "
        f"({args.true_samples} rounds/tick, bandit's recorded track), x "
        f"{args.rate_used * dt:.2f} rd per control step. 50 hits = a kill."
    )
    print("  Trackers hold the trigger down; the `expert` arm uses the env's own PursuitExpert")
    print("  (its own fire gate), so its `rounds` column is its discipline, not the budget.")
    combos = []
    for opp_tier, opp_lbl in ((0, "T3"), (1, "dummy")):
        for arm in args.arms.split(","):
            arm = arm.strip()
            if not arm:
                continue
            combos.append((arm, opp_tier, opp_lbl))
    results = {}
    for cell in args.cells.split(","):
        cell = cell.strip()
        if not cell:
            continue
        print(f"\n  ################ CELL {cell} ################")
        print(
            f"  {'arm':<15} {'opp':<6} | {'minrng':>6} {'absmin':>6} {'in1k':>5} {'in800':>5} "
            f"{'in600':>5} {'in400':>5} {'areq600':>7} {'ata600':>6} | {'tLethTot':>8} "
            f"{'/ep':>6} {'long':>6} | {'E[hits]/ep':>10} {'pmax':>5} | {'shoot':>5} "
            f"{'conv':>6} | {'k':>2} {'dealt':>6} {'rds':>5} {'ends'}"
        )
        for arm, opp_tier, opp_lbl in combos:
            # ⚠ FIXED 2026-09-06 (D3a's verifiers, handoff §9). This was
            #   `args.seed + 1000 * (hash(cell) % 7) + 37 * opp_tier`
            # and it had TWO defects, both load-bearing:
            #   (1) Python's str hashing is PYTHONHASHSEED-randomised, and PYTHONHASHSEED is
            #       unset in this container (`sys.flags.hash_randomization == 1`), so three
            #       invocations of the SAME command drew three different seed sets and NOTHING
            #       this file printed was reproducible. Replaced with a stable digest.
            #   (2) the `+ 37 * opp_tier` term gave the Tier-1 D6 control a DIFFERENT seed than
            #       the T3 arm it controls, so the control was build-matched but not seed-matched
            #       — which is the one property a negative control has to have. Dropped, so the
            #       arm and its control now fly the identical jitter and reset seeds.
            cell_key = int(hashlib.blake2s(cell.encode(), digest_size=4).hexdigest(), 16)
            seed0 = args.seed + 1000 * (cell_key % 7)
            eps = _roll_cell(arm, opp_tier, cell, args, seed0)
            tot_leth = int(sum(int(r["lethal"].sum()) for r in eps))
            longest = max((max(_runs_true(r["lethal"]), default=0) for r in eps), default=0)
            e_all = e_leth = 0.0
            n_eval = n_skip = 0
            p_max = 0.0
            in600 = in800 = in1k = in400 = shoot = conv = 0
            minrngs, areq600, ata600 = [], [], []
            for r in eps:
                a, b, ne, ns, pm, _st = _true_phit_integral(r, args, dt)
                e_all += a
                e_leth += b
                n_eval += ne
                n_skip += ns
                p_max = max(p_max, pm)
                m600 = r["rng"] <= B.LETHAL_RANGE_M
                in600 += int(m600.sum())
                in800 += int((r["rng"] <= 800.0).sum())
                in1k += int((r["rng"] <= 1000.0).sum())
                in400 += int((r["rng"] <= 400.0).sum())
                minrngs.append(float(r["rng"].min()))
                sh = m600 & (r["alpha_req"] <= math.radians(ALPHA_CEIL_DEG))
                shoot += int(sh.sum())
                conv += int((sh & r["lethal"]).sum())
                gate = r["rng"] <= 600.0
                if gate.sum():
                    areq600.extend(np.degrees(r["alpha_req"][gate]).tolist())
                    ata600.extend(np.degrees(r["ata"][gate]).tolist())
            kills = sum(1 for r in eps if r["event"] == "kill" or (r["dealt"] >= C.HP_MAX - 1e-6))
            ends = {}
            for r in eps:
                ends[str(r["event"])] = ends.get(str(r["event"]), 0) + 1
            nep = len(eps)
            conv_pct = f"{100.0 * conv / shoot:5.1f}%" if shoot else "  n/a"
            med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")  # noqa: E731
            print(
                f"  {arm:<15} {opp_lbl:<6} | {med(minrngs):6.0f} {min(minrngs):6.0f} "
                f"{in1k:5d} {in800:5d} {in600:5d} {in400:5d} "
                f"{med(areq600):7.1f} {med(ata600):6.1f} | "
                f"{tot_leth * dt:7.2f}s {tot_leth * dt / nep:5.2f}s {longest * dt:5.2f}s | "
                f"{e_all / nep:10.2f} {p_max:5.3f} | {shoot:5d} {conv_pct:>6} | "
                f"{kills:2d} {sum(r['dealt'] for r in eps) / nep:6.1f} "
                f"{int(sum(r['rounds'] for r in eps) / nep):5d} "
                f"{','.join(f'{k}:{v}' for k, v in sorted(ends.items()))}"
            )
            results[(cell, arm, opp_lbl)] = dict(
                eps=eps,
                tot_leth=tot_leth,
                longest=longest,
                e_all=e_all,
                e_leth=e_leth,
                in600=in600,
                shoot=shoot,
                conv=conv,
                kills=kills,
                n_eval=n_eval,
                n_skip=n_skip,
            )
        print(
            "   cols: minrng = median over eps of the CLOSEST range reached (m); in1k/in800/in600 ="
        )
        print("         control steps inside that range, pooled over eps; areq600/ata600 = median")
        print(
            "         alpha_req / ata over the in-600 steps (deg); tLethTot//ep/long = lethal-tick"
        )
        print(
            "         dwell (s); E[hits]/ep = expected hits per episode if the trigger were held on"
        )
        print(
            "         every in-gate step (50 = a kill); shoot = in-600 steps with alpha_req <= "
            f"{ALPHA_CEIL_DEG} deg"
        )
        print(
            "         (the nose CAN be pointed at the lead point); conv = P(lay_miss<8 | shootable)."
        )
    return results


def sec_a2_breakstats(res: dict, args) -> dict:
    """The T3's MEASURED break -- the input A1 is required to use rather than assume."""
    print()
    print("=" * 100)
    print("A2b  THE T3's MEASURED BREAK (loaded ticks, rng < THREAT_RANGE 2500 m)")
    print("=" * 100)
    print(
        f"  {'cell / arm':<34} | {'nz p50':>6} {'nz p90':>6} {'TAS p50':>8} {'omega p50':>9} "
        f"{'radius':>7} {'alt p50':>8} | {'DEF%':>5} {'ENER%':>6} {'NEU%':>5} {'OFF%':>5}"
    )
    out = {}
    for cell, arm, opp in sorted(res):
        if opp != "T3":
            continue
        eps = res[(cell, arm, opp)]["eps"]
        nz, tas, om, alt = [], [], [], []
        states = {}
        for r in eps:
            gate = r["rng"] < C.THREAT_RANGE
            load = gate & (r["nz_bd"] > B.NZ_LOADED)
            nz.extend(r["nz_bd"][gate].tolist())
            tas.extend(r["tas_bd"][load].tolist())
            om.extend(r["opp_turn_rate"][load][np.isfinite(r["opp_turn_rate"][load])].tolist())
            alt.extend(r["alt_bd"][load].tolist())
            for s in r["opp_state"][: len(r["rng"])]:
                pass
            for i, s in enumerate(r["opp_state"]):
                if r["rng"][i] < C.THREAT_RANGE:
                    states[s] = states.get(s, 0) + 1
        tot = max(1, sum(states.values()))
        p = lambda xs, q: float(np.percentile(xs, q)) if len(xs) else float("nan")  # noqa: E731
        nz50, nz90 = p(nz, 50), p(nz, 90)
        tas50, om50, alt50 = p(tas, 50), p(om, 50), p(alt, 50)
        radius = tas50 / om50 if om50 and np.isfinite(om50) and om50 > 1e-6 else float("nan")
        print(
            f"  {cell + ' / ' + arm:<34} | {nz50:6.2f} {nz90:6.2f} {tas50:8.1f} "
            f"{math.degrees(om50) if np.isfinite(om50) else float('nan'):9.2f} {radius:7.0f} "
            f"{alt50:8.0f} | "
            f"{100.0 * states.get('defensive', 0) / tot:5.1f} "
            f"{100.0 * states.get('energy_recovery', 0) / tot:6.1f} "
            f"{100.0 * states.get('neutral', 0) / tot:5.1f} "
            f"{100.0 * states.get('offensive', 0) / tot:5.1f}"
        )
        out[(cell, arm)] = dict(
            nz50=nz50, nz90=nz90, tas50=tas50, om50=om50, radius=radius, alt50=alt50, states=states
        )
    print("   omega = |d v_hat/dt| of the BANDIT's velocity vector (full 3-D turn rate), on ticks")
    print("   with nz>3 inside 2500 m. radius = TAS/omega. States are Tier3Opponent.state.")
    return out


def sec_a2c(res: dict) -> None:
    """The measured POINTING envelope -- the other input A1 must not assume.

    A1's whole verdict turns on one number, the alpha budget (recorded as 16.7 deg, vs
    `C.ALPHA_REF` = 30 and vs `geometry.RelGeom.alpha_req`'s own docstring "lift-limited to
    ~10.7-15 deg"). This measures what the airframe actually delivered on these very
    trajectories: the FDM's own `alpha_rad`, and the angle between the nose and the velocity
    vector, which is the quantity the alpha cone is a bound on. Added AFTER the main run and
    measured in a second invocation; it is additive and changes no other section."""
    print()
    print("=" * 100)
    print("A2c  MEASURED POINTING ENVELOPE (deg, ticks with rng < 2500 m)")
    print("=" * 100)
    print(
        f"  {'cell / arm / opp':<42} | {'alpha p50':>9} {'p90':>6} {'p99':>6} {'max':>6} | "
        f"{'nose-off-vel p50':>16} {'p90':>6} {'p99':>6} {'max':>6}"
    )
    for cell, arm, opp in sorted(res):
        a, nv = [], []
        for r in res[(cell, arm, opp)]["eps"]:
            g = r["rng"] < C.THREAT_RANGE
            a.extend(np.degrees(r["alpha_me"][g]).tolist())
            nv.extend(np.degrees(r["nose_off_vel"][g]).tolist())
        if not a:
            continue
        q = lambda xs, p: float(np.percentile(xs, p))  # noqa: E731
        print(
            f"  {cell + ' / ' + arm + ' / ' + opp:<42} | {q(a, 50):9.2f} {q(a, 90):6.2f} {q(a, 99):6.2f} "
            f"{max(a):6.2f} | {q(nv, 50):16.2f} {q(nv, 90):6.2f} {q(nv, 99):6.2f} {max(nv):6.2f}"
        )
    print(
        f"   Compare the ALPHA_CEIL_DEG = {ALPHA_CEIL_DEG} used by A1, C.ALPHA_REF = "
        f"{math.degrees(C.ALPHA_REF):.0f} deg, and the 10.7-15 deg in `RelGeom.alpha_req`'s"
    )
    print("   docstring. nose-off-vel is the 3-D angle between the body X axis and the velocity")
    print("   vector: the true radius of the cone the boresight can be placed in.")


# ======================================================================================
# A1 -- the analytic co-turn bound
# ======================================================================================
def _coturn(V: float, n: float, alpha_nose_rad: float, muzzle: float = 1030.0) -> dict:
    """Two identical airframes co-speed on ONE circle (the symmetric steady state).

    Attacker trailing by arc Delta on a common circle of radius r: ata = aspect = Delta/2,
    range = 2 r sin(Delta/2), and the target's velocity component perpendicular to the LOS is
    2 V sin(Delta/2), so the required lead off the LOS is L = asin(2 V sin(Delta/2)/muzzle)
    and the nose must sit Delta/2 + L off the attacker's OWN velocity vector. That offset is
    an angle of attack, so it is bounded by the airframe. Solve for the largest Delta the
    airframe can hold."""
    if n <= 1.0:
        return dict(feasible=False)
    om = G0 * math.sqrt(n * n - 1.0) / V
    r = V / om

    def need(half):
        return half + math.asin(min(1.0, 2.0 * V * math.sin(half) / muzzle))

    lo, hi = 0.0, min(math.pi / 2, alpha_nose_rad)
    if need(hi) <= alpha_nose_rad:
        half = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if need(mid) <= alpha_nose_rad:
                lo = mid
            else:
                hi = mid
        half = lo
    R = 2.0 * r * math.sin(half)
    return dict(feasible=True, omega=om, r=r, half=half, R=R, ata=need(half))


def sec_a1(brk: dict, args) -> None:
    print()
    print("=" * 100)
    print("A1  ANALYTIC BOUND: the sustainable co-turning gun solution for IDENTICAL airframes")
    print("=" * 100)
    print("  Both jets are the same f16 at the same load factor, so neither out-turns the other:")
    print("  the steady state is a common circle with the attacker trailing by arc Delta, which")
    print("  is the recorded 'neutral geometric draw' (ata pinned 74-93 deg). On that circle")
    print("    ata = aspect = Delta/2,  range = 2 r sin(Delta/2),  r = V^2/(g sqrt(n^2-1)),")
    print("  and the required lead off the LOS is L = asin(2 V sin(Delta/2)/1030). The nose must")
    print("  therefore sit (Delta/2 + L) off its OWN velocity vector -- which is an angle of")
    print("  attack, hence bounded by the airframe. In a level turn only alpha*sin(phi) of the")
    print("  offset lies in the turn plane (phi = acos(1/n); 0.985 alpha at 5.8 g), so both the")
    print("  optimistic (alpha) and the level-turn (alpha sin phi) budgets are shown.")
    print("  ==> a tracking solution is SUSTAINABLE only inside R*, and the dwell there is")
    print("      unbounded by geometry (the T3 stays DEFENSIVE while ata_learner < 45 deg and")
    print("      rng < 2500 m, so it never relents; only the deck or ENERGY_RECOVERY ends it).")
    print()
    print(
        f"  {'V m/s':>6} {'n g':>5} {'alpha budget':>13} | {'radius m':>9} {'omega d/s':>9} "
        f"{'Delta/2':>8} {'R* m':>7} {'ata at R*':>9} | {'need at 600 m':>13}"
    )
    rows = []
    for V in [float(x) for x in args.a1_speeds.split(",")]:
        for n in [float(x) for x in args.a1_gees.split(",")]:
            for a_deg, tag in (
                (ALPHA_CEIL_DEG, "alpha"),
                (ALPHA_CEIL_DEG * math.sin(math.acos(min(1.0, 1.0 / n))), "alpha sin phi"),
            ):
                s = _coturn(V, n, math.radians(a_deg))
                if not s["feasible"]:
                    continue
                # what the same geometry would DEMAND at 600 m
                if 600.0 <= 2.0 * s["r"]:
                    half600 = math.asin(600.0 / (2.0 * s["r"]))
                    need600 = math.degrees(
                        half600 + math.asin(min(1.0, 2.0 * V * math.sin(half600) / 1030.0))
                    )
                    need_s = f"{need600:12.1f}d"
                else:
                    need_s = "   R>diameter"
                print(
                    f"  {V:6.0f} {n:5.1f} {a_deg:8.2f}d {tag:>4} | {s['r']:9.0f} "
                    f"{math.degrees(s['omega']):9.2f} {math.degrees(s['half']):7.2f}d "
                    f"{s['R']:7.0f} {math.degrees(s['ata']):8.2f}d | {need_s}"
                )
                rows.append((V, n, a_deg, s))
    print()
    print("  With the T3's MEASURED break parameters from A2b:")
    best = None
    for (cell, arm), b in sorted(brk.items()):
        if not np.isfinite(b["tas50"]) or not np.isfinite(b["nz50"]):
            continue
        n = max(1.05, b["nz50"])
        s = _coturn(b["tas50"], n, math.radians(ALPHA_CEIL_DEG))
        print(
            f"    {cell + ' / ' + arm:<34} V {b['tas50']:5.1f} n {n:4.2f} -> analytic r "
            f"{s['r']:5.0f} m (measured {b['radius']:5.0f} m), R* {s['R']:5.0f} m, "
            f"ata* {math.degrees(s['ata']):5.2f} deg"
        )
        if best is None:
            best = (b["tas50"], n, s)
    if best is not None:
        V, n, s = best
        print()
        print(
            f"  REQUIRED NOSE OFFSET vs RANGE on the measured co-circle (V {V:.0f} m/s, n {n:.2f} g,"
            f" r {s['r']:.0f} m),"
        )
        print(f"  against the {ALPHA_CEIL_DEG} deg alpha budget. `need` = Delta/2 + lead; the")
        print(
            "  boresight must lie inside the alpha cone about the VELOCITY vector, so need > alpha"
        )
        print("  means NO stick input produces the gun solution at that range.")
        print(f"  {'R m':>6} {'Delta/2':>8} {'lead':>7} {'need':>7} {'margin':>7}  verdict")
        for R in (200, 250, 300, 350, 400, 450, 500, 600, 700, 800):
            if R > 2.0 * s["r"]:
                continue
            half = math.asin(R / (2.0 * s["r"]))
            lead = math.asin(min(1.0, 2.0 * V * math.sin(half) / 1030.0))
            need = math.degrees(half + lead)
            marg = ALPHA_CEIL_DEG - need
            print(
                f"  {R:6d} {math.degrees(half):7.2f}d {math.degrees(lead):6.2f}d {need:6.2f}d "
                f"{marg:+6.2f}d  {'HOLDABLE' if marg >= 0 else 'out of the alpha cone'}"
            )
    print()
    print("  ALTITUDE SENSITIVITY. Turn radius scales with ALTITUDE, not entry speed (recorded):")
    print("  a higher block forces a higher sustained TAS at the same load, so r grows as V^2 and")
    print("  R* grows with it. The rows above bracket the block: the fights here run at ~4570 m")
    print("  (alt_ft 15000). R* is proportional to r, so R*/r is altitude-INVARIANT -- the")
    print("  invariant quantity is the RATIO R*/r, and the range band moves with the block.")
    for V, n, a_deg, s in rows:
        if abs(a_deg - ALPHA_CEIL_DEG) < 1e-9 and abs(n - NZ_SUSTAINED_G) < 1e-9:
            print(
                f"    at n={n} g, alpha={a_deg}deg, V={V:.0f}: R*/r = {s['R'] / s['r']:.3f} "
                f"=> R* = {s['R']:.0f} m at r = {s['r']:.0f} m, and "
                f"{s['R'] / s['r'] * RADIUS_RECORDED_M:.0f} m at the board's r = "
                f"{RADIUS_RECORDED_M:.0f} m"
            )


# ======================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="D3a analytic + empirical dwell budget")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--arms", default="expert,lagtrack_fixed,lead_keep,lead_max")
    p.add_argument("--cells", default="chase-asp0,chase-asp30,chase-asp55,merge-lat1400")
    # R1
    p.add_argument("--rate-ticks", type=int, default=240)
    # R2
    p.add_argument("--phit-ranges", default="200,300,350,400,450,500,600,700,900")
    p.add_argument("--phit-misses", default="0,2,4,6,8,12,16,24")
    p.add_argument("--phit-gees", default="1,5,7")
    p.add_argument("--phit-samples", type=int, default=3000)
    p.add_argument("--phit-alt", type=float, default=5000.0)
    p.add_argument("--v-shooter", type=float, default=230.0)
    p.add_argument("--v-target", type=float, default=200.0)
    p.add_argument("--nb-samples", type=int, default=200000)
    # A2 true p_hit
    p.add_argument("--true-samples", type=int, default=1200)
    p.add_argument("--phit-gate-m", type=float, default=600.0)
    p.add_argument("--phit-max-ticks", type=int, default=250)
    p.add_argument(
        "--rate-used",
        type=float,
        default=100.0,
        help="rd/s used in the E[hits] integral; set from R1's continuous read",
    )
    # A1
    p.add_argument("--a1-speeds", default="160,175,200,230")
    p.add_argument("--a1-gees", default="4,5,5.8,7")
    p.add_argument("--parts", default="r0,r1,r2,r3,a2,a1")
    return p.parse_args(argv)


def main():
    args = parse_args()
    parts = {x.strip() for x in args.parts.split(",")}
    print("D3a DWELL BUDGET | " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    print(
        f"one control step = {args.action_repeat}/{args.rate_hz:.0f} Hz = "
        f"{1000 * args.action_repeat / args.rate_hz:.1f} ms ; 12 steps = 0.5 s ; 36 steps = 1.5 s"
    )
    r2 = None
    brk = {}
    if "r0" in parts:
        sec_r0()
    r1 = sec_r1(args) if "r1" in parts else {}
    if r1:
        cont = r1.get("continuous", {}).get("r_on")
        if cont and np.isfinite(cont):
            args.rate_used = float(cont)
            print(
                f"\n  => E[hits] integrals below use the MEASURED continuous rate "
                f"{args.rate_used:.1f} rd/s"
            )
    if "r2" in parts:
        r2 = sec_r2(args)
    if "r3" in parts and r2 is not None:
        sec_r3(r2, args.rate_used, args)
    if "a2" in parts:
        res = sec_a2(args)
        brk = sec_a2_breakstats(res, args)
        if "a2c" in parts:
            sec_a2c(res)
    if "a1" in parts:
        sec_a1(brk, args)
    print("\ndone.")


if __name__ == "__main__":
    main()
