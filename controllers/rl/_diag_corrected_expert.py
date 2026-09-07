"""CLOSED-LOOP A/B of the gun lay: does a controller that STEERS to the corrected boresight
actually kill, where the naive one only "nearly" does? (read-only forensics; throwaway probe)

`_diag_aim_counterfactual.py` answers the question OFFLINE — it re-scores rounds that were already
fired and says c84's magazine was worth Ekills 5.93 flown vs 24.71 at the corrected lay
([[gun-lay-velocity-inheritance]]). That is a rescoring, not a demonstration: it holds the trajectory
fixed and only moves the nose on paper. This probe closes the loop. It flies the SAME scripted
pursuit law twice against the same T3 with the same seeds, changing exactly one thing:

  naive      steer the lift vector at `geometry.lead_point` (bandit.pos + vel·rng/1030) and fire when
             `rng·sin(naive_lead_error) <= GUN_FIRE_MISS_M`     — today's `PursuitExpert`, byte-for-byte
  corrected  steer at the velocity-inheritance-CORRECTED boresight (`gunlay.intercept`) and fire when
             the predicted miss `lay_miss_m <= GUN_FIRE_MISS_M`  — the curriculum-86 lay
  honest     as `corrected`, but the trigger waits for `lay_miss_m <= HIT_RADIUS_M` (the real 8 m
             sphere) — prices whether the looser 10 m gate is wasting rounds

If `corrected` kills materially more than `naive` from identical starts, the aim point is the wall in
the CONTROL loop and not just in the arithmetic, and the reward fix (R2) is worth training on. If the
two are indistinguishable, the offline counterfactual was measuring something the controller cannot
exploit and R2 should be re-thought before a GPU run.

Nothing here trains or writes: it drives the learner slot with a scripted stick and reads state.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from astrum_rl import _paths

DEG = 180.0 / math.pi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lays", default="naive,corrected,honest", help="comma list of lays to fly")
    p.add_argument("--episodes", type=int, default=8, help="episodes per (lay, aspect)")
    p.add_argument("--aspects", default="0,30,45,55")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-heading", type=float, default=None, help="180 = head-on merge")
    p.add_argument(
        "--opponent-tier",
        type=int,
        default=0,
        help="opponent AI inside the T3 START GEOMETRY (make_tier3_env's own `opponent_tier`, so "
        "`aspect_deg` still applies): 0/3 = full Tier3Opponent, 2 = a plain breaker, 1 = a "
        "straight-and-level dummy. A weaker AI is the only way to get a FIRE-RICH arm: vs the "
        "current aggressive T3 ([[aggressive-t3-break]]) the scripted expert never even reaches "
        "its own 300 m trail, so the trigger never opens and a gun-LAY A/B has nothing to measure.",
    )
    p.add_argument("--me-dalt-ft", type=float, default=0.0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--per-episode", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- the two lays


def _aim_point(op, gl, geom, C, me, bandit, lay: str) -> np.ndarray:
    """The point the lift vector is rolled onto and the nose pulled to.

    Both lays are expressed as a POINT at the same slant range so the identical `lead_pursuit`
    gains apply and only the DIRECTION differs — the whole experiment is that one rotation.

    `gate` deliberately steers NAIVE: it swaps only the trigger, so its trajectory is byte-identical
    to the `naive` arm and the comparison isolates round→hit CONVERSION from steering. That matters
    because `lead_pursuit`'s roll/pull gains are tuned for the naive point, and the corrected
    boresight sits further off the LOS — a proportional pursuit law chasing it pulls wider and bleeds
    closure (measured: minRng 190→301 m vs the full T3). A learned policy derives its own steering
    and has no such tuning to break, so `gate` is the arm that speaks to the REWARD fix (R2's
    `in_solution` half) without the hand-tuned-gain confound.
    """
    if lay in ("naive", "gate"):
        return geom.lead_point(me, bandit)
    sol = gl.intercept(me.pos, me.vel, bandit.pos, bandit.vel)
    if not sol.feasible:  # never observed (needs |v_rel| >= 1030 m/s); fall back to today's point
        return geom.lead_point(me, bandit)
    rng = float(np.linalg.norm(bandit.pos - me.pos))
    return me.pos + sol.boresight * rng


def _fire_gate(rg, C, lay: str) -> bool:
    """The trigger. Same shape in every lay — "is my predicted miss inside a tolerance" — but the
    naive lay measures that miss with `rng·sin(θ)` off the NOSE (~19% high, and around the wrong
    point), whereas the corrected lay uses the exact `1030·sin(lead_error)·tof` at the target."""
    if lay == "naive":
        return rg.rng * math.sin(rg.naive_lead_error) <= C.GUN_FIRE_MISS_M
    tol = C.GUN_FIRE_MISS_M if lay == "corrected" else C.HIT_RADIUS_M
    return rg.lay_feasible and rg.lay_miss_m <= tol


def _act(op, gl, geom, C, me, bandit, lay: str) -> np.ndarray:
    """`PursuitExpert.act` with a swappable aim point — raw env action [ail, elev, rud, thr, trig].

    Copied rather than imported so `opponent.py` stays pinned to the naive lay (the scripted T3 is
    the benchmark and must not move); the four control lines are identical to `lead_pursuit`.
    """
    rg = geom.relative(me, bandit)
    closure = op._clamp(C.K_CHASE_RANGE * (rg.rng - C.R_CHASE_TRAIL), 0.0, C.V_CHASE_MARGIN)
    tgt_spd = bandit.airspeed + closure
    g = op.F16Gains()

    off, bank_err, _ = geom.body_bearing(me, _aim_point(op, gl, geom, C, me, bandit, lay))
    roll_gate = min(off / C.PURSUE_ROLL_DEADBAND, 1.0)
    ail = op._clamp(C.K_PURSUE_ROLL * bank_err * roll_gate - g.k_roll_rate * me.p, -1.0, 1.0)
    aligned = max(math.cos(bank_err), 0.0)
    elev = op._corner_limited_pull(op._clamp(-C.K_PURSUE_PULL * off * aligned, -1.0, 0.0), me)
    thr = op._clamp(g.throttle_trim + g.k_spd_to_thr * (tgt_spd - me.airspeed), 0.0, 1.0)

    hot = me.ammo > 0 and me.alive and bandit.alive and rg.in_range and _fire_gate(rg, C, lay)
    return np.asarray([ail, elev, 0.0, thr, 1.0 if hot else 0.0], np.float32), rg


# --------------------------------------------------------------------------- rollout


def _roll(env, base, op, gl, geom, C, inv, low, high, seed: int, lay: str) -> dict:
    obs, _info = env.reset(seed=seed)
    event, done = None, False
    hp0 = float(base._states[base.opponent_id].health)
    ammo0 = int(base._states[base.learner_id].ammo)
    pulls, in_sol_steps, miss_sum, miss_n = 0, 0, 0.0, 0
    best_q, min_rng = 0.0, float("inf")
    while not done:
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        raw, rg = _act(op, gl, geom, C, me, opp, lay)
        pulls += int(raw[4] > 0.5)
        min_rng = min(min_rng, rg.rng)
        best_q = max(best_q, rg.gun_quality)
        if rg.in_range and rg.lay_feasible:
            in_sol_steps += int(rg.lay_miss_m <= C.HIT_RADIUS_M)
            if rg.rng <= C.R_EFF:
                miss_sum += rg.lay_miss_m
                miss_n += 1
        a = inv(raw, low, high)
        _obs, _r, terminated, truncated, info = env.step(a)
        if info.get("event"):
            event = info["event"]
        done = terminated or truncated
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    return {
        "event": event,
        "kill": bool((not opp.alive) or event == "kill"),
        "die": bool((not me.alive) or event in ("killed", "death", "trade", "hard_deck")),
        "dealt": hp0 - float(opp.health),
        "taken": 100.0 - float(me.health),
        "spent": ammo0 - int(me.ammo),
        "pulls": pulls,
        "insol": in_sol_steps,
        "miss": (miss_sum / miss_n) if miss_n else float("nan"),
        "best_q": best_q,
        "min_rng": min_rng,
    }


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight import geometry as geom
    from astrum_dogfight import gunlay as gl
    from astrum_dogfight import opponent as op

    from astrum_rl.envs import invert_tdmpc_action, make_tdmpc_env

    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    lays = [x.strip() for x in args.lays.split(",") if x.strip()]

    print(
        f"CORRECTED-LAY CLOSED LOOP | scripted pursuit vs T{args.opponent_tier or 3} | "
        f"sep {args.separation:.0f}m opp_hdg {args.opp_heading} dalt +{args.me_dalt_ft:.0f}ft "
        f"jitter {args.jitter} | {args.episodes} eps/aspect × {len(aspects)} aspects, "
        f"{args.max_time_s:.0f}s, repeat {args.action_repeat}"
    )
    print(
        f"  gates: naive rng·sin(naive_lead)<={C.GUN_FIRE_MISS_M:.0f}m | corrected "
        f"lay_miss<={C.GUN_FIRE_MISS_M:.0f}m | honest lay_miss<={C.HIT_RADIUS_M:.0f}m "
        f"(sphere {C.HIT_RADIUS_M:.0f}m, 50 hits/kill)"
    )

    table: dict[tuple[str, float], list[dict]] = {}
    for lay in lays:
        for asp in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=asp,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            if args.opp_heading is not None:
                kw["opp_heading_deg"] = args.opp_heading
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            # Always the TIER-3 factory (it owns `posture`/`aspect_deg`); `opponent_tier` swaps only
            # the AI. Routing to `make_tdmpc_env(2, ...)` instead would land in `make_tier2_env`,
            # which reuses `tier1_scenario` and rejects `posture`.
            if args.opponent_tier:
                kw["opponent_tier"] = args.opponent_tier
            env = make_tdmpc_env(3, **kw)
            base = env.unwrapped
            low = np.asarray(base.action_space.low, np.float32)
            high = np.asarray(base.action_space.high, np.float32)
            rows = [
                _roll(env, base, op, gl, geom, C, invert_tdmpc_action, low, high,
                      args.seed + 1000 * int(asp) + ep, lay)
                for ep in range(args.episodes)
            ]
            env.close()
            table[(lay, asp)] = rows

    def agg(rows, k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return float(np.mean(v)) if v else float("nan")

    print("\n=== PER ASPECT — same seeds, same law, only the lay differs ===")
    print(f"{'lay':>10} {'asp':>4} {'kill':>5} {'die':>4} {'dealt':>6} {'taken':>6} {'rds':>5} "
          f"{'pulls':>6} {'inSol':>6} {'miss_m':>7} {'bestQ':>5} {'minR':>5} {'events'}")
    for lay in lays:
        for asp in aspects:
            rows = table[(lay, asp)]
            ev = {}
            for r in rows:
                ev[str(r["event"])] = ev.get(str(r["event"]), 0) + 1
            evs = " ".join(f"{k}×{v}" for k, v in sorted(ev.items(), key=lambda kv: -kv[1]))
            print(f"{lay:>10} {asp:>4.0f} {sum(r['kill'] for r in rows):>3}/{len(rows):<2} "
                  f"{sum(r['die'] for r in rows):>4} {agg(rows, 'dealt'):>6.1f} "
                  f"{agg(rows, 'taken'):>6.1f} {agg(rows, 'spent'):>5.0f} {agg(rows, 'pulls'):>6.0f} "
                  f"{agg(rows, 'insol'):>6.0f} {agg(rows, 'miss'):>7.1f} {agg(rows, 'best_q'):>5.2f} "
                  f"{agg(rows, 'min_rng'):>5.0f} {evs}")
            if args.per_episode:
                for i, r in enumerate(rows):
                    print(f"           ep{i:<2} {str(r['event']):>11} dealt {r['dealt']:>5.1f} "
                          f"taken {r['taken']:>5.1f} rds {r['spent']:>4} miss {r['miss']:>6.1f}")

    print("\n=== TOTALS — the headline A/B ===")
    print(f"{'lay':>10} {'kills':>6} {'deaths':>7} {'dealt/ep':>9} {'taken/ep':>9} {'rds/ep':>7} "
          f"{'HP/100rd':>9} {'miss_m':>7}")
    base_kills = None
    for lay in lays:
        rows = [r for asp in aspects for r in table[(lay, asp)]]
        rds = agg(rows, "spent")
        dealt = agg(rows, "dealt")
        k = sum(r["kill"] for r in rows)
        if base_kills is None:
            base_kills = k
        print(f"{lay:>10} {k:>3}/{len(rows):<2} {sum(r['die'] for r in rows):>7} {dealt:>9.1f} "
              f"{agg(rows, 'taken'):>9.1f} {rds:>7.0f} "
              f"{(100.0 * dealt / rds if rds else float('nan')):>9.1f} {agg(rows, 'miss'):>7.1f}")
    print("\n  Read: if `corrected` beats `naive` on kills AND on HP per 100 rounds, the corrected lay "
          "is exploitable by a CONTROLLER, not just better on paper ⇒ R2 (the reward fix) is worth a "
          "GPU run. `honest` isolates whether the 10 m fire gate wastes rounds vs the real 8 m sphere.")


if __name__ == "__main__":
    main()
