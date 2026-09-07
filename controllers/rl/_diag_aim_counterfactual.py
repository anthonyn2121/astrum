"""Aim counterfactual: of the rounds a checkpoint ALREADY fires, how many SHOULD have hit?

Every run since curriculum-76 ends the same way — the policy presses, tracks, empties the
magazine, and deals ~0 damage. `_diag_chase_bfm.py` established WHERE it shoots from
(250-600 m, 79-86% "hittable" on the reward's own tolerance) but not whether those rounds
were physically capable of hitting. That is the question this probe answers, and it decides
whether the remaining problem is the AIM POINT or the FLYING.

Method: roll the policy ONE SIM TICK AT A TIME (the agent still acts every `--action-repeat`
ticks, exactly as `ActionRepeat` makes it), recording both aircraft every tick. Every trigger
pull is then reconstructed with zero interpolation — the M61's 100 Hz accumulator against a
120 Hz tick lets out at most one round per tick, so an ammo decrement at tick `j` IS one round
fired from tick `j`'s pose. Each round is re-scored under five lays with the sim's own
ballistics (drag + gravity + Coriolis + the 8 mrad cone + the 8 m sphere), sharing dispersion
draws so the comparison is paired:

  flown    the boresight it really used (its nose) — reproduces the sim's own result
  pred     the honest fire-control solution: a full-ballistics intercept computed from the
           bandit's INSTANTANEOUS state (all any real FCS can know), refined against that
           same constant-velocity belief
  clair    the clairvoyant solution: the intercept of the path the bandit really flew
  clamp    `clair` clamped into an α cone about the velocity vector — is the required lay
           even reachable by an airframe that cannot point its nose off its flight path?
  predclm  `pred` clamped the same way — the best HONESTLY ACHIEVABLE lay, and therefore
           the ceiling on what fixing the reward's aim point can possibly buy

Read the output as a chain of differences, each isolating one wall:

  flown → pred      what fixing the AIM MATH buys (the `gun-lay-velocity-inheritance` bug:
                    the sim fires along `ac_vel + 1030·boresight`, so the nose is not the
                    lay, yet `geometry.lead_error` measures from the nose)
  pred  → clair     target-manoeuvre UNPREDICTABILITY — irreducible, ~½·a·tof², which is
                    0.98 m at 150 m but 18 m at 600 m against an 8 g break
  clair → clamp     REACHABILITY: whether the shot is a gun-tracking problem at all
  hits  → Ehits     validation. `Ehits(flown)` must reproduce the damage actually dealt;
                    if it does not, the frame reconstruction is wrong and nothing else in
                    the table means anything. `--audit-hits` gives the harder version of
                    that test: a round whose UNDISPERSED miss exceeds `8 m + 3σ·rng` cannot
                    hit under any dispersion draw, so if the sim landed more hits than there
                    were such rounds, the reconstruction is provably wrong.

`Ekills = Ehits / HITS_TO_KILL` is the honest headline: how many kills' worth of hits the
magazine bought. A run that fires 500 rounds for Ekills(flown)=0.1 but Ekills(pred)=3 is
losing to arithmetic; one where Ekills(pred) is also ~0 is losing to its flying, and no
reward re-shaping will save it.

Frames — this is where an earlier version of this probe was wrong, so it is spelled out.
`resolve_hits` compares Bevy `Transform.translation`, and `observe()` publishes that verbatim
as `pos_e/pos_n/pos_u`; the scenario generators set `origin_alt_m = 0.0`, so the density
lookup `base_alt + pos.z` is just ENU up. Every position here is therefore `AState.pos`
UNCHANGED (`alt_m` is MSL and must not be substituted for `pos_u`). The muzzle is
`pos + 4·nose` (`Gun::m61a1`'s body-X offset) and the round inherits the full aircraft
velocity. Timing: `sync_geo_transform → fire_guns → step_bullets → resolve_hits` are all
`.chain()`ed inside ONE tick, so the sim's first hit test pits the segment
`muzzle → muzzle + v·dt` against the target at the SHOT tick — row 0 of every target table
below is the shot tick, not one tick later.

Run:  cd controllers/rl && uv run python _diag_aim_counterfactual.py \
        --checkpoint runs/curriculum-84/c84_broaden10kill_5000k.pt \
        --aspects 0,30,45,55 --episodes 6 --jitter 0.5 --device cuda
      # head-on merge (c85's geometry):
      ... --aspects 0 --opp-heading 180 --separation 3000 --max-time-s 180
"""

from __future__ import annotations

import argparse
import statistics
from collections import deque
from math import degrees, radians

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

LAYS = ("flown", "pred", "clair", "clamp", "predclm")

# `origin_alt_m = 0.0` in both scenario generators, so the sim's density altitude for a
# round at ENU height z is just z. Recovered per-episode at reset and warned about if the
# scenario ever changes.
ALT_OFFSET = 0.0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-score a checkpoint's fired rounds under five lays.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--policy",
        default="ckpt",
        choices=["ckpt", "expert"],
        help="'expert' flies the scripted PursuitExpert instead — the VALIDATION mode: against "
        "a --opponent-tier 1 dummy its attitudes change slowly, so Ehits(flown) must reproduce "
        "the sim's hits almost exactly, isolating any frame error from policy noise",
    )
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--aspects", default="0,30,45,55")
    p.add_argument("--mode", default="greedy", choices=["greedy", "mpc"])
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--opp-heading", type=float, default=None, help="opponent heading (deg); 180 = head-on merge")
    p.add_argument("--me-dalt-ft", type=float, default=0.0)
    p.add_argument("--opponent-tier", type=int, default=0, help="0/3 = full T3")
    p.add_argument(
        "--action-repeat",
        type=int,
        default=5,
        help="the POLICY's control cadence in sim ticks. The env itself is always built at "
        "action_repeat=1 and the action is held manually, so every tick can be recorded; the "
        "resulting trajectory is identical to the ActionRepeat wrapper's.",
    )
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument(
        "--alpha-max",
        type=float,
        default=15.0,
        help="usable nose-off-velocity cone (deg); the F-16 model's measured usable α is 10.7-15.0°",
    )
    p.add_argument("--samples", type=int, default=300, help="dispersion draws per round scored")
    p.add_argument("--horizon-pad", type=float, default=0.1, help="s added to 2·tof for the scoring window")
    p.add_argument(
        "--hit-radius",
        type=float,
        default=0.0,
        help="override the 8 m hit sphere (0 = the sim's own). Sweeping it answers 'what "
        "effective radius would make Ehits(flown) match the sim?' — a residual-error probe",
    )
    p.add_argument("--per-episode", action="store_true")
    p.add_argument(
        "--audit-hits",
        action="store_true",
        help="the hard validation: undispersed-miss distribution vs the hits the sim landed",
    )
    p.add_argument(
        "--trace-fire",
        action="store_true",
        help="per-control-step detail with the real hits lag-aligned by time of flight",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.policy == "ckpt" and not args.checkpoint:
        p.error("--checkpoint is required unless --policy expert")
    return args


def _ang(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions, degrees."""
    return degrees(float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


# --------------------------------------------------------------------------- rollout


def _record(state, geom) -> dict:
    """One aircraft snapshot in the sim's OWN hit-test frame (ENU, `Transform.translation`)."""
    nose, _, _ = geom.body_axes_enu(state.phi, state.theta, state.psi)
    return {
        "p": np.asarray(state.pos, float).copy(),
        "v": np.asarray(state.vel, float).copy(),
        "nose": nose,
        "ammo": int(state.ammo),
        "hp": float(state.health),
        # `alt_m − pos_u` recovers the ENU-origin altitude the sim uses for density (plus a
        # ~cm curvature term). Only checked, never used: ALT_OFFSET is the authority.
        "base_alt": float(state.alt) - float(state.pos[2]),
    }


def _roll(agent, env, base, geom, repeat: int, k_stack: int, seed: int, policy: str) -> dict:
    """Roll one episode at TICK resolution while acting every `repeat` ticks.

    This reimplements `ActionRepeat` (and `FrameStack`) in the loop body rather than using the
    wrappers: same action held for the same ticks, same early break on done, same
    end-of-step observation fed to the agent — but with `base._states` read after EVERY tick.
    That is what makes the reconstruction exact: the trigger is a per-tick event, so scoring a
    burst from interpolated mid-step attitudes was itself a larger error than the aim bias
    being measured.
    """
    obs, info = env.reset(seed=seed)
    if agent is not None:
        agent._prev_mean.zero_()
    frames: deque = deque([np.asarray(obs, np.float32)] * k_stack, maxlen=k_stack)
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    mine = [_record(me, geom)]
    theirs = [_record(opp, geom)]
    cstep = [0]  # which control step each tick belongs to (for the trace)
    fires: list[tuple[int, int, object]] = []  # (tick, rounds, RelGeom at that tick)
    event = None
    t0 = True
    done = False
    step_i = 0
    while not done:
        stacked = np.concatenate(list(frames)) if k_stack > 1 else frames[-1]
        if policy == "expert":
            # `_LinearActionWrapper` already inverts info's raw `expert_action` into agent
            # space, so it re-maps to the identical env action when flown.
            a = np.asarray(info["expert_action"], np.float32)
        else:
            a = agent.act(stacked, t0=t0, eval_mode=True).numpy()
        t0 = False
        step_i += 1
        for _ in range(repeat):
            obs, _r, terminated, truncated, info = env.step(a)
            if info.get("event"):
                event = info["event"]
            me = base._states[base.learner_id]
            opp = base._states[base.opponent_id]
            mine.append(_record(me, geom))
            theirs.append(_record(opp, geom))
            cstep.append(step_i)
            rounds = mine[-2]["ammo"] - mine[-1]["ammo"]
            if rounds > 0:
                fires.append((len(mine) - 1, rounds, geom.relative(me, opp)))
            if terminated or truncated:
                done = True
                break
        frames.append(np.asarray(obs, np.float32))
    return {
        "mine": mine,
        "theirs": theirs,
        "theirs_p": np.array([r["p"] for r in theirs]),  # (M,3) tick-exact track, sliced per shot
        "hp": np.array([r["hp"] for r in theirs]),  # bandit HP per TICK → when the hits landed
        "cstep": np.array(cstep),
        "fires": fires,
        "event": event,
        "killed": (not opp.alive) or event == "kill",
        "died": (not me.alive) or event in ("killed", "death", "trade", "hard_deck"),
        "dealt": 100.0 - float(opp.health),
        "taken": 100.0 - float(me.health),
        "spent": mine[0]["ammo"] - mine[-1]["ammo"],
    }


# --------------------------------------------------------------------------- scoring


def _score_event(gl, ep, j: int, rounds: int, args, omega, seed: int) -> dict:
    """Score the `rounds` (normally 1) fired from tick `j`'s exact pose."""
    me, them = ep["mine"][j], ep["theirs"][j]
    pos, vel, nose = me["p"], me["v"], me["nose"]
    muzzle = pos + gl.MUZZLE_OFFSET_BODY[0] * nose
    tp, tv = them["p"], them["v"]

    sol = gl.intercept(muzzle, vel, tp, tv)
    tof = sol.tof_s if sol.feasible else float(np.linalg.norm(tp - muzzle)) / gl.MUZZLE_SPEED_MPS
    horizon = min(gl.BULLET_TTL_S, 2.0 * tof + args.horizon_pad)
    n_row = max(2, int(horizon / gl.SIM_DT))

    # Row k is the target position the sim's k-th hit test uses: tick j+k, held at the last
    # recorded tick if the episode ended first (which is what the sim does too — a dead or
    # despawned target stops moving).
    idx = np.clip(j + np.arange(n_row), 0, len(ep["theirs_p"]) - 1)
    tbl_real = ep["theirs_p"][idx]
    ticks = np.arange(n_row) * gl.SIM_DT
    tbl_cv = tp[None, :] + tv[None, :] * ticks[:, None]
    path = np.column_stack([ticks, tbl_real])

    # The honest FCS lay is refined against its OWN constant-velocity belief — refining it
    # against the real path would silently turn a prediction error into zero.
    pred = gl.refine_lay(sol.boresight, muzzle, vel, tbl_cv, omega, alt_offset=ALT_OFFSET)
    clair_0 = gl.intercept_path(muzzle, vel, path).boresight
    clair = gl.refine_lay(clair_0, muzzle, vel, tbl_real, omega, alt_offset=ALT_OFFSET)
    alpha = radians(args.alpha_max)
    d = {
        "flown": nose,
        "pred": pred,
        "clair": clair,
        "clamp": gl.clamp_to_alpha_cone(clair, vel, alpha),
        "predclm": gl.clamp_to_alpha_cone(pred, vel, alpha),
    }
    # The two candidate REWARD targets, scored alongside the wall chain (section 4):
    #   naive  what `geometry.lead_error` measures against today — the nose-to-aim-point
    #          direction for `bandit.pos + bandit.vel·(rng/1030)`, velocity inheritance
    #          ignored. Scoring it answers "if the policy tracked the reward's own aim point
    #          PERFECTLY, would it hit?"
    #   icept  the closed-form velocity-inheritance lay, UNREFINED — the cheapest correct
    #          target a per-tick Python reward could use. `pred` is this plus four drag/
    #          gravity/Coriolis passes, so icept→pred prices the refinement.
    naive_dir = gl._unit(gl.naive_lead_point_from_parts(pos, tp, tv) - pos)
    icept = sol.boresight if sol.feasible else naive_dir
    d["naive"] = naive_dir
    d["icept"] = icept

    rng = float(np.linalg.norm(tp - pos))
    # The undispersed round's own closest approach: the hard bound behind --audit-hits.
    _, d0_miss, _ = gl._closest_approach_table(
        muzzle, vel + nose * gl.MUZZLE_SPEED_MPS, tbl_real, omega, gl.SIM_DT, ALT_OFFSET
    )
    out = {
        "w": float(rounds),
        "rng": rng,
        "tof": tof,
        "d0_miss": d0_miss,
        "hit_env": gl.HIT_RADIUS_M + 3.0 * gl.DISPERSION_RAD * rng,
        # α the honest / clairvoyant solutions demand, vs what the airframe allows.
        "a_pred": _ang(pred, gl._unit(vel)),
        "a_clair": _ang(clair, gl._unit(vel)),
        "a_flown": _ang(nose, gl._unit(vel)),
        "e_flown_pred": _ang(nose, pred),
        "e_pred_clair": _ang(pred, clair),
        # Today's reward signal (should match `RelGeom.lead_error` at this tick) and how far
        # its aim point sits from the correct one — the size of the lie the reward tells.
        "e_flown_naive": _ang(nose, naive_dir),
        "e_naive_pred": _ang(naive_dir, pred),
        "e_icept_pred": _ang(icept, pred),
    }
    for name, lay in d.items():
        p, miss = gl.p_hit_vec(
            lay, muzzle, vel, tp, tv,
            samples=args.samples, lat_rad=ep["lat"], seed=seed,
            tgt_table=tbl_real, alt_offset=ALT_OFFSET,
        )
        out[f"p_{name}"] = p
        out[f"m_{name}"] = miss
    # The flown lay scored against the CONSTANT-VELOCITY belief instead of the real track:
    # the size of the error that scoring (not aiming) with a CV target would introduce.
    out["p_flowncv"], out["m_flowncv"] = gl.p_hit_vec(
        nose, muzzle, vel, tp, tv,
        samples=args.samples, lat_rad=ep["lat"], seed=seed,
        tgt_table=tbl_cv, alt_offset=ALT_OFFSET,
    )
    return out


def _score_episode(gl, ep, args, omega) -> list[dict]:
    events = []
    for j, rounds, rel in ep["fires"]:
        e = _score_event(gl, ep, j, rounds, args, omega, seed=1_000 + j)
        e["tick"] = j
        e["step"] = int(ep["cstep"][j])
        e["lead_deg"] = degrees(rel.lead_error)
        e["aspect_deg"] = degrees(rel.aspect)
        e["closure"] = rel.closure
        events.append(e)
    return events


def _trace_fire(gl, ep, dt_tick: float) -> None:
    """Per-control-step detail, with the real hits LAG-ALIGNED by time of flight.

    The validation that matters: a round fired at tick `j` lands ~tof/dt ticks later (0.36 s
    = 43 ticks at 400 m), so a naive same-step comparison of expected vs actual hits is off by
    most of a second. Depositing each round's expected hits at `j + round(tof/dt)` puts the two
    cumulative curves on the same clock; if they then track, the reconstruction is sound, and
    if they diverge over one burst, that burst's geometry is what to go look at."""
    n = len(ep["hp"])
    actual = np.zeros(n)
    actual[1:] = -np.diff(ep["hp"]) / gl.DAMAGE_PER_HIT
    dep_fl, dep_pr = np.zeros(n), np.zeros(n)
    by_step: dict[int, list[dict]] = {}
    for e in ep["ev"]:
        by_step.setdefault(e["step"], []).append(e)
        k = min(n - 1, e["tick"] + int(round(e["tof"] / dt_tick)))
        dep_fl[k] += e["w"] * e["p_flown"]
        dep_pr[k] += e["w"] * e["p_pred"]
    print(
        f"    {'t':>5} {'rng':>5} {'asp':>4} {'clo':>4} {'rds':>4} {'α°':>5} {'lead°':>6} "
        f"{'n→pr°':>6} {'p_fl':>5} {'p_pr':>5} {'miss':>5} {'d0':>5} | {'Σrds':>5} {'Σact':>5} "
        f"{'Σexp_fl':>7} {'Σexp_pr':>7}"
    )
    cum_r = cum_a = cum_f = cum_p = 0.0
    last_step = -1
    for j in range(1, n):
        cum_a += actual[j]
        cum_f += dep_fl[j]
        cum_p += dep_pr[j]
        step = int(ep["cstep"][j])
        if step == last_step:
            continue
        evs = by_step.get(step)
        if not evs:
            continue
        last_step = step
        mid = evs[len(evs) // 2]  # the mid-step attitude, representative of the burst
        rounds = sum(e["w"] for e in evs)
        cum_r += rounds
        print(
            f"    {j * dt_tick:>5.1f} {mid['rng']:>5.0f} {mid['aspect_deg']:>4.0f} "
            f"{mid['closure']:>4.0f} {rounds:>4.0f} {mid['a_flown']:>5.1f} {mid['lead_deg']:>6.2f} "
            f"{mid['e_flown_pred']:>6.2f} {mid['p_flown']:>5.2f} {mid['p_pred']:>5.2f} "
            f"{mid['m_flown']:>5.1f} {mid['d0_miss']:>5.1f} | {cum_r:>5.0f} {cum_a:>5.1f} "
            f"{cum_f:>7.1f} {cum_p:>7.1f}"
        )


def _totals(events: list[dict]) -> dict:
    w = sum(e["w"] for e in events)
    out = {"rounds": w}
    for lay in LAYS:
        out[f"E_{lay}"] = sum(e["w"] * e[f"p_{lay}"] for e in events)
    return out


def _wmean(events: list[dict], key: str) -> float:
    w = sum(e["w"] for e in events)
    return sum(e["w"] * e[key] for e in events) / w if w > 0 else float("nan")


# --------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import geometry as geom
    from astrum_dogfight import gunlay as gl
    from astrum_dogfight.scenario import _ORIGIN_LAT
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    if args.hit_radius > 0:
        gl.HIT_RADIUS_M = args.hit_radius  # read at call time by p_hit_vec

    agent = None
    if args.policy == "ckpt":
        sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = args.mode == "mpc"
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()

    lat = radians(_ORIGIN_LAT)
    omega = gl.earth_rate_enu(lat)
    dt_tick = 1.0 / args.rate_hz
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]

    who = args.checkpoint if args.policy == "ckpt" else "PursuitExpert (scripted)"
    print(
        f"AIM COUNTERFACTUAL {who} | mode={args.mode} | opp T{args.opponent_tier or 3} | "
        f"sep {args.separation:.0f}m opp_hdg {args.opp_heading if args.opp_heading is not None else 'co'} "
        f"dalt {args.me_dalt_ft:+.0f}ft jitter{args.jitter:g} | {args.episodes} eps/aspect"
    )
    print(
        f"ballistics: muzzle {gl.MUZZLE_SPEED_MPS:.0f}m/s  disp {gl.DISPERSION_RAD * 1e3:.0f}mrad  "
        f"sphere {gl.HIT_RADIUS_M:.0f}m  {gl.HITS_TO_KILL} hits/kill  {gl.GUN_RATE_HZ:.0f}rd/s  "
        f"α cone {args.alpha_max:.0f}°  {args.samples} draws/round  "
        f"TICK-EXACT ({args.rate_hz:.0f}Hz, act every {args.action_repeat})"
    )
    print(
        "Ehits = Σ rounds·P(hit) vs the bandit's REAL per-tick track. Ekills = Ehits/"
        f"{gl.HITS_TO_KILL}. Ehits(flown)≈hits validates the frame."
    )

    all_events: list[dict] = []
    hdr = (
        f"{'aspect':>6} {'kill':>4} {'die':>3} {'dealt':>5} {'hits':>5} {'rds':>5} | "
        f"{'E_flown':>7} {'E_pred':>7} {'E_clair':>7} {'E_clamp':>7} {'E_pclm':>7} | "
        f"{'Ek_fl':>5} {'Ek_pr':>5} {'Ek_cl':>5} | {'val':>5}"
    )
    for ai, aspect in enumerate(aspects):
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=aspect,
            separation_m=args.separation,
            opponent_tier=args.opponent_tier,
            me_dalt_ft=args.me_dalt_ft,
            action_repeat=1,  # the repeat is done in `_roll` so every tick is recorded
            frame_stack=1,  # likewise the stack, at control-step boundaries
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            **({"opp_heading_deg": args.opp_heading} if args.opp_heading is not None else {}),
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
        )
        base = env.unwrapped
        rows, ev_aspect = [], []
        for epi in range(args.episodes):
            ep = _roll(
                agent, env, base, geom, args.action_repeat, max(1, args.frame_stack),
                seed=args.seed + int(aspect) * 1000 + epi, policy=args.policy,
            )
            ep["lat"] = lat
            if ai == 0 and epi == 0 and abs(ep["mine"][0]["base_alt"] - ALT_OFFSET) > 5.0:
                print(
                    f"  WARNING: recovered ENU-origin altitude {ep['mine'][0]['base_alt']:.1f} m "
                    f"≠ ALT_OFFSET {ALT_OFFSET:.1f} — the density lookup is off; fix ALT_OFFSET."
                )
            ev = _score_episode(gl, ep, args, omega)
            ep["ev"] = ev
            rows.append(ep)
            ev_aspect.extend(ev)
            if args.per_episode or args.trace_fire:
                t = _totals(ev)
                print(
                    f"  ep{epi:<2} asp{aspect:<4.0f} {ep['event'] or '-':<12} dealt {ep['dealt']:>5.0f} "
                    f"hits {ep['dealt'] / gl.DAMAGE_PER_HIT:>5.1f} rds {t['rounds']:>5.0f}"
                    f"/spent {ep['spent']:>4} | "
                    + " ".join(f"{lay}{t[f'E_{lay}']:>7.1f}" for lay in LAYS)
                )
            if args.trace_fire and ev:
                _trace_fire(gl, ep, dt_tick)
        env.close()
        all_events.extend(ev_aspect)
        if ai == 0:
            print("\n=== (1) PER-ASPECT: what the magazine bought vs what it could have ===")
            print(hdr)

        t = _totals(ev_aspect)
        hits = sum(r["dealt"] for r in rows) / gl.DAMAGE_PER_HIT
        val = t["E_flown"] / hits if hits > 0 else float("nan")
        print(
            f"{aspect:>6.0f} {sum(r['killed'] for r in rows):>4} {sum(r['died'] for r in rows):>3} "
            f"{statistics.fmean(r['dealt'] for r in rows):>5.0f} {hits:>5.1f} {t['rounds']:>5.0f} | "
            + " ".join(f"{t[f'E_{lay}']:>7.1f}" for lay in LAYS)
            + f" | {t['E_flown'] / gl.HITS_TO_KILL:>5.2f} {t['E_pred'] / gl.HITS_TO_KILL:>5.2f} "
            f"{t['E_clair'] / gl.HITS_TO_KILL:>5.2f} | {val:>5.2f}"
        )

    if not all_events:
        print("\nNO ROUNDS FIRED — nothing to counterfactual. (A hold-fire policy is its own finding.)")
        return

    t = _totals(all_events)
    print(
        f"{'ALL':>6} {'':>4} {'':>3} {'':>5} {'':>5} {t['rounds']:>5.0f} | "
        + " ".join(f"{t[f'E_{lay}']:>7.1f}" for lay in LAYS)
        + f" | {t['E_flown'] / gl.HITS_TO_KILL:>5.2f} {t['E_pred'] / gl.HITS_TO_KILL:>5.2f} "
        f"{t['E_clair'] / gl.HITS_TO_KILL:>5.2f} |"
    )

    print("\n=== (2) PER-ROUND P(hit) BY RANGE — where the rounds go and what they were worth ===")
    print(
        f"{'range band':>12} {'rds':>6} {'%rds':>5} | {'flown':>6} {'pred':>6} {'clair':>6} "
        f"{'clamp':>6} {'pclm':>6} | {'miss_fl':>7} {'miss_pr':>7} | {'σ_trk°':>6} {'lead°':>6} {'clo':>5}"
    )
    for lo, hi in ((0, 150), (150, 250), (250, 400), (400, 600), (600, 900), (900, 1200), (1200, 99999)):
        rec = [e for e in all_events if lo <= e["rng"] < hi]
        w = sum(e["w"] for e in rec)
        if w <= 0:
            continue
        band = f"{lo}-{hi if hi < 99999 else '∞'}m"
        print(
            f"{band:>12} {w:>6.0f} {100.0 * w / t['rounds']:>4.0f}% | "
            + " ".join(f"{_wmean(rec, f'p_{lay}'):>6.3f}" for lay in LAYS)
            + f" | {_wmean(rec, 'm_flown'):>7.1f} {_wmean(rec, 'm_pred'):>7.1f} | "
            f"{degrees(geom.sigma_track((lo + hi) / 2 if hi < 99999 else 1200)):>6.2f} "
            f"{_wmean(rec, 'lead_deg'):>6.2f} {_wmean(rec, 'closure'):>5.0f}"
        )

    print("\n=== (3) THE THREE WALLS — mean angular error, round-weighted ===")
    out_of_cone = sum(e["w"] for e in all_events if e["a_clair"] > args.alpha_max + 1e-9)
    out_of_cone_pred = sum(e["w"] for e in all_events if e["a_pred"] > args.alpha_max + 1e-9)
    print(
        f"  aim math      nose→pred   {_wmean(all_events, 'e_flown_pred'):>6.2f}°   "
        f"(what the reward's lead_error cannot see: it measures from the NOSE)"
    )
    print(
        f"  prediction    pred→clair  {_wmean(all_events, 'e_pred_clair'):>6.2f}°   "
        f"(irreducible ≈½·a·tof²; mean tof {_wmean(all_events, 'tof'):>4.2f}s)"
    )
    print(
        f"  reachability  α required  {_wmean(all_events, 'a_clair'):>6.2f}° clair / "
        f"{_wmean(all_events, 'a_pred'):>6.2f}° pred vs {args.alpha_max:.0f}° cone; "
        f"flown α {_wmean(all_events, 'a_flown'):>5.2f}°"
    )
    print(
        f"                outside the cone: clair {100.0 * out_of_cone / t['rounds']:>4.0f}% of rounds, "
        f"pred {100.0 * out_of_cone_pred / t['rounds']:>4.0f}%"
    )
    gain_aim = t["E_pred"] - t["E_flown"]
    gain_pred = t["E_clair"] - t["E_pred"]
    loss_cone = t["E_clair"] - t["E_clamp"]
    print(
        f"\n  BUDGET (hits): aim math {gain_aim:>+7.1f}   prediction {gain_pred:>+7.1f}   "
        f"α cone {-loss_cone:>+7.1f}   → honest achievable {t['E_predclm']:>6.1f} "
        f"= {t['E_predclm'] / gl.HITS_TO_KILL:.2f} kills"
    )

    print("\n=== (4) WHAT THE REWARD SHOULD AIM AT — each candidate target, tracked perfectly ===")
    print(f"{'target':>8} {'P(hit)':>7} {'Ehits':>7} {'Ekills':>6} {'miss_m':>7}   what it is")
    for name, what in (
        ("flown", "the nose it actually used (the baseline)"),
        ("naive", "TODAY's reward aim point: bandit.pos + vel·rng/1030, from the NOSE"),
        ("icept", "closed-form intercept WITH velocity inheritance, unrefined (cheap)"),
        ("pred", "icept + 4 drag/gravity/Coriolis passes (the honest FCS ceiling)"),
        ("clair", "clairvoyant: the intercept of the path actually flown"),
    ):
        e = sum(x["w"] * x[f"p_{name}"] for x in all_events)
        print(
            f"{name:>8} {_wmean(all_events, f'p_{name}'):>7.3f} {e:>7.1f} "
            f"{e / gl.HITS_TO_KILL:>6.2f} {_wmean(all_events, f'm_{name}'):>7.1f}   {what}"
        )
    print(
        f"  nose→naive {_wmean(all_events, 'e_flown_naive'):>5.2f}° is the error the reward SEES; "
        f"naive→pred {_wmean(all_events, 'e_naive_pred'):>5.2f}° is the error it CANNOT see. "
        f"icept→pred {_wmean(all_events, 'e_icept_pred'):>5.2f}° prices the drag refinement."
    )

    if args.audit_hits:
        print("\n=== (5) HIT-POSSIBILITY AUDIT — can the reconstruction even permit the sim's hits? ===")
        d0 = np.array([e["d0_miss"] for e in all_events])
        envm = np.array([e["hit_env"] for e in all_events])
        wts = np.array([e["w"] for e in all_events])
        possible = float(wts[d0 <= envm].sum())
        certain = float(wts[d0 <= gl.HIT_RADIUS_M].sum())
        print(
            f"  undispersed closest approach (m): p05 {np.percentile(d0, 5):.1f}  "
            f"p25 {np.percentile(d0, 25):.1f}  median {np.median(d0):.1f}  "
            f"p75 {np.percentile(d0, 75):.1f}  p95 {np.percentile(d0, 95):.1f}"
        )
        print(
            f"  rounds inside the 8 m sphere with NO dispersion: {certain:.0f} / {t['rounds']:.0f}; "
            f"inside 8 m + 3σ·rng (dispersion could rescue): {possible:.0f}"
        )
        print(
            "  A round outside that envelope cannot hit under any draw, so the sim's hits must "
            f"be ≤ {possible:.0f}. Ehits(flown) = {t['E_flown']:.1f}. If the observed hits exceed "
            "the envelope count, the reconstruction (frame/timing), not the aim model, is wrong."
        )
        print(
            f"  CV-vs-real scoring: P(hit) flown {_wmean(all_events, 'p_flown'):.4f} real vs "
            f"{_wmean(all_events, 'p_flowncv'):.4f} constant-velocity  (miss "
            f"{_wmean(all_events, 'm_flown'):.1f} vs {_wmean(all_events, 'm_flowncv'):.1f} m) — "
            "how much a CV target would have distorted the scoring alone."
        )


if __name__ == "__main__":
    main()
