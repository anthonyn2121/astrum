"""D4 (surviving form): is the scripted T3's DEFENSIVE release gated on the LEARNER's
instantaneous pointing, and does that gate actually fire while the learner is building
turning room?

    cd controllers/rl && uv run python -u _diag_t3_release.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --cell chase --aspects 30,55 --episodes 6 --jitter 0.5

WHAT IS MEASURED
----------------
A copy of `_bfm_verdict_tmp.py`'s greedy driver (identical env constructor, identical
`Sample` cadence of one sample per POLICY step taken from the state the action was chosen
in, identical `_prev_mean.zero_()` between episodes, identical D6 checkpoint list in ONE
invocation), with four read-only additions per policy step:

  * `Tier3Opponent.state`  BEFORE the control step (`getattr(base.opponent, "state", "?")`,
    exactly as `_finish_trace_tmp.py:60,75-77`) and again AFTER it;
  * `Tier3Opponent._dwell` before and after. `_dwell` increments once per SIM tick inside
    `act()` and is zeroed by `_to()`, and `ActionRepeat` calls `DogfightEnv.step` (hence
    `opponent.act`) exactly `action_repeat` times per control step. So for a non-terminal
    control step, `dwell_after == dwell_before + action_repeat` iff NO transition happened
    anywhere inside the window, and `dwell_after <= action_repeat - 1` iff one did. That
    recovers FSM transitions at the full 120 Hz from a 24 Hz sample, which matters because
    a DEFENSIVE->NEUTRAL->DEFENSIVE "blip" can be a single 8.3 ms tick and would otherwise
    be invisible 4 times out of 5;
  * `rg_t.ata` -- the exact quantity `Tier3Opponent._transition` reads (`opponent.py:384-385`,
    `:480`). ⚠ IDENTITY, verified by source and asserted numerically in the header below:
    `act(s=bandit, opponent=learner)` computes `rg_t = relative(opponent, s)
    = relative(learner, bandit)`, and `relative(me, bandit).ata` is the angle off *me*'s
    nose. So `rg_t.ata` IS the learner's own antenna train angle to the bandit, i.e. it is
    numerically identical to `bfm_detect.Sample.ata`, which the detector already carried.
    The threat picture the T3 reacts to is "how well is the LEARNER pointing at me";
  * the T3's own airspeed and specific energy, so a DEFENSIVE exit into ENERGY_RECOVERY
    (the `V_ENERGY_LOW` channel, closed as C-5) is never mistaken for a pointing release.

Every occupancy / energy / ata statistic is reported BOTH unconditionally and conditioned on
`rng < THREAT_RANGE` (2500 m). The conditioned number is the one that means anything:
`threat_clear = ata > 65 deg OR rng > THREAT_RANGE_EXIT (3500 m)`, so any release measured
without a range condition is manufactured by the range gate.

Because `threat_clear` is a disjunction, a DEFENSIVE->NEUTRAL release observed at
`rng <= 3500 m` is *necessarily* caused by the nose gate (`ata > 65 deg`) -- there is no other
way for the predicate to be true. Releases are therefore bucketed by range at release:
`<2500` (unambiguous pointing gate, inside the threat bubble), `2500-3500` (pointing gate, but
outside the bubble the T3 would re-enter DEFENSIVE from), `>3500` (range gate, uninformative).

WHAT IS NOT MEASURED
--------------------
* Nothing is changed, tuned or added: no new opponent class, no reward touch, no fix. This is
  a descriptive read of the D6 control set's greedy flight. `opponent_tier` stays 0 (the full
  `Tier3Opponent`).
* Not a re-score of c87/c95/c96. Those numbers exist (§9); they are reprinted only so the
  release statistics sit next to the cell they were measured on, and c87 predates B1 so its
  obs column 29 is out of distribution here.
* Not evidence about whether the release HELPS or HURTS the learner. This box asks only
  whether the gate FIRES. The two payoff channels ("the release gifts the bandit energy",
  "the release gifts the bandit a shot") are already closed as C-5 and are not re-litigated.
* Not a statement about D1/D2/D3 counts as progress -- banned. Counts appear here only as
  window *addresses* to condition on.
* `tHit` is never used; lethality comes from `bfm_detect`'s range-gated `lethal` (`tLeth`).
* The per-window `D5hit` column is `lethality_lift`'s own `_hit` predicate (a range-gated lethal
  tick within `D5_WINDOW_S` of the event ENTRY), printed so the release ledger can be read
  against WHICH windows converted. It is descriptive context, not a success criterion, and it
  says nothing about causation in either direction.

PRE-REGISTERED INTERPRETATION
-----------------------------
The question, on the c96 aspect-55 wall cell (the one non-degenerate D5 lift on record: 4 of 9
turning-room windows followed by a lethal tick, 0 of 225 event-free windows), during those D3
turning-room windows:

  Does `rg_t.ata` ever exceed `THREAT_CONE_EXIT_RAD` = 65 deg, and does the T3 actually leave
  DEFENSIVE?

  * FAIL / mechanism VACUOUS: `rg_t.ata` stays under 65 deg through the loaded out-of-plane
    content, or every DEFENSIVE exit is attributable to `rng > 3500` or to `V_ENERGY_LOW`.
    Then the nose gate cannot be cancelling turning room, the residual third channel of the
    T3-release mechanism dies with the other two, and D4 closes for free. This is the outcome
    the geometry makes plausible: chi is the roll of the LIFT VECTOR off the LOS plane and does
    NOT imply a large ata -- the lift vector can be 90 deg out of plane with the nose still
    well inside a 65 deg cone.
  * PASS / VERIFIED DEFECT: `rg_t.ata` exceeds 65 deg inside D3 windows AND the T3 leaves
    DEFENSIVE at `rng <= 3500` m with airspeed >= `V_ENERGY_LOW`. Then the nose gate is
    recorded as a verified defect (it appears in none of the four record locations; both prior
    T3 rewrites touched `_break`'s control law, never `_transition`) and this box STOPS. No fix
    is implemented here.

Two exact, tick-aligned consistency checks are run and printed, because the whole box turns on
`rg_t.ata` being the right quantity. `_transition`'s first evaluation inside a control step
reads exactly the states this harness sampled, so:
  (A) state DEFENSIVE, `dwell_before + 1 >= FSM_MIN_DWELL ticks`, airspeed >= V_ENERGY_LOW and
      `threat_clear` TRUE  =>  a transition MUST occur in that window;
  (B) state in {NEUTRAL, OFFENSIVE, HIGH_YOYO} and `threatened` TRUE  =>  a transition MUST
      occur in that window.
If either check shows violations, the logged quantity is not the one the FSM reads and every
number below is void. That is reported as a hard PASS/FAIL line, not buried.

Runs on CPU so it never contends with a GPU training run.
"""

from __future__ import annotations

import argparse
import math
import statistics

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify

_STATES = ("neutral", "offensive", "high_yoyo", "defensive", "energy_recovery")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D4: T3 DEFENSIVE-release gate vs learner pointing.")
    p.add_argument(
        "--checkpoints",
        required=True,
        help="comma list. RUN first, D6 controls after (c95, c87) — one invocation.",
    )
    p.add_argument("--cell", default="chase", choices=("chase", "headon"))
    p.add_argument("--aspects", default="30,55")
    p.add_argument("--laterals", default="1400")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--separation", type=float, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--me-dalt-ft", type=float, default=None)
    p.add_argument("--opp-tier", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in a verdict")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _cell_defaults(args) -> None:
    if args.cell == "chase":
        d = dict(separation=700.0, max_time_s=60.0, me_dalt_ft=0.0)
    else:
        d = dict(separation=3000.0, max_time_s=180.0, me_dalt_ft=2000.0)
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def _cells(args) -> list[tuple[str, float]]:
    if args.cell == "chase":
        return [(f"asp{float(a):.0f}", float(a)) for a in args.aspects.split(",") if a.strip()]
    out = []
    for x in args.laterals.split(","):
        if not x.strip():
            continue
        lat = float(x)
        out.append((f"lat{lat:.0f}m", math.degrees(math.asin(min(1.0, lat / args.separation)))))
    return out


def _roll(agent, env, base, C, B, args, episodes: int, seed0: int):
    """Greedy episodes -> (outcome_rows, traces, fsm_logs). One entry per POLICY step."""
    from astrum_dogfight.geometry import relative, specific_energy

    dt = args.action_repeat / args.rate_hz
    rows, traces, logs = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0, t = True, 0.0
        event = None
        tr, lg = [], []
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            # Sample the state the action was CHOSEN in, before the step — and this is also
            # exactly the state `Tier3Opponent.act` is handed on the FIRST of the
            # `action_repeat` inner sim ticks, which is what makes checks (A)/(B) exact.
            me = base._states[base.learner_id]
            bd = base._states[base.opponent_id]
            rel = relative(me, bd)  # == `rg_t` inside Tier3Opponent._transition
            tr.append(B.make_sample(t, me, bd, a, rel))
            st_before = str(getattr(base.opponent, "state", "?"))
            dw_before = int(getattr(base.opponent, "_dwell", -1))
            rec = dict(
                t=t,
                ata_t=float(rel.ata),  # rg_t.ata — the learner's ATA, the FSM's threat scalar
                rng=float(rel.rng),
                st_before=st_before,
                dw_before=dw_before,
                v_opp=float(bd.airspeed),
                es_opp=specific_energy(bd),
                es_me=specific_energy(me),
                v_me=float(me.airspeed),
            )
            obs, _r, terminated, truncated, info = env.step(a)
            t += dt
            rec["st_after"] = str(getattr(base.opponent, "state", "?"))
            rec["dw_after"] = int(getattr(base.opponent, "_dwell", -1))
            rec["terminal"] = bool(terminated or truncated)
            lg.append(rec)
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rows.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "event": event,
                "dealt": C.HP_MAX - float(opp.health),
                "taken": C.HP_MAX - float(me.health),
            }
        )
        traces.append(tr)
        logs.append(lg)
    return rows, traces, logs


def _outcome_line(rows: list[dict]) -> str:
    n = len(rows)
    return (
        f"score: {n} eps | kills {sum(r['outcome'] == 'kill' for r in rows)} "
        f"deaths {sum(r['outcome'] == 'death' for r in rows)} "
        f"decks {sum(r['outcome'] == 'deck' for r in rows)} "
        f"t/out {sum(r['outcome'] == 'timeout' for r in rows)} | "
        f"dealt {statistics.fmean(r['dealt'] for r in rows):.1f} "
        f"taken {statistics.fmean(r['taken'] for r in rows):.1f}"
    )


def _pcts(vals: list[float]) -> str:
    if not vals:
        return "  n=0        —      —      —"
    v = np.asarray(vals, float)
    return (
        f"  n={len(v):<6d} p50 {np.percentile(v, 50):6.1f}  p90 {np.percentile(v, 90):6.1f}  "
        f"max {v.max():6.1f}"
    )


def _transitioned(rec: dict, repeat: int) -> bool:
    """Did the FSM change state at least once inside this control step?

    Exact for a non-terminal step: `_dwell` rises by exactly `repeat` iff nothing transitioned.
    On the terminal step the number of inner sim ticks is unknown (ActionRepeat breaks early),
    so fall back to the conservative test `dw_after <= dw_before`, which cannot false-positive
    for any inner count >= 1 and can only UNDER-count.
    """
    if rec["dw_before"] < 0 or rec["dw_after"] < 0:
        return False
    if rec["terminal"]:
        return rec["dw_after"] <= rec["dw_before"]
    return rec["dw_after"] != rec["dw_before"] + repeat


def _analyse(label, logs, traces, B, C, args, out):
    """Print the FSM/release block for one (checkpoint, cell)."""
    cone_exit = math.degrees(C.THREAT_CONE_EXIT_RAD)
    cone = math.degrees(C.THREAT_CONE_RAD)
    min_dwell = int(C.FSM_MIN_DWELL_S * args.rate_hz)
    rep = args.action_repeat

    flat = [(ei, i, r) for ei, lg in enumerate(logs) for i, r in enumerate(lg)]
    near = [(ei, i, r) for ei, i, r in flat if r["rng"] < C.THREAT_RANGE]

    # ---- occupancy -------------------------------------------------------------
    def occ(rows):
        n = max(1, len(rows))
        return {s: sum(1 for _, _, r in rows if r["st_before"] == s) / n for s in _STATES}

    o_all, o_near = occ(flat), occ(near)
    print(
        f"   FSM occupancy (state before each control step)  [ALL n={len(flat)}]"
        f"  [rng<{C.THREAT_RANGE:.0f}m n={len(near)}]"
    )
    for s in _STATES:
        print(f"     {s:<16s} all {o_all[s] * 100:5.1f}%   near {o_near[s] * 100:5.1f}%")
    unk = sum(1 for _, _, r in flat if r["st_before"] not in _STATES)
    if unk:
        print(f"     !! {unk} samples with an unrecognised state string")

    # ---- V / Es by state, conditioned on rng < 2500 ----------------------------
    print(f"   T3 airspeed / Es by state  (rng<{C.THREAT_RANGE:.0f}m only)")
    for s in _STATES:
        v = [r["v_opp"] for _, _, r in near if r["st_before"] == s]
        e = [r["es_opp"] for _, _, r in near if r["st_before"] == s]
        de = [r["es_me"] - r["es_opp"] for _, _, r in near if r["st_before"] == s]
        if not v:
            print(f"     {s:<16s} n=0")
            continue
        print(
            f"     {s:<16s} n={len(v):<6d} V {statistics.fmean(v):6.1f} m/s  "
            f"Es {statistics.fmean(e):7.0f} m  relEs(me-T3) {statistics.fmean(de):+7.0f} m"
        )

    # ---- ata inside vs outside D1/D3 event windows -----------------------------
    # Window address = [i0, i1] of every D1 (displacement roll) and D3 (turning room) event.
    in_win = [set() for _ in traces]
    ev_by_ep = []
    for ei, tr in enumerate(traces):
        d1 = B.detect_displacement_rolls(tr)
        d3 = B.detect_turning_room(tr)
        evs = sorted(d1 + d3, key=lambda e: e.i0)
        ev_by_ep.append(evs)
        for e in evs:
            in_win[ei].update(range(e.i0, e.i1 + 1))
    inside = [math.degrees(r["ata_t"]) for ei, i, r in near if i in in_win[ei]]
    outside = [math.degrees(r["ata_t"]) for ei, i, r in near if i not in in_win[ei]]
    n_ev = sum(len(e) for e in ev_by_ep)
    print(
        f"   rg_t.ata (deg) inside vs outside D1/D3 windows  (rng<{C.THREAT_RANGE:.0f}m only)"
        f"   [{n_ev} D1+D3 windows]"
    )
    print(f"     inside :{_pcts(inside)}")
    print(f"     outside:{_pcts(outside)}")
    over_in = sum(1 for v in inside if v > cone_exit)
    over_out = sum(1 for v in outside if v > cone_exit)
    print(
        f"     ticks with ata > THREAT_CONE_EXIT ({cone_exit:.0f} deg): "
        f"inside {over_in}/{len(inside)}  outside {over_out}/{len(outside)}"
    )

    # ---- the release ledger ----------------------------------------------------
    rel_rows = []
    for ei, i, r in flat:
        if r["st_before"] != "defensive" or not _transitioned(r, rep):
            continue
        rel_rows.append((ei, i, r))
    buckets = {"<2500": 0, "2500-3500": 0, ">3500": 0}
    to_energy = 0
    for ei, i, r in rel_rows:
        if r["st_after"] == "energy_recovery":
            to_energy += 1
            continue
        rr = r["rng"]
        buckets[
            "<2500"
            if rr < C.THREAT_RANGE
            else ("2500-3500" if rr <= C.THREAT_RANGE_EXIT else ">3500")
        ] += 1
    print(
        f"   DEFENSIVE releases (a transition out of DEFENSIVE inside a control step): "
        f"{len(rel_rows)} total"
    )
    print(
        f"     -> ENERGY_RECOVERY (V<{C.V_ENERGY_LOW:.0f} m/s, channel closed as C-5): {to_energy}"
    )
    print(
        f"     -> NEUTRAL by range at release:  rng<2500m {buckets['<2500']}"
        f"   2500-3500m {buckets['2500-3500']}   >3500m {buckets['>3500']}"
    )
    print(
        f"     NOTE threat_clear = (ata>{cone_exit:.0f}deg OR rng>{C.THREAT_RANGE_EXIT:.0f}m), so any "
        f"NEUTRAL release at rng<={C.THREAT_RANGE_EXIT:.0f}m is NECESSARILY the nose gate."
    )
    ptg = [
        (ei, i, r)
        for ei, i, r in rel_rows
        if r["st_after"] != "energy_recovery" and r["rng"] <= C.THREAT_RANGE_EXIT
    ]
    if ptg:
        print("     pointing-gate releases (rng<=3500m), first 20:")
        for ei, i, r in ptg[:20]:
            print(
                f"       ep{ei} t={r['t']:6.2f}s  ata {math.degrees(r['ata_t']):6.1f}  "
                f"rng {r['rng']:6.0f}  V(T3) {r['v_opp']:6.1f}  dwell {r['dw_before']}->"
                f"{r['dw_after']}  {r['st_before']}->{r['st_after']}"
                f"  {'IN-D1/D3-WINDOW' if i in in_win[ei] else ''}"
            )
    # Source-level cross-check independent of the transition detector: a DEFENSIVE sample that
    # is simultaneously dwell-satisfied, fast enough not to divert to ENERGY, and outside the
    # exit cone inside the bubble is a release the FSM is OBLIGED to take.
    obliged = [
        (ei, i, r)
        for ei, i, r in flat
        if r["st_before"] == "defensive"
        and r["dw_before"] + 1 >= min_dwell
        and r["v_opp"] >= C.V_ENERGY_LOW
        and r["rng"] < C.THREAT_RANGE
        and math.degrees(r["ata_t"]) > cone_exit
    ]
    obliged_win = sum(1 for ei, i, _ in obliged if i in in_win[ei])
    print(
        f"     FSM-OBLIGED pointing releases (DEFENSIVE & dwell_ok & V>=V_ENERGY_LOW & "
        f"rng<2500 & ata>{cone_exit:.0f}): {len(obliged)}  (of which inside a D1/D3 window: "
        f"{obliged_win})"
    )

    # ---- per-event table (the pre-registered question) -------------------------
    print("   Per D1/D3 event window (rg_t.ata over the window, FSM occupancy, releases):")
    if n_ev == 0:
        print("     (no D1/D3 events in this cell)")
    for ei, evs in enumerate(ev_by_ep):
        for e in evs:
            lg = logs[ei]
            idx = list(range(e.i0, min(e.i1 + 1, len(lg))))
            atas = [math.degrees(lg[i]["ata_t"]) for i in idx]
            rngs = [lg[i]["rng"] for i in idx]
            sts = {}
            for i in idx:
                sts[lg[i]["st_before"]] = sts.get(lg[i]["st_before"], 0) + 1
            rels = sum(
                1 for i in idx if lg[i]["st_before"] == "defensive" and _transitioned(lg[i], rep)
            )
            rels_ptg = sum(
                1
                for i in idx
                if lg[i]["st_before"] == "defensive"
                and _transitioned(lg[i], rep)
                and lg[i]["st_after"] != "energy_recovery"
                and lg[i]["rng"] <= C.THREAT_RANGE_EXIT
            )
            n_over = sum(1 for v in atas if v > cone_exit)
            occ_s = " ".join(f"{k[:4]}:{v}" for k, v in sorted(sts.items(), key=lambda x: -x[1]))
            # Is this one of the D5 "hit" windows? Same predicate `lethality_lift._hit` uses:
            # any range-gated lethal tick within D5_WINDOW_S of the event ENTRY index. Printed so
            # the release ledger can be read against WHICH windows converted, rather than only
            # against how many exist. Descriptive: not a success criterion, and D1/D3 counts stay
            # banned as evidence.
            trc = traces[ei]
            d5_hit = any(
                trc[k].lethal
                for k in range(e.i0, len(trc))
                if trc[k].t - trc[e.i0].t <= B.D5_WINDOW_S
            )
            print(
                f"     ep{ei} {e.kind:<17s} t {e.t0:6.2f}-{e.t1:6.2f}s ({e.duration_s:4.2f}s, "
                f"{len(idx)} steps) | ata p50 {np.percentile(atas, 50) if atas else float('nan'):5.1f} "
                f"max {max(atas) if atas else float('nan'):5.1f} "
                f"| >{cone_exit:.0f}deg {n_over}/{len(atas)} "
                f"| rng {min(rngs) if rngs else 0:5.0f}-{max(rngs) if rngs else 0:5.0f}m "
                f"| FSM {occ_s} | rel {rels} (ptg {rels_ptg}) "
                f"| D5hit {'Y' if d5_hit else 'n'} "
                f"| dAsp {e.d_aspect_deg:+6.1f} dRng {e.d_rng_m:+6.0f}"
            )

    # ---- exact tick-aligned consistency checks ---------------------------------
    a_tot = a_bad = b_tot = b_bad = 0
    for ei, i, r in flat:
        tc = math.degrees(r["ata_t"]) > cone_exit or r["rng"] > C.THREAT_RANGE_EXIT
        th = math.degrees(r["ata_t"]) < cone and r["rng"] < C.THREAT_RANGE
        if (
            r["st_before"] == "defensive"
            and r["dw_before"] + 1 >= min_dwell
            and r["v_opp"] >= C.V_ENERGY_LOW
            and tc
        ):
            a_tot += 1
            if not _transitioned(r, rep):
                a_bad += 1
        if r["st_before"] in ("neutral", "offensive", "high_yoyo") and th:
            b_tot += 1
            if not _transitioned(r, rep):
                b_bad += 1
    print(
        f"   CONSISTENCY (A) DEFENSIVE&dwell_ok&V_ok&threat_clear must transition: "
        f"{a_tot - a_bad}/{a_tot} ok, {a_bad} violations"
    )
    print(
        f"   CONSISTENCY (B) non-survival state & threatened must transition: "
        f"{b_tot - b_bad}/{b_tot} ok, {b_bad} violations"
    )
    out.append((label, a_tot, a_bad, b_tot, b_bad))


def main() -> None:
    args = parse_args()
    _cell_defaults(args)
    _paths.require_astrum_env()
    from astrum_dogfight import bfm_detect as B
    from astrum_dogfight import constants as C
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    cells = _cells(args)
    ctl_ms = 1000.0 * args.action_repeat / args.rate_hz

    print(
        f"D4 T3-RELEASE | cell {args.cell} | sep {args.separation:.0f}m mt{args.max_time_s:.0f}s "
        f"jitter{args.jitter:g} perch+{args.me_dalt_ft:.0f}ft ar{args.action_repeat} "
        f"opp_tier{args.opp_tier} | {args.episodes} eps/cell | seed {args.seed}"
    )
    print(
        f"one control step = {ctl_ms:.1f} ms ({args.rate_hz / args.action_repeat:.1f} Hz); the FSM "
        f"runs at {args.rate_hz:.0f} Hz = {args.action_repeat} evaluations per control step"
    )
    print(
        f"GATES: THREAT_CONE {math.degrees(C.THREAT_CONE_RAD):.0f}deg  THREAT_CONE_EXIT "
        f"{math.degrees(C.THREAT_CONE_EXIT_RAD):.0f}deg  THREAT_RANGE {C.THREAT_RANGE:.0f}m  "
        f"THREAT_RANGE_EXIT {C.THREAT_RANGE_EXIT:.0f}m  FSM_MIN_DWELL {C.FSM_MIN_DWELL_S:.3f}s = "
        f"{int(C.FSM_MIN_DWELL_S * args.rate_hz)} ticks  V_ENERGY_LOW {C.V_ENERGY_LOW:.0f}m/s"
    )
    print(
        "IDENTITY: Tier3Opponent.act(s=bandit, opponent=learner) => rg_t = relative(learner, "
        "bandit), so rg_t.ata is the LEARNER's ATA (opponent.py:480 + geometry.py:196) and is "
        "numerically identical to bfm_detect.Sample.ata."
    )
    print(
        "D6: the FIRST checkpoint is the run; the rest are controls, same env build, same seeds, "
        "same thresholds, one invocation."
    )

    checks = []
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        tag = "RUN" if ci == 0 else "CONTROL"
        print(
            f"\n{'=' * 100}\n[{tag}] {ck} (stage_idx={int(sd.get('stage_idx', -1))}, "
            f"collected={int(sd.get('collected', -1)):,})\n{'=' * 100}"
        )
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        for label, aspect in cells:
            kw = dict(
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            if args.opp_tier > 0:
                kw["opponent_tier"] = args.opp_tier
            if args.cell == "headon":
                kw["opp_heading_deg"] = 180.0
            env = make_tdmpc_env(3, **kw)
            base = env.unwrapped
            assert type(base.opponent).__name__ == "Tier3Opponent", type(base.opponent).__name__
            rows, traces, logs = _roll(
                agent, env, base, C, B, args, args.episodes, args.seed + int(round(aspect)) * 1000
            )
            env.close()
            print(f"\n-- greedy / {label} (bearing {aspect:.1f}°) --")
            print(f"   {_outcome_line(rows)}")
            for ln in B.format_summary(B.summarize(traces)).splitlines():
                print(f"   {ln}")
            _analyse(
                f"{ck.split('/')[-2] if '/' in ck else ck}/{label}",
                logs,
                traces,
                B,
                C,
                args,
                checks,
            )

    print(
        f"\n{'=' * 100}\nCONSISTENCY ROLL-UP (if any violations appear, rg_t.ata is the wrong "
        f"quantity and every number above is VOID)\n{'=' * 100}"
    )
    bad = 0
    for label, a_tot, a_bad, b_tot, b_bad in checks:
        bad += a_bad + b_bad
        print(f"  {label:<40s} A {a_tot - a_bad}/{a_tot}  B {b_tot - b_bad}/{b_tot}")
    print(f"  TOTAL VIOLATIONS: {bad}  =>  {'INSTRUMENT VALID' if bad == 0 else 'INSTRUMENT VOID'}")


if __name__ == "__main__":
    main()
