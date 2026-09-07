"""ADVERSARIAL VERIFICATION of box D4 (`_diag_t3_release.py`). Three questions the reported
run left open, all answered in ONE invocation with the full D6 control set.

    cd controllers/rl && uv run python -u _diag_d4_verify.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --cell chase --aspects 30,55 --episodes 6 --jitter 0.5

WHAT IS MEASURED
----------------
Same greedy driver, env constructor, per-policy-step `Sample` cadence, `_prev_mean.zero_()`,
seeds (`seed + round(aspect)*1000`) and thresholds as `_diag_t3_release.py`, so every episode is
byte-identical to the reported run (its own two invocations already agree exactly on c96/asp55).
Three additions, each aimed at one hole in the reported conclusion:

  (Q1) The reported ledger flags a pointing release as `IN-D1/D3-WINDOW`. A turning-room window
       spans the loaded out-of-plane HOLD *plus* up to `CHI_RETURN_S` = 4.0 s of the in-plane
       "cash" phase (`bfm_detect.detect_turning_room:474-486`), and the reported table printed
       neither `detail["hold_s"]` nor the per-tick predicate. So "inside a window" does not
       establish "during real loaded out-of-plane content", which is the exact wording of D4's
       FAIL branch. Here every release tick is tested DIRECTLY against the detector's own
       predicates: `chi_valid and |chi| > CHI_ROOM_DEG (45) and nz > NZ_LOADED (3)` (the D3 bar)
       and `|chi| > CHI_OUT_DEG (60) and nz > 3` (the D1 bar). Reported as conditional rates:
       P(ata > 65 deg | loaded-out-of-plane tick, rng < 2500) against P(ata > 65 | rng < 2500).
  (Q2) The release-vs-conversion read (`D5hit`) exists only in a SECOND invocation that dropped
       both controls. c95 and c87 are restored here.
  (Q3) `D5hit` is `lethality_lift`'s `_hit`: a lethal tick within `D5_WINDOW_S` = 8 s of the event
       ENTRY, and `lethal` requires `rng <= LETHAL_RANGE_M` = 600 m. A window whose whole credit
       interval sits outside 600 m CANNOT score a hit whatever the opponent does, so `D5hit n` is
       over-determined by range. `min rng` over each credit interval is printed and the
       release x D5hit table is stratified by it.

WHAT IS NOT MEASURED
--------------------
* Nothing is changed, tuned, added or fixed: no opponent class, no reward, no threshold. Read-only
  re-roll of saved checkpoints at `opponent_tier` 0 (the stock `Tier3Opponent`).
* NOT whether the release helps or hurts the learner. That needs a counterfactual opponent, which
  D4 forbids in this box. Nothing here licenses a causal claim in either direction.
* NOT a re-score of c96/c95/c87 (those are §9's); the outcome line is printed only as the
  replication check that the episodes match the reported run.
* D1/D3 counts are addresses, never progress. `tHit` unused. The D5 *ratio* is never printed.

PRE-REGISTERED INTERPRETATION
-----------------------------
D4's PASS condition is "`rg_t.ata` exceeds 65 deg inside D3 windows AND the T3 leaves DEFENSIVE at
rng <= 3500 m with airspeed >= V_ENERGY_LOW". That is a statement about window MEMBERSHIP and it
either replicates or it does not; this harness cannot overturn it and is not trying to.
What it can settle:
  * If P(ata > 65 | loaded-out-of-plane tick) is at or above P(ata > 65 | any near tick), and some
    pointing release lands ON a loaded out-of-plane tick, then the gate genuinely fires during the
    maneuver and D4's PASS survives at full strength.
  * If P(ata > 65 | loaded-out-of-plane tick) is well BELOW the unconditional rate and no release
    lands on such a tick, then the gate fires in the in-plane cash phase / elsewhere, and D4's PASS
    must be narrowed to "the gate fires inside the window's padding", which is a materially weaker
    statement than "the release cancels turning room being built".
  * If every window carrying a release has `minRng` over its credit interval > 600 m, the reported
    release-vs-`D5hit` anti-correlation is a range artifact and must not be reported as evidence
    that the release cost a conversion.
  * If the controls (c95, c87) produce pointing releases at a comparable rate per near-tick while
    building fewer or zero turning-room windows, then release incidence is not diagnostic of the
    learner doing BFM.
Every one of these can come out either way. CPU only.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.eval_policy import _classify


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial verification of D4.")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--cell", default="chase", choices=("chase", "headon"))
    p.add_argument("--aspects", default="30,55")
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--separation", type=float, default=None)
    p.add_argument("--max-time-s", type=float, default=None)
    p.add_argument("--me-dalt-ft", type=float, default=None)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _cell_defaults(args) -> None:
    d = (
        dict(separation=700.0, max_time_s=60.0, me_dalt_ft=0.0)
        if args.cell == "chase"
        else dict(separation=3000.0, max_time_s=180.0, me_dalt_ft=2000.0)
    )
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def _transitioned(rec: dict, repeat: int) -> bool:
    """FSM changed state at least once inside this control step (dwell arithmetic at 120 Hz)."""
    if rec["dw_before"] < 0 or rec["dw_after"] < 0:
        return False
    if rec["terminal"]:
        return rec["dw_after"] <= rec["dw_before"]
    return rec["dw_after"] != rec["dw_before"] + repeat


def _roll(agent, env, base, C, B, args, episodes: int, seed0: int):
    from astrum_dogfight.geometry import relative

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
            me = base._states[base.learner_id]
            bd = base._states[base.opponent_id]
            rel = relative(me, bd)  # == rg_t inside Tier3Opponent._transition
            tr.append(B.make_sample(t, me, bd, a, rel))
            rec = dict(
                t=t,
                ata=math.degrees(float(rel.ata)),
                rng=float(rel.rng),
                st_before=str(getattr(base.opponent, "state", "?")),
                dw_before=int(getattr(base.opponent, "_dwell", -1)),
                v_opp=float(bd.airspeed),
            )
            obs, _r, terminated, truncated, info = env.step(a)
            t += args.action_repeat / args.rate_hz
            rec["st_after"] = str(getattr(base.opponent, "state", "?"))
            rec["dw_after"] = int(getattr(base.opponent, "_dwell", -1))
            rec["terminal"] = bool(terminated or truncated)
            lg.append(rec)
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        opp = base._states[base.opponent_id]
        rows.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "dealt": C.HP_MAX - float(opp.health),
            }
        )
        traces.append(tr)
        logs.append(lg)
    return rows, traces, logs


def _analyse(logs, traces, B, C, args, roll):
    rep = args.action_repeat
    oop45, oop60 = [], []  # per-episode boolean masks: the detector's own tick predicates
    for tr in traces:
        oop45.append(
            [
                bool(
                    s.chi_valid and abs(math.degrees(s.chi)) > B.CHI_ROOM_DEG and s.nz > B.NZ_LOADED
                )
                for s in tr
            ]
        )
        oop60.append(
            [
                bool(
                    s.chi_valid and abs(math.degrees(s.chi)) > B.CHI_OUT_DEG and s.nz > B.NZ_LOADED
                )
                for s in tr
            ]
        )

    flat = [(ei, i, r) for ei, lg in enumerate(logs) for i, r in enumerate(lg)]
    near = [(ei, i, r) for ei, i, r in flat if r["rng"] < C.THREAT_RANGE]

    def rate(rows, mask=None):
        sel = [r for ei, i, r in rows if mask is None or mask[ei][i]]
        if not sel:
            return 0, 0, float("nan")
        over = sum(1 for r in sel if r["ata"] > 65.0)
        return over, len(sel), 100.0 * over / len(sel)

    print("   (Q1) P(ata > THREAT_CONE_EXIT 65deg | tick), rng<2500m only")
    for name, mask in (
        ("any near tick", None),
        ("loaded |chi|>45 & nz>3 (D3 bar)", oop45),
        ("loaded |chi|>60 & nz>3 (D1 bar)", oop60),
    ):
        o, n, p = rate(near, mask)
        print(f"     {name:<34s} {o:6d}/{n:<6d} = {p:5.1f}%")

    # windows
    in_win = [set() for _ in traces]
    ev_by_ep = []
    for ei, tr in enumerate(traces):
        evs = sorted(
            B.detect_displacement_rolls(tr) + B.detect_turning_room(tr), key=lambda e: e.i0
        )
        ev_by_ep.append(evs)
        for e in evs:
            in_win[ei].update(range(e.i0, e.i1 + 1))

    rel_rows = [
        (ei, i, r)
        for ei, i, r in flat
        if r["st_before"] == "defensive"
        and _transitioned(r, rep)
        and r["st_after"] != "energy_recovery"
        and r["rng"] <= C.THREAT_RANGE_EXIT
    ]
    n_near = max(1, len(near))
    print(
        f"   pointing-gate releases: {len(rel_rows)} in {len(near)} near ticks "
        f"= {1000.0 * len(rel_rows) / n_near:.2f} per 1000 near ticks "
        f"(exposure-normalised, raw counts are not comparable across cells)"
    )
    n_in_win = sum(1 for ei, i, _ in rel_rows if i in in_win[ei])
    n_on_45 = sum(1 for ei, i, _ in rel_rows if oop45[ei][i])
    n_on_60 = sum(1 for ei, i, _ in rel_rows if oop60[ei][i])
    print(
        f"     of which: inside a D1/D3 window {n_in_win}  |  ON a loaded |chi|>45 tick {n_on_45}"
        f"  |  ON a loaded |chi|>60 tick {n_on_60}"
    )
    for ei, i, r in rel_rows:
        tr = traces[ei]
        print(
            f"       ep{ei} t={r['t']:6.2f}s ata {r['ata']:5.1f} rng {r['rng']:6.0f} "
            f"V(T3) {r['v_opp']:6.1f} | |chi| {abs(math.degrees(tr[i].chi)):5.1f} "
            f"valid {int(tr[i].chi_valid)} nz {tr[i].nz:4.1f} "
            f"| in-window {'Y' if i in in_win[ei] else 'n'} "
            f"| loaded-oop45 {'Y' if oop45[ei][i] else 'n'}"
        )

    # per-window ledger with hold_s, credit-window min range, D5hit, releases
    print("   (Q2/Q3) per D1/D3 window: hold, credit-interval min range, D5hit, releases")
    tab = {}  # (has_release, d5hit) -> n ; and range-feasibility split
    for ei, evs in enumerate(ev_by_ep):
        tr, lg = traces[ei], logs[ei]
        for e in evs:
            idx = list(range(e.i0, min(e.i1 + 1, len(lg))))
            credit = [k for k in range(e.i0, len(tr)) if tr[k].t - tr[e.i0].t <= B.D5_WINDOW_S]
            d5 = any(tr[k].lethal for k in credit)
            min_rng = min(tr[k].rng for k in credit) if credit else float("nan")
            feasible = min_rng <= B.LETHAL_RANGE_M
            rels = [
                i
                for i in idx
                if lg[i]["st_before"] == "defensive"
                and _transitioned(lg[i], rep)
                and lg[i]["st_after"] != "energy_recovery"
                and lg[i]["rng"] <= C.THREAT_RANGE_EXIT
            ]
            hold = e.detail.get("hold_s")
            hold_s = f"{hold:4.2f}" if isinstance(hold, (int, float)) else "  - "
            on45 = sum(1 for i in rels if oop45[ei][i])
            key = (bool(rels), bool(d5), bool(feasible))
            tab[key] = tab.get(key, 0) + 1
            print(
                f"     ep{ei} {e.kind:<17s} t {e.t0:6.2f}-{e.t1:6.2f}s hold {hold_s}s "
                f"| oop45 ticks {sum(1 for i in idx if oop45[ei][i]):3d}/{len(idx):3d} "
                f"| >65deg {sum(1 for i in idx if lg[i]['ata'] > 65.0):3d}/{len(idx):3d} "
                f"| creditMinRng {min_rng:6.0f}m lethal-capable {'Y' if feasible else 'n'} "
                f"| D5hit {'Y' if d5 else 'n'} | rel {len(rels)} (on-oop45 {on45})"
            )
    if tab:
        print("   release x D5hit x lethal-capable(creditMinRng<=600m) window counts:")
        for feas in (True, False):
            for relv in (True, False):
                y = tab.get((relv, True, feas), 0)
                n = tab.get((relv, False, feas), 0)
                if y or n:
                    print(
                        f"     lethal-capable {'Y' if feas else 'n'}  release "
                        f"{'Y' if relv else 'n'}:  D5hit Y {y}  D5hit n {n}"
                    )
    roll.append(None)


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
    cells = [(f"asp{float(a):.0f}", float(a)) for a in args.aspects.split(",") if a.strip()]

    print(
        f"D4-VERIFY | cell {args.cell} | sep {args.separation:.0f}m mt{args.max_time_s:.0f}s "
        f"jitter{args.jitter:g} ar{args.action_repeat} | {args.episodes} eps/cell | seed {args.seed}"
    )
    print(
        f"one control step = {1000.0 * args.action_repeat / args.rate_hz:.1f} ms "
        f"({args.rate_hz / args.action_repeat:.1f} Hz); FSM at {args.rate_hz:.0f} Hz"
    )
    print(
        f"bars: CHI_ROOM {B.CHI_ROOM_DEG:.0f} CHI_OUT {B.CHI_OUT_DEG:.0f} NZ_LOADED "
        f"{B.NZ_LOADED:.0f} LETHAL_RANGE {B.LETHAL_RANGE_M:.0f}m D5_WINDOW {B.D5_WINDOW_S:.0f}s "
        f"| gates: CONE_EXIT 65deg RANGE_EXIT {C.THREAT_RANGE_EXIT:.0f}m V_ENERGY_LOW "
        f"{C.V_ENERGY_LOW:.0f}m/s"
    )
    print("D6: first checkpoint = the run, the rest are controls, one invocation, same seeds.")

    roll = []
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        print(
            f"\n{'=' * 100}\n[{'RUN' if ci == 0 else 'CONTROL'}] {ck} "
            f"(stage_idx={int(sd.get('stage_idx', -1))}, collected={int(sd.get('collected', -1)):,})"
            f"\n{'=' * 100}"
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
            print(
                f"   replication check: kills "
                f"{sum(r['outcome'] == 'kill' for r in rows)}/{len(rows)} "
                f"decks {sum(r['outcome'] == 'deck' for r in rows)} "
                f"t/out {sum(r['outcome'] == 'timeout' for r in rows)} | "
                f"lethalTicks {sum(1 for tr in traces for s in tr if s.lethal)}"
            )
            _analyse(logs, traces, B, C, args, roll)


if __name__ == "__main__":
    main()
