"""F8 — re-derive the POLICY-side D1/D2/D3 readings under the REPAIRED detector, and persist the traces.

    cd controllers/rl && ASTRUM_C97_VERTICAL=0 uv run python _diag_f8_policy_d1_reread.py \
        --checkpoints runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt,runs/curriculum-96/run.pt \
        --tag obs41 --episodes 8 --aspects 0,30,55

WHY THIS BOX EXISTS. F7 landed and audited the four-step D1 repair on 15 SCRIPTED arms, and closed with
an explicit ⚠: the POLICY-side readings this program has recorded — c95's `D1 = D2 = D3 = 0.00` on the
chase ([[bfm-policy-flies-no-bfm]]) and D2d's three arms ([[coloured-exploration-landed]]) — were
measured with the PRE-repair detector and were never re-derived. F7 labelled their robustness a
PREDICTION, on the reasoning that those policies fail at `hold_s`, upstream of every clause the repair
touches. A prediction recorded as a result is exactly the failure mode
[[free-tier-overturned-verdicts]] catalogues, so this box measures it.

⛔ THE REASON A RE-DERIVATION IS NOT MERELY CONFIRMATORY. F7 proved the repair is **not monotone in the
event count**: repair 1 is a filter INSIDE the exit search, so the loop `continue`s past a disqualified
exit and keeps scanning to `CHI_RETURN_S`. On `lead_turn`/`coach` it RELOCATED the exit (154.4° →
119.8°) rather than dropping the event. Relocation can also run the other way: if `pre`'s exit died on
a LATER clause — (f) failing at that particular tick — a repaired config can relocate to a tick where
(f) passes and FIRE WHERE `pre` DID NOT. So "the repair only tightens, therefore 0 stays 0" is a
non-argument, and the SHIPPED column has to be read, not assumed.

PRE-REGISTERED, written before the first episode was flown:

  P1  SHIPPED D1 = 0.00 on every checkpoint, matching `pre`. This is F7's prediction and the expected
      outcome. It CLOSES the ⚠ on [[bfm-and-kill-are-independent]]'s Direction A.
  P2  Any config where D1 > 0 while `pre` = 0 is a RELOCATION-CREATED event. It is not evidence of BFM
      until a per-event autopsy (exit ata, exit closure, BOTH currencies) says so — the same standard
      F7 imposed on a nonzero merge D1, and for the same reason: the repaired D1 has NO POSITIVE
      CONTROL, so it can only carry absence.
  P3  If `pre` itself reads nonzero on a checkpoint whose record says 0.00, the RECORD is what is
      wrong, not the repair — this cell/seed set differs from whatever produced the recorded number,
      and the recorded phrasing must be narrowed rather than the new reading discarded.

  ⚠ A null on P1 does NOT show the policies fly BFM badly or well. It shows the recorded zeros survive
    the instrument change. Sensitivity remains unmeasured on every arm ([[bfm-detector-spec]]).

WHAT IT PERSISTS. Traces go to `runs/f8/policy_traces_<tag>.npz` in F7's FROZEN 17-field order, reusing
`_diag_f7_detector_repair`'s `trace_to_array`/`array_to_trace` (which assert the width and say *re-fly,
do not re-score* if `Sample` changed). F7's dividend was that 120 scripted traces made every later
threshold question free; the policy side had the same gap and this closes it. Rescoring all eight
configs after the fly costs seconds, so the 8-column table is free once the episodes exist.

⚠ OBS WIDTH. `ASTRUM_C97_VERTICAL` fixes `OBS_DIM` at import, so a 45-obs checkpoint cannot share a
process with a 41-obs one. Pass `--tag` to keep the two npz files apart; the tag is recorded in the
file so a later reader cannot cross them.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import _bfm_verdict_tmp as V
import _diag_f7_detector_repair as F7
import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

OUT = "/workspace/controllers/rl/runs/f8"
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def parse_args():
    p = argparse.ArgumentParser(description="Policy-side D1 re-read under the repaired detector.")
    p.add_argument("--checkpoints", required=True, help="comma list; the FIRST is the RUN, rest controls")
    p.add_argument("--tag", required=True, help="obs-width tag; names the npz so widths cannot cross")
    p.add_argument("--aspects", default="0,30,55")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--me-dalt-ft", type=float, default=0.0)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in a verdict")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import bfm_detect as B
    from astrum_dogfight import constants as C
    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    os.makedirs(OUT, exist_ok=True)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    aspects = [float(a) for a in args.aspects.split(",") if a.strip()]

    log("#" * 112)
    log("#  F8 — policy-side D1/D2/D3 under the REPAIRED detector (F7's untested prediction, measured)")
    log("#" * 112)
    log(
        f"\n  obs tag {args.tag}   OBS_DIM {C.OBS_DIM}   C97_VERTICAL_OBS={C.C97_VERTICAL_OBS}"
        f"   cell chase sep {args.separation:.0f}m mt{args.max_time_s:.0f}s jitter{args.jitter:g}"
    )
    log(f"  aspects {aspects}   {args.episodes} eps/aspect   greedy, eval_mode=True")
    log(f"  module defaults (SHIPPED): {F7.shipped_defaults()}")
    log("  D6: the FIRST checkpoint is the RUN; the rest are controls.")

    # ---------------------------------------------------------------------------------------------
    # Fly once, persist, then rescore every config off the saved traces.
    # ---------------------------------------------------------------------------------------------
    store: dict = {}
    outcomes: dict = {}
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location=device, weights_only=False)
        name = ck.replace("runs/", "").replace("/run.pt", "").replace(".pt", "").replace("/", "_")
        tagged = "RUN" if ci == 0 else "CONTROL"
        collected = int(sd.get("collected", -1))
        log(f"\n{'=' * 112}")
        log(f"[{tagged}] {ck}  (stage_idx={int(sd.get('stage_idx', -1))}, collected={collected:,})")
        log("=" * 112)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False  # greedy is the out-of-band read; MPC is death-seeking here (E3)
        agent = TDMPC2(cfg, device)
        agent.load_state_dict(sd)
        agent.model.eval()
        roll_args = SimpleNamespace(action_repeat=args.action_repeat, rate_hz=args.rate_hz)
        rows_all = []
        for asp in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=asp,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
                jitter=args.jitter,
            )
            if args.me_dalt_ft:
                kw["me_dalt_ft"] = args.me_dalt_ft
            env = make_tdmpc_env(3, **kw)
            base = env.unwrapped
            rows, traces = V._roll(
                agent, env, base, C, B, roll_args, args.episodes, args.seed + int(round(asp)) * 1000
            )
            env.close()
            rows_all.extend(rows)
            for k, tr in enumerate(traces):
                store[(f"{name}::asp{asp:.0f}", k)] = (F7.trace_to_array(tr), np.zeros(1))
            log(f"  -- asp{asp:.0f}: {V._outcome_line(rows)}")
        outcomes[name] = rows_all
        log(f"  == pooled: {V._outcome_line(rows_all)}")

    # Persist in F7's frozen format, keyed so a width cannot be mixed in later.
    flat = {}
    for (key, ep), (arr, oc) in store.items():
        flat[f"tr::{key}::{ep}"] = arr.astype(np.float32)
        flat[f"oc::{key}::{ep}"] = oc
    path = os.path.join(OUT, f"policy_traces_{args.tag}.npz")
    np.savez_compressed(path, fields=np.asarray(F7.FIELDS), obs_tag=np.asarray([args.tag]), **flat)
    log(f"\n  persisted {len(store)} policy traces -> {path}")

    # ---------------------------------------------------------------------------------------------
    # The table this box exists for: every config, every checkpoint, pooled over all aspects.
    # ---------------------------------------------------------------------------------------------
    by_ck: dict[str, list] = {}
    for (key, _ep), (arr, _oc) in store.items():
        by_ck.setdefault(key.split("::")[0], []).append(F7.array_to_trace(arr))

    log("\n" + "=" * 112)
    log("TABLE — D1 / D2 / D3 per episode, pooled over aspects, under all eight detector configs")
    log("  `pre` reproduces the instrument the RECORDED readings were taken on; SHIPPED is the module now.")
    log("=" * 112)
    cfgs = list(F7.CONFIGS)
    log(f"  {'checkpoint':<26} {'metric':>7} " + " ".join(f"{c['tag']:>9}" for c in cfgs))
    d1_table: dict[str, dict[str, float]] = {}
    for name, trs in by_ck.items():
        cells: dict[str, dict[str, float]] = {}
        for c in cfgs:
            F7.apply_cfg(c)
            s = B.summarize(trs)
            cells[c["tag"]] = dict(d1=s["d1_per_ep"], d2=s["d2_per_ep"], d3=s["d3_per_ep"])
        d1_table[name] = {t: v["d1"] for t, v in cells.items()}
        for metric in ("d1", "d2", "d3"):
            log(
                f"  {name if metric == 'd1' else '':<26} {metric.upper():>7} "
                + " ".join(f"{cells[c['tag']][metric]:>9.2f}" for c in cfgs)
            )
    F7.apply_cfg(F7.CONFIGS[0])

    # ---------------------------------------------------------------------------------------------
    log("\n" + "=" * 112)
    log("VERDICT — against the pre-registration")
    log("=" * 112)
    shipped_tag = "SHIPPED"
    any_fire = False
    any_created = False
    for name, row in d1_table.items():
        pre, ship = row["pre"], row[shipped_tag]
        created = [t for t, v in row.items() if v > 0.0 and pre == 0.0]
        if ship > 0.0:
            any_fire = True
        if created:
            any_created = True
        log(f"  {name:<26} pre {pre:.2f}   SHIPPED {ship:.2f}   configs firing off a zero `pre`: {created or 'none'}")
    log("")
    if not any_fire and not any_created:
        log("  P1 CONFIRMED: SHIPPED D1 = 0.00 on every checkpoint, and NO config fires where `pre` is 0.")
        log("     => F7's prediction holds. The recorded policy-side zeros survive the repair, so")
        log("        [[bfm-and-kill-are-independent]] Direction A no longer carries the un-re-derived ⚠.")
        log("     ⚠ This is ABSENCE evidence only. Sensitivity is still unmeasured (no positive control).")
    elif any_created:
        log("  P2 TRIGGERED: a repaired config fires where `pre` does not — RELOCATION-CREATED.")
        log("     => do NOT report this as BFM. Autopsy each event (exit ata, exit closure, both")
        log("        currencies) before it is believed, exactly as F7 requires for a nonzero merge D1.")
    else:
        log("  P3 TERRITORY: `pre` is itself nonzero here — the recorded 0.00 was a different cell/seed set.")
        log("     => narrow the RECORDED phrasing; do not discard this reading.")

    out = os.path.join(OUT, f"f8_policy_d1_reread_{args.tag}.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
