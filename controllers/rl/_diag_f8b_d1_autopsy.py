"""F8b — per-event autopsy of the FIRST nonzero SHIPPED D1 events ever recorded on a POLICY.

    cd controllers/rl && ASTRUM_C97_VERTICAL=0 uv run python _diag_f8b_d1_autopsy.py --tag obs41

WHY THIS IS MANDATORY AND NOT OPTIONAL. F7 closed with a hard rule, because the repaired D1 has NO
POSITIVE CONTROL — specificity was measured on 15 scripted arms, sensitivity on none: *"a merge/head-on
D1 count is admissible as evidence of ABSENCE, never of presence; a nonzero one is a new observation
requiring a per-event autopsy (exit ata, exit closure, BOTH currencies) before it is reported."*  F8
then produced exactly that new observation on the CHASE: c95 and c87 read D1 = 0.08/ep under SHIPPED
(2 events / 24 episodes each), where c96 and c97-ctl read 0.00.  Reporting "c95 flies displacement
rolls" off that number without opening the events would be the same error F2 caught on the merge, where
D1 fired 8/8 on four arms that each dealt 0.0 HP.

This costs seconds and no JSBSim: F8 persisted all 96 policy traces in F7's frozen 17-field order.

WHAT IT PRINTS, per surviving event: entry/exit time and the hold, chi peak, `nz` over the hold, ata
entry -> exit and `d_ata`, `d_aspect`, closure at entry/exit and over the hold, minimum range, and
whether a LETHAL tick (`rng <= LETHAL_RANGE_M` with a sub-`WEZ_HIT_M` lay, i.e. `Sample.lethal`) occurs
inside the event or within `D5_CREDIT_S` after it.  The last column is the one that decides whether the
event is BFM or bookkeeping.

PRE-REGISTERED READINGS, before opening any event:

  A1  The events are real converting displacement rolls  =>  d_ata clearly negative, exit inside the
      forward hemisphere, closure non-opening across the hold, and a lethal tick following. This would
      be the FIRST positive control the repaired D1 has ever had, on any trajectory, and would upgrade
      merge/head-on D1 counts from absence-only to interpretable.
  A2  The events are chase-cell bookkeeping  =>  no lethal tick follows, and/or the geometry shows the
      roll accomplished nothing (range opening off a minimum, ata barely moving, or the hold sitting at
      the 0.5 s / nz 3 g bars). Then c95's nonzero D1 is another instance of Direction B in
      [[bfm-and-kill-are-independent]] — BFM-shaped motion that does not convert — and the correct
      report is "D1 is nonzero and it still buys nothing", NOT "c95 flies BFM".
  A3  ⚠ Either way this does NOT resolve sensitivity. A2 leaves the repaired D1 with no positive
      control; A1 would give it one only on the chase, where (f) is doing honest work anyway.

⚠ It also re-checks F7's Q2 refutation on POLICY traces, independently of the scripted arms: F8 measured
`r1` RAISING c95's D1 from 0.12 to 0.17, which is the relocation non-monotonicity showing up on a
learned trajectory. The per-config exit table below is where that is visible tick by tick.
"""

from __future__ import annotations

import argparse
import math
import os

import _diag_f7_detector_repair as F7
import numpy as np
from astrum_dogfight import bfm_detect as B

OUT = "/workspace/controllers/rl/runs/f8"
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _deg(r: float) -> float:
    return math.degrees(r)


def parse_args():
    p = argparse.ArgumentParser(description="Per-event autopsy of policy-side D1 events.")
    p.add_argument("--tag", default="obs41")
    p.add_argument("--credit-s", type=float, default=8.0, help="D5 credit window after the event")
    return p.parse_args()


def load(tag: str) -> dict:
    z = np.load(os.path.join(OUT, f"policy_traces_{tag}.npz"), allow_pickle=False)
    got = tuple(str(x) for x in z["fields"])
    assert got == F7.FIELDS, f"saved field order {got} != {F7.FIELDS} — re-fly, do not re-score"
    out = {}
    for k in z.files:
        if not k.startswith("tr::"):
            continue
        _, ck, cell, ep = k.split("::")
        out[(ck, cell, int(ep))] = F7.array_to_trace(z[k])
    return out


def main() -> None:
    args = parse_args()
    os.makedirs(OUT, exist_ok=True)
    traces = load(args.tag)
    cks = sorted({c for c, _, _ in traces})

    log("#" * 118)
    log("#  F8b — autopsy of the first nonzero SHIPPED D1 events on a learned policy")
    log("#" * 118)
    log(f"\n  {len(traces)} saved policy traces, no sim   (tag {args.tag})")
    log(f"  SHIPPED = module defaults: {F7.shipped_defaults()}")
    log(f"  LETHAL_RANGE_M {B.LETHAL_RANGE_M:.0f}   credit window {args.credit_s:.0f} s")

    # ---------------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("TABLE A — every SHIPPED D1 event, opened up")
    log("  `leth in` = lethal tick inside the event | `leth aft` = lethal tick within the credit window")
    log("=" * 118)
    F7.apply_cfg(F7.CONFIGS[-2])  # SHIPPED
    hdr = (
        f"  {'checkpoint':<16} {'cell':>7} {'ep':>3} {'t0':>6} {'t1':>6} {'hold':>5} {'chiPk':>6} "
        f"{'nzMin':>6} {'ataIn':>6} {'ataEx':>6} {'dAta':>7} {'dAsp':>7} {'cloIn':>6} {'cloEx':>6} "
        f"{'rngMin':>7} {'leth in':>7} {'leth aft':>8}"
    )
    log(hdr)
    n_events = 0
    n_lethal = 0
    for ck in cks:
        for (c, cell, ep), tr in sorted(traces.items()):
            if c != ck:
                continue
            for e in B.detect_displacement_rolls(tr):
                n_events += 1
                i0, i1 = e.i0, e.i1
                hold = [tr[k] for k in range(i0, min(i1 + 1, len(tr)))]
                nz_min = min(s.nz for s in hold) if hold else float("nan")
                chi_pk = max(abs(_deg(s.chi)) for s in hold if s.chi_valid) if hold else float("nan")
                leth_in = any(s.lethal for s in hold)
                t_end = e.t1 + args.credit_s
                leth_aft = any(s.lethal for s in tr if e.t1 < s.t <= t_end)
                if leth_in or leth_aft:
                    n_lethal += 1
                log(
                    f"  {ck[:16]:<16} {cell:>7} {ep:>3} {e.t0:>6.2f} {e.t1:>6.2f} "
                    f"{e.t1 - e.t0:>5.2f} {chi_pk:>6.1f} {nz_min:>6.2f} "
                    f"{e.detail['ata_entry_deg']:>6.1f} {e.detail['ata_exit_deg']:>6.1f} "
                    f"{e.detail['d_ata_deg']:>+7.1f} {e.d_aspect_deg:>+7.1f} "
                    f"{e.detail['closure_entry']:>+6.0f} {e.detail['closure_exit']:>+6.0f} "
                    f"{e.detail['rng_min_m']:>7.0f} {str(leth_in):>7} {str(leth_aft):>8}"
                )
    if n_events == 0:
        log("  (no SHIPPED D1 event on any trace)")

    # ---------------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("TABLE B — the same traces under every config: event COUNT and the exit it selects")
    log("  F7's Q2, re-tested on LEARNED trajectories: a constraint inside the exit search can RELOCATE")
    log("  an exit instead of dropping the event, so the count is NOT monotone in added constraints.")
    log("=" * 118)
    log(f"  {'checkpoint':<16} " + " ".join(f"{c['tag']:>9}" for c in F7.CONFIGS))
    per_cfg: dict[str, dict[str, int]] = {}
    for ck in cks:
        trs = [tr for (c, _, _), tr in traces.items() if c == ck]
        row = {}
        for cfg in F7.CONFIGS:
            F7.apply_cfg(cfg)
            row[cfg["tag"]] = sum(len(B.detect_displacement_rolls(tr)) for tr in trs)
        per_cfg[ck] = row
        log(f"  {ck[:16]:<16} " + " ".join(f"{row[c['tag']]:>9}" for c in F7.CONFIGS))

    log("\n  non-monotone rows (a repaired config with MORE events than `pre`):")
    any_nm = False
    for ck, row in per_cfg.items():
        worse = [t for t, v in row.items() if t != "pre" and v > row["pre"]]
        if worse:
            any_nm = True
            log(f"    {ck:<20} pre {row['pre']} -> {[(t, row[t]) for t in worse]}")
    if not any_nm:
        log("    none")
    F7.apply_cfg(F7.CONFIGS[0])

    # ---------------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("VERDICT — against the pre-registration")
    log("=" * 118)
    log(f"  SHIPPED D1 events across all policies: {n_events}; of those, {n_lethal} are followed by a lethal tick.")
    if n_events == 0:
        log("  no event to autopsy.")
    elif n_lethal == 0:
        log("  A2 CONFIRMED — every surviving D1 event is BFM-shaped motion that converts NOTHING.")
        log("     => report as 'policy-side D1 is nonzero and still buys no lethality', which is another")
        log("        instance of Direction B in [[bfm-and-kill-are-independent]]. Do NOT report it as")
        log("        'the policy flies BFM'. The repaired D1 still has NO positive control (A3).")
    else:
        log("  A1 TERRITORY — at least one event is followed by a lethal tick. Read the geometry columns")
        log("     before believing it: a chase event whose range is already inside LETHAL_RANGE_M scores")
        log("     a lethal tick whatever the lift vector did (the D4 range artifact, §9). Stratify by")
        log("     rngMin before any claim.")

    out = os.path.join(OUT, f"f8b_d1_autopsy_{args.tag}.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
