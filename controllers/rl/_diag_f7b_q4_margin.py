"""F7b — the MARGIN behind Q4, scored from F7's saved traces with NO SIM.

F7's TABLE 3 was supposed to answer "how close did the chase come to producing an ata-carried
event", and it could not: its `best d_aspect / best d_ata` columns are computed over
`detect_displacement_rolls` RETURNS under `pre`, i.e. over events that already passed (f).  On the
chase that set is empty, so every cell printed `no cand` — which merely restates that the `pre`
column is 0.  The script said so in its own NOTE, but the VERDICT line still read "Q4 CONFIRMED",
which is stronger than that table can support on its own.

What IS sound in F7: TABLE 1's `r3` column runs the FULL detector with `carrier='ata'` and nothing
else changed, and reads 0 on all ten chase arms.  That is a direct measurement, not a diagnostic —
repair 3 creates no chase events.  So the verdict stands; only the margin was missing.

This box supplies the margin, and it is the first use of the trace persistence F7 added: 120 traces
already on disk, so the whole question costs seconds of CPU and no JSBSim.

For every candidate hold on every arm it records the FIRST failing clause of the PRE-repair
detector (F2's `d1_attribute`, which mirrors it clause-for-clause), and for any candidate that
reaches (f) it prints BOTH currencies — d_aspect and d_ata — so the reader can see directly whether
swapping the carrier would have admitted it.  Two things it can settle that F7 could not:

  M1  Does any chase candidate reach (f) at all?  [[bfm-detector-spec]] records "the chase readings
      survive, sole rejection is (f) at -6.5 deg vs -15 deg" — if nothing reaches (f) here, that
      recorded phrasing is about a DIFFERENT pool (F2's attribution walk, which reports the
      furthest-progressing clause per candidate) and must not be read as "a chase event nearly fired".
  M2  For candidates that do reach (f), is d_ata MORE or LESS permissive than d_aspect?  If d_ata is
      systematically more negative on the chase, repair 3 is a loosening that happens to be masked
      by an upstream clause on these particular arms -- which would make Q4's pass FRAGILE rather
      than structural, and is exactly the distinction the pre-registration cared about.
"""

from __future__ import annotations

import math
import os
from collections import Counter

import _diag_f2_poscontrol as F
import _diag_f7_detector_repair as F7
import numpy as np
from astrum_dogfight import bfm_detect as B

NPZ = "/workspace/controllers/rl/runs/f7/traces.npz"
OUT = "/workspace/controllers/rl/runs/f7"
CFG_PRE = F7.CONFIGS[0]
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _deg(r: float) -> float:
    return math.degrees(r)


def candidates(tr) -> list[dict]:
    """F2's attribution, extended with d_ata wherever the walk got far enough to have an exit."""
    recs = F.d1_attribute(tr)
    for rec in recs:
        if "exit" in rec:
            k, i = rec["exit"], rec["i0"]
            rec["d_ata"] = _deg(tr[k].ata - tr[i].ata)
            rec["ata_in"] = _deg(tr[i].ata)
            rec["ata_ex"] = _deg(tr[k].ata)
            rec["clo_ex"] = tr[k].closure
    return recs


def main() -> None:
    z = np.load(NPZ)
    fields = tuple(str(x) for x in z["fields"])
    assert fields == F7.FIELDS, f"trace field order changed: {fields} != {F7.FIELDS}"
    keys = [k for k in z.files if k.startswith("tr::")]
    traces = {}
    for k in keys:
        _, arm, sd = k.split("::")
        traces[(arm, int(sd))] = F7.array_to_trace(z[k])
    arms = sorted({a for a, _ in traces}, key=lambda a: (not a.startswith("chase"), a))
    log("#" * 118)
    log("#  F7b — the margin behind Q4: which clause stops each candidate, and in WHICH currency")
    log("#" * 118)
    log(f"\n  rescored {len(traces)} saved traces, no sim   ({NPZ})")
    log(
        f"  bars: ASPECT_GAIN {B.ASPECT_GAIN_DEG:.0f} deg   ATA_GAIN {B.ATA_GAIN_DEG:.0f} deg"
        f"   (an event needs d_aspect <= -{B.ASPECT_GAIN_DEG:.0f} or d_ata <= -{B.ATA_GAIN_DEG:.0f})"
    )

    log("\n" + "=" * 118)
    log("TABLE A — first failing clause of the PRE-repair detector, over ALL candidate holds")
    log("=" * 118)
    order = ["hold_s", "(a) lookback", "(c) closing", "(d) rolled back", "(e) same side", "(f) angles", None]
    hdr = ["hold_s", "(a)look", "(c)clos", "(d)roll", "(e)side", "(f)ang", "PASSED"]
    log(f"  {'arm':<26} {'cands':>6} " + " ".join(f"{h:>8}" for h in hdr))
    reach_f: dict[str, list[dict]] = {}
    for arm in arms:
        c: Counter = Counter()
        tot = 0
        for sd in F7.SEEDS:
            for rec in candidates(traces[(arm, sd)]):
                c[rec["fail"]] += 1
                tot += 1
                if rec["fail"] in ("(f) angles", None):
                    reach_f.setdefault(arm, []).append(rec)
        log(f"  {arm:<26} {tot:>6} " + " ".join(f"{c[k]:>8}" for k in order))

    log("\n" + "=" * 118)
    log("TABLE B — M1/M2: every candidate that REACHED (f), in both currencies")
    log("  `aspect ok` / `ata ok` = would that carrier admit it (d <= -GAIN)?")
    log("=" * 118)
    if not reach_f:
        log("  NOTHING on any arm reached (f).")
    for arm in arms:
        recs = reach_f.get(arm)
        if not recs:
            log(f"  {arm:<26} nothing reached (f)")
            continue
        log(f"\n  {arm}  ({len(recs)} candidates over 8 seeds)")
        log(
            f"      {'t0':>6} {'hold':>6} {'chiPk':>6} {'ataIn':>6} {'ataEx':>6} "
            f"{'dAsp':>8} {'dAta':>8} {'cloEx':>7}  {'aspect ok':>9} {'ata ok':>7}  clause"
        )
        for rec in recs[:8]:
            ok_a = "YES" if rec.get("d_aspect", 0.0) <= -B.ASPECT_GAIN_DEG else "no"
            ok_t = "YES" if rec.get("d_ata", 0.0) <= -B.ATA_GAIN_DEG else "no"
            log(
                f"      {rec['t0']:>6.2f} {rec['hold_s']:>6.3f} {rec['chi_peak']:>6.1f} "
                f"{rec.get('ata_in', float('nan')):>6.1f} {rec.get('ata_ex', float('nan')):>6.1f} "
                f"{rec.get('d_aspect', float('nan')):>+8.1f} {rec.get('d_ata', float('nan')):>+8.1f} "
                f"{rec.get('clo_ex', float('nan')):>+7.0f}  {ok_a:>9} {ok_t:>7}  "
                f"{rec['fail'] or 'PASSED'}"
            )
        if len(recs) > 8:
            log(f"      ... {len(recs) - 8} more (seeds are bit-identical on these arms)")

    log("\n" + "=" * 118)
    log("TABLE C — why repair 1 ALONE leaves lead_turn/coach at 8 (F7's Q2 refutation, MEASURED)")
    log("  Repair 1 is a filter INSIDE the exit search, not a rejection: the loop `continue`s past a")
    log("  disqualified exit and keeps scanning to `CHI_RETURN_S`. So on a trajectory that turns back")
    log("  toward the bandit it can RELOCATE the exit later instead of dropping the event -- adding a")
    log("  constraint inside a search is NOT monotone in the event count. Measured below.")
    log("=" * 118)
    log(
        f"  {'arm':<26} {'cfg':>4} {'n':>3} {'exit t':>7} {'ataEx':>7} {'dAta':>8} {'dAsp':>8} "
        f"{'cloEx':>7} {'valid':>6}"
    )
    for arm in [a for a in arms if not a.startswith("chase")]:
        for cfg in F7.CONFIGS:
            if cfg["tag"] not in ("pre", "r1", "r2", "r3"):
                continue
            F7.apply_cfg(cfg)
            tr = traces[(arm, F7.SEEDS[0])]
            evs = B.detect_displacement_rolls(tr)
            if not evs:
                log(f"  {arm:<26} {cfg['tag']:>4} {0:>3}   (no event)")
                continue
            e = evs[0]
            log(
                f"  {arm:<26} {cfg['tag']:>4} {len(evs):>3} {e.t1:>7.2f} "
                f"{e.detail['ata_exit_deg']:>7.1f} {e.detail['d_ata_deg']:>+8.1f} "
                f"{e.d_aspect_deg:>+8.1f} {e.detail['closure_exit']:>+7.0f} "
                f"{str(e.detail['exit_chi_valid']):>6}"
            )
    F7.apply_cfg(CFG_PRE)

    log("\n" + "=" * 118)
    log("VERDICT")
    log("=" * 118)
    chase_f = {a: v for a, v in reach_f.items() if a.startswith("chase")}
    if not chase_f:
        log("  M1  NO chase candidate reaches (f) on any of the ten arms.")
        log("      => F7's TABLE 3 'no cand' was literal, and the recorded phrasing")
        log("         'the sole chase rejection is (f) at -6.5 deg' describes F2's")
        log("         ATTRIBUTION walk, not a candidate that nearly fired. It must not")
        log("         be read as 'a chase event was one threshold away'.")
        log("  M2  vacuous on the chase: no candidate to compare currencies on.")
        log("      => Q4's pass is structural for a DIFFERENT reason than pre-registered:")
        log("         the carrier is unreachable on the chase because (f) is unreachable.")
    else:
        for a, v in chase_f.items():
            worse = sum(1 for r in v if r.get("d_ata", 0.0) < r.get("d_aspect", 0.0))
            log(f"  {a}: {len(v)} reached (f); d_ata more permissive on {worse}")
    m_f = {a: v for a, v in reach_f.items() if not a.startswith("chase")}
    log(f"\n  merge arms reaching (f): {sorted(m_f)} (context for the four repaired rows)")
    path = os.path.join(OUT, "f7b_q4_margin.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
