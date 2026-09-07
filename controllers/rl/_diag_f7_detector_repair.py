"""F7 — DOES THE D1 REPAIR ACTUALLY REPAIR D1?  (the open prerequisite for every merge BFM box)

F2 (2026-09-06) ran the detector's first positive control and the answer was worse than "too strict":
**D1 = 8/8 on FOUR merge arms that each deal exactly 0.0 HP** -- one of which fires 0 rounds and dives
2.4 km -- and **0/8 on the only arm that kills**.  On the merge cell the instrument is *anti*-correlated
with lethality, because two clauses are handed to it free by the geometry of a head-on pass:

  * (d) "rolled back in" never required the bandit to still be IN FRONT (the four arms exit at
    ata 139-171 deg), and it accepted the `chi_valid` SENTINEL (`chi = 0.0` for "dead ahead" satisfies
    `|chi| < CHI_RETURN_DEG`);
  * (f) "aspect improved by >= 15 deg" is automatic on any head-on pass, since aspect is measured off
    the TARGET's tail.

`docs/bfm-program-handoff.md` therefore recorded a four-step repair and the standing ban: *"until 1-3
land, do not run any box whose success criterion is a D1 count on a merge or head-on cell."*  Steps 1-4
are now in `bfm_detect` behind four switches.  This box prices them.

WHAT IT DOES

Re-flies F2's exact 15 arms at F2's exact cells and seeds (the same `_diag_f2_poscontrol.arms` list, so
this cannot drift from the box it is auditing), **persists every trace to
`runs/f7/traces.npz`**, then re-scores the SAME traces under eight detector configurations.  The
persistence is the durable half: `bfm_detect` is documented as a pure function of a per-tick trace, but
no trace has ever been saved, so every past detector question cost a re-fly.  After this run,
re-scoring is free.

  pre      the historical instrument, all four switches off.  Its counts MUST reproduce F2's recorded
           8/8 on the four merge arms -- that is this box's own harness self-check, and if it does not
           reproduce, nothing below is interpretable.
  r1/r2/r3/r4  each step ALONE, against `pre`.  Every column is therefore a true marginal price.
           ⚠ The first draft of this box put r1/r2/r3 on top of r4, and the smoke showed why that is
           wrong: repair 4 is not the pure loosening its -10 m/s floor makes it look like.  It swaps
           the STATISTIC (mean over the hold, not the entry tick), and `merge:lead_turn`'s pass enters
           at +253 m/s and exits at -348, so **repair 4 alone already kills it**.  Stacked on r4, the
           r1/r2/r3 columns would all have read 0 and attributed nothing.
  r123     the three narrowing steps without repair 4, i.e. the repair the handoff doc actually
           demanded ("until 1-3 land...").
  SHIPPED  r1 + r2 + r3 + r4 == the defaults now in `bfm_detect` (asserted, not assumed).
  both     SHIPPED but with (f) requiring aspect AND ata.  Reported to show what the convenient choice
           would have bought, and NOT shipped: a carrier picked because it preserves a previous answer
           is tuning, not repair.

PRE-REGISTERED PREDICTIONS (written before the run):

  Q1  The four 0.0-HP merge arms (`merge:lead_turn`, `merge:coach`, `merge:FLYBY(d1exists)`,
      `merge:d3b_perch0`) go from **D1 = 8/8 to D1 = 0/8** under SHIPPED.  If any of them survives, the
      repair is incomplete and the ban stays in force.
  Q2  **Repair 1 ALONE is sufficient for all four.**  The fly-by dies on the `chi_valid` sentinel and the
      other three die on the ata bound (they exit at 139-171 deg vs the 120 deg bar).  So `r1` should
      already read 0/8 on all four, and r2/r3 are redundant-but-defensive ON THIS CELL.  If instead the
      four only fall under r2 or r3, my account of WHY F2 failed was wrong and must be corrected.
  Q3  `merge:d3b_perch2000` -- the only arm that kills -- stays at **D1 = 0, unchanged in every
      config**.  F2's attribution says its 14 candidates die 11 on `hold_s` and 3 on (a) lookback, both
      upstream of every repaired clause, so no configuration should move it.  A change here means a
      repair step reached somewhere it was not supposed to.
  Q4  THE RISK.  The chase arms' recorded sole rejection is **(f) at d_aspect -6.5 deg vs the -15 deg
      bar**, i.e. they reach (f) and fail it by 8.5 deg.  Swapping the carrier to ata could hand those
      holds a pass and CREATE chase BFM out of the very same flights.  I predict it does not, but I am
      not confident, so the threshold is pre-registered: **if any chase arm goes from 0/8 to >= 4/8
      under SHIPPED, repair 3 must be reported as a LOOSENING, the `both` carrier reconsidered, and
      every "zero BFM on the chase" reading re-derived.**  Between 1 and 3 is reported as a partial and
      resolved by reading the per-event rows, not by moving a threshold.

SCOPE.  This measures the repair against HAND-CODED arms only.  The policy-side chase readings
(c95 D1 = D2 = D3 = 0.00, D2d's three arms) are NOT re-derived here -- they need a checkpoint and the
GPU, which is busy.  They are however predicted to be robust to the repair for a stated reason: those
policies were recorded as failing at the HOLD stage (clauses a/b), upstream of everything repaired, so
no carrier or exit change can resurrect them.  That prediction is untested and must be labelled as such
until a GPU box confirms it.

Usage (inside the rl container):
    python /workspace/controllers/rl/_diag_f7_detector_repair.py            # fly + persist + score
    python /workspace/controllers/rl/_diag_f7_detector_repair.py --rescore  # score saved traces only
"""

from __future__ import annotations

import os
import sys

import _diag_f2_poscontrol as F
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

OUT = "/workspace/controllers/rl/runs/f7"
TRACES = f"{OUT}/traces.npz"
SEEDS = F.SEEDS  # the D3b search seeds, so every row is directly comparable to F2's

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


# =================================================================================================
# Trace persistence — the durable half of this box
# =================================================================================================
# Field order is FROZEN. `bfm_detect.Sample` gaining a field would silently shift every column, so
# the loader asserts the width it read against the width it expects and refuses a stale file rather
# than reconstructing garbage.
FIELDS = (
    "t",
    "chi",
    "chi_valid",
    "gamma",
    "nz",
    "phi",
    "alt",
    "ail",
    "rng",
    "ata",
    "ata_az",
    "aspect",
    "closure",
    "cpa_miss",
    "es_rel",
    "lay_miss_m",
    "lethal",
)
_BOOL = {"chi_valid", "lethal"}


def trace_to_array(tr: list[B.Sample]) -> np.ndarray:
    return np.asarray(
        [[float(getattr(s, f)) for f in FIELDS] for s in tr], np.float64
    ).reshape(len(tr), len(FIELDS))


def array_to_trace(a: np.ndarray) -> list[B.Sample]:
    assert a.ndim == 2 and a.shape[1] == len(FIELDS), (
        f"saved trace is {a.shape[1] if a.ndim == 2 else a.shape} wide, expected {len(FIELDS)} "
        f"({FIELDS}) — `Sample` changed since the file was written; re-fly, do not re-score"
    )
    out = []
    for row in a:
        kw = {f: (bool(row[i]) if f in _BOOL else float(row[i])) for i, f in enumerate(FIELDS)}
        out.append(B.Sample(**kw))
    return out


def save_all(store: dict) -> None:
    """`store[key] = (trace_array, outcome_vector)`. One npz, keys `tr::<arm>::<seed>` / `oc::...`."""
    flat = {}
    for (name, seed), (arr, oc) in store.items():
        flat[f"tr::{name}::{seed}"] = arr.astype(np.float32)
        flat[f"oc::{name}::{seed}"] = np.asarray(oc, np.float64)
    np.savez_compressed(TRACES, fields=np.asarray(FIELDS), **flat)


OC_FIELDS = ("kill", "hp_dealt", "tleth_s", "rng_min", "dalt_m", "rounds", "dead", "steps")


def load_all() -> dict:
    z = np.load(TRACES, allow_pickle=False)
    got = tuple(str(x) for x in z["fields"])
    assert got == FIELDS, f"saved field order {got} != {FIELDS} — re-fly"
    store = {}
    for k in z.files:
        if not k.startswith("tr::"):
            continue
        _, name, seed = k.split("::")
        store[(name, int(seed))] = (z[k], z[f"oc::{name}::{seed}"])
    return store


# =================================================================================================
# Configurations
# =================================================================================================
# Each of r1..r4 differs from `pre` in EXACTLY ONE switch, so its column is a marginal price. See the
# docstring for why they are not stacked on r4.
CONFIGS = (
    dict(tag="pre", fwd=False, nonopen=False, carrier="aspect", clo_hold=False),
    dict(tag="r1", fwd=True, nonopen=False, carrier="aspect", clo_hold=False),
    dict(tag="r2", fwd=False, nonopen=True, carrier="aspect", clo_hold=False),
    dict(tag="r3", fwd=False, nonopen=False, carrier="ata", clo_hold=False),
    dict(tag="r4", fwd=False, nonopen=False, carrier="aspect", clo_hold=True),
    dict(tag="r123", fwd=True, nonopen=True, carrier="ata", clo_hold=False),
    dict(tag="SHIPPED", fwd=True, nonopen=True, carrier="ata", clo_hold=True),
    dict(tag="both", fwd=True, nonopen=True, carrier="both", clo_hold=True),
)


def apply_cfg(cfg: dict) -> None:
    B.D1_REQUIRE_FORWARD_EXIT = cfg["fwd"]
    B.D1_REQUIRE_NONOPENING = cfg["nonopen"]
    B.D1_ANGLE_CARRIER = cfg["carrier"]
    B.D1_CLOSURE_OVER_HOLD = cfg["clo_hold"]


def shipped_defaults() -> dict:
    """Read the module's OWN defaults so the `SHIPPED` row cannot silently diverge from them."""
    return dict(
        tag="module-defaults",
        fwd=B.D1_REQUIRE_FORWARD_EXIT,
        nonopen=B.D1_REQUIRE_NONOPENING,
        carrier=B.D1_ANGLE_CARRIER,
        clo_hold=B.D1_CLOSURE_OVER_HOLD,
    )


# =================================================================================================
# Fly
# =================================================================================================
def fly_all() -> dict:
    d3b = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"]
    flyby = np.array(
        [1.0, 1.0, 0.2, 0.582792, 1.0, 0.2, 1.0, 1.0, 0.2, 0.872769, 1.0, 0.2,
         -0.007301, 1.0, 0.2, 0.312102, 1.0, 0.2],
        float,
    )  # fmt: skip
    store: dict = {}
    for arm in F.arms(d3b, flyby):
        for sd in SEEDS:
            r, _st = F.fly(arm, sd)
            store[(arm["name"], sd)] = (
                trace_to_array(r.trace),
                [r.kill, r.hp_dealt, r.tleth_s, r.rng_min, r.dalt_m, r.rounds, int(r.dead), r.steps],
            )
        log(f"    flew {arm['name']:<26} {len(SEEDS)} seeds")
    return store


# =================================================================================================
# Report
# =================================================================================================
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rescore = "--rescore" in sys.argv
    log("#" * 118)
    log("#  F7 — does the D1 repair repair D1?  (steps 1-4; see docstring for Q1-Q4)")
    log("#" * 118)

    if rescore and os.path.exists(TRACES):
        log(f"\n  RESCORE ONLY — reading {TRACES} (no sim)")
        store = load_all()
    else:
        log(f"\n  flying F2's 15 arms x {len(SEEDS)} seeds, dt {M.CONTROL_DT_S * 1000:.3f} ms")
        store = fly_all()
        save_all(store)
        log(f"  persisted {len(store)} traces -> {TRACES}")

    names = []
    for name, _sd in store:
        if name not in names:
            names.append(name)
    traces = {k: array_to_trace(v[0]) for k, v in store.items()}
    outc = {k: dict(zip(OC_FIELDS, v[1])) for k, v in store.items()}

    # The `SHIPPED` column is only meaningful if it IS the module's defaults. Asserted, not asserted
    # in prose: a default drifting away from the audited configuration is exactly the failure that
    # makes a recorded count untraceable to the code that produced it.
    defaults = shipped_defaults()
    ship = next(c for c in CONFIGS if c["tag"] == "SHIPPED")
    mismatch = {k: (ship[k], defaults[k]) for k in ("fwd", "nonopen", "carrier", "clo_hold") if ship[k] != defaults[k]}
    assert not mismatch, f"SHIPPED config != bfm_detect defaults: {mismatch}"
    log(
        f"\n  module defaults: fwd={defaults['fwd']} nonopen={defaults['nonopen']} "
        f"carrier={defaults['carrier']!r} clo_hold={defaults['clo_hold']}"
        f"  (asserted == the SHIPPED column)"
        f"\n  bars: ATA_EXIT_MAX {B.D1_EXIT_ATA_MAX_DEG:.0f} deg  ATA_GAIN {B.ATA_GAIN_DEG:.0f} deg  "
        f"ASPECT_GAIN {B.ASPECT_GAIN_DEG:.0f} deg  CLOSURE_TOL {B.CLOSURE_TOL_MS:.0f} m/s"
    )

    # ---- TABLE 1: D1 per arm under every configuration -----------------------------------------
    log("\n" + "=" * 118)
    log("TABLE 1 — D1 count per arm (pooled over 8 seeds) under each repair configuration")
    log("  `pre` is the HISTORICAL instrument: it must reproduce F2's 8/8 on the four merge arms.")
    log("=" * 118)
    hdr = f"  {'arm':<26} {'kills':>5} {'hp':>6} " + " ".join(f"{c['tag']:>8}" for c in CONFIGS)
    log(hdr)
    counts: dict[tuple[str, str], int] = {}
    for cfg in CONFIGS:
        apply_cfg(cfg)
        for name in names:
            counts[(name, cfg["tag"])] = sum(
                len(B.detect_displacement_rolls(traces[(name, sd)])) for sd in SEEDS
            )
    for name in names:
        k = sum(int(outc[(name, sd)]["kill"]) for sd in SEEDS)
        hp = float(np.mean([outc[(name, sd)]["hp_dealt"] for sd in SEEDS]))
        log(
            f"  {name:<26} {k:>5} {hp:>6.1f} "
            + " ".join(f"{counts[(name, c['tag'])]:>8}" for c in CONFIGS)
        )

    # ---- TABLE 2: per-event autopsy of the pre-repair merge events ------------------------------
    log("\n" + "=" * 118)
    log("TABLE 2 — every event the HISTORICAL instrument reported on the merge arms, and which")
    log("  repair step kills it. Pool scored under `pre`, because `r4` is NOT a superset of `pre`")
    log("  (it swaps the (c) statistic and can reject as well as admit).")
    log("  r1 = exit chi_valid AND ata <= 120 deg | r2 = exit closure > -10 m/s |")
    log("  r3 = d_ata <= -15 deg | r4 = mean closure over the hold > -10 m/s.")
    log("  'KEPT' = survives all four, i.e. a real event by the repaired bar.")
    log("=" * 118)
    apply_cfg(CONFIGS[0])  # `pre`
    for name in names:
        if not name.startswith("merge"):
            continue
        rows = []
        for sd in SEEDS:
            for ev in B.detect_displacement_rolls(traces[(name, sd)]):
                d = ev.detail
                rows.append(
                    (
                        sd,
                        ev,
                        d["exit_chi_valid"] and d["ata_exit_deg"] <= B.D1_EXIT_ATA_MAX_DEG,
                        d["closure_exit"] > -B.CLOSURE_TOL_MS,
                        d["d_ata_deg"] <= -B.ATA_GAIN_DEG,
                        d["closure_hold"] > -B.CLOSURE_TOL_MS,
                    )
                )
        if not rows:
            log(f"\n  {name}: the historical instrument reported NO events here")
            continue
        log(f"\n  {name}  ({len(rows)} events over {len(SEEDS)} seeds)")
        log(
            f"    {'sd':>4} {'t0':>6} {'hold':>6} {'chiPk':>6} {'nzPk':>5} {'ataIn':>6} {'ataEx':>6} "
            f"{'dAta':>7} {'dAsp':>7} {'cloEx':>7} {'cloHld':>7} {'valid':>6}   "
            f"{'r1':>3} {'r2':>3} {'r3':>3} {'r4':>3}  verdict"
        )
        for sd, ev, r1_ok, r2_ok, r3_ok, r4_ok in rows[:12]:
            d = ev.detail
            flags = (("r1", r1_ok), ("r2", r2_ok), ("r3", r3_ok), ("r4", r4_ok))
            kills = [t for t, ok in flags if not ok]
            log(
                f"    {sd:>4} {ev.t0:>6.2f} {d['hold_s']:>6.3f} {d['chi_peak_deg']:>6.1f} "
                f"{d['nz_peak']:>5.2f} {d['ata_entry_deg']:>6.1f} {d['ata_exit_deg']:>6.1f} "
                f"{d['d_ata_deg']:>+7.1f} {ev.d_aspect_deg:>+7.1f} {d['closure_exit']:>+7.0f} "
                f"{d['closure_hold']:>+7.0f} {str(d['exit_chi_valid']):>6}   "
                + " ".join(f"{'ok' if ok else 'NO':>3}" for _t, ok in flags)
                + "  "
                + ("KEPT" if not kills else "killed by " + "+".join(kills))
            )
        if len(rows) > 12:
            log(f"    ... {len(rows) - 12} more (same structure; all seeds shown in TABLE 1 counts)")

    # ---- TABLE 3: the chase risk (Q4) ----------------------------------------------------------
    log("\n" + "=" * 118)
    log("TABLE 3 — Q4, the chase risk: does the ata carrier CREATE chase events?")
    log("  Pre-registered: any chase arm going 0/8 -> >=4/8 under SHIPPED means repair 3 LOOSENS.")
    log("=" * 118)
    log(
        f"  {'arm':<26} {'pre':>5} {'r3':>4} {'SHIPPED':>8} {'both':>6}   "
        f"{'best d_aspect':>14} {'best d_ata':>11}  (over `pre` candidates)"
    )
    apply_cfg(CONFIGS[0])  # `pre`
    for name in names:
        if not name.startswith("chase"):
            continue
        d_asp, d_ata = [], []
        for sd in SEEDS:
            for ev in B.detect_displacement_rolls(traces[(name, sd)]):
                d_asp.append(ev.d_aspect_deg)
                d_ata.append(ev.detail["d_ata_deg"])
        ba = f"{min(d_asp):+.1f}" if d_asp else "no cand"
        bt = f"{min(d_ata):+.1f}" if d_ata else "no cand"
        log(
            f"  {name:<26} {counts[(name, 'pre')]:>5} {counts[(name, 'r3')]:>4} "
            f"{counts[(name, 'SHIPPED')]:>8} {counts[(name, 'both')]:>6}   {ba:>14} {bt:>11}"
        )
    log(
        "\n  NOTE: these columns are D1 counts on the PASSING side of (f) only. A chase arm reading 0"
        "\n  in every column may be failing upstream (hold_s / lookback) and is NOT evidence that the"
        "\n  carrier is safe — read the 'no cand' marks, which mean nothing even reached (f)."
    )

    # ---- VERDICT --------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("VERDICT — against the pre-registered Q1-Q4")
    log("=" * 118)
    four = [
        "merge:lead_turn",
        "merge:coach",
        "merge:FLYBY(d1exists)",
        "merge:d3b_perch0",
    ]
    four = [n for n in four if n in names]
    log("  Q1  the four 0.0-HP merge arms, pre -> r123 (what the handoff doc demanded) -> SHIPPED:")
    for n in four:
        log(
            f"        {n:<26} pre {counts[(n, 'pre')]:>3} -> r123 {counts[(n, 'r123')]:>3}"
            f" -> SHIPPED {counts[(n, 'SHIPPED')]:>3}"
            f"   {'PASS' if counts[(n, 'SHIPPED')] == 0 else 'FAIL (survives the repair)'}"
        )
    q1 = all(counts[(n, "SHIPPED")] == 0 for n in four)
    q1b = all(counts[(n, "r123")] == 0 for n in four)
    log(f"      => Q1 {'CONFIRMED' if q1 else 'REFUTED — the ban STAYS IN FORCE'}")
    log(
        "         and steps 1-3 alone (no repair 4) are "
        + ("SUFFICIENT" if q1b else "NOT sufficient — repair 4 is load-bearing, which the doc did not anticipate")
    )
    log("  Q2  is repair 1 ALONE sufficient on those four?  (each column is one switch vs `pre`)")
    for n in four:
        log(
            f"        {n:<26} r1 {counts[(n, 'r1')]:>3}  r2 {counts[(n, 'r2')]:>3}  "
            f"r3 {counts[(n, 'r3')]:>3}  r4 {counts[(n, 'r4')]:>3}"
        )
    q2 = all(counts[(n, "r1")] == 0 for n in four)
    log(
        f"      => Q2 {'CONFIRMED (r1 alone reads 0 on all four; r2/r3/r4 are defensive on this cell)' if q2 else 'REFUTED — my account of WHY F2 failed was incomplete; read TABLE 2'}"
    )
    kn = "merge:d3b_perch2000"
    if kn in names:
        vals = {c["tag"]: counts[(kn, c["tag"])] for c in CONFIGS}
        log(f"  Q3  the KILLER arm {kn}: {vals}")
        log(
            f"      => Q3 {'CONFIRMED (0 in every config)' if set(vals.values()) == {0} else 'REFUTED — a repair step reached an arm it should not have'}"
        )
    log("  Q4  chase arms, pre -> SHIPPED (threshold: 0 -> >=4 means repair 3 loosens):")
    worst = 0
    for n in names:
        if not n.startswith("chase"):
            continue
        a, b = counts[(n, "pre")], counts[(n, "SHIPPED")]
        worst = max(worst, b - a)
        if a != b:
            log(f"        {n:<26} {a:>3} -> {b:>3}  MOVED")
    log(
        f"      => Q4 largest increase {worst:+d}: "
        + (
            "CONFIRMED (carrier does not create chase events)"
            if worst < 4
            else "REFUTED — repair 3 is a LOOSENING; re-derive every zero-BFM chase reading"
        )
    )
    log("")
    log("  ⚠ Not tested here: the POLICY-side chase readings (c95, D2d). They need a checkpoint and")
    log("    the GPU. Predicted robust because those policies fail at the HOLD stage, upstream of")
    log("    every repaired clause — but that is a PREDICTION, and must be labelled as one.")

    path = f"{OUT}/f7_detector_repair.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
