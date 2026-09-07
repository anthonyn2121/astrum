"""Independent recomputation of the D3 (turning_room) verdict on seed 28.

Pure stdlib. Re-implements `bfm_detect.detect_turning_room` clause-for-clause against the
`smp_*` columns of the replay dump (the raw `B.make_sample` field set), so the events found
here are the events `B.detect_all` sees. Constants are copied from bfm_detect.py:79-92.
"""

import csv
import math
import sys

CHI_ROOM_DEG = 45.0
CHI_MIN_HOLD_S = 0.5
CHI_RETURN_S = 4.0
NZ_LOADED = 3.0

PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_nokill_seed28.csv")
)

with open(PATH) as fh:
    rows = list(csv.DictReader(fh))
print(f"file={PATH}  rows={len(rows)}")


def f(r, k):
    return float(r[k])


# --- rebuild the trace the detector reads -------------------------------------------------
tr = [
    dict(
        step=int(r["step"]),
        t=f(r, "smp_t"),
        chi=f(r, "smp_chi_rad"),
        chi_valid=f(r, "smp_chi_valid") > 0.5,
        nz=f(r, "smp_nz"),
        ata_az=f(r, "smp_ata_az"),
        cpa_miss=f(r, "smp_cpa_miss"),
        rng=f(r, "smp_rng"),
        aspect=f(r, "smp_aspect"),
        alt=f(r, "smp_alt"),
        es_rel=f(r, "smp_es_rel"),
        lethal=f(r, "smp_lethal") > 0.5,
        alt_msl=f(r, "alt_m"),
    )
    for r in rows
]


def runs(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


hold = [(abs(s["chi"]) > math.radians(CHI_ROOM_DEG) and s["nz"] > NZ_LOADED) for s in tr]
allruns = runs(hold)
print(f"\ntotal (|chi|>45 & nz>3) runs: {len(allruns)}")

print("\n=== every candidate hold, with each clause verdict ===")
print(
    f"{'i':>4} {'j':>4} {'t0':>7} {'t1':>7} {'hold_s':>6} "
    f"{'dur_ok':>6} {'1side':>5} {'cpa_i':>8} {'cpa_j':>8} {'build_m':>9} {'build_%':>8} "
    f"{'built':>5} {'cashed':>6} {'exit':>5} {'cpa_exit':>9}"
)
passing = []
for i, j in allruns:
    hold_s = tr[j]["t"] - tr[i]["t"]
    dur_ok = hold_s >= CHI_MIN_HOLD_S
    sides = {1 if tr[k]["ata_az"] >= 0.0 else -1 for k in range(i, j + 1)}
    one_side = len(sides) == 1
    cpa_i, cpa_j = tr[i]["cpa_miss"], tr[j]["cpa_miss"]
    build = cpa_j - cpa_i
    built = cpa_j > cpa_i
    after = [k for k in range(j + 1, len(tr)) if tr[k]["t"] - tr[j]["t"] <= CHI_RETURN_S]
    cashed = bool(after) and min(tr[k]["cpa_miss"] for k in after) < cpa_j
    exit_idx = min(after, key=lambda k: tr[k]["cpa_miss"]) if after else -1
    cpa_exit = tr[exit_idx]["cpa_miss"] if exit_idx >= 0 else float("nan")
    print(
        f"{i:>4} {j:>4} {tr[i]['t']:>7.3f} {tr[j]['t']:>7.3f} {hold_s:>6.3f} "
        f"{str(dur_ok):>6} {str(one_side):>5} {cpa_i:>8.1f} {cpa_j:>8.1f} {build:>+9.1f} "
        f"{100.0 * build / cpa_i:>+8.2f} {str(built):>5} {str(cashed):>6} {exit_idx:>5} {cpa_exit:>9.1f}"
    )
    if dur_ok and one_side and built and cashed:
        passing.append((i, j, exit_idx))

print(f"\nPASSING D3 events: {passing}")

# --- price every passing event -------------------------------------------------------------
for i, j, e in passing:
    print(f"\n=== EVENT i0={i} j(hold end)={j} i1(exit)={e} ===")
    print(f"  hold_s (tr[j].t-tr[i].t) = {tr[j]['t'] - tr[i]['t']:.3f} s   ticks={j - i + 1}")
    print(f"  event span t {tr[i]['t']:.3f} -> {tr[e]['t']:.3f} s")
    print(
        f"  cpa entry {tr[i]['cpa_miss']:.1f} -> peak {tr[j]['cpa_miss']:.1f} -> exit {tr[e]['cpa_miss']:.1f}"
    )
    print(
        f"  BUILD = {tr[j]['cpa_miss'] - tr[i]['cpa_miss']:+.1f} m "
        f"({100.0 * (tr[j]['cpa_miss'] - tr[i]['cpa_miss']) / tr[i]['cpa_miss']:+.2f}% of entry)"
    )
    print(f"  CASH  = {tr[e]['cpa_miss'] - tr[j]['cpa_miss']:+.1f} m")
    for lab, a, b in (("i0->i1(exit)", i, e), ("i0->j(hold end)", i, j)):
        print(
            f"  priced {lab}: rng {tr[a]['rng']:.0f} -> {tr[b]['rng']:.0f} m "
            f"(d_rng {tr[b]['rng'] - tr[a]['rng']:+.0f})  "
            f"alt_enu {tr[a]['alt']:.0f} -> {tr[b]['alt']:.0f} m ({tr[b]['alt'] - tr[a]['alt']:+.0f})  "
            f"alt_msl {tr[a]['alt_msl']:.0f} -> {tr[b]['alt_msl']:.0f} ({tr[b]['alt_msl'] - tr[a]['alt_msl']:+.0f})  "
            f"aspect {math.degrees(tr[a]['aspect']):.1f} -> {math.degrees(tr[b]['aspect']):.1f} deg "
            f"(d {math.degrees(tr[b]['aspect'] - tr[a]['aspect']):+.1f})  "
            f"es_rel {tr[a]['es_rel']:.0f} -> {tr[b]['es_rel']:.0f} ({tr[b]['es_rel'] - tr[a]['es_rel']:+.0f})"
        )
    print(f"  lethal ticks inside hold  [{i},{j}]: {sum(tr[k]['lethal'] for k in range(i, j + 1))}")
    print(f"  lethal ticks inside event [{i},{e}]: {sum(tr[k]['lethal'] for k in range(i, e + 1))}")
    print(
        f"  nz over hold: {min(tr[k]['nz'] for k in range(i, j + 1)):.2f} .. "
        f"{max(tr[k]['nz'] for k in range(i, j + 1)):.2f} g"
    )
    print(
        f"  |chi| over hold: {min(abs(math.degrees(tr[k]['chi'])) for k in range(i, j + 1)):.1f} .. "
        f"{max(abs(math.degrees(tr[k]['chi'])) for k in range(i, j + 1)):.1f} deg"
    )

# --- is the event during EGRESS? range/closure context ------------------------------------
print("\n=== range & closure context, whole trace (every 2.0 s) ===")
print(f"{'step':>4} {'t':>7} {'rng':>7} {'closure':>8} {'cpa':>8} {'aspect':>7} {'alt_enu':>8}")
for k in range(0, len(tr), 48):
    print(
        f"{tr[k]['step']:>4} {tr[k]['t']:>7.3f} {tr[k]['rng']:>7.0f} "
        f"{float(rows[k]['closure_ms']):>8.1f} {tr[k]['cpa_miss']:>8.1f} "
        f"{math.degrees(tr[k]['aspect']):>7.1f} {tr[k]['alt']:>8.0f}"
    )
k = len(tr) - 1
print(
    f"{tr[k]['step']:>4} {tr[k]['t']:>7.3f} {tr[k]['rng']:>7.0f} "
    f"{float(rows[k]['closure_ms']):>8.1f} {tr[k]['cpa_miss']:>8.1f} "
    f"{math.degrees(tr[k]['aspect']):>7.1f} {tr[k]['alt']:>8.0f}"
)

print("\n=== rng min/max and where ===")
kmin = min(range(len(tr)), key=lambda k: tr[k]["rng"])
kmax = max(range(len(tr)), key=lambda k: tr[k]["rng"])
print(f"  rng min {tr[kmin]['rng']:.0f} m at step {kmin} t {tr[kmin]['t']:.3f}")
print(f"  rng max {tr[kmax]['rng']:.0f} m at step {kmax} t {tr[kmax]['t']:.3f}")
print(f"  lethal ticks whole trace: {sum(s['lethal'] for s in tr)}")
print(f"  fire ticks whole trace: {sum(1 for r in rows if float(r['fire']) > 0.5)}")
print(
    f"  opp_health first {rows[0]['opp_health']} last {rows[-1]['opp_health']}; "
    f"own health first {rows[0]['health']} last {rows[-1]['health']}"
)
