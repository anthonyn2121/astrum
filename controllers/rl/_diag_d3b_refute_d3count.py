"""REFUTATION AUDIT: is the prereg's pooled merge1500 D3 = 3 one event counted three times?

Independent recomputation. Nothing is read from a prior report; every number below is produced
by re-flying the saved genome in the true sim and re-running the detector.
"""

from __future__ import annotations

import math

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
N = 576

z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
g = np.asarray(z["genome"], float)
gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
cell = M.CELLS[str(z["cell"])]

print(
    f"cell={str(z['cell'])}  CHI_ROOM_DEG={B.CHI_ROOM_DEG}  NZ_LOADED={B.NZ_LOADED}  "
    f"CHI_MIN_HOLD_S={B.CHI_MIN_HOLD_S}  CHI_RETURN_S={B.CHI_RETURN_S}"
)
print(f"LETHAL_RANGE_M={B.LETHAL_RANGE_M}")

traces, rolls = {}, {}
for sd in SEEDS:
    env, base = M.build_env(cell, sd, N * M.CONTROL_DT_S)
    r = M.roll_once(env, base, lambda s, me, bd, rel: gm.act(g, s, N, me), N, keep=True)
    env.close()
    traces[sd] = r.trace
    rolls[sd] = r

print("\n=== (1) PER-SEED detect_all, ALL EIGHT seeds (full float repr on D3) ===")
print(f"{'seed':>5} {'kill':>4} {'steps':>5} {'D1':>3} {'D2':>3} {'D2c':>4} {'D3':>3}")
d3 = {}
for sd in SEEDS:
    d = B.detect_all(traces[sd])
    d3[sd] = B.detect_turning_room(traces[sd])
    assert len(d["turning_room"]) == len(d3[sd])
    print(
        f"{sd:>5} {rolls[sd].kill:>4} {rolls[sd].steps:>5} "
        f"{len(d['displacement_rolls']):>3} {len(d['yoyos']):>3} "
        f"{len(d['yoyos_clean']):>4} {len(d['turning_room']):>3}"
    )
print(f"POOLED D3 over 8 seeds = {sum(len(v) for v in d3.values())}   (prereg SEARCH row claims 3)")

print("\n=== (2) THE D3 EVENTS, full precision (repr, not %.2f) ===")
for sd in SEEDS:
    for e in d3[sd]:
        det = e.detail
        print(f"seed {sd}: i0={e.i0} i1={e.i1} t0={e.t0!r} t1={e.t1!r}")
        print(f"    cpa_entry_m={det['cpa_entry_m']!r}")
        print(f"    cpa_peak_m ={det['cpa_peak_m']!r}")
        print(f"    cpa_exit_m ={det['cpa_exit_m']!r}")
        print(f"    hold_s={det['hold_s']!r}  build_m={det['cpa_peak_m'] - det['cpa_entry_m']!r}")
        print(f"    d_aspect_deg={e.d_aspect_deg!r} d_rng_m={e.d_rng_m!r} d_es_m={e.d_es_m!r}")

print("\n=== (3) BITWISE cross-seed identity of the fields the detector READS ===")
FIELDS = [
    "t",
    "chi",
    "chi_valid",
    "nz",
    "ata_az",
    "cpa_miss",
    "alt",
    "rng",
    "aspect",
    "closure",
    "es_rel",
    "phi",
    "gamma",
    "ail",
    "lay_miss_m",
    "ata",
]
ref = 28
for sd in (79, 96):
    n = min(len(traces[ref]), len(traces[sd]))
    print(f"  seed {sd} vs {ref}: len {len(traces[sd])} vs {len(traces[ref])}")
    for f in FIELDS:
        a = np.array([float(getattr(s, f)) for s in traces[ref][:n]])
        b = np.array([float(getattr(s, f)) for s in traces[sd][:n]])
        d = np.max(np.abs(a - b))
        print(f"    max|d {f:<11}| = {d:.3e}{'   <-- NONZERO' if d != 0.0 else ''}")
print("  fields the detector NEVER reads, for contrast (hp dealt / rounds / event):")
for sd in (28, 79, 96):
    r = rolls[sd]
    print(
        f"    seed {sd}: hp_dealt={r.hp_dealt} hp_taken={r.hp_taken} rounds={r.rounds} "
        f"event={r.event} tleth={r.tleth_s:.3f}"
    )

print("\n=== (4) The five KILL seeds: WHY is D3=0? truncation or a different trajectory? ===")
for sd in (11, 45, 62, 113, 130):
    n = min(len(traces[sd]), len(traces[ref]))
    dmax = max(
        np.max(
            np.abs(
                np.array([float(getattr(s, f)) for s in traces[sd][:n]])
                - np.array([float(getattr(s, f)) for s in traces[ref][:n]])
            )
        )
        for f in FIELDS
    )
    print(
        f"  seed {sd}: steps={rolls[sd].steps:>4}  max|d| vs seed28 over the OVERLAP "
        f"({n} steps, all {len(FIELDS)} fields) = {dmax:.3e}   "
        f"D3 event i0=499 reached? {rolls[sd].steps > 499}"
    )

print("\n=== (5) PRICE THE HOLD. The mask run [i,j] that produced the event, on seed 28 ===")
tr = traces[ref]
hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
runs = B._runs(hold)
print(f"  total loaded-out-of-plane runs in the 576-step trace: {len(runs)}")
for i, j in runs:
    dur = tr[j].t - tr[i].t
    sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
    built = tr[j].cpa_miss > tr[i].cpa_miss
    print(
        f"    run [{i:>3},{j:>3}] t {tr[i].t:6.3f}-{tr[j].t:6.3f} dur {dur:5.3f}s  "
        f"dur>=0.5:{dur >= B.CHI_MIN_HOLD_S}  single_side:{len(sides) == 1}  "
        f"cpa_built:{built}  cpa {tr[i].cpa_miss:8.2f}->{tr[j].cpa_miss:8.2f}  "
        f"nz {min(tr[k].nz for k in range(i, j + 1)):5.2f}-{max(tr[k].nz for k in range(i, j + 1)):5.2f}  "
        f"alt {tr[i].alt:7.1f}->{tr[j].alt:7.1f} ({tr[j].alt - tr[i].alt:+.1f} m)  "
        f"Es_rel {tr[i].es_rel:+8.1f}->{tr[j].es_rel:+8.1f}"
    )

for sd, evs in d3.items():
    for e in evs:
        i0, i1 = e.i0, e.i1
        # recover j: the index where cpa_miss == cpa_peak within the run containing i0
        jj = [j for (i, j) in runs if i == i0]
        j = jj[0] if jj else None
        t = traces[sd]
        print(f"\n  seed {sd} EVENT PRICED (i0={i0}, j={j}, exit={i1}):")
        print(
            f"    alt      {t[i0].alt:9.2f} -> {t[j].alt:9.2f} -> {t[i1].alt:9.2f} m   "
            f"(entry->exit {t[i1].alt - t[i0].alt:+.1f} m)"
        )
        print(f"    es_rel   {t[i0].es_rel:+9.2f} -> {t[j].es_rel:+9.2f} -> {t[i1].es_rel:+9.2f} m")
        print(
            f"    cpa_miss {t[i0].cpa_miss:9.2f} -> {t[j].cpa_miss:9.2f} -> {t[i1].cpa_miss:9.2f} m"
        )
        print(f"    rng      {t[i0].rng:9.2f} -> {t[j].rng:9.2f} -> {t[i1].rng:9.2f} m")
        print(f"    nz       {t[i0].nz:9.2f} -> {t[j].nz:9.2f} -> {t[i1].nz:9.2f} g")
        print(
            f"    |chi|deg {abs(t[i0].chi) * DEG:9.2f} -> {abs(t[j].chi) * DEG:9.2f} -> {abs(t[i1].chi) * DEG:9.2f}"
        )
        print(
            f"    build = {t[j].cpa_miss - t[i0].cpa_miss:+.2f} m on a base cpa of "
            f"{t[i0].cpa_miss:.2f} m = {100.0 * (t[j].cpa_miss - t[i0].cpa_miss) / t[i0].cpa_miss:+.4f} %"
        )
        print(
            f"    cpa_miss==rng (i.e. PAST-CPA degenerate clamp) at i0:{abs(t[i0].cpa_miss - t[i0].rng) < 1e-9}"
            f"  j:{abs(t[j].cpa_miss - t[j].rng) < 1e-9}  exit:{abs(t[i1].cpa_miss - t[i1].rng) < 1e-9}"
        )
        deg = sum(1 for k in range(i0, i1 + 1) if abs(t[k].cpa_miss - t[k].rng) < 1e-9)
        print(f"    degenerate ticks in [i0,exit]: {deg}/{i1 - i0 + 1}")
        print(f"    exit_idx == last trace index? {i1 == len(t) - 1}  (len={len(t)})")
        print(
            f"    cpa_exit vs LETHAL_RANGE_M={B.LETHAL_RANGE_M}: "
            f"{t[i1].cpa_miss:.1f} m = {t[i1].cpa_miss / B.LETHAL_RANGE_M:.1f}x the lethal range"
        )
        print(f"    seg_idx at i0={(i0 * 6) // N}  j={(j * 6) // N}  exit={(i1 * 6) // N}")

print("\n=== (6) Does the SAME structural pooling inflate the CONTROL arms? ===")
print(
    "  prereg DETECTOR merge1500 rows: recorded D3=0/8eps, recorded_two D3=8/8eps, inplane D3=0/8eps"
)
for which in ("recorded_two",):
    per = []
    for sd in SEEDS:
        env, base = M.build_env(cell, sd, N * M.CONTROL_DT_S)
        r = M.roll_once(env, base, M.make_control(which), N, keep=True)
        env.close()
        evs = B.detect_turning_room(r.trace)
        per.append(
            (
                sd,
                r.steps,
                len(evs),
                [
                    (
                        e.i0,
                        e.i1,
                        round(e.detail["cpa_entry_m"], 6),
                        round(e.detail["cpa_peak_m"], 6),
                    )
                    for e in evs
                ],
            )
        )
    print(f"  arm={which}: pooled D3 = {sum(p[2] for p in per)} over {len(per)} seeds")
    for sd, st, n, ev in per:
        print(f"    seed {sd:>3} steps {st} D3 {n} {ev}")
    uniq = {tuple(p[3]) for p in per}
    print(f"  DISTINCT event signatures across the 8 seeds: {len(uniq)}")
