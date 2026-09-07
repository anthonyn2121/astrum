"""Independent recomputation of the D4/coverage claim from the replay dumps (pure stdlib).

Reads the per-step CSVs written by _diag_d3b_replay.py and recomputes, WITHOUT importing
bfm_detect, every number the claim cites:
  * the sign-constant loaded |chi| holds at the 20/30/45/60 deg bars, priced in altitude
  * D4's own predicates verbatim (commitment(): bank hold |phi|>45 & nz>3 sign-constant;
    chi hold |chi|>45 & nz>3 & chi_valid, sign-constant), with dalt on tr[].alt == alt_enu_m
  * D3's hold mask (identical bar to D4's chi hold) and each clause it applies, one at a time,
    so "no member able to fire" can be separated from "a clause rejected it"
Run on the HOST: python3 _diag_refute_coverage.py
"""

from __future__ import annotations

import csv
import math
import os
import sys

RUNS = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"
NZ_LOADED = 3.0
CHI_ROOM_DEG = 45.0
CHI_OUT_DEG = 60.0
CHI_MIN_HOLD_S = 0.5
CHI_RETURN_S = 4.0
BANK_COMMIT_DEG = 45.0


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if k in (
                "step",
                "seg_idx",
                "fire",
                "chi_valid",
                "in_range",
                "lethal",
                "smp_chi_valid",
                "smp_lethal",
            ):
                d[k] = int(float(v))
            else:
                d[k] = float(v)
        out.append(d)
    return out


def runs_of(mask):
    """Contiguous True runs, [i, j] inclusive."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def longest_hold(tr, value, bar, need_valid, validkey="smp_chi_valid"):
    """commitment()._longest_hold verbatim: sign-constant, |value|>bar, nz>NZ_LOADED."""
    n = len(tr)
    held, b0, b1 = 0.0, -1, -1

    def ok(s):
        if need_valid and not s[validkey]:
            return False
        return abs(value(s)) > bar and s["nz_g"] > NZ_LOADED

    i = 0
    while i < n:
        if not ok(tr[i]):
            i += 1
            continue
        sgn = 1 if value(tr[i]) > 0 else -1
        j = i
        while j + 1 < n:
            nxt = tr[j + 1]
            if not ok(nxt):
                break
            if (1 if value(nxt) > 0 else -1) != sgn:
                break
            j += 1
        if tr[j]["t_s"] - tr[i]["t_s"] > held:
            held, b0, b1 = tr[j]["t_s"] - tr[i]["t_s"], i, j
        i = j + 1
    return held, b0, b1


def report(tag, path):
    tr = load(path)
    n = len(tr)
    print("=" * 100)
    print(f"{tag}  n={n} steps  t {tr[0]['t_s']:.3f}-{tr[-1]['t_s']:.3f} s")
    print("=" * 100)

    # ---- A. the sign-constant loaded |chi| holds the claim cites -------------------------
    print("\nA. sign-constant loaded |chi| holds (audit-2 C table), recomputed")
    print("   NOTE two altitude columns: alt_m = AState.alt (geodetic, what geometry uses),")
    print("        alt_enu_m = pos[2] = Sample.alt, which is what commitment() prices dalt on.")
    for bar in (20.0, 30.0, 45.0, 60.0):
        held, i, j = longest_hold(tr, lambda s: s["chi_deg"], bar, True)
        if i < 0:
            print(f"   |chi|>{bar:5.1f}: NONE")
            continue
        ticks = j - i + 1
        print(
            f"   |chi|>{bar:5.1f}: steps {i}-{j} ({ticks} ticks) "
            f"t {tr[i]['t_s']:.3f}-{tr[j]['t_s']:.3f} detector={held:.3f}s physical={ticks * (tr[1]['t_s'] - tr[0]['t_s']):.3f}s  "
            f"chi {tr[i]['chi_deg']:+.1f} -> {tr[j]['chi_deg']:+.1f} (peak {max(abs(tr[k]['chi_deg']) for k in range(i, j + 1)):.1f})  "
            f"nz {min(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}-{max(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}  "
            f"dalt_enu {tr[j]['alt_enu_m'] - tr[i]['alt_enu_m']:+.0f} m  dalt_geo {tr[j]['alt_m'] - tr[i]['alt_m']:+.0f} m"
        )

    # ---- B. D4 verbatim -------------------------------------------------------------------
    print("\nB. D4 == commitment() predicates, verbatim, on this trace")
    bh, bi, bj = longest_hold(
        tr, lambda s: math.radians(s["phi_deg"]), math.radians(BANK_COMMIT_DEG), False
    )
    ch, ci, cj = longest_hold(
        tr, lambda s: math.radians(s["chi_deg"]), math.radians(CHI_ROOM_DEG), True
    )
    for nm, h, i, j in (("bank", bh, bi, bj), ("chi ", ch, ci, cj)):
        if i < 0:
            print(f"   {nm} hold: NONE")
            continue
        print(
            f"   {nm} hold {h:.3f} s  steps {i}-{j} ({j - i + 1} ticks) t {tr[i]['t_s']:.3f}-{tr[j]['t_s']:.3f}  "
            f"dalt_enu {tr[j]['alt_enu_m'] - tr[i]['alt_enu_m']:+.0f} m  dalt_geo {tr[j]['alt_m'] - tr[i]['alt_m']:+.0f} m  "
            f"nz {min(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}-{max(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}"
        )

    # ---- C. the nz>3 loaded run(s) --------------------------------------------------------
    print("\nC. every nz>NZ_LOADED run (the audit's 'single loaded run' claim)")
    for i, j in runs_of([s["nz_g"] > NZ_LOADED for s in tr]):
        print(
            f"   steps {i:3d}-{j:3d}  t {tr[i]['t_s']:.3f}-{tr[j]['t_s']:.3f} ({tr[j]['t_s'] - tr[i]['t_s']:.3f} s, {j - i + 1} ticks)  "
            f"nz {min(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}-{max(tr[k]['nz_g'] for k in range(i, j + 1)):.2f}  "
            f"|phi| {min(abs(tr[k]['phi_deg']) for k in range(i, j + 1)):.1f}-{max(abs(tr[k]['phi_deg']) for k in range(i, j + 1)):.1f}  "
            f"dalt_enu {tr[j]['alt_enu_m'] - tr[i]['alt_enu_m']:+.0f} m"
        )

    # ---- D. D3's OWN mask, clause by clause ----------------------------------------------
    print("\nD. D3 detect_turning_room: its hold mask is |chi|>45 & nz>3 (NO chi_valid clause),")
    print("   then (i) hold>=0.5s (ii) ata_az sign-constant (iii) cpa BUILT (iv) cpa CASHED")
    mask = [(abs(s["chi_deg"]) > CHI_ROOM_DEG and s["nz_g"] > NZ_LOADED) for s in tr]
    for i, j in runs_of(mask):
        dur = tr[j]["t_s"] - tr[i]["t_s"]
        sides = {1 if tr[k]["ata_az_deg"] >= 0.0 else -1 for k in range(i, j + 1)}
        built = tr[j]["cpa_miss_m"] > tr[i]["cpa_miss_m"]
        after = [k for k in range(j + 1, n) if tr[k]["t_s"] - tr[j]["t_s"] <= CHI_RETURN_S]
        cashed = bool(after) and min(tr[k]["cpa_miss_m"] for k in after) < tr[j]["cpa_miss_m"]
        print(
            f"   run steps {i:3d}-{j:3d} dur {dur:.3f}s({j - i + 1} ticks) "
            f"| dur>=0.5 {str(dur >= CHI_MIN_HOLD_S):5s} | ata_az one-sided {str(len(sides) == 1):5s} "
            f"| cpa {tr[i]['cpa_miss_m']:.1f}->{tr[j]['cpa_miss_m']:.1f} built {str(built):5s} | cashed {str(cashed):5s}"
            f"  chi {tr[i]['chi_deg']:+.1f}->{tr[j]['chi_deg']:+.1f}"
        )
    print("   D1 detect_displacement_rolls mask is the SAME shape at the 60 deg bar:")
    m1 = [(abs(s["chi_deg"]) > CHI_OUT_DEG and s["nz_g"] > NZ_LOADED) for s in tr]
    for i, j in runs_of(m1):
        print(
            f"   run steps {i:3d}-{j:3d} dur {tr[j]['t_s'] - tr[i]['t_s']:.3f}s({j - i + 1} ticks) "
            f"dur>=0.5 {str(tr[j]['t_s'] - tr[i]['t_s'] >= CHI_MIN_HOLD_S):5s}"
        )

    # ---- E. monotonicity of cpa_miss over the cited hold ---------------------------------
    print("\nE. cpa_miss over steps 88-108 and over the whole trace (the D3 'build' clause)")
    if n > 108:
        seg = [tr[k]["cpa_miss_m"] for k in range(88, 109)]
        mono = all(seg[k + 1] <= seg[k] for k in range(len(seg) - 1))
        print(
            f"   88-108: {seg[0]:.1f} -> {seg[-1]:.1f} m, max {max(seg):.1f}, monotone non-increasing {mono}"
        )
    cpa = [s["cpa_miss_m"] for s in tr]
    print(
        f"   whole trace: first {cpa[0]:.1f} last {cpa[-1]:.1f} max {max(cpa):.1f} at step {cpa.index(max(cpa))}"
    )

    # ---- F. banned-stat checks ------------------------------------------------------------
    valid = [abs(s["chi_deg"]) for s in tr if s["smp_chi_valid"]]
    valid_sorted = sorted(valid)
    med = valid_sorted[len(valid_sorted) // 2] if valid_sorted else float("nan")
    print(
        f"\nF. |chi| median over valid ticks {med:.1f} deg (BANNED as a BFM stat; printed only to show it is not used)"
    )
    print(
        f"   valid_frac {len(valid) / n:.3f}   max|chi| {max(abs(s['chi_deg']) for s in tr):.1f} deg at step "
        f"{max(range(n), key=lambda k: abs(tr[k]['chi_deg']))}"
    )


if __name__ == "__main__":
    for tag, fn in (
        ("KILL seed 62", "replay_kill_seed62.csv"),
        ("NO-KILL seed 28", "replay_nokill_seed28.csv"),
    ):
        p = os.path.join(RUNS, fn)
        if not os.path.exists(p):
            print(f"MISSING {p}", file=sys.stderr)
            continue
        report(tag, p)
