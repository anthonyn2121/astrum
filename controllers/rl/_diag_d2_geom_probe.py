"""Read-only probe of the d3b replay CSVs: entry/exit geometry of the relaxed-D2 window."""

from __future__ import annotations

import csv
import math


def main() -> None:
    rows = list(csv.DictReader(open("runs/d3b/replay_kill_seed62.csv")))
    print("n rows", len(rows))
    for s in (0, 23, 109, 110, 156, 163, 172):
        r = rows[s]
        print(
            f"step {s:>3} t {float(r['t_s']):6.3f} clos {float(r['closure_ms']):+8.1f} "
            f"rng {float(r['rng_m']):7.1f} aspect {float(r['aspect_deg']):7.2f} "
            f"ata {float(r['ata_deg']):7.2f} gamma {float(r['gamma_deg']):+7.2f} "
            f"nz {float(r['nz_g']):5.2f} alt_enu {float(r['alt_enu_m']):8.1f} "
            f"smp_clos {float(r['smp_closure']):+8.1f} "
            f"smp_aspect_deg {math.degrees(float(r['smp_aspect'])):7.2f}"
        )
    cl = [float(r["smp_closure"]) for r in rows]
    print(
        "min smp_closure",
        min(cl),
        "at",
        cl.index(min(cl)),
        "| max",
        max(cl),
        "at",
        cl.index(max(cl)),
    )
    print("ticks <=120:", sum(1 for c in cl if c <= 120.0))
    asp = [math.degrees(float(r["smp_aspect"])) for r in rows]
    print(
        f"aspect entry {asp[0]:.2f} exit110 {asp[110]:.2f} d {asp[110] - asp[0]:+.2f} "
        f"| exit163 {asp[163]:.2f} d {asp[163] - asp[0]:+.2f}"
    )
    alt = [float(r["smp_alt"]) for r in rows]
    print(
        f"dalt 0->110 {alt[110] - alt[0]:+.1f} m | 0->163 {alt[163] - alt[0]:+.1f} m | "
        f"0->172 {alt[172] - alt[0]:+.1f} m"
    )
    g = [float(r["smp_gamma_rad"]) * 180 / math.pi for r in rows]
    print(
        f"gamma max {max(g):+.2f} min {min(g):+.2f} ticks with gamma>0: {sum(1 for x in g if x > 0)}"
    )
    nz = [float(r["smp_nz"]) for r in rows]
    print(f"loaded frac steps 0..109: {sum(1 for x in nz[:110] if x > 3.0) / 110.0:.4f}")
    print(f"loaded frac steps 0..163: {sum(1 for x in nz[:164] if x > 3.0) / 164.0:.4f}")

    r28 = list(csv.DictReader(open("runs/d3b/replay_nokill_seed28.csv")))
    c28 = [float(r["smp_closure"]) for r in r28]
    above = [i for i, c in enumerate(c28) if c > 120.0]
    print(
        f"\nseed28: n {len(r28)} ticks>120 {len(above)} first {above[0]} last {above[-1]} "
        f"contiguous {above == list(range(above[0], above[-1] + 1))}"
    )
    print(f"seed28 closure min {min(c28):+.1f} at {c28.index(min(c28))} max {max(c28):+.1f}")


if __name__ == "__main__":
    main()
