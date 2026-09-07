"""Follow-up: does the validity gate do WORK on seed 28's holds, and where is its D3 event?"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
from _verify_chi_cond import (  # noqa: E402
    KILL,
    NOKILL,
    B,
    build_trace,
    chi_of,
    geom,
    load,
    max_dchi,
)


def runs_of(mask):
    out, n = [], len(mask)
    i = 0
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


def analyse(name, path):
    rows = load(path)
    n = len(rows)
    print("=" * 100)
    print(f"{name}  {n} ticks")
    chi_g, chi_u, nz, t, alt, cpa, ata_v = [], [], [], [], [], [], []
    for r in rows:
        _, _, _, _, los_hat, v_hat, _, av = geom(r)
        c = chi_of(
            los_hat,
            v_hat,
            math.radians(float(r["phi_deg"])),
            math.radians(float(r["theta_deg"])),
            math.radians(float(r["psi_deg"])),
        )
        chi_g.append(float(r["chi_deg"]))
        chi_u.append(math.degrees(c) if c is not None else float("nan"))
        nz.append(float(r["nz_g"]))
        t.append(float(r["t_s"]))
        alt.append(float(r["alt_m"]))
        cpa.append(float(r["cpa_miss_m"]))
        ata_v.append(av)
    chi_g, chi_u, nz = np.array(chi_g), np.array(chi_u), np.array(nz)
    t, alt, cpa, ata_v = map(np.array, (t, alt, cpa, ata_v))

    for bar, lbl in ((B.CHI_ROOM_DEG, "D3 bar 45"), (B.CHI_OUT_DEG, "D1 bar 60")):
        for series, tag in (
            (chi_g, "GATED  (what the detector read)"),
            (chi_u, "UNGATED (gate off)"),
        ):
            m = (np.abs(series) > bar) & (nz > B.NZ_LOADED)
            rr = [(a, b) for a, b in runs_of(m)]
            long = [(a, b) for a, b in rr if t[b] - t[a] >= B.CHI_MIN_HOLD_S]
            print(
                f"  {lbl}  {tag}: {len(rr)} loaded runs, {len(long)} of them >= "
                f"{B.CHI_MIN_HOLD_S} s"
            )
            for a, b in rr:
                dur = t[b] - t[a]
                flag = " <== PASSES HOLD_S" if dur >= B.CHI_MIN_HOLD_S else ""
                print(
                    f"      steps {a:4d}-{b:4d}  t {t[a]:6.3f}-{t[b]:6.3f}  {dur:5.3f} s"
                    f"  dalt {alt[b] - alt[a]:+7.1f} m"
                    f"  cpa {cpa[a]:7.1f}->{cpa[b]:7.1f} m"
                    f"  ata_v {ata_v[a]:6.1f}->{ata_v[b]:6.1f}{flag}"
                )

    # max |chi| gated vs ungated
    print(
        f"  max |chi| GATED   = {np.nanmax(np.abs(chi_g)):.1f} deg at step "
        f"{int(np.nanargmax(np.abs(chi_g)))}  (ata_v {ata_v[int(np.nanargmax(np.abs(chi_g)))]:.2f},"
        f" nz {nz[int(np.nanargmax(np.abs(chi_g)))]:+.2f} g)"
    )
    k = int(np.nanargmax(np.abs(chi_u)))
    print(
        f"  max |chi| UNGATED = {np.nanmax(np.abs(chi_u)):.1f} deg at step {k}"
        f"  (ata_v {ata_v[k]:.2f}, nz {nz[k]:+.2f} g,"
        f" amp {max_dchi(*geom(rows[k])[4:6], math.radians(float(rows[k]['phi_deg'])), math.radians(float(rows[k]['theta_deg'])), math.radians(float(rows[k]['psi_deg'])), 1.0, n=360):.1f} deg/deg)"
    )

    # D3 event provenance
    tr = build_trace(rows, ungate=False)
    for ev in B.detect_turning_room(tr):
        print(
            f"  D3 EVENT: steps {ev.i0}-{ev.i1}  t {ev.t0:.3f}-{ev.t1:.3f}"
            f"  detail {ev.detail}  d_aspect {ev.d_aspect_deg:+.1f}  d_rng {ev.d_rng_m:+.1f}"
        )
        print(
            f"    priced: dalt {alt[ev.i1] - alt[ev.i0]:+.1f} m over {ev.t1 - ev.t0:.3f} s;"
            f"  nz over hold min {nz[ev.i0 : ev.i1 + 1].min():.2f} max {nz[ev.i0 : ev.i1 + 1].max():.2f}"
        )
    tru = build_trace(rows, ungate=True)
    print(
        f"  D3 count gated {len(B.detect_turning_room(tr))}  ungated {len(B.detect_turning_room(tru))}"
    )


analyse("KILL seed 62", KILL)
analyse("NOKILL seed 28", NOKILL)
