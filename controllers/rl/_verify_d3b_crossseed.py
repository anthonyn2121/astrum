"""Cross-seed check of the 'four never-executed segments' profile (segs 2-5, steps 192-575).

Own-ship columns should be bit-identical across the three non-kill seeds (28, 79, 96) because
`Genome.act` reads own-ship attitude only and gun damage does not perturb own-ship dynamics.
Opponent-relative columns (rng, aspect, ...) need not be. This measures which.
"""

from __future__ import annotations

import _diag_d3b_replay as R
import _diag_true_mpc as M
import numpy as np

npz = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
genome = npz["genome"]
cell = M.CELLS["merge1500"]
gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
N = 576

OWN = [
    "phi_deg",
    "theta_deg",
    "psi_deg",
    "alpha_deg",
    "nz_g",
    "alt_m",
    "vt_ms",
    "gamma_deg",
    "roll_rate_dps",
    "es_m",
    "elev",
    "ail",
    "thr_flown",
]
REL = ["rng_m", "aspect_deg", "ata_deg", "closure_ms", "cpa_miss_m", "opp_health"]

out = {}
for sd in (28, 79, 96):
    r, rows, post = R.fly(genome, gm, cell, N, sd)
    out[sd] = rows
    print(
        f"seed {sd}: steps={r.steps} event={r.event} kill={r.kill} hp={r.hp_dealt:.1f} "
        f"own_health={post['health']:.1f} opp_health={post['opp_health']:.1f}"
    )

base = out[28]
print(f"\nrow counts: {[(s, len(out[s])) for s in out]}")
for sd in (79, 96):
    n = min(len(base), len(out[sd]))
    print(f"\n--- seed {sd} vs 28 over {n} rows ---")
    for k in OWN:
        d = max(abs(float(base[i][k]) - float(out[sd][i][k])) for i in range(n))
        print(f"  OWN {k:14s} max|d| = {d:.3e}")
    for k in REL:
        d = max(abs(float(base[i][k]) - float(out[sd][i][k])) for i in range(n))
        print(f"  REL {k:14s} max|d| = {d:.3e}")

print("\n=== the claim's per-seed numbers, recomputed on all three non-kill seeds ===")
hdr = (
    f"{'seed':>5} {'seg2 dEs':>9} {'seg2 dalt':>10} {'seg3 dEs':>9} {'seg3 dalt':>10} "
    f"{'seg4 dEs':>9} {'seg4 dalt':>10} {'seg5 dEs':>9} {'seg5 dalt':>10} "
    f"{'ep dEs':>8} {'ep dalt':>9} {'end gam':>8} {'end vu':>8} "
    f"{'rng384':>8} {'rng479':>8} {'nz>7 ep':>8} {'nz>7 540-575':>13}"
)
print(hdr)
for sd, rows in out.items():

    def f(i, k):
        return float(rows[i][k])

    segs = [(192, 287), (288, 383), (384, 479), (480, 575)]
    cells = []
    for a, b in segs:
        cells.append(f(b, "es_m") - f(a, "es_m"))
        cells.append(f(b, "alt_m") - f(a, "alt_m"))
    n7 = sum(1 for x in rows if float(x["nz_g"]) > 7.0)
    n7w = sum(1 for x in rows[540:576] if float(x["nz_g"]) > 7.0)
    print(
        f"{sd:>5} "
        + " ".join(f"{c:>+9.2f}" if i % 2 == 0 else f"{c:>+10.2f}" for i, c in enumerate(cells))
        + f" {f(575, 'es_m') - f(0, 'es_m'):>+8.2f} {f(575, 'alt_m') - f(0, 'alt_m'):>+9.2f}"
        f" {f(575, 'gamma_deg'):>+8.2f} {f(575, 'vel_u_ms'):>+8.2f}"
        f" {f(384, 'rng_m'):>8.1f} {f(479, 'rng_m'):>8.1f} {n7:>8d} {n7w:>13d}"
    )
