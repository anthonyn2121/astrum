"""Independent recomputation of the 'nose-low slice' claim on the D3b kill genome.

Pure stdlib (host-runnable). Reads the two replay dumps and recomputes every number the
claim cites, plus the controls the claim omits (gravity share, quadrature share,
path-integral vs net-displacement, the non-kill seed, and the episode Es base rate).
"""

from __future__ import annotations

import csv
import math
import os

DT = 5.0 / 120.0  # ACTION_REPEAT / RATE_HZ
NZ_LOADED = 3.0  # bfm_detect.py:74
G = 9.80665
RUN = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def runs_of(flags):
    """Maximal runs of True as (start_idx, end_idx) inclusive."""
    out = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def unwrap_deg(seq):
    out = [seq[0]]
    for v in seq[1:]:
        d = v - out[-1]
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        out.append(out[-1] + d)
    return out


def hold_stats(rows, a, b, label):
    seg = rows[a : b + 1]
    n = len(seg)
    dur_steps = (n - 1) * DT
    dur_span = n * DT
    nz = [r["nz_g"] for r in seg]
    gam = [r["gamma_deg"] for r in seg]
    alt = [r["alt_m"] for r in seg]
    es = [r["es_m"] for r in seg]
    vt = [r["vt_ms"] for r in seg]
    psi = unwrap_deg([r["psi_deg"] for r in seg])

    d_psi_net = psi[-1] - psi[0]
    d_gam_net = gam[-1] - gam[0]
    # net-displacement decomposition exactly as the claim states it
    horiz_net = d_psi_net * math.cos(math.radians(sum(gam) / n))
    horiz_net_percos = sum(
        (psi[k + 1] - psi[k]) * math.cos(math.radians(0.5 * (gam[k] + gam[k + 1])))
        for k in range(n - 1)
    )
    norm_net = math.hypot(horiz_net_percos, d_gam_net)

    # path integrals of the two velocity-vector rotation components
    path_h = sum(
        abs(psi[k + 1] - psi[k]) * math.cos(math.radians(0.5 * (gam[k] + gam[k + 1])))
        for k in range(n - 1)
    )
    path_g = sum(abs(gam[k + 1] - gam[k]) for k in range(n - 1))
    path_tot = sum(
        math.hypot(
            (psi[k + 1] - psi[k]) * math.cos(math.radians(0.5 * (gam[k] + gam[k + 1]))),
            gam[k + 1] - gam[k],
        )
        for k in range(n - 1)
    )

    # gravity's unavoidable share of gamma_dot:  gdot_grav = -g cos(gamma)/V
    grav_dgam = sum(
        math.degrees(-G * math.cos(math.radians(gam[k])) / vt[k] * DT) for k in range(n - 1)
    )

    print(f"\n  --- {label}: steps {a}-{b}  n={n}  ({dur_steps:.3f} s between ends, {dur_span:.3f} s span)")
    print(f"      t {seg[0]['t_s']:.3f} -> {seg[-1]['t_s']:.3f} s   seg_idx {seg[0]['seg_idx']:.0f} -> {seg[-1]['seg_idx']:.0f}")
    print(f"      nz mean {sum(nz)/n:.3f} g  min {min(nz):.3f}  max {max(nz):.3f} g (max at step {a + nz.index(max(nz))})")
    print(f"      nz>{NZ_LOADED} on {sum(1 for v in nz if v > NZ_LOADED)}/{n} steps")
    print(f"      gamma<0 on {sum(1 for v in gam if v < 0)}/{n} steps   gamma {gam[0]:+.2f} -> {gam[-1]:+.2f} deg")
    print(f"      alt {alt[0]:.1f} -> {alt[-1]:.1f} m  dAlt {alt[-1]-alt[0]:+.1f} m  ({(alt[-1]-alt[0])/dur_steps:+.2f} m/s)")
    print(f"      alt monotone non-increasing inside hold: {all(alt[k+1] <= alt[k] for k in range(n-1))}")
    print(f"      Es {es[0]:.1f} -> {es[-1]:.1f} m  dEs {es[-1]-es[0]:+.1f} m  Ps={(es[-1]-es[0])/dur_steps:+.2f} m/s")
    print(f"      NET  dPsi {d_psi_net:+.2f} deg   dPsi*cos(mean gam) {horiz_net:+.2f}   sum dPsi*cos(gam) {horiz_net_percos:+.2f}")
    print(f"      NET  dGamma {d_gam_net:+.2f} deg   |norm| {norm_net:.2f}")
    print(f"      NET  share-of-norm (claim's metric)  |dGam|/norm = {abs(d_gam_net)/norm_net*100:.1f}%")
    print(f"      NET  quadrature share dGam^2/norm^2               = {d_gam_net**2/norm_net**2*100:.1f}%")
    print(f"      PATH integral |heading| {path_h:.2f} deg  |gamma| {path_g:.2f} deg  arc {path_tot:.2f} deg")
    print(f"      PATH share |gamma|/(|heading|+|gamma|) = {path_g/(path_h+path_g)*100:.1f}%   |gamma|/arc = {path_g/path_tot*100:.1f}%")
    print(f"      gravity-only dGamma over the hold  = {grav_dgam:+.2f} deg  ({grav_dgam/d_gam_net*100 if d_gam_net else float('nan'):.1f}% of net dGamma)")
    print(f"      residual (lift-attributable) dGamma = {d_gam_net - grav_dgam:+.2f} deg")
    # what the airframe's nz was actually allocated to, from bank angle
    phi = [r["phi_deg"] for r in seg]
    print(f"      phi {phi[0]:+.1f} -> {phi[-1]:+.1f} deg  mean {sum(phi)/n:+.1f}  |cos phi| mean {sum(abs(math.cos(math.radians(p))) for p in phi)/n:.3f}")
    return dict(n=n, dur=dur_steps, dalt=alt[-1] - alt[0], dpsi=d_psi_net, dgam=d_gam_net)


def episode_stats(rows, name):
    n = len(rows)
    alt = [r["alt_m"] for r in rows]
    es = [r["es_m"] for r in rows]
    gam = [r["gamma_deg"] for r in rows]
    nz = [r["nz_g"] for r in rows]
    psi = unwrap_deg([r["psi_deg"] for r in rows])
    ups = [k for k in range(n - 1) if alt[k + 1] > alt[k]]
    print(f"\n=== EPISODE {name}: {n} rows, {(n-1)*DT:.3f} s between ends")
    print(f"    alt {alt[0]:.1f} -> {alt[-1]:.1f} m   dAlt {alt[-1]-alt[0]:+.1f} m")
    print(f"    steps where alt INCREASES: {len(ups)}  {'(first few: ' + str(ups[:8]) + ')' if ups else ''}")
    print(f"      largest single-step climb {max((alt[k+1]-alt[k]) for k in range(n-1)):+.4f} m")
    print(f"    gamma<0 on {sum(1 for v in gam if v < 0)}/{n} rows; gamma>0 rows {[k for k in range(n) if gam[k] > 0][:10]}")
    print(f"    Es {es[0]:.1f} -> {es[-1]:.1f} m  dEs {es[-1]-es[0]:+.1f} m  mean Ps {(es[-1]-es[0])/((n-1)*DT):+.2f} m/s")
    print(f"    psi NET {psi[-1]-psi[0]:+.2f} deg   psi PATH integral {sum(abs(psi[k+1]-psi[k]) for k in range(n-1)):.2f} deg")
    print(f"    alt price  per NET deg heading  {(alt[-1]-alt[0])/(psi[-1]-psi[0]):+.3f} m/deg")
    print(f"    alt price  per PATH deg heading {(alt[-1]-alt[0])/sum(abs(psi[k+1]-psi[k]) for k in range(n-1)):+.3f} m/deg")
    holds = [(a, b) for (a, b) in runs_of([v > NZ_LOADED for v in nz]) if (b - a) * DT >= 0.5]
    print(f"    nz>{NZ_LOADED} holds >=0.5 s: {holds}")
    for a, b in runs_of([v > NZ_LOADED for v in nz]):
        print(f"       run steps {a}-{b}  {(b-a)*DT:.3f} s  meanNz {sum(nz[a:b+1])/(b-a+1):.2f}  dAlt {alt[b]-alt[a]:+.1f} m  dEs {es[b]-es[a]:+.1f} m")
    return holds


def main():
    for name, fn in (("seed62 KILL", "replay_kill_seed62.csv"), ("seed28 NO-KILL", "replay_nokill_seed28.csv")):
        rows = load(os.path.join(RUN, fn))
        holds = episode_stats(rows, name)
        for i, (a, b) in enumerate(holds, 1):
            hold_stats(rows, a, b, f"{name} auto-hold {i}")

    # the claim's own hold windows, verbatim, on the kill seed
    rows = load(os.path.join(RUN, "replay_kill_seed62.csv"))
    print("\n\n### THE CLAIM'S OWN WINDOWS, VERBATIM (kill seed 62)")
    hold_stats(rows, 28, 115, "claim hold 1 (28-115)")
    hold_stats(rows, 123, 172, "claim hold 2 (123-172)")

    # sanity: is step 100 the nz peak? is nz>3 really true at step 28 and 115?
    print("\n### boundary rows, kill seed 62")
    for k in (26, 27, 28, 29, 114, 115, 116, 117, 121, 122, 123, 124, 99, 100, 101):
        r = rows[k]
        print(f"   step {k:3d} t {r['t_s']:.3f} nz {r['nz_g']:+.3f} gam {r['gamma_deg']:+.2f} alt {r['alt_m']:.1f} es {r['es_m']:.1f} phi {r['phi_deg']:+.1f} psi {r['psi_deg']:.2f}")


if __name__ == "__main__":
    main()
