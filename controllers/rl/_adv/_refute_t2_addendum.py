"""Addendum to the T2 refutation: the five discriminating rows."""

from __future__ import annotations

import csv
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


d = load(KILL)
n = d["_n"]
ar = d["alpha_req_deg"]

hdr("A1  DID THE MANEUVER MAKE THE REQUIRED LAY WORSE BEFORE IT MADE IT BETTER?")
worse = np.where(ar > ar[0])[0]
back = np.where((np.arange(n) > 0) & (ar <= ar[0]))[0]
print(
    f"  alpha_req[0] = {ar[0]:.2f} deg.  Steps with alpha_req ABOVE that: {worse.min()}-{worse.max()} "
    f"count {len(worse)}"
)
print(
    f"  first step back at/below the step-0 value: {int(back.min())} (t {d['t_s'][int(back.min())]:.3f} s, "
    f"rng {d['rng_m'][int(back.min())]:.0f} m)"
)
print(
    f"  peak {ar.max() if False else ar[:100].max():.2f} deg at step {int(ar[:100].argmax())} "
    f"(t {d['t_s'][int(ar[:100].argmax())]:.3f}) = {ar[:100].max() - ar[0]:+.2f} deg vs the initial condition"
)
print(
    f"  so for the first {d['t_s'][int(back.min())]:.3f} s of the manoeuvre the required lay was WORSE than "
    f"doing nothing."
)

hdr("A2  LEAD / PURE / LAG — where was the nose relative to the LOS?")
az = d["ata_az_deg"]
el = d["ata_el_deg"]
cross = [k for k in range(1, n) if az[k - 1] > 0 >= az[k]]
print(f"  ata_az zero crossings (nose passing through the LOS in azimuth): {cross}")
print(
    f"  ata_az range: min {az.min():+.2f} max {az.max():+.2f}; steps with ata_az < -1 deg (nose LEADING "
    f"to the left of a right-hand LOS): {int((az < -1).sum())}"
)
print(
    f"  {'step':>4} {'t_s':>6} {'ata_az':>8} {'ata_el':>8} {'ata':>7} {'aspect':>8} {'hca':>9} {'lead_err':>9}"
)
for k in (0, 24, 48, 72, 96, 120, 140, 156, 164, 168, 172):
    print(
        f"  {k:4d} {d['t_s'][k]:6.3f} {az[k]:+8.2f} {el[k]:+8.2f} {d['ata_deg'][k]:7.2f} "
        f"{d['aspect_deg'][k]:8.2f} {d['hca_deg'][k]:+9.2f} {d['lead_error_deg'][k]:9.2f}"
    )

hdr("A3  WAS THE REQUIRED LAY THE CONSTRAINT AT THE STEPS THAT ACTUALLY DEALT DAMAGE?")
print(
    f"  {'step':>4} {'t_s':>6} {'rng':>7} {'tof':>6} {'lead_err':>9} {'lay_miss':>9} {'alpha_req':>9} "
    f"{'alpha':>7} {'opp_hp':>7} {'lethal':>7}"
)
for k in range(154, n):
    print(
        f"  {k:4d} {d['t_s'][k]:6.3f} {d['rng_m'][k]:7.1f} {d['lay_tof_s'][k]:6.3f} "
        f"{d['lead_error_deg'][k]:9.2f} {d['lay_miss_m'][k]:9.2f} {ar[k]:9.2f} {d['alpha_deg'][k]:7.2f} "
        f"{d['opp_health'][k]:7.1f} {int(d['lethal'][k]):7d}"
    )

hdr("A4  SEED 28 — THE 7 POST-PASS FIRING STEPS: DAMAGE DEALT AT alpha_req ~170 deg")
e = load(NOKILL)
f = np.where(e["fire"][173:] > 0)[0] + 173
print(
    f"  {'step':>4} {'t_s':>7} {'rng':>7} {'tof':>6} {'lead_err':>9} {'lay_miss':>9} {'alpha_req':>9} "
    f"{'ata':>7} {'aspect':>8} {'opp_hp':>7}"
)
for k in f:
    print(
        f"  {k:4d} {e['t_s'][k]:7.3f} {e['rng_m'][k]:7.1f} {e['lay_tof_s'][k]:6.3f} "
        f"{e['lead_error_deg'][k]:9.2f} {e['lay_miss_m'][k]:9.2f} {e['alpha_req_deg'][k]:9.2f} "
        f"{e['ata_deg'][k]:7.2f} {e['aspect_deg'][k]:8.2f} {e['opp_health'][k]:7.1f}"
    )
print(
    f"  opp_health before the first of those steps: {e['opp_health'][int(f.min()) - 1]:.1f}; after the last: "
    f"{e['opp_health'][-1]:.1f}"
)
print(
    f"  => {e['opp_health'][int(f.min()) - 1] - e['opp_health'][-1]:.0f} HP dealt while the required lay sat at "
    f"{e['alpha_req_deg'][f].min():.0f}-{e['alpha_req_deg'][f].max():.0f} deg, i.e. "
    f"{e['alpha_req_deg'][f].min() / 15.0:.0f}-{e['alpha_req_deg'][f].max() / 15.0:.0f}x the 15 deg cone."
)

hdr("A5  ALTITUDE PRICE OF THE 'LEAD TURN' (steps 0-172) AND OF THE FULL GENOME (seed 28)")
print(
    f"  kill seed 62: alt {d['alt_m'][0]:.1f} -> {d['alt_m'][172]:.1f} m ({d['alt_m'][172] - d['alt_m'][0]:+.1f}); "
    f"gamma min {d['gamma_deg'].min():.2f} deg; nz max {d['nz_g'].max():.2f} g"
)
seg0 = slice(0, 96)
print(
    f"  seg0 (the turn, steps 0-95): alt {d['alt_m'][0]:.1f} -> {d['alt_m'][95]:.1f} "
    f"({d['alt_m'][95] - d['alt_m'][0]:+.1f} m), nz {d['nz_g'][seg0].min():.2f}-{d['nz_g'][seg0].max():.2f} g, "
    f"Es {d['es_m'][0]:.1f} -> {d['es_m'][95]:.1f} ({d['es_m'][95] - d['es_m'][0]:+.1f} m)"
)
print(
    f"  seed 28 whole genome: alt {e['alt_m'][0]:.1f} -> {e['alt_m'][-1]:.1f} m "
    f"({e['alt_m'][-1] - e['alt_m'][0]:+.1f}), Es {e['es_m'][0]:.1f} -> {e['es_m'][-1]:.1f} "
    f"({e['es_m'][-1] - e['es_m'][0]:+.1f} m)"
)
