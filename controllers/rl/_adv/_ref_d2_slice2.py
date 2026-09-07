"""Part 2: does clause (d) fail because the flight path NEVER returns, or because the return
arrives after YOYO_RETURN_S? Threshold sensitivity + the altitude price of the 'return'."""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
from astrum_rl import _paths  # noqa: E402

_paths.ensure_on_path()

from astrum_dogfight import bfm_detect as B  # noqa: E402

KILL = "/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv"


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: np.array([float(r[n]) for r in rows]) for n in names}
    d["_n"] = len(rows)
    return d


def trace(d):
    return [
        B.Sample(
            t=d["smp_t"][k],
            chi=d["smp_chi_rad"][k],
            chi_valid=bool(d["smp_chi_valid"][k]),
            gamma=d["smp_gamma_rad"][k],
            nz=d["smp_nz"][k],
            phi=d["smp_phi_rad"][k],
            alt=d["smp_alt"][k],
            ail=d["smp_ail"][k],
            rng=d["smp_rng"][k],
            ata=d["smp_ata"][k],
            ata_az=d["smp_ata_az"][k],
            aspect=d["smp_aspect"][k],
            closure=d["smp_closure"][k],
            cpa_miss=d["smp_cpa_miss"][k],
            es_rel=d["smp_es_rel"][k],
            lay_miss_m=d["smp_lay_miss_m"][k],
            lethal=bool(d["smp_lethal"][k]),
        )
        for k in range(d["_n"])
    ]


k, nk = load(KILL), load(NOKILL)
tk, tn = trace(k), trace(nk)

print("=== the RETURN, step by step, around the 10-deg bar (both traces, same commanded stream)")
print("   step   t_s   seg  gamma_deg  |gam-g0|  alt_m     nz_g   rng_m  aspect  closure  cmd_bank")
for s in list(range(96, 216, 6)):
    print(
        f"   {s:4d} {nk['t_s'][s]:6.3f}  {int(nk['seg_idx'][s])}   {nk['gamma_deg'][s]:+8.2f} "
        f"{abs(nk['gamma_deg'][s] - nk['gamma_deg'][0]):8.2f} {nk['alt_m'][s]:8.1f} "
        f"{nk['nz_g'][s]:6.2f} {nk['rng_m'][s]:7.0f} {nk['aspect_deg'][s]:7.1f} "
        f"{nk['closure_ms'][s]:8.1f}  {nk['cmd_bank_deg'][s]:+8.2f}"
        + ("   <- kill seed ends here" if s == 172 else "")
    )

g0 = nk["smp_gamma_rad"][0]
gam = nk["smp_gamma_rad"]
bar = math.radians(B.YOYO_GAMMA_RETURN_DEG)
first = next(s for s in range(110, 576) if abs(gam[s] - g0) < bar)
print(
    f"\n   first return step {first} t {nk['t_s'][first]:.3f}s gamma {nk['gamma_deg'][first]:+.3f} "
    f"seg {int(nk['seg_idx'][first])}"
)
print(
    f"   last step of segment 1 (step 191, t {nk['t_s'][191]:.3f}s): gamma "
    f"{nk['gamma_deg'][191]:+.3f} deg, |gam-g0| = {abs(nk['gamma_deg'][191] - nk['gamma_deg'][0]):.3f} "
    f"deg -> still {abs(nk['gamma_deg'][191] - nk['gamma_deg'][0]) - 10.0:+.3f} deg outside the bar"
)
print(
    f"   recovery achieved INSIDE the kill episode (steps 109->172): "
    f"{k['gamma_deg'][109]:+.2f} -> {k['gamma_deg'][172]:+.2f} deg = "
    f"{k['gamma_deg'][172] - k['gamma_deg'][109]:+.2f} deg of the "
    f"{abs(k['gamma_deg'][109] - k['gamma_deg'][0]) - 10.0:.2f} deg needed to reach the bar"
)
print(
    f"   ALTITUDE PRICE of the return: alt_m at peak (109) {nk['alt_m'][109]:.1f} -> at return "
    f"({first}) {nk['alt_m'][first]:.1f} = {nk['alt_m'][first] - nk['alt_m'][109]:+.1f} m; "
    f"Es {nk['es_m'][109]:.1f} -> {nk['es_m'][first]:.1f} = {nk['es_m'][first] - nk['es_m'][109]:+.1f} m"
)
a = nk["alt_m"]
da = np.diff(a)
print(
    f"   seed28 alt_m: {a[0]:.1f} -> {a[-1]:.1f} ({a[-1] - a[0]:+.1f} m); steps up "
    f"{int((da > 0).sum())}/{len(da)}, down {int((da < 0).sum())}/{len(da)}; max {a.max():.1f} at step "
    f"{int(a.argmax())}"
)
print(
    f"   kill seed62 alt_m: {k['alt_m'][0]:.1f} -> {k['alt_m'][-1]:.1f} "
    f"({k['alt_m'][-1] - k['alt_m'][0]:+.1f} m)"
)

print("\n=== THRESHOLD SENSITIVITY: monkeypatch the two bars the claim argues about")
for swing, rets in ((30.0, 6.0), (28.0, 6.0), (30.0, 9.0), (28.0, 9.0), (28.0, 12.0), (25.0, 12.0)):
    B.YOYO_GAMMA_SWING_DEG, B.YOYO_RETURN_S = swing, rets
    rk, rn = B.detect_yoyos(tk), B.detect_yoyos(tn)
    print(
        f"   swing={swing:5.1f} return_s={rets:5.1f} -> kill62 D2={len(rk)}  nokill28 D2={len(rn)}"
    )
    for lab, evs in (("kill62", rk), ("nokill28", rn)):
        for e in evs:
            print(
                f"       {lab}: {e.kind} i0={e.i0} i1={e.i1} t {e.t0:.3f}->{e.t1:.3f}s "
                f"swing={e.detail['gamma_swing_deg']:.2f} loadfrac={e.detail['loaded_frac']:.3f} "
                f"clo {e.detail['closure_entry']:.0f}->{e.detail['closure_exit']:.0f} "
                f"d_aspect={e.d_aspect_deg:+.1f} d_rng={e.d_rng_m:+.0f} d_es={e.d_es_m:+.0f} "
                f"failed_exit={e.detail['failed_exit']} failed_aspect={e.detail['failed_aspect']} "
                f"clean={e.detail['clean']}"
            )
B.YOYO_GAMMA_SWING_DEG, B.YOYO_RETURN_S = 30.0, 6.0

print("\n=== was this an OVERSHOOT at all? (D2's entry gate premise)")
for lab, d in (("kill62", k), ("nokill28", nk)):
    print(
        f"   {lab}: step0 rng {d['rng_m'][0]:.0f} m aspect {d['aspect_deg'][0]:.1f} ata "
        f"{d['ata_deg'][0]:.1f} closure {d['closure_ms'][0]:+.1f} m/s; "
        f"rng min {d['rng_m'].min():.0f} at step {int(d['rng_m'].argmin())}"
    )
