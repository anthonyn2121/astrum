import csv
import math
import sys

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import bfm_detect as B

N_STEPS = int(round(24.0 * M.RATE_HZ / M.ACTION_REPEAT))


def runs(f):
    out = []
    i = 0
    n = len(f)
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and f[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def diag(rec, label):
    t, nz, az, ar, cpa, rng, clo, alt, asp = (
        rec[k] for k in ("t", "nz", "az", "ar", "cpa", "rng", "clo", "alt", "asp")
    )
    print(f"  {label}: nz>3 runs and PER-CLAUSE verdicts")
    nev = 0
    for i, j in runs([x > B.NZ_LOADED for x in nz]):
        flips = [k for k in range(i, j) if (az[k] >= 0) != (az[k + 1] >= 0)]
        c = []
        if t[j] - t[i] < 0.5:
            c.append("dur<0.5s")
        if clo[i] <= 0.0:
            c.append("not closing")
        if rng[i] <= B.LETHAL_RANGE_M:
            c.append("inside pass")
        if not flips:
            c.append("no ata_az flip")
        if not ar[j] < ar[i]:
            c.append("alpha_req NOT falling")
        if not cpa[j] < cpa[i]:
            c.append("cpa NOT falling")
        ok = not c
        nev += ok
        print(
            f"    [{i:3d},{j:3d}] dur={t[j] - t[i]:6.3f}s rng_i={rng[i]:7.1f} clo_i={clo[i]:7.1f} "
            f"ar {math.degrees(ar[i]):6.1f}->{math.degrees(ar[j]):6.1f} cpa {cpa[i]:7.1f}->{cpa[j]:7.1f} "
            f"flips={len(flips)} dalt={alt[j] - alt[i]:8.1f} -> {'ACCEPT' if ok else 'reject: ' + ', '.join(c)}"
        )
    print(f"    => {nev} event(s)")
    return nev


def make(kind):
    rec = {k: [] for k in ("t", "nz", "az", "ar", "cpa", "rng", "clo", "alt", "asp")}

    def ctrl(step, me, bd, rel):
        chi, valid = B.out_of_plane_angle(me, bd)
        rec["t"].append(step * M.ACTION_REPEAT / M.RATE_HZ)
        rec["nz"].append(float(me.nz))
        rec["az"].append(float(rel.ata_az))
        rec["ar"].append(float(rel.alpha_req))
        rec["rng"].append(float(rel.rng))
        rec["clo"].append(float(rel.closure))
        rec["alt"].append(float(me.pos[2]))
        rec["asp"].append(float(rel.aspect))
        rec["cpa"].append(B.cpa_miss_m(me.pos, me.vel, bd.pos, bd.vel))
        if kind == "SL":
            return np.array(
                [float(np.clip(bfm.KROLL * (0.0 - float(me.phi)), -1, 1)), 0.0, 0.0, 0.6, -1.0],
                np.float32,
            )
        load = 1.0 if kind == "PP" else 0.5
        ail = float(np.clip(-bfm.KROLL * chi, -1, 1)) if valid else 0.0
        mag = max(math.cos(chi), 0.0) if valid else 1.0
        return np.array([ail, -float(np.clip(load * mag, 0, 1)), 0.0, 1.0, -1.0], np.float32)

    return ctrl, rec


DEF = {k: getattr(B, k) for k in ("CHI_OUT_DEG", "ASPECT_GAIN_DEG")}
env, base = M.build_env(M.CELLS["merge1500"], 62, 24.0)
for kind in ("PP", "PPm", "SL"):
    ctrl, rec = make(kind)
    r = M.roll_once(env, base, ctrl, N_STEPS, keep=False)
    for k, v in DEF.items():
        setattr(B, k, v)
    d0 = B.detect_all(r.trace)
    B.CHI_OUT_DEG, B.ASPECT_GAIN_DEG = 45.0, -9.0
    d1 = B.detect_all(r.trace)
    for k, v in DEF.items():
        setattr(B, k, v)
    print(
        f"\n=== {kind} hp_dealt={r.hp_dealt:.1f} rng_min={r.rng_min:.1f} dalt={r.dalt_m:.1f} | "
        f"D1 default={len(d0['displacement_rolls'])}  D1 LOOSENED(chi45,asp-9)={len(d1['displacement_rolls'])} ==="
    )
    for e in d1["displacement_rolls"]:
        print(
            f"      loosened D1 ev [{e.i0},{e.i1}] dasp={e.d_aspect_deg:+.2f} hold={e.detail['hold_s']:.3f}s nzpk={e.detail['nz_peak']:.2f}"
        )
    diag(rec, kind)
env.close()


# ---- minimal ASPECT_GAIN_DEG for the kill trace, and which D2 clause rejects it ----
def build(path):
    tr = []
    with open(path) as f:
        for r in csv.DictReader(f):
            tr.append(
                B.Sample(
                    t=float(r["smp_t"]),
                    chi=float(r["smp_chi_rad"]),
                    chi_valid=bool(int(float(r["smp_chi_valid"]))),
                    gamma=float(r["smp_gamma_rad"]),
                    nz=float(r["smp_nz"]),
                    phi=float(r["smp_phi_rad"]),
                    alt=float(r["smp_alt"]),
                    ail=float(r["smp_ail"]),
                    rng=float(r["smp_rng"]),
                    ata=float(r["smp_ata"]),
                    ata_az=float(r["smp_ata_az"]),
                    aspect=float(r["smp_aspect"]),
                    closure=float(r["smp_closure"]),
                    cpa_miss=float(r["smp_cpa_miss"]),
                    es_rel=float(r["smp_es_rel"]),
                    lay_miss_m=float(r["smp_lay_miss_m"]),
                    lethal=bool(int(float(r["smp_lethal"]))),
                )
            )
    return tr


K = build("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv")
print("\n--- minimal ASPECT_GAIN_DEG at CHI_OUT_DEG=45 on the KILL trace ---")
for a in (15.0, 5.0, 0.0, -5.0, -8.0, -8.6, -8.7, -9.0, -15.0):
    B.CHI_OUT_DEG, B.ASPECT_GAIN_DEG = 45.0, a
    print(f"   ASPECT_GAIN_DEG={a:6.1f} -> D1={len(B.detect_all(K)['displacement_rolls'])}")
for k, v in DEF.items():
    setattr(B, k, v)

print("\n--- D2 on the KILL trace: which clause rejects? ---")
g = [math.degrees(s.gamma) for s in K]
clo = [s.closure for s in K]
print(f"   closure[0]={clo[0]:.1f} (> {B.CLOSURE_OVERSHOOT} -> D2 entry gate FIRES at i=0)")
print(f"   gamma range over the trace: min={min(g):.2f} max={max(g):.2f} deg; gamma[0]={g[0]:.2f}")
print(
    f"   max |gamma - gamma[0]| within 6.0 s = {max(abs(x - g[0]) for x in g):.2f} deg "
    f"vs YOYO_GAMMA_SWING_DEG={B.YOYO_GAMMA_SWING_DEG}"
)
print(
    f"   n ticks with closure<=120: {sum(1 for c in clo if c <= B.CLOSURE_OVERSHOOT)} / {len(clo)}"
)
