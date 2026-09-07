import csv
import sys

sys.path.insert(0, "/workspace/crates/astrum-env/python")
from astrum_dogfight import bfm_detect as B


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


D = "/workspace/controllers/rl/runs/d3b/"
K = build(D + "replay_kill_seed62.csv")
N = build(D + "replay_nokill_seed28.csv")
print("rebuilt: kill", len(K), "nokill", len(N))

DEF = {
    k: getattr(B, k)
    for k in (
        "CHI_OUT_DEG",
        "CHI_ROOM_DEG",
        "CHI_MIN_HOLD_S",
        "ASPECT_GAIN_DEG",
        "CHI_INPLANE_DEG",
        "CHI_RETURN_DEG",
        "CHI_RETURN_S",
        "CHI_LOOKBACK_S",
        "NZ_LOADED",
        "CLOSURE_OVERSHOOT",
        "YOYO_ASPECT_SLIP_DEG",
        "YOYO_GAMMA_SWING_DEG",
    )
}


def setth(**kw):
    for k, v in DEF.items():
        setattr(B, k, v)
    for k, v in kw.items():
        setattr(B, k, v)


def run(tr, label, **kw):
    setth(**kw)
    d = B.detect_all(tr)
    print(
        f"  {label:52s} D1={len(d['displacement_rolls'])} D2={len(d['yoyos'])} "
        f"D2c={len(d['yoyos_clean'])} D3={len(d['turning_room'])}"
    )
    for e in d["displacement_rolls"]:
        print(
            f"      D1 ev [{e.i0},{e.i1}] t {e.t0:.3f}-{e.t1:.3f}s dasp={e.d_aspect_deg:+.2f} "
            f"drng={e.d_rng_m:+.1f} dEs={e.d_es_m:+.1f} detail={ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.detail.items()} }"
        )
    return d


print("\n--- KILL seed 62 ---")
run(K, "DEFAULTS (validate rebuild vs reported D1=D2=D2c=D3=0)")
run(K, "ASPECT_GAIN_DEG=-180 only", ASPECT_GAIN_DEG=-180.0)
run(K, "CHI_OUT_DEG=45 only", CHI_OUT_DEG=45.0)
run(K, "CHI_OUT_DEG=45 + ASPECT_GAIN_DEG=-180", CHI_OUT_DEG=45.0, ASPECT_GAIN_DEG=-180.0)
run(K, "CHI_OUT_DEG=40 + ASPECT_GAIN_DEG=-180", CHI_OUT_DEG=40.0, ASPECT_GAIN_DEG=-180.0)
run(K, "CHI_OUT_DEG=30 + ASPECT_GAIN_DEG=-180", CHI_OUT_DEG=30.0, ASPECT_GAIN_DEG=-180.0)

print("\n--- D3 threshold sweep on the KILL trace (is D3 really unreachable?) ---")
for room in (45.0, 30.0, 20.0, 10.0, 5.0, 0.0):
    for hold in (0.5, 0.2, 0.05):
        setth(CHI_ROOM_DEG=room, CHI_MIN_HOLD_S=hold)
        d = B.detect_all(K)
        print(
            f"  CHI_ROOM_DEG={room:5.1f} CHI_MIN_HOLD_S={hold:.2f} -> D3={len(d['turning_room'])}"
        )
setth()
print("\n--- NOKILL seed 28 ---")
run(N, "DEFAULTS")
run(N, "CHI_OUT_DEG=45 + ASPECT_GAIN_DEG=-180", CHI_OUT_DEG=45.0, ASPECT_GAIN_DEG=-180.0)
setth()
