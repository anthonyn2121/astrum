import csv
import math

import astrum_rl  # noqa
from astrum_dogfight import bfm_detect as B

DEG = math.degrees


def load(p):
    rows = list(csv.DictReader(open(p)))
    return [
        B.Sample(
            t=float(r["smp_t"]),
            chi=float(r["smp_chi_rad"]),
            chi_valid=r["smp_chi_valid"] in ("1", "True", "true"),
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
            lethal=r["smp_lethal"] in ("1", "True", "true"),
        )
        for r in rows
    ]


R = "/workspace/controllers/rl/runs/d3b"
for lbl, f in (("KILL62", "replay_kill_seed62.csv"), ("NOKILL28", "replay_nokill_seed28.csv")):
    tr = load(f"{R}/{f}")
    g0 = tr[0].gamma
    print(f"\n== {lbl}: g0={DEG(g0):+.3f} deg, YOYO_RETURN_S deadline t<=6.000s (entry step 0)")
    # gamma within the 6 s window
    w = [k for k in range(len(tr)) if tr[k].t - tr[0].t <= B.YOYO_RETURN_S]
    print(
        f"   window steps 0-{w[-1]} (t {tr[w[-1]].t:.3f}); gamma at window end {DEG(tr[w[-1]].gamma):+.2f}"
    )
    pk = max(w, key=lambda k: abs(tr[k].gamma - g0))
    print(
        f"   peak in window: step {pk} t {tr[pk].t:.3f} gamma {DEG(tr[pk].gamma):+.2f} swing {DEG(abs(tr[pk].gamma - g0)):.3f}"
    )
    ret = [
        k
        for k in range(pk + 1, len(tr))
        if tr[k].t - tr[0].t <= B.YOYO_RETURN_S
        and abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG)
    ]
    print(f"   gamma-return candidates inside deadline: {ret[:3] if ret else 'NONE'}")
    # counterfactual: relax the swing bar only
    old = B.YOYO_GAMMA_SWING_DEG
    for bar in (30.0, 28.0, 25.0, 20.0):
        B.YOYO_GAMMA_SWING_DEG = bar
        ev = B.detect_yoyos(tr)
        print(
            f"   swing bar {bar:5.1f} -> D2={len(ev)} "
            + "; ".join(
                f"[{e.kind} {e.i0}->{e.i1} dasp {e.d_aspect_deg:+.2f} kept={not e.detail['failed_aspect']} clean={e.detail['clean']}]"
                for e in ev
            )
        )
    B.YOYO_GAMMA_SWING_DEG = old
