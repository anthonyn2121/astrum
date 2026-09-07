"""PROBE 5 — measure the f16's ACTUAL turn performance in THIS sim vs altitude/speed.

Drives the raw action interface (aileron/elevator/rudder/throttle) directly through
`astrum_env.DogfightEnv` (1 tick = 1/120 s) and records the max-performance turn:

  * MAXPULL<bank>  roll to `bank` deg, then hold elev = -1 (max pull), throttle 1.
  * VERT           wings level, hold elev = -1 (a pure vertical pull / loop).

Reports, per (alt, KCAS): peak Nz, Nz / turn rate / turn radius at t = 1, 2, 5, 10 s
after the pull starts, alpha, airspeed and altitude bleed. Turn rate is measured off
the VELOCITY VECTOR (the thing that sets the turn radius), not the heading.

Run:  PYTHONPATH=/workspace/crates/astrum-env/python:/workspace/controllers/rl/vendor \
        python /workspace/crates/astrum-env/python/_diag_turnperf.py
"""

from __future__ import annotations

import argparse
import math

import astrum_env
import numpy as np

DT = 1.0 / 120.0
FT = 0.3048


def _rho(h_m: float) -> float:
    """ISA density (kg/m^3), troposphere."""
    t = 288.15 - 0.0065 * h_m
    p = 101325.0 * (t / 288.15) ** 5.25588
    return p / (287.053 * t)


def scen(alt_ft: float, kts: float, model: str = "f16") -> str:
    return f"""
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_m = 0.0

[sim]
rate_hz = 120.0
max_time_s = 0.0
realtime = false
seed = 7

[[aircraft]]
id = 1
model = "{model}"
lat_deg = 37.6188
lon_deg = -122.3756
alt_ft = {alt_ft}
heading_deg = 0.0
airspeed_kts = {kts}
team = 0

[[aircraft]]
id = 2
model = "{model}"
lat_deg = 37.9188
lon_deg = -122.3756
alt_ft = {alt_ft}
heading_deg = 0.0
airspeed_kts = {kts}
team = 1
"""


def vel(o):
    # obs vel is local NED (vel_n, vel_e, vel_d)
    return np.array([o.vel_n, o.vel_e, o.vel_d], dtype=float)


def run_case(
    alt_ft: float,
    tas_tgt: float,
    bank_deg: float,
    settle_s=25.0,
    pull_s=14.0,
    hold=False,
    model: str = "f16",
):
    # start at a CAS that lands near the TAS target, then trim onto it with throttle
    rho_ratio = math.sqrt(1.225 / max(1e-3, _rho(alt_ft * FT)))
    env = astrum_env.DogfightEnv(scen(alt_ft, tas_tgt / 0.514444 / rho_ratio, model))
    obs = {o.id: o for o in env.reset()}
    me = obs[1]

    # --- phase 1: settle wings-level at altitude, throttle holding the TAS target ---
    n_settle = int(settle_s / DT)
    for _ in range(n_settle):
        e = max(-1.0, min(1.0, 3.0 * me.theta + 0.6 * me.q))  # elev+ = nose down
        a = max(-1.0, min(1.0, -2.0 * me.phi - 0.3 * me.p))
        v = float(np.linalg.norm(vel(me)))
        thr = max(0.0, min(1.0, 0.45 + 0.02 * (tas_tgt - v)))
        out, _ = env.step([(1, [a, e, 0.0, thr, 0.0]), (2, [0.0, 0.0, 0.0, 0.5, 0.0])])
        me = {o.id: o for o in out}[1]

    v0 = float(np.linalg.norm(vel(me)))
    alt0 = me.alt_m
    rec = []
    prev_v = vel(me)
    n_pull = int(pull_s / DT)
    bank = math.radians(bank_deg)
    for k in range(n_pull):
        # roll onto the commanded bank as fast as the FLCS allows, then full pull
        a = max(-1.0, min(1.0, 2.0 * ((bank - me.phi + math.pi) % (2 * math.pi) - math.pi)))
        if hold:  # throttle holds the target TAS through the pull (instantaneous-turn curve)
            vv = float(np.linalg.norm(vel(me)))
            thr = max(0.0, min(1.0, 0.45 + 0.02 * (tas_tgt - vv)))
        else:
            thr = 1.0
        out, _ = env.step([(1, [a, -1.0, 0.0, thr, 0.0]), (2, [0.0, 0.0, 0.0, 0.5, 0.0])])
        me = {o.id: o for o in out}[1]
        v = vel(me)
        sp = float(np.linalg.norm(v))
        pv = float(np.linalg.norm(prev_v))
        if sp > 1.0 and pv > 1.0:
            c = float(np.dot(v, prev_v) / (sp * pv))
            om = math.acos(max(-1.0, min(1.0, c))) / DT  # rad/s, velocity-vector turn rate
        else:
            om = 0.0
        prev_v = v
        rec.append(
            dict(
                t=(k + 1) * DT,
                nz=me.nz_g,
                alpha=math.degrees(me.alpha_rad),
                V=sp,
                alt=me.alt_m,
                phi=math.degrees(me.phi),
                theta=math.degrees(me.theta),
                q=me.q,
                p=me.p,
                omega=om,
            )
        )
    return v0, alt0, rec


def smooth(rec, key, t, win=0.5):
    lo, hi = t - win / 2, t + win / 2
    vals = [r[key] for r in rec if lo <= r["t"] <= hi]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="85,0")
    ap.add_argument("--alts", default="5000,10000,15000,17000")
    ap.add_argument("--tas", default="150,180,200,230")
    ap.add_argument("--hold", action="store_true", help="throttle holds the target TAS through the pull")
    ap.add_argument(
        "--model",
        default="f16",
        help="JSBSim aircraft name. Checklist item A3 (defect D-C) compares the stock `f16` against\n"
        "two VARIANTS of its `fcs/g-load-pid` -- `f16nopid` (the <trigger> removed) and `f16kp10`\n"
        "(kp 0.3 -> 1.0). The stock file is a third-party asset outside the repo, so an in-place edit\n"
        "would be invisible to git; the variants live in a copied JSBSim root selected with\n"
        "JSBSIM_ROOT=/workspace/.jsbsim-variants. Build it with `make jsbsim-variants`.",
    )
    a = ap.parse_args()
    banks = [float(x) for x in a.banks.split(",")]
    alts = [float(x) for x in a.alts.split(",")]
    tass = [float(x) for x in a.tas.split(",")]

    for bank in banks:
        tag = "VERT (wings level)" if bank == 0 else f"MAXPULL bank={bank:.0f}deg"
        print(
            f"\n===== {tag} : model={a.model}  elev=-1, "
            f"throttle={'TAS-hold' if a.hold else '1'} ====="
        )
        print(
            f"{'alt_ft':>7} {'TASt':>5} {'TAS0':>6} | "
            + " ".join(f"{'t=' + str(t) + 's':>39}" for t in (1, 2, 5, 10))
        )
        print(
            f"{'':>7} {'':>5} {'':>6} | "
            + " ".join(f"{'nz':>5}{'a':>5}{'V':>6}{'w/s':>6}{'R':>6}{'dh':>6}{'q/s':>5}" for _ in range(4))
        )
        for alt in alts:
            for tt in tass:
                v0, alt0, rec = run_case(alt, tt, bank, hold=a.hold, model=a.model)
                row = f"{alt:7.0f} {tt:5.0f} {v0:6.1f} |"
                peak = max(r["nz"] for r in rec)
                for t in (1, 2, 5, 10):
                    nz = smooth(rec, "nz", t)
                    al = smooth(rec, "alpha", t)
                    V = smooth(rec, "V", t)
                    om = smooth(rec, "omega", t)
                    qq = smooth(rec, "q", t)
                    R = V / om if om > 1e-4 else float("inf")
                    dh = smooth(rec, "alt", t) - alt0
                    row += (
                        f" {nz:5.2f}{al:5.1f}{V:6.1f}{math.degrees(om):6.1f}"
                        f"{min(R, 99999):6.0f}{dh / FT:6.0f}{math.degrees(qq):5.0f}"
                    )
                print(row + f"  peakNz={peak:.2f}")


if __name__ == "__main__":
    main()
