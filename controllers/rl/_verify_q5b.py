"""Q5 verification: (A) is D1 expressible at dim 18 / L = 4.0 s?  (B) is D2 reachable in the
L -> 0 limit (a closed-loop, state-triggered bang-bang controller = infinitely fine segments)?

Both probes command ONLY (bank_deg, load, thr) through the same decode as `Genome.act`, so every
action they fly is inside the D3b search space's expressible set at some segment length.
"""

import math
import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

CELL = M.CELLS["merge1500"]
NSTEPS = 576
DT = M.CONTROL_DT_S


def decode(bank_deg, load, thr, me):
    """Byte-for-byte `Genome.act` (param='bank', roll_gate=True, channels=3)."""
    phi_tgt = math.radians(max(-M.BANK_MAX_DEG, min(M.BANK_MAX_DEG, bank_deg)))
    bank_err = phi_tgt - float(me.phi)
    ail = float(np.clip(bfm.KROLL * bank_err, -1.0, 1.0))
    mag = max(math.cos(bank_err), 0.0)
    elev = -float(np.clip(load * mag, 0.0, 1.0))
    return np.array([ail, elev, 0.0, thr, -1.0], np.float32)


def gamma_of(me):
    v = np.asarray(me.vel, float)
    sp = float(np.linalg.norm(v))
    return math.asin(float(np.clip(v[2] / sp, -1.0, 1.0))) if sp > 1e-6 else 0.0


def fly(ctl, seed=11):
    env, base = M.build_env(CELL, seed, NSTEPS * DT)
    try:
        r = M.roll_once(env, base, ctl, NSTEPS, keep=False)
    finally:
        env.close()
    return r


def d2_window(tr):
    """Re-evaluate `detect_yoyos`' clauses on the step-0 closure edge, verbatim thresholds."""
    if not (tr[0].closure > B.CLOSURE_OVERSHOOT):
        return None
    g0 = tr[0].gamma
    win = [k for k in range(len(tr)) if tr[k].t - tr[0].t <= B.YOYO_RETURN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    swing = math.degrees(abs(tr[peak].gamma - g0))
    gpk = math.degrees(tr[peak].gamma)
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(0, peak + 1)]
    lf = sum(loaded) / len(loaded)
    ret = None
    for k in range(peak + 1, len(tr)):
        if tr[k].t - tr[0].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    after = [k for k in win if k > peak]
    closest = min((abs(tr[k].gamma - g0) for k in after), default=float("nan"))
    g_end = next((tr[k].gamma for k in reversed(win)), g0)
    return dict(
        swing=swing,
        gpk=gpk,
        lf=lf,
        peak=peak,
        t_peak=tr[peak].t,
        ret=ret,
        t_ret=(tr[ret].t if ret is not None else float("nan")),
        closest=math.degrees(closest),
        g_end=math.degrees(g_end),
    )


def max_rate(tr, win_s=1.0):
    """Largest |dgamma/dt| averaged over a `win_s` window (deg/s), and its sign."""
    w = max(1, int(round(win_s / DT)))
    best, at = 0.0, -1
    for i in range(len(tr) - w):
        r = math.degrees(tr[i + w].gamma - tr[i].gamma) / (tr[i + w].t - tr[i].t)
        if abs(r) > abs(best):
            best, at = r, i
    return best, at


# =====================================================================================
print("=" * 96)
print("(A)  D1 AT dim 18 / L = 4.0 s -- re-flying the genome from d1_exists_report.txt:15")
print("=" * 96)
G_D1 = np.array(
    [
        1.0,
        1.0,
        0.2,
        0.582792,
        1.0,
        0.2,
        1.0,
        1.0,
        0.2,
        0.872769,
        1.0,
        0.2,
        -0.007301,
        1.0,
        0.2,
        0.312102,
        1.0,
        0.2,
    ]
)
gm = M.Genome(6, 3, "bank", True)
print(f"  dim={gm.dim} segments={gm.segments} -> L = {NSTEPS * DT / gm.segments:.3f} s per segment")
for sd in (11, 62):
    r = fly(lambda step, me, bd, rel: gm.act(G_D1, step, NSTEPS, me), seed=sd)
    d = B.detect_all(r.trace)
    tr = r.trace
    print(
        f"  seed {sd}: steps {r.steps} event {r.event} kill {r.kill} hp {r.hp_dealt:.0f} "
        f"D1 {len(d['displacement_rolls'])} D2 {len(d['yoyos'])} "
        f"D2c {len(d['yoyos_clean'])} D3 {len(d['turning_room'])} "
        f"dalt {tr[-1].alt - tr[0].alt:+.0f} m  score {M.score(r)[0]:+.3f}"
    )
    for e in d["displacement_rolls"]:
        i0, i1 = e.i0, e.i1
        hold = [
            k
            for k in range(i0, i1 + 1)
            if abs(tr[k].chi) > math.radians(B.CHI_OUT_DEG) and tr[k].nz > B.NZ_LOADED
        ]
        h0, h1 = (hold[0], hold[-1]) if hold else (i0, i1)
        print(
            f"    D1 event steps {i0}-{i1} t {tr[i0].t:.3f}-{tr[i1].t:.3f}; "
            f"LOADED |chi|>60 run steps {h0}-{h1} = {tr[h1].t - tr[h0].t:.3f} s "
            f"(bar {B.CHI_MIN_HOLD_S} s, margin {(tr[h1].t - tr[h0].t) / B.CHI_MIN_HOLD_S:.1f}x); "
            f"nz {min(tr[k].nz for k in range(h0, h1 + 1)):.2f}-"
            f"{max(tr[k].nz for k in range(h0, h1 + 1)):.2f} g; "
            f"ALT PRICE of the hold {tr[h1].alt - tr[h0].alt:+.0f} m "
            f"({(tr[h1].alt - tr[h0].alt) / max(1e-9, tr[h1].t - tr[h0].t):+.1f} m/s)"
        )

# =====================================================================================
print()
print("=" * 96)
print("(B)  D2 IN THE L -> 0 LIMIT: closed-loop bang-bang, switch on measured gamma")
print("=" * 96)
print(
    f"  bars: swing >= {B.YOYO_GAMMA_SWING_DEG} deg, |gamma_pk| >= {B.YOYO_GAMMA_EXCURSION_DEG}"
    f" deg, loaded_frac >= {B.LOADED_FRAC}, return |gamma-g0| < {B.YOYO_GAMMA_RETURN_DEG} deg"
    f" within {B.YOYO_RETURN_S} s of the closure edge"
)
print(
    f"  {'dir':>5} {'thr_sw':>7} {'throt':>6} | {'swing':>6} {'gpk':>7} {'lf':>5} "
    f"{'t_peak':>7} {'t_ret':>7} | {'D2':>3} {'closest':>8} {'g@6s':>7} {'maxRate':>8}"
)
rates = []
best = None
for direction in ("up", "down"):
    for thr_sw in (30.5, 31.0):
        for throt in (0.0, 0.5, 1.0):
            st = {"ph": 0}
            sgn = 1.0 if direction == "up" else -1.0

            def ctl(step, me, bd, rel, st=st, sgn=sgn, thr_sw=thr_sw, throt=throt):
                g = gamma_of(me)
                if st["ph"] == 0 and sgn * g >= math.radians(thr_sw):
                    st["ph"] = 1
                if st["ph"] == 1 and sgn * g <= math.radians(8.0):
                    st["ph"] = 2
                if st["ph"] == 0:
                    bank = 0.0 if sgn > 0 else M.BANK_MAX_DEG
                elif st["ph"] == 1:
                    bank = M.BANK_MAX_DEG if sgn > 0 else 0.0
                else:
                    bank = 0.0 if sgn > 0 else M.BANK_MAX_DEG
                return decode(bank, 1.0, throt, me)

            r = fly(ctl)
            d = B.detect_all(r.trace)
            w = d2_window(r.trace)
            mr, _ = max_rate(r.trace)
            rates.append(abs(mr))
            n2, n2c = len(d["yoyos"]), len(d["yoyos_clean"])
            print(
                f"  {direction:>5} {thr_sw:7.1f} {throt:6.2f} | {w['swing']:6.1f} {w['gpk']:+7.1f} "
                f"{w['lf']:5.2f} {w['t_peak']:7.3f} {w['t_ret']:7.3f} | {n2:3d} "
                f"{w['closest']:8.1f} {w['g_end']:+7.1f} {mr:+8.2f}"
            )
            if best is None or w["swing"] > best[0]:
                best = (w["swing"], direction, thr_sw, throt, w)
print(
    f"  largest swing found: {best[0]:.1f} deg ({best[1]}, switch {best[2]:.0f} deg, "
    f"throttle {best[3]:.2f}), return {best[4]['t_ret']}"
)
print(
    f"  max |dgamma/dt| over a 1.0 s window, over all {len(rates)} probes: {max(rates):.2f} deg/s"
)
