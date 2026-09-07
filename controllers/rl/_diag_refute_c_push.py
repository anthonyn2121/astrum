"""Counterfactual for D2 root cause C: is PULL-ONLY what blocks the yo-yo, or the 6 s window?

Cause C says: `Genome.act` builds elev = -clip(load*max(cos(bank_err),0), 0, 1) in [-1, 0], so a
nose-down flight-path change must be flown roll-then-pull, and that roll (measured ~1.2 s for
115 deg) is what spends the D2 window.

The only way to price that is to fly the SAME high yo-yo with the roll removed, i.e. with the push
the decode forfeits. `fcs/elevator-cmd-norm` is clipped by the F-16's own FLCS to [-1, +0.44]
(.jsbsim-variants/aircraft/f16/f16.xml:520-531, "100% up and 44% down" = 4 g negative), and
astrum-sim writes a[1] straight into it (crates/astrum-sim/src/env.rs:270), so +0.44 is available
to the ENV even though the D3b decode can never name it.

Arms, all from the same cell/seed, same trigger (roll_once overwrites a[4] for every arm):
  A  PULL115  up half = bank 0 / load 1 pull, return = roll to +115 and pull   <- the decode
  B  PUSH0    up half identical,             return = elev +1.0 (FLCS -> +0.44), bank stays 0
  C  PULL115_NOGATE  as A but mag = 1 (no roll gate): isolates the gate from the roll itself
  D  PUSH115  up half identical, return = roll to +115 AND push (both levers)

Reported per arm: every clause D2 reads, plus altitude paid, plus the measured nz the push buys.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm as bfm
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


class Sched2:
    """Piecewise-constant (bank_deg, load, thr), load SIGNED: >0 = pull (the decode), <0 = push.

    Pull branch is `Genome.act` verbatim. Push branch sets elev = +|load| (the sign the decode's
    outer clip forbids); `gate` mirrors the decode's roll gate and can be switched off.
    """

    def __init__(self, phases, gate: bool = True):
        self.phases = phases
        self.gate = bool(gate)
        self.edges = np.cumsum([p[0] for p in phases])

    def cmd(self, t: float):
        i = int(np.searchsorted(self.edges, t, side="right"))
        i = min(i, len(self.phases) - 1)
        _, bank, load, thr = self.phases[i]
        return float(np.clip(bank, -M.BANK_MAX_DEG, M.BANK_MAX_DEG)), float(load), float(thr)

    def __call__(self, step, me, bd, rel):
        bank_deg, load, thr = self.cmd(step * E.DT)
        phi_tgt = math.radians(bank_deg)
        bank_err = phi_tgt - float(me.phi)
        ail = float(np.clip(bfm.KROLL * bank_err, -1.0, 1.0))
        mag = max(math.cos(bank_err), 0.0) if self.gate else 1.0
        if load >= 0.0:
            elev = -float(np.clip(load * mag, 0.0, 1.0))
        else:
            elev = float(np.clip(-load * mag, 0.0, 1.0))  # the forfeited PUSH half-range
        return np.array([ail, elev, 0.0, thr, -1.0], np.float32)


def d2_clauses(tr) -> dict:
    """Every quantity detect_yoyos reads at the tick-0 entry edge (bfm_detect.py:407-440)."""
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    out = dict(edges=edges)
    if not edges:
        return out
    i = edges[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
    ret = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    best = None
    for p in win:
        sw = math.degrees(abs(tr[p].gamma - g0))
        if (
            sw < B.YOYO_GAMMA_SWING_DEG
            or abs(math.degrees(tr[p].gamma)) < B.YOYO_GAMMA_EXCURSION_DEG
        ):
            continue
        for k in range(p + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                best = (p, k, round(sw, 1), round(tr[k].t, 3))
                break
        if best:
            break
    # the fastest nose-down flight-path rate achieved AFTER the peak, over 0.25 s (6 steps)
    W = 6
    down = float("nan")
    if peak + W < n:
        down = min(
            (math.degrees(tr[k + W].gamma - tr[k].gamma) / (W * E.DT)) for k in range(peak, n - W)
        )
    out.update(
        swing=math.degrees(abs(tr[peak].gamma - g0)),
        gpk=math.degrees(tr[peak].gamma),
        t_peak=tr[peak].t,
        lf=sum(loaded) / max(1, len(loaded)),
        ret=ret,
        t_ret=(tr[ret].t if ret is not None else float("nan")),
        best=best,
        down_rate=down,
    )
    return out


def main() -> None:
    n_steps = int(round(9.0 / E.DT))
    log("#" * 100)
    log("#  REFUTE-OR-CONFIRM D2 CAUSE C -- does the forfeited PUSH half-range close the yo-yo?")
    log("#" * 100)
    log(
        f"  cell merge1500 seed {E.SEED}; horizon {n_steps * E.DT:.2f} s = {n_steps} steps; "
        f"D2 window = {B.YOYO_RETURN_S:.1f} s from the tick-0 closure edge"
    )
    log(
        "  bars: swing >= 30 deg, |gamma_peak| >= 15 deg, loaded_frac >= 0.5 (nz > 3) entry->peak,\n"
        "        return to within 10 deg of gamma(entry) inside the window."
    )
    hdr = (
        f"  {'arm':<22} {'T_up':>5} {'steps':>5} {'event':>9} {'swing':>6} {'gpk':>7} {'tpk':>5} "
        f"{'lf':>4} {'ret':>4} {'tret':>5} {'dn_rate':>8} {'D2':>3} {'D2c':>4} {'D1':>3} {'D3':>3} "
        f"{'dalt':>7} {'nzMin':>6} {'nzMax':>6} {'elevMax':>7}"
    )
    arms = []
    for t_up in (1.5, 2.0, 2.5, 3.0, 3.5):
        arms.append(("A PULL115", t_up, [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, 1.0, 1.0)], True))
        arms.append(("B PUSH0", t_up, [(t_up, 0.0, 1.0, 1.0), (30.0, 0.0, -1.0, 1.0)], True))
        arms.append(
            ("C PULL115_NOGATE", t_up, [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, 1.0, 1.0)], False)
        )
        arms.append(("D PUSH115", t_up, [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, -1.0, 1.0)], True))
    log("")
    log(hdr)
    for name, t_up, phases, gate in arms:
        ctl = Sched2(phases, gate=gate)
        r, rows = E.fly(ctl, n_steps=n_steps)
        st = d2_clauses(r.trace)
        det = B.detect_all(r.trace)
        nz = [row["nz"] for row in rows]
        ev = [row["elev"] for row in rows]
        log(
            f"  {name:<22} {t_up:5.2f} {r.steps:>5} {str(r.event):>9} "
            f"{st.get('swing', float('nan')):>6.1f} {st.get('gpk', float('nan')):>+7.1f} "
            f"{st.get('t_peak', float('nan')):>5.2f} {st.get('lf', float('nan')):>4.2f} "
            f"{str(st.get('ret')):>4} {st.get('t_ret', float('nan')):>5.2f} "
            f"{st.get('down_rate', float('nan')):>+8.2f} "
            f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4} "
            f"{len(det['displacement_rolls']):>3} {len(det['turning_room']):>3} "
            f"{r.dalt_m:>+7.0f} {min(nz):>+6.2f} {max(nz):>+6.2f} {max(ev):>+7.3f}"
        )
        if st.get("best"):
            log(f"      -> BEST (peak, ret, swing, t_ret) = {st['best']}")

    # --- what the push actually buys, measured tick by tick on the best push arm ------------
    log("\n  PUSH AUTHORITY, measured (arm B, T_up 2.50 s): the g the FLCS delivers at elev > 0")
    ctl = Sched2([(2.5, 0.0, 1.0, 1.0), (30.0, 0.0, -1.0, 1.0)], gate=True)
    r, rows = E.fly(ctl, n_steps=n_steps)
    log(
        f"      {'step':>4} {'t':>6} {'elev':>7} {'phi':>7} {'nz':>6} {'gamma':>7} {'dgam/dt':>8} "
        f"{'alt':>7} {'vt':>6}"
    )
    W = 6
    for k in range(0, min(len(rows), n_steps), 6):
        rate = (
            math.degrees(r.trace[k + W].gamma - r.trace[k].gamma) / (W * E.DT)
            if k + W < len(rows)
            else float("nan")
        )
        row = rows[k]
        log(
            f"      {row['step']:>4} {row['t']:>6.3f} {row['elev']:>+7.3f} {row['phi']:>+7.1f} "
            f"{row['nz']:>+6.2f} {row['gamma']:>+7.2f} {rate:>+8.2f} {row['alt']:>7.0f} "
            f"{row['vt']:>6.1f}"
        )

    path = f"{E.OUT}/refute_c_push.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
