"""Which D3 clause rejects the REFERENCE arm on merge1500? (follow-up to _diag_verify_leadturn_claim)

The claim under test says D3 = 0 on the D3b optimum because the "room must have been BUILT" clause
(bfm_detect.py:481) fails. If that clause is the structural reason, then the arm D3 was designed
around -- which DOES build room -- should pass it. This prints the same clause cascade the detector
audit prints, for LTA1000+ (with APPROACH) and LTAturn0 (TURN from step 0).
"""

import math

import _diag_merge_convert_tmp as CV
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / M.CONTROL_DT_S))
CELL = M.CELLS["merge1500"]


def runs_of(mask):
    out, i = [], 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def cascade(tag, tr):
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    print(f"\n  {tag}: {len(tr)} ticks")
    print(f"    ingredient |chi| > CHI_ROOM_DEG={B.CHI_ROOM_DEG}: "
          f"{sum(1 for s in tr if abs(s.chi) > math.radians(B.CHI_ROOM_DEG))}/{len(tr)}")
    print(f"    ingredient nz > NZ_LOADED={B.NZ_LOADED}: {sum(1 for s in tr if s.nz > B.NZ_LOADED)}/{len(tr)}")
    print(f"    CO-OCCURRENCE (the hold mask): {sum(hold)}/{len(tr)}")
    rs = runs_of(hold)
    print(f"    runs: {rs}")
    print(f"    {'run':>12} {'dur_s':>6} {'chiPk':>6} {'nzPk':>5} {'side':>5} {'cpa_i':>8} {'cpa_j':>8} "
          f"{'built':>6} {'cashed':>7}  verdict")
    for i, j in rs:
        dur = tr[j].t - tr[i].t
        sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}
        built = tr[j].cpa_miss > tr[i].cpa_miss
        after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
        cashed = bool(after) and min(tr[k].cpa_miss for k in after) < tr[j].cpa_miss
        fails = []
        if dur < B.CHI_MIN_HOLD_S:
            fails.append("dur")
        if len(sides) != 1:
            fails.append("side")
        if not built:
            fails.append("built")
        if not cashed:
            fails.append("cashed")
        print(f"    {f'{i}-{j}':>12} {dur:>6.3f} {max(abs(tr[k].chi) for k in range(i, j + 1)) * DEG:>6.1f} "
              f"{max(tr[k].nz for k in range(i, j + 1)):>5.2f} "
              f"{('one' if len(sides) == 1 else 'BOTH'):>5} "
              f"{tr[i].cpa_miss:>8.1f} {tr[j].cpa_miss:>8.1f} {str(built):>6} {str(cashed):>7}  "
              f"{('PASS' if not fails else 'FAIL ' + ','.join(fails))}")
    c = [s.cpa_miss for s in tr]
    print(f"    cpa_miss {c[0]:.1f} -> {c[-1]:.1f} m; max {max(c):.1f}; RISES on "
          f"{sum(1 for k in range(len(c) - 1) if c[k + 1] > c[k])}/{len(c) - 1} ticks")
    nz = [s.nz for s in tr]
    print(f"    nz: max {max(nz):.2f} g; ticks nz>3: {sum(1 for x in nz if x > 3.0)}/{len(nz)}")
    chi = [abs(s.chi) * DEG for s in tr]
    print(f"    |chi| among LOADED (nz>3) ticks: max "
          f"{max([chi[k] for k in range(len(tr)) if nz[k] > 3.0], default=float('nan')):.1f} deg")


def main():
    print("#" * 100)
    print("#  WHICH D3 CLAUSE rejects the reference arm on merge1500 (seed 62)")
    print("#" * 100)
    print(f"  CHI_ROOM_DEG={B.CHI_ROOM_DEG} NZ_LOADED={B.NZ_LOADED} CHI_MIN_HOLD_S={B.CHI_MIN_HOLD_S} "
          f"CHI_RETURN_S={B.CHI_RETURN_S}")
    for tag, turn_at in (("LTA1000+ (APPROACH->TURN)", 1000.0), ("LTAturn0 (TURN from step 0)", float("inf"))):
        env, base = M.build_env(CELL, 62, HORIZON_S)
        arm = CV.LeadTurnArm(turn_at, True)
        phis = []

        def ctl(step, me, bd, rel, _arm=arm, _phis=phis):
            _phis.append((float(me.phi) * DEG, _arm.phase))
            return _arm.act(me, bd, rel)

        r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
        env.close()
        cascade(tag, r.trace)
        ap = np.array([p for p, ph in phis])
        print(f"    |phi| over the episode: min {np.abs(ap).min():.1f} max {np.abs(ap).max():.1f} deg; "
              f"ticks |phi| > 90 (past vertical): {int((np.abs(ap) > 90.0).sum())}/{len(ap)}; "
              f"ticks |phi| >= 110: {int((np.abs(ap) >= 110.0).sum())}/{len(ap)}")
        print(f"    outcome: event={r.event} kill={r.kill} hp={r.hp_dealt:.1f} dalt={r.dalt_m:+.0f} m "
              f"rngMin={r.rng_min:.0f} m rounds={r.rounds}")


if __name__ == "__main__":
    main()
