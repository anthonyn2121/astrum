"""Probe 2: the three things probe 1 left open.

E  SEED ROBUSTNESS. Every published row and every probe-1 row was flown on seed 62 only.
   Re-fly 8 representative schedules on all 8 search seeds and check that swing / ret / #edges
   / D2 are seed-invariant (the opponent is scripted, but the gun PRNG and any opponent
   stochasticity are seeded, and early termination truncates the window).

F  IS D2c REALLY 'FORCED BY THE CELL GEOMETRY'?  The report's basis is that the in-window
   closure FLOOR was +125..+375 m/s over its 16 trajectories, all of which roll to +115.
   `clean` needs closure < 120 AND rng < 1500 at the exit index. Sweep the down-bank over
   the full reachable range (-115..+115) and report, for each, whether ANY index in the
   [0, 6] s window satisfies BOTH sub-gates -- i.e. whether a clean exit is geometrically
   available at all.

G  THE RATE BUDGET, measured. The largest sustained |dgamma/dt| the decode can produce in
   each direction, and the wall time the roll gate costs during a 115 deg reversal
   (elev = -clip(load*max(cos(bank_err),0),0,1) is exactly 0 whenever |bank_err| >= 90 deg).
   Turn that into a lower bound on the time a 30 deg swing + return needs, and compare with
   YOYO_RETURN_S = 6.0 s.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []
WIN_S = B.YOYO_RETURN_S


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def stats(tr) -> dict:
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    if not edges:
        return dict(edges=edges, n=n)
    i = edges[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= WIN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    ret = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > WIN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    ok = [k for k in win if tr[k].closure < B.CLOSURE_OVERSHOOT and tr[k].rng < B.YOYO_EXIT_RNG_M]
    return dict(
        edges=edges,
        n=n,
        swing=math.degrees(abs(tr[peak].gamma - g0)),
        gpk=math.degrees(tr[peak].gamma),
        peak=peak,
        t_peak=tr[peak].t,
        t_win_end=tr[win[-1]].t,
        ret=ret,
        clo_min=min(tr[k].closure for k in win),
        rng_min=min(tr[k].rng for k in win),
        clean_idx=ok[:4],
        n_clean_idx=len(ok),
    )


def main() -> None:
    N10 = int(round(10.0 / E.DT))
    N24 = int(round(24.0 / E.DT))
    log("#" * 118)
    log("#  PROBE 2 -- seed robustness, the D2c 'cell geometry' basis, and the rate budget")
    log("#" * 118)

    # -------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  E  SEED ROBUSTNESS (all 8 D3b search seeds; published grid + probe 1 used seed 62 only)")
    log("=" * 118)
    reps = [
        (
            "HIGH pull 3.0s @bank0 then bank115 pull",
            [(3.0, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
        ),
        ("SEG  4.0s bank0 then 4.0s bank115", [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)]),
        (
            "FINE 0.5 bank0 / 1.0 bank115 / rest bank0",
            [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
        ),
        ("LOW  4.2s bank115 then bank0", [(4.2, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
        ("HIGH 2.8s bank0 then bank115", [(2.8, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)]),
        (
            "HIGH 2.2s bank0 then bank115 (a RETURNING row)",
            [(2.2, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
        ),
        ("LOW- 4.6s bank-115 then bank0", [(4.6, -115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]),
        ("IDLE LOW 4.2s bank115 then bank0", [(4.2, 115.0, 1.0, 0.0), (24.0, 0.0, 1.0, 0.0)]),
    ]
    log(
        f"     {'schedule':<48} {'seed':>4} {'steps':>5} {'#ed':>3} {'swing':>6} {'tPk':>5} {'ret':>5} {'D2':>3} {'D2c':>4}"
    )
    for nm, phases in reps:
        for sd in (11, 28, 45, 62, 79, 96, 113, 130):
            r, rows = E.fly(E.Sched(phases), n_steps=N10, seed=sd)
            st = stats(r.trace)
            det = B.detect_all(r.trace)
            log(
                f"     {nm:<48} {sd:>4} {r.steps:>5} {len(st['edges']):>3} "
                f"{st.get('swing', float('nan')):>6.1f} {st.get('t_peak', float('nan')):>5.2f} "
                f"{str(st.get('ret')):>5} {len(det['yoyos']):>3} {len(det['yoyos_clean']):>4}"
            )

    # -------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log(
        "  F  IS A CLEAN EXIT GEOMETRICALLY AVAILABLE?  down-bank sweep over the full reachable range"
    )
    log("=" * 118)
    log(
        f"     `clean` at the exit index needs closure < {B.CLOSURE_OVERSHOOT:.0f} m/s AND "
        f"rng < {B.YOYO_EXIT_RNG_M:.0f} m. n_ok = # of indices inside [0, 6] s that satisfy BOTH."
    )
    log(
        f"     {'schedule':<48} {'cloMinWin':>9} {'rngMinWin':>9} {'n_ok':>5} {'ok idx':<22} {'swing':>6} {'ret':>5}"
    )
    for d in (2.0, 3.0, 4.0):
        for bank in (-115.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 115.0):
            nm = f"{d:.1f}s bank{bank:+.0f} pull -> bank0 pull"
            r, rows = E.fly(E.Sched([(d, bank, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)]), n_steps=N10)
            st = stats(r.trace)
            log(
                f"     {nm:<48} {st.get('clo_min', float('nan')):>9.0f} "
                f"{st.get('rng_min', float('nan')):>9.0f} {st.get('n_clean_idx', 0):>5} "
                f"{str(st.get('clean_idx')):<22} {st.get('swing', float('nan')):>6.1f} {str(st.get('ret')):>5}"
            )

    # -------------------------------------------------------------------------------------
    log("\n" + "=" * 118)
    log("  G  RATE BUDGET -- measured max |dgamma/dt| per direction and the roll-gate dead time")
    log("=" * 118)
    log(
        "     elev = -clip(load*max(cos(bank_err),0),0,1)  (_diag_true_mpc.py:218-219): the elevator\n"
        "     is EXACTLY ZERO for every step with |bank_err| >= 90 deg, so a 0 -> 115 deg reversal is\n"
        "     flown at zero commanded pull until the bank error falls under 90 deg."
    )
    probes = [
        ("bank 0, max pull (UP rate)", [(24.0, 0.0, 1.0, 1.0)]),
        ("bank 115, max pull (DOWN rate)", [(24.0, 115.0, 1.0, 1.0)]),
        (
            "bank 0 (4 s) -> bank 115 (reversal, roll-gate cost)",
            [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)],
        ),
        (
            "bank 115 (4 s) -> bank 0 (reversal back)",
            [(4.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)],
        ),
    ]
    for nm, phases in probes:
        r, rows = E.fly(E.Sched(phases), n_steps=N24)
        k0 = 0
        if len(phases) > 1:
            k0 = int(round(phases[0][0] / E.DT))
        w = rows[: int(round(8.0 / E.DT))]
        rate = [(w[k + 6]["gamma"] - w[k]["gamma"]) / (6 * E.DT) for k in range(len(w) - 6)]
        # roll-gate dead time after the edge: consecutive steps with |elev| < 0.02
        dead = 0
        for k in range(k0, min(len(rows), k0 + int(round(3.0 / E.DT)))):
            if abs(rows[k]["elev"]) < 0.02:
                dead += 1
            elif dead:
                break
        nzlo = min(rows[k]["nz"] for k in range(k0, min(len(rows), k0 + int(round(1.5 / E.DT)))))
        log(
            f"     {nm:<52} max_up {max(rate):+6.2f} deg/s  max_dn {min(rate):+6.2f} deg/s  "
            f"roll-gate dead {dead * E.DT:5.3f} s (elev==0)  min nz in 1.5 s after edge {nzlo:5.2f}"
        )
    log(
        f"\n     LOWER BOUND on a 30 deg swing + return to within {B.YOYO_GAMMA_RETURN_DEG:.0f} deg, using the\n"
        "     measured max rates above and one 115 deg reversal:\n"
        "       t = 30/|rate_out| + dead_time + 20/|rate_back|\n"
        f"     compare against YOYO_RETURN_S = {B.YOYO_RETURN_S:.1f} s."
    )

    path = f"{E.OUT}/refute_d2_express2.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
